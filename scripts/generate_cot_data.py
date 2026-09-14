#!/usr/bin/env python3
"""Generate verified chain-of-thought traces for the training pool using a
GLM teacher API (OpenAI-compatible endpoint).

Each output record carries the original question, the teacher's CoT with an
anchored `ANSWER:` line, the number of reasoning steps used, and a verified
flag asserting the anchored answer matches the record's gold answers under
the project's own multi-gold normalization (src/evaluation/metrics.py).
Unverifiable traces are retried, then dropped.

The cache is id-keyed and resumable: rerunning skips already-generated ids.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import _normalize_answer, _normalize_references

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$")
STEP_RE = re.compile(r"(?im)^\s*(?:step\s+(\d+))\s*[:.)]")

DEFAULT_BASE_URL = os.environ.get(
    "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
)


def stable_id(record: dict, index: int) -> str:
    import hashlib

    key = record.get("question", "") + "||" + str(
        (record.get("answers") or record.get("targets") or [""])[0]
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{record.get('source_dataset', 'unk')}-{index:06d}-{h}"


def gold_list(record: dict) -> list[str]:
    golds: list[str] = []
    for field in ("answers", "final_answers", "targets"):
        v = record.get(field)
        if isinstance(v, list):
            golds.extend(str(g) for g in v if str(g).strip())
        elif v:
            golds.append(str(v))
    golds.extend(str(a) for a in (record.get("aliases") or []) if str(a).strip())
    return golds or [""]


def build_user_prompt(record: dict, max_steps: int, min_steps: int) -> str:
    parts = []
    subject = str(record.get("subject", "") or "").strip()
    if subject:
        parts.append(f"Context: {subject}")
    parts.append(f"Question: {record['question']}")
    choices = record.get("choices")
    if choices:
        opts = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        parts.append(f"Options:\n{opts}")
    parts.append(f"Gold answer: {gold_list(record)[0]}")
    steps_clause = (
        f"Use between {min_steps} and {max_steps} numbered reasoning steps "
        f"(only as many as the question needs), then give the final answer."
        if max_steps > 1
        else "Give a single brief reasoning step, then the final answer."
    )
    parts.append(
        "Write your reasoning as 'Step 1: ...', 'Step 2: ...', etc. "
        + steps_clause
        + " Finish with a line of exactly the form 'ANSWER: <answer>' where "
        "<answer> is the gold answer phrased as directly as possible."
    )
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are an expert temporal-reasoning teacher. You reason about dates, "
    "durations, event order, and timelines in news text, then state the "
    "final answer in the exact requested format."
)


def extract_answer(cot: str) -> str | None:
    matches = ANSWER_RE.findall(cot)
    return matches[-1].strip() if matches else None


def count_steps(cot: str) -> int:
    nums = [int(n) for n in STEP_RE.findall(cot)]
    return max(nums) if nums else 0


def answer_matches(extracted: str, golds: list[str]) -> bool:
    if extracted is None:
        return False
    norm = _normalize_answer(extracted)
    return norm in _normalize_references(golds)


def call_teacher(client, model: str, user_prompt: str, max_steps: int,
                 temperature: float, max_attempts: int) -> tuple[str | None, str]:
    import requests

    last_err = ""
    for attempt in range(max_attempts):
        try:
            resp = client.post(
                f"{DEFAULT_BASE_URL.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 160 + 90 * max_steps,
                },
                timeout=120,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep((2 ** attempt) + random.random())
                continue
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage", {})
            return (
                body["choices"][0]["message"]["content"],
                "",
                {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            )
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep((2 ** attempt) + random.random())
    return None, last_err, {}


class LocalTeacher:
    """Local HF-model teacher (e.g. models/Qwen3.5-9B) — run with a
    modern-transformers interpreter such as venv_qwen (the frozen eval
    env may not know new architectures)."""

    def __init__(self, model_path: str, max_steps: int,
                 few_shot_file: str = ""):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.truncation_side = "left"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="auto",
            )
        self.model.eval()
        self.max_new = 160 + 90 * max_steps

        # Optional few-shot seeding (STaR A/B pilot arm B): prepend up to 3
        # verified manual traces as REAL user/assistant turns, not text
        # flattened into the system prompt. A chat-tuned model imitates a
        # demonstration that occupies an actual prior turn; the same text
        # dumped into the system message instead reads as background
        # reference, not a format to reproduce -- confirmed by a pilot run
        # where flattened few-shot dropped step-formatted output from 27%
        # (zero-shot) to 1.5% (see docs/session_handoff.md D35).
        self.few_shot_messages: list[dict] = []
        if few_shot_file:
            n = 0
            for line in open(few_shot_file, encoding="utf-8"):
                c = json.loads(line)
                if c.get("verified") and c.get("cot"):
                    q = str(c.get("question") or "").strip()
                    cot = str(c.get("cot") or "").strip()
                    if q and cot:
                        self.few_shot_messages.append({"role": "user", "content": q})
                        self.few_shot_messages.append({"role": "assistant", "content": cot})
                        n += 1
                if n >= 3:
                    break

    def generate(self, user_prompts: list[str], temperature: float,
                 batch_size: int = 8) -> list[str]:
        import torch

        order = sorted(range(len(user_prompts)),
                       key=lambda i: len(user_prompts[i]))
        outs: list[str | None] = [None] * len(user_prompts)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + self.few_shot_messages
                    + [{"role": "user", "content": user_prompts[i]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for i in idx
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4),
                    top_p=0.95,
                )
            for j, i in enumerate(idx):
                outs[i] = self.tok.decode(
                    gen[j][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
        return [o or "" for o in outs]


def process_record(client, model: str, record: dict, index: int,
                   args) -> dict | None:
    rid = stable_id(record, index)
    golds = gold_list(record)
    user_prompt = build_user_prompt(record, args.max_steps, args.min_steps)

    for round_ in range(args.rounds):
        if getattr(args, "local_teacher", None) is not None:
            prompt_i = user_prompt if round_ == 0 else (
                user_prompt + "\n(Previous attempt was invalid; produce a cleaner, "
                "correctly formatted trace.)"
            )
            texts = args.local_teacher.generate(
                [prompt_i], args.temperature, 1,
            )
            cot, err, usage = texts[0], "", {}
        else:
            cot, err, usage = call_teacher(
                client, model, user_prompt, args.max_steps,
                args.temperature, args.retries,
            )
        if cot is None:
            return {
                "id": rid, "index": index, "verified": False,
                "error": err, "model": model,
            }
        extracted = extract_answer(cot)
        steps = count_steps(cot)
        ok = answer_matches(extracted, golds) and 0 < steps <= args.max_steps
        if ok:
            return {
                "id": rid,
                "index": index,
                "question": record.get("question"),
                "gold": golds[0],
                "extracted_answer": extracted,
                "steps_used": steps,
                "cot": cot,
                "verified": True,
                "round": round_,
                "model": model,
                "usage": usage,
            }
    return {
        "id": rid, "index": index, "verified": False,
        "error": "answer_mismatch_after_rounds", "model": model,
    }


def self_test() -> int:
    fake_cot = "Step 1: Check the dates. 2010 to 2019 is ten years.\nStep 2: The employers are listed.\nANSWER: University of Edinburgh"
    assert extract_answer(fake_cot) == "University of Edinburgh"
    assert count_steps(fake_cot) == 2
    assert answer_matches("university of edinburgh.", ["University of Edinburgh"])
    assert not answer_matches("Glasgow University", ["University of Edinburgh"])
    mcq = "Step 1: Compare.\nFINAL ANSWER: B. 44 minutes"
    assert extract_answer(mcq) == "B. 44 minutes"
    print("self-test OK")
    return 0


def run_local(todo, args, out_path) -> int:
    """Batched local-teacher driver: length-sorted batches, incremental
    cache writes; verified results are flushed per batch, failures retry
    next round, final-round failures are stored as verified=false."""
    lt = args.local_teacher
    remaining_pairs = [(i, r) for i, r in todo]
    last_rec: dict[int, dict] = {}
    verified = failed = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for round_ in range(args.rounds):
            if not remaining_pairs:
                break
            prompts = []
            for i, r in remaining_pairs:
                up = build_user_prompt(r, args.max_steps, args.min_steps)
                if round_ > 0:
                    up += ("\n(Previous attempt was invalid; produce a cleaner, "
                           "correctly formatted trace.)")
                prompts.append(up)
            outs = lt.generate(prompts, args.temperature, args.gen_batch)
            still = []
            for (i, r), cot in zip(remaining_pairs, outs):
                golds = gold_list(r)
                extracted = extract_answer(cot)
                steps = count_steps(cot)
                ok = answer_matches(extracted, golds) and 0 < steps <= args.max_steps
                rec = {
                    "id": stable_id(r, i), "index": i,
                    "question": r.get("question"), "gold": golds[0],
                    "extracted_answer": extracted, "steps_used": steps,
                    "cot": cot, "verified": bool(ok), "round": round_,
                    "model": f"local:{args.local_model}", "usage": {},
                }
                if ok:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    verified += 1
                else:
                    still.append((i, r))
                    last_rec[i] = rec
            remaining_pairs = still
            print(f"round {round_ + 1}: verified={verified} "
                  f"pending_retry={len(remaining_pairs)}", flush=True)
        for i, r in remaining_pairs:
            out.write(json.dumps(last_rec[i], ensure_ascii=False) + "\n")
            failed += 1
        out.flush()
    total = verified + failed
    print(
        f"done: {total} processed, {verified} verified "
        f"({100 * verified / total if total else 0:.1f}%), {failed} failed"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/combined_80_20_v2/train.jsonl")
    ap.add_argument("--out", dest="out", default="data/cot/cot_train.jsonl")
    ap.add_argument("--max-steps", type=int, default=5,
                    help="maximum reasoning steps the teacher may use (definable)")
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2,
                    help="regeneration attempts per record on verification failure")
    ap.add_argument("--retries", type=int, default=3,
                    help="network retries per call with backoff")
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "glm-5.3-max"))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sources", default="",
                    help="comma-separated source_dataset filter (default: all)")
    ap.add_argument("--types", default="",
                    help="comma-separated type filter (default: all)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="process only the first N selected records (0 = all)")
    ap.add_argument("--local-model", default="",
                    help="use a local HF model as teacher (STaR self-rationales)")
    ap.add_argument("--few-shot-file", default="",
                    help="verified-CoT jsonl; prepend up to 3 traces as few-shot "
                         "seeds (STaR pilot arm B)")
    ap.add_argument("--gen-batch", type=int, default=8,
                    help="batch size for local teacher generation")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    client = None
    if args.local_model:
        args.local_teacher = LocalTeacher(args.local_model, args.max_steps,
                                          getattr(args, "few_shot_file", "") or "")
        args.concurrency = 1
    else:
        api_key = os.environ.get("GLM_API_KEY", "")
        if not api_key:
            print(
                "ERROR: GLM_API_KEY not set. Put 'GLM_API_KEY=...' in .env or the "
                "environment (see .env.example), or use --local-model.",
                file=sys.stderr,
            )
            return 2

        import requests

        client = requests.Session()
        client.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    records = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    types = {s.strip() for s in args.types.split(",") if s.strip()}
    selected = [
        (i, r)
        for i, r in enumerate(records)
        if (not sources or r.get("source_dataset") in sources)
        and (not types or r.get("type") in types)
    ]
    if args.max_samples:
        selected = selected[: args.max_samples]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [(i, r) for i, r in selected if stable_id(r, i) not in done_ids]

    print(
        f"selected={len(selected)} cached={len(selected) - len(todo)} "
        f"todo={len(todo)} model={args.model} max_steps={args.max_steps}"
    )

    if getattr(args, "local_teacher", None) is not None:
        return run_local(todo, args, out_path)

    verified = failed = 0
    with open(out_path, "a", encoding="utf-8") as out, cf.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {
            pool.submit(process_record, client, args.model, r, i, args): (i, r)
            for i, r in todo
        }
        for n, fut in enumerate(cf.as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as exc:
                result = {"verified": False, "error": f"worker: {exc}"}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if n % 25 == 0 or n == len(todo):
                out.flush()
                print(
                    f"{n}/{len(todo)} verified={verified} failed={failed}",
                    flush=True,
                )
            if result.get("verified"):
                verified += 1
            else:
                failed += 1

    total = verified + failed
    print(
        f"done: {total} processed, {verified} verified "
        f"({100 * verified / total if total else 0:.1f}%), {failed} failed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
