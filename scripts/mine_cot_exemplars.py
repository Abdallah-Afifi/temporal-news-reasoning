#!/usr/bin/env python
"""Mine few-shot CoT exemplars from the TRAINING POOL (D63).

Design (zero-shot-only methodology preserved):
- Candidates come from data/combined_80_20_v6/train.jsonl (canonical llama
  cycle data, audited) -- NEVER from benchmarks: zero leakage.
- Each candidate is prompted exactly like the zero_shot_reasoning arm
  (same _build_cot_prompt + COT_SYSTEM_PROMPT + greedy + 256 tokens) and
  the ANSWER:-anchored extraction is verified against the row's golds
  (targets + final_answers) with the frozen scorer.
- Among verified-correct generations, the SHORTEST trace per kind wins
  (short traces teach termination -- the zs-R failure mode, D59/D61).
- Kinds: 2x phrase_qa, 1x ordering (B,C,A), 1x duration_phrase.

Run on GPU (minutes). Output: data/cot_fewshot_exemplars_mined.json,
same schema as data/cot_fewshot_exemplars.json (the withdrawn
handwritten file stays for reference only).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

from scripts.run_baselines import (  # noqa: E402
    COT_SYSTEM_PROMPT,
    _build_cot_prompt,
    _format_cot_exemplar,
    _postprocess_prediction,
)
from src.data.data_loader import TemporalExample  # noqa: E402
from src.evaluation.date_equivalence import matches_any  # noqa: E402
from src.models.inference import SLMInference  # noqa: E402

POOL = Path("data/combined_80_20_v6/train.jsonl")
OUT = Path("data/cot_fewshot_exemplars_mined.json")
CANDIDATES_PER_KIND = 40
BATCH = 8

# PROMPT BUDGET (added 2026-09-14b after the first mined set was measured).
#
# THE DEFECT THIS EXISTS TO PREVENT. Selection used to be "shortest verified
# trace", scored as `sum(len(s) for s in steps)` -- which ignores `context`
# entirely. Two of the four exemplars it picked carried 20,144- and
# 17,936-character contexts, so the rendered preamble came to 9,393 tokens
# against an inference-time truncation limit of 4,096 with
# `truncation_side="left"`. Consequences, measured on the mined file:
#   * 100% of few-shot prompts exceeded 4,096 on BOTH benchmarks;
#   * the median item kept only 3,964 (TimeBench) / 2,554 (TIME) preamble
#     tokens -- a mid-sentence fragment of the LAST exemplar, never four;
#   * left truncation cuts from the front, so COT_SYSTEM_PROMPT and the chat
#     template's opening were removed from EVERY few-shot item, while the
#     zero-shot CoT arm kept them -- silently confounding the one comparison
#     the arm exists to make.
#
# Budget arithmetic against the 4,096 limit: system prompt + chat scaffolding
# ~90 tokens, and the target block reaches ~2,135 tokens at TIME's p90. A
# 1,600-token preamble therefore leaves every item up to ~2,400 target tokens
# intact. Per-exemplar cap keeps any single example from eating the budget.
MAX_EXEMPLAR_TOKENS = 450
MAX_PREAMBLE_TOKENS = 1600
TRUNCATION_LIMIT = 4096  # src/models/inference.py tokenizer max_length
ORDER_RE = re.compile(r"^[A-D](,[A-D])+$")
DURATION_RE = re.compile(r"\b(day|days|week|weeks|month|months|year|years|decade|hour|hours|minute|minutes)\b", re.I)


def classify(row: dict) -> str | None:
    tgts = [str(t).strip() for t in (row.get("targets") or []) if str(t).strip()]
    if not tgts:
        return None
    q = row.get("question") or ""
    if ORDER_RE.match(tgts[0]):
        return "ordering"
    if len(tgts[0]) == 1 and tgts[0] in "ABCD":
        return None  # letter-MCQ: pool rows carry no choices list; skip
    if DURATION_RE.search(tgts[0]):
        return "duration_phrase"
    return "phrase_qa"


_TOK = None


def _tokenizer():
    """The evaluated model's own tokenizer — budgets must be counted in the
    units the runner truncates in, not in characters."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        from scripts.run_baselines import MODEL_RUNTIME_CONFIG
        _TOK = AutoTokenizer.from_pretrained(
            str(Path(MODEL_RUNTIME_CONFIG["llama"]["model_dir"]).resolve()))
    return _TOK


def n_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


def block_tokens(exemplar: dict) -> int:
    """Rendered size of one exemplar, measured with the SAME renderer the
    runner uses (`_format_cot_exemplar`), so the miner and the runner can
    never drift apart about what an exemplar costs."""
    return n_tokens(_format_cot_exemplar(exemplar))


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already exists; delete it to re-mine"); return
    random.seed(42)
    buckets: dict[str, list[dict]] = {}
    with open(POOL) as fh:
        for line in fh:
            row = json.loads(line)
            kind = classify(row)
            if kind:
                buckets.setdefault(kind, []).append(row)
    quota = {"ordering": 1, "duration_phrase": 1, "phrase_qa": 2}

    # Drop oversized candidates BEFORE generation: an exemplar whose context
    # alone blows the per-exemplar budget can never be selected, so spending
    # GPU on it is waste. ~120 tokens are reserved for the steps + ANSWER line
    # that generation will add.
    STEPS_ALLOWANCE = 120
    dropped: dict[str, int] = {}
    for kind in list(buckets):
        kept = []
        for row in buckets[kind]:
            stub = {"context": row.get("context"), "question": row.get("question"),
                    "choices": None, "steps": [], "answer": ""}
            if block_tokens(stub) + STEPS_ALLOWANCE <= MAX_EXEMPLAR_TOKENS:
                kept.append(row)
        dropped[kind] = len(buckets[kind]) - len(kept)
        buckets[kind] = kept
    print(f"oversized candidates dropped (> {MAX_EXEMPLAR_TOKENS} tok): {dropped}")

    cands: list[tuple[str, dict]] = []
    for kind, n in quota.items():
        pool = buckets.get(kind, [])
        if len(pool) < n:
            raise SystemExit(
                f"pool too small for {kind} after the {MAX_EXEMPLAR_TOKENS}-token "
                f"budget filter: {len(pool)}. Raise MAX_EXEMPLAR_TOKENS only if "
                f"the preamble still fits MAX_PREAMBLE_TOKENS.")
        for row in random.sample(pool, min(CANDIDATES_PER_KIND, len(pool))):
            cands.append((kind, row))
    print(f"candidates: {len(cands)} across {sorted(buckets)}")

    examples = []
    for i, (kind, row) in enumerate(cands):
        examples.append((kind, row, TemporalExample(
            id=f"mine-{i}", question=row.get("question") or "",
            answer=(row.get("targets") or [""])[0], context=row.get("context"),
            choices=None, temporal_type=None, task="Order_Compare", source="pool",
        )))
    prompts = [_build_cot_prompt(ex) for _, _, ex in examples]

    print("loading llama (7B-class GPU job, minutes)...")
    from scripts.run_baselines import MODEL_RUNTIME_CONFIG
    cfg = MODEL_RUNTIME_CONFIG["llama"]
    inf = SLMInference(
        model_key=cfg["inference_key"],
        model_dir=str(Path(cfg["model_dir"]).resolve()),
        system_prompt=COT_SYSTEM_PROMPT,
    )
    raws: list[str] = []
    for s in range(0, len(prompts), BATCH):
        raws.extend(inf.batch_generate(prompts[s:s + BATCH], max_new_tokens=256, temperature=0.0, do_sample=False))
        print(f"  generated {min(s + BATCH, len(prompts))}/{len(prompts)}")

    by_kind: dict[str, list[dict]] = {}
    for (kind, row, ex), raw in zip(examples, raws):
        pred = _postprocess_prediction(raw, None)
        golds = [str(x) for x in (row.get("targets") or [])] + [str(x) for x in (row.get("final_answers") or [])]
        golds = [g for g in golds if g.strip()]
        if not (pred and matches_any(pred, golds)):
            continue
        steps = [l.strip() for l in raw.splitlines() if l.strip().startswith("Step ")]
        if len(steps) < 1 or len(steps) > 4:
            continue
        cand = {
            "kind": kind, "context": row.get("context"), "question": row.get("question"),
            "choices": None, "steps": steps, "answer": pred,
        }
        # Exact rendered size now that the trace exists. The pre-generation
        # filter used an allowance; this is the real measurement.
        if block_tokens(cand) > MAX_EXEMPLAR_TOKENS:
            continue
        by_kind.setdefault(kind, []).append(cand)
    picked: list[dict] = []
    kinds_used: dict[str, int] = {}
    # Priority order with spill-over (declared rule, not cherry-picking):
    # take the kind quota if verified-correct traces exist; unfilled slots
    # spill to phrase_qa (the generic kind). Duration-phrase verification
    # is genuinely hard under CoT (D59/D64: ~25% phrase-bucket rate), so a
    # 0-correct duration draw degrades to a third phrase exemplar rather
    # than failing the arm. Selection within a kind: shortest-trace-first.
    quota = {"ordering": 1, "duration_phrase": 1, "phrase_qa": 2}
    for kind in ("ordering", "duration_phrase", "phrase_qa"):
        avail = sorted(by_kind.get(kind, []), key=lambda e: sum(len(s) for s in e["steps"]))
        take = min(quota[kind], len(avail))
        picked.extend(avail[:take])
        kinds_used[kind] = take
    spill_quota = 4 - len(picked)
    if spill_quota > 0:
        phrase_left = sorted(by_kind.get("phrase_qa", []), key=lambda e: sum(len(s) for s in e["steps"]))[kinds_used.get("phrase_qa", 0):]
        leftover = [e for k in ("ordering", "duration_phrase") for e in sorted(by_kind.get(k, []), key=lambda e: sum(len(s) for s in e["steps"]))[quota[k]:]]
        fill = (phrase_left + leftover)[:spill_quota]
        picked.extend(fill)
        kinds_used["phrase_qa"] = kinds_used.get("phrase_qa", 0) + sum(1 for e in fill if e["kind"] == "phrase_qa")
    if len(picked) < 4:
        raise SystemExit(f"mining failed: only {len(picked)}/4 verified-correct exemplars (kinds: {kinds_used})")

    # HARD BUDGET CHECK. Measure the preamble exactly as the runner renders it
    # (scripts/run_baselines.py::_build_cot_fewshot_prompt) and refuse to write
    # a set that cannot survive inference-time truncation. Without this the
    # arm looks fine everywhere except in the tokenizer.
    preamble = ("Worked examples:\n\n"
                + "\n\n".join(_format_cot_exemplar(e) for e in picked)
                + "\n\nNow solve this one, the same way.\n\n")
    pre_n = n_tokens(preamble)
    print(f"\nrendered preamble: {pre_n} tokens "
          f"(cap {MAX_PREAMBLE_TOKENS}, truncation limit {TRUNCATION_LIMIT})")
    for e in picked:
        print(f"    [{e['kind']:>16}] {block_tokens(e):5,} tok")
    if pre_n > MAX_PREAMBLE_TOKENS:
        raise SystemExit(
            f"REFUSING TO WRITE: preamble is {pre_n} tokens, over the "
            f"{MAX_PREAMBLE_TOKENS} cap. At inference the prompt is truncated to "
            f"{TRUNCATION_LIMIT} tokens from the LEFT, so an oversized preamble is "
            f"silently cut — taking COT_SYSTEM_PROMPT and the leading exemplars "
            f"with it, and leaving a mid-sentence fragment instead of k worked "
            f"examples. This is the defect found on 2026-09-14.")
    OUT.write_text(json.dumps({
        "version": 2,
        "description": "SELF-MINED few-shot CoT exemplars (D63): llama zero_shot_reasoning "
                       "generations on TRAINING-POOL questions (v6 train, never benchmarks), "
                       "answer-verified with the frozen scorer, shortest-trace-first per kind. "
                       "Same traces the v9/STaR pipeline would train on -- placed in-prompt "
                       "instead of in-weights. kinds: 2x phrase_qa, 1x ordering, 1x duration_phrase.",
        "exemplars": picked,
    }, indent=2))
    print(f"wrote {OUT} with {len(picked)} exemplars:")
    for e in picked:
        print(f"  [{e['kind']}] steps={len(e['steps'])} answer={e['answer']!r}")


if __name__ == "__main__":
    main()
