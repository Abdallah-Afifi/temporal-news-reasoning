#!/usr/bin/env python3
"""Build the v3 training data: rebalanced mixture + rehearsal mix +
dialogue/reasoning augmentations + optional CoT overlay.

Changes vs v2 (one bundle, per docs/v3_plan.md):
- synthetic families resized and rebalanced (~15-20% of loaded pool
  instead of v2's 33.5%) and category-weighted toward the measured
  deficits (dialogue > reasoning > nli/mcq > arithmetic)
- + dialogue-format augmentation (AUG_DIALOG)
- + verified two-hop reasoning augmentation (AUG_REASON)
- + rehearsal mix from databricks-dolly-15k (decontaminated)
- optional overlay of verified CoT traces (data/cot/*.jsonl from
  scripts/generate_cot_data.py), joined by question text
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v2_training_data import (
    benchmark_questions,
    make_arithmetic,
    make_mcq,
    make_nli,
    norm,
)

DIALOG_INSTRUCTION = "How should Alex reply? Answer directly."
REHEARSAL_DATASET = "data/rehearsal/databricks-dolly-15k.jsonl"


def first_sentence(text: str, cap: int = 220) -> str:
    t = re.split(r"(?<=[.!?])\s", str(text).strip())[0]
    return t[:cap]


def make_dialogue(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Two-speaker temporal QA: fact stated in dialogue, question asked,
    target = the direct answer. Trains dialogue-format extraction."""
    usable = [
        r
        for r in records
        if r.get("source_dataset") == "TimeQA"
        and r.get("context")
        and r.get("question")
        and r.get("answer")
        and len(str(r["answer"])) < 60
    ]
    rng.shuffle(usable)
    out = []
    for r in usable[:n]:
        fact = first_sentence(r["context"])
        if len(fact) < 40:
            continue
        q = str(r["question"]).strip().rstrip("?")
        a = str(r["answer"]).strip().rstrip(".")
        question = (
            f"In a conversation, Alex says: \"{fact}\"\n"
            f"Sam then asks: \"{q}?\"\n{DIALOG_INSTRUCTION}"
        )
        out.append(
            {
                "source_dataset": "AUG_DIALOG",
                "question": question,
                "context": "",
                "targets": [a],
                "source": "augmented",
            }
        )
    return out


YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20[0-4]\d)\b")


def make_reasoning(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Verified two-hop questions: hop 1 = answer the factual question
    (a year, from context), hop 2 = shift by k years. Target verified by
    construction from the record's gold year."""
    usable = []
    for r in records:
        m = YEAR_RE.findall(str(r.get("answer") or ""))
        if len(m) == 1 and r.get("question"):
            usable.append((r, int(m[0])))
    rng.shuffle(usable)
    out = []
    for r, year in usable[:n]:
        q = str(r["question"]).strip().rstrip("?")
        k = rng.randint(2, 30)
        if rng.random() < 0.5:
            hop = f"{k} years after that"
            target = year + k
        else:
            hop = f"{k} years before that"
            target = year - k
        if target < 1400 or target > 2100:
            continue
        question = (
            f"{q}? Then, considering that answer: in which year was it {hop}?"
        )
        out.append(
            {
                "source_dataset": "AUG_REASON",
                "question": question,
                "context": r.get("context", ""),
                "targets": [str(target)],
                "source": "augmented",
            }
        )
    return out


def load_pool(v2_dir: str) -> list[dict]:
    """Real records from the v2 pool (already decontaminated)."""
    out = []
    for line in open(Path(v2_dir) / "train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("source_dataset") in ("TimeQA", "TLQA"):
            out.append(r)
    return out


def norm_record(r: dict) -> dict:
    """Loader-compatible normalized view keeping every gold answer.

    TLQA and TimeQA are list-answer tasks ("List all positions X held from
    2010 to 2013"), so keeping only ``answers[0]`` trains the model to name
    one item where the gold names ~2.9 — it under-answers at inference and
    is scored wrong against the full list. ``answers`` carries the whole
    list; ``answer`` stays as the joined string for the augmentation passes.
    """
    if r.get("source_dataset") == "TimeQA":
        ans = r.get("targets") or []
        parts = [str(a).strip() for a in ans if str(a).strip()]
        return {
            "source_dataset": "TimeQA",
            "question": r.get("question", ""),
            "context": r.get("context", ""),
            "answer": "; ".join(parts),
            "answers": parts,
        }
    answers = r.get("final_answers", r.get("answers", []))
    parts = [str(a).strip() for a in answers if str(a).strip()]
    return {
        "source_dataset": "TLQA",
        "question": r.get("question", ""),
        "context": "",
        "answer": "; ".join(parts),
        "answers": parts,
    }


def rehearsal_slice(path: str, n: int, rng: random.Random,
                    bench_qs: set[str]) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = []
    for r in rows:
        instr = str(r.get("instruction", "")).strip()
        resp = str(r.get("response", "")).strip()
        ctx = str(r.get("context", "")).strip()
        if not instr or not resp:
            continue
        if len(resp) > 280 or len(instr) > 400 or "```" in resp:
            continue
        if norm(instr) in bench_qs:
            continue
        q = f"{instr}\n\n{ctx}" if ctx else instr
        ok.append(
            {
                "source_dataset": "REHEARSAL",
                "question": q,
                "context": "",
                "targets": [resp],
                "source": "databricks/databricks-dolly-15k",
            }
        )
    rng.shuffle(ok)
    return ok[:n]


def overlay_cot(records: list[dict], cot_path: str) -> tuple[list[dict], int]:
    by_q = {}
    for line in open(cot_path, encoding="utf-8"):
        c = json.loads(line)
        if c.get("verified") and c.get("cot"):
            by_q[norm(str(c.get("question") or ""))] = c["cot"]
    n_overlaid = 0
    for r in records:
        key = norm(str(r.get("question") or ""))
        cot = by_q.get(key)
        if cot:
            r["targets"] = [cot.rstrip()]
            r["cot"] = True
            n_overlaid += 1
    return records, n_overlaid


def stratified_split(records: list[dict], val_frac: float,
                     rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_src: dict[str, list[dict]] = {}
    for r in records:
        by_src.setdefault(r["source_dataset"], []).append(r)
    train, val = [], []
    for src, rows in by_src.items():
        rng.shuffle(rows)
        cut = int(len(rows) * val_frac)
        val.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-dir", default="data/combined_80_20_v2")
    ap.add_argument("--out", default="data/combined_80_20_v3")
    ap.add_argument("--mcq", type=int, default=500)
    ap.add_argument("--arith", type=int, default=400)
    ap.add_argument("--nli", type=int, default=500)
    ap.add_argument("--dialog", type=int, default=800)
    ap.add_argument("--reason", type=int, default=700)
    ap.add_argument("--rehearsal", type=int, default=2600)
    ap.add_argument("--cot-file", default="",
                    help="optional verified-CoT jsonl to overlay")
    ap.add_argument("--cot-sources", default="TimeQA,TLQA",
                    help="comma list of source_dataset values to overlay")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bench_qs = benchmark_questions()
    pool = load_pool(args.v2_dir)
    norm_pool = [norm_record(r) for r in pool]
    print(f"real pool (decontaminated v2): {len(norm_pool)}")

    aug = []
    aug += make_mcq(norm_pool, args.mcq, rng)
    aug += make_arithmetic(args.arith, rng)
    aug += make_nli(norm_pool, args.nli, rng)
    aug += make_dialogue(norm_pool, args.dialog, rng)
    aug += make_reasoning(norm_pool, args.reason, rng)
    for a in aug:
        if norm(str(a.get("question") or "")) in bench_qs:
            raise SystemExit("FATAL: augmentation collided with benchmark")
    print(f"synthetic: {Counter(a['source_dataset'] for a in aug)}")

    reh = rehearsal_slice(REHEARSAL_DATASET, args.rehearsal, rng, bench_qs)
    print(f"rehearsal slice: {len(reh)} (decontaminated)")

    real_records = [
        {
            "source_dataset": r["source_dataset"],
            "question": r["question"],
            "context": r["context"],
            # Every gold answer, not just the first — see norm_record.
            "targets": list(r["answers"]),
            # TLQA's training-loader branch reads final_answers first; emit
            # both so the row survives regardless of which key is read.
            "final_answers": list(r["answers"]),
            "source": "v2_pool",
        }
        for r in norm_pool
        if r["answers"]
    ]
    all_records = real_records + aug + reh

    n_overlaid = 0
    if args.cot_file:
        srcs = {s.strip() for s in args.cot_sources.split(",")}
        eligible = [r for r in all_records if r["source_dataset"] in srcs]
        eligible, n_overlaid = overlay_cot(eligible, args.cot_file)
        by_q = {id(r): r for r in eligible}
        all_records = [by_q.get(id(r), r) for r in all_records]
        print(f"CoT overlaid: {n_overlaid}/{len(eligible)} eligible")

    train, val = stratified_split(all_records, args.val_frac, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rows):
        c = Counter(r["source_dataset"] for r in rows)
        syn = sum(v for k, v in c.items() if k.startswith("AUG_"))
        return {
            "n": len(rows),
            "synthetic_pct": round(100 * syn / len(rows), 1),
            "by_source": dict(c),
        }

    stats = {"train": mix(train), "val": mix(val),
             "cot_overlaid": n_overlaid, "seed": args.seed,
             "sizes": vars(args)}
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
