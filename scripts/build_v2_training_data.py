#!/usr/bin/env python3
"""Build the v2 leakage-free, mixed-format training data (Path B groundwork).

Phases
------
1. Pool the existing 80/20 split (train + val).
2. Remove every record whose question text appears in ANY evaluation
   benchmark (TIME, TimeBench full zip incl. Hypothesis-keyed tasks, TRAM)
   — closes the leakage found by scripts/audit_train_benchmark_overlap.py
   at the source.
3. Augment with the formats the diagnostics showed the adapter lost or
   never had:
     - AUG_MCQ:  pool items converted to multiple-choice (options embedded
       in the question, mirroring the evaluation prompt format)
     - AUG_ARITH: templated date/duration arithmetic with computed golds
     - AUG_NLI:   premise/hypothesis items built from TimeQA contexts
       (Entailment / Contradiction / Neutral), mirroring TimeBench NLI
4. Stratified 80/20 re-split (by source x format, seeded) and write
   ``data/combined_80_20_v2/{train,val}.jsonl`` plus a manifest.

CPU-only, deterministic (seed 42). Original split is left untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

SEED = 42
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def norm(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Benchmark question sets (leakage filter)
# ---------------------------------------------------------------------------

def benchmark_questions() -> set[str]:
    from src.data.data_loader import BenchmarkLoader

    qs: set[str] = set()
    loader = BenchmarkLoader(str(_REPO_ROOT / "data" / "benchmarks"))
    for bench in ("time", "timebench", "tram"):
        try:
            examples = loader.load(bench)
        except Exception as exc:  # pragma: no cover - data availability
            print(f"WARNING: could not load {bench}: {exc}")
            continue
        for e in examples:
            q = norm(e.question)
            qs.add(q)
            if "\n" in e.question:
                qs.add(norm(e.question.split("\n", 1)[0]))
            # hypothesis-keyed tasks (tracie/nli) store the hypothesis in
            # the question field after loading — already covered above.
        print(f"leakage filter: {bench} -> {len(examples):,} questions")
    return qs


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

MCQ_TEMPLATE = "{q}\nChoices:\n{opts}"


def make_mcq(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Convert pool QA items into MCQ items (options embedded in question)."""
    usable = [r for r in records
              if r.get("answer") and len(str(r["answer"])) < 80
              and r.get("source_dataset") in ("TimeQA", "TLQA")]
    rng.shuffle(usable)
    out = []
    for r in usable[: n]:
        gold = str(r["answer"])
        pool = [str(o["answer"]) for o in usable if o is not r and str(o["answer"]) != gold]
        distractors = rng.sample(pool, 3) if len(pool) >= 3 else None
        if not distractors:
            continue
        options = distractors + [gold]
        rng.shuffle(options)
        letter = "ABCD"[options.index(gold)]
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
        out.append({
            "source_dataset": "AUG_MCQ",
            "question": MCQ_TEMPLATE.format(q=r["question"], opts=opts),
            "context": r.get("context", ""),
            "targets": [gold, letter],
            "source": "augmented",
        })
    return out


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def fmt(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def make_arithmetic(n: int, rng: random.Random) -> list[dict]:
    out = []
    for _ in range(n):
        kind = rng.choice(["after_years", "after_months", "before_months",
                           "duration_days", "duration_months"])
        base = date(rng.randint(1900, 2024), rng.randint(1, 12), rng.randint(1, 28))
        if kind == "after_years":
            k = rng.randint(2, 40)
            q = f"What is the date {k} years after {fmt(base)}?"
            a = fmt(add_months(base, 12 * k))
        elif kind == "after_months":
            k = rng.randint(2, 30)
            q = f"What is the date {k} months after {fmt(base)}?"
            a = fmt(add_months(base, k))
        elif kind == "before_months":
            k = rng.randint(2, 30)
            q = f"What is the date {k} months before {fmt(base)}?"
            a = fmt(add_months(base, -k))
        elif kind == "duration_days":
            end = base + timedelta(days=rng.randint(2, 900))
            q = f"How many days are there between {fmt(base)} and {fmt(end)}?"
            a = str((end - base).days)
        else:
            end = add_months(base, rng.randint(2, 36))
            months = (end.year - base.year) * 12 + (end.month - base.month)
            q = f"How many months are there between {fmt(base)} and {fmt(end)}?"
            a = str(months)
        out.append({
            "source_dataset": "AUG_ARITH",
            "question": q, "context": "", "targets": [a], "source": "augmented",
        })
    return out


NLI_INSTRUCTION = ("Read the premise. Does the premise entail the hypothesis? "
                   "Answer with exactly one word: Entailment, Contradiction, or Neutral.")


def make_nli(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Premise/hypothesis pairs from TimeQA contexts.

    - Entailment: a statement binding the question's answer to the subject.
    - Contradiction: the same statement with a corrupted entity/date.
    - Neutral: a statement about a different record's entity.
    """
    usable = [r for r in records
              if r.get("source_dataset") == "TimeQA" and r.get("context")
              and r.get("answer") and r.get("question")]
    rng.shuffle(usable)
    out, i = [], 0
    while len(out) < n and i < len(usable) - 1:
        r = usable[i]
        other = usable[(i + rng.randint(1, 50)) % len(usable)]
        i += 1
        ctx = str(r["context"])[:1200]
        if len(ctx) < 80:
            continue
        subj = str(r.get("subject") or "").strip().rstrip(".") or "the subject"
        ans = str(r["answer"]).strip().rstrip(".")
        q = r["question"].strip().rstrip("?")
        label = rng.choice(["Entailment", "Contradiction", "Neutral"])
        if label == "Entailment":
            hyp = f"{subj} is associated with {ans}."
        elif label == "Contradiction":
            wrong = str(other["answer"]).strip().rstrip(".") if other.get("answer") else "an unrelated entity"
            hyp = f"{subj} is associated with {wrong}."
        else:
            osubj = str(other.get("subject") or "another subject").strip().rstrip(".")
            oans = str(other["answer"]).strip().rstrip(".") if other.get("answer") else "something"
            hyp = f"{osubj} is associated with {oans}."
        out.append({
            "source_dataset": "AUG_NLI",
            "question": f"Hypothesis: {hyp} {NLI_INSTRUCTION}",
            "context": ctx, "targets": [label], "source": "augmented",
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/combined_80_20_v2")
    ap.add_argument("--mcq", type=int, default=2500)
    ap.add_argument("--arith", type=int, default=2000)
    ap.add_argument("--nli", type=int, default=2000)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    rng = random.Random(SEED)

    # 1. pool + leakage filter
    bench_qs = benchmark_questions()
    pool, excluded = [], 0
    for split in ("train", "val"):
        for line in open(_REPO_ROOT / "data/combined_80_20_split" / f"{split}.jsonl"):
            r = json.loads(line)
            q = str(r.get("question", ""))
            if norm(q) in bench_qs or norm(q.split("\n", 1)[0]) in bench_qs:
                excluded += 1
                continue
            pool.append(r)
    print(f"pool after leakage filter: {len(pool):,} (excluded {excluded})")

    # normalised {question, context, answer} for augmentation
    normed = []
    for r in pool:
        ans = r.get("targets", r.get("final_answers", r.get("answers", "")))
        if isinstance(ans, list):
            ans = "; ".join(str(a).strip() for a in ans if str(a).strip())
        if not str(ans).strip():
            continue
        normed.append({**r, "answer": str(ans)})

    # 2. augmentations
    aug = []
    aug += make_mcq(normed, args.mcq, rng)
    aug += make_arithmetic(args.arith, rng)
    aug += make_nli(normed, args.nli, rng)
    print(f"augmented: {len(aug):,} "
          f"({Counter(a['source_dataset'] for a in aug)})")

    everything = pool + aug

    # 3. stratified 80/20 split by source_dataset
    by_src = defaultdict(list)
    for r in everything:
        by_src[r.get("source_dataset", "?")].append(r)
    train, val = [], []
    for src, rows in sorted(by_src.items()):
        rng.shuffle(rows)
        k = max(1, int(len(rows) * args.val_ratio)) if len(rows) >= 5 else 0
        val += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = _REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": SEED,
        "pool_records_after_leakage_filter": len(pool),
        "excluded_leaked_records": excluded,
        "augmented": dict(Counter(a["source_dataset"] for a in aug)),
        "train": len(train),
        "val": len(val),
        "train_by_source": dict(Counter(r.get("source_dataset", "?") for r in train)),
        "val_by_source": dict(Counter(r.get("source_dataset", "?") for r in val)),
        "notes": [
            "leakage filter: normalized question text vs TIME + TimeBench(full zip) + TRAM",
            "rehearsal/general-instruction mix NOT included (requires external dataset)",
            "aug formats mirror the evaluation prompt structures (MCQ choices embedded; NLI hypothesis+instruction)",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2)[:800])
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
