#!/usr/bin/env python3
"""Seed a re-run directory with the predictions the postprocessor fix cannot change.

The MCQ letter-extraction fix (see tests/test_prediction_postprocessing.py)
altered _postprocess_prediction only on the path taken by items that carry
choices. A stored prediction is provably identical under the old and new code
when either:

  * the item has no choices — that path returns first_line in both versions; or
  * the stored prediction is not equal to any of the item's choices. Every
    branch that the fix touched returns a choice string, so a non-choice result
    proves none of them fired, in either version.

Everything else must be regenerated. Seeding a fresh results directory with
only the provably-unaffected rows lets run_baselines.py's normal resume logic
regenerate exactly the remainder — no sampling, no engine change, no change to
any number that survives.

Retained rows carry no ``raw_prediction`` field; rows written by the re-run do.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import BenchmarkLoader  # noqa: E402


def choices_by_id(benchmark: str, benchmark_dir: str) -> dict[str, list[str] | None]:
    loader = BenchmarkLoader(benchmark_dir)
    return {ex.id: ex.choices for ex in loader.load(benchmark)}


def is_unaffected(record: dict, choices: list[str] | None) -> bool:
    if not choices:
        return True
    prediction = (record.get("prediction") or "").strip()
    return not any(prediction == choice.strip() for choice in choices)


def seed_one(src: Path, dst: Path, choices: dict[str, list[str] | None]) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if is_unaffected(record, choices.get(record["id"])):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    return kept, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="results_rerun")
    ap.add_argument("--benchmark-dir", default="data/benchmarks")
    ap.add_argument("--arms", nargs="*", default=[
        "baseline/zero_shot_v2/llama", "baseline/zero_shot_v2/mistral",
        "finetuned/llama", "finetuned/mistral",
        "finetuned_v2/llama", "finetuned_v3/llama",
    ])
    args = ap.parse_args()

    cache: dict[str, dict[str, list[str] | None]] = {}
    grand_kept = grand_total = 0
    for arm in args.arms:
        for src in sorted((Path(args.results_root) / arm).glob("*/*/predictions.jsonl")):
            benchmark = src.parts[-3]
            if benchmark not in cache:
                cache[benchmark] = choices_by_id(benchmark, args.benchmark_dir)
            dst = Path(args.out_root) / src.relative_to(args.results_root)
            kept, total = seed_one(src, dst, cache[benchmark])
            grand_kept += kept
            grand_total += total
            print(f"{arm}/{benchmark:9s} kept {kept:6d} / {total:6d} "
                  f"({100 * kept / total:5.1f}%)  regenerate {total - kept:6d}")

    regen = grand_total - grand_kept
    print(f"\nTOTAL: keep {grand_kept} / {grand_total} ({100 * grand_kept / grand_total:.1f}%) "
          f"— regenerate {regen} ({100 * regen / grand_total:.1f}%)")


if __name__ == "__main__":
    main()
