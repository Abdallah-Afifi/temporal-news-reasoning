#!/usr/bin/env python3
"""Audit overlap between the combined 80/20 training split and benchmark data.

The merge script (src/data/merge_datasets_80_20.py) pooled ALL original
splits (train/dev/test) of TimeQA / Temprel / TLQA before splitting 80/20.
If the same questions later appear in an evaluation benchmark, reported
scores are contaminated. This script quantifies that risk.

Checks, per benchmark dataset found under --benchmarks-dir:
  - exact overlap: normalized question text appears in train/val
  - fuzzy overlap: question similarity >= threshold (difflib, O(n*m) —
    capped via --max-fuzzy-comparisons per benchmark for speed)

Usage (on the machine that holds the data), from the repo root::

    python scripts/audit_train_benchmark_overlap.py
    python scripts/audit_train_benchmark_overlap.py \
        --train data/combined_80_20_split/train.jsonl \
        --val data/combined_80_20_split/val.jsonl \
        --benchmarks-dir data/benchmarks --output results/overlap_audit.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

QUESTION_KEYS = ("question", "query", "input", "prompt")


def normalize_question(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Some archives ship JSONL under .json names
                f.seek(0)
                out = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return out
        if isinstance(data, list):
            records = [r for r in data if isinstance(r, dict)]
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            print("pandas is required for parquet benchmark files — skipping")
            return []
        records = pd.read_parquet(path).to_dict(orient="records")
    return records


def extract_questions(records: list[dict]) -> list[str]:
    questions: list[str] = []
    for rec in records:
        lowered = {k.lower(): v for k, v in rec.items()}
        for key in QUESTION_KEYS:
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                questions.append(value)
                # MCQ benchmarks (e.g. TIME) embed the options in the
                # question text; the first line is the comparable question.
                if "\n" in value:
                    first_line = value.split("\n", 1)[0].strip()
                    if first_line:
                        questions.append(first_line)
                break
    return questions


def load_split_questions(path: Path) -> set[str]:
    if not path.exists():
        print(f"WARNING: split file not found: {path}")
        return set()
    questions = extract_questions(read_records(path))
    return {normalize_question(q) for q in questions}


def find_benchmark_files(benchmarks_dir: Path) -> list[Path]:
    if not benchmarks_dir.exists():
        return []
    return sorted(
        p for p in benchmarks_dir.rglob("*")
        if p.suffix.lower() in {".json", ".jsonl", ".parquet"}
    )


def audit_file(
    bench_path: Path,
    train_qs: set[str],
    val_qs: set[str],
    fuzzy_threshold: float,
    max_fuzzy: int,
) -> dict:
    bench_questions = extract_questions(read_records(bench_path))
    bench_norm = [normalize_question(q) for q in bench_questions]

    exact_train = sum(1 for q in bench_norm if q in train_qs)
    exact_val = sum(1 for q in bench_norm if q in val_qs)

    # Fuzzy pass (capped): compare a SAMPLE of benchmark questions against a
    # sample of the training questions to estimate near-duplicate leakage.
    # Both sides are capped — the full cross product is quadratic and can
    # spin for hours on large benchmarks.
    train_sample = list(train_qs)[:max_fuzzy]
    bench_sample = bench_norm[:max_fuzzy]
    fuzzy_train = 0
    for q in bench_sample:
        if q in train_qs:
            continue
        if any(
            difflib.SequenceMatcher(None, q, t).ratio() >= fuzzy_threshold
            for t in train_sample
        ):
            fuzzy_train += 1

    total = len(bench_norm)
    return {
        "file": str(bench_path),
        "num_questions": total,
        "exact_overlap_with_train": exact_train,
        "exact_overlap_with_val": exact_val,
        f"fuzzy_overlap_with_train(>={fuzzy_threshold})": fuzzy_train,
        "fuzzy_train_sample_size": len(train_sample),
        "exact_overlap_with_train_pct": round(100 * exact_train / total, 2) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=_REPO_ROOT / "data/combined_80_20_split/train.jsonl")
    parser.add_argument("--val", type=Path,
                        default=_REPO_ROOT / "data/combined_80_20_split/val.jsonl")
    parser.add_argument("--benchmarks-dir", type=Path,
                        default=_REPO_ROOT / "data/benchmarks")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9)
    parser.add_argument("--max-fuzzy-comparisons", type=int, default=20000,
                        help="Cap on training questions used in the fuzzy pass.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save the JSON audit report.")
    args = parser.parse_args()

    train_qs = load_split_questions(args.train)
    val_qs = load_split_questions(args.val)
    print(f"Train questions: {len(train_qs):,} | Val questions: {len(val_qs):,}")

    bench_files = find_benchmark_files(args.benchmarks_dir)
    if not bench_files:
        print(
            f"\nNo benchmark files found under {args.benchmarks_dir}.\n"
            f"The benchmark directories are empty (data is gitignored). Run this "
            f"audit on the machine that holds the downloaded benchmark data, e.g.:\n"
            f"  python scripts/download_datasets.py\n"
            f"  python scripts/audit_train_benchmark_overlap.py"
        )
        sys.exit(0)

    results = []
    for bench_path in bench_files:
        result = audit_file(
            bench_path, train_qs, val_qs,
            args.fuzzy_threshold, args.max_fuzzy_comparisons,
        )
        results.append(result)
        print(
            f"\n{bench_path}\n"
            f"  questions: {result['num_questions']}\n"
            f"  exact overlap with train: {result['exact_overlap_with_train']} "
            f"({result['exact_overlap_with_train_pct']}%)\n"
            f"  exact overlap with val:   {result['exact_overlap_with_val']}\n"
            f"  fuzzy overlap with train: {result[f'fuzzy_overlap_with_train(>={args.fuzzy_threshold})']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nAudit report saved: {args.output}")


if __name__ == "__main__":
    main()
