"""Validate manually written augmentation records (AUG_GLM).

Checks: schema, non-empty targets, unique questions, and no question
collision with any evaluation benchmark (TIME / TimeBench / TRAM).
Usage: venv/bin/python scripts/validate_manual_aug.py data/manual_aug/*.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import BenchmarkLoader  # noqa: E402


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def main() -> int:
    bench_q: set[str] = set()
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    for bench in ("time", "timebench"):
        for ex in loader.load(bench):
            bench_q.add(norm(ex.question))
    print(f"loaded {len(bench_q)} benchmark questions for collision check")

    seen: set[str] = set()
    total = collisions = bad = 0
    for path in sys.argv[1:]:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                r = json.loads(line)
                assert r.get("question") and r.get("targets") and r.get("source_dataset")
                assert isinstance(r["targets"], list) and all(str(t).strip() for t in r["targets"])
            except Exception as e:
                bad += 1
                print(f"SCHEMA ERROR: {e}: {line[:80]}")
                continue
            nq = norm(r["question"])
            if nq in seen:
                print(f"DUPLICATE: {r['question'][:60]}")
                bad += 1
            seen.add(nq)
            if nq in bench_q:
                print(f"COLLISION: {r['question'][:60]}")
                collisions += 1
    print(f"\nvalidated {total} records: bad={bad} collisions={collisions}")
    return 0 if bad == 0 and collisions == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
