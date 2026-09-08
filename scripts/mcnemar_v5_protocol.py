"""Paired McNemar tests over the v5 scoring protocol. Zero GPU cost.

`rescore_v5_protocol.py` compares arms with an unpaired two-proportion
z-test.  Every arm is evaluated on the SAME items, so that test throws away
the pairing and is conservative: items both arms get right, or both get
wrong, carry no information about which arm is better, yet they inflate the
unpaired standard error.

This scores each arm exactly as `rescore_v5_protocol.rescore` does -- same
`_postprocess_prediction`, same `matches_any` -- but keeps per-example
correctness keyed by example id, then runs McNemar over the discordant
pairs only.

Usage: ./venv/bin/python scripts/mcnemar_v5_protocol.py [--benchmark timebench]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rescore_v5_protocol import ARMS, choice_map  # noqa: E402
from scripts.run_baselines import _postprocess_prediction  # noqa: E402
from src.evaluation.date_equivalence import matches_any  # noqa: E402


def correct_by_id(path: Path, choices_by_id) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        refs = r["reference"] if isinstance(r["reference"], list) else [r["reference"]]
        raw = r.get("raw_prediction")
        pred = _postprocess_prediction(raw, choices_by_id.get(r["id"])) \
            if raw is not None else r["prediction"]
        out[r["id"]] = bool(matches_any(pred, refs))
    return out


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Return (z, two-sided p). Exact binomial when discordants are few."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    z = (b - c) / math.sqrt(n)
    if n < 25:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return z, min(1.0, 2 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n          # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2))
    return z, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/benchmarks")
    ap.add_argument("--benchmark", default="timebench", choices=["timebench", "time"])
    args = ap.parse_args()

    choices_by_id = choice_map(args.benchmark, args.data_dir)
    arms: dict[str, dict[str, bool]] = {}
    for name, rel in ARMS[args.benchmark]:
        p = PROJECT_ROOT / rel
        if not p.exists():
            print(f"skip {name}: missing {rel}")
            continue
        arms[name] = correct_by_id(p, choices_by_id)
        print(f"loaded {name:10} n={len(arms[name]):>7}  acc={100*sum(arms[name].values())/len(arms[name]):.2f}%")

    full = max(len(v) for v in arms.values())
    partial = [k for k, v in arms.items() if len(v) < full]
    for k in partial:
        print(f"EXCLUDED {k}: {len(arms[k])}/{full} rows -- leg still running, "
              f"and predictions are prompt-length-sorted, so the prefix is not a random sample")
        arms.pop(k)

    names = list(arms)
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"\n=== {args.benchmark}: paired McNemar over {len(ids)} common ids ===")
    print(f"{'A':10} {'B':10} {'A-only':>7} {'B-only':>7} {'disc':>7} "
          f"{'diff pp':>8} {'z':>7} {'p':>10}")
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            A, B = arms[a], arms[bname]
            b = sum(1 for k in ids if A[k] and not B[k])
            c = sum(1 for k in ids if B[k] and not A[k])
            z, p = mcnemar(b, c)
            diff = 100 * (b - c) / len(ids)
            print(f"{a:10} {bname:10} {b:>7} {c:>7} {b+c:>7} "
                  f"{diff:>+8.2f} {z:>+7.2f} {p:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
