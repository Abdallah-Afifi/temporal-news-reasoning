"""Carve a fixed, stratified HPO dev subset out of each benchmark's test pool.

None of TIME, TimeBench or TRAM ships a train/dev split (TRAM's only extra
files are few-shot "shots"). Selecting hyperparameters by TEST accuracy would
leak the test set into the model choice, and any "beats zero-shot" claim made
afterwards would be invalid. So: a subset is drawn ONCE here, every HPO trial
is scored on it and nothing else, and every final number -- zero-shot and
every fine-tuned arm alike -- is reported on test-minus-dev
(rescore_v5_protocol.py --exclude-ids data/hpo_dev/dev_ids.json).

Stratified by the same label the scorer's per-category table uses
(temporal_type or task), plus TIME's `Setting` (gold vs retrieved context,
which moves accuracy by ~23pp), with PROPORTIONAL allocation so the dev
micro-average estimates the test micro-average (the headline v5_pct).

Deterministic: ids are sorted within each stratum and sampled with a
stratum-keyed seed, so the split does not depend on file or load order.

Usage:
    venv/bin/python scripts/make_hpo_dev_split.py            # writes data/hpo_dev/
    venv/bin/python scripts/make_hpo_dev_split.py --check    # verify, no write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import BenchmarkLoader  # noqa: E402

DEFAULT_N = {"time": 5000, "timebench": 2500, "tram": 10000}
SEED = "hpo-dev-v1"
OUT_DIR = PROJECT_ROOT / "data" / "hpo_dev"


def stratum(bench: str, ex) -> str:
    cat = str(ex.temporal_type or ex.task or "unknown")
    if bench == "time":
        return f"{cat}|{(ex.metadata or {}).get('Setting') or ''}"
    return cat


def proportional(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n across strata, proportional to size."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    left = n - sum(alloc.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[:left]:
        alloc[k] += 1
    return alloc


def build(bench: str, n: int, data_dir: str) -> tuple[list[str], dict]:
    examples = BenchmarkLoader(data_dir=data_dir).load(bench)
    ids = [ex.id for ex in examples]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"FATAL: {bench} has duplicate example ids; a dev "
                         f"split keyed on id would be ambiguous.")
    groups: dict[str, list[str]] = defaultdict(list)
    for ex in examples:
        groups[stratum(bench, ex)].append(ex.id)
    alloc = proportional({k: len(v) for k, v in groups.items()}, n)
    chosen: list[str] = []
    for k in sorted(groups):
        pool = sorted(groups[k])
        chosen += random.Random(f"{SEED}|{bench}|{k}").sample(pool, alloc[k])
    chosen.sort()
    meta = {"pool": len(ids), "dev": len(chosen),
            "test_minus_dev": len(ids) - len(chosen),
            "strata": {k: {"pool": len(groups[k]), "dev": alloc[k]}
                       for k in sorted(groups)},
            "sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest()}
    return chosen, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "benchmarks"))
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare with the stored split")
    for b, n in DEFAULT_N.items():
        ap.add_argument(f"--n-{b}", type=int, default=n)
    args = ap.parse_args()

    split, manifest = {}, {"seed": SEED, "benchmarks": {}}
    for b in DEFAULT_N:
        ids, meta = build(b, getattr(args, f"n_{b}"), args.data_dir)
        split[b] = ids
        manifest["benchmarks"][b] = meta
        print(f"{b:10s} pool={meta['pool']:>8,} dev={meta['dev']:>6,} "
              f"test-minus-dev={meta['test_minus_dev']:>8,} "
              f"strata={len(meta['strata'])}")

    ids_path, man_path = OUT_DIR / "dev_ids.json", OUT_DIR / "MANIFEST.json"
    if args.check:
        stored = json.loads(ids_path.read_text())
        same = all(sorted(stored.get(b, [])) == split[b] for b in split)
        print("split matches stored copy" if same else "SPLIT DIFFERS FROM STORED COPY")
        return 0 if same else 1
    if ids_path.exists():
        stored = json.loads(ids_path.read_text())
        if any(sorted(stored.get(b, [])) != split[b] for b in split):
            raise SystemExit(
                f"FATAL: {ids_path} exists and differs from a fresh build. The "
                f"dev split must never change once HPO trials have been scored "
                f"on it. Move it aside deliberately if you really mean to "
                f"re-split (and then discard every trial scored on the old one).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps(split))
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {ids_path} and {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
