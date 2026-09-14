"""Build the v6d mixture — v6's mixture + AUG_DURATION isolated.

THE ONE CHANGE vs v6
--------------------
Add the `AUG_DURATION` slice ("Which of these lasted longer?", built from
TLQA year spans) on top of v6's SHIPPED mixture — the generator is copied
verbatim from `build_v7_training_data.py`, where this slice was the only v7
change that paid: TIME `Duration_Compare` 33.1 (v6) -> 40.1 (v7 with the
slice) on 9,217 items ≈ +0.6pp overall (D56/D57).

Per `docs/v7_corrected_plan.md` "Explicitly out of scope", AUG_DURATION gets
its own single-variable arm. This is it. Control = v6.

DESIGN NOTE — why the core is LOADED, not rebuilt. The current
`build_v6_training_data.py` no longer reproduces the shipped v6 data: the
2026-09-07 audit extended its leakage guard to TRAM, so a re-run filters
~782 different rows (14,854 vs 14,850). Rebuilding would silently change the
control. This script therefore reads `data/combined_80_20_v6/{train,val}.jsonl`
verbatim and appends ONLY the new slice — v6's 14,850 rows and their
train/val assignment are bit-identical by construction, and the delta is
attributable to the slice alone.

The slice draws from a separate `random.Random(seed + 1000)` stream.

KNOWN CO-MOVEMENTS (disclosed, not hidden): rows 14,850 -> ~16,400 (+10.6%),
synthetic share 30.7% -> ~37%, optimizer steps ~2,784 -> ~3,080 at 3 epochs.
These are the unavoidable cost of ADDING a slice; v7's confound was +79% rows
and 45% synthetic, an order of magnitude larger.

Pre-registered criteria (D59): success = TIME Duration_Compare gains >= +4pp
over v6's 33.1 AND overall TIME stays within noise of v6's 42.18 (|z| < 2).
Gains with collateral TIME damage = partial credit, disclosed. No gain =
the slice is spent at this size.

Usage:
    ./venv/bin/python scripts/build_v6d_training_data.py
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
from build_v5_training_data import benchmark_questions, norm, stratified_split  # noqa: E402

# TLQA target entries look like "Governor of Kansas (2011, 2012, 2013)".
_ENTRY_RE = re.compile(r"^(?P<label>.+?)\s*\((?P<years>\s*\d{3,4}(?:\s*,\s*\d{3,4})*\s*)\)\s*$")


def _spans(row: dict) -> list[tuple[str, int, int]]:
    """(label, first year, last year) for each parseable TLQA entry."""
    out = []
    for tgt in row.get("targets") or []:
        m = _ENTRY_RE.match(str(tgt).strip())
        if not m:
            continue
        years = [int(y) for y in re.findall(r"\d{3,4}", m.group("years"))]
        label = m.group("label").strip()
        if years and label:
            out.append((label, min(years), max(years)))
    return out


def build_aug_duration(pool: list[dict], n: int, rng: random.Random,
                       bench_qs: set[str]) -> list[dict]:
    """'Which lasted longer?' — targets Duration_Compare (9,217 items, -7.3pp).

    Copied VERBATIM from build_v7_training_data.py so this is the same slice
    that bought Duration_Compare 33.1 -> 40.1 there.

    Built from TLQA year spans only. The two spans must differ in length or the
    question has no answer; a 1-year gap is required so the item is decidable
    from the text rather than a coin flip.
    """
    cand = []
    for r in pool:
        if r.get("source_dataset") != "TLQA":
            continue
        sp = [s for s in _spans(r) if s[2] > s[1]]
        if len(sp) >= 2:
            cand.append(sp)
    rng.shuffle(cand)
    out: list[dict] = []
    for sp in cand:
        if len(out) >= n:
            break
        a, b = rng.sample(sp, 2)
        da, db = a[2] - a[1], b[2] - b[1]
        if da == db:
            continue
        longer = a if da > db else b
        opts = [f"{a[0]} ({a[1]}-{a[2]})", f"{b[0]} ({b[1]}-{b[2]})"]
        rng.shuffle(opts)
        correct = f"{longer[0]} ({longer[1]}-{longer[2]})"
        q = ("Which of these lasted longer?\nChoices:\n"
             + "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts)))
        if norm(q) in bench_qs:
            continue
        out.append({"source_dataset": "AUG_DURATION", "question": q, "context": "",
                    "targets": [correct], "source": "augmented"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v6-dir", default="data/combined_80_20_v6",
                    help="shipped v6 mixture, loaded verbatim as the core")
    ap.add_argument("--v3-dir", default="data/combined_80_20_v3_fixed",
                    help="TLQA pool (input to the slice generator; read-only)")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="data/combined_80_20_v6d")
    ap.add_argument("--aug-duration", type=int, default=1573,
                    help="AUG_DURATION rows (v7's slice size); 0 disables the arm's one change")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench_qs = benchmark_questions()
    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}

    core = {s: [json.loads(l) for l in open(Path(args.v6_dir) / f"{s}.jsonl",
                                            encoding="utf-8")]
            for s in ("train", "val")}
    print(f"core loaded from {args.v6_dir}: "
          f"train={len(core['train'])} val={len(core['val'])}")
    assert len(core["train"]) == 14850 and len(core["val"]) == 3708, \
        "shipped v6 mixture has unexpected size — refusing to build on it"

    dur_rows: list[dict] = []
    dur_train: list[dict] = []
    dur_val: list[dict] = []
    if args.aug_duration:
        pool = []
        for s in ("train", "val"):
            pool += [json.loads(l) for l in
                     open(Path(args.v3_dir) / f"{s}.jsonl", encoding="utf-8")]
        dur_rng = random.Random(args.seed + 1000)
        dur_rows = build_aug_duration(pool, args.aug_duration, dur_rng, bench_qs)
        dur_q = [norm(str(r.get("question") or "")) for r in dur_rows]
        assert not any(q in probe_q for q in dur_q), "FATAL: probe leak in AUG_DURATION"
        assert not any(q in bench_qs for q in dur_q), "FATAL: benchmark collision in AUG_DURATION"
        seen = {norm(str(r.get("question") or ""))
                for s in ("train", "val") for r in core[s]}
        unique = [r for r, q in zip(dur_rows, dur_q) if q not in seen]
        n_dup = len(dur_rows) - len(unique)
        if n_dup:
            print(f"AUG_DURATION dedup: dropped {n_dup} rows already in the core")
            dur_rows = unique
        dur_train, dur_val = stratified_split(dur_rows, args.val_frac, dur_rng)
        print(f"AUG_DURATION: {len(dur_rows)} (requested {args.aug_duration}; "
              f"Duration_Compare — v6 33.1, zs 40.4, v7-with-slice 40.1)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", core["train"] + dur_train),
                       ("val.jsonl", core["val"] + dur_val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rows):
        c = Counter(r["source_dataset"] for r in rows)
        syn = sum(v for k, v in c.items() if k.startswith("AUG_"))
        return {"n": len(rows), "synthetic_pct": round(100 * syn / len(rows), 1),
                "rehearsal_pct": round(100 * c.get("REHEARSAL", 0) / len(rows), 1),
                "by_source": dict(c)}

    train = core["train"] + dur_train
    val = core["val"] + dur_val
    stats = {
        "cycle": "v6d",
        "variables_vs_v6": [
            f"AUG_DURATION +{len(dur_rows)} (Duration_Compare: v6 33.1, zs 40.4, "
            f"v7-with-slice 40.1; generator copied verbatim from build_v7_training_data.py)",
            "core LOADED from the shipped v6 files (the current v6 builder no "
            "longer reproduces them — its 2026-09-07 TRAM-guard extension "
            "filters ~782 different rows), so v6's rows and train/val "
            "assignment are bit-identical by construction",
            "separate rng stream (seed+1000) for the slice",
        ],
        "train": mix(train), "val": mix(val),
        "core_is_v6_prefix": True,
        "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
