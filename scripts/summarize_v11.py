"""Apply the pre-registered §9 verdict (docs/hpo_v11_protocol.md) to the final
test-minus-dev rescore.

rescore_v5_protocol.py reports each seed as its own arm; §9 needs, per
benchmark: (a) seed-42 micro accuracy above zs-vllm-pinned, (b) McNemar
z > 1.96 vs zs-vllm-pinned, (c) the mean over seeds 42/43/44 above it too.
All three must hold, else the benchmark is "no improvement". Macro and
no-abstain deltas are printed beside the headline, never used for the verdict.

Usage:
    venv/bin/python scripts/summarize_v11.py [results/rescored/v11_test_minus_dev.json]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = "zs-vllm-pinned"
SEEDS = ("v11-best", "v11-best-s43", "v11-best-s44")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/rescored/v11_test_minus_dev.json"
    report = json.loads(path.read_text())
    out: dict = {"source": str(path), "benchmarks": {}}
    print(f"source: {path}\n")
    print(f"{'bench':10s}{'zs':>8s}{'s42':>8s}{'s43':>8s}{'s44':>8s}{'mean':>8s}{'sd':>6s}"
          f"{'Δmean':>8s}{'z(s42)':>8s}{'Δmacro':>8s}{'Δno-ab':>8s}  verdict")
    for bench in ("time", "timebench", "tram"):
        arms = report.get(bench, {})
        if REF not in arms:
            print(f"{bench:10s} reference arm {REF} missing -- cannot judge")
            continue
        zs = arms[REF]
        seeds = {s: arms[s] for s in SEEDS if s in arms}
        if "v11-best" not in seeds:
            print(f"{bench:10s} v11-best missing -- cannot judge")
            continue
        accs = [seeds[s]["v5_pct"] for s in SEEDS if s in seeds]
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else float("nan")
        b, c = seeds["v11-best"]["_paired"]
        z = (c - b) / math.sqrt(b + c) if b + c else 0.0
        s42 = seeds["v11-best"]
        complete = len(accs) == len(SEEDS)
        ok = s42["v5_pct"] > zs["v5_pct"] and z > 1.96 and mean > zs["v5_pct"]
        verdict = ("BEATS zero-shot" if ok else "no improvement") + ("" if complete else
                   f"  (PROVISIONAL: {len(accs)}/3 seeds)")
        cell = lambda s: f"{seeds[s]['v5_pct']:8.2f}" if s in seeds else f"{'—':>8s}"
        print(f"{bench:10s}{zs['v5_pct']:8.2f}{cell('v11-best')}{cell('v11-best-s43')}"
              f"{cell('v11-best-s44')}{mean:8.2f}{sd:6.2f}{mean - zs['v5_pct']:+8.2f}{z:+8.2f}"
              f"{s42['macro_pct'] - zs['macro_pct']:+8.2f}"
              f"{s42['no_abstain_pct'] - zs['no_abstain_pct']:+8.2f}  {verdict}")
        out["benchmarks"][bench] = {
            "zs_pinned": zs["v5_pct"], "seeds": {s: seeds[s]["v5_pct"] for s in seeds},
            "mean": mean, "sd": sd, "delta_mean": mean - zs["v5_pct"],
            "mcnemar_z_seed42": z, "delta_macro_seed42": s42["macro_pct"] - zs["macro_pct"],
            "delta_no_abstain_seed42": s42["no_abstain_pct"] - zs["no_abstain_pct"],
            "n": s42["n"], "seeds_complete": complete, "verdict": verdict}
    print("\nTRAM significance is overstated by its MCQ/SAQ NLI duplication (audit §9);"
          "\nTIME's no-abstain column is the artifact-free comparison (audit §1.1 / D49).")
    dst = path.with_name(path.stem + "_verdict.json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
