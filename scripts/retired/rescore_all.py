"""Rescore ALL arms from stored predictions with the fixed (punctuation-
invariant) scorer. Zero GPU cost. Writes results/rescored/fair_scoring_report.json
and prints the canonical fair-scoring table with 95% CIs and pairwise z-tests.

Re-run after the mistral zs TIME leg completes to include it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import _normalize_answer  # noqa: E402

ARMS = [
    # label, model, benchmark, arm, path
    ("llama zs TIME", "llama", "time", "zs",
     "results/baseline/zero_shot_v2/llama/time/zero_shot/predictions.jsonl"),
    ("llama v1-ft TIME", "llama", "time", "v1",
     "results/finetuned/llama/time/finetuned/predictions.jsonl"),
    ("llama v2-ft TIME", "llama", "time", "v2",
     "results/finetuned_v2/llama/time/finetuned/predictions.jsonl"),
    ("llama v3-ft TIME (HF)", "llama", "time", "v3",
     "results/finetuned_v3/llama/time/finetuned/predictions.jsonl"),
    ("llama v3-ft TIME (vLLM)", "llama", "time", "v3vllm",
     "results/finetuned_v3_vllm/llama/time/finetuned/predictions.jsonl"),
    ("llama zs TimeBench", "llama", "timebench", "zs",
     "results/baseline/zero_shot_v2/llama/timebench/zero_shot/predictions.jsonl"),
    ("llama v1-ft TimeBench", "llama", "timebench", "v1",
     "results/finetuned/llama/timebench/finetuned/predictions.jsonl"),
    ("llama v2-ft TimeBench", "llama", "timebench", "v2",
     "results/finetuned_v2/llama/timebench/finetuned/predictions.jsonl"),
    ("llama v3-ft TimeBench", "llama", "timebench", "v3",
     "results/finetuned_v3/llama/timebench/finetuned/predictions.jsonl"),
    ("llama v3-ft TimeBench (vLLM)", "llama", "timebench", "v3vllm",
     "results/finetuned_v3_vllm/llama/timebench/finetuned/predictions.jsonl"),
    ("mistral zs TimeBench", "mistral", "timebench", "zs",
     "results/baseline/zero_shot_v2/mistral/timebench/zero_shot/predictions.jsonl"),
    ("mistral v1-ft TIME", "mistral", "time", "v1",
     "results/finetuned/mistral/time/finetuned/predictions.jsonl"),
    ("mistral v1-ft TimeBench", "mistral", "timebench", "v1",
     "results/finetuned/mistral/timebench/finetuned/predictions.jsonl"),
    ("mistral zs TIME", "mistral", "time", "zs",
     "results/baseline/zero_shot_v2/mistral/time/zero_shot/predictions.jsonl"),
]

# completeness thresholds: TIME legs must have all 106500 predictions
EXPECTED_N = {"time": 106500, "timebench": 21188}


def score(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    per_cat: dict[str, list[int]] = {}
    correct = 0
    for r in rows:
        refs = r["reference"] if isinstance(r["reference"], list) else [r["reference"]]
        ref_norms = [_normalize_answer(g) for g in refs]
        ok = _normalize_answer(r["prediction"]) in ref_norms
        correct += ok
        cat = str(r.get("category") or r.get("task") or "unknown")
        c = per_cat.setdefault(cat, [0, 0])
        c[0] += ok
        c[1] += 1
    p = correct / n if n else 0.0
    ci = 1.96 * math.sqrt(p * (1 - p) / n) * 100 if n else 0.0
    return {
        "n": n, "accuracy": round(p * 100, 2), "ci95_pp": round(ci, 2),
        "complete": n == EXPECTED_N.get(bench := path.parts[2] if False else "", n),
        "by_category": {k: {"acc": round(100 * v[0] / v[1], 2), "n": v[1]}
                        for k, v in sorted(per_cat.items())},
    }


def ztest(p1: float, n1: int, p2: float, n2: int) -> float:
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    out = {}
    for label, model, bench, arm, path in ARMS:
        p = PROJECT_ROOT / path
        if not p.exists():
            out[label] = {"status": "missing"}
            continue
        s = score(p)
        expected = EXPECTED_N[bench]
        if s and s["n"] != expected:
            s["status"] = f"INCOMPLETE ({s['n']}/{expected})"
        else:
            s["status"] = "ok"
        s.update({"model": model, "benchmark": bench, "arm": arm})
        out[label] = s

    print(f"{'arm':34s} {'acc':>7s} {'±CI':>5s} {'n':>7s}  status")
    for label, s in out.items():
        if s.get("status") == "missing":
            print(f"{label:34s} {'—':>7s} {'':>5s} {'0':>7s}  missing")
        else:
            print(f"{label:34s} {s['accuracy']:6.2f}% {s['ci95_pp']:5.2f} {s['n']:7d}  {s['status']}")

    print("\n== pairwise z-tests (accuracy, fair scorer) ==")
    pairs = [
        ("llama zs TIME", "llama v1-ft TIME"), ("llama zs TIME", "llama v2-ft TIME"),
        ("llama zs TIME", "llama v3-ft TIME (vLLM)"),
        ("llama v2-ft TIME", "llama v3-ft TIME (vLLM)"),
        ("llama v1-ft TIME", "llama v2-ft TIME"),
        ("llama zs TimeBench", "llama v3-ft TimeBench"),
        ("llama zs TimeBench", "llama v2-ft TimeBench"),
        ("llama v3-ft TIME (HF)", "llama v3-ft TIME (vLLM)"),
        ("mistral zs TimeBench", "mistral v1-ft TimeBench"),
    ]
    for a, b in pairs:
        if out[a].get("status", "").startswith("INCOMPLETE") or out[b].get("status", "").startswith("INCOMPLETE"):
            continue
        if "accuracy" not in out[a] or "accuracy" not in out[b]:
            continue
        pa, pb = out[a]["accuracy"] / 100, out[b]["accuracy"] / 100
        z = ztest(pa, out[a]["n"], pb, out[b]["n"])
        d = out[a]["accuracy"] - out[b]["accuracy"]
        print(f"  {a} vs {b}: Δ={d:+.2f}pp z={z:+.1f} {'SIG' if abs(z) > 2 else 'not sig'}")

    dest = PROJECT_ROOT / "results" / "rescored"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fair_scoring_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: results/rescored/fair_scoring_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
