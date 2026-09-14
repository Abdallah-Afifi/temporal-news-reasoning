"""Compare vLLM predictions against the frozen HF engine.

Usage (frozen venv):
  compare_parity.py <parity.jsonl>                     # single file w/ hf_prediction
  compare_parity.py <vllm_predictions.jsonl> <hf_predictions.jsonl>   # two files joined by id
Gate: |delta accuracy| <= 0.1 pp AND normalized prediction agreement >= 99%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import _normalize_answer, accuracy  # noqa: E402


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main() -> int:
    if len(sys.argv) == 3:
        hf_rows = {r["id"]: r for r in load_rows(sys.argv[2])}
        rows = [
            {"prediction": r["prediction"], "hf_prediction": hf_rows[r["id"]]["prediction"],
             "reference": r["reference"], "id": r["id"]}
            for r in load_rows(sys.argv[1]) if r["id"] in hf_rows
        ]
    else:
        rows = load_rows(sys.argv[1])
    n = len(rows)
    if n == 0:
        print("no rows")
        return 1
    exact = sum(r["prediction"] == r["hf_prediction"] for r in rows)
    norm = sum(
        _normalize_answer(r["prediction"]) == _normalize_answer(r["hf_prediction"])
        for r in rows
    )
    preds = [r["prediction"] for r in rows]
    hf = [r["hf_prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    acc_v, acc_h = accuracy(preds, refs), accuracy(hf, refs)
    print(f"n={n}")
    print(f"exact agreement:    {exact / n:.4%}")
    print(f"normalized agreement: {norm / n:.4%}")
    print(f"accuracy vllm={acc_v:.6f} hf={acc_h:.6f} "
          f"delta={100 * (acc_v - acc_h):+.3f} pp")
    diffs = [r for r in rows
             if _normalize_answer(r["prediction"]) != _normalize_answer(r["hf_prediction"])]
    for r in diffs[:10]:
        print(f"DIFF {r['id']} | vllm: {r['prediction'][:50]!r} "
              f"| hf: {r['hf_prediction'][:50]!r} | gold: {str(r['reference'])[:50]!r}")
    verdict = "PASS" if abs(acc_v - acc_h) <= 0.001 and norm / n >= 0.99 else "FAIL"
    print(f"parity gate: {verdict}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
