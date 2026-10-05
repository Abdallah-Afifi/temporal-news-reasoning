"""Score the v13-prog pilot dev predictions with the HPO's own scorer.

Reuses scripts/hpo_v11.py::score verbatim (same dev ids, same v5 rescoring,
same paired-delta math) so the pilot's numbers are directly comparable to
the v11 trial table. References:

  zs       -- results/hpo_v11/zs_dev (zero-shot dev, the objective's base)
  t09      -- results/hpo_v11/trials/t09/preds (v11-best recipe, dev +5.48).
              Confounded as a control: t09 trained on v11 data, the pilot on
              v12 + AUG_PROG (audit 2026-10-04 §3.2).
  v12-data -- the confound-free control: same recipe, v12 data, no AUG_PROG.
              Its full-test predictions contain the dev ids; they are
              filtered to the dev split into a scratch dir before scoring.

Also reports TimeBench without the date_arith subset, which the pilot's
AUG_PROG template copies verbatim (audit §3.1) -- the TimeBench headline is
not general skill without it.

Writes results/hpo_v13_pilot/score.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hpo_v11 import DEV_IDS, score  # noqa: E402  (DEV_IDS + rescore machinery)

PILOT = ROOT / "results" / "hpo_v13_pilot"
T09_PREDS = ROOT / "results" / "hpo_v11" / "trials" / "t09" / "preds"
V12_DATA = {"time": ROOT / "results/corrected/v12_data_vllm/llama/time/finetuned/predictions.jsonl",
            "timebench": ROOT / "results/corrected/v12_data_vllm/llama/timebench/finetuned/predictions.jsonl",
            "tram": ROOT / "results/tram_fixed/v12_data/llama/tram/finetuned/predictions.jsonl"}
DATE_ARITH = "timebench-date_arith-"


def public(block: dict) -> dict:
    """Drop the internal _correct id sets, which are not JSON-serialisable."""
    return {b: {k: v for k, v in s.items() if not k.startswith("_")} for b, s in block.items()}


def dev_filtered_dir(sources: dict[str, Path], dest: Path) -> Path:
    """Copy each benchmark's predictions, keeping only dev ids, into the
    <dest>/llama/<bench>/finetuned/ layout hpo_v11.score() reads."""
    dev = json.loads(DEV_IDS.read_text())
    for b, src in sources.items():
        keep = set(dev[b])
        out = dest / "llama" / b / "finetuned" / "predictions.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
            for line in f:
                try:
                    if json.loads(line).get("id") in keep:
                        g.write(line)
                except json.JSONDecodeError:
                    g.write(line)   # let rescore count it as torn, as it would
    return dest


def without_date_arith(s: dict) -> dict:
    """TimeBench accuracy with the date_arith subset removed (dev split)."""
    dev = [i for i in json.loads(DEV_IDS.read_text())["timebench"]]
    da = {i for i in dev if i.startswith(DATE_ARITH)}
    correct = s["_correct"]
    n = s["n"] - len(da)
    hit = len(correct - da)
    return {"n": n, "date_arith_n": len(da), "v5_pct": 100 * hit / n if n else 0.0,
            "date_arith_pct": 100 * len(correct & da) / len(da) if da else 0.0}


def main() -> int:
    ref = score(ROOT / "results" / "hpo_v11" / "zs_dev", "zero_shot", None)
    t09 = score(T09_PREDS, "finetuned", ref)          # vs zs, keeps _correct
    pilot = score(PILOT, "finetuned", ref)
    vs_t09 = score(PILOT, "finetuned", t09)
    with tempfile.TemporaryDirectory() as tmp:
        v12 = score(dev_filtered_dir(V12_DATA, Path(tmp)), "finetuned", ref)
    vs_v12 = score(PILOT, "finetuned", v12)
    obj = sum(pilot[b]["d_v5_pct"] for b in pilot) / len(pilot)
    obj_v12 = sum(v12[b]["d_v5_pct"] for b in v12) / len(v12)
    tb_noda = {name: without_date_arith(arm["timebench"])
               for name, arm in (("zs", ref), ("t09", t09), ("v12_data", v12), ("pilot", pilot))}
    out = {"objective": obj, "benchmarks": public(pilot), "vs_t09": public(vs_t09),
           "v12_data_objective": obj_v12, "v12_data": public(v12),
           "vs_v12data": public(vs_v12),
           "timebench_without_date_arith": tb_noda,
           "note": "v13-prog pilot: provisional data (v12+AUG_PROG, no GLM); "
                   "dev-split only; NOT quotable as v13. vs_v12data is the "
                   "confound-free control (same recipe, v12 data)."}
    (PILOT / "score.json").write_text(json.dumps(out, indent=1))

    print(f"{'bench':10s} {'zs':>7s} {'t09':>7s} {'v12d':>7s} {'pilot':>7s} "
          f"{'pilot-zs':>9s} {'pilot-t09':>10s} {'pilot-v12d':>11s}")
    for b in ("time", "timebench", "tram"):
        p = pilot[b]
        print(f"{b:10s} {p['v5_pct'] - p['d_v5_pct']:7.2f} {t09[b]['v5_pct']:7.2f} "
              f"{v12[b]['v5_pct']:7.2f} {p['v5_pct']:7.2f} {p['d_v5_pct']:+9.2f} "
              f"{vs_t09[b]['d_v5_pct']:+10.2f} {vs_v12[b]['d_v5_pct']:+11.2f}")
    print("\nTimeBench without date_arith:")
    for name, r in tb_noda.items():
        print(f"  {name:9s} {r['v5_pct']:6.2f}  (n={r['n']}; date_arith {r['date_arith_pct']:.2f})")
    print(f"\nobjective (mean delta vs zs, dev): pilot {obj:+.2f}pp, "
          f"v12-data {obj_v12:+.2f}pp   [t09 was +5.48pp]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
