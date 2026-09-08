"""Rescore every corrected arm under the v5 scoring protocol. Zero GPU cost.

Two defects were found in the v4 post-mortem, both of which made a correct
answer score as wrong whenever the model's output STYLE changed:

1. **Unambiguous partial option naming** — v4 answers "Fact 1" where the
   option reads "Fact 1 happened earlier."  Fixed in
   `scripts/run_baselines._postprocess_prediction`; applied here by
   re-running that function over the stored `raw_prediction`.
2. **Date reformatting** — v4 answers "18 March 1934" where the gold reads
   "March 18, 1934."  Fixed in `src.evaluation.date_equivalence`, applied
   here at scoring time.

Both are applied to EVERY arm — scoring protocol is identical across arms or
the comparison is void. Neither can credit a less precise or different
answer; see the two test modules for the guards.

Writes results/rescored/v5_protocol.json and prints the official table.

Usage: ./venv/bin/python scripts/rescore_v5_protocol.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_baselines import _postprocess_prediction  # noqa: E402
from src.data.data_loader import BenchmarkLoader  # noqa: E402
from src.evaluation.date_equivalence import matches_any  # noqa: E402
from src.evaluation.generation_metrics import best_token_f1  # noqa: E402
from src.evaluation.metrics import _normalize_answer  # noqa: E402

ARMS = {
    "time": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/time/zero_shot/predictions.jsonl"),
        ("v1-ft", "results/corrected/v1/llama/time/finetuned/predictions.jsonl"),
        ("v2-ft", "results/corrected/v2/llama/time/finetuned/predictions.jsonl"),
        ("v3-corr", "results/corrected/v3_fixed/llama/time/finetuned/predictions.jsonl"),
        ("v4", "results/corrected/v4/llama/time/finetuned/predictions.jsonl"),
        ("v5", "results/corrected/v5/llama/time/finetuned/predictions.jsonl"),
        ("v6", "results/corrected/v6/llama/time/finetuned/predictions.jsonl"),
    ],
    # TRAM: added 2026-09-07 for its first ever run (D48). Arms appear here
    # only once their predictions exist; a missing file is reported and
    # skipped, so this is safe to carry before the run.
    "tram": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/tram/zero_shot/predictions.jsonl"),
        ("v6", "results/corrected/v6/llama/tram/finetuned/predictions.jsonl"),
        ("v7", "results/corrected/v7/llama/tram/finetuned/predictions.jsonl"),
    ],
    "timebench": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/timebench/zero_shot/predictions.jsonl"),
        ("v1-ft", "results/corrected/v1/llama/timebench/finetuned/predictions.jsonl"),
        ("v2-ft", "results/corrected/v2/llama/timebench/finetuned/predictions.jsonl"),
        ("v3-corr", "results/corrected/v3_fixed/llama/timebench/finetuned/predictions.jsonl"),
        ("v4", "results/corrected/v4/llama/timebench/finetuned/predictions.jsonl"),
        ("v5", "results/corrected/v5/llama/timebench/finetuned/predictions.jsonl"),
        ("v6", "results/corrected/v6/llama/timebench/finetuned/predictions.jsonl"),
    ],
}


# TIME ships ONE category under two spellings — "Co_temporality" (8,289 items)
# and "Co-temporality" (1,800). It is an upstream inconsistency in
# TIME_Newest.json, not a code bug, but reporting it as two rows splits a
# single category across every per-category table in the thesis. Normalised
# here, at report time only: the underlying data is untouched.
_CATEGORY_ALIASES = {"co-temporality": "Co_temporality"}

# Categories whose gold is a sentence, where exact match is the wrong
# instrument. TimeBench's situated_generation asks for a "temporally grounded
# statement" and its 115 golds are multi-sentence texts a correct answer can
# paraphrase, so EM reports 0.00% for EVERY arm including zero-shot — a metric
# artifact, not a result. Reported ADDITIONALLY as token-F1, never folded into
# the headline: the headline stays exact match for every category so one
# campaign is scored by one rule (see src/evaluation/generation_metrics.py).
_GENERATION_CATEGORIES = {"situated_generation"}


def canon_category(raw: str) -> str:
    """Fold TIME's duplicate category spellings into one label."""
    return _CATEGORY_ALIASES.get(str(raw).strip().lower(), str(raw))


def choice_map(benchmark: str, data_dir: str) -> dict[str, list[str] | None]:
    loader = BenchmarkLoader(data_dir=data_dir)
    return {ex.id: ex.choices for ex in loader.load(benchmark)}


def rescore(path: Path, choices_by_id: dict[str, list[str] | None]) -> dict:
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    gen_f1: dict[str, list[float]] = defaultdict(list)
    stored = fixed = n = rewritten = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        refs = r["reference"] if isinstance(r["reference"], list) else [r["reference"]]
        gold = [_normalize_answer(g) for g in refs]
        raw = r.get("raw_prediction")
        pred = _postprocess_prediction(raw, choices_by_id.get(r["id"])) \
            if raw is not None else r["prediction"]
        rewritten += pred != r["prediction"]
        s = _normalize_answer(r["prediction"]) in gold
        f = matches_any(pred, refs)
        stored += s
        fixed += f
        n += 1
        cat = canon_category(r.get("category") or r.get("task") or "unknown")
        c = per_cat[cat]
        c[0] += s
        c[1] += f
        c[2] += 1
        if cat in _GENERATION_CATEGORIES:
            gen_f1[cat].append(best_token_f1(pred, refs))
    return {"n": n, "stored_pct": 100 * stored / n, "v5_pct": 100 * fixed / n,
            "stored_correct": stored, "v5_correct": fixed, "rewritten": rewritten,
            "by_category": {k: {"stored_pct": round(100 * v[0] / v[2], 2),
                                "v5_pct": round(100 * v[1] / v[2], 2), "n": v[2]}
                            for k, v in per_cat.items()},
            "generation_f1": {k: {"token_f1": round(100 * sum(v) / len(v), 2),
                                  "n": len(v)}
                              for k, v in gen_f1.items() if v}}


def ztest(c1: int, n1: int, c2: int, n2: int) -> float:
    p1, p2 = c1 / n1, c2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/benchmarks")
    ap.add_argument("--out", default="results/rescored/v5_protocol.json")
    args = ap.parse_args()

    report: dict = {}
    for bench in ("time", "timebench", "tram"):
        cmap = choice_map(bench, args.data_dir)
        arms: dict = {}
        partial: dict = {}
        print(f"\n{'=' * 72}\n{bench.upper()}  (stored -> v5 protocol)\n{'=' * 72}")
        print(f"{'arm':12s}{'n':>8s}{'stored':>10s}{'v5':>10s}{'delta':>8s}{'rewritten':>11s}")
        for label, p in ARMS[bench]:
            path = PROJECT_ROOT / p
            if not path.exists():
                print(f"{label:12s}  missing: {p}")
                continue
            s = rescore(path, cmap)
            s["expected_n"] = len(cmap)
            if s["n"] < len(cmap):
                # A leg that is still running, or one that died part-way.
                # Its number is NOT comparable: run_baselines sorts prompts by
                # length, so a partial file is the SHORT-prompt end of the
                # benchmark and scores systematically differently. Report it,
                # never fold it into the table.
                partial[label] = s
                print(f"{label:12s}{s['n']:8d}{'':10s}{'PARTIAL':>9s}"
                      f"   {100 * s['n'] / len(cmap):5.1f}% of {len(cmap):,} "
                      f"— EXCLUDED (in progress or aborted; length-sorted, so "
                      f"not comparable)")
                continue
            arms[label] = s
            print(f"{label:12s}{s['n']:8d}{s['stored_pct']:9.2f}%{s['v5_pct']:9.2f}%"
                  f"{s['v5_pct'] - s['stored_pct']:+7.2f}{s['rewritten']:11d}")

        if not arms:
            print(f"  (no complete arm for {bench}; nothing to compare)")
            report[bench] = {"arms": {}, "partial": {k: v["n"] for k, v in partial.items()}}
            continue

        # Compare the NEWEST arm present against the two references, so this
        # keeps working as cycles are added without editing the formatting.
        new = "v6" if "v6" in arms else ("v5" if "v5" in arms else "v4")
        cats = sorted(arms[new]["by_category"],
                      key=lambda c: -arms[new]["by_category"][c]["n"])
        print(f"\nper-category, v5 protocol:\n{'category':24s}{'n':>7s}"
              + "".join(f"{a:>11s}" for a in arms)
              + f"{new + '-v3':>9s}{new + '-zs':>9s}")
        for cat in cats:
            n = arms[new]["by_category"][cat]["n"]
            row = f"{cat:24s}{n:7d}"
            acc = {}
            for a in arms:
                e = arms[a]["by_category"].get(cat)
                acc[a] = e["v5_pct"] if e else float("nan")
                row += f"{acc[a]:10.1f}%" if e else f"{'—':>11s}"
            ref = 'v3-corr' if 'v3-corr' in acc else new
            print(row + f"{acc[new] - acc[ref]:+8.1f}"
                        f"{acc[new] - acc['zero-shot']:+9.1f}")
        # Generation categories, reported ADDITIONALLY — never folded into the
        # headline. Exact match gives 0.00% for every arm here, which is a
        # metric artifact; token-F1 says what the arms actually do.
        gen_cats = sorted({c for a in arms for c in arms[a].get("generation_f1", {})})
        if gen_cats:
            print(f"\ngeneration categories — token-F1 (ADDITIONAL metric; the")
            print(f"headline above stays exact match, which reads 0.00% for all arms):")
            print(f"{'category':24s}{'n':>7s}" + "".join(f"{a:>11s}" for a in arms))
            for cat in gen_cats:
                e0 = next((arms[a]["generation_f1"][cat] for a in arms
                           if cat in arms[a].get("generation_f1", {})), None)
                row_g = f"{cat:24s}{e0['n'] if e0 else 0:7d}"
                for a in arms:
                    e = arms[a].get("generation_f1", {}).get(cat)
                    row_g += f"{e['token_f1']:10.2f}%" if e else f"{'—':>11s}"
                print(row_g)

        row = f"{'OVERALL':24s}{arms[new]['n']:7d}"
        for a in arms:
            row += f"{arms[a]['v5_pct']:10.2f}%"
        print("-" * (31 + 11 * len(arms)))
        print(row + f"{arms[new]['v5_pct'] - arms['v3-corr']['v5_pct']:+8.2f}"
                    f"{arms[new]['v5_pct'] - arms['zero-shot']['v5_pct']:+9.2f}")

        nw, v3, zs = arms[new], arms["v3-corr"], arms["zero-shot"]
        print(f"\n{new} vs v3-corrected: z = "
              f"{ztest(nw['v5_correct'], nw['n'], v3['v5_correct'], v3['n']):+.2f}")
        print(f"{new} vs zero-shot   : z = "
              f"{ztest(nw['v5_correct'], nw['n'], zs['v5_correct'], zs['n']):+.2f}")
        if new == "v5":
            v4a = arms["v4"]
            print(f"v5 vs v4          : z = "
                  f"{ztest(nw['v5_correct'], nw['n'], v4a['v5_correct'], v4a['n']):+.2f}"
                  f"  ({nw['v5_pct'] - v4a['v5_pct']:+.2f}pp — v5's own variable)")
        report[bench] = arms

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
