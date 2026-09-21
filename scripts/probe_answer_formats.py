"""Per-gold-format emission and accuracy for any set of arms. CPU, ~1 min/arm.

The process change D38 asked for. `audit_output_style.py` reports ONE
aggregate strict-vs-protocol gap, which is why v5's *total* collapse on 10.8%
of TIME (1 valid sequence emitted out of 11,361) passed it as merely "LARGE"
and was only found after a 7-hour evaluation. Averaging hides a collapse that
is confined to one answer shape.

This splits TIME by the SHAPE OF THE GOLD and reports, per shape, both what
the arm emitted and what it scored -- so "the model stopped producing this
kind of answer at all" can never again be averaged into "slightly worse".

    letter sequence    gold "B,C,A"            11,361 items (10.8%)
    MCQ: no-answer     gold is an abstention    2,427 items ( 2.3%)
    MCQ: answerable    gold is a bare letter   46,937 items (44.7%)
    free text          everything else         42,428 items (40.4%)
    paren permutation  gold "(...)"             1,798 items ( 1.7%)  ceiling

Usage:
    ./venv/bin/python scripts/probe_answer_formats.py --arms v6 v5 zero-shot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_baselines import _postprocess_prediction  # noqa: E402
from src.data.data_loader import BenchmarkLoader  # noqa: E402
from src.evaluation.date_equivalence import matches_any  # noqa: E402

ARM_PATHS = {
    "zero-shot": "results/baseline/zero_shot_v3/llama/time/zero_shot/predictions.jsonl",
    "v1-ft": "results/corrected/v1/llama/time/finetuned/predictions.jsonl",
    "v2-ft": "results/corrected/v2/llama/time/finetuned/predictions.jsonl",
    "v3-corr": "results/corrected/v3_fixed/llama/time/finetuned/predictions.jsonl",
    "v4": "results/corrected/v4/llama/time/finetuned/predictions.jsonl",
    "v5": "results/corrected/v5/llama/time/finetuned/predictions.jsonl",
    "v6": "results/corrected/v6/llama/time/finetuned/predictions.jsonl",
    # v7 and v7-corrected only ever ran on vLLM; their HF-era paths
    # (results/corrected/v7) do not exist, which made every probe that asked
    # for them silently print "MISSING — skipped" and still exit OK.
    "v6-vllm": "results/corrected/v6_vllm/llama/time/finetuned/predictions.jsonl",
    "v7": "results/corrected/v7_vllm/llama/time/finetuned/predictions.jsonl",
    "v7-vllm": "results/corrected/v7_vllm/llama/time/finetuned/predictions.jsonl",
    "v7-corrected": "results/corrected/v7_corrected_vllm/llama/time/finetuned/predictions.jsonl",
    "v6d": "results/corrected/v6d_vllm/llama/time/finetuned/predictions.jsonl",
    # v9 — vLLM only, like v7/v7c/v6d. Registered BEFORE the run so the
    # post-run probe cannot silently print "MISSING — skipped": that probe
    # is the pre-registered test of whether a low v9 TIME score is answer
    # style (its golds average ~7.9 words vs TIME's 2.35) or reasoning.
    "v9": "results/corrected/v9_vllm/llama/time/finetuned/predictions.jsonl",
    "v9-vllm": "results/corrected/v9_vllm/llama/time/finetuned/predictions.jsonl",
}

# A letter sequence, with EITHER separator. TIME uses both: "C,B,A" (11,361
# items) and space-separated "A C" / "A B C" (2,614 more). The 2026-09-07
# audit found the comma-only pattern was filing those 2,614 under "free text",
# where the emission check only tests for non-empty output — so a total
# collapse on them would have reported as 100% emitted.
SEQ = re.compile(r"^\s*[A-D](?:\s*[,;]\s*|\s+)[A-D](?:(?:\s*[,;]\s*|\s+)[A-D])*\s*$")
BARE = re.compile(r"^\s*[A-D]\s*$")
PAREN = re.compile(r"^\s*\(.*\)\s*$")
ABSTAIN = re.compile(r"no answer|not (?:be )?determin|does not say|none of the (?:above|options)", re.I)
ORDER = ["MCQ: answerable", "MCQ: no-answer", "letter sequence", "free text", "paren permutation"]


def bucket(golds: list[str]) -> str:
    if any(SEQ.match(g) for g in golds):
        return "letter sequence"
    if any(BARE.match(g) for g in golds):
        return "MCQ: no-answer" if any(ABSTAIN.search(g) for g in golds) else "MCQ: answerable"
    if any(PAREN.match(g) for g in golds):
        return "paren permutation"
    return "free text"


def emitted(bucket_name: str, raw: str, pred: str) -> bool:
    """Did the arm produce an answer OF THE SHAPE the gold demands?

    Sequence shape is judged on the RAW generation, because the whole failure
    mode is the model emitting a bare letter where a sequence was asked for.
    Abstention is judged on the POSTPROCESSED prediction instead: an arm that
    answers "C" has selected the no-answer option without ever writing the
    words, and reading the raw text there would score zero-shot at 3.17%
    emission against its own 52.95% accuracy.
    """
    r = (raw or "").strip()
    if bucket_name == "letter sequence":
        return bool(SEQ.match(r.splitlines()[0].strip() if r else ""))
    if bucket_name == "MCQ: no-answer":
        return bool(ABSTAIN.search(pred or ""))
    return bool(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["v5", "zero-shot"])
    ap.add_argument("--benchmark", default="time",
                    choices=["time", "timebench", "tram"],
                    help="which benchmark's stored predictions to probe")
    ap.add_argument("--data-dir", default="./data/benchmarks")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    choices = {e.id: e.choices for e in BenchmarkLoader(data_dir=args.data_dir).load(args.benchmark)}

    lines: list[str] = []
    stats: dict[str, dict] = {}
    for arm in args.arms:
        path = PROJECT_ROOT / ARM_PATHS.get(arm, arm).replace("/time/", f"/{args.benchmark}/")
        if not path.exists():
            lines.append(f"{arm}: MISSING ({path}) — skipped")
            continue
        per = defaultdict(lambda: [0, 0, 0])   # emitted, correct, n
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write — see rescore_v5_protocol's handling
            refs = r["reference"] if isinstance(r["reference"], list) else [r["reference"]]
            golds = [str(g) for g in refs if g is not None]
            raw = r.get("raw_prediction")
            pred = _postprocess_prediction(raw, choices.get(r["id"])) if raw is not None \
                else r["prediction"]
            b = bucket(golds)
            c = per[b]
            c[0] += emitted(b, str(raw if raw is not None else pred), str(pred))
            c[1] += matches_any(pred, refs)
            c[2] += 1
        stats[arm] = {k: {"emit_pct": round(100 * v[0] / v[2], 2),
                          "acc_pct": round(100 * v[1] / v[2], 2), "n": v[2]}
                      for k, v in per.items()}
        total = sum(v[1] for v in per.values()), sum(v[2] for v in per.values())
        lines.append(f"\n=== {arm} — {args.benchmark.upper()} overall {100 * total[0] / total[1]:.2f}% "
                     f"(n={total[1]}) ===")
        lines.append(f"{'gold shape':22s}{'n':>7s}{'share':>8s}{'emitted':>10s}{'accuracy':>10s}")
        for b in ORDER:
            if b not in per:
                continue
            e, c, n = per[b]
            lines.append(f"{b:22s}{n:7d}{100 * n / total[1]:7.1f}%"
                         f"{100 * e / n:9.2f}%{100 * c / n:9.2f}%")

    # The whole point: flag any shape an arm has stopped producing.
    lines.append("\n--- format alarms (emission collapsed vs the zero-shot reference) ---")
    ref = "zero-shot" if "zero-shot" in stats else None
    alarms = 0
    for arm, s in stats.items():
        if arm == ref:
            continue
        for b, v in s.items():
            base = stats.get(ref, {}).get(b, {}).get("emit_pct", 0) if ref else 0
            # Absolute rule catches a shape the arm stopped producing outright;
            # the relative rule catches a collapse in a shape zero-shot itself
            # only produces sometimes -- abstention, where zero-shot sits at
            # 52.95%, would slip past an absolute >90% test entirely.
            if base > 20 and v["emit_pct"] < 0.5 * base:
                alarms += 1
                lines.append(f"  ALARM {arm}: '{b}' emitted {v['emit_pct']:.2f}% "
                             f"(zero-shot {base:.2f}%) on {v['n']} items — "
                             f"the headline number understates this checkpoint.")
    if not alarms:
        lines.append("  none — every arm still produces every gold shape.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
