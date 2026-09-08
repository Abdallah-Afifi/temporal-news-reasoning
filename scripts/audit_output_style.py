"""Answer-style drift audit — run AFTER training, BEFORE the long evaluation.

Why this exists. v4's headline TIME number came in at 33.62% against
v3-corrected's 37.77% and was read as a 4.15pp capability regression. It was
not. Three quarters of that gap was v4 answering *correctly in a different
surface form* than the scorer expected:

  - "Fact 1"          for the option "Fact 1 happened earlier."   (2,096 items)
  - "18 March 1934"   for the gold   "March 18, 1934."            (2,097 items)

Both are now handled (scripts/run_baselines._postprocess_prediction and
src/evaluation/date_equivalence), but the general lesson stands: a fine-tuned
model can shift its OUTPUT STYLE without shifting its ability, and exact-match
scoring turns that into a fake regression. Nine hours of TIME evaluation were
spent before anyone could see it.

This probe spends ~10 minutes instead. It samples a stratified slice of TIME,
generates with the same prompt and decoding the real evaluation uses, and
reports — against a reference arm's stored predictions on the SAME items:

  1. answer-shape statistics (length, bare-number rate, date format used,
     bare-letter rate),
  2. the strict exact-match score, and
  3. the score under the full v5 protocol.

**A large gap between (2) and (3) is the warning sign.** It means this
checkpoint's answers are being punished for their format, and the headline
number from the full run will understate it. That is informational, not a
gate — a style shift is not by itself a reason to discard a checkpoint.

Usage:
  ./venv/bin/python scripts/audit_output_style.py \
      --adapter checkpoints/llama_v5/final \
      --reference results/corrected/v4/llama/time/finetuned/predictions.jsonl \
      --out logs/queue_v5/style_audit.report
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.date_equivalence import matches_any  # noqa: E402
from src.evaluation.metrics import _normalize_answer  # noqa: E402

_BARE_LETTER = re.compile(r"^[a-d](?:\s+[a-d])*$")
_BARE_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_MONTHS = ("january|february|march|april|may|june|july|august|september|"
           "october|november|december")
_MDY = re.compile(rf"\b({_MONTHS})\s+\d{{1,2}}\b")
_DMY = re.compile(rf"\b\d{{1,2}}\s+({_MONTHS})\b")


def style(preds: list[str]) -> dict:
    n = max(len(preds), 1)
    norms = [_normalize_answer(p) for p in preds]
    return {
        "mean_words": round(sum(len(p.split()) for p in norms) / n, 2),
        "pct_bare_letter": round(100 * sum(bool(_BARE_LETTER.match(p)) for p in norms) / n, 1),
        "pct_bare_number": round(100 * sum(bool(_BARE_NUMBER.match(p)) for p in norms) / n, 1),
        "pct_month_day_year": round(100 * sum(bool(_MDY.search(p)) for p in norms) / n, 1),
        "pct_day_month_year": round(100 * sum(bool(_DMY.search(p)) for p in norms) / n, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--benchmark", default="time")
    ap.add_argument("--data-dir", default="./data/benchmarks")
    ap.add_argument("--reference", default="",
                    help="a previous arm's predictions.jsonl, compared on the "
                         "same sampled ids")
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="must match the evaluation run (run_baselines default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data.data_loader import BenchmarkLoader

    loader = BenchmarkLoader(data_dir=args.data_dir)
    by_cat: dict[str, list] = defaultdict(list)
    for ex in loader.load(args.benchmark):
        by_cat[str(getattr(ex, "temporal_type", None) or ex.task or "unknown")].append(ex)
    rng = random.Random(args.seed)
    sample = []
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        sample.extend(rows[:args.per_category])
    print(f"sampled {len(sample)} items across {len(by_cat)} categories")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    system_prompt = None
    if args.adapter:
        from experiments.finetuning.shared.prompt_templates import TEMPORAL_SYSTEM_PROMPT
        system_prompt = TEMPORAL_SYSTEM_PROMPT

    from scripts.run_baselines import _build_zero_shot_prompt, _postprocess_prediction

    preds, strict_hits, v5_hits = [], 0, 0
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for ex in sample:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": _build_zero_shot_prompt(ex)})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = _postprocess_prediction(raw, ex.choices)
        golds = ex.answer if isinstance(ex.answer, list) else [ex.answer]
        strict = _normalize_answer(pred) in [_normalize_answer(g) for g in golds]
        v5 = matches_any(pred, golds)
        preds.append(pred)
        strict_hits += strict
        v5_hits += v5
        cat = str(getattr(ex, "temporal_type", None) or ex.task or "unknown")
        c = per_cat[cat]
        c[0] += strict
        c[1] += v5
        c[2] += 1

    n = len(sample)
    lines = [
        f"=== answer-style audit: {args.adapter or 'base model'} on {args.benchmark} ===",
        f"sample: {n} items, {args.per_category}/category, seed {args.seed}",
        "",
        f"strict exact-match : {strict_hits}/{n} = {100 * strict_hits / n:.1f}%",
        f"v5 protocol        : {v5_hits}/{n} = {100 * v5_hits / n:.1f}%",
        f"format-only gap    : {100 * (v5_hits - strict_hits) / n:+.1f}pp"
        f"   <-- answers that are RIGHT but scored WRONG by exact match",
        "",
        "answer style (this checkpoint):",
    ]
    st = style(preds)
    for k, v in st.items():
        lines.append(f"  {k:22s} {v}")

    if args.reference and Path(args.reference).exists():
        ids = {ex.id for ex in sample}
        ref = [json.loads(l) for l in open(args.reference, encoding="utf-8")]
        ref = [r for r in ref if r["id"] in ids]
        if ref:
            rst = style([r["prediction"] for r in ref])
            lines += ["", f"reference arm ({Path(args.reference).parts[2]}, "
                          f"{len(ref)} of the same items):"]
            for k, v in rst.items():
                drift = st[k] - v
                flag = "  <-- DRIFT" if abs(drift) >= (1.0 if "mean" in k else 10.0) else ""
                lines.append(f"  {k:22s} {v}   (this run {st[k]}, {drift:+.2f}){flag}")

    lines += ["", "per-category (strict -> v5 protocol):"]
    for cat, (s, v, m) in sorted(per_cat.items(), key=lambda kv: -kv[1][2]):
        gap = 100 * (v - s) / m
        lines.append(f"  {cat:24s} {100 * s / m:5.1f}% -> {100 * v / m:5.1f}%  "
                     f"({gap:+.1f}pp format-only, n={m})")

    gap_pp = 100 * (v5_hits - strict_hits) / n
    lines += ["", "VERDICT: " + (
        f"format-only gap is {gap_pp:.1f}pp — LARGE. This checkpoint's answers are "
        f"being punished for their surface form. Read the full run's headline "
        f"number with that in mind, and rescore with scripts/rescore_v5_protocol.py."
        if gap_pp >= 2.0 else
        f"format-only gap is {gap_pp:.1f}pp — small. Exact match is not "
        f"materially penalising this checkpoint's answer style.")]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
