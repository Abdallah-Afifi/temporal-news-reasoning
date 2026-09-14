"""Letter-format probe: measures letter-emission rate on held-out probe items.

Run AFTER v4 training: venv/bin/python scripts/eval_letter_probe.py \
    --adapter checkpoints/llama_v4/final --out logs/v4_letter_probe.report

Gate: >=95% of predictions must be a bare letter (A-D) or comma letter
sequence (e.g. B,C,A) matching the gold format. A failure here means the
v3 format crash is NOT fixed - do not spend GPU on the full eval.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import _normalize_answer  # noqa: E402

# NOTE: _normalize_answer lowercases and replaces commas with spaces
# ("B, C, A" -> "b c a"), so the regex must match that normalized form.
LETTER_RE = re.compile(r"^[a-d](?:\s+[a-d])*$", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.probe, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Use the SAME system prompt as fine-tuned evaluation (run_baselines
    # injects TEMPORAL_SYSTEM_PROMPT for adapters) so the probe measures
    # emission in the exact deployment prompt format, not an OOD one.
    system_prompt = None
    if args.adapter:
        from experiments.finetuning.shared.prompt_templates import (
            TEMPORAL_SYSTEM_PROMPT,
        )
        system_prompt = TEMPORAL_SYSTEM_PROMPT

    emit_ok = correct = 0
    per_subtype: dict[str, list[int]] = {}
    for r in rows:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": r["question"]})
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=4096, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        pred = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        norm = _normalize_answer(pred)
        is_letter = bool(LETTER_RE.match(norm))
        gold = r["targets"][0]
        hit = norm == _normalize_answer(gold)
        emit_ok += is_letter
        correct += hit
        st = per_subtype.setdefault(r["subtype"], [0, 0, 0])
        st[0] += is_letter
        st[1] += hit
        st[2] += 1

    n = len(rows)
    lines = [
        f"letter-emission rate: {emit_ok}/{n} = {100 * emit_ok / n:.1f}% "
        f"(gate >= 95%)",
        f"probe accuracy:       {correct}/{n} = {100 * correct / n:.1f}%",
    ]
    for st, (e, c, m) in sorted(per_subtype.items()):
        lines.append(f"  {st:20s} emit {100 * e / m:5.1f}%  acc {100 * c / m:5.1f}%  (n={m})")
    verdict = "PASS" if emit_ok / n >= 0.95 else "FAIL"
    lines.append(f"probe verdict: {verdict}")
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
