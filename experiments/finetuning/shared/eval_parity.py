"""Train/eval prompt parity.

Found 2026-09-23 (docs/audit_2026_09_23.md §1): every fine-tuned arm was
TRAINED on ``build_chat_messages`` -- a custom system prompt plus
"Context:/Question:" with any choices left inside the question -- but
EVALUATED on ``run_baselines._build_zero_shot_prompt``, which adds three
instruction lines, a separate "Choices:" block with a letter instruction, and
a Premise/Hypothesis layout for NLI. The fine-tuned model was scored on a
prompt it had never seen, while zero-shot was scored on the instruction
prompt it follows natively.

This module renders a training row through the SAME builder the evaluator
calls, after mapping the row onto the field layout the benchmark loader
produces for the analogous item, so the only difference between a zero-shot
and a fine-tuned run is the weights:

  - "...\\nChoices:\\nA. x\\nB. y"   -> question + choices list (as TIME/TRAM load)
  - nli_saq / nli_mcq "Premise: P\\nHypothesis: H" -> context=P, question=H,
    temporal_type="temporal_nli" (exactly how TRAM's nli_* items load)
  - no system message, and the chat template's "Today Date" pinned to
    PINNED_DATE (the template otherwise stamps the wall-clock date into every
    prompt, so runs on different days see different inputs).
"""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.run_baselines import _build_zero_shot_prompt  # noqa: E402
from src.data.data_loader import TemporalExample  # noqa: E402

# The Llama-3.x template's own fallback when strftime_now is unavailable.
PINNED_DATE = "26 Jul 2024"

_CHOICES_SPLIT = re.compile(r"\n\s*Choices:\s*\n")
_OPTION = re.compile(r"^\s*([A-Z])\.\s+(.*\S)\s*$")
_NLI = re.compile(r"^\s*Premise:\s*(.*?)\s*\n\s*Hypothesis:\s*(.*\S)\s*$", re.S)
NLI_CATEGORIES = {"nli_saq", "nli_mcq"}


def _split_choices(question: str) -> tuple[str, list[str] | None]:
    parts = _CHOICES_SPLIT.split(question, maxsplit=1)
    if len(parts) != 2:
        return question, None
    opts: list[str] = []
    for i, line in enumerate(l for l in parts[1].splitlines() if l.strip()):
        m = _OPTION.match(line)
        # Options must run A, B, C, ... with nothing else after them; anything
        # else is not an option block this function understands, so the
        # question is left exactly as written rather than half-parsed.
        if not m or m.group(1) != chr(65 + i):
            return question, None
        opts.append(m.group(2))
    return (parts[0].rstrip(), opts) if len(opts) >= 2 else (question, None)


def row_to_example(record: dict, question: str, context: str,
                   answer: str) -> TemporalExample:
    """Map a normalised training row onto the benchmark loader's layout."""
    category = record.get("category")
    q, choices = _split_choices(question)
    ctx, ttype = context, None
    if category in NLI_CATEGORIES:
        m = _NLI.match(q)
        if m:
            ctx, q, ttype = m.group(1), m.group(2), "temporal_nli"
    return TemporalExample(id="", question=q, answer=answer, context=ctx,
                           choices=choices, task=category, temporal_type=ttype)


def render_prompt(tokenizer, example: TemporalExample,
                  date_string: str = PINNED_DATE) -> str:
    """Exactly the string run_eval_vllm.py feeds the model (parity mode)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _build_zero_shot_prompt(example)}],
        tokenize=False, add_generation_prompt=True, date_string=date_string)


def with_context(example: TemporalExample, context: str) -> TemporalExample:
    return replace(example, context=context)
