"""Regression tests for the zero-shot-CoT baseline arm (--prompt-style cot).

The CoT prompt must be the STaR arm-A family: identical inputs to the
standard zero-shot prompt (context / question / choices / NLI shape), with
the answer-format instruction replaced by step-then-'ANSWER:' reasoning.
The comparison against v9/STaR is only valid if the prompt family matches
and no gold information is introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import (  # noqa: E402
    COT_SYSTEM_PROMPT,
    _build_cot_prompt,
    _build_zero_shot_prompt,
    _postprocess_prediction,
)
from src.data.data_loader import TemporalExample  # noqa: E402


def _mcq_example() -> TemporalExample:
    return TemporalExample(
        id="t1",
        question="Who held the role from 2010 to 2011?",
        answer="D",
        context="Granoche moved clubs several times.",
        choices=["Club A striker", "Club B keeper", "Club C winger", "Novara"],
        temporal_type=None,
        task="Order_Compare",
        source="time",
    )


def _nli_example() -> TemporalExample:
    return TemporalExample(
        id="t2",
        question="The hypothesis text goes here.",
        answer="Neutral",
        context="The premise text goes here.",
        choices=None,
        temporal_type="temporal_nli",
        task="nli",
        source="timebench",
    )


def _sequence_example() -> TemporalExample:
    return TemporalExample(
        id="t3",
        question=(
            "Order the events. You must output a sequence of uppercase "
            "letters separated by commas, such as 'A,B,C'."
        ),
        answer="B,C,A",
        context="Event one happened after event two.",
        choices=["event one", "event two", "event three"],
        temporal_type=None,
        task="Timeline",
        source="time",
    )


def test_cot_prompt_has_same_inputs_as_standard():
    ex = _mcq_example()
    cot, std = _build_cot_prompt(ex), _build_zero_shot_prompt(ex)
    for fragment in (ex.question, ex.context or ""):
        assert fragment in cot and fragment in std
    assert all(c in cot for c in ex.choices)


def test_cot_prompt_swaps_instruction_not_content():
    ex = _mcq_example()
    cot, std = _build_cot_prompt(ex), _build_zero_shot_prompt(ex)
    assert "ANSWER:" in cot and "Step 1" in cot
    assert "ANSWER:" not in std and "Step 1" not in std
    assert "no explanation" not in cot  # standard's terse rule must be gone


def test_cot_prompt_never_reveals_the_gold():
    ex = _mcq_example()
    cot = _build_cot_prompt(ex)
    # The gold OPTION is one of the four listed — that is MCQ, not leakage.
    # What must never appear is any marker singling it out.
    assert "Novara;" not in cot.replace("Choices:\n", "")
    assert not any(
        marker in cot.lower() for marker in ("gold answer", "correct option", "answer is")
    )


def test_sequence_items_keep_self_specified_format_guard():
    cot = _build_cot_prompt(_sequence_example())
    # The generic single-letter instruction must NOT contradict the task's
    # own 'A,B,C' format instruction; the anchor instruction must remain.
    assert "or a single letter" not in cot
    assert "ANSWER:" in cot


def test_nli_shape_preserved():
    ex = _nli_example()
    cot = _build_cot_prompt(ex)
    assert "Premise:" in cot and "Hypothesis:" in cot
    assert cot.rstrip().endswith("directly as possible.")


def test_postprocessor_scores_the_anchored_line():
    cot_out = "Step 1: Compare durations.\nStep 2: B is longest.\nANSWER: Novara"
    assert _postprocess_prediction(cot_out, None) == "Novara"
    mcq_out = "Step 1: Compare.\nANSWER: D. Novara"
    assert _postprocess_prediction(mcq_out, _mcq_example().choices) == "Novara"


def test_cot_system_prompt_is_the_star_teacher():
    assert "temporal-reasoning teacher" in COT_SYSTEM_PROMPT
