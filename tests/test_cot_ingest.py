"""scripts/ingest_cot_batch.py -- the gate that guards CoT trace quality.

Regression cases are taken from the manual test run 2026-09-24 that found
the original 4-token grounding floor too weak: a fabricated, non-grounded
trace ("He was a famous Australian politician... well known for his long
career", correct answer copied in) was wrongly accepted because a generic
4-word phrase matched the passage by chance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ingest_cot_batch import check_trace, count_steps, extract_answer, touches_passage  # noqa: E402

PASSAGE = (
    "John Sydney Joe Dawkins is an Australian former politician. In 1974, "
    "aged 27, Dawkins was elected to the House of Representatives for the "
    "marginal seat of Tangney. He was defeated at the 1975 election."
)
GOOD_COT = (
    "Step 1: The question asks his role between 1974 and 1975.\n"
    "Step 2: In 1974, Dawkins was elected to the House of Representatives "
    "for the marginal seat of Tangney.\n"
    "ANSWER: the House of Representatives"
)
FABRICATED_COT = (
    "Step 1: He was a famous Australian politician.\n"
    "Step 2: He is well known for his long career.\n"
    "ANSWER: the House of Representatives"
)


def test_extract_answer_takes_the_last_anchor():
    assert extract_answer("Step 1: x\nANSWER: first\nANSWER: second") == "second"


def test_extract_answer_none_when_absent():
    assert extract_answer("Step 1: no anchor here") is None


def test_count_steps_takes_the_max_step_number():
    assert count_steps("Step 1: a\nStep 3: b") == 3
    assert count_steps("no steps") == 0


def test_genuinely_grounded_trace_passes():
    assert touches_passage(GOOD_COT, PASSAGE)


def test_fabricated_trace_with_generic_phrasing_is_rejected():
    """The exact regression this gate exists for."""
    assert not touches_passage(FABRICATED_COT, PASSAGE)


def test_check_trace_accepts_a_good_grounded_answer():
    assert check_trace(GOOD_COT, "the House of Representatives", PASSAGE) == []


def test_check_trace_rejects_wrong_answer():
    errs = check_trace(GOOD_COT, "the Senate", PASSAGE)
    assert any("does not match gold" in e for e in errs)


def test_check_trace_rejects_fabricated_reasoning():
    errs = check_trace(FABRICATED_COT, "the House of Representatives", PASSAGE)
    assert any("grounding-b" in e for e in errs)


def test_check_trace_rejects_answer_stated_in_step_one():
    cot = "Step 1: The answer is the House of Representatives.\nANSWER: the House of Representatives"
    errs = check_trace(cot, "the House of Representatives", PASSAGE)
    assert any("grounding-a" in e for e in errs)


def test_check_trace_rejects_too_many_steps():
    cot = "\n".join(f"Step {i}: x" for i in range(1, 8)) + "\nANSWER: y"
    errs = check_trace(cot, "y", PASSAGE)
    assert any("step-bounds" in e for e in errs)


def test_check_trace_rejects_missing_answer_line():
    errs = check_trace("Step 1: reasoning with no anchor", "y", PASSAGE)
    assert errs == ["no ANSWER: line found (§ answer-match)"]


def test_check_trace_accepts_date_equivalent_answer():
    cot = ("Step 1: In 1974, Dawkins was elected to the House of Representatives.\n"
          "ANSWER: House of Representatives")
    assert check_trace(cot, "the House of Representatives", PASSAGE) == []
