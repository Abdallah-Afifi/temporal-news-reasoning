"""Tests for the GLM batch validator (scripts/ingest_glm_batch.py).

This validator is the only thing between raw model output and a training
mixture, and the pilot showed the model gets things wrong in ways that read
fine: it omitted `targets` entirely on every Computation row, copied golds out
of the context, and produced "May 4, 2000" for Feb 8 2001 minus 9 months and
5 days. Each case below is one of those failures, or a guard against the
validator over-rejecting correct rows.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_glm_batch import check, check_arithmetic, options  # noqa: E402

CTX = (
    "[1] Title: Regional Board Opens Drive, Day: March 4, 2016 Content: The "
    "Eastvale Regional Health Board began its seasonal vaccination drive on "
    "March 4, 2016, opening twelve clinics across the district and reporting "
    "early uptake well ahead of the projections it had published that winter. "
    "[2] Title: Drive Reaches Target, Day: November 25, 2016 Content: The "
    "board announced on November 25, 2016 that the drive had reached its "
    "coverage target, crediting the mobile units that served outlying areas."
)
HINT = ("   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year "
        "2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)")


def row(**kw):
    base = dict(source_dataset="AUG_GLM2", slice="A", category="Localization",
                provenance="wiki", question="When did the drive open?",
                context=CTX, targets=["March 4, 2016"],
                rationale="The passage states the drive opened on March 4, 2016.",
                source="augmented", source_id="")
    base.update(kw)
    return base


# --- schema (§2) -------------------------------------------------------------

def test_a_valid_row_passes() -> None:
    assert check(row()) == []


def test_missing_targets_is_rejected() -> None:
    """The pilot's single most common failure: the answer lived only in
    `rationale` and `targets` was absent on every Computation row."""
    r = row()
    del r["targets"]
    errs = check(r)
    assert errs and "targets" in errs[0]


def test_dual_gold_is_rejected() -> None:
    """[text, LETTER] was removed in v7c; §8 lists it as prohibited. The stale
    v8 validator REQUIRED it, which is why it could not be reused."""
    errs = check(row(targets=["March 4, 2016", "A"]))
    assert any("single non-empty element" in e for e in errs)


def test_unknown_category_is_rejected() -> None:
    assert check(row(category="Extract"))


# --- MCQ structure (§3) ------------------------------------------------------

MCQ_Q = ("Which development came first?\nChoices:\nA. The drive opened\n"
         "B. The target was reached\nC. The review was published\n"
         "D. The clinics closed")


def test_gold_must_be_character_identical_to_an_option() -> None:
    errs = check(row(category="Co_temporality", question=MCQ_Q,
                     targets=["The drive opened."]))          # trailing period
    assert any("CHARACTER-IDENTICAL" in e for e in errs)


def test_matching_option_passes() -> None:
    assert check(row(category="Co_temporality", question=MCQ_Q,
                     targets=["The drive opened"])) == []


def test_bare_letter_gold_is_rejected() -> None:
    errs = check(row(category="Co_temporality", question=MCQ_Q, targets=["A"]))
    assert errs


def test_timeline_letter_sequence_is_the_documented_exception() -> None:
    q = ("Below are 3 facts. Sort them.\nChoices:\nA. one\nB. two\nC. three")
    assert check(row(category="Timeline", question=q, targets=["B,A,C"])) == []
    assert check(row(category="Timeline", question=q, targets=["one"]))


def test_free_text_category_may_not_carry_options() -> None:
    errs = check(row(category="Computation", question=MCQ_Q, targets=["8 days"]))
    assert any("free text" in e for e in errs)


# --- label categories --------------------------------------------------------

def test_nli_saq_takes_a_bare_label_and_no_options() -> None:
    r = row(category="nli_saq", slice="B",
            question="The board reached its target before the drive opened.",
            targets=["contradiction"])
    assert check(r) == []


def test_nli_saq_rejects_a_non_label_gold() -> None:
    assert check(row(category="nli_saq", slice="B",
                     question="The board reached its target.",
                     targets=["yes"]))


def test_no_context_category_must_have_empty_context() -> None:
    q = ("The inquiry ran from May 12, 1980 to September 9, 1980, and the port "
         "authority approved the budget on July 11, 1980, in the middle of it. "
         "What is the relationship?\nChoices:\nA. IDENTITY\nB. BEFORE\nC. DURING")
    assert check(row(category="relation", slice="B", provenance="none",
                     question=q, context="", targets=["DURING"])) == []
    assert check(row(category="relation", slice="B", provenance="none",
                     question=q, context=CTX, targets=["DURING"]))


# --- §9 answer style, the gate the template build kept failing ---------------

def test_long_free_text_gold_is_rejected() -> None:
    long_gold = " ".join(["the"] * 15) + " board announcement"
    errs = check(row(targets=[long_gold]))
    assert any("too long" in e for e in errs)


def test_page_furniture_in_a_gold_is_rejected() -> None:
    assert check(row(targets=["Related Stories Board opens drive"]))


# --- §5.1 Computation --------------------------------------------------------

def test_computation_gold_copied_from_context_is_rejected() -> None:
    r = row(category="Computation", context=CTX, targets=["March 4, 2016"],
            question="What date did the drive open?" + HINT,
            rationale="Stated on March 4, 2016.")
    assert any("copied, not computed" in e for e in check(r))


def test_computation_without_the_hint_is_rejected() -> None:
    r = row(category="Computation", context=CTX, targets=["8 months 21 days"],
            question="How long between the two announcements?",
            rationale="March 4, 2016 to November 25, 2016 is 8 months 21 days.")
    assert any("hint" in e for e in check(r))


# --- §7.4 arithmetic ---------------------------------------------------------

def test_correct_span_is_accepted() -> None:
    r = row(category="Computation", context=CTX,
            question=("How long passed between the drive opening on March 4, "
                      "2016 and the target being reached on November 25, 2016?"
                      + HINT),
            targets=["8 months 21 days"],
            rationale="March 4, 2016 to November 25, 2016 is 8 months 21 days.")
    assert check_arithmetic(r) == []


def test_wrong_span_is_rejected() -> None:
    r = row(category="Computation", context=CTX,
            question=("How long passed between the drive opening on March 4, "
                      "2016 and the target being reached on November 25, 2016?"
                      + HINT),
            targets=["8 months 11 days"],
            rationale="An incorrect subtraction.")
    assert any("arithmetic" in e for e in check_arithmetic(r))


def test_off_by_one_date_offset_is_rejected() -> None:
    """The exact error the pilot produced."""
    r = row(category="Computation", context=CTX,
            question=("A report from February 8, 2001 says the contract was "
                      "signed 9 months and 5 days before the report. On what "
                      "date was it signed?" + HINT),
            targets=["May 4, 2000"],
            rationale="Subtracting the offset from the report date.")
    assert any("arithmetic" in e for e in check_arithmetic(r))


def test_correct_date_offset_is_accepted() -> None:
    r = row(category="Computation", context=CTX,
            question=("A report from February 8, 2001 says the contract was "
                      "signed 9 months and 5 days before the report. On what "
                      "date was it signed?" + HINT),
            targets=["May 3, 2000"],
            rationale="Subtracting the offset from the report date.")
    assert check_arithmetic(r) == []


def test_offset_anchor_never_comes_from_the_rationale() -> None:
    """Regression: anchors were taken from question AND rationale, so the
    answer date became its own anchor and failed a row whose gold was right."""
    r = row(category="Computation", context=CTX,
            question=("A grant was issued on April 10, 2018. A report followed "
                      "exactly 11 months after it was issued. On what date?"
                      + HINT),
            targets=["March 10, 2019"],
            rationale="April 10, 2018 plus 11 months is March 10, 2019.")
    assert check_arithmetic(r) == []


# --- interrogative -----------------------------------------------------------

def test_free_text_row_without_a_question_is_rejected() -> None:
    assert any("interrogative" in e for e in
               check(row(question="The drive opened on March 4, 2016.")))


def test_mcq_row_without_a_question_mark_is_allowed() -> None:
    """Options make the task explicit; requiring '?' rejected valid rows."""
    q = ("The Rosetta probe launched on March 2, 2004. Philae landed on "
         "November 12, 2014.\nChoices:\nA. IDENTITY\nB. BEFORE\nC. DURING")
    assert check(row(category="relation", slice="B", provenance="none",
                     question=q, context="", targets=["BEFORE"])) == []


# --- option parsing ----------------------------------------------------------

def test_options_are_parsed_from_the_choices_block() -> None:
    assert options(MCQ_Q) == ["The drive opened", "The target was reached",
                              "The review was published", "The clinics closed"]
    assert options("No choices here") == []


# --- added after the first live batch (2026-09-19) -----------------------------
#
# The first 20 GLM rows came back with a date in 100% of question stems against
# TIME's 36.7% that print NONE. The generator had been shaped that way because
# the arithmetic checker only read the stem, so a question naming events rather
# than dates could not be verified. The checker now falls back to the rationale,
# which §5.1 requires to state both dates.

def test_span_is_verified_from_the_rationale_when_the_stem_names_no_dates() -> None:
    r = row(category="Computation", context=CTX,
            question=("How long passed between the drive opening and the "
                      "coverage target being announced?" + HINT),
            targets=["8 months 21 days"],
            rationale=("The drive opened on March 4, 2016 and the target was "
                       "announced on November 25, 2016, a span of 8 months "
                       "and 21 days."))
    assert check_arithmetic(r) == []


def test_a_wrong_span_is_still_caught_via_the_rationale() -> None:
    r = row(category="Computation", context=CTX,
            question=("How long passed between the drive opening and the "
                      "coverage target being announced?" + HINT),
            targets=["8 months 11 days"],
            rationale=("The drive opened on March 4, 2016 and the target was "
                       "announced on November 25, 2016."))
    assert any("arithmetic" in e for e in check_arithmetic(r))


def test_abstain_row_needs_no_arithmetic_in_its_rationale() -> None:
    """Demanding a digit here forced a date to be invented to pass the check."""
    r = row(category="Computation", context=CTX,
            question="How long was it between the review and the audit?" + HINT,
            targets=["There is no answer."],
            rationale="The passage never dates the audit, so the span cannot "
                      "be computed.")
    assert check(r) == []


# --- Timeline ordering (added after the first live Timeline batch) ------------
#
# Timeline options are paraphrases of passage events, not quotes, so they cannot
# be matched back to the passage and dated. A fuzzy matcher tried on batch 012
# resolved 1 row in 20 and false-flagged that one -- the gold was correct. The
# rationale now carries letter-keyed ISO dates so the order is checkable, in the
# one category that produced D51.

from ingest_glm_batch import check_timeline  # noqa: E402

_KEYED = "The plans were unveiled, then consultation closed, then approval. " \
         "A=2017-07-09; B=2017-05-06; C=2017-03-11"


def test_timeline_order_matching_the_keyed_dates_passes() -> None:
    assert check_timeline({"category": "Timeline", "targets": ["C,B,A"],
                           "rationale": _KEYED}) == []


def test_timeline_order_contradicting_the_keyed_dates_is_rejected() -> None:
    errs = check_timeline({"category": "Timeline", "targets": ["A,B,C"],
                           "rationale": _KEYED})
    assert errs and "sort to" in errs[0]


def test_timeline_rationale_missing_a_letter_is_rejected() -> None:
    errs = check_timeline({"category": "Timeline", "targets": ["C,B,A"],
                           "rationale": "A=2017-07-09; B=2017-05-06"})
    assert errs and "do not cover" in errs[0]


def test_timeline_without_keyed_dates_is_silent() -> None:
    """Batches written before the keys were introduced must still ingest."""
    assert check_timeline({"category": "Timeline", "targets": ["C,B,A"],
                           "rationale": "no keyed dates here"}) == []


def test_timeline_check_ignores_other_categories() -> None:
    assert check_timeline({"category": "Computation", "targets": ["8 days"],
                           "rationale": "A=2017-07-09"}) == []


# --- Duration_Compare (added after batch 032-037) -----------------------------
#
# v6 scored 33.3% here, exactly the chance rate: its rows carried no signal. The
# comparison is recomputed from the rationale's stated spans so a row only
# trains if the answer genuinely follows from the dates.

from ingest_glm_batch import check_duration_compare  # noqa: E402

_DUR = ("Duration 1 runs from March 14, 2016 to September 3, 2018, a span of "
        "2 years. Duration 2 runs from January 22, 2017 to September 9, 2017.")


def test_duration_compare_correct_direction_passes() -> None:
    assert check_duration_compare(
        {"category": "Duration_Compare", "targets": ["Duration 1 is longer."],
         "rationale": _DUR}) == []


def test_duration_compare_wrong_direction_is_rejected() -> None:
    errs = check_duration_compare(
        {"category": "Duration_Compare", "targets": ["Duration 2 is longer."],
         "rationale": _DUR})
    assert errs and "Duration 2 is longer" in errs[0]


def test_duration_compare_false_tie_is_rejected() -> None:
    errs = check_duration_compare(
        {"category": "Duration_Compare",
         "targets": ["The two durations are approximately the same length."],
         "rationale": _DUR})
    assert errs and "similar" in errs[0]


def test_duration_compare_is_silent_without_parsable_spans() -> None:
    assert check_duration_compare(
        {"category": "Duration_Compare", "targets": ["Duration 1 is longer."],
         "rationale": "Duration 1 is clearly the longer of the two."}) == []


# --- added after longform_free batch 133-136 (2026-09-19) ---------------------
#
# The card gives two templates: interrogative and imperative ("Describe what
# happened between X and Y."). The interrogative-only gate rejected 27
# otherwise-correct rows using the card's own second template.

def test_longform_free_imperative_template_is_allowed() -> None:
    r = row(category="longform_free", slice="C",
            question="Describe what happened between March 2016 and June 2017.",
            targets=["The council approved the budget in March. A review "
                    "followed in June. The scheme opened the next year."])
    assert check(r) == []


def test_longform_free_interrogative_template_still_allowed() -> None:
    r = row(category="longform_free", slice="C",
            question="What developments does the passage report between "
                     "March 2016 and June 2017?",
            targets=["The council approved the budget in March. A review "
                    "followed in June."])
    assert check(r) == []


# --- Order_Compare (added after batch 147, which caught 4 gold/date
# contradictions by hand before this gate existed) --------------------------

from ingest_glm_batch import check_order_compare  # noqa: E402

_OC = "The passage dates the fact to March 3, 2010 and the fact to November 17, 2010."
_OC_CLOSE = "The two facts date to January 19, 2009 and January 20, 2009, a day apart."


def test_order_compare_fact1_earlier_passes() -> None:
    assert check_order_compare(
        {"category": "Order_Compare", "targets": ["Fact 1 happened earlier."],
         "rationale": _OC}) == []


def test_order_compare_wrong_fact_is_rejected() -> None:
    errs = check_order_compare(
        {"category": "Order_Compare", "targets": ["Fact 2 happened earlier."],
         "rationale": _OC})
    assert errs and "Fact 2 earlier" in errs[0]


def test_order_compare_genuine_near_tie_passes() -> None:
    assert check_order_compare(
        {"category": "Order_Compare",
         "targets": ["They happen at almost the same time."],
         "rationale": _OC_CLOSE}) == []


def test_order_compare_false_near_tie_is_rejected() -> None:
    errs = check_order_compare(
        {"category": "Order_Compare",
         "targets": ["They happen at almost the same time."],
         "rationale": _OC})
    assert errs and "days apart" in errs[0]


# --- shape-queue proportioning (found 2026-09-22, round 2) -------------------
#
# The old approach picked a shape per order from `(k*7) % 100`, which gives
# 7,14,21,28,35,42,49,56 for k=1..8 -- every one under the news threshold. Any
# category with <=8 orders got 100% news regardless of where k started. Hit
# twice: Co_temporality/Counterfactual/Order_Compare in round 1, Order_Compare
# again in round 2 (caught before generation).

import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
from make_glm_packets import build_shape_queues  # noqa: E402


def test_a_small_category_is_not_forced_all_news() -> None:
    """Order_Compare's exact round-2 shape: 50 rows / 20 per packet = 3 orders."""
    q = build_shape_queues({"Order_Compare": 50}, per_packet=20)
    shapes = q["Order_Compare"]
    assert len(shapes) == 3
    assert "news" not in shapes or shapes.count("news") < 3, (
        f"category forced all-news: {shapes}")
    assert "wiki" in shapes


def test_every_category_with_three_or_more_orders_gets_a_wiki_row() -> None:
    want = {"Order_Compare": 50, "Explicit_Reasoning": 60, "storytelling": 75}
    q = build_shape_queues(want, per_packet=20)
    for cat, shapes in q.items():
        if len(shapes) >= 3:
            assert "wiki" in shapes, f"{cat}: {shapes}"


def test_a_single_order_category_still_gets_a_shape() -> None:
    q = build_shape_queues({"nli_mcq": 15}, per_packet=20)
    assert q["nli_mcq"] in (["news"], ["wiki"], ["dial"])


def test_no_context_and_dialogue_categories_are_excluded_from_queues() -> None:
    q = build_shape_queues({"relation": 75, "temporal_dialogue": 100}, per_packet=20)
    assert "relation" not in q
    assert "temporal_dialogue" not in q
