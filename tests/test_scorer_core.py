"""Direct unit tests for the load-bearing scorer functions.

Gap closed from the 2026-09-09 audit §6: `_normalize_answer`, `accuracy` and
`_canonical_golds` decide every published number and were only ever pinned
*indirectly*, through the postprocessing and date-equivalence suites. A change
to any of them silently moves every table in the thesis.

Two of these tests pin behaviour that is arguably WRONG (the letter-sequence
branch collapsing ordinary words, and thousands separators not being
canonicalised). They are marked and pinned deliberately: the protocol is
frozen, so the job of a test here is to make a change to it visible, not to
assert that the current rule is ideal. See docs/audit_2026_09_12.md §8.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts.run_baselines import _canonical_golds
from src.data.data_loader import TemporalExample
from src.evaluation.metrics import _normalize_answer, _normalize_references, accuracy


def _ex(answer, choices=None):
    return TemporalExample(
        id="x", question="q", context="", answer=answer, choices=choices,
        source="time", task="t",
    )


class TestNormalizeAnswer:
    @pytest.mark.parametrize("raw,expected", [
        ("The Boston Tea Party", "boston tea party"),
        ("an apple", "apple"),
        ("  Yes.  ", "yes"),
        ("There is no answer.", "there is no answer"),
        ("co-temporality", "co temporality"),
        ("Tuesday , 25th September , 2001", "tuesday 25th september 2001"),
    ])
    def test_squad_style_normalisation(self, raw, expected):
        assert _normalize_answer(raw) == expected

    def test_articles_are_dropped_only_as_whole_words(self):
        # "a" inside a word must survive, or "Cathedral" becomes "Cthedrl".
        assert _normalize_answer("Cathedral at Amiens") == "cathedral at amiens"

    def test_an_answer_that_is_only_an_article_normalises_to_empty(self):
        # Consequence: such an item is dropped by the rescore's empty-gold
        # guard rather than scored. No TIME/TimeBench gold does this today.
        assert _normalize_answer("the") == ""

    def test_non_string_input_is_coerced_not_crashed(self):
        assert _normalize_answer(None) == ""
        assert _normalize_answer(1994) == "1994"

    class TestLetterSequenceBranch:
        """MCQ letter sequences skip article removal ('A' must not be eaten)."""

        @pytest.mark.parametrize("raw,expected", [
            ("B", "b"),
            ("(C)", "c"),
            ("B,C,A", "b c a"),
            ("A C", "a c"),
            ("A; C", "a c"),
        ])
        def test_letter_sequences_are_space_separated(self, raw, expected):
            assert _normalize_answer(raw) == expected

        def test_a_is_not_removed_as_an_article_inside_a_sequence(self):
            # The whole reason the branch exists: "A,B" must not become "b".
            assert _normalize_answer("A,B") == "a b"

        def test_KNOWN_QUIRK_words_made_only_of_abcd_collapse(self):
            # "cab" is indistinguishable from the sequence "C,A,B" after
            # normalisation, so a free-text answer of "cab" scores correct
            # against the gold "C,A,B" and vice versa. Measured impact on the
            # campaign: no TIME/TimeBench/TRAM gold is such a word, so this is
            # latent. Pinned so a future edit to the branch is visible.
            assert _normalize_answer("cab") == "c a b"
            assert _normalize_answer("cab") == _normalize_answer("C,A,B")

    def test_KNOWN_QUIRK_number_separators_are_not_canonicalised(self):
        # Thousands and decimal separators become spaces, so gold "1,000"
        # never matches prediction "1000" (audit 2026-09-09 LOW). Changing
        # this would move published numbers, so it is pinned, not fixed.
        assert _normalize_answer("1,000") == "1 000"
        assert _normalize_answer("1000") == "1000"
        assert _normalize_answer("1,000") != _normalize_answer("1000")


class TestNormalizeReferences:
    def test_single_gold_becomes_a_one_element_list(self):
        assert _normalize_references("March 1995") == ["march 1995"]

    def test_multi_gold_keeps_every_alternative(self):
        assert _normalize_references(["A", "Yes"]) == ["a", "yes"]

    def test_blank_alternatives_are_dropped(self):
        assert _normalize_references(["", "  ", "Yes"]) == ["yes"]

    def test_an_all_blank_gold_yields_the_empty_string_gold(self):
        # This is exactly the trap the empty-gold guard exists for: with
        # [""] as the gold, an empty prediction scores CORRECT. The rescore
        # and the loader both drop such items before they reach here.
        assert _normalize_references([""]) == [""]
        assert accuracy([""], [[""]]) == 1.0


class TestAccuracy:
    def test_exact_match_after_normalisation(self):
        assert accuracy(["The Answer."], ["answer"]) == 1.0

    def test_any_gold_alternative_counts(self):
        assert accuracy(["Yes"], [["A", "Yes"]]) == 1.0

    def test_a_wrong_answer_scores_zero(self):
        assert accuracy(["No"], [["A", "Yes"]]) == 0.0

    def test_mismatched_lengths_raise_rather_than_silently_truncate(self):
        with pytest.raises(ValueError):
            accuracy(["a", "b"], ["a"])

    def test_empty_input_is_zero_not_a_division_error(self):
        assert accuracy([], []) == 0.0

    def test_is_a_fraction_not_a_percentage(self):
        # Every caller multiplies by 100; returning 50.0 here would silently
        # produce 5000% tables.
        assert accuracy(["a", "x"], ["a", "b"]) == 0.5


class TestCanonicalGolds:
    def test_letter_gold_on_an_mcq_also_accepts_the_option_text(self):
        golds = _canonical_golds(_ex("B", ["first", "second", "third"]))
        assert golds == ["B", "second"]

    def test_parenthesised_letter_is_unwrapped(self):
        assert _canonical_golds(_ex("(C)", ["a", "b", "cee"])) == ["(C)", "cee"]

    def test_integer_gold_is_read_as_an_option_index(self):
        assert _canonical_golds(_ex(1, ["first", "second"])) == ["1", "second"]

    def test_free_text_gold_is_returned_unchanged(self):
        assert _canonical_golds(_ex("March 18, 1934")) == ["March 18, 1934"]

    def test_a_letter_gold_with_no_choices_stays_a_letter(self):
        assert _canonical_golds(_ex("B")) == ["B"]

    def test_out_of_range_letter_does_not_index_past_the_options(self):
        assert _canonical_golds(_ex("D", ["a", "b"])) == ["D"]

    def test_multi_gold_is_flattened_and_deduplicated(self):
        golds = _canonical_golds(_ex(["Yes", "Yes", "yes"]))
        assert golds == ["Yes", "yes"]

    def test_blank_gold_yields_the_empty_marker_the_guards_look_for(self):
        assert _canonical_golds(_ex("")) == [""]
        assert _canonical_golds(_ex([None, "  "])) == [""]
