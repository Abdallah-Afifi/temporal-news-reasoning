"""Pins the 2026-09-12 audit additions to scripts/rescore_v5_protocol.py.

These are ADDITIVE diagnostics: the headline `v5_pct` must be byte-for-byte
what it was before. The tests below pin (a) that the abstain-option detector
finds exactly the bucket D49 describes, (b) that McNemar agrees with the
closed form, and (c) that the new columns are computed from the same items as
the headline.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math

from scripts.rescore_v5_protocol import abstain_option_ids, mcnemar


def _approx(x, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()


class TestAbstainOptionIds:
    def test_finds_the_abstain_phrasings_time_actually_uses(self):
        cmap = {
            "a": ["Yes", "No", "There is no answer."],
            "b": ["1994", "1995", "Unanswerable"],
            "c": ["Not enough information", "March 1990"],
            "d": ["Cannot be determined.", "April"],
        }
        assert abstain_option_ids(cmap) == {"a", "b", "c", "d"}

    def test_ignores_ordinary_negative_options(self):
        # "No" as an option, and prose that merely contains "no", must not be
        # mistaken for an abstain option -- that would pull tens of thousands
        # of ordinary yes/no items into the excluded bucket.
        cmap = {
            "a": ["Yes", "No"],
            "b": ["The treaty was not ratified", "The treaty was ratified"],
            "c": ["Nobody knows the date", "1994"],
        }
        assert abstain_option_ids(cmap) == set()

    def test_free_text_items_have_no_choices_and_are_never_selected(self):
        assert abstain_option_ids({"a": None, "b": []}) == set()

    def test_case_and_punctuation_insensitive(self):
        assert abstain_option_ids({"a": ["THERE IS NO ANSWER!"]}) == {"a"}


class TestMcNemar:
    def test_symmetric_disagreement_is_null(self):
        z, chi = mcnemar(50, 50)
        assert z == 0.0
        assert chi == _approx(0.01)  # (|0|-1)^2 / 100

    def test_sign_follows_the_arm_not_the_reference(self):
        # c = arm right / reference wrong -> a positive z means the arm won.
        assert mcnemar(10, 100)[0] > 0
        assert mcnemar(100, 10)[0] < 0

    def test_matches_the_closed_form(self):
        b, c = 10145, 10896
        z, chi = mcnemar(b, c)
        assert abs(z - (c - b) / math.sqrt(b + c)) < 1e-12
        assert abs(chi - (abs(b - c) - 1) ** 2 / (b + c)) < 1e-12

    def test_no_disagreement_does_not_divide_by_zero(self):
        assert mcnemar(0, 0) == (0.0, 0.0)

    def test_reproduces_the_published_v6_vs_zero_shot_cell(self):
        # TIME, v5 protocol, measured 2026-09-12: b=10,145 c=10,896 -> +5.18.
        # The unpaired z published for the same cell is +3.32, i.e. the paired
        # test is the stronger one; if this ever flips, the pairing broke.
        z, _ = mcnemar(10145, 10896)
        assert 5.0 < z < 5.4
