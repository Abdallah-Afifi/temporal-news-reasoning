"""Tests for free-text generation scoring (situated_generation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.generation_metrics import best_token_f1, generation_f1, token_f1

GOLD = "Earth revolves around the sun approximately 365 days."


def test_identical_text_scores_one() -> None:
    assert token_f1(GOLD, GOLD) == 1.0


def test_paraphrase_scores_partial_credit_where_exact_match_gives_zero() -> None:
    """The whole point: a correct paraphrase must not score 0."""
    paraphrase = "The Earth revolves around the sun in approximately 365 days"
    score = token_f1(paraphrase, GOLD)
    assert 0.7 < score < 1.0
    assert paraphrase != GOLD  # exact match would score this 0


def test_unrelated_text_scores_zero() -> None:
    assert token_f1("Napoleon was crowned emperor in 1804.", GOLD) == 0.0


def test_articles_and_punctuation_are_ignored() -> None:
    assert token_f1("the earth revolves", "Earth revolves!") == 1.0


def test_empty_prediction_scores_zero_against_real_gold() -> None:
    assert token_f1("", GOLD) == 0.0


def test_best_of_multiple_golds_is_used() -> None:
    assert best_token_f1(GOLD, ["something else entirely", GOLD]) == 1.0


def test_generation_f1_averages_over_examples() -> None:
    assert generation_f1([GOLD, "unrelated words here"], [GOLD, GOLD]) == 0.5


# --- report-time wiring, added by the 2026-09-07 audit ---------------------
# `generation_metrics` existed but was never called from anywhere, so
# `situated_generation` reported 0.00% for every arm including zero-shot — a
# metric artifact presented as a result. And TIME shipped one category under
# two spellings, splitting it across every per-category table. Both are now
# handled in `scripts/rescore_v5_protocol.py`; these pin that wiring.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_duplicate_time_category_spellings_are_folded() -> None:
    from scripts.rescore_v5_protocol import canon_category

    assert canon_category("Co-temporality") == "Co_temporality"
    assert canon_category("co-temporality") == "Co_temporality"
    assert canon_category("Co_temporality") == "Co_temporality"


def test_unrelated_categories_are_untouched_by_folding() -> None:
    from scripts.rescore_v5_protocol import canon_category

    for c in ("Timeline", "Computation", "Order_Compare", "temporal_nli", "unknown"):
        assert canon_category(c) == c


def test_situated_generation_is_flagged_as_a_generation_category() -> None:
    from scripts.rescore_v5_protocol import _GENERATION_CATEGORIES

    assert "situated_generation" in _GENERATION_CATEGORIES
    # The headline must stay exact match for everything else.
    assert "temporal_nli" not in _GENERATION_CATEGORIES
    assert "Timeline" not in _GENERATION_CATEGORIES
