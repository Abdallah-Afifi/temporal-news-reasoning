"""Tests for the date-equivalence scoring rule (v4 post-mortem).

The rule must credit a reformatted date and must NEVER credit a less precise
one — that asymmetry is the whole reason it is safe to apply to every arm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.date_equivalence import matches_any, same_date


def test_day_month_year_matches_month_day_year() -> None:
    assert same_date("18 March 1934", "March 18, 1934.")
    assert same_date("26 January 2017", "January 26, 2017.")
    assert same_date("1934-03-18", "March 18, 1934.")


def test_less_precise_answer_is_still_wrong() -> None:
    """The v4 failure mode this rule must NOT paper over."""
    assert not same_date("1783", "February 24, 1783.")
    assert not same_date("1887", "December 13, 1887.")
    assert not same_date("March 1934", "March 18, 1934.")


def test_a_different_date_is_wrong() -> None:
    assert not same_date("1 March 2011", "January 23, 2016.")
    assert not same_date("19 March 1934", "March 18, 1934.")


def test_non_date_answer_is_wrong() -> None:
    assert not same_date("2 decades", "December 21, 1955.")
    assert not same_date("4 years", "1963.")
    assert not same_date("at a restaurant", "January 12, 2024")


def test_year_range_ignores_a_leading_preposition() -> None:
    assert same_date("1955 to 1958", "From 1955 to 1958.")
    assert same_date("1939 to 1950", "From 1939 to 1950.")


def test_terse_year_does_not_match_a_much_longer_gold() -> None:
    assert not same_date("1955", "the spring of 1955 through late 1958")


def test_over_answering_is_not_credited() -> None:
    assert not same_date("March 18, 1934 and April 2, 1935", "March 18, 1934.")


def test_matches_any_falls_back_to_plain_exact_match() -> None:
    assert matches_any("Berlin", ["Paris", "Berlin"])
    assert not matches_any("Rome", ["Paris", "Berlin"])
    assert matches_any("18 March 1934", ["A", "March 18, 1934."])


# --- guards against the over-permissive first draft ---------------------------
#
# The first version compared bare years whenever neither side carried a full
# (y, m, d) triple. On TimeBench arithmetic — whose golds are month+year,
# "Sep, 1183" — that ignored the month entirely and credited any answer landing
# in the right year, inflating v3-corrected's arithmetic by ~48pp. These pin the
# level rule that replaced it.

def test_same_year_different_month_is_wrong() -> None:
    assert not same_date("Dec, 1183", "Sep, 1183")
    assert not same_date("February, 1866", "Mar, 1866")
    assert not same_date("July, 1726", "Aug, 1726")


def test_month_year_reformatting_is_still_credited() -> None:
    assert same_date("July, 1653", "Jul, 1653")
    assert same_date("September 1183", "Sep, 1183")


def test_month_year_never_matches_a_bare_year() -> None:
    assert not same_date("1183", "Sep, 1183")
    assert not same_date("Sep, 1183", "1183")


def test_free_text_generation_is_out_of_scope() -> None:
    """situated_generation golds are paragraphs; sharing years is not an answer."""
    gold = ("Leland and Jane Stanford founded Stanford University in 1885. "
            "University of New Brunswick founded in 1780 is older.")
    pred = ("Stanford University was founded in 1885 by Leland and Jane Stanford "
            "in California, USA. The University of New Brunswick dates to 1780.")
    assert not same_date(pred, gold)


def test_bare_year_requires_agreeing_content_words() -> None:
    """A shared year must not credit a semantically different answer.

    Found by the 2026-09-07 audit: the length-only guard scored
    "the 1979 establishment of the Islamic Republic" correct against
    "the 1979 overthrow of the U.S.-backed monarchy". A reformatting may drop
    or add filler words; it may never substitute a content word.
    """
    assert not matches_any("the 1979 establishment", ["the 1979 overthrow"])
    assert not matches_any(
        "the 1979 establishment of the Islamic Republic",
        ["the 1979 overthrow of the U.S.-backed monarchy"],
    )
    # filler-only differences still match
    assert matches_any("in 1945", ["1945"])
    assert matches_any("1945", ["in 1945"])
    assert matches_any("the year 1990", ["1990"])


def test_full_date_requires_agreeing_content_words() -> None:
    """A shared month-year or full date must not credit a different claim.

    The 2026-09-07 guard above covered only BARE YEARS, so a shared month or
    day still licensed any surrounding assertion. Measured on TIME by the
    2026-09-14b audit: 27 items on zero-shot, 26 on v6 and 32 on v7c were
    scored correct for answers that contradict the gold.
    """
    # full date (ymd) — substituted subject
    assert not matches_any(
        "Britain and Ireland suspended flights to Egypt on November 1, 2015",
        ["Russia and Britain suspended flights to Egypt on November 1, 2015."],
    )
    assert not matches_any(
        "A third party mediated the talks on July 8, 2017",
        ["Egypt mediated the talks on July 8, 2017"],
    )
    # the temporal claim itself differs — the project's own subject matter
    assert not matches_any(
        "1 day after September 25, 2015.",
        ["2 days after September 25, 2015"],
    )
    # month-year (ym) — substituted subject
    assert not matches_any(
        "Egyptian military operations in July 2015",
        ["Israeli airstrikes in July 2015"],
    )


def test_genuine_date_reformatting_still_matches() -> None:
    """The new content guard must not cost the reformattings this module exists
    to credit: only SUBSTITUTED content words are refused, never date
    scaffolding (month names, day numbers, ordinals, years)."""
    assert matches_any("18 March 1934", ["March 18, 1934."])
    assert matches_any("March 18, 1934", ["18 March 1934"])
    assert matches_any("1934-03-18", ["March 18, 1934"])
    assert matches_any("September 1183", ["Sep, 1183"])
    # filler may still be dropped or added around a full date
    assert matches_any("on March 18, 1934", ["March 18, 1934"])
    assert matches_any("March 18, 1934", ["on March 18, 1934"])
    # and the level rule is unchanged: less specific never matches more
    assert not matches_any("1783", ["February 24, 1783"])
    assert not matches_any("Dec, 1183", ["Sep, 1183"])


def test_ordinal_day_is_demoted_to_month_year_known_limitation() -> None:
    """PINS A KNOWN LIMITATION, not desired behaviour.

    `_DMY` requires a bare `\\d{1,2}` for the day, so an ordinal form loses its
    day and is demoted to a month-year signature. The level rule then refuses
    it against a full-date gold, and "March 3rd, 2015" is scored wrong for the
    gold "March 3, 2015" — precisely the class of defect this module exists to
    remove.

    Measured on TIME by the 2026-09-14b audit and left unfixed deliberately:
    it costs 2 items on zero-shot, 3 on v6 and 2 on v7c (~0.002pp), it is
    near-uniform across arms so it does not bias any comparison, and widening
    the day pattern risks over-matching for no measurable gain. Revisit only
    if a future arm emits ordinals at a materially higher rate.
    """
    from src.evaluation.date_equivalence import _signature

    assert _signature("1st May 2016") == ("ym", [(2016, 5)])
    assert not matches_any("March 3rd, 2015", ["March 3, 2015."])
    # stripping the suffix is all it takes — the day itself is right
    assert matches_any("March 3, 2015", ["March 3, 2015."])
