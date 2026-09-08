"""Date-format equivalence for exact-match scoring.

TIME's golds are US-format ("March 18, 1934."); a model that answers
"18 March 1934" has named the same day and is scored wrong by exact match.
This is a scoring defect, not a model error, and it is **not** arm-neutral —
measured on the fully-corrected TIME runs it costs zero-shot +0.72pp, v2-ft
+0.47pp and v4 +2.12pp overall (+20.58pp on v4's Localization alone), so it
distorts every cross-arm comparison in the campaign, not just one arm's total.

Discovered in the v4 post-mortem (docs/v5_plan.md § "What v4 actually
measured"), where it accounted for most of an apparent 15.3pp Localization
collapse.

The rule is deliberately strict — it credits a *reformatting*, never a loss
of precision:

- Each side is reduced to a signature at its most specific available level
  — (year, month, day), then (year, month), then bare years — and the two
  match only at the SAME level with the SAME values. A less specific answer
  therefore never matches a more specific gold: "1783" does not match
  "February 24, 1783", and "Dec, 1183" does not match "Sep, 1183".
- Only short, answer-shaped strings are compared (both sides <= 12 words),
  so a free-text generation that merely happens to mention the same years
  is never credited.

An earlier, looser draft compared bare years whenever neither side carried a
full triple. It credited "Dec, 1183" for the gold "Sep, 1183" and inflated
TimeBench arithmetic by ~48pp. The level rule below exists to prevent that
class of over-credit; `tests/test_date_equivalence.py` pins both directions.

Applied on top of, never instead of, the project's existing normalization.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.evaluation.metrics import _normalize_answer

_MONTHS: dict[str, int] = {}
for _i, _m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1
):
    _MONTHS[_m] = _i
    _MONTHS.setdefault(_m[:3], _i)
_MONTHS.setdefault("sept", 9)

_MRE = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "18 march 1934"
_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MRE})\s+(\d{{4}})\b")
# "march 18, 1934" / "march 18 1934"
_MDY = re.compile(rf"\b({_MRE})\s+(\d{{1,2}})\s*,?\s*(\d{{4}})\b")
# "1934-03-18"
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_YEAR = re.compile(r"\b(1[0-9]\d{2}|20\d{2})\b")
# "sep, 1183" / "september 1183" -- month + year, no day
_MY = re.compile(rf"\b({_MRE})\s*,?\s+(\d{{3,4}})\b")


_MAX_WORDS = 12

# A token that is nothing but a 3-4 digit year.
_YEAR_TOKEN_RE = re.compile(r"\d{3,4}")


def _triples(text: str) -> list[tuple[int, int, int]]:
    """Every complete (year, month, day) date in ``text``."""
    t = _normalize_answer(text)
    out: list[tuple[int, int, int]] = []
    for m in _DMY.finditer(t):
        out.append((int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1))))
    for m in _MDY.finditer(t):
        out.append((int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))))
    # ISO is read from the RAW string: _normalize_answer strips the hyphens,
    # and "1934 03 18" is not safely distinguishable from three loose numbers.
    for m in _ISO.finditer(str(text or "")):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((y, mo, d))
    return out


def _month_years(text: str) -> list[tuple[int, int]]:
    """Every (year, month) pair — "Sep, 1183", "September 1183"."""
    t = _normalize_answer(text)
    return [(int(m.group(2)), _MONTHS[m.group(1)])
            for m in _MY.finditer(t)]


def _years(text: str) -> list[int]:
    return [int(y) for y in _YEAR.findall(_normalize_answer(text))]


def _signature(text: str) -> tuple[str, Any] | None:
    """Reduce a date answer to its most specific level, or None."""
    if (triples := _triples(text)):
        return ("ymd", sorted(triples))
    if (pairs := _month_years(text)):
        return ("ym", sorted(pairs))
    if (yrs := _years(text)):
        return ("y", yrs)
    return None


def same_date(prediction: Any, gold: Any) -> bool:
    """True when both strings denote the same date(s) in a different format."""
    pred, ref = str(prediction or ""), str(gold or "")
    np_, nr = _normalize_answer(pred).split(), _normalize_answer(ref).split()
    # Free-text generations are out of scope: matching them on dates alone
    # would credit an essay that merely mentions the right year.
    if len(np_) > _MAX_WORDS or len(nr) > _MAX_WORDS:
        return False
    sp, sg = _signature(pred), _signature(ref)
    if sp is None or sg is None or sp[0] != sg[0] or sp[1] != sg[1]:
        return False
    if sp[0] == "y":
        # A bare year list is weak evidence, for two separate reasons.
        #
        # 1. Verbosity: require comparable length so "1955" cannot match
        #    "the spring of 1955 through late 1958".
        if len(np_) < len(nr) - 1:
            return False
        # 2. Content (added 2026-09-07 audit): the length check alone credited
        #    any two short prose strings sharing a year — "the 1979
        #    establishment of the Islamic Republic" scored correct against
        #    "the 1979 overthrow of the U.S.-backed monarchy". Measured
        #    inflation before this guard: 99 items on zero-shot (0.09pp), 43 on
        #    v3-corrected, 14 on v4, 9 on v5 — small, but it credited answers
        #    that say the opposite of the gold. A REFORMATTED bare year is
        #    still essentially just a year, so require that either the
        #    non-year words agree, or both sides carry almost none.
        cp = {w for w in np_ if not _YEAR_TOKEN_RE.fullmatch(w)}
        cg = {w for w in nr if not _YEAR_TOKEN_RE.fullmatch(w)}
        # A reformatting may DROP or ADD filler ("in 1945" vs "1945"), so a
        # subset either way is fine. It may never SUBSTITUTE a content word,
        # which is what "establishment" -> "overthrow" does.
        return cp <= cg or cg <= cp
    return True


def matches_any(prediction: Any, golds: Iterable[Any]) -> bool:
    """Exact match under the project's normalization, or a date reformatting."""
    pred_norm = _normalize_answer(prediction)
    golds = list(golds)
    if any(pred_norm == _normalize_answer(g) for g in golds):
        return True
    return any(same_date(prediction, g) for g in golds)
