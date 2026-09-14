"""Date parsing, formatting and span arithmetic for the v9 synthetic arm.

All gold computation goes through here so that §7.4's independent
recomputation (audit_v9_aug.py) can share the exact calendar rules the
generator used, while parsing only what is visibly stated in the row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_CANON = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
          6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
          11: "November", 12: "December"}
MONTH_NAMES = _CANON

_MON = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
RE_FULL_US = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
RE_FULL_EU = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+(\d{{4}})\b")
RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
RE_MONTH_YEAR = re.compile(rf"\b{_MON}\.?,?\s+(\d{{4}})\b")
RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

HINT = ("(Hint: Please answer in the form of Month Day, Year. e.g. 1 year 2 "
        "months 3days, or 2 days, or 9 months, or 3 months 15 days.)")


@dataclass
class D:
    """A possibly partial calendar date with its surface form."""

    y: int
    m: int | None = None
    d: int | None = None
    text: str = ""

    @property
    def precision(self) -> int:
        return 3 if self.d else (2 if self.m else 1)

    def key(self) -> tuple:
        return (self.y, self.m or 0, self.d or 0)

    def exact(self) -> date | None:
        if self.d and self.m:
            try:
                return date(self.y, self.m, self.d)
            except ValueError:
                return None
        return None

    def sort_key(self) -> tuple:
        return (self.y, self.m or 0, self.d or 0, 0)


def _mk(y: int, m: int | None, d: int | None, text: str) -> D | None:
    try:
        if m and d:
            date(y, m, d)
        return D(int(y), m, d, text)
    except ValueError:
        return None


def extract_dates(text: str) -> list[D]:
    """All explicit dates in text, in order of appearance, deduped.

    Shorter patterns whose span lies inside a longer match (the bare year
    inside 'March 3, 2015') are suppressed.
    """
    out: list[D] = []
    seen: set[tuple] = set()
    spans: list[tuple[int, int, int, D]] = []

    def add(d: D | None, s0: int, s1: int) -> None:
        if d:
            spans.append((s0, s1, len(m.group(0)), d))

    for m in RE_ISO.finditer(text):
        add(_mk(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)),
            m.start(), m.end())
    for m in RE_FULL_US.finditer(text):
        mon = MONTHS[m.group(1).lower().rstrip(".")]
        add(_mk(int(m.group(3)), mon, int(m.group(2)), m.group(0)),
            m.start(), m.end())
    for m in RE_FULL_EU.finditer(text):
        mon = MONTHS[m.group(2).lower().rstrip(".")]
        add(_mk(int(m.group(3)), mon, int(m.group(1)), m.group(0)),
            m.start(), m.end())
    for m in RE_MONTH_YEAR.finditer(text):
        mon = MONTHS[m.group(1).lower().rstrip(".")]
        add(_mk(int(m.group(2)), mon, None, m.group(0)),
            m.start(), m.end())
    for m in RE_YEAR.finditer(text):
        add(_mk(int(m.group(1)), None, None, m.group(0)), m.start(), m.end())

    kept: list[tuple[int, int, int, D]] = []
    for s0, s1, ln, d in sorted(spans, key=lambda x: -x[2]):
        if any(k0 <= s0 and s1 <= k1 for k0, k1, _, _ in kept):
            continue
        kept.append((s0, s1, ln, d))
    for _s0, _s1, _ln, d in sorted(kept, key=lambda x: x[0]):
        if d.key() not in seen and d.y >= 1:
            seen.add(d.key())
            out.append(d)
    return out


def full_dates(text: str) -> list[D]:
    return [d for d in extract_dates(text) if d.precision == 3]


def fmt_full(d: D | date) -> str:
    y, m, dd = (d.y, d.m, d.d) if isinstance(d, D) else (d.year, d.month, d.day)
    return f"{MONTH_NAMES[m]} {dd}, {y}"


def fmt_month_year(d: D) -> str:
    return f"{MONTH_NAMES[d.m]} {d.y}"


def fmt_auto(d: D) -> str:
    if d.precision == 3:
        return fmt_full(d)
    if d.precision == 2:
        return fmt_month_year(d)
    return str(d.y)


def days_between(a: D, b: D) -> int | None:
    ea, eb = a.exact(), b.exact()
    if ea is None or eb is None:
        return None
    return (eb - ea).days


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def humanize_span(a: D, b: D) -> str | None:
    """'8 days' / '2 months 14 days' / '1 year 3 months' per the TIME hint."""
    ea, eb = a.exact(), b.exact()
    if ea is None or eb is None or eb <= ea:
        return None
    days = (eb - ea).days
    if days < 30:
        return _plural(days, "day")
    rd = relativedelta(eb, ea)
    parts: list[str] = []
    if rd.years:
        parts.append(_plural(rd.years, "year"))
    if rd.months:
        parts.append(_plural(rd.months, "month"))
    if rd.days and not rd.years:
        parts.append(_plural(rd.days, "day"))
    return " ".join(parts) if parts else _plural(days, "day")


def month_gap(a: D, b: D) -> int | None:
    """Exact calendar-month gap for precision>=2 dates (b - a, in months)."""
    if a.m is None or b.m is None:
        return None
    return (b.y * 12 + b.m) - (a.y * 12 + a.m)


def humanize_months(gap: int) -> str:
    y, m = divmod(gap, 12)
    parts: list[str] = []
    if y:
        parts.append(_plural(y, "year"))
    if m:
        parts.append(_plural(m, "month"))
    return " ".join(parts) if parts else _plural(gap, "month")


def parse_human_span(lead: D, text: str) -> D | None:
    """Resolve 'three weeks later' style offsets relative to a lead date."""
    t = text.lower()
    num_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12, "fifteen": 15, "twenty": 20}
    m = re.match(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+(day|week|month|year)s?", t)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else num_words[m.group(1)]
    unit = m.group(2)
    base = lead.exact()
    if base is None:
        return None
    try:
        if unit == "day":
            nd = base + timedelta(days=n)
        elif unit == "week":
            nd = base + timedelta(weeks=n)
        elif unit == "month":
            rd = relativedelta(months=n)
            nd = base + rd
        else:
            nd = base + relativedelta(years=n)
    except (ValueError, OverflowError):
        return None
    return D(nd.year, nd.month, nd.day, fmt_full(nd))


def next_weekday(base: date, weekday: int) -> date:
    d = base + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def strip_date_phrases(sentence: str) -> str:
    """Remove explicit date expressions so Timeline facts order only via context."""
    s = sentence
    for pat in (RE_ISO, RE_FULL_US, RE_FULL_EU, RE_MONTH_YEAR):
        s = pat.sub(" ", s)
    s = re.sub(r"\b(on|in|by|since|until|from|during)\s+(?:the\s+)?(?:following\s+)?(?:day|month|year|week)\b", r"\1 ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s


def in_context(gold: str, context: str) -> bool:
    """Normalized containment: ignores case/whitespace and trailing punctuation."""
    def norm(s: str) -> str:
        s = re.sub(r"[^\w\s]", " ", s.lower())
        return " ".join(s.split())
    return norm(gold) in norm(context)


_JUNK = {"on", "in", "from", "to", "between", "at", "by", "and", "when",
         "until", "since", "the", "a", "an", "of", "as", "for", "with"}


def clean_fact(s: str, min_words: int = 5, max_words: int = 30) -> str | None:
    """Date-stripped fact with dangling connectors trimmed away."""
    f = strip_date_phrases(s)
    words = f.split()
    while words and words[-1].lower().strip(",;:.") in _JUNK:
        words.pop()
    while words and words[0].lower().strip(",;:.") in {"and", "when", "while", "which", "that", "but"}:
        words.pop(0)
    f = " ".join(words).rstrip(" ,;:.").strip()
    if not (min_words <= len(f.split()) <= max_words):
        return None
    if f and not re.match(r"^[A-Z]", f):
        f = f[0].upper() + f[1:]
    return f


_DATED_TAIL = re.compile(
    r"\s+(?:on|in|since|by|as\s+of|until|before|after)\s+"
    r"((?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct"
    r"|Nov|Dec)\.?\s+\d{1,4}(?:,?\s+\d{4})?|\d{4})"
    r"\s*[.,]?\s*$")


def dated_tail_fact(sent: str) -> str | None:
    """Fact = sentence with its final 'on <date>' tail removed (a clean prefix)."""
    m = _DATED_TAIL.search(sent)
    if not m:
        return None
    fact = sent[: m.start()].rstrip(" ,;:").strip()
    if len(fact.split()) < 5:
        return None
    return fact


def decide_span(d1: float, d2: float, unit: str) -> str | None:
    """Canonical A/B/C decision for Duration_Compare (shared by generator
    and audit so boundaries cannot disagree)."""
    if unit == "days":
        tol, ratio = 7.0, 1.15
        if abs(d1 - d2) <= max(tol, 0.08 * max(d1, d2)):
            return "C"
    else:
        tol, ratio = 1.0, 1.25
        if abs(d1 - d2) <= tol:
            return "C"
    if d1 >= d2 * ratio:
        return "A"
    if d2 >= d1 * ratio:
        return "B"
    return None
