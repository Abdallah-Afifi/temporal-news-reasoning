"""Independent re-verification of every math / programmatic row v13 keeps.

Scope (docs/audit_2026_10_04.md; researcher decision 2026-10-05: math rows are
kept, but only if their golds are independently correct):

  1. AUG_TPL3 math -- data/manual_aug_glm_v13/*.jsonl, categories
     Computation, Timeline, Duration_Compare, Order_Compare, Relative_Reasoning;
  2. v12-base script-built math -- rows of data/combined_80_20_v12 whose
     question appears in the script-built data/glm_raw/271..313.txt files and
     whose category is one of the above;
  3. AUG_PROG -- every row of data/prog_aug_v13/prog_*.jsonl.

INDEPENDENCE: the expected gold is derived only from what a model sees
(question + context + choices) with this file's own date parser. It never
reads the row's rationale, the generators, or ingest_glm_batch's checkers.

A row FAILS when the derived gold differs from targets[0] ("wrong") or when
the visible text does not determine a unique answer ("undeterminable":
the event is never dated, two candidate dates for one event, ties, year-only
facts that overlap). A row whose shape this parser cannot read is "unparsed"
and is NOT failed.

Outputs:
  data/v13_verify/failed_questions.json  -- list of row["question"].strip()
  data/v13_verify/report.json            -- per source/category counts + examples

Usage: venv/bin/python scripts/verify_v13_math.py
"""
from __future__ import annotations

import calendar
import datetime as dt
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "v13_verify"
MATH = {"Computation", "Timeline", "Duration_Compare", "Order_Compare",
        "Relative_Reasoning"}

# Thresholds for the three-way compare categories (no generator code read).
# "Approximately the same" is only judged at the clear ends; the band between
# is accepted either way, so only unambiguous label errors fail.
OC_SAME_MAX = 14          # gold "almost the same time" fails if gap > this (days)
DC_SAME_HARD = (0.03, 14)  # rel diff <= 3% or <= 14 days: must be "approximately the same"
DC_DIFF_HARD = (0.20, 90)  # rel diff >= 20% AND >= 90 days: must NOT be "the same"


class Unparsed(Exception):
    pass


class Undet(Exception):
    pass


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------
_MON_FULL = ["january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"]
MONTHS = {m: i + 1 for i, m in enumerate(_MON_FULL)}
MONTHS.update({m[:3]: i + 1 for i, m in enumerate(_MON_FULL)})
MONTHS["sept"] = 9
MON_RE = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
ORD = r"(?:st|nd|rd|th)?"
P_DMY = re.compile(rf"\b(?:the\s+)?(\d{{1,2}}){ORD}\s+(?:of\s+)?({MON_RE}),?\s+(\d{{4}})\b")
P_MDY = re.compile(rf"\b({MON_RE})\s+(\d{{1,2}}){ORD},?\s+(\d{{4}})\b")
P_NUM = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
P_MY = re.compile(rf"\b({MON_RE}),?\s+(\d{{4}})\b")
P_Y = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def _mon(s: str) -> int:
    return MONTHS[s.lower().rstrip(".")[:4] if s.lower().startswith("sept") else
                  s.lower().rstrip(".")[:3]] if not s.lower().startswith("sept") else 9


class Span:
    __slots__ = ("start", "end", "lo", "hi", "kind")

    def __init__(self, start, end, lo, hi, kind):
        self.start, self.end, self.lo, self.hi, self.kind = start, end, lo, hi, kind

    def __repr__(self):
        return f"Span({self.kind},{self.lo},{self.hi})"


def find_dates(text: str, years: bool = True) -> list[Span]:
    """Non-overlapping date mentions, most specific first: full > month > year."""
    spans: list[Span] = []
    taken: list[tuple[int, int]] = []

    def free(a, b):
        return all(b <= s or a >= e for s, e in taken)

    def add(m, lo, hi, kind):
        if free(m.start(), m.end()):
            spans.append(Span(m.start(), m.end(), lo, hi, kind))
            taken.append((m.start(), m.end()))

    for m in P_DMY.finditer(text):
        try:
            add(m, d := dt.date(int(m.group(3)), _mon(m.group(2)), int(m.group(1))), d, "full")
        except ValueError:
            pass
    for m in P_MDY.finditer(text):
        try:
            add(m, d := dt.date(int(m.group(3)), _mon(m.group(1)), int(m.group(2))), d, "full")
        except ValueError:
            pass
    for m in P_NUM.finditer(text):
        try:
            add(m, d := dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))), d, "full")
        except ValueError:
            pass
    for m in P_MY.finditer(text):
        y, mo = int(m.group(2)), _mon(m.group(1))
        add(m, dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1]), "month")
    if years:
        for m in P_Y.finditer(text):
            y = int(m.group(1))
            add(m, dt.date(y, 1, 1), dt.date(y, 12, 31), "year")
    return sorted(spans, key=lambda s: s.start)


def full_dates(text: str) -> list[dt.date]:
    return [s.lo for s in find_dates(text, years=False) if s.kind == "full"]


def parse_single_date(text: str) -> dt.date | None:
    d = [s for s in find_dates(text, years=False) if s.kind == "full"]
    return d[0].lo if len(d) == 1 else None


# ---------------------------------------------------------------------------
# durations
# ---------------------------------------------------------------------------
_DUR = re.compile(r"(\d+)\s*(years?|months?|weeks?|days?)\b", re.I)


def parse_duration(s: str) -> tuple[int, int, int] | None:
    y = m = d = 0
    found = False
    rest = s
    for n, unit in _DUR.findall(s):
        found = True
        n = int(n)
        u = unit.lower()
        if u.startswith("year"):
            y += n
        elif u.startswith("month"):
            m += n
        elif u.startswith("week"):
            d += 7 * n
        else:
            d += n
    rest = _DUR.sub("", s).replace("and", "").strip(" ,.")
    return (y, m, d) if found and not rest else None


def span_tuple(a: dt.date, b: dt.date) -> tuple[int, int, int]:
    lo, hi = sorted((a, b))
    rd = relativedelta(hi, lo)
    return (rd.years, rd.months, rd.days)


# ---------------------------------------------------------------------------
# locating events in a context
# ---------------------------------------------------------------------------
STOP = set("the of a an and its was were is are by to on in at for his her their "
           "first s this that with as from had has be been which who".split())
_SUFFIXES = ("ation", "ition", "ments", "ment", "ings", "ing", "ion", "ied", "ed",
             "es", "s")


def stem(w: str) -> str:
    w = w.lower().replace("'", "")
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w[:6]


def stems(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[A-Za-z']+", text) if w.lower() not in STOP}


TEMPLATE_TITLES = re.compile(
    r"Title: (A Busy Season|Town Affairs|The Board Reports|A Year of Decisions|"
    r"Chronicle of Progress|The Season's Record|The Committee's Work)")


def context_clauses(ctx: str) -> list[tuple[str, dt.date]]:
    """(clause text, date) for every full date in the context that is not a
    passage header ("Day: ...")."""
    out = []
    spans = [s for s in find_dates(ctx, years=False) if s.kind == "full"]
    prev_end = 0
    for s in spans:
        if ctx[max(0, s.start - 5): s.start].endswith("Day: "):
            prev_end = s.end
            continue
        seg_start = prev_end
        for mark in (". ", "| ", "Content: ", "\n", "; "):
            k = ctx.rfind(mark, prev_end, s.start)
            if k >= 0:
                seg_start = max(seg_start, k + len(mark))
        out.append((ctx[seg_start: s.start], s.lo))
        prev_end = s.end
    return out


def _norm_phrase(p: str) -> str:
    p = p.strip().rstrip(".").strip()
    p = re.sub(r"^(the|The)\s+", "", p)
    return re.sub(r"\s+", " ", p).lower()


def locate(phrase: str, ctx: str, template: bool):
    """Return ('ok', date) | raises Undet / Unparsed."""
    p = _norm_phrase(phrase)
    low = ctx.lower()
    dates = set()
    for m in re.finditer(re.escape(p), low):
        tail = ctx[m.end(): m.end() + 70]
        mm = re.match(r"\s*(?:,[^,.]{0,40},\s*)?(?:on|dated|in)\s+", tail)
        if not mm:
            continue
        d = [s for s in find_dates(tail[mm.end():], years=False) if s.kind == "full" and s.start == 0]
        if d:
            dates.add(d[0].lo)
    if len(dates) == 1:
        return dates.pop()
    if len(dates) > 1:
        raise Undet(f"'{p}' dated twice: {sorted(dates)}")
    # fuzzy fallback
    fs = stems(p)
    if not fs:
        raise Unparsed(f"no content words in '{p}'")
    scored = []
    for clause, d in context_clauses(ctx):
        cs = stems(clause)
        scored.append((len(fs & cs) / len(fs), d))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < 0.8:
        if template:
            raise Undet(f"'{p}' never dated in the context")
        raise Unparsed(f"cannot locate '{p}'")
    best = scored[0]
    rivals = [s for s in scored[1:] if s[1] != best[1] and s[0] >= best[0] - 0.2]
    if rivals:
        if template and rivals[0][0] >= best[0]:
            raise Undet(f"'{p}' matches clauses with different dates")
        raise Unparsed(f"ambiguous fuzzy match for '{p}'")
    return best[1]


def event_date(phrase: str, ctx: str, template: bool) -> dt.date:
    """Date stated inside the phrase itself, else located in the context."""
    d = full_dates(phrase)
    if len(d) == 1:
        return d[0]
    if len(d) > 1:
        raise Unparsed("phrase carries two dates")
    if not ctx:
        raise Undet(f"'{phrase}' has no date and there is no context")
    return locate(phrase, ctx, template)


def options(q: str) -> dict[str, str]:
    tail = q.split("Choices:", 1)[1] if "Choices:" in q else q
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^\s*([A-E])\.\s+(.*?)\s*$", tail, re.M)}


# ---------------------------------------------------------------------------
# per-category expected golds
# ---------------------------------------------------------------------------
def v_clock(r):
    m = re.fullmatch(r"What is (\d{1,2}):(\d{2}) ([+-]) (\d{1,2}):(\d{2})\?", r["question"].strip())
    if not m:
        raise Unparsed("clock shape")
    a = int(m.group(1)) * 60 + int(m.group(2))
    b = int(m.group(4)) * 60 + int(m.group(5))
    t = (a + b if m.group(3) == "+" else a - b) % 1440
    return f"{t // 60:02d}:{t % 60:02d}", lambda g, e: g.strip() == e


def v_month(r):
    q = r["question"].strip()
    m = re.fullmatch(rf"What is the time (?:(\d+) years?)?(?: and )?(?:(\d+) months?)? "
                     rf"(after|before) ({MON_RE}),? (\d{{4}})\??", q)
    if not m:
        raise Unparsed("month-arith shape")
    yrs, mos = int(m.group(1) or 0), int(m.group(2) or 0)
    base = dt.date(int(m.group(5)), _mon(m.group(4)), 1)
    delta = relativedelta(years=yrs, months=mos)
    d = base + delta if m.group(3) == "after" else base - delta
    exp = (d.year, d.month)

    def ok(g, e):
        mm = re.fullmatch(rf"({MON_RE}),?\s+(\d{{4}})", g.strip())
        return bool(mm) and (int(mm.group(2)), _mon(mm.group(1))) == e
    return exp, ok


_HINT = re.compile(r"\(Hint:.*?\)\s*$", re.S)


def v_computation(r, template):
    q = _HINT.sub("", r["question"]).strip()
    ctx = r.get("context", "")
    m = re.search(r"date (.+?) (before|after) (.+?)\?$", q)
    if m and parse_duration(m.group(1)):
        y, mo, d = parse_duration(m.group(1))
        anchor = event_date(m.group(3), ctx, template)
        delta = relativedelta(years=y, months=mo, days=d)
        exp = anchor + delta if m.group(2) == "after" else anchor - delta
        return exp, lambda g, e: parse_single_date(g) == e
    qd = full_dates(q)
    if len(qd) >= 2:
        exp = span_tuple(qd[0], qd[1])
    else:
        m = (re.search(r"between (.+?) and (.+?)\?$", q)
             or re.search(r"from (.+?) (?:to|until) (.+?)\?$", q))
        if not m:
            raise Unparsed("computation shape")
        a = event_date(m.group(1), ctx, template)
        b = event_date(m.group(2), ctx, template)
        exp = span_tuple(a, b)
    if exp == (0, 0, 0):
        raise Undet("zero-length span")
    return exp, lambda g, e: parse_duration(g) == e


def _two_events(between: str, ctx: str, template: bool):
    """Split 'X and Y' where either side may contain ' and '."""
    qd = find_dates(between, years=False)
    if len(qd) == 2 and all(s.kind == "full" for s in qd):
        return qd[0].lo, qd[1].lo
    results = set()
    errs = []
    for m in re.finditer(r" and ", between):
        a, b = between[: m.start()], between[m.end():]
        try:
            results.add((event_date(a, ctx, template), event_date(b, ctx, template)))
        except Undet as e:
            errs.append(("undet", e))
        except Unparsed as e:
            errs.append(("unparsed", e))
    if len(results) == 1:
        return results.pop()
    if len(results) > 1:
        raise Unparsed("several ways to split the duration")
    if any(k == "undet" for k, _ in errs):
        raise Undet(str(next(e for k, e in errs if k == "undet")))
    raise Unparsed(str(errs[0][1]) if errs else "duration shape")


def v_duration_compare(r, template):
    q, ctx = r["question"], r.get("context", "")
    m = re.search(r"\*Duration 1:\*\s*Between (.+?)\.\s*\*Duration 2:\*\s*Between (.+?)\.\s*(?:\n|Choices:|$)", q, re.S)
    if not m:
        raise Unparsed("duration-compare shape")
    a1, b1 = _two_events(m.group(1), ctx, template)
    a2, b2 = _two_events(m.group(2), ctx, template)
    d1, d2 = abs((b1 - a1).days), abs((b2 - a2).days)
    gold = r["targets"][0]
    big = max(d1, d2) or 1
    rel, diff = abs(d1 - d2) / big, abs(d1 - d2)
    hard_same = rel <= DC_SAME_HARD[0] or diff <= DC_SAME_HARD[1]
    hard_diff = rel >= DC_DIFF_HARD[0] and diff >= DC_DIFF_HARD[1]
    longer = "Duration 1 is longer." if d1 > d2 else "Duration 2 is longer." if d2 > d1 else None
    info = f"d1={d1}d d2={d2}d rel={rel:.3f}"
    if gold.startswith("The two durations"):
        ok = not hard_diff
    elif hard_same:
        ok = False
    else:
        ok = gold == longer
    return (ok, info)


def v_order_compare(r, template):
    q, ctx = r["question"], r.get("context", "")
    m = re.search(r"For Fact1:\s*(.+?) and Fact2:\s*(.+?), which one happened earlier\?", q, re.S)
    if not m:
        raise Unparsed("order-compare shape")
    d1 = event_date(m.group(1), ctx, template)
    d2 = event_date(m.group(2), ctx, template)
    g = (d2 - d1).days
    gold = r["targets"][0]
    info = f"gap={g}d"
    if gold.startswith("They happen"):
        ok = abs(g) <= OC_SAME_MAX
    elif g == 0:
        ok = False
    elif gold.startswith("Fact 1"):
        ok = g > 0
    elif gold.startswith("Fact 2"):
        ok = g < 0
    else:
        raise Unparsed("order-compare gold")
    return (ok, info)


def _fact_interval(text: str, ctx: str, template: bool):
    sp = find_dates(text, years=True)
    if sp:
        if len(sp) > 1:
            raise Unparsed("option with two dates")
        return sp[0].lo, sp[0].hi
    d = event_date(text, ctx, template)
    return d, d


def v_timeline(r, template):
    q, ctx = r["question"], r.get("context", "")
    opts = options(q)
    if len(opts) < 2:
        raise Unparsed("timeline options")
    iv = {k: _fact_interval(v, ctx, template) for k, v in opts.items()}
    order = sorted(iv, key=lambda k: (iv[k][0], iv[k][1]))
    for a, b in zip(order, order[1:]):
        if not iv[a][1] < iv[b][0]:
            raise Undet(f"facts {a}/{b} overlap or tie: {iv[a]} vs {iv[b]}")
    return ",".join(order), lambda g, e: g.replace(" ", "") == e


def v_relative(r, template):
    q, ctx = r["question"], r.get("context", "")
    stem_q = q.split("Choices:")[0]
    m = re.search(r"(?:after|followed)\s+(.+?)\?", stem_q, re.S)
    if not m:
        raise Unparsed("relative-reasoning shape")
    anchor = event_date(m.group(1), ctx, template)
    gold = r["targets"][0].strip()
    opts = options(q) if "Choices:" in q else {}
    if opts:
        cand = {}
        for k, v in opts.items():
            try:
                cand[v] = event_date(v, ctx, template)
            except Undet:
                if v == gold:
                    raise
                cand[v] = None         # never dated: cannot be "the next event"
        later = {v: d for v, d in cand.items() if d is not None and d > anchor}
        if not later:
            raise Undet("no option dated after the anchor")
        best = min(later.values())
        winners = [v for v, d in later.items() if d == best]
        if len(winners) > 1:
            raise Undet("tie for the next event")
        return winners[0], lambda g, e: g.strip() == e
    # free text: next dated clause in the whole context
    clauses = sorted(((c, d) for c, d in context_clauses(ctx) if d > anchor), key=lambda x: x[1])
    if not clauses:
        raise Undet("nothing dated after the anchor")
    if len(clauses) > 1 and clauses[0][1] == clauses[1][1]:
        raise Undet("tie for the next event")
    gs = stems(gold)
    nxt = clauses[0][0]
    if gs and len(gs & stems(nxt)) / len(gs) >= 0.6:
        return True, f"next={nxt.strip()[:60]}"
    try:
        gd = locate(gold, ctx, template)
    except (Undet, Unparsed):
        raise Unparsed("free-text gold not locatable (paraphrase)")
    return gd == clauses[0][1], f"gold dated {gd}, next is {clauses[0][1]}"


def v_ordering_seq(r):
    q = r["question"].split("Choices:")[0]          # the options also contain "(n)"
    items = re.findall(r"\((\d+)\)\s*(.+?)(?=\s*\(\d+\)|$)", q, re.S)
    if len(items) < 2:
        raise Unparsed("ordering shape")
    ds = {}
    for k, t in items:
        d = full_dates(t)
        if len(d) != 1:
            raise Unparsed("ordering item date")
        ds[k] = d[0]
    order = sorted(ds, key=lambda k: ds[k])
    if len(set(ds.values())) != len(ds):
        raise Undet("tied dates")
    exp = ", ".join(f"({k})" for k in order)
    return exp, lambda g, e: g.replace(" ", "") == e.replace(" ", "")


def v_ordering_tf(r):
    q = r["question"].split("Choices:")[0]
    d = full_dates(q)
    if len(d) != 2:
        raise Unparsed("tf dates")
    if d[0] == d[1]:
        raise Undet("same date")
    exp = "TRUE" if d[0] < d[1] else "FALSE"
    return exp, lambda g, e: g.strip().upper() == e


def _event_extent(sent: str):
    m = re.search(r"ran from (.+)", sent)
    if m:                                  # dates may contain commas ("Jan 20, 1892")
        d = full_dates(m.group(1))
        if len(d) >= 2:
            return d[0], d[1]
    m = re.search(r"spanned the years (\d{4}) to (\d{4})", sent)
    if m:
        return dt.date(int(m.group(1)), 1, 1), dt.date(int(m.group(2)), 12, 31)
    d = full_dates(sent)
    if len(d) == 1:
        return d[0], d[0]
    raise Unparsed("event extent")


def _relate(ev, ref):
    (es, ee), (rs, re_) = ev, ref
    if es == ee and rs == re_:
        return "BEFORE" if es < rs else "AFTER" if es > rs else "SIMULTANEOUS"
    if es != ee and rs == re_:
        return "AFTER" if rs < es else "BEFORE" if rs > ee else "INCLUDES"
    if es == ee and rs != re_:
        return "BEFORE" if es < rs else "AFTER" if es > re_ else "IS_INCLUDED"
    raise Unparsed("interval-interval relation")


def v_relation(r):
    q = r["question"].split("Choices:")[0].strip()
    m = re.search(r"What is the relationship between the event '([^']+)' and the time '([^']+)'\?", q)
    if m:
        body = q[: m.start()]
        dl = re.search(r"[A-Z][A-Z .'-]+, [^()]{4,30}\([^)]*\)\.\s*$", body)
        if not dl:
            raise Unparsed("dateline")
        sent = body[: dl.start()]
        if m.group(1) not in sent:
            raise Unparsed("event verb not in sentence")
        t = find_dates(m.group(2), years=True)
        if len(t) != 1:
            raise Unparsed("time")
        exp = _relate(_event_extent(sent), (t[0].lo, t[0].hi))
    else:
        m = re.match(r"(.*?\d{4})\.\s+(.*?\d{4})\.\s+What is the relationship between the events\?", q, re.S)
        if not m:
            raise Unparsed("event-event shape")
        exp = _relate(_event_extent(m.group(1)), _event_extent(m.group(2)))
    return exp, lambda g, e: g.strip().upper() == e


def v_extract(r):
    q, ctx = r["question"], r.get("context", "")
    opts = options(q)
    if not opts:
        raise Unparsed("extract options")
    ctx_dates = {s.lo for s in find_dates(ctx, years=False) if s.kind == "full"}
    exp = []
    for k in sorted(opts):
        d = parse_single_date(opts[k])
        if d is None:
            raise Unparsed(f"extract option {opts[k]!r}")
        if d in ctx_dates:
            exp.append(k)
    if not exp:
        raise Undet("no option is mentioned in the context")
    return "  ".join(exp), lambda g, e: sorted(g.split()) == sorted(e.split())


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def gold_of(r: dict) -> str:
    t = r.get("targets") or [""]
    if isinstance(t, str):
        import ast
        try:
            t = ast.literal_eval(t)
        except (ValueError, SyntaxError):
            return t
    return str(t[0]) if isinstance(t, list) and t else str(t)


def v_seq(r):
    """AUG_SEQ: 'Sort the following N facts from earliest to latest' with a
    stated year per fact; letters in order."""
    q = r["question"]
    items = re.findall(r"^([A-E])\.\s+(.+)$", q, re.M)
    if len(items) < 2:
        raise Unparsed("seq shape")
    yrs = {}
    for k, t in items:
        ys = re.findall(r"\b(1[0-9]\d\d|20\d\d)\b", t)
        if len(set(ys)) != 1:
            raise Unparsed("seq item year")
        yrs[k] = int(ys[0])
    if len(set(yrs.values())) != len(yrs):
        raise Undet("two facts share a year")
    exp = ",".join(sorted(yrs, key=lambda k: yrs[k]))
    return exp, lambda g, e: g.replace(" ", "") == e


def verify_row(r: dict, source: str) -> tuple[str, str]:
    """('pass'|'wrong'|'undeterminable'|'unparsed', detail)."""
    cat = r.get("category", "")
    ctx = r.get("context", "") or ""
    template = bool(TEMPLATE_TITLES.search(ctx)) or source == "AUG_PROG"
    gold = gold_of(r)
    try:
        if source == "v12_seq":
            exp, ok = v_seq(r)
        elif cat == "prog_arith_clock":
            exp, ok = v_clock(r)
        elif cat == "prog_arith_month":
            exp, ok = v_month(r)
        elif cat in ("Computation", "prog_computation"):
            exp, ok = v_computation(r, template)
        elif cat in ("Duration_Compare", "prog_duration_compare"):
            good, info = v_duration_compare(r, template)
            return ("pass" if good else "wrong"), info
        elif cat in ("Order_Compare", "prog_order_compare"):
            good, info = v_order_compare(r, template)
            return ("pass" if good else "wrong"), info
        elif cat in ("Timeline", "prog_timeline"):
            exp, ok = v_timeline(r, template)
        elif cat == "Relative_Reasoning":
            out = v_relative(r, template)
            if isinstance(out[0], bool):
                return ("pass" if out[0] else "wrong"), out[1]
            exp, ok = out
        elif cat == "prog_ordering_seq":
            exp, ok = v_ordering_seq(r)
        elif cat == "prog_ordering_tf":
            exp, ok = v_ordering_tf(r)
        elif cat == "prog_relation":
            exp, ok = v_relation(r)
        elif cat == "prog_extract":
            exp, ok = v_extract(r)
        else:
            return "unparsed", f"no verifier for {cat}"
    except Undet as e:
        return "undeterminable", str(e)
    except Unparsed as e:
        return "unparsed", str(e)
    return ("pass" if ok(gold, exp) else "wrong"), f"expected {exp!r}"


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/manual_aug_glm_v13/*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("category") in MATH:
                    rows.append(("AUG_TPL3", r))
    scripted = set()
    for n in range(271, 314):
        p = ROOT / f"data/glm_raw/{n:03d}.txt"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "question" in j:
                    scripted.add(j["question"].strip())
    for split in ("train", "val"):
        for line in open(ROOT / f"data/combined_80_20_v12/{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            q = r["question"].strip()
            if r.get("category") in MATH and q in scripted:
                rows.append(("v12_script_built", r))
            elif r.get("source_dataset") == "AUG_GLM2" and r.get("category") in MATH:
                rows.append(("v12_glm2", r))
            elif r.get("source_dataset") == "AUG_SEQ":
                rows.append(("v12_seq", r))
    for f in sorted(glob.glob(str(ROOT / "data/prog_aug_v13/prog_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                rows.append(("AUG_PROG", json.loads(line)))
    return rows


def main() -> int:
    rows = load_rows()
    stats = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))
    failed, unparsed = [], []
    for source, r in rows:
        key = f"{source}/{r.get('category')}"
        verdict, detail = verify_row(r, source)
        stats[key]["checked"] += 1
        stats[key][verdict] += 1
        if verdict in ("wrong", "undeterminable"):
            failed.append(r["question"].strip())
        elif verdict == "unparsed":
            unparsed.append(r["question"].strip())
        if verdict != "pass" and len(examples[key][verdict]) < 5:
            examples[key][verdict].append({"question": r["question"][:400],
                                           "gold": gold_of(r), "detail": detail})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = sorted(set(failed))
    (OUT_DIR / "failed_questions.json").write_text(json.dumps(failed, indent=1, ensure_ascii=False))
    # rows this parser could not read go to the human-style review (make_review_packets.py)
    (OUT_DIR / "unparsed_questions.json").write_text(
        json.dumps(sorted(set(unparsed)), indent=1, ensure_ascii=False))
    report = {"method": __doc__.split("\n\n")[0],
              "thresholds": {"order_compare_same_max_days": OC_SAME_MAX,
                             "duration_compare_hard_same": DC_SAME_HARD,
                             "duration_compare_hard_diff": DC_DIFF_HARD},
              "failed_total": len(failed),
              "by_source_category": {
                  k: {"checked": c["checked"], "passed": c["pass"],
                      "failed_wrong_gold": c["wrong"],
                      "failed_undeterminable": c["undeterminable"],
                      "unparsed": c["unparsed"],
                      "unparsed_pct": round(100 * c["unparsed"] / c["checked"], 1),
                      "examples": examples[k]}
                  for k, c in sorted(stats.items())}}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"{'source/category':42s} {'checked':>7s} {'pass':>6s} {'wrong':>6s} {'undet':>6s} {'unpars':>6s}")
    for k, c in sorted(stats.items()):
        print(f"{k:42s} {c['checked']:7d} {c['pass']:6d} {c['wrong']:6d} {c['undeterminable']:6d} {c['unparsed']:6d}")
    print(f"\nfailed (wrong + undeterminable): {len(failed)} -> {OUT_DIR / 'failed_questions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
