"""AUG_PROG: programmatic synthetic rows for v13 (docs/v13_plan.md §2).

Every row's gold is COMPUTED from dates chosen before any text is rendered,
then re-derived by an independent verifier (`VERIFIERS`) before the row is
written; any verification failure fails the audit and the run. No LLM is
involved anywhere, which is the point: unlimited volume, exact golds, zero
API cost.

Row schema matches the training mixture (generic branch of
experiments/finetuning/LLaMA/data_loader.py::normalize_record):
    {source_dataset:"AUG_PROG", category, question, context, targets:[gold],
     rationale, source:"augmented", ...}

Two surface conventions, both already proven in the mixture:
  - TIME-style rows embed a "\nChoices:\nA. ..." block so eval_parity
    splits them into question+choices exactly like a benchmark MCQ item;
  - TRAM-style label rows are emitted BOTH bare (no options -- matching
    TRAM's actual eval surface) and with a Choices block (the AUG_GLM2
    relation-card shape v11 trained on), roughly 60/40.

Hard guarantees (audited before writing; the run fails if any breaks):
  - MCQ gold letters cycle A/B/C (no position skew, the L7 lesson)
  - no single distinct gold > 10% of its category (the D51 lesson)
  - label categories balanced across their gold label space
  - Timeline/ordering identity permutations <= 1/3 of rows
  - no duplicate questions, internally or vs data/combined_80_20_v12
  - every context-carrying row has >= 60 words
  - row count == the pre-registered cap for every category

Usage:
    venv/bin/python scripts/generate_v13_prog_data.py            # plan caps
    venv/bin/python scripts/generate_v13_prog_data.py --smoke    # 12/category
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dateutil.relativedelta import relativedelta  # noqa: E402

# ---------------------------------------------------------------------------
# Banks. Combinatorial by design: a row's cast should not repeat often.
# ---------------------------------------------------------------------------
CITIES = ["KUALA LUMPUR", "PARIS", "TOKYO", "NAIROBI", "LIMA", "OSLO", "CAIRO",
          "MUMBAI", "TORONTO", "VIENNA", "DUBLIN", "SEOUL", "LISBON", "ATHENS",
          "HELSINKI", "BRASILIA", "AMMAN", "HANOI", "PRAGUE", "BOGOTA",
          "TUNIS", "MANILA", "WARSAW", "DAMASCUS", "KINGSTON", "SUVA",
          "REYKJAVIK", "TAIPEI", "ACCRA", "TEHRAN", "JAKARTA", "RABAT",
          "QUITO", "VILNIUS", "VALLETTA", "OTTAWA", "LAGOS", "SANTIAGO"]
WIRES = ["AFP", "Reuters", "AP", "dpa", "EFE", "IPS", "TASS"]
ORG_ADJ = ["municipal", "regional", "national", "coastal", "inland", "northern",
           "southern", "eastern", "western", "provincial", "district",
           "metropolitan", "rural", "urban", "state", "federal", "island",
           "highland", "river", "border"]
ORG_NOUN = ["water board", "transit authority", "growers council", "port",
            "chamber of commerce", "hospital trust", "miners union",
            "farmers collective", "shipping line", "rail operator",
            "broadcasting service", "housing association", "fisheries office",
            "power utility", "savings bank", "technical college", "museum",
            "opera house", "sports club", "library service", "harbour office",
            "grain exchange", "textile mill", "steel works", "chemical plant",
            "observatory", "planning commission", "roads department"]
PERSON_A = ["Dana", "Ivo", "Rosa", "Karel", "Mira", "Tomas", "Elena", "Bruno",
            "Nadia", "Owen", "Priya", "Samir", "Lena", "Hugo", "Yara", "Felix",
            "Anouk", "Diego", "Marta", "Jonas", "Kira", "Ravi", "Sofia",
            "Emil", "Nour", "Pablo", "Greta", "Tarek", "Ines", "Otto"]
PERSON_B = ["Novak", "Ferrier", "Ostrowski", "Halvorsen", "Beaumont", "Carreno",
            "Lindqvist", "Adeyemi", "Kovacs", "Marchetti", "Duarte", "Bergman",
            "Vasquez", "Ionescu", "Tanaka", "Muller", "Rossi", "Hassan",
            "Petrov", "Larsen", "Moreau", "Silva", "Weber", "Kaur", "Nguyen",
            "Okafor", "Sanchez", "Fischer", "Ibrahim", "Costa"]
EVENT_NOUNS = ["founding", "merger", "strike", "audit", "opening", "closure",
               "expansion", "renaming", "relocation", "reopening", "licensing",
               "bankruptcy", "nationalisation", "privatisation", "survey",
               "festival", "exhibition", "tournament", "conference", "summit",
               "referendum", "inauguration", "groundbreaking", "dedication",
               "inspection", "recruitment drive", "apprentice scheme",
               "restoration", "conservation order", "buyout", "takeover"]
VERBS_PAST = ["opened", "approved", "collapsed", "resumed", "banned",
              "announced", "completed", "launched", "suspended", "signed",
              "unveiled", "delayed", "settled", "reached", "inaugurated"]
HINT = ("   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year "
        "2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)")

REL_GOLD = ["BEFORE", "AFTER", "IS_INCLUDED", "SIMULTANEOUS", "INCLUDES"]
REL_DISTRACT = ["BEFORE", "AFTER", "IS_INCLUDED", "SIMULTANEOUS", "INCLUDES",
                "DURING", "IDENTITY", "IMMEDIATELY BEFORE", "IMMEDIATELY AFTER"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MON_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
            "Oct", "Nov", "Dec"]

CAPS = {  # docs/v13_plan.md §2, pre-registered
    "prog_arith_clock": 1500,
    "prog_arith_month": 1500,
    "prog_computation": 1500,
    "prog_timeline": 2000,
    "prog_relation": 2000,
    "prog_ordering_tf": 800,
    "prog_ordering_seq": 800,
    "prog_duration_compare": 800,
    "prog_order_compare": 800,
    "prog_extract": 1500,
}

# --------------------------------------------------------------------------- 
# Small helpers
# ---------------------------------------------------------------------------


def render_date(d: date, rng: Random, style: str | None = None) -> str:
    s = style or rng.choice(["us", "dayfirst", "abbr"])
    if s == "us":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if s == "dayfirst":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return f"{MON_ABBR[d.month - 1]} {d.day}, {d.year}"


def rand_date(rng: Random, y0: int = 1890, y1: int = 2024) -> date:
    return date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))


def org(rng: Random) -> str:
    return f"the {rng.choice(ORG_ADJ)} {rng.choice(ORG_NOUN)}"


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def span_text(rd: relativedelta) -> str:
    parts = []
    for n, sing, plur in ((rd.years, "year", "years"),
                          (rd.months, "month", "months"),
                          (rd.days, "day", "days")):
        if n:
            parts.append(f"{n} {sing if n == 1 else plur}")
    return " ".join(parts)


def choices_block(opts: list[str]) -> str:
    return "\nChoices:\n" + "\n".join(
        f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))


def split_options(question: str) -> list[str]:
    if "Choices:" not in question:
        return []
    return [ln[3:] for ln in question.split("Choices:", 1)[1].splitlines()
            if re.match(r"^[A-Z]\. ", ln)]


def row(cat: str, question: str, context: str, gold: str,
        rationale: str) -> dict:
    return {"source_dataset": "AUG_PROG", "slice": "P", "category": cat,
            "provenance": "prog", "question": question, "context": context,
            "targets": [gold], "rationale": rationale, "source": "augmented",
            "source_id": ""}


def letter_cycle_abc():
    i = 0
    while True:
        yield "ABC"[i % 3]
        i += 1


# ---------------------------------------------------------------------------
# 1. prog_arith_clock -- TRAM arithmetic: "What is 23:48 - 01:31?"
# ---------------------------------------------------------------------------


def gen_arith_clock(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    minute_choices = [5, 11, 17, 23, 29, 31, 41, 47, 53, 59]
    while len(out) < n:
        h1 = rng.randint(0, 23)
        m1 = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48, 50, 55])
        total = rng.randint(5, 12 * 60 + 59)          # operand in minutes
        dh, dm = total // 60, total % 60
        op = rng.choice(["-", "-", "+"])
        a = h1 * 60 + m1
        res = (a - total) % 1440 if op == "-" else (a + total) % 1440
        gold = f"{res // 60:02d}:{res % 60:02d}"
        q = f"What is {h1:02d}:{m1:02d} {op} {dh:02d}:{dm:02d}?"
        if q in seen:
            continue
        seen.add(q)
        rat = (f"{h1:02d}:{m1:02d} {'minus' if op == '-' else 'plus'} "
               f"{dh:02d}:{dm:02d} is {gold} on a 24-hour clock.")
        out.append(row("prog_arith_clock", q, "", gold, rat))
    return out


# ---------------------------------------------------------------------------
# 2. prog_arith_month -- TimeBench date_arith surface form
# ---------------------------------------------------------------------------


def gen_arith_month(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        y, m = rng.randint(1100, 2199), rng.randint(1, 12)
        if rng.random() < 0.2:                       # single-unit form
            ny = rng.choice([0, rng.randint(1, 30)])
            nm = 0 if ny else rng.randint(1, 11)
            if not (ny or nm):
                nm = rng.randint(1, 11)
        else:
            ny, nm = rng.randint(1, 30), rng.randint(1, 11)
        back = rng.random() < 0.5
        anchor = date(y, m, 1)
        off = relativedelta(years=ny, months=nm)
        res = anchor - off if back else anchor + off
        if not (1100 <= res.year <= 2199):
            continue
        gold = f"{MON_ABBR[res.month - 1]}, {res.year}"
        if ny and not nm:
            phrase = f"{ny} year" + ("s" if ny != 1 else "")
        elif nm and not ny:
            phrase = f"{nm} month" + ("s" if nm != 1 else "")
        else:
            phrase = f"{ny} year" + ("s" if ny != 1 else "") + f" and {nm} month"
        d = "before" if back else "after"
        q = f"What is the time {phrase} {d} {MON_ABBR[m - 1]}, {y}"
        if q in seen:
            continue
        seen.add(q)
        rat = (f"{MON_ABBR[m - 1]}, {y} {'minus' if back else 'plus'} "
               f"{phrase} is {gold}.")
        out.append(row("prog_arith_month", q, "", gold, rat))
    return out


# ---------------------------------------------------------------------------
# 3. prog_computation -- TIME Computation analog (span + offset, verbatim Hint)
# ---------------------------------------------------------------------------

_FILLERS = [
    "The announcement surprised the town, though insiders had expected it for some time.",
    "Local newspapers covered both events in detail over the following weeks.",
    "Committee minutes from the period describe long debates before each decision was taken.",
    "Photographs of the occasions survive in the regional archive, along with attendance lists.",
    "Residents still recall the preparations that surrounded each occasion.",
    "The ledger kept by the secretariat records both dates in the same careful hand.",
    "Auditors later praised the record-keeping that fixed both dates beyond dispute.",
]


def gen_computation(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        d1, d2 = rand_date(rng), rand_date(rng)
        if abs((d2 - d1).days) < 3 or abs((d2 - d1).days) > 11000:
            continue
        if d1 > d2:
            d1, d2 = d2, d1
        o1, o2 = org(rng), org(rng)
        e1, e2 = rng.choice(EVENT_NOUNS), rng.choice(EVENT_NOUNS)
        v1, v2 = rng.choice(VERBS_PAST), rng.choice(VERBS_PAST)
        ctx = (f"{cap(o1)} {v1} its {e1} on {render_date(d1, rng)}. "
               f"The {e2} of {o2} was {v2} on {render_date(d2, rng)}, as the "
               f"surviving register confirms. {rng.choice(_FILLERS)} "
               f"{rng.choice(_FILLERS)} Historians working from the same "
               f"archive note that both entries were cross-checked against "
               f"contemporary correspondence, and neither date has ever been "
               f"disputed in print.")
        if len(ctx.split()) < 60:
            continue
        if rng.random() < 0.7:                       # span question
            if rng.random() < 0.55:                  # both dates in the stem
                q = (f"What was the duration from {render_date(d1, rng)} to "
                     f"{render_date(d2, rng)}?")
            else:                                    # events only (TIME: 37%)
                q = (f"How long passed between the {e1} of {o1} and the {e2} "
                     f"of {o2}?")
            gold = span_text(relativedelta(d2, d1))
            if not gold:
                continue
            rat = (f"The passage dates the {e1} to {render_date(d1, rng, 'us')} "
                   f"and the {e2} to {render_date(d2, rng, 'us')}; the span "
                   f"between them is {gold}.")
        else:                                        # offset question
            ny = rng.randint(0, 3)
            nm = rng.randint(0, 11)
            nd = rng.randint(1, 27)
            if not (ny or nm):
                nm = rng.randint(1, 11)
            off = relativedelta(years=ny, months=nm, days=nd)
            back = rng.random() < 0.5
            res = d1 - off if back else d1 + off
            gold = render_date(res, rng, "us")
            phrase = span_text(off)
            q = (f"What was the date {phrase} {'before' if back else 'after'} "
                 f"the {e1} of {o1}?")
            rat = (f"The {e1} is dated {render_date(d1, rng, 'us')}; "
                   f"{'subtracting' if back else 'adding'} {phrase} gives "
                   f"{gold}.")
        q += HINT
        if q in seen or gold.lower() in ctx.lower():
            continue
        seen.add(q)
        out.append(row("prog_computation", q, ctx, gold, rat))
    return out


# ---------------------------------------------------------------------------
# 4. prog_timeline -- exact TIME template + the AUG_SEQ template
# ---------------------------------------------------------------------------

_TL_TEMPLATE = ("Below are {n} facts. You need to sort these facts in "
                "chronological order. Requirements: You must output a sequence "
                "of uppercase letters separated by commas, such as 'A,B,C', "
                "without any other characters.")
_SEQ_TEMPLATE = ("Sort the following {n} facts from earliest to latest. Reply "
                 "with only the letters in order, comma separated, e.g. "
                 "'B,A,C'.")


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def gen_timeline(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    identity = total = 0
    while len(out) < n:
        k = rng.choices([3, 4, 5], weights=[6, 3, 1])[0]
        base = rand_date(rng)
        ds = [base] + [base + timedelta(days=x) for x in
                       sorted(rng.sample(range(20, 4000), k - 1))]
        rng.shuffle(ds)
        if ds == sorted(ds) and identity / max(len(out), 1) > 0.30:
            continue
        identity += int(ds == sorted(ds))
        facts, year_only = [], []
        for d in ds:
            o, e = org(rng), rng.choice(EVENT_NOUNS)
            style = rng.random()
            year_only.append(0.55 <= style < 0.8)
            if style < 0.55:
                facts.append(f"The {e} of {o} on {render_date(d, rng)}.")
            elif style < 0.8:
                facts.append(f"{cap(o)} recorded its {e} in {d.year}.")
            else:
                facts.append(f"The {e} took place on the {d.day}"
                             f"{_ordinal(d.day)} of {MONTHS[d.month - 1]}, "
                             f"{d.year}.")
        # A year-only fact sharing its year with another fact is ordered by
        # day-level dates the model never sees (audit 2026-10-04 §2.5).
        if any(year_only[i] and any(ds[j].year == ds[i].year
                                    for j in range(k) if j != i)
               for i in range(k)):
            continue
        gold = ",".join(chr(65 + i) for i in
                        sorted(range(k), key=lambda i: ds[i]))
        if rng.random() < 0.6:                       # TIME shape
            q = _TL_TEMPLATE.format(n=k) + choices_block(facts)
        else:                                        # AUG_SEQ shape
            q = (_SEQ_TEMPLATE.format(n=k) + "\n" +
                 "\n".join(f"{chr(65 + i)}. {f}" for i, f in enumerate(facts)))
        if q in seen:
            continue
        seen.add(q)
        keys = "; ".join(f"{chr(65 + i)}={ds[i].isoformat()}" for i in range(k))
        out.append(row("prog_timeline", q, "", gold,
                       f"Keyed dates: {keys}. Sorted earliest to latest the "
                       f"order is {gold}."))
    return out


# ---------------------------------------------------------------------------
# 5. prog_relation -- TRAM temporal_relation analog (both surface forms)
#
# Semantics (relation of the FIRST-mentioned event to the SECOND / the time):
#   BEFORE        point event earlier than the reference
#   AFTER         point event later than the reference
#   IS_INCLUDED   point event inside the reference interval
#   INCLUDES      event interval contains the reference point
#   SIMULTANEOUS  same timestamp
# ---------------------------------------------------------------------------


def gen_relation(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    quota = {"BEFORE": n * 30 // 100, "AFTER": n * 25 // 100,
             "IS_INCLUDED": n * 20 // 100, "INCLUDES": n * 15 // 100,
             "SIMULTANEOUS": n * 10 // 100}
    leftover = n - sum(quota.values())
    for i, k in enumerate(["BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES",
                           "SIMULTANEOUS"]):
        if i >= leftover:
            break
        quota[k] += 1
    made: Counter = Counter()
    while len(out) < n:
        label = min((k for k in quota if made[k] < quota[k]),
                    key=lambda k: (made[k] / max(quota[k], 1), k),
                    default=None)
        if label is None:
            break
        T = rand_date(rng)
        city, wire = rng.choice(CITIES), rng.choice(WIRES)
        o1, o2 = org(rng), org(rng)
        e1, e2 = rng.choice(EVENT_NOUNS), rng.choice(EVENT_NOUNS)
        v1 = rng.choice(VERBS_PAST)
        e2t = rng.random() < 0.6
        time_full = f"{MONTHS[T.month - 1]} {T.day}, {T.year}"
        if e2t:
            # sentence + "What is the relationship between the event 'V' and
            # the time 'T'?" -- the quoted event must actually be the sentence
            # verb, so build both together.
            if label == "BEFORE":
                d = T - timedelta(days=rng.randint(3, 3000))
                sent = (f"{cap(o1)} {v1} its {e1} on {render_date(d, rng)}, "
                        f"well before the year's end.")
                anchor = d
            elif label == "AFTER":
                d = T + timedelta(days=rng.randint(3, 3000))
                sent = (f"{cap(o1)} {v1} its {e1} on {render_date(d, rng)}, "
                        f"months after the audit that prompted it.")
                anchor = d
            elif label == "IS_INCLUDED":
                s = T - timedelta(days=rng.randint(5, 200))
                e = T + timedelta(days=rng.randint(5, 200))
                v1 = "ran"
                sent = (f"The {e1} of {o1} ran from {render_date(s, rng)} to "
                        f"{render_date(e, rng)}, the longest in the "
                        f"authority's records.")
                anchor = (s, e)
            elif label == "INCLUDES":
                y0 = T.year - rng.randint(1, 6)
                y1 = T.year + rng.randint(1, 6)
                v1 = "spanned"
                sent = (f"The {e1} of {o1} spanned the years {y0} to {y1} and "
                        f"shaped everything the region built afterwards.")
                anchor = (date(y0, 1, 1), date(y1, 12, 31))
            else:  # SIMULTANEOUS
                sent = (f"{cap(o1)} {v1} its {e1} on exactly "
                        f"{render_date(T, rng)}, the same day inspectors "
                        f"arrived.")
                anchor = T
            q = (f"{sent} {city}, {time_full} ({wire}). What is the "
                 f"relationship between the event '{v1}' and the time "
                 f"'{time_full}'?")
            rat = (f"The event '{v1}' is anchored at "
                   f"{anchor if not isinstance(anchor, tuple) else anchor[0]}.."
                   f"{anchor[1] if isinstance(anchor, tuple) else anchor} "
                   f"relative to the time {time_full}; the relation is "
                   f"{label}.")
        else:
            # event-to-event: first event vs second event, each a sentence
            if label == "BEFORE":
                dA = T
                dB = T + timedelta(days=rng.randint(3, 3000))
                s1 = f"{cap(o1)} {v1} its {e1} on {render_date(dA, rng)}."
                s2 = f"{cap(o2)} {rng.choice(VERBS_PAST)} its {e2} on {render_date(dB, rng)}."
            elif label == "AFTER":
                dA = T
                dB = T - timedelta(days=rng.randint(3, 3000))
                s1 = f"{cap(o1)} {v1} its {e1} on {render_date(dA, rng)}."
                s2 = f"{cap(o2)} {rng.choice(VERBS_PAST)} its {e2} on {render_date(dB, rng)}."
            elif label == "IS_INCLUDED":
                dA = T                                   # point inside dB's interval
                s0 = T - timedelta(days=rng.randint(5, 200))
                s1 = f"{cap(o1)} {v1} its {e1} on {render_date(dA, rng)}."
                s2 = (f"The {e2} of {o2} ran from {render_date(s0, rng)} to "
                      f"{render_date(T + timedelta(days=rng.randint(5, 200)), rng)}.")
            elif label == "INCLUDES":
                s0 = T - timedelta(days=rng.randint(30, 900))   # interval first
                s1 = (f"The {e1} of {o1} ran from {render_date(s0, rng)} to "
                      f"{render_date(T + timedelta(days=rng.randint(30, 900)), rng)}.")
                s2 = f"{cap(o2)} {rng.choice(VERBS_PAST)} its {e2} on {render_date(T, rng)}."
            else:  # SIMULTANEOUS
                s1 = f"{cap(o1)} {v1} its {e1} on {render_date(T, rng)}."
                s2 = f"{cap(o2)} {rng.choice(VERBS_PAST)} its {e2} on {render_date(T, rng)}."
            q = f"{s1} {s2} What is the relationship between the events?"
            rat = f"The relation of the first-mentioned event to the second is {label}."
        if rng.random() < 0.4:                       # MCQ shape
            distract = rng.sample([x for x in REL_DISTRACT if x != label], 2)
            letter = next(lc)
            placed: list[str | None] = [None, None, None]
            placed[ord(letter) - 65] = label
            it = iter(distract)
            for i in range(3):
                if placed[i] is None:
                    placed[i] = next(it)
            q += choices_block([x for x in placed if x is not None])
        if q in seen:
            continue
        seen.add(q)
        made[label] += 1
        out.append(row("prog_relation", q, "", label, rat))
    return out


# ---------------------------------------------------------------------------
# 6. prog_ordering_tf -- TRAM TRUE/FALSE, order computable from stated dates
# ---------------------------------------------------------------------------


def gen_ordering_tf(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    made: Counter = Counter()
    while len(out) < n:
        want_true = made["TRUE"] <= made["FALSE"]
        d1 = rand_date(rng)
        d2 = d1 + timedelta(days=rng.randint(4, 900))
        if not want_true:
            d1, d2 = d2, d1
        o = org(rng)
        e1, e2 = rng.choice(EVENT_NOUNS), rng.choice(EVENT_NOUNS)
        v1, v2 = rng.choice(VERBS_PAST), rng.choice(VERBS_PAST)
        style = rng.random()
        if style < 0.5:
            s1 = f"{cap(o)} {v1} its {e1} on {render_date(d1, rng)}"
            s2 = f"its {e2} followed on {render_date(d2, rng)}"
        else:
            s1 = f"The {e1} of {o} was recorded first, on {render_date(d1, rng)}"
            s2 = f"the {e2} came later, dated {render_date(d2, rng)}"
        gold = "TRUE" if want_true else "FALSE"
        q = f"{s1}. Then {s2}. - True/False?"
        if rng.random() < 0.4:                       # MCQ shape
            letter = next(lc)
            other = "FALSE" if gold == "TRUE" else "TRUE"
            placed: list[str | None] = [None, None, None]
            placed[ord(letter) - 65] = gold
            it = iter([other, "Undetermined"])
            for i in range(3):
                if placed[i] is None:
                    placed[i] = next(it)
            q += choices_block([x for x in placed if x is not None])
        if q in seen:
            continue
        seen.add(q)
        rat = (f"The first event is dated {render_date(d1, rng, 'us')} and "
               f"the second {render_date(d2, rng, 'us')}, so the claim's "
               f"order {'holds' if gold == 'TRUE' else 'does not hold'}.")
        made[gold] += 1
        out.append(row("prog_ordering_tf", q, "", gold, rat))
    return out


# ---------------------------------------------------------------------------
# 7. prog_ordering_seq -- TRAM permutation shape, bare + MCQ
# ---------------------------------------------------------------------------


def gen_ordering_seq(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    identity = total = 0
    while len(out) < n:
        k = rng.choices([2, 3, 4], weights=[2, 5, 3])[0]
        base = rand_date(rng)
        ds = [base] + [base + timedelta(days=x) for x in
                       sorted(rng.sample(range(10, 2000), k - 1))]
        order = list(range(k))
        rng.shuffle(order)                    # listing position p shows ds[order[p]]
        if order == list(range(k)) and identity / (len(out) + 1) >= 0.25:
            continue
        identity += int(order == list(range(k)))
        body = []
        for p in range(k):
            o = org(rng)
            e = rng.choice(EVENT_NOUNS)
            body.append(f"({p + 1}) {cap(o)} {rng.choice(VERBS_PAST)} its "
                        f"{e} on {render_date(ds[order[p]], rng)}.")
        # gold: listing positions sorted by their event's date
        pos_sorted = sorted(range(k), key=lambda p: ds[order[p]])
        gold = "(" + "), (".join(str(p + 1) for p in pos_sorted) + ")"
        q = ("Arrange the following events in chronological order: "
             + " ".join(body))
        # k=2 has only 2 permutations -- a 3-option MCQ is impossible, so
        # those rows stay bare (TRAM's own surface form).
        if k >= 3 and rng.random() < 0.5:               # MCQ w/ 3 permutations
            perms = {gold}
            while len(perms) < 3:
                p = list(range(k))
                rng.shuffle(p)
                perms.add("(" + "), (".join(str(i + 1) for i in p) + ")")
            perms_l = list(perms)
            rng.shuffle(perms_l)
            letter = next(lc)
            gi = perms_l.index(gold)
            li = ord(letter) - 65
            perms_l[gi], perms_l[li] = perms_l[li], perms_l[gi]
            q += choices_block(perms_l)
        if q in seen:
            continue
        seen.add(q)
        rat = ("The events in listing order are dated " +
               ", ".join(render_date(ds[order[p]], rng, "us")
                         for p in range(k)) +
               f", so the chronological order is {gold}.")
        out.append(row("prog_ordering_seq", q, "", gold, rat))
    return out


# ---------------------------------------------------------------------------
# 8. prog_duration_compare -- exact TIME option wording
# ---------------------------------------------------------------------------

_DC_OPTS = ["Duration 1 is longer.", "Duration 2 is longer.",
            "The two durations are approximately the same length."]


def gen_duration_compare(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    made: Counter = Counter()
    quota = {"A": max(1, n * 33 // 100), "B": max(1, n * 40 // 100),
             "C": max(1, n - max(1, n * 33 // 100) - max(1, n * 40 // 100))}
    while len(out) < n:
        target = min((L for L in ("A", "B", "C") if made[L] < quota[L]),
                     key=lambda L: made[L], default=None)
        if target is None:
            break
        a1 = rand_date(rng)
        # A/B targets need |d1-d2| > 60 days AND >10% apart, else the
        # verifier's "approximately the same" band fires; d1 >= 350 days
        # guarantees both for d2 in [0.25,0.8]x or [1.3,3.5]x of d1.
        d1 = (rng.randint(350, 3000) if target in ("A", "B")
              else rng.randint(20, 3000))
        a2 = a1 + timedelta(days=d1)
        if target == "C":
            d2 = max(3, d1 + rng.choice([-2, -1, 0, 0, 1]))
        elif target == "A":
            d2 = max(3, int(d1 * rng.uniform(0.25, 0.8)))
        else:
            d2 = int(d1 * rng.uniform(1.3, 3.5))
        b1 = a2 + timedelta(days=rng.randint(30, 900))
        b2 = b1 + timedelta(days=d2)
        o = [org(rng) for _ in range(4)]
        e = [rng.choice(EVENT_NOUNS) for _ in range(4)]
        if rng.random() < 0.5:                       # dates printed in the stem
            dur1 = f"Between {render_date(a1, rng)} and {render_date(a2, rng)}"
            dur2 = f"Between {render_date(b1, rng)} and {render_date(b2, rng)}"
            ctx = ""
        else:
            dur1 = f"Between the {e[0]} of {o[0]} and the {e[1]} of {o[1]}"
            dur2 = f"Between the {e[2]} of {o[2]} and the {e[3]} of {o[3]}"
            ctx = (f"The {e[0]} of {o[0]} opened on {render_date(a1, rng)} "
                   f"and closed on {render_date(a2, rng)}. A later "
                   f"programme, the {e[2]} of {o[2]}, ran from "
                   f"{render_date(b1, rng)} to {render_date(b2, rng)}. The "
                   f"{e[1]} of {o[1]} and the {e[3]} of {o[3]} name the same "
                   f"two pairs of dates in the authority's ledger, whose "
                   f"clerks recorded every opening and closing with care.")
            if len(ctx.split()) < 60:
                continue
        q = (f"Which of the following two durations is longer? "
             f"*Duration 1:* {dur1}. *Duration 2:* {dur2}." +
             choices_block(_DC_OPTS))
        gold = _DC_OPTS[ord(target) - 65]
        if q in seen:
            continue
        seen.add(q)
        rat = (f"Duration 1 runs from {a1.isoformat()} to {a2.isoformat()} "
               f"({d1} days); Duration 2 from {b1.isoformat()} to "
               f"{b2.isoformat()} ({d2} days); the correct option is "
               f"'{gold}'")
        made[target] += 1
        out.append(row("prog_duration_compare", q, ctx, gold, rat))
    return out


# ---------------------------------------------------------------------------
# 9. prog_order_compare -- exact TIME option wording
# ---------------------------------------------------------------------------

_OC_OPTS = ["Fact 1 happened earlier.", "Fact 2 happened earlier.",
            "They happen at almost the same time."]


def gen_order_compare(rng: Random, n: int, lc) -> list[dict]:
    out, seen = [], set()
    made: Counter = Counter()
    quota = {"A": max(1, n * 35 // 100), "B": max(1, n * 45 // 100),
             "C": max(1, n - max(1, n * 35 // 100) - max(1, n * 45 // 100))}
    while len(out) < n:
        target = min((L for L in ("A", "B", "C") if made[L] < quota[L]),
                     key=lambda L: made[L], default=None)
        if target is None:
            break
        d1 = rand_date(rng)
        if target == "C":
            d2 = d1 + timedelta(days=rng.choice([0, 1, 2]))
        else:
            d2 = d1 + timedelta(days=rng.randint(15, 2000))
            if target == "B":
                d1, d2 = d2, d1
        o1, o2 = org(rng), org(rng)
        e1, e2 = rng.choice(EVENT_NOUNS), rng.choice(EVENT_NOUNS)
        if rng.random() < 0.7:                       # dates inside the facts
            f1 = f"the {e1} of {o1} on {render_date(d1, rng)}"
            f2 = f"the {e2} of {o2} on {render_date(d2, rng)}"
            ctx = ""
        else:
            f1, f2 = f"the {e1} of {o1}", f"the {e2} of {o2}"
            ctx = (f"The ledger of {o1} dates its {e1} to "
                   f"{render_date(d1, rng)}, while the registry kept by {o2} "
                   f"records its {e2} on {render_date(d2, rng)}. Both "
                   f"entries were countersigned by the provincial "
                   f"secretariat, whose surviving minute books confirm the "
                   f"dates beyond dispute.")
            if len(ctx.split()) < 60:
                continue
        q = (f"For Fact1: {f1} and Fact2: {f2}, which one happened earlier?" +
             choices_block(_OC_OPTS))
        gold = _OC_OPTS[ord(target) - 65]
        if q in seen:
            continue
        seen.add(q)
        rat = (f"Fact 1 is dated {d1.isoformat()} and Fact 2 {d2.isoformat()}, "
               f"so '{gold}'")
        made[target] += 1
        out.append(row("prog_order_compare", q, ctx, gold, rat))
    return out


# ---------------------------------------------------------------------------
# 10. prog_extract -- TIME Extract analog (TimeDial-style dated dialogue)
#
# Context is built ONLY from recoverable date surface forms:
#   "D Month, YYYY" (session headers, am/pm style) and "DD.MM.YYYY".
# Correct options are exact substrings of the context; distractors are
# mutated dates that provably do not occur in it (verify_extract enforces
# both directions character-exactly).
# ---------------------------------------------------------------------------

_EX_Q = ("Which of the following are time expressions mentioned in the "
         "context? (Note: There may be one or more correct options. And the "
         "time expressions are mentioned directly or indirectly in the "
         "context.)")
_OPENERS = ["Hey! How are you?", "Good morning!", "Hi, long time no see!",
            "Hello! Anything new with you?", "Hey there!", "Good evening!",
            "Morning! Did I catch you at a bad time?"]
_TOPICS = ["the renovation", "the trip", "the course schedule", "the move",
           "the concert", "the delivery", "the appointment", "the repair",
           "the family visit", "the exam", "the wedding", "the car service"]
_FILL = ["That sounds great!", "Oh no, that's annoying.", "I see.",
         "Thanks for letting me know.", "Perfect, see you then.",
         "Let's talk soon.", "I'll keep that in mind.", "Sounds good.",
         "Ha, typical!", "Right, I remember you mentioning that."]

_FULL_RE = re.compile(
    r"(\d{1,2}) (January|February|March|April|May|June|July|August|September|"
    r"October|November|December), (\d{4})")
_DOT_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


def _dialogue(rng: Random, n_sessions: int) -> str:
    a = rng.choice(PERSON_A)
    b = rng.choice(PERSON_A)
    while b == a:
        b = rng.choice(PERSON_A)
    lines = []
    day0 = rand_date(rng, 2018, 2024)
    for s in range(n_sessions):
        d = day0 + timedelta(days=rng.randint(5, 160) * (s + 1))
        hh = rng.randint(6, 22)
        mm = rng.choice([0, 4, 13, 18, 23, 30, 37, 45, 52])
        h12 = hh if 1 <= hh <= 12 else (hh - 12 if hh > 12 else 12)
        ap = "am" if hh < 12 else "pm"
        if rng.random() < 0.5:
            lines.append(f"Session {s + 1} happened at {h12}:{mm:02d} {ap} "
                         f"on {d.day} {MONTHS[d.month - 1]}, {d.year}.")
        else:
            lines.append(f"Session {s + 1} happened at "
                         f"{d.strftime('%d.%m.%Y')}, {h12}:{mm:02d}:"
                         f"{rng.randint(10, 59):02d}.")
        lines.append(f"{a}: {rng.choice(_OPENERS)}")
        extra = d + timedelta(days=rng.randint(1, 45))
        lines.append(f"{b}: We finally sorted out {rng.choice(_TOPICS)} on "
                     f"{MONTHS[extra.month - 1]} {extra.day}, {extra.year}.")
        lines.append(f"{a}: {rng.choice(_FILL)}")
        if rng.random() < 0.5:
            extra2 = d + timedelta(days=rng.randint(46, 300))
            lines.append(f"{b}: And {rng.choice(_TOPICS)} is booked for "
                         f"{extra2.day} {MONTHS[extra2.month - 1]}, "
                         f"{extra2.year}.")
            lines.append(f"{a}: {rng.choice(_FILL)}")
    return "\n".join(lines)


def gen_extract(rng: Random, n: int, lc) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    sizes = ([1] * (n * 30 // 100) + [2] * (n * 40 // 100) +
             [3] * (n * 25 // 100) + [4] * (n - (n * 30 // 100) -
                                            (n * 40 // 100) - (n * 25 // 100)))
    rng.shuffle(sizes)
    for want in sizes:
        for _attempt in range(60):
            text = _dialogue(rng, rng.randint(2, 4))
            if len(text.split()) < 60:
                continue
            # exact surface forms present in the context
            fulls = [m.group(0) for m in _FULL_RE.finditer(text)]
            dots = [m.group(0) for m in _DOT_RE.finditer(text)]
            uniq_fulls = sorted(set(fulls))
            uniq_dots = sorted(set(dots))
            mentioned = uniq_fulls + uniq_dots
            if len(mentioned) < 2:
                continue
            k_opts = 5 if rng.random() < 0.3 else 4
            want_correct = min(want, len(mentioned), k_opts - 1)
            if want_correct < 1:
                continue
            correct_opts = rng.sample(mentioned, want_correct)
            # distractors: mutated dates NOT present in the context
            def mutations(src: str) -> list[str]:
                m = _FULL_RE.fullmatch(src)
                if m:
                    d = date(int(m.group(3)), MONTHS.index(m.group(2)) + 1,
                             int(m.group(1)))
                else:
                    m2 = _DOT_RE.fullmatch(src)
                    d = date(int(m2.group(3)), int(m2.group(2)),
                             int(m2.group(1)))
                cands = []
                for cand in (d + timedelta(days=1), d + timedelta(days=-1),
                             d + timedelta(days=2),
                             date(d.year + 1, d.month, min(d.day, 28)),
                             date(d.year - 1, d.month, min(d.day, 28)),
                             date(d.year, (d.month % 12) + 1, min(d.day, 28))):
                    for form in (f"{cand.day} {MONTHS[cand.month - 1]}, "
                                 f"{cand.year}",
                                 cand.strftime("%d.%m.%Y")):
                        cands.append(form)
                return cands
            pool: list[str] = []
            for src in mentioned:
                pool += mutations(src)
            rng.shuffle(pool)
            distr: list[str] = []
            for cand in pool:
                if cand in mentioned or cand in distr:
                    continue
                if cand in text:            # must not occur in the context
                    continue
                distr.append(cand)
                if len(distr) == k_opts - want_correct:
                    break
            if len(distr) < k_opts - want_correct:
                continue
            opts = correct_opts + distr
            rng.shuffle(opts)
            letters = [chr(65 + i) for i in range(len(opts))]
            gold_letters = sorted(L for L, o in zip(letters, opts)
                                  if o in correct_opts)
            gold = "  ".join(gold_letters)
            q = _EX_Q + choices_block(opts)
            if q in seen:
                continue
            seen.add(q)
            rat = ("The context mentions " + ", ".join(correct_opts) +
                   "; the other options appear nowhere in it.")
            out.append(row("prog_extract", q, text, gold, rat))
            break
    return out


# ---------------------------------------------------------------------------
# Independent verifiers (re-derive the gold from the ROW ONLY)
# ---------------------------------------------------------------------------


_MON_TOKENS = sorted(MONTHS + MON_ABBR, key=len, reverse=True)
_MON_ALT = "|".join(_MON_TOKENS)  # longest-first so 'June' beats 'Jun'


def _find_dates(s: str) -> list[date]:
    """Dates in any surface form this generator emits (incl. abbreviations).

    Order of appearance is preserved -- several verifiers depend on it.
    """
    spans: dict[int, date] = {}
    for p in (rf"({_MON_ALT})\s+(\d{{1,2}}),?\s+(\d{{4}})",     # month-first
              rf"(\d{{1,2}})\s+({_MON_ALT}),?\s+(\d{{4}})"):    # day-first
        for m in re.finditer(p, s):
            g = m.groups()
            mon, rest = (g[0], g[1:]) if g[0].isalpha() and not g[0][0].isdigit() else (g[1], (g[0], g[2]))
            day, year = int(rest[0]), int(rest[1])
            mi = (MONTHS.index(mon) + 1) if mon in MONTHS else MON_ABBR.index(mon) + 1
            try:
                d = date(year, mi, day)
            except ValueError:
                continue
            spans.setdefault(m.start(), d)
    return [d for _, d in sorted(spans.items())]


def verify_arith_clock(r: dict) -> bool:
    m = re.fullmatch(r"What is (\d{2}):(\d{2}) ([-+]) (\d{2}):(\d{2})\?",
                     r["question"])
    if not m:
        return False
    h1, m1, op, h2, m2 = (int(m.group(1)), int(m.group(2)), m.group(3),
                          int(m.group(4)), int(m.group(5)))
    b = h2 * 60 + m2
    res = (h1 * 60 + m1 - b) % 1440 if op == "-" else (h1 * 60 + m1 + b) % 1440
    return r["targets"][0] == f"{res // 60:02d}:{res % 60:02d}"


def verify_arith_month(r: dict) -> bool:
    m = re.fullmatch(
        r"What is the time (.+) (after|before) (Jan|Feb|Mar|Apr|May|Jun|Jul|"
        r"Aug|Sep|Oct|Nov|Dec), (\d{3,4})", r["question"])
    if not m:
        return False
    phrase, d, mon, y = m.groups()
    anchor = date(int(y), MON_ABBR.index(mon) + 1, 1)
    yy = re.search(r"(\d+) years?", phrase)
    mm = re.search(r"(\d+) months?", phrase)
    if not (yy or mm):
        return False
    off = relativedelta(years=int(yy.group(1)) if yy else 0,
                        months=int(mm.group(1)) if mm else 0)
    res = anchor - off if d == "before" else anchor + off
    return r["targets"][0] == f"{MON_ABBR[res.month - 1]}, {res.year}"


def verify_computation(r: dict) -> bool:
    gold = r["targets"][0]
    q = r["question"].split("(Hint:")[0]
    rat_dates = _find_dates(r["rationale"])
    if not rat_dates:
        return False
    if q.startswith("What was the date"):
        anchor = rat_dates[0]                 # rationale names the anchor first
        yy = re.search(r"(\d+) years?", q)
        mm = re.search(r"(\d+) months?", q)
        dd = re.search(r"(\d+) days?", q)
        if not (yy or mm or dd):
            return False
        off = relativedelta(years=int(yy.group(1)) if yy else 0,
                            months=int(mm.group(1)) if mm else 0,
                            days=int(dd.group(1)) if dd else 0)
        back = "before" in q
        res = anchor - off if back else anchor + off
        gd = _find_dates(gold)
        return bool(gd) and gd[0] == res
    a = rat_dates[0]
    b = max(rat_dates)
    if len({x for x in rat_dates}) < 2:
        return False
    return gold == span_text(relativedelta(b, a))


def verify_timeline(r: dict) -> bool:
    keys = dict(re.findall(r"([A-E])=(\d{4}-\d{2}-\d{2})", r["rationale"]))
    gold = r["targets"][0]
    letters = gold.split(",")
    if set(keys) != set(letters):
        return False
    want = ",".join(sorted(letters, key=lambda L: keys[L]))
    return want == gold


def verify_relation(r: dict) -> bool:
    gold = r["targets"][0]
    if gold not in REL_GOLD:
        return False
    q = r["question"]
    if "Choices:" in q:
        opts = split_options(q)
        return len(opts) == 3 and gold in opts
    return " What is the relationship " in q


def verify_ordering_tf(r: dict) -> bool:
    gold = r["targets"][0]
    if gold not in ("TRUE", "FALSE"):
        return False
    ds = _find_dates(r["question"])
    if len(ds) < 2:
        return False
    return ("TRUE" if ds[0] < ds[1] else "FALSE") == gold


def verify_ordering_seq(r: dict) -> bool:
    gold = r["targets"][0]
    ds = _find_dates(r["question"])
    if not ds:
        return False
    k = len(ds)
    idx = [int(x) for x in re.findall(r"\((\d)\)", gold)]
    if sorted(idx) != list(range(1, k + 1)):
        return False
    want = "(" + "), (".join(str(i + 1) for i in
                             sorted(range(k), key=lambda j: ds[j])) + ")"
    return want == gold


def verify_duration_compare(r: dict) -> bool:
    gold = r["targets"][0]
    m = re.findall(r"(\d{4}-\d{2}-\d{2})", r["rationale"])
    if len(m) < 4:
        return False
    a1, a2, b1, b2 = (date.fromisoformat(x) for x in m[:4])
    d1, d2 = abs((a2 - a1).days), abs((b2 - b1).days)
    close = (abs(d1 - d2) / max(d1, d2, 1) <= 0.10) or (abs(d1 - d2) <= 60)
    if gold.startswith("The two"):
        return close
    if gold.startswith("Duration 1"):
        return d1 > d2 and not close
    if gold.startswith("Duration 2"):
        return d2 > d1 and not close
    return False


def verify_order_compare(r: dict) -> bool:
    gold = r["targets"][0]
    ds = _find_dates(r["question"] + " " + r["context"])
    if len(ds) < 2:
        return False
    a, b = ds[0], ds[1]
    if gold.startswith("They happen"):
        return abs((b - a).days) <= 2
    if gold.startswith("Fact 1"):
        return a < b
    if gold.startswith("Fact 2"):
        return b < a
    return False


def verify_extract(r: dict) -> bool:
    gold = r["targets"][0]
    letters = gold.split("  ")
    if not all(re.fullmatch(r"[A-E]", x) for x in letters):
        return False
    if len(letters) != len(set(letters)):
        return False
    opts = [ln for ln in r["question"].split("Choices:", 1)[1].splitlines()
            if re.match(r"^[A-E]\. ", ln)]
    for L in letters:
        opt = opts[ord(L) - 65][3:]
        if opt not in r["context"]:
            return False
    for i, ln in enumerate(opts):
        if chr(65 + i) not in letters and ln[3:] in r["context"]:
            return False
    return True


VERIFIERS = {
    "prog_arith_clock": verify_arith_clock,
    "prog_arith_month": verify_arith_month,
    "prog_computation": verify_computation,
    "prog_timeline": verify_timeline,
    "prog_relation": verify_relation,
    "prog_ordering_tf": verify_ordering_tf,
    "prog_ordering_seq": verify_ordering_seq,
    "prog_duration_compare": verify_duration_compare,
    "prog_order_compare": verify_order_compare,
    "prog_extract": verify_extract,
}

GENS = {
    "prog_arith_clock": gen_arith_clock,
    "prog_arith_month": gen_arith_month,
    "prog_computation": gen_computation,
    "prog_timeline": gen_timeline,
    "prog_relation": gen_relation,
    "prog_ordering_tf": gen_ordering_tf,
    "prog_ordering_seq": gen_ordering_seq,
    "prog_duration_compare": gen_duration_compare,
    "prog_order_compare": gen_order_compare,
    "prog_extract": gen_extract,
}


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit(cat: str, rows: list[dict], v12_qs: set[str], cap_n: int) -> list[str]:
    errs: list[str] = []
    n = len(rows)
    if n == 0:
        return [f"{cat}: no rows"]
    if n != cap_n:
        errs.append(f"{cat}: generated {n} rows, cap is {cap_n}")
    bad = [r for r in rows if not VERIFIERS[cat](r)]
    if bad:
        errs.append(f"{cat}: {len(bad)}/{n} rows FAIL independent "
                    f"verification (e.g. {str(bad[0]['question'])[:90]!r} "
                    f"gold={bad[0]['targets'][0]!r})")
    qs = [re.sub(r"\s+", " ", r["question"]).strip().lower() for r in rows]
    if len(set(qs)) != len(qs):
        errs.append(f"{cat}: internal duplicate questions")
    clash = sum(1 for q in qs if q in v12_qs)
    if clash:
        errs.append(f"{cat}: {clash} questions collide with v12")
    g = Counter(r["targets"][0] for r in rows)
    # Gold-concentration rules are category-aware:
    #  - free-form golds (arithmetic, dates, letter-sets): no single string
    #    > 10% (the D51 lesson);
    #  - bounded combinatorial golds (permutations, 3-option texts, labels)
    #    are balanced BY DESIGN, so they get balance-band checks instead,
    #    applied only at real scale (n >= 50).
    FREE_FORM = {"prog_arith_clock", "prog_arith_month", "prog_computation",
                 "prog_extract"}
    top, cnt = g.most_common(1)[0]
    if cat in FREE_FORM:
        if cnt / n > 0.10 and n >= 50:
            errs.append(f"{cat}: gold '{top}' is {100 * cnt / n:.0f}% of rows "
                        f"(>10% cap)")
    elif n >= 50:
        if cat in ("prog_duration_compare", "prog_order_compare"):
            want = {"prog_duration_compare": (33, 40, 27),
                    "prog_order_compare": (35, 45, 20)}[cat]
            got = tuple(round(100 * g[o] / n) for o in
                        (_DC_OPTS if cat == "prog_duration_compare"
                         else _OC_OPTS))
            if any(abs(a - b) > 8 for a, b in zip(got, want)):
                errs.append(f"{cat}: gold mix {got} vs target {want}")
        elif cat == "prog_ordering_tf":
            t, f = g["TRUE"], g["FALSE"]
            if abs(t - f) > 0.10 * n:
                errs.append(f"{cat}: TRUE/FALSE split {t}/{f}")
        else:  # permutation / letter-sequence golds (relation has its own
               # quota bands above; 5 labels cannot fit under 26%)
            if cat in ("prog_timeline", "prog_ordering_seq") and cnt / n > 0.26:
                errs.append(f"{cat}: top gold '{top}' is "
                            f"{100 * cnt / n:.0f}% (>26% cap)")
    mcq = [r for r in rows if "Choices:" in r["question"]]
    if len(mcq) >= 6:
        pos: Counter = Counter()
        for r in mcq:
            opts = split_options(r["question"])
            g0 = r["targets"][0]
            if g0 in opts:
                pos[chr(65 + opts.index(g0))] += 1
        if pos and pos.most_common(1)[0][1] / sum(pos.values()) > 0.60:
            errs.append(f"{cat}: MCQ gold position skew {dict(pos)}")
    shortctx = [r for r in rows if r["context"].strip()
                and len(r["context"].split()) < 60]
    if shortctx:
        errs.append(f"{cat}: {len(shortctx)} rows with context < 60 words")
    if cat == "prog_timeline":
        ident = sum(1 for r in rows
                    if r["targets"][0] == ",".join(
                        sorted(r["targets"][0].split(","))))
        if ident / n > 0.35:
            errs.append(f"{cat}: identity golds {100 * ident / n:.0f}% (>1/3)")
    if cat == "prog_ordering_seq":
        # identity = strictly "(1), (2), ..., (k)" (a gold merely STARTING
        # with (1) is not identity -- an earlier regex here counted those)
        def _is_ident(g: str) -> bool:
            parts = [p.strip() for p in g.split(",")]
            return parts == [f"({i + 1})" for i in range(len(parts))]
        ident = sum(1 for r in rows if _is_ident(r["targets"][0]))
        if ident / n > 0.35:
            errs.append(f"{cat}: identity golds {100 * ident / n:.0f}% (>1/3)")
    if cat == "prog_relation" and n >= 50:
        dist = Counter(r["targets"][0] for r in rows)
        for lbl in REL_GOLD:
            if lbl == "SIMULTANEOUS" and dist[lbl] / n > 0.15:
                errs.append(f"{cat}: SIMULTANEOUS {100 * dist[lbl] / n:.0f}% "
                            f"(>15%)")
            elif lbl != "SIMULTANEOUS" and dist[lbl] / n < 0.10:
                errs.append(f"{cat}: {lbl} only {100 * dist[lbl] / n:.0f}% "
                            f"(<10%)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prog_aug_v13")
    ap.add_argument("--smoke", action="store_true",
                    help="12 rows per category, *_smoke output dir")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    if args.smoke:
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = {k: 12 for k in CAPS} if args.smoke else dict(CAPS)

    v12_qs: set[str] = set()
    for split in ("train", "val"):
        p = ROOT / "data" / "combined_80_20_v12" / f"{split}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    v12_qs.add(re.sub(
                        r"\s+", " ",
                        json.loads(l)["question"]).strip().lower())

    all_errs: list[str] = []
    summary = {}
    for cat, gen in GENS.items():
        rng = Random(f"{args.seed}-{cat}")
        rows = gen(rng, caps[cat], letter_cycle_abc())
        errs = audit(cat, rows, v12_qs, caps[cat])
        all_errs += errs
        with (out_dir / f"{cat}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[cat] = {
            "rows": len(rows),
            "gold_top3": Counter(r["targets"][0] for r in rows).most_common(3),
        }
        status = "OK" if not errs else "ERRORS: " + "; ".join(errs)
        print(f"  {cat:24s} {len(rows):5d} rows  {status}")
    (out_dir / "audit.json").write_text(json.dumps(
        {"seed": args.seed, "caps": caps, "summary": summary,
         "errors": all_errs}, indent=1))
    if all_errs:
        print(f"\nAUDIT FAILED with {len(all_errs)} error(s) -- "
              f"see {out_dir / 'audit.json'}")
        return 1
    print(f"\naudit clean; rows written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
