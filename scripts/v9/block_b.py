"""Block B — class-label emission (ruleset §4 Block B, §5.11).

nli_saq / nli_mcq are Mode A (premises are real CC-News sentences);
relation / ordering are Mode B (self-contained, internally consistent).
`neutral` is trained on for the first time in the campaign (L4).
"""
from __future__ import annotations

import random
import re
from collections import Counter

from .common import make_row, trim_sentence
from .corpus import Article, Event
from .mcq import build_mcq, LetterBalancer
from .temporal import days_between, extract_dates

NLI_CHOICES = ["entailment", "neutral", "contradiction"]
REL_CHOICES = ["IDENTITY", "BEFORE", "DURING"]
ORD_CHOICES = ["TRUE", "Undetermined", "FALSE"]

NEUTRAL_TEMPLATES = [
    "The {p1} took place in the capital.",
    "Officials had anticipated the {p2} for months beforehand.",
    "The {p1} was the first event of its kind that year.",
    "Both the {p1} and the {p2} received extensive television coverage.",
    "The {p1} happened late at night.",
    "The {p2} had been planned years in advance.",
    "The {p1} was more controversial than the {p2}.",
    "Few people paid attention to the {p1} at the time.",
]

R_SUBJ = [
    "the parliamentary committee", "the city council", "the museum board",
    "the football club", "the opera house", "the shipping company",
    "the railway operator", "the university senate", "the mining firm",
    "the television network", "the farmers' cooperative", "the airline",
    "the publishing house", "the theater troupe", "the software studio",
    "the energy utility", "the water authority", "the cycling team",
    "the chamber of commerce", "the heritage foundation", "the port authority",
    "the film studio", "the record label", "the insurance group",
    "the research institute", "the athletics federation", "the bakery chain",
    "the ferry operator", "the construction firm", "the radio station",
]
R_VERB_PAST = [
    "released its annual figures", "opened the new wing", "cancelled the event",
    "launched the campaign", "approved the budget", "announced the merger",
    "unveiled the memorial", "signed the agreement", "paused the project",
    "published the findings", "held the vote", "started the renovation",
    "closed the facility", "awarded the contract", "filed the appeal",
    "hosted the festival", "completed the survey", "inaugurated the bridge",
    "retired the fleet", "staged the concert", "adopted the resolution",
    "unveiled the logo", "expanded the depot", "secured the loan",
]
R_CONTAINER = [
    "the exhibition", "the trade fair", "the inquiry", "the festival season",
    "the auction series", "the lecture programme", "the tournament",
    "the renovation project", "the enrollment period", "the negotiation round",
    "the pilot scheme", "the survey season",
]
R_ALIAS = [
    ("the winter gala, also billed as the season finale,", "the season finale"),
    ("the flagship conference, marketed as the global summit,", "the global summit"),
    ("the opening ceremony, referred to in the press as the inaugural event,", "the inaugural event"),
    ("the anniversary tour, promoted as the farewell series,", "the farewell series"),
    ("the championship final, officially styled the title decider,", "the title decider"),
    ("the redevelopment plan, known locally as the river scheme,", "the river scheme"),
]
R_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]


def _mk_date(rng: random.Random, y0: int = 1960, y1: int = 2024):
    y = rng.randint(y0, y1)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return y, m, d


def _fmt(y: int, m: int, d: int) -> str:
    return f"{R_MONTHS[m - 1]} {d}, {y}"


def _first_ok(e) -> bool:
    ds = [d for d in extract_dates(e.sent) if d.y >= 1900]
    fulls = [d for d in ds if d.precision >= 2]
    return bool(ds) and ds[0].key() == e.date.key() and len(fulls) == 1


class BlockB:
    def __init__(self, arts: list[Article], rng: random.Random) -> None:
        self.rng = rng
        self.arts = arts
        self.bal = LetterBalancer()
        self.usage: Counter[str] = Counter()
        self.seen_q: set[str] = set()
        self.qtoks: list[set[str]] = []
        self.used_pairs: set[tuple] = set()

    def take(self) -> Article | None:
        for _ in range(300):
            a = self.rng.choice(self.arts)
            if self.usage[a.sid] < 6:
                self.usage[a.sid] += 1
                return a
        return None

    def fresh(self, q: str) -> bool:
        nq = " ".join(q.lower().split())
        if nq in self.seen_q:
            return False
        ts = set(re.findall(r"[a-z0-9]+", q.lower()))
        for other in self.qtoks:
            u = len(ts | other)
            if u and len(ts & other) / u > 0.89:
                return False
        self.seen_q.add(nq)
        self.qtoks.append(ts)
        return True

    def _premise_pair(self) -> tuple[Event, Event, str] | None:
        art = self.take()
        if not art:
            return None
        fulls: dict[tuple, Event] = {}
        for e in art.events:
            if e.date.precision == 3 and 8 <= len(e.sent.split()) <= 26:
                fulls.setdefault(e.date.key(), e)
        fulls = {k: e for k, e in fulls.items()
                 if _first_ok(e)}
        fulls = {k: e for k, e in fulls.items() if _first_ok(e)}
        evs = sorted(fulls.values(), key=lambda e: e.date.sort_key())
        if len(evs) < 2:
            return None
        a, b = sorted(self.rng.sample(evs, 2), key=lambda e: e.date.sort_key())
        premise = f"{a.sent} {b.sent}"
        return a, b, premise

    def gen_nli(self, n: int, mcq: bool) -> list[dict]:
        cat = "nli_mcq" if mcq else "nli_saq"
        self.used_pairs.clear()
        third = n // 3
        queue = ["entailment"] * third + ["contradiction"] * third + \
                ["neutral"] * (n - 2 * third)
        self.rng.shuffle(queue)
        rows: list[dict] = []
        guard = 0
        while len(rows) < n and queue and guard < n * 60:
            guard += 1
            want = queue.pop(0)
            pp = self._premise_pair()
            if not pp:
                queue.append(want)
                if guard > n * 55:
                    break
                continue
            a, b, premise = pp
            pkey = tuple(sorted((a.date.key(), b.date.key())))
            if pkey in self.used_pairs:
                queue.append(want)
                continue
            from .temporal import clean_fact as _cf
            p1 = " ".join((_cf(a.lead) or a.lead).split()[:7]).rstrip(" ,;:.")
            p2 = " ".join((_cf(b.lead) or b.lead).split()[:7]).rstrip(" ,;:.")
            if not p1 or not p2 or p1 == p2:
                queue.append(want)
                continue
            if p1 not in premise or p2 not in premise:
                queue.append(want)
                continue
            gap = days_between(a.date, b.date)
            if want == "neutral":
                hyp = self.rng.choice(NEUTRAL_TEMPLATES).format(p1=p1, p2=p2)
                rat = (f"The premise only dates the two events ({a.date.text}, "
                       f"{b.date.text}); the hypothesis adds an unverifiable "
                       f"claim, so the label is neutral.")
            else:
                use_duration = gap is not None and gap >= 4 and self.rng.random() < 0.5
                if use_duration:
                    mid = gap // 2
                    ent = f"More than {mid} days separated the {p1} from the {p2}."
                    con = f"Fewer than {mid} days separated the {p1} from the {p2}."
                else:
                    ent = f"The {p2} came after the {p1}."
                    con = f"The {p2} came before the {p1}."
                hyp = ent if want == "entailment" else con
                rat = (f"The premise dates the {p1} to {a.date.text} and the "
                       f"{p2} to {b.date.text}, so the hypothesis is {want}.")
            if mcq:
                q = hyp + "\nChoices:\n" + "\n".join(
                    f"{l}. {t}" for l, t in zip("ABC", NLI_CHOICES))
            else:
                q = hyp
            if not self.fresh(q):
                queue.append(want)
                continue
            self.used_pairs.add(pkey)
            rows.append(make_row("B", cat, "news", q, premise, want, rat, a.sid))
        return rows[:n]

    def gen_relation(self, n: int = 150) -> list[dict]:
        rows: list[dict] = []
        kinds = ["BEFORE"] * (n // 3) + ["DURING"] * (n // 3) + \
                ["IDENTITY"] * (n - 2 * (n // 3))
        self.rng.shuffle(kinds)
        guard = 0
        while len(rows) < n and guard < n * 40:
            guard += 1
            kind = kinds[len(rows) % len(kinds)]
            if kind == "BEFORE":
                y1, m1, d1 = _mk_date(self.rng)
                span = self.rng.choice([3, 8, 15, 40, 100, 400, 800])
                from datetime import date
                dd = date(y1, m1, d1)
                from datetime import timedelta
                d2 = dd + timedelta(days=span)
                y2, m2, dd2 = d2.year, d2.month, d2.day
                s1, s2 = self.rng.sample(R_SUBJ, 2)
                v1, v2 = self.rng.sample(R_VERB_PAST, 2)
                sent = (f"{s1.capitalize()} {v1} on {_fmt(y1, m1, d1)}, and "
                        f"{s2} {v2} on {_fmt(y2, m2, dd2)}.")
                gold = "BEFORE"
                rat = (f"The first event is dated {_fmt(y1, m1, d1)} and the "
                       f"second {_fmt(y2, m2, dd2)}, so the first happened "
                       f"before the second.")
            elif kind == "DURING":
                y1, m1, d1 = _mk_date(self.rng)
                from datetime import date, timedelta
                start = date(y1, m1, d1)
                end = start + timedelta(days=self.rng.choice([14, 30, 60, 120]))
                mid = start + timedelta(days=(end - start).days // 2)
                cont = self.rng.choice(R_CONTAINER)
                inner_s = self.rng.choice(R_SUBJ)
                inner_v = self.rng.choice(R_VERB_PAST)
                sent = (f"{cont.capitalize()} ran from "
                        f"{_fmt(start.year, start.month, start.day)} to "
                        f"{_fmt(end.year, end.month, end.day)}, and {inner_s} "
                        f"{inner_v} on {_fmt(mid.year, mid.month, mid.day)}, "
                        f"in the middle of it.")
                gold = "DURING"
                rat = (f"The second event ({_fmt(mid.year, mid.month, mid.day)}) "
                       f"falls inside the first event's span "
                       f"({_fmt(start.year, start.month, start.day)} to "
                       f"{_fmt(end.year, end.month, end.day)}), so it happened "
                       f"during it.")
            else:
                y, m, d = _mk_date(self.rng)
                full, alias = self.rng.choice(R_ALIAS)
                v = self.rng.choice(R_VERB_PAST)
                sent = f"{full.capitalize()} {v} on {_fmt(y, m, d)}."
                gold = "IDENTITY"
                rat = (f"The two names refer to the same single event on "
                       f"{_fmt(y, m, d)}, so the relationship is identity.")
            q = (f"{sent} What is the relationship between the events?"
                 "\nChoices:\n"
                 + "\n".join(f"{l}. {t}" for l, t in zip("ABC", REL_CHOICES)))
            gold_text = gold
            if not self.fresh(q):
                continue
            rows.append(make_row("B", "relation", "none", q, "", gold_text,
                                 rat, ""))
        return rows[:n]

    def gen_ordering(self, n: int = 100) -> list[dict]:
        rows: list[dict] = []
        n_und = int(n * 0.26)
        kinds = ["TRUE"] * ((n - n_und) // 2) + ["FALSE"] * ((n - n_und) // 2) \
            + ["Undetermined"] * (n - 2 * ((n - n_und) // 2))
        self.rng.shuffle(kinds)
        guard = 0
        while len(rows) < n and guard < n * 40:
            guard += 1
            kind = kinds[len(rows) % len(kinds)]
            if kind == "Undetermined":
                s1, s2 = self.rng.sample(R_SUBJ, 2)
                v1, v2 = self.rng.sample(R_VERB_PAST, 2)
                e1 = f"{s1.capitalize()} {v1}"
                e2 = f"{s2} {v2}"
                gold = "Undetermined"
                rat = ("Neither statement carries any date or sequencing "
                       "information, so the chronological order cannot be "
                       "determined.")
            else:
                from datetime import date, timedelta
                y1, m1, d1 = _mk_date(self.rng)
                start = date(y1, m1, d1)
                gap = self.rng.choice([1, 4, 9, 20, 60, 200, 500])
                if kind == "TRUE":
                    nxt = start + timedelta(days=gap)
                    e1 = self._ev(start)
                    e2 = self._ev(nxt)
                    gold = "TRUE"
                    rat = (f"The first event is dated "
                           f"{_fmt(start.year, start.month, start.day)} and the "
                           f"second {_fmt(nxt.year, nxt.month, nxt.day)}, so "
                           f"the stated order is correct.")
                else:
                    nxt = start - timedelta(days=gap)
                    e1 = self._ev(start)
                    e2 = self._ev(nxt)
                    gold = "FALSE"
                    rat = (f"The first event is dated "
                           f"{_fmt(start.year, start.month, start.day)} but the "
                           f"second {_fmt(nxt.year, nxt.month, nxt.day)}, which "
                           f"is earlier, so the stated order is wrong.")
            q = (f"{e1}. Then {e2}. - True/False?\nChoices:\n"
                 + "\n".join(f"{l}. {t}" for l, t in zip("ABC", ORD_CHOICES)))
            if not self.fresh(q):
                continue
            rows.append(make_row("B", "ordering", "none", q, "", gold, rat, ""))
        return rows[:n]

    def _ev(self, d) -> str:
        s = self.rng.choice(R_SUBJ)
        v = self.rng.choice(R_VERB_PAST)
        return f"{s} {v} on {_fmt(d.year, d.month, d.day)}"
