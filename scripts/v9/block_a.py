"""Block A — TIME category generators (ruleset §4 Block A, §5 cards).

Every gold is mechanically derived from dates explicitly stated in the
context, so §7.4's recomputation can verify it independently. Contexts are
built first, then events are drawn from the article the context actually
carries — that is what makes the §6.1 noisy-retrieval condition honest.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from datetime import timedelta
from itertools import permutations

from .common import ABSTAIN, dated_tail_fact, make_row, paraphrase, move_date_front, \
    strip_dangling, subject_of, swap_one_token, trim_sentence
from .corpus import Article, Event, TopicIndex
from .ctxbuild import dial_context, dial_context_drop_turns, news_context, \
    news_context_no_gold, wiki_context, wiki_context_drop
from .mcq import LetterBalancer, build_mcq
from .temporal import D, clean_fact, days_between, decide_span, extract_dates, fmt_full, \
    humanize_span, humanize_months, in_context, month_gap, strip_date_phrases

CATS = {
    "Computation":        dict(total=800, mcq=0,   pairs=0),
    "Timeline":           dict(total=650, mcq=650, pairs=0),
    "Localization":       dict(total=550, mcq=0,   pairs=0),
    "Counterfactual":     dict(total=500, mcq=305, pairs=50),
    "Duration_Compare":   dict(total=450, mcq=450, pairs=0),
    "Relative_Reasoning": dict(total=450, mcq=261, pairs=45),
    "Order_Reasoning":    dict(total=400, mcq=232, pairs=35),
    "Co_temporality":     dict(total=300, mcq=150, pairs=25),
    "Explicit_Reasoning": dict(total=200, mcq=122, pairs=21),
    "Order_Compare":      dict(total=100, mcq=100, pairs=0),
}

HINT = ("   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year "
        "2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)")

DC_OPTIONS = ["Duration 1 is longer.", "Duration 2 is longer.",
              "The two durations are approximately the same length."]
OC_OPTIONS = ["Fact 1 happened earlier.", "Fact 2 happened earlier.",
              "They happen at almost the same time."]

TL_HEADER = ("Below are {k} facts. You need to sort these facts in "
             "chronological order. Requirements: You must output a sequence "
             "of uppercase letters separated by commas, such as 'A,B,C', "
             "without any other characters.")

COMP_SPAN_STEMS = [
    "How many days passed between {e1} on {d1} and {e2} on {d2}?",
    "How many days elapsed between {e1} on {d1} and {e2} on {d2}?",
    "What was the span of time between {e1} on {d1} and {e2} on {d2}?",
    "How much time went by between {e1} on {d1} and {e2} on {d2}?",
]
COMP_OFFSET_STEMS = [
    "What date was it {n} days after {d1}, when {e1} took place?",
    "{n} days after {e1} on {d1}, what was the date?",
    "Counting {n} days forward from {e1} on {d1}, what date is it?",
]
COMP_MONTH_STEMS = [
    "How many months passed between {e1} in {d1} and {e2} in {d2}?",
    "How many months went by between {e1} in {d1} and {e2} in {d2}?",
    "Counting from {e1} in {d1} to {e2} in {d2}, how many months passed?",
]
LOC_STEMS = {
    3: ["When did {e}?", "On what date did {e}?", "On which date did {e}?"],
    2: ["In which month and year did {e}?"],
    1: ["In what year did {e}?"],
}
CF_PREMISES = [
    "If the reports of {pre} had turned out to be mistaken",
    "Had the circumstances around {pre} unfolded otherwise",
    "If {pre} had never been made public",
    "Assuming the timing of {pre} had been other than reported",
    "If official accounts of {pre} had been withdrawn",
]
CF_ASKS = [
    "what does the passage record about {ask}?",
    "what is documented about {ask}?",
    "what did the reporting actually establish about {ask}?",
    "what was reported concerning {ask}?",
]
RR_NEXT = [
    "What happened immediately after {a}?",
    "What followed directly after {a}?",
    "Which development came next after {a}?",
]
RR_RECENT = [
    "What was the most recent development after {a}?",
    "Which event was the latest to follow {a}?",
    "After {a}, what was the last reported development?",
]
OR_STEMS = [
    "What was the {ord} development in the {win} coverage of {sub}?",
    "What was the {ord} recorded development for {sub} in {win}?",
    "In {win}, what came {ord} in the sequence of developments for {sub}?",
]
CT_STEMS = [
    "While events around {akey} were still unfolding between {a1} and {a2}, what was reported concerning {bkey}?",
    "During the {akey} developments spanning {a1} through {a2}, what was happening with {bkey}?",
    "Between {a1} and {a2}, while the {akey} situation was ongoing, what development involved {bkey}?",
]
ER_STEMS = [
    "What notable activities did {sub} engage in between {d1} and {d2}?",
    "Between {d1} and {d2}, what activities were recorded for {sub}?",
    "What developments involving {sub} took place between {d1} and {d2}?",
]

REL_PHRASES = {3: "three days later", 4: "four days later", 5: "five days later",
               7: "one week later", 10: "ten days later", 14: "two weeks later",
               21: "three weeks later", 28: "four weeks later"}


def clip(lead: str, n: int) -> str:
    """Word-prefix of a lead that stays a verbatim substring of the passage.

    Trailing function words are dropped (AUDIT 2026-09-16) so the fragment does
    not end mid-clause; dropping words from the end keeps it a substring.
    """
    cut = " ".join(lead.split()[:n]).rstrip(" ,;:")
    return strip_dangling(cut) or cut


def nonverbatim_sent(sent: str, ctx: str) -> str | None:
    """A meaning-preserving rewrite guaranteed absent from the context."""
    t = move_date_front(paraphrase(sent))
    if not in_context(t, ctx):
        return t
    t2 = "Reportedly, " + (t[0].lower() + t[1:] if t else t)
    return None if in_context(t2, ctx) else t2


def nonverbatim_lead(lead: str, date_text: str, ctx: str) -> str | None:
    t = paraphrase(lead)
    if not in_context(t, ctx):
        return t
    t2 = f"{clip(lead, 12)} ({date_text})"
    return None if in_context(t2, ctx) else t2


def _norm_q(q: str) -> str:
    return " ".join(q.lower().split())


def _tokset(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


class BlockA:
    def __init__(self, arts: list[Article], topics: TopicIndex,
                 rng: random.Random) -> None:
        self.rng = rng
        self.arts = arts
        self.topics = topics
        self.usage: Counter[tuple] = Counter()
        self.total_use: Counter[str] = Counter()
        self.bal = LetterBalancer()
        self.seen_q: dict[str, set[str]] = defaultdict(set)
        self.qtoks: dict[str, list[set[str]]] = defaultdict(list)
        self.shortfalls: dict[str, str] = {}

    # -- shared helpers ---------------------------------------------------
    def take(self, pool: list[Article], cat: str = "*", per_cat: int = 2,
             global_cap: int = 20) -> Article | None:
        for _ in range(min(len(pool), 300)):
            art = self.rng.choice(pool)
            if self.usage[(cat, art.sid)] < per_cat and \
                    self.total_use[art.sid] < global_cap:
                self.usage[(cat, art.sid)] += 1
                self.total_use[art.sid] += 1
                return art
        return None

    def fillers(self, art: Article, n: int) -> list[Article]:
        rel = self.topics.related(art, n + 4)
        out = [a for a in rel if a.sid != art.sid]
        while len(out) < n:
            c = self.rng.choice(self.arts)
            if c.sid != art.sid:
                out.append(c)
        return out[:n]

    def fresh(self, cat: str, q: str) -> bool:
        nq = _norm_q(q)
        if nq in self.seen_q[cat]:
            return False
        ts = _tokset(q)
        for other in self.qtoks[cat]:
            if jaccard(ts, other) > 0.89:
                return False
        self.seen_q[cat].add(nq)
        self.qtoks[cat].append(ts)
        return True

    def series(self, art: Article, k: int, precision: int = 2) -> list[Event]:
        by_date: dict[tuple, Event] = {}
        for e in art.events:
            if e.date.precision >= precision:
                by_date.setdefault(e.date.key(), e)
        s = sorted(by_date.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def series_ev(self, art: Article, k: int, maxpos: int,
                  precision: int = 1) -> list[Event]:
        """Distinct-(year,month) series — ordering is never ambiguous."""
        best: dict[tuple, Event] = {}
        for e in self.ctx_events(art, maxpos):
            if e.date.precision >= precision:
                mk = (e.date.y, e.date.m or 0)
                cur = best.get(mk)
                if cur is None or e.date.precision > cur.date.precision:
                    best[mk] = e
        s = sorted(best.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def full_series_ev(self, art: Article, k: int, maxpos: int) -> list[Event]:
        by_date: dict[tuple, Event] = {}
        for e in self.ctx_events(art, maxpos):
            if e.date.precision == 3:
                by_date.setdefault(e.date.key(), e)
        s = sorted(by_date.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def month_series_ev(self, art: Article, k: int, maxpos: int) -> list[Event]:
        by: dict[tuple, Event] = {}
        for e in self.ctx_events(art, maxpos):
            if e.date.precision >= 2:
                by.setdefault((e.date.y, e.date.m), e)
        s = sorted(by.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def span_unit_series_ev(self, art: Article, k: int, maxpos: int
                            ) -> tuple[list[Event], str] | None:
        s = self.full_series_ev(art, k, maxpos)
        if len(s) >= k:
            return s, "days"
        s = self.month_series_ev(art, k, maxpos)
        if len(s) >= k:
            return s, "months"
        return None

    def full_series(self, art: Article, k: int) -> list[Event]:
        by_date: dict[tuple, Event] = {}
        for e in art.events:
            if e.date.precision == 3:
                by_date.setdefault(e.date.key(), e)
        s = sorted(by_date.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def month_series(self, art: Article, k: int) -> list[Event]:
        """Precision>=2 events with distinct (year, month) — unambiguously sortable."""
        by: dict[tuple, Event] = {}
        for e in art.events:
            if e.date.precision >= 2:
                by.setdefault((e.date.y, e.date.m), e)
        s = sorted(by.values(), key=lambda e: e.date.sort_key())
        return s if len(s) >= k else []

    def span_unit_series(self, art: Article, k: int) -> tuple[list[Event], str] | None:
        s = self.full_series(art, k)
        if len(s) >= k:
            return s, "days"
        s = self.month_series(art, k)
        if len(s) >= k:
            return s, "months"
        return None

    def prov_queue(self, total: int) -> list[str]:
        n_news = round(total * 0.60)
        n_wiki = round(total * 0.35)
        q = (["news"] * n_news + ["wiki"] * n_wiki
             + ["dial"] * (total - n_news - n_wiki))
        self.rng.shuffle(q)
        return q

    def relax(self) -> bool:
        return self.rng.random() < 0.15

    def build_context(self, prov: str, art: Article, gold_per: int | None = None
                      ) -> tuple[str, Article, str, int, list]:
        """Returns (context, gold article, sid, gold max-pos, participating
        [(article, its event word-budget)] for pooled fact construction)."""
        if prov == "wiki":
            return (wiki_context(art, art.events, self.rng), art, art.sid,
                    10 ** 9, [(art, 10 ** 9)])
        if prov == "dial":
            evs3 = [e for e in art.events if e.date.precision == 3] or art.events
            ctx, _ = dial_context(art, evs3, self.rng, n_sessions=2)
            return ctx, art, art.sid, 10 ** 9, [(art, 10 ** 9)]
        if self.rng.random() < 0.20:
            fils = self.fillers(art, 4)
            ctx, gper = news_context(fils[0], fils[1:4],
                                     self.rng.randint(0, 2), self.rng,
                                     gold_per=gold_per)
            carts = [(fils[0], gper - 40)] + [(f, 300) for f in fils[1:4]]
            return ctx, fils[0], fils[0].sid, gper - 40, carts
        fils = self.fillers(art, 3)
        ctx, gper = news_context(art, fils,
                                 self.rng.randint(0, 2), self.rng,
                                 gold_per=gold_per)
        carts = [(art, gper - 40)] + [(f, 300) for f in fils]
        return ctx, art, art.sid, gper - 40, carts

    def ctx_events(self, src: Article, maxpos: int) -> list[Event]:
        return [e for e in src.events if e.pos <= maxpos]

    def dial_fact_ok(self, ctx: str, ev: Event) -> bool:
        return in_context(ev.lead, ctx) and in_context(ev.date.text, ctx)

    # -- Computation -------------------------------------------------------
    def gen_computation(self, pool: list[Article], n: int = 800) -> list[dict]:
        rows: list[dict] = []
        kinds = ["days"] * 40 + ["months"] * 40 + ["years"] * 20
        self.rng.shuffle(kinds)
        provs = self.prov_queue(n)
        guard = 0
        while len(rows) < n and guard < n * 300:
            guard += 1
            art = self.take(pool, "Computation", per_cat=4, global_cap=24)
            if not art:
                self.shortfalls.setdefault("Computation", f"{len(rows)}/{n}")
                break
            kind = kinds[len(rows) % len(kinds)]
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art)
            evs = self.full_series_ev(src, 2, maxpos)
            lo, hi = {"days": (2, 29), "months": (35, 800),
                      "years": (380, 4000)}[kind]
            pair = None
            for _ in range(30):
                if len(evs) < 2:
                    break
                a, b = sorted(self.rng.sample(evs, 2),
                              key=lambda e: e.date.sort_key())
                d = days_between(a.date, b.date)
                if d and lo <= d <= hi:
                    pair = (a, b, d)
                    break
            if not pair and kind in ("months", "years"):
                mev = sorted({e.date.key(): e for e in self.ctx_events(src, maxpos)
                              if e.date.precision >= 2}.values(),
                             key=lambda e: e.date.sort_key())
                for _ in range(30):
                    if len(mev) < 2:
                        break
                    a, b = sorted(self.rng.sample(mev, 2),
                                  key=lambda e: e.date.sort_key())
                    if a.date.precision == 3 and b.date.precision == 3:
                        continue
                    g = month_gap(a.date, b.date)
                    want = (2, 24) if kind == "months" else (13, 200)
                    if g and want[0] <= g <= want[1]:
                        pair = (a, b, g)
                        break
            if not pair:
                continue
            a, b, d = pair
            la = strip_date_phrases(clip(a.lead, 12)).rstrip(" ,;:.")
            lb = strip_date_phrases(clip(b.lead, 12)).rstrip(" ,;:.")
            if len(la.split()) < 3:
                la = "the first event"
            if len(lb.split()) < 3:
                lb = "the second event"
            if not (in_context(a.date.text, ctx) and in_context(b.date.text, ctx)
                    and in_context(la, ctx) and in_context(lb, ctx)):
                continue
            if days_between(a.date, b.date) is None:
                gold = humanize_months(d)
                stem = self.rng.choice(COMP_MONTH_STEMS)
                q = stem.format(e1=la, d1=a.date.text, e2=lb,
                                d2=b.date.text) + HINT
                rat = (f"The passage places \"{la}\" in {a.date.text} and "
                       f"\"{lb}\" in {b.date.text}; that is {d} months apart, "
                       f"i.e. {gold}.")
                if in_context(gold, ctx) or not self.fresh("Computation", q):
                    continue
                rows.append(make_row("A", "Computation", prov, q, ctx, gold,
                                     rat, sid))
                continue
            if self.rng.random() < 0.30:
                off = self.rng.randint(3, 90)
                gd = a.date.exact() + timedelta(days=off)
                gold = fmt_full(gd)
                if in_context(gold, ctx):
                    continue
                stem = self.rng.choice(COMP_OFFSET_STEMS)
                q = stem.format(n=off, d1=a.date.text, e1=la) + HINT
                rat = (f"The passage dates \"{la}\" to {a.date.text}; "
                       f"{a.date.text} + {off} days = {gold}.")
            else:
                gold = humanize_span(a.date, b.date)
                if not gold or in_context(gold, ctx):
                    continue
                stem = self.rng.choice(COMP_SPAN_STEMS)
                q = stem.format(e1=la, d1=a.date.text, e2=lb, d2=b.date.text) + HINT
                rat = (f"The passage states {a.date.text} for \"{la}\" and "
                       f"{b.date.text} for \"{lb}\"; the span is {d} days, "
                       f"i.e. {gold}.")
            if not self.fresh("Computation", q):
                continue
            rows.append(make_row("A", "Computation", prov, q, ctx, gold, rat, sid))
        return rows[:n]

    # -- Timeline ------------------------------------------------------------
    def gen_timeline(self, pools: dict[int, list[Article]], n: int = 650) -> list[dict]:
        q3 = ["".join(p) for _ in range(65)
              for p in self.rng.sample(list(permutations("ABC")), 6)]
        q4 = []
        while len(q4) < 195:
            q4 += ["".join(p) for p in
                   self.rng.sample(list(permutations("ABCD")), 24)]
        q5 = ["".join(p) for p in
              self.rng.sample(list(permutations("ABCDE")), 120)][:65]
        queue = [(3, g) for g in q3[:390]] + [(4, g) for g in q4[:195]] \
            + [(5, g) for g in q5]
        rows: list[dict] = []
        provs = self.prov_queue(n)
        guard = 0
        while len(rows) < n and queue and guard < n * 400:
            guard += 1
            k, perm = queue[0]
            art = self.take(pools.get(k) or pools[3], "Timeline", per_cat=6,
                            global_cap=14)
            if not art:
                self.shortfalls.setdefault("Timeline", f"{len(rows)}/{n}")
                break
            prov = provs[guard % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            bym: dict[tuple, Event] = {}
            for cart, cmax in carts:
                for e in self.ctx_events(cart, cmax):
                    mk = (e.date.y, e.date.m or 0)
                    cur = bym.get(mk)
                    if cur is None or e.date.precision > cur.date.precision:
                        bym[mk] = e
            s = sorted(bym.values(), key=lambda e: e.date.sort_key())
            if len(s) < k:
                continue
            pairs_f: list[tuple[Event, str]] = []
            for e in s:
                ds = [d for d in extract_dates(e.sent) if d.y >= 1900]
                if not ds or ds[0].key() != e.date.key():
                    continue
                if prov == "dial":
                    if not (in_context(clean_fact(e.lead) or e.lead, ctx)
                            and in_context(e.date.text, ctx)):
                        continue
                    f = clean_fact(e.lead)
                else:
                    if not in_context(e.sent, ctx):
                        continue
                    f = clean_fact(e.lead) or clean_fact(clip(e.sent, 26))
                if f and len(f.split()) <= 30:
                    pairs_f.append((e, f, e.sent))
            if len(pairs_f) < k or len({f for _, f, _ in pairs_f}) < k:
                continue
            pairs_f = pairs_f[:k]
            facts = [f for _, f, _ in pairs_f]
            if not all(in_context(b, ctx) for _, _, b in pairs_f):
                continue
            listing: list[str] = [""] * k
            for i, fact in enumerate(facts):
                listing["ABCDEF".index(perm[i])] = fact
            gold = ",".join(perm)
            q = TL_HEADER.format(k=k) + "\nChoices:\n" + "\n".join(
                f"{'ABCDEF'[i]}. {f}" for i, f in enumerate(listing))
            if not self.fresh("Timeline", q):
                continue
            rat = "The passage dates the facts: " + "; ".join(
                f"{perm[i]} = {pairs_f[i][0].date.text}"
                for i in range(k)) + \
                f"; chronological order is {gold}."
            rows.append(make_row("A", "Timeline", prov, q, ctx, gold, rat, sid))
            queue.pop(0)
        return rows[:n]

    # -- Localization ---------------------------------------------------------
    def gen_localization(self, pool: list[Article], n: int = 550) -> list[dict]:
        rows: list[dict] = []
        n_rel = int(n * 0.30)
        rel_done = 0
        provs = self.prov_queue(n)
        guard = 0
        from dateutil.relativedelta import relativedelta
        while len(rows) < n and guard < n * 90:
            guard += 1
            art = self.take(pool, "Localization")
            if not art:
                self.shortfalls.setdefault("Localization", f"{len(rows)}/{n}")
                break
            made_rel = False
            if rel_done < n_rel:
                fulls = [e for e in art.events if e.date.precision == 3]
                pair = None
                for _ in range(20):
                    if len(fulls) < 2:
                        break
                    a, b = sorted(self.rng.sample(fulls, 2),
                                  key=lambda e: e.date.sort_key())
                    ea, eb = a.date.exact(), b.date.exact()
                    if not ea or not eb:
                        continue
                    d = (eb - ea).days
                    if d in REL_PHRASES:
                        pair = (a, b, REL_PHRASES[d])
                        break
                    rd = relativedelta(eb, ea)
                    if rd.years == 0 and rd.months == 1 and rd.days == 0:
                        pair = (a, b, "one month later")
                        break
                    if rd.years == 1 and rd.months == 0 and rd.days == 0:
                        pair = (a, b, "one year later")
                        break
                if not pair:
                    from .corpus import _sentences, _clean_sent
                    undated = []
                    for s in _sentences(art.text):
                        cs = _clean_sent(s)
                        if not cs:
                            continue
                        ds = extract_dates(cs)
                        if ds and any(d.y >= 1900 for d in ds):
                            continue
                        if 8 <= len(cs.split()) <= 40:
                            undated.append(cs)
                    fulls2 = [e for e in fulls
                              if len([d for d in extract_dates(e.sent)
                                      if d.y >= 1900]) == 1]
                    if undated and fulls2:
                        a = self.rng.choice(fulls2)
                        N, phrase = self.rng.choice(
                            [(7, "one week later"),
                             (14, "two weeks later"),
                             (21, "three weeks later"),
                             (10, "ten days later"),
                             (28, "four weeks later")])
                        from datetime import timedelta
                        target = a.date.exact() + timedelta(days=N)
                        u = self.rng.choice(undated)
                        u_txt = strip_date_phrases(u)
                        rel_sent = f"{phrase.capitalize()}, " + \
                            (u_txt[0].lower() + u_txt[1:])
                        gold = fmt_full(D(target.year, target.month, target.day))
                        keep = [e for e in art.events
                                if e.sent != u and gold not in e.sent]
                        rctx = wiki_context(art, keep, self.rng,
                                            keep_sentence=rel_sent,
                                            exclude_str=gold)
                        if not in_context(gold, rctx) and \
                                in_context(a.lead, rctx) and \
                                in_context(a.date.text, rctx):
                            q = self.rng.choice(LOC_STEMS[3]).format(
                                e=clip(a.lead, 12) + ", and the follow-up")
                            rat = (f"The passage dates '{clip(a.lead, 10)}' to "
                                   f"{a.date.text} and marks the follow-up "
                                   f"'{phrase}'; that resolves to {gold}.")
                            if self.fresh("Localization", q):
                                rows.append(make_row(
                                    "A", "Localization", "wiki", q, rctx, gold,
                                    rat, art.sid))
                                rel_done += 1
                                made_rel = True
                if pair:
                    a, b, phrase = pair
                    stripped = strip_date_phrases(trim_sentence(b.sent, 22))
                    rel_sent = f"{phrase.capitalize()}, " + \
                        (stripped[0].lower() + stripped[1:])
                    keep = [e for e in art.events
                            if e.sent != b.sent and b.date.text not in e.sent]
                    rctx = wiki_context(art, keep, self.rng,
                                        keep_sentence=rel_sent,
                                        exclude_str=b.date.text)
                    gold = fmt_full(b.date)
                    if not in_context(gold, rctx) and not in_context(b.lead, rctx) \
                            and in_context(a.lead, rctx) \
                            and in_context(a.date.text, rctx):
                        q = self.rng.choice(LOC_STEMS[3]).format(
                            e=clip(a.lead, 12) + ", and the follow-up")
                        rat = (f"The passage dates '{clip(a.lead, 10)}' to "
                               f"{a.date.text} and says the follow-up came "
                               f"{phrase}; that resolves to {gold}.")
                        if self.fresh("Localization", q):
                            rows.append(make_row("A", "Localization", "wiki",
                                                 q, rctx, gold, rat, art.sid))
                            rel_done += 1
                            made_rel = True
            if made_rel:
                continue
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art)
            evs = [e for e in self.ctx_events(src, maxpos)
                   if in_context(e.date.text, ctx)
                   and (in_context(e.lead, ctx) or in_context(e.sent, ctx))]
            if not evs:
                continue
            e = self.rng.choice(evs)
            p = e.date.precision
            if p < 3 and self.rng.random() < 0.6:
                better = [x for x in evs if x.date.precision == 3]
                if better:
                    e = self.rng.choice(better)
                    p = 3
            gold = (fmt_full(e.date) + ".") if p == 3 else e.date.text
            q = self.rng.choice(LOC_STEMS[min(p, 3)]).format(e=clip(e.lead, 14))
            rat = f"The passage states '{clip(e.lead, 12)}' dated {e.date.text}."
            if self.fresh("Localization", q):
                rows.append(make_row("A", "Localization", prov, q, ctx, gold,
                                     rat, sid))
        return rows[:n]

    # -- Duration_Compare -------------------------------------------------------
    def gen_duration_compare(self, pool: list[Article], n: int = 450) -> list[dict]:
        rows: list[dict] = []
        classes = ["A"] * 166 + ["B"] * 180 + ["C"] * 104
        self.rng.shuffle(classes)
        provs = self.prov_queue(n)
        guard = 0
        while len(rows) < n and guard < n * 300:
            guard += 1
            art = self.take(pool, "Duration_Compare", per_cat=8, global_cap=40)
            if not art:
                self.shortfalls.setdefault("Duration_Compare", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            pooled: dict[tuple, Event] = {}
            for cart, cmax in carts:
                for e in self.ctx_events(cart, cmax):
                    if e.date.precision == 3:
                        mk = e.date.key()
                        cur = pooled.get(mk)
                        if cur is None:
                            pooled[mk] = e
            evs = sorted(pooled.values(), key=lambda e: e.date.sort_key())
            unit = "days"
            if len(evs) < 4:
                pooled2: dict[tuple, Event] = {}
                for cart, cmax in carts:
                    for e in self.ctx_events(cart, cmax):
                        if e.date.precision >= 2:
                            mk = (e.date.y, e.date.m or 0)
                            cur = pooled2.get(mk)
                            if cur is None or e.date.precision > cur.date.precision:
                                pooled2[mk] = e
                evs = sorted(pooled2.values(), key=lambda e: e.date.sort_key())
                unit = "months"
            if len(evs) < 4:
                continue
            evs1 = evs
            if len(evs1) < 4:
                continue
            spans = []
            for i in range(len(evs1)):
                for j in range(i + 1, len(evs1)):
                    if unit == "days":
                        d = days_between(evs1[i].date, evs1[j].date)
                        if d and d >= 21:
                            spans.append((d, evs1[i], evs1[j]))
                    else:
                        g = month_gap(evs1[i].date, evs1[j].date)
                        if g and g >= 3:
                            spans.append((g, evs1[i], evs1[j]))
            if len(spans) < 2:
                continue
            want = classes[len(rows) % len(classes)]
            choice = None
            for _ in range(60):
                (d1, a1, b1), (d2, a2, b2) = self.rng.sample(spans, 2)
                keys = {a1.date.key(), b1.date.key(), a2.date.key(), b2.date.key()}
                if len(keys) < 4:
                    continue
                if decide_span(d1, d2, unit) == want:
                    choice = (a1, b1, a2, b2, d1, d2)
                    break
            if not choice:
                continue
            unit_word = unit
            a1, b1, a2, b2, d1, d2 = choice
            evs4 = [a1, b1, a2, b2]
            if any(not clean_fact(x.lead) for x in evs4):
                continue
            leads = [f"{clip(clean_fact(x.lead), 12)}, on "
                     f"{x.date.text}" for x in evs4]
            q = ("Which of the following two durations is longer? "
                 f'*Duration 1:* Between "{leads[0]}" and "{leads[1]}". '
                 f'*Duration 2:* Between "{leads[2]}" and "{leads[3]}".\nChoices:\n'
                 + "\n".join(f"{l}. {t}" for l, t in zip("ABC", DC_OPTIONS)))
            gold = {"A": DC_OPTIONS[0], "B": DC_OPTIONS[1], "C": DC_OPTIONS[2]}[want]
            if not self.fresh("Duration_Compare", q):
                continue
            rat = (f"Duration 1: {a1.date.text} to {b1.date.text} = {d1} {unit_word}; "
                   f"duration 2: {a2.date.text} to {b2.date.text} = {d2} {unit_word}; "
                   f"so {want}.")
            rows.append(make_row("A", "Duration_Compare", prov, q, ctx, gold,
                                 rat, sid))
        return rows[:n]

    # -- Order_Compare ------------------------------------------------------------
    def gen_order_compare(self, pool: list[Article], n: int = 100) -> list[dict]:
        rows: list[dict] = []
        classes = ["A"] * 38 + ["B"] * 47 + ["C"] * 15
        self.rng.shuffle(classes)
        provs = self.prov_queue(n)
        guard = 0
        while len(rows) < n and guard < n * 80:
            guard += 1
            art = self.take(pool, "Order_Compare")
            if not art:
                self.shortfalls.setdefault("Order_Compare", f"{len(rows)}/{n}")
                break
            want = classes[len(rows) % len(classes)]
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art)
            pair = None
            s = [e for e in self.full_series_ev(src, 2, maxpos)
                 if len([d for d in extract_dates(e.sent)
                         if d.precision == 3]) == 1]
            if len(s) >= 2:
                for _ in range(40):
                    e1, e2 = self.rng.sample(s, 2)
                    d = days_between(e1.date, e2.date)
                    if d is None or d == 0:
                        continue
                    if want == "A" and d > 30:
                        pair = (e1, e2)
                    elif want == "B" and d < -30:
                        pair = (e1, e2)
                    elif want == "C" and abs(d) <= 3:
                        pair = (e1, e2)
                    if pair:
                        break
            elif want in ("A", "B"):
                ms = self.month_series_ev(src, 2, maxpos)
                if len(ms) < 2:
                    continue
                for _ in range(20):
                    e1, e2 = self.rng.sample(ms, 2)
                    g = (e2.date.y * 12 + (e2.date.m or 1)) - \
                        (e1.date.y * 12 + (e1.date.m or 1))
                    if want == "A" and g >= 2:
                        pair = (e1, e2)
                    elif want == "B" and g <= -2:
                        pair = (e1, e2)
                    if pair:
                        break
            if not pair:
                continue
            e1, e2 = pair
            if not (in_context(e1.date.text, ctx) and in_context(e2.date.text, ctx)):
                continue
            f1 = dated_tail_fact(e1.sent) or clean_fact(e1.sent, 5, 34)
            f2 = dated_tail_fact(e2.sent) or clean_fact(e2.sent, 5, 34)
            if not f1 or not f2 or extract_dates(f1) or extract_dates(f2):
                continue
            q = (f"For Fact1: {f1} (on {e1.date.text}) and Fact2: {f2} "
                 f"(on {e2.date.text}), which one happened "
                 "earlier?\nChoices:\n"
                 + "\n".join(f"{l}. {t}" for l, t in zip("ABC", OC_OPTIONS)))
            gold = {"A": OC_OPTIONS[0], "B": OC_OPTIONS[1], "C": OC_OPTIONS[2]}[want]
            if not self.fresh("Order_Compare", q):
                continue
            rat = (f"Fact 1 is dated {e1.date.text}; Fact 2 is dated "
                   f"{e2.date.text}; therefore {gold.lower()}")
            rows.append(make_row("A", "Order_Compare", prov, q, ctx, gold,
                                 rat, sid))
        return rows[:n]

    # -- Counterfactual ---------------------------------------------------------
    def _cf_material(self, src: Article, ctx: str, maxpos: int = 10 ** 9
                     ) -> tuple[Event, Event, list[Event]] | None:
        evs = [e for e in self.ctx_events(src, maxpos) if in_context(e.sent, ctx)
               and 8 <= len(e.sent.split()) <= 44]
        if len(evs) < 3:
            return None
        ge = self.rng.choice(evs)
        others = [e for e in evs if e is not ge]
        if not others:
            return None
        se = self.rng.choice(others)
        return ge, se, others

    def gen_counterfactual(self, pool: list[Article], n: int = 500) -> list[dict]:
        cfg = CATS["Counterfactual"]
        rows: list[dict] = []
        n_np = max(1, cfg["total"] - 2 * cfg["pairs"])
        n_free = round(n * (cfg["total"] - cfg["mcq"]) / n_np)
        n_mcq_plain = n - n_free
        verbatim_budget = max(0, round(n * 0.36) - round(cfg["pairs"] * n / n_np))
        n_verbatim = 0
        provs = self.prov_queue(n)
        guard = 0
        need_free, need_mcq = n_free, n_mcq_plain
        while (need_free > 0 or need_mcq > 0) and guard < n * 600:
            guard += 1
            art = self.take(pool, "Counterfactual")
            if not art:
                self.shortfalls.setdefault("Counterfactual", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            mat = self._cf_material(src, ctx, maxpos)
            if not mat:
                continue
            ge, se, others = mat
            premise = self.rng.choice(CF_PREMISES).format(pre=clip(se.lead, 10))
            ask = clip(ge.lead, 10)
            head = f"{premise}, {self.rng.choice(CF_ASKS).format(ask=ask)}"
            verbatim = n_verbatim < verbatim_budget and self.rng.random() < 0.45
            if verbatim:
                gold_long = ge.sent if in_context(ge.sent, ctx) else None
                gold_short = ge.lead if gold_long and in_context(ge.lead, ctx) else gold_long
            else:
                gold_long = nonverbatim_sent(ge.sent, ctx)
                gold_short = nonverbatim_lead(ge.lead, ge.date.text, ctx) or gold_long
            if not gold_long:
                continue
            if need_mcq > 0 and (need_free == 0 or self.rng.random() < 0.61):
                dis: list[str] = []
                for sib in self.rng.sample(others, min(3, len(others))):
                    sw = swap_one_token(trim_sentence(ge.sent, 26), sib, self.rng)
                    if sw and sw != gold_long:
                        dis.append(trim_sentence(sw, 26))
                    elif len(sib.sent.split()) <= 28:
                        dis.append(trim_sentence(sib.sent, 26))
                dis = [d for d in dis if d != gold_long][:3]
                if len(dis) < 3:
                    continue
                row = self.mcq_row("Counterfactual", prov, src, ctx, head,
                                   gold_long, dis,
                                   f"The counterfactual premise does not "
                                   f"change the record; the passage states: "
                                   f"{clip(ge.sent, 18)}", sid, relax=False)
                if row:
                    rows.append(row)
                    need_mcq -= 1
                    n_verbatim += int(verbatim)
            else:
                q = head
                if not self.fresh("Counterfactual", q):
                    continue
                rows.append(make_row("A", "Counterfactual", prov, q, ctx,
                                     gold_short,
                                     f"The counterfactual does not change what "
                                     f"the passage records: {gold_short}", sid))
                need_free -= 1
                n_verbatim += int(verbatim)
        return rows[:n]

    # -- Relative_Reasoning --------------------------------------------------------
    def gen_relative(self, pool: list[Article], n: int = 450) -> list[dict]:
        cfg = CATS["Relative_Reasoning"]
        rows: list[dict] = []
        n_np = max(1, cfg["total"] - 2 * cfg["pairs"])
        n_free = round(n * (cfg["total"] - cfg["mcq"]) / n_np)
        n_mcq_plain = n - n_free
        verbatim_budget = max(0, round(n * 0.48) - round(cfg["pairs"] * n / n_np))
        n_verbatim = 0
        provs = self.prov_queue(n)
        guard = 0
        need_free, need_mcq = n_free, n_mcq_plain
        while (need_free > 0 or need_mcq > 0) and guard < n * 600:
            guard += 1
            art = self.take(pool, "Relative_Reasoning", per_cat=8, global_cap=60)
            if not art:
                self.shortfalls.setdefault("Relative_Reasoning", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            bym: dict[tuple, Event] = {}
            for cart, cmax in carts:
                for e in self.ctx_events(cart, cmax):
                    mk = (e.date.y, e.date.m or 0)
                    cur = bym.get(mk)
                    if cur is None or e.date.precision > cur.date.precision:
                        bym[mk] = e
            s = sorted(bym.values(), key=lambda e: e.date.sort_key())
            if len(s) < 4:
                continue
            i = self.rng.randrange(len(s) - 1)
            anchor, nxt = s[i], s[i + 1]
            mode_recent = i + 1 < len(s) - 1 and self.rng.random() < 0.5
            gold_ev = s[-1] if mode_recent else nxt
            if gold_ev is anchor:
                continue
            a_txt = clip(anchor.lead, 12)
            if not in_context(a_txt, ctx):
                continue
            stems = RR_RECENT if mode_recent else RR_NEXT
            head = self.rng.choice(stems).format(a=a_txt)
            verbatim = n_verbatim < verbatim_budget and self.rng.random() < 0.55
            if verbatim:
                gold_long = gold_ev.sent if in_context(gold_ev.sent, ctx) else None
                gold_short = gold_ev.lead if gold_long and in_context(gold_ev.lead, ctx) else gold_long
            else:
                gold_long = nonverbatim_sent(gold_ev.sent, ctx)
                gold_short = nonverbatim_lead(gold_ev.lead, gold_ev.date.text, ctx) or gold_long
            if not gold_long:
                continue
            siblings = [e for e in s if e is not anchor and e is not gold_ev]
            if need_mcq > 0 and (need_free == 0 or self.rng.random() < 0.58):
                dis = []
                for e in self.rng.sample(siblings, min(3, len(siblings))):
                    t = trim_sentence(e.sent, 24)
                    if t and t != gold_long:
                        dis.append(t)
                if len(dis) < 3:
                    continue
                seq_txt = "; ".join(clip(e.lead, 8) for e in s[i + 1:i + 4])
                row = self.mcq_row("Relative_Reasoning", prov, src, ctx, head,
                                   gold_long, dis,
                                   f"After the anchor, the passage records, "
                                   f"in order: {seq_txt}", sid, relax=False)
                if row:
                    rows.append(row)
                    need_mcq -= 1
                    n_verbatim += int(verbatim)
            else:
                if not self.fresh("Relative_Reasoning", head):
                    continue
                rows.append(make_row("A", "Relative_Reasoning", prov, head, ctx,
                                     gold_short,
                                     f"Anchor: \"{a_txt}\"; the passage then "
                                     f"records, in order, " + "; ".join(
                                         f"{e.lead} ({e.date.text})"
                                         for e in s[i + 1:]) + ".", sid))
                need_free -= 1
                n_verbatim += int(verbatim)
        return rows[:n]

    # -- Order_Reasoning ---------------------------------------------------------
    def gen_order_reasoning(self, pool: list[Article], n: int = 400) -> list[dict]:
        cfg = CATS["Order_Reasoning"]
        rows: list[dict] = []
        n_np = max(1, cfg["total"] - 2 * cfg["pairs"])
        n_free = round(n * (cfg["total"] - cfg["mcq"]) / n_np)
        n_mcq_plain = n - n_free
        provs = self.prov_queue(n)
        guard = 0
        need_free, need_mcq = n_free, n_mcq_plain
        while (need_free > 0 or need_mcq > 0) and guard < n * 600:
            guard += 1
            art = self.take(pool, "Order_Reasoning")
            if not art:
                self.shortfalls.setdefault("Order_Reasoning", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            by_year: dict[int, list[Event]] = defaultdict(list)
            for e in src.events:
                if in_context(e.sent, ctx) and 6 <= len(e.sent.split()) <= 50:
                    by_year[e.date.y].append(e)
            windows = [v for v in by_year.values() if len(v) >= 3]
            if not windows:
                continue
            win = self.rng.choice(windows)
            win = sorted(win, key=lambda e: e.date.sort_key())
            ord_idx = self.rng.randrange(len(win))
            ordinal = ORDINAL_FOR(len(win), ord_idx)
            ge = win[ord_idx]
            sub = subject_of(src)
            head = self.rng.choice(OR_STEMS).format(
                ord=ordinal, win=str(win[0].date.y), sub=sub)
            gold_long = ge.sent
            if len(gold_long.split()) > 30 or not in_context(gold_long, ctx):
                continue
            siblings = [e for e in win if e is not ge]
            if need_mcq > 0 and (need_free == 0 or self.rng.random() < 0.58):
                dis = []
                for e in self.rng.sample(siblings, min(3, len(siblings))):
                    t = e.sent if len(e.sent.split()) <= 30 else trim_sentence(e.sent, 26)
                    if t and t != gold_long and in_context(t, ctx):
                        dis.append(t)
                if len(dis) < 3:
                    continue
                ord_txt = "; ".join(
                    f"{ORDINAL_FOR(len(win), k2)}: {clip(e2.lead, 8)}"
                    for k2, e2 in enumerate(win))
                row = self.mcq_row("Order_Reasoning", prov, src, ctx, head,
                                   gold_long, dis,
                                   f"In {win[0].date.y} the recorded order "
                                   f"was: {ord_txt}", sid, relax=False)
                if row:
                    rows.append(row)
                    need_mcq -= 1
            else:
                gold_short = ge.lead if in_context(ge.lead, ctx) else gold_long
                if not self.fresh("Order_Reasoning", head):
                    continue
                rows.append(make_row("A", "Order_Reasoning", prov, head, ctx,
                                     gold_short,
                                     f"In {win[0].date.y} the recorded order was: "
                                     + "; ".join(f"{ORDINAL_FOR(len(win), i)}: "
                                                 f"{e.lead} ({e.date.text})"
                                                 for i, e in enumerate(win)) + ".",
                                     sid))
                need_free -= 1
        return rows[:n]

    # -- Co_temporality -----------------------------------------------------------
    _DET = {"The", "This", "That", "These", "Those", "His", "Her", "Their",
            "Its", "It", "He", "She", "They", "We", "You", "In", "On", "At",
            "After", "Before", "During", "When", "While", "A", "An", "But",
            "And", "Then", "Later", "Earlier", "Meanwhile", "However", "Some",
            "Many", "Both", "Each", "Another", "Such", "There", "Here", "What",
            "Which", "Who", "Whose", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday", "Monday", "Tuesday", "January", "February",
            "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "New", "First", "Last", "Two",
            "Three", "Four", "Five", "Officials", "Reporters"}

    @classmethod
    def threads(cls, art: Article, maxpos: int = 10 ** 9) -> dict[str, list[Event]]:
        by: dict[str, list[Event]] = defaultdict(list)
        for e in art.events:
            if e.pos > maxpos:
                continue
            caps = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", e.lead)
                    if w not in cls._DET]
            if caps:
                by.setdefault(caps[0], []).append(e)
        return {k: sorted(v, key=lambda e: e.date.sort_key()) for k, v in by.items()}

    def gen_co_temporality(self, pool: list[Article], n: int = 300) -> list[dict]:
        cfg = CATS["Co_temporality"]
        rows: list[dict] = []
        n_np = max(1, cfg["total"] - 2 * cfg["pairs"])
        n_free = round(n * (cfg["total"] - cfg["mcq"]) / n_np)
        n_mcq_plain = n - n_free
        verbatim_budget = max(0, round(n * 0.46) - round(cfg["pairs"] * n / n_np))
        n_verbatim = 0
        provs = self.prov_queue(n)
        guard = 0
        need_free, need_mcq = n_free, n_mcq_plain
        while (need_free > 0 or need_mcq > 0) and guard < n * 1000:
            guard += 1
            art = self.take(pool, "Co_temporality", per_cat=4, global_cap=24)
            if not art:
                self.shortfalls.setdefault("Co_temporality", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art, gold_per=600)
            th = self.threads(src, maxpos)
            keys = list(th)
            pick = None
            for _ in range(60):
                if len(keys) < 2:
                    break
                akey, bkey = self.rng.sample(keys, 2)
                ta, tb = th[akey], th[bkey]
                if len(ta) < 2 or (len(tb) < 2 and bkey == akey):
                    continue
                ea1, ea2 = sorted(self.rng.sample(ta, 2),
                                  key=lambda e: e.date.sort_key())
                a1, a2 = ea1.date, ea2.date
                if a2.key() <= a1.key():
                    continue
                inside = [e for e in tb
                          if a1.key() < e.date.key() < a2.key()]
                outside = [e for e in self.ctx_events(src, maxpos)
                           if not a1.key() <= e.date.key() <= a2.key()]
                if inside:
                    pick = (akey, bkey, a1, a2, inside, outside, ta)
                    break
            if not pick:
                evs_all = self.ctx_events(src, maxpos)
                if not evs_all:
                    continue
                sev = sorted(evs_all, key=lambda e: e.date.sort_key())
                a1, a2 = sev[0].date, sev[-1].date
                if a2.key() <= a1.key():
                    continue
                akey = subject_of(src)
                tb_keys = [k for k in th if k and k.lower() not in
                           akey.lower()]
                if tb_keys:
                    bkey = self.rng.choice(tb_keys)
                    inside = [e for e in th.get(bkey, [])
                              if a1.key() < e.date.key() < a2.key()]
                else:
                    inside = []
                if not inside:
                    mid = [e for e in sev
                           if a1.key() < e.date.key() < a2.key()]
                    if not mid:
                        continue
                    ge_fb = self.rng.choice(mid)
                    gcaps = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b",
                                                   ge_fb.lead)
                             if w not in self._DET]
                    bkey = (gcaps[0] if gcaps else subject_of(src))
                    inside = [ge_fb]
                outside = [e for e in evs_all
                           if not a1.key() <= e.date.key() <= a2.key()]
                ta = sev
                if not inside:
                    continue
            akey, bkey, a1, a2, inside, outside, ta = pick if pick else \
                (akey, bkey, a1, a2, inside, outside, ta)
            ge = self.rng.choice(inside)
            head = self.rng.choice(CT_STEMS).format(
                akey=akey, a1=a1.text, a2=a2.text, bkey=bkey)
            verbatim = n_verbatim < verbatim_budget and self.rng.random() < 0.55
            if verbatim:
                gold_long = ge.sent if in_context(ge.sent, ctx) else None
                gold_short = ge.lead if gold_long and in_context(ge.lead, ctx) else gold_long
            else:
                gold_long = nonverbatim_sent(ge.sent, ctx)
                gold_short = nonverbatim_lead(ge.lead, ge.date.text, ctx) or gold_long
            if not gold_long:
                continue
            dis_pool = outside + [e for e in ta if e.date.key() not in
                                  (a1.key(), a2.key())]
            if len(dis_pool) < 3:
                dis_pool = dis_pool + [e for e in inside if e is not ge]
            if need_mcq > 0 and (need_free == 0 or self.rng.random() < 0.5):
                dis = []
                for e in self.rng.sample(dis_pool, min(3, len(dis_pool))):
                    t = trim_sentence(e.sent, 24)
                    if t and t != gold_long:
                        dis.append(t)
                if len(dis) < 3:
                    continue
                row = self.mcq_row("Co_temporality", prov, src, ctx, head,
                                   gold_long, dis,
                                   f"{akey} spanned {a1.text} to {a2.text}; "
                                   f"inside that window the passage dates "
                                   f"'{clip(ge.lead, 10)}' to {ge.date.text}.",
                                   sid, relax=False)
                if row:
                    rows.append(row)
                    need_mcq -= 1
                    n_verbatim += int(verbatim)
            else:
                if not self.fresh("Co_temporality", head):
                    continue
                rows.append(make_row("A", "Co_temporality", prov, head, ctx,
                                     gold_short,
                                     f"{akey} spanned {a1.text} to {a2.text}; "
                                     f"within that window the passage dates "
                                     f"\"{ge.lead}\" to {ge.date.text}.", sid))
                need_free -= 1
                n_verbatim += int(verbatim)
        return rows[:n]

    # -- Explicit_Reasoning ----------------------------------------------------------
    def gen_explicit(self, pool: list[Article], n: int = 200) -> list[dict]:
        cfg = CATS["Explicit_Reasoning"]
        rows: list[dict] = []
        n_np = max(1, cfg["total"] - 2 * cfg["pairs"])
        n_free = round(n * (cfg["total"] - cfg["mcq"]) / n_np)
        n_mcq_plain = n - n_free
        verbatim_budget = max(0, round(n * 0.67) - round(cfg["pairs"] * n / n_np))
        n_verbatim = 0
        provs = self.prov_queue(n)
        guard = 0
        need_free, need_mcq = n_free, n_mcq_plain
        while (need_free > 0 or need_mcq > 0) and guard < n * 1500:
            guard += 1
            art = self.take(pool, "Explicit_Reasoning", per_cat=8, global_cap=60)
            if not art:
                self.shortfalls.setdefault("Explicit_Reasoning", f"{len(rows)}/{n}")
                break
            prov = provs[len(rows) % len(provs)]
            ctx, src, sid, maxpos, carts = self.build_context(prov, art)
            evs = [e for cart, cmax in carts for e in self.ctx_events(cart, cmax)
                   if in_context(e.sent, ctx)
                   and 7 <= len(e.sent.split()) <= 60]
            if len(evs) < 3:
                continue
            bym2: dict[tuple, Event] = {}
            for cart, cmax in carts:
                for e in self.ctx_events(cart, cmax):
                    mk = (e.date.y, e.date.m or 0)
                    cur = bym2.get(mk)
                    if cur is None or e.date.precision > cur.date.precision:
                        bym2[mk] = e
            fulls = sorted(bym2.values(), key=lambda e: e.date.sort_key())
            if len(fulls) < 2:
                continue
            i = self.rng.randrange(len(fulls) - 1) if len(fulls) > 2 else 0
            j = min(i + self.rng.choice((1, 2)), len(fulls) - 1)
            d1, d2 = fulls[i].date, fulls[j].date
            if d2.key() <= d1.key():
                continue
            inside = [e for e in evs if d1.key() <= e.date.key() <= d2.key()]
            outside = [e for e in evs
                       if not d1.key() <= e.date.key() <= d2.key()]
            if not inside:
                continue
            ge = self.rng.choice(inside)
            sub = subject_of(src)
            head = self.rng.choice(ER_STEMS).format(sub=sub, d1=d1.text, d2=d2.text)
            verbatim = n_verbatim < verbatim_budget and self.rng.random() < 0.75
            if verbatim:
                gold_long = ge.sent if in_context(ge.sent, ctx) else None
                gold_short = ge.lead if gold_long and in_context(ge.lead, ctx) else gold_long
            else:
                gold_long = nonverbatim_sent(ge.sent, ctx)
                gold_short = nonverbatim_lead(ge.lead, ge.date.text, ctx) or gold_long
            if not gold_long:
                continue
            if need_mcq > 0 and (need_free == 0 or self.rng.random() < 0.61):
                dis = []
                for e in self.rng.sample(outside, min(3, len(outside))):
                    t = trim_sentence(e.sent, 26)
                    if t and t != gold_long:
                        dis.append(t)
                if len(dis) < 3:
                    continue
                in_txt = "; ".join(clip(e.lead, 8) for e in inside[:3])
                row = self.mcq_row("Explicit_Reasoning", prov, src, ctx, head,
                                   gold_long, dis,
                                   f"Within {d1.text} to {d2.text} the passage "
                                   f"records: {in_txt}", sid, relax=False)
                if row:
                    rows.append(row)
                    need_mcq -= 1
                    n_verbatim += int(verbatim)
            else:
                if not self.fresh("Explicit_Reasoning", head):
                    continue
                rows.append(make_row("A", "Explicit_Reasoning", prov, head, ctx,
                                     gold_short,
                                     f"Within {d1.text} to {d2.text} the passage "
                                     f"records: " + "; ".join(e.lead for e in inside)
                                     + ".", sid))
                need_free -= 1
                n_verbatim += int(verbatim)
        return rows[:n]

    # -- shared MCQ/free helpers ---------------------------------------------------
    def mcq_row(self, cat: str, prov: str, art: Article, ctx: str,
                head: str, gold_text: str, distractors: list[str],
                rationale: str, sid: str | None = None,
                relax: bool | None = None) -> dict | None:
        opts = [gold_text] + [d for d in distractors if d and d != gold_text][:3]
        if len(opts) < 4:
            return None
        r = self.relax() if relax is None else relax
        q, gold, info = build_mcq(head, opts, 0, cat, self.bal, self.rng, relax=r)
        if not self.fresh(cat, q):
            return None
        return make_row("A", cat, prov, q, ctx, gold, rationale, sid or art.sid)

    # -- §5.12 answerability pairs ---------------------------------------------------
    def gen_pairs(self, pool: list[Article]) -> list[dict]:
        PAIR_PLAN = [
            ("Counterfactual", 50, self._pair_cf),
            ("Relative_Reasoning", 45, self._pair_rr),
            ("Order_Reasoning", 35, self._pair_or),
            ("Co_temporality", 25, self._pair_ct),
            ("Explicit_Reasoning", 21, self._pair_er),
        ]
        rows: list[dict] = []
        for cat, n_pairs, mat_fn in PAIR_PLAN:
            provs = self.prov_queue(n_pairs)
            made = 0
            guard = 0
            while made < n_pairs and guard < n_pairs * 120:
                guard += 1
                art = self.take(pool, cat, per_cat=3, global_cap=24)
                if not art:
                    self.shortfalls.setdefault(f"{cat}/pairs", f"{made}/{n_pairs}")
                    break
                prov = provs[made % len(provs)]
                out = self._build_one_pair(cat, prov, art, mat_fn)
                if out:
                    rows.extend(out)
                    made += 1
        return rows

    def _build_one_pair(self, cat: str, prov: str, art: Article, mat_fn
                        ) -> list[dict] | None:
        ctx1, src, sid, maxpos, carts1 = self.build_context(prov, art, gold_per=600)
        mat = mat_fn(src, ctx1, maxpos)
        if not mat:
            return None
        head1, head2, ge, dis_texts = mat
        gold_long = ge.sent
        if len(gold_long.split()) > 30 or not in_context(gold_long, ctx1):
            return None
        opts = [o for o in dis_texts if o and o != gold_long][:2]
        if len(opts) < 2:
            return None
        if prov == "news":
            fils = self.fillers(art, 3)
            ctx2, _per = news_context_no_gold(fils, self.rng)
        elif prov == "wiki":
            ctx2 = wiki_context_drop(src, src.events, self.rng, ge.sent)
        else:
            ctx2 = dial_context_drop_turns(ctx1, ge.lead)
        if in_context(gold_long, ctx2) or in_context(paraphrase(gold_long), ctx2):
            return None
        q1, gold1, _i1 = build_mcq(head1, [gold_long] + opts + [ABSTAIN], 0,
                                   cat, self.bal, self.rng, relax=self.relax())
        q2, gold2, _i2 = build_mcq(head2, [ABSTAIN] + opts + [gold_long], 0,
                                   cat, self.bal, self.rng, relax=self.relax())
        if not self.fresh(cat, q1) or not self.fresh(cat, q2):
            return None
        row1 = make_row("A", cat, prov, q1, ctx1, gold1,
                        f"The passage states the fact directly: {gold1[:120]}", sid)
        row2 = make_row("A", cat, prov, q2, ctx2, gold2,
                        "The passage never states this — the fact is absent, so "
                        "the abstain option is the gold.", sid)
        return [row1, row2]

    def _pair_cf(self, src: Article, ctx: str, maxpos: int = 10 ** 9):
        mat = self._cf_material(src, ctx, maxpos)
        if not mat:
            return None
        ge, se, others = mat
        premise1 = self.rng.choice(CF_PREMISES[:3]).format(pre=clip(se.lead, 10))
        premise2 = self.rng.choice(CF_PREMISES[3:]).format(pre=clip(se.lead, 10))
        ask = clip(ge.lead, 10)
        head1 = f"{premise1}, what does the passage record about {ask}?"
        head2 = f"{premise2}, what is documented about {ask}?"
        dis = []
        for sib in others[:4]:
            sw = swap_one_token(trim_sentence(ge.sent, 26), sib, self.rng)
            if sw:
                dis.append(trim_sentence(sw, 26))
        dis = [d for d in dis if d][:2]
        if len(dis) < 2:
            return None
        return head1, head2, ge, dis

    def _pair_rr(self, src: Article, ctx: str, maxpos: int = 10 ** 9):
        s = self.series_ev(src, 4, maxpos)
        if len(s) < 4:
            return None
        i = self.rng.randrange(len(s) - 1)
        anchor, ge = s[i], s[i + 1]
        a_txt = clip(anchor.lead, 12)
        if not in_context(a_txt, ctx):
            return None
        head1 = f"What happened immediately after {a_txt}?"
        head2 = f"Which development came next after {a_txt}?"
        sibs = [e for e in s if e is not anchor and e is not ge]
        dis = [trim_sentence(e.sent, 24) for e in self.rng.sample(sibs, 2)]
        if len([d for d in dis if d]) < 2:
            return None
        return head1, head2, ge, dis

    def _pair_or(self, src: Article, ctx: str, maxpos: int = 10 ** 9):
        by_year: dict[int, list[Event]] = defaultdict(list)
        for e in src.events:
            if in_context(e.sent, ctx) and 6 <= len(e.sent.split()) <= 50:
                by_year[e.date.y].append(e)
        windows = [v for v in by_year.values() if len(v) >= 3]
        if not windows:
            return None
        win = sorted(self.rng.choice(windows), key=lambda e: e.date.sort_key())
        idx = self.rng.randrange(len(win))
        ordinal = ORDINAL_FOR(len(win), idx)
        ge = win[idx]
        sub = subject_of(src)
        year = str(win[0].date.y)
        head1 = f"What was the {ordinal} development in the {year} coverage of {sub}?"
        head2 = f"In {year}, what came {ordinal} in the sequence of developments for {sub}?"
        sibs = [e for e in win if e is not ge]
        dis = [trim_sentence(e.sent, 26) for e in self.rng.sample(sibs, 2)]
        if len([d for d in dis if d]) < 2:
            return None
        return head1, head2, ge, dis

    def _pair_ct(self, src: Article, ctx: str, maxpos: int = 10 ** 9):
        evs = self.ctx_events(src, maxpos)
        if len(evs) < 4:
            return None
        sev = sorted(evs, key=lambda e: e.date.sort_key())
        a1, a2 = sev[0].date, sev[-1].date
        if a2.key() <= a1.key():
            return None
        akey = subject_of(src)
        mid = [e for e in sev if a1.key() < e.date.key() < a2.key()]
        if not mid:
            return None
        ge = self.rng.choice(mid)
        gcaps = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", ge.lead)
                 if w not in self._DET]
        bkey = gcaps[0] if gcaps else subject_of(src)
        head1 = (f"While events around {akey} were still unfolding between "
                 f"{a1.text} and {a2.text}, what was reported concerning {bkey}?")
        head2 = (f"During the {akey} developments spanning {a1.text} through "
                 f"{a2.text}, what was happening with {bkey}?")
        outside = [e for e in evs if not a1.key() <= e.date.key() <= a2.key()]
        dis_pool = outside + [e for e in mid if e is not ge]
        dis = [trim_sentence(e.sent, 24) for e in
               self.rng.sample(dis_pool, min(2, len(dis_pool)))]
        if len([d for d in dis if d]) < 2:
            return None
        return head1, head2, ge, dis

    def _pair_er(self, src: Article, ctx: str, maxpos: int = 10 ** 9):
        evs = [e for e in self.ctx_events(src, maxpos) if in_context(e.sent, ctx)
               and 7 <= len(e.sent.split()) <= 40]
        if len(evs) < 4:
            return None
        fulls = sorted({(e.date.y, e.date.m or 0): e for e in evs}.values(),
                       key=lambda e: e.date.sort_key())
        if len(fulls) < 4:
            return None
        i = self.rng.randrange(len(fulls) - 3)
        d1, d2 = fulls[i].date, fulls[i + 2].date
        inside = [e for e in evs if d1.key() <= e.date.key() <= d2.key()]
        outside = [e for e in evs if not d1.key() <= e.date.key() <= d2.key()]
        if not inside or len(outside) < 2:
            return None
        ge = self.rng.choice(inside)
        sub = subject_of(src)
        head1 = f"What notable activities did {sub} engage in between {d1.text} and {d2.text}?"
        head2 = f"What developments involving {sub} took place between {d1.text} and {d2.text}?"
        dis = [trim_sentence(e.sent, 26) for e in self.rng.sample(outside, 2)]
        if len([d for d in dis if d]) < 2:
            return None
        return head1, head2, ge, dis


def ORDINAL_FOR(n: int, idx: int) -> str:
    if idx == n - 1 and n >= 3:
        return "last"
    return {0: "first", 1: "second", 2: "third"}.get(idx, f"{idx + 1}th")
