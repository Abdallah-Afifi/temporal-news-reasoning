"""Block C — anti-forgetting maintenance slices (ruleset §4 Block C).

temporal_dialogue / storytelling / longform_free are Mode A (grounded on
CC-News); duration is Mode B (curated world-knowledge table, internally
consistent). These rows buy no gain; they are insurance (L3).
"""
from __future__ import annotations

import random
import re
from collections import Counter

from .common import make_row, subject_of, swap_one_token, trim_sentence
from .corpus import Article, TopicIndex
from .ctxbuild import SPEAKERS, dial_context, news_context, wiki_context
from .mcq import LetterBalancer, build_mcq
from .temporal import clean_fact, in_context

DURATION_ITEMS = [
    ("World War II", "1939 to 1945", "about 6 years",
     ["about 4 years", "about 8 years", "about 10 years"]),
    ("World War I", "1914 to 1918", "about 4 years",
     ["about 2 years", "about 6 years", "about 8 years"]),
    ("American Civil War", "1861 to 1865", "about 4 years",
     ["about 2 years", "about 6 years", "about 9 years"]),
    ("Korean War", "1950 to 1953", "about 3 years",
     ["about 1 year", "about 5 years", "about 8 years"]),
    ("Cold War", "1947 to 1991", "about 44 years",
     ["about 20 years", "about 30 years", "about 60 years"]),
    ("Hundred Years' War", "1337 to 1453", "about 116 years",
     ["about 60 years", "about 90 years", "about 150 years"]),
    ("Thirty Years' War", "1618 to 1648", "about 30 years",
     ["about 15 years", "about 45 years", "about 60 years"]),
    ("Napoleonic Wars", "1803 to 1815", "about 12 years",
     ["about 6 years", "about 18 years", "about 25 years"]),
    ("Crimean War", "1853 to 1856", "about 3 years",
     ["about 1 year", "about 5 years", "about 8 years"]),
    ("Gulf War of 1991", "August 1990 to February 1991", "about 7 months",
     ["about 2 months", "about 1 year", "about 2 years"]),
    ("Six-Day War", "June 5 to June 10, 1967", "about 6 days",
     ["about 2 days", "about 2 weeks", "about 1 month"]),
    ("Falklands War", "April to June 1982", "about 10 weeks",
     ["about 3 weeks", "about 6 months", "about 1 year"]),
    ("Prohibition in the United States", "1920 to 1933", "about 13 years",
     ["about 6 years", "about 20 years", "about 30 years"]),
    ("Great Depression", "1929 to 1939", "about 10 years",
     ["about 4 years", "about 15 years", "about 25 years"]),
    ("the Soviet Union", "1922 to 1991", "about 69 years",
     ["about 50 years", "about 80 years", "about 100 years"]),
    ("the Berlin Wall", "1961 to 1989", "about 28 years",
     ["about 10 years", "about 40 years", "about 55 years"]),
    ("Queen Elizabeth II's reign", "1952 to 2022", "about 70 years",
     ["about 45 years", "about 60 years", "about 80 years"]),
    ("Queen Victoria's reign", "1837 to 1901", "about 63 years",
     ["about 40 years", "about 55 years", "about 75 years"]),
    ("the Apollo program", "1961 to 1972", "about 11 years",
     ["about 5 years", "about 15 years", "about 20 years"]),
    ("the Apollo 11 mission", "July 16 to July 24, 1969", "about 8 days",
     ["about 2 days", "about 2 weeks", "about 1 month"]),
    ("the Apollo 13 mission", "April 11 to April 17, 1970", "about 6 days",
     ["about 2 days", "about 2 weeks", "about 3 weeks"]),
    ("construction of the Eiffel Tower", "1887 to 1889", "about 2 years",
     ["about 6 months", "about 5 years", "about 10 years"]),
    ("construction of the Empire State Building", "1930 to 1931",
     "about 1 year", ["about 6 months", "about 3 years", "about 5 years"]),
    ("construction of the Panama Canal", "1904 to 1914", "about 10 years",
     ["about 4 years", "about 15 years", "about 25 years"]),
    ("construction of the Suez Canal", "1859 to 1869", "about 10 years",
     ["about 5 years", "about 20 years", "about 30 years"]),
    ("the Titanic's maiden voyage", "April 10 to April 15, 1912",
     "about 5 days", ["about 2 days", "about 2 weeks", "about 1 month"]),
    ("the Beatles' recording career", "1962 to 1970", "about 8 years",
     ["about 4 years", "about 12 years", "about 20 years"]),
    ("the Woodstock festival", "August 15 to August 18, 1969", "about 3 days",
     ["about 1 day", "about 1 week", "about 2 weeks"]),
    ("Margaret Thatcher's premiership", "1979 to 1990", "about 11 years",
     ["about 5 years", "about 15 years", "about 20 years"]),
    ("Ronald Reagan's presidency", "1981 to 1989", "about 8 years",
     ["about 4 years", "about 12 years", "about 16 years"]),
    ("Barack Obama's presidency", "2009 to 2017", "about 8 years",
     ["about 4 years", "about 10 years", "about 12 years"]),
    ("George Washington's presidency", "1789 to 1797", "about 8 years",
     ["about 4 years", "about 12 years", "about 16 years"]),
    ("Abraham Lincoln's presidency", "1861 to 1865", "about 4 years",
     ["about 2 years", "about 8 years", "about 12 years"]),
    ("Franklin D. Roosevelt's presidency", "1933 to 1945", "about 12 years",
     ["about 6 years", "about 16 years", "about 20 years"]),
    ("Moorish rule in Spain", "711 to 1492", "about 781 years",
     ["about 400 years", "about 600 years", "about 900 years"]),
    ("the Ottoman Empire", "1299 to 1922", "about 623 years",
     ["about 300 years", "about 500 years", "about 800 years"]),
    ("the Ming dynasty", "1368 to 1644", "about 276 years",
     ["about 150 years", "about 200 years", "about 350 years"]),
    ("the Qing dynasty", "1644 to 1912", "about 268 years",
     ["about 100 years", "about 200 years", "about 400 years"]),
    ("the Tang dynasty", "618 to 907", "about 289 years",
     ["about 150 years", "about 250 years", "about 400 years"]),
    ("the American Revolutionary War", "1775 to 1783", "about 8 years",
     ["about 4 years", "about 12 years", "about 20 years"]),
    ("the War of 1812", "1812 to 1815", "about 3 years",
     ["about 1 year", "about 6 years", "about 10 years"]),
    ("the Spanish Civil War", "1936 to 1939", "about 3 years",
     ["about 1 year", "about 5 years", "about 7 years"]),
    ("the Russo-Japanese War", "1904 to 1905", "about 18 months",
     ["about 6 months", "about 3 years", "about 5 years"]),
    ("the Watergate scandal", "1972 to 1974", "about 2 years",
     ["about 6 months", "about 4 years", "about 8 years"]),
    ("the Space Shuttle program", "1981 to 2011", "about 30 years",
     ["about 15 years", "about 40 years", "about 50 years"]),
    ("Concorde's passenger service", "1976 to 2003", "about 27 years",
     ["about 12 years", "about 35 years", "about 45 years"]),
    ("Boeing 747 production", "1969 to 2023", "about 54 years",
     ["about 30 years", "about 40 years", "about 60 years"]),
    ("the Franco-Prussian War", "1870 to 1871", "about 10 months",
     ["about 3 months", "about 2 years", "about 4 years"]),
    ("the US involvement in the Vietnam War", "1965 to 1973",
     "about 8 years", ["about 3 years", "about 12 years", "about 20 years"]),
    ("the Mexican Revolution", "1910 to 1920", "about 10 years",
     ["about 4 years", "about 15 years", "about 25 years"]),
    ("the Whitlam government in Australia", "1972 to 1975", "about 3 years",
     ["about 1 year", "about 6 years", "about 10 years"]),
    ("the Weimar Republic", "1919 to 1933", "about 14 years",
     ["about 7 years", "about 20 years", "about 30 years"]),
]

DURATION_STEMS = [
    "How long did {name} last?",
    "The {name} ran from {span}. How long did it last in total?",
    "For roughly how long was {name} ongoing?",
    "The {name} spanned {span}. What was its total duration?",
]

DIAL_Q_WHEN = [
    "When did {sp} say that {ev} took place?",
    "According to the transcript, on what date did {ev} happen?",
    "On what date, as stated in the conversation, did {ev} occur?",
]
DIAL_Q_SESSION = [
    "In which session was {ev} discussed?",
    "Which session of the conversation covered {ev}?",
]
STORY_Q = ("Which of the two endings is the most plausible correct ending "
           "to the story?")
LONGFORM_STEMS = [
    "Describe what happened between {d1} and {d2} according to the passage.",
    "Summarize the sequence of developments concerning {sub} that the passage records.",
    "What developments does the passage report between {d1} and {d2}?",
]


class BlockC:
    def __init__(self, arts: list[Article], topics: TopicIndex,
                 rng: random.Random) -> None:
        self.rng = rng
        self.arts = arts
        self.topics = topics
        self.bal = LetterBalancer()
        self.usage: Counter[str] = Counter()
        self.seen_q: set[str] = set()
        self.qtoks: list[set[str]] = []
        self.sess_gold: Counter[str] = Counter()

    def take(self) -> Article | None:
        return self.take_from(self.arts)

    def take_from(self, pool: list[Article], cap: int = 1) -> Article | None:
        for _ in range(300):
            a = self.rng.choice(pool)
            if self.usage[a.sid] < cap:
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

    def gen_dialogue(self, n: int = 250, pool: list[Article] | None = None) -> list[dict]:
        rows: list[dict] = []
        guard = 0
        use_pool = pool if pool else self.arts
        while len(rows) < n and guard < n * 30:
            guard += 1
            art = self.take_from(use_pool)
            if not art:
                break
            s = sorted({(e.date.y, e.date.m or 0): e for e in art.events
                        if e.date.precision >= 2}.values(),
                       key=lambda e: e.date.sort_key())
            if len(s) < 3:
                continue
            ctx, facts = dial_context(art, s, self.rng,
                                      words_band=(1500, 2500),
                                      n_sessions=self.rng.choice((2, 3)))
            if not facts:
                continue
            f = self.rng.choice(facts)
            ev = (clean_fact(f["event"]) or f["event"]).rstrip(" ,;:.")
            if len(ev.split()) < 4:
                continue
            roll = self.rng.random()
            if roll < 0.55:
                q = self.rng.choice(DIAL_Q_WHEN).format(sp=f["speaker"], ev=ev)
                gold = f["date_str"]
                dis = [g["date_str"] for g in facts
                       if g["date_str"] != gold and g["event"] != ev]
                uniq = list(dict.fromkeys(dis))[:3]
                if len(uniq) < 3:
                    continue
                opts = [gold] + uniq
                row_q, gold_text, _ = build_mcq(
                    q, opts, 0, "temporal_dialogue", self.bal, self.rng,
                    relax=self.rng.random() < 0.22)
                if not self.fresh(row_q):
                    continue
                rows.append(make_row("C", "temporal_dialogue", "dial", row_q,
                                     ctx, gold_text,
                                     f"In session {f['session']}, {f['speaker']} "
                                     f"dates \"{ev}\" to {f['date_str']}.",
                                     art.sid))
            elif roll < 0.80:
                q = self.rng.choice(DIAL_Q_SESSION).format(ev=ev)
                gold = f"Session {f['session']}"
                if self.sess_gold[gold] >= 45:
                    continue
                if not self.fresh(q):
                    continue
                self.sess_gold[gold] += 1
                rows.append(make_row("C", "temporal_dialogue", "dial", q, ctx,
                                     gold,
                                     f"The exchange about \"{ev}\" occurs in "
                                     f"session {f['session']}.", art.sid))
            else:
                others = [g for g in facts if g["event"] != ev][:3]
                if not others:
                    continue
                gold = f["speaker"]
                spk = ({g["speaker"] for g in facts}
                       | {g["asker"] for g in facts})
                dis = [x for x in spk if x != gold]
                for x in SPEAKERS:
                    if len(dis) >= 3:
                        break
                    if x != gold and x not in dis:
                        dis.append(x)
                dis = dis[:3]
                if len(dis) < 3:
                    continue
                q = (f"Which speaker dated \"{ev}\" to {f['date_str']}?")
                row_q, gold_text, _ = build_mcq(
                    q, [gold] + dis, 0, "temporal_dialogue", self.bal,
                    self.rng, relax=self.rng.random() < 0.22)
                if not self.fresh(row_q):
                    continue
                rows.append(make_row("C", "temporal_dialogue", "dial", row_q,
                                     ctx, gold_text,
                                     f"{f['speaker']} states the date in "
                                     f"session {f['session']}.", art.sid))
        return rows[:n]

    def gen_duration(self, n: int = 150) -> list[dict]:
        rows: list[dict] = []
        plan = []
        while len(plan) < n:
            for i in range(len(DURATION_ITEMS)):
                plan.append(i)
        self.rng.shuffle(plan)
        used: Counter[str] = Counter()
        for idx in plan[:n]:
            name, span, gold, dis = DURATION_ITEMS[idx]
            stem = DURATION_STEMS[used[idx] % len(DURATION_STEMS)]
            used[idx] += 1
            q_head = stem.format(name=name, span=span)
            opts = [gold] + dis
            q, gold_text, _ = build_mcq(
                q_head, opts, 0, "duration", self.bal, self.rng,
                relax=self.rng.random() < 0.22)
            if not self.fresh(q):
                continue
            rat = f"The {name} ran from {span}, which is {gold}."
            rows.append(make_row("C", "duration", "none", q, "", gold_text,
                                 rat, ""))
        return rows[:n]

    def gen_storytelling(self, n: int = 150) -> list[dict]:
        rows: list[dict] = []
        guard = 0
        while len(rows) < n and guard < n * 30:
            guard += 1
            art = self.take()
            if not art:
                break
            s = sorted({e.date.key(): e for e in art.events
                        if 7 <= len(e.sent.split()) <= 24}.values(),
                       key=lambda e: e.date.sort_key())
            if len(s) < 4:
                continue
            ctx_events = s[:3]
            ending = s[3]
            intro = (f"{art.title}. The record, in order, reads as follows.")
            body = " ".join(e.sent for e in ctx_events)
            ctx = re.sub(r"\s+", " ", f"{intro} {body}").strip()
            if not in_context(ctx_events[-1].sent, ctx):
                continue
            gold = trim_sentence(ending.sent, 22)
            if len(gold.split()) < 5:
                continue
            wrong = swap_one_token(gold, self.rng.choice(s[:3]), self.rng)
            if not wrong or wrong == gold:
                alt = [e for e in s[4:]]
                if not alt:
                    continue
                wrong = trim_sentence(alt[0].sent, 22)
            if not wrong or wrong == gold:
                continue
            q, gold_text, _ = build_mcq(
                STORY_Q, [gold, wrong], 0, "storytelling", self.bal, self.rng,
                relax=self.rng.random() < 0.22)
            if not self.fresh(q):
                continue
            rows.append(make_row("C", "storytelling", "news", q, ctx,
                                 gold_text,
                                 f"The story's three dated steps lead to the "
                                 f"recorded next event: {gold[:110]}", art.sid))
        return rows[:n]

    def gen_longform(self, n: int = 150) -> list[dict]:
        from .ctxbuild import wiki_context
        rows: list[dict] = []
        guard = 0
        n_news = round(n * 0.6)
        while len(rows) < n and guard < n * 30:
            guard += 1
            art = self.take()
            if not art:
                break
            prov = "news" if len(rows) < n_news else "wiki"
            if prov == "news":
                rel = self.topics.related(art, 4)
                fillers = [a for a in rel if a.sid != art.sid][:3]
                while len(fillers) < 3:
                    fillers.append(self.rng.choice(self.arts))
                ctx, _per = news_context(art, fillers, self.rng.randint(0, 2), self.rng)
                src = art
            else:
                ctx = wiki_context(art, art.events, self.rng)
                src = art
            evs = [e for e in src.events if in_context(e.sent, ctx)]
            if len(evs) < 3:
                continue
            evs = sorted({e.date.key(): e for e in evs}.values(),
                         key=lambda e: e.date.sort_key())
            d1, d2 = evs[0].date, evs[-1].date
            sub = subject_of(src)
            stem = self.rng.choice(LONGFORM_STEMS)
            if "{d1}" in stem:
                q = stem.format(d1=d1.text, d2=d2.text)
            else:
                q = stem.format(sub=sub)
            if not self.fresh(q):
                continue
            seq = [trim_sentence(e.lead, 14).rstrip(" .") for e in evs[:5]]
            gold = " ".join(s + "." for s in seq if s)
            rat = ("The passage records, in order: "
                   + "; ".join(f"{e.lead} ({e.date.text})" for e in evs[:5]) + ".")
            rows.append(make_row("C", "longform_free", prov, q, ctx, gold,
                                 rat, src.sid))
        return rows[:n]
