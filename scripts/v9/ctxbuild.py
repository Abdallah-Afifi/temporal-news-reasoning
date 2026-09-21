"""Context construction for the v9 arm (ruleset §6).

news: exactly three passages in the retriever-output shape.
wiki: one flowing narrative built from a real article's dated facts.
dial: multi-session timestamped transcript over a real article's events.
"""
from __future__ import annotations

import random
import re

from .corpus import Article, Event, TopicIndex, passage_text
from .temporal import D, clean_fact, fmt_full

NEWS_WORDS = (1100, 1300)
WIKI_WORDS = (700, 800)
DIAL_WORDS_A = (1200, 2500)
DIAL_WORDS_C = (1500, 2500)

SPEAKERS = ["Alex", "Jordan", "Sam", "Casey", "Riley", "Morgan", "Taylor",
            "Avery", "Quinn", "Dana", "Ellis", "Harper", "Rowan", "Sasha"]

WIKI_OPENERS = [
    "This account traces the sequence of developments in detail.",
    "The record of events, drawn from contemporary reporting, runs as follows.",
    "Taken together, the reporting reconstructs the timeline below.",
    "The developments, arranged chronologically, are set out here.",
]

WIKI_CONNECT = [
    "Subsequently,", "Later on,", "Around that time,", "Some time afterward,",
    "Not long after,", "In a further development,", "Following that,",
    "More recently,", "Earlier,", "By then,",
]

DIAL_OPENERS = [
    "have you been following this?",
    "what do you make of the latest?",
    "did you see the update on this?",
    "any thoughts on how this has been going?",
]

DIAL_ASK = [
    "When did that happen again?",
    "What was the date on that?",
    "Can you pin that down to a day?",
    "When was that exactly?",
]

DIAL_REACT = [
    "That is sooner than I expected.",
    "Interesting, I had it later in my head.",
    "Good, that matches what I read.",
    "Okay, noted for our timeline.",
    "Right, that slots in before the rest.",
]

# Filler exchanges used ONLY to reach the word band when the source article is
# short on datable events. AUDIT 2026-09-16: the previous implementation
# appended ONE verbatim exchange in a `while` loop, so a short article produced
# up to 76 identical copies of the same two lines -- 7.1% of v9 rows carried
# such a block, and all 250 temporal_dialogue rows did. Training on that
# teaches degenerate repetition and spends the context budget on nothing.
# Each exchange here is used AT MOST ONCE per context; if the pool runs out the
# context is returned SHORT rather than padded (ruleset section 8: report a
# shortfall, never pad).
# Emitted with p=0.3 after an event turn. AUDIT 2026-09-16: this used to be ONE
# hardcoded string, so a transcript with many events repeated it verbatim up to
# 14 times. Drawn from a pool now.
DIAL_ASIDE = [
    "We should watch whether anything further comes of it.",
    "Worth keeping an eye on how that develops.",
    "I expect we will hear more about that one.",
    "That may matter later on, depending on what follows.",
    "I had not connected those two until just now.",
    "It is easy to lose track of which came first.",
]

DIAL_FILLER = [
    ("Anything else you remember from that period?",
     "Only that the reporting kept adding detail with each update."),
    ("Was there any follow-up coverage after that?",
     "A little, though it mostly restated what was already known."),
    ("Did the earlier reports line up with this?",
     "Broadly, yes -- the dates were the part people kept revising."),
    ("How widely was this picked up at the time?",
     "Enough that several outlets ran their own version of it."),
    ("Did anyone dispute the sequence of events?",
     "Not the sequence so much as how long the gaps between them were."),
    ("Was the timing ever clarified afterwards?",
     "Only in passing, when a later piece referred back to it."),
    ("Do you recall what prompted the coverage?",
     "It came up alongside the other developments we went through."),
    ("Anything in the follow-ups that changed the picture?",
     "Nothing that moved the dates we already have."),
]


def _clamp(text: str, lo: int, hi: int) -> bool:
    return lo <= len(text.split()) <= hi


def news_context(gold_art: Article | None, others: list[Article], gold_pos: int,
                 rng: random.Random, per_passage: int = 420,
                 gold_per: int | None = None) -> tuple[str, int]:
    """Three passages; gold passage at ``gold_pos`` (0/1/2), or absent if -1.

    Returns (text, per-passage word budget). When ``gold_per`` is given the
    gold passage gets the larger budget and fillers shrink so the §6.1
    1,100–1,300-word total still holds.
    """
    total = rng.randint(*NEWS_WORDS)
    if gold_per:
        fill = max(260, (total - 40 - gold_per) // 2)
    else:
        fill = max(260, (total - 30) // 3)
        gold_per = fill
    arranged: list[Article | None] = [None, None, None]
    budgets = [fill, fill, fill]
    if gold_art is not None and gold_pos in (0, 1, 2):
        arranged[gold_pos] = gold_art
        budgets[gold_pos] = gold_per
    fl = list(others)
    fi = 0
    for i in range(3):
        if arranged[i] is None:
            arranged[i] = fl[fi % len(fl)]
            fi += 1
    text = "\n".join(f"[{i + 1}] " + passage_text(a, budgets[i])  # type: ignore[arg-type]
                     for i, a in enumerate(arranged))
    return text, gold_per


def news_context_no_gold(others: list[Article], rng: random.Random,
                         per_passage: int = 420) -> tuple[str, int]:
    """§6.1 noisy-retrieval condition: gold passage absent entirely."""
    return news_context(None, others, -1, rng, per_passage)


def wiki_context(art: Article, events: list[Event], rng: random.Random,
                 keep_sentence: str | None = None,
                 exclude_str: str | None = None) -> str:
    ordered = sorted(events, key=lambda e: e.date.sort_key())
    parts = [f"{art.title}. {rng.choice(WIKI_OPENERS)}"]
    used: set[int] = set()
    for i, e in enumerate(ordered):
        sent = e.sent
        if i > 0 and rng.random() < 0.5:
            sent = f"{rng.choice(WIKI_CONNECT)} {sent[0].lower() + sent[1:]}"
        parts.append(sent)
        used.add(i)
    filler = [s for s in re.split(r"(?<=[.!?])\s+", art.text)
              if 8 <= len(s.split()) <= 40
              and s not in {e.sent for e in ordered}
              and not (exclude_str and exclude_str in s)]
    rng.shuffle(filler)
    words = len(" ".join(parts).split())
    for s in filler:
        if words >= rng.randint(*WIKI_WORDS):
            break
        if len(s.split()) + words > WIKI_WORDS[1] + 30:
            continue
        parts.append(re.sub(r"\s+", " ", s).strip())
        words += len(s.split())
    if keep_sentence and keep_sentence not in " ".join(parts):
        parts.insert(2, keep_sentence)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def wiki_context_drop(art: Article, events: list[Event], rng: random.Random,
                      drop_sentence: str) -> str:
    ev2 = [e for e in events if e.sent != drop_sentence]
    ctx = wiki_context(art, ev2, rng)
    ctx = ctx.replace(drop_sentence, "")
    ctx = re.sub(r"\s{2,}", " ", ctx).strip()
    return ctx


def _turn(sp: str, txt: str) -> str:
    return f"{sp}: {txt}"


def dial_context(art: Article, events: list[Event], rng: random.Random,
                 words_band: tuple[int, int] = DIAL_WORDS_A,
                 n_sessions: int = 2) -> tuple[str, list[dict]]:
    """Build a dated transcript; returns (context, session_facts).

    session_facts entries: {session, speaker, event, date, date_str} — the
    question builders draw only from these, so every dial gold is recoverable.
    """
    a, b = rng.sample(SPEAKERS, 2)
    ordered = sorted(events, key=lambda e: e.date.sort_key())
    per = max(1, len(ordered) // max(1, n_sessions))
    chunks = [ordered[i:i + per] for i in range(0, len(ordered), per)][:n_sessions]
    while len(chunks) < n_sessions:
        chunks.append(chunks[-1])
    lines: list[str] = []
    facts: list[dict] = []
    words = 0
    for si, chunk in enumerate(chunks, 1):
        hh = rng.randint(7, 22) % 12 or 10
        mm = rng.choice((0, 4, 12, 15, 20, 30, 45))
        ap = rng.choice(("am", "pm"))
        d0 = chunk[0].date
        date_str = fmt_full(d0) if d0.precision == 3 else d0.text
        stated = _eu_style(date_str, rng)
        lines.append(f"Session {si} happened at {hh}:{mm:02d} {ap} on {stated}.")
        words += 12
        opener = rng.choice(DIAL_OPENERS)
        lines.append(_turn(a, opener))
        lines.append(_turn(b, f"There is quite a bit to catch up on from {d0.y}."))
        words += 18
        for ei, ev in enumerate(chunk):
            frag = clean_fact(ev.lead) or ev.lead
            asker, answerer = (a, b) if ei % 2 == 0 else (b, a)
            ev_stated = _eu_style(ev.date.text, rng) if ev.date.precision == 3 else ev.date.text
            lines.append(_turn(asker, f"About \"{frag}\" — {rng.choice(DIAL_ASK)}"))
            lines.append(_turn(answerer, f"That was on {ev_stated}."))
            lines.append(_turn(asker, rng.choice(DIAL_REACT)))
            words += len(frag.split()) + 22
            facts.append({"session": si, "speaker": answerer, "asker": asker,
                          "event": frag, "date": ev.date,
                          "date_str": ev_stated})
            if rng.random() < 0.3:
                extra = rng.choice(DIAL_ASIDE)
                lines.append(_turn(answerer if ei % 2 == 0 else asker, extra))
                words += len(extra.split()) + 1
        lines.append(_turn(b if si % 2 else a,
                           f"Let us pick this up again after the next update."))
        words += 12
    ctx = "\n".join(lines)
    # Reach the word band with DISTINCT filler exchanges, each used at most
    # once. If the pool is exhausted the context stays short -- padding with
    # repeats is what the 2026-09-16 audit found and removed.
    filler = list(DIAL_FILLER)
    rng.shuffle(filler)
    for ask, reply in filler:
        if words >= words_band[0]:
            break
        ctx += f"\n{a}: {ask}\n{b}: {reply}"
        words += len(ask.split()) + len(reply.split()) + 2
    return ctx, facts


def _eu_style(date_str: str, rng: random.Random) -> str:
    """Benchmark transcripts write '18 January, 2020'; convert US style."""
    m = re.match(r"([A-Z][a-z]+) (\d{1,2}), (\d{4})", date_str)
    if m:
        return f"{m.group(2)} {m.group(1)}, {m.group(3)}"
    return date_str


def dial_context_drop_turns(ctx: str, event_fragment: str) -> str:
    """§5.12 member 2 for dial rows: remove the turns carrying the fact."""
    out = []
    kill = False
    for line in ctx.split("\n"):
        if event_fragment in line:
            kill = True
            continue
        if kill:
            kill = False
            continue
        out.append(line)
    return "\n".join(out)
