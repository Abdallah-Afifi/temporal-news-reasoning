"""CC-News corpus access for the v9 synthetic arm (ruleset §6).

The local corpus snapshot carries no publish metadata (url/date/domain are
stripped), so:
  * ``source_id`` is minted as ``<file-stem>:<line-no>`` — stable for the purge;
  * passage ``Day:`` headers use the article's principal explicitly stated
    date, never a fabricated one.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .temporal import D, extract_dates, fmt_full

CORPUS_FILES = [
    "ccnews_2023_en.jsonl",
    "ccnews_2024_en.jsonl",
]

STOP = set("""a about above after again against all also an and any are as at be
because been before being below between both but by can could did do does doing
down during each few for from further had has have having he her here hers him
his how i if in into is it its just like me more most my no nor not now of off
on once only or other our out over own same she should so some such than that
the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you
your says said say new news report reports according latest update updates
first second third two three one four five six seven eight nine ten make made
get got take took go went come came""".split())

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u201c(\[])")
_ABBREV_DOT = re.compile(r"\b([A-Z][a-z]{1,4})\.(?=\s)")


@dataclass
class Event:
    sid: str
    date: D
    sent: str
    lead: str = ""
    pos: int = 10 ** 9

    @property
    def words(self) -> list[str]:
        return [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-']+", self.sent.lower()) if t not in STOP]


@dataclass
class Article:
    sid: str
    title: str
    text: str
    day: D | None = None
    events: list[Event] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)

    @property
    def words(self) -> int:
        return len(self.text.split())


def _sentences(text: str) -> list[str]:
    out = []
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        prot = _ABBREV_DOT.sub("\\1\u00b7", chunk)
        parts = [s.replace("\u00b7", ".") for s in SENT_SPLIT.split(prot)]
        out.extend(s.strip() for s in parts if s and s.strip())
    return out


def _clean_sent(s: str) -> str | None:
    s = re.sub(r"\s+", " ", s).strip()
    w = s.split()
    if not (6 <= len(w) <= 110):
        return None
    if re.search(r"http|www\.|@|All rights reserved|\bUPI\b|Follow (us|him|her)", s):
        return None
    if not re.search(r"[a-z]", s) or not re.match(r"[A-Z\"'\u201c]", s):
        return None
    return s


def load_articles(root: Path, ban: set[str] | None = None,
                  max_articles: int | None = None) -> list[Article]:
    seen_titles: set[str] = set()
    arts: list[Article] = []
    for fn in CORPUS_FILES:
        path = root / fn
        if not path.exists():
            continue
        stem = fn.split("_en")[0]
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                sid = f"{stem}:{i:06d}"
                if ban and sid in ban:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = (rec.get("title") or "").strip()
                text = (rec.get("text") or "").strip()
                if not title or len(text.split()) < 130:
                    continue
                tkey = re.sub(r"\W", "", title.lower())[:60]
                if not tkey or tkey in seen_titles:
                    continue
                seen_titles.add(tkey)
                art = Article(sid=sid, title=title, text=text)
                _index_article(art)
                if art.events:
                    arts.append(art)
                if max_articles and len(arts) >= max_articles:
                    return arts
    return arts


def _index_article(art: Article) -> None:
    body = art.text.split()
    lead = " ".join(body[:90])
    full = extract_dates(lead)
    art.day = next((d for d in full if d.precision == 3), None) or \
              next((d for d in extract_dates(art.title) + full if d.precision >= 2), None)
    pos = 0
    for sent in _sentences(art.text):
        cs = _clean_sent(sent)
        n_words = len(sent.split())
        if cs:
            dates = extract_dates(cs)
            if dates and any(d.y >= 1900 for d in dates):
                seen: set[tuple] = set()
                for d in dates:
                    if d.key() in seen:
                        continue
                    seen.add(d.key())
                    art.events.append(Event(sid=art.sid, date=d, sent=cs,
                                            lead=_event_lead(cs), pos=pos))
        pos += n_words
    art.tokens = {t for t in re.findall(r"[a-zA-Z]{4,}", (art.title + " " + lead).lower())
                  if t not in STOP}


def _event_lead(sent: str) -> str:
    """A compact verbatim phrase naming the event (for question quotes)."""
    s = re.sub(r"\s+", " ", sent).strip()
    s = re.sub(r"^[A-Z][a-z]+ (Reporter|Correspondent)[:\s]+", "", s)
    clauses = re.split(r"(?<=[,;])\s+(?=[A-Z])", s)
    for c in clauses:
        w = c.split()
        if 5 <= len(w) <= 22:
            return c.strip(" ,;:")
    return " ".join(s.split()[:20]).strip(" ,;:")


class TopicIndex:
    """Token-overlap grouping standing in for the retriever of §6.1."""

    def __init__(self, arts: list[Article]) -> None:
        self.post: dict[str, set[str]] = defaultdict(set)
        self.by_sid: dict[str, Article] = {}
        for a in arts:
            self.by_sid[a.sid] = a
            for t in a.tokens:
                self.post[t].add(a.sid)
        self._pruned = {t: s for t, s in self.post.items() if 2 <= len(s) <= 120}

    def related(self, art: Article, k: int, used: set[str] | None = None) -> list[Article]:
        cand: Counter[str] = Counter()
        for t in art.tokens:
            for sid in self._pruned.get(t, ()):  # type: ignore[union-attr]
                cand[sid] += 1
        out = []
        for sid, _ in cand.most_common():
            if sid == art.sid or (used and sid in used):
                continue
            other = self.by_sid[sid]
            overlap = len(art.tokens & other.tokens)
            if overlap >= 2:
                out.append(other)
            if len(out) >= k:
                break
        return out


def passage_text(art: Article, max_words: int) -> str:
    body = art.text.split()[:max_words]
    day = fmt_full(art.day) if art.day and art.day.precision == 3 else ""
    if not day:
        if art.day and art.day.precision == 2:
            day = f"{art.day.text}"
        else:
            day = "Not stated"
    return f"Title: {art.title}, Day: {day}\nContent: {' '.join(body)}"


def has_full_pair(events: list[Event]) -> bool:
    fulls = [e for e in events if e.date.precision == 3]
    return len({e.date.key() for e in fulls}) >= 2


def series_of(events: list[Event], min_len: int) -> list[list[Event]]:
    """Date-sorted event series for one article, deduped by date."""
    by_date: dict[tuple, Event] = {}
    for e in events:
        by_date.setdefault(e.date.key(), e)
    s = sorted(by_date.values(), key=lambda e: e.date.sort_key())
    return [s] if len(s) >= min_len else []
