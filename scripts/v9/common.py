"""Row assembly shared by all v9 block generators (schema §2)."""
from __future__ import annotations

import re

from .corpus import Article, Event
from .temporal import strip_date_phrases

FIELDS = ["source_dataset", "slice", "category", "provenance", "question",
          "context", "targets", "rationale", "source", "source_id"]

ABSTAIN = "There is no answer."

SYNONYM_SWAPS = [
    ("more than", "over"), ("approximately", "roughly"), ("began", "started"),
    ("ended", "concluded"), ("stated", "said"), ("noted", "said"),
    ("in order to", "to"), ("due to the fact that", "because"),
    ("a number of", "several"), ("prior to", "before"), ("following", "after"),
    ("utilized", "used"), ("purchased", "bought"), ("numerous", "many"),
    ("assistance", "help"), ("additional", "further"), ("rapidly", "quickly"),
]

LEAD_CONNECTIVES = ("However,", "But", "And", "So", "Meanwhile,", "Still,",
                    "Yet", "In addition,", "Moreover,", "Also,", "Then,",
                    "Later,", "Earlier,", "Finally,", "Subsequently,")


def make_row(slice_: str, category: str, provenance: str, question: str,
             context: str, gold: str, rationale: str, source_id: str) -> dict:
    return {
        "source_dataset": "AUG_GLM2",
        "slice": slice_,
        "category": category,
        "provenance": provenance,
        "question": question,
        "context": context,
        "targets": [gold],
        "rationale": rationale,
        "source": "augmented",
        "source_id": source_id,
    }


# Function words a fragment must not end on. AUDIT 2026-09-16: trim_sentence
# and clip both cut at a fixed word count and appended nothing but a period, so
# 128 golds and 187 question stems ended mid-clause ("...the submarine
# disappeared somewhere in.") -- a gold no model can produce and no scorer can
# match.
_DANGLING = {
    "in", "on", "at", "of", "to", "for", "with", "from", "by", "the", "a", "an",
    "and", "or", "that", "as", "into", "after", "before", "when", "who",
    "which", "was", "were", "is", "are", "has", "have", "had", "he", "she",
    "they", "it", "but", "than", "over", "under", "between", "during", "while",
    "its", "his", "her", "their", "this", "these", "those", "about", "against",
}


def strip_dangling(cut: str) -> str:
    """Drop trailing function words so a fragment ends on a content word."""
    w = cut.rstrip(" ,;:.").split()
    while w and w[-1].lower().strip(",;:.") in _DANGLING:
        w.pop()
    return " ".join(w).rstrip(" ,;:")


def trim_sentence(s: str, max_words: int = 28) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if len(words) <= max_words:
        return s
    cut = " ".join(words[:max_words])
    # Prefer ending at a clause boundary inside the window; the original only
    # handled a boundary in the final two tokens.
    #
    # The boundary is taken ONLY if it keeps most of the window. AUDIT
    # 2026-09-19: an earlier version accepted any boundary past the midpoint,
    # which cut distractors well short of their 24-word budget. Golds in
    # Relative_Reasoning are full paraphrased sentences and are NOT capped, so
    # shorter distractors made the gold the longest option in 30% of rows and
    # failed the L7 shortcut probe that had passed at 18%. Keeping >= 80% of
    # the budget preserves the anti-dangling fix without shrinking the option.
    m = None
    for m2 in re.finditer(r"[,;:]", cut):
        if len(cut[: m2.start()].split()) >= int(max_words * 0.8):
            m = m2
            break
    if m:
        cut = cut[: m.start()]
    cut = strip_dangling(cut)
    return (cut + ".") if cut else " ".join(words[:max_words]).rstrip(" ,;:") + "."


def subject_of(art: Article) -> str:
    """A short subject phrase from the title for question stems."""
    t = re.sub(r"[?:!].*$", "", art.title).strip()
    words = t.split()
    for i, w in enumerate(words):
        if w.lower() in ("amid", "after", "before", "as", "over", "following", "amid"):
            return " ".join(words[:i]) if i >= 2 else " ".join(words[:4])
    return " ".join(words[:5])


def paraphrase(s: str, rng=None) -> str:
    """Light meaning-preserving rewrite so a gold can be non-verbatim (L6)."""
    out = s
    for lead in LEAD_CONNECTIVES:
        if out.startswith(lead + " "):
            out = out[len(lead) + 1:]
            break
    for a, b in SYNONYM_SWAPS:
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _dated_tail(s: str) -> str | None:
    m = re.search(r"\b(on|in) ((?:January|February|March|April|May|June|July|August"
                  r"|September|October|November|December)[^.,]{0,20})\.?$", s)
    return m.group(2) if m else None


def move_date_front(s: str) -> str:
    """'X happened on March 3, 2015.' -> 'On March 3, 2015, X happened.'"""
    tail = _dated_tail(s)
    if not tail:
        return s
    head = s[: -(len(tail) + 4)].rstrip(" .")
    if not head:
        return s
    tail_cap = tail[0].upper() + tail[1:]
    head_low = head[0].lower() + head[1:] if head[0].isupper() and not head.split()[0].isupper() else head
    return f"On {tail_cap}, {head_low}."


def swap_one_token(sent: str, pool_event: Event, rng) -> str | None:
    """One-clause swap: date, number, or capitalized entity from a sibling event."""
    from .temporal import extract_dates
    dates = [d.text for d in extract_dates(sent)]
    sib = [d.text for d in extract_dates(pool_event.sent)]
    if dates and sib and set(dates) != set(sib):
        old = rng.choice(dates)
        new = rng.choice([x for x in sib if x != old]) if len(set(sib)) > 1 else None
        if new:
            return sent.replace(old, new, 1)
    nums = re.findall(r"\b\d[\d,.]*\b", sent)
    sib_nums = re.findall(r"\b\d[\d,.]*\b", pool_event.sent)
    if nums:
        old = rng.choice(nums)
        cands = [x for x in sib_nums + [str(int(re.sub(r"[^\d]", "", old) or 3) + max(1, int(re.sub(r"[^\d]", "", old) or 1)))] if x != old]
        if cands:
            return sent.replace(old, rng.choice(cands), 1)
    caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,})?\b", sent)
    sib_caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,})?\b", pool_event.sent)
    if caps and sib_caps:
        old = rng.choice(caps)
        cands = [x for x in sib_caps if x != old]
        if cands:
            return sent.replace(old, rng.choice(cands), 1)
    return None


from .temporal import clean_fact, dated_tail_fact  # noqa: E402,F401
