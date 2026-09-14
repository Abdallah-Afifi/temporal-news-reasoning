"""MCQ assembly with the L7 shortcut constraints baked in (ruleset §3, §7.3)."""
from __future__ import annotations

import random
import re
from collections import defaultdict

LETTERS = "ABCDEFG"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class LetterBalancer:
    """Round-robin gold-letter assignment so letters stay within ±5pp of uniform."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, int], defaultdict[str, int]] = {}

    def pick(self, key: str, n_options: int, rng: random.Random,
             allowed: set[str] | None = None) -> str:
        c = self.counts.setdefault((key, n_options), defaultdict(int))
        letters = [LETTERS[i] for i in range(n_options)]
        if allowed:
            letters = [l for l in letters if l in allowed] or list(LETTERS[:n_options])
        best = min(letters, key=lambda l: (c[l], rng.random()))
        c[best] += 1
        return best


def option_stats(question_head: str, placed: list[str], gold_slot: int) -> dict:
    """Per-row shortcut facts used to constrain placement."""
    qtoks = _tokens(question_head)
    lengths = [len(o.split()) for o in placed]
    overlaps = [len(_tokens(o) & qtoks) for o in placed]
    return {
        "gold_uniquely_longest": lengths[gold_slot] == max(lengths)
        and lengths.count(max(lengths)) == 1,
        "gold_uniquely_max_overlap": overlaps[gold_slot] == max(overlaps)
        and overlaps.count(max(overlaps)) == 1,
    }


def _permute(options: list[str], gold_idx: int, slot: int) -> list[str]:
    n = len(options)
    gold_text = options[gold_idx]
    rest = [o for i, o in enumerate(options) if i != gold_idx]
    out: list[str] = [""] * n
    out[slot] = gold_text
    ri = 0
    for i in range(n):
        if i != slot:
            out[i] = rest[ri]
            ri += 1
    return out


_PADS = [", among other reported details", ", according to the record",
         ", as elsewhere noted", ", in the same account"]


def harden_options(question_head: str, options: list[str], gold_idx: int,
                   rng: random.Random) -> list[str]:
    """L7 fix: length and overlap are placement-invariant, so when the gold
    is uniquely longest / uniquely max-overlap, lengthen or enrich a
    distractor instead of moving letters. Only for sentence-style options."""
    opts = list(options)
    if len(opts) < 3 or sum(len(o.split()) for o in opts) / len(opts) < 8:
        return opts
    lens = [len(o.split()) for o in opts]
    if lens[gold_idx] == max(lens) and lens.count(max(lens)) == 1:
        j = max((i for i in range(len(opts)) if i != gold_idx),
                key=lambda i: lens[i])
        guard = 0
        while len(opts[j].split()) <= lens[gold_idx] and guard < 4:
            opts[j] = opts[j].rstrip(".,;:") + rng.choice(_PADS)
            guard += 1
    head_t = _tokens(question_head)
    ov = [len(_tokens(o) & head_t) for o in opts]
    if ov[gold_idx] == max(ov) and ov.count(max(ov)) == 1 and ov[gold_idx] > 0:
        j = max((i for i in range(len(opts)) if i != gold_idx),
                key=lambda i: ov[i])
        adds = sorted(head_t - _tokens(opts[j]) - _tokens(opts[gold_idx]),
                      key=len, reverse=True)[:2]
        if adds:
            opts[j] = opts[j].rstrip(".,;:") + ", involving " + \
                " and ".join(adds)
    return opts


def build_mcq(question_head: str, options: list[str], gold_idx: int,
              key: str, balancer: LetterBalancer, rng: random.Random,
              relax: bool = False) -> tuple[str, str, dict]:
    """Assemble '<head>\nChoices:\nA. ...' with the gold placed robustly.

    Unless ``relax`` is set, the gold is never placed so that it is uniquely
    the longest option or uniquely the max-token-overlap option; ~22% of rows
    are built relaxed so the inverse cue is not learnable either. Returns
    (question_text, gold_option_text, placement_info); the gold option text is
    returned verbatim and becomes the single target (§2).
    """
    if not relax:
        options = harden_options(question_head, options, gold_idx, rng)
    gold_peek = options[gold_idx]
    if any(o == gold_peek for o in options if o is not gold_peek):
        seen = set()
        dedup: list[str] = []
        for i, o in enumerate(options):
            if o == gold_peek and i != gold_idx:
                continue
            if o in seen and o != gold_peek:
                continue
            seen.add(o)
            dedup.append(o)
        options = dedup
        gold_idx = options.index(gold_peek)
    n = len(options)
    gold_text = options[gold_idx]
    letters = list(LETTERS[:n])
    allowed: set[str] | None = None
    if not relax and n >= 3:
        allowed = set()
        for slot, l in enumerate(letters):
            stats = option_stats(question_head, _permute(options, gold_idx, slot), slot)
            if not stats["gold_uniquely_longest"] and not stats["gold_uniquely_max_overlap"]:
                allowed.add(l)
        if not allowed:
            allowed = None
    letter = balancer.pick(key, n, rng, allowed)
    slot = letters.index(letter)
    placed = _permute(options, gold_idx, slot)
    q = question_head + "\nChoices:\n" + "\n".join(
        f"{LETTERS[i]}. {t}" for i, t in enumerate(placed))
    stats = option_stats(question_head, placed, slot)
    return q, gold_text, {"letter": letter, "gold_slot": slot,
                          "options": placed, **stats}


def mcq_letters(question_text: str, n: int | None = None) -> list[str]:
    """Extract the option letters present in a Choices block."""
    m = re.search(r"Choices:\s*\n((?:[A-G]\..*\n?)+)\s*$", question_text)
    if not m:
        return []
    return re.findall(r"^([A-G])\.", m.group(1), flags=re.M)
