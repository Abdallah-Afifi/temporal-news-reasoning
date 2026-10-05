"""v13 GLM wave: master prompt + ORDERS + API packets (docs/v13_plan.md §3).

Targets the buckets that need language quality (not arithmetic): TRAM
relation/NLI/storytelling/dialogue, TIME Timeline/Relative_Reasoning/.../
Extract (new card), TimeBench dialogue. ~4,500 rows, run on GLM-5.2.

Reuses the v9/v12 brief machinery from make_glm_packets.py, with TWO card
changes:
  - `relation` EXTENDED: half the rows use TRAM's event-to-time surface form
    ("What is the relationship between the event 'dipped' and the time
    'April 1, 1997'?"), which no previous arm trained on;
  - `extract` NEW: the first training analogue of TIME's Extract category
    (multi-select time expressions over TimeDial-style transcripts, gold
    "B  C" two-space joined).

Usage:
    # chat-UI workflow (as v12): master brief + one ORDER per turn
    venv/bin/python scripts/make_v13_glm_orders.py
    # then paste data/glm_packets_v13/MASTER_PROMPT_v13.md into GLM-5.2 chat,
    # send the ORDER lines from ORDERS_v13.txt, save each reply verbatim to
    # data/glm_raw_v13/<NNN>_<category>.txt

    # or API workflow:
    venv/bin/python scripts/make_v13_glm_orders.py --packets
    venv/bin/python scripts/run_v13_glm.py            # needs GLM_API_KEY
    # either way, then:
    venv/bin/python scripts/ingest_glm_batch.py data/glm_raw_v13/*.txt \
        --out data/manual_aug_glm_v13 --plan data/glm_packets_v13/_plan.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_glm_packets import (  # noqa: E402
    CARDS, SHAPES, SLICE, DOMAINS, ERAS, NO_CONTEXT, NEED_DATES, HEADER,
    TASK_CHAT, MASTER_HEAD, MASTER_TAIL,
)

PER_ORDER = 10

# docs/v13_plan.md §3 allocation (pre-registered). (category, rows, shape,
# passages). Shape "dial" = the TimeDial-shaped transcript written per row
# into `context` (like storytelling's story shape); "none" = self-contained.
PLAN = [
    ("relation", 700, "none", 0),
    ("nli_saq", 400, "news", 3),
    ("nli_mcq", 250, "news", 3),
    ("temporal_dialogue", 400, "masked", 0),
    ("storytelling", 400, "story", 0),
    ("Timeline", 400, "news", 3),
    ("Computation", 300, "news", 3),
    ("Relative_Reasoning", 350, "news", 3),
    ("Duration_Compare", 300, "news", 3),
    ("Order_Compare", 200, "news", 3),
    ("duration", 250, "none", 0),
    ("ordering", 250, "none", 0),
    ("extract", 300, "dial", 0),
]

V13_SLICE = dict(SLICE)
V13_SLICE["extract"] = "C"

RELATION_V13 = CARDS["relation"] + """

V13 EXTENSION -- EVENT-TO-TIME SHAPE (use for HALF the rows). TRAM's actual
surface form is not event-vs-event but:
  `<one or two sentences describing an event with explicit dates> What is
  the relationship between the event '<verb from the sentence>' and the time
  '<a date>'?`
Write it exactly like that: quote the sentence's own verb as the event, and
pick a date whose relation to that event is one of the five gold labels:
  - BEFORE: the event happened strictly earlier than the time
  - AFTER: strictly later
  - IS_INCLUDED: the event falls INSIDE a period the time expression names
    (e.g. the event happened on 12 March 1997 and the time is 'March 1997')
  - INCLUDES: the event itself spans a stated interval that CONTAINS the
    time (e.g. 'the festival ran from 1998 to 2004' and the time is 2001)
  - SIMULTANEOUS: the event happened exactly at that time
No Choices block is needed for the event-to-time shape -- TRAM asks it bare
and the model must emit the bare label. The event-vs-event shape (above)
keeps its Choices block. Rotate golds across all five labels in both
shapes; do not let BEFORE dominate."""

EXTRACT_V13 = """TIME's Extract category, which no arm has ever trained on. Multi-select
time expressions over a dated transcript.

For EVERY row write your own short transcript as the `context`, in the
shape of a multi-session chat with dated headers:

  Session 1 happened at 12:04 am on 18 January, 2020.
  Audrey: Hey! Anything new?
  Rosalva: We finally booked the venue on January 27, 2020.

Two to four sessions, 60-120 words total, two named speakers, each session
stamped with a time and date. State further dates naturally inside turns
("on March 2, 2020", "by 14 February, 2020"). Keep every date in ONE of
these surface forms so it can be machine-matched:
  `<Month> <D>, <YYYY>`  (e.g. January 27, 2020)
  `<D> <Month>, <YYYY>`  (e.g. 27 January, 2020)

The `question` field is, VERBATIM:
`Which of the following are time expressions mentioned in the context?
(Note: There may be one or more correct options. And the time expressions
are mentioned directly or indirectly in the context.)`
then a Choices block with FOUR options (five in about one row in three).
Every option is a date in one of the two surface forms above.

GOLD RULES -- the gate recomputes all of this, so follow exactly:
  - Correct options are dates that appear VERBATIM (character-identical) in
    your transcript; distractors are mutated dates (a day/month/year off,
    or a year changed) that appear NOWHERE in it.
  - 1-4 of the options are correct. Vary the count: about 30% of rows have
    one correct option, 40% two, 25% three, 5% four.
  - `targets` holds the correct LETTERS in ascending order joined by TWO
    SPACES, exactly as TIME prints multi-select golds: `B  C` -- never
    `B,C`, never `B C`.
  - Spread which letters are correct across the batch so no position wins
    by default; keep every distractor the same surface form and rough
    length as the correct options.
The `rationale` must list the dates the context actually mentions."""

V13_CARDS = dict(CARDS)
V13_CARDS["relation"] = RELATION_V13
V13_CARDS["extract"] = EXTRACT_V13

V13_NO_CONTEXT = set(NO_CONTEXT) | {"storytelling", "temporal_dialogue", "extract"}
V13_NEED = dict(NEED_DATES)
V13_NEED["extract"] = 3


def build_master_v13() -> str:
    parts = [MASTER_HEAD.replace("all 18 category cards",
                                 "all 19 category cards")]
    for name, body in SHAPES.items():
        parts.append(f"### shape = {name}\n\n{body}\n")
    parts.append("\n## 5. Category cards\n")
    for cat in V13_CARDS:
        parts.append(f"### {cat}  (slice {V13_SLICE.get(cat, '?')})\n\n"
                     f"{V13_CARDS[cat]}\n")
    parts.append(MASTER_TAIL)
    return "\n".join(parts)


def order_lines() -> list[str]:
    rng = random.Random("v13-orders")
    dom = itertools.cycle(rng.sample(DOMAINS, len(DOMAINS)))
    era = itertools.cycle(rng.sample(ERAS, len(ERAS)))
    lines = []
    for cat, rows, shape, passages in PLAN:
        for j in range(rows // PER_ORDER):
            extra = ""
            if cat == "relation":
                extra = (f" relation_shape="
                         f"{'event_to_time' if j % 2 == 0 else 'event_to_event'}")
            if cat == "duration":
                extra = " duration_shape=commonsense" if j % 2 else ""
            if cat == "ordering":
                extra = (f" ordering_shape="
                         f"{'sequence' if j % 2 else 'truefalse'}")
            lines.append(f"ORDER: category={cat} rows={PER_ORDER} "
                         f"passages={passages} shape={shape} "
                         f"domain=\"{next(dom)}\" era={next(era)}{extra}")
    return lines


def build_packets(out_dir: Path, per_packet: int) -> None:
    """Self-contained one-shot packets for the API runner (same prompts as
    the chat workflow, so replies are interchangeable)."""
    rng = random.Random("v13-packets")
    idx = 0
    plan = []
    for cat, rows, shape, passages in PLAN:
        prov = ("none" if cat in V13_NO_CONTEXT
                else "dial" if cat == "extract" else shape)
        remaining = rows
        while remaining > 0:
            n = per_packet if cat not in V13_NO_CONTEXT else per_packet * 5
            n = min(n, remaining)
            remaining -= n
            idx += 1
            if cat in ("relation", "ordering", "duration"):
                src = ('This category carries NO context. Set `context` to "" '
                       'and `source_id` to "". Invent the events yourself; '
                       'they need only be internally consistent.')
            elif cat == "storytelling":
                src = ('Each row carries its own 4-5 sentence everyday story '
                       'directly in `context`, with `"passage":""` and '
                       '`"provenance":"none"`.')
            elif cat == "temporal_dialogue":
                src = ('Each row carries its own `A: ... B: ...` exchange '
                       'with a `<MASK>` in the QUESTION field above the '
                       'Choices block; set `"context":""`, `"passage":""`, '
                       '`"provenance":"dial"`. About half the rows may '
                       'instead be DATED TRANSCRIPT sessions in `context`.')
            elif cat == "extract":
                src = ('Each row carries its own dated transcript in '
                       '`context` per the card; set `"passage":""`, '
                       '`"provenance":"dial"`.')
            else:
                src = (f"### Passage to write ({prov})\n\n{SHAPES[prov]}\n\n"
                       f'Set `source_id` to "" — these passages are written '
                       f"by you, not drawn from a corpus.")
            dom = DOMAINS[(idx * 3) % len(DOMAINS)]
            era = ERAS[(idx * 5) % len(ERAS)]
            src += (f"\n\n### Diversity directive for THIS batch\n\n"
                    f"Subject area: **{dom}**. Dates fall mainly in "
                    f"**{era}**.\nUse different organisations, people and "
                    f"places in every row — a batch that reuses one cast, or "
                    f"lands on one answer repeatedly, is discarded.")
            task = TASK_CHAT.format(n=n, cat=cat, source_block=src,
                                    k=max(passages, 1) if passages else 1)
            body = HEADER.format(cat=cat, slice=V13_SLICE.get(cat, "A"),
                                 prov=prov, sid_hint="",
                                 card=V13_CARDS[cat], n=n, task_block=task)
            name = f"{idx:03d}_{cat}.md"
            (out_dir / name).write_text(body, encoding="utf-8")
            plan.append({"packet": name, "category": cat, "rows": n,
                         "provenance": prov, "domain": dom, "era": era})
    (out_dir / "_plan.json").write_text(json.dumps(
        {"target_rows": sum(r for _, r, _, _ in PLAN), "packets": len(plan),
         "by_category": {c: r for c, r, _, _ in PLAN}, "plan": plan},
        indent=1))
    print(f"{len(plan)} packets -> {out_dir} "
          f"({sum(r for _, r, _, _ in PLAN)} rows planned)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/glm_packets_v13")
    ap.add_argument("--packets", action="store_true",
                    help="emit one-shot .md packets for scripts/run_v13_glm.py")
    ap.add_argument("--per-packet", type=int, default=25)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.packets:
        build_packets(out_dir, args.per_packet)
        return 0

    master = build_master_v13()
    (out_dir / "MASTER_PROMPT_v13.md").write_text(master, encoding="utf-8")
    lines = order_lines()
    (out_dir / "ORDERS_v13.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    (out_dir / "_plan.json").write_text(json.dumps(
        {"target_rows": sum(r for _, r, _, _ in PLAN),
         "by_category": {c: r for c, r, _, _ in PLAN}}, indent=1))
    print(f"master ({len(master.split())} words) + {len(lines)} orders "
          f"({len(lines) * PER_ORDER} rows) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
