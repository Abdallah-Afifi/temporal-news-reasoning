"""Emit ready-to-paste GLM chat packets for the v9 synthetic arm.

WHY THIS EXISTS. The supplied API account cannot run the job: glm-4.6/4.5/
4.5-air return "insufficient balance" (code 1113) and the free glm-4.5-flash
trips a hard rate limit (code 1302) under even light load. Generating in the
GLM chat UI sidesteps both -- and it is what the ruleset always described
(`prompts/glm_news_temporal_items.md`: "Send the block below once per news
article ... Collect the JSONL output"). The 502 genuine AUG_GLM rows in
data/manual_aug/ were made this way, which is why that directory is "manual".

WORKFLOW
  1. venv/bin/python scripts/make_glm_packets.py --rows 3000
       -> data/glm_packets/<NNN>_<Category>.md, each self-contained
  2. Paste one packet into GLM chat. Save the reply verbatim to
       data/glm_raw/<same name>.txt
  3. venv/bin/python scripts/ingest_glm_batch.py data/glm_raw/*.txt
       -> validates against the CURRENT ruleset and writes the accepted rows
          to data/manual_aug_glm/, reporting what each packet still owes.

Every packet carries its own article text, so packets are independent and can
be done in any order, by more than one person, across sessions.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# §4 allocation, scaled by --rows. Mode B categories need no article.
ALLOC = {
    "Computation": 800, "Timeline": 650, "Localization": 550,
    "Counterfactual": 500, "Duration_Compare": 450, "Relative_Reasoning": 450,
    "Order_Reasoning": 400, "Co_temporality": 300, "Explicit_Reasoning": 200,
    "Order_Compare": 100, "nli_saq": 400, "nli_mcq": 250, "relation": 150,
    "ordering": 100, "temporal_dialogue": 250, "duration": 150,
    "storytelling": 150, "longform_free": 150,
}
SLICE = {c: "A" for c in ("Computation", "Timeline", "Localization",
                          "Counterfactual", "Duration_Compare",
                          "Relative_Reasoning", "Order_Reasoning",
                          "Co_temporality", "Explicit_Reasoning",
                          "Order_Compare")}
SLICE.update({"nli_saq": "B", "nli_mcq": "B", "relation": "B", "ordering": "B",
              "temporal_dialogue": "C", "duration": "C", "storytelling": "C",
              "longform_free": "C"})
# Categories whose rows carry NO context at all.
NO_CONTEXT = {"relation", "ordering", "duration"}

# RECORDED DEVIATION (2026-09-16, researcher's decision).
# Ruleset §6.1 says news passages "come from real CC-News articles supplied to
# the generator, not invented". They are now INVENTED by the model, and the
# corpus seed is dropped entirely. Consequences, stated so nobody has to
# rediscover them:
#   + no corpus dependency, packets are small, benchmark collision is ~0 by
#     construction (still gate-checked in §7.1).
#   + the template generator's worst defects came from ingesting real page
#     furniture -- bylines, nav text, CAPTCHA fragments used as gold answers.
#     Invented prose cannot carry those.
#   - invented passages are cleaner than real news, so the train/test surface
#     differs from TIME's genuinely messy contexts.
#   - the D50 risk (one generation came back 60.8% unusable, ONE distinct
#     answer across 4,476 rows) is a diversity collapse, and unseeded
#     generation is more exposed to it. Mitigated by the per-packet domain and
#     era rotation below, and caught by the §7.2 distinct-gold gate.
# The §6 passage SHAPES are kept exactly -- the ruleset calls the three-passage
# retriever shape "the shape no arm has ever trained on and the single largest
# structural gap", and that is about form, not provenance.

# Rotated per packet so 100+ packets do not all describe the same world.
DOMAINS = [
    "national politics and elections", "courts and legal rulings",
    "corporate earnings, mergers and layoffs", "public health and medicine",
    "climate, weather and natural disasters", "professional sport",
    "transport, aviation and rail", "energy and utilities",
    "universities, schools and research", "municipal government and planning",
    "technology products and outages", "agriculture and food supply",
    "labour disputes and strikes", "central banking and inflation",
    "film, music and cultural institutions", "space and astronomy",
    "shipping, ports and trade", "telecoms and infrastructure",
    "insurance and natural-catastrophe claims", "archaeology and heritage",
]
ERAS = ["1890-1935", "1936-1965", "1966-1989", "1990-2004",
        "2005-2014", "2015-2019", "2020-2024"]

# §6 passage shapes. Provenance split inside Block A matches TIME: 60/35/5.
SHAPES = {
"news": """Each passage is EXACTLY THREE sub-passages, concatenated, in this format:

[1] Title: <headline>, Day: <Month D, YYYY> Content: <article body>
[2] Title: <headline>, Day: <Month D, YYYY> Content: <article body>
[3] Title: <headline>, Day: <Month D, YYYY> Content: <article body>

Total roughly 700-1000 words across the three. Rules:
- The three passages are TOPICALLY RELATED BUT NOT REDUNDANT -- the same story
  from different angles or on different days, as a search engine would return.
- Rotate which sub-passage carries the answer: [1] for about a third of
  rows, [2] a third, [3] a third. NEVER always the first.
- In about 1 row in 5, the sub-passage carrying the full answer is ABSENT.
  EXCEPT in `Order_Reasoning`, `Timeline` and `Duration_Compare`, which carry
  NO absent rows -- see their cards.
  SPLIT THOSE ROWS ROUGHLY IN HALF:
    * about half keep a PARTIAL answer as the gold -- the passage still
      supports a coarser answer (the year but not the day, one endpoint but
      not the span). Prefer this whenever any real answer survives.
    * about half abstain, and then the gold must be ONE OF THESE FOUR, varied
      between rows, not the same one every time:
        There is no answer.
        Cannot be determined from the context.
        The passage does not say.
        None of the options is supported by the passage.
  Answerability is decided by the PASSAGE, never by whether an abstain option
  happens to be listed.
  WHY THIS MATTERS: a slice where one abstain string is 20% of the golds
  teaches "when unsure, abstain" instead of teaching the task. A previous arm
  learned exactly that and emitted 2,424 false abstentions against a baseline's
  1,291. No single gold string may exceed 10% of a category.
- Never write `Context: "None"`.""",

"wiki": """Each passage is ONE flowing third-person narrative, roughly
450-700 words: no headers, no bullets, no bylines. Dates woven into the prose
("On April 28, 1965, the institute was founded..."). Biography or
institution-history style.""",

"dial": """Each passage is a dated multi-session transcript, roughly 400-700 words:

Session 1 happened at 12:04 am on 18 January, 2020.
<Speaker A>: ...
<Speaker B>: ...
Session 2 happened at 9:30 pm on 4 March, 2020.
...

Two named speakers, several sessions, each stamped with a time and date, and
dates stated naturally in conversation. EVERY turn must carry content -- do not
pad with a repeated filler exchange.""",
}
# How many dated events an article needs to support the category.
NEED_DATES = {"Computation": 2, "Timeline": 3, "Duration_Compare": 4,
              "Order_Compare": 2, "Relative_Reasoning": 3,
              "Order_Reasoning": 3, "Co_temporality": 2,
              "Explicit_Reasoning": 2, "nli_saq": 2, "nli_mcq": 2,
              "temporal_dialogue": 3, "storytelling": 2, "longform_free": 2,
              "Localization": 1, "Counterfactual": 2}

HINT = ("   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year "
        "2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)")

CARDS = {
"Computation": f"""Date arithmetic over the passage. The gold must be COMPUTED, never copied:
if the answer string appears anywhere in the context, do not emit the row.
Two dates are stated in the passage; ask for the span between them, or for a
date offset from one of them.

MATCH THE BENCHMARK ON WHERE THE DATES APPEAR. In the real benchmark the
question stem prints BOTH dates in 61% of items, and prints NO DATE AT ALL in
37% -- those name the events and leave the model to find their dates in the
passage. So write roughly:
  - 6 rows in 10 naming both dates in the question;
  - 3.5 rows in 10 naming NEITHER date, only the events
    ("How long passed between the drive opening and the target being reached?");
  - the rest naming one.
Do NOT put a date in every stem. A slice that always hands over the dates
teaches subtraction and never teaches reading. Either way the `rationale` must
state both dates and the subtraction -- that is what the arithmetic check
reads.
Append this hint verbatim to the end of every question:
`{HINT.strip()}`
Answer format follows the hint: `8 days`, `2 months 14 days`, `1 year 3 months`.
Vary the scale: ~40% under 30 days, ~40% months, ~20% spanning years.
The `rationale` must state both dates and the subtraction.
Free text: NO Choices block.""",

"Timeline": """Chronological sorting. Question text, verbatim:
`Below are N facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters.`
then a Choices block with the facts as A., B., C., ...
GOLD is the letter sequence, e.g. `C,B,A`.
3 facts (60%), 4 facts (30%), 5 facts (10%).
SHUFFLE the facts before labelling them -- presenting them in article order
makes the identity permutation the gold every time. That defect (D51) produced
ONE distinct answer across 4,476 rows. Vary which permutation is correct.
Every fact's date must be recoverable from the passage.
The `rationale` MUST end with each option's date keyed by letter, in this exact
form, so the ordering can be machine-checked:
  `A=2017-07-09; B=2017-05-06; C=2017-03-11`
Use ISO YYYY-MM-DD there even when the passage words the date differently. If a
fact is dated only to a month, use the first of that month in the key and make
sure the ordering does not depend on the day.""",

"Localization": """"When did X happen?" The gold is a date or date expression, 1-3 words.
~70% of golds appear verbatim in the context; ~30% require resolving a
relative expression ("the following Tuesday", "three weeks later").
NEVER fabricate YYYY-01-01 from a bare year -- if the passage gives only a
year, the gold IS the year. That error was 69.3% of a previous generation.
Free text: NO Choices block.""",

"Counterfactual": """A hypothetical premise prefixed to a question about what the passage records.
The premise must NOT change the answer -- it tests whether the model stays
anchored to the passage.
61% of rows MCQ (4 options), 39% free text.
For MCQ: options are long, near-identical sentences differing in ONE clause.
That is what makes the category hard and what made its +13.1pp real.""",

"Duration_Compare": """Compare two durations. Question shape:
`Which of the following two durations is longer? *Duration 1:* Between <event> and <event>. *Duration 2:* Between <event> and <event>.`
Choices block, these three options in EXACTLY this order and wording:
A. Duration 1 is longer.
B. Duration 2 is longer.
C. The two durations are approximately the same length.
GOLD is the full option text. Gold distribution roughly 33/40/27
(A / B / C). NOT the 40/40/20 an earlier draft asked for: the shortcut probe
caps "always-A" at 38%, and 40% fails it outright. 36% was also tried and
still lands a part-built category at 38.2%. 33% is safe whether or not earlier
batches over-weighted A. The template build recorded the same deviation.
Both durations must be derivable from dates stated in the passage, and must
differ by a real margin.""",

"Relative_Reasoning": """"What was the most recent X after Y?" / "immediately after X, what followed?"
Needs an anchor event plus at least three dated candidates. The anchor must
NEVER itself be the gold.
58% MCQ (4 options), 42% free text.

EXTRACTABILITY IS GATED AT 38-58%, TARGET ~48%. Only about half the golds may
appear in the passage word-for-word; the rest must be PARAPHRASES of what the
passage says, so the row cannot be answered by string-matching. A live batch
came in at 74% verbatim and was rejected for it. Roughly: for every two rows,
write one gold that quotes the passage and one that restates it in different
words.""",

"Order_Reasoning": """Ordinal selection over a dated series: "What was the second workshop X
attended in 2020?" The gold is ALWAYS present in the passage -- this is the
one category where 100% extractability is correct.
58% MCQ (4 options), 42% free text.

NO ABSENT-PASSAGE ROWS IN THIS CATEGORY. The news shape asks for about 1 row
in 5 whose answer is missing from the passage; this category is the documented
exception, because its gate requires 90-100% of golds to appear in the passage
and a partial or abstain gold by definition does not. Write every row fully
answerable and fully extractable. (A live batch carrying 2 partial + 2 abstain
per order landed the category at 80% and failed the gate.)""",

"Co_temporality": """"While X was doing A, what was Y doing?" Requires two overlapping intervals.
Golds are SHORT noun phrases, e.g. `diocesan bishop`.
50% MCQ (4 options), 50% free text.

MATCH DISTRACTOR LENGTH TO THE GOLD. Because golds here are real role titles
("flight dynamics lead", "harbourmaster"), it is easy to write distractors that
are shorter generic roles ("clerk", "chair"). A live batch did exactly that:
the gold was the LONGEST option 44% of the time against a uniform 25%, and ran
+1.56 words on a 2.37-word mean -- "pick the longest" beat chance outright and
the slice failed L7. Every distractor must be a role title of the SAME word
count band as the gold. If the gold is three words, the distractors are two to
four words, not one.

ABSTAIN RATE IS ~10% OF ROWS, NOT 20%. The news shape asks for ~20% of rows to
have the answer ABSENT, and half of those keep a PARTIAL answer; only the other
half abstain. A live batch abstained on all of them.""",

"Explicit_Reasoning": """"What notable activities did X engage in between <date> and <date>?"
Window filtering over dated events. For MCQ, distractors are REAL events from
the passage that fall OUTSIDE the window.
61% MCQ (4 options), 39% free text.""",

"Order_Compare": """Which of two facts happened earlier. Question shape:
`For Fact1: <fact> and Fact2: <fact>, which one happened earlier?`
Choices block, these three options in EXACTLY this order and wording:
A. Fact 1 happened earlier.
B. Fact 2 happened earlier.
C. They happen at almost the same time.
GOLD is the full option text. Gold distribution roughly 35/45/20 (A/B/C) --
B LEADS, NOT A. The ruleset's own figure (45/45/10) puts A at 45%, but the
shortcut probe caps "always-A" at 38% for any 3-option category regardless of
the distribution gate's band -- the exact conflict Duration_Compare hit before
its 40/40/20 was corrected to 33/40/27. Write Fact2 as the correct order
noticeably more often than Fact1, and give "almost the same time" real margin
(within about a day, or a stated approximation) so it carries genuine signal
rather than padding out to 20%.""",

"nli_saq": """Temporal natural-language inference with NO OPTIONS AT ALL. The question is a
premise-and-hypothesis statement; the model must emit the bare label.
GOLD is exactly one of: `entailment`, `neutral`, `contradiction`.
NO Choices block -- that is the whole point of this category.
Balance the three labels within 30-37% each.
`neutral` is TRAM's single most common gold and has never appeared in any
arm's training data, so do not under-produce it.""",

"nli_mcq": """Same as nli_saq but WITH a 3-option Choices block:
A. entailment
B. neutral
C. contradiction
GOLD is the bare label text, e.g. `neutral`.
Balance the three labels within 30-37% each.""",

"relation": """MODE B -- invent the events freely; no source article. Internal consistency
is all that is required.
One sentence describing two events (or an interval and a point inside it),
with explicit dates, then an EXPLICIT interrogative such as
`What is the relationship between the events?`
Choices block, exactly:
A. IDENTITY
B. BEFORE
C. DURING
GOLD is the bare label. Balance the three labels within 30-37% each.
Use varied invented actors (a port authority, a chamber of commerce, a
software studio) and a wide range of years.""",

"ordering": """MODE B -- invent the events freely; no source article.
Two sentences, each with an explicit date, joined as a claim, then the explicit `- True/False?`
Choices block, exactly:
A. TRUE
B. Undetermined
C. FALSE
GOLD is the bare label. NOT a uniform three-way split: roughly TRUE 36%,
FALSE 40%, Undetermined 24%. The ruleset's own target is "Undetermined ~25%,
or the model learns a binary where the benchmark has three classes" -- not a
third. TRUE also sits under the shortcut probe's 38% always-A cap (it is
option A), the same constraint that shaped Duration_Compare and Order_Compare.
`Undetermined` rows are genuinely undecidable from the two stated dates and
the claim's wording -- not simply rare -- so write real ambiguity into them
(e.g. the claim's ordering word is vague, or a date is stated to a coarser
precision that leaves the comparison unresolved), never an artificially
suppressed TRUE/FALSE.""",

"temporal_dialogue": """A dated conversation transcript, then a question about it.
Context shape: sessions with headers like
`Session 1 happened at 4:20 pm on 15 December, 2021.` followed by turns
`Casey: ...` / `Alex: ...` that state dates in conversation.
Ask which session something was discussed in, or on what date an event happened.
MCQ (4 options) for most rows.
DO NOT pad the transcript with a repeated filler line -- a previous generation
repeated one exchange up to 76 times. Every turn must carry content.""",

"duration": """MODE B -- invent freely; no source article. World-knowledge duration lookup.
Shape: `The <thing> ran from <year> to <year>. How long did it last in total?`
or `How long did <well-known event> last?`
Choices block with 4 plausible durations. GOLD is the full option text,
e.g. `about 54 years`. Distractors must be the same TYPE and granularity.""",

"storytelling": """Plausible-ending selection. Question, verbatim:
`Which of the two endings is the most plausible correct ending to the story?`
then a Choices block with exactly TWO options, A. and B.
One ending is consistent with the passage's dates; the other contradicts them
in a single, specific way (a wrong count, a wrong year, an impossible order).
GOLD is the full text of the correct ending.""",

"longform_free": """Free-text summary over a window: "What developments does the passage report
between <date> and <date>?" or "Describe what happened between X and Y."
Answer in 2-4 sentences drawn from the passage, in chronological order.
NO Choices block.""",
}

TASK_INLINE = """Emit EXACTLY {n} JSONL rows for category `{cat}`.
Before you output, check each line has all ten keys including `targets`.
{source_block}
Output the {n} lines and nothing else."""

TASK_CHAT = """{source_block}

### STEP 1 — write {k} passage(s)

Write {k} passage(s), each labelled on its own line EXACTLY like this:

=== PASSAGE P1 ===
<the passage text>
=== PASSAGE P2 ===
<the passage text>

### STEP 2 — write {n} rows

Then emit EXACTLY {n} JSONL rows for category `{cat}`, spread roughly evenly
across the passages.

IN CHAT MODE ONLY: set `"context":""` and add ONE extra key `"passage"` naming
which passage the row uses, e.g. `"passage":"P2"`. Do NOT repeat the passage
text inside the row — it is filled in automatically. Every other rule above
still applies, and every line still needs `targets`.

Output the passages, then the {n} lines, and nothing else."""

HEADER = """You are generating training data for a temporal-reasoning benchmark.
Follow the rules EXACTLY. Output nothing but JSONL.

## OUTPUT FORMAT — one JSON object per line, no markdown fences, no commentary

{{"source_dataset":"AUG_GLM2","slice":"{slice}","category":"{cat}","provenance":"{prov}","question":"<text>","context":"<text or empty string>","targets":["<gold>"],"rationale":"<1-3 sentences>","source":"augmented","source_id":"{sid_hint}"}}

EVERY line MUST contain ALL TEN keys, in this order:
  source_dataset, slice, category, provenance, question, context,
  targets, rationale, source, source_id
A line missing any key is discarded. The most common failure is OMITTING
`targets` and leaving the answer only in `rationale` — do not do that. Write
`targets` for every single row.

WORKED EXAMPLE of a complete line (different category, shown only for shape):
{{"source_dataset":"AUG_GLM2","slice":"A","category":"Localization","provenance":"wiki","question":"When was the Harkness Institute founded?","context":"The Harkness Institute opened its doors on 14 May 1958 after three years of construction ... (a full passage goes here)","targets":["14 May 1958"],"rationale":"The passage states the institute opened on 14 May 1958.","source":"augmented","source_id":""}}

HARD RULES (a row breaking any of these is discarded):
- `targets` has EXACTLY ONE element. Never [text, LETTER]. For an MCQ row the
  single element is the FULL TEXT of the correct option, not its letter —
  except `Timeline`, whose gold is a letter sequence, and the label categories
  (nli_saq, nli_mcq, relation, ordering), whose gold is the bare label.
- The gold must be CHARACTER-IDENTICAL to the option it names.
- MCQ options go inside `question`, in this exact shape:
    <question text>
    Choices:
    A. <option>
    B. <option>
    C. <option>
    D. <option>
  Do not renumber, do not use parentheses, do not write "The answer is".
- ANSWERS MUST BE TERSE. The benchmark's mean gold is 2.35 words, median 1.
  Free-text golds must be SHORT — a date, a number, a noun phrase. Do not
  answer in sentences unless the category card says so.
- Balance gold position: across this batch, A/B/C/D must each be roughly a
  quarter of the MCQ golds.
- The correct option must not be guessable without the passage: not by length,
  not by specificity, not by shared wording with the question, not by being the
  only option that reads as a complete thought. Write every distractor as if it
  were the answer.
- Distractors must be the SAME TYPE, granularity and rough length as the gold.
- Never copy, paraphrase or reconstruct a question from TIME, TimeBench or TRAM.
- `rationale` explains the answer; it must NEVER appear in `question` or `context`.
- Do not invent dates, events or entities that are not in the supplied passage
  (except the MODE B categories, which say so explicitly).
- Every question must be answerable from `context` alone.

## CATEGORY: {cat}

{card}

## YOUR TASK

{task_block}
"""



MASTER_HEAD = """# Temporal-reasoning dataset generation — standing brief

You are generating training data for a temporal-reasoning benchmark. This
message is the STANDING BRIEF: it holds every rule, all 18 category cards and
all three passage shapes. Read it once and keep it in force for the whole
conversation.

After this message I will send short ORDER lines, one per turn, like:

    ORDER: category=Computation rows=10 passages=2 shape=news domain="public health and medicine" era=2015-2019

For each ORDER you reply with the passages and the JSONL rows for that order
ONLY, following everything below. No commentary, no preamble, no summary — the
reply is parsed by a script.

---

## 1. Output format

One JSON object per line. UTF-8. No markdown fences.

{"source_dataset":"AUG_GLM2","slice":"<A|B|C>","category":"<name>","provenance":"<news|wiki|dial|none>","question":"<text>","context":"","passage":"P1","targets":["<gold>"],"rationale":"<1-3 sentences>","source":"augmented","source_id":""}

EVERY line MUST carry ALL TEN keys: source_dataset, slice, category,
provenance, question, context, targets, rationale, source, source_id — plus
`passage` when the order has passages. A line missing any key is discarded.

The most common failure is OMITTING `targets` and leaving the answer only in
`rationale`. Do not do that. Write `targets` on every single row.

WORKED EXAMPLE of a complete line:
{"source_dataset":"AUG_GLM2","slice":"A","category":"Localization","provenance":"wiki","question":"When was the Harkness Institute founded?","context":"","passage":"P1","targets":["14 May 1958"],"rationale":"The passage states the institute opened on 14 May 1958.","source":"augmented","source_id":""}

## 2. Hard rules — a row breaking any of these is discarded

- `targets` has EXACTLY ONE element. Never [text, LETTER]. For an MCQ row the
  single element is the FULL TEXT of the correct option, never its letter —
  except `Timeline`, whose gold is a letter sequence, and the label categories
  (nli_saq, nli_mcq, relation, ordering), whose gold is the bare label.
- The gold must be CHARACTER-IDENTICAL to the option it names — same
  punctuation, same capitalisation, no trailing full stop the option lacks.
- MCQ options live inside `question`, in exactly this shape:

      <question text>
      Choices:
      A. <option>
      B. <option>
      C. <option>
      D. <option>

  Do not renumber, do not use parentheses, never write "The answer is".
- ANSWERS MUST BE TERSE. The benchmark's mean gold is 2.35 words, median 1.
  Free-text golds are a date, a number or a short noun phrase. Do not answer in
  sentences unless the category card says so. This is the single rule a
  previous generation failed worst — its golds averaged 8 words.
- BALANCE GOLD POSITION -- DO THIS BY CONSTRUCTION, NOT BY EYE. Before writing
  each MCQ row, decide its gold letter by CYCLING A, B, C, D, A, B, C, D ...
  through the batch in order, then write the options so the correct answer sits
  at that letter. Do not write the answer first and place it wherever it falls.
  A batch whose golds land mostly on one letter is DISCARDED WHOLE, however good
  the questions are: a live batch shipped 72 of 72 golds at A, which teaches
  "pick A" and nothing else. State the intended letter to yourself for each row
  before writing its options.
- THE CORRECT OPTION MUST NOT BE GUESSABLE WITHOUT THE PASSAGE. Someone shown
  only the question and the options should have no way to prefer the gold: not
  by length, not by specificity, not by how much wording it shares with the
  question, not by being the only one that reads as a complete thought. Write
  every distractor as if it were the answer.
- DISTRACTORS MUST BE TYPE-PLAUSIBLE: same answer type, same granularity, same
  rough length band as the gold. A distractor of a different type makes the row
  solvable without reading.
- ARITHMETIC IS CHECKED BY MACHINE. Every date calculation is recomputed on
  ingest and the row is discarded on any mismatch, including off by one day.
  Work the calculation out explicitly before writing the gold.
- Never copy, paraphrase or reconstruct a question, passage or option from the
  TIME, TimeBench or TRAM benchmarks.
- `rationale` explains the answer and must NEVER appear in `question` or
  `context`.
- Every question must be answerable from its passage alone.
- Vary everything between rows: different organisations, people, places and
  sentence shapes. A batch that reuses one cast, or lands on one answer
  repeatedly, is discarded wholesale.

## 3. How to answer an ORDER

### Step 1 — write the passages

Write `passages=N` passages in the ordered `shape`, each labelled on its own
line EXACTLY like this:

    === PASSAGE P1 ===
    <the passage text>
    === PASSAGE P2 ===
    <the passage text>

### Step 2 — write the rows

Then emit exactly `rows=N` JSONL rows for the ordered category, spread roughly
evenly across the passages. Set `"context":""` and name the passage with
`"passage":"P2"`. Never repeat the passage text inside a row — it is filled in
automatically.

If the ordered category is `relation`, `ordering` or `duration`, write NO
passages: those rows are self-contained. Set `"context":""`, `"passage":""`,
`"provenance":"none"`, and invent the events freely — they need only be
internally consistent.

Output the passages, then the rows, and nothing else.

## 4. Passage shapes

"""

MASTER_TAIL = """
## 6. Slice labels

Use `slice` = "A" for Computation, Timeline, Localization, Counterfactual,
Duration_Compare, Relative_Reasoning, Order_Reasoning, Co_temporality,
Explicit_Reasoning, Order_Compare, nli_saq, nli_mcq; "B" for relation and
ordering; "C" for temporal_dialogue, duration, storytelling, longform_free.

---

Acknowledge this brief in one short line, then wait for the first ORDER.
"""


def build_master() -> str:
    parts = [MASTER_HEAD]
    for name, body in SHAPES.items():
        parts.append(f"### shape = {name}\n\n{body}\n")
    parts.append("\n## 5. Category cards\n")
    for cat in ALLOC:
        parts.append(f"### {cat}  (slice {SLICE[cat]})\n\n{CARDS[cat]}\n")
    parts.append(MASTER_TAIL)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=3000,
                    help="total rows to plan for; §4 allocation is scaled to it")
    ap.add_argument("--per-packet", type=int, default=25,
                    help="rows requested per chat message")
    ap.add_argument("--out", default="data/glm_packets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chat", action="store_true", help=(
        "Chat-friendly packets: the model writes each passage ONCE and rows "
        "reference it by label. A 25-row news packet otherwise needs ~6,000 "
        "words of passage text in one reply and truncates."))
    ap.add_argument("--passages", type=int, default=3,
                    help="passages per chat packet")
    ap.add_argument("--orders", action="store_true", help=(
        "Emit ORDERS.txt: one ORDER line per turn, covering the whole §4 "
        "allocation, to drive the master prompt."))
    ap.add_argument("--master", action="store_true", help=(
        "Emit ONE self-contained MASTER_PROMPT.md carrying every rule, all 18 "
        "category cards and all three passage shapes, instead of per-category "
        "packets. Paste it once per chat session, then drive it one turn at a "
        "time with a short ORDER line. Built from the same CARDS/SHAPES "
        "constants as the packets, so the two cannot drift."))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these categories")
    args = ap.parse_args()

    if args.orders:
        out_dir = PROJECT_ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        scale = args.rows / sum(ALLOC.values())
        want = {c: max(1, int(round(n * scale))) for c, n in ALLOC.items()}
        # Subtract rows already banked, so a re-issued order list never asks
        # for work that is done. Over-shooting a category is not harmless: the
        # §7.2 distribution gate FAILS a category that exceeds its allocation.
        done_dir = PROJECT_ROOT / "data" / "manual_aug_glm"
        import collections as _c
        have: _c.Counter = _c.Counter()
        for f in done_dir.glob("*.jsonl"):
            for line in f.open():
                if line.strip():
                    have[json.loads(line)["category"]] += 1
        if have:
            print(f"already banked: {dict(have)}")
            want = {c: max(0, n - have.get(c, 0)) for c, n in want.items()}
        lines, i = [], 0
        for cat in sorted(want, key=lambda c: -want[c]):
            left = want[cat]
            while left > 0:
                # No-passage categories cost the model ~40 words per row and no
                # passage at all, so a 10-row cap there spends turns for
                # nothing. The reply-length limit that forces small orders
                # applies to the PASSAGE burden, which these do not carry.
                per = args.per_packet * 5 if cat in NO_CONTEXT else args.per_packet
                n = min(per, left)
                left -= n
                i += 1
                if cat in NO_CONTEXT:
                    shape, npass = "none", 0
                elif cat == "temporal_dialogue":
                    shape, npass = "dial", args.passages
                else:
                    r = (i * 7) % 100
                    shape = "news" if r < 60 else ("wiki" if r < 95 else "dial")
                    npass = args.passages
                dom = DOMAINS[(i * 3) % len(DOMAINS)]
                era = ERAS[(i * 5) % len(ERAS)]
                lines.append(
                    f"ORDER: category={cat} rows={n} passages={npass} "
                    f"shape={shape} domain=\"{dom}\" era={era}")
        dest = out_dir / "ORDERS.txt"
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(lines)} orders -> {dest} ({sum(want.values())} rows planned)")
        return 0

    if args.master:
        out_dir = PROJECT_ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "MASTER_PROMPT.md"
        dest.write_text(build_master(), encoding="utf-8")
        print(f"master prompt -> {dest} ({len(build_master().split())} words)")
        return 0

    rng = random.Random(args.seed)
    scale = args.rows / sum(ALLOC.values())
    want = {c: max(1, int(round(n * scale))) for c, n in ALLOC.items()}
    if args.only:
        want = {c: n for c, n in want.items() if c in args.only}

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # MASTER_PROMPT.md is written by --master into the same directory; the
    # earlier blanket glob deleted it on every packet rebuild.
    for old_f in out_dir.glob("*.md"):
        if old_f.name != "MASTER_PROMPT.md":
            old_f.unlink()

    # TIME is 58.8% News / 35.3% Wiki / 4.5% Dial -- match it across Block A
    # packets. temporal_dialogue is dial by definition; the no-context
    # categories get neither a shape nor a provenance.
    def pick_prov(cat: str, k: int) -> str:
        if cat in NO_CONTEXT:
            return "none"
        if cat == "temporal_dialogue":
            return "dial"
        r = (k * 7) % 100          # deterministic, evenly spread
        return "news" if r < 60 else ("wiki" if r < 95 else "dial")

    idx = 0
    plan = []
    for cat in sorted(want, key=lambda c: -want[c]):
        remaining = want[cat]
        k = 0
        while remaining > 0:
            n = min(args.per_packet, remaining)
            remaining -= n
            idx += 1
            k += 1
            prov = pick_prov(cat, idx)
            if cat in NO_CONTEXT:
                src = ('This category carries NO context. Set `context` to "" '
                       'and `source_id` to "". Invent the events yourself; they '
                       'need only be internally consistent.')
            else:
                src = (f"### Passage to write ({prov})\n\n{SHAPES[prov]}\n\n"
                       f'Set `source_id` to "" — these passages are written by '
                       f"you, not drawn from a corpus.")
            dom = DOMAINS[(idx * 3) % len(DOMAINS)]
            era = ERAS[(idx * 5) % len(ERAS)]
            src += (f"\n\n### Diversity directive for THIS batch\n\n"
                    f"Subject area: **{dom}**. Dates fall mainly in **{era}**.\n"
                    f"Use different organisations, people and places in every "
                    f"row — a batch that reuses one cast, or lands on one "
                    f"answer repeatedly, is discarded. Vary sentence structure "
                    f"and headline style between rows.")
            if args.chat and cat not in NO_CONTEXT:
                task = TASK_CHAT.format(n=n, cat=cat, source_block=src,
                                        k=args.passages)
            else:
                task = TASK_INLINE.format(n=n, cat=cat, source_block=src)
            body = HEADER.format(cat=cat, slice=SLICE[cat], prov=prov,
                                 sid_hint="", card=CARDS[cat], n=n,
                                 task_block=task)
            name = f"{idx:03d}_{cat}.md"
            (out_dir / name).write_text(body, encoding="utf-8")
            plan.append({"packet": name, "category": cat, "rows": n,
                         "provenance": prov, "domain": dom, "era": era})

    (out_dir / "_plan.json").write_text(json.dumps(
        {"target_rows": args.rows, "packets": len(plan), "by_category": want,
         "plan": plan}, indent=1))
    print(f"\n{len(plan)} packets -> {out_dir}  (target {args.rows} rows, "
          f"{args.per_packet}/packet)")
    for c in sorted(want):
        print(f"  {c:20s} {want[c]:5d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
