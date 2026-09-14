# Synthetic data ruleset — LLM generation for the v9 arm

> **Status:** written 2026-09-14. This is the instruction document handed to the
> generating LLM, and the acceptance contract the generated data must pass
> before it reaches a trainer. Derived from the measured behaviour of arms
> v1 → v7c (`docs/results_and_methodology.md`), the coverage law in
> `docs/v7_plan.md` §7a, and the distribution-audit incident D50/D51.
>
> **Precedent:** `data/manual_aug/glm_batch*.jsonl` — 502 LLM-written rows that
> moved TIME `Counterfactual` **+13.1pp**. That slice is the proof small
> targeted generation works. This document is that slice done at scale and
> across the whole benchmark surface.

---

## 0. How to use this file

The generator is run **per category, per batch of ~50 rows**, never as one
open-ended "generate temporal data" prompt. Each run gets:

1. §1 (the laws) and §2 (the schema) verbatim,
2. the single §5 category card for the category being generated,
3. one **real source passage** drawn from `data/corpus/ccnews/` (Mode A), or
   nothing at all for the two Mode-B slices marked in §4.

Batches are then concatenated and must pass §7 before training. A batch that
fails §7 is regenerated, not patched.

---

## 1. The nine laws

These are not style preferences. Each one is a measured failure this project
has already paid for.

**L1 — Fixed budget. New rows displace old rows; they never add.**
v7 added +79% rows → **−5.94pp** TIME. v6d added +10.6% rows → **−2.22pp**,
*with no gain on the category its slice targeted*. Growing the mixture at
fixed LR/epochs degrades TIME whatever the rows are. Total generated budget is
**6,000 rows**, swapped into v6's mixture against retired rehearsal and spent
slices. Total training rows stay **≤ 15,000**.

**L2 — Cover everything, or the uncovered categories pay for the covered ones.**
Every TIME category with a slice beat zero-shot; every category without one
lost. The uncovered categories cost **−1.47pp, twice v6's entire margin**.
Allocation is fixed by §4 and no category in it may be dropped for convenience.

**L3 — Maintenance doses are mandatory where zero-shot is already strong.**
v2 took TimeBench `temporal_dialogue` from **78.6% → 46.2%** by not having any.
Categories above 75% zero-shot (`Co-temporality` 79.8, TimeBench `duration`
78.6, `temporal_dialogue` 78.6, TRAM `storytelling` 76.9) get rows whose only
job is to preserve existing behaviour. Block C in §4.

**L4 — The model must keep the ability to emit a bare class label.**
TRAM is 57.5% temporal NLI; zero-shot scores 53.8%; v1 (pure span-extraction
training, no classification slice) loses **10.8pp** of it. "Pure span
extraction is the most format-destructive thing you can do to a classification
task." Worse, `AUG_NLI`'s `neutral` rows were filtered out in v3-corr, so no
arm has ever trained on the label that is TRAM's **most common gold**
(104,106 of 282,134). Block B in §4 fixes this and is not optional.

**L5 — Never buy a bucket instead of a capability.**
`AUG_NOANS` (400 rows) scored **99.88%** on the 2,427 TIME items that offer an
abstain option — where the abstain option is the gold in 2,427 of 2,427. That
bucket *is* v6's entire headline margin, and removing it removes the margin.
Meanwhile v6 scores **0.00%** on the 2,532 items whose gold is "no answer"
but which offer no such option, and emits false abstentions **2,424 times**
against zero-shot's 1,291. It learned "if an abstain option is listed, pick
it." Rule: **answerability is decided by the passage, never by the option
list** (§5.12).

**L6 — Match the benchmark's answer-in-context rate; do not maximise it.**
Golds are extractable at wildly different rates per category, and a slice that
is 100% extractable teaches copying where the benchmark requires computing.
`Computation` is **0.8% extractable**. Per-category targets are in §5 and are
audited in §7.

**L7 — Balance gold position within every category.**
v6's Counterfactual +13.1pp was verified *not* a shortcut precisely because
gold positions were balanced (A/B/C/D = 1,444/1,556/1,440/1,530), "pick the
longest option" was only 26.9% accurate, and v6 agreed with it *less* than
zero-shot. Any slice failing §7's shortcut probes is worthless regardless of
how good the questions read.

**L8 — Distractors must be type-plausible.**
Same answer type, same granularity, same surface length band as the gold. The
v6 builder found implausible distractors teach a shallow cue that does not
transfer.

**L9 — Sample-reading is not an audit.**
D50: the first news generation was **60.8% unusable** — one distinct answer
across all 4,476 `timeline_construction` rows, 100% `false` in `temporal_nli`,
69.3% fabricated `YYYY-01-01` dates — and the 300-row sample "looked fine."
It was caught only by distinct-answer counts and answer-in-context rates.
§7 runs on **every** batch before training.

---

## 2. Output schema

One JSON object per line. UTF-8. No markdown fences, no commentary, no
trailing prose — the batch file is parsed with `json.loads` per line.

```json
{"source_dataset":"AUG_GLM2","slice":"A|B|C","category":"<§4 name>","provenance":"news|wiki|dial|none","question":"<text>","context":"<text or empty string>","targets":["<gold>"],"rationale":"<1-3 sentences>","source":"augmented","source_id":"<ccnews id or empty>"}
```

Field rules:

| field | rule |
|---|---|
| `source_dataset` | always the literal `"AUG_GLM2"` |
| `slice` | `"A"`, `"B"` or `"C"` per §4 |
| `category` | exactly one of the §4 names, case-sensitive |
| `provenance` | how `context` was built (§6); `"none"` for Mode-B rows |
| `question` | the full prompt including options when the row is MCQ (§3) |
| `context` | the passage, or `""` for Mode-B rows. Never the string `"None"` |
| `targets` | **exactly one element** — the gold answer text |
| `rationale` | supervision for the reasoning arm; never leaks into `question` |
| `source_id` | the CC-News article id the row was grounded on, for the purge |

**`targets` is single-element.** Dual gold `[text, LETTER]` was removed in v7c
and that arm is the protocol-robust headline; do not reintroduce it. For MCQ
rows the single element is the **full text of the correct option**, not its
letter — except `Timeline`, whose gold genuinely is a letter sequence, and
Block B, whose gold is a bare class label.

---

## 3. Question formatting

MCQ rows carry their options inside `question`, in this exact shape:

```
<question text>
Choices:
A. <option>
B. <option>
C. <option>
D. <option>
```

Three-option categories (`Order_Compare`, `Duration_Compare`, Block B NLI) use
A–C. Do not renumber, do not use parentheses, do not add "The answer is".

Free-text rows carry no `Choices:` block. Where the benchmark attaches a
format hint, reproduce it — 8,877 `Computation` items carry:

```
(Hint: Please answer in the form of Month Day, Year. e.g. 1 year 2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)
```

---

## 4. Allocation — 6,000 rows

Rows are allocated by **leverage** = category share of the benchmark ×
(1 − zero-shot accuracy), discounted where the ceiling is structural.

### Block A — TIME categories (4,400 rows)

| category | n in TIME | zero-shot | leverage | **rows** | MCQ / free | mode |
|---|---|---|---|---|---|---|
| `Computation` | 9,372 | 18.8% | 7.25pp | **800** | 0 / 100 | A |
| `Timeline` | 13,159 | 16.5% | 10.47pp † | **650** | 100 / 0 | A |
| `Localization` | 9,894 | 41.4% | 5.53pp | **550** | 0 / 100 | A |
| `Counterfactual` | 9,720 | 42.6% | 5.32pp | **500** | 61 / 39 | A |
| `Duration_Compare` | 9,217 | 40.3% | 5.24pp | **450** | 100 / 0 | A |
| `Relative_Reasoning` | 10,265 | 46.9% | 5.19pp | **450** | 58 / 42 | A |
| `Order_Reasoning` | 10,266 | 49.9% | 4.90pp | **400** | 58 / 42 | A |
| `Co_temporality` | 8,289 | 44.4% | 4.39pp | **300** | 50 / 50 | A |
| `Explicit_Reasoning` | 9,720 | 58.5% | 3.84pp | **200** | 61 / 39 | A |
| `Order_Compare` | 9,897 | 61.1% | 3.67pp | **100** | 100 / 0 | A |

† `Timeline` leverage is capped: 1,798 of its items are 8-element parenthesis
permutations with a 1/40,320 chance rate. **Generate none of those.** All 650
rows are the letter-sequence form (`A,B,C` … over 3–5 facts).

**`Extract` is deliberately excluded** (3,340 items, contributed −0.06pp in
v6 — the lowest leverage in the benchmark, 1,800 of it ships
`Context: "None"` and is unanswerable for every arm, and it is multi-select
scored). Excluding it is a decision, not an oversight; record it.

**`Order_Compare` gets a maintenance dose only.** v6 already wins it +7.6pp,
and v7's `AUG_MCQ2` ×3.8 scale-up drove it **68.7% → 40.4%**. More is
actively dangerous here.

### Block B — class-label emission (900 rows)

Serves TRAM (57.5% NLI, the largest single deficit in the campaign) and
TimeBench `temporal_nli` (6,965 items).

| category | shape | **rows** | label set | mode |
|---|---|---|---|---|
| `nli_saq` | premise + hypothesis, **no options at all** | **400** | `entailment` / `neutral` / `contradiction` | A |
| `nli_mcq` | same, 3 options | **250** | same | A |
| `relation` | event pair in one sentence | **150** | `IDENTITY` / `BEFORE` / `DURING` | B |
| `ordering` | two-sentence claim, True/False | **100** | `TRUE` / `Undetermined` / `FALSE` | B |

`nli_saq` carries no options — that is the whole point; the model must emit the
bare token. Label balance per sub-slice must be within **30–37% each**;
`neutral` is the single most common TRAM gold and has never appeared in any
arm's training data.

### Block C — anti-forgetting maintenance (700 rows)

Preserves behaviour where zero-shot is strong and fine-tuning historically
collapses. These rows target no gain; they are insurance.

| target | zero-shot | **rows** | mode |
|---|---|---|---|
| dialogue-transcript QA (`Co-temporality`, TimeBench `temporal_dialogue`) | 79.8 / 78.6% | **250** | A (dial) |
| duration lookup, world-knowledge form (TimeBench `duration`) | 78.6% | **150** | B |
| plausible-ending selection (TRAM `storytelling`) | 76.9% | **150** | A |
| long-form free answer over a passage | — | **150** | A |

### Provenance split inside Block A

TIME is 58.8% News, 35.3% Wiki, 4.5% Dial. Match it:

- **60% `news`** — three-passage retriever output (§6.1). This is the shape no
  arm has ever trained on and it is the single largest structural gap.
- **35% `wiki`** — single narrative passage (§6.2).
- **5% `dial`** — dated dialogue transcript (§6.3).

---

## 5. Category cards

Each card gives the question form, the gold form, the target answer-in-context
rate (L6), and a worked example. Follow the card exactly; the benchmark's own
phrasing is reproduced here from real items.

### 5.1 `Computation` — 800 rows, free text, in-context 0–3%

Date arithmetic over the passage. The gold must be **computed, never copied** —
if the answer string appears anywhere in the context, discard the row. Two
dates are stated in the passage; the question asks for the span between them,
or a date offset from one of them.

```
Q: How many days passed between the initial opposition to Netanyahu's speech
   on February 19, 2015, and the announcement of his address to Congress on
   February 27, 2015?   (Hint: Please answer in the form of Month Day, Year.
   e.g. 1 year 2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)
GOLD: 8 days
```

- Answer format follows the hint: `8 days`, `2 months 14 days`, `1 year 3 months`.
- The `rationale` states the two dates and the subtraction. This is the
  category CoT exists for.
- Vary the unit scale: ~40% under 30 days, ~40% months, ~20% spanning years.

### 5.2 `Timeline` — 650 rows, MCQ letter sequence, in-context n/a

```
Q: Below are 3 facts. You need to sort these facts in chronological order.
   Requirements: You must output a sequence of uppercase letters separated by
   commas, such as 'A,B,C', without any other characters.
   Choices:
   A. <fact>
   B. <fact>
   C. <fact>
GOLD: C,B,A
```

- 3 facts (60%), 4 facts (30%), 5 facts (10%).
- **Shuffle the facts before labelling them.** D51's defect was presenting
  events in article order, making the identity permutation the gold every
  single time — 1 distinct answer across 4,476 rows.
- All permutations must appear near-uniformly (§7).
- Every fact's date must be recoverable from the passage; do not require
  outside knowledge to order them.

### 5.3 `Localization` — 550 rows, free text, in-context ~70%

"When did X happen?" The gold is a date or date expression, typically 1–3 words.

```
Q: When is Israeli Prime Minister Benjamin Netanyahu scheduled to address the
   US Congress?
GOLD: March 3, 2015.
```

- ~70% of golds appear verbatim in the context; the other ~30% require
  resolving a relative expression ("the following Tuesday", "three weeks later").
- **Never fabricate `YYYY-01-01` from a bare year.** If the passage gives only
  a year, the gold is the year. This was 69.3% of a previous generation.

### 5.4 `Counterfactual` — 500 rows, 61% MCQ / 39% free, in-context ~36%

A hypothetical premise prefixed to a question about what the passage records.

```
Q: If the planned address to Congress in April 2015 was coordinated with the
   White House, what was the focus of the political controversy surrounding
   Israeli Prime Minister Benjamin Netanyahu's planned address to Congress in
   April 2015?
   Choices: A. … B. … C. … D. …
GOLD: The controversy focused on Netanyahu's opposition to the US-led nuclear
      negotiations with Iran and the perceived lack of coordination with the
      White House.
```

- Options are long, near-identical sentences differing in **one clause**. This
  is what makes the category hard and what made the +13.1pp real.
- The counterfactual premise must not change the answer — it tests whether the
  model stays anchored to the passage.

### 5.5 `Duration_Compare` — 450 rows, MCQ, 3 fixed options

```
Q: Which of the following two durations is longer? *Duration 1:* Between <event>
   and <event>. *Duration 2:* Between <event> and <event>.
   Choices:
   A. Duration 1 is longer.
   B. Duration 2 is longer.
   C. The two durations are approximately the same length.
GOLD: Duration 2 is longer.
```

Options are **fixed strings in this order**. Gold distribution must be roughly
40 / 40 / 20. v6 sits exactly at the 33.3% chance rate here — this slice has to
carry real signal, so the two durations must differ by a margin that is
genuinely derivable from stated dates.

### 5.6 `Relative_Reasoning` — 450 rows, 58% MCQ / 42% free, in-context ~48%

"What was the most recent X after Y?" / "immediately after X, what followed?"
Requires an anchor event plus at least three dated candidates. The anchor must
never itself be the gold.

### 5.7 `Order_Reasoning` — 400 rows, 58% MCQ / 42% free, in-context ~100%

"What was the second workshop X attended in 2020?" Ordinal selection over a
dated series. The gold is always present in the passage — this is the one
category where 100% extractability is correct.

### 5.8 `Co_temporality` — 300 rows, 50% MCQ / 50% free, in-context ~46%

"While X was doing A, what was Y doing?" Requires two overlapping intervals.

```
Q: When José Antonio Eguren held the position of auxiliary bishop, what
   position did Mauro Morelli hold?
GOLD: diocesan bishop
```

### 5.9 `Explicit_Reasoning` — 200 rows, 61% MCQ / 39% free, in-context ~61%

"What notable activities did X engage in between <date> and <date>?" Window
filtering over dated events. Distractors are real events from the passage that
fall **outside** the window — this is the type-plausibility rule at its
sharpest.

### 5.10 `Order_Compare` — 100 rows, MCQ, 3 fixed options

```
Q: For Fact1: <fact> and Fact2: <fact>, which one happened earlier?
   Choices:
   A. Fact 1 happened earlier.
   B. Fact 2 happened earlier.
   C. They happen at almost the same time.
```

Maintenance dose only. Gold roughly 45 / 45 / 10.

### 5.11 Block B cards

```
nli_saq / nli_mcq
  context:  <premise sentence>
  question: <hypothesis sentence>          ← no "Choices:" block for nli_saq
  GOLD:     neutral
```

Gold is the **lowercase bare token**. The premise/hypothesis pair must turn on
a temporal relation (order, overlap, duration), not on world knowledge.

```
relation   Q: <sentence stating two events>. What is the relationship between
              the events?  Choices: A. IDENTITY  B. BEFORE  C. DURING
           GOLD: BEFORE                       ← uppercase, matches TRAM

ordering   Q: <event 1>. Then <event 2>. - True/False?
              Choices: A. TRUE  B. Undetermined  C. FALSE
           GOLD: TRUE
```

`Undetermined` must be the gold in ~25% of `ordering` rows, or the model learns
a binary where the benchmark has three classes.

### 5.12 Answerability — folded into Blocks A and C, not a separate slice

Replacing `AUG_NOANS` (L5). Generate ~8% of Block A rows as **matched pairs**:

- Pair member 1: passage contains the fact; question asked; gold is the fact.
- Pair member 2: **the same passage with that one fact removed**; the same
  question; gold is `There is no answer.`
- The abstain option is present in the option list of **both** members.

This makes answerability a property of the passage. Any generation where the
abstain option's presence predicts the gold is a failed batch.

---

## 6. Context construction

### 6.1 `news` — the retriever-output shape (60% of Block A)

**Exactly three passages**, concatenated, in this format:

```
[1] Title: <headline>, Day: <Month D, YYYY> Content: <article body>
[2] Title: <headline>, Day: <Month D, YYYY> Content: <article body>
[3] Title: <headline>, Day: <Month D, YYYY> Content: <article body>
```

Total ~1,100–1,300 words. Rules:

- Passages come from **real CC-News articles** supplied to the generator, not
  invented. Dates are the articles' real dates.
- The three passages are **topically related but not redundant** — the same
  story from different angles or days, as a retriever would return.
- The passage carrying the gold is at position 1 in ~33% of rows, position 2 in
  ~33%, position 3 in ~33%. Never always first.
- In **~20% of rows the gold passage is absent entirely** and the question is
  answerable only from partial information — these become the §5.12 abstain
  members or, where a partial answer exists, keep the partial gold. This is the
  noisy-retrieval condition that 59% of TIME is scored on and that no arm has
  ever seen.
- Do not write `Context: "None"`. The 1,800 benchmark items shaped that way are
  a source-data defect.

### 6.2 `wiki` — narrative passage (35% of Block A)

A single flowing third-person account, ~700–800 words, no headers, no bullets,
dates woven into prose ("On April 28, 1965, X was ordained…"). Built from a
CC-News article's facts rewritten as a biography- or institution-history-style
narrative.

### 6.3 `dial` — dated transcript (5% of Block A, and Block C's 250)

```
Session 1 happened at 12:04 am on 18 January, 2020. <Speaker A>: … <Speaker B>: …
Session 2 happened at …
```

Multi-session, each session stamped with time and date, two named speakers.
Benchmark transcripts run ~13,000 words; generate **1,500–2,500 words** — long
enough for the cross-session temporal structure, short enough to train on
inside the 4,226-token `max_model_len`.

---

## 7. Acceptance gates

A batch is training-eligible only when **all** of these pass. Run them per
category, not pooled — pooling is how D50's defects hid.

### 7.1 Machine gates (blocking)

```bash
venv/bin/python scripts/validate_manual_aug.py data/manual_aug_v9/*.jsonl
```
Must report `bad=0 collisions=0`. This checks schema, non-empty targets,
in-file duplicates, and exact-question collision against TIME and TimeBench.

Extend the purge before training — the generator now emits **passages**, not
just questions, so question-level matching is not sufficient:

| check | threshold |
|---|---|
| exact question collision vs TIME / TimeBench / **TRAM** | 0 |
| punctuation-insensitive collision | 0 |
| token-set Jaccard > 0.8 vs any benchmark question | 0 |
| **shared-passage detection** — any 30-token shingle shared with a benchmark context | 0 |
| in-batch near-duplicate questions (Jaccard > 0.9) | 0 |

### 7.2 Distribution audit (blocking) — the D51 gate

Per category:

| metric | threshold |
|---|---|
| distinct golds | ≥ 6, and no single gold > 20% of the category |
| — except fixed-option categories (`Duration_Compare`, `Order_Compare`, Block B) | each class within the band named on its card |
| answer-in-context rate | within **±10pp** of the card's target (L6) |
| fabricated `YYYY-01-01`-style dates | < 2% |
| `Timeline` permutation coverage | all permutations present, none > 1.5× uniform |

### 7.3 Shortcut probes (blocking) — the L7 gate

Per MCQ category:

| probe | threshold |
|---|---|
| gold letter distribution | each letter within ±5pp of uniform |
| "always pick A" accuracy | ≤ 30% (4-option), ≤ 38% (3-option) |
| "pick the longest option" accuracy | ≤ 30% |
| "pick the option sharing most tokens with the question" accuracy | ≤ 35% |
| abstain-option presence predicts gold | must be **uninformative** (§5.12) |

### 7.4 Gold correctness (blocking where recomputable)

`Computation`, `Duration_Compare` and Block B `relation` golds are
mechanically checkable from the stated dates. Precedent: `AUG_DURATION`
verified **0 incorrect of 1,966**, `AUG_ARITH` **0 incorrect of 1,158**.
Anything below **99% verified-correct on the recomputable subset** fails the
batch. Sample 100 rows per non-recomputable category for manual check.

---

## 8. Prohibited

- Copying, paraphrasing, or reconstructing any benchmark question, passage or
  option. The generator never sees benchmark files.
- Inventing dates, events or entities not present in the supplied source
  passage (Mode A). Mode-B slices are self-contained by design and may invent
  freely, but must remain internally consistent.
- Emitting `targets` with more than one element (v7c).
- Emitting the letter alone as the gold for a non-`Timeline` MCQ row.
- Answer-bearing preamble in `question` ("The answer is…", "Note that X
  happened in 1998…").
- Rationale text leaking into `question` or `context`.
- Growing any category past its §4 row count (L1).
- Padding a shortfall. If a category cannot reach its count under these rules,
  **report the shortfall** — `AUG_RELATIVE` delivered 524 of 2,000 requested
  and reporting it was correct.

---

## 9. Answer style

TIME's mean gold is **2.35 words, median 1**. Keep golds terse:

| category | typical gold |
|---|---|
| `Localization` | `March 3, 2015.` |
| `Computation` | `8 days` / `2 months 14 days` |
| `Timeline` | `C,B,A` |
| `Co_temporality` (free) | `diocesan bishop` |
| Block B | `neutral` / `BEFORE` / `TRUE` |
| MCQ categories | the full option text, matching the option verbatim |

The MCQ exception matters: the gold must be **character-identical** to the
option as it appears in `question`, so the scorer's exact match can fire.

---

## 10. Batch protocol

```
data/manual_aug_v9/
  <category>_batch01.jsonl     ~50 rows each
  <category>_batch02.jsonl
  ...
  AUDIT.md                     §7 output per category, kept with the data
```

1. Generate one batch (~50 rows).
2. Run §7.1 and §7.2 on that single batch. Fix the *prompt*, not the rows.
3. Only once a category's first three batches pass clean, generate the rest.
4. Run §7 again over the whole category before it enters a builder.
5. Write the §7 output into `AUDIT.md`. A slice with no recorded audit does not
   go into a trainer — that rule is what D51 cost.

---

## 11. What this data is measured against

The arm built on it is a **swap at fixed size**, and is pre-registered against
these, per category, before training:

| benchmark | metric | current bar |
|---|---|---|
| TIME | `no_abstain_pct`, **stratified by `Setting`** | zero-shot 41.19% pooled; base 27.21%, retrieved ~51% |
| TIME | per-category, all 11 | the §4 zero-shot column |
| TimeBench | overall + per-category | zero-shot 45.18% |
| TRAM | micro **and** macro, `temporal_nli` broken out | zero-shot 46.95% micro, 53.8% on NLI |

A win is: **no category regresses more than 1pp, and the covered categories
gain.** The campaign's entire history says the second half is easy and the
first half is where every arm has died.
