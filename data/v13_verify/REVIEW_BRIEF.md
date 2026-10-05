# v13 data review — reviewer brief

You are checking training rows for a temporal-reasoning dataset. For EVERY
row in your packet decide whether the stored gold answer is correct given
ONLY the text in the row (question + context/evidence). Do not use outside
knowledge to overrule the passage. Be strict but fair: the goal is to remove
rows that would teach a wrong or arbitrary answer, not to rewrite style.

## Output

Write one JSON object per input row, in input order, to the output file you
are given (JSONL, append as you go so partial work survives):

    {"rid": "<rid>", "verdict": "<VERDICT>", "reason": "<= 25 words>"}

Every input rid must appear exactly once. Verdicts:

- `CORRECT` — the gold follows from the given text.
- `WRONG` — the text supports a different answer than the gold (say which).
- `AMBIGUOUS` — two or more options/answers are equally defensible from the
  text, or the question is unanswerable as worded (MCQ with two right
  options, gold depends on an unstated assumption, contradictory passage).
- `MALFORMED` — broken row: gold not among the options of an MCQ, garbled
  question, empty/truncated gold, question and passage about different things.
- `NOT_FOUND` — (TimeQA slice only) the evidence excerpt neither supports nor
  contradicts the gold; the excerpt may simply have cut the relevant sentence.
  This is NOT a removal — such rows get a full-passage second look.

Rows that are `WRONG`, `AMBIGUOUS` or `MALFORMED` will be removed from
training. Do not over-flag: a terse, oddly phrased but correct row is
`CORRECT`.

## Category notes (glm2_lang slice — LLM-written news/wiki/dialogue rows)

- MCQ: the gold is the FULL TEXT of one option (or a bare label for
  nli/relation/ordering). Check the chosen option is the one the passage
  supports and no other option is equally supported.
- nli_saq / nli_mcq: labels entailment / neutral / contradiction (or the
  project's equivalents) for a premise/hypothesis pair about time. Neutral
  is correct when the passage does not settle the hypothesis.
- Counterfactual: the question posits a changed fact; the gold must follow
  from the passage PLUS that change.
- Localization / Explicit_Reasoning: a date or time expression read off the
  passage; it must match the passage exactly (format differences are fine).
- Order_Reasoning: which event came first/after; check dates in the passage.
- duration: commonsense typical duration (no context); the gold must be the
  clearly most plausible option, others clearly less plausible — if two
  options are about equally plausible, AMBIGUOUS.
- storytelling: pick the ending that best continues the story; the gold must
  be the clearly more coherent/plausible one.
- temporal_dialogue: answer follows from the dialogue's stated times/events.
- longform_free: free-text answer; CORRECT if the gold is a faithful short
  answer supported by the passage.
- Some rows describe an answer as absent ("not stated", "no answer", an
  abstain string): CORRECT only if the passage genuinely does not contain it.

## TimeQA slice (Wikipedia, time-scoped facts)

The question asks who/what held a role (or a property) during a time window;
the gold is the answer per the TimeQA convention: the entity that held it
during (some or most of) the window. `evidence` is an excerpt — lead
sentences plus sentences mentioning years near the window or the gold's
name. Judge:
- `CORRECT` if the excerpt shows the gold held the role during the window,
  or for a substantial part of it.
- `WRONG` if the excerpt shows the gold did NOT hold it at any time in the
  window (e.g. the gold's tenure ended before / began after it), or names a
  different entity for the whole window. Name the right one in the reason.
- `NOT_FOUND` if the excerpt doesn't settle it.
- `MALFORMED` if the gold is a fragment / not a valid answer to the question.
