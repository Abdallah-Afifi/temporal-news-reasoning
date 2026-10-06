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

## math_text slice (date arithmetic / comparison written in free text)

Rows a parser could not read. Work the answer out yourself from the context
(dates in the passage), then compare with the gold:
- Computation: elapsed time between two dated events; the gold must equal
  the calendar difference (e.g. "1 year 2 months 3 days"; "24 days").
- Timeline: the letters in chronological order of the facts, each fact
  dated somewhere in the context. If a fact is never dated, or two facts
  can't be ordered from the text, AMBIGUOUS.
- Duration_Compare: compare the two spans; "approximately the same" is
  right only when the spans are within ~10% (or ~2 months) of each other;
  if the gap is borderline, AMBIGUOUS.
- Order_Compare: which of two events happened first / same time.
- Relative_Reasoning: "most recent X after Y" / "immediately after Y" means
  the NEXT event after Y (the TIME benchmark's convention), not the latest
  one overall; "most recent before T" means the closest before T.
- Co_temporality: which event happened at the same time / during another.
A gold that the text gives no way to reach is AMBIGUOUS, not CORRECT.

## short slice (no or tiny context)

- AUG_GLM: short arithmetic / clock / calendar word problems. Recompute the
  answer; WRONG if it differs, AMBIGUOUS if the wording allows two answers.
- AUG_GLM2 relation (BEFORE/AFTER/INCLUDES/IS_INCLUDED/SIMULTANEOUS) and
  ordering rows: check the label against the dates/facts stated in the
  question itself. For "relationship between the event E and the time T":
  E INCLUDES T when E's span contains T; E IS_INCLUDED when E falls inside
  T's span.


## restored slice and glm3 slice (GLM-written rows)

Rows the GLM model wrote one by one (older corpus files and the v13 wave).
Apply the same per-category rules as the glm2_lang slice (and the math_text
rules for any Computation / Timeline / Duration_Compare / Order_Compare /
Relative_Reasoning row). Extra categories here:
- relation: the label must match the dates in the question/context
  (see the short-slice note on INCLUDES vs IS_INCLUDED).
- extract (multi-select): the gold lists, two-space separated (e.g. "B  C"),
  exactly the options that are time expressions mentioned in the context,
  directly or indirectly (a date written in another format still counts).
  A missing or extra letter is WRONG.
- Explicit_Reasoning: a fact or date read directly off the passage.

## Resuming

If your verdicts file already has lines, keep them and continue from the
first rid in the packet that is not yet in the file.
