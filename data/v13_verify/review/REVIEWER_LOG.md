# Review log — who reviewed what

Verdict files in this directory carry a `"reviewer"` field when known:

- (no field) — verdicts written before 2026-10-06, by the project author
  (human) with model assistance; treated as human-reviewed.
- `"ai"` — verdicts written on 2026-10-06 by an AI reviewer (GLM, via
  opencode) following `data/v13_verify/REVIEW_BRIEF.md`, after the user
  chose "Proceed, flag as AI-reviewed". These are pending a human
  spot-check before `review_removed.json` is distilled from them.

## AI pass, 2026-10-06

Scope: completion of all in-progress packets (math_text_01, short_01,
glm2_lang_02), then glm3_01 through glm3_11 except glm3_06 and the
nli_mcq/nli_saq rows of glm3_10 (session budget ended there). Verdict lines
from this pass include `"reviewer": "ai"`.

Method:
- nli / storytelling / temporal_dialogue / extract-question rows: read in
  full, judged per `REVIEW_BRIEF.md`.
- extract (multi-select date) rows: verified with an exact date-match script
  (`scripts/review_v13/check_extract.py`) that recomputes the gold letters
  from the context's literal dates; cross-validated against 100+ rows also
  read by hand. 310 rows, 0 disagreements.
- relation rows: verified with `scripts/review_v13/check_relation.py`
  (span/point semantics per the brief) plus manual reading of the 15
  same-time prose rows it skips. 218 rows, 1 checker-gold disagreement
  adjudicated by hand in favour of the gold.
- duration rows: year arithmetic recomputed
  (`check_rel_dur.py`); 4 commonsense rows read by hand.

Findings of this pass (verdicts not CORRECT): glm3_03 #110 (neutral gold vs
entailed engineer opening the extension), glm3_05 #74 (entailment gold for
"found before the bank holiday" - 29 Aug 1967 is after the 28 Aug holiday),
glm3_04 #117 (AMBIGUOUS: coal train "planned" vs "ran"), glm3_05 #88
(AMBIGUOUS: 25 days vs "three weeks"), math_text_01 2x WRONG (TIME
"most-recent-after" convention), glm2_lang_02 1x WRONG (level-crossing case
date). Overall glm3 error rate found: 3/1,151 rows.

Spot-check guidance: sample ~30 of the AI lines (weight towards
WRONG/AMBIGUOUS/MALFORMED verdicts, since those cause removals); compare
against the row in the matching packet file. If disagreement exceeds ~10%,
re-read the affected slice.

## AI pass 2 (continuation), later on 2026-10-06

Scope: glm3_06 and glm3_10 nli rows (completing the whole glm3 slice);
restored_00..03 (447 rows) and timeqa_01 (225 rows) reviewed by parallel
AI subagents (fresh contexts) given `REVIEW_BRIEF.md` plus the written
conventions above; their output was spot-checked by the lead reviewer
(every WRONG/AMBIGUOUS re-read where feasible plus random CORRECT samples).
One subagent verdict was amended (a262d96268ce, AMBIGUOUS->WRONG: 1978 vs
1981 are different calendar decades). Spot-checks otherwise agreed.

Findings this pass: glm3 slice totals 5 WRONG + 5 AMBIGUOUS over 1,401 rows
(~0.4%); restored slice 6 WRONG + 9 AMBIGUOUS over 447 (~1.3% bad);
timeqa_01 17 WRONG + 8 MALFORMED + 2 AMBIGUOUS + 49 NOT_FOUND over 250
(the NOT_FOUNDs await the full-passage second look, per the brief).

## AI pass 3, later on 2026-10-06

Scope: glm2_lang_03..22 (2,377 rows), math_text_02..11 (~951 rows), and
timeqa_02..03 (500 rows), all via parallel AI subagents with lead-reviewer
spot-checks of flagged rows.

IMPORTANT CONVENTION CORRECTION: the "most recent X after Y = NEXT event"
convention (a TIME-benchmark annotation quirk) had been misapplied to
math_text Relative_Reasoning rows. Those questions LITERALLY ask "which came
most recently", where the correct answer is the LATEST event recorded in the
passage after the anchor. 51 verdicts across math_text_00..05 were re-audited
under literal semantics: 51 WRONG -> CORRECT (verified: distractor choices
not recorded in the passage, gold = latest recorded), 1 stayed WRONG
(53694e4b793f, phrased "came FIRST after" and the gold is not first).
Correct rule going forward: match the question's literal phrasing —
"first X after Y" = next event; "most recent X after Y" = latest recorded.
Edited lines carry "reaudit": "literal-semantics".

Findings this pass: glm2_lang_03..22 ~47 WRONG + 4 AMBIGUOUS over 2,377
(~2% bad — recurring modes: abstain-vs-derivable Localization, window
omissions in longform_free, relative-date arithmetic like "last night");
math_text_02..11 ~11 WRONG + ~14 AMBIGUOUS over ~951 (~1%, mostly calendar
arithmetic and tolerance calls); timeqa_02..03 37 WRONG + 22 MALFORMED +
8 AMBIGUOUS + 88 NOT_FOUND over 500 (~12% bad, consistent with 00/01).

## AI pass 4, later on 2026-10-06 — READING REVIEW COMPLETE

Scope: timeqa_04..13 (2,442 rows) via five parallel AI subagents;
lead-reviewer spot-checks of flagged rows confirmed the subagents' verdicts
(Sarkozy pre-presidency window, gold literally "died", Deputy-PM-for-PM,
etc.). Final totals over all 10,278 packet rows:
CORRECT 9,080 / WRONG 346 / AMBIGUOUS 100 / NOT_FOUND 601 / MALFORMED 151.
Removal candidates = WRONG + MALFORMED (497); AMBIGUOUS (100) need a human
call; NOT_FOUND (601) are NOT removals — they await a full-passage second
look, for which the candidate passage source is
`data/benchmarks/timebench/TimeBench/TimeBench-subset-7553/TimeQA/timeqa_{easy,hard}_timebench.jsonl`
(verify those files contain full passages and can be matched to row
questions/evidence before relying on them).

## Second look + blind re-verification, 2026-10-06 (checks 1 and 2)

Second look (NOT_FOUND): all 601 NOT_FOUND rows were matched to full
passages in `data/combined_80_20_v12/{train,val}.jsonl` (work files under
`data/v13_verify/review_second_look/`), re-judged by 3 subagents and
settled in place (marker "secondpass": "full-passage"): 450 CORRECT,
42 WRONG, 109 AMBIGUOUS. No NOT_FOUND verdicts remain.

Blind re-verification: all 748 non-CORRECT verdicts (539 removal
candidates + 209 AMBIGUOUS) were re-judged fresh and blind by 5
independent subagents (work files and re-verdicts under
`data/v13_verify/review_verify/`). Agreement on removal candidates: ~79%.
Reconciliation policy (mechanical, conservative):
- verifier CORRECT vs original WRONG/MALFORMED -> downgraded to AMBIGUOUS,
  kept in training (59 rows, marker "verify": "disagreed-kept")
- verifier AMBIGUOUS vs original WRONG/MALFORMED -> AMBIGUOUS, kept
  (29 rows, "verify": "context-insufficient")
- both non-CORRECT -> removal confirmed (451 rows, "verify": "confirmed")
- AMBIGUOUS + verifier CORRECT -> committed CORRECT (80 rows)
- AMBIGUOUS + verifier WRONG/MALFORMED -> stays AMBIGUOUS, leans-bad
  (35 rows); AMBIGUOUS + AMBIGUOUS stays (94 rows)

FINAL verdict totals: 9,610 CORRECT / 326 WRONG / 217 AMBIGUOUS /
125 MALFORMED. Removal candidates (WRONG+MALFORMED, all confirmed by two
independent reviews): 451. AMBIGUOUS (217) kept in training, flagged for
human disposition. Recurring verifier disagreements: in-window tenure
starts, ~2-month duration tolerance, form-vs-fact (MALFORMED vs WRONG).
