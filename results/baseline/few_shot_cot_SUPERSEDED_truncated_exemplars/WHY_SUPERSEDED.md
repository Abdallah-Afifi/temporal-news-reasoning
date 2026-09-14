# few-shot CoT arm — SUPERSEDED IN FULL, 2026-09-14

Do not score. Do not quote. Not an arm. Kept as provenance for ~26 GPU-hours.

Legs present: llama/timebench (21,188 rows), mistral/timebench (21,075),
llama/time (35,269 of 104,939 — stopped mid-run 2026-09-14 20:07).

## Why every row is unusable

**1. The exemplars never reached the model.** `_build_cot_fewshot_prompt`
puts the worked examples at the FRONT of the prompt. Inference tokenizes with
`max_length=4096` and `truncation_side="left"`
(`src/models/inference.py:67,150`), so anything over the limit is cut from the
front. The mined exemplar preamble was **9,393 tokens**. Measured:

| | TimeBench | TIME |
|---|---|---|
| prompts over the 4,096 limit | **100%** | **100%** |
| median preamble tokens surviving | 3,964 (42%) | 2,554 (27%) |
| items keeping ZERO exemplar text | 0 | ~4% |

The model never saw four worked examples. It saw a mid-sentence fragment of
the last one. Root cause: `mine_cot_exemplars.py` selected on
`sum(len(s) for s in steps)` — trace length only — and ignored `context`, so
it picked exemplars carrying 20,144- and 17,936-character contexts.

**2. Left truncation also deleted the persona.** `COT_SYSTEM_PROMPT` is the
first message in the chat template, so it was cut from EVERY few-shot item
while the zero-shot CoT arm kept it. The script header claimed "the ONLY delta
is the visible exemplars"; in fact the arm differed by a missing system prompt,
a malformed chat opening, and fragmentary exemplars.

**3. Token budget.** All three legs ran at `--max-new-tokens 256`, the defect
of `docs/audit_2026_09_14.md` §2. Anchor-missing: llama/TB 2.47%,
mistral/TB 0.30%, llama/TIME 9.27% — those items score ~0%.

**4. llama/timebench additionally predates a harness change.** It ran
2026-09-13 17:58–23:22; `scripts/run_baselines.py` was replaced at 18:53. That
leg carries 21,188 rows (the 113 blank-gold timeqa items included), every leg
after it carries 21,075. The old version is unrecoverable — no git history for
the file, no backup newer than 2026-05-11.

## Fixes applied before any re-run

- `mine_cot_exemplars.py`: budget enforced at selection (`MAX_EXEMPLAR_TOKENS`
  450, `MAX_PREAMBLE_TOKENS` 1,600), measured with the model's own tokenizer
  through the runner's own renderer, plus a hard refuse-to-write check.
- `run_baselines.py::_load_cot_exemplars`: refuses an over-budget file.
- `run_baselines.py`: records `prompt_style` and `max_new_tokens` per row and
  refuses to resume across a change in either (`--allow-budget-change` opts in
  for the salvage workflow).
- `run_fewshot_cot.sh` / `run_zs_cot.sh`: `.done` markers stamped `OK:<budget>`
  so a budget change invalidates its own markers.
