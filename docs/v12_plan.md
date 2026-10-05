# v12 plan — data fixes + two single-variable checks (pre-registered 2026-09-25)

Written before any v12 row was generated or any v12 arm trained. Background:
v11's per-category results (`results_and_methodology.md` §10.3) and the
2026-09-25 follow-up audit (`audit_2026_09_23.md`, last section).

## 1. What was wrong in the data

| Category | v11 Δ vs zero-shot | Cause (measured) |
|---|---|---|
| TRAM storytelling | −11.31 | Card taught "the wrong ending contradicts a stated date" with narrator endings ("The story ends with …"); TRAM is commonsense plausibility over plain story sentences. v11 picks the wrong ending 34.4% vs 23.1%. |
| TimeBench temporal_dialogue | −18.45 | Card only had dated multi-session transcripts; TimeBench's TimeDial is a short A/B exchange with a `<MASK>` span, and 48% of its options use number words. v11 answers "48 hours" where the option reads "forty-eight hours". |
| TimeBench duration | −7.74 | Card only had world-knowledge spans ("ran from 1891 to 1916"); DurationQA/McTaco ask everyday typical durations with a context sentence. |
| TRAM relation / ordering | (fixed in v11 by deletion) | The **ingest gate itself** certified the wrong label spaces ({IDENTITY, BEFORE, DURING}, {TRUE, Undetermined, FALSE}). TRAM golds are {BEFORE, AFTER, IS_INCLUDED, SIMULTANEOUS, INCLUDES} and {TRUE, FALSE}, and ordering has a second, numbered-sequence shape the card never covered. |

## 2. What changed (code, 2026-09-25)

- `scripts/ingest_glm_batch.py`: TRAM gold label spaces; relation options
  restricted to TRAM's vocabulary; both ordering shapes validated (sequence
  options must be permutations); narrator-style storytelling endings rejected;
  masked dialogue may have an empty context; duration may carry a context
  sentence. 12 new tests (`tests/test_glm_ingest.py`).
- `scripts/make_glm_packets.py`: rewritten cards for relation, ordering,
  storytelling, temporal_dialogue (+ masked shape), duration (+ commonsense
  shape); brief rules for `shape=story` / `shape=masked`.
- Label FREQUENCIES are not matched to any benchmark's test distribution —
  only label spaces and formats, which are task definitions.

## 3. Generation

    venv/bin/python scripts/make_glm_packets.py --master --out data/glm_packets_v12
    venv/bin/python scripts/make_v12_orders.py      # -> data/glm_packets_v12/ORDERS_v12.txt

61 orders / 610 rows: storytelling 150 (replaces all 150 old rows), relation
120, ordering 100, temporal_dialogue 120 (masked), duration 120 (commonsense).
Save replies to `data/glm_raw_v12/`, ingest with
`scripts/ingest_glm_batch.py --out data/manual_aug_glm_v12 data/glm_raw_v12/*.txt`
— a separate corpus, so v11's `data/manual_aug_glm/` is never modified.

## 4. Build

`scripts/build_v12_training_data.py`: v11's mixture unchanged except the old
AUG_GLM2 storytelling rows are dropped; the new rows are re-gated, passed
through v11's contamination filter, split 80/20 by stable hash. Writes
`data/combined_80_20_v12/` and `data/training_versions/v12/`.

## 5. Arms — one variable each, t09's exact recipe otherwise

| Arm | Config | Changes vs its control | Control |
|---|---|---|---|
| v11-2ep | `config_v11_2ep.yaml` | 2 epochs instead of 1 | v11-best |
| v12-data | `config_v12_data.yaml` | v12 data | v11-best |
| v12-ctx | `config_v12_ctx.yaml` | max_seq_length 4096 (batch 1 × accum 16, same effective batch) | v12-data |

Hyperparameters are NOT re-searched: re-searching on the new data would add a
second variable to v12-data.

## 6. Evaluation and decision rule

`scripts/run_schedule_v12.sh` (after v11's schedule has finished): each arm on
the full benchmarks, then `rescore_v5_protocol.py --exclude-ids
data/hpo_dev/dev_ids.json --reference zs-vllm-pinned` →
`results/rescored/v12_test_minus_dev.json`. Test-minus-dev only.

An arm "helps" if its micro accuracy beats its control's on a benchmark by
more than the seed-to-seed spread measured by v11's seeds 42/43/44 (the
noise floor); a smaller difference is reported as "within seed noise", not
as a gain. The targeted categories (§1) are reported separately, and so is
every category that moves the other way.

## Amendments

2026-09-26, before any v12 arm was trained (data facts, not rule changes):
- All 610 rows generated, 0 gate rejects, 0 contaminated;
  `data/combined_80_20_v12`: 11,804 train / 2,971 val (150 old storytelling
  rows dropped, 610 added).
- Storytelling tone: the card's tone rule was added after batch 1. Across all
  150 rows the wrong ending is the more negative one in 47 rows and the right
  ending in 27 (crude negative-word count; 76 neutral) — reduced from batch 1's
  75/25 split but not fully balanced. Recorded as a known residual.
- Storytelling length: the gate's 60-word context minimum overrides the card's
  "40-60 words"; stories average 72 words (TRAM's are shorter).
