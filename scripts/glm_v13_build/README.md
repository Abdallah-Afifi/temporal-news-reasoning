# GLM v13 packet build scripts

Everything used to generate the v13 data-collection replies, moved out of
/tmp so the build is reproducible. The finished artifacts are:

- `data/glm_raw_v13/*.txt`   - all 109 GLM reply files (passages + JSONL rows)
- `data/manual_aug_glm_v13/` - ingested output (47 batch jsonl files + ledger)
- final status: 4519/4500 rows, all 13 categories at/above target

## Generators (this sitting)

| script | packets | category | notes |
|---|---|---|---|
| `gen_reasoning.py` | 042-102 (58) | Timeline, Computation, Relative_Reasoning, Duration_Compare, Order_Compare | seeded event pools per domain/era; verifies every row with the ingest gate's own `check()` (context injected from the passage map, mirroring the parser) |
| `gen_dialogue.py` | 033-035 | temporal_dialogue | arithmetic A/B dialogues, gold position shuffled (L7) |
| `gen_stories.py` | 037-039 | storytelling | 15-template scenario bank, slot-filled endings, auto-padded contexts |
| `gen_rel.py` | 003-006 | relation | per-packet era/entity pools, MCQ gold rotation |

Re-run with the project venv from the repo root, e.g.:
    venv/bin/python scripts/glm_v13_build/gen_reasoning.py 042
(packet ids as args; no args = all remaining packets). Output files are
overwritten deterministically (per-packet seeds); re-ingest afterwards with
`scripts/ingest_glm_batch.py <file> --out data/manual_aug_glm_v13
--plan data/glm_packets_v13/_plan.json`.

## Earlier sitting scripts

`build105.py`/`build106.py` (ordering), `build_dc*.py`/`dc2_lib.py`
(Duration_Compare), `build_er*.py`/`er_*.py` (extract), `build_st1.py`/`st_lib.py`
(storytelling), `build_td*.py`/`td_lib.py` (temporal_dialogue),
`build_nm*.py`/`nli_m_lib.py` (nli_mcq), `build_lf_oc.py`/`lf_oc_lib.py`,
`build_oc2.py`, `syn_gen.py`, `t1.py`/`t2.py`, `blk.py`, `validate_v12.py` -
iterative builders and validators from the hand-writing phase.

## parts/

Intermediate passage/row fragments (`NNN_a.txt` passages, `NNN_b.txt` rows)
that were concatenated into `data/glm_raw_v13/NNN_category.txt`. Kept for
provenance; the concatenated files are authoritative.

## Gate lessons baked into the generators

- MCQ gold-position skew (L7): reject any batch where one letter holds >60%
  of golds; shuffle options / rotate endings.
- Computation golds are recomputed (relativedelta must match exactly);
  Timeline needs `A=YYYY-MM-DD` keys covering the gold sequence;
  Duration_Compare rationales state both spans as "Month D, YYYY" (near-tie
  = within 10% or 60 days); Order_Compare rationale's first two dates must
  be Fact1 then Fact2 (gap >14 days for "same time" golds is an error).
- Context floor of 60 words; `source_dataset` must be present on every row;
  DANGLE check rejects golds ending in a preposition/article.
