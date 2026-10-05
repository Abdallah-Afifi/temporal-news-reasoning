# data/ — taxonomy

| Dir | What it is | Status |
|---|---|---|
| `benchmarks/` | TIME (canonical `TIME_Newest.json`, 104,939 scored), TimeBench (`TimeBench-full-19000.zip`, 21,075 scored; no MenatQA in that archive), TRAM (980,918) | **eval sets — never train on these** |
| `hpo_dev/` | **v11 HPO dev split** carved from the test pools (`dev_ids.json`: TIME 5,000 / TimeBench 2,500 / TRAM 10,000; `MANIFEST.json`). Built by `scripts/make_hpo_dev_split.py`, frozen | selection only; excluded from every quoted number |
| `combined_80_20_split` | v1 training data (original mixture) | frozen |
| `combined_80_20_v2 … v7` | training data per cycle | frozen |
| `combined_80_20_v9`, `_v9_glm` | synthetic-swap mixtures (template-generated / GLM-generated AUG_GLM2) | frozen |
| `combined_80_20_v10_glm` | v10 mixture (5,954 AUG_GLM2 rows) | **superseded before use** — 561 benchmark-contaminated TimeQA rows (audit 2026-09-23 §7) |
| `combined_80_20_v11` | v10 minus contamination and 124 label-space rows (`scripts/build_v11_parity_data.py`, `manifest.json` lists every removal) | v11 |
| `manual_aug_glm/` | the GLM-chat-generated AUG_GLM2 corpus (invented per order, no source corpus) + `AUDIT.md` | source of AUG_GLM2 |
| `glm_packets/`, `glm_raw/` | generation brief (`MASTER_PROMPT.md`), `ORDERS.txt`, raw GLM replies | generation provenance |
| `glm_packets_v12/`, `glm_raw_v12/`, `manual_aug_glm_v12/` | v12 corrected-card brief + `ORDERS_v12.txt` (610 rows), raw replies, and the separate banked corpus (`docs/v12_plan.md`) | v12 inputs; v11's corpus untouched |
| `combined_80_20_v12` | v11 minus old storytelling + the v12 rows (`scripts/build_v12_training_data.py`) | v12 |
| `prog_aug_v13/` | AUG_PROG: 13,200 programmatic rows (clock/month arithmetic, computation, timeline, relation, ordering, duration/order compare, extract) — `scripts/generate_v13_prog_data.py`; `audit.json` | v13 input (math data kept deliberately) |
| `glm_packets_v13/`, `glm_raw_v13/`, `manual_aug_glm_v13/` | v13 wave: packets, replies, ingested rows. **Produced by the coding agent's Python templates (`scripts/glm_v13_build/`), not GLM chat** — relabelled `AUG_TPL3` in the build (`docs/audit_2026_10_04.md` §1.1) | v13 input |
| `combined_80_20_v13` | v12 + AUG_TPL3 + AUG_PROG, quality-filtered (`scripts/build_v13_training_data.py`, filter counts in `manifest.json`) | **current** |
| `training_versions/` | per-version folders: verified data copy + measured manifest + story. Originals remain canonical in `combined_80_20_*` | start here for v1–v7 |
| `corpus/` | CC-News 2023-24 + CNN stories — source corpus for the old template generator | v8/v9 fuel |
| `synthetic/` | generated news items (v8a) | historical |
| `rehearsal/` | dolly / HotpotQA / DROP / CoQA pools used by the builders | builder input |
| `prepared_v4/` | large external pools (cotcoll 575k, slimorca, squad) | mostly unused |
| `cot/` | STaR self-generation pilot (`scripts/generate_cot_data.py`; only 16% verified yield) + the 98-row manual GLM pilot (`pilot_manual_glm53.jsonl`) that proved the GLM-chat approach | historical / reference |
| `cot_packets/` | **CoT v-next generation brief** (protocol: `docs/cot_generation_protocol.md`) — `MASTER_PROMPT.md` + `ORDERS_grounded*.txt` (batches 1–19, 1,900 ids keyed; real TimeQA question+passage+gold from `combined_80_20_v11`, already decontaminated) + `answer_key.json` (id → real fields, re-attached at ingest so GLM never re-transcribes them) | prep for a later v-cycle; banked traces in `cot_verified/` (author per trace: `provenance.json`) |
| `cot_raw/` | raw GLM replies for the CoT packets, one file per batch | generation provenance (gitignored) |
| `cot_verified/` | `grounded.jsonl` / `invented.jsonl` — traces that passed `scripts/ingest_cot_batch.py`'s gate (answer-match + grounding checks) + `_rejected.jsonl` with reasons | verified CoT pool (gitignored; not yet built into a training mixture) |
| `raw/` | original dataset downloads | provenance only |
