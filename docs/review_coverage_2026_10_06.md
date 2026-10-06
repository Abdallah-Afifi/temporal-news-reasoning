# Review coverage — what has and has not been reviewed (2026-10-06)

Companion to [`audit_2026_10_04.md`](audit_2026_10_04.md). Honest scope
statement: **all code that produces a published or planned number has been
reviewed; not all data has.** Two small code areas outside the evaluated
chain were never reviewed (§1.2). Everything still open is either data
(§3) or doc corrections (§4).

Status words: **REVIEWED** = read/checked in full; **PARTIAL** = some of it;
**NOT REVIEWED** = not looked at in this audit cycle.

---

## 1. Code

### 1.1 REVIEWED (audit 2026-10-04 + fixes 2026-10-05; suite 282 passed / 6 skipped)

| Area | Files | Result |
|---|---|---|
| Training, both models | `experiments/finetuning/{LLaMA,Mistral}/{train.py,data_loader.py}`, `experiments/finetuning/shared/eval_parity.py` | Clean: prompt parity, `seed=`, answer-only loss mask, gold-truncation drop, external checkpoint selection identical in both |
| Configs | `config_v11*`, `config_v12_*`, `config_v13_*`, `config_mistral_*` | Clean; pilot config repointed to its preserved data |
| Run scripts | `run_schedule_{v11,v12,v13,mistral}.sh`, `run_mistral_seeds.sh`, `run_v13_prog_pilot.sh` | Fixed: GPU-busy guard, fail-fast markers, build gate |
| Evaluation | `scripts/run_eval_vllm.py`, `scripts/merge_lora.py` | Clean: date pin (LLaMA) / none (Mistral), no double BOS, chat template copied |
| Scoring | `scripts/rescore_v5_protocol.py` (arm registry, reference pairing, McNemar) | Fixed: missing reference = error, `_paired_reference` saved |
| HPO / selection | `scripts/hpo_v11.py`, `scripts/hpo_mistral.py`, `scripts/score_v13_pilot.py` | Fixed pilot scorer (serialisation bug, confound-free control) |
| Soup | `scripts/lora_soup.py` | Fixed (exact concatenation soup, rebuilt) |
| Data pipeline | `scripts/ingest_glm_batch.py`, `scripts/build_v13_training_data.py`, `scripts/generate_v13_prog_data.py`, `scripts/glm_v13_build/*` | Fixed: provenance labels, filters, timeline guard |
| New verification code | `scripts/verify_v13_math.py`, `scripts/review_v13/*` | Written this cycle; parser misses hand-checked |
| Sync / ops | `scripts/push_to_second_pc.sh`, `TRANSFER_README.md` | Fixed (anchored excludes, `--update`, lock-file rebuild) |
| Tests | `tests/` | Run in full; 30+ tests added for the fixed parts |

### 1.2 NOT REVIEWED in this cycle

| Area | Why it matters / doesn't |
|---|---|
| Scorer internals: `src/evaluation/metrics`, answer-extraction rules, TIME `date_equivalence` | Feeds every published number. Reviewed by the earlier audits (`audit_2026_09_12.md`, `_09_14`, `_09_23`) and unchanged since; this cycle only reviewed how it is called. **The one code gap that touches results.** |
| Benchmark loaders `src/data/data_loader.py` (`CANONICAL_FILES`, TRAM CSV parsing) | Also covered by the 09-12/09-23 audits, unchanged since. |
| RAG system: `temporal_rag/`, `src/rag/`, `src/temporal/`, `src/pipeline/` | Not part of any evaluated number (README says so). |
| `src/prompting/` (stubs), `src/training/` (deprecated second trainer) | Not used. |
| Pre-v11 scripts and configs (v1–v10) | Historical; their results are superseded. |
| CoT tooling beyond the gate docs (`build_cot_packets.py`, `cot_evidence.py` logic) | CoT data is not in any training mix yet. |

## 2. Results — REVIEWED in full

30 prediction files (10 arms × 3 benchmarks): no corruption, identical id
sets (pairing valid), dev split genuinely excluded. Every number quoted in
`results_and_methodology.md` §10–12 and `README.md` re-derived from the
rescored JSONs. Mistral 3-seed result added. v12 verdict corrected.

## 3. Data — the part still open

The v13 training mixture is v12 + the v13 wave + AUG_PROG. Each source,
how it was checked, and where it stands:

| Source | Rows | How checked | Status |
|---|---|---|---|
| AUG_PROG (programmatic math, kept by decision) | 13,200 | Independent mechanical re-derivation of every gold (`verify_v13_math.py`), 0 unparsed | **REVIEWED** — 941 removed (relation mislabels 251, extract 51, duration_compare 412, timeline 227) |
| v13 wave, template rows (AUG_TPL3) | 2,877 | Non-math: removed as template output. Math: mechanical | **REVIEWED** — 1,450 non-math removed; math 1,427 checked, 105 unparsed are in the math_text review |
| v13 wave, GLM-written rows (AUG_GLM3) | 1,642 | Math: mechanical. Non-math: reading review (`glm3` slice) | **REVIEWED** — all 1,401 packet rows judged (125 by the author; the rest by the 10-06 AI passes, see `data/v13_verify/review/REVIEWER_LOG.md`); 5 WRONG + 5 AMBIGUOUS |
| v12 base, AUG_GLM2 math categories | ~2,560 | Mechanical | **REVIEWED** where parsable (1,186 pass, 15 removed); unparsable ones are in `math_text` |
| v12 base, AUG_GLM2 language categories | 2,752 | Reading review (`glm2_lang`) | **PARTIAL** — 375 read (packets 00–02; 02 finished by the 10-06 AI pass), ~2% bad |
| v12 base, free-text math + Co_temporality | 1,451 | Reading review (`math_text`) | **PARTIAL** — 250 read (packets 00–01; 01 finished by the 10-06 AI pass), ~6% bad |
| v12 base, GLM-written rows from glm_raw 271–313 (wrongly removed, now restored) | 447 to read | Reading review (`restored`) | **REVIEWED** — all 447 judged on 10-06 (AI pass 2, spot-checked); 6 WRONG + 9 AMBIGUOUS |
| v12 base, template rows from glm_raw 296/297/309 | 138 | Template output, non-math | Removed |
| AUG_SEQ (sort by stated years) | 1,273 | Mechanical | **REVIEWED** — 1,267 pass, 6 in review |
| AUG_GLM (short word problems) + AUG_GLM2 bare relation/ordering | 785 | Reading review (`short`) | **REVIEWED** — all 785 read (packet 01 finished by the 10-06 AI pass), 1.5% bad |
| TimeQA (Wikipedia, external dataset) | 5,642 | 2,200 by the CoT pass (233 wrong golds removed); rest by reading review (`timeqa`) | **REVIEWED** — all 3,442 packet rows judged on 2026-10-06 (AI passes with spot-checks); 204 WRONG + 118 MALFORMED + 61 AMBIGUOUS + 376 NOT_FOUND (NOT_FOUNDs await the full-passage second look, not removals) |
| TLQA (external, list answers, no passage) | 1,068 | Nothing in the row to check against | **NOT REVIEWED — cannot be verified from the data itself** |
| CoT trace bank (not in any training mix) | 1,713+ | Audit sample of 30 + statistics on all | Sampled only |
| The benchmarks themselves (TIME / TimeBench / TRAM) | — | External evaluation data | Not reviewed (out of scope) |

Reading-review progress: **COMPLETE — all 10,278 rows judged, verified, and
second-looked** (2026-10-06). The 601 TimeQA NOT_FOUND rows were settled
against full passages (450 CORRECT / 42 WRONG / 109 AMBIGUOUS), and all 748
non-CORRECT verdicts were blind re-verified by independent reviewers (~79%
agreement on removal candidates); disagreements were reconciled
conservatively (see `data/v13_verify/review/REVIEWER_LOG.md` and
`data/v13_verify/review_verify/`). Final verdict totals: 9,610 CORRECT /
326 WRONG / 217 AMBIGUOUS / 125 MALFORMED. Removal candidates
(WRONG+MALFORMED, all doubly confirmed): **451**; AMBIGUOUS kept in
training: 217. Pass 3 corrected a misapplied convention on math_text
("most recent X after Y" there means the LATEST recorded after-event; 51
verdicts re-audited WRONG -> CORRECT). Remaining before the v13 rebuild:
human sign-off on the 451 removals (and the 217 AMBIGUOUS dispositions),
then write `data/v13_verify/review_removed.json` from the verdicts and
rebuild via `build_v13_training_data.py`. No removals have been applied
yet.

## 4. Docs still to correct

The provenance finding was revised on 2026-10-05: most of the "builder"
data (v13 packets 001–002, 007–032, 036, 040, 041/057/069/083/095, 103–104,
107–109; glm_raw 271–313 except 296/297/309) was **written row by row by
the GLM model** and only formatted by scripts — it is GLM-generated, not
template output. Only the random slot-filling generators (gen_rel,
gen_dialogue, gen_stories, gen_reasoning, build105/106, syn_gen) are
templates. These documents still state the older, too-broad claim and need
correcting: `audit_2026_10_04.md` §1.1–1.2 and §8, `data/manual_aug_glm/PROVENANCE.md`,
`data/manual_aug_glm/AUDIT.md`, `data/README.md`, `docs/v13_plan.md` §9,
`docs/session_state.md`, `docs/V13_HANDOFF.md`, `results_and_methodology.md` §10.1.

## 5. Effect on published results

None of the open items can change a published number. v11, v12 and Mistral
were measured on held-out test data with no contamination; bad training
rows can only have held them back, not inflated them. The open data review
affects only v13, which has not been trained.
