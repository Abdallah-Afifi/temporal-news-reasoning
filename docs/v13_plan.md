# v13 — targeted scale-up: category-targeted synthetic data, free wins, capacity

Status: PRE-REGISTERED 2026-10-03. Not launched. Everything below is the
plan; deviations get recorded here as they happen, in the style of
docs/v12_plan.md. The reference results this plan attacks are
`results/rescored/v12_test_minus_dev.json` (test-minus-dev, v5 protocol).

## 0. Why v13, in one table

v11/v12 beat zero-shot everywhere, but the per-category deltas show two
things: huge untapped buckets, and four categories where fine-tuning still
LOSES to zero-shot.

| Bucket (test-minus-dev) | n | zs -> best ft | What v13 does about it |
|---|---|---|---|
| TIME Computation | 8,926 | 18.1 -> 50.5 | programmatic arithmetic at scale (§2) |
| TIME Timeline | 12,530 | 16.5 -> 22.7 | programmatic + GLM sorting rows (§2, §3) |
| TIME Extract | 3,180 | 9.6 -> 6.1 (LOSS) | FIRST EVER training analogue (§2.4, §3.4) |
| TIME Order_Compare | 9,426 | 60.4 -> 56.7 (LOSS) | programmatic + GLM rows, exact card (§2.5) |
| TimeBench temporal_dialogue | 1,274 | 78.2 -> 61.9 (LOSS) | GLM-5.2 masked-span + transcript wave (§3.2) |
| TimeBench arithmetic | 3,526 | 7.0 -> 22.3 | programmatic month/clock arithmetic (§2.1) |
| TRAM temporal_relation | 202,825 | 26.1 -> 31.2 | programmatic + GLM relation rows, incl. TRAM's event-to-time surface form (§2.3) |
| TRAM storytelling | 66,519 | 76.7 -> 68.9 (LOSS) | corrected-card GLM wave, bigger (§3.3) |
| TRAM arithmetic | 30,850 | 19.9 -> 31.4 | programmatic clock arithmetic (§2.1) |
| TRAM temporal_nli | 558,516 | 53.2 -> 65.6 | GLM nli_saq/nli_mcq wave (§3.1) |

Design rule kept from v11/v12: every new row is rendered through the SAME
`eval_parity` path at training time, completion-only loss, and the frozen v5
scorer decides. Nothing here changes prompts, engine, decoding, or scoring.

## 1. Arms (pre-registered)

All arms train on `data/combined_80_20_v13` (§4) with the t09 recipe
(lr 1.77e-4, LoRA r16/a32, 7 target modules, eff. batch 16, cosine,
warmup 0.0819, wd 0.0109, seed 42, bf16, completion-only loss,
`prompt_format: eval_parity`) except the single variable named:

| Arm | Variable | Config | Why |
|---|---|---|---|
| `soup-v11` | NO training: average the v11 seed adapters (s42/s43/s44) | scripts/lora_soup.py | free; model soups typically +0.3-1pp; also gives a 4th seed-averaged reference point |
| `v13-base` | v13 data, everything else = t09 | config_v13_base.yaml | the headline arm: does targeted scale-up move the pooled number |
| `v13-2ep` | 2 epochs | config_v13_2ep.yaml | v11-2ep won TimeBench/TRAM but starved on 11k rows; with ~3x data the trade-off should flip positive on TIME too |
| `v13-ctx` | max_seq_length 4096 (bs 1 x accum 16) | config_v13_ctx.yaml | 56.8% of TIME prompts exceed 2048; v12-ctx's TimeBench gain was real (+0.84 over v12-data) |
| `v13-r32` | LoRA r=32 (alpha 64) | config_v13_r32.yaml | Notes.md anticipated r=64 "if underfitting"; with 3x data the r=16 capacity ceiling becomes testable cheaply. RUNS LAST, drop if the schedule overruns |

Selection rule (fixed now): the arm that wins is the one with the best mean
delta-vs-zs across the three benchmarks on test-minus-dev; per-benchmark
winners may be reported separately but the headline arm is one model.
McNemar z vs `zs-vllm-pinned` per benchmark, and the v11 3-seed noise floor
(0.4-0.7pp) is the yardstick for "real".

## 2. AUG_PROG — programmatic synthetic rows (exact golds by construction)

New generator: `scripts/generate_v13_prog_data.py`, seeded (42), writes
`data/prog_aug_v13/<Category>_jsonl` rows in the training schema
(`source_dataset: "AUG_PROG"`, question/context/targets/rationale/source).
No LLM anywhere in this path; every gold is COMPUTED before the text is
rendered, then re-verified by an independent checker at the end. Caps per
category are pre-registered below so the mixture cannot drown the real
TimeQA/TLQA rows (which stay at 100%).

Target mixture (rows, train+val before the 80/20 hash split):

| Category (AUG_PROG) | rows | Analog of | Gold |
|---|---|---|---|
| `prog_arith_clock` | 1,500 | TRAM arithmetic ("What is 23:48 - 01:31?") | HH:MM, mod-24h, computed |
| `prog_arith_month` | 1,500 | TimeBench arithmetic ("What is the time 6 year and 4 month after Nov, 1185") | "Mon, YYYY", calendar-exact |
| `prog_computation` | 1,500 | TIME Computation (span + offset, with the verbatim Hint) | "N years M months D days" / date, calendar-exact, mini-context |
| `prog_timeline` | 2,000 | TIME Timeline / AUG_SEQ (exact "Below are N facts" template) | letter sequence, computed from the dates |
| `prog_relation` | 2,000 | TRAM temporal_relation, BOTH surface forms: event-to-time (the dominant TRAM form) and event-to-event incl. intervals | TRAM gold labels only: BEFORE/AFTER/IS_INCLUDED/SIMULTANEOUS/INCLUDES |
| `prog_ordering_tf` | 800 | TRAM ordering TRUE/FALSE | computed from stated dates |
| `prog_ordering_seq` | 800 | TRAM ordering permutation shape | computed |
| `prog_duration_compare` | 800 | TIME Duration_Compare (exact 3-option card wording) | computed from stated dates; ~33/40/27 A/B/C |
| `prog_order_compare` | 800 | TIME Order_Compare (exact 3-option card wording) | computed; ~35/45/20 A/B/C |
| `prog_extract` | 1,500 | TIME Extract (multi-select time expressions, TimeDial-style contexts) | letters joined by TWO spaces, exactly as TIME ("B  C") |
| **total** | **14,200** | | |

Hard constraints the generator enforces (audited at the end, gate fails the
run if violated):

- MCQ gold letters cycle A/B/C/D (no position skew; the L7 lesson).
- Label categories: gold labels balanced (relation roughly equal across the
  five TRAM gold labels, SIMULTANEOUS capped at ~10% since TRAM rarely uses
  it; nli-like TRUE/FALSE ~50/50; ordering permutations never identity >1/3).
- No single distinct gold string > 10% of its category (the D51 lesson).
- Every row's gold is re-derived by an independent verifier function before
  the row is written; any mismatch aborts the generator.
- Dates drawn 1890-2024 for event rows, 1100-2199 for month-arithmetic
  (TimeBench's own range), varied surface forms ("March 3, 2019",
  "3 March 2019", "Mar 3rd, 2019"), digits/number-words mixed where the
  benchmark mixes them.
- Dedup by normalized question hash against AUG_PROG itself AND against
  every row already in data/combined_80_20_v12.

Why short invented contexts (60-90 words) instead of full three-passage news
shapes: AUG_PROG teaches the SKILL (calendar arithmetic, ordering, letter
multi-select), not the retrieval shape; the AUG_GLM2 rows (kept at full
length) already cover the shape. This is the same split of labor as
AUG_GLM (no-context arithmetic) + AUG_GLM2 (passage rows) in v11, which
moved Computation +32pp.

## 3. AUG_GLM3 — GLM-5.2 targeted wave (~4,500 rows)

Generator: `scripts/make_v13_glm_orders.py` (packets -> data/glm_packets_v13/,
raw replies -> data/glm_raw_v13/, ingested -> data/manual_aug_glm_v13/).
RUNS ON GLM-5.2 (`scripts/run_v13_glm.py`, default model `glm-5.2`,
same client, same never-stored-key policy). The ingest gate is the v12
one (`ingest_glm_batch.py`, minimally extended for `extract`).

Allocation (pre-registered; targets the buckets in §0 that need language
quality, not arithmetic):

| Category | rows | Notes |
|---|---|---|
| `relation` | 700 | card EXTENDED: half the rows use TRAM's event-to-time surface form |
| `nli_saq` | 400 | TRAM's biggest bucket; `neutral` never under-produced |
| `nli_mcq` | 250 | |
| `temporal_dialogue` | 400 | the -16pp regression; masked-span + dated transcripts |
| `storytelling` | 400 | corrected everyday-story card (v12-data already moved it -11.3 -> -8.1) |
| `Timeline` | 400 | passages + paraphrased facts, rationale letter-dated |
| `Computation` | 300 | passage-buried dates (37% of stems name NO date, as in TIME) |
| `Relative_Reasoning` | 350 | flat at 46.6 — paraphrase golds, 48% extractability |
| `Duration_Compare` | 300 | |
| `Order_Compare` | 200 | |
| `duration` | 250 | commonsense typical durations (needs a language model) |
| `ordering` | 250 | commonsense TRUE/FALSE half needs language |
| `extract` | 300 | NEW CARD: TimeDial-style dated dialogue, multi-select time expressions, gold "B  C" two-space joined |
| **total** | **4,500** | |

Ingested rows are relabelled `AUG_GLM3` when the v13 mixture is built (the
generator still emits `AUG_GLM2` because the gate's schema check requires
it — recorded deviation, kept for gate compatibility).

## 4. The v13 mixture (`scripts/build_v13_training_data.py`)

- Start from `data/combined_80_20_v12/{train,val}.jsonl` UNCHANGED.
- Add every ingested AUG_GLM3 row (gate-checked again, benchmark-
  contamination filter from build_v11_parity_data applied).
- Add AUG_PROG rows capped at §2 numbers.
- New rows split 80/20 by the SAME stable sha1(question) hash rule as v12,
  so a later rebuild with GLM rows added cannot move AUG_PROG rows between
  splits.
- Manifest + data/training_versions/v13/ folder, as v12 did.
- Provisional build WITHOUT GLM rows is allowed (`--allow-no-glm`) for
  pipeline validation only; the schedule refuses to train arms on a
  provisional build (manifest.glm_rows must be > 0 unless
  V13_ALLOW_NO_GLM=1 is exported on the training shell, recorded).

Expected size: v12 11,804 + ~4,000 GLM (after gate losses ~10-15%) +
~14,200 prog ~= ~29,000 rows train+val (~2.4x v12).

## 5. Soup arm (no training)

`scripts/lora_soup.py` averages the safetensors of
`checkpoints/hpo_v11/t09/final` (s42), `checkpoints/llama_v11_best_s43/final`,
`checkpoints/llama_v11_best_s44/final` -> `checkpoints/llama_v11_soup/final`,
then the standard merge -> vLLM eval -> rescore path. Same adapters, same
recipe, so this is a pure measurement of soup gain. If it wins anywhere it
becomes a v14 candidate (soup of v13 seeds).

## 6. Schedule (`scripts/run_schedule_v13.sh`)

Gate: refuses to run before `logs/sched_v12/schedule_complete.marker`
exists; arms run strictly serially on the single 3090; each arm =
5-step dry-run -> train -> merge_lora -> vLLM eval (time, timebench, tram;
`--system-prompt none --date-string "26 Jul 2024"`) -> merged weights
deleted -> final rescore to `results/rescored/v13_test_minus_dev.json`
(--exclude-ids data/hpo_dev/dev_ids.json --reference zs-vllm-pinned).
Order: soup (cheap) -> v13-base -> v13-2ep -> v13-ctx -> v13-r32 (last,
droppable). Resumable via .done markers, as v12.

GLM-5.2 generation runs BEFORE the schedule, from any shell with
GLM_API_KEY exported (the key is never stored — repo policy):
  1. `venv/bin/python scripts/make_v13_glm_orders.py` (packets + ORDERS)
  2. `venv/bin/python scripts/run_v13_glm.py` (or paste packets into the
     GLM-5.2 chat UI and save replies to data/glm_raw_v13/ — same workflow
     as v12 if the API account rate-limits)
  3. `venv/bin/python scripts/ingest_glm_batch.py data/glm_raw_v13/*.txt
     --out data/manual_aug_glm_v13`
  4. `venv/bin/python scripts/build_v13_training_data.py`

## 7. Explicitly NOT in v13 (parked for v14, with reasons)

- CoT/STaR distillation (the 803 verified traces in data/cot_verified):
  the v5 scorer takes the FIRST line, so visible-CoT outputs would need a
  protocol change; v13 keeps parity. v14 candidate: rationale-conditioned
  SFT with answer-only targets.
- Self-consistency / sampling at eval: violates the frozen greedy protocol.
- Mistral seed replicates: orthogonal, scheduled separately (docs/mistral_plan.md §7).
- Qwen under the protocol: same.
- External calculator tool at inference: protocol change, parked with the
  Fine_Tuning/external tool.md note.
- NEFTune: one-line Trainer flag but it changes the loss surface; if
  v13-r32 shows underfitting is the binding constraint, NEFTune joins a
  v14 HPO instead of being smuggled into v13.

## 8. Success criteria (fixed now)

- Headline: v13-base beats v11-best mean delta (currently +5.27/+1.48/+6.54)
  on at least 2 of 3 benchmarks, outside the 0.4-0.7pp seed noise floor.
- Targeted buckets: TIME Extract and Order_Compare, TimeBench
  temporal_dialogue, TRAM storytelling all move to >= 0 delta vs zs (they
  are the four active losses).
- Stretch: TRAM temporal_relation >= 35 (from 31.2), TIME Timeline >= 30
  (from 22.7), TimeBench arithmetic >= 30 (from 22.3).
- No bucket that v11/v12 won regresses by more than the noise floor.
- Soup: reported even if it loses; it is one merged eval, not an arm that
  consumes training time.

## 9. RECORDED DEVIATIONS (2026-10-05, `docs/audit_2026_10_04.md`)

Recorded before any v13 arm trains, so nothing below is post-hoc.

1. **AUG_GLM3 is not GLM-authored.** The §3 wave was produced by the
   coding agent's Python templates and builders (`scripts/glm_v13_build/`),
   not GLM-5.2 completions. Relabelled in the build as `AUG_TPL3`
   (`provenance: "agent-template"`). AUG_PROG keeps its label
   (`provenance: "programmatic"`). The programmatic math data is kept in
   full by the researcher's decision — it was deliberately non-GLM and is
   needed. §3's premise (LLM-diverse language for the language-heavy
   categories) is therefore NOT met; v13 tests "v12 + agent templates +
   programmatic data", and any conclusion about LLM-written data is out of
   scope.
2. **Quality filters added to the build** (counts in
   `data/training_versions/v13/manifest.json`): storytelling length shortcut
   removed (−263, gold-longer now exactly 50%); relation rows whose label
   contradicts their dates (−16); duration broken-template (−102) and
   unit-only-option rows (−20); prog_timeline under-determined same-year
   rows (−227); prog_duration_compare incoherent event-named rows (−412);
   TimeQA rows the CoT pass found contradicted/unsupported (−207). Final
   build 24,870 train / 6,376 val.
3. **TimeBench is reported with AND without `date_arith`.** AUG_PROG
   reproduces that subset's template verbatim (pilot: 5.3% → 100% on dev),
   and `date_arith` is ~19% of TimeBench test-minus-dev. The §8 TimeBench
   criterion is judged on the without-`date_arith` number; the with-number
   is reported alongside. Same disclosure for TIME Computation (copies TIME's
   Hint text) and TRAM arithmetic.
4. **Control = v11 three-seed mean**, not seed 42 (seed 42 is the HPO winner
   and the best seed on TIME/TRAM — selection bias, audit §5.1). The §8
   headline ("beat v11-best mean delta +5.27/+1.48/+6.54") already uses the
   mean; arm-vs-arm calls use noise sd·√(4/3) vs the mean or sd·√2 vs a
   single run.
5. **Soup arm rebuilt correctly.** The first build averaged LoRA A and B
   separately (wrong: ≈ half-strength plus cross-seed noise). Now an exact
   soup by concatenation (rank 48, alpha 96); the old build is kept as
   `checkpoints/llama_v11_soup/final_naive_AB_mean` and was never evaluated.
6. **Pilot control corrected.** The pilot's confound-free control is
   v12-data restricted to dev: pilot − v12-data = TIME −2.50, TimeBench
   +18.16 (≈ +1.5 without `date_arith`), TRAM +3.05 (dev only, not
   quotable). The pilot's data is preserved at
   `data/training_versions/v13_provisional/` and its config points there.
