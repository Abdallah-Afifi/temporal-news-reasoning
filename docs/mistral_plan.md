# Mistral plan — porting the v11 audit fixes to Mistral-7B-Instruct-v0.3 (2026-09-28)

Everything the LLaMA campaign fixed on 2026-09-23 (`audit_2026_09_23.md`) had
never been ported to `experiments/finetuning/Mistral/`. Mistral's own
fine-tuning pipeline was still in the pre-audit state LLaMA was in before
v11: trained on a different prompt than it was evaluated on, no seed fix, no
dev-selected hyperparameters. This doc records what was found, what changed,
and how the first properly-fixed Mistral arm is run and scored.

## 1. What was missing (found 2026-09-27/28)

| Audit item | LLaMA fix (v11) | Mistral, before this pass |
|---|---|---|
| §1 train/eval prompt parity | `prompt_format: eval_parity` | absent -- `data_loader.py` had no parity path at all |
| §2 chat-template date confound | `date_string` pinned to "26 Jul 2024" | **not applicable** -- Mistral-7B-Instruct-v0.3's chat template has no date field (verified 2026-09-28 by dumping the template); the pin is a no-op for this model, not a missing fix |
| §4 seed never reached the Trainer | `seed=` added to `TrainingArguments` | absent -- confirmed by grep, `set_seed()` at the top only reached LoRA init, exactly the LLaMA defect |
| §5 checkpoint selection on in-distribution eval_loss | selection moved OUTSIDE the Trainer entirely (`eval_strategy`/`save_strategy`: "no", `load_best_model_at_end: false`); the whole config becomes one HPO-searched point | config.yaml still had `load_best_model_at_end: true`, `metric_for_best_model: eval_loss` |
| §7 contamination / label-space fixes | `data/combined_80_20_v11` / `v12` | **N/A** -- these fixes are at the row-text level (word-token shingle matching, not a model tokenizer), so they apply to any model. Mistral reuses `data/combined_80_20_v12` directly, no rebuild needed. |

## 2. Code changes

- `experiments/finetuning/Mistral/data_loader.py`: added `tokenize_example_parity`
  (ported verbatim from `experiments/finetuning/LLaMA/data_loader.py` --
  `experiments/finetuning/shared/eval_parity.py` is model-agnostic, it only
  calls `tokenizer.apply_chat_template` on the shared zero-shot builder's
  output) and threaded `prompt_format` through `normalize_and_tokenize` /
  `load_flat_datasets`. `normalize_record`'s three branches now also return
  `answer_parts` (needed by the parity function's gold-truncation check).
- `experiments/finetuning/Mistral/train.py`: added `seed=int(config.get("seed", 42))`
  to `TrainingArguments`; `prompt_format` read from config and passed to
  `load_flat_datasets`.
- `experiments/finetuning/Mistral/config_mistral_parity.yaml` (new): the v11
  fixes applied -- `prompt_format: eval_parity`, `data/combined_80_20_v12`,
  `eval_strategy`/`save_strategy: "no"`, `load_best_model_at_end: false`,
  `load_in_4bit: true` (7B is more than double LLaMA-3B's footprint; keeps
  this inside the 24GB budget the old config.yaml documented).
- `tests/test_eval_parity_mistral.py` (new, 5 tests): the same load-bearing
  checks as `tests/test_eval_parity.py`, adjusted for Mistral's tokenizer --
  `</s>` not `<|eot_id|>`, and a test that `date_string` has no effect
  (documents the no-date-field fact rather than asserting a pin). All pass
  against the real local Mistral tokenizer (CPU only). A dry run of
  `tokenize_example_parity` over 800 real `combined_80_20_v12` train rows:
  776 ok, 24 `gold_truncated` (3.0%) -- in line with LLaMA's v11 drop rate.
- `scripts/rescore_v5_protocol.py`: registered `zs-vllm-mistral` and
  `mistral-best` in the `time`/`timebench` ARMS lists; generalised the
  `--tram-root` redirect map's tuples from `(dir, sub)` to
  `(dir, sub, model_key="llama")` (it hardcoded `llama` in the path
  template, which every arm until now happened to be) and added the two
  Mistral TRAM entries; added `results/hpo_mistral/` to `_NOT_ARMS`.

## 3. The D46/D53 "mistral-vLLM banned" decision — RESOLVED, not overridden

`rescore_v5_protocol.py` carried a standing note that Mistral was "banned
from vLLM for a 2.20pp parity failure (D46/D53)" and scored Mistral only on
the HF engine as a side experiment. Investigated before reusing vLLM here,
since the whole point of this plan is an engine-consistent (vLLM throughout)
arm, same as LLaMA's v11.

**What the original report actually measured**: `logs/vllm_parity_chain.sh`
(2026-09-03) ran the SAME comparison for both models, but the Mistral
`--reference` path was `results/finetuned/mistral/time/finetuned/predictions.jsonl`
-- the FINE-TUNED model's HF predictions -- not a zero-shot HF baseline. The
LLaMA line, by contrast, correctly referenced
`results/baseline/zero_shot_v2/llama/.../zero_shot/predictions.jsonl`. So the
"parity failure" compared vLLM's BASE-model zero-shot output against a
DIFFERENT (LoRA-tuned) model's answers and called the mismatch an engine bug.

This matches what `docs/session_handoff.md`'s D57 entry (2026-09-09) already
suspected: tokenizer-level causes (double-BOS, chat-template drift across
`transformers` versions) were eliminated on CPU, byte-identical rendering
confirmed across both venvs, and the leading theory became "the test ran a
pre-fix encode path" -- `run_eval_vllm.py`'s mtime was 2h45 AFTER the failing
report. D57 called for a ~20-minute GPU re-test with corrected code; it was
never run, so the ban stood on the books for 19 days across the v9/v10/v11/v12
cycles despite already being the leading suspect.

**Re-checked here, CPU-only, no GPU needed** (2026-09-28): rebuilt the
comparison using the CORRECT zero-shot reference
(`results/baseline/zero_shot_v3/mistral/time/zero_shot/predictions.jsonl`,
104,950 usable rows) against the same stored vLLM predictions
(`results/parity_vllm/mistral_time_parity.jsonl`, 2,000 items, 1,971
overlapping ids):

| | agreement | accuracy delta |
|---|---|---|
| original report (wrong reference) | 55.15% | -2.70pp |
| corrected reference | **95.18%** | **-0.051pp** |

D57's own resolution criterion was "if agreement lands near llama's 94%, the
ban was an artifact of stale code and mistral-vLLM reopens." 95.18% clears
that bar, and -0.051pp is smaller than LLaMA's own HF/vLLM gap (audit
§6, 0.05pp on TIME). **The ban is lifted**: `zs-vllm-mistral` and
`mistral-best` are first-class vLLM arms, scored by the same frozen v5
protocol as everything else.

## 4. Hyperparameters — HPO-lite, not a full search

`scripts/hpo_mistral.py`, ported from `scripts/hpo_v11.py`: same dev split
(`data/hpo_dev/dev_ids.json` -- carved from benchmark test pools by item id,
not tied to any model), same objective (mean Δv5-accuracy vs zero-shot across
TIME/TimeBench/TRAM on dev), same no-adaptive-sampling discipline (a fixed
plan written before any trial runs).

**Reduced scope, documented rather than hidden**: only
`(learning_rate, num_train_epochs, lora_r)` are searched -- v11's
`aug_fraction` data-mixture knob is not re-opened, since v11/v12 already
answered that question for this data and Mistral reuses
`data/combined_80_20_v12` whole. Default N = 5 sampled trials + 1 anchor (v11
used 10 + 1); Mistral-7B trains markedly slower per step than LLaMA-3B, and
each trial's dev-only evaluation (not full test) keeps the per-trial cost
down, but a full v11-sized budget would still cost several days on one GPU.
The anchor is LLaMA's own HPO-winning trial (t09: lr 1.77e-4, 1 epoch,
lora_r 16), transplanted -- the best available prior evidence for what this
exact data+prompt combination rewards, analogous to how v11's anchor reused
v10-glm's hyperparameters under the new prompt.

## 5. Schedule

`scripts/run_schedule_mistral.sh`:
1. Smoke test (5 steps, `config_mistral_parity.yaml`) -- this whole pipeline
   has never touched a GPU; fail cheap before HPO commits real hours to it.
2. `hpo_mistral.py plan` / `run` / `report` -- 6 trials, dev split only.
3. Zero-shot Mistral through vLLM, full test sets, no adapter
   (`zs-vllm-mistral`).
4. The winning trial's own adapter (seed 42 IS the winning trial, same trick
   `hpo_v11.py final-config` uses -- no retrain), merged, evaluated on the
   full test sets (`mistral-best`).
5. `rescore_v5_protocol.py --tram-root results/tram_fixed --exclude-ids
   data/hpo_dev/dev_ids.json --reference zs-vllm-mistral --out
   results/rescored/mistral_test_minus_dev.json`.

**This Claude Code session has no GPU access** (re-confirmed 2026-09-28:
`/dev/nvidia0` exists but `torch.cuda.init()` raises "Found no NVIDIA driver
on your system"; same boundary `docs/audit_2026_09_12.md` D33 already
documented, including the 2026-09-05 incident where a job launched from
inside this sandbox silently fell back to CPU and thrashed). Launch from a
shell with real GPU access:

```
mkdir -p logs/sched_mistral
nohup bash scripts/run_schedule_mistral.sh > logs/sched_mistral/driver.log 2>&1 &
```

Resumable: re-running skips every stage already marked OK in
`logs/sched_mistral/`.

## 6. What "beats zero-shot" will mean here

Same standard as v11 (`hpo_v11_protocol.md` §9): `mistral-best`'s test-minus-dev
accuracy must beat `zs-vllm-mistral` on each benchmark, with McNemar
significance. There is no seed-replicate budget in this reduced plan (v11 used
seeds 42/43/44 to measure a noise floor before trusting a single-digit delta;
that cost ~3x a single arm's training+eval time and is not included here), so
a Mistral result should be read as a first measurement, not a noise-floor-
checked one, unless replicated later the same way v11 was.

## 7. RESULT (2026-10-02)

Schedule finished end-to-end, unattended, ~4.5 days total (launched
2026-09-28 17:57, `schedule_complete.marker` 2026-10-02 06:24) -- mostly
`zs-vllm-mistral`'s full-test-set run (Mistral-7B generates markedly slower
than LLaMA-3B: TIME alone took ~9h41 vs LLaMA's ~3h33 for the same benchmark).

HPO-lite leaderboard (`results/hpo_mistral/best.json`, dev split, objective =
mean Δv5-accuracy over TIME/TimeBench/TRAM vs zero-shot):

| id | lr | epochs | lora_r | objective | TIME Δ | TimeBench Δ | TRAM Δ |
|---|---|---|---|---|---|---|---|
| **m05 (winner)** | 3.72e-05 | 1 | 16 | **+7.84** | +7.82 | +10.88 | +4.81 |
| m02 | 2.18e-04 | 1 | 8 | +7.78 | +9.84 | +8.36 | +5.13 |
| m00 (anchor, LLaMA's t09 lr) | 1.77e-04 | 1 | 16 | +5.40 | +8.10 | +4.68 | +3.41 |
| m01 | 1.47e-04 | 2 | 32 | +5.29 | +6.42 | +6.08 | +3.36 |
| m03 | 4.07e-04 | 2 | 16 | -5.05 | -0.08 | -9.24 | -5.82 |
| m04 | 3.57e-04 | 1 | 32 | -5.26 | -4.32 | -2.96 | -8.50 |

m05 and m02 are within 2 SE of each other -- the dev split does not resolve
which is truly better, a difference from v11 where the winning margin was
clearer. The two high-LR trials (m03, m04) failed outright (worse than
zero-shot on most benchmarks): learning rate matters far more than epoch
count or LoRA rank for this model, and the LLaMA-transplanted anchor (m00)
was not the best choice, though it still beat zero-shot comfortably.
m05's lr (3.72e-05) is roughly 5x lower than LLaMA's winning rate (1.77e-4) --
a different model wanting a different learning rate was exactly the reason
§4 pre-registered a search instead of reusing t09's numbers outright.

**Final test-minus-dev result** (`results/rescored/mistral_test_minus_dev.json`,
`--reference zs-vllm-mistral`):

| Benchmark | zero-shot | mistral-best | Δ | McNemar z |
|---|---|---|---|---|
| TIME | 37.25% | 45.34% | +8.09 | +52.07 |
| TimeBench | 45.40% | 55.84% | +10.44 | +28.72 |
| TRAM | 52.13% | 56.55% | +4.42 | +96.32 |

Beats zero-shot on all three benchmarks, every delta far past McNemar
significance -- the same verdict pattern v11 reached for LLaMA. No seed
replicates (§6), so read this as a first measurement, not a noise-floor-
checked one.

**Not a LLaMA-vs-Mistral comparison**: Mistral's own zero-shot is lower than
LLaMA's zero-shot on TIME (37.25% vs 41.15%) and TRAM (52.13% vs 46.55%
-- here Mistral's zero-shot is actually HIGHER) and TimeBench (45.40% vs
44.83%, also higher) -- the two baselines differ enough, and the models
differ enough in every other respect (size, pretraining, instruction
tuning), that nothing here licenses ranking the two base models against
each other. The claim this section supports is narrower and model-internal:
fine-tuning under this protocol helps Mistral too, by a comparable
order of magnitude to LLaMA's v11 gains.

## 8. Seed replicates — RESULT (2026-10-05)

Seeds 43/44 of the exact m05 recipe (`config_mistral_best_s4{3,4}.yaml`,
seed the only variable; `scripts/run_mistral_seeds.sh`). Test-minus-dev,
vs `zs-vllm-mistral`:

| Benchmark | zs | s42 | s43 | s44 | mean ± sd | Δ mean |
|---|---|---|---|---|---|---|
| TIME | 37.25 | 45.34 | 45.88 | 46.11 | 45.78 ± 0.40 | +8.53 |
| TimeBench | 45.40 | 55.84 | 55.40 | 56.19 | 55.81 ± 0.40 | +10.41 |
| TRAM | 52.13 | 56.55 | 57.14 | 56.52 | 56.73 ± 0.35 | +4.61 |

Every seed beats zero-shot on every benchmark (McNemar z ≥ +28), so the
§6 caveat ("no seed-replicate budget") is closed. Seed sd 0.35–0.40pp is
Mistral's own noise floor. Caveat added by `audit_2026_10_04.md` §4.4:
adapters are trained against the 4-bit base and evaluated merged into the
fp16 base, consistently for every arm. Full write-up:
`results_and_methodology.md` §12.2a.
