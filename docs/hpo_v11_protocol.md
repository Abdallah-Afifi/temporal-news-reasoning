# v11 hyperparameter search — pre-registered protocol

Written 2026-09-23, **before any v11 trial was run**. Nothing below may be
changed after the first trial result exists; any deviation must be added as a
dated amendment at the bottom, with its reason. Background:
[`audit_2026_09_23.md`](audit_2026_09_23.md).

## 1. The question

Does LoRA fine-tuning of LLaMA-3.2-3B-Instruct beat the zero-shot base model
on TIME, TimeBench and TRAM **when the only difference between the two is the
weights** — same prompt, same template date, same engine, same scorer, same
items — and with hyperparameters chosen without looking at the test items?

## 2. What is held identical between zero-shot and fine-tuned

Prompt `run_baselines._build_zero_shot_prompt`, no system message, chat
template date pinned to "26 Jul 2024", vLLM fp16, greedy, 128 new tokens,
4,096-token left truncation, the frozen v5 scorer (`rescore()`), the same item
ids. The fine-tuned model is trained on that exact prompt
(`prompt_format: eval_parity`).

## 3. Data

`data/combined_80_20_v11` (`scripts/build_v11_parity_data.py`): v10-glm's
mixture minus 561 benchmark-contaminated TimeQA rows and 124 relation/ordering
rows outside TRAM's gold label space. 11,446 train / 2,869 val rows before
parity tokenisation, which drops a further 306 TimeQA rows whose gold would be
truncated out of the context.

## 4. The dev split (selection happens here and only here)

None of the three benchmarks has a dev split, so one is carved out of the test
pools **once**: `scripts/make_hpo_dev_split.py` →
`data/hpo_dev/dev_ids.json` (+ `MANIFEST.json` with per-stratum counts and a
sha256 per list). Stratified by the scorer's category label (plus TIME's
retrieval `Setting`), proportional allocation, deterministic (stratum-keyed
seed over sorted ids; `--check` reproduces it exactly).

| | pool | dev | test-minus-dev (reported) |
|---|---|---|---|
| TIME | 104,939 | 5,000 | 99,939 |
| TimeBench | 21,075 | 2,500 | 18,575 |
| TRAM | 980,918 | 10,000 | 970,918 |

Calibration before any trial: v9-glm's existing predictions give dev deltas
of −1.06 / +1.12 / −2.98pp vs its full-test −0.60 / +0.03 / −2.48pp — all
within about one paired SE (0.69 / 1.13 / 0.51pp). The objective's SE is ≈0.5pp.

## 5. Objective (primary, used for selection)

    objective = mean over {TIME, TimeBench, TRAM} of
                (trial v5 micro accuracy − zero-shot v5 micro accuracy) on dev

Zero-shot = the base model run by the same script on the same dev ids.
Micro stays primary for continuity with every published number. Recorded but
**not** used for selection: macro-over-categories deltas (TRAM's micro is
NLI-weighted, audit §9), no-abstain deltas (TIME abstain artefact), TIME by
retrieval setting, McNemar b/c per benchmark.

## 6. Search space and plan (fixed in `results/hpo_v11/plan.json`)

Written by `scripts/hpo_v11.py plan` (seed 20260923) before any trial ran;
no adaptive sampling:

- **t00 anchor**: v10's hyperparameters under the parity prompt (lr 4.62e-4,
  3 epochs, r 32) — answers "does the prompt fix alone change the sign?"
- **t01–t10**: learning rate log-uniform [1e-5, 5e-4]; epochs {1, 2}; LoRA r
  {8, 16, 32} with alpha = 2r; `aug_fraction` {0, 0.5, 1} = share of the
  AUG_GLM2 slice kept (the data hyperparameter; 0 is the real-data-only
  control).
- Fixed: warmup 0.082, weight decay 0.011, dropout 0.05, cosine, batch 2 × 8,
  max_seq_length 2048, seed 42. The last step of training is the model — no
  val-loss checkpoint selection.

Budget ≈ 35–45 GPU-hours on one RTX 3090.

## 7. Selection rule

The trial with the highest objective. If the top two are within 2 SE on dev,
`report` says so, and the tie is reported as unresolved rather than claimed —
but the argmax is still the one taken forward, so the rule stays mechanical.

## 8. Final evaluation (the only numbers that may be quoted)

1. The best trial's own adapter (seed 42) on the full test pools.
2. The same configuration retrained with seeds 43 and 44 (variance).
3. `zs-vllm-pinned`: zero-shot, same engine and date, full test pools.
4. `rescore_v5_protocol.py --tram-root results/tram_fixed --exclude-ids
   data/hpo_dev/dev_ids.json --reference zs-vllm-pinned
   --out results/rescored/v11_test_minus_dev.json`.

Every final number is on **test-minus-dev**; the dev items are removed from
every arm alike, zero-shot included.

## 9. What counts as "beats zero-shot" (decided now)

Per benchmark, v11-best beats zero-shot if its test-minus-dev micro accuracy
is higher **and** McNemar vs `zs-vllm-pinned` gives z > 1.96, **and** the
mean over the three seeds is also above zero-shot. TRAM significance carries
the NLI-duplication caveat (audit §9). A benchmark where these do not all
hold is reported as "no improvement" regardless of the point estimate. The
macro and no-abstain views are reported beside the headline; a result that
holds on micro but reverses on macro is reported as such.

## 10. Honest limits stated in advance

- The dev split comes from the benchmark distribution, so the chosen
  hyperparameters are tuned to these benchmarks' *distribution* (not their
  test items). That is standard dev-set practice and is what makes the
  comparison fair to zero-shot, which needs no tuning.
- 11 trials is a small search; the result is "best of a pre-registered
  plan", not a global optimum.
- The answer-extraction rules were partly developed on earlier fine-tuned
  arms' test outputs (audit §11); they are frozen and applied identically.

## Amendments

(none to the protocol itself)

Implementation note, 2026-09-25: the schedule's rescore reports each seed as
a separate arm and did not compute §9's three-seed mean. That is now done by
`scripts/summarize_v11.py` (run after `run_schedule_v11.sh` completes), which
applies §9 exactly as written above. No threshold, metric or arm changed.
