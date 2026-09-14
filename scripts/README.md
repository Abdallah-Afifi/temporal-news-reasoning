# Scripts

Run everything from the repository root, with the project venv:
`./venv/bin/python scripts/<name>.py` (vLLM legs use `./venv_vllm/bin/python`).

> Re-written 2026-09-13, reverted by the two-PC merge, restored 2026-09-14.
> The previous version listed five scripts and advertised
> `run_full_finetuned_eval.sh`, which is **retired** for running off-protocol
> and writing into the v1 arm's directory.

## The evaluated chain

| script | what it does |
|---|---|
| `run_baselines.py` | HF evaluation runner (TIME/TimeBench for arms v1–v6, and the CoT arms). Greedy, length-sorted batches under a token budget. Stores raw predictions, tagged with engine/model_dir/adapter_dir, and refuses to resume into another arm's file. |
| `run_eval_vllm.py` | vLLM runner — same prompts, same postprocessing, same schema. Requires a **merged** checkpoint, refuses a base-model/adapter mismatch, refuses a cross-arm resume, and holds an exclusive write lock. |
| `merge_lora.py` | Merge a LoRA adapter into the base for vLLM (~4 min CPU, ~6 GB, regenerable). Requires shards, not just a config, before calling a merge done. |
| `rescore_v5_protocol.py` | **The scorer.** Recomputes every published number on CPU from the immutable `predictions.jsonl` files. Also emits the abstain-free column, the retrieval-`Setting` breakdown, strict/rule-credit diagnostics, macro averages, `anchor_missing_pct` for CoT arms, and paired McNemar. `--tram-root` scores the 2026-09-13 re-run. |
| `mcnemar_v5_protocol.py` | Paired significance between two arms (time / timebench / tram). |
| `build_v*_training_data.py` | One builder per arm. Hard asserts: 0 letter-probe leakage, 0 benchmark collisions, dedup, near-duplicate purge. **New arm ⇒ new builder**; earlier ones are never edited. |

## Schedules (launch from a shell with real GPU access)

| script | status |
|---|---|
| `run_schedule_tram_rerun.sh` | TRAM re-run on corrected prompts — **COMPLETE** 2026-09-13, all 10 arms. |
| `run_schedule_v7_corrected.sh`, `run_schedule_v7.sh`, `run_schedule_v6d.sh` | per-arm campaign schedules, complete |
| `run_queue_v4/v5/v6.sh`, `run_queue_llama.sh`, `run_baseline_v3.sh` | earlier queues, complete. Completion markers are now conditional on every leg reporting OK. |

Each schedule is idempotent: per-stage `logs/<sched>/<stage>.done` files holding
`OK`/`FAIL` are the source of truth, and a stage marked `OK` is skipped.

## Chain-of-Thought prompting arms (base models, no training)

| script | status |
|---|---|
| `run_zs_cot.sh` | zero-shot CoT, llama + mistral × TIME + TimeBench. **Run, but at the old 256-token budget — needs re-running at 768**: 22.1% of llama's TIME outputs were truncated before the `ANSWER:` line and score ~0 (`docs/audit_2026_09_14.md` §2). |
| `run_fewshot_cot.sh` | few-shot CoT. Correct, gated on the zero-shot marker; **not yet run**. Uses `data/cot_fewshot_exemplars_mined.json`. |
| `mine_cot_exemplars.py` | mines exemplars from the **v6 training pool**, never from benchmarks (verified: 0 benchmark collisions). Picks the shortest verified-correct trace per kind. |
| `rename_zero_shot_reasoning.sh` | one-shot rename `zero_shot_cot` → `zero_shot_reasoning`; refuses while any driver is alive. Not yet run. |
| ~~`run_zs_tram_llama.sh`~~ | **RETIRED 2026-09-14.** Ran the HF engine into the vLLM TRAM baseline directory (audit H4) and targeted stale data. Refuses to run. |

## v9 synthetic arm (ruleset `docs/synthetic_data_ruleset.md`)

`generate_v9_aug.py` builds the AUG_GLM2 dataset (6,000-row budget) from
`data/corpus/ccnews` via the `v9/` package — deterministic, mechanically
verified golds, three context shapes (news/wiki/dial), §5.12 answerability
pairs. `audit_v9_aug.py` runs every §7 gate per category and writes
`data/manual_aug_v9/AUDIT.md` (a slice with no recorded audit does not go
into a trainer). `build_v9_training_data.py` swaps the audited rows into
v6's mixture at the L1 cap of 15,000 rows -> `data/combined_80_20_v9/`.

## Analysis and audit

`probe_answer_formats.py`, `audit_output_style.py`, `audit_tram.py`,
`audit_train_benchmark_overlap.py`, `eval_letter_probe.py`, `compare_parity.py`,
`probe_perturbation_consistency.py` (a GPU driver, not a unit test).

## Setup

`download_models.py`, `download_datasets.py`, `install_heideltime.sh`,
`verify_install.py`.

## `scripts/retired/`

Dead or dangerous tools, kept for history. **Do not run them.** Includes
`run_full_finetuned_eval.sh` (off-protocol batch size and token budget, writes
into the v1 arm directory), `run_zeroshot_tram.sh` (would resume a vLLM
baseline file with HF predictions), `rescore_all.py`, `run_queue_v7.sh`,
`chain_after_v6.sh`, `seed_rerun_from_unaffected.py`.
