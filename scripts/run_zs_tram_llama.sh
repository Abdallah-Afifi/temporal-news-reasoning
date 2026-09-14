#!/bin/bash
# ===========================================================================
# RETIRED 2026-09-14 — DO NOT RUN. Superseded, and it reintroduced a hazard
# that had already been retired once.
# ===========================================================================
# What it did: ran `run_baselines.py --benchmark tram` (the **HF** engine)
# into `results/baseline/zero_shot_v3/llama/tram/zero_shot/`, which is the
# **vLLM** TRAM baseline directory that every TRAM arm is compared against.
#
# That is defect H4 from the 2026-09-09 audit verbatim — it is exactly why
# `scripts/retired/run_zeroshot_tram.sh` was retired on that date. Writing one
# results directory with two engines, with resume keyed on `id`, silently
# mixes engines inside the baseline.
#
# It is also REDUNDANT and points at STALE data. The zero-shot TRAM baseline
# was re-run on corrected prompts on 2026-09-13 and lives at
#     results/tram_fixed/zero_shot/llama/tram/zero_shot/predictions.jsonl
# (vLLM, 46.95%). The path this script targeted holds the superseded
# broken-prompt predictions (35.20%) — see docs/audit_2026_09_12.md §0a and
# session_handoff D65.
#
# If a zero-shot TRAM run is ever needed again, use
# `scripts/run_eval_vllm.py` into a NEW results root, the way
# `scripts/run_schedule_tram_rerun.sh` does.
# ===========================================================================
echo "RETIRED: this script would write HF predictions into the vLLM TRAM" >&2
echo "baseline directory (audit H4), and the corrected baseline already" >&2
echo "exists at results/tram_fixed/zero_shot/. Refusing to run." >&2
exit 1
