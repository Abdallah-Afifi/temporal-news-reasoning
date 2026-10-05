#!/bin/bash
# ===========================================================================
# MISTRAL SCHEDULE -- ports the audit_2026_09_23 / v11 fixes to Mistral
# (prompt parity, seed fix, dev-selected hyperparameters; no date pin needed
# -- Mistral-7B-Instruct-v0.3's chat template has no date field). See
# docs/mistral_plan.md.
#
# Stages:
#   1. HPO-lite on dev only (scripts/hpo_mistral.py): anchor (LLaMA's t09
#      hyperparameters, transplanted) + 5 sampled trials, same dev split
#      v11 used. ~6 short trials, evaluated on the 5k/2.5k/10k dev split only
#      (not full test) -- much cheaper than a full-benchmark arm each.
#   2. zero-shot Mistral through vLLM, full test sets, no adapter -- becomes
#      the "zs-vllm-mistral" reference (analogous to zs-vllm-pinned for
#      LLaMA). No prior Mistral run has ever gone through vLLM.
#   3. The winning HPO trial's own adapter (seed 42 IS the winning trial --
#      no retrain, same trick hpo_v11.py's final-config uses), merged and
#      evaluated on the full test sets.
#   4. One rescore on test-minus-dev.
#
# LAUNCH from the GPU shell (this Claude Code session has none -- re-verified
# 2026-09-28, same boundary as docs/audit_2026_09_12.md D33):
#   mkdir -p logs/sched_mistral
#   nohup bash scripts/run_schedule_mistral.sh > logs/sched_mistral/driver.log 2>&1 &
# Resumable: re-running skips every stage marked OK in logs/sched_mistral/.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_mistral
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
BASE=models/Mistral-7B-Instruct-v0.3
export VLLM_USE_FLASHINFER_SAMPLER=0   # mandatory on CUDA 13 (D55)

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }

# ---- Stage 0: smoke test (5 steps, base config) ---------------------------
# This whole pipeline (prompt_format=eval_parity, seed fix, 4-bit QLoRA) is
# NEW for Mistral (ported from LLaMA's v11 2026-09-28) and has never run on a
# GPU. Fail cheap here, before HPO commits real GPU-hours to it.
if ! stage_done smoke; then
  $PY experiments/finetuning/Mistral/train.py \
      --config experiments/finetuning/Mistral/config_mistral_parity.yaml --dry-run \
      > "$Q/smoke.log" 2>&1
  mark smoke $?
  rm -rf checkpoints/mistral_parity_dry_run
fi
stage_done smoke || { echo "ABORT: smoke test failed -- see $Q/smoke.log"; exit 1; }

# ---- Stage 1: HPO-lite (dev split only) ----------------------------------
if ! stage_done hpo_plan; then
  $PY scripts/hpo_mistral.py plan > "$Q/hpo_plan.log" 2>&1
  mark hpo_plan $?
fi
stage_done hpo_plan || { echo "ABORT: hpo_mistral plan failed -- see $Q/hpo_plan.log"; exit 1; }

if ! stage_done hpo_run; then
  echo "$(date '+%F %T') START hpo_run"
  $PY scripts/hpo_mistral.py run > "$Q/hpo_run.log" 2>&1
  mark hpo_run $?
fi
stage_done hpo_run || { echo "ABORT: hpo_mistral run failed -- see $Q/hpo_run.log"; exit 1; }

if ! stage_done hpo_report; then
  $PY scripts/hpo_mistral.py report > "$Q/hpo_report.log" 2>&1
  mark hpo_report $?
fi
stage_done hpo_report || { echo "ABORT: hpo_mistral report failed -- see $Q/hpo_report.log"; exit 1; }
BEST_ID=$($PY -c "import json;print(json.load(open('results/hpo_mistral/best.json'))['trial']['id'])")
echo "$(date '+%F %T') best trial: $BEST_ID"

# ---- Stage 2: zero-shot Mistral through vLLM, full test sets -------------
for b in timebench time tram; do
  leg="zs_$b"
  stage_done "$leg" && continue
  dir=results/baseline/zero_shot_vllm; [ "$b" = tram ] && dir=results/tram_fixed/zs_vllm_mistral
  echo "$(date '+%F %T') START $leg -> $dir/mistral/$b"
  $VLLM scripts/run_eval_vllm.py --model-dir "$BASE" --benchmark "$b" --results-dir "$dir" \
      --model-name mistral --system-prompt none > "$Q/$leg.log" 2>&1
  mark "$leg" $?
done

# ---- Stage 3: merge the winning trial's adapter, full-test eval ----------
ADAPTER="checkpoints/hpo_mistral/$BEST_ID/final"
MERGED="checkpoints/mistral_best/merged_fp16"
if [ ! -f "$MERGED/config.json" ]; then
  $PY scripts/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" >> "$Q/merge.log" 2>&1
fi
for b in timebench time tram; do
  leg="best_$b"
  stage_done "$leg" && continue
  dir=results/corrected/mistral_best_vllm; [ "$b" = tram ] && dir=results/tram_fixed/mistral_best
  echo "$(date '+%F %T') START $leg -> $dir"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$b" --results-dir "$dir" \
      --model-name mistral --system-prompt none --adapter-dir "$ADAPTER" > "$Q/$leg.log" 2>&1
  mark "$leg" $?
done
rm -rf "$MERGED"

# ---- Stage 4: rescore (test-minus-dev) ------------------------------------
echo "$(date '+%F %T') START rescore (test-minus-dev)"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
    --exclude-ids data/hpo_dev/dev_ids.json --reference zs-vllm-mistral \
    --out results/rescored/mistral_test_minus_dev.json > "$Q/rescore.log" 2>&1
mark rescore $?
echo "$(date '+%F %T') DONE" | tee "$Q/schedule_complete.marker"
