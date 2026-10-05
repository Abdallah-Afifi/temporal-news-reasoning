#!/bin/bash
# ===========================================================================
# v11 SCHEDULE -- prompt-parity arm: dev-set HPO -> final arm -> test-minus-dev
#
# Protocol: docs/hpo_v11_protocol.md (pre-registered 2026-09-23).
#   A. HPO on data/hpo_dev/dev_ids.json (scripts/hpo_v11.py plan/run/report)
#   B. Zero-shot reference on the FULL test pools, vLLM, date pinned
#   C. Best config: full test eval of the best trial's adapter (seed 42)
#   D. Seed replicates 43/44 of the best config (RUN_SEEDS=0 skips)
#   E. Rescore on TEST-MINUS-DEV only, paired against zs-vllm-pinned
#
# LAUNCH from a shell with the GPU (a Claude Code Bash session cannot see it):
#   mkdir -p logs/sched_v11
#   nohup bash scripts/run_schedule_v11.sh > logs/sched_v11/driver.log 2>&1 &
#
# Budget on one RTX 3090 (from v9-glm's measured stage times, ~2.2 h per
# epoch on ~11k rows, ~5.7 h per full 3-benchmark eval): A ~40-45 h,
# B ~6 h, C ~6 h, D ~2 x (2-7 h train + 6 h eval). Every stage is resumable:
# re-running the script skips anything already marked OK.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v11
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
DATE="26 Jul 2024"
DEV=data/hpo_dev/dev_ids.json
export VLLM_USE_FLASHINFER_SAMPLER=0   # mandatory on CUDA 13 (D55)

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }

# ---- 0. preconditions ------------------------------------------------------
for f in data/combined_80_20_v11/train.jsonl $DEV experiments/finetuning/LLaMA/config_v11_parity.yaml; do
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done
$PY scripts/make_hpo_dev_split.py --check > "$Q/devsplit_check.log" 2>&1 \
  || { echo "ABORT: dev split does not reproduce -- see $Q/devsplit_check.log"; exit 1; }

# 5-step training smoke test: a pipeline error would otherwise fail all 11
# trials identically, one after another.
if ! stage_done smoke; then
  $PY experiments/finetuning/LLaMA/train.py --config experiments/finetuning/LLaMA/config_v11_parity.yaml \
      --dry-run > "$Q/smoke.log" 2>&1
  mark smoke $?
  stage_done smoke || { echo "ABORT: training smoke test failed -- see $Q/smoke.log"; exit 1; }
  rm -rf checkpoints/llama_v11_dry_run
fi

# ---- A. HPO ------------------------------------------------------------------
[ -f results/hpo_v11/plan.json ] || $PY scripts/hpo_v11.py plan > "$Q/hpo_plan.log" 2>&1
if stage_done hpo; then echo "$(date '+%F %T') skip hpo"; else
  echo "$(date '+%F %T') START hpo"
  $PY scripts/hpo_v11.py run > "$Q/hpo_run.log" 2>&1
  $PY scripts/hpo_v11.py report > "$Q/hpo_report.log" 2>&1
  mark hpo $?
fi
stage_done hpo || { echo "hpo failed; stopping"; exit 1; }
cat "$Q/hpo_report.log"

# ---- helper: full-test eval of one model on all three benchmarks -------------
full_eval () {  # name model_dir adapter_dir(or "") time_tb_results_dir tram_results_dir
  local name=$1 model=$2 adapter=$3 out=$4 tram_out=$5
  for b in timebench time tram; do
    local leg="${name}_$b"
    stage_done "$leg" && { echo "$(date '+%F %T') skip $leg"; continue; }
    local dir=$out; [ "$b" = tram ] && dir=$tram_out
    local extra=(); [ -n "$adapter" ] && extra=(--adapter-dir "$adapter")
    echo "$(date '+%F %T') START $leg -> $dir"
    $VLLM scripts/run_eval_vllm.py --model-dir "$model" --benchmark "$b" \
        --results-dir "$dir" --model-name llama --system-prompt none \
        --date-string "$DATE" "${extra[@]}" > "$Q/$leg.log" 2>&1
    mark "$leg" $?
  done
}

merge () {  # adapter out
  [ -f "$2/config.json" ] && return 0
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct --adapter "$1" --out "$2" \
      >> "$Q/merge.log" 2>&1
}

# ---- B. zero-shot reference, full test, pinned date ---------------------------
full_eval zs_pinned models/Llama-3.2-3B-Instruct "" \
    results/baseline/zero_shot_vllm_pinned results/tram_fixed/zs_vllm_pinned

# ---- C. best trial, full test --------------------------------------------------
BEST_ADAPTER=$($PY scripts/hpo_v11.py final-config --seed 42)
merge "$BEST_ADAPTER" checkpoints/llama_v11_best/merged_fp16
full_eval v11_best checkpoints/llama_v11_best/merged_fp16 "$BEST_ADAPTER" \
    results/corrected/v11_best_vllm results/tram_fixed/v11_best
rm -rf checkpoints/llama_v11_best/merged_fp16

# ---- D. seed replicates --------------------------------------------------------
if [ "${RUN_SEEDS:-1}" = "1" ]; then
  for s in 43 44; do
    if ! stage_done "train_s$s"; then
      CFG=$($PY scripts/hpo_v11.py final-config --seed $s)
      echo "$(date '+%F %T') START train_s$s ($CFG)"
      $PY experiments/finetuning/LLaMA/train.py --config "$CFG" > "$Q/train_s$s.log" 2>&1
      mark "train_s$s" $?
    fi
    stage_done "train_s$s" || continue
    merge checkpoints/llama_v11_best_s$s/final checkpoints/llama_v11_best_s$s/merged_fp16
    full_eval v11_best_s$s checkpoints/llama_v11_best_s$s/merged_fp16 \
        checkpoints/llama_v11_best_s$s/final \
        results/corrected/v11_best_s${s}_vllm results/tram_fixed/v11_best_s$s
    rm -rf checkpoints/llama_v11_best_s$s/merged_fp16
  done
fi

# ---- E. rescore on test-minus-dev ------------------------------------------------
echo "$(date '+%F %T') START rescore (test-minus-dev)"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
    --exclude-ids "$DEV" --reference zs-vllm-pinned \
    --out results/rescored/v11_test_minus_dev.json > "$Q/rescore.log" 2>&1
mark rescore $?

failed=""
for f in "$Q"/*.done; do [ "$(cat "$f")" = OK ] || failed="$failed $(basename "$f" .done)"; done
if [ -z "$failed" ]; then echo "$(date '+%F %T') ALL STAGES OK" | tee "$Q/schedule_complete.marker"
else echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" | tee "$Q/schedule_complete.marker"; fi
