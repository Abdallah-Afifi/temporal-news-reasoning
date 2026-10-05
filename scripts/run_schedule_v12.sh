#!/bin/bash
# ===========================================================================
# v12 SCHEDULE -- three single-variable arms (docs/v12_plan.md)
#
#   v11-2ep  : t09 recipe, v11 data, 2 epochs            (vs v11-best)
#   v12-data : t09 recipe, v12 data                       (vs v11-best)
#   v12-ctx  : t09 recipe, v12 data, max_seq_length 4096  (vs v12-data)
#
# Each arm: 5-step smoke test -> train -> merge -> full eval on all three
# benchmarks (same prompt/engine/date as zs-vllm-pinned) -> merged weights
# deleted. Then one rescore on test-minus-dev. v11-2ep needs no new data and
# runs first; the v12 arms run only once data/combined_80_20_v12 exists
# (scripts/build_v12_training_data.py, after the ORDERS_v12 rows are banked).
#
# LAUNCH from the GPU shell, ONLY after run_schedule_v11.sh has finished:
#   mkdir -p logs/sched_v12
#   nohup bash scripts/run_schedule_v12.sh > logs/sched_v12/driver.log 2>&1 &
# Resumable: re-running skips every stage marked OK in logs/sched_v12/.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v12
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
DATE="26 Jul 2024"
CFG=experiments/finetuning/LLaMA
export VLLM_USE_FLASHINFER_SAMPLER=0   # mandatory on CUDA 13 (D55)

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }

# One GPU: never overlap v11's seeds/eval.
if [ ! -f logs/sched_v11/schedule_complete.marker ]; then
  echo "ABORT: run_schedule_v11.sh has not finished (no logs/sched_v11/schedule_complete.marker)"
  exit 1
fi

run_arm () {  # name config
  local name=$1 cfg=$2
  local out; out=$(grep '^output_dir:' "$cfg" | awk '{print $2}')
  if ! stage_done "${name}_smoke"; then
    $PY $CFG/train.py --config "$cfg" --dry-run > "$Q/${name}_smoke.log" 2>&1
    mark "${name}_smoke" $?
    rm -rf "${out}_dry_run"
  fi
  stage_done "${name}_smoke" || { echo "$name: smoke test failed -- see $Q/${name}_smoke.log"; return 1; }
  if ! stage_done "${name}_train"; then
    echo "$(date '+%F %T') START ${name}_train"
    $PY $CFG/train.py --config "$cfg" > "$Q/${name}_train.log" 2>&1
    mark "${name}_train" $?
  fi
  stage_done "${name}_train" || return 1
  local merged="$out/merged_fp16"
  if [ ! -f "$merged/config.json" ]; then
    $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct --adapter "$out/final" \
        --out "$merged" >> "$Q/merge.log" 2>&1
  fi
  local d=${name//-/_}
  for b in timebench time tram; do
    local leg="${name}_$b"
    stage_done "$leg" && continue
    local dir=results/corrected/${d}_vllm; [ "$b" = tram ] && dir=results/tram_fixed/$d
    echo "$(date '+%F %T') START $leg -> $dir"
    $VLLM scripts/run_eval_vllm.py --model-dir "$merged" --benchmark "$b" --results-dir "$dir" \
        --model-name llama --system-prompt none --date-string "$DATE" \
        --adapter-dir "$out/final" > "$Q/$leg.log" 2>&1
    mark "$leg" $?
  done
  rm -rf "$merged"
}

run_arm v11-2ep $CFG/config_v11_2ep.yaml
if [ -s data/combined_80_20_v12/train.jsonl ]; then
  run_arm v12-data $CFG/config_v12_data.yaml
  run_arm v12-ctx  $CFG/config_v12_ctx.yaml
else
  echo "$(date '+%F %T') SKIP v12-data / v12-ctx: data/combined_80_20_v12 not built yet"
fi

echo "$(date '+%F %T') START rescore (test-minus-dev)"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
    --exclude-ids data/hpo_dev/dev_ids.json --reference zs-vllm-pinned \
    --out results/rescored/v12_test_minus_dev.json > "$Q/rescore.log" 2>&1
mark rescore $?
echo "$(date '+%F %T') DONE" | tee "$Q/schedule_complete.marker"
