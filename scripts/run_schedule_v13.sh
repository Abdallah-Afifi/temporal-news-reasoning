#!/bin/bash
# ===========================================================================
# v13 SCHEDULE -- soup + four single-variable arms (docs/v13_plan.md §1)
#
#   soup-v11 : NO training; average of the three v11 seed adapters
#   v13-base : t09 recipe, v13 data (v12 + AUG_GLM3 + AUG_PROG)
#   v13-2ep  : + 2 epochs          (vs v13-base)
#   v13-ctx  : + max_seq_length 4096 (vs v13-base)
#   v13-r32  : + LoRA r=32/a64    (vs v13-base)  [runs LAST, droppable]
#
# Each arm: 5-step dry-run -> train -> merge -> full vLLM eval on the three
# benchmarks (same prompt/engine/date as zs-vllm-pinned) -> merged weights
# deleted. One rescore at the end, test-minus-dev, vs zs-vllm-pinned.
#
# GATES (refuse to start):
#   - logs/sched_v12/schedule_complete.marker must exist (one GPU, serially);
#   - no other GPU job may be running (gpu_guard; ALLOW_GPU_SHARE=1 overrides);
#   - data/combined_80_20_v13 must exist and be non-provisional
#     (scripts/v13_build_status.py: provisional != true and template_rows > 0), unless
#     V13_ALLOW_NO_GLM=1 is exported -- a provisional build's results are NOT
#     quotable as v13.
#
# COMPLETION: schedule_complete.marker only if every arm succeeded; otherwise
# schedule_failed.marker lists the failed arms (the rescore still runs over
# whatever exists, and says so).
#
# LAUNCH from the GPU shell:
#   mkdir -p logs/sched_v13
#   nohup bash scripts/run_schedule_v13.sh > logs/sched_v13/driver.log 2>&1 &
# Resumable: re-running skips every stage marked OK in logs/sched_v13/.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v13
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
DATE="26 Jul 2024"
CFG=experiments/finetuning/LLaMA
export VLLM_USE_FLASHINFER_SAMPLER=0   # mandatory on CUDA 13 (D55)

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }

# GPU-busy guard (audit 2026-10-04 §4.3): one GPU, one job. Refuse to start
# while another training/eval process or schedule driver is alive.
# ALLOW_GPU_SHARE=1 overrides (deliberate, e.g. a CPU-only rerun).
gpu_guard () {
  local tmp; tmp=$(mktemp)
  pgrep -af 'experiments/finetuning/[A-Za-z]+/train\.py|scripts/run_eval_vllm\.py|run_mistral_seeds\.sh|run_schedule_[A-Za-z0-9_]+\.sh|run_v13_prog_pilot\.sh' > "$tmp" || true
  local others; others=$(awk -v me=$$ -v pp=$PPID '$1 != me && $1 != pp' "$tmp")
  rm -f "$tmp"
  if [ -n "$others" ] && [ "${ALLOW_GPU_SHARE:-0}" != "1" ]; then
    echo "ABORT: another GPU job is running (export ALLOW_GPU_SHARE=1 to override):"
    echo "$others"
    exit 1
  fi
}
FAILED=()
gpu_guard

if [ ! -f logs/sched_v12/schedule_complete.marker ]; then
  echo "ABORT: run_schedule_v12.sh has not finished (no logs/sched_v12/schedule_complete.marker)"
  exit 1
fi

if [ ! -s data/combined_80_20_v13/train.jsonl ]; then
  echo "ABORT: data/combined_80_20_v13 not built -- scripts/build_v13_training_data.py"
  exit 1
fi
BUILD=$($PY scripts/v13_build_status.py data/combined_80_20_v13/manifest.json 2>/dev/null || echo "PROVISIONAL unreadable-manifest")
if [ "${BUILD%% *}" != "FINAL" ] && [ "${V13_ALLOW_NO_GLM:-0}" != "1" ]; then
  echo "ABORT: v13 build is not final ($BUILD). Rebuild with the non-programmatic rows,"
  echo "       or export V13_ALLOW_NO_GLM=1 to train anyway (results not quotable as v13)."
  exit 1
fi
echo "$(date '+%F %T') v13 build: $BUILD"

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
        --out "$merged" >> "$Q/merge.log" 2>&1 || { echo "$name: merge failed"; return 1; }
  fi
  local d=${name//-/_} failed_legs=0
  for b in timebench time tram; do
    local leg="${name}_$b"
    stage_done "$leg" && continue
    local dir=results/corrected/${d}_vllm; [ "$b" = tram ] && dir=results/tram_fixed/$d
    echo "$(date '+%F %T') START $leg -> $dir"
    $VLLM scripts/run_eval_vllm.py --model-dir "$merged" --benchmark "$b" --results-dir "$dir" \
        --model-name llama --system-prompt none --date-string "$DATE" \
        --adapter-dir "$out/final" > "$Q/$leg.log" 2>&1
    mark "$leg" $?
    stage_done "$leg" || failed_legs=1
  done
  rm -rf "$merged"
  return $failed_legs
}

# --- soup arm (no training; adapter built by scripts/lora_soup.py) --------
if ! stage_done "soup_v11_soup"; then
  $PY scripts/lora_soup.py > "$Q/soup_v11_soup.log" 2>&1
  mark "soup_v11_soup" $?
fi
stage_done "soup_v11_soup" || FAILED+=(soup-v11)
if stage_done "soup_v11_soup"; then
  SOUP=checkpoints/llama_v11_soup/final
  SOUPM=checkpoints/llama_v11_soup/merged_fp16
  if [ ! -f "$SOUPM/config.json" ]; then
    $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct --adapter "$SOUP" \
        --out "$SOUPM" >> "$Q/merge.log" 2>&1 || echo "soup-v11: merge failed (legs will fail)"
  fi
  for b in timebench time tram; do
    leg="soup_v11_$b"
    stage_done "$leg" && continue
    dir=results/corrected/soup_v11_vllm; [ "$b" = tram ] && dir=results/tram_fixed/soup_v11
    echo "$(date '+%F %T') START $leg -> $dir"
    $VLLM scripts/run_eval_vllm.py --model-dir "$SOUPM" --benchmark "$b" --results-dir "$dir" \
        --model-name llama --system-prompt none --date-string "$DATE" \
        --adapter-dir "$SOUP" > "$Q/$leg.log" 2>&1
    mark "$leg" $?
    stage_done "$leg" || FAILED+=("$leg")
  done
  rm -rf "$SOUPM"
fi

# --- training arms ---------------------------------------------------------
run_arm v13-base $CFG/config_v13_base.yaml || FAILED+=(v13-base)
run_arm v13-2ep  $CFG/config_v13_2ep.yaml  || FAILED+=(v13-2ep)
run_arm v13-ctx  $CFG/config_v13_ctx.yaml  || FAILED+=(v13-ctx)
run_arm v13-r32  $CFG/config_v13_r32.yaml  || FAILED+=(v13-r32)
[ ${#FAILED[@]} -gt 0 ] && echo "$(date '+%F %T') FAILED arms/legs: ${FAILED[*]} -- rescoring what exists"

echo "$(date '+%F %T') START rescore (test-minus-dev)"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
    --exclude-ids data/hpo_dev/dev_ids.json --reference zs-vllm-pinned \
    --out results/rescored/v13_test_minus_dev.json > "$Q/rescore.log" 2>&1
mark rescore $?
stage_done rescore || FAILED+=(rescore)
rm -f "$Q/schedule_complete.marker" "$Q/schedule_failed.marker"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "$(date '+%F %T') DONE" | tee "$Q/schedule_complete.marker"
else
  echo "$(date '+%F %T') FAILED: ${FAILED[*]}" | tee "$Q/schedule_failed.marker"
  exit 1
fi
