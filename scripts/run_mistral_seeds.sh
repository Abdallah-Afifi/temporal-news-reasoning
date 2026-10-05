#!/bin/bash
# ===========================================================================
# MISTRAL SEED REPLICATES -- the parked follow-up from mistral_plan.md §7.
#
# mistral-best (m05) was a single-seed measurement: no noise floor, unlike
# v11's 42/43/44. This trains seeds 43 and 44 of the EXACT m05 recipe
# (config_mistral_best_s4X.yaml: lr 3.72e-05, 1 epoch, r16, QLoRA, v12
# data -- the ONLY variable is the seed), full-test evals each, and rescores
# against zs-vllm-mistral. Output arms: mistral-best-s43 / mistral-best-s44
# (registered in rescore_v5_protocol.py).
#
# GPU etiquette: one GPU, serial. Waits for the v13-prog pilot to release the
# GPU first (logs/v13_prog_pilot/pilot_{complete,failed}.marker) -- pass
# NO_WAIT=1 to skip the wait -- then refuses to start if any other GPU job is
# alive (gpu_guard; ALLOW_GPU_SHARE=1 overrides). schedule_complete.marker only
# if both seeds and the rescore succeeded, else schedule_failed.marker.
# If the v13 schedule needs the GPU while this runs, interrupt it (resumable
# via .done markers) and give the GPU to run_schedule_v13.sh.
#
#   mkdir -p logs/mistral_seeds
#   nohup bash scripts/run_mistral_seeds.sh > logs/mistral_seeds/driver.log 2>&1 &
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/mistral_seeds
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
BASE=models/Mistral-7B-Instruct-v0.3
MCFG=experiments/finetuning/Mistral
export VLLM_USE_FLASHINFER_SAMPLER=0

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

if [ "${NO_WAIT:-0}" != "1" ]; then
  while [ ! -f logs/v13_prog_pilot/pilot_complete.marker ] && [ ! -f logs/v13_prog_pilot/pilot_failed.marker ]; do
    echo "$(date '+%F %T') waiting for v13-prog pilot to release the GPU..."
    sleep 300
  done
fi
gpu_guard

run_seed () {  # seed
  local s=$1
  local cfg=$MCFG/config_mistral_best_s$s.yaml
  local out=checkpoints/mistral_best_s$s
  if ! stage_done "s${s}_smoke"; then
    $PY $MCFG/train.py --config "$cfg" --dry-run > "$Q/s${s}_smoke.log" 2>&1
    mark "s${s}_smoke" $?
    rm -rf "${out}_dry_run"
  fi
  stage_done "s${s}_smoke" || { echo "s$s smoke failed"; return 1; }
  if ! stage_done "s${s}_train"; then
    echo "$(date '+%F %T') START s${s}_train"
    $PY $MCFG/train.py --config "$cfg" > "$Q/s${s}_train.log" 2>&1
    mark "s${s}_train" $?
  fi
  stage_done "s${s}_train" || return 1
  local merged="$out/merged_fp16"
  if [ ! -f "$merged/config.json" ]; then
    $PY scripts/merge_lora.py --base "$BASE" --adapter "$out/final" \
        --out "$merged" >> "$Q/merge.log" 2>&1 || { echo "s$s merge failed"; return 1; }
  fi
  local failed_legs=0
  for b in timebench time tram; do
    stage_done "s${s}_$b" && continue
    dir=results/corrected/mistral_best_s${s}_vllm; [ "$b" = tram ] && dir=results/tram_fixed/mistral_best_s$s
    echo "$(date '+%F %T') START s${s} $b -> $dir"
    $VLLM scripts/run_eval_vllm.py --model-dir "$merged" --benchmark "$b" --results-dir "$dir" \
        --model-name mistral --system-prompt none --adapter-dir "$out/final" > "$Q/s${s}_${b}.log" 2>&1
    mark "s${s}_$b" $?
    stage_done "s${s}_$b" || failed_legs=1
  done
  rm -rf "$merged"
  return $failed_legs
}

run_seed 43 || FAILED+=(s43)
run_seed 44 || FAILED+=(s44)
[ ${#FAILED[@]} -gt 0 ] && echo "$(date '+%F %T') FAILED seeds: ${FAILED[*]} -- rescoring what exists"

echo "$(date '+%F %T') START rescore (test-minus-dev, vs zs-vllm-mistral)"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
    --exclude-ids data/hpo_dev/dev_ids.json --reference zs-vllm-mistral \
    --out results/rescored/mistral_test_minus_dev.json > "$Q/rescore.log" 2>&1
mark rescore $?
stage_done rescore || FAILED+=(rescore)
rm -f "$Q/schedule_complete.marker" "$Q/schedule_failed.marker"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "$(date '+%F %T') DONE" | tee "$Q/schedule_complete.marker"
else
  echo "$(date '+%F %T') FAILED: ${FAILED[*]}" | tee "$Q/schedule_failed.marker"
  exit 1
fi
