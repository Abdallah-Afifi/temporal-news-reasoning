#!/bin/bash
# ===========================================================================
# v13-PROG PILOT -- dev-split ablation of AUG_PROG, no GLM rows yet.
#
# Answers ONE question while the GLM-5.2 wave waits for its API key: do the
# 13,200 programmatic rows move dev accuracy on top of v12?  Trains
# config_v13_prog_pilot.yaml (t09 recipe, provisional v13 data = v12 +
# AUG_PROG), merges, dev-evals all three benchmarks (--ids-file, same
# prompt/date/engine as zs-vllm-pinned) and scores with the HPO's own
# score() against results/hpo_v11/zs_dev -- directly comparable to t09's
# dev objective (+5.48pp mean delta).
#
# This is a PILOT: results feed arm selection only; nothing here is
# quotable as v13 (docs/v13_plan.md §4 provisional-build rule).
#
#   mkdir -p logs/v13_prog_pilot
#   nohup bash scripts/run_v13_prog_pilot.sh > logs/v13_prog_pilot/driver.log 2>&1 &
#
# COMPLETION (audit 2026-10-04 §4.3): pilot_complete.marker only when every
# stage is OK; any other exit writes pilot_failed.marker. Either marker means
# the GPU is released (run_mistral_seeds.sh waits on either).
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/v13_prog_pilot
mkdir -p "$Q"
PY=venv/bin/python
VLLM=venv_vllm/bin/python
DATE="26 Jul 2024"
CFG=experiments/finetuning/LLaMA/config_v13_prog_pilot.yaml
OUT=checkpoints/llama_v13_prog_pilot
RES=results/hpo_v13_pilot
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
COMPLETE=0
finish () {
  if [ "$COMPLETE" != "1" ]; then
    rm -f "$Q/pilot_complete.marker"
    echo "$(date '+%F %T') PILOT FAILED -- see $Q/*.done" | tee "$Q/pilot_failed.marker"
  fi
}
trap finish EXIT
gpu_guard

if [ ! -s data/combined_80_20_v13/train.jsonl ]; then
  echo "ABORT: no v13 data"; exit 1
fi

if ! stage_done smoke; then
  $PY experiments/finetuning/LLaMA/train.py --config "$CFG" --dry-run > "$Q/smoke.log" 2>&1
  mark smoke $?
  rm -rf "${OUT}_dry_run"
fi
stage_done smoke || { echo "smoke failed"; exit 1; }

if ! stage_done train; then
  echo "$(date '+%F %T') START train"
  $PY experiments/finetuning/LLaMA/train.py --config "$CFG" > "$Q/train.log" 2>&1
  mark train $?
fi
stage_done train || exit 1

MERGED="$OUT/merged_fp16"
if [ ! -f "$MERGED/config.json" ]; then
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
      --adapter "$OUT/final" --out "$MERGED" >> "$Q/merge.log" 2>&1 || exit 1
fi

for b in timebench time tram; do
  stage_done "eval_$b" && continue
  echo "$(date '+%F %T') START dev eval $b"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$b" \
      --results-dir "$RES" --model-name llama --system-prompt none \
      --date-string "$DATE" --ids-file data/hpo_dev/dev_ids.json \
      --adapter-dir "$OUT/final" > "$Q/eval_$b.log" 2>&1
  mark "eval_$b" $?
done
rm -rf "$MERGED"
for b in timebench time tram; do stage_done "eval_$b" || exit 1; done

if ! stage_done score; then
  echo "$(date '+%F %T') START score (vs results/hpo_v11/zs_dev)"
  $PY scripts/score_v13_pilot.py > "$Q/score.log" 2>&1
  mark score $?
fi
cat "$Q/score.log" 2>/dev/null | tail -20
stage_done score || exit 1
COMPLETE=1
rm -f "$Q/pilot_failed.marker"
echo "$(date '+%F %T') PILOT DONE" | tee "$Q/pilot_complete.marker"
