#!/bin/bash
# Fresh zero-shot baseline (v3 protocol), 2026-09-03.
#
# Supersedes results/baseline/zero_shot_v2, which was generated through the
# MCQ letter-extraction bug (predictions rewritten into options the model
# never named) and against the TIME+TIME-Lite pooled set.
#
# Protocol frozen for this campaign -- changing any of it changes the numbers:
#   * TIME = TIME_Newest.json only (104,951). TIME-Lite is a duplicate
#     distribution: all 1,549 of its questions repeat a TIME question and 118
#     carry a conflicting gold.
#   * TimeBench = full loaded set (21,188).
#   * greedy (temperature 0.0), max_new_tokens 128, left truncation at 4096.
#   * batch size and token budget are per-model and FIXED: batch composition
#     changes greedy output (measured 89.25% agreement when the same items
#     are regenerated in different batches), so these are protocol, not tuning.
#   * resume regenerates partially-complete batches on purpose, to preserve
#     batch composition. Never "optimise" it to skip pending items only.
set -u
cd /home/g02-s26/Mohamed/temporal-news-reasoning || exit 1
OUT=./results/baseline/zero_shot_v3
LOG=logs/baseline_v3

run () {  # model benchmark batch budget
  local m=$1 b=$2 bs=$3 tb=$4
  echo "=== $(date '+%F %T') start $m/$b (batch=$bs budget=$tb) ==="
  venv/bin/python scripts/run_baselines.py \
    --model "$m" --benchmark "$b" --mode zero_shot \
    --results-dir "$OUT" --batch-size "$bs" --token-budget "$tb" \
    > "$LOG/${m}_${b}.log" 2>&1 \
    && echo OK > "$LOG/${m}_${b}.done" \
    || { echo FAIL > "$LOG/${m}_${b}.done"; echo "FAILED $m/$b"; }
  echo "=== $(date '+%F %T') end   $m/$b ==="
}

# LLaMA-3.2-3B first (fast, ~10 h total), then Mistral-7B (~30 h).
run llama   timebench 32 57344
run llama   time      32 57344
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run mistral timebench 20 20480
run mistral time      20 20480

echo DONE > "$LOG/baseline_v3_complete.marker"
echo "=== $(date '+%F %T') baseline v3 complete ==="
