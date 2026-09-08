#!/usr/bin/env bash
# 60-step OOM + throughput pilot for the v7 speed settings.
#
# The ONLY gate on config_v7_fast.yaml is whether gradient_checkpointing:false
# fits in 25.3 GB on this data. Gradients are mathematically identical either
# way, so if it fits there is no accuracy question to answer — it is a pure
# speed/memory trade. v4 OOMed with it off, but v4 carried full
# HotpotQA/DROP/CoQA passages; v7's mixture is different, so this must be
# measured rather than assumed in either direction.
#
# Takes ~10 min. Run from a shell with real GPU access (D32/D33).
#   bash scripts/pilot_speed_v7.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/pilot_v7

run () {  # label config
  local label=$1 cfg=$2
  echo "=== $label ($cfg) ==="
  timeout 1800 venv/bin/python experiments/finetuning/LLaMA/train.py \
      --config "$cfg" --max-steps 60 \
      > "logs/pilot_v7/$label.log" 2>&1
  local rc=$?
  if grep -qi "out of memory\|CUDA out of memory" "logs/pilot_v7/$label.log"; then
    echo "  RESULT: OOM  -> do NOT use this config"
  elif [ $rc -ne 0 ]; then
    echo "  RESULT: FAILED rc=$rc (see logs/pilot_v7/$label.log)"
  else
    echo "  RESULT: OK"
    tr '\r' '\n' < "logs/pilot_v7/$label.log" | grep -oE "[0-9.]+s/it" | tail -3 \
      | sed 's/^/    /'
    grep -oE "GPU memory: [0-9.]+ GB allocated, [0-9.]+ GB reserved" \
      "logs/pilot_v7/$label.log" | tail -1 | sed 's/^/    /'
  fi
  rm -rf checkpoints/llama_v7_pilot
  echo
}

run baseline experiments/finetuning/LLaMA/config_v7.yaml
run fast     experiments/finetuning/LLaMA/config_v7_fast.yaml

echo "Compare the s/it figures. If 'fast' is OK, launch v7 with"
echo "config_v7_fast.yaml (edit scripts/run_queue_v7.sh line ~74)."
echo "If 'fast' OOMs, try gradient_checkpointing:true + group_by_length:true"
echo "— that keeps the ~40% padding win with no memory cost."
