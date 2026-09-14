#!/usr/bin/env bash
# Zero-shot llama on TRAM (980,918 items), STANDARD prompt, HF engine
# (user directive 2026-09-11: HF not vLLM).
#
# Gate: the few_shot_cot chain must complete first (GPU serialization).
# Results: results/baseline/zero_shot_v3/llama/tram/zero_shot/
# Batch: 32 / 57,344 (proven llama ceilings, D8/D55). TRAM items are
# context-free (median 0 chars) so prompts are SHORT: expect ~3-6 ex/s,
# ETA ~2-3 days. Idempotent via the runner's own resume.
set -u
cd "$(dirname "$0")/.."
Q=logs/zs_tram_llama
mkdir -p "$Q"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE=logs/fewshot_cot/complete.marker
if [ ! -f "$GATE" ]; then
  echo "$(date '+%F %T') waiting for $GATE (polling every 5 min, no GPU)"
  while [ ! -f "$GATE" ]; do sleep 300; done
fi
echo "$(date '+%F %T') gate passed: few_shot_cot chain complete"

if [ -f "$Q/tram.done" ] && [ "$(cat "$Q/tram.done")" = "OK" ]; then
  echo "already done"; exit 0
fi
echo "$(date '+%F %T') START zs-llama TRAM (980,918 items)"
venv/bin/python scripts/run_baselines.py \
  --model llama --benchmark tram \
  --results-dir ./results/baseline/zero_shot_v3 \
  --batch-size 32 --token-budget 57344 \
  > "$Q/tram.log" 2>&1
rc=$?
echo "$([ $rc -eq 0 ] && echo OK || echo FAIL:$rc)" > "$Q/tram.done"
echo "$(date '+%F %T') END TRAM rc=$rc"
