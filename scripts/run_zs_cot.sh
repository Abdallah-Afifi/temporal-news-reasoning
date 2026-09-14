#!/usr/bin/env bash
# TOKEN BUDGET (fixed 2026-09-14). --max-new-tokens was 256, which is NOT
# enough for the reasoning this prompt asks for: measured on the completed
# zero-shot CoT run, 22.1% of llama's TIME outputs and 7.5% of its TimeBench
# outputs ran out of budget before emitting the required "ANSWER:" line, and
# those items score ~0.08% -- a budget artifact, not a reasoning result.
# Conditional on reaching the anchor, llama CoT scores 49.41% on TIME and
# 53.01% on TimeBench, against 41.46%/45.18% for standard zero-shot.
# 768 gives the long tail room to terminate. See docs/audit_2026_09_14.md.
# Zero-shot-CoT baselines (the v9/STaR comparison targets), HF engine only.
#
# ARMS: llama and mistral, timebench and time, --prompt-style cot,
#       --max-new-tokens 768, results into results/baseline/zero_shot_cot/
#       (their own arm -- never mixed with standard-prompt zero-shot dirs).
#
# ENGINE: HF for all four legs. Mistral-on-vLLM is banned (2.20pp parity
#       failure, D46); llama follows mistral so each benchmark's arms share
#       one engine (D53). If v9 evaluates on vLLM later, re-run the llama
#       legs there for a matched check (llama-vLLM parity is validated).
#
# BATCH RULES (frozen per model + D55 ceilings):
#       llama      : batch 32, budget 57,344 (both benchmarks)
#       mistral tb : batch 24, budget 26,624 (proven 2026-09-08)
#       mistral time: batch 20, budget 20,480 (proven Sept-1; batch 24 OOMs)
#
# This script WAITS for the zs-mistral chain to complete before starting
# (watches logs/zs_mistral_v3/complete.marker; no GPU used while waiting).
# Idempotent per leg (.done markers); run_baselines resumes from its own
# predictions.jsonl. Launched 2026-09-09 by the opencode session.

set -u
cd "$(dirname "$0")/.."
Q=logs/zs_cot
mkdir -p "$Q"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE=logs/zs_mistral_v3/complete.marker
if [ ! -f "$GATE" ]; then
  echo "$(date '+%F %T') waiting for $GATE (polling every 5 min, no GPU)"
  while [ ! -f "$GATE" ]; do sleep 300; done
fi
echo "$(date '+%F %T') gate passed: zs-mistral chain complete"

# The token budget is declared ONCE and stamped into every .done marker, so a
# budget change invalidates its own markers instead of silently skipping legs
# that completed under the old one. See audit 2026-09-14b.
MAXNEW=768

done_ok() { [ -f "$Q/$1.done" ] && [ "$(cat "$Q/$1.done")" = "OK:$MAXNEW" ]; }
mark()    { echo "$2" > "$Q/$1.done"; }

leg() { # name model benchmark budget batch
  local name="$1" model="$2" bench="$3" budget="$4" batch="${5:-20}"
  if done_ok "$name"; then echo "$(date '+%F %T') skip $name"; return 0; fi
  echo "$(date '+%F %T') START $name (model=$model bench=$bench budget=$budget batch=$batch maxnew=$MAXNEW)"
  venv/bin/python scripts/run_baselines.py \
    --model "$model" --benchmark "$bench" \
    --results-dir ./results/baseline/zero_shot_cot \
    --prompt-style cot --max-new-tokens "$MAXNEW" \
    --batch-size "$batch" --token-budget "$budget" \
    --allow-budget-change \
    > "$Q/$name.log" 2>&1
  local rc=$?
  mark "$name" "$([ $rc -eq 0 ] && echo "OK:$MAXNEW" || echo FAIL:$rc)"
  echo "$(date '+%F %T') END $name rc=$rc"
  return $rc
}

# Cheapest first (D47): a broken setup surfaces in minutes, not hours.
leg llama_timebench    llama   timebench 57344 32 || { echo "llama tb failed; stopping"; exit 1; }
leg mistral_timebench  mistral timebench 26624 24 || { echo "mistral tb failed; stopping"; exit 1; }
leg llama_time         llama   time      57344 32 || { echo "llama time failed; stopping"; exit 1; }
leg mistral_time       mistral time      20480 20 || { echo "mistral time failed; stopping"; exit 1; }

echo "$(date '+%F %T') ALL COT LEGS COMPLETE" | tee "$Q/complete.marker"
