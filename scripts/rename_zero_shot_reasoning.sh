#!/usr/bin/env bash
# Rename the misleading "zero_shot_cot" naming to "zero_shot_reasoning"
# (D61 terminology: untrained think-then-answer = zs-R; traces-in-prompt
# = few_shot_cot; trained-on-traces = STaR/v9).
#
# REFUSES to run while any evaluation/driver process is alive (renaming
# under a live writer corrupts results). Run only after BOTH the zs-R
# chain (logs/zs_cot) and the few-shot chain (logs/fewshot_cot) complete.
#
# What moves:
#   results/baseline/zero_shot_cot  -> results/baseline/zero_shot_reasoning
#   logs/zs_cot                     -> logs/zsr
# Historical D-log entries keep their original wording (append-only);
# this script prints the mapping for the session_state header.

set -u
cd "$(dirname "$0")/.."

if pgrep -f "run_baselines.py" >/dev/null 2>&1; then
  echo "REFUSING: run_baselines.py is running. Rename only when idle."; exit 1
fi
if pgrep -f "run_zs_cot.sh|run_fewshot_cot.sh" >/dev/null 2>&1; then
  echo "REFUSING: a CoT driver is alive. Rename only when idle."; exit 1
fi

if [ -d results/baseline/zero_shot_cot ] && [ ! -e results/baseline/zero_shot_reasoning ]; then
  mv results/baseline/zero_shot_cot results/baseline/zero_shot_reasoning
  echo "moved: results/baseline/zero_shot_cot -> results/baseline/zero_shot_reasoning"
fi
if [ -d logs/zs_cot ] && [ ! -e logs/zsr ]; then
  mv logs/zs_cot logs/zsr
  echo "moved: logs/zs_cot -> logs/zsr"
fi
echo "done. mapping: zs-CoT (D56-D60) == zero_shot_reasoning (zs-R)."
