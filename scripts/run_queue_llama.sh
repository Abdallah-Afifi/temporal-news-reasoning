#!/bin/bash
# Confirmed LLaMA run queue. Mistral is NOT queued -- deferred by decision
# until explicitly requested (see docs/v4_plan.md, "Model scope").
#
# Waits for the in-flight baseline TIME leg, stops the baseline chain before
# it can start Mistral, then evaluates the v1/v2/v3 checkpoints under the
# corrected harness. Evaluation only: no training, no data rebuild, and no
# change to the v4 mixture (frozen until the v2/v3 issues are understood).
#
# Safe to re-run after a crash: each leg resumes from its own
# predictions.jsonl, and a leg whose .done marker reads OK is skipped.
set -u
# Portable repo-root cd (2026-09-14). This hardcoded
# /home/g02-s26/Mohamed/temporal-news-reasoning, which is THIS machine's
# path only: on mo-linux the project lives at
# ~/thesis/Thesis2/temporal-news-reasoning, so the script silently ran in
# the wrong directory (or died) there.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
Q=logs/queue_llama
mkdir -p "$Q"

leg () {  # name results_dir adapter benchmark
  local name=$1 out=$2 adapter=$3 bench=$4
  if [ "$(cat "$Q/$name.done" 2>/dev/null)" = "OK" ]; then
    echo "$(date '+%F %T') skip $name (already OK)"; return
  fi
  echo "$(date '+%F %T') START $name"
  local args=(--model llama --benchmark "$bench" --results-dir "$out"
              --batch-size 32 --token-budget 57344)
  if [ -n "$adapter" ]; then args+=(--adapter-dir "$adapter"); fi
  if venv/bin/python scripts/run_baselines.py "${args[@]}" > "$Q/$name.log" 2>&1
  then echo OK > "$Q/$name.done"; else echo FAIL > "$Q/$name.done"; fi
  echo "$(date '+%F %T') END   $name -> $(cat "$Q/$name.done")"
}

echo "$(date '+%F %T') waiting for baseline llama/time to finish"
until [ -f logs/baseline_v3/llama_time.done ]; do sleep 60; done
pkill -f run_baseline_v3.sh; sleep 2
pkill -f "run_baselines.py --model mistral"; sleep 3
rm -rf results/baseline/zero_shot_v3/mistral
echo "$(date '+%F %T') baseline complete; mistral skipped by decision"

# ---------------------------------------------------------------------------
# Order rationale: cheap diagnostics first, then the retrain (which is on the
# critical path to v4), then the slow TIME legs. Evaluating the BROKEN v3
# checkpoint is deliberate and diagnostic -- comparing it against v3's
# published number isolates how much of the -13.8 pp came from the harness
# bugs, and comparing it against v3-corrected isolates how much came from the
# corrupted training data. But it must not delay the retrain, so only its
# 12-minute TimeBench leg runs before training; its 9-hour TIME leg runs after.
# ---------------------------------------------------------------------------

# ---- stage 1a: TimeBench legs, ~12 min each -------------------------------
leg v1_timebench results/corrected/v1 checkpoints/llama/final    timebench
leg v2_timebench results/corrected/v2 checkpoints/llama_v2/final timebench
leg v3_timebench results/corrected/v3 checkpoints/llama_v3/final timebench
echo DONE > "$Q/timebench_stage_complete.marker"

# ---- stage 2: v3-corrected TRAIN, then its TimeBench ----------------------
# v1 and v2 need no retraining: their training data was intact (audited
# 2026-09-03 -- see docs/v4_plan.md). v3 does: its checkpoint was trained
# without its TLQA slice, on answers truncated to the first item, and on 381
# incoherent two-hop rows that only existed because of that truncation.
# config_v3_fixed.yaml differs from config_v3.yaml in data paths and
# output_dir only, so this run isolates the data fix.
if [ "$(cat "$Q/v3fixed_train.done" 2>/dev/null)" = "OK" ]; then
  echo "$(date '+%F %T') skip v3fixed_train (already OK)"
else
  echo "$(date '+%F %T') START v3fixed_train"
  if venv/bin/python experiments/finetuning/LLaMA/train.py \
       --config experiments/finetuning/LLaMA/config_v3_fixed.yaml \
       > "$Q/v3fixed_train.log" 2>&1
  then echo OK > "$Q/v3fixed_train.done"; else echo FAIL > "$Q/v3fixed_train.done"; fi
  echo "$(date '+%F %T') END   v3fixed_train -> $(cat "$Q/v3fixed_train.done")"
fi

if [ "$(cat "$Q/v3fixed_train.done" 2>/dev/null)" = "OK" ]; then
  leg v3fixed_timebench results/corrected/v3_fixed checkpoints/llama_v3_fixed/final timebench
else
  echo "$(date '+%F %T') v3-corrected training did not succeed; skipping its evals"
fi
echo DONE > "$Q/train_stage_complete.marker"

# ---- stage 3: TIME legs, ~9 h each ----------------------------------------
leg v1_time results/corrected/v1 checkpoints/llama/final    time
leg v2_time results/corrected/v2 checkpoints/llama_v2/final time
leg v3_time results/corrected/v3 checkpoints/llama_v3/final time
if [ "$(cat "$Q/v3fixed_train.done" 2>/dev/null)" = "OK" ]; then
  leg v3fixed_time results/corrected/v3_fixed checkpoints/llama_v3_fixed/final time
fi

# Completion marker (audit 2026-09-09 M8, fixed 2026-09-12). This used to be
# written unconditionally, so a queue in which a leg recorded FAIL still ended
# with a file saying "complete" -- and the .done files are the source of truth
# the next stage reads. Write the marker only when no leg failed; otherwise
# name the failures and exit non-zero so a driver or `&&` chain stops here.
_failed=""
for _d in "$Q"/*.done; do
  [ -e "$_d" ] || continue
  case "$(cat "$_d" 2>/dev/null)" in
    OK) ;;
    *) _failed="$_failed $(basename "$_d" .done)";;
  esac
done
if [ -n "$_failed" ]; then
  echo "$(date '+%F %T') QUEUE FAILED — legs:$_failed" \
    | tee "$Q/queue_failed.marker"
  rm -f "$Q/queue_complete.marker"
  exit 1
fi
echo DONE > "$Q/queue_complete.marker"
echo "$(date '+%F %T') LLaMA queue complete"
