#!/bin/bash
# V5 launch chain: train -> style audit (informational) -> TimeBench -> TIME.
# Per docs/v5_plan.md. One variable vs v4: rehearsal 3,344 -> 10,032 rows.
#
# Safe to re-run after a crash: each leg resumes from its own
# predictions.jsonl, and a leg whose .done marker reads OK is skipped.
#
# Must be launched from a shell with real GPU access (a normal terminal, or
# another agent session that has it) -- NOT from a Claude Code Bash-tool
# session on this machine, which cannot see the GPU at all (session_handoff
# D32/D33).
#
#   mkdir -p logs/queue_v5    # the redirect target must exist BEFORE the
#                             # shell opens it; the script's own mkdir is
#                             # too late
#   nohup bash scripts/run_queue_v5.sh > logs/queue_v5/driver.log 2>&1 &
#
# Killing it safely -- never `pkill -f run_queue_v5`, that pattern matches
# your own shell:
#   ps -eo pid,cmd --no-headers | awk '$2=="bash" && $3=="scripts/run_queue_v5.sh" {print $1}'
set -u
cd /home/g02-s26/Mohamed/temporal-news-reasoning || exit 1
Q=logs/queue_v5
mkdir -p "$Q"

leg () {  # name results_dir adapter benchmark
  local name=$1 out=$2 adapter=$3 bench=$4
  if [ "$(cat "$Q/$name.done" 2>/dev/null)" = "OK" ]; then
    echo "$(date '+%F %T') skip $name (already OK)"; return
  fi
  echo "$(date '+%F %T') START $name"
  # Batch size, token budget and the prompt-length sort are PROTOCOL --
  # identical to v1-v4. Changing any of them changes the numbers.
  local args=(--model llama --benchmark "$bench" --results-dir "$out"
              --batch-size 32 --token-budget 57344 --adapter-dir "$adapter")
  if venv/bin/python scripts/run_baselines.py "${args[@]}" > "$Q/$name.log" 2>&1
  then echo OK > "$Q/$name.done"; else echo FAIL > "$Q/$name.done"; fi
  echo "$(date '+%F %T') END   $name -> $(cat "$Q/$name.done")"
}

# ---- stage 1: TRAIN (~10-12 h; 18,180 rows, longer than v4's 12,830) -------
if [ "$(cat "$Q/v5_train.done" 2>/dev/null)" = "OK" ]; then
  echo "$(date '+%F %T') skip v5_train (already OK)"
else
  echo "$(date '+%F %T') START v5_train"
  if venv/bin/python experiments/finetuning/LLaMA/train.py \
       --config experiments/finetuning/LLaMA/config_v5.yaml \
       > "$Q/v5_train.log" 2>&1
  then echo OK > "$Q/v5_train.done"; else echo FAIL > "$Q/v5_train.done"; fi
  echo "$(date '+%F %T') END   v5_train -> $(cat "$Q/v5_train.done")"
fi

if [ "$(cat "$Q/v5_train.done" 2>/dev/null)" != "OK" ]; then
  echo "$(date '+%F %T') v5 training did not succeed; stopping (no eval spent on a bad checkpoint)"
  exit 1
fi

# ---- stage 2: answer-style audit (INFORMATIONAL, ~10 min) ------------------
# THE lesson of the v4 post-mortem. v4's TIME number read 33.62% and was
# taken for a 4.15pp regression against v3-corrected; three quarters of that
# gap was v4 answering correctly in a different surface form ("Fact 1" for
# "Fact 1 happened earlier.", "18 March 1934" for "March 18, 1934."). Nine
# hours of evaluation were spent before that was visible. This probe makes it
# visible in ten minutes, on 300 stratified items, by reporting the strict
# exact-match score next to the v5-protocol score for the SAME generations.
#
# It is deliberately NOT a gate: a style shift is not a reason to discard a
# checkpoint, and the v4 letter probe already showed what happens when a
# diagnostic written for a superseded design is allowed to stop the chain.
if [ "$(cat "$Q/style_audit.done" 2>/dev/null)" != "" ]; then
  echo "$(date '+%F %T') style_audit already run -> $(cat "$Q/style_audit.done") (informational)"
else
  echo "$(date '+%F %T') START style_audit (informational)"
  if venv/bin/python scripts/audit_output_style.py \
       --adapter checkpoints/llama_v5/final \
       --reference results/corrected/v4/llama/time/finetuned/predictions.jsonl \
       --out "$Q/style_audit.report" > "$Q/style_audit.log" 2>&1
  then echo OK > "$Q/style_audit.done"; else echo FAIL > "$Q/style_audit.done"; fi
  echo "$(date '+%F %T') END   style_audit -> $(cat "$Q/style_audit.done") (informational)"
fi

# ---- stage 3: letter probe (INFORMATIONAL) --------------------------------
# Kept for lineage continuity: v4 emitted a bare letter on 0/100 items, and
# v4_plan.md asked for v1/v2/v3-corrected to be re-probed to establish
# whether 0% is universal. Recorded, never gating -- see the v4 queue script
# for the full false-negative account.
if [ "$(cat "$Q/letter_probe.done" 2>/dev/null)" != "" ]; then
  echo "$(date '+%F %T') letter_probe already run -> $(cat "$Q/letter_probe.done") (informational)"
else
  echo "$(date '+%F %T') START letter_probe (informational)"
  if venv/bin/python scripts/eval_letter_probe.py \
       --adapter checkpoints/llama_v5/final \
       --out "$Q/letter_probe.report" > "$Q/letter_probe.log" 2>&1
  then echo OK > "$Q/letter_probe.done"; else echo FAIL > "$Q/letter_probe.done"; fi
  echo "$(date '+%F %T') END   letter_probe -> $(cat "$Q/letter_probe.done") (informational)"
fi

# ---- stage 4: TimeBench (~12 min), then TIME (~9 h) -----------------------
leg v5_timebench results/corrected/v5 checkpoints/llama_v5/final timebench
leg v5_time       results/corrected/v5 checkpoints/llama_v5/final time

# ---- stage 5: rescore every arm under the v5 protocol (CPU, ~2 min) -------
# v5's headline numbers must be produced by the SAME scorer as every other
# arm, so this reruns the whole table rather than scoring v5 alone.
echo "$(date '+%F %T') START rescore_v5_protocol"
venv/bin/python scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && echo "$(date '+%F %T') END   rescore_v5_protocol -> OK" \
  || echo "$(date '+%F %T') END   rescore_v5_protocol -> FAIL"

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
echo "$(date '+%F %T') v5 queue complete" > "$Q/queue_complete.marker"
