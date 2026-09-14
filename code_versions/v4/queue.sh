#!/bin/bash
# V4 launch chain: train -> letter-probe gate -> TimeBench -> TIME.
# Per docs/audit_report.md's v4 launch checklist and docs/v4_plan.md Stage 4.
# GPU is free (the LLaMA v1-v3 queue finished 2026-09-05 07:02:59) -- this
# chain starts immediately, no gate file to wait on.
#
# Safe to re-run after a crash: each leg resumes from its own
# predictions.jsonl, and a leg whose .done marker reads OK is skipped.
#
# Must be launched from a shell with real GPU access (a normal terminal, or
# another agent session that has it) -- not from a Claude Code Bash-tool
# session on this machine, which cannot see the GPU at all (confirmed
# docs/session_handoff.md D32/D33).
set -u
cd /home/g02-s26/Mohamed/temporal-news-reasoning || exit 1
Q=logs/queue_v4
mkdir -p "$Q"

leg () {  # name results_dir adapter benchmark
  local name=$1 out=$2 adapter=$3 bench=$4
  if [ "$(cat "$Q/$name.done" 2>/dev/null)" = "OK" ]; then
    echo "$(date '+%F %T') skip $name (already OK)"; return
  fi
  echo "$(date '+%F %T') START $name"
  local args=(--model llama --benchmark "$bench" --results-dir "$out"
              --batch-size 32 --token-budget 57344 --adapter-dir "$adapter")
  if venv/bin/python scripts/run_baselines.py "${args[@]}" > "$Q/$name.log" 2>&1
  then echo OK > "$Q/$name.done"; else echo FAIL > "$Q/$name.done"; fi
  echo "$(date '+%F %T') END   $name -> $(cat "$Q/$name.done")"
}

# ---- stage 1: TRAIN --------------------------------------------------------
if [ "$(cat "$Q/v4_train.done" 2>/dev/null)" = "OK" ]; then
  echo "$(date '+%F %T') skip v4_train (already OK)"
else
  echo "$(date '+%F %T') START v4_train"
  if venv/bin/python experiments/finetuning/LLaMA/train.py \
       --config experiments/finetuning/LLaMA/config_v4.yaml \
       > "$Q/v4_train.log" 2>&1
  then echo OK > "$Q/v4_train.done"; else echo FAIL > "$Q/v4_train.done"; fi
  echo "$(date '+%F %T') END   v4_train -> $(cat "$Q/v4_train.done")"
fi

if [ "$(cat "$Q/v4_train.done" 2>/dev/null)" != "OK" ]; then
  echo "$(date '+%F %T') v4 training did not succeed; stopping (no eval spent on a bad checkpoint)"
  exit 1
fi

# ---- stage 2: letter probe (INFORMATIONAL — no longer a gate) --------------
# It ran on 2026-09-05 and reported 0/100 = 0.0% letter emission, which
# stopped the chain. That was a FALSE NEGATIVE and the gate has been
# downgraded to informational. Evidence, measured from v3-corrected's own
# stored predictions (no GPU needed to reproduce):
#   - 49,364 of TIME's 104,951 items (47%) have a bare LETTER as their gold.
#   - v3-corrected scores 56.2% on exactly those items — above its 37.77%
#     overall — while only 3 of its 49,364 predictions (0.0%) were themselves
#     a bare letter. It answers in option TEXT and the fixed postprocessor in
#     run_baselines.py maps that text back to the right option ("Exact choice
#     text wins over any letter reading").
# So ~0% letter emission is normal for this whole lineage, not a v4
# regression: v3-corrected — the arm that produced the project's best
# results — would fail this same gate. The probe was written for the
# SUPERSEDED v4 draft that carried AUG_LETTER's 1,300 rows and letter-primary
# AUG_MCQ targets; docs/mistake_ledger.md established that the "v3 format
# crash" it guards against was a harness bug, already fixed, which is exactly
# why AUG_LETTER was dropped from the shipped v4 (v4_plan.md decision 1).
# The probe result is still recorded (not falsified) — it is a true
# measurement of a capability v4 deliberately does not have.
if [ "$(cat "$Q/letter_probe.done" 2>/dev/null)" != "" ]; then
  echo "$(date '+%F %T') letter_probe already run -> $(cat "$Q/letter_probe.done") (informational, not gating)"
else
  echo "$(date '+%F %T') START letter_probe (informational)"
  if venv/bin/python scripts/eval_letter_probe.py \
       --adapter checkpoints/llama_v4/final \
       --out "$Q/letter_probe.report" > "$Q/letter_probe.log" 2>&1
  then echo OK > "$Q/letter_probe.done"; else echo FAIL > "$Q/letter_probe.done"; fi
  echo "$(date '+%F %T') END   letter_probe -> $(cat "$Q/letter_probe.done") (informational, not gating)"
fi

# ---- stage 3: TimeBench (~12 min), then TIME (~9 h) ------------------------
leg v4_timebench results/corrected/v4 checkpoints/llama_v4/final timebench
leg v4_time       results/corrected/v4 checkpoints/llama_v4/final time

echo "$(date '+%F %T') v4 queue complete" > "$Q/queue_complete.marker"
