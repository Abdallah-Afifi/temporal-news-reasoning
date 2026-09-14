#!/bin/bash
# ===========================================================================
# v7-CORRECTED SCHEDULE — train -> merge -> vLLM eval (TimeBench, TIME, TRAM)
#                          -> rescore
#
# COPY OF run_schedule_v7.sh (2026-09-09), adapted. Per the project convention
# (D58), a new arm gets NEW files; scripts/run_schedule_v7.sh is left untouched
# so v7 stays reproducible.
#
# THE ARM. v6's mixture with one variable: every dual-gold MCQ target
# `[text, LETTER]` becomes `[text]` (AUG_MCQ, AUG_MCQ2, AUG_NOANS — 2,000 train
# rows). Same 14,850 rows, same slice sizes, same seed, and config_v7_corrected
# .yaml differs from config_v6.yaml in three path lines only. Control = v6.
# See docs/v7_corrected_plan.md and docs/session_handoff.md D56/D57.
#
# LAUNCH (the log dir must exist BEFORE the shell opens the redirect):
#   mkdir -p logs/sched_v7_corrected
#   nohup bash scripts/run_schedule_v7_corrected.sh \
#         > logs/sched_v7_corrected/driver.log 2>&1 &
#
# Must run from a shell with real GPU access — a Claude Code Bash-tool session
# can neither see the GPU nor signal these processes (D32/D33/D55).
#
# WHAT THIS DROPS vs run_schedule_v7.sh, and why: the v6 vLLM re-runs, the
# Mistral parity test and the zero-shot TRAM baseline all completed OK on
# 2026-09-09 and are not repeated. Their results stand.
#
# DISK. A merged fp16 3B is ~6 GB and is NOT deleted automatically — remove
# checkpoints/llama_v7_corrected/merged_fp16 once the vLLM legs are done.
#
# WHY A MERGE IS NEEDED. run_eval_vllm.py loads whatever is at --model-dir;
# its --adapter-dir only selects the training system prompt, it does NOT load
# LoRA weights into vLLM. The adapter must be merged into the base first.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v7_corrected
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
MERGED=checkpoints/llama_v7_corrected/merged_fp16

# MANDATORY for every vLLM leg (D55). flashinfer 0.6.16.post3 is
# source-incompatible with CUDA 13.0 — its sampling kernel calls
# cub::BlockAdjacentDifference::FlagHeads, removed in CUDA 13's CUB — so the
# JIT build fails and the engine never starts. Cannot change any number: the
# protocol is greedy (temperature 0.0), so top-p/top-k sampling is never
# exercised and vLLM's native path takes the identical argmax.
export VLLM_USE_FLASHINFER_SAMPLER=0

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }
free_gb () { df -P /home 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'; }

# ---- 1. TRAIN -------------------------------------------------------------
# config_v7_corrected.yaml is config_v6.yaml with three path lines changed, so
# the recipe stays bit-identical to v1-v6 and the arm needs no caveat.
# ~14,850 rows -> ~2,787 optimizer steps, the same as v6 (v7's 4,983 steps on a
# 26,570-row mixture is a confound this arm deliberately avoids — D57).
if stage_done v7c_train; then echo "$(date '+%F %T') skip v7c_train"; else
  echo "$(date '+%F %T') START v7c_train"
  echo "  config: experiments/finetuning/LLaMA/config_v7_corrected.yaml"
  $PY experiments/finetuning/LLaMA/train.py \
      --config experiments/finetuning/LLaMA/config_v7_corrected.yaml \
      > "$Q/v7c_train.log" 2>&1
  mark v7c_train $?
fi
stage_done v7c_train || { echo "$(date '+%F %T') training failed; stopping"; exit 1; }

# ---- 2. MERGE -------------------------------------------------------------
if stage_done v7c_merge; then echo "$(date '+%F %T') skip v7c_merge"; else
  if [ "$(free_gb)" -lt 10 ]; then
    echo "$(date '+%F %T') ABORT v7c_merge — only $(free_gb) GB free, need ~6 GB + headroom"
    exit 1
  fi
  echo "$(date '+%F %T') START v7c_merge ($(free_gb) GB free)"
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
      --adapter checkpoints/llama_v7_corrected/final --out "$MERGED" \
      > "$Q/v7c_merge.log" 2>&1
  mark v7c_merge $?
fi
stage_done v7c_merge || { echo "$(date '+%F %T') merge failed; stopping"; exit 1; }

# ---- 3. vLLM EVAL ---------------------------------------------------------
# Order changed vs run_schedule_v7.sh: TimeBench (~4 min) -> TIME -> TRAM.
# TIME is this arm's PRIMARY metric (docs/v7_corrected_plan.md), so it lands
# before the ~2 h TRAM leg rather than after it.
vleg () {  # name benchmark
  local name=$1 bench=$2
  stage_done "$name" && { echo "$(date '+%F %T') skip $name"; return; }
  echo "$(date '+%F %T') START $name (vLLM)"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$bench" \
       --results-dir results/corrected/v7_corrected_vllm --model-name llama \
       --adapter-dir checkpoints/llama_v7_corrected/final > "$Q/$name.log" 2>&1
  mark "$name" $?
}
vleg v7c_timebench timebench
vleg v7c_time      time
vleg v7c_tram      tram

# ---- 4. rescore + per-shape format probe ----------------------------------
# THE PRE-REGISTERED CHECK: the `rewritten` column for this arm must be ~0.
# 0 rewrites + a score at or near v6's 42.19% means the trailing-letter scorer
# rule is not load-bearing and the objection is answered. See the criteria
# table in docs/v7_corrected_plan.md — decided before the run, not after.
echo "$(date '+%F %T') START rescore"
$PY scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && { echo "$(date '+%F %T') rescore -> OK"; echo OK > "$Q/rescore.done"; } \
  || { echo "$(date '+%F %T') rescore -> FAIL"; echo FAIL > "$Q/rescore.done"; }
$PY scripts/probe_answer_formats.py --arms v7-corrected v7 v6 zero-shot \
     --out "$Q/format_probe.report" > "$Q/format_probe.log" 2>&1 \
  && echo "$(date '+%F %T') format_probe -> OK" \
  || echo "$(date '+%F %T') format_probe -> FAIL"

# ---- done -----------------------------------------------------------------
# The marker records the ACTUAL outcome. run_schedule_v7.sh wrote "schedule
# complete" unconditionally, which read as success on a pass where nearly
# every leg had FAILED (D55/D56, and the same trap as the STaR pilot in D32).
failed=""
for s in v7c_train v7c_merge v7c_timebench v7c_time v7c_tram rescore; do
  [ "$(cat "$Q/$s.done" 2>/dev/null)" = "OK" ] || failed="$failed $s"
done
if [ -z "$failed" ]; then
  echo "$(date '+%F %T') ALL STAGES OK" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') ALL STAGES DONE — every stage OK."
else
  echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') DONE, BUT THESE STAGES DID NOT SUCCEED:$failed"
fi
