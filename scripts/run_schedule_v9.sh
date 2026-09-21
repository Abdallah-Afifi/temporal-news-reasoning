#!/bin/bash
# ===========================================================================
# v9 SCHEDULE — train -> merge -> vLLM eval (TimeBench, TIME, TRAM) -> rescore
#
# New file per the project convention (D58); run_schedule_v7_corrected.sh is
# left untouched so v7c stays reproducible. This is that script adapted.
#
# THE ARM. v6's mixture with the spent synthetic slices retired and replaced by
# AUG_GLM2 (~5,990 rows, 39.9% of the 12,002-row train split), capped at the
# L1 total of 15,000. Hyperparameters are byte-identical to config_v6.yaml
# outside the three path lines, so the arm stays comparable to v1-v7c.
# Control = v6.
#
# ---------------------------------------------------------------------------
# READ BEFORE QUOTING ANY NUMBER THIS PRODUCES (audit 2026-09-16)
#
# data/manual_aug_v9/AUDIT.md reports OVERALL: FAIL. The blocking gate is §9
# answer style: AUG_GLM2's free-text golds average ~7.9 words against TIME's
# 2.35, and 14.1% run to 20+ words against TIME's 0.95%. Because AUG_GLM2 is
# ~40% of the mixture, this arm is trained toward an answer shape the frozen
# exact-match scorer does not reward.
#
# So a LOW score on TIME does NOT establish that the synthetic data is bad —
# answer-style mismatch and data quality are confounded in this run. A HIGH
# score is still informative (it would clear both at once). The clean
# disambiguation, if the result comes back low, is scripts/probe_answer_formats
# .py: if v9's answers are long where zero-shot's are terse, the style
# explanation is live. That probe is run automatically in stage 4.
#
# The run proceeds at the researcher's explicit direction with the gate red.
# ---------------------------------------------------------------------------
#
# LAUNCH (the log dir must exist BEFORE the shell opens the redirect):
#   mkdir -p logs/sched_v9
#   nohup bash scripts/run_schedule_v9.sh > logs/sched_v9/driver.log 2>&1 &
#
# Must run from a shell with real GPU access — a Claude Code Bash-tool session
# can neither see the GPU nor signal these processes (D32/D33/D55).
#
# DISK. A merged fp16 3B is ~6 GB and is NOT deleted automatically — remove
# checkpoints/llama_v9/merged_fp16 once the vLLM legs are done.
#
# WHY A MERGE IS NEEDED. run_eval_vllm.py loads whatever is at --model-dir;
# its --adapter-dir only selects the training system prompt, it does NOT load
# LoRA weights into vLLM. The adapter must be merged into the base first.
#
# WHERE RESULTS GO, and why the two roots differ:
#   TIME / TimeBench -> results/corrected/v9_vllm/   (the ARMS["time"] path)
#   TRAM             -> results/tram_fixed/v9/       (the --tram-root path)
# rescore_v5_protocol.py REPLACES its TRAM arm list when --tram-root is given,
# so a TRAM leg written under results/corrected/ would be silently dropped.
# Both paths are registered in that script.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v9
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
MERGED=checkpoints/llama_v9/merged_fp16

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

# ---- 0. DATA PRECONDITION -------------------------------------------------
# The mixture must have been rebuilt from the REGENERATED augmentation set.
# The first v9 build (2026-09-14) used aug data with 426 repetition-padded
# contexts and 128 truncated golds; those files were replaced on 2026-09-16.
if [ ! -s data/combined_80_20_v9/train.jsonl ]; then
  echo "$(date '+%F %T') ABORT — data/combined_80_20_v9/train.jsonl missing."
  echo "  run: $PY scripts/build_v9_training_data.py"
  exit 1
fi
if [ -f data/combined_80_20_v9/STALE_DO_NOT_TRAIN.md ]; then
  echo "$(date '+%F %T') ABORT — the mixture is marked stale; rebuild it first."
  exit 1
fi
echo "$(date '+%F %T') data OK: $(wc -l < data/combined_80_20_v9/train.jsonl) train / $(wc -l < data/combined_80_20_v9/val.jsonl) val rows"

# ---- 1. TRAIN -------------------------------------------------------------
# 12,002 rows -> ~2,250 optimizer steps, comparable to v6's 2,784 on 14,850.
if stage_done v9_train; then echo "$(date '+%F %T') skip v9_train"; else
  echo "$(date '+%F %T') START v9_train"
  echo "  config: experiments/finetuning/LLaMA/config_v9.yaml"
  $PY experiments/finetuning/LLaMA/train.py \
      --config experiments/finetuning/LLaMA/config_v9.yaml \
      > "$Q/v9_train.log" 2>&1
  mark v9_train $?
fi
stage_done v9_train || { echo "$(date '+%F %T') training failed; stopping"; exit 1; }

# ---- 2. MERGE -------------------------------------------------------------
if stage_done v9_merge; then echo "$(date '+%F %T') skip v9_merge"; else
  if [ "$(free_gb)" -lt 10 ]; then
    echo "$(date '+%F %T') ABORT v9_merge — only $(free_gb) GB free, need ~6 GB + headroom"
    exit 1
  fi
  echo "$(date '+%F %T') START v9_merge ($(free_gb) GB free)"
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
      --adapter checkpoints/llama_v9/final --out "$MERGED" \
      > "$Q/v9_merge.log" 2>&1
  mark v9_merge $?
fi
stage_done v9_merge || { echo "$(date '+%F %T') merge failed; stopping"; exit 1; }

# ---- 3. vLLM EVAL ---------------------------------------------------------
# Cheapest first: TimeBench (~4 min) -> TIME (~25 min) -> TRAM (~2 h).
vleg () {  # name benchmark results-dir
  local name=$1 bench=$2 out=$3
  stage_done "$name" && { echo "$(date '+%F %T') skip $name"; return; }
  echo "$(date '+%F %T') START $name (vLLM) -> $out"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$bench" \
       --results-dir "$out" --model-name llama \
       --adapter-dir checkpoints/llama_v9/final > "$Q/$name.log" 2>&1
  mark "$name" $?
}
vleg v9_timebench timebench results/corrected/v9_vllm
vleg v9_time      time      results/corrected/v9_vllm
vleg v9_tram      tram      results/tram_fixed/v9

# ---- 4. rescore + answer-format probe -------------------------------------
# --tram-root IS REQUIRED for a quotable TRAM column: without it the scorer
# reads the pre-2026-09-12 predictions, whose prompts omitted the NLI
# hypothesis and the storytelling passage for 85.3% of items, and prints a
# STALE banner.
echo "$(date '+%F %T') START rescore"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
     > "$Q/rescore.log" 2>&1 \
  && { echo "$(date '+%F %T') rescore -> OK"; echo OK > "$Q/rescore.done"; } \
  || { echo "$(date '+%F %T') rescore -> FAIL"; echo FAIL > "$Q/rescore.done"; }

# THE DISAMBIGUATION named in the header. If v9 scores low on TIME, this says
# whether it is answer style or reasoning. Pre-registered, not chosen after.
$PY scripts/probe_answer_formats.py --arms v9 v6 v7-corrected zero-shot \
     --out "$Q/format_probe.report" > "$Q/format_probe.log" 2>&1 \
  && echo "$(date '+%F %T') format_probe -> OK" \
  || echo "$(date '+%F %T') format_probe -> FAIL"

# ---- done -----------------------------------------------------------------
# The marker records the ACTUAL outcome; an unconditional "complete" line once
# read as success on a pass where nearly every leg had FAILED (D55/D56).
failed=""
for s in v9_train v9_merge v9_timebench v9_time v9_tram rescore; do
  [ "$(cat "$Q/$s.done" 2>/dev/null)" = "OK" ] || failed="$failed $s"
done
if [ -z "$failed" ]; then
  echo "$(date '+%F %T') ALL STAGES OK" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') ALL STAGES DONE — every stage OK."
else
  echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') DONE, BUT THESE STAGES DID NOT SUCCEED:$failed"
fi
