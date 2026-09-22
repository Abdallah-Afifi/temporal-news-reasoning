#!/bin/bash
# ===========================================================================
# v10-glm SCHEDULE — train -> merge -> vLLM eval (TimeBench, TIME, TRAM) -> rescore
#
# New file per the project convention (D58); run_schedule_v9_glm.sh is left
# untouched so v9-glm stays reproducible.
#
# THE ARM, and why it is a KNOWING three-way confound (researcher's explicit
# choice, 2026-09-22 — see config_v10_glm.yaml for the full rationale):
#   (a) AUG_GLM2 doubles 3,000 -> 6,000 rows (the ruleset's original plan).
#   (b) The `news` shape gains a near-miss-passage mechanism, motivated by
#       v9-glm's own result: +3.81pp on TIME's gold-context items, -2.1 to
#       -2.9pp on retriever-supplied ones (59.6% of TIME) -- the old
#       cleanly-absent-passage mechanism never taught distinguishing the
#       right passage from a plausible wrong one.
#   (c) lora_r 16 -> 32, lora_alpha 32 -> 64. NOT the HPO-searched
#       learning_rate (left untouched) -- lora_r was held fixed at 16
#       through that whole search and every arm since, so it is the one
#       lever with no existing evidence behind its value.
# If v10-glm's result differs from v9-glm's (TIME 40.79%, TimeBench 45.05%,
# TRAM 44.45% -- all vLLM, results/rescored/v5_protocol.json), THIS ARM ALONE
# CANNOT SAY WHICH OF (a)/(b)/(c) CAUSED IT. A follow-up ablation would be
# needed before attributing the change to one cause. Control = v9-glm.
#
# ---------------------------------------------------------------------------
# DATA QUALITY (round 1 verified 2026-09-21/22, data/manual_aug_glm/AUDIT.md;
# round 2 must be re-verified the same way before this launches)
#
# Round 1 (3,000 rows): 0 contamination, 0 shingle overlap, all shortcut
# probes pass, 0 padding, 0 truncation, free-text answer style mean 3.12
# words / 0.0% >=20w (TIME itself: 2.35 / 0.95%). AUDIT.md's OVERALL line
# reads FAIL on two gates confirmed to be tool-convention mismatches, not
# data defects (boilerplate-heavy near-duplicate detection; a recomputer
# built for the template generator's conventions) -- documented in
# config_v9_glm.yaml, do not requote the bare FAIL without that context.
# Known limitation, not fully corrected: Block A provenance skews toward
# news vs the ruleset's 60/35/5 target.
# ---------------------------------------------------------------------------
#
# LAUNCH (the log dir must exist BEFORE the shell opens the redirect):
#   mkdir -p logs/sched_v10_glm
#   nohup bash scripts/run_schedule_v10_glm.sh > logs/sched_v10_glm/driver.log 2>&1 &
#
# Must run from a shell with real GPU access — a Claude Code Bash-tool session
# can neither see the GPU nor signal these processes (D32/D33/D55).
#
# DISK. A merged fp16 3B is ~6 GB and is NOT deleted automatically — remove
# checkpoints/llama_v10_glm/merged_fp16 once the vLLM legs are done.
#
# WHY A MERGE IS NEEDED. run_eval_vllm.py loads whatever is at --model-dir;
# its --adapter-dir only selects the training system prompt, it does NOT load
# LoRA weights into vLLM. The adapter must be merged into the base first.
#
# WHERE RESULTS GO, and why the two roots differ:
#   TIME / TimeBench -> results/corrected/v10_glm_vllm/   (the ARMS["time"] path)
#   TRAM             -> results/tram_fixed/v10_glm/       (the --tram-root path)
# rescore_v5_protocol.py REPLACES its TRAM arm list when --tram-root is given,
# so a TRAM leg written under results/corrected/ would be silently dropped.
# Both paths are registered in that script.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v10_glm
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
MERGED=checkpoints/llama_v10_glm/merged_fp16

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
if [ ! -s data/combined_80_20_v10_glm/train.jsonl ]; then
  echo "$(date '+%F %T') ABORT — data/combined_80_20_v10_glm/train.jsonl missing."
  echo "  run: $PY scripts/build_v9_training_data.py"
  exit 1
fi
if [ -f data/combined_80_20_v10_glm/STALE_DO_NOT_TRAIN.md ]; then
  echo "$(date '+%F %T') ABORT — the mixture is marked stale; rebuild it first."
  exit 1
fi
echo "$(date '+%F %T') data OK: $(wc -l < data/combined_80_20_v10_glm/train.jsonl) train / $(wc -l < data/combined_80_20_v10_glm/val.jsonl) val rows"

# ---- 1. TRAIN -------------------------------------------------------------
# 12,002 rows -> ~2,250 optimizer steps, comparable to v6's 2,784 on 14,850.
if stage_done v10_train; then echo "$(date '+%F %T') skip v10_train"; else
  echo "$(date '+%F %T') START v10_train"
  echo "  config: experiments/finetuning/LLaMA/config_v10_glm.yaml"
  $PY experiments/finetuning/LLaMA/train.py \
      --config experiments/finetuning/LLaMA/config_v10_glm.yaml \
      > "$Q/v10_train.log" 2>&1
  mark v10_train $?
fi
stage_done v10_train || { echo "$(date '+%F %T') training failed; stopping"; exit 1; }

# ---- 2. MERGE -------------------------------------------------------------
if stage_done v10_merge; then echo "$(date '+%F %T') skip v10_merge"; else
  if [ "$(free_gb)" -lt 10 ]; then
    echo "$(date '+%F %T') ABORT v10_merge — only $(free_gb) GB free, need ~6 GB + headroom"
    exit 1
  fi
  echo "$(date '+%F %T') START v10_merge ($(free_gb) GB free)"
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
      --adapter checkpoints/llama_v10_glm/final --out "$MERGED" \
      > "$Q/v10_merge.log" 2>&1
  mark v10_merge $?
fi
stage_done v10_merge || { echo "$(date '+%F %T') merge failed; stopping"; exit 1; }

# ---- 3. vLLM EVAL ---------------------------------------------------------
# Cheapest first: TimeBench (~4 min) -> TIME (~3h33m) -> TRAM (~2h6m). These
# are v9-glm's OWN measured times on this exact pipeline (logs/sched_v9_glm/
# driver.log, 2026-09-22), not the "~25 min TIME" figure that turned out to
# have no log evidence anywhere in the project when checked -- see
# scripts/run_zs_vllm_time_timebench.sh for that correction. lora_r=32 vs
# v9-glm's 16 may shift these slightly (larger adapter), but not by an order
# of magnitude.
vleg () {  # name benchmark results-dir
  local name=$1 bench=$2 out=$3
  stage_done "$name" && { echo "$(date '+%F %T') skip $name"; return; }
  echo "$(date '+%F %T') START $name (vLLM) -> $out"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$bench" \
       --results-dir "$out" --model-name llama \
       --adapter-dir checkpoints/llama_v10_glm/final > "$Q/$name.log" 2>&1
  mark "$name" $?
}
vleg v10_timebench timebench results/corrected/v10_glm_vllm
vleg v10_time      time      results/corrected/v10_glm_vllm
vleg v10_tram      tram      results/tram_fixed/v10_glm

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
for s in v10_train v10_merge v10_timebench v10_time v10_tram rescore; do
  [ "$(cat "$Q/$s.done" 2>/dev/null)" = "OK" ] || failed="$failed $s"
done
if [ -z "$failed" ]; then
  echo "$(date '+%F %T') ALL STAGES OK" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') ALL STAGES DONE — every stage OK."
else
  echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') DONE, BUT THESE STAGES DID NOT SUCCEED:$failed"
fi
