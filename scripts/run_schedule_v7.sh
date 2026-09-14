#!/usr/bin/env bash
# Full v7 schedule, in the order the user asked for (2026-09-08):
#   1. v7 training
#   2. v7 evaluation ON vLLM   (TimeBench -> TRAM -> TIME, cheapest first)
#   3. Mistral 2,000-sample vLLM parity re-test
#   4. zero-shot TRAM (llama)
#   5. rescore + format probe
#
# Launch from a shell with real GPU access, NOT a Claude Code Bash session:
#   mkdir -p logs/sched_v7
#   nohup bash scripts/run_schedule_v7.sh > logs/sched_v7/driver.log 2>&1 &
#
# Every stage is idempotent: a stage whose .done reads OK is skipped, and the
# eval legs resume from their own predictions.jsonl.
#
# ============================ READ BEFORE RUNNING ==========================
#
# ENGINE CONSISTENCY. v1-v6 were evaluated on HF. Evaluating v7 on vLLM makes
# the results table MIXED-ENGINE. That is permitted for llama (parity is
# measured at 0.05pp: 94.40% agreement, and full TimeBench 45.64 vLLM vs 45.65
# HF) but every thesis table cell must then record `engine: vllm 0.28.0`.
# Stage 3b re-runs v6 on vLLM so at least the v6-vs-v7 comparison — the one
# this cycle actually turns on — is engine-consistent. Skip it only if you
# accept comparing across engines.
#
# DISK. /home is at 99% (~15 G free). A merged fp16 3B is ~6 G. Stage 2 checks
# for headroom and aborts rather than filling the disk. The merged model is
# NOT deleted automatically — remove checkpoints/llama_v7/merged_fp16 when the
# vLLM legs are done if you need the space back.
#
# WHY A MERGE IS NEEDED. run_eval_vllm.py loads whatever is at --model-dir;
# its --adapter-dir only selects the training system prompt, it does NOT load
# LoRA weights into vLLM. So the adapter must be merged into the base first.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_v7
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
MERGED=checkpoints/llama_v7/merged_fp16

# MANDATORY for every vLLM leg (added 2026-09-08 after the whole vLLM stage
# failed seven times across four scheduler passes).
#
# flashinfer 0.6.16.post3 is SOURCE-INCOMPATIBLE with the installed CUDA 13.0:
# its sampling kernel calls cub::BlockAdjacentDifference::FlagHeads, which no
# longer exists in CUDA 13's CUB, so the JIT build fails ->
#   sampling.cuh(623): error: class "cub::_V_300302_SM_860::
#     BlockAdjacentDifference<__nv_bool, 1024, 1, 1>" has no member "FlagHeads"
#   -> RuntimeError: Engine core initialization failed
# Every vLLM leg died ~30 s in with that. It is not a disk, memory or model
# problem, and it is not fixable by config — flashinfer would have to be
# downgraded to a CUDA-13-compatible build.
#
# Disabling flashinfer's sampler CANNOT change any number here: the whole
# protocol is greedy (temperature 0.0), so top-p/top-k sampling is never
# exercised — vLLM's native PyTorch path takes the identical argmax. Verified
# empirically: the one leg that passed (v7_timebench, 20:50) is exactly the
# one that ran with this variable set, and it logged
#   "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0."
# It was set ad hoc in that shell; exporting it here makes it apply to every
# leg on every future run instead of depending on the launching environment.
export VLLM_USE_FLASHINFER_SAMPLER=0

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }
free_gb () { df -P /home 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'; }

# ---- 1. TRAIN ------------------------------------------------------------
if stage_done v7_train; then echo "$(date '+%F %T') skip v7_train"; else
  echo "$(date '+%F %T') START v7_train"
  # TRAINING CANNOT USE vLLM — vLLM is an inference engine with no training
  # path. Any training speedup has to come from the config.
  #
  # DEFAULT IS config_v7.yaml (user decision, 2026-09-08). It is the MEASURED,
  # known quantity: 8.82 s/step on a 60-step pilot = 12.2 h for 4,980 steps,
  # and it is bit-recipe-identical to v1-v6, so v7 needs no methodological
  # caveat.
  #
  # What the alternatives turned out to be worth:
  #   gradient_checkpointing: false — NOT AVAILABLE. OOMs on this 24 GB card
  #     (22.44 of 23.53 GiB used). v4 OOMed the same way in 2026-09-05, so
  #     that is twice on two different mixtures. Do not retry.
  #   group_by_length: true — available (config_v7_fast.yaml), padding waste
  #     40.6% -> 0.5%, but UNMEASURED: HF's LengthGroupedSampler runs the
  #     longest batches FIRST, so a 60-step pilot samples only the slowest
  #     region and cannot price it. It also changes which examples share an
  #     optimizer step (like a reseed), so it is not bit-comparable to v1-v6.
  #
  # To use it anyway:
  #   TRAIN_CFG=experiments/finetuning/LLaMA/config_v7_fast.yaml bash scripts/run_schedule_v7.sh
  TRAIN_CFG="${TRAIN_CFG:-experiments/finetuning/LLaMA/config_v7.yaml}"
  echo "  config: $TRAIN_CFG"
  $PY experiments/finetuning/LLaMA/train.py \
      --config "$TRAIN_CFG" > "$Q/v7_train.log" 2>&1
  mark v7_train $?
fi
stage_done v7_train || { echo "training failed; stopping"; exit 1; }

# ---- 2. MERGE (required for vLLM) ----------------------------------------
if stage_done v7_merge; then echo "$(date '+%F %T') skip v7_merge"; else
  if [ "$(free_gb)" -lt 10 ]; then
    echo "$(date '+%F %T') ABORT: only $(free_gb) GB free; a merged fp16 3B needs ~6 GB."
    echo "  Free space, then re-run. Nothing has been deleted."
    exit 1
  fi
  echo "$(date '+%F %T') START v7_merge ($(free_gb) GB free)"
  $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
      --adapter checkpoints/llama_v7/final --out "$MERGED" > "$Q/v7_merge.log" 2>&1
  mark v7_merge $?
fi
stage_done v7_merge || { echo "merge failed; stopping"; exit 1; }

# ---- 3. vLLM EVAL: TimeBench -> TRAM -> TIME (cheapest first) ------------
vleg () {  # name benchmark
  local name=$1 bench=$2
  stage_done "$name" && { echo "$(date '+%F %T') skip $name"; return; }
  echo "$(date '+%F %T') START $name (vLLM)"
  $VLLM scripts/run_eval_vllm.py --model-dir "$MERGED" --benchmark "$bench" \
       --results-dir results/corrected/v7_vllm --model-name llama \
       --adapter-dir checkpoints/llama_v7/final > "$Q/$name.log" 2>&1
  mark "$name" $?
}
vleg v7_timebench timebench
vleg v7_tram      tram
vleg v7_time      time

# ---- 3b. v6 on vLLM, so v6-vs-v7 is engine-consistent --------------------
# Comment this block out if you accept a mixed-engine comparison.
if stage_done v6_merge; then echo "$(date '+%F %T') skip v6_merge"; else
  if [ "$(free_gb)" -ge 10 ]; then
    echo "$(date '+%F %T') START v6_merge"
    $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
        --adapter checkpoints/llama_v6/final --out checkpoints/llama_v6/merged_fp16 \
        > "$Q/v6_merge.log" 2>&1
    mark v6_merge $?
  else
    echo "$(date '+%F %T') SKIP v6_merge — only $(free_gb) GB free"
  fi
fi
if stage_done v6_merge; then
  for b in timebench time; do
    n="v6_${b}_vllm"
    stage_done "$n" && { echo "$(date '+%F %T') skip $n"; continue; }
    echo "$(date '+%F %T') START $n (vLLM)"
    $VLLM scripts/run_eval_vllm.py --model-dir checkpoints/llama_v6/merged_fp16 \
         --benchmark "$b" --results-dir results/corrected/v6_vllm --model-name llama \
         --adapter-dir checkpoints/llama_v6/final > "$Q/$n.log" 2>&1
    mark "$n" $?
  done
fi

# ---- 4. Mistral 2,000-sample vLLM parity re-test -------------------------
# The 2.20pp failure that banned Mistral from vLLM has the signature of a
# PROMPT BUG, not engine noise: 44.9% disagreement with substantively
# different answers and vLLM outputs 57% longer, versus llama's 5.6% with
# matched lengths. Suspects: chat-template application and double-BOS (a
# defect this project has shipped before). This re-runs the 2,000-item test.
# If it still fails, diff the token IDs each path feeds for one identical item
# before concluding anything about the engine.
if stage_done mistral_parity; then echo "$(date '+%F %T') skip mistral_parity"; else
  echo "$(date '+%F %T') START mistral_parity (2,000 items)"
  $VLLM scripts/vllm_parity_test.py \
       --model-dir models/Mistral-7B-Instruct-v0.3 --benchmark time --n 2000 \
       --reference results/parity_vllm/mistral_time_parity.jsonl \
       --out results/parity_vllm/mistral_time_parity_2026-09-08.jsonl \
       > "$Q/mistral_parity.log" 2>&1
  mark mistral_parity $?
  $PY scripts/compare_parity.py \
       results/parity_vllm/mistral_time_parity_2026-09-08.jsonl \
       > "$Q/mistral_parity.report" 2>&1 || true
fi

# ---- 5. zero-shot TRAM (llama) ON vLLM — the MISSING baseline ------------
# Without this, v7's TRAM number has nothing to beat (D48).
#
# ON vLLM, and that is the CONSISTENT choice rather than a compromise: TRAM
# has never been evaluated by anyone here, so there is no HF TRAM precedent to
# match. Every TRAM arm in this project — zero-shot, v6, v7 — will be vLLM, so
# TRAM is internally engine-consistent from its first run. (Engine consistency
# is required WITHIN a benchmark, which is what makes arms comparable; TIME
# and TimeBench keep their own engines.)
#
# No merge step: zero-shot uses the BASE model, so there is no adapter to
# merge. Passing no --adapter-dir also correctly omits the training system
# prompt, matching how the other zero-shot arms were built.
if stage_done zs_tram; then echo "$(date '+%F %T') skip zs_tram"; else
  echo "$(date '+%F %T') START zs_tram on vLLM (980,918 items)"
  $VLLM scripts/run_eval_vllm.py --model-dir models/Llama-3.2-3B-Instruct \
       --benchmark tram --results-dir results/baseline/zero_shot_v3 \
       --model-name llama > "$Q/zs_tram.log" 2>&1
  mark zs_tram $?
fi

# ---- 6. rescore + per-shape format probe --------------------------------
echo "$(date '+%F %T') START rescore"
$PY scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && echo "$(date '+%F %T') rescore -> OK" || echo "$(date '+%F %T') rescore -> FAIL"
$PY scripts/probe_answer_formats.py --arms v7 v6 v5 zero-shot \
     --out "$Q/format_probe.report" > "$Q/format_probe.log" 2>&1 \
  && echo "$(date '+%F %T') format_probe -> OK" || echo "$(date '+%F %T') format_probe -> FAIL"

echo "$(date '+%F %T') schedule complete" > "$Q/schedule_complete.marker"
echo "$(date '+%F %T') ALL STAGES DONE. TRAM random baseline = 18.92%; TIME = 13.56%."
