#!/bin/bash
# ===========================================================================
# TRAM for the classic arms — v1, v2, v3-corrected, v4, v5 (2026-09-11)
#
# Fills the last gap in the TRAM column: only zs/v6/v7/v7c/v6d have TRAM
# numbers. This runs the five HF-era arms ON vLLM so the whole TRAM column
# is engine-consistent (every TRAM number in the project is vLLM — D48/D53).
#
# Per arm: merge LoRA into the base (~4 min CPU) -> TRAM leg (~2 h GPU,
# 980,918 items) -> DELETE the merged dir after the leg is OK (disk is at
# ~40 G free; the merge is regenerable from <adapter>/final in minutes).
#
# SKIPPED, deliberately:
#   - v3 (checkpoints/llama_v3) — BROKEN checkpoint (corrupted training
#     data; superseded by llama_v3_fixed). No scientific value, 2 h saved.
#   - mistral (lora_mistral) — banned from vLLM pending the prompt-bug
#     investigation (D53/D46).
#
# Consistency review done before launch (D61): the training loader has
# always applied TEMPORAL_SYSTEM_PROMPT (include_system=True default), so
# --adapter-dir's prompt selection matches what v1-v5 trained with; greedy
# protocol; dataset-order chunks of 1000 like every prior TRAM leg;
# per-arm results dirs (results/corrected/<arm>_vllm) so nothing resumes
# onto another arm's predictions; provenance tags written per record.
#
# LAUNCH (detached, survives shell exit):
#   mkdir -p logs/sched_tram_classics
#   setsid nohup bash scripts/run_tram_classics.sh \
#         > logs/sched_tram_classics/driver.log 2>&1 < /dev/null &
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_tram_classics
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python

# MANDATORY for every vLLM leg (D55): flashinfer 0.6.16.post3 is
# source-incompatible with CUDA 13.0 and the engine dies ~30 s in without
# this. Cannot change any number — the protocol is greedy (temp 0.0).
export VLLM_USE_FLASHINFER_SAMPLER=0

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }
free_gb () { df -P /home 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'; }

# label|adapter|merged_dir|results_dir
ARMS=(
  "v1_tram|checkpoints/llama/final|checkpoints/llama/merged_fp16|results/corrected/v1_vllm"
  "v2_tram|checkpoints/llama_v2/final|checkpoints/llama_v2/merged_fp16|results/corrected/v2_vllm"
  "v3c_tram|checkpoints/llama_v3_fixed/final|checkpoints/llama_v3_fixed/merged_fp16|results/corrected/v3c_vllm"
  "v4_tram|checkpoints/llama_v4/final|checkpoints/llama_v4/merged_fp16|results/corrected/v4_vllm"
  "v5_tram|checkpoints/llama_v5/final|checkpoints/llama_v5/merged_fp16|results/corrected/v5_vllm"
)

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name adapter merged results_dir <<< "$spec"

  if stage_done "$name"; then echo "$(date '+%F %T') skip $name"; continue; fi

  # ---- merge (idempotent; requires shards, not just config.json) --------
  merge_name="${name%%_tram}_merge"
  if stage_done "$merge_name"; then
    echo "$(date '+%F %T') skip $merge_name"
  else
    if [ "$(free_gb)" -lt 12 ]; then
      echo "$(date '+%F %T') ABORT $merge_name — only $(free_gb) GB free, need ~6 GB + headroom"
      exit 1
    fi
    echo "$(date '+%F %T') START $merge_name ($(free_gb) GB free)"
    $PY scripts/merge_lora.py --base models/Llama-3.2-3B-Instruct \
        --adapter "$adapter" --out "$merged" > "$Q/$merge_name.log" 2>&1
    mark "$merge_name" $?
  fi
  stage_done "$merge_name" || { echo "$(date '+%F %T') merge failed for $name; stopping"; exit 1; }

  # ---- TRAM leg ----------------------------------------------------------
  echo "$(date '+%F %T') START $name (vLLM, 980,918 items)"
  $VLLM scripts/run_eval_vllm.py --model-dir "$merged" --benchmark tram \
       --results-dir "$results_dir" --model-name llama \
       --adapter-dir "$adapter" > "$Q/$name.log" 2>&1
  mark "$name" $?

  # ---- reclaim the 6 GB merged model once the leg is OK ------------------
  # Regenerable any time: scripts/merge_lora.py --adapter <adapter> --out
  # <merged> (~4 min CPU). The final adapter is NEVER touched.
  if stage_done "$name" && [ -d "$merged" ]; then
    echo "$(date '+%F %T') reclaiming $merged ($(free_gb) GB free before)"
    rm -rf "$merged"
  fi
done

# ---- rescore: the canonical table grows the five TRAM rows ---------------
echo "$(date '+%F %T') START rescore"
$PY scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && { echo "$(date '+%F %T') rescore -> OK"; echo OK > "$Q/rescore.done"; } \
  || { echo "$(date '+%F %T') rescore -> FAIL"; echo FAIL > "$Q/rescore.done"; }

failed=""
for s in v1_tram v2_tram v3c_tram v4_tram v5_tram rescore; do
  [ "$(cat "$Q/$s.done" 2>/dev/null)" = "OK" ] || failed="$failed $s"
done
if [ -z "$failed" ]; then
  echo "$(date '+%F %T') ALL STAGES OK" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') ALL STAGES DONE — every stage OK."
else
  echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') DONE, BUT THESE STAGES DID NOT SUCCEED:$failed"
fi
