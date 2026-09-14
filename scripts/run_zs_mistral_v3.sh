#!/usr/bin/env bash
# Zero-shot MISTRAL under the frozen protocol, HF engine only.
#   leg 1: timebench (batch 20, token budget 26,624 — D8)
#   leg 2: time      (batch 20, token budget 20,480 — D8)
# Results -> results/baseline/zero_shot_v3/mistral/<bench>/zero_shot/
# (same layout the llama zero-shot arms use, so rescore can pick them up.)
#
# Idempotent: a leg whose .done reads OK is skipped; run_baselines itself
# resumes from its own predictions.jsonl, so a killed leg just continues.
#
# BATCH SETTINGS (user decision, 2026-09-08, per D47 doctrine — a never-run
# arm may pick its batch once, for speed, then freeze it for comparability):
#   PER-LEG settings. timebench: batch 24 / 26,624 (completed 32 min, 15:00
#   2026-09-08, DONE). time: batch 20 / 20,480 — the empirically proven
#   Sept-1 setting (19.5 h clean). A batch-24/26,624 attempt on TIME OOMd
#   at 16:57 (asked 1,008 MiB, 970 free) — do not retry it. This was the
#   measured-stable operating point for short prompts (21.4 GB), RTX 3090
#   23.53 GiB, expandable_segments on). Mistral OOMs at budget 57,344 (D8);
#   do NOT raise past 26,624 without a fresh memory watch. HF engine only —
#   mistral on vLLM is forbidden (2.20pp parity failure, D46).
# The batch-20 first attempt (3k timebench rows) was killed and deleted
# 2026-09-08; its log is kept as mistral_timebench.batch20_aborted.log.
#
# Launched 2026-09-08 by the opencode session at the user's request.
# DO NOT run training or any other GPU job while this chain is active.

set -u
cd "$(dirname "$0")/.."   # repo root
Q=logs/zs_mistral_v3
mkdir -p "$Q"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

done_ok() { [ -f "$Q/$1.done" ] && [ "$(cat "$Q/$1.done")" = "OK" ]; }
mark()    { echo "$2" > "$Q/$1.done"; }

leg() { # name benchmark budget batch
  local name="$1" bench="$2" budget="$3" batch="${4:-20}"
  if done_ok "$name"; then echo "$(date '+%F %T') skip $name"; return 0; fi
  echo "$(date '+%F %T') START $name (bench=$bench budget=$budget batch=$batch)"
  venv/bin/python scripts/run_baselines.py \
    --model mistral --benchmark "$bench" \
    --results-dir ./results/baseline/zero_shot_v3 \
    --batch-size "$batch" --token-budget "$budget" \
    > "$Q/$name.log" 2>&1
  local rc=$?
  mark "$name" "$([ $rc -eq 0 ] && echo OK || echo FAIL:$rc)"
  echo "$(date '+%F %T') END $name rc=$rc"
  return $rc
}

leg mistral_timebench timebench 26624 24 || { echo "timebench failed; stopping"; exit 1; }
leg mistral_time      time      20480 20 || { echo "time failed; stopping";     exit 1; }

echo "$(date '+%F %T') ALL LEGS COMPLETE" | tee "$Q/complete.marker"
