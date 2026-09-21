#!/bin/bash
# ===========================================================================
# ZERO-SHOT LLAMA, vLLM ENGINE — TIME + TimeBench only
#
# Fills a real gap: zero-shot llama exists on HF for TIME/TimeBench
# (results/baseline/zero_shot_v3/) and on vLLM for TRAM
# (results/tram_fixed/zero_shot/, 980,918 rows, already complete -- NOT
# rerun here, that would cost ~2h GPU for no new information). There has
# never been a vLLM-native zero-shot run for TIME or TimeBench.
#
# WHY IT MATTERS: v7, v7c, v6d, v9, v9-glm and any other vLLM-only arm is
# currently compared against the HF zero-shot baseline. That is licensed by
# measured HF/vLLM parity (v6 dual-engine re-run: identical to 0.02pp on both
# benchmarks, item agreement 99.55%/99.58% TIME, 99.25%/99.49% TimeBench --
# docs/results_and_methodology.md §5.1) -- but a direct vLLM-native zero-shot
# column removes the need to invoke that parity argument at all.
#
# No adapter, no merge: --model-dir points straight at the base model.
#
# LAUNCH (the log dir must exist BEFORE the shell opens the redirect):
#   mkdir -p logs/zs_vllm
#   nohup bash scripts/run_zs_vllm_time_timebench.sh \
#         > logs/zs_vllm/driver.log 2>&1 &
#
# Must run from a shell with real GPU access — a Claude Code Bash-tool
# session can neither see the GPU nor signal these processes (D32/D33/D55).
#
# EXPECTED RUNTIME: ~4 min TimeBench + ~25 min TIME (v9's schedule measured
# these orders of magnitude on the same engine/model/hardware).
#
# WHERE RESULTS GO:
#   results/baseline/zero_shot_vllm/llama/<bench>/zero_shot/predictions.jsonl
# Registered in scripts/rescore_v5_protocol.py as arm label "zs-vllm" on both
# benchmarks (TRAM's vLLM zero-shot is already reachable there via
# --tram-root's existing "zero-shot" -> zero_shot/zero_shot mapping, so no
# new TRAM registration is needed).
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/zs_vllm
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
BASE=models/Llama-3.2-3B-Instruct
OUT=results/baseline/zero_shot_vllm

# MANDATORY for every vLLM leg (D55). flashinfer 0.6.16.post3 is
# source-incompatible with CUDA 13.0 -- its sampling kernel calls
# cub::BlockAdjacentDifference::FlagHeads, removed in CUDA 13's CUB -- so the
# JIT build fails and the engine never starts. Cannot change any number: the
# protocol is greedy (temperature 0.0), so top-p/top-k sampling is never
# exercised and vLLM's native path takes the identical argmax.
export VLLM_USE_FLASHINFER_SAMPLER=0

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }

vleg () {  # name benchmark
  local name=$1 bench=$2
  stage_done "$name" && { echo "$(date '+%F %T') skip $name"; return; }
  echo "$(date '+%F %T') START $name (vLLM, zero-shot, no adapter)"
  $VLLM scripts/run_eval_vllm.py --model-dir "$BASE" --benchmark "$bench" \
       --results-dir "$OUT" --model-name llama > "$Q/$name.log" 2>&1
  mark "$name" $?
}

# Cheapest first.
vleg zs_timebench timebench
vleg zs_time      time

# ---- rescore, so the new zs-vllm column is immediately quotable ----------
echo "$(date '+%F %T') START rescore"
$PY scripts/rescore_v5_protocol.py --tram-root results/tram_fixed \
     > "$Q/rescore.log" 2>&1 \
  && { echo "$(date '+%F %T') rescore -> OK"; echo OK > "$Q/rescore.done"; } \
  || { echo "$(date '+%F %T') rescore -> FAIL"; echo FAIL > "$Q/rescore.done"; }

# ---- done ------------------------------------------------------------------
failed=""
for s in zs_timebench zs_time rescore; do
  [ "$(cat "$Q/$s.done" 2>/dev/null)" = "OK" ] || failed="$failed $s"
done
if [ -z "$failed" ]; then
  echo "$(date '+%F %T') ALL STAGES OK" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') ALL STAGES DONE — every stage OK."
else
  echo "$(date '+%F %T') DONE_WITH_FAILURES:$failed" > "$Q/schedule_complete.marker"
  echo "$(date '+%F %T') DONE, BUT THESE STAGES DID NOT SUCCEED:$failed"
fi
