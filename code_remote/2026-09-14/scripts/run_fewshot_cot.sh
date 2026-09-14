#!/usr/bin/env bash
# Few-shot CoT baselines (k=4 worked traces in-prompt), HF engine only.
#
# ARM: llama and mistral, timebench and time, --prompt-style cot_fewshot
#      (data/cot_fewshot_exemplars_mined.json -- SELF-MINED from the v6
#      training pool, D63: leakage-free by construction, verified-correct),
#      --max-new-tokens 256, results into results/baseline/few_shot_cot/.
#      Same prompt family, persona, batching and budgets as the
#      zero_shot_reasoning arm (formerly zero_shot_cot) -- the ONLY delta
#      is the visible exemplars, so the fewshot-vs-zs-R comparison is clean.
#
# HYPOTHESIS (D59/D61): the zs-R failure modes were 256-token truncation
#      before ANSWER and deliberation misordering. SHORT exemplar traces
#      (2 steps, always terminated) teach exactly that discipline.
#
# BATCH RULES: identical to run_zs_cot.sh (D55 ceilings). The token-budget
#      packer self-shrinks batches for the +~700 exemplar tokens; mistral
#      time packs ~6/batch and is the long pole (~3 days). It is the LAST
#      leg -- killing it early for the GPU is acceptable if needed.
#
# WAITS for the zero_shot_reasoning chain (logs/zs_cot/complete.marker).
# Idempotent per leg. Launched 2026-09-11.

set -u
cd "$(dirname "$0")/.."
Q=logs/fewshot_cot
mkdir -p "$Q"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE=logs/zs_cot/complete.marker
if [ ! -f "$GATE" ]; then
  echo "$(date '+%F %T') waiting for $GATE (polling every 5 min, no GPU)"
  while [ ! -f "$GATE" ]; do sleep 300; done
fi
echo "$(date '+%F %T') gate passed: zero_shot_reasoning chain complete"

# Step 0: self-mine the exemplars (D63) -- llama zs-R generations on
# TRAINING-POOL questions, answer-verified, shortest-trace-first. Minutes
# on GPU. The handwritten exemplars are retired (researcher-authorship
# confound); data/cot_fewshot_exemplars.json stays for reference only.
MINED=data/cot_fewshot_exemplars_mined.json
if [ ! -f "$MINED" ]; then
  echo "$(date '+%F %T') mining exemplars from the v6 training pool..."
  venv/bin/python scripts/mine_cot_exemplars.py > "$Q/mine.log" 2>&1 || {
    echo "exemplar mining FAILED (see $Q/mine.log)"; exit 1; }
fi

done_ok() { [ -f "$Q/$1.done" ] && [ "$(cat "$Q/$1.done")" = "OK" ]; }
mark()    { echo "$2" > "$Q/$1.done"; }

leg() { # name model benchmark budget batch
  local name="$1" model="$2" bench="$3" budget="$4" batch="${5:-20}"
  if done_ok "$name"; then echo "$(date '+%F %T') skip $name"; return 0; fi
  echo "$(date '+%F %T') START $name (model=$model bench=$bench budget=$budget batch=$batch)"
  venv/bin/python scripts/run_baselines.py \
    --model "$model" --benchmark "$bench" \
    --results-dir ./results/baseline/few_shot_cot \
    --prompt-style cot_fewshot --max-new-tokens 256 \
    --cot-exemplars data/cot_fewshot_exemplars_mined.json \
    --batch-size "$batch" --token-budget "$budget" \
    > "$Q/$name.log" 2>&1
  local rc=$?
  mark "$name" "$([ $rc -eq 0 ] && echo OK || echo FAIL:$rc)"
  echo "$(date '+%F %T') END $name rc=$rc"
  return $rc
}

# Cheapest first (D47).
leg llama_timebench    llama   timebench 57344 32 || { echo "llama tb failed; stopping"; exit 1; }
leg mistral_timebench  mistral timebench 26624 24 || { echo "mistral tb failed; stopping"; exit 1; }
leg llama_time         llama   time      57344 32 || { echo "llama time failed; stopping"; exit 1; }
leg mistral_time       mistral time      20480 20 || { echo "mistral time failed; stopping"; exit 1; }

echo "$(date '+%F %T') ALL FEWSHOT LEGS COMPLETE" | tee "$Q/complete.marker"
