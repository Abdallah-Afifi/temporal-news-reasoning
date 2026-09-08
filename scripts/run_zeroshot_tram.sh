#!/usr/bin/env bash
# Zero-shot LLaMA on TRAM — the baseline that does not exist yet.
#
# WHY THIS IS A SEPARATE SCRIPT: `results/baseline/zero_shot_v3/llama/` holds
# only `time` and `timebench`. TRAM has NEVER been run for llama under the
# corrected harness (the tram results on disk are mistral/qwen under the OLD
# broken harness, in results/baseline/zero_shot/). Without this baseline a v7
# TRAM number has nothing to be compared against.
#
# The zero-shot arm is adapter-independent, so this does not need to wait for
# v7 — run it in any idle GPU window. ~3 h at batch 256.
#
# BATCH 256 is deliberate and must match the fine-tuned TRAM legs exactly
# (D47): TRAM prompts are p50 44 / p90 94 / p99 211 tokens, so batch 32 would
# leave the GPU idle and cost ~23 h instead of ~3 h. TRAM has never been run,
# so no prior number constrains the choice — but once chosen it is FROZEN for
# every arm, or the arms are not comparable.
#
# PREREQUISITE, already done 2026-09-07: the 31,626 duplicate TRAM ids were
# fixed in src/data/data_loader.py. run_baselines keys its resume set on `id`,
# so before that fix a full TRAM run would have silently skipped ~31,626
# items. Pinned by tests/test_benchmark_loader.py::test_tram_ids_are_unique.
#
# Usage — from a shell with real GPU access, NOT a Claude Code Bash session:
#   mkdir -p logs/zs_tram
#   nohup bash scripts/run_zeroshot_tram.sh > logs/zs_tram/driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
Q=logs/zs_tram
mkdir -p "$Q"

if [ "$(cat "$Q/zs_tram.done" 2>/dev/null)" = "OK" ]; then
  echo "$(date '+%F %T') skip zs_tram (already OK)"; exit 0
fi

echo "$(date '+%F %T') START zs_tram (980,918 items, batch 256, ~3 h)"
if venv/bin/python scripts/run_baselines.py \
     --model llama --benchmark tram \
     --results-dir results/baseline/zero_shot_v3 \
     --batch-size 256 --token-budget 57344 \
     > "$Q/zs_tram.log" 2>&1
then echo OK > "$Q/zs_tram.done"; else echo FAIL > "$Q/zs_tram.done"; fi
echo "$(date '+%F %T') END   zs_tram -> $(cat "$Q/zs_tram.done")"

# Report immediately: TRAM's random baseline is 18.92% (not 25% — 28.8% of
# TRAM is short-answer nli_saq where guessing scores nothing). Quote it beside
# the result or a reader cannot tell skill from chance.
echo "$(date '+%F %T') TRAM random baseline = 18.92%; MCQ share 53.7%"
venv/bin/python scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && echo "$(date '+%F %T') rescore -> OK" || echo "$(date '+%F %T') rescore -> FAIL"
