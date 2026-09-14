#!/bin/bash
# V6 ("v5-corrected") launch chain: train -> style audit -> letter probe
# -> format probe -> TimeBench -> TIME -> rescore. Per docs/v6_plan.md.
# THREE data variables vs v5 (combined arm, aimed at beating zero-shot):
#   rehearsal 8,026 -> 2,600 (half long-form) | AUG_SEQ +1,273 |
#   AUG_NOANS +500 with a 4:1 AUG_MCQ2 counterweight.
#
# Safe to re-run after a crash: each leg resumes from its own
# predictions.jsonl, and a leg whose .done marker reads OK is skipped.
#
# Must be launched from a shell with real GPU access (a normal terminal, or
# another agent session that has it) -- NOT from a Claude Code Bash-tool
# session on this machine, which cannot see the GPU at all (session_handoff
# D32/D33).
#
#   mkdir -p logs/queue_v7    # the redirect target must exist BEFORE the
#                             # shell opens it; the script's own mkdir is
#                             # too late
#   nohup bash scripts/run_queue_v7.sh > logs/queue_v7/driver.log 2>&1 &
#
# EVAL ORDER (user decision 2026-09-07, D47): TimeBench -> TRAM -> TIME,
# CHEAPEST FIRST. Measured: TimeBench ~12 min, TRAM ~3 h at batch 256,
# TIME ~10.5 h.
#
# BATCH SIZES ARE NOT INTERCHANGEABLE:
#   TIME / TimeBench -> 32. FROZEN. These have published numbers, and batch
#     composition changes greedy output (89.25% agreement measured when 800
#     identical items were regenerated in different batches). DO NOT CHANGE.
#   TRAM -> 256. A free choice: TRAM has never been run, so no prior number
#     constrains it. Measured on 3,000 TRAM prompts (p50 44, p90 94, p99 211):
#        batch  items/batch  padding  batches    KV     est. total
#           32           32     2.2%   30,654  0.5GB      ~23 h
#          128          128     6.4%    7,663  2.0GB
#          256          256     8.0%    3,832  3.9GB       ~3 h   <-- chosen
#          512   271 (capped by the 57,344 token budget)  19.2%   3,620  4.2GB
#     Past 256 the token budget caps the effective batch near 271, so 512 buys
#     ~5% fewer batches for 2.4x the padding waste. 256 is the knee.
#   Batch size does NOT improve accuracy -- greedy decoding is deterministic
#   and batching only perturbs floating-point reduction order. It is chosen
#   for SPEED, then frozen for comparability across arms.
#
# Killing it safely -- never `pkill -f run_queue_v6`, that pattern matches
# your own shell:
#   ps -eo pid,cmd --no-headers | awk '$2=="bash" && $3=="scripts/run_queue_v7.sh" {print $1}'
set -u
cd /home/g02-s26/Mohamed/temporal-news-reasoning || exit 1
Q=logs/queue_v7
mkdir -p "$Q"

leg () {  # name results_dir adapter benchmark [batch_size]
  local name=$1 out=$2 adapter=$3 bench=$4 batch=${5:-32}
  if [ "$(cat "$Q/$name.done" 2>/dev/null)" = "OK" ]; then
    echo "$(date '+%F %T') skip $name (already OK)"; return
  fi
  echo "$(date '+%F %T') START $name (batch $batch)"
  # Batch size, token budget and the prompt-length sort are PROTOCOL for
  # TIME and TimeBench -- identical to v1-v6, and changing any of them
  # changes the numbers. The 5th argument exists ONLY for TRAM, which has
  # never been run and therefore has no prior number to stay consistent with
  # (D47). Never pass it for time or timebench.
  local args=(--model llama --benchmark "$bench" --results-dir "$out"
              --batch-size "$batch" --token-budget 57344 --adapter-dir "$adapter")
  if venv/bin/python scripts/run_baselines.py "${args[@]}" > "$Q/$name.log" 2>&1
  then echo OK > "$Q/$name.done"; else echo FAIL > "$Q/$name.done"; fi
  echo "$(date '+%F %T') END   $name -> $(cat "$Q/$name.done")"
}

# ---- stage 1: TRAIN (~7 h; 14,850 rows, SHORTER than v5's 18,180) ---------
if [ "$(cat "$Q/v7_train.done" 2>/dev/null)" = "OK" ]; then
  echo "$(date '+%F %T') skip v7_train (already OK)"
else
  echo "$(date '+%F %T') START v7_train"
  if venv/bin/python experiments/finetuning/LLaMA/train.py \
       --config experiments/finetuning/LLaMA/config_v7.yaml \
       > "$Q/v7_train.log" 2>&1
  then echo OK > "$Q/v7_train.done"; else echo FAIL > "$Q/v7_train.done"; fi
  echo "$(date '+%F %T') END   v7_train -> $(cat "$Q/v7_train.done")"
fi

if [ "$(cat "$Q/v7_train.done" 2>/dev/null)" != "OK" ]; then
  echo "$(date '+%F %T') v6 training did not succeed; stopping (no eval spent on a bad checkpoint)"
  exit 1
fi

# ---- stage 2: answer-style audit (INFORMATIONAL, ~10 min) ------------------
# THE lesson of the v4 post-mortem. v4's TIME number read 33.62% and was
# taken for a 4.15pp regression against v3-corrected; three quarters of that
# gap was v4 answering correctly in a different surface form ("Fact 1" for
# "Fact 1 happened earlier.", "18 March 1934" for "March 18, 1934."). Nine
# hours of evaluation were spent before that was visible. This probe makes it
# visible in ten minutes, on 300 stratified items, by reporting the strict
# exact-match score next to the v5-protocol score for the SAME generations.
#
# It is deliberately NOT a gate: a style shift is not a reason to discard a
# checkpoint, and the v4 letter probe already showed what happens when a
# diagnostic written for a superseded design is allowed to stop the chain.
if [ "$(cat "$Q/style_audit.done" 2>/dev/null)" != "" ]; then
  echo "$(date '+%F %T') style_audit already run -> $(cat "$Q/style_audit.done") (informational)"
else
  echo "$(date '+%F %T') START style_audit (informational)"
  if venv/bin/python scripts/audit_output_style.py \
       --adapter checkpoints/llama_v7/final \
       --reference results/corrected/v6/llama/time/finetuned/predictions.jsonl \
       --out "$Q/style_audit.report" > "$Q/style_audit.log" 2>&1
  then echo OK > "$Q/style_audit.done"; else echo FAIL > "$Q/style_audit.done"; fi
  echo "$(date '+%F %T') END   style_audit -> $(cat "$Q/style_audit.done") (informational)"
fi

# ---- stage 3: letter probe (INFORMATIONAL) --------------------------------
# Kept for lineage continuity: v4 emitted a bare letter on 0/100 items, and
# v4_plan.md asked for v1/v2/v3-corrected to be re-probed to establish
# whether 0% is universal. Recorded, never gating -- see the v4 queue script
# for the full false-negative account.
if [ "$(cat "$Q/letter_probe.done" 2>/dev/null)" != "" ]; then
  echo "$(date '+%F %T') letter_probe already run -> $(cat "$Q/letter_probe.done") (informational)"
else
  echo "$(date '+%F %T') START letter_probe (informational)"
  if venv/bin/python scripts/eval_letter_probe.py \
       --adapter checkpoints/llama_v7/final \
       --out "$Q/letter_probe.report" > "$Q/letter_probe.log" 2>&1
  then echo OK > "$Q/letter_probe.done"; else echo FAIL > "$Q/letter_probe.done"; fi
  echo "$(date '+%F %T') END   letter_probe -> $(cat "$Q/letter_probe.done") (informational)"
fi

# ---- stage 4: CHEAPEST FIRST -- TimeBench, then TRAM, then TIME ----------
# Reordered by user decision 2026-09-07 (D47). Legs are idempotent and resume
# from their own predictions.jsonl, so ordering costs nothing and a broken
# checkpoint now surfaces in ~12 min instead of after a 10-hour TIME leg.
leg v7_timebench results/corrected/v7 checkpoints/llama_v7/final timebench

# TRAM -- FULL RUN, all 980,918 items (user decision 2026-09-07, D48).
# No subset, so the stratification question is moot and the `--max-samples`
# trap cannot apply. Batch 256 per D47: ~3,832 batches, 8.0% padding, ~3.9 GB
# KV, ~3 h. The 31,626 duplicate ids were FIXED in src/data/data_loader.py on
# 2026-09-07 -- mandatory here, because run_baselines keys resume on `id` and
# would otherwise have silently skipped every duplicate. Pinned by
# tests/test_benchmark_loader.py::test_tram_ids_are_unique.
leg v7_tram      results/corrected/v7 checkpoints/llama_v7/final tram 256

leg v7_time       results/corrected/v7 checkpoints/llama_v7/final time

# ---- stage 5: rescore every arm under the v5 protocol (CPU, ~2 min) -------
# v5's headline numbers must be produced by the SAME scorer as every other
# arm, so this reruns the whole table rather than scoring v5 alone.
echo "$(date '+%F %T') START rescore_v5_protocol"
venv/bin/python scripts/rescore_v5_protocol.py > "$Q/rescore.log" 2>&1 \
  && echo "$(date '+%F %T') END   rescore_v5_protocol -> OK" \
  || echo "$(date '+%F %T') END   rescore_v5_protocol -> FAIL"

# ---- stage 4b: format probe (INFORMATIONAL, CPU, ~1 min) ------------------
# The process change D38 asked for. audit_output_style.py reports ONE
# aggregate strict-vs-protocol gap, which is why v5's total collapse on 10.8%
# of TIME passed it as merely "LARGE". This reads per-gold-format emission
# rates off the finished TIME predictions -- sequence, no-answer, MCQ, free
# text -- against zero-shot and v5, so the two behaviours v6 exists to fix
# are reported explicitly instead of being averaged away.
echo "$(date '+%F %T') START format_probe"
venv/bin/python scripts/probe_answer_formats.py \
     --arms v6 v5 v3-corr zero-shot --out "$Q/format_probe.report" \
     > "$Q/format_probe.log" 2>&1 \
  && echo "$(date '+%F %T') END   format_probe -> OK" \
  || echo "$(date '+%F %T') END   format_probe -> FAIL (informational)"

echo "$(date '+%F %T') v6 queue complete" > "$Q/queue_complete.marker"
