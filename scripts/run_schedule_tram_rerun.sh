#!/bin/bash
# ===========================================================================
# TRAM RE-RUN — all 10 arms, on the FIXED prompts (2026-09-12)
# ===========================================================================
# WHY. The audit of 2026-09-12 found a prompt-content bug in
# BenchmarkLoader._tram_csv_to_example: TRAM's NLI `Hypothesis` column
# (564,268 items, 57.5% of the benchmark) and storytelling's `Story` column
# (67,204) never reached the prompt, and `context` fell back to the `Source`
# provenance string for relation_* (204,914). 836,386 of 980,918 items
# (85.3%) were therefore evaluated on prompts the fix changes. Every stored
# TRAM prediction is stale. See docs/audit_2026_09_12.md §0a and
# docs/results_and_methodology.md §9 (the column is WITHDRAWN until this
# schedule completes and the rescore is re-run).
#
# WHAT CHANGED IN THE PROTOCOL: nothing. Same engine (vLLM 0.28), same greedy
# decoding, same seed, same max_new_tokens, same postprocessing, same scorer.
# The only difference is that the prompt now contains the question. Arms,
# checkpoints and adapters are the ones already trained — NO training happens
# here.
#
# OUTPUT GOES TO A NEW ROOT: results/tram_fixed/<arm>/...
# It must NOT be results/corrected/<arm>_vllm — those files hold the stale
# predictions, the model_dir/adapter_dir tags would MATCH, and the runner
# would happily resume and append new predictions onto old ones, blending two
# prompt regimes inside one arm. The stale files are kept as the record.
#
# LAUNCH (the log dir must exist BEFORE the shell opens the redirect):
#   mkdir -p logs/sched_tram_rerun
#   nohup bash scripts/run_schedule_tram_rerun.sh \
#         > logs/sched_tram_rerun/driver.log 2>&1 &
#
# MUST BE LAUNCHED FROM A SHELL WITH REAL GPU ACCESS. A Claude Code Bash-tool
# session has none — confirmed in D33 even with the tool's sandbox flag
# disabled — and a job launched from there silently falls back to CPU and
# thrashes (the 2026-09-05 incident). From Claude Code, prefix the launch with
# `!` so it runs in the user's own terminal.
#
# COST. ~2h per arm measured on the old prompts (127 ex/s over 980,918 items);
# the fixed prompts are longer, so budget ~2h15 per arm and ~22-24h total for
# ten arms, plus ~4 min CPU per merge and ~10 min for the final rescore.
#
# DISK. A merged fp16 3B is ~6.0 GB; five arms (v1, v2, v3c, v4, v5) have no
# merge yet. This script merges -> evaluates -> DELETES each merge it creates,
# so peak extra usage stays near 6 GB + ~650 MB of predictions per arm
# (~6.5 GB total). Merges that already existed (v6, v6d, v7, v7c) are reused
# and never deleted. The run aborts before any merge if free space is under
# MIN_FREE_GB.
#
# RESUMABLE. Every stage writes logs/sched_tram_rerun/<stage>.done containing
# OK or FAIL; a stage marked OK is skipped. Safe to re-launch after a crash.
# run_eval_vllm.py itself resumes by id within a leg and holds an exclusive
# lock, so a double-launch is refused rather than interleaved.
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
Q=logs/sched_tram_rerun
mkdir -p "$Q"
VLLM=venv_vllm/bin/python
PY=venv/bin/python
BASE=models/Llama-3.2-3B-Instruct
OUT_ROOT=results/tram_fixed
MIN_FREE_GB=12

# MANDATORY for every vLLM leg (D55). flashinfer 0.6.16.post3 is
# source-incompatible with CUDA 13.0 — its sampling kernel calls
# cub::BlockAdjacentDifference::FlagHeads, removed in CUDA 13's CUB — so the
# JIT build fails and the engine never starts. Cannot change any number: the
# protocol is greedy (temperature 0.0), so sampling is never exercised and
# vLLM's native path takes the identical argmax.
export VLLM_USE_FLASHINFER_SAMPLER=0

stage_done () { [ "$(cat "$Q/$1.done" 2>/dev/null)" = "OK" ]; }
mark () { if [ "$2" -eq 0 ]; then echo OK > "$Q/$1.done"; else echo FAIL > "$Q/$1.done"; fi
          echo "$(date '+%F %T') END   $1 -> $(cat "$Q/$1.done")"; }
free_gb () { df -P /home 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'; }
say () { echo "$(date '+%F %T') $*"; }

# arm:checkpoint-dir   (zero-shot is handled separately: base model, no adapter)
ARMS="v1:llama v2:llama_v2 v3c:llama_v3_fixed v4:llama_v4 v5:llama_v5 \
v6:llama_v6 v7:llama_v7 v7c:llama_v7_corrected v6d:llama_v6d"

say "TRAM re-run on fixed prompts — start. free=$(free_gb)G"

# ---- GPU GATE -------------------------------------------------------------
# D33: a job launched from a shell without GPU access does not fail cleanly —
# it falls back to CPU and thrashes (the 2026-09-05 incident: ~1h at 0.00 ex/s,
# 56 threads, 20+ GB RSS, competing with a real GPU job). A Claude Code
# Bash-tool session is such a shell, and the boundary does not lift with its
# sandbox flag disabled. Fail here, in 2 seconds, instead.
$VLLM -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('NO GPU VISIBLE to this shell (torch.cuda.is_available() is False).\n'
             'This schedule would fall back to CPU and thrash for hours.\n'
             'Launch it from a terminal with real GPU access — from Claude Code, '
             'prefix the command with \'!\' so it runs in the user shell.')
print(f'gpu gate: {torch.cuda.device_count()} device(s) — {torch.cuda.get_device_name(0)}')
" || { say "ABORT: GPU gate failed"; exit 1; }

$PY - <<'PRECHECK' || { say "PRECHECK FAILED — aborting"; exit 1; }
import sys
sys.path.insert(0, ".")
from src.data.data_loader import BenchmarkLoader
ex = BenchmarkLoader("data/benchmarks").load("tram")
n = len(ex)
uniq = len({(e.question, e.context, e.answer) for e in ex})
print(f"precheck: {n:,} TRAM items, {uniq:,} unique (context, question, answer)")
# The whole point of this re-run. Before the loader fix this read 244,064;
# TRAM's paper documents 526,668 problems. If this assert fires, the fix is
# not in place and the re-run would reproduce the broken prompts.
assert n == 980_918, f"unexpected TRAM size {n}"
assert uniq > 500_000, (
    f"ONLY {uniq:,} unique problems — the 2026-09-12 loader fix is NOT active. "
    f"Re-running now would regenerate the broken prompts. Aborting."
)
print("precheck: loader fix confirmed active")
PRECHECK

# ---- 0. ZERO-SHOT (base model, no adapter, no system prompt) --------------
# The baseline every arm is compared against. No --adapter-dir: that flag
# selects the training system prompt, which the baseline must not get.
if stage_done zs_tram; then say "zs_tram already OK — skipping"; else
  say "START zs_tram (~2h15)"
  $VLLM scripts/run_eval_vllm.py \
      --model-dir "$BASE" \
      --benchmark tram \
      --results-dir "$OUT_ROOT/zero_shot" \
      --model-name llama \
      > "$Q/zs_tram.log" 2>&1
  mark zs_tram $?
fi

# ---- 1..9. THE FINE-TUNED ARMS -------------------------------------------
for pair in $ARMS; do
  arm="${pair%%:*}"; ckpt="${pair##*:}"
  ADAPTER="checkpoints/$ckpt/final"
  MERGED="checkpoints/$ckpt/merged_fp16"
  PREEXISTING=no; [ -d "$MERGED" ] && PREEXISTING=yes

  if stage_done "${arm}_tram"; then say "${arm}_tram already OK — skipping"; continue; fi

  # --- merge (only if this arm has no merged model yet) --------------------
  if [ ! -d "$MERGED" ]; then
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
      say "ABORT: only $(free_gb)G free, need >= ${MIN_FREE_GB}G to merge $ckpt"
      echo FAIL > "$Q/${arm}_merge.done"; break
    fi
    say "START ${arm}_merge (~4 min CPU, ~6.0G)"
    $PY scripts/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
        > "$Q/${arm}_merge.log" 2>&1
    mark "${arm}_merge" $?
    stage_done "${arm}_merge" || { say "merge failed for $arm — skipping its eval"; continue; }
  else
    say "${arm}: reusing existing $MERGED (not created here, will NOT be deleted)"
  fi

  # --- evaluate ------------------------------------------------------------
  say "START ${arm}_tram (~2h15)"
  $VLLM scripts/run_eval_vllm.py \
      --model-dir "$MERGED" \
      --adapter-dir "$ADAPTER" \
      --benchmark tram \
      --results-dir "$OUT_ROOT/$arm" \
      --model-name llama \
      > "$Q/${arm}_tram.log" 2>&1
  mark "${arm}_tram" $?

  # --- reclaim -------------------------------------------------------------
  # Only merges THIS script created are removed, and only after the leg that
  # needed them succeeded. A merge is regenerable in ~4 min, a leg is 2h.
  if [ "$PREEXISTING" = "no" ] && stage_done "${arm}_tram"; then
    say "reclaiming $MERGED (~6.0G; regenerable via scripts/merge_lora.py)"
    rm -rf "$MERGED"
  fi
  say "free=$(free_gb)G"
done

# ---- 10. RESCORE ----------------------------------------------------------
# --tram-root repoints only the TRAM arms at the new predictions; TIME and
# TimeBench are untouched and keep reproducing their published numbers.
if stage_done rescore; then say "rescore already OK — skipping"; else
  say "START rescore (~10 min CPU)"
  $PY scripts/rescore_v5_protocol.py \
      --tram-root "$OUT_ROOT" \
      --out results/rescored/v5_protocol_tram_fixed.json \
      > "$Q/rescore.log" 2>&1
  mark rescore $?
fi

# ---- completion -----------------------------------------------------------
_failed=""
for _d in "$Q"/*.done; do
  [ -e "$_d" ] || continue
  case "$(cat "$_d" 2>/dev/null)" in
    OK) ;;
    *) _failed="$_failed $(basename "$_d" .done)";;
  esac
done
if [ -n "$_failed" ]; then
  say "SCHEDULE FAILED — legs:$_failed" | tee "$Q/schedule_failed.marker"
  rm -f "$Q/schedule_complete.marker"
  exit 1
fi
say "TRAM re-run complete — all stages OK" > "$Q/schedule_complete.marker"
say "TRAM re-run complete. Next: compare results/rescored/v5_protocol_tram_fixed.json"
say "against docs/audit_2026_09_12.md §0a's unaffected-items table, then un-withdraw §9."
