#!/usr/bin/env bash
# Push the HARNESS CODE to mo-linux so it can run zero-shot Mistral.
#
#   host  mo-linux  (100.95.193.19 / mo-linux.tail4d8373.ts.net, MagicDNS)
#   user  mohamed-khaled
#   path  ~/thesis/Thesis2/temporal-news-reasoning
#
# CODE ONLY — user decision 2026-09-08. The Mistral base model (28 GB) and the
# benchmarks are already on mo-linux and are NOT touched. Nothing here moves
# model weights or data.
#
#   bash scripts/push_to_mo_linux.sh            # dry run first, then asks
#   FORCE=1 bash scripts/push_to_mo_linux.sh    # skip the prompt
#
# WHY THE CODE MUST BE OVERWRITTEN: the local copy carries fixes the remote
# one predates — double-BOS tokenisation, first-gold-only scoring, missing NLI
# task instructions, id collisions, and `CANONICAL_FILES` (which is what makes
# TIME load 104,951 items instead of double-counting TIME-Lite to 106,500,
# D23). Without these the remote numbers are not comparable to anything.
set -eu
cd "$(dirname "$0")/.."

USER_AT="${REMOTE_USER:-mohamed-khaled}@${REMOTE_HOST:-mo-linux}"
RPATH="${REMOTE_PATH:-~/thesis/Thesis2/temporal-news-reasoning}"
DST="$USER_AT:$RPATH"
# The file list is DERIVED FROM GIT, not hand-maintained. A hand-written list
# silently drifted once already: the 2026-09-08 push sent
# (scripts src experiments docs requirements.txt) and therefore MISSED
# tests/ (5 files), prompts/ and .gitignore — so mo-linux would have pushed
# the scoring fixes WITHOUT the tests that pin them.
#
# `git add -An` lists exactly what a commit would contain, already filtered by
# .gitignore, so it cannot include predictions or datasets.
mapfile -t ITEMS < <(git add -An 2>/dev/null \
  | sed "s/^add //; s/^remove //" | tr -d "'" \
  | grep -vE "^\.stfolder/" | sort -u)
if [ "${#ITEMS[@]}" -eq 0 ]; then echo "nothing to sync"; exit 0; fi
echo "=== ${#ITEMS[@]} files to sync (from git add -An) ==="
printf "  %s\n" "${ITEMS[@]}" | head -12
[ "${#ITEMS[@]}" -gt 12 ] && echo "  ... and $(( ${#ITEMS[@]} - 12 )) more"

echo "=== remote state ==="
ssh "$USER_AT" "cd $RPATH && \
  echo 'model      :' \$(du -sh models/Mistral-7B-Instruct-v0.3 2>/dev/null | cut -f1) && \
  echo 'bench/time :' \$(du -sh data/benchmarks/time 2>/dev/null | cut -f1) && \
  echo 'bench/tbench:' \$(du -sh data/benchmarks/timebench 2>/dev/null | cut -f1) && \
  echo 'CANONICAL_FILES in loader :' \$(grep -c CANONICAL_FILES src/data/data_loader.py 2>/dev/null || echo 0) && \
  echo 'trailing-letter rule      :' \$(grep -c _TRAILING_LETTER_RE scripts/run_baselines.py 2>/dev/null || echo 0) && \
  echo 'venv       :' \$(ls venv/bin/python 2>/dev/null || echo MISSING)"

echo
echo "=== dry run: what the code sync WOULD change (nothing transferred) ==="
printf "%s\n" "${ITEMS[@]}" > /tmp/_mo_files.txt
rsync -an --itemize-changes --files-from=/tmp/_mo_files.txt . "$DST/" | head -40
echo "  (>f = file to update, cd = dir to create; empty means already current)"

if [ "${FORCE:-0}" != "1" ]; then
  echo
  read -r -p "Push the code now? [yes] " a
  [ "$a" = "yes" ] || { echo "aborted; nothing sent"; exit 1; }
fi

echo "=== pushing code (~3 MB) ==="
rsync -a --info=progress2 --files-from=/tmp/_mo_files.txt . "$DST/"

echo
echo "=== verify the fixes landed ==="
ssh "$USER_AT" "cd $RPATH && \
  echo 'CANONICAL_FILES :' \$(grep -c CANONICAL_FILES src/data/data_loader.py) && \
  echo 'trailing letter :' \$(grep -c _TRAILING_LETTER_RE scripts/run_baselines.py) && \
  echo 'date equiv      :' \$(grep -c 'def matches_any' src/evaluation/date_equivalence.py)"

cat <<'NEXT'

=== NEXT, on mo-linux ===
  cd ~/thesis/Thesis2/temporal-news-reasoning

  # venv only if the check above said MISSING, or after a requirements change
  python3 -m venv venv && venv/bin/pip install -r requirements.txt

  # sanity: the loader must report 104,951 TIME items, not 106,500
  venv/bin/python -c "import sys; sys.path.insert(0,'.'); \
from src.data.data_loader import BenchmarkLoader; \
print(len(BenchmarkLoader(data_dir='./data/benchmarks').load('time')))"

  # ZERO-SHOT MISTRAL — HF only, never vLLM (2.20pp parity failure).
  # Token budgets are PROTOCOL (D8) and are NOT interchangeable:
  #   timebench 26,624   |   time 20,480
  # llama's 57,344 OOMs on Mistral; it did, ~1.5 h into TIME's 4k head.
  mkdir -p logs
  nohup venv/bin/python scripts/run_baselines.py --model mistral \
      --benchmark timebench --results-dir results/baseline/zero_shot_v3 \
      --batch-size 32 --token-budget 26624 > logs/zs_mistral_timebench.log 2>&1 &

  # then, once that finishes:
  nohup venv/bin/python scripts/run_baselines.py --model mistral \
      --benchmark time --results-dir results/baseline/zero_shot_v3 \
      --batch-size 32 --token-budget 20480 > logs/zs_mistral_time.log 2>&1 &

=== BRING RESULTS BACK (run on THIS machine) ===
  rsync -a mohamed-khaled@mo-linux:~/thesis/Thesis2/temporal-news-reasoning/results/baseline/zero_shot_v3/mistral/ \
           results/baseline/zero_shot_v3/mistral/
  ./venv/bin/python scripts/rescore_v5_protocol.py
NEXT
