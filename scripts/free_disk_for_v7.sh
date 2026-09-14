#!/usr/bin/env bash
# Reclaim ~7 GB by deleting ONLY the predictions.jsonl of the broken-harness
# runs. Every report JSON, and therefore every published number, is kept.
#
# WHY THESE ARE SAFE: standing decision 5 — "Never resume onto
# results/baseline/zero_shot_v2 or results/finetuned*. Those were produced by
# the broken harness." They are already forbidden as inputs. Their reports
# (25 JSON files) stay, so the historical numbers remain auditable; only the
# raw generations go.
#
# WHY IT IS NEEDED: run_schedule_v7.sh writes ~13.87 GB (v7 merged 6 GB, v6
# merged 6 GB, ~1.9 GB of predictions) and /home has ~14 GB free.
#
# DESTRUCTIVE AND IRREVERSIBLE — it prints what it would remove and requires
# an explicit "yes". Re-running any of those arms would need the old harness,
# which no longer exists, so this cannot be undone by re-running.
set -u
cd "$(dirname "$0")/.."

echo "About to delete these files (predictions only; reports are kept):"
total=0
for d in results/baseline/zero_shot_v2 results/finetuned results/finetuned_v2 \
         results/finetuned_v3 results/finetuned_v3_vllm; do
  [ -d "$d" ] || continue
  while IFS= read -r f; do
    sz=$(stat -c %s "$f"); total=$((total + sz))
    printf "  %8.2f GB  %s\n" "$(echo "$sz" | awk '{print $1/1073741824}')" "$f"
  done < <(find "$d" -name "predictions.jsonl")
done
printf "\n  TOTAL: %.2f GB\n" "$(echo "$total" | awk '{print $1/1073741824}')"
echo "  Reports kept: $(find results/baseline/zero_shot_v2 results/finetuned* -name '*report*.json' 2>/dev/null | wc -l) JSON files"
echo
read -r -p "Type 'yes' to delete: " ans
[ "$ans" = "yes" ] || { echo "aborted, nothing deleted"; exit 1; }

for d in results/baseline/zero_shot_v2 results/finetuned results/finetuned_v2 \
         results/finetuned_v3 results/finetuned_v3_vllm; do
  [ -d "$d" ] || continue
  find "$d" -name "predictions.jsonl" -delete
done
echo "done. free now:"
df -h /home | tail -1
