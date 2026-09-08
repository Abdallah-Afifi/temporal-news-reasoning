#!/usr/bin/env bash
# Wait for the v6 queue to finish, then run the zero-shot LLaMA TRAM baseline.
#
# Chained rather than run now because v6's TIME leg owns the GPU until ~05:35.
# This script itself uses no GPU while waiting — it polls for v6's completion
# marker and only then launches.
#
# Launch from a shell with real GPU access, NOT a Claude Code Bash session
# (D32/D33):
#   mkdir -p logs/zs_tram
#   nohup bash scripts/chain_after_v6.sh > logs/zs_tram/chain.log 2>&1 &
#
# Safe to start at any time, including while v6 is mid-run.
set -u
cd "$(dirname "$0")/.."

MARKER=logs/queue_v6/queue_complete.marker
echo "$(date '+%F %T') waiting for $MARKER"

# Poll every 5 minutes. The v6 queue writes the marker as its last action.
while [ ! -f "$MARKER" ]; do
  # Bail out if the v6 driver has died without writing the marker, so this
  # does not wait forever on a crashed run.
  if ! ps -eo cmd --no-headers | grep -q "[r]un_queue_v6.sh"; then
    echo "$(date '+%F %T') v6 driver is gone and no marker was written."
    echo "$(date '+%F %T') Check logs/queue_v6/driver.log before starting TRAM."
    exit 1
  fi
  sleep 300
done

echo "$(date '+%F %T') v6 complete. Starting the zero-shot TRAM baseline."
exec bash scripts/run_zeroshot_tram.sh
