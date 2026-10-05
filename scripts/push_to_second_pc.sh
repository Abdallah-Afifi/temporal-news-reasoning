#!/usr/bin/env bash
# Incremental sync of NEW WORK to the second PC (which already has an older
# copy of this repo from a prior sync -- pre-Mistral). Pushed FROM this
# machine (the one with the live v13/mistral work in progress) over the
# tailnet.
#
#   host  Second_PC  (100.72.118.72 on the tailnet; 192.168.55.3 on the
#         direct Ethernet link -- use REMOTE_HOST=192.168.55.3, Tailscale
#         relays through DERP at ~1MB/s, the cable does ~190MB/s)
#   user  g02-s26
#   path  set REMOTE_PATH below, or override: REMOTE_PATH=... bash scripts/push_to_second_pc.sh
#
# Excluded, and why (2026-10-04 sizing pass -- remote only has 37GB free):
# (all anchored to the repo root with a leading "/", so a nested directory
#  that happens to share a name -- e.g. a results/ inside data/ -- is still sent)
#   venv/ venv_vllm/ venv_qwen/   -- machine-specific, must be rebuilt (TRANSFER_README.md)
#   */merged_fp16/                -- reproducible (base model + the adapter,
#                                     which IS synced, via scripts/merge_lora.py);
#                                     cut checkpoints/ from 25.7GB to 4.8GB.
#                                     One of these (mistral_best_s43) is also
#                                     the LIVE file the running vLLM eval is
#                                     reading right now and gets rm -rf'd the
#                                     moment that eval finishes.
#   gemma-4-e4b-model/            -- public base weights, not generated work,
#   Qwen2.5-3B-Instruct/             not used by any current v11/v12/v13/
#                                     mistral pipeline; re-download from HF if
#                                     ever needed again (15G + 5.8G).
#   results/                      -- real generated output, deliberately
#                                     deferred to a second pass once this
#                                     smaller push lands and disk is rechecked.
# Mistral-7B-Instruct-v0.3 needed NO exclusion -- it was already on the
# remote from the prior sync (confirmed: dry-run size didn't change when
# excluded).
#
# --update: a file that is NEWER on the remote is left alone, never
# overwritten by an older local copy. No --delete: nothing on the remote is
# ever removed.
#
# Resumable: rsync --partial, safe to re-run if it dies partway. Run it from
# your own terminal on this PC, not from the Claude Code sandbox -- the
# sandbox has no ssh/rsync access (verified), same boundary as its no-GPU rule.
#
#   bash scripts/push_to_second_pc.sh            # dry run first, then asks
#   FORCE=1 bash scripts/push_to_second_pc.sh    # skip the confirmation prompt
set -eu
cd "$(dirname "$0")/.."

REMOTE_USER="${REMOTE_USER:-g02-s26}"
REMOTE_HOST="${REMOTE_HOST:-100.72.118.72}"
REMOTE_PATH="${REMOTE_PATH:-~/Mohamed/temporal-news-reasoning}"
USER_AT="$REMOTE_USER@$REMOTE_HOST"
DST="$USER_AT:$REMOTE_PATH"

EXCLUDES=(
  --exclude /venv/ --exclude /venv_vllm/ --exclude /venv_qwen/
  --exclude __pycache__/ --exclude '*.pyc'
  --exclude .ipynb_checkpoints/
  --exclude '/checkpoints/*/merged_fp16/'
  --exclude '/models/gemma-4-e4b-model/'
  --exclude '/models/Qwen2.5-3B-Instruct/'
  --exclude '/results/'
)

echo "=== checking rsync on both ends ==="
command -v rsync >/dev/null 2>&1 || { echo "rsync not installed HERE -- install it first (e.g. sudo apt install rsync)"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=6 "$USER_AT" "command -v rsync" >/dev/null 2>&1 \
  || { echo "rsync not installed on $USER_AT -- install it there first (sudo apt install rsync), then re-run"; exit 1; }

echo "=== remote free space ==="
ssh "$USER_AT" "mkdir -p $REMOTE_PATH && df -h \$(eval echo $REMOTE_PATH) | tail -1"
echo "(expect roughly 9-10GB after exclusions: checkpoints adapters ~4.8G, data ~2.75G, small models/code deltas)"

echo
echo "=== dry run (nothing transferred yet) ==="
rsync -an --update --info=stats2 "${EXCLUDES[@]}" ./ "$DST/"

if [ "${FORCE:-0}" != "1" ]; then
  echo
  read -r -p "Proceed with the full mirror now? [yes] " a
  [ "$a" = "yes" ] || { echo "aborted; nothing sent"; exit 1; }
fi

echo "=== transferring (resumable; safe to Ctrl-C and re-run) ==="
rsync -a --update --partial --info=progress2 "${EXCLUDES[@]}" ./ "$DST/"

echo
echo "=== verifying ==="
rsync -an -c --update --info=stats2 "${EXCLUDES[@]}" ./ "$DST/" | tail -15
echo "(an empty/zero-diff 'Number of files transferred' above means the mirror is complete)"
echo
cat <<EOF
On $REMOTE_HOST, rebuild BOTH venvs from the lock files (TRANSFER_README.md)
before running anything -- requirements.txt is unpinned and is NOT the
environment that produced the published numbers:
  cd $REMOTE_PATH
  # training / rescoring venv (torch 2.10.0+cu128, transformers 4.57.6, peft 0.18.1)
  python3.12 -m venv venv && venv/bin/pip install -U pip
  grep -v '^flash_attn @' requirements.lock.venv.txt > /tmp/lock_venv.txt
  venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/lock_venv.txt
  venv/bin/pip install wheels/flash_attn-*.whl          # optional, Ampere+ GPU only
  # evaluation venv (vllm 0.28.0) -- every eval leg runs in this one
  python3.12 -m venv venv_vllm && venv_vllm/bin/pip install -U pip
  venv_vllm/bin/pip install -r requirements.lock.venv_vllm.txt
  export VLLM_USE_FLASHINFER_SAMPLER=0   # mandatory on CUDA 13 (D55); schedules set it
EOF
