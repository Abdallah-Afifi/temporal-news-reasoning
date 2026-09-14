# Cycle code - v3

BROKEN cycle - TLQA rows trained empty (D21). Kept for the record.

Files: builder.py, config.yaml, queue.sh

Data: data/training_versions/v3/ (copy) / data/combined_80_20_v3/ (canonical)
Results: results/corrected/v3
Adapter: checkpoints/llama_v3 (broken; merged_fp16 was a STaR teacher only)

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
