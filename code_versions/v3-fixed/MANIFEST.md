# Cycle code - v3-fixed

D21 fix applied; the v3 data rebuilt with full TLQA targets. Best TimeBench arm (48.19%). Builder shown is the post-fix state (in-place fix - pre-fix state not recoverable, git history predates it).

Files: builder.py, config.yaml, queue.sh

Data: data/training_versions/v3-fixed/ (copy) / data/combined_80_20_v3-fixed/ (canonical)
Results: results/corrected/v3_fixed
Adapter: checkpoints/llama_v3_fixed

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
