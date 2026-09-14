# Cycle code - v5

Single variable vs v4: rehearsal x3. Sequence-emission collapse found (D37/D38).

Files: builder.py, config.yaml, queue.sh

Data: data/training_versions/v5/ (copy) / data/combined_80_20_v5/ (canonical)
Results: results/corrected/v5
Adapter: checkpoints/llama_v5

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
