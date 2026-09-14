# Cycle code - v6

Format arm: AUG_SEQ + AUG_NOANS + rehearsal cap (D39/D40). First arm to lead zs on TIME (+0.73pp, artifact - D49).

Files: builder.py, config.yaml, queue.sh

Data: data/training_versions/v6/ (copy) / data/combined_80_20_v6/ (canonical)
Results: results/corrected/v6
Adapter: checkpoints/llama_v6

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
