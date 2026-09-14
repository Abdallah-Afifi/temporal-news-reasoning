# Cycle code - v4

Single variable vs v3-fixed: rehearsal swapped to HotpotQA/DROP/CoQA. Builder rewritten per the mistake ledger (D36).

Files: builder.py, config.yaml, queue.sh

Data: data/training_versions/v4/ (copy) / data/combined_80_20_v4/ (canonical)
Results: results/corrected/v4
Adapter: checkpoints/llama_v4

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
