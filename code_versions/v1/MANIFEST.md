# Cycle code - v1

Original cycle (llama v1 + the mistral adapter). No versioned builder - data came from the v1-era standardizer/merge tooling (standardizer.py here is that era's copy).

Files: config.yaml, queue.sh, standardizer.py

Data: data/training_versions/v1/ (copy) / data/combined_80_20_split/ (canonical)
Results: results/corrected/v1
Adapter: checkpoints/llama (v1-era) + checkpoints/lora_mistral

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
