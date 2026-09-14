# Cycle code - v7

Coverage arm: single-target MCQ (D45), shortcut-free NOANS, rehearsal restored, AUG_ARITH/DURATION/RELATIVE. TRAINING NOW on the second PC. queue.sh = run_schedule_v7.sh (the full schedule); run_queue_v7.sh = eval-only chain; config_v7_fast.yaml = the declined speed variant (D54).

Files: builder.py, config.yaml, config_v7_fast.yaml, queue.sh, run_queue_v7.sh

Data: data/training_versions/v7/ (copy) / data/combined_80_20_v7/ (canonical)
Results: pending - in flight
Adapter: checkpoints/llama_v7 (in flight)

Shared code (train.py, data_loader.py, run_baselines.py, scorers) evolved across cycles and is NOT snapshotted per version here; the per-cycle behavior changes are documented in the D-log (docs/session_handoff.md). Git history predates the cycles (last code commit 2026-05-14) - these copies are the per-cycle record.
