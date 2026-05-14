`src/evaluation/` — evaluation harness and metrics

Purpose
- Evaluate model predictions on benchmarks and produce templated reports.

Key files
- `src/evaluation/evaluate.py`: `EvaluationHarness` driver used by runners.
- `src/evaluation/metrics.py`: metric computations and the `TemporalEvaluator`.

Run examples
```bash
# Evaluate saved predictions using the harness
python -m src.evaluation.evaluate --predictions results/predictions.jsonl --references data/benchmarks/time/ref.jsonl

# Run baseline zero-shot pipeline (loads models and writes reports)
python scripts/run_baselines.py --model qwen --benchmark time
```

Outputs
- Reports and predictions are written to `results/` (see `configs/evaluation_config.yaml` for defaults).

Notes
- The per-model `experiments/finetuning/*/evaluate.py` scripts wrap these utilities for fine-tuned adapters.
