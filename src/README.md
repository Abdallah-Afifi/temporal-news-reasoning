Short overview of the `src/` package

This package contains the core implementation of the Temporal News Reasoning project.

Top-level modules
- `data/`        : data loaders, corpus processors, synthetic data utilities
- `models/`      : model inference interfaces (SLMInference) and adapters
- `training/`    : LoRA training helpers and curriculum orchestration
- `evaluation/`  : evaluation harness, metrics, and reporting
- `pipeline/`    : end-to-end temporal reasoning pipeline
- `rag/`         : retrieval components (indexing, retriever, temporal filters)
- `prompting/`   : prompt templates and self-consistency helpers
- `temporal/`    : temporal intent and metadata extraction utilities

Quick start (repo root):

```bash
# run zero-shot baselines (uses SLMInference and evaluation harness)
python scripts/run_baselines.py --model qwen --benchmark time

# run evaluation harness directly
python -m src.evaluation.evaluate --predictions predictions.jsonl --references references.jsonl
```

Notes
- Models and adapter files should be placed under `models/` or referenced via `model_dir`/`adapter_dir` settings in scripts.
- Experimental fine-tuning runs are in `experiments/finetuning/`.
