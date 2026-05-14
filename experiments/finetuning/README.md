# Finetuning Experiments

This folder contains model-specific LoRA fine-tuning pipelines and helpers used during experiments.

Structure
- `LLaMA/`, `Mistral/`, `qwen3.5-9b-model/` — per-model training, inference, and evaluation scripts.
- `shared/` — utilities shared across model pipelines.

Notes
- The pipelines are experiments only and may contain large `Progress/` and `results/` folders which are ignored by `.gitignore`.
- Prefer using the canonical training and evaluation entrypoints in `scripts/` and `src/training` when available.

Quick links
- Qwen 3.5 evaluator: `experiments/finetuning/qwen3.5-9b-model/evaluate.py`
- LLaMA README: `experiments/finetuning/LLaMA/README.md`

If you want to run or reproduce these experiments, read the model README within each subfolder.
