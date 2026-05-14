`src/training/` — LoRA fine-tuning guide

Purpose
- LoRA fine-tuning utilities and curriculum orchestration.

Key files
- `src/training/train_lora.py`: main LoRA training entry used by the project.
- `scripts/run_lora_ablation.py`: ablation utilities and helpers.

Configs
- Training configs are YAML files (examples in `experiments/finetuning/<model>/config.yaml`).

Quick command
```bash
python src/training/train_lora.py path/to/config.yaml
# or use the per-model fine-tuning scripts under experiments/finetuning
python experiments/finetuning/qwen3.5-9b-model/train.py --config experiments/finetuning/qwen3.5-9b-model/config.yaml
```

Output
- LoRA adapters and checkpoints are written to `checkpoints/` by default. Keep these directories gitignored.
