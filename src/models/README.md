`src/models/` — model inference and adapters

Purpose
- Provide a unified inference interface for SLMs used in the project.

Key files
- `src/models/inference.py`: `SLMInference` — load HF models, optional PEFT/LoRA adapters, and generate/batch_generate helpers.

Where to place models
- Put local model copies under `models/` (e.g. `models/Qwen2.5-3B-Instruct/`).
- Put LoRA adapters under `checkpoints/lora/<adapter-name>/` or supply `--adapter-dir` to runners.

Example usage (zero-shot baselines):
```bash
python scripts/run_baselines.py --model qwen2.5 --benchmark time --model-root ./models --adapter-dir ./checkpoints/lora/my_adapter
```

Notes
- `SLMInference` supports loading in 4-bit quantization via `--load-in-4bit` flags used in runner scripts.
- For custom models that require `trust_remote_code`, set `model_dir` or `trust_remote_code` in the runtime config.
