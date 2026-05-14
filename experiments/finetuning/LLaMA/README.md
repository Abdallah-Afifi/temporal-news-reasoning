# LLaMA-3.2-3B-Instruct — LoRA Fine-Tuning

This directory contains the complete LoRA fine-tuning pipeline for
**meta-llama/Llama-3.2-3B-Instruct** on the temporal-news-reasoning curriculum.

---

## Directory Structure

```
experiments/finetuning/LLaMA/
├── config.yaml       # Hyperparameters, LoRA settings, curriculum stage paths
├── data_loader.py    # LLaMA-specific tokenization & data formatting
├── train.py          # Main training script (curriculum loop)
├── evaluate.py       # Evaluation on TIME / TIMEBENCH / TRAM benchmarks
├── inference.py      # Quick interactive / batch inference testing
└── README.md         # This file
```

---

## Model Card

| Property | Value |
|----------|-------|
| **Base model** | `meta-llama/Llama-3.2-3B-Instruct` |
| **Parameters** | 3.21 B |
| **Context window** | 128K tokens |
| **Architecture** | Transformer, GQA (8 KV heads, 24 query heads) |
| **Positional encoding** | RoPE |
| **LoRA rank** | 16 (configurable) |
| **LoRA target modules** | q_proj, k_proj, v_proj, o_proj |
| **Trainable parameters** | ~25M / 3.21B = 0.78% |
| **Required VRAM** | ≥10 GB (bfloat16) or ≥6 GB (4-bit QLoRA) |

> **License**: LLaMA 3.2 requires accepting the Meta license at
> [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
> before downloading.

---

## Curriculum Stages

| Stage | Data file | Focus | LR |
|-------|-----------|-------|----|
| Stage 1 | `data/training/stage1_explicit.jsonl` | Explicit dates, year arithmetic | 2e-4 |
| Stage 2 | `data/training/stage2_implicit.jsonl` | Implicit refs ("yesterday", "last month") | 1e-4 |
| Stage 3 | `data/training/stage3_complex.jsonl` | Timelines, fast-changing facts, ordering | 5e-5 |

---

## Quick Start

### 0. Prerequisites

```bash
# Accept LLaMA license, then:
huggingface-cli login

# Install dependencies (from repo root)
pip install -r requirements.txt
```

### 1. Verify data and tokenizer (dry-run)

```bash
# From repo root
python experiments/finetuning/LLaMA/data_loader.py --preview --stage stage1_explicit

# Verify training pipeline with 5 steps only
python experiments/finetuning/LLaMA/train.py --dry-run
```

### 2. Run full curriculum training

```bash
# Standard (bfloat16, ≥10 GB VRAM)
python experiments/finetuning/LLaMA/train.py

# Low-VRAM (4-bit QLoRA, ≥6 GB VRAM)
python experiments/finetuning/LLaMA/train.py --load-in-4bit

# Resume from stage 2 after an interruption
python experiments/finetuning/LLaMA/train.py --resume-from-stage stage2_implicit

# Custom config path
python experiments/finetuning/LLaMA/train.py --config experiments/finetuning/LLaMA/config.yaml
```

Training produces checkpoints under `checkpoints/llama/`:
```
checkpoints/llama/
├── stage1_explicit/final/
├── stage2_implicit/final/
├── stage3_complex/final/
└── final/                  ← final merged adapter (use this)
```

### 3. Quick inference test

```bash
# Run demo covering all 3 curriculum stages
python experiments/finetuning/LLaMA/inference.py --demo

# Interactive mode
python experiments/finetuning/LLaMA/inference.py --interactive

# Single question
python experiments/finetuning/LLaMA/inference.py \
    --question "The CEO resigned last Tuesday. The article was published on March 21, 2025. When did the CEO resign?" \
    --context "TechCorp CEO John Smith stepped down last Tuesday citing personal reasons."
```

### 4. Evaluate on benchmarks

```bash
# Full evaluation on TIME benchmark
python experiments/finetuning/LLaMA/evaluate.py --benchmark time

# Quick test (first 100 examples)
python experiments/finetuning/LLaMA/evaluate.py --benchmark time --max-examples 100

# Evaluate a specific stage checkpoint on TIMEBENCH
python experiments/finetuning/LLaMA/evaluate.py \
    --adapter-path checkpoints/llama/stage3_complex/final \
    --benchmark timebench \
    --stage stage3_complex

# Save report to custom path
python experiments/finetuning/LLaMA/evaluate.py \
    --output results/llama_final_time_eval.json
```

Results are saved to `results/llama/` by default.

---

## Configuration Reference (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | `meta-llama/Llama-3.2-3B-Instruct` | HF model ID |
| `model.local_path` | `""` | Local model dir (overrides `name` if set) |
| `model.dtype` | `bfloat16` | Training precision |
| `lora.r` | `16` | LoRA rank |
| `lora.lora_alpha` | `32` | Scaling factor (alpha/r = 2.0) |
| `lora.target_modules` | `[q,k,v,o]_proj` | Layers to adapt |
| `lora.lora_dropout` | `0.05` | Adapter dropout |
| `training.per_device_train_batch_size` | `4` | Reduce to 2 if OOM |
| `training.gradient_accumulation_steps` | `4` | Effective batch = 16 |
| `training.max_seq_length` | `2048` | Truncation length |
| `training.bf16` | `true` | bfloat16 mixed precision |
| `quantization.load_in_4bit` | `false` | Enable QLoRA |

---

## LLaMA 3.x Chat Template

LLaMA 3.x uses special header tokens — do **not** mix with LLaMA 2's `[INST]` format:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{assistant_response}
```

The `data_loader.py` and `shared/prompt_templates.py::format_llama_prompt` apply
this automatically.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OutOfMemoryError` | Reduce `per_device_train_batch_size` to 2, enable `--load-in-4bit`, or reduce `max_seq_length` to 1024 |
| `401 Unauthorized` from HF | Run `huggingface-cli login` and accept the LLaMA license |
| `pad_token` warning | Expected — LLaMA has no dedicated pad token; we use EOS as pad during training |
| Slow tokenization | Ensure `use_fast=True` in `load_tokenizer()` (default) |
| Checkpoint not found | Check `checkpoints/llama/final/` exists; run training first |
| `trust_remote_code` error | Not needed for LLaMA — only Qwen requires this flag |

---

## Expected Results (reference)

| Stage | Benchmark | Accuracy | Notes |
|-------|-----------|----------|-------|
| Baseline (0-shot) | TIME | ~40% | Before fine-tuning |
| After Stage 1 | TIME | ~55% | Explicit temporal improvement |
| After Stage 3 | TIME | ~65%+ | Target after full curriculum |

> Actual results will vary with hardware, data quality, and hyperparameters.
> Refer to `results/llama/` after running evaluation.
