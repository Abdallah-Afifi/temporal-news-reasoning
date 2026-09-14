# LLaMA-3.2-3B-Instruct — LoRA Fine-Tuning

This directory contains the complete LoRA fine-tuning pipeline for
**meta-llama/Llama-3.2-3B-Instruct** on the temporal-news-reasoning curriculum.

---

## Directory Structure

```
experiments/finetuning/LLaMA/
├── config.yaml              # Flat-training hyperparameters (HPO-derived) + LoRA settings
├── data_loader.py           # LLaMA tokenization & chat-template formatting (label masking)
├── train.py                 # Main training script (flat training, no curriculum stages)
├── evaluate.py              # Evaluation on combined val split or TIME/TIMEBENCH/TRAM
├── inference.py             # Quick interactive / batch inference testing
├── hyperparameter_search.py # Optuna HPO (uses the same data pipeline as train.py)
├── HPO/                     # Archived copy of the HPO script + results
└── README.md                # This file
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
| **LoRA target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainable parameters** | ~25M / 3.21B = 0.78% |
| **Required VRAM** | ≥10 GB (bfloat16) or ≥6 GB (4-bit QLoRA) |

> **License**: LLaMA 3.2 requires accepting the Meta license at
> [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
> before downloading.

---

## Training Data

Training uses the flat **combined_80_20_split** dataset (80/20 train/val) built by
`src/data/merge_datasets_80_20.py` from three sources:

| Source | Schema | Notes |
|--------|--------|-------|
| **TLQA** | `question`, `answers`/`final_answers`, `subject` | No context |
| **TimeQA** | `question`, `context`, `targets` | Context-grounded |
| **Temprel** | tlinks/timeline | **Skipped** — not QA format |

Each example is formatted with the LLaMA chat template (system + user + assistant)
and prompt tokens are masked (`labels = -100`), so loss is computed only on the
assistant's answer.

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
# Inspect dataset sizes with the data loader CLI
python experiments/finetuning/LLaMA/data_loader.py \
    --train data/combined_80_20_split/train.jsonl \
    --val data/combined_80_20_split/val.jsonl \
    --model models/Llama-3.2-3B-Instruct

# Verify training pipeline with 5 steps only
python experiments/finetuning/LLaMA/train.py --dry-run
```

### 2. Run full training

```bash
# Standard (bfloat16, ≥10 GB VRAM)
python experiments/finetuning/LLaMA/train.py

# Low-VRAM (4-bit QLoRA, ≥6 GB VRAM) — force on/off overrides config
python experiments/finetuning/LLaMA/train.py --load-in-4bit

# Custom config path
python experiments/finetuning/LLaMA/train.py --config experiments/finetuning/LLaMA/config.yaml
```

Training produces the adapter under `checkpoints/llama/`:
```
checkpoints/llama/
├── checkpoint-*/           # per-epoch checkpoints (best kept via load_best_model_at_end)
└── final/                  ← final adapter (use this)
```

### 3. Quick inference test

```bash
# Run demo questions
python experiments/finetuning/LLaMA/inference.py --demo

# Interactive mode
python experiments/finetuning/LLaMA/inference.py --interactive

# Single question
python experiments/finetuning/LLaMA/inference.py \
    --question "The CEO resigned last Tuesday. The article was published on March 21, 2025. When did the CEO resign?" \
    --context "TechCorp CEO John Smith stepped down last Tuesday citing personal reasons."
```

> Inference prompts are rendered with the SAME chat template + system prompt
> used during training (`shared/prompt_templates.py::format_inference_prompt`).

### 4. Evaluate

```bash
# Default: held-out combined validation split (recommended first check)
python experiments/finetuning/LLaMA/evaluate.py

# Quick test (first 100 examples)
python experiments/finetuning/LLaMA/evaluate.py --max-examples 100

# Custom data file / report path
python experiments/finetuning/LLaMA/evaluate.py \
    --data data/combined_80_20_split/val.jsonl \
    --output results/llama_final_val_eval.json

# Benchmark mode (requires data downloaded via scripts/download_datasets.py)
python experiments/finetuning/LLaMA/evaluate.py --benchmark time
```

Reports are saved to `experiments/finetuning/LLaMA/results/` by default.

---

## Configuration Reference (`config.yaml`)

Flat keys (not nested):

| Key | Default | Description |
|-----|---------|-------------|
| `model_path` | `models/Llama-3.2-3B-Instruct` | Local model dir (falls back to HF id) |
| `dtype` | `bfloat16` | Training precision |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRA adapter settings |
| `target_modules` | `[q,k,v,o,gate,up,down]_proj` | Layers to adapt |
| `learning_rate` | `4.62e-4` | HPO-derived |
| `num_train_epochs` | `3` | With early stopping (patience 2) |
| `per_device_train_batch_size` | `8` | Reduce to 2 if OOM |
| `gradient_accumulation_steps` | `2` | Effective batch = 16 |
| `max_seq_length` | `2048` | Truncation length |
| `bf16` | `true` | bfloat16 mixed precision |
| `load_in_4bit` | `false` | Enable QLoRA |

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

The `data_loader.py` and `shared/prompt_templates.py::format_inference_prompt`
apply this automatically.

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

| Configuration | Benchmark | Accuracy | Notes |
|-------|-----------|----------|-------|
| Baseline (zero-shot) | TIME | ~39% | Before fine-tuning (`scripts/run_baselines.py`) |
| Fine-tuned (LoRA) | combined val | see `results/` | Evaluate with `evaluate.py` |

> Actual results will vary with hardware, data quality, and hyperparameters.
> Note: earlier `results/finetuned/` reports were produced with a zero-shot
> prompt format (no training system prompt) — re-evaluate before comparing
> against baselines.
