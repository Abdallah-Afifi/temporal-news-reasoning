# Mistral-7B-Instruct-v0.3 — LoRA Fine-Tuning

LoRA fine-tuning pipeline for **mistralai/Mistral-7B-Instruct-v0.3**
on the temporal-news-reasoning curriculum.

---

## Model Card

| Property | Value |
|----------|-------|
| **Base model** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Parameters** | 7.24 B |
| **Context window** | 32K (SWA window: 4096 tokens) |
| **Architecture** | Transformer, GQA (8 KV / 32 query heads), SWA |
| **Chat template** | `[INST]` / `[/INST]` (no separate system-prompt token) |
| **LoRA rank** | 16 |
| **Trainable params** | ~42M / 7.24B = 0.58% |
| **Required VRAM** | ≥18 GB (bfloat16) or ≥10 GB (4-bit QLoRA) |
| **trust_remote_code** | Not required |

---

## Quick Start

```bash
# 1. Dry-run (5 steps only, pipeline check)
python experiments/finetuning/Mistral/train.py --dry-run

# 2. Full training (requires A100 40 GB or 2× 24 GB)
python experiments/finetuning/Mistral/train.py

# Low-VRAM (4-bit QLoRA, ≥10 GB)
python experiments/finetuning/Mistral/train.py --load-in-4bit

# 3. Inference demo
python experiments/finetuning/Mistral/inference.py --demo

# 4. Benchmark evaluation
python experiments/finetuning/Mistral/evaluate.py --benchmark time
python experiments/finetuning/Mistral/evaluate.py --benchmark timebench --max-examples 200
```

---

## Mistral Chat Format

Mistral v0.3 has NO dedicated system-prompt token — system instructions
are prepended inside `[INST]` automatically by `format_mistral_prompt()`:

```
<s>[INST] {system_instructions}

{user_question} [/INST] {assistant_answer}
```

Do **not** use the LLaMA `<|begin_of_text|>` or Qwen `<|im_start|>` templates.

---

## VRAM Tips

| Setup | Approx. VRAM | Config change |
|-------|-------------|---------------|
| bfloat16 full | ~18–20 GB | Default |
| 4-bit QLoRA | ~10–12 GB | `--load-in-4bit` |
| Reduce batch further | ~8 GB | Set `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 16` |

---

## Checkpoint Structure

```
checkpoints/mistral/
├── stage1_explicit/final/
├── stage2_implicit/final/
├── stage3_complex/final/
└── final/                  ← final adapter (use for evaluation/inference)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OutOfMemoryError` | Use `--load-in-4bit` or reduce batch to 1 |
| System prompt ignored | Expected — prepend it inside [INST] via `format_mistral_prompt` |
| Slow training vs 3B models | Expected: 7B has 2× params; use multi-GPU + DeepSpeed ZeRO-2 |
| `trust_remote_code` error | Not needed — remove the flag if it appears in your code |
