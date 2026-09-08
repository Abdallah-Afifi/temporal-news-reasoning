"""
experiments/finetuning/shared/generation.py
=================================
Single canonical batch-generation implementation shared by all
LLaMA / Qwen / Mistral evaluation and inference scripts.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_answer_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.1,
) -> list[str]:
    """Generate answers for a batch of fully formatted prompt strings.

    Uses left-padding so causal-LM decoding is aligned across the batch,
    and decodes ONLY the newly generated tokens.
    """
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    # Over-long prompts drop the OLDEST tokens, keeping the question and the
    # chat-template generation header intact (right-truncation would cut them).
    tokenizer.truncation_side = "left"

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            # Prompts come from apply_chat_template, which already emits
            # BOS/special tokens; re-adding them duplicates BOS for
            # tokenizers with add_bos_token=True (LLaMA/Mistral).
            add_special_tokens=False,
        ).to(model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        outputs = model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
