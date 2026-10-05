"""
experiments/finetuning/Mistral/data_loader.py
=====================================
Data loading and tokenization for Mistral-7B-Instruct-v0.3 LoRA fine-tuning.

Handles the three data schemas in combined_80_20_split:
  - **TLQA**:   question, answers/final_answers, subject  (no context)
  - **TimeQA**: question, context, targets
  - **Temprel**: tlinks, timeline, entities — **SKIPPED** (no QA format)

Each valid record is normalised to ``{question, context, answer}`` and then
formatted using the Mistral chat template with label masking so that the loss
is computed only on the assistant's response tokens.

Mistral tokenizer nuances
--------------------------
1. No system-prompt token: Mistral v0.3 uses [INST] / [/INST] tokens but
   has NO dedicated system prompt slot. System instructions are prepended
   INSIDE the [INST] block via apply_chat_template.
2. Sliding Window Attention: Mistral uses SWA with a 4096-token window.
   Training sequences of 1024 tokens fit comfortably within one window.
3. trust_remote_code: NOT needed for Mistral.
4. BOS token: Mistral uses '<s>' as the beginning-of-sequence token.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.finetuning.shared.prompt_templates import build_chat_messages
from experiments.finetuning.shared.utils import setup_logger, read_jsonl
from experiments.finetuning.shared.eval_parity import (
    PINNED_DATE, render_prompt, row_to_example, with_context,
)
from scripts.run_baselines import _build_zero_shot_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEY = "mistral"
HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

SKIPPED_DATASETS = {"Temprel"}  # no question field — not usable for QA training


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name_or_path: str = HF_MODEL_NAME,
) -> AutoTokenizer:
    """Load and configure the Mistral tokenizer.

    - ``pad_token`` → ``eos_token`` (Mistral v0.3 may not ship with one).
    - ``padding_side`` → ``"right"`` for training.
    - ``trust_remote_code=False`` (Mistral does not require it).
    """
    log = setup_logger(f"{MODEL_KEY}.tokenizer")
    log.info("Loading Mistral tokenizer from: %s", model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=False,  # Not needed for Mistral
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token set to eos_token (%r)", tokenizer.eos_token)

    tokenizer.padding_side = "right"

    log.info(
        "Tokenizer loaded | vocab_size=%d | model_max_length=%d",
        tokenizer.vocab_size,
        tokenizer.model_max_length,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Record normalisation  (3 schemas → {question, context, answer})
# ---------------------------------------------------------------------------

def _first_present(record: dict, *keys: str):
    """Return the value of the first key present with a non-empty value.

    Builders have written the answer list under different keys across data
    versions — v3 emitted TLQA answers as ``targets`` only — so each branch
    checks every known key instead of one canonical name.
    """
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return []


def normalize_record(record: dict) -> Optional[dict]:
    """Convert a raw JSONL record to ``{question, context, answer}``.

    Returns ``None`` for records that cannot be converted (e.g. Temprel).
    """
    source = record.get("source_dataset", "")

    # ── Skip unsupported datasets ────────────────────────────────
    if source in SKIPPED_DATASETS:
        return None

    # ── TLQA: question + answers (list), no context ─────────────
    if source == "TLQA":
        question = record.get("question", "").strip()
        if not question:
            return None
        answers = _first_present(record, "final_answers", "answers", "targets")
        if isinstance(answers, list):
            answer_parts = [str(a).strip() for a in answers if str(a).strip()]
            answer = "; ".join(answer_parts)
        else:
            answer = str(answers)
            answer_parts = [answer] if answer.strip() else []
        if not answer.strip():
            return None
        return {"question": question, "context": "", "answer": answer,
                "answer_parts": answer_parts}

    # ── TimeQA: question + context + targets ────────────────────
    if source == "TimeQA":
        question = record.get("question", "").strip()
        if not question:
            return None
        context = record.get("context", "")
        targets = _first_present(record, "targets", "final_answers", "answers")
        if isinstance(targets, list):
            answer_parts = [str(t).strip() for t in targets if str(t).strip()]
            answer = "; ".join(answer_parts)
        else:
            answer = str(targets)
            answer_parts = [answer] if answer.strip() else []
        if not answer.strip():
            return None
        return {"question": question, "context": context, "answer": answer,
                "answer_parts": answer_parts}

    # ── Unknown source — best-effort extraction ─────────────────
    question = record.get("question", "").strip()
    if not question:
        return None
    context = record.get("context", "")
    raw_ans = record.get("answer", record.get("answers", record.get("targets", "")))
    if isinstance(raw_ans, list):
        answer_parts = [str(a).strip() for a in raw_ans if str(a).strip()]
        answer = "; ".join(answer_parts)
    else:
        answer = str(raw_ans)
        answer_parts = [answer] if answer.strip() else []
    if not answer.strip():
        return None
    return {"question": question, "context": context, "answer": answer,
            "answer_parts": answer_parts}


# ---------------------------------------------------------------------------
# Tokenisation with label masking
# ---------------------------------------------------------------------------

def _truncate_context(context: str, max_chars: int) -> str:
    """Truncate context to approximately *max_chars* characters on a sentence
    boundary so the model still sees coherent text."""
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars // 2:
        truncated = truncated[: last_period + 1]
    return truncated + " [truncated]"


def tokenize_example(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> Optional[dict]:
    """Tokenise a normalised example using the Mistral chat template.

    If the context is too long, it is **truncated** (not discarded) so the
    answer always fits within ``max_length``.  Labels are set to ``-100`` for
    all prompt tokens (system + user) — loss is computed only on the
    assistant's response.

    Returns a dict with ``input_ids``, ``attention_mask``, ``labels``.
    """
    question = example["question"]
    context = example["context"]
    answer = example["answer"]

    from experiments.finetuning.shared.prompt_templates import TEMPORAL_SYSTEM_PROMPT
    
    # ------------------------------------------------------------------
    # Prepend system prompt to user content manually for Mistral
    # because its chat template drops the system role when an assistant
    # role is present.
    # ------------------------------------------------------------------
    user_content_base = f"{TEMPORAL_SYSTEM_PROMPT}\n\n"
    
    # ------------------------------------------------------------------
    # Exact context truncation: use the tokenizer to measure the overhead
    # so the answer always fits precisely.
    # ------------------------------------------------------------------
    empty_context_content = user_content_base + (f"Context:\n\n\nQuestion: {question}" if context else f"Question: {question}")
    full_messages_empty = [{"role": "user", "content": empty_context_content}, {"role": "assistant", "content": answer}]
    full_text_empty = tokenizer.apply_chat_template(full_messages_empty, tokenize=False, add_generation_prompt=False)
    overhead_tokens = len(tokenizer(full_text_empty, add_special_tokens=False)["input_ids"])
    
    max_context_tokens = max_length - overhead_tokens - 10  # safety margin
    
    if context:
        context_tokens = tokenizer(context, add_special_tokens=False)["input_ids"]
        if len(context_tokens) > max_context_tokens:
            if max_context_tokens > 0:
                truncated_context_tokens = context_tokens[:max_context_tokens]
                context = tokenizer.decode(truncated_context_tokens)
            else:
                context = ""
    
    # Build final content
    if context:
        user_content = user_content_base + f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = user_content_base + f"Question: {question}"

    full_messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": answer}]
    prompt_messages = [{"role": "user", "content": user_content}]

    try:
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

    full_enc = tokenizer(
        full_text, truncation=True, max_length=max_length,
        add_special_tokens=False,
    )
    prompt_enc = tokenizer(
        prompt_text, truncation=True, max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Mask prompt tokens in labels
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    if len(labels) < len(input_ids):
        labels += input_ids[len(labels):]
    labels = labels[: len(input_ids)]

    # Final safety: skip only if answer is truly empty
    n_loss_tokens = sum(1 for l in labels if l != -100)
    if n_loss_tokens < 2:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_example_parity(
    record: dict,
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> tuple[Optional[dict], str]:
    """Tokenise with the EXACT evaluation prompt (see shared/eval_parity.py).

    Ported verbatim from LLaMA/data_loader.py (audit 2026-09-23 §1, applied to
    Mistral 2026-09-28) -- this function is model-agnostic: it only calls
    tokenizer.apply_chat_template on the shared eval builder's output. Context
    is truncated by tokens from the end, never the question or instructions.
    A row whose extractive gold sits in the dropped part of its context is
    DROPPED rather than trained on. No " [truncated]" marker is appended --
    evaluation contexts never carry one.
    """
    ex = row_to_example(record, example["question"], example["context"],
                        example["answer"])
    answer = example["answer"]

    def render(e):
        prompt = render_prompt(tokenizer, e)
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": _build_zero_shot_prompt(e)},
             {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False, date_string=PINNED_DATE)
        return prompt, full

    prompt_text, full_text = render(ex)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        if not ex.context:
            return None, "too_long"
        over = len(full_ids) - max_length
        ctx_ids = tokenizer(ex.context, add_special_tokens=False)["input_ids"]
        keep = len(ctx_ids) - over - 16  # margin for boundary re-tokenisation
        if keep < 32:
            return None, "too_long"
        ctx = tokenizer.decode(ctx_ids[:keep])
        cut = ctx.rfind(". ")
        if cut > len(ctx) // 2:
            ctx = ctx[: cut + 1]
        parts = example.get("answer_parts") or [answer]
        full_ctx = ex.context.lower()
        if any(p and p.lower() in full_ctx and p.lower() not in ctx.lower()
               for p in parts):
            return None, "gold_truncated"
        ex = with_context(ex, ctx)
        prompt_text, full_text = render(ex)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_length:
            return None, "too_long"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None, "prefix_mismatch"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if sum(1 for l in labels if l != -100) < 2:
        return None, "empty_answer"
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}, "ok"


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------

def normalize_and_tokenize(
    records: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    logger: logging.Logger | None = None,
    prompt_format: str = "legacy",
) -> Dataset:
    """Normalise raw JSONL records, tokenise, and return a HF Dataset.

    Skips Temprel and any record that cannot be converted.
    ``prompt_format``: "legacy" (original Mistral arms) or "eval_parity"
    (train on the exact evaluation prompt; shared/eval_parity.py).
    """
    if prompt_format not in ("legacy", "eval_parity"):
        raise ValueError(f"unknown prompt_format {prompt_format!r}")
    log = logger or setup_logger("mistral.data_loader")
    skipped_by_source: dict[str, int] = {}
    total_by_source: dict[str, int] = {}
    tokenized: list[dict] = []

    for rec in records:
        src = rec.get("source_dataset", "unknown")
        total_by_source[src] = total_by_source.get(src, 0) + 1
        norm = normalize_record(rec)
        if norm is None:
            skipped_by_source[src] = skipped_by_source.get(src, 0) + 1
            continue

        if prompt_format == "eval_parity":
            tok, why = tokenize_example_parity(rec, norm, tokenizer, max_length)
            if tok is None:
                key = f"parity_{why}"
                skipped_by_source[key] = skipped_by_source.get(key, 0) + 1
                continue
        else:
            tok = tokenize_example(norm, tokenizer, max_length)
        if tok is None:
            skipped_by_source["tokenize_error"] = skipped_by_source.get("tokenize_error", 0) + 1
            continue

        tokenized.append(tok)

    for src, count in sorted(skipped_by_source.items()):
        log.info("  Skipped %d records from source=%s", count, src)

    # A source losing a large share of its rows means the builder wrote a
    # schema this loader does not read. That is how the v3 cycle lost its
    # entire TLQA slice (3,253 rows) while the log showed only a benign-
    # looking "tokenize_error" count.
    for src, count in sorted(skipped_by_source.items()):
        if src in SKIPPED_DATASETS or src == "tokenize_error" or src.startswith("parity_"):
            continue
        total = total_by_source.get(src, 0)
        if total and count / total > 0.05:
            log.warning(
                "  %s: dropped %d/%d rows (%.1f%%) — check that the builder "
                "writes an answer key this loader reads",
                src, count, total, 100 * count / total,
            )
    log.info("Tokenised %d / %d records", len(tokenized), len(records))

    return Dataset.from_dict({
        "input_ids":      [t["input_ids"] for t in tokenized],
        "attention_mask":  [t["attention_mask"] for t in tokenized],
        "labels":          [t["labels"] for t in tokenized],
    })


def load_flat_datasets(
    train_path: str | Path,
    val_path: str | Path,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    logger: logging.Logger | None = None,
    prompt_format: str = "legacy",
) -> tuple[Dataset, Dataset]:
    """Load and tokenise train + val JSONL files for flat training.

    Returns (train_dataset, val_dataset).
    """
    log = logger or setup_logger("mistral.data_loader")

    log.info("Loading training data: %s", train_path)
    train_records = read_jsonl(train_path)
    log.info("Loading validation data: %s", val_path)
    val_records = read_jsonl(val_path)

    log.info("--- Training set ---")
    train_ds = normalize_and_tokenize(train_records, tokenizer, max_length, log,
                                      prompt_format=prompt_format)
    log.info("--- Validation set ---")
    val_ds = normalize_and_tokenize(val_records, tokenizer, max_length, log,
                                    prompt_format=prompt_format)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# CLI — quick inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mistral data loader inspection")
    parser.add_argument("--train", default="../../data/combined_80_20_split/train.jsonl")
    parser.add_argument("--val", default="../../data/combined_80_20_split/val.jsonl")
    parser.add_argument("--model", default="../../models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    tok = load_tokenizer(args.model)
    train_ds, val_ds = load_flat_datasets(args.train, args.val, tok, args.max_length)
    print(f"Train: {len(train_ds)} examples")
    print(f"Val:   {len(val_ds)} examples")
    print(f"Sample input_ids length: {len(train_ds[0]['input_ids'])}")
