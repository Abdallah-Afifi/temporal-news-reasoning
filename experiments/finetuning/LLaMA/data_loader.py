"""
experiments/finetuning/LLaMA/data_loader.py
==================================
Data loading and tokenization for LLaMA-3.2-3B-Instruct LoRA fine-tuning.

Handles the three data schemas in combined_80_20_split:
  - **TLQA**:   question, answers/final_answers, subject  (no context)
  - **TimeQA**: question, context, targets
  - **Temprel**: tlinks, timeline, entities — **SKIPPED** (no QA format)

Each valid record is normalised to ``{question, context, answer}`` and then
formatted using the LLaMA chat template with label masking so that the loss
is computed only on the assistant's response tokens.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEY = "llama"
HF_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BOS_TOKEN_ID = 128000   # <|begin_of_text|>
EOT_TOKEN_ID = 128009   # <|eot_id|>

SKIPPED_DATASETS = {"Temprel"}  # no question field — not usable for QA training


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name_or_path: str = HF_MODEL_NAME,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Load and configure the LLaMA 3.x tokenizer.

    - ``pad_token`` → ``eos_token`` (LLaMA has no native pad token).
    - ``padding_side`` → ``"right"`` for training.
    """
    log = setup_logger(f"{MODEL_KEY}.tokenizer")
    log.info("Loading LLaMA tokenizer from: %s", model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
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
    # Try to cut at a sentence boundary
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
    """Tokenise a normalised example using the LLaMA chat template.

    If the context is too long, it is **truncated** (not discarded) so the
    answer always fits within ``max_length``.  Labels are set to ``-100`` for
    all prompt tokens (system + user) — loss is computed only on the
    assistant's response.

    Returns a dict with ``input_ids``, ``attention_mask``, ``labels``.
    """
    question = example["question"]
    context = example["context"]
    answer = example["answer"]

    # ------------------------------------------------------------------
    # Smart context truncation: estimate token overhead and shrink context
    # so the answer always fits.  Rough heuristic: 1 token ≈ 3.5 chars.
    # ------------------------------------------------------------------
    CHARS_PER_TOKEN = 3.5
    # Overhead = system prompt + chat-template markup + question + answer
    overhead_chars = 600 + len(question) + len(answer)
    overhead_tokens = int(overhead_chars / CHARS_PER_TOKEN) + 50  # safety margin
    max_context_tokens = max_length - overhead_tokens
    if max_context_tokens < 50:
        max_context_tokens = 50  # keep at least something
    max_context_chars = int(max_context_tokens * CHARS_PER_TOKEN)

    if len(context) > max_context_chars:
        context = _truncate_context(context, max_context_chars)

    # ------------------------------------------------------------------
    # Build chat messages & tokenise
    # ------------------------------------------------------------------
    full_messages = build_chat_messages(
        question=question, context=context, answer=answer, include_system=True,
    )
    prompt_messages = build_chat_messages(
        question=question, context=context, answer=None, include_system=True,
    )

    try:
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
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


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------

def normalize_and_tokenize(
    records: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    logger: logging.Logger | None = None,
) -> Dataset:
    """Normalise raw JSONL records, tokenise, and return a HF Dataset.

    Skips Temprel and any record that cannot be converted.
    """
    log = logger or setup_logger("llama.data_loader")
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
        if src in SKIPPED_DATASETS or src == "tokenize_error":
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
    max_length: int = 2048,
    logger: logging.Logger | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and tokenise train + val JSONL files for flat training.

    Returns (train_dataset, val_dataset).
    """
    log = logger or setup_logger("llama.data_loader")

    log.info("Loading training data: %s", train_path)
    train_records = read_jsonl(train_path)
    log.info("Loading validation data: %s", val_path)
    val_records = read_jsonl(val_path)

    log.info("--- Training set ---")
    train_ds = normalize_and_tokenize(train_records, tokenizer, max_length, log)
    log.info("--- Validation set ---")
    val_ds = normalize_and_tokenize(val_records, tokenizer, max_length, log)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# LLaMADataLoader class (backward compat for evaluate.py / inference.py)
# ---------------------------------------------------------------------------

class LLaMADataLoader:
    """Backward-compatible wrapper — delegates to flat loading."""

    def __init__(self, data_root: str | Path = ".", include_system_prompt: bool = True):
        self.data_root = Path(data_root)
        self.log = setup_logger("llama.data_loader")

    def load_tokenizer(self, model_name_or_path: str = HF_MODEL_NAME) -> AutoTokenizer:
        return load_tokenizer(model_name_or_path)

    def load_stage(
        self, stage_name: str, tokenizer: AutoTokenizer,
        max_length: int = 2048, eval_ratio: float = 0.02,
        seed: int = 42, stage_path: Optional[str | Path] = None,
    ) -> tuple[Dataset, Optional[Dataset]]:
        from experiments.finetuning.shared.data_loader import CurriculumDataLoader
        base = CurriculumDataLoader(model_key=MODEL_KEY, data_root=self.data_root, logger=self.log)
        return base.load_stage(stage_name, tokenizer, max_length, eval_ratio, seed, stage_path)


# ---------------------------------------------------------------------------
# CLI — quick inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaMA data loader inspection")
    parser.add_argument("--train", default="../../data/combined_80_20_split/train.jsonl")
    parser.add_argument("--val", default="../../data/combined_80_20_split/val.jsonl")
    parser.add_argument("--model", default="../../models/Llama-3.2-3B-Instruct")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tok = load_tokenizer(args.model)
    train_ds, val_ds = load_flat_datasets(args.train, args.val, tok, args.max_length)
    print(f"Train: {len(train_ds)} examples")
    print(f"Val:   {len(val_ds)} examples")
    print(f"Sample input_ids length: {len(train_ds[0]['input_ids'])}")
