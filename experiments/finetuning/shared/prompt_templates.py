"""
experiments/finetuning/shared/prompt_templates.py
=======================================
Prompt formatting utilities for temporal reasoning fine-tuning.

Provides chat-message builders that work with any HuggingFace tokenizer's
``apply_chat_template()`` method.
"""

from __future__ import annotations

TEMPORAL_SYSTEM_PROMPT = (
    "You are a temporal reasoning assistant specialized in understanding "
    "time-related questions. Analyze temporal information carefully, "
    "considering dates, durations, sequences, and temporal relationships. "
    "Provide precise, well-reasoned answers."
)


def build_chat_messages(
    question: str,
    context: str = "",
    answer: str | None = None,
    include_system: bool = True,
) -> list[dict[str, str]]:
    """Build a messages list for ``tokenizer.apply_chat_template()``.

    Args:
        question:       The question text.
        context:        Optional context passage.
        answer:         If provided, appended as the assistant turn (for training).
        include_system: Whether to prepend the temporal system prompt.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts.
    """
    messages: list[dict[str, str]] = []

    if include_system:
        messages.append({"role": "system", "content": TEMPORAL_SYSTEM_PROMPT})

    user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    messages.append({"role": "user", "content": user_content})

    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return messages


def example_to_training_text(
    example: dict,
    model_key: str = "llama",
    include_system_prompt: bool = True,
) -> str:
    """Format a normalized example as a plain-text training string.

    This is a fallback for cases where a tokenizer is not available.
    Prefer :func:`build_chat_messages` + ``tokenizer.apply_chat_template()``.
    """
    question = example.get("question", "")
    context = example.get("context", "")
    answer = example.get("answer", "")

    parts = []
    if include_system_prompt:
        parts.append(f"System: {TEMPORAL_SYSTEM_PROMPT}")

    user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    parts.append(f"User: {user_content}")
    parts.append(f"Assistant: {answer}")

    return "\n\n".join(parts)


def build_temporal_cot_prompt(
    model_key: str,
    question: str,
    context: str = "",
    reference_date: str | None = None,
) -> str:
    """Build a chain-of-thought prompt string for inference.

    Returns the raw user-content string.  For actual inference, pass this
    through ``tokenizer.apply_chat_template()`` with appropriate messages.
    """
    parts: list[str] = []
    if context:
        parts.append(f"Context:\n{context}")
    if reference_date:
        parts.append(f"Reference Date: {reference_date}")
    parts.append(f"Question: {question}")
    parts.append("\nThink step by step about the temporal aspects, then provide your answer.")
    return "\n\n".join(parts)


def build_implicit_reference_prompt(
    model_key: str,
    question: str,
    context: str = "",
) -> str:
    """Build a prompt for implicit temporal reference resolution."""
    return build_temporal_cot_prompt(model_key, question, context)


def format_prompt(
    model_key: str,
    question: str,
    context: str = "",
) -> str:
    """Generic prompt formatter."""
    return build_temporal_cot_prompt(model_key, question, context)
