"""Regression tests for the finetuning data loader's answer-key handling.

The v3 cycle trained on 26% less data than it was given: its builder wrote
TLQA answers under ``targets`` only, the loader's TLQA branch read
``final_answers``/``answers``, and every one of the 3,253 rows normalized to
an empty answer. Because the branch returned a dict rather than ``None``, the
rows were dropped later by the tokenizer guard and reported as
``tokenize_error`` — a benign-looking bucket that hid the loss for a whole
training run. These tests pin both halves of that failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.finetuning.LLaMA.data_loader import normalize_record


def test_tlqa_answers_read_from_targets_key() -> None:
    """v3-shaped rows carry the answer list under ``targets`` only."""
    record = {
        "source_dataset": "TLQA",
        "question": "List all positions Guido Westerwelle held from 2010 to 2013.",
        "targets": ["Vice-Chancellor of Germany", "Federal Minister for Foreign Affairs"],
    }

    normalized = normalize_record(record)

    assert normalized is not None
    assert normalized["answer_parts"] == [
        "Vice-Chancellor of Germany",
        "Federal Minister for Foreign Affairs",
    ]


def test_tlqa_prefers_final_answers_over_targets() -> None:
    """v4-shaped rows carry both keys; final_answers stays authoritative."""
    record = {
        "source_dataset": "TLQA",
        "question": "List all employers.",
        "final_answers": ["University of Glasgow", "University of Aberdeen"],
        "targets": ["University of Glasgow"],
    }

    normalized = normalize_record(record)

    assert normalized["answer_parts"] == ["University of Glasgow", "University of Aberdeen"]


def test_timeqa_answers_read_from_final_answers_key() -> None:
    """The fallback is symmetric — TimeQA rows survive either schema."""
    record = {
        "source_dataset": "TimeQA",
        "question": "Which country did Svetogorsk belong to in 1991?",
        "context": "Svetogorsk is a town.",
        "final_answers": ["Russia"],
    }

    normalized = normalize_record(record)

    assert normalized is not None
    assert normalized["answer"] == "Russia"


def test_every_gold_answer_is_kept() -> None:
    """TLQA is a list task: truncating to the first answer under-trains it."""
    record = {
        "source_dataset": "TLQA",
        "question": "List all positions held.",
        "final_answers": ["First", "Second", "Third"],
    }

    normalized = normalize_record(record)

    assert normalized["answer"] == "First; Second; Third"
    assert len(normalized["answer_parts"]) == 3


def test_empty_answer_is_dropped_and_attributed_to_its_source() -> None:
    """An unanswerable row must return None so the skip report names TLQA,
    rather than surfacing later as an anonymous ``tokenize_error``."""
    record = {
        "source_dataset": "TLQA",
        "question": "List all positions held.",
        "targets": [""],
    }

    assert normalize_record(record) is None


def test_timeqa_empty_targets_are_dropped() -> None:
    record = {
        "source_dataset": "TimeQA",
        "question": "Which country did Svetogorsk belong to from 1991 to 1992?",
        "context": "Svetogorsk is a town.",
        "targets": [""],
    }

    assert normalize_record(record) is None
