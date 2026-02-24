"""Evaluation metrics for temporal reasoning."""


def accuracy(predictions: list, references: list) -> float:
    """Compute accuracy."""
    raise NotImplementedError


def f1_score(predictions: list, references: list) -> float:
    """Compute F1 score."""
    raise NotImplementedError


def exact_match(predictions: list, references: list) -> float:
    """Compute exact match score."""
    raise NotImplementedError


def temporal_f1(predictions: list, references: list) -> float:
    """Compute temporal-specific F1 (correct temporal extraction)."""
    raise NotImplementedError


def timeline_order_accuracy(predicted_timeline: list, reference_timeline: list) -> float:
    """Compute timeline ordering accuracy (Kendall's tau)."""
    raise NotImplementedError
