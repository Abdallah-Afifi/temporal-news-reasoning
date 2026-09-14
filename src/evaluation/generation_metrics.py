"""Metrics for free-text generation categories.

Exact match is the wrong instrument for a task whose gold is a sentence:
TimeBench's ``situated_generation`` asks the model to "generate a temporally
grounded statement", and its 115 golds are multi-sentence texts that a correct
answer can paraphrase without reproducing. Scored by EM every arm reports
0.00%, which says nothing about the model.

These functions are deliberately kept out of ``metrics.py`` and out of the live
evaluation path. The headline metric stays exact match for every category, so
one campaign is scored by one rule; generation categories are reported
*additionally*, from stored predictions, by every arm at once.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _tokens(text: Any) -> list[str]:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    s = "" if text is None else str(text).lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return s.split()


def token_f1(prediction: Any, reference: Any) -> float:
    """Token-overlap F1 between one prediction and one reference."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        # Two empty strings agree; one empty string does not.
        return float(not pred and not ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: Any, references: Any) -> float:
    """Token F1 against the best of several acceptable golds."""
    refs = references if isinstance(references, (list, tuple)) else [references]
    refs = [r for r in refs if str(r).strip()] or [""]
    return max(token_f1(prediction, r) for r in refs)


def generation_f1(
    predictions: Sequence[Any], references: Sequence[Any]
) -> float:
    """Mean best-token-F1 over a set of generation examples."""
    if not predictions:
        return 0.0
    return sum(
        best_token_f1(p, r) for p, r in zip(predictions, references)
    ) / len(predictions)
