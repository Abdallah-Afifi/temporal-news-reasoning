"""Module 1: Temporal Intent Detection.

Classifies queries into temporal categories and extracts temporal constraints.
"""

from enum import Enum


class TemporalIntent(Enum):
    """Types of temporal intent in a query."""
    RECENCY = "recency"       # "What happened recently..."
    PAST = "past"             # "What happened in 2020..."
    FUTURE = "future"         # "What will happen..."
    ATEMPORAL = "atemporal"   # "What is the capital of..."


class TemporalIntentDetector:
    """Detect temporal intent and extract constraints from queries."""

    def __init__(self):
        pass

    def detect(self, query: str) -> dict:
        """Detect temporal intent in a query.

        Args:
            query: User query string.

        Returns:
            Dict with 'intent', 'constraints', and 'reference_time'.
        """
        raise NotImplementedError
