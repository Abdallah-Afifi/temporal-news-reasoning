"""Resolve implicit temporal references to concrete dates."""


class TemporalReferenceResolver:
    """Resolve expressions like 'recently', 'last week' to date ranges."""

    def __init__(self):
        pass

    def resolve(self, expression: str, reference_time: str | None = None) -> dict:
        """Resolve a temporal expression to a concrete date range.

        Args:
            expression: Temporal expression (e.g., 'recently', 'last month').
            reference_time: Reference time for resolution (ISO format).

        Returns:
            Dict with 'start_date', 'end_date', 'confidence'.
        """
        raise NotImplementedError
