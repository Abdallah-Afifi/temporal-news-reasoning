"""HeidelTime wrapper for temporal metadata extraction."""


class TemporalMetadataExtractor:
    """Extract temporal expressions from text using HeidelTime."""

    def __init__(self, heideltime_path: str | None = None):
        self.heideltime_path = heideltime_path

    def extract(self, text: str, document_date: str | None = None) -> list[dict]:
        """Extract temporal expressions from text.

        Args:
            text: Input text.
            document_date: Publication date of the document (ISO format).

        Returns:
            List of extracted temporal expressions with metadata.
        """
        raise NotImplementedError
