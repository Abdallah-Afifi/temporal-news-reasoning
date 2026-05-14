import logging
from typing import Tuple

from py_heideltime import heideltime

logger = logging.getLogger(__name__)


class HeidelTimeWrapper:
    """Wrapper for the HeidelTime temporal text processing engine."""

    def __init__(self, language: str = "English", document_type: str = "news"):
        """Initialize the HeidelTime wrapper.

        Args:
            language: The language of the text to be processed. Defaults to "English".
            document_type: The document type, which influences HeidelTime heuristics.
                Supported values: "news", "narrative", "colloquial", "scientific". Defaults to "news".
        """
        self.language = language
        self.document_type = document_type

    def _extract_min_max_dates(self, results: list[dict], fallback_date: str) -> Tuple[str, str]:
        """Extract the minimum and maximum boundaries from the TIMEX3 extraction results.

        Args:
            results: The JSON output from py_heideltime.heideltime() containing extracted tags.
            fallback_date: Date to fallback to if no valid dates are found in the extraction results.

        Returns:
            Tuple with min (start) date and max (end) date in ISO-8601 format.
        """
        valid_dates = []

        for item in results:
            if item.get("type") == "DATE":
                val = item.get("value")
                if val and not val.startswith("UNDEF"):  # Ignore undefined/vague dates.
                    valid_dates.append(val)

        if not valid_dates:
            return fallback_date, fallback_date

        sorted_dates = sorted(valid_dates)
        return sorted_dates[0], sorted_dates[-1]

    def extract_temporal_bounds(self, text: str, document_creation_time: str) -> Tuple[str, str]:
        """Extract temporal bounds (T_start, T_end) from text chunks during data ingestion.

        Args:
            text: The chunk of text to process.
            document_creation_time: Document Creation Time (DCT, normally publication date) to act
                as an anchor for relative time expressions like "last week". Must be ISO-8601 format (e.g. "YYYY-MM-DD").

        Returns:
            Tuple containing the T_start and T_end boundaries in ISO-8601 format. If no dates are found, both
            fall back to the document_creation_time.
        """
        try:
            results = heideltime(
                text,
                language=self.language,
                document_type=self.document_type,
                dct=document_creation_time
            )
            return self._extract_min_max_dates(results, document_creation_time)
        except Exception as e:
            logger.error(f"HeidelTime extraction failed during bound processing: {e}")
            return document_creation_time, document_creation_time

    def parse_query_range(self, query: str, reference_date: str) -> Tuple[str, str]:
        """Parse natural language queries and extract temporal search boundaries.

        Args:
            query: The natural language search query.
            reference_date: The date to use as a contemporary anchor for relative queries (e.g. today's date). Must be ISO-8601.

        Returns:
            Tuple containing the start date and end date bounds (q_start, q_end) in ISO-8601 format. If no bound is found,
            it falls back to the reference_date.
        """
        try:
            results = heideltime(
                query,
                language=self.language,
                document_type="colloquial",  # Queries are typically colloquial/conversational.
                dct=reference_date
            )
            return self._extract_min_max_dates(results, reference_date)
        except Exception as e:
            logger.error(f"HeidelTime extraction failed during query parsing: {e}")
            return reference_date, reference_date
