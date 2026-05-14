from src.rag.heideltime_wrapper import HeidelTimeWrapper


class TestHeidelTimeWrapper:
    
    def setup_method(self):
        self.wrapper = HeidelTimeWrapper()

    def test_extract_temporal_bounds_single_date(self):
        text = "The event happened on 2023-10-15."
        start, end = self.wrapper.extract_temporal_bounds(text, document_creation_time="2023-10-16")
        assert start == "2023-10-15"
        assert end == "2023-10-15"

    def test_extract_temporal_bounds_multiple_dates(self):
        text = "The war began in 1939 and ended in 1945."
        start, end = self.wrapper.extract_temporal_bounds(text, document_creation_time="2020-01-01")
        assert start == "1939"
        assert end == "1945"

    def test_extract_temporal_bounds_relative_date(self):
        text = "I went to the store yesterday."
        start, end = self.wrapper.extract_temporal_bounds(text, document_creation_time="2023-10-16")
        assert start == "2023-10-15"
        assert end == "2023-10-15"

    def test_extract_temporal_bounds_fallback(self):
        text = "There are no dates in this specific sentence."
        fallback_date = "2023-10-16"
        start, end = self.wrapper.extract_temporal_bounds(text, document_creation_time=fallback_date)
        assert start == fallback_date
        assert end == fallback_date

    def test_parse_query_range(self):
        query = "news from last year"
        reference_date = "2024-05-15"
        start, end = self.wrapper.parse_query_range(query, reference_date=reference_date)
        assert start == "2023"
        assert end == "2023"

    def test_parse_query_range_fallback(self):
        query = "what is the capital of france"
        reference_date = "2024-05-15"
        start, end = self.wrapper.parse_query_range(query, reference_date=reference_date)
        assert start == reference_date
        assert end == reference_date
