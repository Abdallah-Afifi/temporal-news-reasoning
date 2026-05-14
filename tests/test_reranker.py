"""Tests for temporal reranker."""

from src.rag.temporal_reranker import TemporalReranker


class TestTemporalReranker:
    def test_placeholder(self):
        reranker = TemporalReranker()
        assert reranker is not None
