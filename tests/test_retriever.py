"""Tests for semantic retriever."""

from src.rag.semantic_retriever import SemanticRetriever


class TestSemanticRetriever:
    def test_placeholder(self):
        retriever = SemanticRetriever()
        assert retriever is not None
