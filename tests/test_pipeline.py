"""Tests for end-to-end pipeline."""

from src.pipeline.temporal_reasoning_pipeline import TemporalReasoningPipeline


class TestPipeline:
    def test_placeholder(self):
        pipeline = TemporalReasoningPipeline()
        assert pipeline is not None
