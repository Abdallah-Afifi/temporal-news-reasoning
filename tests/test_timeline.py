"""Tests for timeline builder."""

from src.rag.timeline_builder import TimelineBuilder


class TestTimelineBuilder:
    def test_placeholder(self):
        builder = TimelineBuilder()
        assert builder is not None
