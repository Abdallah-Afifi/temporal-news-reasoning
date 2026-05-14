 """Module 4: Temporal prompting templates.

Contains templates for:
- Timeline-based Chain-of-Thought
- Temporal context injection
- Self-consistency with temporal checking
"""


class TemporalPromptTemplates:
    """Manage temporal reasoning prompt templates."""

    @staticmethod
    def zero_shot(query: str) -> str:
        """Basic zero-shot prompt."""
        raise NotImplementedError

    @staticmethod
    def few_shot(query: str, examples: list[dict]) -> str:
        """Few-shot prompt with temporal examples."""
        raise NotImplementedError

    @staticmethod
    def timeline_cot(query: str, timeline: list[dict], context: str) -> str:
        """Timeline-based Chain-of-Thought prompt."""
        raise NotImplementedError

    @staticmethod
    def temporal_context_injection(query: str, context: str, temporal_metadata: dict) -> str:
        """Prompt with injected temporal context and metadata."""
        raise NotImplementedError
