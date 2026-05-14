"""End-to-end temporal reasoning pipeline.

Query → Intent Detection → RAG Retrieval → Timeline → SLM Inference → Answer
"""


class TemporalReasoningPipeline:
    """Complete temporal reasoning pipeline."""

    def __init__(self, config_path: str = "./configs/rag_config.yaml"):
        self.config_path = config_path

    def run(self, query: str) -> dict:
        """Run the full pipeline on a query.

        Args:
            query: User query about news events.

        Returns:
            Dict with 'answer', 'confidence', 'timeline', 'sources'.
        """
        raise NotImplementedError
