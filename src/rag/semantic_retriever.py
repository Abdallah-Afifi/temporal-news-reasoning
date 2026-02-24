"""Stage 1: Semantic search using Sentence-BERT embeddings."""


class SemanticRetriever:
    """Retrieve documents using dense semantic similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def encode(self, texts: list[str]):
        """Encode texts to embeddings."""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        """Retrieve semantically similar documents."""
        raise NotImplementedError
