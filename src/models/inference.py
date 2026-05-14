"""Module 4: Unified inference interface for all SLMs."""


class TemporalModelInference:
    """Unified inference for Qwen, Phi, and LLaMA models."""

    SUPPORTED_MODELS = ["qwen", "phi", "llama"]

    def __init__(self, model_name: str, lora_path: str | None = None):
        self.model_name = model_name
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer."""
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """Generate a response from the model."""
        raise NotImplementedError
