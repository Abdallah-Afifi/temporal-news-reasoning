"""Ablation study runner."""


class AblationStudy:
    """Run ablation studies to isolate component contributions."""

    COMPONENTS = [
        "base_only",
        "base_lora",
        "base_rag",
        "base_prompting",
        "base_lora_rag",
        "base_lora_prompting",
        "base_rag_prompting",
        "full_system",
    ]

    def __init__(self, config_path: str = "./configs/evaluation_config.yaml"):
        self.config_path = config_path

    def run(self, components: list[str] | None = None):
        """Run ablation studies for specified components."""
        raise NotImplementedError
