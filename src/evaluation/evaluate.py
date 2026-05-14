"""Evaluation harness — run models against benchmarks."""


class EvaluationHarness:
    """Run evaluation across benchmarks and models."""

    def __init__(self, config_path: str = "./configs/evaluation_config.yaml"):
        self.config_path = config_path

    def evaluate(self, model_name: str, benchmark: str, task: str | None = None):
        """Evaluate a model on a benchmark task."""
        raise NotImplementedError

    def evaluate_all(self):
        """Run all configured evaluations."""
        raise NotImplementedError
