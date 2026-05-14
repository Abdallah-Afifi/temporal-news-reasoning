"""Unified benchmark data loader for TIME, TIMEBENCH, and TRAM."""

from pathlib import Path


class BenchmarkLoader:
    """Load and standardize benchmark datasets."""

    SUPPORTED_BENCHMARKS = ["time", "timebench", "tram"]

    def __init__(self, data_dir: str = "./data/benchmarks"):
        self.data_dir = Path(data_dir)

    def load(self, benchmark: str, task: str | None = None, split: str = "test"):
        """Load a benchmark dataset.

        Args:
            benchmark: One of 'time', 'timebench', 'tram'.
            task: Specific task within the benchmark (optional).
            split: Data split to load.

        Returns:
            List of examples in standardized format.
        """
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Supported: {self.SUPPORTED_BENCHMARKS}"
            )
        raise NotImplementedError(f"Loader for {benchmark} not yet implemented.")
