"""Download benchmark datasets."""

import os


DATASETS = {
    "time": {
        "source": "huggingface",
        "repo": "TBD",
        "output": "./data/benchmarks/time",
    },
    "timebench": {
        "source": "github",
        "url": "https://github.com/zchuz/TimeBench",
        "output": "./data/benchmarks/timebench",
    },
    "tram": {
        "source": "github",
        "url": "https://github.com/TRAM-benchmark/TRAM",
        "output": "./data/benchmarks/tram",
    },
    "cnn_dailymail": {
        "source": "huggingface",
        "repo": "cnn_dailymail",
        "output": "./data/corpus/cnn_dailymail",
    },
}


def download_dataset(name: str):
    """Download a dataset."""
    config = DATASETS.get(name)
    if not config:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    os.makedirs(config["output"], exist_ok=True)
    print(f"Downloading {name} to {config['output']}...")
    # Implementation depends on source type
    raise NotImplementedError(f"Download logic for {name} not yet implemented.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            download_dataset(name)
    else:
        print("Available datasets:", list(DATASETS.keys()))
