"""LoRA fine-tuning script for temporal reasoning."""


def train(config_path: str):
    """Run LoRA fine-tuning with the given configuration.

    Args:
        config_path: Path to YAML configuration file.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        print("Usage: python train_lora.py <config.yaml>")
