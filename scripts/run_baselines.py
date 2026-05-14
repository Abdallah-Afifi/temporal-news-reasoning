"""Run baseline evaluations (zero-shot, few-shot) on benchmarks."""

import click


@click.command()
@click.option("--model", type=click.Choice(["qwen", "phi", "llama", "all"]), default="all")
@click.option("--benchmark", type=click.Choice(["time", "timebench", "tram", "all"]), default="all")
@click.option("--mode", type=click.Choice(["zero_shot", "few_shot", "all"]), default="all")
def run_baselines(model: str, benchmark: str, mode: str):
    """Run baseline evaluations."""
    print(f"Running baselines: model={model}, benchmark={benchmark}, mode={mode}")
    raise NotImplementedError


if __name__ == "__main__":
    run_baselines()
