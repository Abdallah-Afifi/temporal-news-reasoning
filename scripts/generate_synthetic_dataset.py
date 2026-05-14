"""Generate a news-specific synthetic temporal reasoning dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_templates import SyntheticDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="./data/synthetic/news_temporal_reasoning.jsonl")
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument(
        "--sources",
        type=str,
        default="ccnews,cnn_dailymail,cnn_stories",
        help="Comma-separated list of source corpora to use.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-articles-per-source", type=int, default=4000)
    parser.add_argument("--no-cot", action="store_true", help="Exclude rationale text from assistant messages.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Write train/validation/test splits instead of a single file.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    generator = SyntheticDataGenerator(
        seed=args.seed,
        max_articles_per_source=args.max_articles_per_source,
    )
    examples = generator.generate(
        num_examples=args.num_examples,
        sources=sources,
        include_cot=not args.no_cot,
    )
    summary = generator.validate_examples(examples)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.split:
        splits = generator.split_examples(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        output_dir = Path(args.output)
        if output_dir.suffix:
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, records in splits.items():
            write_jsonl(output_dir / f"{split_name}.jsonl", records)
    else:
        write_jsonl(Path(args.output), examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
