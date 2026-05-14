#!/usr/bin/env python3
"""Merge TimeQA, Temprel, and TLQA into a single 80/20 train/val split.

The script reads each dataset from its native format, tags every record with
its source dataset, shuffles once, and writes two JSONL files:
train.jsonl and val.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "combined_80_20_split"
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_TOTAL_SIZE = 15000


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

    return [record if isinstance(record, dict) else {"value": record} for record in data]


def load_parquet_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def iter_dataset_records(dataset_dir: Path) -> Iterable[dict[str, Any]]:
    name = dataset_dir.name

    for json_path in sorted(dataset_dir.glob("*.json")):
        split_name = json_path.stem
        for record in load_json_records(json_path):
            record["source_dataset"] = name
            record["source_split"] = split_name
            yield record

    for parquet_path in sorted(dataset_dir.glob("*.parquet")):
        split_name = parquet_path.stem
        for record in load_parquet_records(parquet_path):
            record["source_dataset"] = name
            record["source_split"] = split_name
            yield record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sample_records(records: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count >= len(records):
        return list(records)
    return rng.sample(records, count)


def compute_balanced_quotas(dataset_sizes: dict[str, int], total_size: int) -> dict[str, int]:
    dataset_names = list(dataset_sizes)
    dataset_count = len(dataset_names)
    if dataset_count == 0:
        return {}

    base_target = total_size // dataset_count
    quotas = {name: min(size, base_target) for name, size in dataset_sizes.items()}
    assigned = sum(quotas.values())
    remaining = total_size - assigned

    if remaining <= 0:
        return quotas

    spare_capacity = {
        name: max(0, dataset_sizes[name] - quotas[name])
        for name in dataset_names
    }

    while remaining > 0:
        available = {name: spare for name, spare in spare_capacity.items() if spare > 0}
        if not available:
            break

        total_spare = sum(available.values())
        provisional: dict[str, int] = {}
        fractional_parts: list[tuple[float, str]] = []

        for name, spare in available.items():
            exact_share = remaining * spare / total_spare
            extra = min(spare, math.floor(exact_share))
            provisional[name] = extra
            fractional_parts.append((exact_share - extra, name))

        allocated = sum(provisional.values())
        if allocated == 0:
            for name, _ in sorted(available.items(), key=lambda item: (-item[1], item[0])):
                if remaining == 0:
                    break
                quotas[name] += 1
                spare_capacity[name] -= 1
                remaining -= 1
            continue

        for name, extra in provisional.items():
            quotas[name] += extra
            spare_capacity[name] -= extra
        remaining -= allocated

        if remaining <= 0:
            break

        for _, name in sorted(fractional_parts, reverse=True):
            if remaining == 0:
                break
            if spare_capacity[name] <= 0:
                continue
            quotas[name] += 1
            spare_capacity[name] -= 1
            remaining -= 1

    return quotas


def compute_split_counts(quotas: dict[str, int], train_ratio: float) -> dict[str, int]:
    if not quotas:
        return {}

    train_targets = {name: quota * train_ratio for name, quota in quotas.items()}
    train_counts = {name: math.floor(target) for name, target in train_targets.items()}
    remaining = int(round(sum(quotas.values()) * train_ratio)) - sum(train_counts.values())

    if remaining > 0:
        remainders = sorted(
            ((train_targets[name] - train_counts[name], name) for name in quotas),
            reverse=True,
        )
        for _, name in remainders:
            if remaining == 0:
                break
            train_counts[name] += 1
            remaining -= 1

    return train_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train.jsonl and val.jsonl will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used before splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of the merged data to place in train.",
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=DEFAULT_TOTAL_SIZE,
        help="Target total number of examples to keep across all datasets.",
    )
    args = parser.parse_args()

    dataset_dirs = [ROOT / "TimeQA", ROOT / "Temprel", ROOT / "TLQA"]
    missing_dirs = [str(path) for path in dataset_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing dataset directories: {', '.join(missing_dirs)}")

    dataset_records: dict[str, list[dict[str, Any]]] = {}
    for dataset_dir in dataset_dirs:
        dataset_records[dataset_dir.name] = list(iter_dataset_records(dataset_dir))

    if not any(dataset_records.values()):
        raise ValueError("No records were found across the three datasets.")

    dataset_sizes = {name: len(records) for name, records in dataset_records.items()}
    total_available = sum(dataset_sizes.values())
    target_total = min(args.total_size, total_available)
    quotas = compute_balanced_quotas(dataset_sizes, target_total)

    rng = random.Random(args.seed)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    train_counts = compute_split_counts(quotas, args.train_ratio)

    for dataset_name, records in dataset_records.items():
        quota = quotas.get(dataset_name, 0)
        sampled = sample_records(records, quota, rng)
        sampled_counts[dataset_name] = len(sampled)
        rng.shuffle(sampled)

        train_count = min(train_counts.get(dataset_name, 0), len(sampled))
        train_records.extend(sampled[:train_count])
        val_records.extend(sampled[train_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "val.jsonl", val_records)

    summary = {
        "total_available": total_available,
        "target_total": target_total,
        "sampled_total": len(train_records) + len(val_records),
        "train": len(train_records),
        "val": len(val_records),
        "quotas": quotas,
        "train_counts_by_dataset": train_counts,
        "sampled_counts": sampled_counts,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()