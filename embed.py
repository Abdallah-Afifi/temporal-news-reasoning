#!/usr/bin/env python3
"""
embed.py
=========
Build step 7: build the ``embed`` string that the sentence-transformer will encode.

Simple summary
--------------
Each chunk already has a ``text`` field. This script copies that text directly
into the ``embed`` field. Title and date prefixes are not added because the
temporal filter already handles time using T_start/T_end, and the chunk text
already contains the topic. Using raw text gives the encoder the full 512-token
budget for actual content.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Temporal IE output JSONL]  →  THIS MODULE  →  [Encoder (encoder.py)]

What it writes
--------------
Every input JSON object is copied to the output with one new field added:

    ``embed`` - the raw chunk text sent to the encoder, e.g.:
        "Football Manager is back for another year and this time..."

You can process one file (``--input`` / ``--output``) or many numbered shards under
``--input-dir``. Writes a small stats JSON so you can confirm every row looks valid.

Paths are resolved from the repo root via ``PROJECT_ROOT``.

Usage
-----
    # Single file:
    python -m temporal_rag.embed \\
        --input  temporal_ie_output.jsonl \\
        --output embed_output.jsonl

    # Sharded input directory:
    python -m temporal_rag.embed --input-dir chunks_rag_clean_output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT


DEFAULT_INPUT_DIR = Path("chunks_rag_clean_output")
DEFAULT_OUTPUT_DIR = Path("hunks_rag_clean_output_embed")
DEFAULT_STATS_NAME = "stats_embed.json"
DEFAULT_FIELD_NAME = "embed"


@dataclass
class FileMetrics:
    """Counts reads/writes and embed lengths for one shard."""

    input_file: str
    output_file: str
    rows_read: int = 0
    rows_written: int = 0
    rows_skipped_empty_text: int = 0
    rows_missing_title: int = 0
    rows_missing_published_date: int = 0
    min_embed_length: int | None = None
    max_embed_length: int | None = None
    total_embed_length: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI: either one input/output pair or a range of shard files."""
    parser = argparse.ArgumentParser(
        description=(
            "Add an embed field to each JSON line. "
            "Use --input and --output for one file, "
            "or --input-dir / --output-dir for numbered shard files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Single input JSONL. Needs --output. Same Title:/Date: prefix rules as shards."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single output JSONL when --input is set.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing chunks_00001.jsonl to chunks_00009.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where transformed JSONL shards will be written.",
    )
    parser.add_argument(
        "--stats-name",
        default=DEFAULT_STATS_NAME,
        help="Stats JSON filename written inside the output directory.",
    )
    parser.add_argument(
        "--field-name",
        default=DEFAULT_FIELD_NAME,
        help="Name of the embedding-ready field added to each record.",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=1,
        help="First chunk shard number to process, inclusive.",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=9,
        help="Last chunk shard number to process, inclusive.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional global record limit, useful for sample validation runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate inputs without writing output files.",
    )
    parser.add_argument(
        "--missing-metadata-mode",
        choices=("omit", "placeholder"),
        default="omit",
        help="How to handle missing title or published_date values in the embed field.",
    )
    parser.add_argument(
        "--title-placeholder",
        default="Unknown Title",
        help="Placeholder used when title is missing and placeholder mode is enabled.",
    )
    parser.add_argument(
        "--date-placeholder",
        default="unknown",
        help="Placeholder used when published_date is missing and placeholder mode is enabled.",
    )
    args = parser.parse_args()
    if (args.input is not None) ^ (args.output is not None):
        parser.error("--input and --output must be passed together (or omit both for shard mode).")
    return args


def resolve_path(base_dir: Path, path: Path) -> Path:
    """If ``path`` is relative, anchor it under ``base_dir``."""

    if path.is_absolute():
        return path
    return base_dir / path


def chunk_shard_paths(input_dir: Path, start_shard: int, end_shard: int) -> list[Path]:
    """List shard files ``chunks_{start}`` … ``chunks_{end}``; fail if any absent."""

    if start_shard > end_shard:
        raise ValueError("start_shard cannot be greater than end_shard")

    paths: list[Path] = []
    missing: list[str] = []

    for shard_num in range(start_shard, end_shard + 1):
        shard_path = input_dir / f"chunks_{shard_num:05d}.jsonl"
        if shard_path.exists():
            paths.append(shard_path)
        else:
            missing.append(shard_path.name)

    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing required shard files: {missing_list}")

    return paths


def embed_job_paths(
    base_dir: Path, args: argparse.Namespace
) -> tuple[list[tuple[Path, Path]], Path]:
    """Plan jobs: one (input, output) pair, or every shard file mapped into ``output_dir``."""
    if args.input is not None:
        assert args.output is not None
        in_path = resolve_path(base_dir, args.input)
        out_path = resolve_path(base_dir, args.output)
        if not in_path.is_file():
            raise FileNotFoundError(f"Input file not found: {in_path}")
        return [(in_path, out_path)], out_path.parent

    input_dir = resolve_path(base_dir, args.input_dir)
    output_dir = resolve_path(base_dir, args.output_dir)
    shard_paths = chunk_shard_paths(input_dir, args.start_shard, args.end_shard)
    pairs = [(p, output_dir / p.name) for p in shard_paths]
    return pairs, output_dir


def safe_string(value: Any) -> str:
    """Coerce JSON values to string; ``None`` becomes empty."""
    if value is None:
        return ""
    return str(value)


def build_embed_text(
    title: str,
    published_date: str,
    text: str,
    missing_metadata_mode: str,
    title_placeholder: str,
    date_placeholder: str,
) -> str:
    """Return the chunk text as the embedding input. Title and date are excluded.

    The temporal filter handles time filtering using T_start/T_end, so date
    prefixes in the vector are not needed. The chunk text already contains
    the topic, so the title prefix is also not needed. Using raw text gives
    the model the full 512-token budget for actual content.
    """
    return text


def update_length_metrics(metrics: FileMetrics, embed_length: int) -> None:
    """Update the min, max, and total embed length counters inside ``metrics``."""
    metrics.total_embed_length += embed_length
    if metrics.min_embed_length is None or embed_length < metrics.min_embed_length:
        metrics.min_embed_length = embed_length
    if metrics.max_embed_length is None or embed_length > metrics.max_embed_length:
        metrics.max_embed_length = embed_length


def percentile(sorted_values: list[int], pct: float) -> float | None:
    """Return the value at position ``pct`` (0.0–1.0) in a sorted list.

    For example, ``pct=0.50`` gives the median, ``pct=0.99`` gives the 99th percentile.
    Returns ``None`` when the list is empty.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * fraction


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path``."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_embed(args: argparse.Namespace) -> dict[str, Any]:
    """Scan JSONL shards, attach ``embed`` strings, accumulate validation stats, write outputs."""
    base_dir = PROJECT_ROOT
    job_paths, stats_parent = embed_job_paths(base_dir, args)
    stats_path = stats_parent / args.stats_name

    if not args.dry_run:
        stats_parent.mkdir(parents=True, exist_ok=True)

    file_metrics_list: list[FileMetrics] = []
    embed_lengths: list[int] = []
    sample_outputs: list[dict[str, Any]] = []

    total_records_read = 0
    total_records_written = 0
    total_records_skipped_empty_text = 0
    missing_title_count = 0
    missing_published_date_count = 0
    missing_both_count = 0
    embed_non_empty_count = 0
    embed_ends_with_text_count = 0
    title_prefix_present_count = 0
    date_prefix_present_count = 0
    title_prefix_expected_count = 0
    date_prefix_expected_count = 0
    records_limited = False

    single_file = args.input is not None

    for shard_path, output_path in job_paths:
        metrics = FileMetrics(
            input_file=shard_path.name, output_file=output_path.name
        )
        file_metrics_list.append(metrics)

        output_file = None
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path.open("w", encoding="utf-8")

        try:
            with shard_path.open("r", encoding="utf-8") as input_file:
                for line_no, raw_line in enumerate(input_file, start=1):
                    if (
                        args.max_records is not None
                        and total_records_written >= args.max_records
                    ):
                        records_limited = True
                        break

                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {shard_path.name} line {line_no}: {exc}"
                        ) from exc

                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Expected JSON object in {shard_path.name} line {line_no}"
                        )

                    metrics.rows_read += 1
                    total_records_read += 1

                    text = safe_string(record.get("text"))
                    title = safe_string(record.get("title"))
                    published_date = safe_string(record.get("published_date"))

                    title_present = bool(title.strip())
                    published_date_present = bool(published_date.strip())
                    text_present = bool(text.strip())

                    if not title_present:
                        metrics.rows_missing_title += 1
                        missing_title_count += 1
                    if not published_date_present:
                        metrics.rows_missing_published_date += 1
                        missing_published_date_count += 1
                    if not title_present and not published_date_present:
                        missing_both_count += 1

                    if not text_present:
                        metrics.rows_skipped_empty_text += 1
                        total_records_skipped_empty_text += 1
                        continue

                    embed = build_embed_text(
                        title=title,
                        published_date=published_date,
                        text=text,
                        missing_metadata_mode=args.missing_metadata_mode,
                        title_placeholder=args.title_placeholder,
                        date_placeholder=args.date_placeholder,
                    )

                    output_record = dict(record)
                    output_record[args.field_name] = embed

                    if output_file is not None:
                        output_file.write(
                            json.dumps(output_record, ensure_ascii=False) + "\n"
                        )

                    metrics.rows_written += 1
                    total_records_written += 1

                    embed_length = len(embed)
                    embed_lengths.append(embed_length)
                    update_length_metrics(metrics, embed_length)

                    if embed:
                        embed_non_empty_count += 1
                    if embed.endswith(text):
                        embed_ends_with_text_count += 1

                    if title_present:
                        title_prefix_expected_count += 1
                        if embed.startswith(f"Title:{title}."):
                            title_prefix_present_count += 1
                    if published_date_present:
                        date_prefix_expected_count += 1
                        date_fragment = f"Date:{published_date}."
                        if date_fragment in embed:
                            date_prefix_present_count += 1

                    if len(sample_outputs) < 3:
                        sample_outputs.append(
                            {
                                "chunk_id": output_record.get("chunk_id"),
                                "title": title,
                                "published_date": published_date,
                                "text_preview": text[:120],
                                "embed_preview": embed[:180],
                            }
                        )
        finally:
            if output_file is not None:
                output_file.close()

        avg_embed_length = None
        if metrics.rows_written:
            avg_embed_length = metrics.total_embed_length / metrics.rows_written

        print(
            f"Processed {metrics.input_file}: "
            f"read={metrics.rows_read} "
            f"written={metrics.rows_written} "
            f"skipped_empty_text={metrics.rows_skipped_empty_text} "
            f"avg_embed_length={avg_embed_length if avg_embed_length is not None else 'n/a'}"
        )

        if records_limited:
            break

    sorted_lengths = sorted(embed_lengths)
    avg_embed_length = None
    if total_records_written:
        avg_embed_length = sum(embed_lengths) / total_records_written

    input_dir = resolve_path(base_dir, args.input_dir)
    output_dir = resolve_path(base_dir, args.output_dir)

    stats: dict[str, Any] = {
        "dataset_job": "embed_input_builder",
        "guide_reference": str(
            (
                base_dir.parent / "second_stage" / "embed_input_builder_guide.pdf"
            ).resolve()
        ),
        "config": {
            "mode": "single_file" if single_file else "shards",
            "input_file": str(resolve_path(base_dir, args.input))
            if single_file
            else None,
            "output_file": str(resolve_path(base_dir, args.output))
            if single_file
            else None,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "stats_path": str(stats_path),
            "field_name": args.field_name,
            "start_shard": args.start_shard,
            "end_shard": args.end_shard,
            "max_records": args.max_records,
            "dry_run": args.dry_run,
            "missing_metadata_mode": args.missing_metadata_mode,
        },
        "files": {
            "requested": len(job_paths),
            "processed": len(file_metrics_list),
            "written": 0 if args.dry_run else len(file_metrics_list),
        },
        "records": {
            "read": total_records_read,
            "written": total_records_written,
            "skipped_empty_text": total_records_skipped_empty_text,
            "missing_title": missing_title_count,
            "missing_published_date": missing_published_date_count,
            "missing_both_title_and_published_date": missing_both_count,
            "limited_by_max_records": records_limited,
        },
        "embed_validation": {
            "records_with_non_empty_embed": embed_non_empty_count,
            "records_where_embed_ends_with_original_text": embed_ends_with_text_count,
            "records_expected_to_have_title_prefix": title_prefix_expected_count,
            "records_with_title_prefix_present": title_prefix_present_count,
            "records_expected_to_have_date_prefix": date_prefix_expected_count,
            "records_with_date_prefix_present": date_prefix_present_count,
            "all_written_records_have_non_empty_embed": embed_non_empty_count
            == total_records_written,
            "all_expected_title_prefixes_present": title_prefix_expected_count
            == title_prefix_present_count,
            "all_expected_date_prefixes_present": date_prefix_expected_count
            == date_prefix_present_count,
            "all_embeds_end_with_original_text": embed_ends_with_text_count
            == total_records_written,
        },
        "embed_length_characters": {
            "min": sorted_lengths[0] if sorted_lengths else None,
            "max": sorted_lengths[-1] if sorted_lengths else None,
            "avg": avg_embed_length,
            "p50": percentile(sorted_lengths, 0.50),
            "p90": percentile(sorted_lengths, 0.90),
            "p99": percentile(sorted_lengths, 0.99),
        },
        "sample_outputs": sample_outputs,
        "per_file_breakdown": [],
    }

    for metrics in file_metrics_list:
        avg_length = None
        if metrics.rows_written:
            avg_length = metrics.total_embed_length / metrics.rows_written

        stats["per_file_breakdown"].append(
            {
                **asdict(metrics),
                "avg_embed_length": avg_length,
            }
        )

    if not args.dry_run:
        write_json(stats_path, stats)

    return stats


def main() -> None:
    """Parse command-line arguments, run the embed builder, and print a final summary."""
    args = parse_args()
    stats = process_embed(args)

    print("\n=== Final Summary ===")
    print(f"Shards processed: {stats['files']['processed']}")
    print(f"Records read: {stats['records']['read']}")
    print(f"Records written: {stats['records']['written']}")
    print(f"Missing title count: {stats['records']['missing_title']}")
    print(f"Missing published_date count: {stats['records']['missing_published_date']}")
    print(
        "Embed validation all green: "
        f"{stats['embed_validation']['all_written_records_have_non_empty_embed'] and stats['embed_validation']['all_expected_title_prefixes_present'] and stats['embed_validation']['all_expected_date_prefixes_present'] and stats['embed_validation']['all_embeds_end_with_original_text']}"
    )
    if not args.dry_run:
        print(f"Stats written to: {Path(stats['config']['stats_path'])}")


if __name__ == "__main__":
    main()
