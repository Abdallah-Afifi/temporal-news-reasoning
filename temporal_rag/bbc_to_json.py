#!/usr/bin/env python3
"""
Convert BBC parquet files into one JSONL file per article.

What each output line looks like::

    {"title": "...", "plain_text": "...", "published_date": "YYYY-MM-DD"}

Rows are de-duplicated by hashing the article body so the same story is not written twice.

Examples::

    python -m temporal_rag.bbc_to_json --input train-00000-of-00001.parquet --output bbc_standardized.jsonl
    python -m temporal_rag.bbc_to_json --input train-00000-of-00001.parquet --output out.jsonl --max-rows 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FileFailure:
    """One parquet file path plus the error string when reading fails."""

    path: str
    error: str


def normalize_text(value: Any) -> str:
    """Return text from a table cell, or empty string if missing."""

    if value is None or pd.isna(value):
        return ""
    return str(value)


def normalize_date(value: Any) -> str | None:
    """Return a calendar date as YYYY-MM-DD, or None if parsing fails."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        # Keep already standardized dates as-is.
        if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
            return stripped

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def main() -> None:
    """Read CLI flags, copy parquet rows into JSONL, dedupe by body hash."""
    parser = argparse.ArgumentParser(description="Convert BBC parquet → JSONL")
    parser.add_argument(
        "--input", type=Path, default=Path("train-00000-of-00001.parquet"),
        help="Path to the parquet file (default: train-00000-of-00001.parquet)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("bbc_standardized.jsonl"),
        help="Output JSONL path (default: bbc_standardized.jsonl)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit to first N articles (default: all)"
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    max_rows: int | None = args.max_rows

    # Accept one parquet path or a directory containing several parquet files.
    if input_path.is_dir():
        parquet_files = sorted(input_path.rglob("*.parquet"))
    else:
        parquet_files = [input_path]

    total_files = len(parquet_files)
    file_failures: list[FileFailure] = []
    row_fail_reasons: dict[str, int] = {}

    files_processed = 0
    files_failed = 0
    total_rows_found = 0
    total_rows_imported = 0
    total_rows_failed = 0
    total_rows_duplicates = 0
    rows_remaining = max_rows  # None means no row limit.
    seen_hashes: set[str] = set()  # SHA1 of body text to skip duplicate articles.

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, parquet_path in enumerate(parquet_files, start=1):
            if rows_remaining is not None and rows_remaining <= 0:
                break

            path_str = str(parquet_path)
            added_in_file = 0
            failed_rows_in_file = 0

            try:
                df = pd.read_parquet(parquet_path)
                files_processed += 1
                rows_in_file = len(df)
                total_rows_found += rows_in_file

                # Apply row limit per file
                if rows_remaining is not None:
                    df = df.iloc[:rows_remaining]

                title_col = df["title"] if "title" in df.columns else pd.Series([None] * len(df))
                content_col = df["content"] if "content" in df.columns else pd.Series([None] * len(df))
                published_col = (
                    df["published_date"]
                    if "published_date" in df.columns
                    else pd.Series([None] * len(df))
                )

                for title_val, content_val, pub_val in zip(title_col, content_col, published_col):
                    try:
                        plain_text = normalize_text(content_val)

                        # Deduplicate by SHA1 hash of the article body
                        content_hash = hashlib.sha1(
                            plain_text.encode("utf-8", errors="ignore")
                        ).hexdigest()
                        if content_hash in seen_hashes:
                            total_rows_duplicates += 1
                            continue
                        seen_hashes.add(content_hash)

                        record = {
                            "title": normalize_text(title_val),
                            "plain_text": plain_text,
                            "published_date": normalize_date(pub_val),
                        }
                        json.dump(record, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        added_in_file += 1
                    except Exception as row_exc:
                        failed_rows_in_file += 1
                        reason = f"row_transform_error: {type(row_exc).__name__}"
                        row_fail_reasons[reason] = row_fail_reasons.get(reason, 0) + 1

            except Exception as file_exc:
                files_failed += 1
                file_failures.append(FileFailure(path=path_str, error=str(file_exc)))

            total_rows_imported += added_in_file
            total_rows_failed += failed_rows_in_file
            if rows_remaining is not None:
                rows_remaining -= added_in_file

            print(
                f"[{i}/{total_files}] {path_str} | "
                f"added={added_in_file} | failed_rows={failed_rows_in_file}"
            )

    print("\n=== Summary ===")
    print(f"Files processed: {files_processed} / {total_files}")
    if file_failures:
        for failure in file_failures[:20]:
            print(f"  FAILED: {failure.path}: {failure.error}")
    print(f"Articles written   : {total_rows_imported}")
    print(f"Duplicates skipped : {total_rows_duplicates}")
    if max_rows:
        print(f"(limited to first {max_rows} unique rows as requested)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
