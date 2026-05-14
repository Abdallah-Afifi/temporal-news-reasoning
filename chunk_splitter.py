#!/usr/bin/env python3
"""
chunk_splitter.py
==================
Build step 2: split raw articles into overlapping text chunks.

Simple summary
--------------
Reads a JSONL file of articles (one per line). For each article, it calls
``RecursiveCharacterSplitter`` to break the text into smaller windows. Each
window becomes one row in the output JSONL with its own ``chunk_id``.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Raw Article JSONL]  →  THIS MODULE  →  [Coref Resolver (optional) / GLiNER Extractor]

Output fields per chunk
-----------------------
    chunk_id         TEXT  - unique ID (e.g. ``doc_0_chunk_3``)
    text             TEXT  - the chunk text
    title            TEXT  - article title
    published_date   TEXT  - YYYY-MM-DD (from input)
    source_doc_id    TEXT  - which article this came from
    chunk_index      INT   - position of this chunk inside the article (0-based)
    total_chunks     INT   - how many chunks this article produced
    token_count      INT   - number of tokens in this chunk

Usage
-----
    python -m temporal_rag.chunk_splitter \\
        --input  bbc_standardized.jsonl \\
        --output chunks.jsonl

    # Process only the first 50 articles (useful for testing):
    python -m temporal_rag.chunk_splitter \\
        --input  bbc_standardized.jsonl \\
        --output chunks.jsonl \\
        --max-docs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recursive_splitter import RecursiveCharacterSplitter


def main() -> None:
    """Command-line entry: read JSONL articles and write JSONL chunk rows."""
    parser = argparse.ArgumentParser(description="Split articles into chunks")
    parser.add_argument(
        "--input", type=Path, default=Path("bbc_standardized.jsonl"),
        help="JSONL file produced by bbc_to_json.py (default: bbc_standardized.jsonl)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("chunks.jsonl"),
        help="Output JSONL of chunks (default: chunks.jsonl)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Process only the first N articles (default: all)"
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=512,
        help="Target chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=50,
        help="Overlap between consecutive chunks in tokens (default: 50)"
    )
    args = parser.parse_args()

    splitter = RecursiveCharacterSplitter(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    total_docs = 0
    total_chunks = 0

    with args.input.open("r", encoding="utf-8") as in_f, \
         args.output.open("w", encoding="utf-8") as out_f:

        for line in in_f:
            line = line.strip()
            if not line:
                continue
            if args.max_docs is not None and total_docs >= args.max_docs:
                break

            article = json.loads(line)
            text = article.get("plain_text", "").strip()
            title = article.get("title", "")
            published_date = article.get("published_date")

            if not text:
                print(f"  [doc {total_docs}] Skipping empty article: {title!r}")
                total_docs += 1
                continue

            source_doc_id = f"doc_{total_docs}"

            chunks = splitter.split(
                text=text,
                source_doc_id=source_doc_id,
                title=title,
                published_date=published_date,
            )

            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "title": title,
                    "published_date": published_date,
                    "source_doc_id": source_doc_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "token_count": chunk.token_count,
                }
                json.dump(record, out_f, ensure_ascii=False)
                out_f.write("\n")
                total_chunks += 1

            print(
                f"  [doc {total_docs}] '{title[:60]}' -> {len(chunks)} chunk(s)"
            )
            total_docs += 1

    print(f"\n=== Chunking complete ===")
    print(f"Documents processed : {total_docs}")
    print(f"Total chunks written: {total_chunks}")
    print(f"Output              : {args.output}")


if __name__ == "__main__":
    main()
