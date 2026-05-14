#!/usr/bin/env python3
"""Quick test: embed one question and print nearest chunks from FAISS.

Uses model ``all-MiniLM-L6-v2`` (384 floats), same as the main encoder. Expects vectors
stored for inner-product search (often L2-normalized at index build time).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .paths import PROJECT_ROOT


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the FAISS query tool."""
    p = argparse.ArgumentParser(description="Query the FAISS index built by build_faiss_database.py.")
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (if omitted, reads one line from stdin).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory with faiss_hnsw_ip.index, faiss_vector_lookup.jsonl, faiss_index_manifest.json.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of neighbors to return.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Must match encoder model.")
    p.add_argument(
        "--model-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "encoder/model_cache",
        help="Sentence-transformers cache (same as encoder).",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Optional JSONL (e.g. output/deref.jsonl) to show text previews by chunk_id.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='"auto", "cpu", or "cuda" / "cuda:0" (same idea as encoder).',
    )
    return p.parse_args()


def choose_device(requested: str) -> str:
    """Return the compute device to use: honour an explicit request, or auto-detect cuda/cpu."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Divide each row vector by its L2 norm so inner-product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm query vector; cannot normalize for IP search")
    return vectors / norms


def load_lookup(path: Path) -> list[dict]:
    """Read faiss_vector_lookup.jsonl and return rows sorted by global_vector_id."""
    rows_by_id: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gid = int(r["global_vector_id"])
            rows_by_id[gid] = r
    if not rows_by_id:
        raise SystemExit(f"No rows in {path}")
    n = len(rows_by_id)
    expected = set(range(n))
    got = set(rows_by_id.keys())
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise SystemExit(
            f"Lookup global_vector_id must be 0..{n - 1} contiguous. "
            f"missing sample {missing} extra sample {extra}"
        )
    return [rows_by_id[i] for i in range(n)]


def load_chunk_texts(path: Path) -> dict[int, str]:
    """Return a mapping of chunk_id (int) to text string from a JSONL file."""
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cid = o.get("chunk_id")
            if cid is not None:
                out[int(cid)] = str(o.get("text", ""))
    return out


def main() -> None:
    """Embed the query, search the FAISS index, and print the top matching chunks."""
    args = parse_args()
    q = args.query
    if q is None:
        q = sys.stdin.readline().strip()
    if not q:
        print("Usage: query_faiss.py 'your question here'", file=sys.stderr)
        print("   or: echo 'your question' | query_faiss.py", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir
    index_path = out_dir / "faiss_hnsw_ip.index"
    lookup_path = out_dir / "faiss_vector_lookup.jsonl"
    manifest_path = out_dir / "faiss_index_manifest.json"

    for p in (index_path, lookup_path, manifest_path):
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            sys.exit(2)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dim = int(manifest.get("vector_dim", EXPECTED_DIM))
    if manifest.get("normalize_before_add") is not True:
        print(
            "Warning: index was built with normalize_before_add!=true; "
            "query normalization may not match.",
            file=sys.stderr,
        )

    lookup_rows = load_lookup(lookup_path)
    index = faiss.read_index(str(index_path))
    if index.d != dim:
        print(f"Index dim {index.d} != manifest {dim}", file=sys.stderr)
        sys.exit(2)
    if index.ntotal != len(lookup_rows):
        print(
            f"Warning: index.ntotal={index.ntotal} but lookup rows={len(lookup_rows)}",
            file=sys.stderr,
        )

    device = choose_device(args.device)
    model = SentenceTransformer(
        args.model,
        device=device,
        cache_folder=str(args.model_cache_dir),
    )
    if model.get_sentence_embedding_dimension() != dim:
        print("Model embedding dim does not match index dim.", file=sys.stderr)
        sys.exit(2)

    vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
    vec = l2_normalize_rows(vec).astype(np.float32)
    k = min(args.top_k, index.ntotal) if index.ntotal else 0
    if k <= 0:
        print("Index is empty.", file=sys.stderr)
        sys.exit(2)

    scores, ids = index.search(vec, k)
    scores_row, ids_row = scores[0], ids[0]

    texts: dict[int, str] = {}
    if args.chunks and args.chunks.is_file():
        texts = load_chunk_texts(args.chunks)

    print(f"Query: {q!r}")
    print(f"Model: {args.model} | device: {device} | metric: inner_product (higher is better)")
    print("-" * 72)
    rank = 0
    for vid, sc in zip(ids_row, scores_row):
        rank += 1
        if vid < 0:
            continue
        meta = lookup_rows[int(vid)]
        cid = meta.get("chunk_id")
        title = meta.get("title", "")
        line = (
            f"{rank}. score={float(sc):.4f}  chunk_id={cid}  "
            f"doc={meta.get('source_doc_id')!r}  {title[:76]!r}"
        )
        print(line)
        if cid is not None and int(cid) in texts:
            preview = texts[int(cid)].replace("\n", " ")[:220]
            print(f"   text: {preview!r}...")

    print("-" * 72)
    print("If titles look relevant to your question, FAISS + encoder are wired correctly.")


if __name__ == "__main__":
    main()
