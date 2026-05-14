#!/usr/bin/env python3
"""
encoder.py
===========
Build step 8: turn the ``embed`` text strings into fixed-size float vectors.

Simple summary
--------------
Reads JSONL files where each row has an ``embed`` field (built by ``embed.py``).
Passes each string through the ``all-MiniLM-L12-v2`` sentence-transformer model
and saves the resulting 384-dimensional float vectors to ``.npy`` files.
A matching ``chunk_mapping_*.jsonl`` is saved beside each vector file so that
``build_faiss_database.py`` can look up which chunk_id a vector belongs to.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Embed Input Builder (embed.py)]  →  THIS MODULE  →  [FAISS Builder (build_faiss_database.py)]

Output files (one pair per input shard)
---------------------------------------
    embeddings_00001.npy          - 2-D float32 array of shape (N, 384)
    chunk_mapping_00001.jsonl     - one JSON row per vector with chunk_id, title, dates, etc.
    stats_encoder.json            - detailed counts and validation results for the whole run

Usage (CLI)
-----------
    # Full run (reads INPUT_FILES list defined in this file):
    python -m temporal_rag.encoder

    # Single file:
    python -m temporal_rag.encoder --input output/embed.jsonl

    # Test on first 500 rows:
    python -m temporal_rag.encoder --max-records 500

Defaults assume MiniLM and paths under ``encoder/output``. Change the constants at the
top of this file if your layout differs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .paths import PROJECT_ROOT


# ---------- knobs you often edit ----------

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
EXPECTED_DIM = 384
BATCH_SIZE = 128

# Set to an integer like 1000 to test only the first X chunks across ALL files.
# Set to None to process everything.
TRY_FIRST_X_CHUNKS: int | None = None

# Input files: chunks_00001.jsonl ... chunks_00009.jsonl
INPUT_FILES = [
    PROJECT_ROOT / "output" / "embed_output.jsonl",
]

# Output folder required by you
OUTPUT_DIR = PROJECT_ROOT / "encoder/output"
MODEL_CACHE_DIR = PROJECT_ROOT / "encoder/model_cache"

# Device: "auto", "cpu", "cuda", "cuda:0", ...
DEVICE = "auto"

# If True, embeddings are L2-normalized before saving.
# Keep False unless you explicitly want normalized vectors saved.
NORMALIZE_EMBEDDINGS = False

REENCODE_VALIDATION_ROWS = 3
VECTOR_PREVIEW_VALUES = 8


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class FileStats:
    """Statistics collected while encoding one input shard file.

    Attributes
    ----------
    input_file:
        Name of the JSONL shard that was read.
    output_embedding_file:
        Name of the ``.npy`` file that was written.
    output_mapping_file:
        Name of the ``chunk_mapping_*.jsonl`` file that was written.
    rows_counted:
        How many non-empty, valid rows were found before encoding started.
        (Used to pre-allocate the output array.)
    rows_read:
        How many rows were actually read and sent to the model.
    rows_encoded:
        How many rows came back from the model successfully.
    rows_written:
        How many rows were saved to the ``.npy`` file.
    rows_skipped:
        Rows that had an empty ``embed`` field and were skipped.
    faulty_records_count:
        Rows that could not be parsed or were missing required fields.
    faulty_record_references:
        Details about each faulty row (file name, line number, reason).
    embedding_rows_verified / mapping_rows_verified:
        Row counts confirmed by re-reading the saved files after writing.
    mapping_row_index_sequence_ok:
        ``True`` if the saved mapping file has sequential ``row_index`` values (0, 1, 2, …).
    mapping_first_row_index / mapping_last_row_index:
        First and last ``row_index`` values in the mapping file.
    batches:
        How many model-inference batches were run.
    min/max/avg_embedding_norm:
        L2 norm statistics of the raw (un-normalized) output vectors.
    sample_reencode_*:
        Results of re-encoding a few rows and comparing to the saved vectors
        (a sanity check that the file was written correctly).
    output_embedding_shape / dtype / size_bytes:
        Shape, element type, and file size of the saved ``.npy`` file.
    output_mapping_size_bytes:
        File size of the saved mapping JSONL.
    first_vector_preview / first_vector_norm:
        First 8 values and L2 norm of the very first saved vector (for quick visual inspection).
    elapsed_seconds / throughput_records_per_second:
        Timing and speed for this shard.
    """
    input_file: str
    output_embedding_file: str
    output_mapping_file: str
    rows_counted: int = 0
    rows_read: int = 0
    rows_encoded: int = 0
    rows_written: int = 0
    rows_skipped: int = 0
    faulty_records_count: int = 0
    faulty_record_references: list[dict[str, Any]] = field(default_factory=list)
    embedding_rows_verified: int = 0
    mapping_rows_verified: int = 0
    mapping_row_index_sequence_ok: bool = True
    mapping_first_row_index: int | None = None
    mapping_last_row_index: int | None = None
    batches: int = 0
    min_embedding_norm: float | None = None
    max_embedding_norm: float | None = None
    avg_embedding_norm: float | None = None
    sample_reencode_rows_checked: int = 0
    sample_reencode_max_abs_diff: float | None = None
    sample_reencode_mean_abs_diff: float | None = None
    sample_reencode_allclose: bool | None = None
    output_embedding_shape: list[int] = field(default_factory=list)
    output_embedding_dtype: str | None = None
    output_embedding_size_bytes: int = 0
    output_mapping_size_bytes: int = 0
    first_vector_preview: list[float] = field(default_factory=list)
    first_vector_norm: float | None = None
    elapsed_seconds: float = 0.0
    throughput_records_per_second: float = 0.0


@dataclass
class RunConfig:
    """All settings for one encoder run, filled from command-line arguments.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"sentence-transformers/all-MiniLM-L12-v2"``.
    expected_dim:
        The number of dimensions the chosen model outputs.  Used to validate each batch.
    batch_size:
        How many ``embed`` strings to send to the model at once.
    max_records:
        Stop after encoding this many rows across all input files (``None`` = no limit).
    input_files:
        List of JSONL shards to process, in order.
    output_dir:
        Folder where ``.npy`` and ``chunk_mapping_*.jsonl`` files are written.
    model_cache_dir:
        Local folder where downloaded model weights are cached.
    device_request:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:0"`` etc.
    normalize_embeddings:
        If ``True``, L2-normalize each vector before saving.  Keep ``False`` so
        ``build_faiss_database.py`` can normalize at index time instead.
    resume_existing:
        If ``True``, skip shards whose output files already exist.
    """
    model_name: str
    expected_dim: int
    batch_size: int
    max_records: int | None
    input_files: list[Path]
    output_dir: Path
    model_cache_dir: Path
    device_request: str
    normalize_embeddings: bool
    resume_existing: bool


# =========================
# HELPERS
# =========================

def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the encoder step."""
    parser = argparse.ArgumentParser(
        description="Encode the embed field from JSONL shards using Sentence-BERT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Single JSONL with an 'embed' field (e.g. output/embed.jsonl). "
            "When set, replaces the default INPUT_FILES list."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=TRY_FIRST_X_CHUNKS,
        help="Encode only the first N non-empty records across all shards.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used during encoding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where embeddings, mapping files, and stats are written.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=MODEL_CACHE_DIR,
        help="Local cache directory for the sentence-transformers model.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help='Device to use: "auto", "cpu", "cuda", "cuda:0", ...',
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=NORMALIZE_EMBEDDINGS,
        help="L2-normalize embeddings before saving them.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse already-generated shard outputs in the output directory and continue with missing shards.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    """Validate parsed arguments and return a ``RunConfig`` object."""
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive when provided")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.input is not None:
        input_path = (
            args.input if args.input.is_absolute() else Path.cwd() / args.input
        ).resolve()
        input_files: list[Path] = [input_path]
    else:
        input_files = INPUT_FILES

    return RunConfig(
        model_name=MODEL_NAME,
        expected_dim=EXPECTED_DIM,
        batch_size=args.batch_size,
        max_records=args.max_records,
        input_files=input_files,
        output_dir=args.output_dir,
        model_cache_dir=args.model_cache_dir,
        device_request=args.device,
        normalize_embeddings=args.normalize_embeddings,
        resume_existing=args.resume_existing,
    )


def choose_device(requested: str) -> str:
    """Return the device string to use.  ``"auto"`` picks CUDA if available, else CPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_faulty_record_reference(
    *,
    path: Path,
    line_no: int,
    reason: str,
    chunk_id: Any = None,
    null_bytes_found: int = 0,
) -> dict[str, Any]:
    """Build a small dict describing one bad input row so it can be logged in the stats."""
    return {
        "input_file": path.name,
        "line_no": line_no,
        "reason": reason,
        "chunk_id": chunk_id,
        "null_bytes_found": null_bytes_found,
        "action": "skipped",
    }


def iter_jsonl(
    path: Path,
    faulty_record_references: list[dict[str, Any]] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` pairs from a JSONL file.

    Skips blank lines, NUL-byte lines, and lines with invalid JSON.
    When ``faulty_record_references`` is a list, bad rows are appended to it
    and skipped silently.  When it is ``None``, a bad row raises ``ValueError``.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            null_bytes_found = stripped.count("\x00")
            if null_bytes_found:
                if faulty_record_references is not None:
                    faulty_record_references.append(
                        make_faulty_record_reference(
                            path=path,
                            line_no=line_no,
                            reason="Invalid JSON line contains raw NUL bytes",
                            null_bytes_found=null_bytes_found,
                        )
                    )
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if faulty_record_references is not None:
                    faulty_record_references.append(
                        make_faulty_record_reference(
                            path=path,
                            line_no=line_no,
                            reason=f"Invalid JSON: {exc.msg}",
                        )
                    )
                    continue
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc

            if not isinstance(record, dict):
                if faulty_record_references is not None:
                    faulty_record_references.append(
                        make_faulty_record_reference(
                            path=path,
                            line_no=line_no,
                            reason="Expected JSON object",
                        )
                    )
                    continue
                raise ValueError(f"Expected JSON object in {path} line {line_no}")

            yield line_no, record


def validate_record_schema(record: dict[str, Any], path: Path, line_no: int) -> list[str]:
    """Return a list of required field names that are missing from ``record``.

    An empty list means the record is valid.  Currently checks for ``chunk_id`` and ``embed``.
    """
    required_fields = ("chunk_id", "embed")
    return [field for field in required_fields if field not in record]


def safe_embed_text(record: dict[str, Any]) -> str:
    """Return the ``embed`` field as a string; returns ``""`` when the field is missing or ``None``."""
    value = record.get("embed", "")
    if value is None:
        return ""
    return str(value)


def output_paths(output_dir: Path, shard_num: int) -> tuple[Path, Path]:
    """Return the ``(embeddings_*.npy, chunk_mapping_*.jsonl)`` paths for a given shard number."""
    emb_path = output_dir / f"embeddings_{shard_num:05d}.npy"
    mapping_path = output_dir / f"chunk_mapping_{shard_num:05d}.jsonl"
    return emb_path, mapping_path


def shard_num_for_input(path: Path, file_index: int) -> int:
    """Return the shard number for an input file.

    For files named like ``chunks_00007.jsonl`` the trailing number (7) is used.
    For files with non-numeric suffixes (e.g. ``embed.jsonl``) the 1-based
    ``file_index`` position in the input list is used instead.
    """
    suffix = path.stem.split("_")[-1]
    try:
        return int(suffix)
    except ValueError:
        return file_index


def resolve_local_model_snapshot(model_name: str, model_cache_dir: Path) -> Path | None:
    """Return the path to a locally cached model snapshot, or ``None`` if not found.

    Looks inside the HuggingFace Hub cache layout under ``model_cache_dir/hub/``.
    When found, we pass this path directly to ``SentenceTransformer`` so it never
    needs a network connection.
    """
    repo_dir = model_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = repo_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return snapshot_dir

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        candidates = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    return None


def mapping_record(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Build the metadata row that maps one vector (by ``row_index``) back to its chunk."""
    return {
        "row_index": row_index,
        "chunk_id": record.get("chunk_id"),
        "source_doc_id": record.get("source_doc_id"),
        "chunk_index": record.get("chunk_index"),
        "total_chunks": record.get("total_chunks"),
        "title": record.get("title"),
        "published_date": record.get("published_date"),
    }


def count_valid_rows(
    path: Path,
    remaining_limit: int | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Do a first pass over ``path`` to count how many rows are valid and encodable.

    Returns ``(valid_count, skipped_empty_text_count, faulty_record_list)``.
    This pre-count is needed so we can allocate the output ``.npy`` array with
    the exact right shape before we start encoding.
    """
    counted = 0
    skipped = 0
    faulty_record_references: list[dict[str, Any]] = []

    for line_no, record in iter_jsonl(path, faulty_record_references):
        missing = validate_record_schema(record, path, line_no)
        if missing:
            faulty_record_references.append(
                make_faulty_record_reference(
                    path=path,
                    line_no=line_no,
                    reason=f"Missing required fields: {missing}",
                    chunk_id=record.get("chunk_id"),
                )
            )
            continue
        text = safe_embed_text(record)
        if not text.strip():
            skipped += 1
            continue

        counted += 1
        if remaining_limit is not None and counted >= remaining_limit:
            break

    return counted, skipped, faulty_record_references


def validate_vectors(vectors: np.ndarray, expected_dim: int) -> dict[str, Any]:
    """Check that a batch of vectors has the right shape and contains finite values.

    Returns a small dict with keys ``rows``, ``dim``, ``all_finite``,
    ``cosine_check_ok``, and ``sample_cosine``.
    Raises ``ValueError`` on shape or dtype problems.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {vectors.shape}")
    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}"
        )
    if not np.issubdtype(vectors.dtype, np.number):
        raise ValueError(f"Embeddings must be numeric, got {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or inf values")

    cosine_value = None
    cosine_ok = True
    if vectors.shape[0] >= 2:
        a = vectors[0]
        b = vectors[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            cosine_ok = False
        else:
            cosine_value = float(np.dot(a, b) / denom)
            cosine_ok = math.isfinite(cosine_value)

    return {
        "rows": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "all_finite": True,
        "cosine_check_ok": cosine_ok,
        "sample_cosine": cosine_value,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path``."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    """Read and return the JSON object stored at ``path``."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_encoder_stats(stats_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load per-file stats and sample validations from a previous encoder run's stats file.

    Returns ``(per_file_stats_by_input_name, sample_validations_list)``.
    Used by the ``--resume-existing`` mode to avoid re-encoding shards that are already done.
    Returns empty structures when the stats file does not exist.
    """
    if not stats_path.exists():
        return {}, []

    payload = read_json(stats_path)
    per_file = {
        item["input_file"]: item
        for item in payload.get("per_file_breakdown", [])
        if "input_file" in item
    }
    sample_validations = payload.get("sample_validations", [])
    return per_file, sample_validations


def validate_existing_output_pair(
    *,
    path: Path,
    shard_num: int,
    config: RunConfig,
    existing_stats_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check that a previously saved shard output is complete and correct.

    Reads the ``.npy`` and ``.jsonl`` files from a prior run, confirms their shapes
    and row counts match, and returns a stats dict.
    Raises ``FileNotFoundError`` or ``ValueError`` if anything looks wrong.
    """
    emb_path, mapping_path = output_paths(config.output_dir, shard_num)
    if not emb_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Existing output pair is incomplete")

    loaded = np.load(emb_path, mmap_mode="r")
    if loaded.ndim != 2 or loaded.shape[1] != config.expected_dim:
        raise ValueError(
            f"Existing embedding file {emb_path.name} has invalid shape {loaded.shape}"
        )
    if str(loaded.dtype) != "float32":
        raise ValueError(
            f"Existing embedding file {emb_path.name} has invalid dtype {loaded.dtype}"
        )

    mapping_rows = 0
    first_row_index = None
    last_row_index = None
    with mapping_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_index = record.get("row_index")
            if not isinstance(row_index, int):
                raise ValueError(f"Existing mapping file {mapping_path.name} has non-int row_index")
            if first_row_index is None:
                first_row_index = row_index
            last_row_index = row_index
            if row_index != mapping_rows:
                raise ValueError(
                    f"Existing mapping file {mapping_path.name} has non-sequential row_index"
                )
            mapping_rows += 1

    if loaded.shape[0] != mapping_rows:
        raise ValueError(
            f"Existing output mismatch for shard {shard_num}: "
            f"{emb_path.name} has {loaded.shape[0]} rows but {mapping_path.name} has {mapping_rows}"
        )

    if existing_stats_entry is not None:
        merged = dict(existing_stats_entry)
        merged["output_embedding_shape"] = [int(loaded.shape[0]), int(loaded.shape[1])]
        merged["output_embedding_dtype"] = str(loaded.dtype)
        merged["output_embedding_size_bytes"] = emb_path.stat().st_size
        merged["output_mapping_size_bytes"] = mapping_path.stat().st_size
        merged["mapping_rows_verified"] = mapping_rows
        merged["mapping_row_index_sequence_ok"] = True
        merged["mapping_first_row_index"] = first_row_index
        merged["mapping_last_row_index"] = last_row_index
        merged["embedding_rows_verified"] = int(loaded.shape[0])
        merged.setdefault("rows_written", int(loaded.shape[0]))
        merged.setdefault("rows_encoded", int(loaded.shape[0]))
        merged.setdefault("rows_read", int(loaded.shape[0]))
        merged.setdefault("rows_counted", int(loaded.shape[0]))
        merged.setdefault("rows_skipped", 0)
        merged.setdefault("faulty_records_count", 0)
        merged.setdefault("faulty_record_references", [])
        return merged

    return asdict(
        FileStats(
            input_file=path.name,
            output_embedding_file=emb_path.name,
            output_mapping_file=mapping_path.name,
            rows_counted=int(loaded.shape[0]),
            rows_read=int(loaded.shape[0]),
            rows_encoded=int(loaded.shape[0]),
            rows_written=int(loaded.shape[0]),
            rows_skipped=0,
            faulty_records_count=0,
            faulty_record_references=[],
            embedding_rows_verified=int(loaded.shape[0]),
            mapping_rows_verified=mapping_rows,
            mapping_row_index_sequence_ok=True,
            mapping_first_row_index=first_row_index,
            mapping_last_row_index=last_row_index,
            output_embedding_shape=[int(loaded.shape[0]), int(loaded.shape[1])],
            output_embedding_dtype=str(loaded.dtype),
            output_embedding_size_bytes=emb_path.stat().st_size,
            output_mapping_size_bytes=mapping_path.stat().st_size,
        )
    )


def validate_saved_outputs(
    *,
    emb_path: Path,
    mapping_path: Path,
    stats: FileStats,
    expected_dim: int,
    model: SentenceTransformer,
    normalize_embeddings: bool,
    batch_size: int,
    sampled_reencode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """After writing, re-read the saved files and verify they are correct.

    Checks:
    - The saved ``.npy`` array has the right shape and contains finite values.
    - The mapping JSONL has the same number of rows and sequential ``row_index`` values.
    - A few sampled rows, when re-encoded, produce vectors very close to what was saved.

    Updates ``stats`` in place and returns a reencode validation dict.
    Raises ``ValueError`` on any mismatch.
    """
    loaded = np.load(emb_path, mmap_mode="r")

    stats.embedding_rows_verified = int(loaded.shape[0])
    stats.output_embedding_shape = [int(x) for x in loaded.shape]
    stats.output_embedding_dtype = str(loaded.dtype)
    stats.output_embedding_size_bytes = emb_path.stat().st_size
    stats.output_mapping_size_bytes = mapping_path.stat().st_size

    if loaded.shape != (stats.rows_written, expected_dim):
        raise ValueError(
            f"Saved embedding shape mismatch for {emb_path}: "
            f"expected {(stats.rows_written, expected_dim)}, got {loaded.shape}"
        )

    if loaded.size > 0:
        sample_indices = sorted(
            {0, max(0, stats.rows_written // 2), max(0, stats.rows_written - 1)}
        )
        sampled_vectors = np.asarray(loaded[sample_indices], dtype=np.float32)
        if not np.isfinite(sampled_vectors).all():
            raise ValueError(f"Saved embedding file contains non-finite values: {emb_path}")

        first_vector = np.asarray(loaded[0], dtype=np.float32)
        stats.first_vector_norm = float(np.linalg.norm(first_vector))
        stats.first_vector_preview = [
            round(float(x), 6) for x in first_vector[:VECTOR_PREVIEW_VALUES]
        ]

    expected_row_index = 0
    stats.mapping_row_index_sequence_ok = True
    with mapping_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_index = record.get("row_index")
            if not isinstance(row_index, int):
                stats.mapping_row_index_sequence_ok = False
                break
            if stats.mapping_first_row_index is None:
                stats.mapping_first_row_index = row_index
            stats.mapping_last_row_index = row_index
            if row_index != expected_row_index:
                stats.mapping_row_index_sequence_ok = False
            expected_row_index += 1

    stats.mapping_rows_verified = expected_row_index

    if stats.mapping_rows_verified != stats.rows_written:
        raise ValueError(
            f"Mapping rows mismatch for {mapping_path}: "
            f"expected {stats.rows_written}, got {stats.mapping_rows_verified}"
        )

    if stats.rows_written > 0:
        if stats.mapping_first_row_index != 0:
            stats.mapping_row_index_sequence_ok = False
        if stats.mapping_last_row_index != stats.rows_written - 1:
            stats.mapping_row_index_sequence_ok = False

    if not stats.mapping_row_index_sequence_ok:
        raise ValueError(f"Mapping row_index sequence is invalid in {mapping_path}")

    reencode_validation: dict[str, Any] = {
        "rows_checked": 0,
        "allclose": True,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "sample_chunk_ids": [],
    }

    if sampled_reencode_rows:
        texts = [item["text"] for item in sampled_reencode_rows]
        saved_indices = [item["row_index"] for item in sampled_reencode_rows]
        reencoded = model.encode(
            texts,
            batch_size=min(batch_size, len(texts)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )
        saved_vectors = np.asarray(loaded[saved_indices], dtype=np.float32)
        reencoded = reencoded.astype(np.float32, copy=False)
        diff = np.abs(reencoded - saved_vectors)

        stats.sample_reencode_rows_checked = len(sampled_reencode_rows)
        stats.sample_reencode_max_abs_diff = float(diff.max())
        stats.sample_reencode_mean_abs_diff = float(diff.mean())
        stats.sample_reencode_allclose = bool(
            np.allclose(reencoded, saved_vectors, rtol=1e-4, atol=1e-6)
        )

        if not stats.sample_reencode_allclose:
            raise ValueError(
                f"Saved embeddings do not match re-encoded validation rows for {emb_path}"
            )

        reencode_validation = {
            "rows_checked": stats.sample_reencode_rows_checked,
            "allclose": stats.sample_reencode_allclose,
            "max_abs_diff": stats.sample_reencode_max_abs_diff,
            "mean_abs_diff": stats.sample_reencode_mean_abs_diff,
            "sample_chunk_ids": [item["chunk_id"] for item in sampled_reencode_rows],
        }

    del loaded
    return reencode_validation


def build_validation_summary(
    *,
    detected_dim: int | None,
    config: RunConfig,
    per_file_stats: list[dict[str, Any]],
    all_sample_validations: list[dict[str, Any]],
    total_rows_encoded: int,
    total_rows_written: int,
) -> dict[str, Any]:
    """Build a flat dict of True/False checks that confirm the encoding run is correct.

    Each key is a human-readable assertion (e.g. ``"all_embedding_dtypes_float32"``).
    ``"overall_validation_passed"`` is ``True`` only when every single check passes.
    Saved inside ``stats_encoder.json`` for a quick pass/fail check after the run.
    """
    summary = {
        "detected_dim_matches_expected": detected_dim == config.expected_dim,
        "output_counts_match_encoded_counts": total_rows_encoded == total_rows_written,
        "all_mapping_counts_match": all(
            item["mapping_rows_verified"] == item["rows_written"] for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_mapping_row_indexes_sequential": all(
            item["mapping_row_index_sequence_ok"] for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_embedding_shapes_match_rows_and_dim": all(
            item["output_embedding_shape"] == [item["rows_written"], config.expected_dim]
            for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_embedding_dtypes_float32": all(
            item["output_embedding_dtype"] == "float32" for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_files_have_non_negative_counts": all(
            item["rows_encoded"] >= 0 and item["rows_written"] >= 0 and item["rows_read"] >= 0
            for item in per_file_stats
        )
        if per_file_stats
        else True,
        "sample_validations_all_finite": all(
            item["validation"]["all_finite"]
            for item in all_sample_validations
            if item.get("type") == "batch_validation"
        )
        if all_sample_validations
        else True,
        "sample_validations_dim_ok": all(
            item["validation"]["dim"] == config.expected_dim
            for item in all_sample_validations
            if item.get("type") == "batch_validation"
        )
        if all_sample_validations
        else True,
        "sample_validations_cosine_ok": all(
            item["validation"]["cosine_check_ok"]
            for item in all_sample_validations
            if item.get("type") == "batch_validation"
        )
        if all_sample_validations
        else True,
        "sample_reencodes_match_saved_vectors": all(
            item["allclose"]
            for item in all_sample_validations
            if item.get("type") == "saved_vs_reencoded"
        )
        if all_sample_validations
        else True,
    }
    summary["overall_validation_passed"] = all(summary.values())
    return summary


def build_stats_payload(
    *,
    status: str,
    started_at_utc: str,
    finished_at_utc: str,
    config: RunConfig,
    device_used: str | None,
    detected_dim: int | None,
    files_processed: int,
    total_rows_counted: int,
    total_rows_read: int,
    total_rows_encoded: int,
    total_rows_written: int,
    total_rows_skipped: int,
    elapsed_seconds: float,
    per_file_stats: list[dict[str, Any]],
    all_sample_validations: list[dict[str, Any]],
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect all run data into the big JSON object written to ``stats_encoder.json``.

    Includes config settings, per-shard stats, validation summary, any error, and
    a list of removed (faulty) rows.  This file is the single source of truth for
    confirming the encoding run succeeded.
    """
    avg_rps = total_rows_encoded / elapsed_seconds if elapsed_seconds > 0 else 0.0
    removed_faulty_records = [
        faulty
        for item in per_file_stats
        for faulty in item.get("faulty_record_references", [])
    ]
    validation_summary = build_validation_summary(
        detected_dim=detected_dim,
        config=config,
        per_file_stats=per_file_stats,
        all_sample_validations=all_sample_validations,
        total_rows_encoded=total_rows_encoded,
        total_rows_written=total_rows_written,
    )

    return {
        "job_name": "sentence_encoder",
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "run_mode": "sample" if config.max_records is not None else "full",
        "model_name": config.model_name,
        "device_requested": config.device_request,
        "device_used": device_used,
        "model_cache_dir": str(config.model_cache_dir),
        "local_model_snapshot_found": str(
            resolve_local_model_snapshot(config.model_name, config.model_cache_dir)
        )
        if resolve_local_model_snapshot(config.model_name, config.model_cache_dir) is not None
        else None,
        "expected_embedding_dim": config.expected_dim,
        "detected_embedding_dim": detected_dim,
        "batch_size": config.batch_size,
        "normalize_embeddings": config.normalize_embeddings,
        "try_first_x_chunks": config.max_records,
        "input_files": [str(path) for path in config.input_files],
        "output_dir": str(config.output_dir),
        "files_processed": files_processed,
        "total_rows_counted": total_rows_counted,
        "total_rows_read": total_rows_read,
        "total_rows_encoded": total_rows_encoded,
        "total_rows_written": total_rows_written,
        "total_rows_skipped": total_rows_skipped,
        "total_faulty_records_removed": len(removed_faulty_records),
        "elapsed_seconds": elapsed_seconds,
        "average_chunks_per_second": avg_rps,
        "validation_summary": validation_summary,
        "sample_validations": all_sample_validations[:20],
        "removed_faulty_records": removed_faulty_records,
        "per_file_breakdown": per_file_stats,
        "error": error,
    }


# =========================
# CORE
# =========================

def encode_one_file(
    *,
    path: Path,
    shard_num: int,
    model: SentenceTransformer,
    config: RunConfig,
    remaining_limit: int | None,
) -> tuple[FileStats, list[dict[str, Any]]]:
    """Read one input JSONL shard, encode all rows, and write the output pair.

    Steps:
    1. Count valid rows (to pre-allocate the NumPy memory-mapped array).
    2. Stream rows in batches through the sentence-transformer model.
    3. Write vectors to ``embeddings_*.npy`` and metadata to ``chunk_mapping_*.jsonl``.
    4. Validate the saved files and re-encode a few rows to confirm correctness.

    Returns ``(FileStats, sample_validations_list)``.
    """
    rows_counted, skipped_during_count, faulty_record_references = count_valid_rows(
        path, remaining_limit
    )
    emb_path, mapping_path = output_paths(config.output_dir, shard_num)

    stats = FileStats(
        input_file=path.name,
        output_embedding_file=emb_path.name,
        output_mapping_file=mapping_path.name,
        rows_counted=rows_counted,
        rows_skipped=skipped_during_count,
        faulty_records_count=len(faulty_record_references),
        faulty_record_references=faulty_record_references,
    )

    sampled_validations: list[dict[str, Any]] = []
    sampled_reencode_rows: list[dict[str, Any]] = []

    if rows_counted == 0:
        arr = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float32, shape=(0, config.expected_dim)
        )
        del arr
        with mapping_path.open("w", encoding="utf-8"):
            pass
        stats.output_embedding_shape = [0, config.expected_dim]
        stats.output_embedding_dtype = "float32"
        stats.output_embedding_size_bytes = emb_path.stat().st_size
        stats.output_mapping_size_bytes = mapping_path.stat().st_size
        return stats, sampled_validations

    memmap_arr = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float32, shape=(rows_counted, config.expected_dim)
    )

    row_index = 0
    batch_texts: list[str] = []
    batch_records: list[dict[str, Any]] = []
    norm_sum = 0.0
    batch_index = 0

    start_time = time.perf_counter()
    progress = tqdm(
        total=rows_counted,
        desc=path.name,
        unit="chunk",
        dynamic_ncols=True,
    )

    with mapping_path.open("w", encoding="utf-8") as mapping_file:
        for line_no, record in iter_jsonl(path, []):
            missing = validate_record_schema(record, path, line_no)
            if missing:
                continue

            text = safe_embed_text(record)
            if not text.strip():
                continue

            batch_texts.append(text)
            batch_records.append(record)
            stats.rows_read += 1

            should_flush = (
                len(batch_texts) >= config.batch_size or stats.rows_read == rows_counted
            )
            if not should_flush:
                continue

            vectors = model.encode(
                batch_texts,
                batch_size=config.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=config.normalize_embeddings,
            )

            validation = validate_vectors(vectors, config.expected_dim)
            if batch_index < 5:
                sampled_validations.append(
                    {
                        "type": "batch_validation",
                        "batch_index": batch_index + 1,
                        "validation": validation,
                        "sample_chunk_ids": [r.get("chunk_id") for r in batch_records[:5]],
                    }
                )

            norms = np.linalg.norm(vectors, axis=1).astype(np.float64, copy=False)
            batch_min = float(norms.min())
            batch_max = float(norms.max())
            norm_sum += float(norms.sum())

            if stats.min_embedding_norm is None or batch_min < stats.min_embedding_norm:
                stats.min_embedding_norm = batch_min
            if stats.max_embedding_norm is None or batch_max > stats.max_embedding_norm:
                stats.max_embedding_norm = batch_max

            next_row_index = row_index + vectors.shape[0]
            memmap_arr[row_index:next_row_index] = vectors.astype(np.float32, copy=False)

            for local_i, batch_record in enumerate(batch_records):
                absolute_row_index = row_index + local_i
                if len(sampled_reencode_rows) < REENCODE_VALIDATION_ROWS:
                    sampled_reencode_rows.append(
                        {
                            "row_index": absolute_row_index,
                            "chunk_id": batch_record.get("chunk_id"),
                            "text": batch_texts[local_i],
                        }
                    )

                mapping_file.write(
                    json.dumps(
                        mapping_record(batch_record, absolute_row_index),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            written_now = vectors.shape[0]
            row_index = next_row_index
            stats.rows_encoded += written_now
            stats.rows_written += written_now
            stats.batches += 1
            batch_index += 1

            progress.update(written_now)

            batch_texts.clear()
            batch_records.clear()

            if remaining_limit is not None and stats.rows_encoded >= remaining_limit:
                break

    progress.close()
    memmap_arr.flush()
    del memmap_arr

    stats.elapsed_seconds = time.perf_counter() - start_time
    if stats.elapsed_seconds > 0:
        stats.throughput_records_per_second = stats.rows_encoded / stats.elapsed_seconds
    if stats.rows_encoded > 0:
        stats.avg_embedding_norm = norm_sum / stats.rows_encoded

    reencode_validation = validate_saved_outputs(
        emb_path=emb_path,
        mapping_path=mapping_path,
        stats=stats,
        expected_dim=config.expected_dim,
        model=model,
        normalize_embeddings=config.normalize_embeddings,
        batch_size=config.batch_size,
        sampled_reencode_rows=sampled_reencode_rows,
    )
    sampled_validations.append(
        {
            "type": "saved_vs_reencoded",
            "input_file": path.name,
            **reencode_validation,
        }
    )

    return stats, sampled_validations


def main() -> None:
    """Load the model, encode all input shards, validate outputs, and save the stats file."""
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)

    stats_path = config.output_dir / "stats_encoder.json"
    existing_stats_by_input: dict[str, dict[str, Any]] = {}
    existing_sample_validations: list[dict[str, Any]] = []
    if config.resume_existing:
        existing_stats_by_input, existing_sample_validations = load_existing_encoder_stats(stats_path)
    started_at_utc = now_utc_iso()
    finished_at_utc = started_at_utc

    status = "failed"
    error: dict[str, Any] | None = None
    device_used: str | None = None
    detected_dim: int | None = None
    per_file_stats: list[dict[str, Any]] = []
    all_sample_validations: list[dict[str, Any]] = list(existing_sample_validations)
    total_start = time.perf_counter()

    try:
        for path in config.input_files:
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")

        device_used = choose_device(config.device_request)
        print(f"[info] device = {device_used}")
        print(f"[info] model = {config.model_name}")
        print(f"[info] cache = {config.model_cache_dir.resolve()}")
        print(f"[info] output = {config.output_dir.resolve()}")
        print(f"[info] try_first_x_chunks = {config.max_records}")

        local_snapshot = resolve_local_model_snapshot(config.model_name, config.model_cache_dir)
        model_source = str(local_snapshot) if local_snapshot is not None else config.model_name
        if local_snapshot is not None:
            print(f"[info] using local snapshot = {local_snapshot}")

        model = SentenceTransformer(
            model_source,
            device=device_used,
            cache_folder=str(config.model_cache_dir),
            local_files_only=local_snapshot is not None,
        )
        detected_dim = model.get_sentence_embedding_dimension()

        if detected_dim != config.expected_dim:
            raise ValueError(
                f"Detected dim {detected_dim} does not match EXPECTED_DIM {config.expected_dim}"
            )

        remaining_limit = config.max_records

        for file_index, path in enumerate(config.input_files, start=1):
            if remaining_limit is not None and remaining_limit <= 0:
                break

            shard_num = shard_num_for_input(path, file_index)
            emb_path, mapping_path = output_paths(config.output_dir, shard_num)

            if config.resume_existing and emb_path.exists() and mapping_path.exists():
                print(f"\n[info] reusing existing outputs for {path.name}")
                existing_file_stats = validate_existing_output_pair(
                    path=path,
                    shard_num=shard_num,
                    config=config,
                    existing_stats_entry=existing_stats_by_input.get(path.name),
                )
                per_file_stats.append(existing_file_stats)
                print(
                    f"[skip] {path.name} | existing_rows={existing_file_stats['rows_written']} "
                    f"| embeddings={emb_path.name}"
                )
                continue

            print(f"\n[info] starting {path.name}")
            file_stats, file_validations = encode_one_file(
                path=path,
                shard_num=shard_num,
                model=model,
                config=config,
                remaining_limit=remaining_limit,
            )

            per_file_stats.append(asdict(file_stats))
            all_sample_validations.extend(file_validations)

            if remaining_limit is not None:
                remaining_limit = max(0, remaining_limit - file_stats.rows_encoded)

            print(
                f"[done] {path.name} | encoded={file_stats.rows_encoded} "
                f"| elapsed={file_stats.elapsed_seconds:.2f}s "
                f"| throughput={file_stats.throughput_records_per_second:.2f} chunks/s"
            )

        status = "completed"
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-12:],
        }
        raise
    finally:
        files_processed = len(per_file_stats)
        total_rows_counted = sum(int(item.get("rows_counted", 0)) for item in per_file_stats)
        total_rows_read = sum(int(item.get("rows_read", 0)) for item in per_file_stats)
        total_rows_encoded = sum(int(item.get("rows_encoded", 0)) for item in per_file_stats)
        total_rows_written = sum(int(item.get("rows_written", 0)) for item in per_file_stats)
        total_rows_skipped = sum(int(item.get("rows_skipped", 0)) for item in per_file_stats)
        finished_at_utc = now_utc_iso()
        payload = build_stats_payload(
            status=status,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            config=config,
            device_used=device_used,
            detected_dim=detected_dim,
            files_processed=files_processed,
            total_rows_counted=total_rows_counted,
            total_rows_read=total_rows_read,
            total_rows_encoded=total_rows_encoded,
            total_rows_written=total_rows_written,
            total_rows_skipped=total_rows_skipped,
            elapsed_seconds=time.perf_counter() - total_start,
            per_file_stats=per_file_stats,
            all_sample_validations=all_sample_validations,
            error=error,
        )
        write_json(stats_path, payload)

        print("\n========== FINAL SUMMARY ==========")
        print(f"Status:                 {status}")
        print(f"Files processed:        {files_processed}")
        print(f"Rows encoded:           {total_rows_encoded}")
        print(f"Detected embedding dim: {detected_dim}")
        print(f"Stats file:             {stats_path.resolve()}")
        print("===================================")


if __name__ == "__main__":
    main()
