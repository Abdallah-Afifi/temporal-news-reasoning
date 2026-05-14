#!/usr/bin/env python3
"""
build_faiss_database.py
========================
Build step 9: build the FAISS vector index used at query time.

Simple summary
--------------
Takes the ``.npy`` embedding files and mapping JSONL files that ``encoder.py`` wrote,
adds all vectors into a FAISS index, and saves four files to the output folder:

  faiss_hnsw_ip.index          - the search index (vectors stored here)
  faiss_vector_lookup.jsonl    - one row per vector: chunk_id, title, dates, etc.
  faiss_index_manifest.json    - short summary of what was indexed (dim, shard list)
  stats_faiss_database.json    - detailed run statistics and validation results

The default index type is HNSW with inner-product (cosine) metric on CPU.
If a GPU-enabled FAISS build is found, it switches to FlatIP on GPU automatically.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Encoder (encoder.py)]  →  THIS MODULE  →  [Query / interactive_query.py]

Usage (CLI)
-----------
    # Full build from encoder output:
    python -m temporal_rag.build_faiss_database

    # Test with just the first 1000 vectors:
    python -m temporal_rag.build_faiss_database --max-vectors 1000

    # Resume a previous run (skip already-indexed shards):
    python -m temporal_rag.build_faiss_database --resume-existing

If the folder ``faiss_dependencies`` exists, its ``site-packages`` directory is put on
``sys.path`` so FAISS can import without a global pip install.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT


DEFAULT_INPUT_DIR = PROJECT_ROOT / "encoder" / "output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_FAISS_ENV_DIR = PROJECT_ROOT / "faiss_dependencies"


def bootstrap_local_faiss() -> None:
    """If vendored wheels live under ``faiss_dependencies``, put them on ``sys.path``."""
    lib_dir = DEFAULT_FAISS_ENV_DIR / "lib"
    for site_packages in sorted(lib_dir.glob("python*/site-packages"), reverse=True):
        if site_packages.exists():
            sys.path.insert(0, str(site_packages))
            return


bootstrap_local_faiss()

import faiss  # noqa: E402  # imported after bootstrap may tweak sys.path
import numpy as np  # noqa: E402


INDEX_FILENAME = "faiss_hnsw_ip.index"
LOOKUP_FILENAME = "faiss_vector_lookup.jsonl"
MANIFEST_FILENAME = "faiss_index_manifest.json"
STATS_FILENAME = "stats_faiss_database.json"

DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 128
DEFAULT_ADD_BATCH_SIZE = 2048
SELF_SEARCH_VALIDATION_ROWS = 25


@dataclass
class InputShardStats:
    """Statistics collected while indexing one shard (one .npy + .jsonl pair).

    Attributes
    ----------
    embedding_file:
        Name of the ``.npy`` file that held the vectors for this shard.
    mapping_file:
        Name of the ``.jsonl`` file that held the chunk metadata for this shard.
    rows_available:
        How many vectors existed in the embedding file.
    rows_indexed:
        How many of those vectors were actually added to the index
        (may be less than ``rows_available`` when ``--max-vectors`` is used).
    row_offset_start:
        Global vector ID of the first vector from this shard.
    row_offset_end:
        Global vector ID of the last vector from this shard.
    embedding_shape:
        Shape of the embedding array, e.g. ``[5000, 384]``.
    embedding_dtype:
        NumPy dtype string, usually ``"float32"``.
    embedding_size_bytes / mapping_size_bytes:
        File sizes on disk in bytes.
    mapping_rows_consumed:
        How many rows were read from the mapping JSONL (should equal ``rows_indexed``).
    mapping_row_index_sequence_ok:
        ``True`` if every ``row_index`` in the mapping file was in order (0, 1, 2, …).
    first_mapping_row_index / last_mapping_row_index:
        First and last ``row_index`` values seen in the mapping file.
    vectors_norm_min/max/avg_before_normalize:
        L2 norm statistics of the raw vectors before any normalization step.
    elapsed_seconds:
        Wall-clock time to process this shard.
    throughput_vectors_per_second:
        Indexing speed for this shard.
    """
    embedding_file: str
    mapping_file: str
    rows_available: int = 0
    rows_indexed: int = 0
    row_offset_start: int = 0
    row_offset_end: int = -1
    embedding_shape: list[int] = field(default_factory=list)
    embedding_dtype: str | None = None
    embedding_size_bytes: int = 0
    mapping_size_bytes: int = 0
    mapping_rows_consumed: int = 0
    mapping_row_index_sequence_ok: bool = True
    first_mapping_row_index: int | None = None
    last_mapping_row_index: int | None = None
    vectors_norm_min_before_normalize: float | None = None
    vectors_norm_max_before_normalize: float | None = None
    vectors_norm_avg_before_normalize: float | None = None
    elapsed_seconds: float = 0.0
    throughput_vectors_per_second: float = 0.0


@dataclass
class RunConfig:
    """All settings for one build run, filled from command-line arguments.

    Attributes
    ----------
    input_dir:
        Folder that contains the ``embeddings_*.npy`` and ``chunk_mapping_*.jsonl`` files
        written by ``encoder.py``.
    output_dir:
        Folder where the index, lookup, manifest, and stats files are saved.
    max_vectors:
        Stop after indexing this many vectors across all shards (``None`` = no limit).
    add_batch_size:
        How many vectors to add to the index at once.  Larger batches are faster.
    hnsw_m:
        HNSW graph connectivity.  Higher M gives better recall but more memory.
    ef_construction:
        HNSW build-time search depth.  Higher = better index quality, slower build.
    ef_search:
        HNSW search-time depth used during the self-search validation step.
    normalize_before_add:
        If ``True``, L2-normalize every vector before adding it.  Required for
        cosine similarity via inner-product search.
    requested_device:
        ``"auto"``, ``"cpu"``, or ``"gpu"`` (chosen by the user on the CLI).
    requested_index_kind:
        ``"auto"``, ``"hnsw"``, or ``"flatip"`` (chosen by the user on the CLI).
    resume_existing:
        If ``True``, skip shards that are already in the existing index file.
    """
    input_dir: Path
    output_dir: Path
    max_vectors: int | None
    add_batch_size: int
    hnsw_m: int
    ef_construction: int
    ef_search: int
    normalize_before_add: bool
    requested_device: str
    requested_index_kind: str
    resume_existing: bool


@dataclass
class ExecutionPlan:
    """The resolved device and index type that will actually be used.

    ``requested_*`` fields hold what the user asked for on the CLI.
    ``actual_*`` fields hold what will really happen after auto-detection.

    Attributes
    ----------
    requested_device / requested_index_kind:
        Raw CLI choices before any auto-detection logic runs.
    actual_device:
        ``"cpu"`` or ``"gpu"`` (the device that will be used).
    actual_index_kind:
        ``"hnsw"`` or ``"flatip"`` (the index type that will be built).
    metric:
        Always ``"inner_product"`` for this pipeline.
    gpu_faiss_available:
        ``True`` if a GPU-enabled FAISS build was detected.
    gpu_reason:
        Human-readable note explaining why a fallback happened (e.g. HNSW not
        supported on GPU), or ``None`` when no fallback was needed.
    gpu_device_id:
        Which GPU to use (0 = first GPU).
    """
    requested_device: str
    requested_index_kind: str
    actual_device: str
    actual_index_kind: str
    metric: str
    gpu_faiss_available: bool
    gpu_reason: str | None = None
    gpu_device_id: int = 0


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def print_progress(message: str) -> None:
    """Print ``message`` to stdout with a ``[HH:MM:SS]`` timestamp prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the FAISS build step."""
    parser = argparse.ArgumentParser(
        description="Build a FAISS HNSW index from encoder/output embeddings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing embeddings_*.npy and chunk_mapping_*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the FAISS index, lookup, manifest, and stats are written.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        help="Optional limit on how many vectors to index across all shards.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=DEFAULT_ADD_BATCH_SIZE,
        help="How many vectors to add to the index per batch.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=DEFAULT_HNSW_M,
        help="HNSW M parameter.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=DEFAULT_EF_CONSTRUCTION,
        help="HNSW efConstruction parameter.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=DEFAULT_EF_SEARCH,
        help="HNSW efSearch parameter used for validation search.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before adding vectors to FAISS.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Execution target for index build. GPU requires a GPU-enabled FAISS build.",
    )
    parser.add_argument(
        "--index-kind",
        choices=("auto", "hnsw", "flatip"),
        default="auto",
        help="Index type. Auto prefers GPU-friendly FlatIP when GPU FAISS is available.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume from an existing FAISS output directory by appending only newly discovered shards.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    """Validate parsed arguments and return a ``RunConfig`` object."""
    if args.max_vectors is not None and args.max_vectors <= 0:
        raise ValueError("--max-vectors must be positive when provided")
    if args.add_batch_size <= 0:
        raise ValueError("--add-batch-size must be positive")
    if args.hnsw_m <= 0:
        raise ValueError("--hnsw-m must be positive")
    if args.ef_construction <= 0 or args.ef_search <= 0:
        raise ValueError("--ef-construction and --ef-search must be positive")

    return RunConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_vectors=args.max_vectors,
        add_batch_size=args.add_batch_size,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        normalize_before_add=not args.no_normalize,
        requested_device=args.device,
        requested_index_kind=args.index_kind,
        resume_existing=args.resume_existing,
    )


def discover_input_pairs(input_dir: Path) -> list[tuple[Path, Path, int]]:
    """Find all (embedding_file, mapping_file, shard_number) triples in ``input_dir``.

    Scans for ``embeddings_*.npy`` files and expects a matching
    ``chunk_mapping_*.jsonl`` file beside each one.
    Raises ``FileNotFoundError`` if any mapping file is missing.
    """
    pairs: list[tuple[Path, Path, int]] = []
    for emb_path in sorted(input_dir.glob("embeddings_*.npy")):
        shard_suffix = emb_path.stem.split("_")[-1]
        mapping_path = input_dir / f"chunk_mapping_{shard_suffix}.jsonl"
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Missing mapping file for {emb_path.name}: expected {mapping_path.name}"
            )
        pairs.append((emb_path, mapping_path, int(shard_suffix)))

    if not pairs:
        raise FileNotFoundError(f"No embeddings_*.npy files found in {input_dir}")

    return pairs


def faiss_gpu_available() -> bool:
    """Return ``True`` if this FAISS build includes GPU support."""
    return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")


def choose_execution_plan(config: RunConfig) -> ExecutionPlan:
    """Decide the actual device and index type to use based on what is available.

    Rules (in plain English):
    - HNSW always runs on CPU (FAISS does not support HNSW on GPU).
    - FlatIP can run on GPU if GPU-enabled FAISS is installed and the user
      asked for ``"gpu"`` or ``"auto"``.
    - If the user asked for ``"auto"``, we pick the fastest available option.
    """
    gpu_available = faiss_gpu_available()
    requested_device = config.requested_device
    requested_index_kind = config.requested_index_kind
    reason: str | None = None

    if requested_index_kind == "auto":
        if requested_device == "cpu":
            actual_index_kind = "hnsw"
        elif gpu_available:
            actual_index_kind = "flatip"
        else:
            actual_index_kind = "hnsw"
            reason = "GPU FAISS APIs not available; using CPU HNSW fallback"
    else:
        actual_index_kind = requested_index_kind

    if actual_index_kind == "hnsw":
        actual_device = "cpu"
        if requested_device == "gpu":
            if requested_index_kind == "hnsw":
                reason = "HNSW is not supported on GPU in FAISS; using CPU"
            elif not gpu_available:
                raise RuntimeError(
                    "GPU was requested but GPU-enabled FAISS is not installed in faiss_dependencies"
                )
        elif requested_device == "auto" and reason is None:
            reason = "HNSW uses CPU because FAISS does not support HNSW on GPU"
    else:
        if requested_device == "cpu":
            actual_device = "cpu"
        elif gpu_available:
            actual_device = "gpu"
            if requested_device == "auto":
                reason = "Using GPU FlatIP because GPU-enabled FAISS APIs are available"
        elif requested_device == "gpu":
            raise RuntimeError(
                "GPU was requested but GPU-enabled FAISS is not installed in faiss_dependencies"
            )
        else:
            actual_device = "cpu"
            reason = "GPU FlatIP requested by auto mode, but GPU FAISS is unavailable; using CPU FlatIP"

    return ExecutionPlan(
        requested_device=requested_device,
        requested_index_kind=requested_index_kind,
        actual_device=actual_device,
        actual_index_kind=actual_index_kind,
        metric="inner_product",
        gpu_faiss_available=gpu_available,
        gpu_reason=reason,
    )


def make_index(dim: int, config: RunConfig, plan: ExecutionPlan) -> tuple[faiss.Index, Any | None]:
    """Create a new empty FAISS index with the correct type and dimension.

    Returns ``(index, gpu_resources)``.  ``gpu_resources`` is ``None`` when
    running on CPU; it must be kept alive for as long as the index is used on GPU.
    """
    if plan.actual_index_kind == "flatip":
        cpu_index = faiss.IndexFlatIP(dim)
    elif plan.actual_index_kind == "hnsw":
        cpu_index = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        cpu_index.hnsw.efConstruction = config.ef_construction
        cpu_index.hnsw.efSearch = config.ef_search
    else:
        raise ValueError(f"Unsupported index kind: {plan.actual_index_kind}")

    if plan.actual_device == "gpu":
        gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, plan.gpu_device_id, cpu_index)
        return gpu_index, gpu_resources

    return cpu_index, None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path``."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read and return the JSON object stored at ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_faiss_state(
    *,
    index_file: Path,
    lookup_file: Path,
    manifest_file: Path,
    stats_file: Path,
) -> tuple[faiss.Index | None, list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Load a previously built index and its metadata so a resume run can continue.

    Returns ``(index, manifest_rows, per_file_stats_by_name, lookup_row_count)``.
    Returns ``(None, [], {}, 0)`` if any of the four files is missing.
    """
    if not (index_file.exists() and lookup_file.exists() and manifest_file.exists() and stats_file.exists()):
        return None, [], {}, 0

    loaded_index = faiss.read_index(str(index_file))
    manifest_payload = read_json(manifest_file)
    stats_payload = read_json(stats_file)
    existing_manifest_rows = manifest_payload.get("shards", [])
    existing_per_file_stats = {
        item["embedding_file"]: item
        for item in stats_payload.get("per_file_breakdown", [])
        if "embedding_file" in item
    }
    existing_lookup_rows = int(stats_payload.get("lookup_rows_written", loaded_index.ntotal))
    return loaded_index, existing_manifest_rows, existing_per_file_stats, existing_lookup_rows


def normalize_rows_inplace(batch: np.ndarray) -> None:
    """Divide each row vector by its L2 norm so all vectors have length 1.

    Modifies ``batch`` in place.  Raises ``ValueError`` if any row is a zero vector
    (a zero vector has no direction and cannot be normalized).
    """
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    zero_mask = norms <= 0.0
    if np.any(zero_mask):
        raise ValueError("Zero-norm vector encountered; cannot normalize for cosine/IP search")
    batch /= norms


def summarize_validation(
    *,
    status: str,
    dim: int | None,
    total_vectors_indexed: int,
    lookup_rows_written: int,
    index_ntotal: int,
    per_file_stats: list[dict[str, Any]],
    self_search_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a flat dict of True/False checks that confirm the index is correct.

    Each key is a human-readable assertion (e.g. ``"lookup_rows_match_vectors_indexed"``).
    The final key ``"overall_validation_passed"`` is ``True`` only when every check passes.
    These checks are saved inside the stats JSON so you can verify the build at a glance.
    """
    summary = {
        "status_is_completed": status == "completed",
        "dimension_detected": dim is not None and dim > 0,
        "all_mapping_sequences_valid": all(
            item["mapping_row_index_sequence_ok"] for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_mapping_rows_match_indexed_rows": all(
            item["mapping_rows_consumed"] == item["rows_indexed"] for item in per_file_stats
        )
        if per_file_stats
        else True,
        "all_embedding_shapes_match_dim": all(
            item["embedding_shape"][1] == dim
            for item in per_file_stats
            if len(item["embedding_shape"]) == 2 and dim is not None
        )
        if per_file_stats and dim is not None
        else dim is not None,
        "lookup_rows_match_vectors_indexed": lookup_rows_written == total_vectors_indexed,
        "faiss_ntotal_matches_vectors_indexed": index_ntotal == total_vectors_indexed,
        "self_search_identity_match": bool(
            self_search_validation and self_search_validation["all_top1_match"]
        )
        if self_search_validation is not None
        else False,
        "self_search_scores_finite": bool(
            self_search_validation and self_search_validation["all_scores_finite"]
        )
        if self_search_validation is not None
        else False,
    }
    summary["overall_validation_passed"] = all(summary.values())
    return summary


def build_stats_payload(
    *,
    status: str,
    started_at_utc: str,
    finished_at_utc: str,
    config: RunConfig,
    discovered_pairs: list[tuple[Path, Path, int]],
    dim: int | None,
    total_vectors_available: int,
    total_vectors_indexed: int,
    lookup_rows_written: int,
    index_ntotal: int,
    index_file: Path,
    lookup_file: Path,
    manifest_file: Path,
    per_file_stats: list[dict[str, Any]],
    self_search_validation: dict[str, Any] | None,
    elapsed_seconds: float,
    error: dict[str, Any] | None,
    plan: ExecutionPlan,
) -> dict[str, Any]:
    """Collect all run data into the big JSON object written to ``stats_faiss_database.json``.

    Includes config settings, shard-level stats, the validation summary, the self-search
    results, and any error that occurred.  This file is the single source of truth for
    confirming a build succeeded.
    """
    validation_summary = summarize_validation(
        status=status,
        dim=dim,
        total_vectors_indexed=total_vectors_indexed,
        lookup_rows_written=lookup_rows_written,
        index_ntotal=index_ntotal,
        per_file_stats=per_file_stats,
        self_search_validation=self_search_validation,
    )

    return {
        "job_name": "faiss_database_builder",
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "run_mode": "sample" if config.max_vectors is not None else "full",
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "script_path": str(Path(__file__).resolve()),
        "local_faiss_env_detected": str(DEFAULT_FAISS_ENV_DIR) if DEFAULT_FAISS_ENV_DIR.exists() else None,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "requested_device": plan.requested_device,
        "requested_index_kind": plan.requested_index_kind,
        "actual_device": plan.actual_device,
        "actual_index_kind": plan.actual_index_kind,
        "gpu_faiss_available": plan.gpu_faiss_available,
        "gpu_plan_reason": plan.gpu_reason,
        "index_type": (
            "IndexFlatIP"
            if plan.actual_index_kind == "flatip"
            else "IndexHNSWFlat"
            if plan.actual_index_kind == "hnsw"
            else "unknown"
        ),
        "metric": plan.metric,
        "normalize_before_add": config.normalize_before_add,
        "hnsw_m": config.hnsw_m,
        "ef_construction": config.ef_construction,
        "ef_search": config.ef_search,
        "max_vectors": config.max_vectors,
        "add_batch_size": config.add_batch_size,
        "discovered_input_pairs": [
            {
                "embedding_file": str(emb_path),
                "mapping_file": str(mapping_path),
                "shard_num": shard_num,
            }
            for emb_path, mapping_path, shard_num in discovered_pairs
        ],
        "files_processed": len(per_file_stats),
        "vector_dim": dim,
        "total_vectors_available": total_vectors_available,
        "total_vectors_indexed": total_vectors_indexed,
        "lookup_rows_written": lookup_rows_written,
        "faiss_index_ntotal": index_ntotal,
        "elapsed_seconds": elapsed_seconds,
        "average_vectors_per_second": (
            total_vectors_indexed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "index_file": str(index_file),
        "index_file_size_bytes": index_file.stat().st_size if index_file.exists() else 0,
        "lookup_file": str(lookup_file),
        "lookup_file_size_bytes": lookup_file.stat().st_size if lookup_file.exists() else 0,
        "manifest_file": str(manifest_file),
        "manifest_file_size_bytes": manifest_file.stat().st_size if manifest_file.exists() else 0,
        "validation_summary": validation_summary,
        "self_search_validation": self_search_validation,
        "per_file_breakdown": per_file_stats,
        "error": error,
    }


def main() -> None:
    """Run the full FAISS build: discover shards, index vectors, validate, and save."""
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    index_file = config.output_dir / INDEX_FILENAME
    lookup_file = config.output_dir / LOOKUP_FILENAME
    manifest_file = config.output_dir / MANIFEST_FILENAME
    stats_file = config.output_dir / STATS_FILENAME

    started_at_utc = now_utc_iso()
    status = "failed"
    error: dict[str, Any] | None = None
    discovered_pairs: list[tuple[Path, Path, int]] = []
    per_file_stats: list[dict[str, Any]] = []
    self_search_validation: dict[str, Any] | None = None
    total_vectors_available = 0
    total_vectors_indexed = 0
    lookup_rows_written = 0
    dim: int | None = None
    index_ntotal = 0
    total_start = time.perf_counter()
    plan = ExecutionPlan(
        requested_device=config.requested_device,
        requested_index_kind=config.requested_index_kind,
        actual_device="unknown",
        actual_index_kind="unknown",
        metric="inner_product",
        gpu_faiss_available=faiss_gpu_available(),
    )

    try:
        plan = choose_execution_plan(config)
        discovered_pairs = discover_input_pairs(config.input_dir)
        print_progress(f"discovered {len(discovered_pairs)} embedding/mapping pair(s)")
        print_progress(
            f"execution plan: device={plan.actual_device}, index={plan.actual_index_kind}, "
            f"gpu_faiss_available={plan.gpu_faiss_available}"
        )
        if plan.gpu_reason:
            print_progress(plan.gpu_reason)

        existing_index: faiss.Index | None = None
        existing_manifest_rows: list[dict[str, Any]] = []
        existing_stats_by_embedding: dict[str, dict[str, Any]] = {}
        existing_lookup_rows = 0
        if config.resume_existing:
            (
                existing_index,
                existing_manifest_rows,
                existing_stats_by_embedding,
                existing_lookup_rows,
            ) = load_existing_faiss_state(
                index_file=index_file,
                lookup_file=lookup_file,
                manifest_file=manifest_file,
                stats_file=stats_file,
            )
            if existing_index is not None:
                print_progress(
                    f"resuming existing FAISS state with ntotal={existing_index.ntotal} "
                    f"and {len(existing_manifest_rows)} existing shard(s)"
                )
                per_file_stats.extend(existing_stats_by_embedding.values())
                lookup_rows_written = existing_lookup_rows
                index_ntotal = int(existing_index.ntotal)
                total_vectors_indexed = int(existing_index.ntotal)
                dim = int(existing_index.d)

        if existing_index is not None and plan.actual_device == "gpu":
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, plan.gpu_device_id, existing_index)
        else:
            index = existing_index
        index_for_write: faiss.Index | None = None
        if existing_index is None or plan.actual_device != "gpu":
            gpu_resources: Any | None = None
        global_vector_id = int(existing_index.ntotal) if existing_index is not None else 0
        sample_query_vectors: list[np.ndarray] = []
        sample_query_ids: list[int] = []
        manifest_rows: list[dict[str, Any]] = list(existing_manifest_rows)
        remaining_limit = config.max_vectors
        existing_shard_nums = {int(item["shard_num"]) for item in existing_manifest_rows}

        lookup_mode = "a" if config.resume_existing and lookup_file.exists() else "w"
        with lookup_file.open(lookup_mode, encoding="utf-8") as lookup_out:
            for emb_path, mapping_path, shard_num in discovered_pairs:
                if remaining_limit is not None and remaining_limit <= 0:
                    break

                emb_probe = np.load(emb_path, mmap_mode="r")
                rows_available_probe = int(emb_probe.shape[0])
                total_vectors_available += rows_available_probe
                if config.resume_existing and shard_num in existing_shard_nums:
                    print_progress(
                        f"reusing existing FAISS shard {shard_num:05d}: {emb_path.name} already indexed"
                    )
                    continue

                shard_start = time.perf_counter()
                emb = emb_probe
                if emb.ndim != 2:
                    raise ValueError(f"{emb_path.name} must be 2D, got shape {emb.shape}")

                shard_dim = int(emb.shape[1])
                if dim is None:
                    dim = shard_dim
                elif dim != shard_dim:
                    raise ValueError(
                        f"Dimension mismatch: expected {dim}, got {shard_dim} in {emb_path.name}"
                    )

                if index is None:
                    index, gpu_resources = make_index(dim, config, plan)
                    if plan.actual_index_kind == "hnsw":
                        print_progress(
                            f"initialized FAISS index with dim={dim}, HNSW M={config.hnsw_m}"
                        )
                    else:
                        print_progress(
                            f"initialized FAISS index with dim={dim}, FlatIP on {plan.actual_device}"
                        )

                rows_available = rows_available_probe
                rows_to_index = rows_available
                if remaining_limit is not None:
                    rows_to_index = min(rows_to_index, remaining_limit)

                shard_stats = InputShardStats(
                    embedding_file=emb_path.name,
                    mapping_file=mapping_path.name,
                    rows_available=rows_available,
                    row_offset_start=global_vector_id,
                    embedding_shape=[int(emb.shape[0]), int(emb.shape[1])],
                    embedding_dtype=str(emb.dtype),
                    embedding_size_bytes=emb_path.stat().st_size,
                    mapping_size_bytes=mapping_path.stat().st_size,
                )

                print_progress(
                    f"building shard {shard_num:05d}: indexing {rows_to_index} / {rows_available} vectors"
                )

                norm_sum = 0.0
                norm_count = 0
                next_expected_mapping_row = 0

                with mapping_path.open("r", encoding="utf-8") as mapping_in:
                    while shard_stats.rows_indexed < rows_to_index:
                        batch_start = shard_stats.rows_indexed
                        batch_end = min(batch_start + config.add_batch_size, rows_to_index)

                        batch = np.array(emb[batch_start:batch_end], dtype=np.float32, copy=True)
                        if batch.ndim != 2 or batch.shape[1] != dim:
                            raise ValueError(
                                f"Invalid batch shape from {emb_path.name}: {batch.shape}"
                            )
                        if not np.isfinite(batch).all():
                            raise ValueError(f"Non-finite values found in {emb_path.name}")

                        original_norms = np.linalg.norm(batch, axis=1).astype(np.float64, copy=False)
                        batch_min = float(original_norms.min()) if len(original_norms) else None
                        batch_max = float(original_norms.max()) if len(original_norms) else None
                        if batch_min is not None:
                            if (
                                shard_stats.vectors_norm_min_before_normalize is None
                                or batch_min < shard_stats.vectors_norm_min_before_normalize
                            ):
                                shard_stats.vectors_norm_min_before_normalize = batch_min
                        if batch_max is not None:
                            if (
                                shard_stats.vectors_norm_max_before_normalize is None
                                or batch_max > shard_stats.vectors_norm_max_before_normalize
                            ):
                                shard_stats.vectors_norm_max_before_normalize = batch_max
                        norm_sum += float(original_norms.sum())
                        norm_count += len(original_norms)

                        if config.normalize_before_add:
                            normalize_rows_inplace(batch)

                        index.add(batch)

                        for local_idx in range(batch_start, batch_end):
                            raw_line = mapping_in.readline()
                            if not raw_line:
                                raise ValueError(
                                    f"Unexpected end of mapping file {mapping_path.name} at row {local_idx}"
                                )
                            mapping_record = json.loads(raw_line)
                            row_index = mapping_record.get("row_index")
                            if row_index != next_expected_mapping_row:
                                shard_stats.mapping_row_index_sequence_ok = False
                                raise ValueError(
                                    f"row_index mismatch in {mapping_path.name}: "
                                    f"expected {next_expected_mapping_row}, got {row_index}"
                                )

                            if shard_stats.first_mapping_row_index is None:
                                shard_stats.first_mapping_row_index = int(row_index)
                            shard_stats.last_mapping_row_index = int(row_index)
                            next_expected_mapping_row += 1
                            shard_stats.mapping_rows_consumed += 1

                            lookup_record = {
                                "global_vector_id": global_vector_id,
                                "shard_num": shard_num,
                                "embedding_file": emb_path.name,
                                "mapping_file": mapping_path.name,
                                "local_row_index": int(row_index),
                                "chunk_id": mapping_record.get("chunk_id"),
                                "source_doc_id": mapping_record.get("source_doc_id"),
                                "chunk_index": mapping_record.get("chunk_index"),
                                "total_chunks": mapping_record.get("total_chunks"),
                                "title": mapping_record.get("title"),
                                "published_date": mapping_record.get("published_date"),
                            }
                            for field_name in (
                                "T_start",
                                "T_end",
                                "T_start_epoch",
                                "T_end_epoch",
                                "temporal_source",
                                "temporal_backend",
                                "all_resolved_dates",
                                "span_to_date",
                                "temporal_mention_count",
                            ):
                                if field_name in mapping_record:
                                    lookup_record[field_name] = mapping_record.get(field_name)
                            lookup_out.write(json.dumps(lookup_record, ensure_ascii=False) + "\n")

                            if len(sample_query_vectors) < SELF_SEARCH_VALIDATION_ROWS:
                                sample_query_vectors.append(np.array(batch[local_idx - batch_start], copy=True))
                                sample_query_ids.append(global_vector_id)

                            global_vector_id += 1
                            lookup_rows_written += 1

                        shard_stats.rows_indexed = batch_end
                        total_vectors_indexed += len(batch)

                        print_progress(
                            f"progress {emb_path.name}: {shard_stats.rows_indexed}/{rows_to_index} indexed"
                        )

                    if rows_to_index == rows_available:
                        trailing_mapping_rows = 0
                        for raw_line in mapping_in:
                            if raw_line.strip():
                                trailing_mapping_rows += 1
                        if trailing_mapping_rows != 0:
                            raise ValueError(
                                f"Mapping file {mapping_path.name} has {trailing_mapping_rows} extra rows "
                                f"beyond the embedding count"
                            )

                shard_stats.row_offset_end = global_vector_id - 1 if shard_stats.rows_indexed else global_vector_id
                shard_stats.elapsed_seconds = time.perf_counter() - shard_start
                if shard_stats.elapsed_seconds > 0:
                    shard_stats.throughput_vectors_per_second = (
                        shard_stats.rows_indexed / shard_stats.elapsed_seconds
                    )
                if norm_count > 0:
                    shard_stats.vectors_norm_avg_before_normalize = norm_sum / norm_count

                manifest_rows.append(
                    {
                        "shard_num": shard_num,
                        "embedding_file": emb_path.name,
                        "mapping_file": mapping_path.name,
                        "rows_available": rows_available,
                        "rows_indexed": shard_stats.rows_indexed,
                        "global_id_start": shard_stats.row_offset_start,
                        "global_id_end": shard_stats.row_offset_end,
                    }
                )
                per_file_stats.append(asdict(shard_stats))

                if remaining_limit is not None:
                    remaining_limit = max(0, remaining_limit - shard_stats.rows_indexed)

        if index is None or dim is None:
            raise ValueError("No vectors were indexed")

        print_progress("writing FAISS index to disk")
        index_ntotal = int(index.ntotal)
        if plan.actual_device == "gpu":
            index_for_write = faiss.index_gpu_to_cpu(index)
        else:
            index_for_write = index
        faiss.write_index(index_for_write, str(index_file))

        manifest_payload = {
            "index_file": index_file.name,
            "lookup_file": lookup_file.name,
            "vector_dim": dim,
            "requested_device": plan.requested_device,
            "requested_index_kind": plan.requested_index_kind,
            "actual_device": plan.actual_device,
            "actual_index_kind": plan.actual_index_kind,
            "gpu_faiss_available": plan.gpu_faiss_available,
            "metric": plan.metric,
            "normalize_before_add": config.normalize_before_add,
            "hnsw_m": config.hnsw_m,
            "ef_construction": config.ef_construction,
            "ef_search": config.ef_search,
            "total_vectors_indexed": total_vectors_indexed,
            "files_processed": len(per_file_stats),
            "shards": manifest_rows,
        }
        write_json(manifest_file, manifest_payload)

        print_progress("running validation checks on saved index")
        loaded_index = faiss.read_index(str(index_file))
        if loaded_index.ntotal != total_vectors_indexed:
            raise ValueError(
                f"Saved index ntotal mismatch: expected {total_vectors_indexed}, got {loaded_index.ntotal}"
            )

        if loaded_index.d != dim:
            raise ValueError(f"Saved index dim mismatch: expected {dim}, got {loaded_index.d}")

        if sample_query_vectors:
            queries = np.vstack(sample_query_vectors).astype(np.float32, copy=False)
            distances, indices = loaded_index.search(queries, 1)
            top1_matches = [int(indices[i, 0]) == sample_query_ids[i] for i in range(len(sample_query_ids))]
            score_values = [float(distances[i, 0]) for i in range(len(sample_query_ids))]
            self_search_validation = {
                "rows_checked": len(sample_query_ids),
                "all_top1_match": all(top1_matches),
                "top1_match_rate": sum(top1_matches) / len(top1_matches),
                "all_scores_finite": all(np.isfinite(score_values)),
                "min_score": min(score_values),
                "max_score": max(score_values),
                "sample_results": [
                    {
                        "query_global_vector_id": sample_query_ids[i],
                        "retrieved_global_vector_id": int(indices[i, 0]),
                        "score": float(distances[i, 0]),
                        "top1_match": bool(top1_matches[i]),
                    }
                    for i in range(min(10, len(sample_query_ids)))
                ],
            }
            if not self_search_validation["all_top1_match"]:
                raise ValueError("Self-search validation failed: top-1 identity mismatch detected")
            if not self_search_validation["all_scores_finite"]:
                raise ValueError("Self-search validation failed: non-finite scores detected")
        elif existing_index is not None:
            previous_stats_payload = read_json(stats_file) if stats_file.exists() else {}
            self_search_validation = previous_stats_payload.get("self_search_validation")
            if self_search_validation is None:
                raise ValueError("No validation queries were collected and no previous validation exists")
        else:
            self_search_validation = {
                "rows_checked": 0,
                "all_top1_match": False,
                "top1_match_rate": 0.0,
                "all_scores_finite": False,
                "min_score": None,
                "max_score": None,
                "sample_results": [],
            }
            raise ValueError("No validation queries were collected")

        status = "completed"
        print_progress("FAISS database stage completed successfully")
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-12:],
        }
        raise
    finally:
        finished_at_utc = now_utc_iso()
        payload = build_stats_payload(
            status=status,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            config=config,
            discovered_pairs=discovered_pairs,
            dim=dim,
            total_vectors_available=total_vectors_available,
            total_vectors_indexed=total_vectors_indexed,
            lookup_rows_written=lookup_rows_written,
            index_ntotal=index_ntotal,
            index_file=index_file,
            lookup_file=lookup_file,
            manifest_file=manifest_file,
            per_file_stats=per_file_stats,
            self_search_validation=self_search_validation,
            elapsed_seconds=time.perf_counter() - total_start,
            error=error,
            plan=plan,
        )
        write_json(stats_file, payload)
        print_progress(f"stats file written to {stats_file}")


if __name__ == "__main__":
    main()
