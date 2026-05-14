"""
gliner_extractor.py
====================
Build step 4: find mentions of dates, times, events, and entities in each chunk.


GLiNER is a zero-shot tagger: we give it label names (DATE, TIME, EVENT, ENTITY) and it
returns spans with scores. Downstream stages use those spans for HeidelTime and graphs.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Recursive Character Splitter]  →  THIS MODULE  →  [Rule-Based Normalizer (HeidelTime)]

What it does
------------
Runs every text chunk through the GLiNER model (a bi-encoder NER model that
accepts arbitrary label sets at inference time, no fine-tuning required).
Extracts DATE, TIME, EVENT, and ENTITY spans to boost temporal recall for all
downstream stages.

Input  : JSONL produced by the character splitter (one chunk object per line).
Output : Same JSONL, each object enriched with an ``entities`` list field:

    {
        ...all original splitter fields...,
        "entities": [
            {"text": "October 2016", "label": "DATE",   "start": 12, "end": 24, "score": 0.92},
            {"text": "World Cup",    "label": "EVENT",  "start": 35, "end": 43, "score": 0.81},
            {"text": "FIFA",         "label": "ENTITY", "start": 50, "end": 54, "score": 0.76}
        ]
    }

Usage
-----
# Full run:
    python -m temporal_rag.gliner_extractor \\
        --input  chunks.jsonl \\
        --output gliner_output.jsonl

# Smoke-test on first 20 chunks only:
    python -m temporal_rag.gliner_extractor \\
        --input  chunks.jsonl \\
        --output gliner_output.jsonl \\
        --max-chunks 20 \\
        --log-level DEBUG

# Custom labels / threshold:
    python -m temporal_rag.gliner_extractor \\
        --input  ... \\
        --output ... \\
        --labels DATE TIME EVENT ENTITY PERSON ORGANIZATION LOCATION \\
        --threshold 0.40 \\
        --batch-size 4

Dependencies
------------
    pip install "gliner>=0.1.6,<0.2" tqdm

    If you see ``ValueError: ... upgrade torch to at least v2.6`` when loading
    DeBERTa, your **transformers** is too new for **torch 2.2.x** (common on Intel
    Macs where PyPI only ships older torch builds). Fix:

        pip install "transformers>=4.41,<4.48"

    **Important:** the default Hub checkpoints ``urchade/gliner_*-v2.1`` use the
    *legacy* GLiNER architecture (gliner 0.1.x).  If you install **gliner 0.2+**,
    weights will not match and ``from_pretrained`` loads mostly uninitialized
    parameters; you get thousands of nonsense ``DATE`` spans with ~0.446 scores.
    Either stay on gliner 0.1.x with these models, or upgrade torch and switch to a
    checkpoint built for gliner 0.2 (different Hub repos).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ── Optional heavy imports guarded so the file is importable for introspection ──
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class GLiNERConfig:
    """
    Configuration for the GLiNER Neural Span Extractor.

    Attributes
    ----------
    model_name:
        HuggingFace Hub model identifier.  ``urchade/gliner_medium-v2.1``
        gives the best accuracy / speed tradeoff for English news text
        when using **gliner 0.1.x**.  For research use ``urchade/gliner_large-v2.1``.
    entity_labels:
        Ordered list of span labels passed to GLiNER at inference time.
        The model is zero-shot, so you can add or remove labels freely.
    threshold:
        Minimum confidence score to keep a predicted span (0.0–1.0).
        0.35 keeps most valid predictions while filtering noise.
        Raise to 0.50 for higher precision; lower to 0.25 for higher recall.
    batch_size:
        Number of text chunks processed per model call.  Increase on GPU-rich
        machines (32, 64); decrease to 1–4 on CPU-only.
    device:
        ``"cuda"`` | ``"cpu"`` | ``"mps"`` (Apple Silicon) | ``None`` for
        auto-detection.
    max_length:
        Maximum token length fed to the model.  GLiNER default is 384.
        Increase to 512 for longer chunks at the cost of memory.
    """
    model_name: str = "urchade/gliner_medium-v2.1"
    entity_labels: list[str] = field(default_factory=lambda: [
        "DATE", "TIME", "EVENT", "ENTITY"
    ])
    threshold: float = 0.35
    batch_size: int = 8
    device: str | None = None          # None → auto-detect
    max_length: int = 384


# ────────────────────────────────────────────────────────────────────────────────
# Main extractor class
# ────────────────────────────────────────────────────────────────────────────────

class GLiNERExtractor:
    """
    Zero-shot neural span extractor wrapping the GLiNER model.

    Example
    -------
    >>> config = GLiNERConfig(threshold=0.40)
    >>> extractor = GLiNERExtractor(config)
    >>> spans = extractor.extract_spans("The 2024 Olympics opened in Paris on 26 July.")
    >>> for s in spans:
    ...     print(s)
    {'text': '2024 Olympics', 'label': 'EVENT', 'start': 4, 'end': 17, 'score': 0.91}
    {'text': 'Paris',         'label': 'ENTITY', 'start': 27, 'end': 32, 'score': 0.88}
    {'text': '26 July',       'label': 'DATE',   'start': 36, 'end': 43, 'score': 0.95}
    """

    def __init__(self, config: GLiNERConfig | None = None) -> None:
        self.config = config or GLiNERConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None   # lazy-loaded on first use

    # ── Model loading ────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Pick the best available compute device: cuda → mps → cpu."""
        if self.config.device is not None:
            return self.config.device
        if _TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load GLiNER model (called lazily on first inference)."""
        import warnings
        warnings.filterwarnings("ignore", message=".*truncated to.*")

        try:
            import gliner  # type: ignore[import]
            from gliner import GLiNER  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "GLiNER is not installed. Run: pip install gliner"
            ) from exc

        gl_major, gl_minor = (0, 0)
        try:
            parts = gliner.__version__.split(".")
            gl_major, gl_minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            pass
        if gl_major == 0 and gl_minor >= 2:
            raise RuntimeError(
                f"gliner=={gliner.__version__} is incompatible with the default "
                f"checkpoint {self.config.model_name!r} (legacy weight layout). "
                f"Install: pip install 'gliner>=0.1.6,<0.2'"
            )

        device = self._resolve_device()
        self.logger.info(
            "Loading GLiNER model '%s' on device '%s' …",
            self.config.model_name, device,
        )
        t0 = time.perf_counter()
        # Always map weights to CPU first (stable across backends); move after.
        try:
            self._model = GLiNER.from_pretrained(
                self.config.model_name,
                map_location="cpu",
                strict=True,
            )
        except TypeError:
            self._model = GLiNER.from_pretrained(
                self.config.model_name,
                map_location="cpu",
            )

        self._model.eval()
        print("  Checkpoint loaded. Running sanity inference (a few seconds) …", flush=True)
        self._sanity_check_loaded_model()
        print("  Sanity check OK.", flush=True)
        if device != "cpu" and _TORCH_AVAILABLE:
            self._model = self._model.to(device)
            print(f"  Model moved to {device!r}.", flush=True)
        elapsed = time.perf_counter() - t0
        self.logger.info("Model ready in %.1f s.", elapsed)

    def _sanity_check_loaded_model(self) -> None:
        """Fail fast if weights look uninitialized (degenerate span scores)."""
        text = (
            "The 2024 Summer Olympics opened in Paris on 26 July 2024. "
            "John Smith met the mayor on Monday."
        )
        raw = self._model.predict_entities(
            text,
            self.config.entity_labels,
            threshold=self.config.threshold,
        )
        if len(raw) > 80:
            raise RuntimeError(
                "GLiNER returned an implausible number of spans on a short sanity "
                "text; weights likely did not load. "
                "Use gliner 0.1.x with urchade/gliner_*-v2.1 models."
            )
        if len(raw) >= 12:
            scores = {round(float(e["score"]), 3) for e in raw}
            labels = {e["label"] for e in raw}
            if len(scores) <= 2 and len(labels) == 1:
                raise RuntimeError(
                    "GLiNER sanity check failed (uniform scores / single label). "
                    "See gliner_extractor.py docstring for the correct gliner version."
                )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ── Single-text inference ────────────────────────────────────────────────

    def extract_spans(self, text: str) -> list[dict]:
        """
        Run GLiNER on a single text string.

        Parameters
        ----------
        text : str
            Raw text of the chunk (not truncated; model handles max_length).

        Returns
        -------
        list[dict]
            Sorted by start offset. Each dict has keys:
            ``text``, ``label``, ``start``, ``end``, ``score``.
        """
        if not text or not text.strip():
            return []

        # GLiNER's predict_entities API: (text, labels, threshold)
        raw_entities = self.model.predict_entities(
            text,
            self.config.entity_labels,
            threshold=self.config.threshold,
        )

        # Normalise to a consistent internal schema
        spans = []
        for ent in raw_entities:
            spans.append({
                "text":  ent["text"],
                "label": ent["label"],
                "start": ent["start"],
                "end":   ent["end"],
                "score": round(float(ent["score"]), 4),
            })

        # Sort by start character offset
        spans.sort(key=lambda x: x["start"])
        return spans

    # ── Batch inference ──────────────────────────────────────────────────────

    def process_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Enrich a list of chunk dicts with an ``entities`` key.

        Each chunk must have a ``"text"`` field (as produced by the splitter).
        All other fields are preserved unchanged.

        Parameters
        ----------
        chunks : list[dict]
            Batch of chunk objects from the character splitter.

        Returns
        -------
        list[dict]
            Same objects, each with ``entities`` added / overwritten.
        """
        texts = [c.get("text", "") for c in chunks]

        # Batch predict: gliner 0.2+ uses predict_entities_batch; 0.1.x uses batch_predict_entities.
        if hasattr(self.model, "predict_entities_batch"):
            try:
                batch_results = self.model.predict_entities_batch(
                    texts,
                    self.config.entity_labels,
                    threshold=self.config.threshold,
                    batch_size=self.config.batch_size,
                )
            except TypeError:
                batch_results = [
                    self.model.predict_entities(
                        t, self.config.entity_labels, threshold=self.config.threshold
                    )
                    for t in texts
                ]
        elif hasattr(self.model, "batch_predict_entities"):
            batch_results = self.model.batch_predict_entities(
                texts,
                self.config.entity_labels,
                threshold=self.config.threshold,
            )
        else:
            batch_results = [
                self.model.predict_entities(
                    t, self.config.entity_labels, threshold=self.config.threshold
                )
                for t in texts
            ]

        enriched = []
        for chunk, raw_entities in zip(chunks, batch_results):
            spans = []
            for ent in raw_entities:
                spans.append({
                    "text":  ent["text"],
                    "label": ent["label"],
                    "start": ent["start"],
                    "end":   ent["end"],
                    "score": round(float(ent["score"]), 4),
                })
            spans.sort(key=lambda x: x["start"])
            enriched.append({**chunk, "entities": spans})

        return enriched

    # ── JSONL streaming pipeline ─────────────────────────────────────────────

    def _iter_jsonl(self, path: Path) -> Iterator[dict]:
        """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    self.logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)

    def process_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        max_chunks: int | None = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Stream chunks from ``input_path``, run GLiNER, write to ``output_path``.

        Parameters
        ----------
        input_path : str | Path
            Path to the character-splitter JSONL output.
        output_path : str | Path
            Path where enriched JSONL will be written (created/overwritten).
        max_chunks : int | None
            If set, stop after processing this many chunks (useful for testing).
        show_progress : bool
            Display a tqdm progress bar (requires tqdm installed).

        Returns
        -------
        dict
            Summary statistics:
            ``{"total_chunks": N, "total_entities": M, "elapsed_s": T,
               "chunks_per_sec": R, "label_counts": {...}}``
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.logger.info("Input  : %s", input_path)
        self.logger.info("Output : %s", output_path)
        self.logger.info(
            "Config : model=%s  labels=%s  threshold=%.2f  batch=%d",
            self.config.model_name,
            self.config.entity_labels,
            self.config.threshold,
            self.config.batch_size,
        )

        # ── Trigger model load before streaming ──
        _ = self.model

        total_chunks = 0
        total_entities = 0
        label_counts: dict[str, int] = {}
        t_start = time.perf_counter()

        # ── Set up optional progress bar ──
        if show_progress and _TQDM_AVAILABLE:
            progress = _tqdm(desc="GLiNER extraction", unit="chunk")
        else:
            progress = None

        with output_path.open("w", encoding="utf-8") as out_fh:
            batch: list[dict] = []

            def _flush_batch(batch: list[dict]) -> None:
                nonlocal total_chunks, total_entities
                if not batch:
                    return
                enriched = self.process_batch(batch)
                for obj in enriched:
                    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n_ents = len(obj.get("entities", []))
                    total_entities += n_ents
                    total_chunks  += 1
                    if progress is not None:
                        progress.update(1)
                    for ent in obj.get("entities", []):
                        lbl = ent["label"]
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1

            for chunk in self._iter_jsonl(input_path):
                if max_chunks is not None and total_chunks >= max_chunks:
                    break
                batch.append(chunk)
                if len(batch) >= self.config.batch_size:
                    _flush_batch(batch)
                    batch = []

            # Flush remaining
            if max_chunks is None or total_chunks < max_chunks:
                _flush_batch(batch)

        if progress is not None:
            progress.close()

        elapsed = time.perf_counter() - t_start
        stats = {
            "total_chunks":   total_chunks,
            "total_entities": total_entities,
            "elapsed_s":      round(elapsed, 2),
            "chunks_per_sec": round(total_chunks / elapsed, 1) if elapsed > 0 else 0,
            "label_counts":   label_counts,
        }

        self.logger.info("─" * 55)
        self.logger.info("Done.  %d chunks → %d entities in %.1f s (%.1f chunks/s)",
                         total_chunks, total_entities, elapsed, stats["chunks_per_sec"])
        self.logger.info("Label counts: %s", label_counts)
        return stats


# ────────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ────────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the GLiNER extractor CLI."""
    p = argparse.ArgumentParser(
        prog="gliner_extractor",
        description=(
            "Build step 4: GLiNER Neural Span Extractor for the Temporal RAG pipeline.\n"
            "Enriches a character-splitter JSONL with an 'entities' field per chunk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to the input JSONL file (character-splitter output).",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Path for the enriched output JSONL file.",
    )
    p.add_argument(
        "--model", "-m", default="urchade/gliner_medium-v2.1",
        help="GLiNER model name on HuggingFace Hub. (default: urchade/gliner_medium-v2.1)",
    )
    p.add_argument(
        "--labels", "-l", nargs="+",
        default=["DATE", "TIME", "EVENT", "ENTITY"],
        metavar="LABEL",
        help="Entity labels to extract. (default: DATE TIME EVENT ENTITY)",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=0.35,
        help="Confidence score cutoff [0–1]. (default: 0.35)",
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=8, dest="batch_size",
        help="Chunks per model call. (default: 8)",
    )
    p.add_argument(
        "--device", "-d", default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device. Auto-detected if omitted.",
    )
    p.add_argument(
        "--max-chunks", "-n", type=int, default=None, dest="max_chunks",
        help="Stop after N chunks (useful for smoke tests).",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Logging setup ────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ── Build config ─────────────────────────────────────────────────────────
    config = GLiNERConfig(
        model_name=args.model,
        entity_labels=args.labels,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ── Run ──────────────────────────────────────────────────────────────────
    extractor = GLiNERExtractor(config)
    stats = extractor.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        max_chunks=args.max_chunks,
        show_progress=not args.no_progress,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  GLiNER Extraction Complete")
    print("═" * 50)
    print(f"  Chunks processed : {stats['total_chunks']}")
    print(f"  Entities found   : {stats['total_entities']}")
    print(f"  Elapsed          : {stats['elapsed_s']} s")
    print(f"  Throughput       : {stats['chunks_per_sec']} chunks/s")
    if stats["label_counts"]:
        print("  Label breakdown  :")
        for lbl, cnt in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
            print(f"      {lbl:<12} {cnt:>6}")
    print("═" * 50)
    print(f"  Output → {args.output}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
