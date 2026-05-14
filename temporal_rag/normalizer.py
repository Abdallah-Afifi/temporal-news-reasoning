"""
normalizer.py
=============
Build step 5: resolve GLiNER DATE/TIME spans and chunk text to calendar bounds (`T_start`, `T_end`).

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [GLiNER] → THIS MODULE → [Temporal IE]

What it does
------------
Takes GLiNER-enriched chunks and resolves DATE/TIME spans to concrete calendar dates and
Unix epoch timestamps (HeidelTime), merged with dates found on the full chunk text.

Strategy
--------
  1. Extract DATE/TIME span texts from GLiNER entities.
  2. Run HeidelTime on each span with DCT = published_date (e.g. "last Monday", "Q3 2017").
  3. Run HeidelTime on the full chunk text and merge unique dates with (2).
  4. ``T_start`` / ``T_end`` = min / max over the merged list.
  5. Fallback: if nothing resolves, ``T_start`` = ``T_end`` = ``published_date``.

Output schema additions per chunk
----------------------------------
    "T_start"         : "YYYY-MM-DD"
    "T_end"           : "YYYY-MM-DD"
    "T_start_epoch"   : int
    "T_end_epoch"     : int
    "temporal_source" : str ("gliner_spans" | "full_text" | "pub_date")

Usage (library)
---------------
    from temporal_rag.normalizer import TemporalNormalizer

    norm = TemporalNormalizer()
    norm.normalize_chunk(gliner_chunk)
    stats = norm.process_jsonl("gliner_output.jsonl", "normalizer_output.jsonl")

Usage (CLI)
-----------
    python -m temporal_rag.normalizer \\
        --input  gliner_output.jsonl \\
        --output normalizer_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .utils.heideltime_wrapper import HeidelTimeWrapper, iso_to_epoch

logger = logging.getLogger(__name__)

# GLiNER often tags "year" / "month" alone as DATE; HeidelTime on that token is useless
# compared to the same word in "that year" on the full question — so we do not skip
# full-text HeidelTime for ad-hoc queries when *every* temporal span is one of these.
_VAGUE_TIME_SPAN_RE = re.compile(
    r"(?i)^(year|month|week|day|today|yesterday|tomorrow)s?$"
)


def _vague_temporal_span_text(text: str) -> bool:
    return bool(_VAGUE_TIME_SPAN_RE.match((text or "").strip()))

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizerConfig:
    """
    Configuration for the Temporal Normalizer.

    Attributes
    ----------
    language:
        Language passed to HeidelTime. Default "English".
    document_type:
        HeidelTime document type for full-text fallback. Default "news".
    span_labels:
        GLiNER entity labels treated as temporal. Only DATE and TIME spans
        are sent to HeidelTime for normalization.
    min_span_score:
        Minimum GLiNER confidence to consider a span for normalization.
        Spans below this are ignored to avoid noise.
    """
    language: str = "English"
    document_type: str = "news"
    span_labels: tuple[str, ...] = ("DATE", "TIME")
    min_span_score: float = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────────────────────────────────────

class TemporalNormalizer:
    """
    Rule-Based Temporal Normalizer (HeidelTime / TIMEX3-fallback).

    Enriches GLiNER-annotated chunk dicts with resolved temporal bounds:
    T_start, T_end (ISO-8601) and their Unix epoch equivalents.

    Example
    -------
    >>> norm = TemporalNormalizer()
    >>> chunk = {
    ...     "text": "The summit was held on October 18, 2016.",
    ...     "published_date": "2016-10-20",
    ...     "entities": [{"text": "October 18, 2016", "label": "DATE", "score": 0.95, ...}]
    ... }
    >>> result = norm.normalize_chunk(chunk)
    >>> result["T_start"], result["T_start_epoch"]
    ('2016-10-18', 1476748800)
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self._ht = HeidelTimeWrapper(
            language=self.config.language,
            document_type=self.config.document_type,
        )

    # ── Core logic ────────────────────────────────────────────────────────────

    def _resolve_spans(
        self, span_texts: list[str], pub_date: str
    ) -> tuple[list[str], dict[str, str]]:
        """
        Run HeidelTime on each span text.

        Returns
        -------
        all_dates : sorted list of all unique resolved ISO-8601 dates
        span_to_date : mapping of original span text → resolved date
                       (used by TemporalIE for per-span linking)
        """
        all_dates: list[str] = []
        span_to_date: dict[str, str] = {}
        for span in span_texts:
            dates = self._ht.extract_resolved_dates(span, pub_date)
            if dates:
                span_to_date[span] = dates[0]   # first resolved date for this span
                all_dates.extend(dates)
        # Deduplicate and sort
        all_dates = sorted(set(all_dates))
        return all_dates, span_to_date

    def normalize_chunk(self, chunk: dict) -> dict:
        """
        Add T_start / T_end / epoch / all_resolved_dates / span_to_date fields.

        The chunk must have:
          • "published_date" : str  (YYYY-MM-DD, used as DCT and ultimate fallback)
          • "entities"       : list (GLiNER entities, may be empty)
          • "text"           : str  (raw chunk text, used for full-text fallback)

        New fields added
        ----------------
        all_resolved_dates : list of ALL dates found (not just min/max).
                             TemporalIE uses this to link events to every
                             candidate date, avoiding wrong single-date assignment.
        span_to_date       : dict mapping each GLiNER span text to its resolved
                             ISO-8601 date (enables precise per-span linking).
        """
        pub_date = chunk.get("published_date", "")
        entities = chunk.get("entities", [])
        text = chunk.get("text", "")

        t_start: str = pub_date
        t_end:   str = pub_date
        source:  str = "pub_date"
        all_resolved: list[str] = []
        span_to_date: dict[str, str] = {}

        # ── Stage 1: HeidelTime on GLiNER DATE/TIME spans ────────────────────
        span_texts = [
            e["text"]
            for e in entities
            if e.get("label") in self.config.span_labels
            and e.get("score", 0.0) >= self.config.min_span_score
        ]

        had_span_dates = False
        if span_texts:
            all_resolved, span_to_date = self._resolve_spans(span_texts, pub_date)
            if all_resolved:
                had_span_dates = True
                t_start = all_resolved[0]
                t_end   = all_resolved[-1]
                source  = "gliner_spans"
                logger.debug(
                    "chunk %s: gliner_spans → T_start=%s T_end=%s | all=%s",
                    chunk.get("chunk_id"), t_start, t_end, all_resolved,
                )

        # ── Stage 2: full chunk text — always merge when possible ─────────────
        # GLiNER misses some years/expressions; previously we only ran this when
        # *no* span dates resolved, so chunks like chunk_id 40 lost e.g. "2020"
        # even though HeidelTime finds it in the full text.
        #
        # Exception: user *questions* (chunk_id "query") already supply GLiNER
        # DATE/TIME spans — running HeidelTime again on the full question merges
        # deictic phrases ("that weekend") resolved relative to DCT, which can
        # explode q_end into the wrong year. Skip full-text merge when spans
        # already yielded dates — unless every span is a vague word like "year",
        # in which case the full sentence ("that year") must be processed.
        is_user_query = str(chunk.get("chunk_id", "")) == "query"
        query_temporal_spans_all_vague = (
            is_user_query
            and span_texts
            and all(_vague_temporal_span_text(s) for s in span_texts)
        )
        skip_full_text_for_query = (
            is_user_query
            and had_span_dates
            and not query_temporal_spans_all_vague
        )
        if text.strip() and not skip_full_text_for_query:
            full_dates = self._ht.extract_resolved_dates(text, pub_date)
            if full_dates:
                merged = sorted(set(all_resolved + full_dates))
                if merged:
                    span_set = set(all_resolved)
                    extra = [d for d in full_dates if d not in span_set]
                    all_resolved = merged
                    t_start = all_resolved[0]
                    t_end   = all_resolved[-1]
                    if had_span_dates:
                        source = "gliner_spans+full_text" if extra else "gliner_spans"
                    else:
                        source = "full_text"
                    logger.debug(
                        "chunk %s: full_text merge → T_start=%s T_end=%s | extra=%s",
                        chunk.get("chunk_id"), t_start, t_end, extra,
                    )

        # ── Stage 3: Ultimate pub_date fallback (already set as default) ─────
        if source == "pub_date":
            logger.debug("chunk %s: pub_date fallback (no temporal expressions found)", chunk.get("chunk_id"))

        return {
            **chunk,
            "T_start":            t_start,
            "T_end":              t_end,
            "T_start_epoch":      iso_to_epoch(t_start),
            "T_end_epoch":        iso_to_epoch(t_end),
            "temporal_source":    source,
            "all_resolved_dates": all_resolved,   # ALL dates, not just min/max
            "span_to_date":       span_to_date,   # span text → resolved date
        }

    # ── JSONL streaming ───────────────────────────────────────────────────────

    def _iter_jsonl(self, path: Path) -> Iterator[dict]:
        """Yield parsed JSON objects from a JSONL file, skipping blank and malformed lines."""
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping bad JSON at line %d: %s", lineno, exc)

    def process_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        max_chunks: int | None = None,
    ) -> dict:
        """
        Stream GLiNER-enriched JSONL → normalize → write output JSONL.

        Returns summary statistics dict.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        logger.info("Normalizer  input : %s", input_path)
        logger.info("Normalizer  output: %s", output_path)

        source_counts: dict[str, int] = {"gliner_spans": 0, "full_text": 0, "pub_date": 0}
        total = 0
        t0 = time.perf_counter()

        with output_path.open("w", encoding="utf-8") as out_fh:
            for chunk in self._iter_jsonl(input_path):
                if max_chunks is not None and total >= max_chunks:
                    break
                enriched = self.normalize_chunk(chunk)
                out_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                source_counts[enriched["temporal_source"]] = (
                    source_counts.get(enriched["temporal_source"], 0) + 1
                )
                total += 1

        elapsed = time.perf_counter() - t0
        stats = {
            "total_chunks": total,
            "elapsed_s": round(elapsed, 2),
            "source_counts": source_counts,
        }
        logger.info(
            "Normalizer done: %d chunks in %.1f s | sources: %s",
            total, elapsed, source_counts,
        )
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="normalizer",
        description="Build step 5: Rule-Based Temporal Normalizer (HeidelTime + TIMEX3 fallback).",
    )
    p.add_argument("--input",  "-i", required=True, help="GLiNER-enriched input JSONL.")
    p.add_argument("--output", "-o", required=True, help="Normalized output JSONL.")
    p.add_argument("--language", default="English",  help="HeidelTime language (default: English).")
    p.add_argument("--doc-type", default="news",     help="HeidelTime document type (default: news).")
    p.add_argument("--max-chunks", "-n", type=int, default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    cfg = NormalizerConfig(language=args.language, document_type=args.doc_type)
    norm = TemporalNormalizer(cfg)
    stats = norm.process_jsonl(args.input, args.output, max_chunks=args.max_chunks)

    print("\n" + "═" * 50)
    print("  Temporal Normalizer Complete")
    print("═" * 50)
    print(f"  Chunks processed : {stats['total_chunks']}")
    print(f"  Elapsed          : {stats['elapsed_s']} s")
    print("  Source breakdown :")
    for src, cnt in stats["source_counts"].items():
        print(f"      {src:<16} {cnt:>6}")
    print("═" * 50)
    print(f"  Output → {args.output}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
