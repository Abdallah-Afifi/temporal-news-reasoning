"""
temporal_ie.py
===============
Build step 6: link events and entities to dates inside each chunk.

Simple summary
--------------
After normalization we know which calendar dates appear in a chunk. This file builds
small graphs: event-to-date, entity-to-date, and event-to-entity edges. Those edges are
saved as lists on each chunk row and later copied into Neo4j by ``kg_store``.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Rule-Based Normalizer]  →  THIS MODULE  →  [Embed Input Builder]

What it does
------------
Produces three types of canonical KG edges from GLiNER + Normalizer output:

  Edge type 1: EVENT to DATE   (event_date_edges)
      (Football Manager 2017) --[:HAPPENED_ON]--> (2016-11-04)
      Answers: "when did this event happen?"

  Edge type 2: ENTITY to DATE  (entity_date_edges)
      (eBay) --[:MENTIONED_ON]--> (2016-10-18)
      Answers: "when was this entity active / mentioned?"

  Edge type 3: EVENT to ENTITY  (event_entity_edges)
      (Football Manager 2017) --[:INVOLVES]--> (Miles Jacobson)
      Answers: "which entities are associated with this event?"

All three are also collected in a unified ``kg_triples`` list in the format:
    { "subject", "predicate", "object", "object_type", "chunk_id" }
ready for direct import into Neo4j via MERGE statements.

Linking strategy (for edge types 1 & 2)
-----------------------------------------
Link to ALL HeidelTime-resolved dates in the chunk (not just the nearest one).
Single-date heuristics (nearest by char, same sentence) are unreliable because
a scene-setting phrase like "November is approaching" appears close to an event
that actually releases on "November 4" three sentences later.
Linking to all dates lets the retrieval layer decide relevance at query time.
same_sentence_dates is stored as a hint for KG edge weighting.

Output schema additions per chunk
----------------------------------
    "event_date_edges"   : list  - EVENT --[:HAPPENED_ON]-->  DATE
    "entity_date_edges"  : list  - ENTITY --[:MENTIONED_ON]--> DATE
    "event_entity_edges" : list  - EVENT --[:INVOLVES]-->     ENTITY
    "kg_triples"         : list  - unified (subject, predicate, object) list
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

from .utils.heideltime_wrapper import iso_to_epoch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalIEConfig:
    """
    Configuration for Temporal IE.

    Attributes
    ----------
    event_label:
        GLiNER label treated as an event.
    entity_label:
        GLiNER label treated as a named entity.
    date_labels:
        GLiNER labels that carry temporal information.
    min_event_score:
        Minimum GLiNER confidence for EVENT spans to be included.
    min_entity_score:
        Minimum GLiNER confidence for ENTITY spans to be included.
    """
    event_label: str = "EVENT"
    entity_label: str = "ENTITY"
    date_labels: tuple[str, ...] = ("DATE", "TIME")
    min_event_score: float = 0.35
    min_entity_score: float = 0.40


# ──────────────────────────────────────────────────────────────────────────────
# Sentence splitter (no NLTK dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """
    Split *text* into sentences using punctuation heuristics.

    Returns list of (start_char, end_char, sentence_text) tuples.
    Keeps all characters so offsets align with the original text.
    """
    # Split on . ! ? followed by whitespace + capital, or end of string
    pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
    parts: list[tuple[int, int, str]] = []
    prev = 0
    for m in pattern.finditer(text):
        end = m.start() + 1          # include the punctuation
        parts.append((prev, end, text[prev:end]))
        prev = m.end()
    if prev < len(text):
        parts.append((prev, len(text), text[prev:]))
    return parts


def _find_sentence_idx(char_start: int, sentences: list[tuple[int, int, str]]) -> int:
    """Return the index of the sentence that contains char_start."""
    for idx, (s, e, _) in enumerate(sentences):
        if s <= char_start < e:
            return idx
    return len(sentences) - 1   # clamp to last sentence


# ──────────────────────────────────────────────────────────────────────────────
# Temporal IE
# ──────────────────────────────────────────────────────────────────────────────

class TemporalIE:
    """
    Full Temporal IE: produces EVENT→DATE, ENTITY→DATE, and EVENT→ENTITY edges.

    All edges are also collected in ``kg_triples``, a unified
    (subject, predicate, object) list ready for Neo4j MERGE import.
    """

    def __init__(self, config: TemporalIEConfig | None = None) -> None:
        self.config = config or TemporalIEConfig()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _date_link(
        self,
        all_resolved: list[str],
        t_start: str,
        span_to_date: dict,
        date_spans: list[dict],
        subject_span: dict,
        sentences: list,
    ) -> tuple[list[str], list[int], list[str], str]:
        """
        Return (linked_dates, linked_epochs, same_sentence_dates, link_type)
        for any subject span (EVENT or ENTITY).
        """
        subj_sent_idx = _find_sentence_idx(subject_span.get("start", 0), sentences)

        same_sent_dates = sorted(set(
            span_to_date[d["text"]]
            for d in date_spans
            if _find_sentence_idx(d.get("start", 0), sentences) == subj_sent_idx
            and d["text"] in span_to_date
        ))

        if all_resolved:
            linked    = all_resolved
            link_type = "all_chunk_dates"
        else:
            linked    = [t_start]
            link_type = "pub_date_fallback"

        return linked, [iso_to_epoch(d) for d in linked], same_sent_dates, link_type

    # ── Core logic ────────────────────────────────────────────────────────────

    def link_chunk(self, chunk: dict) -> dict:
        """
        Add all three edge types + unified kg_triples to a normalizer chunk.

        Produces:
          event_date_edges   - EVENT  --[:HAPPENED_ON]-->  DATE
          entity_date_edges  - ENTITY --[:MENTIONED_ON]--> DATE
          event_entity_edges - EVENT  --[:INVOLVES]-->     ENTITY
          kg_triples         - unified list of all above for Neo4j import
        """
        chunk_id     = chunk.get("chunk_id")
        entities     = chunk.get("entities", [])
        t_start      = chunk.get("T_start", chunk.get("published_date", ""))
        all_resolved = chunk.get("all_resolved_dates", [])
        span_to_date = chunk.get("span_to_date", {})
        text         = chunk.get("text", "")

        event_spans  = [
            e for e in entities
            if e.get("label") == self.config.event_label
            and e.get("score", 0.0) >= self.config.min_event_score
        ]
        entity_spans = [
            e for e in entities
            if e.get("label") == self.config.entity_label
            and e.get("score", 0.0) >= self.config.min_entity_score
        ]
        date_spans   = [
            e for e in entities if e.get("label") in self.config.date_labels
        ]

        sentences = _split_sentences(text)

        # ── Edge type 1: EVENT --[:HAPPENED_ON]--> DATE ───────────────────────
        event_date_edges: list[dict] = []
        for event in event_spans:
            linked, epochs, same_sent, link_type = self._date_link(
                all_resolved, t_start, span_to_date, date_spans, event, sentences
            )
            event_date_edges.append({
                "event":               event["text"],
                "event_start":         event.get("start"),
                "event_end":           event.get("end"),
                "event_score":         event.get("score"),
                "sentence_idx":        _find_sentence_idx(event.get("start", 0), sentences),
                "sentence":            sentences[_find_sentence_idx(event.get("start", 0), sentences)][2].strip() if sentences else "",
                "linked_dates":        linked,
                "linked_epochs":       epochs,
                "same_sentence_dates": same_sent,
                "link_type":           link_type,
            })

        # ── Edge type 2: ENTITY --[:MENTIONED_ON]--> DATE ─────────────────────
        entity_date_edges: list[dict] = []
        # Deduplicate entity texts so "eBay" appearing 3× creates one KG node
        seen_entities: set[str] = set()
        for entity in entity_spans:
            entity_text = entity["text"].strip()
            if entity_text in seen_entities:
                continue
            seen_entities.add(entity_text)

            linked, epochs, same_sent, link_type = self._date_link(
                all_resolved, t_start, span_to_date, date_spans, entity, sentences
            )
            entity_date_edges.append({
                "entity":              entity_text,
                "entity_score":        entity.get("score"),
                "linked_dates":        linked,
                "linked_epochs":       epochs,
                "same_sentence_dates": same_sent,
                "link_type":           link_type,
            })

        # ── Edge type 3: EVENT --[:INVOLVES]--> ENTITY ────────────────────────
        event_entity_edges: list[dict] = []
        unique_entity_texts = list(dict.fromkeys(
            e["text"].strip() for e in entity_spans
        ))
        for event in event_spans:
            for entity_text in unique_entity_texts:
                event_entity_edges.append({
                    "event":  event["text"],
                    "entity": entity_text,
                })

        # ── Unified kg_triples — all edges in one flat list ───────────────────
        kg_triples: list[dict] = []

        for edge in event_date_edges:
            for d in edge["linked_dates"]:
                kg_triples.append({
                    "subject":     edge["event"],
                    "predicate":   "HAPPENED_ON",
                    "object":      d,
                    "object_type": "Date",
                    "chunk_id":    chunk_id,
                })

        for edge in entity_date_edges:
            for d in edge["linked_dates"]:
                kg_triples.append({
                    "subject":     edge["entity"],
                    "predicate":   "MENTIONED_ON",
                    "object":      d,
                    "object_type": "Date",
                    "chunk_id":    chunk_id,
                })

        for edge in event_entity_edges:
            kg_triples.append({
                "subject":     edge["event"],
                "predicate":   "INVOLVES",
                "object":      edge["entity"],
                "object_type": "Entity",
                "chunk_id":    chunk_id,
            })

        logger.debug(
            "chunk %s: %d event_date + %d entity_date + %d event_entity = %d triples",
            chunk_id,
            len(event_date_edges), len(entity_date_edges),
            len(event_entity_edges), len(kg_triples),
        )

        return {
            **chunk,
            "event_date_edges":   event_date_edges,
            "entity_date_edges":  entity_date_edges,
            "event_entity_edges": event_entity_edges,
            "kg_triples":         kg_triples,
        }

    # ── JSONL streaming ───────────────────────────────────────────────────────

    def _iter_jsonl(self, path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Bad JSON at line %d: %s", lineno, exc)

    def process_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        max_chunks: int | None = None,
    ) -> dict:
        """Stream normalizer JSONL → link events → write output JSONL."""
        input_path  = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        logger.info("TemporalIE  input : %s", input_path)
        logger.info("TemporalIE  output: %s", output_path)

        total_chunks = 0
        total_edges  = 0
        link_type_counts: dict[str, int] = {}
        t0 = time.perf_counter()

        event_date_total   = 0
        entity_date_total  = 0
        event_entity_total = 0
        kg_triple_total    = 0

        with output_path.open("w", encoding="utf-8") as out_fh:
            for chunk in self._iter_jsonl(input_path):
                if max_chunks is not None and total_chunks >= max_chunks:
                    break
                enriched = self.link_chunk(chunk)
                out_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_chunks     += 1
                event_date_total  += len(enriched.get("event_date_edges",   []))
                entity_date_total += len(enriched.get("entity_date_edges",  []))
                event_entity_total+= len(enriched.get("event_entity_edges", []))
                kg_triple_total   += len(enriched.get("kg_triples",         []))

        elapsed = time.perf_counter() - t0
        stats = {
            "total_chunks":       total_chunks,
            "event_date_edges":   event_date_total,
            "entity_date_edges":  entity_date_total,
            "event_entity_edges": event_entity_total,
            "kg_triples":         kg_triple_total,
            "elapsed_s":          round(elapsed, 2),
        }
        logger.info(
            "TemporalIE done: %d chunks | %d event_date + %d entity_date + "
            "%d event_entity = %d total triples (%.1f s)",
            total_chunks, event_date_total, entity_date_total,
            event_entity_total, kg_triple_total, elapsed,
        )
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments, run the Temporal IE linker, and print a summary."""
    p = argparse.ArgumentParser(
        prog="temporal_ie",
        description="Build step 6: Temporal IE - sentence-aware Event-Time Linking.",
    )
    p.add_argument("--input",  "-i", required=True)
    p.add_argument("--output", "-o", required=True)
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

    tie = TemporalIE()
    stats = tie.process_jsonl(args.input, args.output, max_chunks=args.max_chunks)

    print("\n" + "═" * 55)
    print("  Temporal IE Complete")
    print("═" * 55)
    print(f"  Chunks            : {stats['total_chunks']}")
    print(f"  Event→Date edges  : {stats['event_date_edges']}")
    print(f"  Entity→Date edges : {stats['entity_date_edges']}")
    print(f"  Event→Entity edges: {stats['event_entity_edges']}")
    print(f"  KG triples total  : {stats['kg_triples']}")
    print(f"  Elapsed           : {stats['elapsed_s']} s")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
