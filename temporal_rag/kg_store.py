"""
kg_store.py
============
Build step 11: load chunk graphs into Neo4j.

Simple summary
--------------
Reads ``temporal_ie`` JSONL rows and creates Chunk / Entity / Event / Date nodes plus
the relationships expected by the query UI (CONTAINS_ENTITY, HAPPENED_ON, etc.).

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Temporal IE output JSONL]  →  THIS MODULE  →  [Neo4j]

Node types created
------------------
  (:Chunk   {chunk_id, source_doc_id, title, text, published_date,
              T_start, T_end, T_start_epoch, T_end_epoch})
  (:Date    {date, epoch})
  (:Event   {text})
  (:Entity  {text})

Relationship types created
--------------------------
  (:Chunk)  -[:CONTAINS_EVENT]->  (:Event)
  (:Chunk)  -[:CONTAINS_ENTITY]-> (:Entity)
  (:Event)  -[:HAPPENED_ON]->     (:Date)
  (:Entity) -[:MENTIONED_ON]->    (:Date)
  (:Event)  -[:INVOLVES]->        (:Entity)

Usage (CLI)
-----------
    # From repository root:
    python -m temporal_rag.kg_store \\
        --input   temporal_ie_output.jsonl \\
        --uri     bolt://localhost:7687 \\
        --user    neo4j \\
        --password secret

    # Dry-run (just prints triple counts):
    python -m temporal_rag.kg_store --input temporal_ie_output.jsonl --dry-run

Usage (library)
---------------
    from temporal_rag.kg_store import KGStore, KGStoreConfig
    store = KGStore(KGStoreConfig(uri="bolt://localhost:7687",
                                   user="neo4j", password="secret"))
    store.setup_schema()
    stats = store.load_jsonl("temporal_ie_output.jsonl")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .utils.heideltime_wrapper import iso_to_epoch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cypher statements
# ─────────────────────────────────────────────────────────────────────────────

# Indexes / constraints: run once at startup.
_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT chunk_id   IF NOT EXISTS FOR (c:Chunk)  REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT date_val   IF NOT EXISTS FOR (d:Date)   REQUIRE d.date IS UNIQUE",
    "CREATE INDEX event_text      IF NOT EXISTS FOR (e:Event)  ON (e.text)",
    "CREATE INDEX entity_text     IF NOT EXISTS FOR (e:Entity) ON (e.text)",
]

# Chunk node
_MERGE_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET   c.source_doc_id  = $source_doc_id,
      c.title          = $title,
      c.text           = $text,
      c.published_date = $published_date,
      c.T_start        = $T_start,
      c.T_end          = $T_end,
      c.T_start_epoch  = $T_start_epoch,
      c.T_end_epoch    = $T_end_epoch
"""

# Date node
_MERGE_DATE = """
MERGE (d:Date {date: $date})
SET   d.epoch = $epoch
"""

# Event node + link to Chunk
_MERGE_EVENT_CHUNK = """
MERGE (e:Event {text: $event_text})
WITH  e
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (c)-[:CONTAINS_EVENT]->(e)
"""

# Entity node + link to Chunk
_MERGE_ENTITY_CHUNK = """
MERGE (e:Entity {text: $entity_text})
WITH  e
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (c)-[:CONTAINS_ENTITY]->(e)
"""

# EVENT -[:HAPPENED_ON]-> DATE
_MERGE_EVENT_DATE = """
MATCH (ev:Event  {text: $event_text})
MATCH (d:Date    {date: $date})
MERGE (ev)-[:HAPPENED_ON]->(d)
"""

# ENTITY -[:MENTIONED_ON]-> DATE
_MERGE_ENTITY_DATE = """
MATCH (en:Entity {text: $entity_text})
MATCH (d:Date    {date: $date})
MERGE (en)-[:MENTIONED_ON]->(d)
"""

# EVENT -[:INVOLVES]-> ENTITY
_MERGE_EVENT_ENTITY = """
MATCH (ev:Event  {text: $event_text})
MATCH (en:Entity {text: $entity_text})
MERGE (ev)-[:INVOLVES]->(en)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KGStoreConfig:
    """Connection settings for the Neo4j Knowledge Graph.

    Attributes
    ----------
    uri:
        Bolt connection URL for Neo4j (e.g. ``"bolt://localhost:7687"``).
    user:
        Neo4j username.
    password:
        Neo4j password.
    database:
        Name of the Neo4j database to write to (usually ``"neo4j"``).
    batch_size:
        How many chunks to commit in one database transaction.
        Larger values are faster but use more memory.
    """
    uri:        str = "bolt://localhost:7687"
    user:       str = "neo4j"
    password:   str = "temporalrag"
    database:   str = "neo4j"
    batch_size: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

class KGStore:
    """
    Loads Temporal IE JSONL into a Neo4j Temporal Knowledge Graph.

    Creates Chunk, Date, Event, Entity nodes and the five relationship types
    described in the module docstring.

    Example
    -------
    >>> cfg   = KGStoreConfig(uri="bolt://localhost:7687",
    ...                        user="neo4j", password="secret")
    >>> store = KGStore(cfg)
    >>> store.setup_schema()
    >>> stats = store.load_jsonl("temporal_ie_output.jsonl")
    """

    def __init__(self, config: KGStoreConfig | None = None) -> None:
        self.config = config or KGStoreConfig()
        self._driver = None

    # ── Driver ─────────────────────────────────────────────────────────────

    def _get_driver(self):
        """Return the Neo4j driver, creating it on first call (lazy init)."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise ImportError(
                    "neo4j driver not installed. Run: pip install neo4j"
                ) from exc
            cfg = self.config
            self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver and release the connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Schema ─────────────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """Create constraints and indexes (idempotent)."""
        driver = self._get_driver()
        with driver.session(database=self.config.database) as session:
            for stmt in _SCHEMA_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception as exc:
                    # Older Neo4j versions may not support IF NOT EXISTS syntax
                    logger.debug("Schema stmt skipped (%s): %s", exc, stmt[:60])
        logger.info("Neo4j schema ready.")

    # ── JSONL reader ───────────────────────────────────────────────────────

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict]:
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

    # ── Per-chunk writer ───────────────────────────────────────────────────

    @staticmethod
    def _write_chunk(tx, chunk: dict) -> dict:
        """
        Write one chunk and all its KG edges inside a transaction.
        Returns counts for stats.
        """
        chunk_id     = str(chunk.get("chunk_id", ""))
        pub_date     = chunk.get("published_date", "")
        t_start      = chunk.get("T_start",       pub_date)
        t_end        = chunk.get("T_end",         pub_date)
        t_start_ep   = chunk.get("T_start_epoch", 0)
        t_end_ep     = chunk.get("T_end_epoch",   0)

        # ── Chunk node ───────────────────────────────────────────────────
        tx.run(_MERGE_CHUNK, {
            "chunk_id":       chunk_id,
            "source_doc_id":  chunk.get("source_doc_id", ""),
            "title":          chunk.get("title", ""),
            "text":           chunk.get("text", ""),
            "published_date": pub_date,
            "T_start":        t_start,
            "T_end":          t_end,
            "T_start_epoch":  t_start_ep,
            "T_end_epoch":    t_end_ep,
        })

        # ── Date nodes (from all_resolved_dates) ─────────────────────────
        all_dates = chunk.get("all_resolved_dates", [])
        if not all_dates:
            all_dates = [t_start]
        for d in all_dates:
            tx.run(_MERGE_DATE, {"date": d, "epoch": iso_to_epoch(d)})

        # ── Event nodes + CHUNK→EVENT + EVENT→DATE edges ─────────────────
        event_date_count = 0
        for edge in chunk.get("event_date_edges", []):
            ev_text = edge.get("event", "").strip()
            if not ev_text:
                continue
            tx.run(_MERGE_EVENT_CHUNK, {"event_text": ev_text, "chunk_id": chunk_id})
            for d in edge.get("linked_dates", []):
                tx.run(_MERGE_EVENT_DATE, {"event_text": ev_text, "date": d})
                event_date_count += 1

        # ── Entity nodes + CHUNK→ENTITY + ENTITY→DATE edges ──────────────
        entity_date_count = 0
        for edge in chunk.get("entity_date_edges", []):
            en_text = edge.get("entity", "").strip()
            if not en_text:
                continue
            tx.run(_MERGE_ENTITY_CHUNK, {"entity_text": en_text, "chunk_id": chunk_id})
            for d in edge.get("linked_dates", []):
                tx.run(_MERGE_ENTITY_DATE, {"entity_text": en_text, "date": d})
                entity_date_count += 1

        # ── EVENT→ENTITY edges ────────────────────────────────────────────
        event_entity_count = 0
        for edge in chunk.get("event_entity_edges", []):
            ev_text = edge.get("event",  "").strip()
            en_text = edge.get("entity", "").strip()
            if ev_text and en_text:
                tx.run(_MERGE_EVENT_ENTITY,
                       {"event_text": ev_text, "entity_text": en_text})
                event_entity_count += 1

        return {
            "event_date":   event_date_count,
            "entity_date":  entity_date_count,
            "event_entity": event_entity_count,
        }

    # ── Loader ─────────────────────────────────────────────────────────────

    def load_jsonl(
        self,
        input_path: str | Path,
        max_chunks: int | None = None,
    ) -> dict:
        """
        Stream Temporal IE JSONL and write all nodes / edges to Neo4j.

        Returns summary statistics dict.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        driver = self._get_driver()
        total_chunks      = 0
        event_date_total  = 0
        entity_date_total = 0
        event_entity_total= 0
        t0 = time.perf_counter()

        chunk_buffer: list[dict] = []

        def _flush(buf: list[dict]) -> None:
            nonlocal event_date_total, entity_date_total, event_entity_total
            with driver.session(database=self.config.database) as session:
                for ch in buf:
                    counts = session.execute_write(KGStore._write_chunk, ch)
                    event_date_total   += counts["event_date"]
                    entity_date_total  += counts["entity_date"]
                    event_entity_total += counts["event_entity"]

        for chunk in self._iter_jsonl(input_path):
            if max_chunks is not None and total_chunks >= max_chunks:
                break
            chunk_buffer.append(chunk)
            total_chunks += 1
            if len(chunk_buffer) >= self.config.batch_size:
                _flush(chunk_buffer)
                chunk_buffer = []
                logger.info("Committed %d chunks so far …", total_chunks)

        if chunk_buffer:
            _flush(chunk_buffer)

        elapsed = time.perf_counter() - t0
        stats = {
            "total_chunks":       total_chunks,
            "event_date_edges":   event_date_total,
            "entity_date_edges":  entity_date_total,
            "event_entity_edges": event_entity_total,
            "elapsed_s":          round(elapsed, 2),
        }
        logger.info(
            "KG store done: %d chunks | %d ev→date + %d en→date + %d ev→en  (%.2f s)",
            total_chunks, event_date_total, entity_date_total,
            event_entity_total, elapsed,
        )
        return stats

    # ── Dry-run ────────────────────────────────────────────────────────────

    @staticmethod
    def dry_run(input_path: str | Path) -> dict:
        """Count triples from JSONL without connecting to Neo4j."""
        input_path = Path(input_path)
        total_chunks = 0
        ev_date = en_date = ev_en = 0
        with input_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    c = json.loads(raw)
                    total_chunks += 1
                    for e in c.get("event_date_edges", []):
                        ev_date += len(e.get("linked_dates", []))
                    for e in c.get("entity_date_edges", []):
                        en_date += len(e.get("linked_dates", []))
                    ev_en += len(c.get("event_entity_edges", []))
                except json.JSONDecodeError:
                    pass
        total_triples = ev_date + en_date + ev_en
        print(f"\nDry-run: {total_chunks} chunks → {total_triples} triples total")
        print(f"  EVENT→DATE   : {ev_date}")
        print(f"  ENTITY→DATE  : {en_date}")
        print(f"  EVENT→ENTITY : {ev_en}\n")
        return {"total_chunks": total_chunks,
                "event_date_edges": ev_date,
                "entity_date_edges": en_date,
                "event_entity_edges": ev_en}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the KG store CLI."""
    p = argparse.ArgumentParser(
        prog="kg_store",
        description="Build step 11: Load Temporal IE JSONL into Neo4j Temporal KG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",    "-i", required=True, help="Temporal IE output JSONL.")
    p.add_argument("--uri",            default="bolt://localhost:7687")
    p.add_argument("--user",     "-u", default="neo4j")
    p.add_argument("--password", "-p", default="temporalrag")
    p.add_argument("--database", "-d", default="neo4j")
    p.add_argument("--batch-size",     type=int, default=100)
    p.add_argument("--max-chunks", "-n", type=int, default=None)
    p.add_argument("--dry-run",    action="store_true",
                   help="Count triples without connecting to Neo4j.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.dry_run:
        KGStore.dry_run(args.input)
        return

    cfg   = KGStoreConfig(
        uri=args.uri, user=args.user, password=args.password,
        database=args.database, batch_size=args.batch_size,
    )
    store = KGStore(cfg)
    store.setup_schema()
    stats = store.load_jsonl(args.input, max_chunks=args.max_chunks)
    store.close()

    print("\n" + "═" * 55)
    print("  KG Store Complete")
    print("═" * 55)
    print(f"  Chunks written     : {stats['total_chunks']}")
    print(f"  EVENT→DATE edges   : {stats['event_date_edges']}")
    print(f"  ENTITY→DATE edges  : {stats['entity_date_edges']}")
    print(f"  EVENT→ENTITY edges : {stats['event_entity_edges']}")
    print(f"  Elapsed            : {stats['elapsed_s']} s")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
