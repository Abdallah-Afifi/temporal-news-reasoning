"""
sql_store.py
=============
Build step 10: load chunk rows into PostgreSQL.

Simple summary
--------------
Reads ``temporal_ie`` JSONL and upserts one ``chunks`` table row per ``chunk_id`` with
text, publication date, GLiNER-normalized ``T_start`` / ``T_end``, and epoch copies.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Temporal IE output JSONL]  →  THIS MODULE  →  [PostgreSQL]

What it stores
--------------
One row per chunk in a ``chunks`` table:

    chunk_id          TEXT PRIMARY KEY
    source_doc_id     TEXT
    title             TEXT
    chunk_index       INT
    total_chunks      INT
    text              TEXT          - dereferenced chunk text
    token_count       INT
    published_date    TEXT          - YYYY-MM-DD
    T_start           TEXT          - earliest resolved date (YYYY-MM-DD)
    T_end             TEXT          - latest  resolved date  (YYYY-MM-DD)
    T_start_epoch     BIGINT        - UTC Unix seconds
    T_end_epoch       BIGINT        - UTC Unix seconds
    temporal_source   TEXT          - "gliner_spans" | "full_text" | "pub_date"

Usage (CLI)
-----------
    python -m temporal_rag.sql_store \\
        --input   temporal_ie.jsonl \\
        --host    localhost \\
        --port    5432 \\
        --dbname  temporal_rag \\
        --user    postgres \\
        --password secret

    # Dry-run (no DB write, just prints row count):
    python -m temporal_rag.sql_store --input temporal_ie.jsonl --dry-run

Usage (library)
---------------
    from temporal_rag.sql_store import SQLStore, SQLStoreConfig
    store = SQLStore(SQLStoreConfig(host="localhost", dbname="temporal_rag",
                                    user="postgres", password="secret"))
    store.setup_schema()
    stats = store.load_jsonl("temporal_ie_output.jsonl")
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

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT        PRIMARY KEY,
    source_doc_id   TEXT,
    title           TEXT,
    chunk_index     INT,
    total_chunks    INT,
    text            TEXT,
    token_count     INT,
    published_date  TEXT,
    T_start         TEXT,
    T_end           TEXT,
    T_start_epoch   BIGINT,
    T_end_epoch     BIGINT,
    temporal_source TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_T_start_epoch ON chunks (T_start_epoch);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_T_end_epoch   ON chunks (T_end_epoch);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source_doc    ON chunks (source_doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_pub_date      ON chunks (published_date);",
]

# Existing DBs may have been created before new columns were added to _CREATE_TABLE.
# CREATE TABLE IF NOT EXISTS does not fix old tables: add ALTER snippets below instead.
_ALTER_ADD_MISSING_COLUMNS = [
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS temporal_source TEXT;",
]

_UPSERT = """
INSERT INTO chunks
    (chunk_id, source_doc_id, title, chunk_index, total_chunks,
     text, token_count, published_date,
     T_start, T_end, T_start_epoch, T_end_epoch, temporal_source)
VALUES
    (%(chunk_id)s, %(source_doc_id)s, %(title)s, %(chunk_index)s, %(total_chunks)s,
     %(text)s, %(token_count)s, %(published_date)s,
     %(T_start)s, %(T_end)s, %(T_start_epoch)s, %(T_end_epoch)s, %(temporal_source)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    text            = EXCLUDED.text,
    T_start         = EXCLUDED.T_start,
    T_end           = EXCLUDED.T_end,
    T_start_epoch   = EXCLUDED.T_start_epoch,
    T_end_epoch     = EXCLUDED.T_end_epoch,
    temporal_source = EXCLUDED.temporal_source;
"""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SQLStoreConfig:
    """
    Settings for connecting to PostgreSQL and controlling batch inserts.

    Attributes
    ----------
    host:
        Database server hostname. Default ``"localhost"``.
    port:
        Port the Postgres server listens on. Default ``5432``.
    dbname:
        Name of the database. Default ``"temporal_rag"``.
    user:
        Database user. Default ``"postgres"``.
    password:
        Password for the database user. Default ``""`` (empty).
    batch_size:
        Number of rows to collect before committing to the database.
        Larger values are faster but use more memory. Default ``500``.
    """
    host:     str = "localhost"
    port:     int = 5432
    dbname:   str = "temporal_rag"
    user:     str = "postgres"
    password: str = ""
    batch_size: int = 500


# ─────────────────────────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_row(chunk: dict) -> dict:
    """Extract the SQL columns from a Temporal IE chunk dict."""
    pub  = chunk.get("published_date", "")
    return {
        "chunk_id":       str(chunk.get("chunk_id", "")),
        "source_doc_id":  chunk.get("source_doc_id", ""),
        "title":          chunk.get("title", ""),
        "chunk_index":    chunk.get("chunk_index", 0),
        "total_chunks":   chunk.get("total_chunks", 1),
        "text":           chunk.get("text", ""),
        "token_count":    chunk.get("token_count", 0),
        "published_date": pub,
        "T_start":        chunk.get("T_start",       pub),
        "T_end":          chunk.get("T_end",         pub),
        "T_start_epoch":  chunk.get("T_start_epoch", 0),
        "T_end_epoch":    chunk.get("T_end_epoch",   0),
        "temporal_source": chunk.get("temporal_source", "pub_date"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

class SQLStore:
    """
    Loads Temporal IE JSONL into a PostgreSQL ``chunks`` table.

    Example
    -------
    >>> cfg   = SQLStoreConfig(host="localhost", dbname="temporal_rag",
    ...                         user="postgres", password="secret")
    >>> store = SQLStore(cfg)
    >>> store.setup_schema()
    >>> stats = store.load_jsonl("temporal_ie_output.jsonl")
    >>> print(stats)
    {'total_rows': 85, 'elapsed_s': 0.42}
    """

    def __init__(self, config: SQLStoreConfig | None = None) -> None:
        self.config = config or SQLStoreConfig()
        self._conn  = None

    # ── Connection ─────────────────────────────────────────────────────────

    def _get_conn(self):
        """Return the PostgreSQL connection, creating it on first call (lazy init)."""
        if self._conn is None or self._conn.closed:
            try:
                import psycopg2
            except ImportError as exc:
                raise ImportError(
                    "psycopg2 is not installed. Run: pip install psycopg2-binary"
                ) from exc
            cfg = self.config
            self._conn = psycopg2.connect(
                host=cfg.host, port=cfg.port,
                dbname=cfg.dbname, user=cfg.user, password=cfg.password,
            )
        return self._conn

    def close(self) -> None:
        """Close the database connection and release the socket."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── Schema ─────────────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """Create chunks table and indexes if they don't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
            for alter_sql in _ALTER_ADD_MISSING_COLUMNS:
                cur.execute(alter_sql)
            for idx_sql in _CREATE_INDEXES:
                cur.execute(idx_sql)
        conn.commit()
        logger.info("Schema ready (chunks table + indexes).")

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

    # ── Loader ─────────────────────────────────────────────────────────────

    def load_jsonl(
        self,
        input_path: str | Path,
        max_chunks: int | None = None,
    ) -> dict:
        """
        Stream Temporal IE JSONL and insert into PostgreSQL.

        Returns
        -------
        dict with keys: total_rows, elapsed_s
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        conn = self._get_conn()
        total = 0
        t0 = time.perf_counter()
        batch: list[dict] = []

        def _flush(batch: list[dict]) -> None:
            with conn.cursor() as cur:
                for row in batch:
                    cur.execute(_UPSERT, row)
            conn.commit()

        for chunk in self._iter_jsonl(input_path):
            if max_chunks is not None and total >= max_chunks:
                break
            batch.append(_chunk_to_row(chunk))
            total += 1
            if len(batch) >= self.config.batch_size:
                _flush(batch)
                batch = []
                logger.info("Flushed %d rows so far …", total)

        if batch:
            _flush(batch)

        elapsed = time.perf_counter() - t0
        stats = {"total_rows": total, "elapsed_s": round(elapsed, 2)}
        logger.info("SQL store done: %d rows in %.2f s", total, elapsed)
        return stats

    # ── Dry-run ────────────────────────────────────────────────────────────

    @staticmethod
    def dry_run(input_path: str | Path) -> dict:
        """Parse JSONL and count rows without connecting to any database."""
        input_path = Path(input_path)
        total = 0
        t0 = time.perf_counter()
        with input_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    try:
                        json.loads(raw)
                        total += 1
                    except json.JSONDecodeError:
                        pass
        elapsed = time.perf_counter() - t0
        print(f"\nDry-run: {total} rows would be inserted  ({elapsed:.2f} s)\n")
        return {"total_rows": total, "elapsed_s": round(elapsed, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the SQL store CLI."""
    p = argparse.ArgumentParser(
        prog="sql_store",
        description="Build step 10: Load Temporal IE JSONL into PostgreSQL chunks table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",    "-i", required=True, help="Temporal IE output JSONL.")
    p.add_argument("--host",           default="localhost")
    p.add_argument("--port",     "-P", type=int, default=5432)
    p.add_argument("--dbname",   "-d", default="temporal_rag")
    p.add_argument("--user",     "-u", default="postgres")
    p.add_argument("--password", "-p", default="")
    p.add_argument("--batch-size",     type=int, default=500,
                   help="Rows per commit batch. (default: 500)")
    p.add_argument("--max-chunks", "-n", type=int, default=None)
    p.add_argument("--dry-run",    action="store_true",
                   help="Parse and count rows without writing to DB.")
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
        SQLStore.dry_run(args.input)
        return

    cfg   = SQLStoreConfig(
        host=args.host, port=args.port,
        dbname=args.dbname, user=args.user, password=args.password,
        batch_size=args.batch_size,
    )
    store = SQLStore(cfg)
    store.setup_schema()
    stats = store.load_jsonl(args.input, max_chunks=args.max_chunks)
    store.close()

    print("\n" + "═" * 50)
    print("  SQL Store Complete")
    print("═" * 50)
    print(f"  Rows inserted : {stats['total_rows']}")
    print(f"  Elapsed       : {stats['elapsed_s']} s")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
