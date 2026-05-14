#!/usr/bin/env python3
"""
Ask one question in the terminal and watch each pipeline stage print logs.

Simple picture
--------------
  Step 0: Read dates from the question (GLiNER + HeidelTime) -> ``q_start`` / ``q_end``.
  Step 1: Find similar chunks with FAISS (vector search only).
  Step 1b: Print the full candidate table (scores and titles).
  Step 2: Keep chunks whose saved date window overlaps the question window (PASS/DROP).
  Step 3: Pull graph-style facts from Neo4j, or from ``temporal_ie.jsonl`` if Neo4j is off.
  Step 4: Show text for chunks that survived Step 2.

Useful environment variables (set before ``python3``)::

  QUERY_DCT_DATE          Reference day for phrases like "yesterday" (YYYY-MM-DD).
  HEIDELTIME_TREETAGGER_HOME   Folder where TreeTagger is installed (Mac users often need this).
  ONLINE_QUERY_PARSER     Set to 0 to skip GLiNER + normalizer on the question (faster tests).
  FAISS_CANDIDATE_POOL    How many vector hits to consider before the time gate (default 10).

Database URLs use the usual ``NEO4J_*`` and ``PG*`` names (see README).

Run from repo root::

  python3 -m temporal_rag.interactive_query
"""

from __future__ import annotations

import os

# macOS: HF tokenizers plus OpenMP/MKL can crash if threads fight; force single-thread BLAS.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import calendar
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import faiss
from sentence_transformers import SentenceTransformer

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from .paths import PROJECT_ROOT


def _maybe_set_java_home_for_heideltime() -> None:
    """Pick JDK 11 on macOS when JAVA_HOME is missing so HeidelTime can start."""
    if os.environ.get("JAVA_HOME") or sys.platform != "darwin":
        return
    try:
        import subprocess

        out = subprocess.run(
            ["/usr/libexec/java_home", "-v", "11"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        home = (out.stdout or "").strip()
        if out.returncode == 0 and home and Path(home).is_dir():
            os.environ["JAVA_HOME"] = home
            jb = Path(home) / "bin"
            if (jb / "java").is_file():
                os.environ["PATH"] = str(jb) + os.pathsep + os.environ.get(
                    "PATH", ""
                )
    except (OSError, subprocess.TimeoutExpired):
        pass


_maybe_set_java_home_for_heideltime()

# --- Default paths for vectors, chunk metadata, and cached models ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_CACHE = PROJECT_ROOT / "encoder/model_cache"
INDEX_PATH = PROJECT_ROOT / "output/faiss_hnsw_ip.index"
LOOKUP_PATH = PROJECT_ROOT / "output/faiss_vector_lookup.jsonl"
TEMPORAL_IE_PATH = PROJECT_ROOT / "output/temporal_ie.jsonl"
DEREF_PATH = PROJECT_ROOT / "output/deref.jsonl"

# Demo: how many floats to show from the query embedding (full dim is model-dependent, usually 384).
DEMO_EMBEDDING_PREVIEW_DIMS = int(os.environ.get("DEMO_EMBEDDING_PREVIEW_DIMS", "16"))
# Slide diagrams often use k=100; raise for a closer match (slower to read in demos).
FAISS_CANDIDATE_POOL = 10
# How many survivors to keep after temporal overlap (FAISS order). Override at runtime: BOUNCER_TOP_K=3
BOUNCER_TOP_K = 1


def _bouncer_top_k() -> int:
    """How many PASS chunks to keep after overlap filter (reads ``BOUNCER_TOP_K``)."""
    try:
        return max(1, int(os.environ.get("BOUNCER_TOP_K", str(BOUNCER_TOP_K)) or "1"))
    except ValueError:
        return 1


def _online_query_parser_enabled() -> bool:
    """Whether GLiNER + normalizer should run on the question."""
    return os.environ.get("ONLINE_QUERY_PARSER", "1").lower() not in (
        "0", "false", "no", "off",
    )


def _transient_parser_failure(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a flaky network error worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError, InterruptedError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", 0) in (
        32,  # broken pipe (some platforms)
        50,  # network down
        51,  # unreachable
        54,  # connection reset (mac BSD)
        104,  # ECONNRESET
        110,  # ETIMEDOUT
    ):
        return True
    tn = type(exc).__name__
    if tn in ("RemoteDisconnected", "ProtocolError"):
        return True
    msg = str(exc).lower()
    needles = (
        "remote end closed connection",
        "connection aborted",
        "connection reset",
        "temporarily unavailable",
        "eof occurred in violation of protocol",
        "broken pipe",
        "max retries exceeded",
        "name or service not known",
    )
    return any(n in msg for n in needles)


# ONLINE_QUERY_PARSER=0 skips GLiNER + normalizer on the question (fast smoke tests).
def _query_dct_iso() -> str:
    """Reference calendar day for phrases like \"yesterday\" (env or today)."""

    raw = (os.environ.get("QUERY_DCT_DATE") or "").strip()
    if raw:
        return raw
    return date.today().isoformat()


_GLINER_EXTRACTOR = None
_TEMPORAL_NORMALIZER = None


def _get_gliner_extractor():
    """Create GLiNER on first use and cache it."""
    global _GLINER_EXTRACTOR
    if _GLINER_EXTRACTOR is None:
        from .gliner_extractor import GLiNERExtractor

        _GLINER_EXTRACTOR = GLiNERExtractor()
    return _GLINER_EXTRACTOR


def _get_temporal_normalizer():
    """Create TemporalNormalizer on first use and cache it."""
    global _TEMPORAL_NORMALIZER
    if _TEMPORAL_NORMALIZER is None:
        from .normalizer import TemporalNormalizer

        _TEMPORAL_NORMALIZER = TemporalNormalizer()
    return _TEMPORAL_NORMALIZER


_ST_MODEL: SentenceTransformer | None = None


def _load_chunk_text_previews() -> dict[str, str]:
    """Map chunk_id string -> short text preview (from deref or temporal_ie)."""
    out: dict[str, str] = {}
    for path in (DEREF_PATH, TEMPORAL_IE_PATH):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                cid = str(o.get("chunk_id", ""))
                if not cid:
                    continue
                text = str(o.get("text", "")).replace("\n", " ")
                if cid not in out and text:
                    out[cid] = text[:280] + ("..." if len(text) > 280 else "")
        if out:
            break
    return out


_TEXT_PREVIEW_CACHE: dict[str, str] | None = None


def text_preview_for_chunk(chunk_id: str) -> str:
    global _TEXT_PREVIEW_CACHE
    if _TEXT_PREVIEW_CACHE is None:
        _TEXT_PREVIEW_CACHE = _load_chunk_text_previews()
    return _TEXT_PREVIEW_CACHE.get(chunk_id, "")


def parse_query_temporal(query: str) -> dict:
    """Guess which calendar window the question asks about.

    Runs GLiNER on the question, then HeidelTime through TemporalNormalizer (same idea as
    offline chunks). Sets ``q_start`` / ``q_end`` when dates resolve; leaves them empty
    when parsing is off or HeidelTime only sees publication fallback.

    Year/month/day shape follows HeidelTime output (via ``HeidelTimeWrapper``). Mac users
    need a working TreeTagger path (``HEIDELTIME_TREETAGGER_HOME`` or ``vendor/treetagger-install``).
    """
    dct = _query_dct_iso()
    q_start: str | None = None
    q_end: str | None = None
    meta: dict = {
        "dct": dct,
        "mode": "none",
        "entities": [],
        "all_resolved": [],
        "span_to_date": {},
        "temporal_source": None,
    }

    if _online_query_parser_enabled() and query.strip():
        for attempt in range(3):
            try:
                gl = _get_gliner_extractor()
                spans = gl.extract_spans(query)
                meta["entities"] = spans
                norm = _get_temporal_normalizer()
                nchunk = norm.normalize_chunk(
                    {
                        "text": query,
                        "published_date": dct,
                        "entities": spans,
                        "chunk_id": "query",
                    }
                )
                meta["temporal_source"] = nchunk.get("temporal_source")
                meta["all_resolved"] = list(nchunk.get("all_resolved_dates", []))
                meta["span_to_date"] = dict(nchunk.get("span_to_date", {}))
                if nchunk.get("temporal_source") != "pub_date":
                    q_start = str(nchunk["T_start"])
                    q_end = str(nchunk["T_end"])
                    meta["mode"] = "gliner_normalizer"
                else:
                    meta["mode"] = "gliner_no_temporal_signal"
                break
            except Exception as exc:
                if attempt < 2 and _transient_parser_failure(exc):
                    time.sleep(0.6 * (2**attempt))
                    continue
                meta["parser_error"] = str(exc)
                meta["mode"] = "parser_error"
                break
    else:
        meta["mode"] = "online_parser_off" if not _online_query_parser_enabled() else "empty_query"

    return {
        "q_start": q_start,
        "q_end": q_end,
        "dct": dct,
        "meta": meta,
    }


def print_query_parse_banner(parsed: dict) -> None:
    m = parsed["meta"]
    print("\n[STEP 0] Query parser")
    print("-" * 72)
    print(f"   DCT: {parsed['dct']}")
    print(f"   Mode: {m.get('mode', '—')}")
    if m.get("temporal_source"):
        print(f"   temporal_source: {m['temporal_source']}")
    ents = m.get("entities") or []
    for e in ents:
        lab = e.get("label", "—")
        txt = e.get("text", "")
        print(f"   span  {lab}: {txt!r}")
    resolved = m.get("all_resolved") or []
    if resolved:
        print(f"   all_resolved_dates: {resolved}")
    std = m.get("span_to_date") or {}
    for span, iso in std.items():
        print(f"   span_to_date  {span!r} → {iso}")
    if parsed["q_start"] and parsed["q_end"]:
        print(f"   q_start, q_end  ({parsed['q_start']} .. {parsed['q_end']})")
    else:
        print("   q_start, q_end  (none)")
    if m.get("parser_error"):
        print(f"   error: {m['parser_error']}")
    print("-" * 72)


def _print_query_embedding_demo(vec: np.ndarray) -> None:
    v = np.asarray(vec, dtype=np.float64).ravel()
    n_prev = max(0, min(DEMO_EMBEDDING_PREVIEW_DIMS, len(v)))
    av = np.asarray(vec)
    print("   Query embedding (L2-normalized)")
    print(f"   shape: {tuple(av.shape)}   dtype: {av.dtype}")
    print(f"   L2 norm: {float(np.linalg.norm(v)):.6f}")
    if n_prev:
        head = ", ".join(f"{float(x):.5f}" for x in v[:n_prev])
        print(f"   first {n_prev} values: [{head}]")
    if len(v) > n_prev:
        print(f"   … +{len(v) - n_prev} dims (DEMO_EMBEDDING_PREVIEW_DIMS={DEMO_EMBEDDING_PREVIEW_DIMS})")
    print(
        f"   min={float(v.min()):.5f}  max={float(v.max()):.5f}  mean={float(v.mean()):.5f}"
    )


def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ST_MODEL = SentenceTransformer(
            MODEL_NAME, device=device, cache_folder=str(MODEL_CACHE)
        )
    return _ST_MODEL


def query_faiss(query: str, candidate_pool: int = FAISS_CANDIDATE_POOL) -> list[dict]:
    """Embed the query and search the FAISS index, returning up to ``candidate_pool`` chunks."""
    print(
        f"\n[STEP 1] FAISS: embed query and take up to {candidate_pool} "
        "candidate chunks..."
    )
    time.sleep(0.1)

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not LOOKUP_PATH.is_file():
        raise FileNotFoundError(f"Missing lookup: {LOOKUP_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    lookup = [
        json.loads(l)
        for l in LOOKUP_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    print(f"   Model: {MODEL_NAME}")
    model = _get_st_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        batch_size=1,
    )
    vec = (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)
    _print_query_embedding_demo(vec[0])

    k = min(candidate_pool, index.ntotal) if index.ntotal else 0
    if k <= 0:
        return []

    scores, ids = index.search(vec, k)

    results = []
    print(f"   Retrieved {k} candidates (index ntotal={index.ntotal}).")
    for vid, sc in zip(ids[0], scores[0]):
        if vid < 0:
            continue
        meta = lookup[int(vid)]
        cid = str(meta["chunk_id"])
        title = str(meta.get("title", ""))
        doc_id = str(meta.get("source_doc_id", ""))
        body_prev = text_preview_for_chunk(cid) or title[:200]
        results.append(
            {
                "vector_id": int(vid),
                "chunk_id": cid,
                "source_doc_id": doc_id,
                "score": float(sc),
                "title": title,
                "text_preview": body_prev,
            }
        )
    return results


def print_faiss_candidate_table(candidates: list[dict]) -> None:
    """Print a formatted table of FAISS candidates showing scores, IDs, and text previews."""
    print("\n[STEP 1b] FAISS candidate pool (retrieval order)")
    print("-" * 72)
    if not candidates:
        print("   (empty)")
        print("-" * 72)
        return
    for i, c in enumerate(candidates, 1):
        cid = c["chunk_id"]
        vid = c.get("vector_id")
        doc = c.get("source_doc_id") or "—"
        sc = c.get("score", 0.0)
        title = (c.get("title") or "")[:72]
        vi = vid if vid is not None else "—"
        print(
            f"   {i:2}. faiss_id={vi}  chunk_id={cid}  doc={doc}  ip_score={sc:.6f}"
        )
        if title:
            print(f"       title: {title}{'…' if len(c.get('title') or '') > 72 else ''}")
        prev = (c.get("text_preview") or "").strip()
        if prev:
            tail = "…" if len(prev) >= 280 else ""
            print(f"       text_preview: {prev[:220]}{tail}")
    print("-" * 72)


def _bouncer_verdict_for_range(
    cid: str,
    windows: dict[str, dict],
    q_start: str,
    q_end: str,
) -> bool:
    """Overlap iff T_start ≤ q_end AND T_end ≥ q_start (ISO string compare)."""
    w = windows.get(cid)
    if not w:
        return False
    ts = w.get("T_start", "")
    te = w.get("T_end", "")
    if str(ts).strip() == "" or str(te).strip() == "":
        return False
    ts_s, te_s = str(ts), str(te)
    return ts_s <= q_end and te_s >= q_start


def _fetch_sql_windows_all(chunk_ids: list[str]) -> dict[str, dict]:
    """Load T_start/T_end for all requested chunk_ids from PostgreSQL (any overlap)."""
    out: dict[str, dict] = {}
    if not psycopg2 or not chunk_ids:
        return out
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "temporal_rag"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
    )
    try:
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(chunk_ids))
        cur.execute(
            f"""
            SELECT chunk_id, T_start, T_end, published_date, temporal_source
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            chunk_ids,
        )
        for row in cur.fetchall():
            cid = str(row[0])
            out[cid] = {
                "T_start": row[1],
                "T_end": row[2],
                "published": row[3],
                "temporal_source": row[4] or "pub_date",
            }
    finally:
        conn.close()
    return out


def _load_temporal_windows_for_ids(chunk_ids: set[str]) -> dict[str, dict]:
    """chunk_id -> T_start, T_end, published_date from temporal_ie.jsonl."""
    out: dict[str, dict] = {}
    if not TEMPORAL_IE_PATH.is_file():
        return out
    with TEMPORAL_IE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            cid = str(o.get("chunk_id", ""))
            if cid not in chunk_ids:
                continue
            pub = o.get("published_date", "") or ""
            ts = o.get("T_start", "") or pub
            te = o.get("T_end", "") or pub
            out[cid] = {
                "T_start": ts,
                "T_end": te,
                "published": pub,
                "temporal_source": o.get("temporal_source", "pub_date"),
            }
    return out


def _shift_month(iso_date: str, delta: int) -> str:
    """Shift an ISO date string by delta months, clamping to the last day if needed."""
    try:
        d = date.fromisoformat(iso_date)
        month = d.month - 1 + delta
        year = d.year + month // 12
        month = month % 12 + 1
        day = min(d.day, calendar.monthrange(year, month)[1])
        return date(year, month, day).isoformat()
    except Exception:
        return iso_date


def _expand_pub_date_window(window: dict) -> dict:
    """Expand T_start/T_end by one month each side for chunks with only a pub_date fallback."""
    if window.get("temporal_source") != "pub_date":
        return window
    ts = window.get("T_start", "")
    te = window.get("T_end", "")
    if not ts or not te:
        return window
    return {**window, "T_start": _shift_month(ts, -1), "T_end": _shift_month(te, 1)}


def temporal_bouncer_sql(
    candidates: list[dict],
    q_start: str | None,
    q_end: str | None,
) -> tuple[list[dict], dict[str, dict], str]:
    """
    Return (filtered candidates in FAISS order, metadata for survivors, source_note).

    Loads T_start/T_end for every FAISS id (PostgreSQL if available, else temporal_ie.jsonl),
    prints one PASS/DROP line per candidate (with chunk temporal window), then keeps
    up to BOUNCER_TOP_K survivors in FAISS order (env, default 1).
    When q_start/q_end are missing, skips filtering and keeps top FAISS hits.

    Chunks whose temporal_source is "pub_date" (no dates resolved by HeidelTime) have
    their window expanded by one month on each side before the overlap check. This
    handles news articles that were published after the events they describe.
    """
    if not candidates:
        return [], {}, "empty"

    top_k = _bouncer_top_k()
    chunk_ids = [c["chunk_id"] for c in candidates]
    pg_meta: dict[str, dict] = {}
    passing: set[str] = set()
    source = "postgresql"
    windows: dict[str, dict] = {}
    gate = (
        q_start is not None
        and q_end is not None
        and str(q_start).strip() != ""
        and str(q_end).strip() != ""
    )

    if psycopg2:
        try:
            windows = _fetch_sql_windows_all(chunk_ids)
        except Exception as e:
            print(
                f"   PostgreSQL read error: {e} — "
                "using temporal_ie.jsonl for T_start/T_end."
            )
            source = "temporal_ie_jsonl_fallback"
            windows = _load_temporal_windows_for_ids(set(chunk_ids))
    else:
        print("   psycopg2 not installed — temporal windows from temporal_ie.jsonl.")
        source = "temporal_ie_jsonl_fallback"
        windows = _load_temporal_windows_for_ids(set(chunk_ids))

    windows = {cid: _expand_pub_date_window(w) for cid, w in windows.items()}

    if not gate:
        print("\n[STEP 2] Temporal bouncer — skipped (no query ISO range).")
        print(f"   Using top {top_k} FAISS candidate(s) without PASS/DROP.")
        filtered = candidates[:top_k]
        for c in filtered:
            cid = c["chunk_id"]
            w = windows.get(cid, {})
            row = {
                "T_start": w.get("T_start", ""),
                "T_end": w.get("T_end", ""),
                "published": w.get("published", ""),
            }
            if not str(row.get("T_start") or "").strip():
                row = _load_temporal_windows_for_ids({cid}).get(
                    cid, {"T_start": "N/A", "T_end": "N/A", "published": "N/A"}
                )
            pg_meta[cid] = row
        return filtered, pg_meta, source + "|no_gate"

    qs, qe = str(q_start), str(q_end)
    print(f"\n[STEP 2] Temporal bouncer — each FAISS candidate")
    print(
        f"   Rule: T_start ≤ {qe} AND T_end ≥ {qs}  —  window source: {source}"
    )
    print("-" * 72)
    for rank, c in enumerate(candidates, 1):
        cid = c["chunk_id"]
        ok = _bouncer_verdict_for_range(cid, windows, qs, qe)
        tag = "PASS" if ok else "DROP"
        sc = float(c.get("score", 0.0))
        doc = c.get("source_doc_id") or "—"
        w = windows.get(cid) or {}
        ts_disp = str(w.get("T_start", "") or "N/A")
        te_disp = str(w.get("T_end", "") or "N/A")
        print(
            f"   {rank:2}. chunk_id={cid} doc={doc} score={sc:.4f}  [{tag}]  "
            f"T_start={ts_disp}  T_end={te_disp}"
        )
        if ok:
            passing.add(cid)
            w = windows.get(cid, {})
            pg_meta[cid] = {
                "T_start": w.get("T_start", ""),
                "T_end": w.get("T_end", ""),
                "published": w.get("published", ""),
            }
    before = len(candidates)
    survivors = len([c for c in candidates if c["chunk_id"] in passing])
    dropped = before - survivors
    filtered = [c for c in candidates if c["chunk_id"] in passing][:top_k]

    print("-" * 72)
    print(
        f"   Summary: {before} FAISS candidates → {survivors} pass overlap → "
        f"{dropped} dropped → keep top-{top_k} by FAISS order → {len(filtered)} chunk(s)."
    )

    for c in filtered:
        cid = c["chunk_id"]
        if cid not in pg_meta:
            pg_meta[cid] = _load_temporal_windows_for_ids({cid}).get(
                cid, {"T_start": "N/A", "T_end": "N/A", "published": "N/A"}
            )

    return filtered, pg_meta, source


def _logic_triples_from_jsonl(
    chunk_ids: list[str],
    q_start: str | None,
    q_end: str | None,
) -> list[dict]:
    """Same triples Neo4j would expose, from temporal_ie kg_triples."""
    want = set(chunk_ids)
    apply_range = (
        q_start is not None
        and q_end is not None
        and str(q_start).strip()
        and str(q_end).strip()
    )
    ys, ye = str(q_start or ""), str(q_end or "")
    out: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    if not TEMPORAL_IE_PATH.is_file():
        return out
    with TEMPORAL_IE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ch = json.loads(line)
            cid = str(ch.get("chunk_id", ""))
            if cid not in want:
                continue
            for t in ch.get("kg_triples", []):
                sub = str(t.get("subject", "")).strip()
                pred = str(t.get("predicate", "")).strip()
                obj = str(t.get("object", "")).strip()
                if not (sub and pred and obj):
                    continue
                if t.get("object_type") == "Date" and apply_range:
                    if not (obj >= ys and obj <= ye):
                        continue
                key = (cid, sub, pred, obj)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {"chunk_id": cid, "subject": sub, "predicate": pred, "object": obj}
                )
    return out


def query_neo4j_graph_and_logic(
    chunk_ids: list[str],
    q_start: str | None,
    q_end: str | None,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Step 3 (Neo4j): neighborhood + logic-filler triples (X, PREDICATE, Y) for the SLM.

    When q_start and q_end are set, Date objects must fall in that inclusive ISO range.
    """
    print("\n[STEP 3] Neo4j — graph neighborhood + logic filler (triples for SLM)...")
    time.sleep(0.1)

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "temporalrag")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    apply_date_filter = (
        q_start is not None
        and q_end is not None
        and str(q_start).strip() != ""
        and str(q_end).strip() != ""
    )
    ys, ye = (str(q_start), str(q_end)) if apply_date_filter else ("", "")

    neo_results: dict[str, dict] = {
        cid: {"entities": [], "events": [], "dates": []} for cid in chunk_ids
    }
    triples: list[dict] = []
    seen_t: set[tuple[str, str, str, str]] = set()

    def add_triple(cid: str, subj: str, pred: str, obj: str) -> None:
        subj, pred, obj = subj.strip(), pred.strip(), obj.strip()
        if not (subj and pred and obj):
            return
        key = (cid, subj, pred, obj)
        if key in seen_t:
            return
        seen_t.add(key)
        triples.append(
            {"chunk_id": cid, "subject": subj, "predicate": pred, "object": obj}
        )

    if not GraphDatabase:
        print("   (neo4j driver not installed — temporal_ie.jsonl)")
        return simulate_neo4j_with_json(chunk_ids), _logic_triples_from_jsonl(
            chunk_ids, q_start, q_end
        )

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"   Neo4j unreachable ({e}). temporal_ie.jsonl fallback.")
        return simulate_neo4j_with_json(chunk_ids), _logic_triples_from_jsonl(
            chunk_ids, q_start, q_end
        )

    try:
        with driver.session(database=database) as session:
            q_ent = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS cid, collect(e.text) AS entities
                """,
                chunk_ids=chunk_ids,
            )
            for record in q_ent:
                k = str(record["cid"])
                if k in neo_results:
                    neo_results[k]["entities"] = list(record["entities"] or [])

            q_ev = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_EVENT]->(e:Event)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS cid, collect(e.text) AS events
                """,
                chunk_ids=chunk_ids,
            )
            for record in q_ev:
                k = str(record["cid"])
                if k in neo_results:
                    neo_results[k]["events"] = list(record["events"] or [])

            q_dates = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY|CONTAINS_EVENT]->(n)
                      -[:MENTIONED_ON|HAPPENED_ON]->(d:Date)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS cid, collect(DISTINCT d.date) AS dates
                """,
                chunk_ids=chunk_ids,
            )
            for record in q_dates:
                k = str(record["cid"])
                if k in neo_results:
                    neo_results[k]["dates"] = list(record["dates"] or [])

            # Logic filler: explicit edges for SLM (slide Step 3)
            q1 = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)-[:MENTIONED_ON]->(d:Date)
                WHERE c.chunk_id IN $chunk_ids
                  AND (NOT $apply_date_filter OR (d.date >= $ys AND d.date <= $ye))
                RETURN c.chunk_id AS cid, e.text AS subj, 'MENTIONED_ON' AS pred, d.date AS obj
                """,
                chunk_ids=chunk_ids,
                apply_date_filter=apply_date_filter,
                ys=ys,
                ye=ye,
            )
            for record in q1:
                add_triple(str(record["cid"]), str(record["subj"]), record["pred"], str(record["obj"]))

            q2 = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_EVENT]->(ev:Event)-[:HAPPENED_ON]->(d:Date)
                WHERE c.chunk_id IN $chunk_ids
                  AND (NOT $apply_date_filter OR (d.date >= $ys AND d.date <= $ye))
                RETURN c.chunk_id AS cid, ev.text AS subj, 'HAPPENED_ON' AS pred, d.date AS obj
                """,
                chunk_ids=chunk_ids,
                apply_date_filter=apply_date_filter,
                ys=ys,
                ye=ye,
            )
            for record in q2:
                add_triple(str(record["cid"]), str(record["subj"]), record["pred"], str(record["obj"]))

            q3 = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_EVENT]->(ev:Event)-[:INVOLVES]->(en:Entity)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS cid, ev.text AS subj, 'INVOLVES' AS pred, en.text AS obj
                """,
                chunk_ids=chunk_ids,
            )
            for record in q3:
                add_triple(str(record["cid"]), str(record["subj"]), record["pred"], str(record["obj"]))

        print(f"   Graph OK. Triples collected: {len(triples)}.")
        if apply_date_filter:
            print(f"   Date filter on triples: [{ys} .. {ye}]")
        else:
            print("   Date filter: off")
        return neo_results, triples
    except Exception as e:
        print(f"   Neo4j query error: {e}. JSONL fallback.")
        return simulate_neo4j_with_json(chunk_ids), _logic_triples_from_jsonl(
            chunk_ids, q_start, q_end
        )
    finally:
        if driver:
            driver.close()


def simulate_neo4j_with_json(chunk_ids: list[str]) -> dict[str, dict]:
    """Read entity, event, and date data from temporal_ie.jsonl when Neo4j is not available."""
    results: dict[str, dict] = {
        cid: {"entities": set(), "events": set(), "dates": set()} for cid in chunk_ids
    }
    if not TEMPORAL_IE_PATH.is_file():
        return {cid: {"entities": [], "events": [], "dates": []} for cid in chunk_ids}
    with TEMPORAL_IE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            cid = str(chunk["chunk_id"])
            if cid not in results:
                continue
            for edge in chunk.get("entity_date_edges", []):
                ent = (edge.get("entity") or "").strip()
                if ent:
                    results[cid]["entities"].add(ent)
                for d in edge.get("linked_dates", []):
                    results[cid]["dates"].add(d)
            for edge in chunk.get("event_date_edges", []):
                ev = (edge.get("event") or "").strip()
                if ev:
                    results[cid]["events"].add(ev)
                for d in edge.get("linked_dates", []):
                    results[cid]["dates"].add(d)

    return {
        cid: {
            "entities": sorted(results[cid]["entities"]),
            "events": sorted(results[cid]["events"]),
            "dates": sorted(results[cid]["dates"]),
        }
        for cid in chunk_ids
    }


def main() -> None:
    """Run the interactive terminal loop: read a question, run all pipeline steps, print context."""
    os.chdir(PROJECT_ROOT)

    while True:
        try:
            query = input("\nQuestion (or quit): ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break

            print("\n" + "-" * 72)
            print(f"Query: {query!r}")
            print("-" * 72)

            parsed = parse_query_temporal(query)
            print_query_parse_banner(parsed)
            faiss_candidates = query_faiss(query, FAISS_CANDIDATE_POOL)
            print_faiss_candidate_table(faiss_candidates)

            if not faiss_candidates:
                print("No FAISS candidates.")
                continue

            q_start, q_end = parsed["q_start"], parsed["q_end"]
            kept, pg_data, _src = temporal_bouncer_sql(faiss_candidates, q_start, q_end)

            if not kept:
                print(
                    "\nAfter temporal bouncer: no chunks left. "
                    "Try another year or broader question."
                )
                continue

            chunk_ids = [c["chunk_id"] for c in kept]
            neo_data, logic_triples = query_neo4j_graph_and_logic(
                chunk_ids, q_start, q_end
            )

            print("\n[STEP 3b] Logic filler: triples")
            print("-" * 72)
            if not logic_triples:
                print("   (none)")
            else:
                cap = 50
                for t in logic_triples[:cap]:
                    print(
                        f"   [chunk {t['chunk_id']}] "
                        f"{t['subject']} — {t['predicate']} → {t['object']}"
                    )
                if len(logic_triples) > cap:
                    print(f"   ... +{len(logic_triples) - cap} more triples")

            print("\n[STEP 4] Final context")
            print("=" * 72)
            for i, c in enumerate(kept, 1):
                cid = c["chunk_id"]
                pg = pg_data.get(cid, {})
                neo = neo_data.get(cid, {})

                print(f"\n#{i}  chunk_id={cid}  faiss_score={c['score']:.4f}")
                print(f"    title : {c.get('title', '')[:100]}")
                prev = c.get("text_preview", "")
                if prev:
                    print(f"    text  : {prev}")
                print(
                    f"    SQL window: T_start={pg.get('T_start', 'N/A')} "
                    f"T_end={pg.get('T_end', 'N/A')} "
                    f"pub={pg.get('published', 'N/A')}"
                )
                ents = neo.get("entities", [])
                evts = neo.get("events", [])
                dts = neo.get("dates", [])
                print(
                    f"    graph : {len(ents)} entities, {len(evts)} events, {len(dts)} dates"
                )
                if ents:
                    print(
                        f"            entities: {', '.join(ents[:5])}{'...' if len(ents) > 5 else ''}"
                    )
                if evts:
                    print(
                        f"            events  : {', '.join(evts[:5])}{'...' if len(evts) > 5 else ''}"
                    )
                if dts:
                    print(
                        f"            dates   : {', '.join(dts[:5])}{'...' if len(dts) > 5 else ''}"
                    )
            print("\n" + "=" * 72)

        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
