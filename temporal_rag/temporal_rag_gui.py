"""
Web UI for Temporal RAG (Streamlit).

Does the same stages as ``interactive_query.py``: parse question dates, search vectors,
filter by calendar overlap, load graph triples, show chunk text. The sidebar holds DCT,
parser switches, and pool size so you can experiment without editing Python files.

How to run::

    python3 run_gui.py

Always ``cd`` to the repo root first so ``encoder/``, ``output/``, and ``.streamlit/`` exist.

Needs: ``pip install streamlit pandas`` plus packages listed in README.
"""

from __future__ import annotations

import contextlib
import html
import io
import json
import os
import re
import sys
from pathlib import Path

# Repo root (parent of this ``temporal_rag`` package folder).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Streamlit must resolve paths like ``output/`` from here.
os.chdir(PROJECT_ROOT)


def _maybe_set_java_home_for_heideltime() -> None:
    """If JAVA_HOME is empty on macOS, point it at JDK 11 so HeidelTime starts cleanly."""
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

import streamlit as st

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import temporal_rag.interactive_query as iq

_HIGHLIGHT_STOPWORDS = frozenset(
    """
    a an the and or but if as at by for from in into is it its of on onto
    to too so than that this these those was were are be been being with
    not no nor do does did done has had have having will would could should
    may might must can about above after again against all almost also
    although always among another any anyone anything around as at away
    back because been before being below between both each few more most
    other some such than then there these they those through to under until
    very what when where which while who whom whose why how
    """.split()
)


def _load_full_chunk_text(chunk_id: str) -> str:
    """Full `text` for a chunk from temporal_ie.jsonl, then deref.jsonl."""
    cid = str(chunk_id)
    for path in (iq.TEMPORAL_IE_PATH, iq.DEREF_PATH):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(o.get("chunk_id", "")) != cid:
                    continue
                t = o.get("text", "")
                if t is not None and str(t).strip():
                    return str(t)
    return ""


def _best_sentence_for_query(full_text: str, query: str) -> str:
    """One sentence that shares the most wording with the question (simple overlap)."""
    if not full_text.strip() or not query.strip():
        return ""
    words = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", query)
        if len(w) > 2
    }
    if not words:
        return ""
    normalized = full_text.replace("\r\n", "\n")
    parts = re.split(r"(?<=[.!?])\s+", normalized.replace("\n", " "))
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    best, best_score = "", -1
    for p in parts:
        low = p.lower()
        score = sum(1 for w in words if w in low)
        if score > best_score or (score == best_score and len(p) > len(best)):
            best_score, best = score, p
    return best if best_score > 0 else parts[0]


def _merge_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) character spans into a sorted non-overlapping list."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda se: (se[0], se[1]))
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _phrase_echoes_query(phrase: str, query: str) -> bool:
    """True if the phrase appears as its own token run in the question (skip redundant echo)."""
    if not phrase.strip() or not query.strip():
        return False
    return re.search(rf"(?<![\w]){re.escape(phrase.strip())}(?![\w])", query, re.I) is not None


def _collect_graph_phrases(neo: dict | None) -> list[str]:
    """Flatten entities, events, and dates from a Neo4j result dict into one list of strings."""
    if not neo:
        return []
    out: list[str] = []
    for key in ("entities", "events", "dates"):
        for x in neo.get(key) or []:
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _greedy_phrase_spans(text: str, phrases: list[str], query: str) -> list[tuple[int, int]]:
    """Longest phrases first; non-overlapping matches (chunk graph labels + similar)."""
    seen_lower: set[str] = set()
    ordered: list[str] = []
    for p in sorted(phrases, key=len, reverse=True):
        k = p.lower()
        if k in seen_lower:
            continue
        seen_lower.add(k)
        ordered.append(p)

    occupied: list[tuple[int, int]] = []

    def conflicts(s: int, e: int) -> bool:
        return any(s < pe and ps < e for ps, pe in occupied)

    def occupy(s: int, e: int) -> None:
        occupied.append((s, e))

    for p in ordered:
        if len(p) < 2:
            continue
        if _phrase_echoes_query(p, query):
            continue
        if " " in p or not re.fullmatch(r"[\w'-]+", p, re.I):
            pat = re.compile(re.escape(p), re.I)
        else:
            pat = re.compile(rf"(?<![\w]){re.escape(p)}(?![\w])", re.I)
        for m in pat.finditer(text):
            if not conflicts(m.start(), m.end()):
                occupy(m.start(), m.end())

    # numbers / counts not literally present in the question
    for m in re.finditer(r"\b\d[\d,]*\b", text):
        raw = m.group()
        if raw in query or raw.replace(",", "") in query.replace(",", ""):
            continue
        if conflicts(m.start(), m.end()):
            continue
        occupy(m.start(), m.end())

    return _merge_intervals(occupied)


def _fallback_novel_spans(text: str, query: str, max_runs: int = 3) -> list[tuple[int, int]]:
    """When the graph is empty: highlight a few spans of words that are not echoed from the question."""
    if not text.strip() or not query.strip():
        return []
    q_terms = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2}
    words = list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text))
    runs: list[tuple[int, int]] = []
    cur_s: int | None = None
    cur_e: int | None = None
    for m in words:
        w = m.group()
        low = w.lower()
        novel = low not in q_terms and low not in _HIGHLIGHT_STOPWORDS and len(w) >= 3
        if novel:
            if cur_s is None:
                cur_s, cur_e = m.start(), m.end()
            else:
                gap = text[cur_e : m.start()]
                if gap.strip() == "" and "\n" not in gap:
                    cur_e = m.end()
                else:
                    runs.append((cur_s, cur_e))
                    cur_s, cur_e = m.start(), m.end()
        elif cur_s is not None:
            runs.append((cur_s, cur_e))
            cur_s = cur_e = None
    if cur_s is not None:
        runs.append((cur_s, cur_e))

    scored = sorted(((e - s, s, e) for s, e in runs), reverse=True)
    picked: list[tuple[int, int]] = []
    for _ln, s, e in scored[:max_runs]:
        if not any(s < pe and ps < e for ps, pe in picked):
            picked.append((s, e))
    picked.sort(key=lambda se: se[0])
    return _merge_intervals(picked)


def _highlight_answer_hints(text: str, query: str, neo: dict | None) -> str:
    """
    Highlight likely answer bits: graph entities/events/dates from the chunk,
    plus numbers absent from the question; else short runs of novel words.
    """
    if not text:
        return ""
    q = (query or "").strip()
    phrases = _collect_graph_phrases(neo)
    spans = _greedy_phrase_spans(text, phrases, q)
    if not spans and q:
        spans = _fallback_novel_spans(text, q)
    if not spans:
        return html.escape(text)
    parts: list[str] = []
    last = 0
    for s, e in spans:
        parts.append(html.escape(text[last:s]))
        parts.append(f'<mark class="chunk-hl">{html.escape(text[s:e])}</mark>')
        last = e
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def _capture_stdout(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` and return (result, text_printed_to_stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs), buf.getvalue()


def _css():
    """Inject custom CSS into the Streamlit page (font, layout, highlight colours)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        html, body, [class*="css"] {
            font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
        }
        .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 920px; }
        h1 { font-weight: 700; letter-spacing: -0.03em; }
        div.stExpander details summary { font-weight: 600; }
        [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdown"]) p {
            line-height: 1.55;
        }
        .chunk-wrap {
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            line-height: 1.6;
            font-size: 0.92rem;
            padding: 0.75rem 0;
            max-width: 100%;
        }
        mark.chunk-hl {
            background-color: rgba(255, 214, 102, 0.55);
            color: inherit;
            padding: 0 0.12em;
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Build and render the Streamlit web UI: sidebar, query input, and Steps 0-4 output panels."""
    os.environ["ONLINE_QUERY_PARSER"] = "1"
    os.environ["BOUNCER_TOP_K"] = "1"

    st.set_page_config(
        page_title="Temporal RAG",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()

    st.title("Temporal RAG")
    st.markdown("**Online retrieval** · GLiNER / HeidelTime · FAISS · temporal gate · Neo4j / JSONL")

    with st.sidebar:
        st.subheader("Query options")
        dct = st.text_input(
            "DCT",
            value=os.environ.get("QUERY_DCT_DATE", "") or "",
            placeholder="YYYY-MM-DD · empty = today",
        )
        if dct.strip():
            os.environ["QUERY_DCT_DATE"] = dct.strip()

        st.divider()
        st.caption("Backends (env vars)")
        st.caption("Neo4j: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE")
        st.caption("Postgres: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD")
        st.caption("CLI backup: `python3 -m temporal_rag.interactive_query`")

    with st.form("pipeline_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            height=100,
            placeholder="write your question",
        )
        submitted = st.form_submit_button("Run pipeline", type="primary", use_container_width=True)

    if not submitted:
        st.info("Enter a question and click **Run pipeline**.")
        return

    q = (question or "").strip()
    if not q:
        st.warning("Please enter a non-empty question.")
        return

    st.divider()
    st.markdown(f"**Query:** {q}")

    progress = st.progress(0, text="Parsing…")
    try:
        parsed = iq.parse_query_temporal(q)
    except Exception as e:
        progress.empty()
        st.exception(e)
        return

    progress.progress(15, text="FAISS…")

    m = parsed.get("meta") or {}
    with st.expander("Step 0 — Query parse", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("DCT (doc creation time)", parsed.get("dct", "—"))
        c2.metric("Mode", str(m.get("mode", "—")))
        c3.metric(
            "Temporal source",
            str(m.get("temporal_source") or "—"),
        )
        qr = (
            f"`{parsed['q_start']}` → `{parsed['q_end']}`"
            if parsed.get("q_start") and parsed.get("q_end")
            else "*(no ISO gate — bouncer uses top FAISS without overlap filter)*"
        )
        st.markdown(f"**q_start … q_end:** {qr}")
        if m.get("parser_error"):
            if parsed.get("q_start") and parsed.get("q_end"):
                st.warning(
                    f"Query parser raised a network/load error (often Hugging Face or HTTP). "
                    f"**Recovered:** using the year in your question for `q_start`/`q_end`. "
                    f"Details: {m['parser_error']}"
                )
            else:
                st.error(str(m["parser_error"]))
        ents = m.get("entities") or []
        if ents:
            rows = [{"label": e.get("label"), "text": e.get("text")} for e in ents]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        if m.get("all_resolved"):
            st.caption("all_resolved_dates")
            st.json(m["all_resolved"])
        if m.get("span_to_date"):
            st.caption("span_to_date")
            st.json(m["span_to_date"])

    try:
        faiss_candidates, faiss_log = _capture_stdout(iq.query_faiss, q, iq.FAISS_CANDIDATE_POOL)
    except Exception as e:
        progress.empty()
        st.exception(e)
        return

    progress.progress(45, text="Temporal gate…")

    with st.expander("Step 1 — FAISS + embedding", expanded=False):
        st.code(faiss_log.strip() or "(no stdout)", language=None)

    if not faiss_candidates:
        progress.empty()
        st.error("No FAISS candidates (check index and lookup under `output/`).")
        return

    tbl = []
    for i, c in enumerate(faiss_candidates, 1):
        tbl.append(
            {
                "#": i,
                "chunk_id": c["chunk_id"],
                "source_doc_id": c.get("source_doc_id"),
                "ip_score": round(float(c.get("score", 0.0)), 6),
                "title": (c.get("title") or "")[:120],
            }
        )
    with st.expander(f"Step 1b — Top {len(tbl)} candidates", expanded=True):
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    q_start, q_end = parsed["q_start"], parsed["q_end"]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kept, pg_data, bouncer_src = iq.temporal_bouncer_sql(
                faiss_candidates, q_start, q_end
            )
        gate_log = buf.getvalue()
    except Exception as e:
        progress.empty()
        st.exception(e)
        return

    progress.progress(70, text="Graph / triples…")

    with st.expander("Step 2 — Temporal bouncer", expanded=True):
        st.caption(f"Window source: **{bouncer_src}**")
        st.code(gate_log.strip() or "(no stdout)", language=None)

    if not kept:
        progress.empty()
        st.warning("No chunks passed the temporal gate. Try another year or broader phrasing.")
        return

    chunk_ids = [c["chunk_id"] for c in kept]
    try:
        buf_n = io.StringIO()
        with contextlib.redirect_stdout(buf_n):
            neo_data, logic_triples = iq.query_neo4j_graph_and_logic(
                chunk_ids, q_start, q_end
            )
        neo_log = buf_n.getvalue()
    except Exception as e:
        progress.empty()
        st.exception(e)
        return

    progress.progress(100, text="Done")
    progress.empty()

    with st.expander("Step 3 — Neo4j / graph", expanded=False):
        st.code(neo_log.strip() or "(no stdout)", language=None)

    with st.expander("Step 3b — Logic triples", expanded=True):
        if not logic_triples:
            st.caption("No triples (or filter removed all).")
        else:
            trip_rows = [
                {
                    "chunk": t["chunk_id"],
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"],
                }
                for t in logic_triples[:200]
            ]
            st.dataframe(trip_rows, use_container_width=True, hide_index=True)
            if len(logic_triples) > 200:
                st.caption(f"+ {len(logic_triples) - 200} more triples")

    st.subheader("Step 4 — Final context")
    for i, c in enumerate(kept, 1):
        cid = c["chunk_id"]
        pg = pg_data.get(cid, {})
        neo = neo_data.get(cid, {})
        full_chunk = _load_full_chunk_text(cid)
        snippet = _best_sentence_for_query(full_chunk, q) if full_chunk else ""
        with st.container():
            st.markdown(f"**#{i}** · chunk `{cid}` · FAISS score `{c['score']:.4f}`")
            st.markdown(f"*{c.get('title', '')[:200]}*")
            if snippet:
                st.markdown("**Sentence closest to your question**")
                st.markdown(
                    f'<div class="chunk-wrap">{_highlight_answer_hints(snippet, q, neo)}</div>',
                    unsafe_allow_html=True,
                )
            if full_chunk:
                with st.expander("Full chunk text (from corpus)", expanded=True):
                    st.markdown(
                        f'<div class="chunk-wrap">{_highlight_answer_hints(full_chunk, q, neo)}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                prev = (c.get("text_preview") or "").strip()
                if prev:
                    st.caption("Short preview only (full text not found on disk for this chunk_id).")
                    st.text(prev)
            st.markdown(
                f"`T_start` **{pg.get('T_start', 'N/A')}** · `T_end` **{pg.get('T_end', 'N/A')}** · "
                f"`published` **{pg.get('published', 'N/A')}**"
            )
            ents = neo.get("entities", [])
            evts = neo.get("events", [])
            dts = neo.get("dates", [])
            st.caption(f"Graph: {len(ents)} entities · {len(evts)} events · {len(dts)} dates")
            if ents:
                st.text("Entities: " + ", ".join(str(x) for x in ents[:12]) + ("…" if len(ents) > 12 else ""))
            if evts:
                st.text("Events: " + ", ".join(str(x) for x in evts[:12]) + ("…" if len(evts) > 12 else ""))
            if dts:
                st.text("Dates: " + ", ".join(str(x) for x in dts[:12]) + ("…" if len(dts) > 12 else ""))
            st.divider()


# Streamlit executes this module top-to-bottom; entry is function-style grouping above.
main()
