#!/usr/bin/env python3
"""Manual check that FAISS, Postgres, and Neo4j all answer for the same chunk ids.

1. Load the saved vector index and print top matches for your question.
2. Read ``T_start`` / ``T_end`` from Postgres for those chunk ids (needs ``psycopg2``).
3. Read entity/event/date links from Neo4j, or rebuild a rough view from ``temporal_ie.jsonl``.

Edit the connection strings in this file if your database uses different credentials or ports.
Not used by the Streamlit app; only for local debugging after data is loaded.
"""

import sys
import json
import faiss
import numpy as np
try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
except ImportError:
    GraphDatabase = None

from sentence_transformers import SentenceTransformer

from .paths import PROJECT_ROOT


def query_faiss(query: str, top_k: int = 3):
    """Embed ``query`` and run FAISS inner-product search; return chunk ids + scores."""
    print(f"\n[1] FAISS: Encoding question and searching vectors for top {top_k} chunks...")
    
    # Load index and lookup
    index_path = PROJECT_ROOT / "output/faiss_hnsw_ip.index"
    lookup_path = PROJECT_ROOT / "output/faiss_vector_lookup.jsonl"
    index = faiss.read_index(str(index_path))
    lookup = [json.loads(l) for l in lookup_path.open(encoding="utf-8") if l.strip()]
    
    # Encode query
    cache = str(PROJECT_ROOT / "encoder/model_cache")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=cache,
    )
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    vec = (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)
    
    scores, ids = index.search(vec, top_k)
    
    results = []
    for vid, sc in zip(ids[0], scores[0]):
        if vid < 0: continue
        meta = lookup[int(vid)]
        results.append({
            "chunk_id": str(meta["chunk_id"]),
            "score": float(sc),
            "text": meta.get("title", "")
        })
    return results


def query_postgres(chunk_ids: list[str]):
    """Load ``T_start`` / ``T_end`` / ``published_date`` rows from Postgres for chunk ids."""
    print(f"[2] PostgreSQL: Fetching time windows for {len(chunk_ids)} chunks...")
    if not psycopg2:
        print("  -> psycopg2 not installed.")
        return {}
    
    try:
        conn = psycopg2.connect(dbname='temporal_rag', user='postgres', host='localhost', port=5432)
        cur = conn.cursor()
        
        # Format the IN clause
        placeholders = ','.join(['%s'] * len(chunk_ids))
        query = f"""
            SELECT chunk_id, T_start, T_end, published_date
            FROM chunks
            WHERE chunk_id IN ({placeholders})
        """
        cur.execute(query, chunk_ids)
        rows = cur.fetchall()
        
        pg_results = {}
        for row in rows:
            pg_results[str(row[0])] = {
                "T_start": row[1],
                "T_end": row[2],
                "published": row[3]
            }
        conn.close()
        return pg_results
    except Exception as e:
        print(f"  -> PostgreSQL error: {e}")
        return {}


def query_neo4j(chunk_ids: list[str]):
    """Query Neo4j for graph neighborhoods; if unavailable, mimic via ``temporal_ie.jsonl``."""
    print(f"[3] Neo4j: Fetching entities, events, and dates linked to {len(chunk_ids)} chunks...")
    if not GraphDatabase:
        print("  -> neo4j driver not installed.")
        return {}
    
    try:
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'temporalrag'))
        # Quick check if it's up
        driver.verify_connectivity()
    except Exception as e:
        print(f"  -> Neo4j not running or unreachable: {e}")
        print("  -> Falling back to reading temporal_ie.jsonl to simulate Neo4j response...")
        return simulate_neo4j_with_json(chunk_ids)

    try:
        neo_results = {cid: {"entities": [], "events": [], "dates": []} for cid in chunk_ids}
        with driver.session(database="neo4j") as session:
            # Match entities
            res_entities = session.run("""
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id as cid, collect(e.text) as entities
            """, chunk_ids=chunk_ids)
            for record in res_entities:
                neo_results[record["cid"]]["entities"] = record["entities"]

            # Match events
            res_events = session.run("""
                MATCH (c:Chunk)-[:CONTAINS_EVENT]->(e:Event)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id as cid, collect(e.text) as events
            """, chunk_ids=chunk_ids)
            for record in res_events:
                neo_results[record["cid"]]["events"] = record["events"]
                
            # Match dates directly attached to events/entities in this chunk
            res_dates = session.run("""
                MATCH (c:Chunk)-[:CONTAINS_ENTITY|CONTAINS_EVENT]->(n)-[:MENTIONED_ON|HAPPENED_ON]->(d:Date)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id as cid, collect(DISTINCT d.date) as dates
            """, chunk_ids=chunk_ids)
            for record in res_dates:
                neo_results[record["cid"]]["dates"] = record["dates"]

        driver.close()
        return neo_results
    except Exception as e:
        print(f"  -> Neo4j query error: {e}")
        return simulate_neo4j_with_json(chunk_ids)

def simulate_neo4j_with_json(chunk_ids):
    """Fallback when Neo4j is offline: scan ``temporal_ie.jsonl`` for the same edge lists."""
    results = {cid: {"entities": set(), "events": set(), "dates": set()} for cid in chunk_ids}
    temporal_ie = PROJECT_ROOT / "output/temporal_ie.jsonl"
    with temporal_ie.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunk = json.loads(line)
            cid = str(chunk["chunk_id"])
            if cid in chunk_ids:
                for edge in chunk.get("entity_date_edges", []):
                    results[cid]["entities"].add(edge.get("entity", ""))
                    for d in edge.get("linked_dates", []):
                        results[cid]["dates"].add(d)
                for edge in chunk.get("event_date_edges", []):
                    results[cid]["events"].add(edge.get("event", ""))
                    for d in edge.get("linked_dates", []):
                        results[cid]["dates"].add(d)
    
    # Convert sets to lists
    for cid in results:
        results[cid]["entities"] = list(results[cid]["entities"])
        results[cid]["events"] = list(results[cid]["events"])
        results[cid]["dates"] = list(results[cid]["dates"])
    return results


def main():
    """CLI entry: optional query string argument; prints fused vector + SQL + graph summary."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "migrants crossing English Channel"
        
    print("="*80)
    print(f"  END-TO-END TEMPORAL RAG QUERY VERIFICATION")
    print(f"  Question: {query!r}")
    print("="*80)

    # 1. FAISS
    faiss_chunks = query_faiss(query, top_k=3)
    chunk_ids = [c["chunk_id"] for c in faiss_chunks]
    
    if not chunk_ids:
        print("No chunks found.")
        return

    # 2. PostgreSQL
    pg_data = query_postgres(chunk_ids)

    # 3. Neo4j
    neo_data = query_neo4j(chunk_ids)

    # Combine and print
    print("\n" + "="*80)
    print("  FINAL ASSEMBLED RESULTS")
    print("="*80)
    
    for i, c in enumerate(faiss_chunks, 1):
        cid = c["chunk_id"]
        pg = pg_data.get(cid, {})
        neo = neo_data.get(cid, {})
        
        print(f"\n--- Result #{i} (Chunk ID: {cid}) ---")
        print(f"  FAISS Score  : {c['score']:.4f}")
        print(f"  Title        : {c['text']}")

        print(f"  Postgres     : T_start={pg.get('T_start', 'N/A')}  T_end={pg.get('T_end', 'N/A')}  pub={pg.get('published', 'N/A')}")

        ents = neo.get("entities", [])
        evts = neo.get("events", [])
        dts = neo.get("dates", [])

        print(f"  Neo4j links  : {len(ents)} entities, {len(evts)} events, {len(dts)} dates")
        if ents: print(f"    entities: {', '.join(ents[:5])}{'...' if len(ents) > 5 else ''}")
        if evts: print(f"    events  : {', '.join(evts[:5])}{'...' if len(evts) > 5 else ''}")
        if dts:  print(f"    dates   : {', '.join(dts[:5])}{'...' if len(dts) > 5 else ''}")

if __name__ == "__main__":
    main()
