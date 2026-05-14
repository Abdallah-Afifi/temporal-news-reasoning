# Temporal RAG

## What this system is

Most search systems find text that uses **similar words**. They do not check whether the events in that text happened at the **same time** the question is asking about. This causes two common failures: returning an old article that uses the right words, or missing a recent article because the words differ slightly.

**Temporal RAG** solves this by adding a **calendar filter** to the standard Retrieval-Augmented Generation pipeline. Every document chunk is tagged with a date range (**`T_start`** to **`T_end`**). When you ask a question, the system finds what time period your question refers to (**`q_start`**, **`q_end`**) and drops any chunk whose stored date range does not overlap that window. Only chunks that are both **topically similar** and **temporally relevant** reach the language model.

This thesis implements the full pipeline: ingesting raw news articles, extracting and normalizing temporal expressions (using **GLiNER** + **HeidelTime**), building a tri-store index (**FAISS** vectors + **PostgreSQL** metadata + **Neo4j** knowledge graph), and querying through a live web interface.

---

## How it works at query time

1. Parses **when** the question is about (**`q_start`**, **`q_end`**) using **GLiNER** and a **temporal normalizer** backed by **HeidelTime**.
2. Retrieves similar chunks with **FAISS** (vectors).
3. **Filters** by calendar overlap: keep chunks whose range intersects **`[q_start, q_end]`** (when those bounds exist).
4. Optionally enriches answers from **PostgreSQL**, **Neo4j**, or **`output/temporal_ie.jsonl`**.

This README is written so someone new can **install, run, and debug** without reading the code first. For a click-by-click tour of the web UI, use **`docs/demo.md`**. For a **visual architecture** (offline ingestion + online retrieval), open **`docs/RAG_Implementation_Status_May.html`** in a browser.

---

## Section 1: What you need before you start

| Requirement | Notes |
|-------------|--------|
| **Python** | **3.10+** recommended (**`python3`**). |
| **This repo on disk** | You should see **`run_gui.py`**, **`temporal_rag/`**, **`encoder/`**, **`output/`** in the same folder. |
| **Pre-built `output/`** | At minimum you usually want a FAISS index + lookup JSONL, and **`temporal_ie.jsonl`** for times and graph-style triples (unless you rely entirely on Postgres). |
| **Java 11** | Needed for **HeidelTime** (temporal tagging). JDK 11 is most reliable. See **Section 8** for install steps. |
| **TreeTagger** | Needed for **HeidelTime** on macOS. The Linux bundle inside **`py_heideltime`** does not work on Mac. See **Section 8** for install steps. |
| **Internet** | First run may download embedding and GLiNER models from HuggingFace. |

You do **not** need Postgres or Neo4j to try the demo if **`output/temporal_ie.jsonl`** is present.

---

## Section 2: Install Python packages

Install the required packages with:

**Minimum to run the web app and query pipeline**

```bash
cd /path/to/this/repo   # folder that contains run_gui.py

python3 -m pip install --upgrade pip

python3 -m pip install streamlit pandas numpy torch sentence-transformers faiss-cpu \
  psycopg2-binary neo4j py_heideltime gliner tiktoken tqdm
```

**To also run the full offline build pipeline (chunk → encode → index)**

```bash
python3 -m pip install fastcoref
```

`fastcoref` is only needed for the optional coreference resolution step (Build step 3). You can skip it if you are not running that step.

**Notes**

- Use **`faiss-gpu`** instead of **`faiss-cpu`** only if you have a matching GPU setup.
- GLiNER version: if you see checkpoint errors, try **`pip install 'gliner>=0.1.6,<0.2'`** (some checkpoints target the 0.1 line).
- If **`psycopg2-binary`** or **`neo4j`** fail to install, you can skip them and rely on **JSONL** files for times and triples; the app falls back when drivers are missing or databases are down.
- **`tiktoken`** is required by the chunk splitter to count tokens. **`tqdm`** is required by the encoder to show progress bars.

---

## Section 3: Run the web app (recommended first run)

1. Open a terminal.
2. **`cd`** to the **repository root** (the directory that contains **`run_gui.py`**).
3. Confirm **`output/`** has your artifacts (for example **`faiss_hnsw_ip.index`**, **`faiss_vector_lookup.jsonl`**, **`temporal_ie.jsonl`**).
4. Start Streamlit:

```bash
cd /path/to/this/repo
python3 run_gui.py
```

5. Open the URL printed in the terminal (often **`http://localhost:8501`**).
6. If your question uses **yesterday**, **last week**, etc., set **DCT** (reference date) in the sidebar. See **`docs/demo.md`** for examples.
7. Stop the server with **Ctrl+C** in that terminal.

**Important:** Starting **`run_gui.py`** only starts the **web UI**. It does **not** start **Neo4j**, **Postgres**, or **Docker**. Those are separate programs you must launch yourself if you want live database connections.

---

## Section 4: Run the text-only CLI

Same **`cd`** as Section 3, then:

```bash
python3 -m temporal_rag.interactive_query
```

Type **`quit`**, **`exit`**, or **`q`** to exit. This module prints Steps 0 through 4 in the terminal and uses the same parsing and FAISS logic as the GUI.

---

## Section 5: What each UI step means (mental model)

| Step | Idea |
|------|------|
| **Step 0** | **Query parser:** GLiNER finds temporal spans; **TemporalNormalizer** runs **HeidelTime** so the question gets **`q_start` / `q_end`** when possible. |
| **Step 1** | **FAISS:** embedding similarity only (similar words, not guaranteed same dates). |
| **Step 2** | **Temporal gate:** **PASS** / **DROP** by overlap **`T_start…T_end`** vs **`q_start…q_end`** (from Postgres or **`temporal_ie.jsonl`**). Chunks that fell back to **`pub_date`** (no dates found in text) get their window expanded by one month on each side before the check, to account for articles published after the events they describe. |
| **Step 3** | **Graph neighborhood + logic triples:** prefers **Neo4j** if reachable; otherwise **`temporal_ie.jsonl`**. |
| **Step 4** | **Final context:** surviving chunk text and metadata. |

If Step 0 leaves **`q_start` / `q_end`** empty, Step 2 cannot apply a strict time filter. See **`docs/demo.md`** Section 10.

---

## Section 6: Optional PostgreSQL

Use Postgres when you want **`T_start` / `T_end`** (and chunk rows) served from a database instead of only JSONL.

In the **same terminal** where you launch the app:

```bash
export PGHOST="127.0.0.1"
export PGPORT="5432"
export PGDATABASE="temporal_rag"
export PGUSER="your_db_user"
export PGPASSWORD="your_db_password"
```

Ensure the server is running and the **`chunks`** table is loaded (see **`temporal_rag.sql_store`**). If the driver is missing or the DB is down, the pipeline falls back to **`temporal_ie.jsonl`** when possible.

---

## Section 7: Optional Neo4j (and why you see “connection refused”)

Neo4j provides live graph traversal and Date-linked triples in Step 3. **It is optional.** If Neo4j is not running on **`bolt://127.0.0.1:7687`**, you will see **connection refused** (`Errno 61` on macOS). That is normal when no server is listening; the code continues using **`output/temporal_ie.jsonl`** for graph-style triples.

**Neo4j is not started by Streamlit.** You must start Neo4j separately (Neo4j Desktop, Docker, or a service).

Default credentials expected by **`temporal_rag.interactive_query`** (override with env vars):

| Variable | Default |
|----------|---------|
| **`NEO4J_URI`** | **`bolt://127.0.0.1:7687`** |
| **`NEO4J_USER`** | **`neo4j`** |
| **`NEO4J_PASSWORD`** | **`temporalrag`** |
| **`NEO4J_DATABASE`** | **`neo4j`** |

**Docker example** (after Docker Desktop is installed and running):

```bash
docker rm -f neo4j-temporal 2>/dev/null

docker run -d \
  --name neo4j-temporal \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/temporalrag \
  neo4j:5
```

Browser UI: **`http://localhost:7474`**. Then load graph data with **`python3 -m temporal_rag.kg_store --help`** when you are ready.

**Check Bolt is up:**

```bash
nc -z 127.0.0.1 7687 && echo "Bolt port open" || echo "Nothing on 7687 yet"
```

---

## Section 8: Installing Java 11 and TreeTagger (macOS)

**HeidelTime** needs two things to work on macOS: **Java 11** and **TreeTagger**. The Linux TreeTagger bundle inside `py_heideltime` does not work on Mac, so you must install them separately.

---

**Step 1: Install Java 11**

The easiest way is with Homebrew. If you do not have Homebrew, install it first from **`brew.sh`**.

```bash
brew install openjdk@11
```

Then tell your shell where Java 11 is:

```bash
export JAVA_HOME=$(brew --prefix openjdk@11)
export PATH="$JAVA_HOME/bin:$PATH"
```

Add those two lines to your **`~/.zshrc`** (or **`~/.bash_profile`**) so they apply every time you open a terminal.

Verify it works:

```bash
java -version   # should say: openjdk version "11.x.x"
```

---

**Step 2: Install TreeTagger**

1. Go to **`https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/`**
2. Download the **macOS binary** (file named something like **`tree-tagger-MacOSX-*.tar.gz`**)
3. Download the **English parameter file** (**`english-utf8.par.gz`**)
4. Create the folder **`vendor/treetagger-install`** inside this repo
5. Extract the binary into **`vendor/treetagger-install/bin/`** so you have **`vendor/treetagger-install/bin/tree-tagger`**
6. Extract the parameter file into **`vendor/treetagger-install/lib/`** so you have **`vendor/treetagger-install/lib/english-utf8.par`**

The system will find TreeTagger automatically from that location. If you put it somewhere else, set the environment variable instead:

```bash
export HEIDELTIME_TREETAGGER_HOME=/path/to/your/treetagger
```

---

**Verify HeidelTime works:**

```bash
python3 -c "from temporal_rag.utils.heideltime_wrapper import HeidelTimeWrapper; \
h = HeidelTimeWrapper(); print(h.extract_resolved_dates(‘October 2016’, ‘2016-10-01’))"
```

You should see a list containing **`’2016-10’`** or **`’2016-10-01’`**. If you see an empty list or an error, check Java 11 and TreeTagger paths.

---

## Section 9: Environment variables (quick reference)

Set these **before** **`python3 run_gui.py`** in the same shell if you need non-defaults.

| Variable | Meaning |
|----------|---------|
| **`NEO4J_URI`**, **`NEO4J_USER`**, **`NEO4J_PASSWORD`**, **`NEO4J_DATABASE`** | Neo4j Bolt connection. |
| **`PGHOST`**, **`PGPORT`**, **`PGDATABASE`**, **`PGUSER`**, **`PGPASSWORD`** | Postgres connection for chunk times. |
| **`QUERY_DCT_DATE`** | Reference calendar date **`YYYY-MM-DD`** for the question (needed for **yesterday**, **last Tuesday**, etc.). |
| **`HEIDELTIME_TREETAGGER_HOME`** | TreeTagger installation root (optional if **`vendor/treetagger-install`** exists). |
| **`ONLINE_QUERY_PARSER`** | **`0`** disables GLiNER + normalizer on the question; **`1`** or unset enables them. |
| **`BOUNCER_TOP_K`** | How many overlap-passing chunks to keep after the temporal gate. |

---

## Section 10: Repository layout

| Path | Role |
|------|------|
| **`run_gui.py`** | Launches Streamlit from the repo root. |
| **`temporal_rag/`** | Python package: ingestion, query path, GUI. |
| **`temporal_rag/utils/heideltime_wrapper.py`** | HeidelTime + TreeTagger integration fixes. |
| **`encoder/`** | Model cache and embedding outputs from offline runs. |
| **`output/`** | FAISS index, lookup JSONL, **`temporal_ie.jsonl`**, etc. |
| **`docs/demo.md`** | First-time UI walkthrough and sample questions. |
| **`docs/RAG_Implementation_Status_May.html`** | Architecture diagram (offline tri-store + online retrieval path). |

Useful module entry points:

```bash
python3 -m temporal_rag.kg_store --help
python3 -m temporal_rag.build_faiss_database --help
python3 -m temporal_rag.query_full_stack --help
```

---

## Section 11: Build pipeline

Run the steps below **in order** from the repository root. Each step reads the output of the previous one. Exact CLI flags are documented at the top of each `temporal_rag/*.py` file.

**Step 1: Convert raw articles to JSONL**

This step depends on which dataset you are using:

- **CC-News** — already in JSONL format. No conversion needed. Go straight to Step 2.

- **BBC (parquet files)** — must be converted first using `bbc_to_json.py`:

```bash
python3 -m temporal_rag.bbc_to_json \
    --input  train-00000-of-00001.parquet \
    --output bbc_standardized.jsonl
```

Either way, the output must be a JSONL file where each line has at least `plain_text`, `title`, and `published_date`.

**Step 2: Split articles into chunks**

```bash
python3 -m temporal_rag.chunk_splitter \
    --input  bbc_standardized.jsonl \
    --output chunks.jsonl
```

Output: `chunks.jsonl` - one chunk per line, each with `chunk_id`, `text`, `token_count`, `published_date`.

Each chunk is **512 tokens** with a **50-token overlap** between consecutive chunks. The overlap means the last 50 tokens of one chunk are repeated at the start of the next. This ensures that sentences near a chunk boundary are not cut off mid-context — if an answer spans the boundary between two chunks, both chunks will contain enough of the surrounding text to be useful.

**Step 3 (optional): Resolve pronouns**

```bash
python3 -m temporal_rag.coref_resolver \
    --input  chunks.jsonl \
    --output chunks_deref.jsonl
```

Replaces pronouns like "he" with real names so NER and embeddings work better. This also serves as an extra verification step — after dereferencing, each chunk can be read on its own and still make sense, which confirms the 50-token overlap is carrying enough context across chunk boundaries.

**Step 4: Extract named entities with GLiNER**

```bash
python3 -m temporal_rag.gliner_extractor \
    --input  chunks_deref.jsonl \
    --output gliner_output.jsonl
```

Adds an `entities` list to each chunk (DATE, TIME, EVENT, ENTITY spans).

**Step 5: Resolve dates with HeidelTime**

```bash
python3 -m temporal_rag.normalizer \
    --input  gliner_output.jsonl \
    --output normalizer_output.jsonl
```

Turns date phrases like "last Monday" into real calendar dates. Adds `T_start`, `T_end`, and epoch timestamps.

**Step 6: Build knowledge graph edges**

```bash
python3 -m temporal_rag.temporal_ie \
    --input  normalizer_output.jsonl \
    --output temporal_ie.jsonl
```

Links events and entities to their dates. Output goes to `output/temporal_ie.jsonl`.

**Step 7: Build embeddings**

```bash
python3 -m temporal_rag.embed \
    --input  temporal_ie.jsonl \
    --output embed_output.jsonl
```

Adds the `embed` field (raw chunk text only) that the encoder will vectorize. Title and date prefixes are not included — the temporal filter handles time separately.

**Step 8: Encode vectors**

```bash
python3 -m temporal_rag.encoder
```

Writes `.npy` embedding arrays and chunk mapping JSONL files.

**Step 9: Build FAISS index**

```bash
python3 -m temporal_rag.build_faiss_database
```

Writes `output/faiss_hnsw_ip.index` and `output/faiss_vector_lookup.jsonl`.

**Step 10 (optional): Load into PostgreSQL**

```bash
python3 -m temporal_rag.sql_store \
    --input temporal_ie.jsonl \
    --host localhost --dbname temporal_rag --user your_user --password your_password
```

**Step 11 (optional): Load into Neo4j**

```bash
python3 -m temporal_rag.kg_store \
    --input temporal_ie.jsonl
```

---

## Section 12: Architecture HTML vs this codebase

**`docs/RAG_Implementation_Status_May.html`** is a **diagram** of the *Final Temporal RAG* design: **tri-store** ingestion (**FAISS** vectors, **PostgreSQL** metadata and windows, **Neo4j** / temporal KG) and an **online** path with query parsing, retrieval, and enrichment.

**What the shipped Python GUI and `interactive_query` implement today** matches the **Step 0-4** flow in Section 5 above (FAISS retrieval, temporal overlap gate, Neo4j-with-JSONL-fallback, chunk context). Some boxes in the diagram describe **additional or experimental retrieval ideas** (for example extra reranking layers) that are **not necessarily present** as separate components in the current scripts. Treat the HTML as the **big picture**, and **`docs/demo.md`** plus this README as the **ground truth for what to run locally**.

---

## Section 13: Troubleshooting

| Symptom | What to check |
|---------|----------------|
| **Missing FAISS / lookup files** | Build **`output/`** using the pipeline (Section 11). Always **`cd`** to repo root. |
| **Neo4j unreachable / connection refused** | Start Neo4j (Section 7) or ignore and use **`temporal_ie.jsonl`**. |
| **Postgres errors** | Section 6 credentials; server running; or rely on JSONL windows. |
| **Empty `q_start` / `q_end`** | Set **DCT** for relative phrases; ensure HeidelTime + TreeTagger work (Section 8). See **`docs/demo.md`** Section 10. |
| **HeidelTime / TreeTagger errors on Mac** | **`vendor/treetagger-install`** or **`HEIDELTIME_TREETAGGER_HOME`**; JDK **11** available. |
| **Streamlit styling** | Run from repo root so **`.streamlit/config.toml`** applies. |

---

## Section 14: Read more

- **`docs/demo.md`**: hands-on UI guide and sample questions.
- **`docs/RAG_Implementation_Status_May.html`**: open in a browser for the architecture diagram.
- Module docstrings at the top of each **`temporal_rag/*.py`** file list inputs, outputs, and example commands.
