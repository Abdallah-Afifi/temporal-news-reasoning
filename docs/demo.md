# Temporal RAG demo guide

This file is a hands-on walkthrough for someone opening the web app for the **first time**. It covers how to start the app, what each numbered step on screen means, and provides three copy-paste questions to test the system.

Read **`README.md`** first for installation, folder layout, and how to build the data. Use **this file** while the app is open in your browser.

---

## Section 1: Start here

Before you begin:

- You **do not** need to change any code for this walkthrough.
- Complete the install steps in **`README.md`** first.
- Make sure the **`output/`** folder has the FAISS index and lookup files.

**What is DCT?** The **DCT** box in the sidebar is the **reference calendar date** for your question, in **`YYYY-MM-DD`** format. The time parser uses it to turn relative phrases like **yesterday**, **last Monday**, or **that weekend** into real dates. If your question uses those phrases and DCT is wrong or blank, Step 0 will produce empty or incorrect **`q_start` / `q_end`** values.

This guide walks through: starting the app, reading Steps 0 to 4 on screen, and three sample questions: a strong word match, a wrong-year drop, and a **yesterday** question with two different DCT values.

---

## Section 2: What happens when you click Run

Each run ties five ideas together:

1. **Find time words in the question** (models plus rules; Java piece may run in the background).
2. **Find similar chunks** using saved vectors (similar words, not guaranteed same dates).
3. **Apply the time rule:** drop chunks whose saved date range does not overlap the question’s date range.
4. **Pull extra facts** about people, events, and dates (live database or files).
5. **Show text** from the chunks that survived step 3.

You still need real data: FAISS index, **`temporal_ie.jsonl`**, optional Postgres, built ahead of time.

---

## Section 3: Before you start (checklist)

1. **Folder.** Your terminal should be in the folder that contains **`run_gui.py`**, **`temporal_rag/`**, **`output/`**, **`encoder/`**. If not, **`cd`** until it does.
2. **Files.** Check that **`output/`** has your vector index and lookup files, and usually **`temporal_ie.jsonl`** unless Postgres holds all times.
3. **Python.** Use the same environment where **`import faiss`** and **`import sentence_transformers`** already worked once.
4. **Databases (optional).** Start Postgres and Neo4j **before** the app if you plan to use them. Set variables as in **`README.md`** Section 9 in **that same terminal**.

---

## Section 4: Database passwords (optional)

Skip if you demo with files only.

In the terminal where you will run the app, run once (use your real secrets):

```bash
export NEO4J_URI="bolt://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD='YOUR_NEO4J_PASSWORD'
export NEO4J_DATABASE="neo4j"

export PGHOST="127.0.0.1"
export PGPORT="5432"
export PGDATABASE="temporal_rag"
export PGUSER="YOUR_DB_USER"
export PGPASSWORD="YOUR_DB_PASSWORD"
```

Do not commit real passwords to git.

---

## Section 5: Start and stop the app

1. Finish Section 3.
2. Section 4 only if you need databases.
3. Run:

```bash
cd /path/to/your/repo/root
python3 run_gui.py
```

4. Open the **`http://localhost`** URL it prints (often port **8501**).
5. To quit: click the terminal and press **Ctrl+C**. Wait until you see the shell prompt again.

---

## Section 6: Sidebar box **DCT**

**What it is.** One calendar day for the question, **`YYYY-MM-DD`**. Think **reference day** or **“today” for parsing**.

**Why it matters.** **Yesterday** and **last week** depend on which day you call **today**. Scenario **C** uses two different **DCT** values on purpose.

**If you leave it blank.** Many setups treat empty **DCT** as **today’s real calendar date**.

---

## Section 7: Steps **0** to **4** on screen

After you submit a question, you see expandable blocks.

### Section 7.1 Step **0** (question time)

Shows **mode**, **`q_start`**, **`q_end`**, and errors if any.

**Good:** **`q_start`** and **`q_end`** are filled when the question had a clear time.

**Bad:** Both empty. Then the time filter in Step 2 does little. See Section 10.

### Section 7.2 Step **1** / **1b** (similar chunks)

Table of **`chunk_id`** and scores. High score means similar words. Step 2 checks dates.

### Section 7.3 Step **2** (time filter)

**PASS** or **DROP** per row. Scenario **B** wants **DROP** for all rows when the question year does not match stored chunks.

### Section 7.4 Step **3** / **3b** (facts)

Lists of entities, events, dates. Neo4j or file-backed output both count as OK depending on your setup.

### Section 7.5 Step **4** (text)

Highlighted line plus full chunk text when files on disk support it.

---

## Section 8: Three sample questions

These fit **English news-style** indexes (for example CC-News). Your index may differ.

### Section 8.1 Scenario **A** (strong word match)

**DCT:** **`2025-06-01`**

```text
The Home Office figures showed 1,195 migrants arrived in 19 boats on Saturday and Defence Secretary John Healey said Britain had lost control of the borders.
```

**What you hope to see.** Step **4** text mentions **1195**, **19 boats**, **Healey**, **Saturday** if your index contains that story.

### Section 8.2 Scenario **B** (wrong year)

**DCT:** **`2025-06-01`**

```text
How many wrong-side road collisions did Transport Scotland report in 1995?
```

**What you hope to see.** Step **1** may still show Scotland chunks. Step **2** should **DROP** them when stored ranges do not cover **1995**.

### Section 8.3 Scenario **C** (**yesterday**)

Same question twice. Change **only DCT**.

```text
What happened yesterday regarding migrants and small boats in the Channel?
```

| Run | **DCT** | Often resolves **yesterday** to | What to notice |
|-----|---------|--------------------------------|----------------|
| **C1** | **`2025-06-02`** | **`2025-06-01`** | Step **0** dates and Step **2** passes. |
| **C2** | **`2020-03-10`** | **`2020-03-09`** | Compare Step **0** and Step **2** to **C1**. |

---

## Section 9: Same thing in the terminal

```bash
cd /path/to/your/repo/root
python3 -m temporal_rag.interactive_query
```

**`ONLINE_QUERY_PARSER=0`** turns off heavy models on the question (see **`README.md`** Section 9). Use only when you mean to.

---

## Section 10: Empty **`q_start` / `q_end`**

**What you see.** Step **0** shows no question dates. The app may say it runs FAISS **without** the time overlap filter. The time step cannot do its job without a window.

**What to try:**

1. Add a clear date in the question, for example **June 2024** or **2024-06-15**.
2. If you use **yesterday** or **last week**, set **DCT** in the sidebar to the reference day you intend.
3. If Step **0** shows **parser_error** or red text, fix that first (for example GLiNER install issues). Read the error message.

---

## Section 11: When you are done

1. Stop the app with **Ctrl+C**.
2. If you took screenshots for your thesis or slides, save them to your local folder.
3. Before changing database tables or JSON field names, read **`README.md`** Sections 6-7 (databases) and 11 (build pipeline).

---

_End of demo guide._
