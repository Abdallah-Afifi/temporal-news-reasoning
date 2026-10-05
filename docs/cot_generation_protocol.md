# CoT/STaR trace generation — protocol (2026-09-24)

Prep work for a future v-cycle. **Explicitly not part of v11** — the
researcher's own instruction: leave v11 running as-is, do CoT on a later
version. Nothing here touches the GPU or the v11 search.

## Why GLM chat, not self-generation

Two techniques exist in the repo for producing chain-of-thought training
traces:

1. **STaR self-generation** (`scripts/generate_cot_data.py`, the model
   fine-tuned so far acting as its own "teacher"; pilots in
   `data/cot/star_pilot_A.jsonl` / `star_pilot_B.jsonl`). Cheap — no manual
   work — but the measured yield was only **16% verified** (correct answer
   AND non-circular reasoning). Spot-checking `star_pilot_A.jsonl` found
   traces that guess a fact, restate it, guess again, and land on the right
   answer without ever citing the passage.
2. **GLM chat generation** (this protocol). Same manual workflow as the
   AUG_GLM2 campaign: paste a standing brief, feed short orders, verify and
   ingest replies. `data/cot/pilot_manual_glm53.jsonl` (98 rows, GLM-5.3)
   already proved this produces genuinely passage-grounded reasoning — spot
   checked against the real TimeQA source document (Philip Tartaglia /
   Archdiocese of Saint Andrews and Edinburgh) and confirmed the cited fact
   is really in the passage, just further in than a shallow read would find.

This protocol scales technique 2. Technique 1's files are kept for the
record, not deleted, but nothing here depends on them.

## Two generation modes

Both share one output shape: `{"id": ..., "cot": "Step 1: ...\nANSWER: <answer>"}`
(plus extra fields for invented mode — see below). Full instructions for GLM
are in `data/cot_packets/MASTER_PROMPT.md`.

### Grounded

`ORDER: mode=grounded ids=[...]` — real TimeQA question+passage+gold triples,
sampled from `data/combined_80_20_v11/train.jsonl` (`scripts/build_cot_packets.py`),
so already benchmark-decontaminated (audit 2026-09-23 §7) and within a
15,000-char context cap (excludes the top ~5% longest Wikipedia pages).
**GLM writes ONLY the `cot` field** — it never re-transcribes the question,
passage or gold, so it cannot introduce a mismatched passage or a wrong
answer; those are re-attached at ingest time from `answer_key.json` by id.
Current pilot: 100 rows, 10 order blocks of 10 (`ORDERS_grounded.txt`).

### Invented

`ORDER: mode=invented rows=<n> category=<name> domain="..." era=YYYY-YYYY` —
GLM invents the passage, question and gold together, same shapes and rules
as the main AUG_GLM2 brief (`data/glm_packets/MASTER_PROMPT.md`), but the
rationale becomes a full step-by-step trace instead of one sentence. Not yet
scaled to a packet batch (no `ORDERS_invented.txt` built) — write orders by
hand from the category cards when ready, following the format spelled out in
`MASTER_PROMPT.md`'s Mode INVENTED section.

## Verification gate (`scripts/ingest_cot_batch.py`)

Every trace must pass, machine-checked, before it's banked:

1. **Answer-match** — the `ANSWER:` line, under the project's own scorer
   normalization (`src/evaluation/metrics._normalize_answer` +
   `src/evaluation/date_equivalence.matches_any` — the SAME functions
   `rescore_v5_protocol.py` uses) must equal the real gold. A trace that
   "verifies" here would also score correct at eval time.
2. **Step bounds** — 1 to 5 `Step N:` lines.
3. **Grounding (a)** — step 1 must not *announce* the answer as a conclusion
   ("the answer is X") instead of deriving it. Checked by a giveaway-phrase
   regex, not plain containment — containment alone false-flagged legitimate
   traces where step 1 naturally names the correct entity as part of real
   extraction (found and fixed during testing, `tests/test_cot_ingest.py`).
   **What is actually enforced (clarified 2026-10-04,
   `audit_2026_10_04.md` §2.7):** a trace is rejected only if step 1 contains
   the gold *and* a giveaway phrase. Gold-in-step-1 on its own always passes
   — 92% of the coding-agent traces and 37% of the chat-era traces have it —
   so this gate does not stop "answer in step 1, pad step 2".
4. **Grounding (b)** — some step must share a ≥6-token run with the passage
   that includes a digit or a proper noun. A weak floor, not a faithfulness
   check — it exists to catch a trace that never touches the source text at
   all. **Found live during testing**: a deliberately fabricated trace ("He
   was a famous Australian politician... well known for his long career",
   correct answer copied in with no real grounding) passed an earlier
   4-token, no-specificity-required version of this check by chance overlap
   with generic connective phrasing. Raised to 6 tokens + a specificity
   requirement; the fabricated trace is now correctly rejected
   (`tests/test_cot_ingest.py::test_fabricated_trace_with_generic_phrasing_is_rejected`).

None of these four checks is a substitute for actually reading a sample of
accepted traces before deciding to scale past the pilot — they catch the
specific failure modes discovered so far, not "is this good reasoning" in
general.

## Data separation guarantee (researcher's requirement, 2026-09-24)

`data/cot_verified/{grounded,invented}.jsonl` is the permanent, standalone
CoT pool. When a future training mixture incorporates it (a v12 or later
config), that build script must READ from `cot_verified/` and WRITE a
separate `combined_80_20_v<N>/` directory — same pattern as every mixture
build so far (`build_v11_parity_data.py` reads `manual_aug_glm/` +
`combined_80_20_v10_glm/`, both left untouched, itself untouched by v11
training). **`cot_verified/` must never be modified, filtered in place, or
merged into another directory by a mixture-build step.** The researcher
wants to always be able to get the CoT data alone, independent of whatever
it eventually gets mixed into.

## Status (2026-09-24)

100 grounded packets built, not yet sent to GLM. 0 traces verified/banked.
`data/cot_verified/` does not exist until the first batch is ingested.

## Provenance note (2026-10-05, `audit_2026_10_04.md` §1.3)

Not every banked trace was written under this protocol. Per-trace authorship
is in `data/cot_verified/provenance.json` (each id → raw file, author,
evidence). Bank at that date: **1,713 traces** (1,693 grounded + 20
invented).

| author | traces | raw files | 2 steps | gold verbatim in step 1 | mean words |
|---|---|---|---|---|---|
| `glm-chat` | 906 (886 grounded + 20 invented) | 001–102, inv_001–004 (2026-09-24/25) | 801 (88%) | 37.0% | 83 |
| `coding-agent` | 807 | 103–197 (2026-10-04) | 801 (99%) | 92.1% | 67 |

- **Boundary**, verified from both mtimes and style rather than assumed: the
  raw files jump from 2026-09-25 00:41 (`102.txt`) to 2026-10-04 17:20
  (`103.txt`); gold-in-step-1 per file goes from about 36% (001–090) to about
  72% (091–102, which are stylistically in between and stay `glm-chat` by
  date) to about 92% (103–197).
- **How the coding-agent traces were made:** the GLM-backed coding-agent
  session wrote them using `scripts/glm_v13_build/cot_evidence.py`. That
  script ranks passage sentences by overlap with the gold (+3 per gold
  token), so the evidence was **selected knowing the answer**. These traces
  are correct and grounded, but thinner: almost all have 2 steps and most
  name the answer in step 1.
- **Delivery of the `glm-chat` traces:** the text is GLM-authored, but
  whether it came through the chat UI or through the GLM-backed agent can't
  be verified from disk.
- **Using it later:** a future CoT v-cycle can filter or weight by `author`
  using `provenance.json` without touching `cot_verified/`, which the data
  separation guarantee above leaves read-only.

## Next steps, in order

1. Generate + ingest the 100-row grounded pilot; read a sample of accepted
   traces by hand (the gate is a floor, not a quality bar).
2. Decide scale: more grounded rows (pool has 3,709 TimeQA candidates total,
   §"grounded pool" in `build_cot_packets.py`'s own output) and/or write
   `ORDERS_invented.txt` for Mode INVENTED.
3. Only then: design the training mixture and config for the CoT v-cycle
   (not started — no `config_v12*.yaml` or `combined_80_20_v12` exists yet).
