# GLM generation prompt — news-domain temporal reasoning items (v8)

**How to use.** Send the block below **once per news article** (or once per
batch of 3-5 articles), pasting the article text where marked. Collect the
JSONL output into `data/manual_aug/glm_news_batch<N>.jsonl`. Validate every
batch with `./venv/bin/python scripts/validate_glm_news.py <file>` **before**
it goes anywhere near a training run — this project has shipped two bad
synthetic slices before (`AUG_NLI` label leakage, `AUG_REASON` incoherence),
both of which cost a full cycle.

**Targets:** `Counterfactual` ~3,000 items · `Computation` ~2,000 ·
`Unanswerable` ~500. Rationale in `docs/v8_plan.md` § 6.

**Source corpus requirement.** Use news articles that are **not** from the
TIME, TimeBench or TRAM benchmarks. Record the corpus name and date window
for every batch so the D27/D28 leakage purge can verify disjointness. If in
doubt, prefer articles published **after 2023**.

---

## THE PROMPT (paste from here)

You are building a training dataset for temporal reasoning over news. I will
give you a news article. Generate multiple-choice questions that require
reasoning about **time** — when things happened, in what order, how long
apart, and what would have followed if the timing had been different.

Produce exactly:
- **4 `Counterfactual` items** — the answer requires reasoning about a
  hypothetical change to the timeline ("if X had happened before Y instead
  of after…", "had the announcement come a week earlier…"). The answer must
  be genuinely derivable from the article's actual timeline, not invented.
- **3 `Computation` items** — the answer requires **calculating** a date,
  duration, or interval from information in the article (days between two
  events, the date N weeks after an event, how long a period lasted). **The
  answer must NOT appear verbatim in the article** — it must be computed.
- **1 `Unanswerable` item** — a question that sounds answerable from the
  article but genuinely is not, because the article does not contain the
  needed information. The correct answer is the "cannot be determined"
  option.

### Output format

Output **JSONL only** — one JSON object per line, no markdown fences, no
commentary before or after. Each line must be exactly this shape:

```
{"source_dataset":"AUG_GLM_NEWS","category":"Counterfactual","question":"<question text>\nChoices:\nA. <option>\nB. <option>\nC. <option>\nD. <option>","context":"<the relevant 200-800 words of the article>","targets":["<full text of the correct option>","<letter A-D>"],"rationale":"<2-4 sentences of reasoning that ends at the answer>","source":"glm_news","corpus":"<corpus name>","published":"<YYYY-MM>"}
```

### Hard requirements — an item that breaks any of these is unusable

1. **`targets` must be a two-element list: the full correct option text
   first, then its letter.** Example: `["March 18, 2024.", "C"]`. Not the
   letter alone. Not the text alone.
2. **All four distractors must be the same TYPE as the correct answer.** If
   the answer is a date, all four options are dates. If it is a person, all
   four are people's names. If it is a duration, all four are durations.
   **Never mix types** — an option list where only one entry is a date makes
   the question solvable without reading, and the model learns that shortcut
   instead of the reasoning.
3. **Distractors must be plausible**, i.e. wrong but not absurd. Dates should
   be near the correct one (same year or adjacent months). Names should be
   people who could plausibly appear in such an article.
4. **Answers must be terse** — at most about 10 words. Prefer a date, a name,
   a duration, or a short phrase.
5. **Exactly one option is correct.** No "both A and B", no "all of the
   above".
6. **The `Unanswerable` item's correct option** must be one of exactly these
   four strings: `"There is no answer."`, `"Cannot be determined from the
   context."`, `"The passage does not say."`, `"None of the options is
   supported by the passage."` — pick one, and place it at a **random**
   letter position (not always A, not always D).
7. **For `Computation` items the rationale must show the arithmetic** — e.g.
   "The article dates the summit to 3 March and says the follow-up came 16
   days later; 3 March + 16 days = 19 March."
8. **`context` must contain everything needed** to answer, except for the
   `Unanswerable` item, where it deliberately must not.
9. **Do not reuse the same event** for more than two items in a batch.
10. **Dates in options must be formatted consistently** with how the article
    writes them.

### Worked examples of the shape (invent your own from the article)

```
{"source_dataset":"AUG_GLM_NEWS","category":"Computation","question":"If the recall was announced 12 days after the first illness was reported, on what date was the recall announced?\nChoices:\nA. November 6, 2023.\nB. November 18, 2023.\nC. October 25, 2023.\nD. December 2, 2023.","context":"Health officials said the first illness was reported on November 6, 2023 ...","targets":["November 18, 2023.","B"],"rationale":"The article gives the first reported illness as November 6, 2023. Adding 12 days gives November 18, 2023.","source":"glm_news","corpus":"cc-news","published":"2023-11"}
{"source_dataset":"AUG_GLM_NEWS","category":"Counterfactual","question":"Had the safety advisory been issued before the shipments began rather than after, which consequence described in the article would most likely not have occurred?\nChoices:\nA. The distributor expanding into new regions.\nB. The interstate spread of contaminated product.\nC. The quarterly earnings report being delayed.\nD. The appointment of a new safety director.","context":"...","targets":["The interstate spread of contaminated product.","B"],"rationale":"The article attributes the interstate spread to product shipped before the advisory. An advisory issued first would have halted those shipments, so that specific consequence depends on the ordering.","source":"glm_news","corpus":"cc-news","published":"2023-11"}
```

### The article

<<<PASTE THE NEWS ARTICLE HERE>>>

## (paste to here)

---

## After generation

```bash
./venv/bin/python scripts/validate_glm_news.py data/manual_aug/glm_news_batch1.jsonl
```
The validator checks the schema, the dual-gold shape, distractor **type
homogeneity**, answer length, letter-position balance (to catch a positional
bias the model could exploit), whether `Computation` answers wrongly appear in
the context, and collisions against all three benchmarks. Fix or drop what it
rejects — do not train on an unvalidated batch.
