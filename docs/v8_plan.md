# v8 — the DOMAIN arm (news-native data + CoT)

> **⚠ Premise partly superseded (2026-09-13) — re-read before executing.**
> This plan targets the *domain* gap (Wikipedia training vs news benchmark).
> Two later findings change what a v8 should optimise:
>
> 1. **`Counterfactual` is no longer where fine-tuning loses** — v6 and v7c
>    *gain* +13.1pp there (a position-bias correction, verified not a
>    shortcut). The claim below that it is a top loss bucket was true of v1–v5
>    only. `Computation` remains a real loss (v7c 1.8% vs zero-shot 18.8%).
> 2. **The sharpest split is not domain, it is context provenance.** TIME is
>    40.4% `base` (gold context) and 59.6% retriever-supplied, and fine-tuning
>    is **+2.2 to +2.8pp on gold context and −2.2 to −3.0pp on retrieved
>    context** (`docs/audit_2026_09_12.md` §0a-bis, `results_and_methodology`
>    §1.0). A v8 aimed at robustness to *noisy retrieved passages* addresses a
>    larger, measured effect than one aimed at news vocabulary.
>
> Also note the v8 target below ("~49.5% on TIME") was set against a pooled
> TIME number that includes the abstain artifact (§1.1). Re-derive it.
>
> Written 2026-09-07. **v8 is where the domain mismatch gets fixed** — the thing
v1-v7 all leave untouched. Target **~49.5%** on TIME.

---

## 1. The problem v8 exists to solve

D41 measured a train/test mismatch with two halves. v7 fixes the **format**
half. This is the **domain** half:

| | TIME benchmark | training core (TimeQA + TLQA) |
|---|---|---|
| domain | **news articles** (JCPOA, Qatar, FPÖ, Gaza) | **Wikipedia** (football clubs, politicians' offices) |
| multiple-choice | 60.7% | 0.0% *(v7 fixes this)* |
| mean context | 1,653 words | 1,135 words, and 39.6% have none |

`Counterfactual` (9,720 items) and `Computation` (9,372) are where fine-tuning
loses most, and **nothing in the pool teaches either**. Wikipedia factoid
lookup cannot.

## 2. The asset that already exists — and was never used

**`scripts/generate_synthetic_dataset.py` + `src/data/corpus_processor.py` +
`src/data/synthetic_templates.py` + `scripts/download_ccnews.py` is a complete
news-domain generation pipeline that has only ever produced a 300-row sample:
`data/synthetic/news_temporal_reasoning_sample.jsonl`.**

That sample is exactly the right shape. It is built from **CC-News 2023**, is
tagged `benchmark_targets: [TIME, TimeBench, TRAM]`, and already carries a
`rationale` field (i.e. CoT supervision) plus `temporal_anchor` and `messages`:

| category in the sample | n (of 300) | maps to TIME's |
|---|---|---|
| `timeline_construction` | 112 | Timeline / letter sequence |
| `explicit_date_extraction` | 77 | Localization |
| `implicit_reference` | 52 | Relative_Reasoning |
| `temporal_ordering` | 19 | Order_Reasoning, Order_Compare |
| `arithmetic` | 18 | **Computation** |
| `duration` | 12 | Duration_Compare |
| `temporal_nli` | 10 | Co_temporality |

**CORRECTION 2026-09-08 — this step is NOT free, and the earlier wording here
was wrong.** It said "no download beyond CC-News", which buried the blocker:

- **`data/corpus/` does not exist on this machine.** `corpus_processor.py`
  expects `./data/corpus`; nothing is there.
- **`scripts/download_ccnews.py` hardcodes the OLD machine's path** —
  `SAVE_DIR = "/home/g2/temporal-news-reasoning/data/corpus/ccnews"`. That is
  PC 1, not this one. It must be repointed before it will write anywhere
  useful.
- It streams `stanford-oval/ccnews` from HuggingFace at **100,000 articles per
  year across 2000-2025**. That is a large, long download, not a step that
  fits inside a cycle.

**So v8a's real first task is: repoint the downloader, pull a bounded slice
(one or two recent years is plenty for 30k generated items), then run
`generate_synthetic_dataset.py --num-examples 30000`.** The generator itself
works and its 300-row sample is the right shape — but the corpus has to exist
first. Budget a download session, not an afternoon.

**MANDATORY leakage check before use.** TIME is itself news-derived. CC-News
2023 is a different window from TIME's visible content (2015-2017 era), but
this must be *verified*, not assumed: run the D27/D28 purge (exact-question
match, J>0.8 token-set Jaccard, shared-passage detection) against **TIME,
TimeBench and TRAM**, and record the counts. If overlap is non-trivial,
restrict the CC-News window rather than filtering item-by-item.

## 3. The second half: CoT

The free-text bucket is 40.4% of TIME and **only 46.0% of its golds appear in
the context at all**:

| category | n | gold present in context |
|---|---|---|
| Order_Reasoning | 4,296 | 100.0% |
| Localization | 9,894 | 70.3% |
| Explicit_Reasoning | 3,750 | 60.9% |
| Relative_Reasoning | 4,295 | 48.2% |
| Co_temporality | 4,119 | 46.2% |
| Counterfactual | 3,750 | 35.6% |
| Extract | 2,940 | 20.4% |
| **Computation** | **9,372** | **0.8%** |

**`Computation` is 0.8% extractable — it is pure date arithmetic and must be
computed, never copied.** That is precisely what CoT is for, and there is
already evidence it works: TimeBench `arithmetic` went **6.8% -> 21.8%** on
`AUG_ARITH` alone.

Sources, in order of preference:
1. **The CC-News generator's own `rationale` field** — free, already aligned,
   scales with § 2.
2. **`data/prepared_v4/cotcoll.jsonl`** — 575k temporally-filtered
   CoT-Collection rows, 711 MB, on disk, never used. Rehearsal-grade CoT.
3. **STaR self-generation** (`scripts/generate_cot_data.py`) — the pilot
   verified 32/200 (16.0%) zero-shot vs 17/200 (8.5%) few-shot, both bugs
   fixed (D34/D35). Use the zero-shot recipe. 16% yield means ~19k
   generations for 3k traces — expensive, so use it last.

**v8 needs its own eval protocol:** an `ANSWER:` anchor and
`max_new_tokens > 128`. `_postprocess_prediction` **already handles the
anchor** (verified inert on current arms: 0 predictions contain it), so only
the token budget changes. That budget change makes v8's numbers **not
directly comparable** to v1-v7 — it must be disclosed, and ideally v8 is also
scored at 128 tokens for one comparable column.

## 4. Sequencing — v8 is two arms, not one

Three variables at once (news domain + CoT + token budget) would be
uninterpretable.

| arm | change | isolates |
|---|---|---|
| **v8a** | CC-News data at scale, **no** CoT, 128 tokens | the **domain** effect, comparable to v7 |
| **v8b** | v8a + CoT rationales + `ANSWER:` + 512 tokens | the **reasoning** effect |

Run v8a first. It is free, it is comparable to every earlier arm, and it
answers the project's biggest open question — *does news-domain data fix
TIME?* — on its own.

## 5. Expected outcome

| bucket | v5 today | v7 target | **v8 target** | reasoning |
|---|---|---|---|---|
| MCQ answerable | 64.50% | 68.0% | **72.0%** | news-domain MCQ, on-distribution at last |
| MCQ no-answer | 0.74% | 40.0% | **55.0%** | more abstention signal in generated data |
| letter sequence | 0.00% | 17.0% | **25.0%** | `timeline_construction` is real news chronology, unlike `AUG_SEQ`'s dated facts |
| free text | 23.71% | 25.0% | **33.0%** | CoT on `Computation` (0.8% extractable) + news-domain extraction |
| paren permutation | 0.22% | 0.22% | 0.22% | hard ceiling, ignore |
| **TIME overall** | **38.45%** | **43.29%** | **49.52%** | |
| zero-shot | 41.49% | | | |

## 6. What to ask GLM for — the exact specification

Only after § 2 is exhausted, and only for what templates cannot produce:
**genuine counterfactual and computational reasoning over news text.**

- **Priority 1 — `Counterfactual`, ~3,000 items.** 9,720 benchmark items;
  every fine-tuned arm loses 6-10pp here; templates cannot generate
  "what would have followed if X had not happened". 4-option MCQ.
- **Priority 2 — `Computation`, ~2,000 items.** Date arithmetic over a news
  passage with the arithmetic shown in the rationale. Only 0.8% of these are
  extractable, so the rationale is the whole point.
- **Priority 3 — abstention, ~500 items.** News passages where the question
  genuinely cannot be answered from the text, with a plausible distractor set.

Required row schema (matches the existing pipeline, so it drops straight in):
```json
{"source_dataset":"AUG_GLM_NEWS","category":"Counterfactual",
 "question":"<question>\nChoices:\nA. ...\nB. ...\nC. ...\nD. ...",
 "context":"<news passage, 200-800 words>",
 "targets":["<correct option text>","<LETTER>"],
 "rationale":"<2-4 sentences of reasoning ending in the answer>",
 "source":"glm_news"}
```
Constraints that must hold or the slice is worthless:
- **Distractors type-plausible** (same answer type as the gold) — the v6
  builder found that implausible distractors teach a shallow cue that does
  not transfer.
- **Dual gold `[text, LETTER]`**, matching `AUG_MCQ`; the loader joins them
  with `"; "` and the scorer reads that shape (D39).
- **Source articles disjoint from TIME/TimeBench/TRAM** — state the corpus
  and date window so the purge can verify it.
- **Answers terse** (TIME's mean gold is 2.35 words).

## 7. Can this reach 80-90%? No — and here is the arithmetic

| | TIME |
|---|---|
| v5 today | 38.45% |
| zero-shot | 41.49% |
| v7 target | 43.29% |
| v8 target | 49.52% |
| **aggressive long-run ceiling** (MCQ 78%, no-answer 70%, sequence 35%, free text 52%) | **61.32%** |
| what 80% would require | 80.01% |

80% needs 83,961 of 104,951 items correct. Even granting MCQ **90%**,
no-answer **90%**, sequence **60%** and paren **10%**, free text would still
have to reach **76.7%** — a bucket where **54% of golds are not in the
context** and `Computation` is **0.8% extractable**. And `paren permutation`
(1,798 items, 8-element permutations, chance 1/40,320) is a permanent ~1.7%
deduction for every model.

**Realistic and worth stating in the thesis: ~43% (v7) is a genuine win over
zero-shot; ~50% (v8) would be a strong result for a 3B model; ~60% is the
aggressive ceiling. 80-90% is not reachable at this model scale on this
benchmark, and any pipeline claiming it should be suspected of leakage.**
