# Results & methodology — the LLaMA fine-tuning campaign (v1 → v6d)

Written 2026-09-11 (D72). This is the single reference for: what each arm
is, what data it trained on, which inference engine produced each number,
how the numbers are scored, and where every artifact lives. Numbers are the
**v5-protocol canonical table** (`results/rescored/v5_protocol.json`),
regenerated 2026-09-10 after the audit fixes — denominators TIME 104,939 ·
TimeBench 21,075 · TRAM 980,918.

---

> **READ FIRST — 2026-09-23 audit ([`audit_2026_09_23.md`](audit_2026_09_23.md)).**
> Every fine-tuned arm in this document (v1 → v9-glm) was **trained on one
> prompt and evaluated on another**: training used a custom system prompt and
> "Context:/Question:" with options inside the question; evaluation used
> `run_baselines._build_zero_shot_prompt` — three instruction lines, a
> separate `Choices:` block with a letter instruction, and a Premise/
> Hypothesis layout for NLI — which zero-shot follows natively and the
> fine-tuned models never saw. Every "fine-tuned vs zero-shot" delta below
> therefore measures **fine-tuning + a prompt switch**, not fine-tuning. The
> numbers are correct as measured; that is how they must be read.
> Also found: the chat template stamps the run day's date into every prompt;
> 561 TimeQA training rows share passages with TimeBench test items (298 with
> the identical gold, up to ~1.4pp of fine-tuned TimeBench); TIME/TimeBench
> "zero-shot" is the HF run while fine-tuned arms are vLLM (use `zs-vllm`);
> TRAM counts each NLI pair twice (MCQ + SAQ). The fixed arm is **v11**
> (parity prompt, decontaminated data, dev-selected hyperparameters):
> [`hpo_v11_protocol.md`](hpo_v11_protocol.md). v11 numbers are quoted only
> on test-minus-dev (`results/rescored/v11_test_minus_dev.json`).

## 1. Headline table

(**TRAM column re-run 2026-09-13** after the prompt-content bug — the figures
below are the CORRECTED ones, §9. Any TRAM number in an older document is
stale by 5–12pp and ranks the arms differently.)

> **Scorer correction, 2026-09-14b — the table below moved by ≤0.04pp.**
> `same_date` applied its content-substitution guard only to BARE YEARS, so a
> shared month or full date licensed any surrounding claim: "Britain and
> Ireland suspended flights…" was scored correct against "Russia and Britain
> suspended flights… on November 1, 2015", and "1 day after September 25,
> 2015" against "2 days after September 25, 2015". The guard now applies at
> every date level (`docs/audit_2026_09_14b.md` §A1).
>
> Effect, measured on all 32 cells: **TIME −0.01 to −0.04pp, TRAM −0.01 to
> −0.03pp, TimeBench bit-identical.** No arm changes rank, and the v6-vs-
> zero-shot gap is 0.72pp before and after. The table above carries the
> corrected values.
>
> **Figures elsewhere in this document have NOT been swept** and may still
> show the pre-correction value in the second decimal (e.g. "41.46" for
> zero-shot TIME in §1.1, §3 and §8, and the derived deltas such as
> "−8.04 vs zs"). Every one of them is within 0.04pp of correct and none
> changes a sign, a rank or a significance verdict. The authoritative source
> is `results/rescored/v5_protocol.json` / `..._tram_fixed.json`, regenerated
> by `scripts/rescore_v5_protocol.py --tram-root results/tram_fixed`.

| Arm | One-line identity | TIME | TimeBench | TRAM (re-run) |
|---|---|---|---|---|
| zero-shot | base Llama-3.2-3B-Instruct, no training | **41.44%** | 45.18% | **46.93%** |
| v1-ft | first arm: TimeQA+TLQA+Temprel core | 33.39% | 38.88% | 38.69% |
| v2-ft | + AUG_MCQ/ARITH/NLI scale-ups | 37.46% | 45.42% | 41.29% |
| v3 | **broken** (corrupted data) — superseded | — | — | — |
| v3-corr | v3 rebuilt on repaired data | 39.96% | **48.46%** | 41.48% |
| v4 | rehearsal source swap (dolly/hotpot/coqa/drop) | 40.07% | 46.72% | 43.79% |
| v5 | rehearsal ×3 volume (8,026 rows) | 38.43% | 48.07% | 42.67% |
| **v6** | rehearsal cap 2,600 + AUG_SEQ/NOANS/MCQ2 | **42.16%** | 47.44% | 42.76% |
| v7 | six bundled changes (confounded) | 36.11% | 45.60% | 39.02% |
| **v7c** | v6 + dual-gold removal (the clean arm) | **41.88%** | 45.68% | **43.89%** |
| v6d | v6 + AUG_DURATION slice isolated | 39.94% | 47.24% | 42.85% |

Random baselines: TIME 13.56%, TRAM 18.92% — but see §8.1, the TIME figure
needs a recheck for two categories. Key significances: v6 and v7c beat
zero-shot on TIME (z = +3.4 / +2.1 unpaired; McNemar +5.18 / +3.30 — see
§8), **but not once §1.1's artifact is excluded**; v7's loss is real
(z = −24.5); v6d's loss is real (z = −10.3 vs v6); the TRAM column was
re-run on corrected prompts 2026-09-13 (§9): **no fine-tuned arm beats
zero-shot**, by −3.0 to −8.2pp — a larger specialisation cost than the stale
table showed.

### 1.0 TIME must be reported stratified by retrieval setting (2026-09-12)

TIME ships a `Setting` field the loader discards and this project never used:
`base` 42,380 rows (40.4%, gold context) and `bm25` / `vector` / `hybrid`
20,857 each (59.6% combined, context supplied by a retriever). **Every TIME
number in this document is a pooled average over those four conditions**,
which sit ~23pp apart. With the §1.1 abstain artifact excluded:

| Δ vs zero-shot | base (n=42,173) | bm25 | vector | hybrid | pooled |
|---|---|---|---|---|---|
| v6 | **+2.18** (z=+11.6) | −1.73 | −2.33 | −2.44 | −0.38 |
| v7c | **+2.76** (z=+14.5) | −2.94 | −3.08 | −3.00 | −0.64 |

(zero-shot: base 27.21%, bm25 50.07%, vector 51.19%, hybrid 51.62%. McNemar on
the retrieved half: v6 z=−11.2, v7c z=−16.0, n=60,339.)

> **Fine-tuning delivers a real, significant gain on gold context and a real,
> significant loss on retrieved context, and the pooled number reports their
> cancellation as ≈0.** Both effects hold across all three retrievers
> independently and in both headline arms.

This is the campaign's strongest positive result and it is directly on the
project's own subject. Read §1.1's "no arm beats zero-shot" as a statement
about the *pooled* metric; stratified, v6 and v7c clearly beat zero-shot on
`base` and clearly lose on retrieved context. Evidence and method:
`docs/audit_2026_09_12.md` §0a-bis.

Also from that sweep: 1,800 `Extract` rows (1.7% of TIME) ship
`Context: "None"` — the retriever returned nothing — and are unanswerable for
every arm. A source-data defect, not a loader bug; candidate for the same drop
treatment as the 113 empty-gold TimeBench rows.

### 1.1 The TIME column with the abstain artifact removed — READ THIS FIRST

2,427 TIME items offer an abstain-phrased option ("There is no answer"), and
the abstain option is the gold in **2,427 of 2,427** — the benchmark never
presents one as a wrong answer. "If an abstain option is present, pick it"
therefore scores ~100% on that bucket with no temporal reasoning at all. D49
found this on 2026-09-08 and it applies to every arm trained with
`AUG_NOANS`. This table (regenerated 2026-09-12 by
`scripts/rescore_v5_protocol.py`, `no_abstain_pct`) excludes the bucket for
every arm alike:

| Arm | TIME (headline) | abstain bucket | TIME without it (n=102,512) | Δ vs zs | McNemar z |
|---|---|---|---|---|---|
| zero-shot | 41.46% | 52.95% | **41.19%** | — | — |
| v6 | **42.18%** | 99.88% | 40.81% | **−0.38** | +5.18 |
| v7c | 41.91% | 99.18% | 40.56% | **−0.64** | +3.30 |
| v6d | 39.96% | 97.20% | 38.61% | −2.58 | −10.66 |
| v4 | 40.08% | 7.4% | 40.87% | −0.32 | −10.57 |
| v3-corr | 39.98% | 7.21% | 40.75% | −0.44 | −11.19 |

(TimeBench and TRAM contain no abstain-option items, so their `no_abstain_pct`
equals their headline — the artifact is TIME-only, and on those two benchmarks
no fine-tuned arm leads zero-shot anyway, except on TimeBench where the gain is
confined to the `arithmetic` category the augmentation targets: §4.2.)

**So the campaign's honest top-line is: no fine-tuned arm beats the zero-shot
baseline on any of the three benchmarks once the abstain artifact is removed.**
The +0.73pp (v6) and +0.45pp (v7c) headline margins are that bucket and
nothing else. Quote the headline column only together with this one.

This does not make the campaign a failure — it makes it a **negative result
about specialisation**, which §4.2 and §9 already support from two other
directions. What genuinely improved is narrower and real: see §3 v6 on
`Counterfactual` (a position-bias correction, verified not to be a shortcut).

**Campaign verdicts:** v6 = best raw TIME arm; v7c = the **protocol-robust**
arm — it is the only arm that leads zero-shot under BOTH the v5 protocol
(41.91 vs 41.46) and a strict first-line-only reading (41.70 vs 40.49), where
v6 collapses to 29.22%; v3-corr = best TimeBench arm; TRAM inverts the ranking
(best TIME arm = worst TRAM arm) under micro-averaging — see §9. Under the
no-abstain column above, **no arm leads zero-shot on TIME at all.**

> Correction (2026-09-12): earlier revisions of this section credited v7c with
> "0 scorer-rule firings". That came from the rescore's `rewritten` field,
> which counts disagreement between the postprocessor NOW and the one that ran
> at generation time — protocol drift, not rule firings — so it necessarily
> reads 0 for any arm generated after the postprocessor froze. Measured
> properly (`rules_fired` / `strict_pct`, added 2026-09-12), extraction rules
> fire on 16.9% of v7c's TIME items, 49.3% of v6's and 49.4% of zero-shot's,
> and are worth **+0.21pp to v7c, +12.97pp to v6 and +1.00pp to zero-shot**.
> The v7c conclusion survives and is in fact stronger than the old wording
> claimed; the v6 one does not.
>
> Full `strict_pct` / `rule_credit_pp` columns are in
> `results/rescored/v5_protocol.json`. One number there is worth flagging:
> on **TRAM the extraction rules were worth +24.25pp to zero-shot** (10.95% →
> 35.20%) and only +5.11pp to v7c — the opposite asymmetry to TIME. (Those two
> figures are from the STALE TRAM predictions; the asymmetry itself is a
> property of the scorer, not of the prompts, but the magnitudes should be
> recomputed from `results/tram_fixed/` before being quoted.) Free-form
> generation scored by exact match makes every one of these numbers a joint
> measurement of the model and the parser; a likelihood-over-options protocol
> would remove that dependency for the MCQ majority of all three benchmarks.

## 1.5 The Chain-of-Thought arms (base-model prompting, 2026-09-09/14)

A prompting campaign on the **base models — no training, no adapters** — synced
in from the second machine. Audit: `docs/audit_2026_09_14.md`; decision: D77.

| arm | TIME | TimeBench |
|---|---|---|
| standard zero-shot (llama) | 41.46% | 45.18% |
| **zero-shot CoT (llama)** | 38.52% | **49.11%** |
| zero-shot CoT (mistral) | 33.65% | 36.08% |
| few-shot CoT | **not run** | not run |

**These numbers are not yet quotable.** Both runners used
`--max-new-tokens 256`, and scoring depends on a final `ANSWER:` line that a
truncated generation never reaches. **22.1%** of llama's TIME outputs and 7.5%
of its TimeBench outputs ran out of budget mid-reasoning and score **0.08%** —
a budget artifact, not a reasoning result. Conditional on reaching the anchor:

| | as scored | conditional on the anchor |
|---|---|---|
| llama · TIME | 38.52% | **49.41%** (+7.95 vs baseline) |
| llama · TimeBench | 49.11% | **53.01%** (+7.83) |

The conditional column is an **upper bound** — items needing more reasoning are
plausibly harder — so the truth lies between the columns and only a re-run at
the corrected 768-token budget settles it.

Two cautions when reading these arms:

- **The comparison is bundled, not attributable.** The CoT arm changes the
  system prompt (none → `COT_SYSTEM_PROMPT`), the answer instruction, *and* the
  token budget (128 → 256/768) at once. "CoT is worth +7.95pp" is really
  "persona + step instruction + 2–6× budget". Two legs would attribute it; see
  D77 §3. This is the v7 mistake in a new place.
- **`strict_pct` does not mean the same thing here.** It reads ~0% for every
  CoT arm because the first line is "Step 1: …" by construction; the `ANSWER:`
  anchor is the *designed* extraction point, specified in the prompt in
  advance. Do not compare `strict_pct` across prompting styles.

Even so, note what the as-scored TimeBench figure already is: **49.11% beats
v3-corrected's 48.46%**, the best fine-tuned arm in this campaign, with no
training at all. If the re-run holds even half the conditional gain, the
project's strongest finding may be that prompting the base model outperforms
every fine-tune it produced.

## 2. The invariant recipe (v1 → v6d, bit-identical)

| Component | Value |
|---|---|
| Base model | `models/Llama-3.2-3B-Instruct` (fp16) |
| Method | LoRA r=16, α=32 (24.3M trainable params, 0.75%) |
| LR / schedule | 4.62e-4 (HPO-selected), 3 epochs |
| Batch | 2 × 8 (grad accum), gradient checkpointing on |
| Early stopping | patience 2 (epochs), best-checkpoint restore |
| Prompting | chat template, `TEMPORAL_SYSTEM_PROMPT` always on; loss masked to completion only (`-100` on prompt tokens) |
| Seed | 42 (data builders and trainer) |
| Trainer | `experiments/finetuning/LLaMA/train.py` (per-arm `config_v*.yaml` — identical except 3 path lines) |

Every arm differs from its control by **data only** (except v7's config
which was still recipe-identical). No arm changed LR/epochs/batch/prompt.
Verified 2026-09-12: `diff config_v6.yaml config_v7_corrected.yaml` is exactly
three lines (two data paths + output dir).

**Caveat on `metric_for_best_model: eval_loss` (2026-09-12).** Checkpoint
selection uses validation loss on a split drawn from each arm's *own
synthetic mixture*. Across arms that signal is uninformative-to-inverted with
respect to the target metric:

| arm | best eval_loss | TIME |
|---|---|---|
| v6d | 0.5323 | 39.96 |
| v7 | 0.5329 | 36.15 |
| v7c | 0.5855 | 41.91 |
| v6 | 0.5864 | 42.18 |
| v4 | 0.6213 | 40.08 |

More synthetic augmentation → lower in-distribution val loss → **worse** TIME.
This is a cleaner statement of the size hypothesis (D69/D70) than the row-count
correlation, and it argues for selecting checkpoints on a small held-out slice
of the *target* distribution. (Within each arm the mechanism is sound: every
run bottoms out at epoch 2 and rises at epoch 3, and best-checkpoint restore
picks epoch 2.)

**Caveat on arm selection.** Ten arms were designed, trained and compared
against the same TIME/TimeBench sets and the headline is the maximum over
them; there is no held-out confirmation split. Combined with the absence of
seed replicates (§8.1) that is the standard multiple-comparisons hazard, and
it is why §1.1 matters: a max-of-ten margin of +0.45pp that disappears when
one artifact bucket is removed is not a robust finding.

## 3. Per-arm chapters

### v1-ft — the baseline fine-tune
- **Data** (`data/combined_80_20_split/`): 12,000 train rows — TimeQA 7,337
  + TLQA 4,066 + TempRel 597. No synthetic slices.
- **Engine**: HF (`scripts/run_baselines.py`) for TIME/TimeBench; TRAM on
  vLLM (D71, re-run on corrected prompts 2026-09-13) — **38.71%, the WORST
  fine-tuned TRAM score in the campaign** (−8.24 vs zero-shot). Under the
  broken prompts it read 33.48% and looked like the *best* FT arm; §9 explains
  the reversal.
- **Result**: TIME 33.42% (−8.04 vs zs), TimeBench 38.88% (−6.27).
- **Lesson**: naive fine-tuning on Wikipedia-style temporal QA hurts the
  news-domain benchmarks; this is the arm every later change is measured
  against.

### v2-ft — the first augmentation wave
- **Data** (`combined_80_20_v2/`): 16,557 rows — v1 core + AUG_MCQ 2,000 +
  AUG_ARITH 1,600 + AUG_NLI 1,600.
- **Engine**: HF for TIME/TimeBench; TRAM vLLM (D71, re-run 2026-09-13).
- **Result**: TIME 37.48%, TimeBench 45.42% (+0.25 vs zs — first parity),
  TRAM 41.30% (stale 31.50%).
- **Lesson**: part of v1's "catastrophe" was a scoring artifact (dual-gold
  shape, D39); AUG_NLI carried label leakage ("another" → Neutral).

### v3 → v3-corrected — the corruption incident
- v3 trained on **corrupted data** (builder bug); its numbers are void and
  its checkpoint is quarantined (`checkpoints/llama_v3/`, marked BROKEN).
- **v3-corr data** (`combined_80_20_v3_fixed/`): 12,067 rows — core 10,269
  (TimeQA 5,005 + TLQA 3,253) + REHEARSAL 2,080 + AUG_REASON 381 +
  AUG_DIALOG 597 + AUG_MCQ 400 + AUG_NLI 400 (Neutral rows filtered).
- **Engine**: HF for TIME/TimeBench; TRAM vLLM (D71, re-run 2026-09-13).
- **Result**: TIME 39.98%, TimeBench **48.46%** (still the best),
  TRAM 41.50% (stale 30.18%).
- **Lesson**: rebuild-on-repaired-data recovered +2.5pp on TIME and set the
  TimeBench record; five simultaneous changes made v3's original loss
  unattributable — the mistake that created the one-variable discipline.

### v4 — rehearsal source swap
- **Data** (`combined_80_20_v4/`): 12,830 rows — v3-corr core, REHEARSAL
  re-sourced to prepared_v4 (dolly long-form + hotpot/coqa/drop spans).
- **Engine**: HF for TIME/TimeBench; TRAM vLLM (D71, re-run 2026-09-13).
- **Result**: TIME 40.08%, TimeBench 46.72%, TRAM 43.80% (stale 32.01%).
- **Lesson**: long-form rehearsal relieves the short-span collapse (D38);
  first arm within 1.4pp of zero-shot on TIME.

### v5 — rehearsal volume ×3
- **Data** (`combined_80_20_v5/`): 18,180 rows — same as v4 but REHEARSAL
  8,026 (44% of mixture).
- **Engine**: HF for TIME/TimeBench; TRAM vLLM (D71, re-run 2026-09-13).
- **Result**: TIME 38.45% (−1.6 vs v4), TimeBench 48.07%, TRAM 42.68%
  (stale 31.57%).
- **Lesson**: the rehearsal lever is spent — more volume does not help TIME
  and re-dilutes sequence emission; taught the MCQ/format scoring defects
  that became the v5 protocol.

### v6 — the best arm
- **Data** (`combined_80_20_v6/`, `build_v6_training_data.py`): 14,850
  rows, 30.7% synthetic — REHEARSAL capped 2,600 (50% long-form) + NEW
  AUG_SEQ 1,019 (letter-ordering shape) + AUG_NOANS 400 / AUG_MCQ2 1,200
  (abstention handling) + v3-corr core.
- **Engine**: HF for TIME/TimeBench (`results/corrected/v6`); **also
  re-run on vLLM** (`v6_vllm`) for engine parity — identical to 0.02pp
  (42.18/42.18 TIME, 47.44/47.44 TimeBench); TRAM vLLM **42.78%**
  (re-run 2026-09-13; the stale figure was 31.15%).
- **Result**: TIME **42.18%** — the first and only arm to beat zero-shot
  (+0.73, z=+3.4).
- **Caveat (D66)**: part of the margin is Order_Compare prior-exploitation
  (emits the 56.9%-majority option 69.6% of the time); carries real signal
  too (68.7% > 56.9%).
- **Caveat (D49, restated 2026-09-12 — this one removes the margin
  entirely)**: v6 scores 99.88% on the 2,427 abstain-option items where the
  abstain option is always the gold. Without them v6 is 40.81% against
  zero-shot's 41.19% (§1.1). `AUG_NOANS` bought that bucket and nothing else:
  on the 2,532 items whose gold *is* "There is no answer" but which offer no
  such option, **every arm scores 0.00%**, while v6 emits a false "no answer"
  2,424 times against zero-shot's 1,291.
- **What is genuinely v6's (verified 2026-09-12)**: `Counterfactual`
  42.6% → 55.7%, and on its MCQ half 62.40% → **79.92%**. Checked and *not* a
  shortcut — gold positions are balanced (A/B/C/D = 1,444/1,556/1,440/1,530),
  "pick the longest option" is only 26.9% accurate and v6 agrees with it
  *less* than zero-shot (24.1% vs 32.1%). The mechanism is **position-bias
  correction**: zero-shot over-picks A (1,749 A vs 1,100 D), v6 is nearly flat
  (1,522/1,697/1,486/1,262). This is the strongest positive finding in the
  campaign and deserves to be reported separately from the artifact above.

### v7 — the confounded scale-up (negative result)
- **Data** (`combined_80_20_v7/`): 26,570 rows (+79% vs v6), 45% synthetic
  — SIX changes at once: rehearsal ×3.1, AUG_MCQ2 ×3.8, AUG_ARITH ×7.3,
  NEW AUG_DURATION 1,573 + AUG_RELATIVE 420, dual-gold targets removed.
- **Engine**: vLLM only (all three benchmarks, `v7_vllm`).
- **Result**: TIME 36.15% (−5.94 vs v6, z=−24.5 vs zs), TimeBench 45.60%
  (ns), TRAM **39.04%** (re-run; stale 31.40%).
- **Lesson**: methodological goal met (0 trailing-letter rewrites) but the
  accuracy collapsed; every scale-up failed at its own target category;
  steps 4,983 vs 2,787 at fixed LR = over-training/drift signature (D66).

### v7c — v6 + dual-gold removal (the clean arm)
- **Data** (`combined_80_20_v7_corrected/`, `build_v7_corrected_training_
  data.py`): v6's mixture verbatim; the single change is
  `AUG_MCQ`/`AUG_MCQ2`/`AUG_NOANS` targets `[text, LETTER]` → `[text]`
  (2,000 train + 499 val rows; verified bit-identical otherwise).
- **Engine**: vLLM (all three).
- **Result**: TIME 41.91% (z=−1.25 vs v6 — parity; z=+2.1 vs zs), TimeBench
  45.68%, TRAM **43.92% — the best fine-tuned arm on TRAM** (re-run; stale
  32.00%). v7c is therefore both the protocol-robust TIME arm and the best
  TRAM arm.
- **Verdict (pre-registered, D67/D68)**: best-case cell — the "you trained
  a nonstandard format then widened the scorer" objection removed for free
  on TIME; the price is a real −1.76pp on TimeBench. Supersedes v6 as the
  headline arm; v6 keeps the best raw score.

### v6d — v6 + AUG_DURATION isolated (negative result, decisive)
- **Data** (`combined_80_20_v6d/`, `build_v6d_training_data.py`): v6's
  **shipped files loaded verbatim** (bit-identical prefix — the current v6
  builder no longer reproduces them after the Sep-7 TRAM-guard extension)
  + the 1,573 AUG_DURATION rows, generator copied verbatim from v7's
  builder, separate rng stream (seed+1000).
- **Engine**: vLLM (all three).
- **Result**: TIME 39.96% (−2.22 vs v6, z=−10.3), Duration_Compare 32.5%
  (v6: 33.1 — **no gain on its own target**), TimeBench 47.24% (duration
  79.3%, best ever), TRAM **42.86%** (re-run; stale 31.56%).
- **Verdict (pre-registered, D69/D70)**: no-gain cell — **v7's
  Duration_Compare +7.0 was an artifact of its confounded bundle**, not the
  slice. Second data point for the size hypothesis (+10.6% rows → −2.22pp;
  v7's +79% → −5.94pp): growing the mixture at fixed hyper-parameters
  degrades TIME whatever the rows are.

## 4. Data methodology (all arms)

- **Sources**: only the TimeQA / TLQA **training pools** + prepared_v4
  rehearsal corpora (dolly/hotpot/coqa/drop) + synthetic AUG_* slices
  generated from those pools. **No benchmark item, question or answer is
  ever read** by any builder.
- **Guards in every builder from v2 onward** (hard asserts, run stops on
  violation): letter-probe leakage = 0, benchmark-question collisions = 0
  (guard extended to TRAM on 2026-09-07), in-file dedup, near-duplicate purge.
  Re-verified independently 2026-09-12 on every shipped split: 0 collisions
  under both exact and punctuation-insensitive matching.
  **Exception — v1.** `data/combined_80_20_split/` predates the guard and
  contains **49 verbatim benchmark questions** (of 15,000 rows). The effect on
  v1's reported numbers is negligible (49 items against denominators of
  104,939 / 21,075) but v1 is the control every later arm is measured against
  and currently the best TRAM arm, so the contamination must be stated, not
  covered by the blanket "every builder" claim.
  (`results/overlap_audit_fullzip.json` — 104/10,013 TimeBench questions,
  concentrated in timeqa_easy 8.6% / menatqa_scope 5.84% — is a **v1-era**
  audit; it does not describe v2+.)
- **Splits**: stratified 80/20 by source, seed 42; manifests written per
  arm (`data/*/manifest.json`) with per-source counts and answer-style
  stats.
- **Single-variable discipline**: each arm changes ONE thing vs a named
  control; v7 broke this rule once (six changes) and its regression took
  two further arms (v7c, v6d) to attribute. New arms get NEW builder +
  config + schedule files; earlier ones are never edited (reproducibility).
- **Bit-identity verification**: v7c and v6d cores were diffed row-by-row
  against v6's shipped files before training (exact-prefix property).

## 5. Evaluation methodology

### 5.1 Engines

| Arm | TIME | TimeBench | TRAM |
|---|---|---|---|
| zero-shot | HF | HF | vLLM |
| v1…v5 | HF | HF | vLLM (D71 campaign, complete) |
| v6 | HF + vLLM parity re-run | HF + vLLM parity re-run | vLLM |
| v7, v7c, v6d | vLLM | vLLM | vLLM |

- **HF path** (`scripts/run_baselines.py`): transformers generate, greedy,
  length-sorted batches under a 57,344-token budget (batch 32 for
  TIME/TimeBench per D47).
- **vLLM path** (`scripts/run_eval_vllm.py`): v0.28, greedy (temp 0.0),
  seed 42, dataset-order chunks of 1000, fp16, max_model_len 4226,
  `VLLM_USE_FLASHINFER_SAMPLER=0` (D55 flashinfer/CUDA-13 fix — provably
  number-neutral under greedy decoding).
- **Parity evidence**: v6 HF vs vLLM identical to 0.02pp on both benchmarks.
  Item-level agreement measured on the FULL runs (2026-09-12), not the old
  2,000-item pilot: **TIME 99.55% exact / 99.58% normalized; TimeBench 99.25%
  / 99.49%** — both **pass** the ≥99% gate coded in `compare_parity.py:57`.
  (Earlier revisions quoted 94.4% from the pilot, which is also what audit
  2026-09-09 M4 called a gate failure; both are superseded by these numbers.
  Residual differences are batch-composition effects under greedy.)
  **Mistral is banned from vLLM** (2.20pp parity failure with prompt-bug
  signature, D46/D53).
- **Prompting identical on both engines**: chat template, training system
  prompt for fine-tuned arms (verified in-training back to v1), left
  truncation at 4096, `add_special_tokens=False` (no double-BOS),
  `max_new_tokens=128`.
- **Rule**: consistency is required WITHIN a benchmark; TIME/TimeBench
  keep per-arm engines, TRAM is all-vLLM. Every cell must record the
  engine (predictions carry `engine`/`model_dir`/`adapter_dir` tags since
  2026-09-09).
- **Numeric precision**: training runs in **bf16** (`dtype: bfloat16` in
  every `config_v*.yaml`); evaluation runs in **fp16** on both engines
  (`src/models/inference.py` `torch_dtype=float16`, `run_eval_vllm.py`
  `dtype="float16"`). Uniform across every arm, so no comparison is affected,
  but it is a real train/eval precision change and belongs in the record.
- **Merging**: vLLM evaluates merged fp16 models
  (`scripts/merge_lora.py`, ~4 min CPU, regenerable); `--adapter-dir`
  selects the training system prompt only — LoRA weights are never loaded
  by vLLM directly.

### 5.2 Scoring protocol (the v5 protocol + 2026-09-09 fixes)

Exact match on normalized answers, with these rules (order matters):

1. `_normalize_answer` (lowercase, articles, punctuation).
2. Postprocessing of raw predictions: exact option match → bare/prefixed
   letter → **containment** (word-boundary, negation-guarded, last-match —
   fixed 2026-09-09; previously substring, which credited denied options)
   → unambiguous partial prefix (≥5 chars) → trailing-letter
   self-declaration with letters-only guard.
3. `matches_any` date equivalence (`src/evaluation/date_equivalence.py`).
4. **Empty-gold items dropped for all arms alike** (2026-09-09): 113
   TimeBench `timeqa` source rows with `answer == ['']` and 12 TIME
   Timeline rows — hence the denominators 21,075 / 104,939.
5. `situated_generation` (115 items) reported additionally as token-F1,
   never folded into the EM headline.

Protocol history: raw scorer → v5 protocol (2026-09-06/07: partial-option,
date equivalence, trailing-letter) → 2026-09-09 audit fixes (above) →
**2026-09-12: diagnostics only, scoring unchanged.** Every protocol change is
applied **uniformly to every arm** by re-running
`scripts/rescore_v5_protocol.py` (~8 min CPU) from the immutable
`predictions.jsonl` files — raw predictions are never modified. Torn-write
lines are counted and skipped, never silently truncated.

The 2026-09-12 pass added **four reported columns and changed no score**:

| field | meaning |
|---|---|
| `rules_fired` / `rules_fired_pct` | how often `_postprocess_prediction` rewrote the model's first line — the real "scorer-rule firing" count |
| `strict_pct` / `rule_credit_pp` | accuracy with NO extraction rules (bare first line), and how many pp the rules are worth to this arm |
| `no_abstain_pct` | the headline protocol with the abstain-option bucket excluded (§1.1) |
| `macro_pct` | unweighted mean over categories, for §9's aggregation question |
| `_paired` + `mcnemar()` | paired McNemar cells and z against zero-shot |

`rewritten` is retained for old readers but is now also emitted as
`protocol_drift`, which is what it always measured — see the §1.1 correction.

## 6. Checkpoints

Full manifest: `checkpoints/README.md` (one row per arm: contents, size,
verdict, reclaimable space). Every arm's `final/` is the adapter its
published numbers used; mid-checkpoints are per-epoch saves; `merged_fp16/`
dirs are vLLM-loadable full models, regenerable in ~4 min. v3 is
quarantined as BROKEN. ~12G reclaimable (v3+v7 merges) without losing any
number.

## 7. Code organization

- `experiments/finetuning/LLaMA/` — the real trainer (`train.py`,
  `data_loader.py` with prompt-loss masking), per-arm configs
  (`config_v*.yaml`), HPO artifacts.
- `scripts/` — builders (`build_v*_training_data.py`, one per arm),
  `probe_perturbation_consistency.py` (moved out of `tests/` on 2026-09-12 —
  it is a GPU driver that defined no test),
  eval runners (`run_baselines.py` HF, `run_eval_vllm.py` vLLM),
  schedule drivers (`run_schedule_*.sh` — per-arm, idempotent, `.done`
  files are the source of truth), scoring (`rescore_v5_protocol.py`,
  `mcnemar_v5_protocol.py`), analysis (`probe_answer_formats.py`,
  `audit_*.py`), and `scripts/retired/` (six dead/dangerous tools kept
  for history — including the engine-mixing TRAM baseline script).
- `src/` — ALIVE: `data/data_loader.py` (BenchmarkLoader),
  `evaluation/{metrics,date_equivalence,generation_metrics}.py`,
  `models/inference.py`. DEAD (documented): `rag/`, `pipeline/`,
  `temporal/`, `prompting/`, `demo/`, `training/` (a divergent twin
  trainer — warned in its README, do not reproduce from it).
- `tests/` — 133 tests (127 pass, 6 skip without a JVM) pinning the
  scoring/postprocessing/loader contracts
  (incl. the 2026-09-09 denial-guard, the empty-gold behavior, and the
  2026-09-12 abstain-bucket / McNemar diagnostics, the TRAM prompt-content
  fix, and direct unit tests for `_normalize_answer` / `accuracy` /
  `_canonical_golds`). Command:
  `./venv/bin/python -m pytest tests/ -q`. Note 4 of them
  (`test_heideltime_wrapper.py`) need a JVM and fail wherever `java` is not
  installed — as it is not on this machine; nothing in the thesis chain
  depends on HeidelTime.
- Logs: per-schedule under `logs/sched_*/` with driver logs, per-leg logs,
  and `.done` markers.

## 8. Reference numbers and significance anchors

| Comparison | Value |
|---|---|
| Random baselines | TIME 13.56% · TRAM 18.92% (but see §8.1) |
| v6 vs zero-shot (TIME) | +0.73pp, z = +3.4 unpaired / **McNemar +5.18** — **−0.38pp without the abstain bucket** |
| v7c vs v6 (TIME) | −0.27pp, z = −1.25 (parity) |
| v7c vs zero-shot (TIME) | +0.45pp, z = +2.1 unpaired / **McNemar +3.30** — **−0.64pp without the abstain bucket** |
| v7 vs zero-shot (TIME) | −5.32pp, z = −24.5 unpaired / McNemar −34.88 |
| v6d vs v6 (TIME) | −2.22pp, z = −10.3 |
| Best TRAM FT vs zero-shot | −3.03pp (v7c 43.92 vs 46.95), fixed prompts, §9 |
| Engine parity (v6) | 0.00pp TIME · 0.00pp TimeBench (42.18/42.18, 47.44/47.44); item agreement 99.55% / 99.25% |

### 8.1 How to read the significance numbers (2026-09-12)

Three caveats that apply to every z in this document:

1. **The published z is the wrong test, conservatively.** `ztest()` in
   `rescore_v5_protocol.py` is an **unpaired** two-proportion z applied to
   **paired** data (identical items, different arms), so it ignores the ~80%
   of items on which two arms agree and over-states the standard error.
   McNemar is the correct test and is now reported alongside it; it is
   uniformly stronger, so no significance call flips.
2. **n is not a count of independent trials.** TIME's 104,939 items come from
   **49,316 unique contexts** and **56,229 unique question strings** (one
   template repeats 11,361 times, one context 1,800 times). TRAM's 980,918
   items are ~465k problems counted **twice**, once as MCQ and once as SAQ
   (`arithmetic_mcq` 15,584 / `arithmetic_saq` 15,584; `nli_mcq` 282,134 /
   `nli_saq` 282,134; and so on). "Significant at n = 980,918" is therefore
   not informative — cluster on context (TIME) / problem (TRAM), or report per
   task.
3. **There is no variance estimate anywhere.** One training run per arm, one
   seed (42), no replicate, no confidence interval — flagged in
   `docs/audit_report.md:82` and still open. The decisive margins are
   +0.45–0.73pp and nothing establishes they exceed run-to-run variance.
   **One seed-43 replicate of v6 (~8 GPU-hours) is the highest-value run
   left in the project.**

Also: the TIME random baseline of 13.56% assumes 1/n-choices per item, which
is wrong for two categories — `Timeline` is a permutation task (mostly 3
facts → 1/6 = **14.4%**, so every arm's 16.5–18.8% is barely above chance on
12.5% of the benchmark) and `Extract` is **multi-select** over 4 options
(1/15 ≈ **6.7%**, so its 7–10% is slightly above chance, not far below it).

## 9. The TRAM column — RE-RUN AND RESTORED (2026-09-13)

The 2026-09-12 prompt-content bug (`docs/audit_2026_09_12.md` §0a) is fixed and
**all ten arms were re-run on corrected prompts**: `scripts/run_schedule_tram_rerun.sh`,
2026-09-12 14:51 → 2026-09-13 13:19, every stage OK, 980,918 items per arm,
protocol otherwise unchanged (vLLM, greedy, seed 42, same scorer). Canonical
numbers: `results/rescored/v5_protocol.json` (the stale table is kept as
`v5_protocol.stale_tram.json`). **TIME and TimeBench are unaffected — all 22
of their cells reproduce exactly.**

| rank | Arm | TRAM (fixed) | stale | shift | vs zero-shot | macro |
|---|---|---|---|---|---|---|
| 1 | zero-shot | **46.95%** | 35.20% | +11.75 | — | 47.76% |
| 2 | **v7c** | **43.92%** | 32.00% | +11.92 | **−3.03** | 47.42% |
| 3 | v4 | 43.80% | 32.01% | +11.79 | −3.15 | 47.78% |
| 4 | v6d | 42.86% | 31.56% | +11.30 | −4.09 | 46.59% |
| 5 | v6 | 42.78% | 31.15% | +11.63 | −4.17 | 46.59% |
| 6 | v5 | 42.68% | 31.57% | +11.11 | −4.27 | 45.96% |
| 7 | v3-corr | 41.50% | 30.18% | +11.32 | −5.45 | 45.73% |
| 8 | v2 | 41.30% | 31.50% | +9.80 | −5.65 | 46.14% |
| 9 | v7 | 39.04% | 31.40% | +7.64 | −7.91 | 44.25% |
| 10 | **v1** | 38.71% | 33.48% | +5.23 | **−8.24** | 43.42% |

### Readings (these REPLACE the 2026-09-11 set)

1. **No fine-tuned arm beats zero-shot on TRAM — and the cost is LARGER than
   the stale table showed**: −3.0 to −8.2pp, against the stale −1.7 to −5.0pp.
   The specialisation–generalisation trade-off is a firmer result than before,
   not a weaker one.
2. **The stale ranking was close to meaningless.** Spearman ρ between the
   stale and corrected orderings of the nine fine-tuned arms is **+0.17**.
3. **v1 inverts from best to worst** (rank 1 → 9). The 2026-09-11 reading
   *"v1's mild fine-tune is the best TRAM arm — the mildest intervention costs
   the least off-domain"* is **withdrawn and reversed**. See the mechanism
   below.
4. **v7c is the best fine-tuned arm on TRAM** (−3.03pp) as well as the
   protocol-robust arm on TIME (§1.1). The headline now converges on one arm
   instead of the v6/v7c split.
5. **The micro/macro divergence closes.** Macro-average: zero-shot 47.76,
   v4 47.78, v7c 47.42 — a tie at the top, not the reversal the stale data
   showed. §4.1's TRAM caveat is resolved.

### Mechanism: it is all temporal NLI, and it is a format effect

| task | n | %TRAM | zero-shot | v1 | v4 | v1 contrib | v4 contrib |
|---|---|---|---|---|---|---|---|
| temporal_nli | 564,268 | 57.5% | **53.8%** | 43.0% | 50.3% | **−6.17** | −2.02 |
| temporal_relation | 204,914 | 20.9% | 26.1% | 19.8% | 20.9% | −1.32 | −1.09 |
| storytelling | 67,204 | 6.9% | 76.9% | 68.4% | 73.6% | −0.59 | −0.23 |
| all others | 144,532 | 14.7% | — | — | — | −0.16 | +0.19 |

Zero-shot scores **53.8%** on temporal NLI, not the 34.6% the broken prompts
reported — it was never at chance; the task was simply unanswerable. Fine-tuning
costs 10.8pp of that for v1 and 3.5pp for v4, and NLI's 57.5% weight turns that
single task into the whole ranking.

The explanation is consistent with D38's short-span collapse: **v1 trains purely
on span-extraction QA (TimeQA/TLQA) with no classification slice, so it loses
the ability to emit a class label at all** (entailment / neutral /
contradiction). Every arm that added MCQ and NLI augmentation keeps it, and v4
and v7c — the arms with the most classification-shaped data — keep the most.
So the mild intervention is not the cheapest: **pure span extraction is the most
format-destructive thing you can do to a classification task.**

### The section as written on 2026-09-11 (retained for the record, NOT valid)

D71 campaign finished 2026-09-11 22:51, every stage OK.

| Arm | TRAM | vs zero-shot | TRAM rank |
|---|---|---|---|
| zero-shot | **35.20%** | — | 1 |
| **v1** (plain FT) | **33.48%** | −1.72 | **2 (best FT)** |
| v7c | 32.00% | −3.20 | 3 |
| v4 | 32.01% | −3.19 | 4 |
| v6d | 31.56% | −3.64 | 5 |
| v5 | 31.57% | −3.63 | 6 |
| v7 | 31.40% | −3.80 | 7 |
| v2 | 31.50% | −3.70 | 8 |
| **v6** (best TIME) | 31.15% | −4.05 | 9 |
| **v3-corr** (best TimeBench) | **30.18%** | **−5.02** | 10 (worst) |

**Aggregation warning (2026-09-12).** Note this compounds with the withdrawal
above: the numbers below are the stale ones. The column above is a
**micro**-average,
and TRAM is 57.5% `temporal_nli` plus 20.9% `temporal_relation` — so it is
close to "NLI accuracy", not "temporal reasoning accuracy". Averaging over
the ten tasks instead:

| | zero-shot | v4 | v7c | v6d | v6 | v3-corr |
|---|---|---|---|---|---|---|
| micro (above) | **35.20** | 32.01 | 32.00 | 31.56 | 31.15 | 30.18 |
| macro over 10 tasks | 44.98 | **45.09** | 44.40 | 43.71 | 43.49 | 42.63 |

Reading 1 below holds **only under micro-averaging**; under macro-averaging
**v4 beats zero-shot**. Neither choice is wrong, but the conclusion is a
property of the aggregation, not of the models — state the choice explicitly
wherever this column is quoted. (`macro_pct` is emitted per arm by
`scripts/rescore_v5_protocol.py` as of 2026-09-12.)

Readings (differences are large relative to sampling error, but see §8.1 on
what n does and does not mean here):

1. **No fine-tuned arm beats zero-shot on TRAM** — now n=9 arms, −1.7 to
   −5.0pp. Fine-tuning on Wikipedia temporal QA costs broad temporal QA,
   before any augmentation is added.
2. **The big damage happened in the augmentation era, early**: v1 33.48 →
   v2 31.50 (MCQ/ARITH/NLI wave) → v3-corr 30.18 (rehearsal + more slices).
   The v4–v7c mixture tuning moved TRAM within a ~1pp band (~31.5) while
   TIME swung 38.5–42.2.
3. **The sharpest specialization points coincide**: v3-corr holds BOTH the
   TimeBench record (48.46) and the TRAM floor (30.18); v6 holds the TIME
   record and second-worst TRAM. Specialisation–generalisation trade-off,
   demonstrated twice.
4. v1's mild fine-tune is the best TRAM arm — the mildest intervention
   costs the least off-domain. Dose-responsive at the extremes.

---

## 10. v11 — the prompt-parity arm (2026-09-25)

Protocol: [`hpo_v11_protocol.md`](hpo_v11_protocol.md) (pre-registered). Fixes
from [`audit_2026_09_23.md`](audit_2026_09_23.md): train/eval prompt parity,
pinned template date, decontaminated data, hyperparameters selected on a
held-out dev split. Every number below is on **test-minus-dev** (the 17,500
dev items removed from every arm alike), vLLM, date pinned, paired against
`zs-vllm-pinned` — the base model under the identical prompt and engine.

### 10.1 Selection

11 pre-registered trials (`results/hpo_v11/plan.json`); 10 of 11 beat
zero-shot on dev. Winner **t09**: lr 1.77e-4, 1 epoch, LoRA r=16/alpha=32,
100% of AUG_GLM2 (dev objective +5.48pp; AUG_GLM2's authorship is not
confirmed GLM chat — 15% script-built, most of the rest written by the
GLM-backed coding agent, see `data/manual_aug_glm/PROVENANCE.md`; the
measured gains are unaffected). The anchor t00 (v10's never-searched
lr 4.62e-4 × 3 epochs) is the only loser (−2.00): the old recipe overtrained.

### 10.2 Final verdict — three seeds (2026-09-26, `scripts/summarize_v11.py`)

Pre-registered rule (§9 of the protocol): seed-42 above zero-shot, McNemar
z > 1.96, and the three-seed mean above zero-shot. **All three hold on all
three benchmarks**, and every individual seed is above zero-shot everywhere.

| Benchmark | zs-vllm-pinned | seed 42 | seed 43 | seed 44 | mean ± sd | Δ mean |
|---|---|---|---|---|---|---|
| TIME | 41.15 | 46.80 | 45.95 | 46.51 | 46.42 ± 0.43 | **+5.27** |
| TimeBench | 44.83 | 46.48 | 46.58 | 45.85 | 46.30 ± 0.39 | **+1.48** |
| TRAM | 46.55 | 53.74 | 52.38 | 53.17 | 53.10 ± 0.68 | **+6.54** |

Seed-to-seed sd (0.4–0.7pp) is the noise floor used by `v12_plan.md` §6.
Source: `results/rescored/v11_test_minus_dev.json` + `_verdict.json`.

### 10.2a Seed 42 in detail

| Benchmark | n | zs-vllm-pinned | v11-best | Δ | McNemar z | Δ macro | Δ no-abstain |
|---|---|---|---|---|---|---|---|
| TIME | 99,939 | 41.15 | 46.80 | **+5.65** | +39.05 | +5.33 | **+5.05** |
| TimeBench | 18,575 | 44.83 | 46.48 | **+1.65** | +4.21 | +0.67 | +1.65 |
| TRAM | 970,918 | 46.55 | 53.74 | **+7.19** | +138.5 | **+4.99** | +7.19 |

TIME gains hold in every retrieval setting (no-abstain Δ: base +5.06, bm25
+5.52, hybrid +5.04, vector +4.56) — the gold-vs-retrieved sign flip that
cancelled every earlier arm (§1.0) is gone. Read TIME via the no-abstain
column: ~0.6pp of the headline comes from the abstain-option bucket (§1.1;
v11 85.4% vs zero-shot 54.4% on those 2,427 items). TRAM significance is
overstated by the MCQ/SAQ NLI duplication; its macro (+4.99) is the safer
effect size.

**Not a scorer artifact.** Zero-shot leans on the answer-extraction rescue
rules far more than v11 does (rule credit: TIME 0.98 vs 0.23pp, TimeBench
10.95 vs 0.02, TRAM 27.49 vs 0.11). Scored strictly on the model's first
line, with no rules, the v11 lead widens: +6.42 / +12.93 / +34.55.

### 10.3 Where v11 is WORSE than zero-shot (real regressions)

| Category | zs | v11 | Δ | Mechanism (measured) |
|---|---|---|---|---|
| TimeBench temporal_dialogue | 78.18 | 59.73 | −18.45 | on-list-but-wrong 31.4% vs 22.3%; 8.4% off-list, largely paraphrases ("48 hours" vs option "forty-eight hours") |
| TRAM storytelling | 76.67 | 65.36 | −11.31 | on-list-but-wrong 34.4% vs 23.1% — a choice error, not format. Likely cause: the AUG_GLM2 storytelling card teaches "the wrong ending contradicts a stated date", while TRAM's endings are commonsense plausibility (ROCStories) — a different decision rule |
| TimeBench duration | 77.80 | 70.06 | −7.74 | on-list-but-wrong 25.5% vs 20.9%; 3.9% off-list incl. learned abstention |
| TIME Extract | 9.59 | 6.95 | −2.64 | multi-select golds ("B  C"); no training analogue (audit §11) |

Largest gains: TIME Computation +30.63, TIME Counterfactual +10.82, TRAM
ambiguity +13.49, TRAM NLI +10.21, TimeBench temporal_qa +12.78.

### 10.4 Versus published GPT results (NOT a like-for-like comparison)

GPT was not run under this protocol. The numbers below are the benchmark
papers' own, with different prompts, item samples and metrics — they bound
the picture, they do not support a "beats GPT" or "loses to GPT by X" claim.

| Benchmark | Published (paper's protocol) | v11-best (this protocol) |
|---|---|---|
| TRAM (macro over 10 tasks; paper samples 300/task) | GPT-4 zero-shot 80.1, 5-shot CoT 84.4; GPT-3.5 zero-shot 69.4; Llama-2-70B 61.4; human 95.2 | 52.44 macro |
| TimeBench (paper mixes accuracy and option-level EM/F1) | GPT-4 zero-shot 68.3; GPT-3.5 57.4; LLaMA2-70B 44.1, 13B 42.6, 7B 34.3; human 91.5 (few-shot) | 46.48 EM (45.78 macro) |
| TIME (paper: GPT-4o zero-shot on TIME-Lite, F1) | GPT-4o mean of 11 task scores ≈ 60.3 | 45.16 macro EM on full TIME |

Reading: on TRAM and TimeBench v11 is well below GPT-4 and GPT-3.5, and the
gap (≈20–28pp) is too large for metric differences to plausibly close. On
TimeBench it sits numerically above the paper's LLaMA2-70B zero-shot figure,
but the metrics differ. On TIME, per-task results are mixed against GPT-4o
(nominally higher on Computation, Order_Reasoning, Co_temporality,
Counterfactual, Timeline; much lower on Localization, Order_Compare,
Explicit_Reasoning, Duration_Compare, Extract) — but TIME-Lite is a different,
manually verified subset scored with F1, so no ranking is claimed. A valid
comparison needs GPT run on the same items with the same prompt and scorer
(e.g. a stratified sample of test-minus-dev), or v11 evaluated on TIME-Lite
with the paper's F1. Sources: TRAM arXiv:2310.00835, TimeBench
arXiv:2311.17667, TIME arXiv:2505.12891 (PDFs also in `/home/g2/Thesis/Papers`).
The thesis proposal's "GPT-4 ~75% on TIME-News" is a planning estimate, not a
measured figure.

### 10.5 Known protocol weakness (both arms alike)

56.8% of TIME prompts are longer than the 2,048-token training window, so v11
is evaluated on context lengths it never trained on; ~4.9% exceed the
4,096-token eval window and are left-truncated, which cuts the instruction
lines and chat header. Identical for both arms, so the comparison is fair,
but a next cycle should train at a longer max_seq_length or truncate the
context rather than the whole prompt.

---

## 11. v12 — three single-variable arms vs v11-best (2026-09-28)

Protocol: [`v12_plan.md`](v12_plan.md) (pre-registered). Three single-variable
arms, each changing exactly one thing from its control, same v11 protocol
otherwise (test-minus-dev, vLLM, pinned date `26 Jul 2024`, paired against
`zs-vllm-pinned`):

- **v11-2ep** — v11's own data/recipe, 2 epochs instead of 1. Control: v11-best.
- **v12-data** — t09 recipe, corrected storytelling/relation/ordering/
  temporal_dialogue/duration cards (the arms chasing v11's §10.3 regressions).
  Control: v11-best.
- **v12-ctx** — v12-data + `max_seq_length` 2048→4096. Control: v12-data.

### 11.1 Versus zero-shot — all three win everywhere

| Benchmark | zs-vllm-pinned | v11-best | v11-2ep | v12-data | v12-ctx |
|---|---|---|---|---|---|
| TIME | 41.15 | 46.80 (+5.65) | 46.12 (+4.97) | 46.31 (+5.16) | 46.04 (+4.89) |
| TimeBench | 44.83 | 46.48 (+1.65) | 49.27 (+4.44) | 46.93 (+2.10) | 47.77 (+2.94) |
| TRAM | 46.55 | 53.74 (+7.19) | 55.39 (+8.83) | 54.05 (+7.49) | 54.17 (+7.61) |

McNemar z on every cell exceeds +34 (TIME), +4.2 (TimeBench) and +138
(TRAM) — all four arms clear zero-shot by a wide margin, same as v11.
Source: `results/rescored/v12_test_minus_dev.json`.

### 11.2 Versus each arm's own control (corrected 2026-10-04, `audit_2026_10_04.md` §5.1)

Beating zero-shot was already true of v11-best; the question v12 asks is
whether each change beats its *own* control past v11's seed noise.

**The control is the v11 three-seed mean (46.42 / 46.30 / 53.10), not seed
42.** Seed 42 is the HPO winner reused directly and the best of the three
seeds on TIME and TRAM, so comparing against it alone is biased against
every v12 arm. A first version of this section did exactly that and
reported a "real TIME loss" for v11-2ep that is an artifact of the lucky
seed. Noise for a single run vs a 3-seed mean ≈ sd·√(4/3): 0.50 / 0.45 /
0.79pp; for single run vs single run ≈ sd·√2: 0.61 / 0.55 / 0.96pp.

| Arm vs control | TIME | TimeBench | TRAM |
|---|---|---|---|
| v11-2ep vs v11 mean | −0.30 (noise) | **+2.97** | **+2.29** |
| v12-data vs v11 mean | −0.11 (noise) | +0.63 (marginal) | +0.95 (marginal) |
| v12-ctx vs v12-data | −0.27 (noise) | +0.84 (marginal) | +0.12 (noise) |

- **v11-2ep is a clean win**: real TimeBench and TRAM gains, TIME neutral
  (46.12 sits inside the v11 seed range 45.95–46.80).
- **v12-data**: the targeted TRAM categories moved the right way
  (storytelling −11.3→−8.1pp vs zero-shot, ordering −0.5→+1.9pp), pooled
  gains are marginal (≈1.2–1.4× noise), TIME neutral.
- **v12-ctx**: a marginal further TimeBench gain over v12-data, nothing on
  TIME or TRAM.
- No arm moves TIME; the epoch change is the only clearly effective one.

### 11.3 Known caveat

v12 ran one seed per arm (seed 42, like v1–v10) and borrows v11's
three-seed noise floor rather than measuring its own — the "past noise" /
"within noise" calls above assume v12's run-to-run variance resembles
v11's. Not independently verified.

---

## 12. Mistral-7B-Instruct-v0.3 — same protocol, second model (2026-10-02)

Protocol: [`mistral_plan.md`](mistral_plan.md). The v11 audit fixes (prompt
parity via `eval_parity`, the `seed=` Trainer fix, external dev-selected
checkpoints) ported to `experiments/finetuning/Mistral/`; no date-pin fix
needed — Mistral-7B-Instruct-v0.3's chat template carries no date field at
all, so audit §2 is inapplicable rather than unfixed.

**The standing "Mistral banned from vLLM" restriction (D46/D53) was
resolved, not worked around.** The original 2026-09-03 parity check
compared vLLM's zero-shot output against a *different* (fine-tuned)
model's HF predictions — a broken `--reference` path in
`logs/vllm_parity_chain.sh`, not an engine bug (matching D57's 2026-09-09
leading theory, never GPU-confirmed until now). Re-checked against the
correct reference: 95.18% agreement, −0.051pp delta — engine-consistent.
`zs-vllm-mistral` and `mistral-best` are now first-class arms in
`rescore_v5_protocol.py`.

### 12.1 HPO-lite selection (dev split, 6 pre-registered trials)

Reduced-scope HPO vs v11's 11 trials (no `aug_fraction` knob, fixed seed
20260928, anchor = LLaMA's t09 hyperparameters transplanted):

| id | lr | epochs | r | objective ± SE | ΔTIME | ΔTimeBench | ΔTRAM | Δmacro |
|---|---|---|---|---|---|---|---|---|
| **m05 (winner)** | 3.72e-05 | 1 | 16 | **+7.84** ± 0.42 | +7.82 | +10.88 | +4.81 | +9.51 |
| m02 | 2.18e-04 | 1 | 8 | +7.78 ± 0.44 | +9.84 | +8.36 | +5.13 | +8.89 |
| m00 (anchor) | 1.77e-04 | 1 | 16 | +5.40 ± 0.45 | +8.10 | +4.68 | +3.41 | +7.31 |
| m01 | 1.47e-04 | 2 | 32 | +5.29 ± 0.45 | +6.42 | +6.08 | +3.36 | +5.67 |
| m03 | 4.07e-04 | 2 | 16 | −5.05 ± 0.49 | −0.08 | −9.24 | −5.82 | −4.31 |
| m04 | 3.57e-04 | 1 | 32 | −5.26 ± 0.49 | −4.32 | −2.96 | −8.50 | −4.25 |

**m05 and m02 are within 2 SE of each other on dev** — the ranking between
them is not resolved by this dev set; m05 was taken as the winner per the
pre-registered argmax rule, not because the difference is established.
Notably, m05's learning rate (3.72e-5) is ~5x *lower* than LLaMA's winning
t09 (1.77e-4) — confirms the decision to re-search Mistral's hyperparameters
from scratch rather than transplant LLaMA's was the right call; the anchor
(m00, LLaMA's recipe verbatim) placed third. Source: `results/hpo_mistral/`.

### 12.2 Final verdict — one seed (test-minus-dev)

The winning trial's own adapter was reused directly for the full-test eval
(no retrain — same trick `hpo_v11.py`'s final-config step uses).

| Benchmark | n | zs-vllm-mistral | mistral-best | Δ | McNemar z | Δ macro | Δ no-abstain |
|---|---|---|---|---|---|---|---|
| TIME | 99,939 | 37.25 | 45.34 | **+8.09** | +52.07 | +7.38 | +7.82 |
| TimeBench | 18,575 | 45.40 | 55.84 | **+10.44** | +28.72 | +12.10 | +10.44 |
| TRAM | 970,918 | 52.13 | 56.55 | **+4.42** | +96.32 | +7.32 | +4.42 |

Beats zero-shot on all three benchmarks, by a wide McNemar margin on
every one — the same verdict pattern v11 established for LLaMA.
Source: `results/rescored/mistral_test_minus_dev.json`.

### 12.2a Three seeds (2026-10-05)

Seeds 43/44 of the exact m05 recipe (seed the only variable;
`scripts/run_mistral_seeds.sh`). Same pre-registered rule as v11 (§10.2):
every seed above zero-shot, McNemar z > 1.96, mean above zero-shot —
**all three hold on all three benchmarks.**

| Benchmark | zs-vllm-mistral | seed 42 | seed 43 | seed 44 | mean ± sd | Δ mean |
|---|---|---|---|---|---|---|
| TIME | 37.25 | 45.34 | 45.88 | 46.11 | 45.78 ± 0.40 | **+8.53** |
| TimeBench | 45.40 | 55.84 | 55.40 | 56.19 | 55.81 ± 0.40 | **+10.41** |
| TRAM | 52.13 | 56.55 | 57.14 | 56.52 | 56.73 ± 0.35 | **+4.61** |

McNemar z per seed: TIME +52.1 / +55.7 / +57.7, TimeBench +28.7 / +28.0 /
+29.0, TRAM +96.3 / +113.8 / +98.2. Seed sd (0.35–0.40pp) is Mistral's own
noise floor, slightly tighter than LLaMA's (0.4–0.7pp). Seed 42 — the HPO
winner, quoted alone until now — sits within 0.5pp of the mean everywhere,
so unlike v11 it was not a notably lucky draw.

### 12.3 Not a LLaMA-vs-Mistral comparison

Mistral's own zero-shot differs from LLaMA's zero-shot in *both*
directions across benchmarks (TIME 37.25 vs 41.15, TimeBench 45.40 vs
44.83, TRAM 52.13 vs 46.55) — different base models internalize the same
prompt differently, and no shared item-level pairing was computed between
the two models' predictions. This result licenses only a model-internal
claim: the identical fine-tuning methodology that helped LLaMA also helps
Mistral. It does not license any claim about which base model is better,
or by how much.

### 12.4 Known caveats

- **Reading the results file.** Read Mistral arms only from
  `results/rescored/mistral_test_minus_dev.json` — that file pairs *every*
  arm, LLaMA included, against Mistral's zero-shot.
- **QLoRA→fp16.** Adapters are trained against the 4-bit NF4 base and
  evaluated merged into the fp16 base (consistently for HPO, every seed, and
  the fp16 zero-shot), so the deltas are valid but measure "fp16 base +
  QLoRA adapter", not the trained 4-bit model.
