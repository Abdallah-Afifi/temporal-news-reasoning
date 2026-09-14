# v7 — the MCQ / format arm

Written 2026-09-07 (D42). **Design provisional until v6's
`format_probe.report` lands** — see § 6 for the one number that confirms it.

**Goal:** beat zero-shot on TIME (41.46%). Target **43.3%**.

> ## MANDATORY in v7 — fix the training target, not the scorer
>
> `AUG_MCQ`/`AUG_MCQ2` carry a dual gold `[text, LETTER]`, which the
> finetuning loader joins into the training target **`"<option text>; B"`**.
> That is why every fine-tuned arm answers in a shape the benchmark does not
> expect, and why the scorer needed a trailing-letter rule at all (D39).
>
> **The objection this invites is fair:** *you trained a nonstandard output
> format, then widened the scorer to accept it.* It applies only to the letter
> rule — date equivalence is unimpeachable (`audit_2026_09_07.md` § 13).
>
> **v7 must remove it at the source: give `AUG_MCQ`/`AUG_MCQ2` a SINGLE
> target — option text only.** One line in
> `build_v6_training_data.py::build_noans_and_mcq`:
> `"targets": [correct, letter]` -> `"targets": [correct]`.
>
> **Pre-registered check:** count `_TRAILING_LETTER_RE` firings on v7's stored
> predictions. **~0 firings + a good score = the result provably does not
> depend on the scoring change.** Frequent firings = the mixture still teaches
> the shape and the objection stands.
>
> **Risk, stated in advance:** v5's answerable-MCQ bucket (64.49%, best in the
> project) was achieved *with* the dual-gold format, so this is a real
> variable and may cost some of it. Worth paying for a result nobody can wave
> away. Report both columns regardless.

---

## 1. Why this arm, in one table

What one percentage point on each bucket is worth to the overall TIME score:

| bucket | n | share | 1pp there = | v6 aimed at it? |
|---|---|---|---|---|
| **MCQ answerable** | 46,937 | 44.7% | **0.45pp** | no — and its rehearsal cap RISKS it |
| free text | 42,428 | 40.4% | 0.40pp | partly (long-form restore) |
| letter sequence | 11,361 | 10.8% | 0.11pp | yes (`AUG_SEQ`) |
| MCQ no-answer | 2,427 | 2.3% | **0.02pp** | yes (`AUG_NOANS`) |

v6 spent two of three variables on the two lowest-leverage buckets. v7 fixes
the allocation. **And fine-tuning already beats the base model on the big
bucket** — v5 scores **64.50%** on answerable MCQ against zero-shot's 63.56%
— so this is a matter of scaling what works, not of overcoming a base-model
ability.

## 2. The mixture

| slice | v5 | v6 | **v7** | why |
|---|---|---|---|---|
| `REHEARSAL` (context-grounded) | 8,026 (44.2%) | 2,600 (14.0%) | **~8,000** | MCQ accuracy tracks it: v3c ~0% -> 62.20%, v4 20.9% -> 62.93%, v5 44.2% -> **64.50%**. DROP/HotpotQA/CoQA are read-a-passage-answer-briefly tasks — the closest match in the pool to TIME's MCQ format (D41). |
| `AUG_MCQ` + `AUG_MCQ2` | 400 | 1,600 | **~6,000** | The benchmark is **60.7% multiple-choice; the training core is 0%**. This is the untested lever (D41). |
| `AUG_SEQ` | 0 | 1,273 | **1,273** | Keep v6's fix — it costs nothing and buys emission. |
| `AUG_NOANS` | 0 | 500 | **500** | Keep, but do not grow: 0.02pp per point. |
| core `TimeQA`+`TLQA` | 10,269 | 10,269 | 10,269 | Unchanged, as in every cycle. |

**v7 = v5's rehearsal + v6's format slices + `AUG_MCQ` scaled ~15x.**

```bash
./venv/bin/python scripts/build_v6_training_data.py \
    --out data/combined_80_20_v7 \
    --rehearsal 8000 --rehearsal-long-frac 0.15 \
    --aug-mcq2 6000 --aug-noans 500 --aug-seq 1500
```
The generator already exists and already samples **type-plausible
distractors** within each question's type bucket. Check the run's manifest:
`AUG_MCQ2` must actually reach ~6,000 (TimeQA has 4,963 core rows but the
generator draws from the whole pool, so verify rather than assume).

## 3. Expected outcome

| scenario | TIME | vs zero-shot |
|---|---|---|
| MCQ merely holds v5's 64.50% | 41.55% | **+0.06pp** |
| MCQ 66.5% | 42.45% | **+0.96pp** |
| MCQ 68.0% (target) | **43.29%** | **+1.80pp** |

**The key property: v7 beats zero-shot even if the MCQ scale-up adds
nothing**, purely by not sacrificing the 0.45pp/pp bucket to buy the
0.11pp/pp one. That is a much better risk profile than v6's ~35-40%.

## 3b. REQUIRED in v7 — single-target MCQ, to remove the methodological objection

`AUG_MCQ`/`AUG_MCQ2` currently carry a dual gold `[text, LETTER]`, which the
finetuning loader joins into the training target `"<option text>; B"`. That is
why every fine-tuned arm answers in a shape the benchmark does not expect, and
why the scorer needed a trailing-letter rule at all (D39).

**The objection this invites is fair:** *you trained a nonstandard output
format, then widened the scorer to accept it.* It applies only to the letter
rule, not to date equivalence — see `docs/audit_2026_09_07.md` § 13.

**v7 must remove it at the source: give `AUG_MCQ`/`AUG_MCQ2` a SINGLE target
(option text only).** The model then answers in the benchmark's own form.

**Pre-registered check:** count `_TRAILING_LETTER_RE` firings on v7's stored
predictions. **~0 firings + a good score = the result provably does not depend
on the scoring change.** Frequent firings = the mixture still teaches the
shape and the objection stands.

**Risk to weigh:** v5's answerable-MCQ bucket (64.49%, best in the project)
was achieved *with* the dual-gold format, so this is a real variable and could
cost some of it. It is worth paying for a result nobody can wave away — but
report both columns regardless.

## 4. Honest limits

- **v7 fixes FORMAT, not DOMAIN.** `AUG_MCQ` is generated from TimeQA, i.e.
  **Wikipedia**. TIME is **news**. So do not expect `Counterfactual` (9,720
  items, fine-tuning loses 6-10pp) or `Computation` (9,372) to move. The
  domain fix is v8.
- **Over-scaling risk.** At 6,000 rows `AUG_MCQ2` becomes ~25% of the
  mixture. Watch for the model answering MCQ-style on free-text items; the
  format probe's `free text` row is the tripwire.
- **Three variables again** (rehearsal restore, MCQ scale, keep v6 slices),
  but as in v6 they act on largely disjoint buckets, so the format probe
  attributes them.

## 5. Budget

~18,000-24,000 train rows, so ~8-10 h training plus ~7.5 h eval ≈ **16-18 h**.
Reuse `run_queue_v6.sh` with the paths changed, and keep
`probe_answer_formats.py` as stage 4b.

## 6. The one number that confirms this design

From v6's `format_probe.report`: **if `MCQ: answerable` returns 62-63%
instead of v5's 64.50%**, D41's rehearsal correlation is causal, the cap was
the mistake, and v7 as specified is right. If instead it *holds* at 64.5%,
rehearsal volume is not what drives MCQ — in that case keep v6's cap (it is
cheaper to train) and make v7 purely the `AUG_MCQ` scale-up.


---

## 7. BUILT — 2026-09-08, ready to train

`data/combined_80_20_v7/` — **26,570 train / 6,636 val**. Built by
`scripts/build_v7_training_data.py` (defaults are the values used).

### 7a. COVERAGE — the pattern that justified adding three more slices

v6 produced an **exceptionless** result: every TIME category with a dedicated
training slice beat zero-shot, and every category without one lost.

| | categories | v6 vs zero-shot |
|---|---|---|
| **has a slice** | Counterfactual (AUG_GLM, **502 rows**), Order_Compare, Localization, Timeline | **all 4 WIN**: +13.1, +7.6, +5.3, +1.4 |
| **uncovered / thin** | Duration_Compare, Computation, Relative_Reasoning, Extract, Co_temporality | **all 5 LOSE**: −7.3, −6.3, −2.1, −1.9, −1.8 |

The uncovered categories are **22,822 items (21.7% of TIME) costing −0.91pp**,
plus Computation's **−0.56pp** — together **−1.47pp, twice v6's entire
margin**. `AUG_GLM` moving Counterfactual **+13.1pp on 502 rows** is the
evidence that small targeted slices work.

| new slice | rows | targets | v6's gap there |
|---|---|---|---|
| `AUG_ARITH` (scaled 400 -> 2,899) | +2,500 | Computation (9,372) | **−6.3pp** |
| `AUG_DURATION` (new) | 1,966 | Duration_Compare (9,217) | **−7.3pp** |
| `AUG_RELATIVE` (new) | 524 | Relative_Reasoning (10,265) | −2.1pp |

`AUG_DURATION` asks "which of these lasted longer?" from TLQA year spans;
`AUG_RELATIVE` asks "immediately after X, which came next?" from TLQA
timelines with >= 3 distinct start years. Both are pool-built, never
benchmark-derived. `AUG_ARITH` is fully synthetic (`make_arithmetic`), so it
carries no leakage risk at all — and TIME's Computation items are **0.8%
extractable**, i.e. they must be *computed*, which is exactly what it teaches.

**`AUG_RELATIVE` delivered 524 of 2,000 requested** — TLQA has a limited
number of timelines with 3+ distinct start years plus 2 spare distractors. The
builder reports the shortfall rather than padding it.

**Deliberately left uncovered:** `Extract` (3,340 items, −0.06pp — lowest
leverage in the benchmark) and `Co_temporality`, whose only slice `AUG_NLI` is
the degenerate one; fixing that is a separate change.

**Gold correctness verified independently:** `AUG_DURATION` **0 incorrect** of
1,966 (gold is always the longest span); `AUG_RELATIVE` 524/524 on all three
checks (gold among options, 3 distinct options, anchor never equals gold);
`AUG_ARITH` **0 incorrect** across the 1,158 rows that could be recomputed
from the question text.

| slice | v5 | v6 | **v7** |
|---|---|---|---|
| REHEARSAL (of which context-grounded) | 8,026 (8,026) | 2,600 (1,300) | **7,999 (6,800)** |
| AUG_MCQ + AUG_MCQ2 | 500 | 1,600 | **6,255** |
| AUG_SEQ | 0 | 1,273 | 1,273 |
| AUG_NOANS | 0 | 500 | 500 |
| **AUG_ARITH** | 400 | 400 | **2,899** |
| **AUG_DURATION** | 0 | 0 | **1,966** |
| **AUG_RELATIVE** | 0 | 0 | **524** |
| core TimeQA + TLQA | 10,269 | 10,269 | 10,269 |

### The three fixes, each verified on the built data

| fix | check | result |
|---|---|---|
| **D45 single-target MCQ** | MCQ-family rows with a non-single target | **0** |
| | training targets ending in `"; <LETTER>"` | **0** |
| | loader output | `'Queens Park Rangers'`, `'There is no answer.'`, `'C,B,A'` |
| **abstain shortcut broken** | `P(correct \| option is abstain-phrased)` | **49.3%** (v6: 100.0%) |
| **rehearsal restored** | context-grounded rows | **6,800** (v6: 1,300) |

**The abstain fraction was TUNED, not guessed.** The first attempt at
`--noans-distractor-frac 0.25` gave 25.5%, which merely **inverts** the cue —
the model would learn "abstain usually wrong" and under-abstain on TIME, where
an offered abstain option is always correct. **0.09 lands at 49.3%: the
phrasing carries no information and the model must read the context.**

Also verified: AUG_SEQ 1,273 rows with **0 incorrect golds**; **0 rows dropped
by the finetuning loader**; guards passed against **TIME + TimeBench + TRAM**
(0 leaks, 0 collisions, 99 dupes and 39 near-dupes purged);
`config_v7.yaml` differs from v6's in exactly `train_data`, `val_data`,
`output_dir`. 75 tests pass.

### Expect the no-answer bucket to FALL, deliberately

v6 scored **99.88%** there by pattern-matching. v7 cannot, by construction.
That costs up to **−1.09pp** and must be paid for by the MCQ restoration
(+1.14pp expected) and the AUG_MCQ scale-up. **Report v6 and v7 side by side
so the artifact's size stays visible.**

### Run it

```bash
cd /home/g02-s26/Mohamed/temporal-news-reasoning
mkdir -p logs/queue_v7          # BEFORE the redirect opens
nohup bash scripts/run_queue_v7.sh > logs/queue_v7/driver.log 2>&1 &
```

Eval order is **TimeBench -> TRAM (batch 256, full 980,918) -> TIME**
(cheapest first, D47). Budget: **12.2 h train — MEASURED**, not estimated (8.82 s/step on a 60-step
pilot x 4,980 steps; 26,570 rows, +79% vs v6's 14,850). Runs on
`config_v7.yaml`: `gradient_checkpointing: false` OOMs on this card (D54) and
`group_by_length` is unproven and not bit-comparable, so v7 stays
recipe-identical to v1-v6 and needs no methodological caveat. + ~12 min
TimeBench + ~3 h TRAM + ~7.5 h TIME ≈ **26 h**. Real GPU shell only (D32/D33).

**Run `scripts/run_zeroshot_tram.sh` first or in parallel** — without a
zero-shot TRAM baseline v7's TRAM number has nothing to beat (D48).
