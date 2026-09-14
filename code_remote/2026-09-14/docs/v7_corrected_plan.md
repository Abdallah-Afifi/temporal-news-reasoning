# v7-corrected — one variable, on the right base

> **⚠ Superseded in part (2026-09-12).** This document quotes TRAM figures
> that are **withdrawn**: 85.3% of the TRAM column was generated from prompts
> that omitted the question (`BenchmarkLoader._tram_csv_to_example` never read
> the NLI `Hypothesis` or storytelling's `Story`). A re-run is in progress.
> TIME figures here are also pooled over four retrieval `Setting`s that differ
> by ~23pp and in which the fine-tuning effect changes sign. See
> `docs/audit_2026_09_12.md` and session_handoff **D64**. Left unedited
> otherwise — it is the append-only record of what was known at the time.

Written 2026-09-09, after v7's post-mortem (`docs/session_handoff.md` D56).
Status: **planned, not built.**

## Why this arm exists

v7 lost **5.94pp on TIME** against v6 (36.25% vs 42.19%) and **cannot say
why**, because it changed six things at once. v7-corrected does the one thing
v7 was actually mandated to do, on the best available base, as a **single
variable**.

Note the name is an analogy to v3-corrected, not a repeat of it. v3-corrected
fixed *data corruption*. v7 has no corrupted data — its mixture is sound. What
v7 lacks is **attribution**. So v7-corrected is not "v7 with bugs fixed"; it is
**"v7's mandated change, isolated."**

## What v7 got right, and must be preserved

**v7's pre-registered check passed.** `rewritten` (trailing-letter rule
firings) = **0** on both benchmarks, versus 22,870 for v5 on TIME. The
objection v7 existed to remove — *"you trained a nonstandard output format
(`"<option text>; B"`), then widened the scorer to accept it"* — **is
genuinely removed.** That is a real methodological gain and the reason to
retry rather than abandon the idea.

## The evidence v7-corrected is designed around

**1. v7 damaged almost everything, not one slice.** 10 of 11 TIME categories
regressed:

| Category | n | zero-shot | v6 | v7 | v7−v6 |
|---|---|---|---|---|---|
| Order_Compare | 9,897 | 61.1 | 68.7 | 40.4 | **−28.3** |
| Localization | 9,894 | 46.5 | 51.8 | 41.7 | −10.1 |
| Counterfactual | 9,720 | 42.6 | 55.7 | 47.6 | −8.0 |
| Explicit_Reasoning | 9,720 | 58.6 | 57.2 | 51.2 | −6.0 |
| Computation | 9,372 | 18.8 | 12.5 | 6.6 | −5.9 |
| Relative_Reasoning | 10,265 | 47.0 | 44.9 | 40.7 | −4.1 |
| Order_Reasoning | 10,266 | 49.9 | 48.8 | 44.9 | −3.9 |
| Co_temporality | 10,089 | 50.7 | 48.9 | 46.4 | −2.5 |
| Timeline | 13,171 | 16.5 | 17.9 | 17.5 | −0.3 |
| Extract | 3,340 | 9.6 | 7.7 | 6.6 | −1.1 |
| **Duration_Compare** | 9,217 | 40.4 | 33.1 | **40.1** | **+7.0** |

**2. Every scale-up failed at its own stated target.** This is the decisive
point — the slices were grown to buy specific categories, and did not:

| v7 change | intended target | actual result |
|---|---|---|
| `AUG_ARITH` 320 → 2,320 (7.3x) | arithmetic | TimeBench arithmetic **26.9 → 20.7 (−6.2)** |
| `AUG_MCQ2` 1,200 → 4,604 (3.8x) | MCQ buckets | Order_Compare **68.7 → 40.4** |
| `REHEARSAL` 2,077 → 6,400 (3.1x) | MCQ via context-grounding | Localization −10.1, Counterfactual −8.0 |
| `AUG_DURATION` 0 → 1,573 (new) | duration | TimeBench duration −0.3, but **TIME Duration_Compare +7.0** |
| `AUG_RELATIVE` 0 → 420 (new) | relative reasoning | Relative_Reasoning **−4.1** |
| dual-gold removed | scorer objection | **worked — 0 rewrites** |

Only `AUG_DURATION` bought anything, and only on TIME.

**3. A systemic confound sits under all of it.** The recipe is bit-identical
between v6 and v7 (same LR 4.62e-04, 3 epochs, 2x8, LoRA r16/α32), but the
mixture is not:

| | v6 | v7 |
|---|---|---|
| rows | 14,850 | 26,570 (+79%) |
| synthetic share | 30.7% | **45.0%** |
| optimizer steps | ~2,787 | **~4,983 (1.79x)** |

v7 took **79% more gradient updates at the same peak LR**, on a mixture
**half again as synthetic**. Near-uniform degradation across ten categories is
the signature of over-training / distribution drift away from the base model —
not of any one slice being wrong. **Any v7 retry must therefore control
mixture size, not just composition.**

## The design

**v7-corrected = v6's mixture, with every dual-gold target changed from
`[text, LETTER]` to `[text]`. Nothing else.**

**Correction to an earlier draft of this plan (2026-09-09):** the change
covers **three** slices, not two. `AUG_NOANS` (400 train / 100 val) is built
by the same generator as `AUG_MCQ2` and carries the same dual gold
(`['Cannot be determined from the context.', 'A']`). Leaving it would keep
teaching the `"<text>; <LETTER>"` shape and would undermine the arm's whole
purpose — the pre-registered 0-rewrites check could fail on a shape learned
from `AUG_NOANS` alone. **v7 stripped all three as well**, so this also keeps
v7-corrected directly comparable to v7. Rows changed: 2,000 train
(`AUG_MCQ` 400 + `AUG_MCQ2` 1,200 + `AUG_NOANS` 400) and 499 val.

| Property | value | rationale |
|---|---|---|
| Base mixture | **v6's, unchanged** (14,850 rows) | v6 is the only arm that beats zero-shot on TIME (42.19% vs 41.46%) |
| Slice sizes | **identical to v6** | holds rows, synthetic share (30.7%) and step count (~2,787) constant, removing the confound above |
| The one change | `AUG_MCQ`/`AUG_MCQ2`: `"targets": [correct, letter]` → `"targets": [correct]` | the single methodologically-mandated change, per v7_plan.md's MANDATORY block |
| Config | **`config_v6.yaml` verbatim**, new `output_dir` only | keeps the recipe bit-identical to v1-v6 |
| Control | **v6** | same mixture, same size, same recipe — the delta is the dual gold and nothing else |

Implementation is one line in the builder, exactly as v7_plan.md specified —
applied to v6's builder rather than to v7's expanded mixture.

## Pre-registered success criteria

Decide these now, not after seeing the numbers.

| Outcome | Reading |
|---|---|
| TIME ≈ 42.19% (within noise) **and** `rewritten` = 0 | **Best case.** The scorer objection is removed for free; v7-corrected supersedes v6 as the headline arm. |
| TIME meaningfully < 42.19%, `rewritten` = 0 | The dual gold was load-bearing. We now know its exact price, which is itself a publishable, honest finding. Report both arms. |
| TIME ≈ 42.19% but `rewritten` > 0 | The mixture still teaches the `"text; LETTER"` shape from elsewhere — find the source before claiming the objection is gone. |
| TIME collapses like v7 | The dual-gold removal alone caused v7's regression, and the five scale-ups were incidental. Strong, clean result. |

Primary metric: **TIME**, against v6 as control and zero-shot (41.46%) as
baseline. Report TimeBench and the `Order_Compare` distribution alongside —
v7's failure was visible in the emitted-answer distribution before it was
visible in the score.

**Also pre-register the distribution check.** For `Order_Compare`, report the
share of predictions on each option against the gold prior
(`"Fact 1 happened earlier."` = 56.9% of golds). Any arm emitting a rare
option at >20% is collapsing, whatever its headline score says. Note that **v6
itself emits the majority class 69.6% of the time** — above the 56.9% base
rate — so part of v6's advantage is prior-exploitation, and v7-corrected
should be read with that in mind rather than treated as a pure reasoning
comparison.

## Explicitly out of scope

Not bundled in, so the arm stays single-variable:

- **The five scale-ups.** All are reverted to v6 levels. Four of them
  measurably hurt; none met its target.
- **`AUG_DURATION`.** The one v7 change that paid (+7.0 on Duration_Compare,
  9,217 items ≈ +0.6pp overall). Genuinely promising — and therefore worth
  its **own** single-variable arm, not a passenger in this one.
- **Anything targeting Computation** (18.8% zero-shot → 12.5% v6 → 6.6% v7),
  still the worst unexplained deficit in the project.
- **The mixture-size/step-count hypothesis.** If v7-corrected lands near v6,
  that supports size/over-training as v7's real cause; testing it properly
  needs its own arm (v6's mixture scaled up with composition held constant).

## Sequencing

1. **v7-corrected** — this plan.
2. **v6 + `AUG_DURATION`** — the one salvageable v7 idea, isolated.
3. **TRAM for v6** (and v7's one missing item) — the best arm still has no
   TRAM number, and TRAM now has a 35.20% baseline to beat.

Do **not** build v8 on v7. v6 is the base.
