# v6 ("v5-corrected") — plan, rationale and honest limits

Written 2026-09-07. Supersedes the "Next after v5" ranking in
`docs/session_state.md`, which was drawn up before the trailing-letter
scoring fix and before the abstention bucket was found.

**Goal, set by the user:** beat zero-shot on TIME, without training on
benchmark data and without violating the project's methodology.

**Naming (D40).** `v6` had been reserved for the CoT/STaR cycle in the v4/v5
planning docs. This cycle took the name before that was noticed, and by then
it was already training with the name baked into `config_v6.yaml`,
`data/combined_80_20_v6/`, `checkpoints/llama_v6/` and `logs/queue_v6/`.
**`v6` is therefore this corrected-format cycle, and the CoT cycle is `v7`.**

---

## 1. What changed in the diagnosis (and why the old plan was wrong)

Two findings on 2026-09-07 rewrote the problem.

**a. The MCQ "collapse" was a scoring artifact, not a regression.**
`AUG_MCQ` targets are dual-gold lists — `['Christopher Quinten', 'D']` — and
the finetuning `data_loader` joins a target list with `"; "`. So the training
target is literally **`"Christopher Quinten; D"`**, and v5 faithfully answers
`"<option text>; <LETTER>"` on 26,192 TIME MCQ items. The scorer could not
read that shape, so a correct answer scored zero.

`_postprocess_prediction` now reads it (see § 2). Under identical scoring for
every arm:

| arm | TIME before | TIME after |
|---|---|---|
| zero-shot | 41.49% | **41.49%** (emits the shape 0 times — gains nothing) |
| v1-ft | 33.46% | 33.46% |
| v2-ft | 32.70% | **37.59%** |
| v3-corrected | 38.68% | **40.05%** |
| v4 | 37.63% | **40.13%** |
| v5 | 36.62% | **38.45%** |

**b. With that fixed, v5 already beats zero-shot where most of TIME lives.**

| gold shape | n | share | zero-shot | v5 | costs v5 |
|---|---|---|---|---|---|
| MCQ answerable | 46,937 | 44.7% | 63.56% | **64.50%** | **+0.42pp** |
| MCQ "no answer" | 2,427 | 2.3% | 52.95% | 0.74% | **−1.21pp** |
| letter sequence | 11,361 | 10.8% | 19.05% | 0.00% | **−2.06pp** |
| free text | 42,428 | 40.4% | 24.18% | 23.71% | −0.19pp |
| paren permutation | 1,798 | 1.7% | 0.22% | 0.22% | +0.00pp |
| **OVERALL** | 104,951 | | **41.49%** | **38.45%** | **−3.03pp** |

**The entire remaining deficit is two output behaviours the model stopped
producing.** Neither is a reasoning deficit. Fine-tuning has been *improving*
temporal MCQ accuracy all along; it was hidden under a format mismatch.

---

## 2. Change already made and verified (no GPU)

**Trailing-letter rule in `scripts/run_baselines._postprocess_prediction`.**
Purely additive: it runs only where the function previously gave up and
returned the raw line, so it cannot alter any decision the existing rules
already made. It retries the text half on its own first, and reads the letter
only if the text resolves to nothing — so a resolvable text answer always
wins, exactly as elsewhere in that function.

- 7 new tests in `tests/test_prediction_postprocessing.py` (35 pass in that
  file plus `test_date_equivalence.py`; 68 pass repo-wide — the 4
  `heideltime` failures are the pre-existing Java-subprocess permission
  issue, unrelated).
- Applied to **every** arm by `scripts/rescore_v5_protocol.py`, as protocol
  requires. Zero-shot and v1 gain nothing because they never emit the shape.

**Disclosure the thesis must carry:** this rule benefits fine-tuned arms and
not zero-shot. That is not a thumb on the scale — it corrects a measurement
error created by the project's own training-target format, and the same code
scores every arm. It must be reported as a scoring-protocol change with its
per-arm effect shown, exactly as the table in § 1a shows it.

---

## 3. What v6 changes in the data (three variables)

Built by `scripts/build_v6_training_data.py` into `data/combined_80_20_v6/`
(14,850 train / 3,708 val).

| # | Change | Rows | Targets which bucket |
|---|---|---|---|
| 1 | REHEARSAL capped 8,026 → **2,600** (44.2% → 14.0%), **half long-form dolly** | −5,426 | letter sequence, free text |
| 2 | **AUG_SEQ** — comma-separated-letter ordering, new | +1,273 | letter sequence |
| 3 | **AUG_NOANS** +500 with an **AUG_MCQ2** +1,500 answerable counterweight | +2,000 | MCQ "no answer" |

**Why (1) should restore sequence emission.** D38 established the collapse is
dilution, not erosion: emission ran 99.97% (zs) → 2.64% (v2) → **84.92%
(v3-corrected)** → 62.01% (v4) → 0.01% (v5), and **no mixture has ever
contained a sequence-shaped target — v3-corrected included**. What tracks
emission is long/multi-item supervision surviving in the mixture. TLQA's
3,253 rows at ~19.9w were never removed; they were out-voted as REHEARSAL
went 17.2% → 20.9% → 44.2% at ~2.0 words. Capping it restores the ratio that
produced 84.92%.

**Why (2) as well.** (1) alone reached 84.92% emission but only 15.35%
accuracy — below zero-shot's 19.05%. (2) teaches the shape directly, which no
cycle has ever done.

**Why (3) is dangerous and is deliberately counterweighted.** The answerable
bucket is **19x larger** than the no-answer bucket. A mixture that taught
abstention too eagerly would trade 44.7% of TIME for 2.3% of it. AUG_NOANS is
therefore held to 20% of the new MCQ slice, the builder warns below a 3:1
ratio, and `AUG_MCQ2`'s distractors are sampled **within the question's type
bucket** so abstention cannot be solved by the shallow cue "these options
look unrelated" — which would not transfer, because TIME's distractors are
plausible.

---

## 4. Methodology — how this stays clean

- **No benchmark data is read.** Every new row is generated from the
  TimeQA/TLQA training pool. The builder asserts **0 benchmark collisions**
  and **0 letter-probe leaks**, dedupes, and re-runs the D27/D28 near-
  duplicate purge (J>0.8 token-set Jaccard or shared passage). Measured on
  the built mixture: 0 leaks, 0 collisions, 99 exact dupes and 39 near-dupes
  dropped.
- **AUG_SEQ instruction wording is paraphrased across four templates**, none
  of them the benchmark's string, so what is learned is the general
  convention "order chronologically → emit comma-separated letters", not one
  memorised prompt.
- **AUG_SEQ facts carry explicit years.** It therefore teaches *output
  format*, **not** news chronology. This must be stated plainly: no arm in
  this project orders events better than ~3pp over chance (§ 6), and v6 will
  not change that.
- **Hyperparameters are untouched.** `config_v6.yaml` differs from
  `config_v5.yaml` in exactly three lines: `train_data`, `val_data`,
  `output_dir` (verified by diff).
- **Gold correctness verified:** all 1,273 AUG_SEQ golds re-derived
  independently from the option years — **0 incorrect**.

---

## 5. Attribution — v6 is a combined arm, and that is a deliberate cost

The project's rule has been one variable per arm. v6 breaks it on purpose:
the user's target is to beat zero-shot, three independent deficits are
measured, and each cycle costs ~17 h. **A win by v6 therefore does not
attribute to any one change.** If attribution is needed for the thesis, run
these after v6, cheapest first — each is data-only and reuses this builder:

| arm | flags | isolates |
|---|---|---|
| v6a | `--aug-seq 0 --aug-noans 0 --aug-mcq2 0` | rehearsal cap alone |
| v6b | `--rehearsal 8026 --aug-noans 0 --aug-mcq2 0` | AUG_SEQ alone |
| v6c | `--rehearsal 8026 --aug-seq 0` | abstention alone |

v6a is the one worth running regardless: it is the direct test of D38's
dilution claim.

---

## 6. What to expect — stated before the run, so it can be checked honestly

Arithmetic on the § 1b table, assuming each fix reaches the stated rate and
nothing else moves:

| step | gain | TIME |
|---|---|---|
| v5 as scored today | — | 38.45% |
| sequence emission restored to zero-shot's 19.05% | +2.06pp | 40.51% |
| abstention restored to zero-shot's 52.95% | +1.21pp | 41.72% |
| free text back to v4's 26.02% (rehearsal cap) | +0.74pp | **42.46%** |
| **zero-shot** | | **41.49%** |

**So beating zero-shot is arithmetically reachable, with ~1pp of margin —
and only if essentially every fix lands in full.** Honest risk register:

- **Most likely partial outcome:** sequence emission returns but accuracy
  lands near chance (16.67%), worth +1.89pp not +2.06pp. Fine.
- **Real risk — over-abstention.** If v6 abstains on even 3% of the 46,937
  answerable items it loses ~0.9pp, cancelling the abstention win. **Check
  this first** in the format probe's `MCQ: answerable` accuracy row.
- **Real risk — AUG_SEQ teaches date-sorting, not ordering.** Its facts carry
  years; TIME's do not. Emission should transfer; accuracy may not.
- **v5's MCQ lead (64.50%) is the thing to protect.** It is worth more than
  both fixes combined. Any change that costs more than ~1pp there is a net
  loss — this is why rehearsal is capped rather than removed.

**If v6 lands between 40% and 41.5%**, the honest thesis claim is unchanged
in kind but much stronger in degree: fine-tuning beats zero-shot on TimeBench
and closes the TIME gap from −8.03pp (v1) to ~−1pp, with the residual
attributed to output format on 13% of items rather than to reasoning.

---

## 7. Running it

```bash
cd /home/g02-s26/Mohamed/temporal-news-reasoning
mkdir -p logs/queue_v6          # BEFORE the redirect opens
nohup bash scripts/run_queue_v6.sh > logs/queue_v6/driver.log 2>&1 &
```

Must be launched from a shell with real GPU access — **not** a Claude Code
Bash-tool session (D32/D33). Never `pkill -f run_queue_*`; use
`ps -eo pid,cmd --no-headers | awk '$2=="bash" && $3 ~ /run_queue/ {print $1}'`.

Budget: ~7 h train (14,850 rows, 18% fewer than v5, and the long passages are
the ones removed) + ~3 min style audit + ~1 min letter probe + ~12 min
TimeBench + ~7.3 h TIME + ~2 min rescore + ~1 min format probe ≈ **15 h**.

**LAUNCHED 2026-09-07 10:13:03.** Startup verified: 14,839/14,850 rows
tokenised, 11 `tokenize_error`, **0 dropped from any source** — the three new
`source_dataset` names have no dedicated loader branch and correctly fall
through to the generic `targets` reader (this was the D21 risk). Target
strings confirmed via `normalize_record`: `AUG_SEQ` -> `'A,B,C'`,
`AUG_NOANS` -> `'Cannot be determined from the context.; A'`, `AUG_MCQ2` ->
`'Illinois Central Gulf Railroad; C'`. 2,784 steps, 24,313,856 trainable
params (0.75%). Rate measured at step 39: **9.6-10.0 s/it**, ETA ~7 h 20 m,
so ~7.5 h training and ~15.5 h for the cycle. Live status and the
post-run check order: `docs/session_state.md` § "v6 IS RUNNING".

**Progress 2026-09-07 14:18:** step 1400/2784 (50.3%), epoch 1.51, ETA ~18:20.
Epoch-1 `eval_loss` **0.6089**, `checkpoint-928` saved; train loss ~0.63
through epoch 1 then ~0.40 in epoch 2, flat. **v6's `eval_loss` is not
comparable to earlier cycles** — 17.6% of its val rows are the new
`AUG_SEQ`/`AUG_MCQ2`/`AUG_NOANS` slices, and `AUG_SEQ`'s three-character
targets pull per-token loss down mechanically. Expect the usual U-curve with
the minimum at epoch 2; `load_best_model_at_end` discards epoch 3.

New in the chain: **`scripts/probe_answer_formats.py`** (stage 4b), the
process change D38 asked for. It reports emission AND accuracy per gold
shape and raises an explicit ALARM when an arm stops producing one. Run
against v5 it fires on both of v6's targets — which is exactly the report
that would have caught the collapse in one minute instead of after a 7-hour
evaluation.
