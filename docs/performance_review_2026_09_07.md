# Speed review — eval and training, without changing the numbers

> **⚠ Superseded in part (2026-09-12).** This document quotes TRAM figures
> that are **withdrawn**: 85.3% of the TRAM column was generated from prompts
> that omitted the question (`BenchmarkLoader._tram_csv_to_example` never read
> the NLI `Hypothesis` or storytelling's `Story`). A re-run is in progress.
> TIME figures here are also pooled over four retrieval `Setting`s that differ
> by ~23pp and in which the fine-tuning effect changes sign. See
> `docs/audit_2026_09_12.md` and session_handoff **D74**. Left unedited
> otherwise — it is the append-only record of what was known at the time.

Written 2026-09-07 at the user's request, while v6's TIME leg ran. Every
figure below is measured on this machine, not estimated from documentation.

**Bottom line: there is no completely free lunch.** The eval loop is already
well built — the adapter is merged (`merge_and_unload`), prompts are
length-sorted, batching is token-budget aware, and outputs are short. Every
remaining speedup changes the instrument slightly and therefore needs a parity
check. Two are strongly worth it, one is nearly free, and two common ideas
turn out to be dead ends here.

**None of this applies to the running v6 — do not touch it.** All of it is
for v7 and later.

---

## 1. Where the time actually goes

### Evaluation is PREFILL-bound, not decode-bound

Measured on v6's live TIME run (2.76 ex/s):

| | |
|---|---|
| items per batch | **~26** — the 57,344 token budget binds, **not** `--batch-size 32` |
| seconds per batch | 9.4 |
| prefill | 26 x ~2,200 tokens = **~57,200 tokens**, ~75% of the wall clock |
| decode | ~40 steps for a **mean output of 7.2 tokens** |

Generated output length, measured over 20,000 v6 predictions:

| p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|
| 4 | 12 | 19 | 26 | 41 | **7.2** |

**TIME prompts are long (mean context 1,653 words) and answers are tiny.** The
cost is reading the prompt, not writing the answer.

### Training wastes ~40% of its compute on padding

Token lengths of v6's training rows (1,500-row sample, `max_seq_length` 2048):

| p10 | p25 | p50 | p75 | p90 | p99 | mean |
|---|---|---|---|---|---|---|
| 54 | 76 | 453 | 1,544 | 2,965 | 11,967 | 1,192 |

The spread is enormous — a 54-token row can share a micro-batch with a
2,000-token row, and the short one is padded to the long one's length.
Measured padding waste at micro-batch 2:

| ordering | padding waste |
|---|---|
| random (current) | **40.6%** |
| length-sorted | **0.5%** |

---

## 2. Two ideas that DON'T work here — check these off the list

**Cutting `max_new_tokens` from 128.** p99 output is 41 tokens, so batches
almost never approach the 128 budget; HF stops when every sequence in the
batch has emitted EOS. Cutting to 64 would save almost nothing and would
change ~1% of outputs. **Not worth it.**

**Stopping generation at the first newline.** The postprocessor only reads
`first_line`, so this looked free — but **0.0% of v6's outputs contain a
newline**. There is nothing to truncate. **No gain.**

---

## 3. What actually would work, ranked

### A. vLLM for TIME and TRAM — biggest win, and already validated

| | |
|---|---|
| speedup | **5-10x** on long-context batch inference |
| accuracy risk | **already measured for LLaMA**: 94.40% prediction agreement, **0.05pp** delta, and full TimeBench **45.64 (vLLM) vs 45.65 (HF)** on n=21,188 |
| verdict in `pc2_vllm_setup.md` | **vLLM allowed for llama**; **FORBIDDEN for mistral** (2.20pp delta) |

**Why it wins especially here:** HF batching waits for the *longest* sequence
in the batch. With outputs ranging p50=4 to p99=41, most sequences idle while
one finishes. vLLM's continuous batching retires each sequence as it ends —
exactly the mismatch this workload has.

**The condition:** engine consistency. Do not mix HF and vLLM across arms in
one table. The clean move, already proposed in `pc2_vllm_setup.md` § 2: run
**every** arm on vLLM to produce a complete engine-consistent column beside
the HF one. At 5-10x that is ~6 h for all six arms, against ~60 h on HF.

**vLLM is not installed in the eval venv** (`import vllm` -> ModuleNotFoundError);
the build procedure is `pc2_vllm_setup.md` § 5.

### B. Length-bucketed training batches — up to ~40% faster training

Sorting rows by length before batching takes padding waste from **40.6% to
0.5%**. On v6's 8h18m run that is worth roughly **2-3 hours**.

**Is it "a variable"?** It changes which examples share a micro-batch, so the
optimisation trajectory changes — like changing the shuffle seed. It does not
advantage the model and it does not touch the recipe (LR, epochs, batch shape,
LoRA rank all unchanged). Note that **example grouping already differs between
every arm**, because every arm has different data, so this introduces no new
*class* of difference.

**The real risk, and the mitigation:** naive sorting creates correlated
batches — all the short `AUG_SEQ` rows together, all the long rehearsal
passages together — which gives each optimizer step a less diverse gradient.
**Use bucket-then-shuffle:** sort into length buckets, form batches within
buckets, then shuffle the batch order. Validate with a 200-step pilot
comparing loss curves before committing a full cycle.

### C. `gradient_checkpointing: false` — ~30% faster, mathematically identical

Activation checkpointing recomputes activations in the backward pass to save
memory. **Gradients are identical either way** — it is a pure speed/memory
trade and does *not* change the recipe (`config_v5.yaml` says so in its own
comments).

**Why it is off the table today:** v4's first launch OOMed with it disabled,
because v4's rehearsal slice carried full HotpotQA/DROP/CoQA passages.
**But v6's data is different** — rehearsal is capped at 2,600 rows and v6 ran
at 6.7 GB allocated / 20.1 GB reserved of 25.3 GB.

**Test it, do not flip it blindly:** run 50 steps of v7 with
`gradient_checkpointing: false`. If it survives, keep it — this is the only
speedup on this list that provably cannot change the result.

### D. flash-attention — targets the 75% that is prefill

`flash_attn` is **not installed** (`ImportError`). Attention is O(n²) in
prompt length, and these prompts run 2,000-4,000 tokens, so FA2 typically
gives 1.5-3x on prefill-heavy work — precisely this workload.

**Same caveat as vLLM:** a different attention kernel is a numerically
different instrument. It needs the same parity protocol — a 2,000-item
agreement check against SDPA before any headline number uses it.
`docs/evaluation_caveats_report.md` already records an SDPA/FA2 blend as a
known caveat, so this must be done carefully or not at all.

### E. Already done — do not "fix" these

- **The adapter is merged.** `src/models/inference.py:113` calls
  `merge_and_unload()`, so there is no per-layer LoRA overhead at eval.
- **Prompts are length-sorted** and batching is token-budget aware
  (`run_baselines.py` ~line 604) — big batches for short prompts, small for
  long, which is already the right design.
- **Resume regenerates partial batches on purpose.** Do not "optimise" it:
  batch composition changes greedy output (measured 89.25% agreement when 800
  identical items were regenerated in different batches).

---

## 4. Recommendation

| priority | change | gain | risk | gate |
|---|---|---|---|---|
| 1 | **vLLM for TIME/TRAM, all arms** | **5-10x** | already measured at 0.05pp | run every arm on it; keep the HF column |
| 2 | **bucket-then-shuffle training batches** | ~40% of training | trajectory changes like a reseed | 200-step loss-curve pilot |
| 3 | **`gradient_checkpointing: false`** | ~30% of training | **none — identical gradients** | 50-step OOM test |
| 4 | flash-attention at eval | ~1.5-3x prefill | different kernel | 2,000-item parity check |

Doing 2 and 3 together would cut a v7 training run from ~8 h to roughly
**4-5 h**. Doing 1 as well would take a full cycle from ~19 h to **~6 h**, and
would make the TRAM run (`docs/v8_plan.md`, D44) cheap enough to do at full
scale rather than on a 50k subset.

**Nothing here should be applied to v6, which is running.**

---

## 5. Can Mistral use vLLM? (asked 2026-09-07)

**Under the current rule, no** — `pc2_vllm_setup.md` rule 2: *"Any mistral leg
runs on HF only, regardless of machine."* The parity check measured a **2.20pp**
delta (31.15 vLLM vs 33.35 HF) and 196/2000 correctness flips.

**But the ban rests on a parity failure that looks like a BUG, not an engine
limitation.** Re-examined `results/parity_vllm/mistral_time_parity.jsonl`
(n=2,000) against the llama file:

| | mistral | llama |
|---|---|---|
| disagreements | **44.9%** | 5.6% |
| vLLM output is a prefix of HF | 9 | 3 |
| HF output is a prefix of vLLM | 54 | 25 |
| mean words, vLLM | **16.0** | 8.2 |
| mean words, HF | **10.2** | 8.7 |

**Llama's profile is what engine noise looks like:** few disagreements, matched
output lengths, occasional argmax tie-breaks.

**Mistral's is different in kind** — substantively different answers, plus a
systematic **57% length increase** under vLLM:

```
vLLM: 'Eraviperoor'                       HF: 'Cherthala'
vLLM: "...foreign policy towards Israel"  HF: "...nuclear deal with Iran"
vLLM: 'Michele Paramatti'                 HF: 'The thirteenth person to play
                                               for U.S. Russi is Manuel Marani.'
```

Numerical differences do not turn "Cherthala" into "Eraviperoor", and they do
not make one engine consistently more verbose. Note the third example: HF
answers in a full sentence, vLLM with a bare name — **a different
instruction-following mode**, which is the signature of a different chat
template or system prompt reaching the model.

**Suspects, in order:**
1. **Chat template applied differently.** Mistral uses `[INST] … [/INST]`; if
   one path calls `apply_chat_template` and the other passes raw text, the
   model sees a different prompt.
2. **Double-BOS.** Mistral's tokenizer adds `<s>` by default; if vLLM adds
   another the prompt starts `<s><s>[INST]`. **This project has already
   shipped a double-BOS defect once** — it is in the original defect
   inventory.
3. vLLM's LoRA kernels vs HF's merged adapter (`merge_and_unload`).

**The diagnostic, ~1-2 h:** dump the exact token IDs each path feeds for the
same item and diff them. If they differ it is a bug — fix it, re-run the
2,000-item parity, and Mistral qualifies for vLLM exactly as llama did.
`config.json` rules out one common cause: `sliding_window` is `None`, so
sliding-window attention is not the explanation.

**Not urgent.** Mistral is deferred (standing decision 1), its two fine-tuned
numbers predate the current protocol and are not comparable, and it is a
single final replication arm rather than a cycle — one HF run is tolerable
even at 7B.

**Plan on HF for Mistral, but treat the vLLM ban as PROVISIONAL, not
permanent.** Re-test it before accepting a slow HF run as the only option.

---

## 6. TRAM runtime — the item count is misleading (measured 2026-09-07)

| | items | mean prompt tokens | **total prompt tokens** |
|---|---|---|---|
| TIME | 104,951 | **2,461** | **258.3M** |
| TRAM | 980,918 | **55** | **54.0M** |

**TRAM has 9.4x more items but 4.8x FEWER prompt tokens** — its prompts are
45x shorter (p50 45 tokens, p90 96), mostly context-free one-liners, and
14.6% carry no context at all.

**Consequence: TRAM is batch-count bound, not prefill bound — the opposite of
TIME.**

| | items/batch | batches | est. s/batch | est. total |
|---|---|---|---|---|
| TIME (measured) | ~26 (token budget binds) | 4,037 | 9.4 | **10.5 h** |
| TRAM at `--batch-size 32` | 32 (**the CAP binds**) | 30,654 | ~2.7 | **~23 h** |
| TRAM at `--batch-size 256` | 256 (~25k tokens at p90, inside budget) | 3,832 | ~2.7 | **~3 h** |

At today's settings TRAM costs **more than twice** TIME. Raised to a batch
size its prompt lengths actually justify, it costs **less than a third** of
TIME.

**This is a FREE choice, not a protocol violation.** Batch size is protocol
for TIME/TimeBench because changing it would break comparability with numbers
already published from this project. **TRAM has never been run** — there is no
prior TRAM number to stay consistent with. Choose the batch size once,
document it, and use the same value for every arm on TRAM.

On D44's recommended **50k stratified subset** the question is nearly moot —
minutes either way — but the same reasoning says to pick the larger batch.

**Do not raise `--batch-size` for TIME or TimeBench.** Those have published
numbers and batch composition changes greedy output (89.25% agreement measured
when 800 identical items were regenerated in different batches).


---

## 7. MEASURED 2026-09-08 — what the training speedups were actually worth

The § 3 recommendations were projections. Piloted, they came out as follows.

| # | change | projected | **measured** |
|---|---|---|---|
| B | `group_by_length` | ~40% | **unpriceable by a short pilot** — HF's `LengthGroupedSampler` runs the LONGEST batches first, so 60 steps read 13.8 s/step (worse than baseline) purely as an artifact. Runs fine (6.7 GB alloc / 15.8 GB reserved); the 40.6% -> 0.5% padding figure still predicts a real gain, but only a substantial run can confirm it |
| C | `gradient_checkpointing: false` | ~30%, "no accuracy risk" | **NOT AVAILABLE.** OOMs at 22.44 of 23.53 GiB. v4 failed identically on 2026-09-05 — twice, two different mixtures. A property of 2,048-token sequences on a 24 GB card |

**Baseline, measured:** 60 steps in 9m21s = **8.82 s/step**; v7's 4,980 steps
= **12.2 h**. (The earlier "~15 h" was a projection.) Cross-check: v6 ran
14,850 rows -> 2,784 steps -> 8.3 h on the same recipe; v7 has 79% more data
and takes 1.47x the time.

**§ 3C claimed "the one speedup on the list with NO accuracy risk". That was
true about its MATHEMATICS and wrong about its FEASIBILITY** — gradients are
indeed identical, but the config does not fit in memory, which the entry
flagged as needing a test and which the test then failed.

**Two defects in the pilot itself, both fixed:** `--max-steps` left
`eval_strategy: epoch` on, so each 60-step run was followed by a 24-minute
evaluation over 6,636 val rows; and a 60-step pilot structurally cannot price
a length-grouped config. Both are now recorded in `train.py` and the pilot
script.

**Net: the only training speedup available on this hardware is
`group_by_length`, and it is unproven.** v7 runs on `config_v7.yaml` at a
known 12.2 h. **The real time savings are on the eval side**, which is what
moving to vLLM buys — see § 6 and D53.
