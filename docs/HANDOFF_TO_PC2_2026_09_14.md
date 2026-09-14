# Briefing for the Claude session on PC-2 — 2026-09-14

You are on the machine running the CoT campaign. This is written by the session
on PC-1 (the audit/fine-tuning machine) after auditing the merged project.
Everything below was measured, not assumed.

---

## 1. DO THIS FIRST — your in-flight CoT run is probably producing garbage

```bash
grep max-new-tokens scripts/run_zs_cot.sh scripts/run_fewshot_cot.sh
```

**If it says 256, stop the run and restart it at 768.**

Why: the CoT prompt tells the model to reason in steps and finish with a line
`ANSWER: <answer>`. Scoring reads that anchor. At 256 tokens the generation runs
out *before* reaching it, and those items score ~0.08% — a budget artifact, not
a reasoning result. Measured on the completed zero-shot CoT run here:

| arm | truncated | accuracy on truncated |
|---|---|---|
| llama · TIME | **22.1%** (23,152 items) | **0.08%** |
| mistral · TIME | 12.8% | 0.11% |
| llama · TimeBench | 7.5% | 0.70% |
| mistral · TimeBench | 13.1% | 0.00% |

It **reverses the sign of the TIME result**:

| | as scored | conditional on reaching the anchor | standard zero-shot |
|---|---|---|---|
| llama · TIME | 38.52% (−2.94) | **49.41%** (+7.95) | 41.46% |
| llama · TimeBench | 49.11% (+3.93) | **53.01%** (+7.83) | 45.18% |

(The conditional column is an upper bound — items needing more reasoning are
plausibly harder — so the truth is between the columns. Only a re-run settles
it, which is why the budget matters.)

PC-1 already changed both runners to `--max-new-tokens 768`. **That change is
not on your machine unless you have synced since 2026-09-14.**

---

## 2. Take PC-1's copies of these files at the next sync

The **last** merge silently reverted two of PC-1's files and left 34 stale
cross-references. Verify each of these rather than assuming the merge got it
right:

| file | what PC-1 changed |
|---|---|
| `scripts/run_zs_cot.sh`, `scripts/run_fewshot_cot.sh` | `--max-new-tokens` 256 → **768**, with the measurement in the header |
| `scripts/rescore_v5_protocol.py` | CoT + mistral arms added to `ARMS`; `anchor_missing_pct`; tree-vs-list check; torn-line rule |
| `src/data/data_loader.py` | the TRAM prompt fix + `Setting` preservation (do **not** take an older copy) |
| `scripts/run_baselines.py`, `scripts/run_eval_vllm.py` | resume guards, provenance tags, base-model/adapter guard, write lock |
| `scripts/run_zs_tram_llama.sh` | **retired** — see §4 |
| `scripts/README.md`, `src/README.md` | rewritten; **the last merge reverted both** |

Quick check after any sync:

```bash
grep -c "question = hypothesis" src/data/data_loader.py     # must be 1
grep -c "anchor_missing"        scripts/rescore_v5_protocol.py  # must be >0
grep -c "max-new-tokens 768"    scripts/run_zs_cot.sh       # must be >0
./venv/bin/python -m pytest tests/ -q                        # 143 passed, 6 skipped
```

---

## 3. Current canonical numbers (all verified on PC-1, 2026-09-14)

`results/rescored/v5_protocol.json` — 14 arms on TIME/TimeBench, 10 on TRAM.
All 32 previously-published cells reproduce exactly after the sync.

**Do not quote any TRAM number written before 2026-09-13.** That column was
re-run after a prompt-content bug (the NLI `Hypothesis` and storytelling `Story`
never reached the prompt — 85.3% of the column). Zero-shot TRAM is **46.95%**,
not 35.20%, and **v1 went from best fine-tuned arm to worst**.

Three results that supersede older documents:

1. **"v6 beats zero-shot on TIME (+0.73pp)" is a benchmark artifact** — 2,427
   items where an abstain option is always the gold. Excluded, no fine-tuned
   arm leads zero-shot on TIME.
2. **TIME must be read stratified by retrieval `Setting`** (`base` = gold
   context, 40.4%; `bm25`/`vector`/`hybrid` = retrieved, 59.6%). Fine-tuning is
   **+2.2 to +2.8pp on gold context and −2.2 to −3.0pp on retrieved context**;
   pooling reports ≈0. This is the campaign's strongest positive result.
3. **`zsCoT-llama` on TimeBench is 49.11%**, beating v3-corrected's 48.46% —
   the best fine-tuned arm in the project — **with no training at all**, and
   still depressed by 7.5% truncation.

---

## 4. Do not run `scripts/run_zs_tram_llama.sh`

It ran the **HF** engine into the **vLLM** TRAM baseline directory. That is
audit defect **H4**, the exact reason `scripts/retired/run_zeroshot_tram.sh`
was retired on 2026-09-09. It is also redundant and aimed at stale data — the
corrected zero-shot TRAM baseline is `results/tram_fixed/zero_shot/` (46.95%),
not `results/baseline/zero_shot_v3/.../tram/` (35.20%). PC-1 has made it refuse
to run; if your copy still executes, replace it.

---

## 5. A methodology note for whatever you run next

The CoT arm changes **three things at once** versus the standard baseline:
system prompt (none → `COT_SYSTEM_PROMPT`), answer instruction, and token
budget (128 → 256/768). So "CoT is worth +7.95pp" is really "persona + step
instruction + 2–6× budget". That is the v7 mistake in a new place, and the
project's own single-variable discipline (D67) should extend to prompting arms.

Two cheap legs attribute it:
1. standard prompt + `COT_SYSTEM_PROMPT` + 128 tokens → isolates the persona;
2. standard prompt + no system prompt + 768 tokens → isolates the budget.

Also: **`strict_pct` means something different for a CoT arm.** It reads ~0%
because the first line is "Step 1: …" by construction; the `ANSWER:` anchor is
the *designed* extraction point. Do not compare `strict_pct` across prompting
styles.

---

## 6. Two things that are still unrun and are the highest-value GPU work

Neither has ever been done, and the first is the bigger gap:

1. **A seed-43 replicate of v6.** There is **no variance estimate anywhere** in
   this project — one run per arm, one seed, no confidence intervals — while
   the decisive margins are ±0.5–2.8pp. ~8 GPU-hours.
2. **A prompt-matched zero-shot control** — the baseline re-run with
   `TEMPORAL_SYSTEM_PROMPT`, which closes the last confound between the
   baseline and the fine-tuned arms. ~7 GPU-hours.

---

## 7. Read these, in order

1. `docs/session_handoff.md` — start at the header banner, then **D77** and its
   two addenda, then **D75**, then **D74**.
2. `docs/audit_2026_09_14.md` — the audit of your CoT work (what is right in
   it, and the four defects).
3. `docs/results_and_methodology.md` §1.0, §1.1, §1.5, §9.

Note the merge renumbered decisions: PC-2's D56–D65 kept their numbers, PC-1's
D56–D65 became **D66–D75**. `docs/session_state.md` documents the mapping.
