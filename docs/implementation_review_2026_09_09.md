# Implementation Review — whole thesis project (2026-09-09, CPU-only)

Scope: everything under `/home/g2/Mohamed/temporal-news-reasoning`.
Method: static analysis, import maps, compile-all, live CPU checks, test
suite. No GPU touched (zs-mistral TIME was running throughout).

## Verdict

The **live thesis pipeline is healthy and reproducible**; the risks sit in
(a) the Mistral finetuning package's data_loader divergence — directly
relevant to the planned mistral transfer arm — and (b) environment pinning.
The RAG-era code is unused scaffold, as D25 already ruled.

## What the project actually is (import-mapped)

| Layer | Files/Lines | Status |
|---|---|---|
| `scripts/` (43 py, 7.9k L) | runners, builders v2→v7, audits | **LIVE** |
| `src/data` + `src/evaluation` + `src/models` | loaders, scorers, inference | **LIVE** (17/9/2 active-runner imports) |
| `experiments/finetuning/LLaMA` + `shared` | training path (v1–v7) | **LIVE** |
| `tests/` (15 files, 86 pass) | protocol regression net | **LIVE** |
| `src/{rag,pipeline,prompting,temporal,training,demo}` | | **unused scaffold** (0 active imports; `src.rag` has 1 hit from `process_samples.py` only) — D25 stands |
| `temporal_rag/` (18 py, 9.0k L) | older RAG stack | **unused** (0 imports from anything) |
| root-level legacy scripts | 6 files, byte-identical duplicates of `temporal_rag/` copies | stale clutter |
| `experiments/finetuning/{Mistral,qwen3.5-9b-model}` | | Mistral = v1-era; qwen3.5 = no adapter ever |

## Verified correct (live checks, not doc claims)

- **compileall: 0 syntax errors** across 118 .py files.
- **86/86 tests pass** (incl. the 7 CoT-prompt tests added today).
- Canonical TIME loader returns **104,951** (D23 fix live).
- HF-vs-vLLM tokenizer parity **proven identical** (D57, CPU).
- **FA2 chain works**: ctypes preload of `libcudart.so.13` →
  `flash_attn 2.8.3` imports (a bare import fails *by design*; the preload
  in `src/models/inference.py:86` is required — do not "simplify" it away).
- Configs v2→v7 (+`v7_fast`) **all resolve** to existing
  `data/combined_80_20_v*/` splits — every cycle is re-runnable today.
- Training and evaluation entry modules import cleanly CPU-side.

## Risks / debt, ranked

1. **`experiments/finetuning/Mistral/data_loader.py` has diverged** from
   the LLaMA one (376 vs 394 lines, different md5; LLaMA's carries the
   v4→v7-era fixes; `shared/data_loader.py` is an 89-line third variant).
   **Blocker-grade for the planned mistral transfer arm**: port the LLaMA
   loader's current behavior before training mistral on v7-style data, or
   the arm trains on subtly wrong targets (the D21 class of bug).
2. **`requirements.txt` is unpinned** (`torch>=2.1.0` …) while the working
   env is torch 2.10.0+cu128 / transformers 4.57.6 / peft 0.18.1 /
   flash-attn 2.8.3-cu13. A fresh install pulls torch 2.11/transformers
   5.x and breaks flash-attn (documented twice already). Fix: commit a
   `pip freeze` as `requirements-frozen.txt`.
3. **Hardcoded absolute paths** in data-acquisition scripts
   (`scraper.py`, `download_bbc.py`, `download_ccnews.py` → the stale
   `/home/g2/temporal-news-reasoning/...` root). Legacy tools only — the
   live eval/train path is clean.
4. Root-level duplicates of `temporal_rag/` files — delete one copy
   (whichever) to remove confusion; **ask before deleting**.
5. `results_rerun/` (2.8 GB, pre-D22 rescore artifacts) — hygiene
   candidate; disk is fine (179 GB free). **Ask before deleting.**
6. Test-coverage gaps: runners/builders rely on inline audits rather than
   pytest; `train.py` config plumbing untested. Acceptable — the protocol
   net (86 tests) covers the scoring-critical paths that burned us before.
7. **Two live copies of the repo** (this machine + the other PC), each
   editing its own docs — a sync/merge session is needed before the
   thesis write-up pulls numbers from either side.

## Recommended actions (order)

1. Port LLaMA's data_loader to the Mistral package (before any mistral
   training; ~1 h + tests).
2. Commit `requirements-frozen.txt`.
3. The 20-min GPU mistral-vLLM parity re-test (D57) in the next idle
   window.
4. Cleanup pass (root duplicates, `results_rerun/`) after user approval.
5. Cross-machine doc/code sync before thesis writing.
