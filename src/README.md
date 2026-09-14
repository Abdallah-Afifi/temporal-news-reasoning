# `src/` — core package

> Re-written 2026-09-13, reverted by the two-PC merge, restored 2026-09-14.

**Not everything here is in the evaluated chain.** The 2026-09-12 audit checked
importers; the split below is load-bearing, because reproducing "from `src/`"
silently produces non-comparable results.

## ALIVE — every published number passes through these

| module | role |
|---|---|
| `data/data_loader.py` | `BenchmarkLoader` for TIME / TimeBench / TRAM. Fixed 2026-09-12: TRAM's NLI `Hypothesis` and storytelling `Story` now reach the prompt; TIME's retrieval `Setting` is preserved. |
| `evaluation/metrics.py` | `_normalize_answer`, `accuracy` — the scorer core. |
| `evaluation/date_equivalence.py` | date-reformatting equivalence (`matches_any`). |
| `evaluation/generation_metrics.py` | token-F1 for the generation categories. |
| `evaluation/evaluate.py` | live report harness. **Not the canonical scorer** — it uses a different code path and a different denominator; quote `scripts/rescore_v5_protocol.py`. |
| `models/inference.py` | `SLMInference` — the HF generation path. |
| `data/synthetic_templates.py` | used by the data builders. |

## DEAD for the thesis — zero non-test importers

`rag/`, `pipeline/`, `temporal/`, `prompting/`, `demo/app.py`,
`data/corpus_processor.py`, `data/merge_datasets_80_20.py`.

`prompting/templates.py` and `prompting/self_consistency.py` are
`raise NotImplementedError` stubs despite being advertised as key components in
the top-level README's original text.

## DO NOT USE — `training/`

`training/train_lora.py` is a **divergent second trainer**: no prompt-loss
masking, `### Instruction` format, lr 2e-4, batch 4×4, and its shipped configs
point at data that does not exist. The real trainer is
`experiments/finetuning/LLaMA/train.py`. See `src/training/README.md`.
