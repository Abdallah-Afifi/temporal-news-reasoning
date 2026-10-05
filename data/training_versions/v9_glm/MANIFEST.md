# Training data — v9_glm

Dir: `data/combined_80_20_v9_glm/` (80.3 MB) — train 12,002 / val 2,998 | synthetic 31.8% | general rehearsal 0.0%

Synthetic-swap arm with GENUINE GLM-chat AUG_GLM2 (3,000 rows, replacing the template-generated v9 slice). LEGACY prompt: trained on a different prompt from the evaluation prompt (audit 2026-09-23 §1); TimeQA pool carries the 561 contaminated rows (§7).

Results: TIME 40.79 / TimeBench 45.05 / TRAM 44.45 (vLLM, full pools, legacy prompt) — below zero-shot on all three.

Full composition in `manifest.json`. A sha256-verified copy of the data is in `data/`; the canonical dir is `data/combined_80_20_v9_glm/` (configs point there — if the copy ever diverges, the original wins).
