# Training data — v13

Dir: `data/combined_80_20_v13/` (94.1 MB) — train 22,093 / val 5,678 | synthetic 76.8% | general rehearsal 0.0%

v12 (minus 207 TimeQA rows whose gold the CoT pass found contradicted or unsupported by the passage, and minus the non-math rows that came from the agent's script-built glm_raw files 271-313) + AUG_TPL3 MATH ONLY: agent-written Python template rows (scripts/glm_v13_build/; NOT GLM-5.2 chat -- audit_2026_10_04 §1.1) for Computation, Timeline, Relative_Reasoning, Duration_Compare, Order_Compare; every non-math template row removed by researcher decision + AUG_PROG programmatic math/temporal data (deliberately non-LLM, kept). Every kept math row passed an independent re-verification (scripts/verify_v13_math.py); filter counts in data/manifest.json.

Results: pending

Full composition in `manifest.json`. A sha256-verified copy of the data is in `data/`; the canonical dir is `data/combined_80_20_v13/` (configs point there — if the copy ever diverges, the original wins).
