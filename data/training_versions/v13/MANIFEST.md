# Training data — v13

Dir: `data/combined_80_20_v13/` (100.3 MB) — train 24,870 / val 6,376 | synthetic 79.3% | general rehearsal 0.0%

v12 (minus 207 TimeQA rows whose gold the CoT pass found contradicted or unsupported by the passage) + AUG_TPL3: agent-written Python template/builder rows (scripts/glm_v13_build/; NOT GLM-5.2 chat, despite the plan -- audit_2026_10_04 §1.1) for relation, nli, dialogue, storytelling (gold-longer share balanced to 50%), extract, timeline, computation, relative/duration/order compare + AUG_PROG programmatic math/temporal data (deliberately non-LLM, kept), with the audit's quality filters applied (counts in data/manifest.json).

Results: pending

Full composition in `manifest.json`. A sha256-verified copy of the data is in `data/`; the canonical dir is `data/combined_80_20_v13/` (configs point there — if the copy ever diverges, the original wins).
