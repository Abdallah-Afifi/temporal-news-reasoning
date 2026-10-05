# Training data versions — index

Each version folder holds a **verified copy** of its data (`data/` subdir: train/val jsonl + builder manifest), plus `MANIFEST.md` + `manifest.json` (measured composition). The **canonical originals stay in `../../combined_80_20_*`** — 9 configs and the second PC's in-flight v7 reference those paths; if copies ever diverge, originals win.

| v | dir | train | val | synthetic | general | key fact |
|---|---|---|---|---|---|---|
| v1 | combined_80_20_split | 12,000 | 3,000 | 0.0% | 0.0% | no synthetic; Temprel still included |
| v2 | combined_80_20_v2 | 16,557 | 4,138 | 31.4% | 0.0% | leakage purge + first AUGs |
| v3 | combined_80_20_v3 | 12,436 | 3,108 | 16.9% | 16.7% | BROKEN (TLQA empty targets) |
| v3-fixed | combined_80_20_v3_fixed | 12,067 | 3,016 | 14.3% | 17.2% | D21 fix; best TimeBench |
| v4 | combined_80_20_v4 | 12,830 | 3,204 | 15.1% | 20.9% | rehearsal source swap |
| v5 | combined_80_20_v5 | 18,180 | 4,542 | 10.7% | 44.1% | rehearsal x3; sequence collapse |
| v6 | combined_80_20_v6 | 14,850 | 3,708 | 30.7% | 14.0% | format arm (+0.73 zs, artifact) |
| v7 | combined_80_20_v7 | 26,570 | 6,636 | 45.0% | 24.1% | coverage arm |
| v9_glm | combined_80_20_v9_glm | 12,002 | 2,998 | 31.8% | 0.0% | genuine GLM AUG_GLM2 (3,000); legacy prompt — below zero-shot |
| v10_glm | combined_80_20_v10_glm | 12,003 | 2,997 | 51.5% | 0.0% | AUG_GLM2 5,954; **superseded before training** (contamination + legacy prompt) |
| **v11** | combined_80_20_v11 | 11,446 | 2,869 | 53.2% | 0.0% | **prompt parity + decontaminated; beats zero-shot on all three** (seed 42) |
| v12 | combined_80_20_v12 | 11,804 | 2,971 | — | — | corrected storytelling/relation/ordering/dialogue/duration cards; beats zero-shot on all three (`docs/v12_plan.md`, results §11) |
| v13_provisional | (snapshot) | 22,314 | 5,660 | — | — | **PROVISIONAL**: v12 + AUG_PROG only — the v13-prog pilot's data (dev-only ablation) |
| v13 | combined_80_20_v13 | 24,870 | 6,376 | — | — | v12 + AUG_TPL3 (agent-template rows, NOT GLM) + AUG_PROG, quality-filtered (`docs/v13_plan.md` §9); not trained yet |

v9_glm / v10_glm / v11 folders were added 2026-09-25 by
`scripts/make_version_folder.py` (sha256-verified copies; the measured
composition, story and results are in each folder's `manifest.json` /
`MANIFEST.md`). Numbers for v11 are on test-minus-dev; see
`docs/results_and_methodology.md` §10.

**Audit 2026-09-09 (D60):** all copies sha-verified byte-identical; 0 JSON
errors; empty-response rows v1:1,620 / v2:1,035 (= the documented dropped
TimeQA records) / v3+:0; v7 MCQ 0 dual-gold (D45); AUG_SEQ golds
well-formed in `targets`; 0 exact-question collisions vs benchmarks
(v6, v7); v4+ split hygiene clean. Early-version overlaps (v1–v3-fixed)
are frozen history, pre builder-guards.
