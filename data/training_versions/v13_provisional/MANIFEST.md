# Training data — v13_provisional

**PROVISIONAL: v12 + AUG_PROG only, 0 template/GLM rows — the data behind the v13-prog pilot (+11.26pp dev).**

train 22,314 / val 5,660. Composition: {'AUG_PROG': 10510, 'AUG_GLM2': 5024, 'TimeQA': 4504, 'AUG_SEQ': 1019, 'TLQA': 855, 'AUG_GLM': 402}.
The 5,024 `AUG_GLM2` rows are v12's own corpus (unchanged), not a v13 wave.

This was `data/combined_80_20_v13/` as of 2026-10-03 17:20 (sha256 in `manifest.json`).
That directory has since been rebuilt as the real v13 mixture, so **the copy in `data/`
here is the only remaining copy and is authoritative** (unlike other version folders,
where the canonical `combined_80_20_*` dir wins). `config_v13_prog_pilot.yaml` still
points at `data/combined_80_20_v13`; to reproduce the pilot, point it here instead.

Corrected 2026-10-05 per `docs/audit_2026_10_04.md` §6.3 (the earlier text wrongly said
the GLM-5.2 wave was included).
