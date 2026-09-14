# Retired 2026-09-09 (audit docs/audit_2026_09_09.md)

Do not run these. Kept for history, append-only.

- run_zeroshot_tram.sh — writes the SAME zero-shot TRAM baseline file as
  scripts/run_schedule_v7.sh but on the HF engine; a resume would silently
  mix engines inside the file every TRAM arm is compared against (H4).
- run_queue_v7.sh — superseded HF-engine v7 chain writing a phantom
  results/corrected/v7 arm that no live tool reads (M12); header/kill notes
  still said v6.
- rescore_all.py — first-generation rescorer with an always-true completeness
  check and a stale expected TIME count (M14). Canonical:
  scripts/rescore_v5_protocol.py.
- run_full_finetuned_eval.sh — off-protocol legacy runner (batch 16, no
  token budget, writes into the v1 arm dir) (M6).
- seed_rerun_from_unaffected.py — one-shot v5-era repair; doc/code mismatch
  (L10), no longer needed.
- chain_after_v6.sh — waited on the v6 queue marker to launch
  run_zeroshot_tram.sh; both endpoints are done/retired.
