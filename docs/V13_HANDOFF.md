# V13 HANDOFF — full state of work for a successor session

Written 2026-10-03 20:08 by the session that implemented v13. Repo root for
everything below: `/home/g2/Mohamed/temporal-news-reasoning` (the newest
working copy; `git status` shows uncommitted work — nobody was asked to
commit). Read this top to bottom before touching anything.

---

## 1. Project context (30 seconds)

LoRA fine-tuning campaign for temporal reasoning on news. Three benchmarks
(TIME ~100k items, TimeBench ~18.6k, TRAM ~971k), frozen evaluation
protocol ("v5 protocol"): one pinned zero-shot reference (`zs-vllm-pinned`),
identical prompts/engine/date for zero-shot and fine-tuned runs, greedy
vLLM, test-minus-dev scoring. The canonical results file is
`results/rescored/v12_test_minus_dev.json`; the single reference doc is
`docs/results_and_methodology.md` (§10 = v11).

Current best (test-minus-dev, v5 micro accuracy %):

> **Corrections 2026-10-04/05** (`docs/audit_2026_10_04.md`): the "GLM-5.2
> wave" below was produced by the coding agent's Python templates, not GLM
> chat (§1.1) — relabelled `AUG_TPL3` in the build; the v11-best row was
> mislabelled as a 3-seed mean; Mistral is listed separately because it is
> not comparable to the LLaMA rows.

LLaMA-3.2-3B (comparable rows):

| Arm | TIME | TimeBench | TRAM |
|---|---|---|---|
| zs-vllm-pinned (zero-shot) | 41.15 | 44.83 | 46.55 |
| v11-best, seed 42 | 46.80 | 46.48 | 53.74 |
| v11-best, mean of seeds 42/43/44 | 46.42 | 46.30 | 53.10 |
| v11-2ep | 46.12 | **49.27** | **55.39** |

Mistral-7B (own zero-shot only — NOT comparable to the rows above):
zs-vllm-mistral 37.25 / 45.40 / 52.13; mistral-best 3-seed mean
45.78 / 55.81 / 56.73.

Remaining weaknesses v13 attacks (v11-2ep's by_category numbers, from v12
rescored):
TIME Computation 50.5, Timeline 22.7, Extract 6.1 (regression),
Order_Compare 56.7 (regression); TimeBench temporal_dialogue 61.9
(−16 regression), arithmetic 22.3; TRAM temporal_relation 31.2 (n=202,825!),
storytelling 68.9 (−8 regression).

## 2. The v13 plan (pre-registered in `docs/v13_plan.md`)

1. **AUG_PROG** — 13,200 programmatic training rows (no LLM): clock
   arithmetic, month arithmetic, TIME-style Computation/Timeline/
   Duration_Compare/Order_Compare, TRAM-style relation (incl. the
   event-to-time surface form), ordering TRUE/FALSE + permutations, and the
   FIRST-ever training analogue for TIME Extract (multi-select, gold
   `"B  C"` two-space joined).
2. **AUG_GLM3** — ~4,500 GLM-5.2-generated rows targeted at the weak
   language-y buckets: relation (extended card), nli_saq/nli_mcq,
   temporal_dialogue, storytelling, Timeline, Computation,
   Relative_Reasoning, Duration_Compare, Order_Compare, duration, ordering,
   extract (new card).
3. **v13 mixture** = v12 unchanged + AUG_GLM3 + AUG_PROG (~29k rows
   planned; provisional build without GLM = 22,314/5,660 exists).
4. **Five arms**: `soup-v11` (no training: average of the three v11 seed
   adapters), `v13-base` (t09 recipe: lr 1.77e-4, 1 epoch, LoRA r16/a32, 7
   target modules, eff. batch 16, completion-only loss, prompt parity),
   `v13-2ep`, `v13-ctx` (4096 ctx), `v13-r32` (r32/a64, runs last,
   droppable).
5. Success criteria fixed in plan §8 (headline: beat v11-best's mean delta
   +5.27/+1.48/+6.54 on ≥2 of 3 benchmarks beyond the 0.4–0.7pp seed noise;
   flip the four active regressions to ≥0; stretch targets per bucket).
6. Parked for v14 (plan §7): CoT distillation into direct answers,
   self-consistency, external calculator, NEFTune.

## 3. What exists on disk (all created + tested 2026-10-03)

### Data
| Path | What |
|---|---|
| `data/prog_aug_v13/*.jsonl` (10 files) | 13,200 AUG_PROG rows; `audit.json` = clean generator audit (verifiers, letter balance, gold caps, dedup vs v12) |
| `data/prog_aug_v13_smoke/` | 12-rows-per-category smoke set |
| `data/combined_80_20_v13/{train,val}.jsonl` | **PROVISIONAL** v13 build (v12 + AUG_PROG only, 22,314/5,660; `manifest.json` says `provisional: true`, `glm_rows: 0`) |
| `data/glm_packets_v13/` | `MASTER_PROMPT_v13.md` (4,321 words, 19 cards), `ORDERS_v13.txt` (450 orders), 109 one-shot API packets `NNN_<Category>.md`, `_plan.json` |
| `data/glm_raw_v13/`, `data/manual_aug_glm_v13/` | EMPTY — await the GLM run |

### Scripts (new)
| Path | What |
|---|---|
| `scripts/generate_v13_prog_data.py` | the AUG_PROG generator; every gold computed then re-derived by independent verifiers; `--smoke` for 12/category |
| `scripts/make_v13_glm_orders.py` | builds the v13 master/orders/packets; new `extract` card + relation event-to-time extension |
| `scripts/run_v13_glm.py` | GLM-5.2 API runner (default `--model glm-5.2`), resumable |
| `scripts/build_v13_training_data.py` | builds `combined_80_20_v13`; refuses GLM-less builds unless `--allow-no-glm` or `V13_ALLOW_NO_GLM=1` |
| `scripts/lora_soup.py` | adapter averaging; output already built at `checkpoints/llama_v11_soup/final` (s42+s43+s44 uniform mean) |
| `scripts/run_schedule_v13.sh` | the 5-arm schedule; gates on v12 marker + non-provisional v13; resumable `.done` markers |
| `scripts/run_v13_prog_pilot.sh` | **RUNNING NOW** — trains on provisional v13, dev-evals, auto-scores |
| `scripts/score_v13_pilot.py` | scores the pilot with `hpo_v11.score()` vs `zs_dev` AND vs t09's dev preds (paired) |
| `scripts/run_mistral_seeds.sh` | **WAITING NOW** — mistral m05 seeds 43/44, waits for the pilot marker |

### Scripts (modified)
| Path | Change |
|---|---|
| `scripts/ingest_glm_batch.py` | +`extract` category (multi-letter gold, two-sided context check), +bare `relation`/`ordering` allowed, `extract` in ALLOC_NAMES |
| `scripts/glm_client.py` | `GLM_API_URL` env override (endpoint was hardcoded) |
| `scripts/rescore_v5_protocol.py` | registered 9 future arms: `mistral-best-s43/s44`, `soup-v11`, `v13-base/2ep/ctx/r32` (missing-file arms are skipped, not errors) |

### Configs
`experiments/finetuning/LLaMA/config_v13_{base,2ep,ctx,r32,prog_pilot}.yaml`
and `experiments/finetuning/Mistral/config_mistral_best_{s43,s44}.yaml`
(seed-only variables of the m05 recipe).

### Tests / docs
- `tests/test_v13.py` (9 tests). Full suite: **248 passed** (run
  `venv/bin/python -m pytest tests/ -q`).
- `docs/v13_plan.md` (the pre-registration), `docs/v13_glm_runbook.md`
  (one-page GLM run card), `docs/session_state.md` (appended v13 entries),
  `checkpoints/README.md` (v13 rows added).

### Validation already done
- AUG_PROG parity spot-check: all 10 categories render through
  `eval_parity`/`tokenize_example_parity` byte-exact; labels decode to
  exactly `gold + <|eot_id|>`; zero rows dropped.
- Ingest gate validated end-to-end with simulated GLM replies (valid rows
  accepted; wrong join / missing correct option / present distractor
  rejected; bare relation accepted).
- 5-step training dry-run on v13 data: passes (loss ~1.06).
- Pilot dev references verified complete: `results/hpo_v11/zs_dev` and
  `results/hpo_v11/trials/t09/preds` (5,000/2,500/10,000 per bench).

## 4. What is running RIGHT NOW (do not double-book the GPU)

One RTX 3090, serial house rule. Launched via `setsid nohup`, resumable:

1. **v13-prog pilot** — `logs/v13_prog_pilot/` (driver
   `scripts/run_v13_prog_pilot.sh`). At 20:07: training step 868/1376
   (~63%, ETA ~1h08m). Then: merge → dev evals (time/timebench/tram with
   `--ids-file data/hpo_dev/dev_ids.json`) → `scripts/score_v13_pilot.py`
   → marker `logs/v13_prog_pilot/pilot_complete.marker`. Output:
   `results/hpo_v13_pilot/score.json` (pilot vs zs objective — compare to
   t09's **+5.48pp** — and pilot vs t09 paired). It is an ABLATION
   (AUG_PROG only); NOT quotable as v13.
2. **Mistral seed replicates** — `logs/mistral_seeds/` (driver
   `scripts/run_mistral_seeds.sh`). Sleeping in a 300s loop until the
   pilot marker appears; then s43 → s44 (train ~2–3h each + full-test
   evals ~4–6h each) → rescore to
   `results/rescored/mistral_test_minus_dev.json`. Gives Mistral its
   3-seed noise floor.

## 5. The one blocker + exact unblock runbook

`GLM_API_KEY` is not set in any shell (repo policy: the key is NEVER
stored, read from env at call time only). When available:

```bash
export GLM_API_KEY='<key>'
cd /home/g2/Mohamed/temporal-news-reasoning
venv/bin/python scripts/run_v13_glm.py                          # ~109 packets on glm-5.2
venv/bin/python scripts/ingest_glm_batch.py data/glm_raw_v13/*.txt \
    --out data/manual_aug_glm_v13 --plan data/glm_packets_v13/_plan.json
venv/bin/python scripts/build_v13_training_data.py              # rebuild v13 WITH GLM rows
mkdir -p logs/sched_v13
nohup bash scripts/run_schedule_v13.sh > logs/sched_v13/driver.log 2>&1 &
```

Fallback if the API account is out of balance (glm_client measured code
1113 on paid models / 1302 rate limits on flash): the chat-UI route —
paste `MASTER_PROMPT_v13.md` into GLM-5.2 chat, send `ORDERS_v13.txt`
lines one per turn, save replies verbatim to `data/glm_raw_v13/*.txt`,
then ingest as above. Details: `docs/v13_glm_runbook.md`.

GPU priority if things collide: pilot → **v13 schedule** → mistral seeds
(interrupt the seeds driver; it is fully resumable via `.done` markers).

## 6. House rules the next session must keep

- Never change prompts/engine/decoding/scoring for fine-tuned vs zero-shot
  arms (the 2026-09-23 audit: prompt mismatch was the whole reason arms
  v1–v9 lost; `eval_parity` + pinned date "26 Jul 2024" are load-bearing).
- Quotable numbers come ONLY from
  `scripts/rescore_v5_protocol.py --exclude-ids data/hpo_dev/dev_ids.json
  --reference zs-vllm-pinned` (test-minus-dev). Dev numbers are for arm
  selection only.
- Never train/quote a PROVISIONAL v13 build as v13 (`V13_ALLOW_NO_GLM=1`
  exists for pipeline tests only).
- Pre-register changes in `docs/v13_plan.md` (recorded deviations section)
  and `docs/session_state.md`; the project's credibility is its audit trail.
- One GPU job at a time; every driver is resumable — check
  `logs/<queue>/*.done` before assuming a stage needs starting.
- GLM data goes through `scripts/ingest_glm_batch.py` — never straight
  into the mixture. Expect ~10–15% rejects; regenerate skewed batches.
- The old repo copy at `/home/g2/temporal-news-reasoning` is historical
  (its "results" predate the protocol); do not quote it.
