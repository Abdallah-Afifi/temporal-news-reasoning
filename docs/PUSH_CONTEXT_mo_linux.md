# Context for the mo-linux session — first push of the v1..v7 cycle work

Written 2026-09-08 by the session on the Alienware box, for a Claude Code
session started on **mo-linux**. Read this first, then `docs/session_handoff.md`
(§ 4, § 4b, and D39-D54) if you need the research history.

---

## 1. The three machines

| role | host / user | repo path |
|---|---|---|
| **compute (v7 training NOW)** | `g02-s26@cse-p07-2178-g9f-Alienware-Aurora-R12` | `~/Mohamed/temporal-news-reasoning` |
| **you are here — git push box** | `mohamed-khaled@mo-linux` (100.95.193.19, tailnet) | `~/thesis/Thesis2/temporal-news-reasoning` |
| **Mistral compute (zero-shot)** | a third PC, holds the model + benchmarks | pulls code from mo-linux |

mo-linux is a **relay + git box**. It holds **no model weights and no
benchmark data**, and does not need any. Verified 2026-09-08:
`models/Mistral*` absent, `data/benchmarks/time` and `timebench` are empty
4.0K dirs, no `venv`.

## 2. The task

**Commit and push the v1..v7 cycle work. The GitHub repo has never seen any of
it.**

| | |
|---|---|
| remote | `https://github.com/Abdallah-Afifi/temporal-news-reasoning` |
| branch | `main` |
| HEAD on mo-linux | `e944a05` "Fine-Tuning Results" — **identical to origin/main, 0 behind** |
| uncommitted on mo-linux | **95 modified + 73 untracked = 168** |
| deletions | **none** (checked — nothing will be removed from the repo) |
| files > 1 MB | **none** as of the check, which means the re-sync had not yet run |

The user is a collaborator, so push access is fine.

## 3. What must NOT be pushed — already handled by .gitignore

`.gitignore` already excludes all of it; **do not weaken it**:

```
results/**/*.json   results/**/*.jsonl   results/**/*.csv
data/benchmarks/*/*      data/combined_80_20_v*/     data/corpus/
data/prepared_v4/        data/raw/    data/manual_aug/   data/letter_format/
venv/  venv_vllm/  venv_qwen/     (models/ and checkpoints/ ignored too)
```

Sizes on the compute box, for scale: `results/` **15 GB**, `models/` ~65 GB,
`data/benchmarks/` 1.3 GB, `data/corpus/` 1 GB. **A `git add -A` is safe only
because these rules exist.** Verified on the compute box: `git add -An` stages
**136 files / 29 MB**, of which 28.23 MB is one file
(`data/synthetic/news_temporal_reasoning_v8a.jsonl`) and nothing else exceeds
30 KB.

The 43 `results/` entries in the change set are **small already-tracked report
JSONs** showing as modified. No `predictions.jsonl` can slip in.

## 4. A sync gap that was found and must be closed first

The first rsync to mo-linux used a hand-written list
`(scripts src experiments docs requirements.txt)` and therefore **missed 7
files**:

```
tests/test_benchmark_loader.py           <- pins the TRAM id-collision fix
tests/test_date_equivalence.py           <- pins the bare-year guard
tests/test_prediction_postprocessing.py  <- pins the trailing-letter guard
tests/test_generation_metrics.py
tests/test_finetuning_data_loader.py
prompts/glm_news_temporal_items.md
.gitignore
```

**Pushing without these would commit the scoring fixes without the tests that
prove them.** `scripts/push_to_mo_linux.sh` now derives its file list from
`git add -An` instead, so it cannot drift again.

**Check on mo-linux before staging:**
```bash
ls tests/test_benchmark_loader.py tests/test_date_equivalence.py \
   tests/test_prediction_postprocessing.py prompts/glm_news_temporal_items.md
grep -c stfolder .gitignore
```
If any are missing, re-run `bash scripts/push_to_mo_linux.sh` **on the compute
box**, then return here.

## 5. The gate before committing

```bash
venv/bin/python -m pytest tests/ -q 2>&1 | tail -3
```
**Expect 75 passed, 4 failed.** The 4 are pre-existing `heideltime`
Java-subprocess permission failures, unrelated to any of this work. **More
than 4 means the sync is incoherent — do not commit.**

(mo-linux has no `venv`. Either build one — `python3 -m venv venv &&
venv/bin/pip install -r requirements.txt` — or run the tests on the compute
box, where they currently pass 75/4.)

## 6. What the commit contains

**Scoring — applied by identical code to every arm:**
- trailing-letter rule + letters-only guard in `_postprocess_prediction`
  (`scripts/run_baselines.py`)
- bare-year content guard in `src/evaluation/date_equivalence.py` — this one
  *removed* wrongly-credited answers and **lowered** every arm
- token-F1 reported for `situated_generation`; `Co_temporality` label folded
- partial-run guard in `scripts/rescore_v5_protocol.py`

**Data / builders:**
- `scripts/build_v7_training_data.py` — v7 mixture: single-target MCQ,
  abstain shortcut broken (100% -> 49.3%), rehearsal restored to 6,800
  context-grounded, plus `AUG_DURATION`/`AUG_RELATIVE`/`AUG_ARITH` coverage
- `src/data/synthetic_templates.py` — news generator fixed; unusable output
  **60.8% -> 0%**
- `src/data/data_loader.py` — **TRAM id collision fixed**; 31,626 duplicate ids
  were being silently skipped by resume, so a full TRAM run would have
  evaluated ~31,626 fewer items than it reported

**Harness:** `probe_answer_formats.py`, `validate_glm_news.py`,
`merge_lora.py`, `run_schedule_v7.sh`, `pilot_speed_v7.sh`,
`run_zeroshot_tram.sh`, `free_disk_for_v7.sh`, `push_to_mo_linux.sh`;
`group_by_length` and `--max-steps` added to `train.py` (both default-off, so
v1-v6 behaviour is unchanged); leakage guard widened to TRAM.

**Docs:** D39-D54 in `session_handoff.md`, plus `audit_2026_09_07.md`,
`performance_review_2026_09_07.md`, `v6_plan.md`, `v7_plan.md`, `v8_plan.md`.

## 7. Do not disturb: v7 is training on the compute box

Launched 2026-09-08 05:15:07, `config_v7.yaml`, 3 epochs, ~12-14 h.
Progress at the time of writing: **step 2450 / 4980 (49.2%)**.
Driver `logs/sched_v7/driver.log`. After training it continues unattended:
merge -> **vLLM** eval (TimeBench -> TRAM -> TIME) -> v6 on vLLM -> Mistral
parity -> zero-shot TRAM -> rescore. **Nothing on mo-linux affects it.**

## 8. Suggested commit

```bash
git add -A
git status --short | grep -vE '^ ?M|^ ?A|^\?\?' | head    # expect empty (no D)
git commit -m "v7 cycle: scoring fixes, coverage slices, vLLM schedule, TRAM id fix"
git push origin main
```

If anything unexpected appears, stage explicitly instead:
```bash
git add scripts src experiments tests docs prompts .gitignore requirements.txt data/synthetic
```
