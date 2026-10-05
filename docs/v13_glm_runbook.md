# v13 GLM-5.2 wave — runbook (one page)

Everything below is pre-registered in `docs/v13_plan.md` §3/§6. Order file:
`data/glm_packets_v13/ORDERS_v12.txt`-style, i.e. `ORDERS_v13.txt`
(450 orders, 10 rows each, 4,500 rows planned across 13 categories).

## Route A — API (preferred, unattended)

```bash
export GLM_API_KEY='<your key>'      # never written anywhere; repo policy
cd /home/g2/Mohamed/temporal-news-reasoning
venv/bin/python scripts/run_v13_glm.py               # 109 packets, glm-5.2
```

- Resumable: existing non-empty replies in `data/glm_raw_v13/*.txt` are
  skipped.
- `glm_client.py` serialises calls and backs off on code 1302 (rate limit).
  If the account returns code 1113 (no balance) on glm-5.2, either top up
  or switch: `--model glm-4.5-flash` (slower, rate-limited) — the packets
  are model-agnostic.
- Alternative endpoint: `export GLM_API_URL=...` (default is
  `https://open.bigmodel.cn/api/paas/v4/chat/completions`).

## Route B — GLM chat UI (the v12 fallback, works when the API can't)

1. Paste `data/glm_packets_v13/MASTER_PROMPT_v13.md` once per session.
2. Send the ORDER lines from `data/glm_packets_v13/ORDERS_v13.txt` one per
   turn (several people can share: packets are independent).
3. Save each reply verbatim to `data/glm_raw_v13/<NNN>_<Category>.txt`.

## Ingest (either route)

```bash
venv/bin/python scripts/ingest_glm_batch.py data/glm_raw_v13/*.txt \
    --out data/manual_aug_glm_v13 --plan data/glm_packets_v13/_plan.json
venv/bin/python scripts/ingest_glm_batch.py --status \
    --out data/manual_aug_glm_v13   # progress vs the plan
```

Gate notes (v13 changes): `extract` rows (multi-select, gold `B  C`) are
validated two-sided (correct options must appear in the context,
distractors must not); `relation`/`ordering` may be bare (TRAM's surface).
Expect ~10-15% rejects on a first wave — that is the gate doing its job;
regenerate skewed batches (the tool prints which).

## Rebuild + launch

```bash
venv/bin/python scripts/build_v13_training_data.py    # adds AUG_GLM3 rows
mkdir -p logs/sched_v13
nohup bash scripts/run_schedule_v13.sh > logs/sched_v13/driver.log 2>&1 &
```

The schedule refuses a PROVISIONAL build (manifest glm_rows=0) unless
`V13_ALLOW_NO_GLM=1` — do not export it for the quotable v13 run.

## While the wave runs

The v13-prog pilot (`logs/v13_prog_pilot/`) is the AUG_PROG-only ablation,
dev-scored against `results/hpo_v11/zs_dev` (t09 reference: +5.48pp mean
delta). It shares the GPU with nothing — let it finish before launching the
v13 schedule (one GPU, serial by house rule).
