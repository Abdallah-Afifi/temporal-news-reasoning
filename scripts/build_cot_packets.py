"""Build GLM chat packets for CoT/STaR trace generation (a future v-cycle,
explicitly NOT part of v11 -- the user's own instruction, 2026-09-24).

Two independent packet types, matching data/cot/pilot_manual_glm53.jsonl's
proven format (98 hand-verified traces already exist there) but scaled via
the same GLM-chat generation workflow already used for AUG_GLM2:

  grounded  -- real TimeQA rows already in data/combined_80_20_v11/train.jsonl
               (so already decontaminated against every benchmark, §7 of
               audit_2026_09_23.md). GLM is given the REAL question, the REAL
               full passage and the REAL gold, and writes ONLY the reasoning
               trace -- it never re-transcribes question/context/gold, so
               there is no way for it to introduce a mismatched passage or a
               wrong gold. The ingester re-attaches the original fields by id.

  invented  -- GLM invents passage + question + gold + trace together, same
               as AUG_GLM2 (MASTER_PROMPT.md's category cards), but the
               rationale field becomes a full "Step 1: ... ANSWER: <gold>"
               trace instead of one sentence.

Usage:
    venv/bin/python scripts/build_cot_packets.py --grounded-rows 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "cot_packets"
TRAIN = PROJECT_ROOT / "data" / "combined_80_20_v11" / "train.jsonl"
PER_PACKET = 10
MAX_CONTEXT_CHARS = 15_000  # excludes the top ~5% (long-tail Wikipedia pages)
SEED = "cot-v1"

MASTER_PROMPT = """\
# Chain-of-thought trace generation -- standing brief

You are writing CHAIN-OF-THOUGHT REASONING TRACES for a temporal-reasoning
training set. This message is the STANDING BRIEF: it holds every rule for
both order types. Read it once and keep it in force for the whole
conversation. After this message I will send short ORDER lines; reply with
JSONL for that order ONLY -- no commentary, no markdown fences, no preamble.

## Output format (every trace, both modes)

One JSON object per line:

    {"id": "<the id from the order>", "cot": "<the trace>"}

The trace is 2-5 lines of the exact form:

    Step 1: <reasoning>
    Step 2: <reasoning>
    ...
    ANSWER: <final answer>

Hard rules:
- The LAST line is always `ANSWER: <answer>`, nothing after it.
- Every step must do REAL reasoning work grounded in the passage you were
  given (mode GROUNDED) or wrote yourself (mode INVENTED) -- cite the
  specific dates/facts the answer follows from. A step that just restates
  the question or announces the answer early is worthless and will be
  rejected; the gate checks for exactly this.
- The `ANSWER:` line's answer must be the CHARACTER-IDENTICAL gold you were
  given (mode GROUNDED) or the gold you invented (mode INVENTED) -- same
  wording, no rephrasing, no added units the gold lacks.
- 2-5 steps. Use as many as the question genuinely needs, not a fixed count.

## Mode GROUNDED

    ORDER: mode=grounded ids=[id1, id2, ...]

Below the order I paste a block per id:

    ### <id>
    Question: <question>
    Context: <passage>
    Gold: <gold>

Write ONLY `{"id": ..., "cot": ...}` for each -- do NOT repeat the question,
context or gold in your reply. Your reasoning must be traceable to specific
sentences in the given passage; do not use outside/pretrained knowledge to
fill a gap the passage doesn't cover -- if the passage genuinely doesn't
support the gold, skip that id and say so in one line instead of guessing.

## Mode INVENTED

    ORDER: mode=invented rows=<n> category=<name> domain="..." era=YYYY-YYYY

Same invention rules as the main AUG_GLM2 brief (data/glm_packets/MASTER_PROMPT.md):
invent the passage and events freely, no source article, same per-category
shapes (news/wiki 3-part passages with [1]/[2]/[3] sub-sections, dial for
temporal_dialogue). For THIS brief, output one extra field per row:

    {"id": "<cotNNN, sequential across the WHOLE run -- never restart the
            counter per order>", "category": "<name>",
     "passage": "<your invented passage, same shape as AUG_GLM2>",
     "question": "<question>", "gold": "<answer>",
     "cot": "Step 1: ...\\nANSWER: <gold>"}

Standing conventions for invented orders (agreed 2026-09-24, do not change
without the orchestrator):
- ids run cot001, cot002, ... continuously across every invented order in the
  run. The ingest banks by id and drops rows whose id it has already seen, so
  restarting at cot001 in a later order would silently discard that order.
- Orders default to rows=5, not 10: each invented row embeds its own full
  passage, so a 10-row order would roughly double the reply size.
- Outside a mechanics pilot, follow the AUG_GLM2 category cards IN FULL,
  including the news-shape absent-answer rule (about 1 row in 5 whose answer
  sub-passage is absent; half keep a partial gold, half abstain with a VARIED
  abstain string) and the separate near-miss sub-passage rule (about 1 row in
  5, on different rows), with their documented exceptions (Order_Reasoning,
  Timeline and Duration_Compare carry no absent rows). For an abstain row the
  ANSWER: line carries the abstain gold character-identically, and the steps
  must still cite what the passage does and does not say.

The `cot`'s reasoning must cite specific sentences of the `passage` you just
wrote (dates, names) exactly as GROUNDED mode requires -- inventing the
scenario does not relax the grounding rule, it just moves the source text
from "given to you" to "written by you in the same reply".
"""


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    pool = [r for r in rows if r["source_dataset"] == "TimeQA"
            and 0 < len(r.get("context", "")) <= MAX_CONTEXT_CHARS
            and r.get("final_answers")]
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded-rows", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1,
                    help="batch 1 rewrites the v1 files; batch >=2 appends a "
                         "disjoint block of orders (ORDERS_grounded_bN.txt) and "
                         "merges into answer_key.json")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "MASTER_PROMPT.md").write_text(MASTER_PROMPT, encoding="utf-8")

    pool = load_pool()
    print(f"grounded pool (TimeQA, v11 train, <= {MAX_CONTEXT_CHARS} chars): {len(pool)}")

    key_path = out / "answer_key.json"
    if args.batch <= 1:
        answer_key: dict[str, dict] = {}
        orders_path = out / "ORDERS_grounded.txt"
        id_offset = 0
        seed = SEED
        picked = random.Random(seed).sample(pool, min(args.grounded_rows, len(pool)))
    else:
        answer_key = json.loads(key_path.read_text(encoding="utf-8"))
        done = {(v["question"], v["context"]) for v in answer_key.values()}
        remaining = [r for r in pool if (r["question"], r["context"]) not in done]
        id_offset = max(int(k[1:]) for k in answer_key) + 1
        seed = f"{SEED}-b{args.batch}"
        orders_path = out / f"ORDERS_grounded_b{args.batch}.txt"
        picked = random.Random(seed).sample(remaining, min(args.grounded_rows, len(remaining)))
        print(f"batch {args.batch}: {len(remaining)} rows remain eligible after "
              f"excluding {len(answer_key)} already-keyed ids; sampled {len(picked)}")

    with open(orders_path, "w", encoding="utf-8") as orders_f:
        for pi in range(0, len(picked), PER_PACKET):
            chunk = picked[pi: pi + PER_PACKET]
            ids = []
            block_lines = []
            for i, r in enumerate(chunk):
                cid = f"g{id_offset + pi + i:04d}"
                gold = r["final_answers"][0] if isinstance(r["final_answers"], list) else str(r["final_answers"])
                answer_key[cid] = {"question": r["question"], "context": r["context"], "gold": gold}
                ids.append(cid)
                block_lines.append(f"### {cid}\nQuestion: {r['question']}\n"
                                   f"Context: {r['context']}\nGold: {gold}\n")
            orders_f.write(f"ORDER: mode=grounded ids={ids}\n\n" + "\n".join(block_lines) + "\n---\n\n")

    (out / "answer_key.json").write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    print(f"{len(picked)} grounded rows -> {orders_path} ({len(picked) // PER_PACKET + 1} order blocks)")
    print(f"answer key -> {out / 'answer_key.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
