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
     "cot": "Step 1: ...\nANSWER: <gold>"}

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
