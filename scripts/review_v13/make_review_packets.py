"""Build review packets for the non-mechanical rows of the v13 base data.

Two slices that no script can verify from the visible text:

  glm2_lang -- AUG_GLM2 rows (the LLM-written part of the v12 corpus) in the
               language categories; reviewed with full question + context.
  timeqa    -- TimeQA rows NOT already reviewed by the CoT pass
               (data/cot_packets/answer_key.json); the Wikipedia passage is
               cut to an evidence window (sentences mentioning a year in the
               asked range +-1, the answer's name tokens, and the lead) so a
               reviewer can judge it. A NOT_FOUND verdict is NOT a removal: it
               goes to a second, full-passage round.

  math_text -- math/date rows the mechanical verifier could not parse
               (data/v13_verify/unparsed_questions.json: free-text v12 GLM
               math, a few template rows) plus AUG_GLM2 Co_temporality; the
               reviewer derives the answer from the context.
  short     -- short no-context rows: AUG_GLM (arithmetic/time word problems)
               and AUG_GLM2 bare relation / ordering rows.

Rows the mechanical verifier parsed are already settled
(scripts/verify_v13_math.py), the script-built non-math rows are removed by
the build outright, and TLQA (no context: list answers from the external
TLQA dataset) cannot be checked against any text in the row, so none of
those appear here.

Each packet row: {"rid", "slice", "source_dataset", "category", "question",
"context" | "evidence", "gold"}. rid = sha1(question)[:12], stable.
Output: data/v13_verify/review_packets/<slice>_NN.jsonl
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_v13_training_data import (MATH_CATEGORIES, SCRIPT_BUILT_FILES,  # noqa: E402
                                     script_built_questions)

SRC = ROOT / "data" / "combined_80_20_v12"
OUT = ROOT / "data" / "v13_verify" / "review_packets"
LANG = {"Counterfactual", "Explicit_Reasoning", "Localization", "Order_Reasoning",
        "duration", "longform_free", "nli_mcq", "nli_saq", "storytelling",
        "temporal_dialogue"}
PER_PACKET = {"glm2_lang": 125, "timeqa": 250, "math_text": 125, "short": 400,
              "restored": 125, "glm3": 125}
V13 = ROOT / "data" / "combined_80_20_v13"
EVIDENCE_CAP = 2200
_SENT = re.compile(r"(?<=[.!?])\s+")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def rid(q: str) -> str:
    return hashlib.sha1(q.strip().encode()).hexdigest()[:12]


def gold_of(r: dict) -> str:
    t = r.get("targets")
    if isinstance(t, str):
        try:
            t = eval(t, {"__builtins__": {}})          # stored as a python list repr
        except Exception:
            return t
    return " | ".join(map(str, t)) if isinstance(t, list) else str(t)


def evidence(question: str, context: str, gold: str) -> str:
    sents = _SENT.split(context)
    years = [int(y) for y in _YEAR.findall(question)]
    lo, hi = (min(years) - 1, max(years) + 1) if years else (None, None)
    gold_toks = {w for w in re.findall(r"[A-Z][\w'\-]{2,}", gold)}
    keep = set(range(min(2, len(sents))))
    for i, s in enumerate(sents):
        ys = [int(y) for y in _YEAR.findall(s)]
        if (lo is not None and any(lo <= y <= hi for y in ys)) or \
                any(t in s for t in gold_toks):
            keep.update({i - 1, i, i + 1} & set(range(len(sents))))
    out, n = [], 0
    for i in sorted(keep):
        if n + len(sents[i]) > EVIDENCE_CAP:
            out.append("[...]")
            break
        out.append(sents[i])
        n += len(sents[i])
    return " ".join(out)


def main() -> int:
    reviewed = {(v["question"], v["context"]) for v in
                json.loads((ROOT / "data/cot_packets/answer_key.json").read_text()).values()}
    # the original four slices exclude every glm_raw 271-313 row (kept as-is so
    # their packets stay byte-identical); those rows are GLM-written except the
    # syn_gen template files and get their own "restored" slice below.
    scripted = script_built_questions(files=range(271, 314))
    template_files = script_built_questions(files=SCRIPT_BUILT_FILES)
    unparsed = set(json.loads((ROOT / "data/v13_verify/unparsed_questions.json").read_text()))
    slices: dict[str, list[dict]] = {"glm2_lang": [], "timeqa": [], "math_text": [], "short": [],
                                     "restored": [], "glm3": []}
    seen = set()
    for split in ("train", "val"):
        for line in open(SRC / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            q = r["question"].strip()
            if q in seen:
                continue
            seen.add(q)
            src, cat = r.get("source_dataset"), r.get("category")
            base = {"rid": rid(q), "source_dataset": src, "category": cat,
                    "question": r["question"], "gold": gold_of(r)}
            if src == "AUG_GLM2" and cat in LANG and q not in scripted:
                slices["glm2_lang"].append({**base, "slice": "glm2_lang",
                                            "context": r.get("context", "")})
            elif src == "TimeQA" and (r["question"], r.get("context", "")) not in reviewed:
                slices["timeqa"].append({**base, "slice": "timeqa",
                                         "evidence": evidence(q, r.get("context", ""), base["gold"])})
            elif q in scripted:
                if q not in template_files and (cat not in MATH_CATEGORIES or q in unparsed):
                    slices["restored"].append({**base, "slice": "restored",
                                               "context": r.get("context", "")})
                continue
            elif q in unparsed or (src == "AUG_GLM2" and cat == "Co_temporality"):
                slices["math_text"].append({**base, "slice": "math_text",
                                            "context": r.get("context", "")})
            elif src == "AUG_GLM" or (src == "AUG_GLM2" and cat in ("relation", "ordering")):
                slices["short"].append({**base, "slice": "short",
                                        "context": r.get("context", "")})
    # template math rows the verifier could not parse (they live outside the v12 base)
    for f in sorted((ROOT / "data/manual_aug_glm_v13").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["question"].strip()
            if q in unparsed and q not in seen and r.get("category") in MATH_CATEGORIES:
                seen.add(q)
                slices["math_text"].append({"rid": rid(q), "slice": "math_text",
                                            "source_dataset": "AUG_TPL3",
                                            "category": r.get("category"),
                                            "question": r["question"], "gold": gold_of(r),
                                            "context": r.get("context", "")})
    # GLM-written v13 rows, as built (math rows only if the verifier couldn't parse them)
    for split in ("train", "val"):
        for line in open(V13 / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            q = r["question"].strip()
            if r.get("source_dataset") != "AUG_GLM3" or q in seen:
                continue
            if r.get("category") in MATH_CATEGORIES and q not in unparsed:
                continue
            seen.add(q)
            slices["glm3"].append({"rid": rid(q), "slice": "glm3", "source_dataset": "AUG_GLM3",
                                   "category": r.get("category"), "question": r["question"],
                                   "gold": gold_of(r), "context": r.get("context", "")})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in slices.items():
        k = PER_PACKET[name]
        for i in range(0, len(rows), k):
            p = OUT / f"{name}_{i // k:02d}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for row in rows[i:i + k]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows -> {(len(rows) + k - 1) // k} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
