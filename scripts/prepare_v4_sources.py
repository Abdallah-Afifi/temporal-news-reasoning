"""Prepare raw external datasets into the project training template.

Converts each downloaded source into {source_dataset, question, context,
targets, source} JSONL under data/prepared_v4/, with:
  - native-schema conversion (SQuAD / CoQA / DROP / CoT-Collection /
    HotpotQA / SlimOrca)
  - temporal relevance filtering where the source is general-purpose
  - self-contained question construction (dialogue history embedded)
  - dedup within source (normalized question) + benchmark collision guard
  - per-source stats printed and stored in data/prepared_v4/stats.json

Capping and final mixture assembly happen later in
build_v4_training_data.py; this preparer emits full filtered pools.

Usage: venv/bin/python scripts/prepare_v4_sources.py --sources squad,coqa,drop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_v2_training_data import benchmark_questions, norm  # noqa: E402

OUT_DIR = REPO / "data" / "prepared_v4"
RAW = REPO / "data" / "raw"

TEMPORAL_RE = re.compile(
    r"\b(\d{4}|18\d{2}|19\d{2}|20\d{2}|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|yesterday|tomorrow|today|week|month|year|decade|century|"
    r"hour|minute|day|before|after|during|between|earlier|later|first|last|"
    r"dawn|noon|midnight|season|spring|summer|autumn|winter|calendar|clock|"
    r"anniversary|birthday|ago|hence)\b",
    re.IGNORECASE,
)


def rec(question: str, context: str, targets, source: str) -> dict:
    return {
        "source_dataset": source,
        "question": question.strip(),
        "context": (context or "").strip(),
        "targets": targets,
        "source": "real",
    }


def prepare_squad() -> list[dict]:
    d = json.load(open(RAW / "squad_train-v1.1.json"))
    out = []
    for art in d["data"]:
        for para in art["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                q = qa["question"].strip()
                ans = [a["text"] for a in qa["answers"] if a.get("text")]
                if not q or not ans:
                    continue
                # temporal relevance: question OR answer carries time signal
                if not (TEMPORAL_RE.search(q) or TEMPORAL_RE.search(ans[0])):
                    continue
                out.append(rec(q, ctx, ans[:3], "SQUAD_T"))
    return out


def prepare_coqa() -> list[dict]:
    d = json.load(open(RAW / "coqa-train-v1.0.json"))
    out = []
    for story in d["data"]:
        ctx = story["story"]
        qs, ans_list = story["questions"], story["answers"]
        history: list[str] = []
        for i, (q, a) in enumerate(zip(qs, ans_list)):
            turn_ans = a.get("input_text") or a.get("span_text") or ""
            if not turn_ans or turn_ans.lower() in ("unknown", "yes", "no") and i < 2:
                pass  # keep short answers; they are legitimate
            q_text = q["input_text"].strip()
            if not q_text:
                continue
            # embed recent dialogue history so the record is self-contained
            if history:
                q_full = "Conversation so far:\n" + "\n".join(history[-6:]) + \
                         "\nQuestion: " + q_text
            else:
                q_full = q_text
            out.append(rec(q_full, ctx, [turn_ans], "COQA"))
            history.append(f"Q: {q_text}\nA: {turn_ans}")
    return out


def prepare_drop() -> list[dict]:
    z = zipfile.ZipFile(RAW / "drop_dataset.zip")
    name = [n for n in z.namelist() if "train" in n][0]
    d = json.loads(z.read(name))
    out = []
    for pid, block in d.items():
        ctx = block["passage"]
        for qa in block["qa_pairs"]:
            q = qa["question"].strip()
            ans = qa.get("answer", {})
            if ans.get("number"):
                targets = [str(ans["number"])]
            elif ans.get("date"):
                dt = ans["date"]
                targets = [" ".join(str(v) for v in dt.values() if v)]
            else:
                spans = [
                    s if isinstance(s, str) else s.get("text", "")
                    for s in (ans.get("spans") or [])
                ]
                spans = [s for s in spans if s]
                targets = spans[:3]
            if q and targets:
                out.append(rec(q, ctx, targets, "DROP"))
    return out


def prepare_cot_collection() -> list[dict]:
    candidates = list((RAW / "cot_collection").rglob("CoT_collection_en.json"))
    if not candidates:
        raise FileNotFoundError("CoT_collection_en.json not under data/raw/cot_collection")
    d = json.load(open(candidates[0]))
    out = []
    for key, item in d.items():
        q = (item.get("source") or item.get("question") or "").strip()
        a = (item.get("target") or item.get("answer") or "").strip()
        if not q or not a:
            continue
        r = rec(q, "", [a], "COTCOLL")
        r["rationale"] = (item.get("rationale") or "")[:1500]
        r["task"] = item.get("task") or ""
        out.append(r)
    return out


def prepare_hotpot() -> list[dict]:
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(RAW / "hotpot_qa" / "**" / "*.parquet"),
                             recursive=True))
    out = []
    for fp in files:
        tbl = pq.read_table(fp, columns=["question", "answer", "context"])
        data = tbl.to_pylist()
        for item in data:
            q = item["question"].strip()
            a = item.get("answer", "")
            ctx_paras = item.get("context") or {}
            ctx = ""
            if isinstance(ctx_paras, dict):
                titles = ctx_paras.get("title") or []
                sents = ctx_paras.get("sentences") or []
                parts = []
                for t, ss in list(zip(titles, sents))[:4]:
                    parts.append(f"{t}. {' '.join(ss)}")
                ctx = " ".join(parts)[:4000]
            if q and a and a.lower() not in ("yes", "no"):
                out.append(rec(q, ctx, [a], "HOTPOT"))
    return out


def prepare_slimorca() -> list[dict]:
    import glob

    files = [
        f for f in glob.glob(str(RAW / "slimorca" / "**" / "*.jsonl"),
                             recursive=True)
        if "/.cache/" not in f
    ]
    out = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            convs = row.get("conversations", [])
            if len(convs) < 2:
                continue
            human = next((c for c in convs if c.get("from") == "human"), None)
            gpt = next((c for c in convs if c.get("from") == "gpt"), None)
            if not human or not gpt:
                continue
            q, a = human["value"].strip(), gpt["value"].strip()
            if not q or not a or len(a) > 600:
                continue
            out.append(rec(q, "", [a], "SLIMORCA"))
    return out


PREPARERS = {
    "squad": prepare_squad,
    "coqa": prepare_coqa,
    "drop": prepare_drop,
    "cotcoll": prepare_cot_collection,
    "hotpot": prepare_hotpot,
    "slimorca": prepare_slimorca,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="squad,coqa,drop")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = benchmark_questions()
    print(f"collision guard loaded: {len(bench)} benchmark questions")

    stats = {}
    if (OUT_DIR / "stats.json").exists():
        stats = json.load(open(OUT_DIR / "stats.json"))

    for src in args.sources.split(","):
        src = src.strip().lower()
        fn = PREPARERS.get(src)
        if fn is None:
            print(f"unknown source: {src}")
            continue
        print(f"preparing {src} ...", flush=True)
        try:
            rows = fn()
        except FileNotFoundError as e:
            print(f"  SKIP ({e})")
            continue
        seen, kept, coll = set(), [], 0
        for r in rows:
            nq = norm(r["question"])
            if nq in seen or nq in bench:
                coll += (nq in bench)
                continue
            seen.add(nq)
            kept.append(r)
        out_path = OUT_DIR / f"{src}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[src] = {"records": len(kept), "collisions": coll,
                      "raw": len(rows)}
        print(f"  {src}: raw={len(rows)} kept={len(kept)} "
              f"collisions-removed={coll} -> {out_path.name}")

    json.dump(stats, open(OUT_DIR / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
