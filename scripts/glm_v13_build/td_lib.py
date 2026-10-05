from __future__ import annotations
import re, json
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")

def build(fno, SUBS, ROWS, band=(400, 700)):
    BASE = {pid: (s if isinstance(s, str) else "\n".join(s)) for pid, s in SUBS.items()}
    OUT = ROOT / f"data/glm_raw/{fno}.txt"
    out = []
    for pid, txt in BASE.items():
        out += [f"=== PASSAGE {pid} ===", txt]
    for r in ROWS:
        if r["kind"] == "mcq":
            opts = list(r["opts"])
            slot = r.get("slot")
            if slot is not None:
                g = opts.pop(r["gold_idx"]); opts.insert(slot, g)
            else:
                g = opts[r["gold_idx"]]
            gold = g
            q = r["stem"] + "\nChoices:\n" + "\n".join(
                f"{L}. {t}" for L, t in zip("ABCD", opts))
        else:
            q = r["q"]; gold = r["gold"]
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "temporal_dialogue", "provenance": "dial",
            "question": q, "context": "", "passage": r["pid"],
            "targets": [gold], "rationale": r["rat"],
            "source": "augmented", "source_id": ""}, ensure_ascii=True))
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")

    wc = lambda t: len(t.split())
    for pid, txt in BASE.items():
        assert band[0] <= wc(txt) <= band[1], ("base words", pid, wc(txt))
        assert all(ord(c) < 128 for c in txt)
    seen = set()
    for r in ROWS:
        key = re.sub(r"\s+", " ",
                     r.get("stem", "") + r.get("q", "") + r.get("gold", "")).strip().lower()
        assert key not in seen, ("dup", key[:60]); seen.add(key)
        head = (r.get("stem", "") + r.get("q", ""))[:len(r["rat"]) + 50]
        assert r["rat"] not in head
        if r["kind"] == "mcq":
            assert len(r["opts"]) == 4 and len(set(r["opts"])) == 4
            assert r["gold_idx"] < 4
        if r["kind"] == "ft":
            assert "?" in r["q"]
            assert len(r["gold"].split()) <= 12
        if r.get("guard"):
            assert r["guard"] in BASE[r["pid"]], ("guard", r["guard"])
    mcq = sum(1 for r in ROWS if r["kind"] == "mcq")
    print(f"{fno}: mcq={mcq} ft={len(ROWS) - mcq} rows={len(ROWS)}")
    print(f"     wrote {OUT}")

def sess_mcq(pid, stem, gold_session, opt_sessions, slot, rat, guard):
    opts = list(opt_sessions)
    assert gold_session in opts
    gi = opts.index(gold_session)
    return dict(kind="mcq", pid=pid, stem=stem, opts=opts, gold_idx=gi,
                rat=rat, slot=slot, guard=guard)

def sess_ft(pid, q, gold, rat, guard):
    return dict(kind="ft", pid=pid, q=q, gold=gold, rat=rat, guard=guard)
