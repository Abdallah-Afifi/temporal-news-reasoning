from __future__ import annotations
import re, json
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")

def build_st(fno, prov, SUBS, ROWS, band=(700, 1000)):
    news = prov == "news"
    if news:
        BASE = {pid: "\n".join(s) for pid, s in SUBS.items()}
    else:
        BASE = dict(SUBS)
    OUT = ROOT / f"data/glm_raw/{fno}.txt"
    out = []
    for pid, txt in BASE.items():
        out += [f"=== PASSAGE {pid} ===", txt]
    for r in ROWS:
        opts = [r["eA"], r["eB"]]
        q = ("Which of the two endings is the most plausible correct ending "
             "to the story?\nChoices:\n"
             + "\n".join(f"{L}. {t}" for L, t in zip("AB", opts)))
        gold = r["eA"] if r["slot"] == 0 else r["eB"]
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "storytelling", "provenance": prov,
            "question": q, "context": "", "passage": r["pid"],
            "targets": [gold], "rationale": r["rat"],
            "source": "augmented", "source_id": ""}, ensure_ascii=True))
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")

    wc = lambda t: len(t.split())
    for pid, txt in BASE.items():
        assert band[0] <= wc(txt) <= band[1], ("base words", pid, wc(txt))
        if news:
            for n in (1, 2, 3):
                assert f"[{n}]" in txt
        assert all(ord(c) < 128 for c in txt)
    seen = set()
    for r in ROWS:
        key = r["eA"][:100].strip().lower()
        assert key not in seen, ("dup", key[:60]); seen.add(key)
        assert r["eA"] != r["eB"]
        assert len(r["eA"].split()) <= 60
        assert r["guard"] in BASE[r["pid"]], ("guard not in ctx", r["guard"])
        assert str(r["correct"]) in BASE[r["pid"]], ("detail not in ctx", r["correct"])
    from collections import Counter
    c = Counter(r["slot"] for r in ROWS)
    print(f"{fno}: rows={len(ROWS)} slots={dict(sorted(c.items()))}")
    print(f"     wrote {OUT}")

def st_row(pid, frame, correct, wrong, slot):
    eA = frame.format(v=correct)
    eB = frame.format(v=wrong)
    rat = (f"The passage states the detail the endings differ on: "
           f"{correct}. The ending carrying {correct} matches the "
           f"passage; {wrong} contradicts it.")
    return dict(pid=pid, eA=eA, eB=eB, rat=rat, slot=slot,
                correct=correct, guard=str(correct))
