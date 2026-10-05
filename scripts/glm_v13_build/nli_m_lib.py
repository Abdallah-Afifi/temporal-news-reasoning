from __future__ import annotations
import re, json
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
LABELS = ["entailment", "neutral", "contradiction"]

def build(fno, prov, SUBS, ROWS, band=(700, 1000)):
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
        opts = list(LABELS)
        gi = opts.index(r["label"])
        slot = r.get("slot", 0)
        if slot != gi:
            g = opts.pop(gi); opts.insert(slot, g)
        q = (f"Premise: {r['prem']}\nHypothesis: {r['hyp']}\nChoices:\n"
             + "\n".join(f"{L}. {t}" for L, t in zip("ABC", opts)))
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "B",
            "category": "nli_mcq", "provenance": prov,
            "question": q, "context": "", "passage": r["pid"],
            "targets": [r["label"]], "rationale": r["rat"],
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
        assert r["label"] in LABELS
        q = f"Premise: {r['prem']}\nHypothesis: {r['hyp']}"
        key = re.sub(r"\s+", " ", q).strip().lower()
        assert key not in seen, ("dup", key[:60]); seen.add(key)
        assert r["rat"] and r["rat"] not in q[:len(r["rat"]) + 50]
        assert r["prem"][:40] not in r["hyp"]
        if r.get("guard"):
            assert r["guard"] in BASE[r["pid"]], ("guard", r["guard"])
        assert 0 <= r.get("slot", 0) <= 2
    from collections import Counter
    c = Counter(r["label"] for r in ROWS)
    s = Counter(r.get("slot", 0) for r in ROWS)
    print(f"{fno}: rows={len(ROWS)} labels={dict(c)} slots={dict(sorted(s.items()))}")
    print(f"     wrote {OUT}")
