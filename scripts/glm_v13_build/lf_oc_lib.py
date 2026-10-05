from __future__ import annotations
import re, json
from datetime import date
from dateutil import parser as dp
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")

def passages_out(out, BASE):
    for pid, txt in BASE.items():
        out += [f"=== PASSAGE {pid} ===", txt]
    return out

def check_band(fno, prov, BASE, band, rows):
    wc = lambda t: len(t.split())
    for pid, txt in BASE.items():
        assert band[0] <= wc(txt) <= band[1], ("base words", pid, wc(txt))
        if prov == "news":
            for n in (1, 2, 3):
                assert f"[{n}]" in txt
        assert all(ord(c) < 128 for c in txt)
    seen = set()
    for r in rows:
        k = re.sub(r"\s+", " ", r.get("q", r.get("e1", "") + r.get("d1", "") + r.get("e2", "") + r.get("d2", ""))).strip().lower()
        assert k not in seen, ("dup", k[:60]); seen.add(k)

def build_lf(fno, prov, SUBS, ROWS, band=(700, 1000)):
    news = prov == "news"
    BASE = {pid: "\n".join(s) for pid, s in SUBS.items()} if news else dict(SUBS)
    out = passages_out([], BASE)
    for r in ROWS:
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "longform_free", "provenance": prov,
            "question": r["q"], "context": "", "passage": r["pid"],
            "targets": [r["gold"]], "rationale": r["rat"],
            "source": "augmented", "source_id": ""}, ensure_ascii=True))
    OUT = ROOT / f"data/glm_raw/{fno}.txt"
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")
    check_band(fno, prov, BASE, band, ROWS)
    for r in ROWS:
        assert "?" in r["q"]
        for d in r["dates"]:
            assert d in BASE[r["pid"]], ("date not in ctx", d)
        a, b = dp.parse(r["d1"]).date(), dp.parse(r["d2"]).date()
        for d in r["dates"]:
            dd = dp.parse(d).date()
            assert a <= dd <= b, "event outside window"
    print(f"{fno}: rows={len(ROWS)}")
    print(f"     wrote {OUT}")

def lf_row(pid, q, d1, d2, evs):
    """evs: [(sentence, date)] chronological, all within (d1, d2)."""
    gold = " ".join(s for s, _ in evs)
    rat = ("The passage reports these developments in order: "
           + "; ".join(d for _, d in evs) + ".")
    return dict(pid=pid, q=q, d1=d1, d2=d2, gold=gold, rat=rat,
                dates=[d for _, d in evs])

def build_oc(fno, prov, SUBS, ROWS, band=(700, 1000)):
    news = prov == "news"
    BASE = {pid: "\n".join(s) for pid, s in SUBS.items()} if news else dict(SUBS)
    out = passages_out([], BASE)
    OPTS = ["Fact 1 happened earlier.", "Fact 2 happened earlier.",
            "They happen at almost the same time."]
    for r in ROWS:
        opts = list(OPTS)
        if r["slot"] != r["gi"]:
            g = opts.pop(r["gi"]); opts.insert(r["slot"], g)
        gold = opts[r["slot"]]
        q = (f"For Fact1: {r['e1']} and Fact2: {r['e2']}, which one "
             "happened earlier?\nChoices:\n"
             + "\n".join(f"{L}. {t}" for L, t in zip("ABC", opts)))
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "Order_Compare", "provenance": prov,
            "question": q, "context": "", "passage": r["pid"],
            "targets": [gold], "rationale": r["rat"],
            "source": "augmented", "source_id": ""}, ensure_ascii=True))
    OUT = ROOT / f"data/glm_raw/{fno}.txt"
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")
    check_band(fno, prov, BASE, band, ROWS)
    from collections import Counter
    c = Counter()
    for r in ROWS:
        for d in (r["d1"], r["d2"]):
            assert d in BASE[r["pid"]], ("date not in ctx", d)
        c[r["gi"]] += 1
    print(f"{fno}: rows={len(ROWS)} dist(F1/F2/same)={c[0]}/{c[1]}/{c[2]}")
    print(f"     wrote {OUT}")

def oc_row(pid, e1, d1, e2, d2, slot):
    a, b = dp.parse(d1).date(), dp.parse(d2).date()
    diff = abs((a - b).days)
    if diff == 0:
        gi = 2
        rat = (f"The passage dates the fact that {e1} to {d1} and the fact "
               f"that {e2} to {d2}; the two dates are identical, so they "
               f"happen at almost the same time.")
    elif diff <= 14:
        gi = 2
        rat = (f"The passage dates the fact that {e1} to {d1} and the fact "
               f"that {e2} to {d2}; the two dates are only {diff} days "
               f"apart, so they happen at almost the same time.")
    elif a < b:
        gi = 0
        rat = (f"The passage dates the fact that {e1} to {d1} and the fact "
               f"that {e2} to {d2}; the first date precedes the second, so "
               f"Fact 1 happened earlier.")
    else:
        gi = 1
        rat = (f"The passage dates the fact that {e1} to {d1} and the fact "
               f"that {e2} to {d2}; the second date precedes the first, so "
               f"Fact 2 happened earlier.")
    return dict(pid=pid, e1=e1, d1=d1, e2=e2, d2=d2, gi=gi, slot=slot,
                rat=rat)
