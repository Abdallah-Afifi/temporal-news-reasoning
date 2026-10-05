from __future__ import annotations
import re, json
from datetime import date
from dateutil import parser as dp
from dateutil.relativedelta import relativedelta
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
OPTS = ["Duration 1 is longer.", "Duration 2 is longer.",
        "The two durations are approximately the same length."]

def span(a: str, b: str) -> str:
    rd = relativedelta(dp.parse(b).date(), dp.parse(a).date())
    parts = []
    if rd.years: parts.append(f"{rd.years} year{'s' if rd.years != 1 else ''}")
    if rd.months: parts.append(f"{rd.months} month{'s' if rd.months != 1 else ''}")
    if rd.days: parts.append(f"{rd.days} day{'s' if rd.days != 1 else ''}")
    return ", ".join(parts) if parts else "0 days"

def row(pid, e1a, e1b, e2a, e2b, slot):
    s1 = span(e1a[1], e1b[1])
    s2 = span(e2a[1], e2b[1])
    d1 = (dp.parse(e1b[1]).date() - dp.parse(e1a[1]).date()).days
    d2 = (dp.parse(e2b[1]).date() - dp.parse(e2a[1]).date()).days
    if abs(d1 - d2) <= 60 or abs(d1 - d2) / max(d1, d2, 1) <= 0.10:
        gold = OPTS[2]; gi = 2
    elif d1 > d2:
        gold = OPTS[0]; gi = 0
    else:
        gold = OPTS[1]; gi = 1
    q = ("Which of the following two durations is longer? "
         f"*Duration 1:* Between the {e1a[0]} and the {e1b[0]}. "
         f"*Duration 2:* Between the {e2a[0]} and the {e2b[0]}.\nChoices:\n"
         + "\n".join(f"{L}. {t}" for L, t in zip("ABC", OPTS)))
    rat = (f"Duration 1 runs from {e1a[1]} to {e1b[1]}, a span of {s1}. "
           f"Duration 2 runs from {e2a[1]} to {e2b[1]}, a span of {s2}.")
    guards = [e[1] for e in (e1a, e1b, e2a, e2b)]
    return dict(kind="mcq", pid=pid, q=q, gold=gold, gi=gi, rat=rat,
                slot=slot, guards=guards)

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
        opts = list(OPTS)
        if r["slot"] != r["gi"]:
            g = opts.pop(r["gi"]); opts.insert(r["slot"], g)
        gold = opts[r["slot"]]
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "Duration_Compare", "provenance": prov,
            "question": r["q"], "context": "", "passage": r["pid"],
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
    from collections import Counter
    cnt = Counter()
    for r in ROWS:
        key = re.sub(r"\s+", " ", r["q"]).strip().lower()
        assert key not in seen, ("dup", key[:60]); seen.add(key)
        for d in r["guards"]:
            assert d in BASE[r["pid"]], ("date not in ctx", d)
        assert r["q"].count("*Duration 1:*") == 1
        cnt[r["gi"]] += 1
    print(f"{fno}: rows={len(ROWS)} dist(1st/2nd/same)={cnt[0]}/{cnt[1]}/{cnt[2]}")
    print(f"     wrote {OUT}")
