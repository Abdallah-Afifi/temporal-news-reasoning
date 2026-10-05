from __future__ import annotations
import re, json
from pathlib import Path
from dateutil import parser as dp

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")

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
        opts = list(r["opts"])
        if r["slot"] is not None and r["slot"] != r["gold_idx"]:
            g = opts.pop(r["gold_idx"]); opts.insert(r["slot"], g)
        gold = opts[r["slot"] if r["slot"] is not None else r["gold_idx"]]
        q = (f"What notable things did {r['subj']} do between {r['d1']} "
             f"and {r['d2']}?\nChoices:\n"
             + "\n".join(f"{L}. {t}" for L, t in zip("ABCD", opts)))
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": "Explicit_Reasoning", "provenance": prov,
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
        key = re.sub(r"\s+", " ", r["stem_key"] + r["gold_out"]).strip().lower()
        assert key not in seen, ("dup", key[:60]); seen.add(key)
        for d in r["guards"]:
            assert d in BASE[r["pid"]], ("date not in ctx", d)
        assert len(r["opts"]) == 4 and len(set(r["opts"])) == 4
        gd = dp.parse(r["gold_date"]).date()
        a, b = dp.parse(r["d1"]).date(), dp.parse(r["d2"]).date()
        assert a <= gd <= b, "gold outside window"
        for o, od in r["outside"]:
            od_ = dp.parse(od).date()
            assert not (a <= od_ <= b), "distractor inside window"
    print(f"{fno}: rows={len(ROWS)}")
    print(f"     wrote {OUT}")

def erow(pid, subj, d1, d2, events, gold_i, out_is, slot):
    """events: [(phrase, date)] chronologically; gold_i in window (d1,d2);
    out_is: 3 indices outside window."""
    ev = events
    opts = [ev[gold_i][0]] + [ev[i][0] for i in out_is]
    inside = [f"{p} on {d}" for p, d in ev
              if dp.parse(d).date() <= dp.parse(d2).date()]
    listing = ", ".join(inside)
    rat = (f"Between {d1} and {d2} the passage records these activities of "
           f"{subj}: {ev[gold_i][0]} on {ev[gold_i][1]}. The other options "
           f"are dated outside the window: "
           + "; ".join(f"{ev[i][0]} on {ev[i][1]}" for i in out_is) + ".")
    return dict(pid=pid, subj=subj, d1=d1, d2=d2, opts=opts,
                gold_idx=0, gold_out=ev[gold_i][0],
                gold_date=ev[gold_i][1], rat=rat, slot=slot,
                guards=[ev[gold_i][1]] + [ev[i][1] for i in out_is],
                outside=[(ev[i][0], ev[i][1]) for i in out_is],
                stem_key=f"{subj}{d1}{d2}")
