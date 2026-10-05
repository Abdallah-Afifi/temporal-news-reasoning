#!/usr/bin/env python3
"""Generate the three remaining temporal_dialogue packets (125 rows each).
Arithmetic rows in dialogue form; every row verified with the ingest gate."""
import importlib.util
import json
import random
import re
import sys
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

SPEC = importlib.util.spec_from_file_location(
    "ing", "/home/g2/Mohamed/temporal-news-reasoning/scripts/ingest_glm_batch.py")
ING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ING)

MONTHS = ["January","February","March","April","May","June","July",
          "August","September","October","November","December"]
NUMW = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",
        8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",
        14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",
        19:"nineteen",20:"twenty"}
def nw(n):
    return NUMW.get(n, str(n))

PACKETS = [
 ("033_temporal_dialogue", "archaeology", 2005, 2014),
 ("034_temporal_dialogue", "corporate", 1966, 1989),
 ("035_temporal_dialogue", "sport", 1890, 1935),
]

SUBJ = {
 "archaeology": ["the excavation", "the museum's new gallery", "the dig season",
    "the conservation programme", "the field survey", "the heritage grant",
    "the finds report", "the scheduled monument", "the society's lecture series", "the summer training dig"],
 "corporate": ["the merger", "the debenture issue", "the plant closure",
    "the record dividend", "the takeover bid", "the wage settlement",
    "the new factory", "the export drive", "the board reshuffle",
    "the centenary campaign"],
 "sport": ["the cup run", "the championship season", "the new grandstand",
    "the manager's reign", "the promotion campaign", "the overseas tour",
    "the floodlight project", "the signing of the centre-forward",
    "the ground move", "the anniversary match"],
}
SENT = {
 "archaeology": ["began", "opened", "closed", "was reported", "was scheduled",
    "was completed", "was extended", "was launched"],
 "corporate": ["was announced", "took effect", "was completed", "was settled",
    "began", "was deferred", "was ratified", "collapsed"],
 "sport": ["began", "ended", "opened", "finished", "was abandoned", "started",
    "was completed", "was postponed"],
}

def dstr(d): return f"{d.day} {MONTHS[d.month-1]} {d.year}"

def fmt_span(rd):
    parts=[]
    if rd.years: parts.append(f"{nw(rd.years)} year"+("s" if rd.years>1 else ""))
    if rd.months: parts.append(f"{nw(rd.months)} month"+("s" if rd.months>1 else ""))
    if rd.days: parts.append(f"{nw(rd.days)} day"+("s" if rd.days>1 else ""))
    return " ".join(parts) or "zero days"

SEEN=set()

def row_span(rng, subj, s, a, b):
    rd = relativedelta(b, a)
    span = fmt_span(rd)
    total = (b-a).days
    # distractors: perturb the unit magnitude wildly
    cands = set()
    if rd.days and not rd.months and not rd.years:
        cands.update([f"{nw(rd.days)} hours", f"{nw(rd.days)} weeks"] if rd.days>1
                     else ["a minute","a month"])
        if rd.days <= 10: cands.add(f"{nw(rd.days)} months")
        if rd.days <= 5:  cands.add(f"{nw(rd.days)} years")
    elif rd.months and not rd.years:
        cands.update([f"{nw(rd.months)} weeks", f"{nw(rd.months)} days"])
        if rd.months<=6: cands.add(f"{nw(rd.months)} years")
        cands.add(f"{nw(max(1,rd.months//2))} months" if rd.months>2 else "a decade")
    else:
        cands.update([f"{nw(rd.years)} months", f"{nw(rd.years)} weeks"])
        if rd.years<=10: cands.add(f"{nw(rd.years*2)} years")
    cands.discard(span)
    cands.discard(None)
    fallback = ["a moment", "a decade", "a century", "half an hour",
                "three centuries", "a week last winter"]
    while len(cands) < 3:
        cands.add(fallback[len(cands)])
    dis = rng.sample(sorted(cands), 3)
    opts=[span]+dis
    rng.shuffle(opts)
    q = (f"A: The {subj} {s} on {dstr(a)} and finished on {dstr(b)}. "
         f"B: So it lasted <MASK>?\nChoices:\n" + "\n".join(
             f"{'ABCD'[i]}. {o}" for i,o in enumerate(opts)))
    rat = (f"From {dstr(a)} to {dstr(b)} is {span}, so the {subj} lasted {span}; "
           f"the other options are on entirely different scales.")
    return q, span, rat

def row_offset(rng, subj, s, a, rd, back=False):
    target = a - rd if back else a + rd
    span = fmt_span(rd)
    correct = f"{MONTHS[target.month-1]} {target.year}"
    dis = set()
    off2 = relativedelta(years=rd.years+1, months=rd.months, days=rd.days) if rd.years else None
    wrong_y = target.year + rng.choice([-2,-1,1,3])
    dis.add(f"{MONTHS[target.month-1]} {wrong_y}")
    other_m = (target.month % 12) + 1
    dis.add(f"{MONTHS[other_m-1]} {target.year}")
    dis.add("next Tuesday" if rng.random()<0.5 else "a fortnight")
    opts=[correct]+sorted(dis)
    rng.shuffle(opts)
    q = (f"A: The {subj} {s} on {dstr(a)} and ran for {span}. "
         f"B: So it {'started' if back else 'finished'} in <MASK>?\nChoices:\n"
         + "\n".join(f"{'ABCD'[i]}. {o}" for i,o in enumerate(opts)))
    rat = (f"{dstr(a)} {'minus' if back else 'plus'} {span} is {MONTHS[target.month-1]} "
           f"{target.year}; the other options change the scale or the year.")
    return q, correct, rat

def make_packet(fname, domain, y0, y1, n=125):
    rng = random.Random(fname)
    rows, guard = [], 0
    while len(rows) < n:
        guard += 1
        if guard > 4000:
            raise RuntimeError(f"{fname}: stalled at {len(rows)}")
        subj = rng.choice(SUBJ[domain])
        s = rng.choice(SENT[domain])
        a = date(rng.randint(y0, y1-1), rng.randint(1,12), rng.randint(1,28))
        kind = rng.random()
        if kind < 0.45:
            b = a + timedelta(days=rng.choice([3,5,7,9,11,14,17,21,25,40,60,90]))
            if b.year > y1+1: continue
            q, gold, rat = row_span(rng, subj, s, a, b)
        else:
            style = rng.random()
            if style < 0.4:
                rd = relativedelta(years=rng.randint(1,6))
            elif style < 0.75:
                rd = relativedelta(months=rng.choice([2,3,5,7,9,11,18]))
            else:
                rd = relativedelta(days=rng.choice([45,60,80,100,120,150]))
            back = rng.random() < 0.35
            q, gold, rat = row_offset(rng, subj, s, a, rd, back)
        nq = re.sub(r"\s+"," ",q).strip().lower()
        if nq in SEEN: continue
        r = {"source_dataset":"AUG_GLM2","slice":"C","category":"temporal_dialogue",
             "provenance":"dial","question":q,"context":"","passage":"",
             "targets":[gold],"rationale":rat,"source":"augmented","source_id":""}
        errs = ING.check(r)
        if errs:
            raise RuntimeError(f"{fname}: {errs}\n{json.dumps(r,indent=1)}")
        SEEN.add(nq)
        rows.append(r)
    out = "/home/g2/Mohamed/temporal-news-reasoning/data/glm_raw_v13/" + fname + ".txt"
    with open(out,"w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"wrote {out}: {len(rows)} rows")

for fname, dom, y0, y1 in PACKETS:
    make_packet(fname, dom, y0, y1)
