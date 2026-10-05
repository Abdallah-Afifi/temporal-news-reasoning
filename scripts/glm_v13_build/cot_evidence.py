#!/usr/bin/env python3
"""For each item in a CoT order, print question/gold plus the context
sentences most relevant to answering (matching gold tokens or the time
window). Used to ground hand-written traces without reading whole wikis."""
import re
import sys

args = sys.argv[1:]
PACKET = "data/cot_packets/ORDERS_grounded_b11.txt"
if args and args[0].endswith(".txt"):
    PACKET = args.pop(0)
raw = open(PACKET, encoding="utf-8", errors="replace").read()
items = {}
for m in re.finditer(r"### (g\d+)\nQuestion: (.*?)\nContext: (.*?)\nGold: (.*?)\n", raw, re.S):
    items[m.group(1)] = (m.group(2).strip(), m.group(3).strip(), m.group(4).strip())

STOP = set("the a an of to in for from at on by with which what who whom whose was were is are had has have did does do between during until before after and or his her their its it he she they team club school university college name named called held hold helds position title employer work works worked playing played play member served serving which what whom".split())

def sentences(ctx):
    return re.split(r"(?<=[.!?])\s+", ctx)

def score(s, qtok, gtok, years):
    sc = 0
    low = s.lower()
    for t in gtok:
        if t in low: sc += 3
    for t in qtok:
        if t in low: sc += 1
    for y in years:
        if y in s: sc += 2
    return sc

def relevant(q, ctx, gold, k=4):
    qtok = {t for t in re.findall(r"[a-z]+", q.lower()) if t not in STOP}
    gtok = {t for t in re.findall(r"[a-z]+", gold.lower()) if t not in STOP and len(t) > 2}
    years = re.findall(r"\b(1[6-9]\d\d|20[0-2]\d)\b", q)
    sents = sentences(ctx)
    ranked = sorted(((score(s, qtok, gtok, years), i, s) for i, s in enumerate(sents)), reverse=True)
    top = [s for sc, i, s in ranked[:k] if sc > 0]
    top.sort(key=lambda s: sents.index(s))
    return top

for gid in args:
    if gid not in items:
        print(f"### {gid}: NOT FOUND"); continue
    q, ctx, gold = items[gid]
    print(f"### {gid}")
    print(f"Q: {q}")
    print(f"GOLD: {gold}")
    for s in relevant(q, ctx, gold):
        print(f"  EV: {s.strip()}")
    print()
