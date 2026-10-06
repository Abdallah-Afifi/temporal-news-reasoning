"""Check relation and duration rows in glm3 packets.

relation: parse dates/spans from the question; E1-vs-E2 order, or event-span
vs time-point (INCLUDES / IS_INCLUDED / BEFORE / AFTER) per the brief.
duration: 'ran from X to Y' -> year difference vs gold option.

Prints disagreements for manual review.
Usage: python3 check_rel_dur.py <packet>
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
PACKETS = ROOT / "data" / "v13_verify" / "review_packets"

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
D_RE = re.compile(r"(\d{1,2})\s+(January|February|March|April|May|June|July|"
                  r"August|September|October|November|December)\s*,?\s*(\d{4})", re.I)
Y_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")

NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def date_tuples(text):
    return [(int(y), MONTHS[m.lower()], int(d)) for d, m, y in D_RE.findall(text)]


def year_ints(text):
    return [int(y) for y in Y_RE.findall(text)]


def check_relation(r):
    q = r["question"]
    gold = r["gold"].strip().upper()
    dates = date_tuples(q)
    ev_event = "between the event '" in q and "' and the time" in q
    if ev_event and len(dates) >= 2:
        # span event [d1..d2] vs time point (last-mentioned year or date)
        span = sorted(dates[:2])
        rest = dates[2:]
        if rest:
            t = rest[0]
        else:
            ys = year_ints(q)
            if not ys:
                return None
            t = (ys[-1], 6, 15)
        lo, hi = span[0], span[1]
        if lo <= t <= hi and (t[0],) < (lo[0],):  # never true; placeholder
            pass
        if lo < t < hi or lo <= t <= hi:
            expect = "INCLUDES"
        else:
            expect = None
        if expect is None:
            if t < lo:
                expect = "BEFORE"
            else:
                expect = "AFTER"
        return expect, gold
    if len(dates) >= 2:
        expect = "BEFORE" if dates[0] < dates[1] else "AFTER"
        return expect, gold
    return None


def check_duration(r):
    q = r["question"]
    gold = r["gold"].strip().lower()
    ys = year_ints(q)
    if "from" in q and len(set(ys)) >= 2:
        lo, hi = min(ys), max(ys)
        diff = hi - lo
        # match against options containing that number
        opts = re.findall(r"([A-E])\.\s*(.+)", q)
        for _, text in opts:
            t = text.lower()
            m = re.match(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+year", t)
            if m:
                n = int(m.group(1)) if m.group(1).isdigit() else NUMW[m.group(1)]
                if n == diff:
                    return (gold == t), t
        return None
    return None


def main():
    packet = sys.argv[1]
    ok = bad = skip = 0
    for l in open(PACKETS / f"{packet}.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        cat = r.get("category")
        res = None
        if cat == "relation":
            res = check_relation(r)
        elif cat == "duration":
            res = check_duration(r)
        if res is None:
            skip += 1
            continue
        if isinstance(res, tuple) and res[0] is True:
            ok += 1
        elif isinstance(res, tuple) and res[0] is False:
            bad += 1
            print(f"DURATION-DISAGREE {r['rid']}: gold={r['gold']!r} checker={res[1]!r}")
        else:
            expect, gold = res
            if expect == gold:
                ok += 1
            else:
                bad += 1
                print(f"REL-DISAGREE {r['rid']}: checker={expect} gold={gold}")
                print("   Q:", r["question"][:220].replace("\n", " "))
    print(f"# {packet}: {ok} agree, {bad} disagree, {skip} skipped")


if __name__ == "__main__":
    main()
