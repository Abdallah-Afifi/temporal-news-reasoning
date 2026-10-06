"""Check AUG_GLM3 relation rows with the corpus semantics.

Events: a sentence (or date-pair) with from/to/until/between -> span event;
otherwise each D-M-Y date -> point event. E1 = first event, E2 = second.
Event-to-time: quoted time T (date, Month-YYYY or YYYY) vs the single event.

Relation of E1 to E2:
  point/point: SIMULTANEOUS (equal) else BEFORE/AFTER
  span/point:  INCLUDES if inside, else BEFORE (span ends first) / AFTER
  point/span:  IS_INCLUDED if inside, else BEFORE/AFTER
  span/span:   INCLUDES / IS_INCLUDED / SIMULTANEOUS (equal) / BEFORE/AFTER
"""
import json
import re
import sys
import calendar
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
PACKETS = ROOT / "data" / "v13_verify" / "review_packets"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
D_RE = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|"
                  r"August|September|October|November|December)\s*,?\s*(\d{4})\b", re.I)
MY_RE = re.compile(r"\b(January|February|March|April|May|June|July|"
                   r"August|September|October|November|December)\s+(\d{4})\b", re.I)
SPAN_KW = re.compile(r"\b(from|until|between|ran|covered|operated|lasted|served|"
                     r"took place|in force|to)\b", re.I)


def dmy(m):
    return (int(m[2]), MONTHS[m[1].capitalize()], int(m[0]))


def month_year(m):
    y, mo = int(m[1]), MONTHS[m[0].capitalize()]
    last = calendar.monthrange(y, mo)[1]
    return (y, mo, 1), (y, mo, last)


def parse_events(head):
    sents = [s for s in re.split(r"(?<=[.;])\s+", head) if s.strip()]
    evs = []
    for s in sents:
        dates = [dmy(m) for m in D_RE.findall(s)]
        mys = month_year_pairs = [month_year(m) for m in MY_RE.findall(s)]
        if dates and len(dates) >= 2 and SPAN_KW.search(s):
            evs.append(("span", min(dates), max(dates)))
        elif dates and len(dates) >= 2 and not SPAN_KW.search(s):
            evs.append(("point", dates[0], dates[0]))
            evs.append(("point", dates[-1], dates[-1]))
        elif dates:
            evs.append(("point", dates[0], dates[0]))
        elif len(mys) >= 2 and SPAN_KW.search(s):
            lo = min(p[0] for p in mys)
            hi = max(p[1] for p in mys)
            evs.append(("span", lo, hi))
        elif mys:
            evs.append(("span", mys[0][0], mys[0][1]))
    return evs


def parse_time(q):
    m = re.search(r"the time '([^']+)'", q)
    if not m:
        return None
    t = m.group(1)
    d = D_RE.findall(t)
    if d:
        p = dmy(d[0])
        return ("point", p, p)
    my = MY_RE.findall(t)
    if my:
        lo, hi = month_year(my[0])
        return ("span", lo, hi)
    y = re.fullmatch(r"\s*(\d{4})\s*", t)
    if y:
        return ("span", (int(y.group(1)), 1, 1), (int(y.group(1)), 12, 31))
    return None


def rel(e1, e2):
    k1, l1, h1 = e1
    k2, l2, h2 = e2
    if k1 == "point" and k2 == "point":
        if l1 == l2:
            return "SIMULTANEOUS"
        return "BEFORE" if l1 < l2 else "AFTER"
    if k1 == "span" and k2 == "point":
        if l1 <= l2 <= h1:
            return "INCLUDES"
        return "BEFORE" if h1 < l2 else "AFTER"
    if k1 == "point" and k2 == "span":
        if l2 <= l1 <= h2:
            return "IS_INCLUDED"
        return "BEFORE" if l1 < l2 else "AFTER"
    # span/span
    if l1 <= l2 and h2 <= h1:
        return "INCLUDES"
    if l2 <= l1 and h1 <= h2:
        return "IS_INCLUDED"
    if (l1, h1) == (l2, h2):
        return "SIMULTANEOUS"
    return "BEFORE" if h1 < l2 else "AFTER"


def check(r):
    q = r["question"]
    gold = r["gold"].strip().upper()
    head = q.split("What is the relationship")[0]
    if "the time '" in q:
        evs = parse_events(head)
        t = parse_time(q)
        if not evs or not t:
            return None
        return rel(evs[0], t), gold
    evs = parse_events(head)
    if len(evs) < 2:
        return None
    return rel(evs[0], evs[1]), gold


def main():
    packet = sys.argv[1]
    ok = bad = skip = 0
    for l in open(PACKETS / f"{packet}.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("category") != "relation":
            continue
        res = check(r)
        if res is None:
            skip += 1
            continue
        expect, gold = res
        if expect == gold:
            ok += 1
        else:
            bad += 1
            print(f"DISAGREE {r['rid']}: checker={expect} gold={gold}")
            print("   Q:", r["question"][:200].replace("\n", " "))
    print(f"# {packet} relation: {ok} agree, {bad} disagree, {skip} skipped")


if __name__ == "__main__":
    main()
