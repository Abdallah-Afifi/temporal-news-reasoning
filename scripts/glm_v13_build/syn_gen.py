from __future__ import annotations
import json, random
from datetime import date
from dateutil import parser as dp
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")

def write_rows(fno, rows):
    OUT = ROOT / f"data/glm_raw/{fno}.txt"
    out = []
    for r in rows:
        out.append(json.dumps({
            "source_dataset": "AUG_GLM2", "slice": "A",
            "category": r["cat"], "provenance": "none",
            "question": r["q"], "context": "", "passage": "",
            "targets": [r["gold"]], "rationale": r["rat"],
            "source": "augmented", "source_id": ""}, ensure_ascii=True))
    OUT.write_text("\n".join(out) + "\n", encoding="ascii")
    seen = set()
    for r in rows:
        k = r["q"][:120].lower()
        assert k not in seen, ("dup", k[:60])
        seen.add(k)
    from collections import Counter
    c = Counter(r["gold"] for r in rows)
    print(f"{fno}: rows={len(rows)} golds={dict(c)}")
    print(f"     wrote {OUT}")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

def md(d):
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"

# ---------------- relation (296): space domain 2015-2019
SPACE_POINTS = [
 "the Kestrel array's first dish was commissioned",
 "the mission control room was certified",
 "the outreach planetarium opened its dome",
 "the tracking station's first lock was achieved",
 "the observatory's open evening programme began",
 "the cubesat's launch licence was granted",
 "the deep-space network membership was ratified",
 "the student rocketry lab was inaugurated",
 "the mirror coating plant passed its tests",
 "the southern sky survey started its first season",
]
SPACE_INTERVALS = [
 "the calibration campaign",
 "the instrument commissioning window",
 "the survey's second season",
 "the tracking handover trial",
 "the antenna renewal programme",
]

def relation_rows(n_each=25):
    rng = random.Random(296)
    rows = []
    used = set()
    def fresh_pt():
        while True:
            p = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))]
            y = rng.randrange(2015, 2020)
            m = rng.randrange(1, 13)
            d = date(y, m, rng.randrange(1, 28))
            if (p, d) not in used:
                used.add((p, d))
                return p, d
    # IDENTITY
    for _ in range(n_each):
        p1, d1 = fresh_pt()
        p2, d2 = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))], d1
        while p2 == p1:
            p2 = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))]
        q = (f"{p1.capitalize()} on {md(d1)}, and {p2} on the same day. "
             "What is the relationship between the events' dates?\nChoices:\n"
             "A. IDENTITY\nB. BEFORE\nC. DURING")
        rat = (f"Both events are dated {md(d1)}: the two dates are "
               f"identical, so the relation is IDENTITY.")
        rows.append(dict(cat="relation", q=q, gold="IDENTITY", rat=rat))
    # BEFORE
    for _ in range(n_each):
        p1, d1 = fresh_pt()
        p2, d2 = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))], None
        while p2 == p1:
            p2 = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))]
        d2 = d1.fromordinal(d1.toordinal() + rng.randrange(5, 700))
        q = (f"{p1.capitalize()} on {md(d1)}, and {p2} on {md(d2)}. "
             "What is the relationship between the events' dates?\nChoices:\n"
             "A. IDENTITY\nB. BEFORE\nC. DURING")
        rat = (f"The first event is dated {md(d1)} and the second "
               f"{md(d2)}; the first precedes the second, so the relation "
               f"is BEFORE.")
        rows.append(dict(cat="relation", q=q, gold="BEFORE", rat=rat))
    # DURING
    for _ in range(n_each):
        iv = SPACE_INTERVALS[rng.randrange(len(SPACE_INTERVALS))]
        y1 = rng.randrange(2015, 2019)
        d1 = date(y1, rng.randrange(1, 13), rng.randrange(1, 28))
        span = rng.randrange(120, 500)
        d2 = d1.fromordinal(d1.toordinal() + span)
        dp_ = d1.fromordinal(d1.toordinal() + rng.randrange(10, span - 10))
        p2 = SPACE_POINTS[rng.randrange(len(SPACE_POINTS))]
        q = (f"{iv.capitalize()} ran from {md(d1)} to {md(d2)}, and {p2} "
             f"on {md(dp_)}. What is the relationship between the events' "
             "dates?\nChoices:\nA. IDENTITY\nB. BEFORE\nC. DURING")
        rat = (f"The interval runs from {md(d1)} to {md(d2)}, and the "
               f"point event falls on {md(dp_)}, which lies within the "
               f"interval, so the relation is DURING.")
        rows.append(dict(cat="relation", q=q, gold="DURING", rat=rat))
    return rows

# ---------------- duration (297): insurance domain 1990-2004
ERAS = [
 ("the mutual's storm pool", 1991, 1999),
 ("the coastal reinsurance treaty", 1992, 2003),
 ("the hail claims excess scheme", 1990, 1996),
 ("the flood map programme", 1994, 2002),
 ("the fraud task force", 1993, 1997),
 ("the broker accreditation era", 1990, 2004),
 ("the windstorm levy window", 1995, 2001),
 ("the life book expansion", 1996, 2003),
 ("the marine war risks pool", 1991, 1995),
 ("the catastrophe modelling pilot", 1998, 2003),
 ("the panel repair network", 1992, 2000),
 ("the premium trust restructuring", 1997, 2004),
 ("the actuarial reserving project", 1993, 1999),
 ("the regional mutual compact", 1990, 1998),
 ("the winter freeze campaign", 1994, 1997),
 ("the claims transformation programme", 1999, 2004),
 ("the direct writing experiment", 1995, 1999),
 ("the solvency margin rebuild", 1992, 1998),
 ("the rural agency network", 1990, 2003),
 ("the loss adjusters' charter", 1996, 2001),
 ("the household tariff review", 1991, 1994),
 ("the commercial combined rollout", 1998, 2004),
 ("the data quality crusade", 1997, 2002),
 ("the catastrophe season watch", 1993, 2004),
 ("the emerging risks forum", 1999, 2003),
 ("the legacy system migration", 2000, 2004),
 ("the branch refurbishment cycle", 1992, 1996),
 ("the apprentice underwriters' scheme", 1994, 2001),
 ("the motor panel agreement", 1990, 1997),
 ("the drought research grant", 1995, 2000),
 ("the safety award initiative", 1996, 1999),
 ("the offshore energy book", 1992, 2002),
 ("the public liability review", 1991, 1996),
 ("the telematics feasibility study", 2001, 2004),
 ("the weather derivative trial", 1999, 2002),
 ("the claims ombudsman liaison", 1993, 2000),
 ("the windscreen network tender", 1997, 2001),
 ("the bancassurance partnership", 1998, 2003),
 ("the group captive formation", 1994, 2000),
 ("the reserve strengthening era", 2000, 2003),
]

def duration_rows(n=75):
    rng = random.Random(297)
    rows = []
    tmpl = [
        "{E} ran from {Y1} to {Y2}. How long did it last in total?",
        "{E} stretched from {Y1} to {Y2}. In total, how long did it last?",
        "{E} operated from {Y1} to {Y2}. How long did it last?",
        "{E} was in force from {Y1} to {Y2}. How long did it last in total?",
    ]
    i = 0
    while len(rows) < n:
        name, y1, y2 = ERAS[i % len(ERAS)]
        cap = name.capitalize() if not name.startswith("the") else name
        span = y2 - y1
        t = tmpl[(i // len(ERAS)) % len(tmpl)]
        q0 = t.format(E=cap, Y1=y1, Y2=y2)
        opts = [(span, "gold"), (span + 1, "p1"), (span - 1, "m1"),
                (span * 2, "d2")]
        rng.shuffle(opts)
        gold = f"about {span} years"
        letters = []
        gold_letter = None
        for L, (v, kind) in zip("ABCD", opts):
            letters.append(f"{L}. about {v} years")
            if kind == "gold":
                gold_letter = L
        q = q0 + "\nChoices:\n" + "\n".join(letters)
        rat = (f"{cap.rstrip('.') if cap.endswith('.') else cap} ran from "
               f"{y1} to {y2}, a span of {span} years, so the duration is "
               f"about {span} years.")
        rows.append(dict(cat="duration", q=q, gold=gold, rat=rat))
        i += 1
    return rows

# ---------------- ordering (309): film domain 1890-1935
FILM_EVENTS = [
 ("the Tivoli's first night", 0), ("the Empire's opening", 0),
 ("the Picturedrome's licence", 0), ("the bijou hall's conversion", 0),
 ("the orchestral pit's first score", 0), ("the talkies' arrival", 0),
 ("the winter garden season", 0), ("the film society's founding", 0),
 ("the projection booth rebuild", 0), ("the cinema's centenary gala", 0),
 ("the silent classic's premiere", 0), ("the ushers' ball", 0),
 ("the safety curtain fitting", 0), ("the matinee club's first singalong", 0),
 ("the newsreel contract signing", 0), ("the stars' visit", 0),
]

def ordering_rows(n=50):
    rng = random.Random(309)
    rows = []
    counts = {"TRUE": 0, "FALSE": 0, "Undetermined": 0}
    plan = ["TRUE"] * 19 + ["FALSE"] * 19 + ["Undetermined"] * 12
    rng.shuffle(plan)
    used = set()
    for i, want in enumerate(plan):
        e1 = FILM_EVENTS[i % len(FILM_EVENTS)][0]
        e2 = FILM_EVENTS[(i + 5) % len(FILM_EVENTS)][0]
        while e2 == e1:
            e2 = FILM_EVENTS[rng.randrange(len(FILM_EVENTS))][0]
        y = 1890 + rng.randrange(0, 45)
        d1 = date(y, rng.randrange(1, 13), rng.randrange(1, 28))
        d2 = date(y + rng.randrange(0, 3), rng.randrange(1, 13), rng.randrange(1, 28))
        while d2 == d1:
            d2 = date(y + rng.randrange(0, 3), rng.randrange(1, 13), rng.randrange(1, 28))
        if want == "Undetermined":
            m_only = MONTHS[d1.month - 1]
            s1 = f"{e1.capitalize()} in {m_only} {y}"
            s2 = f"{e2} on {md(d2)}"
            claim_first = rng.random() < 0.5
            if claim_first:
                claim = f"{e1} happened before {e2}"
            else:
                claim = f"{e2} happened before {e1}"
            q = (f"{s1}. {s2}. Claim: {claim} - True/False?\nChoices:\n"
                 "A. TRUE\nB. Undetermined\nC. FALSE")
            rat = (f"The first event is stated only to {m_only} {y}, which "
                   f"spans days both before and after {md(d2)}, so the "
                   f"order cannot be determined from the dates given.")
            gold = "Undetermined"
        else:
            s1 = f"{e1.capitalize()} on {md(d1)}"
            s2 = f"{e2} on {md(d2)}"
            claim_first = rng.random() < 0.5
            if claim_first:
                claim = f"{e1} happened before {e2}"
                truth = d1 < d2
            else:
                claim = f"{e2} happened before {e1}"
                truth = d2 < d1
            q = (f"{s1}. {s2}. Claim: {claim} - True/False?\nChoices:\n"
                 "A. TRUE\nB. Undetermined\nC. FALSE")
            gold = "TRUE" if truth else "FALSE"
            rat = (f"The first event is dated {md(d1)} and the second "
                   f"{md(d2)}; the claim's ordering "
                   f"{'holds' if truth else 'does not hold'}, so the answer "
                   f"is {gold}.")
        key = q[:80].lower()
        if key in used:
            continue
        used.add(key)
        counts[gold] += 1
        rows.append(dict(cat="ordering", q=q, gold=gold, rat=rat))
    while len(rows) < n:
        i += 1
        e1 = FILM_EVENTS[i % len(FILM_EVENTS)][0]
        e2 = FILM_EVENTS[(i + 3) % len(FILM_EVENTS)][0]
        y = 1890 + rng.randrange(0, 45)
        d1 = date(y, rng.randrange(1, 13), rng.randrange(1, 28))
        d2 = date(y + rng.randrange(0, 2), rng.randrange(1, 13), rng.randrange(1, 28))
        if d2 == d1:
            continue
        s1 = f"{e1.capitalize()} on {md(d1)}"
        s2 = f"{e2} on {md(d2)}"
        truth = d1 < d2
        claim = f"{e1} happened before {e2}" if truth else f"{e2} happened before {e1}"
        q = (f"{s1}. {s2}. Claim: {claim} - True/False?\nChoices:\n"
             "A. TRUE\nB. Undetermined\nC. FALSE")
        gold = "TRUE" if truth else "FALSE"
        rat = (f"The first event is dated {md(d1)} and the second {md(d2)}; "
               f"the claim's ordering {'holds' if truth else 'does not hold'}, "
               f"so the answer is {gold}.")
        key = q[:80].lower()
        if key in used:
            continue
        used.add(key)
        counts[gold] += 1
        rows.append(dict(cat="ordering", q=q, gold=gold, rat=rat))
    return rows

if __name__ == "__main__":
    write_rows(296, relation_rows())
    write_rows(297, duration_rows())
    write_rows(309, ordering_rows())
