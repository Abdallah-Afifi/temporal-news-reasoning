"""Generate letter-format training data (AUG_LETTER) for v4.

Purpose: v3 collapsed on TIME letter-gold categories (Order_Compare 61->9,
Duration_Compare 40->0) because training targets were spans/dates only and
MCQ items had letters as secondary golds. This generator produces items whose
ONLY acceptable answer is a bare letter (or comma letter sequence), covering:
  order_compare      Which happened first?           gold A/B
  duration_compare   Which lasted longer?            gold A/B
  relative_letter    Before/after framed as options  gold A/B
  timeline_seq       Order 3 events                  gold e.g. B,C,A
  date_mcq           When did X happen? 3 options    gold A/B/C

Outputs data/letter_format/{train,probe}.jsonl (probe = held-out format test,
never trained on) + dedup + benchmark-question collision guard.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import BenchmarkLoader  # noqa: E402

rng = random.Random(42)
probe_rng = random.Random(4242)

ENTITIES = [
    "the Riverside Film Festival", "the Aurora Theatre Company", "the Northgate Marathon",
    "the Silverline Railway", "the Cedar Valley Vineyard", "the Helios Observatory",
    "the Marlowe Publishing House", "the Eastport Shipyard", "the Bluewater Aquarium",
    "the Kestrel Cycling Club", "the Larkspur Botanical Gardens", "the Ironbridge Foundry",
    "the Westhaven Philharmonic", "the Falcon Ridge Ski Resort", "the Amber Museum",
    "the Stonebrook Library", "the Harbor Light Lighthouse Trust", "the Oakfield Football Club",
    "the Meridian Cartographic Society", "the Pinewood Chess Academy", "the Quartz Mining Cooperative",
    "the Windmill Lane Recording Studio", "the Sable Island Ferry Service", "the Copperfield Brewery",
    "the Rosewood Equestrian Center", "the Beacon Hill Dental Clinic", "the Foxglove Apiary",
    "the Grandview Hotel Group", "the Saltmarsh Bird Sanctuary", "the Lantern Street Night Market",
    "the Highfield Grammar School", "the Redcliff Lifeguard Association", "the Willow Creek Sawmill",
    "the Palegrove Art Collective", "the Thornbury Cricket Club", "the Glasswing Aviation School",
    "the Duneside Golf Links", "the Montague Chess Open", "the Fairwind Sailing Club",
    "the Old Mill Chocolate Works", "the Brightwater Rowing Club", "the Cedar Court Chamber Orchestra",
    "the Greylock Hiking Society", "the Sunfield Solar Array", "the Blackfriars Print Shop",
    "the Meadowbrook Dairy Cooperative", "the Silver Birch Theater Troupe", "the Kingsway Tram Heritage Line",
    "the Ashford Pottery Guild", "the Longvale Eisteddfod Council",
]
EVENT_TEMPLATES = [
    "{e} was founded", "the inaugural season of {e} took place", "{e} opened its main site",
    "{e} hosted its first international event", "{e} was renamed", "{e} merged with a rival organization",
    "{e} completed its major renovation", "{e} launched its annual programme",
    "{e} received the regional heritage award", "{e} closed for reconstruction",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

Q_ORDER = [
    "Which of the following happened first?\nChoices:\n{opts}",
    "Put aside later events: which one is the earliest?\nChoices:\n{opts}",
    "Two events are described below. Which occurred before the other?\nChoices:\n{opts}",
    "Order matters here: which happened earlier in time?\nChoices:\n{opts}",
]
Q_DURATION = [
    "Which of the two lasted longer?\nChoices:\n{opts}",
    "Which period was the longer one?\nChoices:\n{opts}",
    "Which of these spans more time?\nChoices:\n{opts}",
    "Pick the longer duration:\nChoices:\n{opts}",
]
Q_RELATIVE = [
    "Which statement is correct?\nChoices:\n{opts}",
    "Regarding the order of events, which is true?\nChoices:\n{opts}",
    "Which option correctly describes the sequence?\nChoices:\n{opts}",
]
Q_SEQ = [
    "List the events in chronological order, earliest first.\nChoices:\nA. {o0}\nB. {o1}\nC. {o2}",
    "Arrange these three events from earliest to latest.\nChoices:\nA. {o0}\nB. {o1}\nC. {o2}",
    "Give the correct chronological sequence of the events.\nChoices:\nA. {o0}\nB. {o1}\nC. {o2}",
]
Q_DATE = [
    "When did {ev}?\nChoices:\n{opts}",
    "On which date did {ev}?\nChoices:\n{opts}",
    "Pick the correct date: {ev}?\nChoices:\n{opts}",
]


def fmt(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def rand_date(lo: int = 1950, hi: int = 2024) -> date:
    start = date(lo, 1, 1)
    return start + timedelta(days=rng.randrange((date(hi, 1, 1) - start).days))


def perturb(d: date) -> date:
    delta = rng.choice([1, 2, 3, 7, 14, 30, 60, 365, 730]) * rng.choice([-1, 1])
    try:
        return d + timedelta(days=delta)
    except (OverflowError, ValueError):
        return d


def entity_pair():
    a, b = rng.sample(ENTITIES, 2)
    return a, b


# Enforced exact gold-letter balance: when the running imbalance reaches 3,
# force the minority letter so train/probe end ~50/50 by construction.
_BAL = {"order_compare": 0, "duration_compare": 0}


def _pick_letter(subtype: str) -> str:
    return "A" if _BAL[subtype] <= 0 else "B"


def make_order_compare():
    a, b = entity_pair()
    da, db = rand_date(), rand_date()
    while db == da:
        db = rand_date()
    ev_a = f"{rng.choice(EVENT_TEMPLATES).format(e=a)} ({fmt(da)})"
    ev_b = f"{rng.choice(EVENT_TEMPLATES).format(e=b)} ({fmt(db)})"
    first_is_a = da < db
    want = _pick_letter("order_compare")
    # place the earlier event at slot `want`, the later at the other slot
    earlier, later = (ev_a, ev_b) if first_is_a else (ev_b, ev_a)
    other = "B" if want == "A" else "A"
    slots = {want: earlier, other: later}
    opts = [slots["A"], slots["B"]]
    gold = want
    _BAL["order_compare"] += 1 if gold == "A" else -1
    q = rng.choice(Q_ORDER).format(opts="\n".join(f"{l}. {o}" for l, o in zip("AB", opts)))
    return {"question": q, "targets": [gold], "subtype": "order_compare"}


def make_duration_compare():
    a, b = entity_pair()
    d1 = rng.randrange(2, 48)
    d2 = d1 + rng.choice([1, 2, 3, 5, 8, 13, 21])
    unit = rng.choice(["months", "weeks", "days"])
    want = _pick_letter("duration_compare")
    big, small = (d2, d1) if d2 > d1 else (d1, d2)
    da_ = big if want == "A" else small
    db_ = small if want == "A" else big
    p1 = f"the licence of {a} ran for {da_} {unit}"
    p2 = f"the licence of {b} ran for {db_} {unit}"
    longer = "A" if da_ > db_ else "B"
    _BAL["duration_compare"] += 1 if longer == "A" else -1
    opts = [p1, p2]
    q = rng.choice(Q_DURATION).format(opts="\n".join(f"{l}. {o}" for l, o in zip("AB", opts)))
    return {"question": q, "targets": [longer], "subtype": "duration_compare"}


def make_relative_letter():
    a, b = entity_pair()
    da, db = rand_date(), rand_date()
    while db == da:
        db = rand_date()
    eva = rng.choice(EVENT_TEMPLATES).format(e=a)
    evb = rng.choice(EVENT_TEMPLATES).format(e=b)
    s1 = f"{eva} on {fmt(da)} happened before {evb} on {fmt(db)}"
    s2 = f"{evb} on {fmt(db)} happened before {eva} on {fmt(da)}"
    gold = "A" if da < db else "B"
    opts = [s1, s2]
    q = rng.choice(Q_RELATIVE).format(opts="\n".join(f"{l}. {o}" for l, o in zip("AB", opts)))
    return {"question": q, "targets": [gold], "subtype": "relative_letter"}


def make_timeline_seq():
    ents = rng.sample(ENTITIES, 3)
    ds = sorted({rand_date() for _ in range(3)})
    while len(ds) < 3:
        ds = sorted({rand_date() for _ in range(3)})
    evs = [(rng.choice(EVENT_TEMPLATES).format(e=e), d) for e, d in zip(ents, ds)]
    perm = list(range(3))
    rng.shuffle(perm)
    opts = [evs[i][0] + f" ({fmt(evs[i][1])})" for i in perm]
    gold = ",".join("ABC"[perm.index(i)] for i in range(3))
    q = rng.choice(Q_SEQ).format(o0=opts[0], o1=opts[1], o2=opts[2])
    return {"question": q, "targets": [gold], "subtype": "timeline_seq"}


def make_date_mcq():
    a = rng.choice(ENTITIES)
    d = rand_date()
    ev = rng.choice(EVENT_TEMPLATES).format(e=a)
    wrong = {perturb(d)}
    while len(wrong) < 2:
        wrong.add(perturb(d))
    distractors = list(wrong)[:2]
    all_dates = [d] + distractors
    rng.shuffle(all_dates)
    gold = "ABC"[all_dates.index(d)]
    q = rng.choice(Q_DATE).format(ev=ev, opts="\n".join(f"{l}. {fmt(x)}" for l, x in zip("ABC", all_dates)))
    return {"question": q, "targets": [gold], "subtype": "date_mcq"}


MAKERS = [
    (make_order_compare, 350), (make_duration_compare, 250), (make_relative_letter, 250),
    (make_timeline_seq, 300), (make_date_mcq, 150),
]
PROBE_MAKERS = [
    (make_order_compare, 25), (make_duration_compare, 20), (make_relative_letter, 20),
    (make_timeline_seq, 25), (make_date_mcq, 10),
]


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def build(makers, out_path: Path, n_expected: int):
    bench_q: set[str] = set()
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    for bench in ("time", "timebench"):
        for ex in loader.load(bench):
            bench_q.add(norm(ex.question))
    rows, seen = [], set()
    collisions = 0
    for maker, n in makers:
        made = 0
        while made < n:
            r = maker()
            k = norm(r["question"])
            if k in seen:
                continue
            if k in bench_q:
                collisions += 1
                continue
            seen.add(k)
            rows.append({
                "source_dataset": "AUG_LETTER",
                "question": r["question"],
                "context": "",
                "targets": r["targets"],
                "source": "letter_format",
                "subtype": r["subtype"],
            })
            made += 1
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{out_path.name}: {len(rows)} rows | collisions skipped: {collisions}")
    print("  by subtype:", dict(Counter(r["subtype"] for r in rows)))
    letters = Counter(r["targets"][0] for r in rows if "," not in r["targets"][0])
    print("  letter balance:", dict(letters))
    seqs = Counter(r["targets"][0] for r in rows if "," in r["targets"][0])
    print("  seq balance:", dict(seqs))
    assert len(rows) == n_expected, f"expected {n_expected}, got {len(rows)}"


def main() -> int:
    out = PROJECT_ROOT / "data" / "letter_format"
    out.mkdir(parents=True, exist_ok=True)
    global rng
    rng = random.Random(42)
    build(MAKERS, out / "train.jsonl", 1300)
    rng = probe_rng
    build(PROBE_MAKERS, out / "probe.jsonl", 100)
    train_q = {norm(json.loads(l)["question"]) for l in open(out / "train.jsonl")}
    probe_q = {norm(json.loads(l)["question"]) for l in open(out / "probe.jsonl")}
    print(f"train/probe overlap: {len(train_q & probe_q)} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
