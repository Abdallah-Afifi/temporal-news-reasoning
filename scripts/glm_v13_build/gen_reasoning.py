#!/usr/bin/env python3
"""Generate the remaining reasoning-category packets (Timeline, Computation,
Relative_Reasoning, Duration_Compare, Order_Compare) for the v13 collection.
Passages and rows are seeded; every row is verified with the ingest gate's own
check() before being written."""
import importlib.util
import json
import random
import re
import sys
from datetime import date, timedelta

SPEC = importlib.util.spec_from_file_location(
    "ing", "/home/g2/Mohamed/temporal-news-reasoning/scripts/ingest_glm_batch.py")
ING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ING)

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

def dstr(d):
    return f"{d.day} {MONTHS[d.month-1]} {d.year}"

def dstr_c(d):
    return f"{MONTHS[d.month-1]} {d.day}, {d.year}"

LEX = {
 "transport": (["railway company", "tramway board", "harbour authority",
    "shipping line", "airfield operator", "carriage works", "port railway",
    "docks board", "ferry service", "canal carriers"],
   ["opened a new station", "doubled the evening service", "took delivery of new engines",
    "suspended the winter sailings", "electrified the suburban stretch",
    "built the goods depot", "rerouted the main line", "retired the paddle steamers",
    "installed the signal box", "cut the freight rates", "launched a night mail run",
    "repainted the whole fleet"]),
 "municipal": (["borough council", "county surveyor", "town planning committee",
    "water committee", "parks department", "housing association", "rates office",
    "civic trust", "market board", "liberation baths committee"],
   ["approved the bypass route", "opened the new estate", "declared a housing zone",
    "rebuilt the town hall steps", "extended the tram shelters", "planted the avenue",
    " surveyed the slum streets", "widened the high street", "lit the promenade",
    "tarmacked the lane", "sank the new well", "let the market stalls"]),
 "labour": (["foundry union branch", "dockers' lodge", "weavers' association",
    "engineers' district", "tramway staff committee", "miners' federation",
    "clerks' guild", "builders' labourers", "printers' chapel", "militia of works"],
   ["balloted for strike action", "won the wage claim", "walked out over piece rates",
    "settled the dispute", "affiliated to the trades council", "founded a benevolent fund",
    "struck for the eight-hour day", "returned to work", "elected new officers",
    "levied the members", "marched on the guildhall", "joined the general stoppage"]),
 "space": (["space agency", "observatory board", "planetary society",
    "astronaut corps", "tracking station", "institute of astronomy",
    "rocket range", "satellite office", "science ministry", "launch consortium"],
   ["launched the orbiter", "commissioned the new telescope", "tracked the comet",
    "published the star survey", "trained the second cohort", "landed the probe",
    "lost contact with the satellite", "opened the visitor centre",
    "flew the balloon experiment", "certified the launch vehicle",
    "mapped the far side", "retired the old antenna"]),
 "insurance": (["insurers' association", "loss adjusters' panel", "fire office",
    "life mutual", "hail pool", "cattle insurance club", "marine underwriters",
    "catastrophe bureau", "actuarial office", "premium fund"],
   ["paid out on the flood claims", "repriced the region's cover",
    "withdrew from the wildfire line", "settled the storm bill",
    "opened the catastrophe bureau", "audited the mortality tables",
    "launched the burglary policy", "refused the subsidence claims",
    "merged the county funds", "declared the annual bonus",
    "invested in the railway debentures", "wrote the reinsurance treaty"]),
 "courts": (["crown court", "county bench", "assize judges", "chancery division",
    "magistrates' bench", "appeal court", "probate registry", "coroner's court",
    "quarter sessions", "king's counsel circle"],
   ["quashed the conviction", "delivered the landmark ruling", "sat for the first time",
    "sentenced the forgers", "admitted the new evidence", "struck out the claim",
    "upheld the appeal", "fined the railway company", "granted the injunction",
    "settled the boundary case", "opened the new registry", "retired the old clerk"]),
 "climate": (["weather bureau", "river board", "flood committee", "moorkeepers",
    "waterworks company", "storm observatory", "glacier survey", "harbour master's office",
    "farmers' weather club", "meteorological society"],
   ["recorded the great frost", "measured the record rainfall",
    "declared the drought", "rebuilt the sea wall", "warned of the spring floods",
    "logged the highest temperature", "charted the storm's track",
    "planted the shelter belt", "dredged the silted channel",
    "issued the gale warning", "counted the lost sheep",
    "marked the river gauge"]),
 "energy": (["electricity board", "gas company", "colliery company",
    "power station works", "turbine fund", "grid authority", "coke works",
    "hydro board", "oil distribution depot", "wind farm partnership"],
   ["switched on the town's supply", "built the cooling towers",
    "sank the new shaft", "capped the old workings", "doubled the turbine hall",
    "piped the gas to the villages", "striking miners stopped the coke ovens",
    "erected the first windmill", "buried the cable across the moor",
    "raised the price of coal", "lit the street lamps", "closed the gasworks"]),
 "technology": (["telephone company", "telegraph office", "radio corporation",
    "computer bureau", "calculator agency", "exchange engineers",
    "software house", "broadcast relay firm", "data processing centre",
    "appliance manufacturer"],
   ["installed the automatic exchange", "cut over the new switchboard",
    "demonstrated the calculating machine", "launched the home terminal",
    "suffered the great outage", "restored the service",
    "doubled the modem bank", "shipped the first thousand units",
    "patched the operating system", "recalled the faulty batch",
    "opened the service bureau", "trained the punch operators"]),
 "banking": (["central bank", "clearing banks' committee", "treasury",
    "note issue department", "exchange control office", "discount house",
    "savings bank trustees", "currency board", "bullion office",
    "regional reserve bank"],
   ["raised the bank rate", "suspended convertibility", "devalued the pound",
    "imposed the credit squeeze", "published the annual report",
    "opened the new clearing house", "withdrew the old notes",
    "capped the hire purchase terms", "eased the restrictions",
    "audited the gold reserves", "stabilised the exchange",
    "lectured on inflation"]),
 "shipping": (["steamship company", "port authority", "dock board",
    "shipbuilders' yard", "harbour commissioners", "container terminal",
    "pilot office", "tug owners' association", "trade winds line",
    "graving dock company"],
   ["launched the new steamer", "deepened the channel", "opened the container berth",
    "laid the keel of the liner", "closed the old graving dock",
    "took the cranes into service", "lost the mail contract",
    "inaugurated the winter route", "welded the first hull section",
    "retired the coal burners", "bunkered the fleet", "salvaged the wreck"]),
 "archaeology": (["county archaeologist", "antiquarian society", "university dig",
    "museum trustees", "ancient monuments board", "field survey team",
    "heritage centre committee", "treasure trove inquest", "local history group",
    "royal commission"],
   ["excavated the roman villa", "scheduled the barrow",
    "uncovered the saxon hoard", "opened the new gallery",
    "published the excavation report", "mapped the ancient field system",
    "conserved the mosaic", "held the summer training dig",
    "recorded the lost chapel", "dated the timber hall",
    "returned the finds to the county", "rewrote the site guide"]),
 "corporate": (["milling combine", "iron and steel group", "brewery company",
    "cotton spinners' trust", "insurance conglomerate", "motor works",
    "chemical alliance", "railway wagon works", "paper mill company",
    "merchant bank house"],
   ["declared the record dividend", "announced the merger",
    "closed the loss-making plant", "floated on the exchange",
    "took over the rival works", "issued the new debentures",
    "cut the workforce", "opened the overseas agency",
    "wound up the subsidiary", "audited the false books",
    "restructured the board", "celebrated the centenary"]),
 "sport": (["county cricket club", "rovers football club", "athletic association",
    "swimming baths committee", "cycling club", "boxing gymnasium",
    "rowing regatta committee", "greyhound track", "speedway promoters",
    "lawn tennis club"],
   ["won the county championship", "signed the record transfer",
    "opened the new grandstand", "sacked the manager", "sealed promotion",
    "toured the colonies", "staged the first floodlit match",
    "founded the junior section", "retired the famous number nine",
    "moved to the new ground", "lifted the cup", "withdrew from the league"]),
 "universities": (["university senate", "college trustees", "research council",
    "school board", "grammar school governors", "polytechnic committee",
    "library syndicate", "laboratory building fund", "examination board",
    "students' union"],
   ["founded the new chair", "built the science block", "opened the halls of residence",
    "admitted women to the degrees", "awarded the honorary doctorate",
    "burned the old syllabus", "inspected the schools",
    "raised the school leaving age", "endowed the scholarship",
    "catalogued the manuscripts", "extended the term", "struck the new quad"]),
 "culture": (["philharmonic society", "repertory theatre", "opera house trust",
    "arts council panel", "cinema circuit", "literary festival committee",
    "choral union", "picture gallery", "poetry bookshop", "brass band contest"],
   ["staged the first night", "commissioned the new symphony",
    "opened the winter season", "restored the auditorium",
    "toured the mining towns", "banned the controversial play",
    "launched the festival", "hung the new exhibition",
    "recorded the anniversary concert", "founded the youth orchestra",
    "printed the first edition", "closed for the war"]),
 "telecoms": (["telephone area board", "cable company", "wireless authority",
    "television relay service", "trunk network engineers", "satellite ground station",
    "subscribers' exchange", "microwave link team", "telex network office",
    "fibre optics consortium"],
   ["connected the thousandth subscriber", "laid the submarine cable",
    "opened the television transmitter", "boosted the relay signal",
    "automated the trunk dialling", "erected the microwave tower",
    "rented the first telex lines", "buried the fibre spine",
    "rang the last manual call", "extended the rural network",
    "licensed the second operator", "surveyed the new route"]),
 "politics": (["council majority", "reform league", "suffrage committee",
    "chamber's opposition bench", "constituency association",
    "boundary commission", "cabinet committee", "trades council political arm",
    "primrose league", "labour representation committee"],
   ["won the by-election", "published the manifesto",
    "passed the reform act", "redivided the constituencies",
    "held the mass meeting", "split over the treaty",
    "formed the coalition", "resigned the safe seat",
    "registered the new voters", "canvassed the estates",
    "petitioned parliament", "chose the new leader"]),
 "health": (["board of guardians", "fever hospital committee", "medical officer",
    "sanitary inspector", "public health laboratory", "nurses' training home",
    "maternity charity", "vaccination officer", "isolation hospital board",
    "dispensary physicians"],
   ["opened the fever hospital", "notified the smallpox case",
    "started the vaccination drive", "cleansed the infected streets",
    "built the maternity wing", "appointed the first matron",
    "traced the typhoid well", "pasted the quarantine notices",
    "funded the district nurses", "inspected the dairies",
    "reported the falling death rate", "closed the common well"]),
 "agriculture": (["farmers' union branch", "agricultural society", "ploughing match club",
    "root crop committee", "threshing ring", "allotments association",
    "creamery company", "sheep breeders' society", "grain cooperative",
    "estate foresters"],
   ["held the annual show", "introduced the new binder", "ploughed the old pasture",
    "fenced the common", "founded the creamery", "imported the pedigree bulls",
    "suffered the harvest failure", "marketed the first crop",
    "drained the water meadow", "won the root prize",
    "rationed the winter feed", "planted the orchard"]),
}
DOMAIN_KEY = {
 "transport, aviation and rail": "transport",
 "municipal government and planning": "municipal",
 "labour disputes and strikes": "labour",
 "space and astronomy": "space",
 "insurance and natural-catastrophe claims": "insurance",
 "courts and legal rulings": "courts",
 "climate, weather and natural disasters": "climate",
 "energy and utilities": "energy",
 "technology products and outages": "technology",
 "central banking and inflation": "banking",
 "shipping, ports and trade": "shipping",
 "archaeology and heritage": "archaeology",
 "corporate earnings, mergers and layoffs": "corporate",
 "professional sport": "sport",
 "universities, schools and research": "universities",
 "film, music and cultural institutions": "culture",
 "telecoms and infrastructure": "telecoms",
 "national politics and elections": "politics",
 "public health and medicine": "health",
 "agriculture and food supply": "agriculture",
}
OPENERS = ["", "That same season, ", "Soon afterwards, ", "By the following spring, ",
 "In the months that followed, ", "Later that decade, ", "The town remembered, ",
 "Meanwhile, ", "The committee's minutes record that ", "It was reported that "]

HINT = ("(Hint: Please answer in the form of Month Day, Year. e.g. "
        "1 year 2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)")

def make_events(rng, domain, era, n, used_global=None):
    orgs, acts = LEX[DOMAIN_KEY[domain]]
    y0, y1 = [int(x) for x in era.split("-")]
    out, used = [], set()
    while len(out) < n:
        org, act = rng.choice(orgs), rng.choice(acts)
        phrase = f"the {org} {act}"
        if phrase in used or (used_global is not None and phrase in used_global):
            continue
        used.add(phrase)
        if used_global is not None:
            used_global.add(phrase)
        y = rng.randint(y0, y1 - 1)
        d = date(y, rng.randint(1, 12), rng.randint(1, 28))
        out.append({"p": phrase, "d": d})
    return out

def passage_text(rng, pid, events):
    blocks = []
    for i in range(3):
        chunk = sorted(events[i*6:(i+1)*6], key=lambda e: e["d"])
        if not chunk:
            continue
        sents = []
        for j, ev in enumerate(chunk):
            lead = rng.choice(OPENERS) if j else ""
            s = f"{lead}The {ev['p'].split(' ', 1)[1].capitalize()}" \
                if False else f"{lead}{ev['p'][0].upper() + ev['p'][1:]} on {dstr(ev['d'])}."
            sents.append(s[0].upper() + s[1:])
        titles = ["A Year of Decisions", "The Committee's Work",
                  "Chronicle of Progress", "The Season's Record",
                  "Town Affairs", "The Board Reports", "A Busy Season"]
        blocks.append(f"[{i+1}] Title: {rng.choice(titles)}, Day: {dstr(chunk[0]['d'])} "
                      f"Content: {' '.join(sents)}")
    return f"=== PASSAGE P{pid} ===\n" + "\n".join(blocks) + "\n"

def fmt_span(rd):
    parts = []
    if rd.years:
        parts.append(f"{rd.years} year" + ("s" if rd.years > 1 else ""))
    if rd.months:
        parts.append(f"{rd.months} month" + ("s" if rd.months > 1 else ""))
    if rd.days:
        parts.append(f"{rd.days} day" + ("s" if rd.days > 1 else ""))
    return " ".join(parts) or "0 days"

def cap(p):
    return p[0].upper() + p[1:]

def base_row(cat, pno):
    return {"source_dataset": "AUG_GLM2", "slice": "A", "category": cat,
            "provenance": "news", "question": "", "context": "",
            "passage": f"P{pno}", "targets": [], "rationale": "",
            "source": "augmented", "source_id": ""}

# ---------------- row makers ----------------
def row_timeline(rng, pno, evs):
    trio = rng.sample(evs, 3)
    letters = ["A", "B", "C"]
    rng.shuffle(letters)
    chosen = dict(zip(letters, trio))
    gold = ",".join(sorted(letters, key=lambda L: chosen[L]["d"]))
    q = ("Below are 3 facts. You need to sort these facts in chronological order. "
         "Requirements: You must output a sequence of uppercase letters separated by "
         "commas, such as 'A,B,C', without any other characters.\nChoices:\n"
         + "\n".join(f"{L}. {cap(chosen[L]['p'])}" for L in "ABC"))
    order_s = ", ".join(f"{cap(e['p'])} in {e['d'].year}" for e in sorted(trio, key=lambda e: e["d"]))
    rat = f"Chronologically: {order_s}. " + "; ".join(
        f"{L}={chosen[L]['d'].isoformat()}" for L in "ABC")
    r = base_row("Timeline", pno)
    r.update(question=q, targets=[gold], rationale=rat)
    return r

def row_computation(rng, pno, evs):
    a, b = rng.sample(evs, 2)
    if a["d"] > b["d"]:
        a, b = b, a
    from dateutil.relativedelta import relativedelta
    span = fmt_span(relativedelta(b["d"], a["d"]))
    q = (f"{cap(a['p'])} on {dstr(a['d'])}, and {b['p']} on {dstr(b['d'])}. "
         f"How long passed from {a['p']} to {b['p']}? {HINT}")
    rat = (f"{cap(a['p'])} on {dstr(a['d'])} and {b['p']} on {dstr(b['d'])}; "
           f"from {dstr(a['d'])} to {dstr(b['d'])} is {span}.")
    r = base_row("Computation", pno)
    r.update(question=q, targets=[span], rationale=rat)
    return r

def row_relative(rng, pno, evs):
    pool = sorted(evs, key=lambda e: e["d"])
    anchor = rng.choice(pool[1:-2])
    ai = pool.index(anchor)
    later = [e for e in pool[ai+1:]]
    if not later:
        return None
    if rng.random() < 0.55:
        gold = later[0]
        others = [e for e in pool if e is not anchor and e is not gold
                  and (e["d"] < anchor["d"] or e["d"] > gold["d"])]
        if len(others) < 3:
            return None
        opts = [gold] + rng.sample(others, 3)
        q = f"What happened immediately after {anchor['p']}?"
        rat = (f"{cap(anchor['p'])} on {dstr(anchor['d'])} and the next step, "
               f"{gold['p']}, came on {dstr(gold['d'])}; the other options either "
               f"precede the anchor or follow later still.")
    else:
        cands = later[:4]
        if len(cands) < 3:
            return None
        gold = min(cands, key=lambda e: e["d"])
        rest = [e for e in cands if e is not gold][:3]
        opts = [gold] + rest
        q = f"Which event came soonest after {anchor['p']}?"
        rat = (f"{cap(anchor['p'])} was on {dstr(anchor['d'])}; among the options, "
               f"{gold['p']} on {dstr(gold['d'])} is the nearest in time.")
    rng.shuffle(opts)
    letters = "ABCD"
    q += "\nChoices:\n" + "\n".join(
        f"{letters[i]}. {cap(o['p'])}" for i, o in enumerate(opts))
    r = base_row("Relative_Reasoning", pno)
    r.update(question=q, targets=[cap(gold["p"])], rationale=rat)
    return r

def row_duration(rng, pno, evs, want):
    tries = 0
    while tries < 5000:
        tries += 1
        e1, e2, e3, e4 = rng.sample(evs, 4)
        s1 = abs((e2["d"] - e1["d"]).days)
        s2 = abs((e4["d"] - e3["d"]).days)
        diff, rel = abs(s1 - s2), abs(s1 - s2) / max(s1, s2, 1)
        close = rel <= 0.10 or diff <= 60
        if want == "C" and close:
            gold = "The two durations are approximately the same length."
            break
        if want == "A" and s1 > s2 and not close:
            gold = "Duration 1 is longer."
            break
        if want == "B" and s2 > s1 and not close:
            gold = "Duration 2 is longer."
            break
    else:
        return None
    q = (f"Which of the following two durations is longer? "
         f"*Duration 1:* Between {e1['p']} and {e2['p']}. "
         f"*Duration 2:* Between {e3['p']} and {e4['p']}.\nChoices:\n"
         f"A. Duration 1 is longer.\nB. Duration 2 is longer.\n"
         f"C. The two durations are approximately the same length.")
    rat = (f"Duration 1 ran from {dstr_c(min(e1['d'], e2['d']))} to "
           f"{dstr_c(max(e1['d'], e2['d']))}; Duration 2 ran from "
           f"{dstr_c(min(e3['d'], e4['d']))} to {dstr_c(max(e3['d'], e4['d']))}. "
           f"The spans are {s1} and {s2} days, so {gold[0].lower() + gold[1:]}"
           .replace("the two durations", "the two durations"))
    r = base_row("Duration_Compare", pno)
    r.update(question=q, targets=[gold], rationale=rat)
    return r

def row_order(rng, pno, evs, want):
    tries = 0
    while tries < 5000:
        tries += 1
        f1, f2 = rng.sample(evs, 2)
        gap = abs((f1["d"] - f2["d"]).days)
        if want == "C" and 3 <= gap <= 14:
            gold = "They happen at almost the same time."
            break
        if want != "C" and gap > 20:
            gold = ("Fact 1 happened earlier." if f1["d"] < f2["d"]
                    else "Fact 2 happened earlier.") if want == "d" else None
            if want == "d":
                gold = ("Fact 1 happened earlier." if f1["d"] < f2["d"]
                        else "Fact 2 happened earlier.")
                break
            else:
                want_gold = ("Fact 1 happened earlier." if want == "F1"
                             else "Fact 2 happened earlier.")
                actual = ("Fact 1 happened earlier." if f1["d"] < f2["d"]
                          else "Fact 2 happened earlier.")
                if want_gold == actual:
                    gold = actual
                    break
    else:
        return None
    q = (f"For Fact1: {cap(f1['p'])} and Fact2: {cap(f2['p'])}, which one "
         f"happened earlier?\nChoices:\nA. Fact 1 happened earlier.\n"
         f"B. Fact 2 happened earlier.\nC. They happen at almost the same time.")
    rat = (f"{cap(f1['p'])} on {dstr_c(f1['d'])}, while {f2['p']} was on "
           f"{dstr_c(f2['d'])}, so {gold[0].lower() + gold[1:]}")
    r = base_row("Order_Compare", pno)
    r.update(question=q, targets=[gold], rationale=rat)
    return r

# ---------------- packet assembly ----------------
SEEN_Q = set()

def make_packet(pid, cat, domain, era, seed):
    rng = random.Random(f"{pid}-{seed}")
    shared = set()
    events = [make_events(rng, domain, era, 18, shared) for _ in range(3)]
    # force a few near-ties inside each pool so "almost same time" rows exist
    for pool in events:
        for _ in range(3):
            a, b = rng.sample(pool, 2)
            b["d"] = a["d"] + timedelta(days=rng.randint(4, 12))
    passages = "\n".join(passage_text(rng, i + 1, evs) for i, evs in enumerate(events))
    pmap = {}
    for i, evs in enumerate(events):
        pmap[f"P{i+1}"] = passage_text(rng, i + 1, evs).split("===", 1)[-1] \
            .split("\n", 1)[1].strip() if False else None
    # rebuild plain passage bodies (no banner) for context injection
    pmap = {}
    for i, evs in enumerate(events):
        txt = passage_text(rng, i + 1, evs)
        pmap[f"P{i+1}"] = txt.split("\n", 1)[1].strip()
    rows = []
    if cat == "Timeline":
        makers = [row_timeline] * 25
    elif cat == "Computation":
        makers = [row_computation] * 25
    elif cat == "Relative_Reasoning":
        makers = [row_relative] * 25
    elif cat == "Duration_Compare":
        wants = (["A"] * 9 + ["B"] * 9 + ["C"] * 7)
        rng.shuffle(wants)
        makers = [(lambda rng, pno, evs, w=w: row_duration(rng, pno, evs, w)) for w in wants]
    else:
        wants = ["F1"] * 10 + ["F2"] * 10 + ["C"] * 5
        rng.shuffle(wants)
        makers = [(lambda rng, pno, evs, w=w: row_order(rng, pno, evs, w)) for w in wants]
    guard = 0
    it = iter(makers)
    for mk in makers:
        guard = 0
        while True:
            guard += 1
            if guard > 400:
                raise RuntimeError(f"{pid}: could not build row")
            pno = rng.randint(1, 3)
            r = mk(rng, pno, events[pno - 1])
            if r is None:
                continue
            nq = re.sub(r"\s+", " ", r["question"]).strip().lower()
            if nq in SEEN_Q:
                continue
            r2 = dict(r)
            ref = r2.pop("passage", "")
            r2["context"] = pmap.get(ref, "")
            errs = ING.check(r2)
            if errs:
                raise RuntimeError(f"{pid} row failed gate: {errs}\n{json.dumps(r, indent=1)}")
            SEEN_Q.add(nq)
            rows.append(r)
            break
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    return passages + "\n" + body + "\n", len(rows)

PACKETS = json.load(open(
    "/home/g2/Mohamed/temporal-news-reasoning/data/glm_packets_v13/_plan.json"))["plan"]
DONE = {"041_Timeline", "057_Computation", "069_Relative_Reasoning",
        "083_Duration_Compare", "095_Order_Compare"}
CATS = {"Timeline", "Computation", "Relative_Reasoning",
        "Duration_Compare", "Order_Compare"}

def main(only=None):
    outdir = "/home/g2/Mohamed/temporal-news-reasoning/data/glm_raw_v13"
    made = 0
    for x in PACKETS:
        pid = x["packet"].split("_")[0]
        fname = x["packet"].replace(".md", ".txt")
        if x["category"] not in CATS or fname in DONE:
            continue
        if only and pid not in only:
            continue
        text, n = make_packet(pid, x["category"], x["domain"], x["era"],
                              seed=int(pid))
        with open(f"{outdir}/{fname}", "w") as fh:
            fh.write(text)
        made += 1
        print(f"wrote {fname}: {n} rows")
    print(f"total packets written: {made}")

if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
