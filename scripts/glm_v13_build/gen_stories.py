#!/usr/bin/env python3
"""Generate the three remaining storytelling packets (125 rows each):
slot-filled story templates with a plausible and an absurd ending."""
import importlib.util, json, random, re, sys

SPEC = importlib.util.spec_from_file_location(
    "ing", "/home/g2/Mohamed/temporal-news-reasoning/scripts/ingest_glm_batch.py")
ING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ING)

# ---- story bank: (context, good ending, absurd ending, rationale) ----
S = [
 ("{N} had been the {O}'s treasurer for eleven years and knew every figure by heart, but the annual audit fell in the same week as the harvest festival, and the two ledgers sat side by side on the same desk through three sleepless nights. On the fourth morning {N} found the error: a single transposed digit in the subscriptions that had travelled undetected through four quarters. {N} wrote out the correction, walked it to the chair's house before breakfast, and offered to read out both figures at the festival itself.",
  "The correction was read from the platform, the members nodded at the honest arithmetic, and {N} slept properly for the first time in a week.",
  "The transposed digit was elected chair by acclamation, and {N} sailed away on a clipper to found a new republic of accountants.",
  "{N} found and owned the error, so a public correction closes the story naturally; a digit taking office is fantasy."),
 ("The {O} was promised to the village by {P}'s owner in a will nobody could find, and the search through the solicitor's cellar took the whole of a wet February. The box was there, watermark intact, and the signature matched the parish register exactly. The solicitor telephoned the committee at dusk, and the keys changed hands on the step in the rain.",
  "The committee dried the deeds by the fire, opened the {O} to the village the following spring, and hung the owner's portrait in the hall.",
  "The deeds turned out to be a recipe for marmalade, and the village lived on marmalade proceeds for a century, with {N} as chief cook.",
  "A will found and verified naturally passes to its beneficiaries; marmalade fortunes belong to another genre."),
 ("{N} trained the colliery band's young cornet player through one entire winter, meeting in the warmth of the boiler house after shifts. The contest at the town hall came, and the boy walked to the platform with the instrument shaking. He played the solo through, one note cracked in the second movement, and the adjudicator's pencil moved once.",
  "The band took second place, the cracked note was forgiven by everyone who had ever played one, and {N} booked the boiler house for the following winter.",
  "The cracked note was heard three counties away, and the adjudicator retired on the spot to keep lighthouses with {N}.",
  "A nervous solo ending in second place fits the story; audible counties and lighthouse keepers do not."),
 ("When the {O}'s roof failed in the night storm, {N} was first through the door with buckets and tarpaulin, and by morning the books and the seed collection stood dry in the kitchen. The insurers sent an assessor who measured everything and doubted the tarpaulin. {N} kept the receipts from the hardware shop in a biscuit tin and handed them over without a word.",
  "The assessor paid out in full, the roof was slated by midsummer, and the biscuit tin went into the {O}'s little museum.",
  "The assessor was so moved by the tarpaulin that he left insurance altogether and joined a travelling circus as the tarpaulin act, billing {N} as the understudy.",
  "Receipts and honest repair close an insurance story properly; circus conversions are absurd."),
 ("The touring company arrived with two plays and one actor short, and {N}, the {O}'s stagekeeper, had half of the second play by heart from years of prompting. The manager, desperate at the curtain's edge, asked the question the whole house could hear. {N} said yes before thinking, was dressed in the dead man's costume in four minutes, and spoke the first line from the wings.",
  "The play finished to a standing house, the manager paid {N} a full actor's wage, and the prompt book retired with honours.",
  "The audience elected {N} to parliament during the interval, and the second play was never spoken of again.",
  "A stagekeeper stepping up is a natural theatre ending; parliamentary elections at the interval are not."),
 ("{N} kept the {O}'s bees on the allotment behind the sheds, and the summer the pesticide drifted from the neighbouring field, three hives sickened in a week. {N} photographed the dead bees daily, dated every frame, and wrote to the farmer with the evidence and no anger at all. The farmer came to look at the hives himself on the Sunday.",
  "The spray drift was moved to the windward evenings, the hives recovered by August, and the farmer left a sack of sugar at {N}'s gate each spring.",
  "The bees unionised, negotiated a four-day working week, and bought the neighbouring farm outright, installing {N} as mascot.",
  "Evidence and a civil letter is how drift disputes resolve; unionised bees are fantasy."),
 ("The {O}'s minibus failed its test on the morning of the regional final, and {N} the caretaker owned the only vehicle in the street — a bread van with a warm smell and a sliding door. The team captain asked first, then the whole committee asked together, and {N} looked at the clock and then at the twelve waiting players.",
  "The bread van reached the final with ten minutes to spare, the players laughed the whole way, and the {O} bought {N} a new sliding door for Christmas.",
  "The bread van was canonised by the county council, and bread was declared a sport in its honour at the {O}'s expense.",
  "A borrowed van getting the team to the final is the plausible close; canonised vehicles are absurd."),
 ("At the archive {N} indexed the {O}'s oldest box and found a letter that contradicted the founding date carved over the door. The committee met on the matter twice and split both times. {N} photocopied the letter for every member, and the third meeting began with all of them reading it in silence.",
  "The carving was corrected with a modest brass plate, the earlier date entered the records, and {N} framed the letter beside the new plate.",
  "The door was burned for firewood, the letter flew away, and the founding date settled itself by arbitration in Geneva, over the {O}'s objection.",
  "Institutions correct their records with small brass plates; self-settling dates and Geneva arbitrations are not."),
 ("{N} ran the {O}'s canteen on a budget the auditors called impossible and everyone else called Tuesday, and the year the grant was cut by a fifth, the menu was rewritten around the wholesale market's odd lots. The first week's experiment was nettle soup and nobody said anything. The second week was better.",
  "By the summer the canteen broke even, the nettle soup became a local legend, and the grant was restored with a small apology addressed to {N} by name.",
  "The nettle soup gained the power of speech and took over the {O}'s committee meetings entirely.",
  "Thrift ending in solvency is the natural arc; sentient soup is fantasy."),
 ("The flood took the {O}'s ground floor on the Tuesday and the volunteers began on the Wednesday, {N} first with a kettle and a list. The insurance wanted photographs of damage that was already being repaired, so {N} photographed the work itself, mud and all, and sent both sets.",
  "The assessors accepted the photographs of the work as evidence of the damage, the claim was paid, and {N}'s kettle was retired to a shelf with a small plaque.",
  "The flood apologised in writing, returned on Thursday to apologise again, and the {O}'s claim was paid in dolphins.",
  "Documented repair supporting a claim is plausible; apologetic floods and dolphin settlements are not."),
 ("{N} had promised the {O}'s choir a new anthem by Michaelmas and wrote nothing until the last week, when the tune arrived whole on a bus journey and the words followed on the walk back. The copyist worked through the night, and the choir saw the pages for the first time at the rehearsal itself.",
  "The anthem was sung on the day, slightly rough at the bars the copyist had inked last, and {N} kept it in the folder for another twenty years.",
  "The anthem was sung by the buses instead, in traffic, and the {O}'s choir took a permanent holiday on the proceeds.",
  "A last-minute piece sung through is a realistic ending; singing buses are absurd."),
 ("The {O}'s telephone system failed on the first morning of the membership drive, and {N}, who had wired the building thirty years before, was called out of a retirement lunch to find the fault. It took four hours and it was one joint, green with age, in the cellar. {N} closed the joint and stayed to answer the phones.",
  "The drive made its target by Friday, {N} accepted a lifetime membership instead of a fee, and the joint was labelled and left visible for the next thirty years.",
  "The faulty joint was exhibited at the national gallery, and the {O}'s telephones learned to answer themselves in Latin.",
  "An old wireman fixing an old joint is the plausible close; gallery joints and Latin phones are not."),
 ("{N} captained the {O}'s second eleven, which had lost every match for two seasons, and the season the fixture list brought the champions to the little ground, {N} picked the youngest side in the club's history on a whim of hope. It rained, then cleared, and the match went to the last over twice.",
  "The second eleven won by two runs, the colts were promoted to the first team together, and the {O}'s losing streak was never mentioned again.",
  "The champions resigned from the sport forever, and the rain was fined for interference by the {O}'s disciplinary panel.",
  "An upset win for a young side fits; mass resignations and fined weather do not."),
 ("The {O}'s exhibition was hung and labelled when the curator noticed the smallest painting was a copy, the original having been swapped at some point no one could fix. {N} the framer spotted the difference at once on being asked, the rebate screws telling the whole story, and said so plainly.",
  "The police recovered the original within a month, the copy was hung in the office with an honest label, and the framer's eye passed into legend at the {O}.",
  "The copy was declared more original than the original, and both paintings agreed to share a frame in the {O}'s store room.",
  "A framer's eye and a police recovery close the story properly; agreeing paintings are fantasy."),
 ("{N} delivered the parish magazine by bicycle for nineteen years and knew which letterboxes bit, and the month the print run doubled with the festival supplement, the rounds took until dark. The editor offered to halve the drop. {N} looked at the piles and asked only for a better lamp.",
  "The whole double run was delivered by ten o'clock, the lamp became part of the magazine's logo, and {N}'s round map was framed at the {O}'s AGM.",
  "The bicycle was knighted for services to distribution, and the {O}'s letterboxes apologised collectively to {N}.",
  "Finishing a doubled round with a better lamp is the natural close; knighted bicycles are absurd."),
 ("The {O}'s boiler was as old as the building and failed in the coldest week, and {N} the caretaker nursed it through each night with a mattress against the door and a thermometer in a jam jar. The council's contractor quoted nine weeks. {N} quoted a list of parts and a fortnight, and the committee, freezing, chose the caretaker.",
  "The boiler ran again in twelve days, the contractor's quote was framed in the boiler room beside {N}'s jam jar.",
  "The boiler achieved sentience, thanked the committee, and demanded a pension from the {O}'s trustees.",
  "A caretaker's repair beating a contractor is plausible; sentient boilers are not."),
]

FILL = {
 "agriculture": {
   "N": ["Ellen Marsh","Tom Brierley","Sioned Prys","Alec Fenwick","Mary Dowd",
         "Rhys Cadwallader","Iris Harkness","Bert Tanwick","Nora Selby","Evan Pryce",
         "Flo Cardew","Stan Ockleton"],
   "O": ["young farmers' club","grain cooperative","village creamery","root show committee",
         "sheep breeders' society","allotments association","harvest festival committee",
         "ploughing match club","threshing ring","dairy herdlings' fund"],
   "extra": "1990s farm prices and subsidy cheques"},
 "culture": {
   "N": ["Cecily Vane","Harold Trumpington","Doris Ashbee","Reggie Mowbray",
         "Vera Lestrange","Alf Pettingale","Constance Birch","Wilf Sopwith",
         "Mabel Ferrers","Ernie Cudlipp","Sybil Marchmont","Gus Hartnell"],
   "O": ["repertory theatre","philharmonic society","opera house trust",
         "little gallery","choral union","arts circle","picture palace",
         "amateur operatics","poetry circle","civic orchestra"],
   "extra": "mid-century playbills and ration-era costumes"},
 "telecoms": {
   "N": ["Priya Raghavan","Callum Docherty","Yara Haddad","Tom Whitlock",
         "Sanne Vos","Deji Adeyemi","Marit Lindqvist","Owen Brackley",
         "Hana Kovac","Fergus Tierney","Lena Ostrowski","Kwame Boateng"],
   "O": ["fibre rollout crew","community radio station","mast maintenance team",
         "broadband co-op","openreach depot","village network group",
         "switchboard heritage project","cable-laying gang","relay engineers' club",
         "digital inclusion hub"],
   "extra": "modern splicing vans and node cabinets"},
}
PACKETS = [("037_storytelling","agriculture"),("038_storytelling","culture"),
           ("039_storytelling","telecoms")]
QFIX = ("Which of the two endings is the most plausible correct ending to the "
        "story?\nChoices:\nA. {a}\nB. {b}")

def subst(t, name, org):
    return t.replace("{N}", name).replace("{O}", org)

def make_packet(fname, dom, n=125):
    rng = random.Random(fname)
    fill = FILL[dom]
    rows, guard, flip = [], 0, False
    seen_endings = set()
    want = n
    while len(rows) < want:
        guard += 1
        if guard > 8000:
            raise RuntimeError(f"{fname}: stalled at {len(rows)}")
        tmpl = S[len(rows) % len(S)]
        name = rng.choice(fill["N"])
        org = rng.choice(fill["O"])
        ctx = subst(tmpl[0], name, org)
        padding = [f" The affair was still talked over at the {org}'s meetings for seasons afterwards.",
                   f" Nobody in the {org} quite forgot how the week had gone.",
                   f" The story entered the {org}'s unofficial chronicle the same year."]
        pi = 0
        while len(ctx.split()) < 68:
            ctx += padding[pi % len(padding)]
            pi += 1
        good = subst(tmpl[1], name, org)
        absurd = subst(tmpl[2], name, org)
        if (good, absurd) in seen_endings:
            continue
        seen_endings.add((good, absurd))
        flip = not flip
        a, b = (absurd, good) if flip else (good, absurd)
        q = QFIX.format(a=a, b=b)
        rat = tmpl[3].replace("{N}", name)
        r = {"source_dataset":"AUG_GLM2","slice":"C","category":"storytelling",
             "provenance":"none","question":q,"context":ctx,"passage":"",
             "targets":[good],"rationale":rat,"source":"augmented","source_id":""}
        errs = ING.check(r)
        if errs:
            raise RuntimeError(f"{fname}: {errs}\n{json.dumps(r,indent=1)[:800]}")
        rows.append(r)
    out = f"/home/g2/Mohamed/temporal-news-reasoning/data/glm_raw_v13/{fname}.txt"
    with open(out,"w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"wrote {out}: {len(rows)} rows")

for fname, dom in PACKETS:
    make_packet(fname, dom)
