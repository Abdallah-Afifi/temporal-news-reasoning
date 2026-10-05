import json, random
from collections import Counter

EVENTS=[("Storm Ansel's claims were filed",2015),("the catastrophe bond was issued",2016),
("the flood model was recalibrated",2017),("the hail adjusters finished the vineyard round",2019),
("the wildfire subrogation suit began",2018),("the reinsurance renewal concluded",2016),
("the storm surge hotline opened",2017),("the drought endorsement was drafted",2015),
("the risk consortium admitted three members",2016),("the freeze payouts were closed",2016),
("the coastal erosion study reported",2018),("the perils pool's accounts were published",2019),
("the windstorm retrocession was placed",2018),("the claims tribunal sat for the flood",2017),
("the cat bond matured",2019),("the hail swath was mapped",2015),
("the loss adjuster's report was delivered",2016),("the business interruption test case settled",2019),
("the flood survey reached the eastern parishes",2017),("the lightning claims spiked",2018),
("the quota share was renewed early",2015),("the aggregation dispute was arbitrated",2018),
("the storm chase data was licensed",2016),("the parametric trigger was agreed",2017),
("the catastrophe reserve was strengthened",2019),("the subsidence claims surge began",2018),
("the hurricane season exceeded the model",2017),("the marine cargo claims closed",2015),
("the smoothing of premiums was announced",2019),("the ice storm mutual aid concluded",2015),
("the reforestation grant was paid",2016),("the wildfire camp claims were denied",2018),
("the flood fair drew the insurers",2019),("the mobile claims units deployed",2017),
("the broker's cyber cover pilot launched",2016),("the tornado verification started",2019),
("the crop hail board sat",2015),("the storm naming committee expanded",2018),
("the facultative market hardened",2018),("the flood zone remap draft appeared",2019)]

rng=random.Random(20601)
rows=[]
# --- TRUE/FALSE rows (62) ---
pairs=[]
i=0
used=set()
for a in range(len(EVENTS)):
    for b in range(a+1,len(EVENTS)):
        pass
# build pairs by sampling distinct combos
combos=[(a,b) for a in range(len(EVENTS)) for b in range(a+1,len(EVENTS))]
rng.shuffle(combos)
chosen=combos[:62]
tf_gold_true=0
for k,(a,b) in enumerate(chosen):
    ea,ya=EVENTS[a]; eb,yb=EVENTS[b]
    if ya==yb: continue
    claim_fwd = (k%2==0)
    if claim_fwd:
        claim=f"{ea[0].upper()+ea[1:]} before {eb}"
        truth = ya<yb
    else:
        claim=f"{eb[0].upper()+eb[1:]} before {ea}"
        truth = yb<ya
    gold = "TRUE" if truth else "FALSE"
    if gold=="TRUE": tf_gold_true+=1
    opts=["TRUE","FALSE","Undetermined"]
    rot=k%3
    opts=[opts[(i-rot)%3] for i in range(3)]
    q=(f"{ea[0].upper()+ea[1:]} in {ya}. {eb[0].upper()+eb[1:]} in {yb}. {claim} - True/False?\n"
       f"Choices:\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}")
    ra=f"{'The first event' if claim_fwd else 'The second-listed event'} is dated {ya if claim_fwd else yb} and the other {yb if claim_fwd else ya}."
    rows.append({"source_dataset":"AUG_GLM2","slice":"B","category":"ordering","provenance":"none",
      "question":q,"context":"","targets":[gold],
      "rationale":f"The claim compares events dated {ya} and {yb}; the ordering makes the claim {'true' if truth else 'false'}.",
      "source":"augmented","source_id":""})
print("tf rows:",len(rows),"TRUE:",tf_gold_true)
# --- sequence rows (63) ---
seqcount=0
trip=set()
while seqcount<63 and len(trip)<400:
    tr=tuple(sorted(rng.sample(range(len(EVENTS)),3)))
    if tr in trip: continue
    trip.add(tr)
    evs=[EVENTS[i] for i in tr]
    ys=[y for _,y in evs]
    if len(set(ys))<3: continue
    order=sorted(range(3),key=lambda i:ys[i])
    # presentation shuffled: identity permutation at most ~1/3 of the time
    pres=list(order)
    while True:
        rng.shuffle(pres)
        if list(pres)!=order or rng.random()<0.33:
            break
    # gold letters = chronological order of displayed positions
    disp=[evs[j] for j in pres]           # displayed events
    dispdates=[ys[j] for j in pres]
    goldseq=sorted(range(3),key=lambda j:dispdates[j])  # indices into displayed
    def fmt(p): return ", ".join(f"({j+1})" for j in p)
    gold=fmt(goldseq)
    # two distractor perms
    import itertools
    perms=[list(p) for p in itertools.permutations(range(3))]
    goldp=goldseq
    others=[p for p in perms if p!=goldp]
    rng.shuffle(others)
    dis=others[:2]
    opts=[fmt(goldp),fmt(dis[0]),fmt(dis[1])]
    rot=seqcount%3
    opts=[opts[(i-rot)%3] for i in range(3)]
    q=("Arrange the following events in chronological order: "
       + " ".join(f"({n+1}) {t} in {y}" for n,(t,y) in enumerate(disp))
       + f"\nChoices:\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}")
    rows.append({"source_dataset":"AUG_GLM2","slice":"B","category":"ordering","provenance":"none",
      "question":q,"context":"","targets":[gold],
      "rationale":"The dates in chronological order are "+", ".join(f"{disp[j][0]} in {dispdates[j]}" for j in goldseq)+".",
      "source":"augmented","source_id":""})
    seqcount+=1
print("seq rows:",seqcount,"total:",len(rows))
pc=Counter()
for r in rows:
    ch=[x[3:] for x in r["question"].split("Choices:\n")[1].splitlines()]
    g=r["targets"][0]
    if g in ch: pc["ABC"[ch.index(g)]]+=1
print("gold positions:",dict(pc))
import io
open('data/glm_raw_v13/106_ordering.txt','w').write("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n")
