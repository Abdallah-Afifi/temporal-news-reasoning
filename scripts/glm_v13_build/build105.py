import json, random
from collections import Counter

EVENTS=[("the Kesselberg Observatory was founded",1892),("Comet Reyes was discovered",1903),
("the Meridian Peak spectrograph was installed",1912),("the great mirror was cast at the Delft foundry",1919),
("the eclipse expedition sailed from Bristol",1898),("the first photographic atlas of the moon appeared",1905),
("the Astronomical Society's journal was launched",1893),("the meteorite fell at Hollow Fen",1921),
("the great aurora was observed across Europe",1909),("the double star Eta Corvi was catalogued",1895),
("the mountain station at Portillo was completed",1926),("the wireless time signal began from the tower",1927),
("the planetoid Vesper was first detected",1923),("the comet of the century brightened the spring skies",1910),
("the observatory's dome was rebuilt in steel",1928),("the eclipse of the long totality crossed the county",1919),
("the transit circle was mounted",1899),("the star atlas's second volume was published",1902),
("the meteor section was formed",1904),("the variable star programme began",1908),
("the jubilee exhibition of instruments opened",1917),("the Schmidt camera was patented",1931),
("the night school's astronomy class enrolled its first hundred",1922),("the observatory received its royal charter",1907),
("the sunspot count reached its recorded maximum",1917),("the spectroscopic binary was announced",1914),
("the lunar map was revised at last",1930),("the comet seeker's medal was struck",1896),
("the transit of Mercury was observed",1907),("the nebular hypothesis debate flared at the congress",1906),
("the photographic plate vault was built",1913),("the clock drive was electrified",1924),
("the satellite of Minerva was confirmed",1916),("the society's library reached ten thousand volumes",1925),
("the polar observatory was abandoned",1915),("the eclipse flight was attempted by aeroplane",1932),
("the cataloguing of the southern stars finished",1920),("the radio echoes from the moon were claimed",1933),
("the dome's shutters were motorised",1929),("the photographic zenith tube entered service",1911)]

rng=random.Random(20501)
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
open('data/glm_raw_v13/105_ordering.txt','w').write("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n")
