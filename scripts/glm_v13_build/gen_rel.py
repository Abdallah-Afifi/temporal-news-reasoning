import json, random, itertools
from collections import Counter

MONTHS=["January","February","March","April","May","June","July","August","September","October","November","December"]
VOCAB=["BEFORE","AFTER","IS_INCLUDED","SIMULTANEOUS","INCLUDES","DURING","IDENTITY","IMMEDIATELY BEFORE","IMMEDIATELY AFTER"]

def dstr(y,m,d): return f"{d} {MONTHS[m-1]} {y}"

def rand_date(rng,y0,y1,force2d=True):
    y=rng.randint(y0,y1)
    m=rng.randint(1,12)
    d=rng.randint(10,28) if force2d else rng.randint(1,28)
    return (y,m,d)

def after(rng,ev):
    # a full date strictly after period start choices; returns (y,m)
    y,m,_=ev
    ny=y+rng.randint(1,4)
    nm=rng.randint(1,12)
    if ny>y: return (ny,nm)
    return (ny,max(m+1,12) if m<12 else 12)

POOLS={
"003_relation":dict(
 seed=3031, era=(1936,1965), rows=125,
 orgs=["the borough of Alderway","the city of Varnmouth","Whitcombe district council","the county of Sorby","the parish of Netherbridge","the corporation of Saltrow","the urban district of Quayle","the rural district of Owles","the council of Fenwick","the borough of Martello","Kestrel city council","the combined area authority","the borough of Corrafold","the municipality of Trelver","the wards of Bishopbrigg","the town of Harkness","the city surveyor's department","the Watchet utilities board","the governors of the gas undertaking","the listeners of the civic wireless","the borough of Duncairn","the district of Marlin","the city of Port Ferris","the borough of Grimsholm"],
 people=["Councillor Ada Fenn","Mayor Otto Kelham","Clerk Wilhelmina Rye","Alderman Silas Punt","Surveyor Nora Quill","Councillor Emil Vasser","Mayor Beatrice Corr","Treasurer Hal Dunwoody","Councillor Gwen Paskell","Borough Engineer Tomas Rilik","Mayor Aldric Vane","Councillor Rosa Whitlock"],
 things=["civic hall","swimming baths","estate office","housing estate","branch library","fire station","abattoir","council depot","cemetery extension","parks nursery","municipal laundry","welfare clinic","bus station","art gallery","sewage works","water tower","allotment gardens","mortuary chambers","weights and measures office","rating office","road widening scheme","town planning scheme","clearance area","ring road","main sewer","footbridge","memorial garden","civic restaurant","school clinic","maternity home","trolleybus route","tram extension","gasworks retort house","electricity showrooms","dust destructor","refuse tip","cattle market","corn exchange","public baths slipper wing","air raid precaution post","civic film unit","museum annexe"],
 point_vps=[("opened","opened the new {t}"),("unveiled","unveiled the plans for the {t}"),("adopted","adopted the scheme for the {t}"),("sealed","sealed the byelaws for the {t}"),("licensed","licensed the {t}"),("held","held the opening ceremony of the {t}"),("cut","cut the first sod of the {t}"),("laid","laid the foundation stone of the {t}"),("signed","signed the contract for the {t}"),("declared","declared the {t} open")],
 int_vps=[("ran","ran the {t} programme"),("operated","operated the {t}"),("maintained","maintained the {t}"),("administered","administered the {t} scheme"),("conducted","conducted the {t} survey"),("kept","kept the {t} register"),("staffed","staffed the {t}"),("managed","managed the {t} undertaking")]),
"004_relation":dict(
 seed=3041, era=(2020,2024), rows=125,
 orgs=["the Granville logistics hub","Nordfield hospital trust","the university of Solsgirth","the Corrin food plant","the transit workers' branch","the Ashville colleges union","the mail sorting centre at Brimm","the Quayle shipyard workforce","the county care workers","the Dylvar smelter staff","the cinema workers' association","the brewery employees at Halstead","the port health officers","the school support staff of Wendmoor","the refuse collectors of Trelech","the court ushers' branch","the museum workers at Penhale","the airport ground crew","the bakery night shift at Ovate","the telecoms engineers' forum","the water board operatives","the harvest workers' guild","the print finishers at Delve","the bus garage staff"],
 people=["steward Marta Silvan","convener Bohumil Krejcar","branch secretary Ruby Osei","negotiator Felicity Shaw","union lead Tomás Brehm","rep Ingrid Halvorsen","organiser Yusuf Dembé","chair Fatima El Mansour","steward Samuel Adeyemi","officer Ilse Verhoeven","president Nadia Rahimi","spokesman Owen Blackwell"],
 things=["pay offer","recognition agreement","strike ballot","walkout","overtime ban","picket rota","hardship fund","strike notice","return-to-work deal","consultation ballot","travel allowance","annual hours deal","shift premium claim","redundancy framework","fitting allowance","closed-shop clause","union learning fund","safety steward role","joint council","pay formula"],
 point_vps=[("balloted","balloted the members on the {t}"),("launched","launched the {t}"),("suspended","suspended the {t}"),("signed","signed the {t}"),("posted","posted the {t}"),("ratified","ratified the {t}"),("tabled","tabled the {t}"),("withdrew","withdrew the {t}"),("announced","announced the {t}"),("counted","counted the {t} votes")],
 int_vps=[("ran","ran the {t}"),("operated","operated the {t}"),("maintained","maintained the {t}"),("pursued","pursued the {t}"),("conducted","conducted the {t}"),("held","held the {t}"),("staffed","staffed the {t}"),("administered","administered the {t}")]),
"005_relation":dict(
 seed=3051, era=(2005,2014), rows=125,
 orgs=["the Auralia space agency","the Meridian observatory","Helix launch services","the deep space network at Kestrel","the planetary science institute","the Vellby telescope consortium","the astronaut corps of Sorland","the solar physics laboratory","the radio array at Portillo","the comet centre at Halvorsen","the orbital mechanics group","the planetary protection office","the launch range at Corrafold","the astronomy faculty at Duncairn","the satellite control centre","the meteor office space desk","the exoplanet screening group","the astrometry service","the test range at Owles","the student rocketry collective","the balloon physics unit","the eclipse working group","the space medicine board","the tracking station network"],
 people=["director Amara Osei","professor Elard Rieux","engineer Mei-Ling Chua","observatory head Halim Corr","mission lead Greta Lindqvist","navigator Paulo Andrade","scientist Nadia Rahimi","programme chief Owen Blackwell","flight director Clara Witzen","optician Rafael Ochoa","payload manager Sam Adeyemi","chair Ingrid Halvorsen"],
 things=["lunar probe","solar observatory","radio dish","test payload","rover prototype","sample return capsule","plasma monitor","star tracker","launch tower","clean room","mirror blank","frequency licence","landing zone","tracking campaign","spectrum survey","orbit insertion burn","attitude control suite","deep space antenna","altitude record attempt","student cubesat"],
 point_vps=[("launched","launched the {t}"),("commissioned","commissioned the {t}"),("detected","detected the first signal from the {t}"),("imaged","imaged the {t}"),("aborted","aborted the {t}"),("funded","approved the funding for the {t}"),("certified","certified the {t}"),("decommissioned","decommissioned the {t}"),("calibrated","calibrated the {t}"),("tracked","first tracked the {t}")],
 int_vps=[("ran","ran the {t} campaign"),("operated","operated the {t}"),("supported","supported the {t}"),("conducted","conducted the {t} survey"),("maintained","maintained the {t}"),("tracked","tracked the {t}"),("commissioned","commissioned the {t} network"),("upgraded","upgraded the {t}")]),
"006_relation":dict(
 seed=3061, era=(1966,1989), rows=75,
 orgs=["the Cornish fire mutual","Lakesure underwriters","the hail pool of the eastern counties","the storm officers of Saltrow","the earthquake commission","the frost mutual of Netherbridge","the marine board at Quayle","the perils bureau","the householders' assurance office","the crop insurers of Owles","the windstorm syndicate","the loss adjusters' panel","the cat risk consortium","the Surehaven company","the municipal insurers' fund"],
 people=["chief adjuster Marta Silvan","actuary Felicity Shaw","assessor Hal Dunwoody","claims manager Rolf Neumann","underwriter Gwen Paskell","surveyor Bohumil Krejcar","registrar Nadia Rahimi","adjuster Paulo Andrade"],
 things=["windstorm cover","flood claims file","hail endorsement","freeze payout scheme","subsidence register","cat bond prospectus","reinsurance treaty","storm loss adjustment","crop hail policy","fire reinsurance slip","loss survey","claims hotline","retrocessional cover","perils map","risk retention group"],
 point_vps=[("opened","opened the {t}"),("settled","settled the {t}"),("issued","issued the {t}"),("priced","priced the {t}"),("filed","filed the {t}"),("paid","paid the {t}"),("renewed","renewed the {t}"),("quantified","quantified the {t}"),("reported","reported the {t}"),("arbitrated","arbitrated the {t}")],
 int_vps=[("ran","ran the {t}"),("operated","operated the {t}"),("paid","paid the {t} round"),("administered","administered the {t}"),("maintained","maintained the {t}"),("conducted","conducted the {t}"),("funded","funded the {t}"),("kept","kept the {t} open")]),
}

def build(pid):
    P=POOLS[pid]; rng=random.Random(P["seed"]); y0,y1=P["era"]
    rows=[]; seen=set()
    def subj():
        return rng.choice(P["orgs"]+P["people"])
    def pev():  # (sentence, verb, date)
        t=rng.choice(P["things"]); verb,vp=rng.choice(P["point_vps"])
        s=rng.choice(P["orgs"]+P["people"]); d=rand_date(rng,y0,y1)
        return f"{s} {vp.format(t=t)} on {dstr(*d)}.", verb, d
    def iev(): # interval
        t=rng.choice(P["things"]); verb,vp=rng.choice(P["int_vps"])
        s=rng.choice(P["orgs"]+P["people"])
        while True:
            a=rand_date(rng,y0,y1); b=rand_date(rng,y0,y1)
            if b>a and (b[0]-a[0])>=1: break
        return f"{s} {vp.format(t=t)} from {dstr(*a)} to {dstr(*b)}.", verb, a, b
    def deduce(e1,e2):
        # e: ("p",d) or ("i",a,b) -> label of e1 vs e2
        if e1[0]=="p" and e2[0]=="p":
            d1,d2=e1[1],e2[1]
            return "BEFORE" if d1<d2 else ("AFTER" if d1>d2 else "SIMULTANEOUS")
        if e1[0]=="i" and e2[0]=="p":
            a,b,p=e1[1],e1[2],e2[1]
            return "INCLUDES" if a<=p<=b else ("BEFORE" if b<p else "AFTER")
        if e1[0]=="p" and e2[0]=="i":
            p,a,b=e1[1],e2[1],e2[2]
            return "IS_INCLUDED" if a<=p<=b else ("BEFORE" if p<a else "AFTER")
        a1,b1,a2,b2=e1[1],e1[2],e2[1],e2[2]
        if a1<=a2 and b2<=b1: return "INCLUDES"
        if a2<=a1 and b1<=b2: return "IS_INCLUDED"
        return "BEFORE" if b1<a2 else "AFTER"
    def point_in(a,b):
        for _ in range(300):
            y=rng.randint(a[0],b[0])
            mlo=a[1] if y==a[0] else 1
            mhi=b[1] if y==b[0] else 12
            if mlo>mhi: continue
            m=rng.randint(mlo,mhi)
            dlo=a[2] if (y==a[0] and m==a[1]) else 10
            dhi=b[2] if (y==b[0] and m==b[1]) else 28
            if dlo>dhi: continue
            d=rng.randint(dlo,dhi)
            p=(y,m,d)
            if a<=p<=b: return p
        return None
    def interval():
        a=rand_date(rng,y0,y1-1)
        by=min(y1,a[0]+rng.randint(1,3))
        b=(by,rng.randint(1,12),rng.randint(10,28))
        return a,b
    def make_ev(shape,label):
        if shape=="pp":
            d1=rand_date(rng,y0,y1)
            if label=="BEFORE":
                if d1[0]==y1 and d1[1]==12 and d1[2]>20: d1=(y1,12,20)
                while True:
                    d2=rand_date(rng,y0,y1)
                    if d2>d1: break
            elif label=="AFTER":
                if d1[0]==y0 and d1[1]==1 and d1[2]<20: d1=(y0,1,20)
                while True:
                    d2=rand_date(rng,y0,y1)
                    if d2<d1: break
            else:
                d2=d1
            return ("p",d1),("p",d2)
        a,b=interval()
        p=point_in(a,b)
        while p is None:
            a,b=interval(); p=point_in(a,b)
        if shape=="ip": return ("i",a,b),("p",p)
        return ("p",p),("i",a,b)
    # ---------- MCQ rows ----------
    mcq_plan=Counter()
    n=P["rows"]
    n_mcq=(n*63)//125 if n==125 else (n*63)//125
    for L in ["BEFORE","AFTER","IS_INCLUDED","SIMULTANEOUS","INCLUDES"]:
        mcq_plan[L]={"BEFORE":13,"AFTER":13,"IS_INCLUDED":12,"SIMULTANEOUS":12,"INCLUDES":13}[L] if n==125 else round((n*63)//125/5)+1
    mcq=[]
    for L,c in mcq_plan.items(): mcq += [L]*c
    mcq=mcq[:n_mcq]; rng.shuffle(mcq)
    pos=0
    for label in mcq:
        if label=="SIMULTANEOUS":
            e1,e2=make_ev("pp",label)
            s1,v1=pev()[0],None
            # build same-day sentence
            t1=rng.choice(P["things"]); verb1,vp1=rng.choice(P["point_vps"])
            t2=rng.choice(P["things"]); verb2,vp2=rng.choice(P["point_vps"])
            s1=rng.choice(P["orgs"]+P["people"]); s2=rng.choice(P["orgs"]+P["people"])
            while s2==s1: s2=rng.choice(P["orgs"]+P["people"])
            d=e1[1]
            qtext=(f"On {dstr(*d)}, {s1} {vp1.format(t=t1)}, and {s2} {vp2.format(t=t2)} on that same date.")
            rat=f"Both events bear the same date, {dstr(*d)}, so they happened at the same time."
        else:
            if label=="BEFORE" or label=="AFTER":
                e1,e2=make_ev("pp",label)
                a1,vb1,_=pev(); a2,vb2,_=pev()
                s1=f"{pev_sentence(P,rng,e1)}" if False else None
                # generate sentences carrying the exact dates
                def psent(e):
                    t=rng.choice(P["things"]); verb,vp=rng.choice(P["point_vps"])
                    s=rng.choice(P["orgs"]+P["people"])
                    return f"{s} {vp.format(t=t)} on {dstr(*e[1])}."
                qtext=psent(e1)+" "+psent(e2)
            elif label=="INCLUDES":
                e1,e2=make_ev("ip",label)
                def isent(e):
                    t=rng.choice(P["things"]); verb,vp=rng.choice(P["int_vps"])
                    s=rng.choice(P["orgs"]+P["people"])
                    return f"{s} {vp.format(t=t)} from {dstr(*e[1])} to {dstr(*e[2])}."
                def psent(e):
                    t=rng.choice(P["things"]); verb,vp=rng.choice(P["point_vps"])
                    s=rng.choice(P["orgs"]+P["people"])
                    return f"{s} {vp.format(t=t)} on {dstr(*e[1])}."
                qtext=isent(e1)+" "+psent(e2)
            else: # IS_INCLUDED
                e1,e2=make_ev("pi",label)
                def isent(e):
                    t=rng.choice(P["things"]); verb,vp=rng.choice(P["int_vps"])
                    s=rng.choice(P["orgs"]+P["people"])
                    return f"{s} {vp.format(t=t)} from {dstr(*e[1])} to {dstr(*e[2])}."
                def psent(e):
                    t=rng.choice(P["things"]); verb,vp=rng.choice(P["point_vps"])
                    s=rng.choice(P["orgs"]+P["people"])
                    return f"{s} {vp.format(t=t)} on {dstr(*e[1])}."
                qtext=psent(e1)+" "+isent(e2)
            assert deduce(e1,e2)==label,(e1,e2,label,deduce(e1,e2))
            # rationale
            if label=="BEFORE": rat=(f"The first event is dated {evd(e1)} and the second {evd(e2)}, so the first came earlier.")
            elif label=="AFTER": rat=(f"The first event is dated {evd(e1)} and the second {evd(e2)}, so the first came later.")
            elif label=="INCLUDES": rat=(f"The first event spans {evd(e1)} to {evd(e1,True)} and the second falls on {evd(e2)} inside that span, so the first includes it.")
            else: rat=(f"The first event falls on {evd(e1)} and the second spans {evd(e2)} to {evd(e2,True)}, so the first is included in it.")
        # options
        dis=rng.sample([v for v in VOCAB if v!=label],2)
        opts=[label]+dis
        gi=pos%3; opts=[opts[(i-gi)%3] for i in range(3)]; pos+=1
        q=qtext+" What is the relationship between the events?\nChoices:\nA. "+opts[0]+"\nB. "+opts[1]+"\nC. "+opts[2]
        key=q[:140].lower()
        if key in seen: continue
        seen.add(key)
        rows.append({"source_dataset":"AUG_GLM2","slice":"B","category":"relation","provenance":"none",
          "question":q,"context":"","targets":[label],"rationale":rat,"source":"augmented","source_id":""})
    # ---------- bare rows ----------
    bare=mcq_plan
    n_bare=n-len(rows)
    bl=[]
    base={"BEFORE":12,"AFTER":12,"IS_INCLUDED":13,"SIMULTANEOUS":12,"INCLUDES":13} if n==125 else {"BEFORE":7,"AFTER":7,"IS_INCLUDED":8,"SIMULTANEOUS":7,"INCLUDES":8}
    for L,c in base.items(): bl += [L]*c
    bl=bl[:n_bare]; rng.shuffle(bl)
    for label in bl:
        t=rng.choice(P["things"])
        if label in ("BEFORE","AFTER","SIMULTANEOUS"):
            verb,vp=rng.choice(P["point_vps"]); s=rng.choice(P["orgs"]+P["people"])
            d=rand_date(rng,y0,y1)
            if label=="SIMULTANEOUS":
                time=dstr(*d); rat=f"The event is dated {time}, exactly the time named."
            elif label=="BEFORE":
                ny=d[0]+rng.randint(1,4); nm=rng.randint(1,12)
                time=f"{MONTHS[nm-1]} {ny}"
                if (ny,nm,1)<=d: time=f"{ny}"
                rat=f"The event is dated {dstr(*d)}, which is before {time}."
            else:
                ny=d[0]-rng.randint(1,4); nm=rng.randint(1,12)
                time=f"{MONTHS[nm-1]} {ny}"
                if (ny,nm,28)>=d: time=f"{ny}"
                rat=f"The event is dated {dstr(*d)}, which is after {time}."
            sent=f"{s} {vp.format(t=t)} on {dstr(*d)}."
        elif label=="IS_INCLUDED":
            verb,vp=rng.choice(P["point_vps"]); s=rng.choice(P["orgs"]+P["people"])
            d=rand_date(rng,y0,y1); time=f"{MONTHS[d[1]-1]} {d[0]}"
            rat=f"The event is dated {dstr(*d)}, which falls inside {time}."
            sent=f"{s} {vp.format(t=t)} on {dstr(*d)}."
        else: # INCLUDES
            verb,vp=rng.choice(P["int_vps"]); s=rng.choice(P["orgs"]+P["people"])
            a=rand_date(rng,y0,y1); b=(a[0]+rng.randint(2,5),rng.randint(1,12),rng.randint(10,28))
            if b[0]>y1: b=(a[0]+1,rng.randint(1,12),rng.randint(10,28))
            my=rng.randint(a[0]+ (1 if a[1]==12 else 0), b[0]- (1 if b[1]==1 else 0))
            mm=rng.randint(1,12)
            time=f"{MONTHS[mm-1]} {my}"
            if not (a<=(my,mm,15)<=b): time=f"{my}"
            rat=f"The event spans {dstr(*a)} to {dstr(*b)}, an interval that contains {time}."
            sent=f"{s} {vp.format(t=t)} from {dstr(*a)} to {dstr(*b)}."
        q=f"{sent} What is the relationship between the event '{verb}' and the time '{time}'?"
        key=q[:140].lower()
        if key in seen: continue
        seen.add(key)
        rows.append({"source_dataset":"AUG_GLM2","slice":"B","category":"relation","provenance":"none",
          "question":q,"context":"","targets":[label],"rationale":rat,"source":"augmented","source_id":""})
    return rows

def evd(e,second=False):
    if e[0]=="p": return dstr(*e[1])
    return dstr(*(e[2] if second else e[1]))

def pev_sentence(P,rng,e):
    t=rng.choice(P["things"]); verb,vp=rng.choice(P["point_vps"])
    s=rng.choice(P["orgs"]+P["people"])
    return f"{s} {vp.format(t=t)} on {dstr(*e[1])}."

if __name__=="__main__":
    import sys
    for pid,out in [("003_relation","data/glm_raw_v13/003_relation.txt"),
                    ("004_relation","data/glm_raw_v13/004_relation.txt"),
                    ("005_relation","data/glm_raw_v13/005_relation.txt"),
                    ("006_relation","data/glm_raw_v13/006_relation.txt")]:
        rows=build(pid)
        print(pid,"rows:",len(rows),
              "labels:",dict(Counter(r["targets"][0] for r in rows)))
        open(out,'w').write("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
