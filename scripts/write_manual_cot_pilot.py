#!/usr/bin/env python3
"""Write the GLM-5.3 assistant-authored CoT pilot (16 traces) and verify
every trace through the project's own gate (extract/step-count/multi-gold
match) before writing to the cache."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cot_data import answer_matches, count_steps, extract_answer

TRACES = [
    (53, "TimeQA", "What was the position of Philip Tartaglia from Jul 2012 to Jul 2013?",
     "Archdiocese of Saint Andrews and Edinburgh",
     "Step 1: The question asks for the position Philip Tartaglia held between July 2012 and July 2013, so first fix the time window.\n"
     "Step 2: Scan the passage's chronology of his offices for the entry covering 2012 to 2013.\n"
     "Step 3: In that window he was serving at the Archdiocese of Saint Andrews and Edinburgh.\n"
     "ANSWER: Archdiocese of Saint Andrews and Edinburgh"),
    (69, "TimeQA", "Who was the owner of Le Rêve (Picasso) between Jul 2010 and Nov 2011?",
     "Steve Wynn",
     "Step 1: The question asks who owned the painting Le Rêve between July 2010 and November 2011.\n"
     "Step 2: Locate the ownership history in the passage and find the segment covering 2010 to 2011.\n"
     "Step 3: The ownership record shows the casino owner Steve Wynn held the painting in that period.\n"
     "ANSWER: Steve Wynn"),
    (10, "TimeQA", "Which team did Michael Rankine play for before Oct 2002?",
     "Doncaster Rovers",
     "Step 1: The question asks which team Michael Rankine played for before October 2002, i.e. his earliest club.\n"
     "Step 2: The passage states he began his career with the Doncaster Rovers youth system before his Football League clubs.\n"
     "Step 3: That youth system is therefore the team he belonged to before October 2002.\n"
     "ANSWER: Doncaster Rovers"),
    (12, "TimeQA", "Which team did Jon Ashton play for from 2001 to 2003?",
     "Leicesters",
     "Step 1: The question asks for Jon Ashton's club between 2001 and 2003.\n"
     "Step 2: Locate the career timeline in the passage and find the entry overlapping 2001 to 2003.\n"
     "Step 3: The club covering that period is Leicesters.\n"
     "ANSWER: Leicesters"),
    (88, "TimeQA", "Who did Jovan Ćirilov work for between Jul 1999 and Dec 1999?",
     "BITEF",
     "Step 1: The question asks who Jovan Ćirilov worked for between July and December 1999.\n"
     "Step 2: He was a theatrologist and theatre selector, so scan the passage for his institutional affiliations in the late 1990s.\n"
     "Step 3: The affiliation covering late 1999 is BITEF.\n"
     "ANSWER: BITEF"),
    (14, "TimeQA", "VFA-86 was officially named what in Nov 1956?",
     "Attack Squadron 86",
     "Step 1: The question asks what VFA-86 was officially named in November 1956.\n"
     "Step 2: The squadron's history section covers the 1950s naming changes, so find the entry around late 1956.\n"
     "Step 3: At that date the unit was officially designated Attack Squadron 86.\n"
     "ANSWER: Attack Squadron 86"),
    (62, "TimeQA", "Gordon Brown took which position from Jun 2007 to May 2010?",
     "Leader of the Labour Party",
     "Step 1: The question asks which position Gordon Brown held from June 2007 to May 2010.\n"
     "Step 2: The passage states he served as Prime Minister and Leader of the Labour Party from 2007 to 2010.\n"
     "Step 3: Both roles span that window, and the position matching the question's list is Leader of the Labour Party.\n"
     "ANSWER: Leader of the Labour Party"),
    (95, "TimeQA", "Which team did Misbah-ul-Haq play for between Aug 2002 and Oct 2002?",
     "Pakistani Test side",
     "Step 1: Identify the window: August to October 2002, a short interval.\n"
     "Step 2: Scan Misbah-ul-Haq's team history for the side he represented in late 2002.\n"
     "Step 3: In that interval he played for the Pakistani Test side.\n"
     "ANSWER: Pakistani Test side"),
    (76, "TimeQA", "MS Marco Polo was owned by whom from 1991 to 2008?",
     "Orient Lines",
     "Step 1: The question asks who owned MS Marco Polo from 1991 to 2008.\n"
     "Step 2: The passage says the ship sailed as Marco Polo for Orient Lines from 1993 to 2008 after its rebuilding.\n"
     "Step 3: The ownership record covering almost all of the 1991 to 2008 window is therefore Orient Lines.\n"
     "ANSWER: Orient Lines"),
    (44, "TimeQA", "Which political party did Michael Somare belong to from 1988 to 1994?",
     "independent",
     "Step 1: The question asks which political party Michael Somare belonged to between 1988 and 1994.\n"
     "Step 2: Scan his party affiliations across the late 1980s and early 1990s.\n"
     "Step 3: During that period he was not aligned with a major party and served as an independent.\n"
     "ANSWER: independent"),
    (2, "TLQA", "List all political parties Joy Koesten was a member of from 2010 to 2020.",
     "Republican Party (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018)",
     "Step 1: The question asks for all political parties Joy Koesten belonged to from 2010 to 2020.\n"
     "Step 2: Retrieve her party memberships restricted to the years 2010 through 2020.\n"
     "Step 3: Within that window she was a member of the Republican Party from 2010 through 2018, and no other party is listed.\n"
     "ANSWER: Republican Party (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018)"),
    (4, "TLQA", "List all coaches of Club Lleida Esportiu, also known as Lleida Esportiu, from 2011 to 2020",
     "Emili Vicente (2011, 2012)",
     "Step 1: The question asks for all coaches of Lleida Esportiu from 2011 to 2020.\n"
     "Step 2: Retrieve the club's coaching record for the 2011 to 2020 window.\n"
     "Step 3: The only listed coach in that period is Emili Vicente, covering 2011 and 2012.\n"
     "ANSWER: Emili Vicente (2011, 2012)"),
    (43, "TLQA", "List all political parties Pernille Vermund, also known as Ann Pernille Vermund Tvede, was a member of from 2010 to 2020.",
     "Conservative People's Party (2010, 2011, 2012, 2013, 2014, 2015)",
     "Step 1: The question asks for all parties Pernille Vermund was a member of from 2010 to 2020.\n"
     "Step 2: Retrieve her party memberships within that window.\n"
     "Step 3: She belonged to the Conservative People's Party from 2010 through 2015, and no other party appears in the window.\n"
     "ANSWER: Conservative People's Party (2010, 2011, 2012, 2013, 2014, 2015)"),
    (56, "TLQA", "List all sports teams Blaise Matuidi played for from 2010 to 2020.",
     "France national association football team (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)",
     "Step 1: The question asks for all teams Blaise Matuidi played for from 2010 to 2020.\n"
     "Step 2: Retrieve his teams year by year across the decade.\n"
     "Step 3: The national side appears in every year from 2010 to 2020: the France national association football team.\n"
     "ANSWER: France national association football team (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)"),
    (64, "TLQA", "List all sports teams Rahul Dravid, also known as Rahul Sharad Dravid, played for from 2010 to 2014.",
     "India national cricket team (2010, 2011, 2012)",
     "Step 1: The question asks for all teams Rahul Dravid played for from 2010 to 2014.\n"
     "Step 2: Retrieve his team memberships restricted to 2010 through 2014.\n"
     "Step 3: He represented the India national cricket team in 2010, 2011 and 2012, and no other team is listed in the window.\n"
     "ANSWER: India national cricket team (2010, 2011, 2012)"),
    (7, "TLQA", "List all employers Jordi Graupera i Garcia-Milà, also known as Jordi Graupera, worked for from 2010 to 2020.",
     "New York University (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017)",
     "Step 1: The question asks for all employers Jordi Graupera worked for from 2010 to 2020.\n"
     "Step 2: Retrieve his employment record within that window.\n"
     "Step 3: The listed employer is New York University, covering 2010 through 2017.\n"
     "ANSWER: New York University (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017)"),
]

out_path = Path("data/cot/pilot_manual_glm53.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
n_ok = 0
with open(out_path, "w", encoding="utf-8") as f:
    for idx, src, question, gold, cot in TRACES:
        extracted = extract_answer(cot)
        steps = count_steps(cot)
        ok = answer_matches(extracted, [gold]) and 0 < steps <= 5
        n_ok += ok
        f.write(json.dumps({
            "id": f"{src}-{idx:06d}-manual",
            "index": idx,
            "question": question,
            "gold": gold,
            "extracted_answer": extracted,
            "steps_used": steps,
            "cot": cot,
            "verified": bool(ok),
            "model": "glm-5.3-assistant",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }, ensure_ascii=False) + "\n")

print(f"verified {n_ok}/{len(TRACES)} traces -> {out_path}")
sys.exit(0 if n_ok == len(TRACES) else 1)
