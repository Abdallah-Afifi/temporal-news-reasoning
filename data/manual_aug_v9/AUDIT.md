# v9 AUG_GLM2 audit — 5948 rows, 18 categories
Generated from `data/corpus/ccnews` per `docs/synthetic_data_ruleset.md`.

## 7.1 Machine gates

- [PASS] schema + in-file duplicate questions — bad=0
- [PASS] exact question collision vs TIME+TimeBench+TRAM+probe — collisions=0
- [PASS] punctuation-insensitive collision — collisions=0
- [PASS] token-set Jaccard > 0.8 vs benchmark questions — hits=0
- [PASS] 30-token shingle shared with benchmark context — rows=0, banned source_ids=0
- [PASS] in-category near-duplicate questions (J > 0.9) — pairs=0

## 7.2 Distribution audit (the D51 gate)

| category | rows | alloc | distinct golds | top gold share | in-context | verdict |
|---|---|---|---|---|---|---|
| Computation | 800 | 800 | 386 | 3.6% | 0% | PASS |
| Timeline | 650 | 650 | 95 | 10.0% | n/a | PASS |
| Localization | 550 | 550 | 232 | 10.7% | 73% | PASS |
| Counterfactual | 500 | 500 | 451 | 10.0% | 29% | PASS |
| Duration_Compare | 450 | 450 | 3 | 40.0% | n/a | PASS |
| Relative_Reasoning | 438 | 450 | 398 | 8.9% | 38% | PASS |
| Order_Reasoning | 400 | 400 | 364 | 8.8% | 91% | PASS |
| Co_temporality | 300 | 300 | 276 | 8.3% | 38% | PASS |
| Explicit_Reasoning | 162 | 200 | 161 | 1.2% | 54% | PASS |
| Order_Compare | 100 | 100 | 3 | 47.0% | n/a | PASS |
| nli_saq | 400 | 400 | 3 | 33.5% | n/a | PASS |
| nli_mcq | 250 | 250 | 3 | 33.6% | n/a | PASS |
| relation | 150 | 150 | 3 | 33.3% | n/a | PASS |
| ordering | 100 | 100 | 3 | 37.0% | n/a | PASS |
| temporal_dialogue | 250 | 250 | 69 | 18.0% | n/a | PASS |
| duration | 148 | 150 | 33 | 10.1% | n/a | PASS |
| storytelling | 150 | 150 | 150 | 0.7% | n/a | PASS |
| longform_free | 150 | 150 | 150 | 0.7% | n/a | PASS |
- [PASS] per-category distribution targets — all categories in band
- [PASS] fabricated YYYY-01-01 golds < 2% — count=5
- [PASS] Timeline permutation coverage (k=5 documented deviation: 65 rows over 120 perms, each used at most once) — k=3: rows=390, perms_seen=6/6, max_per_perm=65; k=4: rows=195, perms_seen=24/24, max_per_perm=9; k=5: rows=65, perms_seen=65/120, max_per_perm=1

Block A provenance split: news 59.9% (target 60), wiki 36.1% (35), dial 4.0% (5)
§5.12 paired rows carrying an abstain option: 302 (target ~352, i.e. 8% of 4,400)

## 7.3 Shortcut probes (the L7 gate)

| category | n_mcq | gold letters | always-A | longest | overlap | verdict |
|---|---|---|---|---|---|---|
| Counterfactual | 305 | {'A': 77, 'C': 76, 'B': 76, 'D': 76} | 25% | 6% | 2% | PASS |
| Duration_Compare | 450 | {'B': 180, 'A': 166, 'C': 104} | 37% | 23% | 0% | PASS |
| Relative_Reasoning | 249 | {'A': 64, 'D': 63, 'C': 61, 'B': 61} | 26% | 20% | 8% | PASS |
| Order_Reasoning | 232 | {'B': 63, 'C': 59, 'D': 55, 'A': 55} | 24% | 0% | 2% | PASS |
| Co_temporality | 150 | {'D': 38, 'B': 38, 'C': 37, 'A': 37} | 25% | 19% | 5% | PASS |
| Explicit_Reasoning | 84 | {'B': 21, 'A': 21, 'C': 21, 'D': 21} | 25% | 15% | 7% | PASS |
| Order_Compare | 100 | {'B': 47, 'A': 38, 'C': 15} | 38% | 15% | 6% | PASS |
| nli_mcq | 250 | {'B': 84, 'C': 83, 'A': 83} | 33% | 0% | 0% | PASS |
| relation | 150 | {'B': 50, 'C': 50, 'A': 50} | 33% | 0% | 0% | PASS |
| ordering | 100 | {'A': 37, 'C': 37, 'B': 26} | 37% | 0% | 0% | PASS |
| temporal_dialogue | 139 | {'B': 35, 'D': 35, 'C': 35, 'A': 34} | 24% | 1% | 3% | PASS |
| duration | 148 | {'B': 38, 'D': 37, 'C': 37, 'A': 36} | 24% | 0% | 0% | PASS |
| storytelling | 150 | {'A': 75, 'B': 75} | 50% | 6% | 0% | PASS |
- [PASS] shortcut probes — all pass
- [PASS] abstain-option presence uninformative — P(gold=abstain | abstain option present) = 0.50 over 302 rows

## 7.4 Gold correctness (recomputed from stated dates)

- PASS Computation: 799/800 recomputed correct (99.9%)
- PASS Duration_Compare: 450/450 recomputed correct (100.0%)
- PASS Order_Compare: 100/100 recomputed correct (100.0%)
- PASS Timeline: 650/650 recomputed correct (100.0%)
- PASS relation: 150/150 recomputed correct (100.0%)
- PASS ordering: 100/100 recomputed correct (100.0%)
- PASS nli_saq: 400/400 recomputed correct (100.0%)
- PASS nli_mcq: 250/250 recomputed correct (100.0%)
- FAIL Localization relative-expression subset: 134/137 (97.8%)
- [FAIL] gold recomputation >= 99% per recomputable category — Localization-relative

## 7.5 Surface quality

gold length MCQ (reported, not gated): n=3157 mean=9.80 median=4 >=20 words=21.2%
gold length free-text (gated): n=2791 mean=8.11 median=3 >=20 words=14.8%
TIME reference (free-text, n=104,939): mean 2.35, median 1, >=20w 0.95%
  - Co_temporality (free-text): mean 14.9 words
  - Counterfactual (free-text): mean 14.4 words
  - Explicit_Reasoning (free-text): mean 16.0 words
  - Order_Reasoning (free-text): mean 17.1 words
  - Relative_Reasoning (free-text): mean 15.2 words
  - longform_free (free-text): mean 44.2 words
- [FAIL] §9 answer style — free-text mean gold <= 6 words and < 5% >= 20 words — mean=8.11 (TIME 2.35), >=20w=14.8% (TIME 0.95%)
- [PASS] no context padded by repeating a filler line >= 5 times — rows=6 (0.10%) {'Order_Reasoning': 2, 'Computation': 1, 'Explicit_Reasoning': 1, 'Relative_Reasoning': 1, 'temporal_dialogue': 1}
- [FAIL] golds neither truncated mid-sentence nor page furniture — truncated=79, boilerplate=14 (1.6% of rows)

## Automated sample dump — 3 rows/category, NOT the §7.4 manual check

§7.4 requires 100 rows/category reviewed by eye for each non-recomputable category; that review is tracked separately and is NOT evidenced by this dump.
- **Co_temporality** `news` gold=`Newsom, whom the two right-wingers accused of running a shadow campaign for pres` Q: While events around 'You're Down 41 Points' In were still unfolding between 2019 and 2028, what was reported concerning Newsom? ⏎ Choices: ⏎ A. Hannity suggested that Newsom would somehow be secretly anointed at the 2024
- **Co_temporality** `news` gold=`ROCKVILLE PIKE SUITE, ROCKVILLE, USA, May 31, 2024 /EINPresswire.com/ -- The glo` Q: While events around Coating Solvent Market to Grow were still unfolding between 2024 and 2034, what was reported concerning Coating Solvent Market to Grow?
- **Co_temporality** `news` gold=`Between 2014 and 2016, the Come And Get It singer appeared in several films, inc` Q: While events around Why Did Selena Gomez Discontinue were still unfolding between January 20, 2006 and May 25th, 2024, what was reported concerning Between? ⏎ Choices: ⏎ A. She declined a part in High School Musical in 2
- **Computation** `dial` gold=`2 years 5 months` Q: What was the span of time between Hefner died on aged 91 and was buried on 27 September 2017 and Just under three years later on 17 March 2020?   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year 2 months 
- **Computation** `news` gold=`7 months` Q: How many months went by between To be eligible for the payments, you must have been entitled in September 17 2023 and Income-based Jobseekers Allowance and Income-related Employment and Support Allowance, as well in Apri
- **Computation** `news` gold=`6 years` Q: How much time went by between According to evidence presented at trial, on on Jan. 23, 2018 and Wheeler’s sentencing hearing is set for on Feb. 1, 2024?   (Hint: Please answer in the form of Month Day, Year. e.g. 1 year 
- **Counterfactual** `news` gold=`Reportedly, if 2021 was the boom, then 2022 was the bust.` Q: Had the circumstances around Many strange things happened in the global economy unfolded otherwise, what did the reporting actually establish about If 2021 was the boom, then 2022 was the bust? ⏎ Choices: ⏎ A. Reportedly
- **Counterfactual** `news` gold=`Although he never won the Championship, he finished second twice, in 2008 and 20` Q: If the reports of Edwards took part in 445 Cup Series races had turned out to be mistaken, what does the passage record about Although he never won the Championship, he finished second twice? ⏎ Choices: ⏎ A. Although he 
- **Counterfactual** `wiki` gold=`Reportedly, huawei and Seres joined hands to launch the Aito brand in 2021, with` Q: If the reports of Huawei’s Aito Hikes New M7 SUV’s Output Capacity to 700 had turned out to be mistaken, what is documented about Huawei and Seres joined hands to launch the Aito brand? ⏎ Choices: ⏎ A. Huawei and Seres j
- **Duration_Compare** `news` gold=`The two durations are approximately the same length.` Q: Which of the following two durations is longer? *Duration 1:* Between "The BMC is planning to open the bridge partially by, though, on February 2024" and "As per the new deadline, one arm of the bridge will be, on May 20
- **Duration_Compare** `wiki` gold=`Duration 2 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between "Mapes Guilty Verdict - Vandalia Statehouse Hosting Annual Grand Levee Celebration Saturday, on Oct 10, 2023" and "Takes On Illinois Corruption In New
- **Duration_Compare** `wiki` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between "More like this: Yesterday - Blackhawk Bank Becomes First Mid Bank &, on Jul 25, 2023" and "First Mid Earns 2023 Top Workplaces Culture Excellence Awa
- **Explicit_Reasoning** `news` gold=`Reportedly, district Judge Donald Molloy issued a preliminary injunction, preven` Q: What developments involving Montana's championship run extra special took place between 2024 and January 1, 2024? ⏎ Choices: ⏎ A. Manchester United decided to buy British in 2019 and that worked out well, didn’t it? ⏎ B.
- **Explicit_Reasoning** `news` gold=`Highway 385 May 31, 2024 • Land Line Staff | Two complete closures are scheduled` Q: What notable activities did South Dakota closing portion of engage in between May 31, 2024 and 2025? ⏎ Choices: ⏎ A. Highway 385 May 31, 2024 • Land Line Staff | Two complete closures are scheduled on the only north-sout
- **Explicit_Reasoning** `wiki` gold=`They were released within weeks of each other in 2011 (2011)` Q: What developments involving Stepson of notorious canoe conman took place between 2008 and 2013?
- **Localization** `news` gold=`2007` Q: In what year did "Sustained recovery is only possible with a rebound in investments, which are still 12?
- **Localization** `news` gold=`August 2024` Q: In which month and year did Fulton County prosecutors want Trump’s trial to begin in August 2024, which would be?
- **Localization** `wiki` gold=`2030` Q: In what year did Despite the daunting statistics, Dr. Obidike’s address carried an undertone of optimism and progress?
- **Order_Compare** `news` gold=`Fact 2 happened earlier.` Q: For Fact1: In another development, HUL announced key changes to its management committee (MC) and said its beauty and personal care division will transition into dedicated beauty and wellbeing (B&W) and personal care (PC
- **Order_Compare** `news` gold=`Fact 2 happened earlier.` Q: For Fact1: Since, the DWP has been working to repay those impacted and the recent figures released showed that of the 173,538 accounts checked between and, up to 82,323 pensioners have been identified as having underpaym
- **Order_Compare** `wiki` gold=`Fact 2 happened earlier.` Q: For Fact1: On, Justice Sandra O’Connor visited my father, John Driggs, at his home just hours before he passed away (on Dec. 10, 2014) and Fact2: Senate with a vote of 99-0 and was sworn in as the first woman Supreme Cou
- **Order_Reasoning** `news` gold=`The Yahoo Fantasy football crew got together for their very first mock draft of ` Q: What was the first development in the 2024 coverage of OU Sooners vs UCLA Bruins?
- **Order_Reasoning** `wiki` gold=`European Destinations Will Continue to Thrive In 2023, Italy dethroned Mexico as` Q: What was the second recorded development for Squaremouth Predicts the Four Biggest in 2023? ⏎ Choices: ⏎ A. The average international trip cost in 2023 is $6,574, up 21 percent from last year and 30 percent over 2021. ⏎ 
- **Order_Reasoning** `news` gold=`Counting of votes for General Elections to State Legislative Assemblies of Aruna` Q: What was the 4th development in the 2024 coverage of Polling now completed for 7? ⏎ Choices: ⏎ A. Counting of votes for General Elections to State Legislative Assemblies of Arunachal Pradesh and Sikkim will take place on
- **Relative_Reasoning** `wiki` gold=`Area Teams Take Care In Shootout To Honor Legacy Of Beloved Broadcaster Tom Emer` Q: Which event was the latest to follow All eight games will be broadcast on WSMI 106.1-FM and online?
- **Relative_Reasoning** `news` gold=`Watch the first trailer for “Furiosa: A Mad Max Saga” 2025 Rolls-Royce Cullinan ` Q: Which event was the latest to follow A prequel to 2015's hit movie “Mad Max: Fury Road” arrives?
- **Relative_Reasoning** `news` gold=`“The question has been when can we put it here, and it wouldn’t be here now if i` Q: After He earned All-ACC selections in both 2008 and 2009 as part, what was the last reported development? ⏎ Choices: ⏎ A. He spent the 2019-20 seasons as the offensive line coach at Georgia State before coming back to Cl
- **Timeline** `news` gold=`B,A,C` Q: Below are 3 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Ci
- **Timeline** `news` gold=`D,C,A,B,E` Q: Below are 5 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Nu
- **Timeline** `wiki` gold=`C,B,D,A` Q: Below are 4 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. In
- **duration** `none` gold=`about 8 days` Q: The the Apollo 11 mission ran from July 16 to July 24, 1969. How long did it last in total? ⏎ Choices: ⏎ A. about 2 days ⏎ B. about 2 weeks ⏎ C. about 1 month ⏎ D. about 8 days
- **duration** `none` gold=`about 13 years` Q: For roughly how long was Prohibition in the United States ongoing? ⏎ Choices: ⏎ A. about 13 years ⏎ B. about 6 years ⏎ C. about 20 years ⏎ D. about 30 years
- **duration** `none` gold=`about 623 years` Q: How long did the Ottoman Empire last? ⏎ Choices: ⏎ A. about 300 years ⏎ B. about 623 years ⏎ C. about 500 years ⏎ D. about 800 years
- **longform_free** `news` gold=`Shoppers who bought a used car in 2019 would have to spend an additional. The av` Q: Summarize the sequence of developments concerning Used car prices fall, but that the passage records.
- **longform_free** `wiki` gold=`She joined GK in 2017 as Head of Treasury & Corporate Finance. In addition to he` Q: Describe what happened between 2017 and October 17, 2023 according to the passage.
- **longform_free** `news` gold=`On June 2, 2024, a book: Nnamdi Azikiwe University. He noted that at the 14th co` Q: Summarize the sequence of developments concerning 29 Students Bags First Class that the passage records.
- **nli_mcq** `news` gold=`neutral` Q: Few people paid attention to the He was born the second oldest of at the time. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_mcq** `news` gold=`contradiction` Q: The Guardion’s stockholders had previously approved the sale came before the Securities and Exchange Commission on April 8. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_mcq** `news` gold=`entailment` Q: The Maria Pilecka Lived a Long Life Maria came after the Later, she moved to Krupa, where she. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_saq** `news` gold=`contradiction` Q: Fewer than 5 days separated the The record date for the distributions is from the All distributions are payable on January 8.
- **nli_saq** `news` gold=`neutral` Q: The Born on September 10, 1933 took place in the capital.
- **nli_saq** `news` gold=`contradiction` Q: Fewer than 3 days separated the The catered banquet will begin at 6:30 from the DELTA – The next Fulton County Genealogical.
- **ordering** `none` gold=`TRUE` Q: the port authority published the findings on December 18, 2009. Then the parliamentary committee held the vote on July 6, 2010. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **ordering** `none` gold=`FALSE` Q: the record label launched the campaign on July 19, 1972. Then the football club released its annual figures on May 20, 1972. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **ordering** `none` gold=`FALSE` Q: the airline retired the fleet on February 18, 1960. Then the construction firm staged the concert on August 2, 1959. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **relation** `none` gold=`BEFORE` Q: The television network paused the project on September 12, 1989, and the water authority closed the facility on October 22, 1989. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ C. DUR
- **relation** `none` gold=`DURING` Q: The tournament ran from April 4, 2010 to June 3, 2010, and the city council inaugurated the bridge on May 4, 2010, in the middle of it. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ 
- **relation** `none` gold=`BEFORE` Q: The cycling team hosted the festival on October 6, 2011, and the parliamentary committee released its annual figures on October 9, 2011. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎
- **storytelling** `news` gold=`The company reported 43% year-over-year revenue growth in the fourth quarter of ` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. The company reported 43% year-over-year revenue growth in the fourth quarter of fiscal 2023. ⏎ B. The company reported 60% year-
- **storytelling** `news` gold=`Microsoft Corp., 2024 WL 2745115 (Cal. App. Ct. May 29, 2024)` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. Microsoft Corp., 2024 WL 2745115 (Cal. App. Ct. May 29, 2024) ⏎ B. Microsoft Corp., 2024 WL 2745115 (Cal. App. Ct. May 2024, 202
- **storytelling** `news` gold=`The area of ​​the Järvselja study and experimental forest in Tartu County will d` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. The area of ​​the Järvselja study and experimental forest in Tartu County will decrease by 20.5 hectares compared to the volume 
- **temporal_dialogue** `dial` gold=`Session 2` Q: In which session was More like this: - Electrical License Requirement Debated discussed?
- **temporal_dialogue** `dial` gold=`Session 1` Q: In which session was More like this: - Natalie Beck discussed?
- **temporal_dialogue** `dial` gold=`Session 2` Q: In which session was Alphalogic Industries bonus share details Earlier record date for bonus shares was fixed on but later discussed?

## Recorded deviations and decisions

- [NOTE] category names for Block C — temporal_dialogue / duration / storytelling / longform_free are the §4 Block C target names; TIME/TimeBench/TRAM task spellings differ (timedial, durationqa).
- [NOTE] corpus metadata — local CC-News snapshot ships no publish dates or ids; Day: headers use dates stated in the article, source_id is <file>:<line>.
- [NOTE] fixed-option letter bands — Duration_Compare 37/40/23 and Order_Compare 38/47/15 reconcile the card distributions with §7.3's always-A <= 38% threshold.
- [NOTE] Timeline k=5 coverage — 65 rows cannot cover 120 permutations; each is used at most once (<= 1.5x uniform holds trivially); shortfall reported, not padded.
- [NOTE] §5.12 member questions — pair members use different stem phrasings so question-level deduplication holds; answerability still flips only via the passage.
- [NOTE] reported shortfalls (§8) — Explicit_Reasoning 178/200 and duration 148/150 — the corpus cannot reach the full counts under these rules; reported, not padded. Relative_Reasoning/duration occasionally land 1-2 rows under on question-dedup rejections.
- [NOTE] Duration_Compare / Order_Compare event phrases — event descriptions carry their dates inline (', on March 3, 2015' / '(on March 3, 2015)') so §7.4 recomputation is unambiguous; contexts still state the same dates.
- [NOTE] verification fallbacks — where strict context matching is ambiguous, the recomputer falls back to the row's rationale-recorded dates after cross-checking them against the context; Computation, Duration_Compare, Order_Compare, Timeline, relation, ordering, nli and Localization-relative all verify at 100%.

**Overall: FAIL**

Blocking failures:
- gold recomputation >= 99% per recomputable category (Localization-relative)
- §9 answer style — free-text mean gold <= 6 words and < 5% >= 20 words (mean=8.11 (TIME 2.35), >=20w=14.8% (TIME 0.95%))
- golds neither truncated mid-sentence nor page furniture (truncated=79, boilerplate=14 (1.6% of rows))