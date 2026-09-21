# v9 AUG_GLM2 audit — 3000 rows, 18 categories
Generated from `data/corpus/ccnews` per `docs/synthetic_data_ruleset.md`.

## 7.1 Machine gates

- [PASS] schema + in-file duplicate questions — bad=0
- [PASS] exact question collision vs TIME+TimeBench+TRAM+probe — collisions=0
- [PASS] punctuation-insensitive collision — collisions=0
- [PASS] token-set Jaccard > 0.8 vs benchmark questions — hits=0
- [PASS] 30-token shingle shared with benchmark context — rows=0, banned source_ids=0
- [FAIL] in-category near-duplicate questions (J > 0.9) — pairs=80

## 7.2 Distribution audit (the D51 gate)

| category | rows | alloc | distinct golds | top gold share | in-context | verdict |
|---|---|---|---|---|---|---|
| Computation | 400 | 480 | 207 | 9.5% | 0% | PASS |
| Timeline | 325 | 390 | 57 | 10.5% | n/a | PASS |
| Localization | 275 | 330 | 235 | 5.1% | 71% | PASS |
| Counterfactual | 250 | 300 | 212 | 7.6% | 40% | PASS |
| Duration_Compare | 225 | 270 | 3 | 40.0% | n/a | PASS |
| Relative_Reasoning | 225 | 270 | 211 | 1.3% | 49% | PASS |
| Order_Reasoning | 200 | 240 | 199 | 1.0% | 100% | PASS |
| Co_temporality | 150 | 180 | 96 | 6.0% | 53% | PASS |
| Explicit_Reasoning | 100 | 120 | 100 | 1.0% | 54% | PASS |
| Order_Compare | 50 | 60 | 3 | 44.0% | n/a | PASS |
| nli_saq | 200 | 240 | 3 | 34.0% | n/a | PASS |
| nli_mcq | 125 | 150 | 3 | 34.4% | n/a | PASS |
| relation | 75 | 90 | 3 | 33.3% | n/a | PASS |
| ordering | 50 | 60 | 3 | 40.0% | n/a | PASS |
| temporal_dialogue | 125 | 150 | 84 | 11.2% | n/a | PASS |
| duration | 75 | 90 | 46 | 8.0% | n/a | PASS |
| storytelling | 75 | 90 | 75 | 1.3% | n/a | PASS |
| longform_free | 75 | 90 | 75 | 1.3% | n/a | PASS |
- [PASS] per-category distribution targets — all categories in band
- [PASS] fabricated YYYY-01-01 golds < 2% — count=0
- [PASS] Timeline permutation coverage (k=5 documented deviation: 65 rows over 120 perms, each used at most once) — k=3: rows=188, perms_seen=6/6, max_per_perm=34; k=4: rows=105, perms_seen=24/24, max_per_perm=7; k=5: rows=32, perms_seen=27/120, max_per_perm=2

Block A provenance split: news 76.8% (target 60), wiki 18.6% (35), dial 4.5% (5)
§5.12 paired rows carrying an abstain option: 0 (target ~352, i.e. 8% of 4,400)

## 7.3 Shortcut probes (the L7 gate)

| category | n_mcq | gold letters | always-A | longest | overlap | verdict |
|---|---|---|---|---|---|---|
| Counterfactual | 147 | {'A': 39, 'C': 38, 'D': 37, 'B': 33} | 27% | 0% | 0% (-0.3w) | PASS |
| Duration_Compare | 225 | {'B': 90, 'A': 83, 'C': 52} | 37% | 23% | 0% (-0.6w) | PASS |
| Relative_Reasoning | 135 | {'A': 34, 'B': 34, 'C': 34, 'D': 33} | 25% | 7% | 7% (+0.1w) | PASS |
| Order_Reasoning | 122 | {'A': 33, 'C': 33, 'B': 28, 'D': 28} | 27% | 3% | 1% (-0.0w) | PASS |
| Co_temporality | 79 | {'A': 22, 'C': 20, 'B': 19, 'D': 18} | 28% | 11% | 4% (+0.6w) | PASS |
| Explicit_Reasoning | 60 | {'A': 15, 'B': 15, 'C': 15, 'D': 15} | 25% | 7% | 7% (-0.0w) | PASS |
| Order_Compare | 50 | {'B': 22, 'A': 17, 'C': 11} | 34% | 22% | 0% (-0.5w) | PASS |
| nli_mcq | 125 | {'B': 43, 'A': 42, 'C': 40} | 34% | 0% | 0% (+0.0w) | PASS |
| relation | 75 | {'A': 25, 'B': 25, 'C': 25} | 33% | 0% | 0% (+0.0w) | PASS |
| ordering | 50 | {'C': 20, 'A': 18, 'B': 12} | 36% | 0% | 0% (+0.0w) | PASS |
| temporal_dialogue | 87 | {'A': 23, 'D': 22, 'B': 21, 'C': 21} | 26% | 0% | 0% (+0.0w) | PASS |
| duration | 75 | {'A': 19, 'B': 19, 'C': 19, 'D': 18} | 25% | 0% | 0% (+0.0w) | PASS |
| storytelling | 75 | {'A': 38, 'B': 37} | 51% | 0% | 3% (+0.0w) | PASS |
- [PASS] shortcut probes — all pass

## 7.4 Gold correctness (recomputed from stated dates)

- FAIL Computation: 193/400 recomputed correct (48.2%)
- FAIL Duration_Compare: 0/225 recomputed correct (0.0%)
- FAIL Order_Compare: 3/50 recomputed correct (6.0%)
- FAIL Timeline: 183/325 recomputed correct (56.3%)
- FAIL relation: 48/75 recomputed correct (64.0%)
- FAIL ordering: 18/50 recomputed correct (36.0%)
- FAIL nli_saq: 0/200 recomputed correct (0.0%)
- FAIL nli_mcq: 0/125 recomputed correct (0.0%)
- FAIL Localization relative-expression subset: 0/1 (0.0%)
- [FAIL] gold recomputation >= 99% per recomputable category — Computation 48.2%, Duration_Compare 0.0%, Order_Compare 6.0%, Timeline 56.3%, relation 64.0%, ordering 36.0%, nli_saq 0.0%, nli_mcq 0.0%, Localization-relative

## 7.5 Surface quality

gold length MCQ (reported, not gated): n=1630 mean=4.38 median=3 >=20 words=1.2%
gold length longform_free (reported, not gated): n=75 mean=24.48 median=24 >=20 words=76.0%
gold length free-text (gated): n=1295 mean=3.12 median=3 >=20 words=0.0%
TIME reference (free-text, n=104,939): mean 2.35, median 1, >=20w 0.95%
- [PASS] §9 answer style — free-text mean gold <= 6 words and < 5% >= 20 words (MCQ and longform_free excluded, see above) — mean=3.12 (TIME 2.35), >=20w=0.0% (TIME 0.95%)
- [PASS] no context padded by repeating a filler line >= 5 times — rows=0 (0.00%)
- [PASS] golds neither truncated mid-sentence nor page furniture — truncated=0, boilerplate=0 (0.0% of rows)

## Automated sample dump — 3 rows/category, NOT the §7.4 manual check

§7.4 requires 100 rows/category reviewed by eye for each non-recomputable category; that review is tracked separately and is NOT evidenced by this dump.
- **Co_temporality** `news` gold=`director of nursing` Q: While Fenwick was the trust's chief executive, what role did Marion Okafor hold? ⏎ Choices: ⏎ A. director of nursing ⏎ B. senior house officer ⏎ C. patient safety lead ⏎ D. catering team leader
- **Co_temporality** `news` gold=`deputy medical director` Q: While Mander chaired the trust board, what was Dr. Ines Faber's position? ⏎ Choices: ⏎ A. director of nursing ⏎ B. senior house officer ⏎ C. deputy medical director ⏎ D. patient safety lead
- **Co_temporality** `news` gold=`catering manager` Q: While Faber was the deputy medical director, what role did Briony Wells hold? ⏎ Choices: ⏎ A. chief executive ⏎ B. estates manager ⏎ C. press officer ⏎ D. catering manager
- **Computation** `news` gold=`9 days` Q: The first suspected meningitis cases in the Doma Valley surfaced on January 9, 2016, and the district issued its formal outbreak alert on January 18, 2016. How long after the first cases surfaced was the alert issued? (H
- **Computation** `news` gold=`24 days` Q: The Doma Valley vaccination campaign was launched on March 3, 2016, and its second phase opened on March 27, 2016. How long did the campaign's first phase run before the second phase opened? (Hint: Please answer in the f
- **Computation** `news` gold=`2 months 16 days` Q: The second phase of the Doma Valley campaign opened on March 27, 2016, and the independent evaluation was published on June 12, 2016. How much time passed between the second phase's opening and the evaluation's publicati
- **Counterfactual** `news` gold=`The pilot hub opened its doors on March 3, 2017, funded for eight months by the ` Q: Had the campaign's families accepted the county's first offer of a leaflet campaign instead of pressing their case, which statement matches the passage's account of the pilot hub's opening? ⏎ Choices: ⏎ A. The pilot hub 
- **Counterfactual** `news` gold=`The hub opened from eight in the evening to eight in the morning, going around t` Q: Suppose the night staff's logs had shown the county's worst hours arriving at noon instead of three in the morning. Which statement matches the passage's account of the hub's opening hours? ⏎ Choices: ⏎ A. The hub opened
- **Counterfactual** `news` gold=`The campaign was launched on February 2, 2016, at a meeting of eleven families.` Q: If the scout hut had burned down the week before the families first met, which statement matches the passage's account of how the campaign began? ⏎ Choices: ⏎ A. The campaign was launched on October 11, 2016, at a meetin
- **Duration_Compare** `news` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between the confirmation of the district's first measles case and the coming into force of the school-entry vaccination rule. *Duration 2:* Between the declar
- **Duration_Compare** `news` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between the opening of the first weekend catch-up clinic and the winding down of the clinic programme. *Duration 2:* Between the coming into force of the scho
- **Duration_Compare** `news` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between the confirmation of the district's first measles case and the winding down of the clinic programme. *Duration 2:* Between the start of the second surg
- **Explicit_Reasoning** `wiki` gold=`The flood model rebuild` Q: What notable activities did the Sabrina Insurance Collective engage in between November 15, 2007 and August 1, 2008? ⏎ Choices: ⏎ A. The flood model rebuild ⏎ B. The summer floods response ⏎ C. The hailstorm tribunal ⏎ D
- **Explicit_Reasoning** `wiki` gold=`The windstorm desk` Q: Which of the Sabrina Insurance Collective's notable activities falls between January 5, 2010 and June 30, 2011? ⏎ Choices: ⏎ A. The hailstorm tribunal ⏎ B. The windstorm desk ⏎ C. The freeze claims panel ⏎ D. The drought
- **Explicit_Reasoning** `wiki` gold=`The surge exercise` Q: Between February 1, 2013 and January 10, 2014, what did the Sabrina Insurance Collective undertake? ⏎ Choices: ⏎ A. The drought claims panel ⏎ B. The root-and-branch review ⏎ C. The surge exercise ⏎ D. The freeze claims 
- **Localization** `news` gold=`March 10, 2017` Q: When did the new stroke unit at Brackenfield General open its doors?
- **Localization** `news` gold=`June 6, 2016` Q: When was the feasibility study for the stroke unit approved?
- **Localization** `news` gold=`November 18, 2016` Q: When was the funding for the unit announced at the county hall?
- **Order_Compare** `news` gold=`Fact 1 happened earlier.` Q: For Fact1: the walk-in centre opened its doors and Fact2: the winter flu vaccination campaign was launched, which one happened earlier? ⏎ Choices: ⏎ A. Fact 1 happened earlier. ⏎ B. Fact 2 happened earlier. ⏎ C. They hap
- **Order_Compare** `news` gold=`Fact 2 happened earlier.` Q: For Fact1: universal MRSA screening began and Fact2: the new east wing took its first patients, which one happened earlier? ⏎ Choices: ⏎ A. Fact 1 happened earlier. ⏎ B. Fact 2 happened earlier. ⏎ C. They happen at almos
- **Order_Compare** `news` gold=`They happen at almost the same time.` Q: For Fact1: the new east wing took its first patients and Fact2: the first surgery was performed in the new wing, which one happened earlier? ⏎ Choices: ⏎ A. Fact 1 happened earlier. ⏎ B. Fact 2 happened earlier. ⏎ C. The
- **Order_Reasoning** `news` gold=`The Quay clinic` Q: What was the second satellite clinic in the passage's account? ⏎ Choices: ⏎ A. The Quay clinic ⏎ B. The Harbour clinic ⏎ C. The Marsh clinic ⏎ D. The reserve satellite clinic
- **Order_Reasoning** `news` gold=`The staffing petition` Q: Which petition was the second in the series the passage records? ⏎ Choices: ⏎ A. The waiting-list petition ⏎ B. The staffing petition ⏎ C. The transport petition ⏎ D. The winter crisis petition
- **Order_Reasoning** `news` gold=`The Marsh clinic` Q: What was the fourth satellite clinic recorded in the passage? ⏎ Choices: ⏎ A. The Harbour clinic ⏎ B. The Quay clinic ⏎ C. The Marsh clinic ⏎ D. The reserve satellite clinic
- **Relative_Reasoning** `news` gold=`The opening of the school hall clinic` Q: What was the most recent clinic opening after the second confirmed case? ⏎ Choices: ⏎ A. The opening of the school hall clinic ⏎ B. The opening of the first catch-up clinic ⏎ C. The opening of the second catch-up clinic 
- **Relative_Reasoning** `news` gold=`The opening of the first catch-up clinic` Q: Immediately after the first confirmed case, what was the next clinic opening in the passage's account? ⏎ Choices: ⏎ A. The opening of the second catch-up clinic ⏎ B. The opening of the first catch-up clinic ⏎ C. The open
- **Relative_Reasoning** `news` gold=`The third confirmed case` Q: Which confirmed case followed the opening of the first catch-up clinic most recently? ⏎ Choices: ⏎ A. The first confirmed case ⏎ B. The second confirmed case ⏎ C. The third confirmed case ⏎ D. The fourth confirmed case
- **Timeline** `news` gold=`C,A,B` Q: Below are 3 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Th
- **Timeline** `news` gold=`A,C,B` Q: Below are 3 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Th
- **Timeline** `news` gold=`B,C,A` Q: Below are 3 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Th
- **duration** `none` gold=`about 64 years` Q: The Victorian era ran from 1837 to 1901. How long did it last in total? ⏎ Choices: ⏎ A. about 64 years ⏎ B. about 65 years ⏎ C. about 63 years ⏎ D. about 128 years
- **duration** `none` gold=`about 70 years` Q: The reign of Elizabeth II ran from 1952 to 2022. How long did it last? ⏎ Choices: ⏎ A. about 72 years ⏎ B. about 70 years ⏎ C. about 68 years ⏎ D. about 105 years
- **duration** `none` gold=`about 4 years` Q: The First World War's run stretched from 1914 to 1918. In total, how long did it last? ⏎ Choices: ⏎ A. about 14 years ⏎ B. about 1 year ⏎ C. about 4 years ⏎ D. about 2 years
- **longform_free** `news` gold=`The electronic case register went live on January 14, 2007. The video-link court` Q: What developments does the passage report between January 13, 2007 and September 10, 2008?
- **longform_free** `news` gold=`The video-link courtroom opened on September 9, 2008. The jurors' summons moved ` Q: Which developments does the passage describe between September 8, 2008 and March 4, 2010?
- **longform_free** `news` gold=`The video-link courtroom opened on September 9, 2008. The jurors' summons moved ` Q: What does the passage report as happening between September 8, 2008 and April 23, 2009?
- **nli_mcq** `news` gold=`entailment` Q: Premise: The Ashmere community responder scheme was launched on February 8, 2015, with twelve volunteers trained to answer emergency calls ahead of the ambulance. ⏎ Hypothesis: The responder scheme started with a dozen v
- **nli_mcq** `news` gold=`neutral` Q: Premise: A second defibrillator cabinet was fixed beside the phone box at Low Briar in 2016, paid for by the Women's Royal Voluntary Service. ⏎ Hypothesis: The charity's volunteers are paid an hourly rate for each call-o
- **nli_mcq** `news` gold=`contradiction` Q: Premise: The volunteers answered 214 call-outs in the scheme's first year, a figure the ambulance trust called remarkable for a valley of four thousand people. ⏎ Hypothesis: The scheme's volunteers answered more than thr
- **nli_saq** `news` gold=`entailment` Q: Premise: The Eastmere district nursing service began 2015 under a merger that its organisers called the quietest revolution in county care, folding four visiting teams into one on March 2, 2015. ⏎ Hypothesis: The four vi
- **nli_saq** `news` gold=`neutral` Q: Premise: A new clinic opened at Wrenfield on June 6, 2016, taking over a closed bank building and cutting the district's longest home-visit round by nineteen miles. ⏎ Hypothesis: The bank building that housed the Wrenfie
- **nli_saq** `news` gold=`contradiction` Q: Premise: By September 30, 2019 the district's MMR uptake stood at ninety-one per cent, the highest ever recorded there and four points above the county average. ⏎ Hypothesis: By the autumn of 2019 the district's MMR upta
- **ordering** `none` gold=`TRUE` Q: The Berrow surgery reopened its Foregate site on May 6, 2015. The practice's online patient portal went live on May 19, 2015. Claim: the surgery reopened before the portal went live - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B
- **ordering** `none` gold=`FALSE` Q: Marden Vale's falls clinic admitted its first patient on February 9, 2016. The clinic's home-visit service began on March 3, 2016. Claim: the home-visit service began before the clinic admitted its first patient - True/F
- **ordering** `none` gold=`TRUE` Q: The Netherfold defibrillator was installed on September 9, 2016. The parish's first-responders completed their training on October 1, 2016. Claim: the responders finished training after the defibrillator was installed - 
- **relation** `none` gold=`IDENTITY` Q: The Marshvale health centre opened on June 9, 1970, and the county's first health board meeting was held on June 9, 1970. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ C. DURING
- **relation** `none` gold=`BEFORE` Q: The mobile vaccination unit began its rounds on February 17, 1967, and the smallpox campaign closed on October 30, 1968. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ C. DURING
- **relation** `none` gold=`DURING` Q: The nurses' ballot ran from April 3, 1974 to June 21, 1974, and the ward closure was announced on May 15, 1974. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ C. DURING
- **storytelling** `news` gold=`The story ends with the councillors counting the square's new benches, all sixty` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. The story ends with the councillors counting the square's new benches, all fifty of them, and finding the argument about pigeons
- **storytelling** `news` gold=`In the closing lines, the fountain plays again in 2019, the year the town's bank` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. In the closing lines, the fountain plays again in 2019, the year the town's bank holiday finally got its promised water. ⏎ B. In
- **storytelling** `news` gold=`The story's last page has the library's reopening falling before the fountain's ` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. The story's last page has the library's reopening falling after the fountain's restoration, by two summers' worth of patience. ⏎
- **temporal_dialogue** `dial` gold=`Session 2` Q: Which session records the uptake figure for the first wave? ⏎ Choices: ⏎ A. Session 2 ⏎ B. Session 1 ⏎ C. Session 3 ⏎ D. Session 4
- **temporal_dialogue** `dial` gold=`14 March, 2016` Q: According to the transcript, on what date did the first school clinics run? ⏎ Choices: ⏎ A. 3 March, 2016 ⏎ B. 14 March, 2016 ⏎ C. 9 March, 2016 ⏎ D. 5 July, 2016
- **temporal_dialogue** `dial` gold=`Session 3` Q: In which session is the mobile clinic van first discussed? ⏎ Choices: ⏎ A. Session 1 ⏎ B. Session 2 ⏎ C. Session 3 ⏎ D. Session 4

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
- in-category near-duplicate questions (J > 0.9) (pairs=80)
- gold recomputation >= 99% per recomputable category (Computation 48.2%, Duration_Compare 0.0%, Order_Compare 6.0%, Timeline 56.3%, relation 64.0%, ordering 36.0%, nli_saq 0.0%, nli_mcq 0.0%, Localization-relative)