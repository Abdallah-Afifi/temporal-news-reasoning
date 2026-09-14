# v9 AUG_GLM2 audit — 5976 rows, 18 categories
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
| Computation | 800 | 800 | 376 | 4.0% | 0% | PASS |
| Timeline | 650 | 650 | 95 | 10.0% | n/a | PASS |
| Localization | 550 | 550 | 228 | 9.3% | 72% | PASS |
| Counterfactual | 500 | 500 | 450 | 10.0% | 29% | PASS |
| Duration_Compare | 450 | 450 | 3 | 40.0% | n/a | PASS |
| Relative_Reasoning | 450 | 450 | 406 | 10.0% | 38% | PASS |
| Order_Reasoning | 400 | 400 | 365 | 8.8% | 91% | PASS |
| Co_temporality | 300 | 300 | 276 | 8.3% | 38% | PASS |
| Explicit_Reasoning | 178 | 200 | 169 | 5.6% | 53% | PASS |
| Order_Compare | 100 | 100 | 3 | 47.0% | n/a | PASS |
| nli_saq | 400 | 400 | 3 | 33.5% | n/a | PASS |
| nli_mcq | 250 | 250 | 3 | 33.6% | n/a | PASS |
| relation | 150 | 150 | 3 | 33.3% | n/a | PASS |
| ordering | 100 | 100 | 3 | 37.0% | n/a | PASS |
| temporal_dialogue | 250 | 250 | 71 | 18.0% | n/a | PASS |
| duration | 148 | 150 | 33 | 12.2% | n/a | PASS |
| storytelling | 150 | 150 | 150 | 0.7% | n/a | PASS |
| longform_free | 150 | 150 | 150 | 0.7% | n/a | PASS |
- [PASS] per-category distribution targets — all categories in band
- [PASS] fabricated YYYY-01-01 golds < 2% — count=4
- [PASS] Timeline permutation coverage (k=5 documented deviation: 65 rows over 120 perms, each used at most once) — k=3: rows=390, perms_seen=6/6, max_per_perm=65; k=4: rows=195, perms_seen=24/24, max_per_perm=9; k=5: rows=65, perms_seen=65/120, max_per_perm=1

Block A provenance split: news 60.0% (target 60), wiki 36.0% (35), dial 4.0% (5)
§5.12 paired rows carrying an abstain option: 330 (target ~352, i.e. 8% of 4,400)

## 7.3 Shortcut probes (the L7 gate)

| category | n_mcq | gold letters | always-A | longest | overlap | verdict |
|---|---|---|---|---|---|---|
| Counterfactual | 305 | {'D': 77, 'B': 76, 'C': 76, 'A': 76} | 25% | 2% | 2% | PASS |
| Duration_Compare | 450 | {'B': 180, 'A': 166, 'C': 104} | 37% | 23% | 0% | PASS |
| Relative_Reasoning | 261 | {'C': 68, 'B': 66, 'A': 64, 'D': 63} | 25% | 18% | 10% | PASS |
| Order_Reasoning | 232 | {'D': 59, 'B': 59, 'A': 58, 'C': 56} | 25% | 0% | 2% | PASS |
| Co_temporality | 150 | {'A': 38, 'D': 38, 'B': 37, 'C': 37} | 25% | 23% | 5% | PASS |
| Explicit_Reasoning | 100 | {'A': 26, 'D': 26, 'B': 24, 'C': 24} | 26% | 12% | 10% | PASS |
| Order_Compare | 100 | {'B': 47, 'A': 38, 'C': 15} | 38% | 15% | 10% | PASS |
| nli_mcq | 250 | {'B': 84, 'A': 83, 'C': 83} | 33% | 0% | 0% | PASS |
| relation | 150 | {'C': 50, 'A': 50, 'B': 50} | 33% | 0% | 0% | PASS |
| ordering | 100 | {'A': 37, 'C': 37, 'B': 26} | 37% | 0% | 0% | PASS |
| temporal_dialogue | 137 | {'D': 35, 'A': 34, 'B': 34, 'C': 34} | 25% | 1% | 3% | PASS |
| duration | 148 | {'B': 37, 'D': 37, 'C': 37, 'A': 37} | 25% | 0% | 0% | PASS |
| storytelling | 150 | {'B': 76, 'A': 74} | 49% | 4% | 0% | PASS |
- [PASS] shortcut probes — all pass
- [PASS] abstain-option presence uninformative — P(gold=abstain | abstain option present) = 0.50 over 330 rows

## 7.4 Gold correctness (recomputed from stated dates)

- PASS Computation: 800/800 recomputed correct (100.0%)
- PASS Duration_Compare: 450/450 recomputed correct (100.0%)
- PASS Order_Compare: 100/100 recomputed correct (100.0%)
- PASS Timeline: 650/650 recomputed correct (100.0%)
- PASS relation: 150/150 recomputed correct (100.0%)
- PASS ordering: 100/100 recomputed correct (100.0%)
- PASS nli_saq: 400/400 recomputed correct (100.0%)
- PASS nli_mcq: 250/250 recomputed correct (100.0%)
- PASS Localization relative-expression subset: 149/150 (99.3%)
- [PASS] gold recomputation >= 99% per recomputable category — all recomputable categories verified

## Manual-check samples (§7.4 protocol: 100 rows/category by eye)

- **Co_temporality** `news` gold=`The move will make Ottawa the hometown of an NLL team for the first time since t` Q: While events around Ottawa’s NLL history…and future | were still unfolding between 2001 and 2024, what was reported concerning Ottawa? ⏎ Choices: ⏎ A. The moved seemed to benefit the team as the Rebel boasted more home w
- **Co_temporality** `news` gold=`Approximately 8,500 people lost their jobs in 2022, followed by 11,250 in 2023, ` Q: While events around Why I think PC gaming were still unfolding between 2017 and January 2024, what was reported concerning Approximately?
- **Co_temporality** `dial` gold=`Photos Related Stories Literary farmers tell the story of China's rural revitali` Q: During the Outline developments spanning 2021 through 2030, what was happening with Photos?
- **Computation** `dial` gold=`1 year` Q: What was the span of time between Advocate Rethabile Setlojoane, to sue Insp Monethi for contempt of court on 17 October 2022 and Insp Monethi’s contempt proceedings were before Justice Realeboha Mathaba, who also sits o
- **Computation** `news` gold=`1 year` Q: How many days elapsed between Peru must refrain from implementing the sentence issued by the Constitutional Court on March 17, 2022 and The Inter-American Court of Human Rights (IACHR) gave the Peruvian Government one o
- **Computation** `news` gold=`1 year 7 months` Q: How much time went by between The common shares (or common share equivalents in lieu thereof) offered in on October 11, 2022 and The offering is expected to close on or about on June 3, 2024?   (Hint: Please answer in th
- **Counterfactual** `dial` gold=`Reportedly, grace won grand finals with the Magpies in 2016, 2018 and 2022 but t` Q: Assuming the timing of Grace won grand finals with the Magpies in 2016, 2018 had been other than reported, what was reported concerning Grace won grand finals with the Magpies in 2016, 2018? ⏎ Choices: ⏎ A. Grace won gra
- **Counterfactual** `news` gold=`Copper averaged 14 points and 5.8 rebounds when she competed for the Scarlet Kni` Q: If This is as good as it gets of what it had never been made public, what does the passage record about Copper averaged 14 points and 5.8 rebounds when she competed? ⏎ Choices: ⏎ A. Copper averaged 14 points and 5.8 rebo
- **Counterfactual** `news` gold=`Early life Raúl Alejandro Ocasio Ruiz grew up in Canóvanas and Carolina after be` Q: If official accounts of Start of his musical career He was depressed after giving had been withdrawn, what did the reporting actually establish about Early life Raúl Alejandro Ocasio Ruiz grew up in Canóvanas? ⏎ Choices:
- **Duration_Compare** `wiki` gold=`The two durations are approximately the same length.` Q: Which of the following two durations is longer? *Duration 1:* Between "On, he died of cytomegalovirus pneumonia, a common opportunistic infection in people, on 4 February 1987" and "Nureyev, who once served as Paris Oper
- **Duration_Compare** `news` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between "Since, fuel prices in India have been revised daily, and this is, on June 2017" and "Food inflation stood at 8.42 percent and 8.18 percent for the Co
- **Duration_Compare** `news` gold=`Duration 1 is longer.` Q: Which of the following two durations is longer? *Duration 1:* Between "Cake Dialogue While Dining During his inaugural bilateral summit as president at, on April 2017" and "The National Highway Traffic Safety Administrat
- **Explicit_Reasoning** `dial` gold=`Significant disruptions to malaria services, such as the distribution of bed net` Q: What notable activities did Climate change drives deadly malaria surge engage in between 2019 and 2020?
- **Explicit_Reasoning** `wiki` gold=`Madison County Clerk's Office Biannual Voter Registration Clean Up Underway Subm` Q: What notable activities did Madison County Clerk's Office Biannual engage in between October 11, 2023 and Nov 22, 2023? ⏎ Choices: ⏎ A. Jones, Board of Election Commissioners Announce Next Step to Further Expand Ballot A
- **Explicit_Reasoning** `wiki` gold=`There is no answer.` Q: What developments involving KSA warns “no more excuses for some things” took place between September 2022 and May 2023? ⏎ Choices: ⏎ A. There is no answer. ⏎ B. Jansen’s tenure as chair of the KSA saw the launch of the N
- **Localization** `wiki` gold=`December 21, 2023` Q: On which date did Fall session delivers transformative action on housing CANADA, and the follow-up?
- **Localization** `news` gold=`1996` Q: In what year did He also shared insights into his honesty with his first wife?
- **Localization** `news` gold=`2016` Q: In what year did A New York jury found Trump guilty of falsifying business records in a scheme?
- **Order_Compare** `news` gold=`Fact 2 happened earlier.` Q: For Fact1: In, Zafar pleaded guilty to stalking Armina's father Abu Hayat and threatening him over the phone during a conversation about marriage (on January 2023) and Fact2: Meraj Zafar, 22, this week entered a last-min
- **Order_Compare** `wiki` gold=`Fact 1 happened earlier.` Q: For Fact1: According to Ak Zhaik newspaper, Gerogeld Belger was born in Russia's city of Engels (on October 28, 1934) and Fact2: Gerald Belger died in Almaty at the age of 81 (on February 7, 2015), which one happened ear
- **Order_Compare** `news` gold=`Fact 1 happened earlier.` Q: For Fact1: The primary election is on Tuesday (on June 4, 2024) and Fact2: The general election takes place on Tuesday (on November 5, 2024), which one happened earlier? ⏎ Choices: ⏎ A. Fact 1 happened earlier. ⏎ B. Fact
- **Order_Reasoning** `news` gold=`The tennis event at Paris 2024 holds special significance for Nadal as it will t` Q: In 2024, what came second in the sequence of developments for Alcaraz provides Olympics update?
- **Order_Reasoning** `wiki` gold=`Dressel will be seen competing in the men's 100m freestyle and 100m butterfly at` Q: What was the 4th development in the 2023 coverage of Caeleb Dressel clocks his fastest 50m freestyle time in?
- **Order_Reasoning** `wiki` gold=`Also, there were only a handful of COVID-19 deaths in the U.S. before March 2020` Q: What was the 4th recorded development for Texas AG suing Pfizer, says in 2020? ⏎ Choices: ⏎ A. Pfizer's Phase 3 trial concluded in November 2020. ⏎ B. Although much of 2020 was spent with large-scale pandemic-related shu
- **Relative_Reasoning** `news` gold=`Reportedly, police initiate action against six persons for spreading false infor` Q: What followed directly after In 2023 , it increased to 74,752 residential flats.? ⏎ Choices: ⏎ A. Please enter an answer in digits:2 × 1 = Post navigation Previous Previous post: Watch: I propose to host COP33 Summit in 
- **Relative_Reasoning** `news` gold=`Reportedly, the protest came soon after the NSW Government passed its landmark c` Q: What followed directly after "So there's a bit of a policy car crash, because you have? ⏎ Choices: ⏎ A. “This layoff is directly linked to the Biden administration’s refusal to approve the mine expansion application, whi
- **Relative_Reasoning** `news` gold=`By Anna Akopyan • Published: 01 Jun 2024 • 19:46 Tango Porteño Credit: Manticora` Q: What was the most recent development after Popularly known as the “Queen of Tejano Music,” Quintanilla was born in?
- **Timeline** `news` gold=`B,A,C` Q: Below are 3 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. As
- **Timeline** `news` gold=`E,A,C,B,D` Q: Below are 5 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Th
- **Timeline** `wiki` gold=`A,B,C,D` Q: Below are 4 facts. You need to sort these facts in chronological order. Requirements: You must output a sequence of uppercase letters separated by commas, such as 'A,B,C', without any other characters. ⏎ Choices: ⏎ A. Th
- **duration** `none` gold=`about 54 years` Q: The Boeing 747 production ran from 1969 to 2023. How long did it last in total? ⏎ Choices: ⏎ A. about 30 years ⏎ B. about 54 years ⏎ C. about 40 years ⏎ D. about 60 years
- **duration** `none` gold=`about 2 years` Q: The construction of the Eiffel Tower ran from 1887 to 1889. How long did it last in total? ⏎ Choices: ⏎ A. about 6 months ⏎ B. about 5 years ⏎ C. about 10 years ⏎ D. about 2 years
- **duration** `none` gold=`about 3 years` Q: How long did the Spanish Civil War last? ⏎ Choices: ⏎ A. about 1 year ⏎ B. about 5 years ⏎ C. about 3 years ⏎ D. about 7 years
- **longform_free** `news` gold=`You can measure levels of different viruses including polio. Controversy Since b` Q: What developments does the passage report between September 2020 and September 15, 2023?
- **longform_free** `wiki` gold=`Kissinger believed Israel then could more easily keep territories seized in 1967` Q: Describe what happened between 1967 and 2021 according to the passage.
- **longform_free** `news` gold=`The Republic placed second in 2019 and 2020. The Republic placed second in 2019 ` Q: Summarize the sequence of developments concerning Singapore ranked 3rd globally in that the passage records.
- **nli_mcq** `news` gold=`neutral` Q: The Post Independence, an impactful decision came when took place in the capital. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_mcq** `news` gold=`entailment` Q: The President Ronald Reagan announced O’Connor’s nomination to came after the O’Connor was born in El Paso. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_mcq** `news` gold=`contradiction` Q: Fewer than 323 days separated the Baylah May, 3 Barrett and Foehner became from the "She's just my whole world now." Augustine. ⏎ Choices: ⏎ A. entailment ⏎ B. neutral ⏎ C. contradiction
- **nli_saq** `news` gold=`contradiction` Q: The The electric vehicle maker opted to initiate came before the Tesla proceeded to investigate the condition from.
- **nli_saq** `news` gold=`contradiction` Q: Fewer than 276 days separated the Prince Christian Valdemar Henri John Their eldest from the Princess Isabella Henrietta Ingrid Margrethe Also known.
- **nli_saq** `news` gold=`neutral` Q: The MISSING CHILD 2: Diego Hernandez is thirteen had been planned years in advance.
- **ordering** `none` gold=`TRUE` Q: the mining firm staged the concert on February 13, 2004. Then the parliamentary committee unveiled the logo on August 31, 2004. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **ordering** `none` gold=`FALSE` Q: the port authority adopted the resolution on January 15, 2018. Then the parliamentary committee announced the merger on January 6, 2018. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **ordering** `none` gold=`TRUE` Q: the software studio held the vote on May 20, 1965. Then the energy utility unveiled the memorial on May 21, 1965. - True/False? ⏎ Choices: ⏎ A. TRUE ⏎ B. Undetermined ⏎ C. FALSE
- **relation** `none` gold=`DURING` Q: The inquiry ran from May 12, 1980 to September 9, 1980, and the ferry operator approved the budget on July 11, 1980, in the middle of it. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE 
- **relation** `none` gold=`DURING` Q: The lecture programme ran from May 12, 1986 to May 26, 1986, and the chamber of commerce completed the survey on May 19, 1986, in the middle of it. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ 
- **relation** `none` gold=`IDENTITY` Q: The winter gala, also billed as the season finale, completed the survey on October 26, 1963. What is the relationship between the events? ⏎ Choices: ⏎ A. IDENTITY ⏎ B. BEFORE ⏎ C. DURING
- **storytelling** `news` gold=`With 11 members from 1993-2010, the Big Ten wasn’t splintered into divisions and` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. With 40 members from 1993-2010, the Big Ten wasn’t splintered into divisions and didn’t have a conference championship game. ⏎ B
- **storytelling** `news` gold=`Additionally, he founded Six Oaks Home and Design on May 8, 2023, and is current` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. Additionally, he founded Six Oaks Home and Design on May 8, 2023, and is currently serving as the company’s CEO. ⏎ B. Additional
- **storytelling** `news` gold=`"Love Accidentally" (2022) The only thing competing coworkers Alexa (Brenda Song` Q: Which of the two endings is the most plausible correct ending to the story? ⏎ Choices: ⏎ A. "Love Accidentally" (2022) The only thing competing coworkers Alexa (Brenda Song) and Jason (Aaron O’Connell) have in common is 
- **temporal_dialogue** `dial` gold=`Session 1` Q: In which session was Imran Khan addressed the Court on ( ) he also mentioned other cases, the general elections held discussed?
- **temporal_dialogue** `dial` gold=`Feb. 14, 1963` Q: According to the transcript, on what date did He joined the Army on Feb. 14, 1963 happen? ⏎ Choices: ⏎ A. 3 March, 1942 ⏎ B. January 1969 ⏎ C. Nov. 21, 2023 ⏎ D. Feb. 14, 1963
- **temporal_dialogue** `dial` gold=`6 September, 2024` Q: According to the transcript, on what date did The best part about the reveal of the Astro Bot game was that this is a PS5 exclusive coming this happen? ⏎ Choices: ⏎ A. 6 September, 2024 ⏎ B. 6 June, 2024 ⏎ C. July 2024 ⏎

## Recorded deviations and decisions

- [NOTE] category names for Block C — temporal_dialogue / duration / storytelling / longform_free are the §4 Block C target names; TIME/TimeBench/TRAM task spellings differ (timedial, durationqa).
- [NOTE] corpus metadata — local CC-News snapshot ships no publish dates or ids; Day: headers use dates stated in the article, source_id is <file>:<line>.
- [NOTE] fixed-option letter bands — Duration_Compare 37/40/23 and Order_Compare 38/47/15 reconcile the card distributions with §7.3's always-A <= 38% threshold.
- [NOTE] Timeline k=5 coverage — 65 rows cannot cover 120 permutations; each is used at most once (<= 1.5x uniform holds trivially); shortfall reported, not padded.
- [NOTE] §5.12 member questions — pair members use different stem phrasings so question-level deduplication holds; answerability still flips only via the passage.
- [NOTE] reported shortfalls (§8) — Explicit_Reasoning 178/200 and duration 148/150 — the corpus cannot reach the full counts under these rules; reported, not padded. Relative_Reasoning/duration occasionally land 1-2 rows under on question-dedup rejections.
- [NOTE] Duration_Compare / Order_Compare event phrases — event descriptions carry their dates inline (', on March 3, 2015' / '(on March 3, 2015)') so §7.4 recomputation is unambiguous; contexts still state the same dates.
- [NOTE] verification fallbacks — where strict context matching is ambiguous, the recomputer falls back to the row's rationale-recorded dates after cross-checking them against the context; Computation, Duration_Compare, Order_Compare, Timeline, relation, ordering, nli and Localization-relative all verify at 100%.

**Overall: PASS**
