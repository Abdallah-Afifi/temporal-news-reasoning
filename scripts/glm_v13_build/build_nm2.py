import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nli_m_lib import build

def N(pid, prem, hyp, label, rat, slot=0, guard=None):
    return dict(pid=pid, prem=prem, hyp=hyp, label=label, rat=rat, slot=slot, guard=guard)

# =================== 275: nli_mcq news, labour 2015-2019
D1 = ("[1] Title: Recognition Agreement Signed, Day: February 11, 2016 "
"Content: The recognition agreement was signed on February 8, 2016, the "
"warehouse's first, its single page giving the union its check-off and "
"its reps a room of their own, and the signing's pen, kept in the rep "
"room's drawer, was used again for every later term of the decade. "
"The union learning centre was opened on October 1, 2018, in the "
"warehouse's old canteen, its first term enrolling a third of the "
"workforce and its first certificate, the centre's own record notes, "
"awarded to the warehouse's longest-serving picker, forty years of "
"labels counted and now one of his own. The pay deal was struck on "
"June 19, 2017, the first the union had negotiated under the "
"agreement, its terms a real-terms rise for every grade and the "
"night shift's premium restored, the deal's own arithmetic, the "
"union's bulletin printed in full, the county's strongest of the "
"decade, the deal's ratification ballot returning nine to one on "
"a turnout the returning officer called the warehouse's own "
"referendum.")

D2 = ("[2] Title: The Strike Ballot, Day: March 8, 2018 "
"Content: The strike ballot was held on March 5, 2018, the "
"warehouse's first in a generation, its question the agency "
"workers' unequal holiday terms and its result, carried by four "
"to one on a turnout of eighty per cent, giving the union the "
"mandate the law required, and the mandate's first use, the "
"bulletin recorded, was not a picket line but a tea room, the "
"company's own directors invited to the learning centre to be "
"taught the warehouse's day by the pickers themselves, the "
"company's chair attending and the holiday terms equalised at "
"the following month's negotiating table without one hour "
"struck, the ballot's whole account, the union's historian "
"wrote, the county's plainest demonstration of a strong "
"mandate's quietest use. The tea room's own account, the union's bulletin printed in full beneath the ballot's figures, gives the day's whole curriculum: the directors issued each with a picker's handset and a picker's targets, the chair's own morning, the bulletin's reporter noted, spent at the returns bay where the holiday terms' unequal arithmetic had been counted, and the directors' joint letter to the workforce, printed beside the account, closed with the chair's single line of thanks, that the warehouse had taught its owners the warehouse's own day, the mandate's quietest use, the letter's last sentence says, the county's loudest lesson. The bulletin's own centenary of the decade, printed at the accord's signing and given to every member of the sector, reproduced the tea room's account, the ballot's figures and the accord's first clause on a single broadsheet, the decade's three papers in one sheet, the union's whole doctrine, the broadsheet's editor wrote, kept in the order the decade taught it: agree, measure, then talk.")

D3 = ("[3] Title: The Sector Accord, Day: April 26, 2019 "
"Content: The sector-wide accord was signed on April 23, 2019, the "
"region's warehouses whole under one set of terms the union and "
"the employers' body had drafted jointly, and the accord's first "
"clause, the union's own recognition agreement reproduced entire, "
"gave every warehouse in the region the room and the pen the "
"first warehouse had kept. The accord's own signing, held in the "
"learning centre with the certificate class of the year "
"attending, was photographed beside the canteen's hatch, the "
"region's whole warehouse sector, the bulletin's last page said, "
"organised from one rep room, one drawer and one pen, the "
"decade's whole labour account, the union's closing editorial "
"wrote, three dates, two ballots and one continuous conversation, "
"the accord's fair copy, framed beside the drawer, carrying the "
"first agreement's and the deal's and the ballot's signatures "
"beneath its own, four pages of the decade in one frame. The frame's own unveiling, held at the accord's first anniversary with the sector's whole workforce represented, was attended by the recognition agreement's original directors, the company's chair of the tea room year among them, and the chair's second letter, read at the unveiling and bound as the frame's fifth page, gave the decade's employers' account its own single line: that the conversation had been continuous, the letter says, because the union had made the table worth sitting at, the accord's whole sector, the bulletin's closing editorial wrote, one warehouse's practices, one room wide and one decade long.")

ROWS275 = [
 N("P1", "The recognition agreement was signed on February 8, 2016.",
   "The union won recognition at the warehouse in the winter of 2016.",
   "entailment",
   "February 8, 2016 falls in winter.",
   0, "February 8, 2016"),
 N("P1", "The learning centre opened in the warehouse's old canteen.",
   "The learning centre's first term enrolled most of the workforce.",
   "contradiction",
   "The passage says a third of the workforce, not most.",
   1, "October 1, 2018"),
 N("P1", "The learning centre's first certificate went to the longest-serving picker.",
   "The first certificate was awarded to a new employee.",
   "contradiction",
   "The passage says the longest-serving picker, forty years of labels.",
   2, "October 1, 2018"),
 N("P1", "The pay deal was struck on June 19, 2017, restoring the night premium.",
   "The night shift's premium had been removed before June 2017.",
   "entailment",
   "Restoring the premium implies it had been removed.",
   0, "June 19, 2017"),
 N("P1", "The deal's ratification ballot returned nine to one.",
   "The pay deal was ratified by a large majority.",
   "entailment",
   "Nine to one is a large majority.",
   1, "June 19, 2017"),
 N("P1", "The strike ballot was held on March 5, 2018, over agency workers' holiday terms.",
   "The strike ballot's issue concerned unequal holiday terms.",
   "entailment",
   "The ballot's question was the agency workers' unequal holiday terms.",
   2, "March 5, 2018"),
 N("P1", "The ballot carried by four to one on an eighty per cent turnout.",
   "A minority of the workforce voted for strike action.",
   "neutral",
   "Four to one of an eighty per cent turnout is a majority of those voting and most of the workforce; whether a minority of the whole workforce supported it cannot be fixed without exact figures.",
   0, "March 5, 2018"),
 N("P1", "The holiday terms were equalised without one hour struck.",
   "No strike took place at the warehouse in 2018.",
   "entailment",
   "The terms were settled at the negotiating table without strike hours.",
   1, "March 5, 2018"),
 N("P1", "The company's directors were taught the warehouse's day by the pickers.",
   "The union's first use of the mandate was industrial action.",
   "contradiction",
   "The passage says the first use was a tea room, not action.",
   2, "March 5, 2018"),
 N("P1", "The sector accord was signed on April 23, 2019.",
   "The accord was signed in the same season as the strike ballot.",
   "neutral",
   "The ballot was March 2018 and the accord April 2019, different years, so not the same season.",
   0, "April 23, 2019"),
 N("P1", "The accord's first clause reproduced the recognition agreement entire.",
   "Every warehouse in the region gained the union recognition terms.",
   "entailment",
   "Reproducing the agreement in the sector-wide accord extends it to every warehouse.",
   1, "April 23, 2019"),
 N("P1", "The accord's signing was held in the learning centre.",
   "The accord was signed in the building where the union taught its members.",
   "entailment",
   "The learning centre is the union's teaching venue.",
   2, "April 23, 2019"),
 N("P1", "The signing pen was kept in the rep room's drawer and reused.",
   "The decade's agreements were signed with the same pen.",
   "entailment",
   "The passage says the pen was used again for every later term.",
   0, "February 8, 2016"),
 N("P1", "The accord's fair copy carries four pages of signatures.",
   "The accord's fair copy was signed by four people.",
   "neutral",
   "Four pages of signatures do not state the number of signatories.",
   1, "April 23, 2019"),
 N("P1", "The pay deal was the county's strongest of the decade.",
   "Every other warehouse in the county got a stronger deal in 2017.",
   "contradiction",
   "The county's strongest deal cannot be exceeded by other deals in the same county.",
   2, "June 19, 2017"),
 N("P1", "The ballot's turnout was eighty per cent.",
   "Two thirds of the workforce voted in the ballot.",
   "contradiction",
   "Eighty per cent exceeds two thirds.",
   0, "March 5, 2018"),
 N("P1", "The learning centre opened on October 1, 2018.",
   "The learning centre opened before the strike ballot was held.",
   "entailment",
   "October 1, 2018... the strike ballot was March 5, 2018, which precedes the centre's opening.",
   1, "March 5, 2018"),
 N("P1", "The accord was drafted jointly by the union and the employers' body.",
   "The union wrote the accord alone and presented it to the employers.",
   "contradiction",
   "The passage says the terms were drafted jointly.",
   2, "April 23, 2019"),
 N("P1", "The union's historian wrote the ballot's account.",
   "The strike ballot's history was recorded before the accord was signed.",
   "neutral",
   "The passage does not date the historian's account.",
   0, "March 5, 2018"),
 N("P1", "The recognition agreement gave the union's reps a room.",
   "Before 2016 the union's representatives had no dedicated room at the warehouse.",
   "entailment",
   "The first agreement giving a room implies none existed before.",
   1, "February 8, 2016"),
]

build(275, "news", {"P1": (D1, D2, D3), "P2": (D1, D2, D3)}, ROWS275)

# =================== 276: nli_mcq news, space 1990-2004
E1 = ("[1] Title: Radio Array Commissioned, Day: September 15, 1993 "
"Content: The radio array was commissioned on September 12, 1993, the "
"county's first listening instrument, its four dishes strung along "
"the old railway cutting and its first observation, the institute's "
"log records, a whole night spent on the transit of the county's "
"own chosen pulsar, the target named by the sixth form that had "
"won the institute's first schools competition. The outreach "
"centre was opened on June 7, 1996, the array's control building "
"given a public wing, and the centre's first year, the institute's "
"annual report records, hosted every school in the county and "
"eleven from beyond it, the array's own data, the report notes, "
"shown live to the visitors on the control room's spare screens, "
"the county's radio sky, the centre's first visitor book said, "
"hung where the trains had run. The centre's own first exhibit, the array's first night's paper chart hung where the cutting's departures board had been, gave the visitors the county's radio sky in the county's own hand-ruled ink, and the chart's second life, the report notes, as the centre's most borrowed teaching sheet, made the sixth form's pulsar the county's best-known star, the schools competition's whole legacy, the institute's evaluator wrote, one star, one chart and one competition, the county's radio astronomy taught from its own first night. The chart's own second edition, ruled by the centre's first volunteer from the array's second year of nights, added the competition's second winner's target and the centre's own moon experiments, the county's radio curriculum, the evaluator's second report noted, growing one competition and one chart at a time.")

E2 = ("[2] Title: The Comet Launch, Day: March 6, 2001 "
"Content: The comet mission was launched on March 3, 2001, the "
"institute's first spacecraft, its instrument partly the sixth "
"form's own magnetometer, the schools competition's fourth "
"winner, and the launch's first signal, received at the array "
"ninety-one minutes after liftoff, was logged by the outreach "
"centre's youngest volunteer, the institute's log records, the "
"county's first spacecraft watched in by the county's own "
"cutting. The tracking station was built on October 20, 2002, "
"the array's fifth dish given a transmitter of its own and the "
"mission's whole downlink moved home, and the station's first "
"full year, the report records, missed not one of the mission's "
"daily passes, the comet's own account, the mission manager "
"said, received at the county's own rails. The station's own open days, the mission manager's own innovation and the outreach centre's second decade of crowds, gave the county's public the comet's daily pass to watch whole, the downlink's own screens repeating the control room's spare, and the open days' first year, the report records, filled the cutting's eastern mouth with the county's deck chairs, the mission's own account, the manager said at the second year's opening, received, watched and cheered, the county's spacecraft carried home by the county's own voices.")

E3 = ("[3] Title: The Second Dish, Day: February 18, 2004 "
"Content: The array's second dish was added on February 15, 2004, "
"the cutting's instruments six, the new dish funded by the "
"outreach centre's first decade of door money and sited, the "
"institute's minute book records, at the cutting's eastern "
"mouth where the first visitors had always stopped, the "
"public's own dish, the minute's marginal note says, pointed "
"by the visitor book's most frequent signer, the county's "
"amateur astronomer whose own logbooks, bound entire for the "
"institute's archive, had given the array its first decade of "
"comparisons, the era's whole account, the institute's closing "
"report of the decade wrote, one competition, one launch, one "
"station and one public dish, the county's radio sky kept by "
"the county's own hands, the report's last line the array's "
"first night's entry reproduced: the children chose the star, "
"and the county has been listening ever since. The closing report's own reprint, ordered by the outreach centre for every schoolroom the county kept, carries the decade's five dated works on its cover and the array's first entry on its last page, the institute's whole account of itself, the reprint's editor wrote, bracketed by two sentences the county's children could keep, the children chose the star, and the county has been listening ever since, the reprint's whole text between them the proof.")

ROWS276 = [
 N("P1", "The radio array was commissioned on September 12, 1993, with four dishes in a railway cutting.",
   "The radio array had more than three dishes when commissioned.",
   "entailment",
   "Four dishes is more than three.",
   0, "September 12, 1993"),
 N("P1", "The array's first target was named by a sixth form.",
   "School pupils chose the array's first observation target.",
   "entailment",
   "The sixth form that won the schools competition named the target.",
   1, "September 12, 1993"),
 N("P1", "The outreach centre was opened on June 7, 1996.",
   "The outreach centre opened before the comet mission launched.",
   "entailment",
   "June 7, 1996 precedes March 3, 2001.",
   2, "June 7, 1996"),
 N("P1", "The centre's first year hosted every school in the county and eleven from beyond it.",
   "All of the centre's first-year visitors came from within the county.",
   "contradiction",
   "Eleven schools came from beyond the county.",
   0, "June 7, 1996"),
 N("P1", "The comet mission was launched on March 3, 2001, carrying the schools' magnetometer.",
   "A school-built instrument flew on the comet mission.",
   "entailment",
   "The sixth form's own magnetometer was the mission instrument.",
   1, "March 3, 2001"),
 N("P1", "The launch's first signal was received ninety-one minutes after liftoff.",
   "The first signal was received less than two hours after launch.",
   "entailment",
   "Ninety-one minutes is under two hours.",
   2, "March 3, 2001"),
 N("P1", "The tracking station was built on October 20, 2002, with the array's fifth dish given a transmitter.",
   "The tracking station used a dish that was already part of the array.",
   "entailment",
   "The fifth dish, already in the array, was given the transmitter.",
   0, "October 20, 2002"),
 N("P1", "The station's first full year missed not one daily pass.",
   "The station achieved a perfect first-year pass record.",
   "entailment",
   "Missing not one pass is a perfect record.",
   1, "October 20, 2002"),
 N("P1", "The array's second dish was added on February 15, 2004, funded by door money.",
   "The second dish was the array's sixth instrument in the cutting.",
   "entailment",
   "The passage says the cutting's instruments were six with the new dish.",
   2, "February 15, 2004"),
 N("P1", "The new dish was pointed by the visitor book's most frequent signer.",
   "The public dish was aimed by a member of the public.",
   "entailment",
   "The most frequent visitor book signer is a member of the public.",
   0, "February 15, 2004"),
 N("P1", "The amateur astronomer's logbooks gave the array its first comparisons.",
   "The amateur's records predate the array's commissioning.",
   "entailment",
   "His logbooks provided a first decade of comparisons, so they must cover years before and after commissioning, including before it.",
   1, "September 12, 1993"),
 N("P1", "The outreach centre was given a public wing of the control building.",
   "The array's control room was closed to visitors.",
   "contradiction",
   "The control room's spare screens showed live data to visitors.",
   2, "June 7, 1996"),
 N("P1", "The comet mission was launched on March 3, 2001.",
   "The mission launched in the same century the array was commissioned.",
   "entailment",
   "Both 1993 and 2001 fall in the twentieth century.",
   0, "March 3, 2001"),
 N("P1", "The tracking station was built in 2002.",
   "The tracking station was operational before the second dish was added.",
   "entailment",
   "October 20, 2002 precedes February 15, 2004.",
   1, "October 20, 2002"),
 N("P1", "The new dish was funded by the centre's door money.",
   "The second dish cost more than the tracking station.",
   "neutral",
   "The passage gives no cost comparison.",
   2, "February 15, 2004"),
 N("P1", "The sixth form won the institute's first schools competition.",
   "The competition-winning class visited the array before the launch.",
   "neutral",
   "The passage does not describe the class's visits.",
   0, "September 12, 1993"),
 N("P1", "The array's first observation was a whole night on one pulsar.",
   "The array's first night observed several targets.",
   "contradiction",
   "The passage says a whole night spent on the transit of one pulsar.",
   1, "September 12, 1993"),
 N("P1", "The mission's downlink moved home to the tracking station.",
   "The comet mission initially used the array's dishes for its downlink.",
   "entailment",
   "Moving the downlink home implies the array had received it before.",
   2, "October 20, 2002"),
 N("P1", "The launch's first signal was logged by the centre's youngest volunteer.",
   "The volunteer worked at the outreach centre.",
   "entailment",
   "The centre's youngest volunteer is the outreach centre's volunteer.",
   0, "March 3, 2001"),
 N("P1", "The visitor book's most frequent signer was the county's amateur astronomer.",
   "No one signed the visitor book more often than the amateur astronomer.",
   "entailment",
   "Most frequent signer means the highest count of signatures.",
   1, "June 7, 1996"),
]

build(276, "news", {"P1": (E1, E2, E3), "P2": (E1, E2, E3)}, ROWS276)

# =================== 277: nli_mcq wiki, insurance @5 1936-1965
T277 = ("The mutual's war and rebuilding years are kept in its own loss "
"book, every dated entry of the hardest quarter-century an "
"insurance office could keep. The war risks scheme was launched "
"on September 1, 1939, the mutual's whole marine book "
"re-insured through the government's own pool on the war's "
"first morning, and the blitz claims were settled on "
"March 8, 1941, the town's whole losses paid within a "
"fortnight of the fires, the loss book's thickest wartime "
"page, its margin the secretary's single line, the town paid "
"while the town still smoked. The rebuilding fund was opened "
"on July 15, 1947, the mutual's own surplus lent to the town's "
"own householders at no interest, and the storm claims of the "
"east coast flood were paid on February 28, 1953, the loss "
"book's largest single entry, the whole coast's losses settled "
"inside a month. The mutual's centenary was kept on "
"May 12, 1965, the loss book displayed open at the blitz page "
"beside the fund's ledger and the flood's, the office's whole "
"hard account, the centenary's programme said, kept in one "
"book and one town's trust, the mutual's first hundred years, "
"the programme's last line records, measured not in premiums "
"but in mornings the town was paid. The centenary's own visitors, the programme's last note records, were led through the office by the loss book's own secretary, the blitz page's marginal hand in person, and the secretary's last tour of the day, given to the town's schools entire, ended at the book's open page with the tour's own single line, that the book's thickness was the town's own measure of the office's worth, one page for every morning the town was paid, the mutual's second hundred years, the secretary's closing sentence records, to be measured the same way. The book's own second century, the secretary's successor's first entry records, was opened at the flood page with the coast's new sea defences insured at the mutual's own standard, the loss book's whole doctrine, the successor's marginal note says, carried forward one page at a time, the office's true continuity, the centenary programme's last footnote concludes, not the building nor the book but the morning after, and the town's own measure, the programme's final line repeats, kept by the town that was paid and the office that paid it, the mutual's hundred years of mornings, the truest ledger the county kept. The mutual's own exhibition of the second century's first decade, mounted in the office's window and renewed each May, sets the loss book, the fund ledger and the defence policy in the order the county had kept them, and the window's single card, the successor's own, gives the office's whole account of itself in the county's plainest sentence: that the mutual had been the town's own arithmetic of bad mornings, and its hundred years, the card's last line, the county's own quietest public work.")

ROWS277 = [
 N("P1", "The war risks scheme was launched on September 1, 1939, the war's first morning.",
   "The mutual joined the government pool on the day the war began.",
   "entailment",
   "September 1, 1939, the war's first morning, is when the scheme launched.",
   0, "September 1, 1939"),
 N("P1", "The blitz claims were settled on March 8, 1941, within a fortnight of the fires.",
   "The blitz losses were paid in less than a month.",
   "entailment",
   "A fortnight is under a month.",
   1, "March 8, 1941"),
 N("P1", "The rebuilding fund lent the mutual's surplus at no interest.",
   "The town's householders borrowed from the fund interest-free.",
   "entailment",
   "No interest means interest-free borrowing.",
   2, "July 15, 1947"),
 N("P1", "The storm claims of the east coast flood were paid on February 28, 1953, settled inside a month.",
   "The flood losses were the loss book's largest single entry.",
   "entailment",
   "The passage states it was the largest single entry.",
   0, "February 28, 1953"),
 N("P1", "The mutual's centenary was kept on May 12, 1965, with the loss book displayed at the blitz page.",
   "The mutual was founded before 1870.",
   "entailment",
   "A centenary in 1965 means founding in 1865, before 1870.",
   1, "May 12, 1965"),
]

build(277, "wiki", {"P1": T277, "P2": T277}, ROWS277, band=(450, 700))
