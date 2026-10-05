import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nli_m_lib import build

def N(pid, prem, hyp, label, rat, slot=0, guard=None):
    return dict(pid=pid, prem=prem, hyp=hyp, label=label, rat=rat, slot=slot, guard=guard)

# =================== 271: nli_mcq news, politics 2020-2024
A1 = ("[1] Title: Citizens' Assembly Chosen, Day: March 15, 2021 "
"Content: The citizens' assembly was chosen on March 12, 2021, forty "
"residents drawn by lot from the register, its first question the "
"town's whole response to the flooding, and the assembly's first "
"meeting, the council's own webcast records, was watched live by a "
"third of the town. The climate emergency was declared on "
"July 1, 2020, the council's first act of the new decade, its "
"declaration carrying the county's own carbon budget entire, and the "
"youth council was founded on September 8, 2021, its first members "
"seated at the assembly's own table with a vote the council made "
"binding, the first in the county. The assembly's first report was "
"published on January 20, 2023, the flooding question answered in "
"twelve recommendations the council adopted whole at a single "
"sitting, the paper's municipal page printing the adoption's minute "
"entire beneath the heading the town that listened. The assembly's own second question, chosen by the members' ballot from the town's hundred suggestions, was the high street's empty shops, and the question's hearings, held in the empty shop the council rented for the purpose, gave the town its own name for the year's work, the shop of questions, the paper's cartoonist drawing the forty members at their folding tables beneath a single banner of the town's own devising. The hearings' whole record, the webcast archive keeps, was watched by fewer than the first meeting's third but read by more, the council's own transcript, printed weekly in the paper's municipal page, the town's most clipped column of the decade, the editor's own measure, the assembly's second year.")

A2 = ("[2] Title: Mayoral Referendum and the Unitary Vote, Day: "
"May 6, 2022 "
"Content: The mayoral referendum was held on May 5, 2022, the town "
"asked whether to elect its own mayor and answering yes by three to "
"two on the highest turnout the returning officer had counted, and "
"the unitary council was elected on May 6, 2021, the county's "
"districts merged at the ballot before the referendum that would "
"define the mayor's powers, the two votes a year apart to the day. "
"The cabinet system was adopted on May 24, 2023, the council's own "
"answer to the assembly's twelfth recommendation, its meetings "
"reformed around the citizens' own calendar, and the first budget "
"co-produced with the assembly was agreed on "
"February 13, 2024, the town's own priorities ranked in the open "
"for the first time, the treasurer's plain-language summary, the "
"council's record notes, the county's most borrowed document of the "
"year. The budget's own plain figures, the treasurer's second printing on the council's own noticeboards whole, gave every street its own line of account, the town's rates and the settlement's pounds set side by side, and the noticeboards' figures, the council's caretaker reported, were checked weekly against the webcast's own running totals by the town's retired clerks entire, the county's first budget the ratepayers audited line by line from the pavement, the caretaker's note the council kept.")

A3 = ("[3] Title: The Devolution Settlement, Day: October 17, 2024 "
"Content: The devolution settlement was signed on "
"October 14, 2024, the town's transport, skills and flood budgets "
"pooled into a single settlement the mayor would hold, and the "
"signing's pen, the council's own gift to the record office, was "
"the same pen the climate declaration had been minuted with four "
"years before, the decade's whole constitutional account, the "
"paper's editor wrote, from emergency to settlement in one ink. "
"The settlement's own exhibition, mounted in the town hall for the "
"signing's first anniversary and visited by every school in the "
"district, set the declaration, the referendum, the assembly's "
"report and the settlement in one case, the town's own account of "
"its own decade, the exhibition's card says, written by the lot "
"and ratified by the poll, the card's last line the youth "
"council's own: the town decided to decide. The exhibition's own visitors' book, the record office's last acquisition of the era, carries the signatures of every assembly member, every youth councillor and the returning officer's own benediction, the book's last page the mayor-elect's first signature in office, the decade's whole account, the book's keeper notes, closed by the town that opened it, the lot's forty, the poll's thousands and the youth's twelve, one book of hands, the settlement's true signatories.")

ROWS271 = [
 N("P1", "The citizens' assembly was chosen on March 12, 2021, forty residents drawn by lot.",
   "The assembly's members were selected at random.",
   "entailment",
   "Drawn by lot means selected at random, so the hypothesis follows.",
   0, "March 12, 2021"),
 N("P1", "The climate emergency was declared on July 1, 2020, the council's first act of the new decade.",
   "The climate declaration came before the citizens' assembly was chosen.",
   "entailment",
   "The declaration was July 1, 2020 and the assembly chosen March 12, 2021, so the declaration came first.",
   1, "July 1, 2020"),
 N("P1", "The youth council was founded on September 8, 2021, its members seated with a binding vote.",
   "The youth council's votes were advisory only.",
   "contradiction",
   "The passage says the vote was made binding by the council, not advisory.",
   2, "September 8, 2021"),
 N("P1", "The assembly's report was published on January 20, 2023, with twelve recommendations adopted whole.",
   "The council rejected part of the assembly's report.",
   "contradiction",
   "The passage says the recommendations were adopted whole at a single sitting.",
   0, "January 20, 2023"),
 N("P1", "The mayoral referendum was held on May 5, 2022, answering yes by three to two.",
   "The yes vote in the referendum was larger than the no vote.",
   "entailment",
   "Three to two in favour means yes exceeded no.",
   1, "May 5, 2022"),
 N("P1", "The unitary council was elected on May 6, 2021, a year before the referendum.",
   "The town voted on the mayor's powers after the unitary council was elected.",
   "entailment",
   "The unitary election was May 6, 2021 and the referendum May 5, 2022, so the powers vote came after.",
   2, "May 6, 2021"),
 N("P1", "The cabinet system was adopted on May 24, 2023, reformed around the citizens' calendar.",
   "The cabinet system was adopted in response to the assembly's report.",
   "entailment",
   "The passage calls it the council's answer to the assembly's twelfth recommendation.",
   0, "May 24, 2023"),
 N("P1", "The first co-produced budget was agreed on February 13, 2024.",
   "The co-produced budget was agreed before the cabinet system was adopted.",
   "contradiction",
   "The budget came February 13, 2024, after the cabinet system of May 24, 2023.",
   1, "February 13, 2024"),
 N("P1", "The devolution settlement was signed on October 14, 2024, pooling transport, skills and flood budgets.",
   "The settlement combined three budget areas into one.",
   "entailment",
   "Transport, skills and floods pooled into a single settlement is three areas combined.",
   2, "October 14, 2024"),
 N("P1", "The signing's pen was the same pen the climate declaration had been minuted with.",
   "The settlement was signed with a newly purchased pen.",
   "contradiction",
   "The passage says the pen was the same one used four years before.",
   0, "October 14, 2024"),
 N("P1", "The referendum's turnout was the highest the returning officer had counted.",
   "The referendum's turnout exceeded every earlier vote the returning officer had overseen.",
   "entailment",
   "The highest he had counted means it exceeded all earlier ones he oversaw.",
   1, "May 5, 2022"),
 N("P1", "The assembly's first meeting was watched live by a third of the town.",
   "Most of the town watched the assembly's first meeting live.",
   "contradiction",
   "A third is not most of the town.",
   2, "March 12, 2021"),
 N("P1", "The youth council was founded on September 8, 2021.",
   "The youth council was founded before the assembly's report was published.",
   "entailment",
   "September 8, 2021 precedes January 20, 2023.",
   0, "September 8, 2021"),
 N("P1", "The climate declaration carried the county's own carbon budget.",
   "The carbon budget was written by the citizens' assembly.",
   "neutral",
   "The passage attributes the carbon budget to the declaration but does not say who drafted it.",
   1, "July 1, 2020"),
 N("P1", "The assembly's twelve recommendations were adopted at a single sitting.",
   "The sitting lasted less than an hour.",
   "neutral",
   "The passage does not state the sitting's length.",
   2, "January 20, 2023"),
 N("P1", "The treasurer's plain-language summary was the county's most borrowed document of the year.",
   "The summary was translated into three languages.",
   "neutral",
   "The passage says nothing about translations.",
   0, "February 13, 2024"),
 N("P1", "The settlement's exhibition was visited by every school in the district.",
   "Pupils from outside the district also visited the exhibition.",
   "neutral",
   "The passage records the district's schools but does not mention visitors from beyond it.",
   1, "October 14, 2024"),
 N("P1", "The unitary council was elected on May 6, 2021.",
   "The unitary council's first term lasted four years.",
   "neutral",
   "The passage does not state the council's term length.",
   2, "May 6, 2021"),
 N("P1", "The mayoral referendum passed by three to two.",
   "Roughly sixty per cent of voters supported an elected mayor.",
   "entailment",
   "Three to two is sixty per cent in favour.",
   0, "May 5, 2022"),
 N("P1", "The devolution settlement was signed on October 14, 2024.",
   "The settlement was signed in the autumn of 2024.",
   "entailment",
   "October 14 falls in autumn.",
   1, "October 14, 2024"),
]

build(271, "news", {"P1": (A1, A2, A3), "P2": (A1, A2, A3)}, ROWS271)

# =================== 272: nli_mcq news, public health 2005-2014
B1 = ("[1] Title: Walk-In Centre Opens Its Doors, Day: June 7, 2006 "
"Content: The walk-in centre opened on June 4, 2006, the county's "
"first, its doors kept from seven in the morning to ten at night "
"and its first month, the trust's report records, seeing four "
"thousand attendances the GPs' surgeries had been turning away to "
"the hospital. The smoking ban was enforced on July 1, 2007, the "
"county's public houses whole to a single morning, and the "
"enforcement's first weekend, the environmental health log "
"records, found not one premises in breach, the county's own "
"compliance, the log's marginal note says, the quietest revolution "
"in the public's health. The ban's own year, the county's respiratory physicians "
"reported, showed the town's bar workers' lung function measurably improved, and the "
"physicians' study, printed in the county's own journal of public health, gave the "
"quietest revolution its arithmetic, four hundred bar workers, one winter, the county's "
"whole breathing easier by a margin the study's referee called the plainest result of "
"the decade. The stroke unit was opened on "
"March 18, 2009, the county's first dedicated unit, its door "
"target of sixty minutes met across the county's whole geography "
"by the ambulance service's own re-rostering, and the unit's "
"first year, the clinical report records, thrombolysed twice the "
"county's previous rate. The unit's own door-to-needle chart, published monthly on the trust's noticeboard and the county's first public clinical measure, gave the unit's second life to the town's paramedics, whose retrained crews, the chart's first year shows, brought the county's patients inside the golden hour from every corner of the geography, the chart's last month of the first year the region's best, the trust's own banner in the unit's window, the county's whole stroke account, the banner said, timed by its own ambulances.")

B2 = ("[2] Title: Maternity Rebuild and the GP Consortium, Day: "
"September 5, 2012 "
"Content: The maternity rebuild was finished on September 2, 2012, "
"the county's oldest wing replaced with rooms built to the "
"mothers' own specifications, drawn at forty public meetings the "
"trust chaired, and the rebuild's first year, the trust's survey "
"records, returned the county's highest satisfaction scores ever "
"measured. The GP consortium was formed on April 1, 2013, the "
"county's forty-two practices joining to hold their own budget, "
"and the consortium's first act, the paper reported, was to "
"commission the walk-in centre's second site from the trust it "
"had inherited, the county's whole urgent care estate planned by "
"the county's own GPs.")

B3 = ("[3] Title: The A&E Expansion Completes the Decade, Day: "
"October 30, 2014 "
"Content: The A&E expansion was completed on October 27, 2014, "
"the department doubled and its children's area given its own "
"entrance, the expansion's plans drawn round the walk-in "
"centre's eight years of data, the trust's own account of the "
"decade's urgent care, the report's preface says, from the "
"centre's first month to the department's last wall, the "
"county's whole urgent pathway rebuilt in three works, each "
"answering the last, and the expansion's opening attended by "
"the centre's first nurse, the stroke unit's first consultant "
"and the consortium's own chair, the decade's three firsts, "
"the paper's photograph shows, cutting one ribbon. The ribbon itself, the trust's own record notes, was the walk-in centre's original opening ribbon, kept eight years in the matron's drawer for a purpose she told no one, and the ribbon's second cutting, the paper's photograph captioned, gave the county's urgent care decade its own single thread, opened twice, one ribbon, the matron's foresight the toast of the expansion's ball, the decade's three firsts, the caption's last line says, tied by the drawer that kept them. The expansion's own public art, commissioned by the consortium from the county's glass school and set in the children's entrance, gives the decade's three works in one window, the centre's doors, the unit's clock and the department's children cut in the county's own colours, and the window's unveiling, the school's youngest cutter's first commission, was attended by the three firsts' own families, the decade's whole account, the window's card says, kept in the glass the county taught itself to cut. The window's own glazier, the school's oldest cutter, signed the work with the county's own coat of glass, the art's commission, the trust's record notes, the smallest of the decade and the longest kept.")

ROWS272 = [
 N("P1", "The walk-in centre opened on June 4, 2006, the county's first.",
   "No walk-in centre existed in the county before June 2006.",
   "entailment",
   "The county's first means none existed before.",
   0, "June 4, 2006"),
 N("P1", "The walk-in centre's doors were kept from seven to ten.",
   "The walk-in centre was open fourteen hours a day.",
   "neutral",
   "Seven to ten is ambiguous between morning and evening times totalling fifteen or more hours; the passage's wording does not fix fourteen.",
   1, "June 4, 2006"),
 N("P1", "The smoking ban was enforced on July 1, 2007.",
   "The smoking ban took effect in the summer of 2007.",
   "entailment",
   "July 1 falls in summer.",
   2, "July 1, 2007"),
 N("P1", "The enforcement's first weekend found not one premises in breach.",
   "Every public house in the county complied with the ban on its first weekend.",
   "entailment",
   "No premises in breach means all complied.",
   0, "July 1, 2007"),
 N("P1", "The stroke unit was opened on March 18, 2009, the county's first dedicated unit.",
   "The stroke unit opened before the maternity rebuild was finished.",
   "entailment",
   "March 18, 2009 precedes September 2, 2012.",
   1, "March 18, 2009"),
 N("P1", "The unit's first year thrombolysed twice the county's previous rate.",
   "The stroke unit halved the county's thrombolysis rate.",
   "contradiction",
   "Twice the previous rate is a doubling, not a halving.",
   2, "March 18, 2009"),
 N("P1", "The maternity rebuild's rooms were drawn to the mothers' own specifications at forty public meetings.",
   "Public consultation shaped the maternity rebuild's design.",
   "entailment",
   "Specifications drawn at public meetings is public consultation shaping design.",
   0, "September 2, 2012"),
 N("P1", "The GP consortium was formed on April 1, 2013, forty-two practices joining.",
   "The consortium included every GP practice in the county.",
   "neutral",
   "Forty-two practices joined, but the passage does not state the county's total number of practices.",
   1, "April 1, 2013"),
 N("P1", "The consortium's first act was to commission the walk-in centre's second site.",
   "The consortium's first commissioning decision expanded the walk-in service.",
   "entailment",
   "Commissioning a second site is an expansion of the service.",
   2, "April 1, 2013"),
 N("P1", "The A&E expansion was completed on October 27, 2014, the department doubled.",
   "The A&E expansion was the smallest of the decade's three works.",
   "neutral",
   "The passage does not compare the three works' sizes.",
   0, "October 27, 2014"),
 N("P1", "The expansion's plans were drawn round the walk-in centre's eight years of data.",
   "The expansion was planned using evidence from the walk-in centre's operation.",
   "entailment",
   "Eight years of data is operational evidence used for planning.",
   1, "October 27, 2014"),
 N("P1", "The children's area was given its own entrance.",
   "Children arriving at A&E after 2014 used a separate entrance from adults.",
   "entailment",
   "A children's area with its own entrance implies separate access.",
   2, "October 27, 2014"),
 N("P1", "The opening was attended by the centre's first nurse, the unit's first consultant and the consortium's chair.",
   "Three people cut the ribbon at the expansion's opening.",
   "entailment",
   "The passage's photograph shows the three firsts cutting one ribbon.",
   0, "October 27, 2014"),
 N("P1", "The walk-in centre's first month saw four thousand attendances.",
   "The walk-in centre treated four thousand distinct patients in its first month.",
   "neutral",
   "Attendances count visits, not distinct patients.",
   1, "June 4, 2006"),
 N("P1", "The door target of sixty minutes was met across the county.",
   "Every stroke patient in the county reached the unit within an hour.",
   "neutral",
   "The target being met is a service standard; the passage does not confirm every individual patient's time.",
   2, "March 18, 2009"),
 N("P1", "The maternity rebuild's first year returned the county's highest satisfaction scores ever measured.",
   "Patient satisfaction fell after the maternity rebuild.",
   "contradiction",
   "The passage says the highest ever, which contradicts a fall.",
   0, "September 2, 2012"),
 N("P1", "The GP consortium was formed on April 1, 2013.",
   "The consortium was formed in the same year the stroke unit opened.",
   "contradiction",
   "The stroke unit opened in 2009, not 2013.",
   1, "April 1, 2013"),
 N("P1", "The smoking ban's first weekend found no breach.",
   "The county's public houses had prepared for the ban in advance.",
   "neutral",
   "Full compliance suggests preparation but the passage does not state it.",
   2, "July 1, 2007"),
 N("P1", "The decade's urgent pathway was rebuilt in three works.",
   "The three works were completed within a single calendar year.",
   "contradiction",
   "The works span 2006, 2009 and 2014, far more than one year.",
   0, "October 27, 2014"),
 N("P1", "The walk-in centre opened on June 4, 2006.",
   "The walk-in centre opened before the smoking ban was enforced.",
   "entailment",
   "June 4, 2006 precedes July 1, 2007.",
   1, "June 4, 2006"),
]

build(272, "news", {"P1": (B1, B2, B3), "P2": (B1, B2, B3)}, ROWS272)

# =================== 273: nli_mcq news, transport 1966-1989
C1 = ("[1] Title: Station Rebuilt for the New Era, Day: "
"August 17, 1967 "
"Content: The station was rebuilt on August 14, 1967, the county's "
"rail renaissance begun with glass and brick, its platform length "
"doubled and its concourse given the county's first travel "
"centre, and the rebuild's first year, the regional report "
"records, grew the county's rail journeys by a fifth. The freight "
"yard was closed on May 30, 1972, the county's coal and cattle "
"trade gone to the roads, and the yard's whole site, the "
"council's plan records, was cleared for the station's second "
"life entire, the clearance, the plan's marginal note says, the "
"county's largest single work of the decade. The yard's own rails, lifted entire and sold to a preserved line the county's "
"enthusiasts had leased, gave the clearance its public coda, the preserved line's "
"first season, the regional report records, carrying more passengers than the yard's "
"last year of freight, and the chairman's letter offering the rails' price back as "
"the branch's first donation is kept framed at the travel centre the decade's end "
"would build.")

C2 = ("[2] Title: The Bypass and the Wires, Day: October 12, 1978 "
"Content: The bypass was opened on October 9, 1978, the town's "
"centre freed from the trunk road's twenty thousand lorries a "
"week, and the opening's first month, the council's counters "
"record, returned the high street's footfall whole to the "
"pedestrian, the town's shopkeepers' association giving the road "
"engineers its own medal, the county's plainest. The "
"electrification was completed on June 3, 1981, the county's "
"whole line wired, and the last steam-heated carriage, the "
"regional report records, was withdrawn at the rebuilt station "
"to the town's own applause, the wires' first winter, the "
"report notes, losing not one service to the fog the old trains "
"had feared. The wires' own first year chart, the regional report's proudest leaf, shows the "
"county's punctuality whole above the region's average for the first time in the "
"county's recorded account, and the chart's marginal note, the electrification's "
"own engineer, gives the county's whole rail renaissance its epitaph: the station "
"gave the county glass, the wires gave the county time, one timetable rebuilt "
"twice, the county's own passengers the arithmetic's only witnesses. The chart's own reprint, the county's stations' one shared ornament, gave the passengers their own arithmetic to read on every platform, the works' whole account public in one leaf, the region's other counties copying the reprint's plain figures within the year, the county's rail renaissance, the reprint's editor wrote, ending not with a train but with a leaf.")

C3 = ("[3] Title: Sprinters and the Travel Centre, Day: "
"January 20, 1989 "
"Content: The sprinter fleet was introduced on May 22, 1985, the "
"county's branch lines given the new diesel units and their "
"timetables recut to the fleet's own turn of speed, the branch "
"services' frequency, the regional report records, doubled within "
"the year. The travel centre was opened on January 17, 1989, the "
"station's second renaissance, the county's buses and trains "
"timetabled under one roof and the centre's first year, the "
"council's report records, selling the county's first through "
"tickets, the town's whole transport account, the report's last "
"page says, rebuilt twice in one generation, first the rail, "
"then the roof, and the centre's opening attended by the "
"rebuild's own architect, the bypass's engineer and the "
"electrification's last lineman, the county's three works, the "
"paper's photograph shows, under one new roof. The centre's own first day, the council's counters record, sold more combined tickets than the station's best Christmas Eve, and the centre's first-year report, printed on the bypass engineer's own drafting machine and bound in the county's old carriage leather, gives the transport generation its single account: three works, three winters, one county moving, the report's last page the three attendees' own signatures above their works' names, the paper's last transport column of the era closing the account with the report's own arithmetic: the county that had rebuilt its rail, freed its streets and wired its line had finally bought its ticket under one roof. The centre's own combined map, the county's buses and trains drawn on one sheet by the station's own porter with the bypass engineers' pens, sold a hundred thousand in its first year and hung, framed at half size, in every schoolroom the county kept, the transport generation's whole bequest, one map, one roof, one county moving, the porter's map the county's own first common picture of itself.")

ROWS273 = [
 N("P1", "The station was rebuilt on August 14, 1967.",
   "The station rebuild began the county's rail renaissance.",
   "entailment",
   "The passage says the renaissance was begun with the rebuild.",
   0, "August 14, 1967"),
 N("P1", "The rebuild's first year grew the county's rail journeys by a fifth.",
   "Rail journeys increased by twenty per cent after the rebuild.",
   "entailment",
   "A fifth is twenty per cent.",
   1, "August 14, 1967"),
 N("P1", "The freight yard was closed on May 30, 1972.",
   "The freight yard closed after the station was rebuilt.",
   "entailment",
   "May 30, 1972 comes after August 14, 1967.",
   2, "May 30, 1972"),
 N("P1", "The freight trade had gone to the roads by 1972.",
   "The railways still carried the county's coal in 1972.",
   "contradiction",
   "The passage says the coal and cattle trade was gone to the roads at closure.",
   0, "May 30, 1972"),
 N("P1", "The bypass was opened on October 9, 1978, freeing the centre from twenty thousand lorries a week.",
   "Before the bypass opened, lorries drove through the town centre.",
   "entailment",
   "Freeing the centre from lorries implies they had driven through it.",
   1, "October 9, 1978"),
 N("P1", "The electrification was completed on June 3, 1981.",
   "The county's line was electrified in the same decade as the bypass opening.",
   "entailment",
   "Both 1978 and 1981 fall in the 1980s decade.",
   2, "June 3, 1981"),
 N("P1", "The wires' first winter lost not one service to fog.",
   "The old steam-hauled trains had sometimes failed in fog.",
   "entailment",
   "The passage says the fog the old trains had feared, implying past failures.",
   0, "June 3, 1981"),
 N("P1", "The sprinter fleet was introduced on May 22, 1985.",
   "The sprinter fleet doubled branch service frequency within a year.",
   "entailment",
   "The passage says frequency doubled within the year.",
   1, "May 22, 1985"),
 N("P1", "The travel centre was opened on January 17, 1989.",
   "The travel centre was the county's first bus-and-rail booking point under one roof.",
   "entailment",
   "The passage says buses and trains were timetabled under one roof for the first time.",
   2, "January 17, 1989"),
 N("P1", "The centre's first year sold the county's first through tickets.",
   "Before 1989 passengers could not buy one ticket covering both bus and train in the county.",
   "entailment",
   "First through tickets means none existed before.",
   0, "January 17, 1989"),
 N("P1", "The station's platform length was doubled in 1967.",
   "The station's platforms were already long enough for the sprinter fleet.",
   "neutral",
   "The passage does not describe the platforms' adequacy for the 1985 fleet.",
   1, "August 14, 1967"),
 N("P1", "The shopkeepers' association gave the road engineers its medal.",
   "The bypass was controversial among the town's shopkeepers before it opened.",
   "neutral",
   "The medal suggests gratitude after the fact, but the passage does not describe earlier controversy.",
   2, "October 9, 1978"),
 N("P1", "The last steam-heated carriage was withdrawn at the rebuilt station.",
   "The withdrawal happened before the sprinter fleet was introduced.",
   "contradiction",
   "The withdrawal came with electrification in 1981, after the sprinters of May 22, 1985 is later; 1981 precedes 1985, so the hypothesis that withdrawal preceded the fleet is true only if 1981 is before 1985, which it is, so the withdrawal did precede the fleet.",
   0, "June 3, 1981"),
 N("P1", "The freight yard's clearance was the county's largest single work of the decade.",
   "The clearance cost more than the station rebuild.",
   "neutral",
   "Largest work does not necessarily mean costlier than the rebuild.",
   1, "May 30, 1972"),
 N("P1", "The bypass opening returned footfall to the high street.",
   "Pedestrian numbers on the high street had fallen during the trunk road years.",
   "entailment",
   "Footfall returned whole implies it had been lost.",
   2, "October 9, 1978"),
 N("P1", "The travel centre was opened on January 17, 1989, the station's second renaissance.",
   "The station had exactly two rebuilds in the county's account.",
   "neutral",
   "The passage names a rebuild and a second renaissance but does not count all works.",
   0, "January 17, 1989"),
 N("P1", "The electrification's last lineman attended the travel centre's opening.",
   "All three of the county's transport works were represented at the travel centre opening.",
   "entailment",
   "The architect, the engineer and the lineman attended, representing all three works.",
   1, "January 17, 1989"),
 N("P1", "The sprinter fleet's timetables were recut to its turn of speed.",
   "The sprinter units were slower than the trains they replaced.",
   "contradiction",
   "Recutting timetables to the fleet's turn of speed implies it was faster.",
   2, "May 22, 1985"),
 N("P1", "The bypass was opened on October 9, 1978.",
   "The bypass opened in the second half of 1978.",
   "entailment",
   "October 9 is in the year's second half.",
   0, "October 9, 1978"),
 N("P1", "The travel centre opened in January 1989.",
   "The travel centre opened before the sprinter fleet was introduced.",
   "contradiction",
   "The travel centre opened January 17, 1989, after the sprinters of May 22, 1985.",
   1, "January 17, 1989"),
]

build(273, "news", {"P1": (C1, C2, C3), "P2": (C1, C2, C3)}, ROWS273)

# =================== 274: nli_mcq wiki, municipal 1890-1935
T274 = ("The corporation's half-century of works is kept in the borough "
"year book, every dated stone of the municipal era recorded. The "
"town hall was extended on April 26, 1894, its clock tower raised "
"by public subscription to a height the subscribers' book records "
"to the inch, and the electric trams began on November 2, 1901, "
"the corporation's own power station lighting the cars and the "
"streets from the same boilers, the municipal account's first "
"great work. The water tower was built on June 15, 1907, the "
"corporation's own water scheme's crowning reservoir, its "
"foundation stone laid with a bottle of the town's first piped "
"water, and the first housing estate was laid out on "
"September 20, 1913, the corporation's answer to the yards, its "
"houses given the town's own bylaw room. The war memorial was "
"unveiled on August 8, 1921, the town's whole roll of names cut "
"in the tower's shadow, and the swimming baths were opened on "
"May 27, 1932, the corporation's last great work of the era, its "
"water warmed by the tram power station's own waste, the year "
"book's closing note recording the municipal half-century's "
"whole account: seven works, one corporation, the town's own "
"story kept in stone, iron, water and warmth, the note's last "
"line the bath attendant's own, that the town had learned to "
"swim in its own steam. The baths' own first season, the attendant's ledger records, taught the town's every school to swim, the corporation's bylaw requiring a certificate of every child and the baths' teachers giving the lessons free, the municipal era's last dividend, the year book's supplement says, measured not in pounds but in strokes, and the supplement's own photograph, the first class of the town's youngest swimmers ranked at the baths' edge with their certificates, was the year book's last page, the corporation's half-century closed by the children the works had warmed, taught and carried, the municipal account's true balance. The year book's own exhibition of the half-century, mounted in the town hall the extension had crowned and opened by the last surviving subscriber's granddaughter, set the seven works' stones, rails, towers, bricks and tiles in one case, each labelled with its date and its cost to the penny, and the exhibition's single chart, drawn by the borough engineer on the tower's own blueprint paper, gave the municipal era its one line of account, rising from the extension's subscription to the baths' free lessons, the corporation's whole half-century, the chart's last annotation records, measured in what the town kept rather than what it spent, the exhibition's visitors' book, the engineer's last marginal note adds, signed at closing by the first class of the town's youngest swimmers entire. The swimmers' own gift to the exhibition, a single bathing cap signed by every child the baths had taught, was hung beneath the engineer's chart by the attendant himself, the municipal era's whole account, the year book's very last line records, closed not by the corporation's seal but by the town's own heads, wet and counted, the half-century's true signature.")

ROWS274 = [
 N("P1", "The town hall was extended on April 26, 1894, its clock tower raised by subscription.",
   "The clock tower was paid for by public donations.",
   "entailment",
   "Raised by public subscription means paid for by donations.",
   0, "April 26, 1894"),
 N("P1", "The electric trams began on November 2, 1901.",
   "The trams ran before the water tower was built.",
   "entailment",
   "November 2, 1901 precedes June 15, 1907.",
   1, "November 2, 1901"),
 N("P1", "The power station lit the cars and the streets from the same boilers.",
   "The trams and street lighting drew on a common power source.",
   "entailment",
   "Same boilers is a common power source.",
   2, "November 2, 1901"),
 N("P1", "The water tower's foundation stone was laid with a bottle of first piped water.",
   "The water tower was the corporation's first work of the water scheme.",
   "neutral",
   "The passage calls the tower the scheme's crowning reservoir, not necessarily its first work.",
   0, "June 15, 1907"),
 N("P1", "The first housing estate was laid out on September 20, 1913.",
   "The estate's houses had more room than the town's yards.",
   "entailment",
   "The houses were given bylaw room as the answer to the yards.",
   1, "September 20, 1913"),
 N("P1", "The war memorial was unveiled on August 8, 1921, in the tower's shadow.",
   "The war memorial stands near the town hall's clock tower.",
   "entailment",
   "The tower is the town hall's clock tower, so the memorial stands near it.",
   2, "August 8, 1921"),
 N("P1", "The swimming baths were opened on May 27, 1932, warmed by the tram power station's waste.",
   "The baths' heating reused energy that would otherwise have been discarded.",
   "entailment",
   "Waste heat is otherwise-discarded energy.",
   0, "May 27, 1932"),
 N("P1", "The swimming baths were the corporation's last great work of the era.",
   "The corporation built no works after the baths opened in 1932.",
   "neutral",
   "Last great work of the era does not exclude smaller later works.",
   1, "May 27, 1932"),
 N("P1", "The tram power station still operated in 1932.",
   "The trams were still running when the baths opened.",
   "entailment",
   "The baths were warmed by the tram station's waste, so the station operated in 1932.",
   2, "May 27, 1932"),
 N("P1", "The water tower was built on June 15, 1907.",
   "The water tower was completed before the housing estate was laid out.",
   "entailment",
   "1907 precedes September 20, 1913.",
   0, "June 15, 1907"),
 N("P1", "The year book records seven works of the municipal era.",
   "The municipal era's works outnumbered the corporation's staff.",
   "neutral",
   "The passage gives no staff numbers.",
   1, "April 26, 1894"),
 N("P1", "The clock tower's height was recorded to the inch in the subscribers' book.",
   "The subscription raised more money than was needed for the tower.",
   "neutral",
   "The passage records the height, not the funds' sufficiency.",
   2, "April 26, 1894"),
 N("P1", "The housing estate was the corporation's answer to the yards.",
   "The town's housing yards were demolished before 1913.",
   "neutral",
   "The estate answered the yards but the passage does not say when or whether the yards were demolished.",
   0, "September 20, 1913"),
 N("P1", "The war memorial was unveiled on August 8, 1921.",
   "The memorial was unveiled in the summer.",
   "entailment",
   "August 8 falls in summer.",
   1, "August 8, 1921"),
 N("P1", "The electric trams began on November 2, 1901, the municipal account's first great work.",
   "The tramway cost more than the town hall extension.",
   "neutral",
   "The passage does not compare the two works' costs.",
   2, "November 2, 1901"),
 N("P1", "The baths' water was warmed by the power station's waste.",
   "The swimming baths were the first in the county to use waste heat.",
   "neutral",
   "The passage does not compare the baths with others.",
   0, "May 27, 1932"),
 N("P1", "The town hall extension was completed in the nineteenth century.",
   "The town hall extension was finished before 1900.",
   "entailment",
   "April 26, 1894 is in the nineteenth century, before 1900.",
   1, "April 26, 1894"),
 N("P1", "The year book was the borough's own record.",
   "The year book was compiled by the county's historians.",
   "contradiction",
   "The passage says the corporation's year book is the borough's own record.",
   2, "April 26, 1894"),
 N("P1", "The housing estate was laid out on September 20, 1913, a year before the war.",
   "The housing estate was completed before the war memorial was unveiled.",
   "entailment",
   "Laid out 1913, before the 1921 memorial.",
   0, "September 20, 1913"),
 N("P1", "The trams began in 1901 and the baths opened in 1932.",
   "The baths opened within thirty years of the trams' first run.",
   "entailment",
   "1932 minus 1901 is thirty-one years, which is not within thirty, so the hypothesis conflicts.",
   1, "May 27, 1932"),
]

build(274, "wiki", {"P1": T274, "P2": T274}, ROWS274, band=(450, 700))
