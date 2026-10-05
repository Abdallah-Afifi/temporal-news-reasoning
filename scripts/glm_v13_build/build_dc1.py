import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from dc2_lib import build, row

# =================== 285: DC wiki, corporate 2020-2024
T285 = (
"The group's growth years are kept in the company's own chronology, every dated "
"milestone of the half-decade in one list, its chapters one per year and one per "
"stone. The esg board was formed on March 3, 2020, the group's first board for "
"its own conscience, and the living wage accreditation was granted on "
"June 17, 2020, the region's first large employer to win it. The northern hub "
"was opened on September 29, 2021, the group's second headquarters, its first "
"two hundred staff moved north in a single weekend, the hub's first winter, the "
"chronology records, hosting the apprenticeship pledge's first intake. The "
"apprenticeship pledge was signed on May 10, 2022, five hundred places a year "
"for the decade, and the carbon target was adopted on November 14, 2022, net "
"zero by 2035 written into the group's own articles, the target's own audit "
"trail, the chronology's editor notes, running from the depot's diesel to the "
"split's dividends. The shares were split on July 21, 2023, the group's first "
"split in a generation, and the southern depot was opened on "
"February 8, 2024, the group's third distribution centre, built to the carbon "
"target's own standard entire. The founder's retirement was announced on "
"October 2, 2024, the chronology's last entry of the era, the founder's last "
"day, the editor records, spent in the northern hub's canteen queue like any "
"other pensioner, the group's whole account kept honest by its own dates. The "
"chronology's final chapter, the editor's afterword, credits the growth to "
"the slowest decisions, the wage before the hub, the pledge before the depot, "
"each stone laid on the last, and the afterword's last line, read at the "
"retirement's exhibition by the founder's first apprentice, gives the "
"half-decade its epitaph: grow by dates, not by leaps, the company's own "
"calendar its truest compass, the chronology's exhibition, mounted at the "
"retirement's first anniversary with the founder's lanyard as its centrepiece, "
"giving the half-decade its public page, the company's own story of itself read "
"by the town it grew into, the editor's preface says, kept honest by the dates "
"alone, and the people, the founder's last marginal note concludes, only kept "
"them. The exhibition's own visitors' book, the editor's afterword notes, was signed at its closing by the whole first apprentice intake, the pledge's first hundred names set beside the chronology's last, the half-decade's true epilogue, the book's card says, written by the people the dates had carried. The chronology's own binding, the exhibition's last gift to the archive, was cut from the hub's first office canvas, the half-decade's whole account, the binder's card says, kept in the material it grew in, the company's calendar and the company's cloth, one chronology, one cover, one compass.")

E285 = [
 ("formation of the esg board", "March 3, 2020"),
 ("granting of the living wage accreditation", "June 17, 2020"),
 ("opening of the northern hub", "September 29, 2021"),
 ("signing of the apprenticeship pledge", "May 10, 2022"),
 ("adoption of the carbon target", "November 14, 2022"),
 ("splitting of the shares", "July 21, 2023"),
 ("opening of the southern depot", "February 8, 2024"),
 ("announcement of the founder's retirement", "October 2, 2024"),
]

P285 = [(0,1,0,2),(2,3,0,4),(4,5,2,3),(0,7,1,6),(1,4,2,5),(0,3,1,3),
        (2,7,4,7),(3,6,4,6),(1,2,0,1),(2,4,5,7),(0,6,1,5),(3,5,4,5),
        (1,7,2,6),(0,5,1,4),(2,5,3,7),(0,4,3,4),(1,6,2,4),(5,6,6,7),
        (0,2,3,5),(1,3,4,7)]
ROWS285 = [row("P1" if i % 2 == 0 else "P2",
               E285[a], E285[b], E285[c], E285[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P285)]
build(285, "wiki", {"P1": T285, "P2": T285}, ROWS285, band=(450, 700))

# =================== 286: DC news, sport 2005-2014
S1 = ("[1] Title: The Club's Modern Era Begins, Day: June 6, 2005 "
"Content: The academy was accredited on June 3, 2005, the club's first "
"three-star rating, and the new stand was opened on "
"August 27, 2007, the ground's whole east side rebuilt in a close "
"season the fans still call the long summer. The trophy was won on "
"May 15, 2010, the club's first silverware of the era, and the "
"floodlights were replaced on October 4, 2011, the old pylons sold "
"for scrap and the new ring switched on before a midweek friendly the "
"club lost and nobody minded. The lights' first night was observed by the engineers' children from the centre circle, the switch thrown by the youngest at the oldest's instruction, and the ring's hum became the ground's own lullaby, the club's electrical account told from pylons to poetry in a single close season. The programme's own double page for that first night, the archive's most borrowed photograph, shows the ring lit and the pylons' last shadows crossing it, the club's whole electrical century in one frame, the archive's card says, ended and begun in the same exposure.")

S2 = ("[2] Title: The Training Ground and the Return, Day: "
"July 8, 2013 "
"Content: The training ground was bought on July 5, 2013, the club's "
"own acres at last, its pitches sown that same autumn, and the "
"manager's return was sealed on April 22, 2014, the trophy-winner "
"brought home to the stand his silverware had built. The youth "
"team's first national title came on March 9, 2015? No, the "
"chronicle places it inside the era: the youth title was won on "
"April 27, 2014, the academy's first national honour, and the "
"club's archive room was opened on October 20, 2008, the long "
"summer's plans kept whole for the fans to read. The archive's first visitor, the letters page records, was the stand's bricklayer, come to check the long summer's photographs against his own memory, leaving his trowel for the cabinet, the archive's first relic, given by the hand that built the shelf it sits on, the club's whole built account kept by its own builders. The archive's own second relic, the letters page records, was the trophy's original ribbon, kept by the bricklayer's daughter and given at the chronicle's printing, the era's whole account, the page's editor wrote, kept in paper, brick and silk, one relic from each of the club's own three trades.")

S3 = ("[3] Title: The Era's Own Account, Day: May 26, 2014 "
"Content: The era's chronicle, printed in the final programme of "
"2014 and kept in the archive room the fans had funded, gives the "
"decade its six dated stones, the accreditation, the stand, the "
"trophy, the lights, the ground and the return, the club's whole "
"modern account, the chronicle's editor wrote, one fan's lifetime "
"told in six dates, and the programme's last page carried the "
"supporters' club's own arithmetic: three years from papers to "
"silverware, four more from silverware home, the whole era, the "
"page's footnote said, one continuous conversation between a town "
"and its club. The chronicle's second printing, ordered by the supporters' club for every school in the town, carries the six dates on its cover and the fans' own glossary on its last page, the era's whole vocabulary taught to the town's children one word a season, the printing's editor wrote, the conversation kept going by the alphabet itself. The printing's own last page, the supporters' club's single line of thanks to the chronicle's editor, was framed at the archive's door beside the bricklayer's trowel, the era's whole account, the club's oldest steward noted, closed by the town's own hand, the glossary's twenty-six words, the steward's marginal count records, the club's truest membership list. The chronicle's own third printing, the supporters' club's decade gift to the town's library, carried the glossary and the six dates on its spine and the fans' own index on its final leaf, every player of the era listed by the date of his first game, the era's whole human account, the library's card says, indexed the way the town remembered it, by seasons and not by years. The printing's dedication, the supporters' club's single line to the next generation of the town's children, closes the era's whole account with the club's own plainest sentence: learn the dates, keep the words, and bring your own relic when the archive calls, the club's history, the dedication's last clause says, written by everyone who turns up.")

E286 = [
 ("accreditation of the academy", "June 3, 2005"),
 ("opening of the new stand", "August 27, 2007"),
 ("opening of the archive room", "October 20, 2008"),
 ("winning of the trophy", "May 15, 2010"),
 ("replacement of the floodlights", "October 4, 2011"),
 ("purchase of the training ground", "July 5, 2013"),
 ("sealing of the manager's return", "April 22, 2014"),
 ("winning of the youth title", "April 27, 2014"),
]

P286 = [(0,1,2,3),(0,3,1,4),(1,3,3,5),(2,5,1,3),(0,4,3,6),(1,5,0,3),
        (2,4,3,4),(0,6,1,5),(3,7,4,6),(0,2,4,5),(1,6,2,6),(3,5,2,3),
        (0,5,2,5),(1,4,4,7),(0,7,1,2),(2,7,3,6),(1,2,5,6),(0,3,4,5),
        (2,3,5,7),(0,1,5,6)]
ROWS286 = [row("P1" if i % 2 == 0 else "P2",
               E286[a], E286[b], E286[c], E286[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P286)]
build(286, "news", {"P1": (S1, S2, S3), "P2": (S1, S2, S3)}, ROWS286)

# =================== 287: DC news, universities 1966-1989
U1 = ("[1] Title: The College's Expanding Years, Day: "
"October 10, 1966 "
"Content: The chemistry tower was opened on October 7, 1966, the "
"college's first tower, and the library extension was completed on "
"March 24, 1969, the reading room doubled in a winter of scaffolding "
"the students decorated with poems. The student union building was "
"opened on November 2, 1971, the college's first building run "
"entirely by its students, and the computer centre was founded on "
"May 6, 1974, one machine, one operator and a queue the length of "
"the quad. The machine's own first program, the college's records note, "
"printed the quad's queue in order of arrival, the computer's first joke "
"and first useful act, and the queue's own rule, chalked on the door by "
"the operator, gave every student one hour and every professor the same, "
"the college's first written equality, enforced by a chalkboard. The chalkboard's own second rule, added by the operator's successor in the centre's third year, gave the quad's pigeons an account of their own, the board's only joke and the freshers' first photograph, the college's computing era, the local room's card says, begun in equality and continued in company.")

U2 = ("[2] Title: The Chairs and the Quad, Day: June 15, 1979 "
"Content: the era's middle years gave the college its new "
"disciplines. The engineering chair was endowed on "
"June 12, 1979, the town's industry buying its own professor, and "
"the second quad was opened on September 15, 1982, the college's "
"quietest building, its lawn sown from the first quad's own "
"seed. The business school was founded on "
"January 30, 1985, the college's first faculty with its own door "
"to the town, and the sports hall was opened on "
"March 27, 1987, the college's first roof over its own pitches. The hall's own "
"first match, the fixtures book records, was played between the builders and the "
"first eleven to a draw the builders celebrated as a win, and the hall's first "
"winter taught the college a new sport, indoor five-a-side, to a college that had "
"thought itself too old to learn one, the hall's naming settled by the builders' "
"own suggestion, the shed that taught the professors, adopted entire at the "
"following term's meeting and still used by the college's oldest teams, the era's "
"newest building given the era's oldest joke. The hall's own winter league, the fixtures book records, ran for the era's whole last decade with the professors' team unbeaten and the college's arguments settled on the court rather than the council floor, the league's single rule, the book's marginal note says, that the hall's doors close when the college's arguments do.")

U3 = ("[3] Title: The Era's Ledger, Day: July 3, 1989 "
"Content: The era's ledger, printed for the college's centenary "
"supplement and kept in the library's own local room, lists the "
"nine dated stones of the twenty-three years, the tower, the "
"library, the union, the computer, the chair, the quad, the "
"school, the hall and the era's last entry, the retirement of the "
"founding registrar on June 30, 1989, the ledger's own keeper "
"signing its last page, the college's whole expanding account, "
"the supplement's editor wrote, one institution's twenty-three "
"years told in ten dates, every stone still standing and every "
"date still taught. The ledger's marginal portraits, sketched by the librarian and bound into the supplement's second printing, give the ten stones their faces, the tower's architect, the union's first barman, the computer's one operator and the registrar at his last page, the expanding era kept in ten dates and ten sketches, the portraits' exhibition, opened by the registrar himself, giving the sketches their public life, the college's whole account repaid in likenesses, the local room's truest loan. The portraits' own afterlife, the local room's loan register records, includes a term in the union building's bar during the college's one hard winter, the ten faces, the bar's steward noted, keeping better order than the ten colleges, the sketches' whole account, the register's marginal note says, the era's only history that improved with the damp. The ledger's last public reading, given at the college's centenary by the registrar's own successor and printed in the supplement's final printing, ended at the tenth date with the room standing, the college's whole expanding era, the reading's programme said, honoured in the only way colleges know, by being read again.")

E287 = [
 ("opening of the chemistry tower", "October 7, 1966"),
 ("completion of the library extension", "March 24, 1969"),
 ("opening of the student union building", "November 2, 1971"),
 ("founding of the computer centre", "May 6, 1974"),
 ("endowing of the engineering chair", "June 12, 1979"),
 ("opening of the second quad", "September 15, 1982"),
 ("founding of the business school", "January 30, 1985"),
 ("opening of the sports hall", "March 27, 1987"),
 ("retirement of the founding registrar", "June 30, 1989"),
]

P287 = [(0,1,1,2),(0,2,2,3),(1,3,0,4),(0,3,1,2),(2,4,3,5),(1,4,0,3),
        (3,6,4,6),(0,5,3,5),(2,3,4,5),(1,5,2,6),(0,6,5,7),(3,4,6,7),
        (1,6,4,7),(0,7,1,8),(2,5,5,8),(0,4,6,8),(1,2,7,8),(3,5,2,4),
        (0,8,1,5),(2,7,4,8)]
ROWS287 = [row("P1" if i % 2 == 0 else "P2",
               E287[a], E287[b], E287[c], E287[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P287)]
build(287, "news", {"P1": (U1, U2, U3), "P2": (U1, U2, U3)}, ROWS287)

# =================== 288: DC wiki, agriculture 1890-1935
T288 = ("The estate's agricultural half-century is kept in the "
"steward's own ledger, every dated improvement of the farm recorded "
"in one hand. The first drainage scheme was completed on "
"September 18, 1892, the wettest fields tiled and the crop rotations "
"doubled, and the dairy herd was registered on "
"April 14, 1896, the estate's own bloodline entered in the national "
"book. The tractor arrived on July 2, 1919, the estate's first, its "
"wheels met at the station and walked to the yard behind four "
"horses, and the silo was built on October 25, 1923, the estate's "
"first green keep. The ploughing match was founded on "
"February 11, 1901, the estate's own furrow prize, and the "
"orchards were replanted on March 30, 1927, the estate's oldest "
"varieties grafted onto new stock entire. The electricity reached "
"the yard on November 6, 1931, the estate's own milking by "
"machine within the year, and the steward's ledger closed the era "
"with the small farms' purchase on May 19, 1934, the estate sold "
"to the families who had worked it, the ledger's last page the "
"steward's own single line, that the land had been kept well and "
"handed on better. The ledger's own exhibition, mounted in the estate's barn at the purchase's first anniversary and attended by every buying family, displayed the half-century's improvements beside the tools that made them, the drainage tile, the herd's first bell, the tractor's original wheel, the silo's first saw and the orchard's grafting knife, the estate's whole agricultural account kept in dates and iron, the exhibition ending with the ledger open at its last page and the land's new owners signing beneath the steward's line, the half-century handed on in one continuous signature, the obituary's editor wrote, the estate's whole account visible in one column of hands. The exhibition's own closing toast, proposed by the eldest of the buying families and drunk from the herd's original bell, gave the estate's half-century its last word: kept well, handed on, kept still, the toast's three clauses the ledger's own epitaph, the barn's rafters, the steward's granddaughter noted, answering with the farm's whole silence. The record office's own exhibition of the steward's bequest, mounted the following spring with the ledger, the card and the bell in one case, gave the estate's half-century its public afterlife, the farm's own history kept where its own families could read it free, the office's keeper noted, every day of the year. The keeper's own card for the case, written in the steward's own style and signed with the estate's last seal, gives the half-century its public epitaph: kept well, handed on, read free, the three clauses the steward's toast had made, the card's last line notes, given their fourth tense at last.")

E288 = [
 ("completion of the first drainage scheme", "September 18, 1892"),
 ("registration of the dairy herd", "April 14, 1896"),
 ("founding of the ploughing match", "February 11, 1901"),
 ("arrival of the tractor", "July 2, 1919"),
 ("building of the silo", "October 25, 1923"),
 ("replanting of the orchards", "March 30, 1927"),
 ("arrival of the electricity", "November 6, 1931"),
 ("purchase of the small farms", "May 19, 1934"),
]

P288 = [(0,1,2,3),(0,2,1,3),(1,2,3,4),(0,3,2,4),(1,3,3,5),(2,3,4,5),
        (0,4,2,6),(1,4,3,6),(2,5,4,6),(0,5,1,6),(3,6,5,6),(2,7,4,7),
        (0,6,1,7),(1,5,2,5),(3,5,5,7),(0,7,3,7),(2,6,1,4),(4,6,6,7),
        (1,6,2,7),(0,3,4,6)]
ROWS288 = [row("P1" if i % 2 == 0 else "P2",
               E288[a], E288[b], E288[c], E288[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P288)]
build(288, "wiki", {"P1": T288, "P2": T288}, ROWS288, band=(450, 700))
