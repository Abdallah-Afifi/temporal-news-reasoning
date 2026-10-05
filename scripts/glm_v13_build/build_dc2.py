import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from dc2_lib import build, row

# =================== 289: DC news, film/culture 2015-2019
F1 = ("[1] Title: The Little Screen's Big Decade, Day: April 10, 2016 "
"Content: The community cinema was licensed on April 7, 2016, the "
"town's first screen since the Regent closed, its projector bought by "
"the film society's hundred members and its first season, the "
"society's records note, sold out before the first reel. The "
"festival was founded on August 21, 2017, three days of the town's "
"own hall given to the county's film-makers, and the festival's "
"first competition, the records note, was won by the projectionist's "
"own short, the projector's first star. The arts centre's studio "
"was fitted on January 16, 2019, the society's teaching wing, its "
"first class the town's youngest critics, and the archive project "
"began on May 30, 2018, the society's whole collection scanned by "
"the members themselves, the town's film memory kept whole for the "
"archive's first public season. The archive's own open evenings, held in the library's hall every first Friday and staffed by the scanners themselves, gave the town its memory back one reel at a time, the evenings' first year, the society's records note, returning four hundred films to the town that had made them, the archive's whole doctrine, the evenings' card said, scan, show, and sign, the membership's thousandth name, the card's last line noted, only a matter of time.")

F2 = ("[2] Title: The Screen Goes Digital, Day: March 3, 2018 "
"Content: The digital projector was installed on February 27, 2018, "
"the community cinema's whole catalogue opened by one delivery, and "
"the society's silent season, the records note, played to the town's "
"own pianist entire, the cinema's first live score. The youth "
"section was founded on September 8, 2018, the town's teenagers "
"given the projector's second weekend, their first programme, the "
"records note, drawn entire from the archive project's first year, "
"the town's own memory curated by its newest hands, and the "
"society's membership passed its thousandth name on "
"November 12, 2019, the thousandth member, the records note, the "
"pianist's own granddaughter, the cinema's whole account, the "
"society's minute book says, three generations wide. The membership's own roll, printed in the society's last minute book of the era and read at the third hall's opening, carries the thousand names in the order they joined, the pianist's family three times, the projectionist's twice, and the founders' single line at the head, the cinema's whole human account, the roll's editor wrote, one continuous audience, the decade's truest ledger, kept by the people who paid to be in it.")

F3 = ("[3] Title: The Decade's Last Picture Show, Day: "
"December 14, 2019 "
"Content: The cinema's third hall was opened on October 5, 2019, "
"the society's own building bought and fitted in the town's "
"high street, its first film the projector's first season's "
"sell-out returned, and the decade's last screening, the "
"society's records note, was given to the archive project's "
"whole first year, the members' own scans shown to the town "
"that had made them, the society's closing minute of the era "
"giving the cinema its own epitaph: a hundred members, one "
"projector, three dates and a thousand names, the town's "
"whole film account, the minute's last line says, kept by "
"the people who turned up. The minute book's own last page, left deliberately half-ruled by the society's secretary for the next decade's first entry, carries the era's closing sentence alone, the cinema's account to be continued by whoever comes next, the secretary's marginal note instructing the book's successor to begin where the projector began, with the town's own first sell-out, the society's whole doctrine, the note says, continuity by attendance, the cinema's first century of town, the note's last clause adds, promised to whoever queues. The projector's own first season booklet, reprinted for the third hall's opening with the original hundred members' names in gold, gave the cinema's account its frontispiece and its promise, the town's screen, the booklet's cover says, kept lit by the same hundred names and all who followed, the society's truest sentence, the reprint's editor wrote, unchanged in three halls. The reprint's own second edition, given free to every seat at the third hall's first screening, carried the youth section's own afterword, the cinema's next hundred names predicted in the section's own hand, the society's whole account, the afterword's last line says, one booklet from the first gold to the next, the town's screen kept by the town's own ink.")

E289 = [
 ("licensing of the community cinema", "April 7, 2016"),
 ("founding of the festival", "August 21, 2017"),
 ("start of the archive project", "May 30, 2018"),
 ("installation of the digital projector", "February 27, 2018"),
 ("founding of the youth section", "September 8, 2018"),
 ("fitting of the arts centre studio", "January 16, 2019"),
 ("opening of the cinema's third hall", "October 5, 2019"),
 ("passing of the thousandth member", "November 12, 2019"),
]

P289 = [(0,1,2,3),(0,2,1,3),(1,2,3,4),(0,3,2,4),(1,3,3,5),(3,2,4,5),
        (0,4,2,6),(1,4,3,6),(2,5,4,6),(0,5,1,6),(3,6,5,6),(2,7,4,7),
        (0,6,1,7),(1,5,2,5),(3,5,5,7),(0,7,3,7),(2,6,1,4),(4,6,6,7),
        (1,6,2,7),(0,3,4,6)]
ROWS289 = [row("P1" if i % 2 == 0 else "P2",
               E289[a], E289[b], E289[c], E289[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P289)]
build(289, "news", {"P1": (F1, F2, F3), "P2": (F1, F2, F3)}, ROWS289)

# =================== 290: DC news, telecoms @5 1990-2004
T290a = ("[1] Title: The Cell Arrives on the Hill, Day: June 9, 1991 "
"Content: The first cell mast was raised on June 6, 1991, the "
"county's first, its signal covering the town and the valley's "
"whole rim, and the town's first car phone call, the gazette "
"records, was made by the haulier to his own wife from the hill "
"itself. The fibre ring was completed on October 14, 1993, the "
"town's whole exchange wired in a summer, and the internet "
"service was launched on February 20, 1996, the county's first "
"dial-up, its first hundred users, the gazette notes, logging "
"on from the library's own hall. The library's own first terminal, kept behind the counter and booked by the half hour, gave the town its first hundred internet users and the gazette its first technology column, written by the librarian herself and titled the counter's edge, the column's first year, the gazette's archive records, answering four hundred questions from the town, the library's whole account of the dial-up years kept at the counter where it began.")

T290b = ("[3] Title: Broadband and Beyond, Day: April 17, 2003 "
"Content: The broadband upgrade was switched on April 14, 2003, "
"the town's whole ring lit for the internet's second age, and "
"the town's own portal, the gazette's records note, took its "
"first bookings the same morning, the town's whole account of "
"itself, the portal's editor wrote, kept online from the first "
"day of the broadband, the exchange's own log giving the era "
"its three dated stones, the mast, the ring and the switch, "
"the county's whole telecom story, the log's last line says, "
"told in one hill's three ascents. The hill's own three masts, each replacing none and standing with all, were given the town's own names by the gazette's readers, the first the haulier's hill, the second the library's, the third the portal's, the county's telecom story told in three nicknames, the gazette's last column of the era wrote, each ascent kept by the people it connected, the hill's fourth name, the column's last line records, reserved by the readers for whoever climbs next. The gazette's own exhibition of the three masts' photographs, mounted in the library's hall for the broadband's first anniversary and visited by every school the county could send, gave the hill's three ascents their public account, the haulier's call, the library's terminal and the portal's morning, the county's whole telecom story told in one hill and three masts, the exhibition's card wrote, a hundred years of the town's own curiosity, the card's last line the librarian's, that the town had climbed the hill three times and found itself at the top each time. The exhibition's own visitors' book, the library's second oldest after the terminal's booking sheet, was signed at its closing by the haulier's own grandson on the portal's own keyboard, the county's whole telecom account, the book's card says, closed by three generations of one family's calls, the hill's fourth name, the book's last line records, still reserved, the town's own future kept off the page on purpose. The book's own last ruled line, the card's postscript notes, was left blank by the librarian's own instruction, the town's next call, the postscript says, to be entered by the hand that makes it, the county's telecom chronicle held open at its happiest page for the reader who climbs next, the exhibition's whole doctrine, the postscript concludes, the future booked by the half hour like the terminal before it.")

E290 = [
 ("raising of the first cell mast", "June 6, 1991"),
 ("completion of the fibre ring", "October 14, 1993"),
 ("launch of the internet service", "February 20, 1996"),
 ("switching on of the broadband upgrade", "April 14, 2003"),
]

P290 = [(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,2,3),(0,1,1,2)]
ROWS290 = [row("P1" if i % 2 == 0 else "P2",
               E290[a], E290[b], E290[c], E290[d], [0, 2, 1][i % 3])
           for i, (a, b, c, d) in enumerate(P290)]
T290c = ("[2] Title: The Dial-Up Years, Day: February 23, 1996 "
"Content: The internet service's first winter, the gazette's records "
"note, was spent teaching the town itself, the librarian's column "
"answering the first hundred users' questions one Friday at a time, "
"and the column's own popularity, the gazette's editor wrote, gave "
"the paper its youngest readership in a generation, the dial-up "
"years' whole account kept in print beside the terminals, the "
"town's own learning curve, the editor's last line said, published "
"weekly for anyone to climb. The column's own reprint, bound by the library at the dial-up era's close and shelved beside the terminal's booking sheet, gave the town's learning curve its permanence, the reprint's index, the librarian's own, listing every question by the Friday it was answered, the town's whole first internet year recoverable one week at a time, the library's truest teaching record and the gazette's youngest column, both kept in one binding, the reprint's preface says, for the next town that climbs.")

build(290, "news", {"P1": (T290a, T290c, T290b), "P2": (T290a, T290c, T290b)}, ROWS290)
