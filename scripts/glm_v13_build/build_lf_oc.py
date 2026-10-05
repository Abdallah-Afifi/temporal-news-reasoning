import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lf_oc_lib import build_lf, lf_row, build_oc, oc_row
from dateutil import parser as dp

# ---------- 302: longform news, central banking 1890-1935
B1 = ("[1] Title: The Bank's Charter Century, Day: June 8, 1892 "
"Content: The bank's charter was renewed on June 5, 1892, the county's "
"oldest institution given its second hundred years, and the first "
"branch was opened on October 21, 1897, the market town's own counter "
"cut from the head office's retired mahogany. The note issue was "
"consolidated on March 14, 1901, the county's whole paper brought "
"under one signature, and the gold reserve was reached on "
"September 30, 1907, the bank's own target met a year early. The renewal's own account, printed in the bank's centenary guide, gives the four decades their arithmetic: the charter's second hundred years begun in debt and ended in gold, the county's paper under one signature, the war's loan raised in the county's own morning, the branches doubled for the county's new towns, the old parity restored and the two ledgers bound, the bank's whole middle years, the guide's editor wrote, told in the numbers the county itself had trusted. The guide's own second volume, written by the bank's retired secretary and given to every branch's counter, carried the arithmetic beside the anecdotes, the charter's facsimile and the merger's pen photographed whole, the bank's middle years, the secretary's preface says, kept public by the branch network that made them, the volume's third printing ordered by the depositors themselves at the exhibition's closing. The closing's own toast, proposed by the bank's oldest depositor and drunk from the charter's own commemorative glass, gave the middle years their last word: trusted, renewed, kept, the toast's three clauses the guide's true epilogue, the glass itself, the cashier's last note records, refilled at every branch's opening since. The glass's own case, cut from the exhibition's oak and hinged by the county's last silversmith, carries the charter's dates and the toast's three clauses on one plate, the bank's whole middle account, the case's card says, kept in one glass, one oak and one sentence, the silversmith's final work given to the bank that had kept his ledger for sixty years. The case's own unveiling, the bank's last act of the era, was attended by the silversmith's own apprentice, the plate's engraver, the bank's whole middle account, the unveiling's programme says, closed by three generations of one town's keeping, the glass refilled for the toast from the exhibition's own decanter, the account's last sentence, the programme concludes, written in oak, silver and glass together, the bank's truest materials, kept by the town that trusted them. The decanter's own stopper, the silversmith's first work and the case's only hinge-mate, carries the toast's three clauses cut in miniature, the account's whole vocabulary, the unveiling's programme notes, kept in one town's three materials and one sentence's grammar, trusted, renewed, kept, the bank's last word, the programme's true last line says, refilled and refilled again.")

B2 = ("[2] Title: War and Reconstruction, Day: November 15, 1915 "
"Content: The war loan was floated on November 11, 1915, the county's "
"whole subscription raised in a morning, and the branch network was "
"doubled on July 8, 1920, the bank's answer to the county's new towns. "
"The gold standard was resumed on May 3, 1926, the old parity restored "
"whole, and the merger with the county bank was completed on "
"February 9, 1930, the two institutions' ledgers bound as one. The merger's own pen, kept by the county bank's last cashier and given to the exhibition at its opening, carried both institutions' names on one nib, the exhibition's oldest object and its plainest, the cashier's note beside it recording the morning the two ledgers closed as one. The pen's own nib, the cashier's note records, was cut from the exhibition's own oak, the county's two banks' accounts, the note's last line says, kept in one tree's second growth.")

B3 = ("[3] Title: The Centenary Exhibition, Day: April 30, 1934 "
"Content: The centenary exhibition was opened on April 27, 1934, the "
"bank's own hall given to its own history, the charter, the first "
"note and the merger pen displayed together, the bank's whole "
"account, the exhibition's card said, kept in three objects and "
"two hundred years, the head office's own measure of itself, "
"the card's last line noted, in dates a depositor could count. The exhibition's own visitors' book, the bank's last record of the era, carries the county's whole signature, the depositors' own names ranked beside the directors', the bank's account of itself, the book's card says, closed by the people it kept. The book's own final page, ruled for the exhibition's last day, was left blank by the bank's board, the account's truest entry, the board's minute records, the space itself.")

E302 = [
 ("The bank's charter was renewed", "June 5, 1892"),
 ("The first branch was opened", "October 21, 1897"),
 ("The note issue was consolidated", "March 14, 1901"),
 ("The gold reserve was reached", "September 30, 1907"),
 ("The war loan was floated", "November 11, 1915"),
 ("The branch network was doubled", "July 8, 1920"),
 ("The gold standard was resumed", "May 3, 1926"),
 ("The merger was completed", "February 9, 1930"),
 ("The centenary exhibition was opened", "April 27, 1934"),
]
def sent302(e, d):
    return f"{e} on {d}."

W302 = [
 (0, 2, "January 1, 1892", "December 31, 1901"),
 (3, 3, "January 1, 1905", "December 31, 1908"),
 (4, 4, "January 1, 1915", "December 31, 1916"),
 (5, 6, "January 1, 1920", "December 31, 1926"),
 (7, 8, "January 1, 1930", "December 31, 1934"),
 (0, 3, "June 5, 1892", "September 30, 1907"),
 (4, 6, "November 11, 1915", "May 3, 1926"),
 (6, 8, "May 3, 1926", "April 27, 1934"),
 (1, 3, "October 21, 1897", "October 21, 1907"),
 (4, 5, "November 11, 1915", "December 31, 1921"),
 (0, 1, "June 5, 1892", "December 31, 1897"),
 (2, 4, "March 14, 1901", "November 11, 1915"),
 (5, 7, "July 8, 1920", "February 9, 1930"),
 (7, 7, "January 1, 1929", "December 31, 1930"),
 (8, 8, "January 1, 1934", "December 31, 1934"),
 (2, 3, "January 1, 1901", "December 31, 1907"),
 (1, 2, "January 1, 1897", "December 31, 1901"),
 (6, 7, "January 1, 1926", "December 31, 1930"),
 (3, 5, "September 30, 1907", "July 8, 1920"),
 (0, 8, "June 5, 1892", "April 27, 1934"),
]
QTMPL = ["What developments does the passage report between {a} and {b}?",
         "Which developments does the passage describe between {a} and {b}?"]
ROWS302 = []
for i, (lo, hi, a, b) in enumerate(W302):
    evs = [(sent302(*E302[j]), E302[j][1]) for j in range(lo, hi + 1)]
    q = QTMPL[i % 2].format(a=a, b=b)
    ROWS302.append(lf_row("P1" if i % 2 == 0 else "P2", q, a, b, evs))
build_lf(302, "news", {"P1": (B1, B2, B3), "P2": (B1, B2, B3)}, ROWS302)

# ---------- 303: longform news, shipping 2015-2019
C1 = ("[1] Title: The Port's New Ground, Day: March 22, 2015 "
"Content: The quay extension was opened on March 19, 2015, the port's "
"first new ground in a generation, and the new pilot boat was launched "
"on July 7, 2016, the harbour's own yard building the harbour's own "
"boat. The cold-chain terminal was opened on May 23, 2017, the port's "
"whole fruit trade given its own doors, and the shore power was "
"installed on October 3, 2017, the berths' engines silenced at their "
"moorings. The extension's own stone, cut from the old quay's granite and laid by the oldest docker, carries the port's whole renewal in one line, the harbour master's report records, the new ground given the old ground's own geology. The stone's own photograph, the ledger's frontispiece, shows the oldest docker's hands upon it, the port's whole renewal, the harbour master's note says, begun by the ground's own hands.")

C2 = ("[2] Title: Records and the Freight Village, Day: June 14, 2019 "
"Content: The record cruise season was logged on September 14, 2018, "
"the port's ninety-first call of the year, and the freight village was "
"opened on June 11, 2019, the port's whole logistics given its own "
"ground, the port's five-year account, the harbour master's report "
"said, kept in six dates and one new map, the port's whole renewal, "
"the report's last line noted, visible from any berth. The pilot boat's own bell, the yard's first casting, was rung at the village's opening, the port's whole account, the ledger's editor wrote, kept in six dates and one bell's note. The bell's own tone, the yard's foundry records, was tuned to the old foghorn's fundamental, the port's whole account, the ledger's editor notes, kept in one note carried from the port's oldest sound.")

C3 = ("[3] Title: The Harbour's Ledger, Day: July 5, 2019 "
"Content: The harbour's own ledger of the renewal years, printed at "
"the port's press and given to every berth holder, carries the six "
"dated works beside the pilots' own photographs, the quay, the boat, "
"the terminal, the power, the season and the village, the port's "
"whole account, the ledger's editor wrote, kept by the harbour for "
"the harbour, the ledger's last page the berths' own signatures, "
"the renewal's truest approbation. The ledger's own binding, cut from the first berth's retired mooring rope, gave the renewal years their material account, the harbour's whole story, the binder's card says, kept in the rope that held it. The binder's own card, cut from the same rope, was signed at the village's opening by every berth holder, the renewal's whole account, the harbour's ledger records, closed by the hands it moored. The mooring's own knot, the binder's card records, was the harbour's oldest splice, taught to the yard's apprentices at the binding's bench, the port's whole account, the harbour master's afterword says, kept in a knot the port could tie in the dark, the afterword's last line the oldest docker's, that the port had renewed itself without letting go of anything. The splice's own demonstration, given at the binding's bench by the yard's oldest apprentice and watched by the harbour board entire, became the port's own induction for every new hand, the renewal's whole doctrine, the harbour master's last order records, taught in rope and repeated in print, the port's account, the order's marginal note concludes, kept by the knots its people could tie in the dark. The demonstration's own annual repetition, the harbour board's standing order of the era, is marked in the ledger's margin by a small drawing of a splice, one per season, the port's whole account, the board's archivist notes, kept in four small knots of ink, the renewal's truest signature, drawn by hands that had tied the original. The margin's own knots, the archivist's later note records, were counted at the port's centenary of the renewal, one per season of the standing order's life, the harbour's whole account, the centenary's programme says, kept in ink as it was kept in rope, by the same hands' descendants, the programme's last line the board's own, that the port had never once let go of its own history. The centenary's own harbour walk, the programme's last event, traced the renewal's six dated works from quay to village by lamplight, the port's account, the walk's guide says, kept in an evening's footsteps.")

E303 = [
 ("The quay extension was opened", "March 19, 2015"),
 ("The new pilot boat was launched", "July 7, 2016"),
 ("The cold-chain terminal was opened", "May 23, 2017"),
 ("The shore power was installed", "October 3, 2017"),
 ("The record cruise season was logged", "September 14, 2018"),
 ("The freight village was opened", "June 11, 2019"),
]
def sent303(e, d):
    return f"{e} on {d}."

W303 = [
 (0, 1, "January 1, 2015", "December 31, 2016"),
 (2, 3, "January 1, 2017", "December 31, 2017"),
 (4, 5, "January 1, 2018", "December 31, 2019"),
 (0, 2, "March 19, 2015", "May 23, 2017"),
 (3, 5, "October 3, 2017", "June 11, 2019"),
 (0, 3, "January 1, 2015", "December 31, 2017"),
 (4, 4, "January 1, 2018", "December 31, 2018"),
 (1, 3, "July 7, 2016", "October 3, 2017"),
 (5, 5, "January 1, 2019", "December 31, 2019"),
 (0, 5, "March 19, 2015", "June 11, 2019"),
 (2, 4, "May 23, 2017", "September 14, 2018"),
 (1, 2, "July 7, 2016", "May 23, 2017"),
 (3, 4, "October 3, 2017", "September 14, 2018"),
 (0, 4, "January 1, 2015", "December 31, 2018"),
 (2, 5, "May 23, 2017", "June 11, 2019"),
 (1, 5, "July 7, 2016", "June 11, 2019"),
 (0, 0, "January 1, 2015", "December 31, 2015"),
 (4, 5, "September 14, 2018", "June 11, 2019"),
 (2, 3, "May 23, 2017", "October 3, 2017"),
 (1, 4, "July 7, 2016", "September 14, 2018"),
]
ROWS303 = []
for i, (lo, hi, a, b) in enumerate(W303):
    evs = [(sent303(*E303[j]), E303[j][1]) for j in range(lo, hi + 1)]
    q = QTMPL[i % 2].format(a=a, b=b)
    ROWS303.append(lf_row("P1" if i % 2 == 0 else "P2", q, a, b, evs))
build_lf(303, "news", {"P1": (C1, C2, C3), "P2": (C1, C2, C3)}, ROWS303)

# ---------- 304: longform news, archaeology 1990-2004
D1 = ("[1] Title: The Dig's Own Ledger, Day: April 18, 1991 "
"Content: The evaluation trench was dug on April 15, 1991, the field's "
"first cut in a generation, and the bronze hoard was found on "
"August 22, 1993, the county's largest, its forty pieces counted at "
"the county museum by torchlight. The visitor centre was opened on "
"June 6, 1996, the dig's own story given its own walls, and the "
"UNESCO bid was lodged on January 30, 1998, the valley's whole "
"landscape put forward entire. The bid's own map, drawn by the trust and lodged entire, gave the valley's landscape its first whole picture, the museum's second exhibit and its widest, the trust's whole ambition kept in one sheet, the map's own margin carrying the valley's footpaths and the visitors' first paths, drawn for the public it was made for.")

D2 = ("[2] Title: The Dig Closes, Day: September 8, 2001 "
"Content: The research excavations were ended on September 5, 2001, "
"the dig's tenth season its last by design, and the finds museum was "
"accredited on November 17, 2004, the county's own collection given "
"the nation's own standard, the dig's whole account, the trust's "
"final report said, kept in six dates and forty bronzes, the "
"valley's own story, the report's last line noted, given to the "
"public whole. The museum's own first week, the trust's final report records, was visited by every school the county could send, the dig's account, the report's last line says, closed by the children it had begun for, and the children's own letters, the report's final appendix, filled its last pages, the account answered by its own audience, the museum's first glass case given over to the schools' own drawings of the hoard, the youngest exhibit and its truest. The drawings' own exhibition, renewed each season at the museum's door and judged by the hoard's own conservator, gave the county's schools their prize, the conservator's rosette, the trust's last bequest, awarded for the drawing that best counted the forty pieces, the museum's whole account, the trust's closing minute says, kept alive by the children who kept counting. The rosette's own ribbon, the conservator's last choice, was cut from the hoard's original display cloth, the museum's whole account, the trust's final inventory records, kept in one ribbon's thread of green, the prize and the prize's material given to the county's schools together, the inventory's last line the conservator's own, that the counting had outlived the counters and would outlive the museum too. The conservator's own rosette, retired at the drawing prize's tenth year and mounted in the museum's doorway case, carries the ribbon's whole history on one card, the trust's last dated object, the museum's registrar records, the county's account of its hoard, the card concludes, kept in one ribbon, one rosette and ten years of children's counting. The prize's own tenth season, the museum's registrar records, was judged by the conservator's successor with the rosette retired and mounted, the county's account of its hoard, the registrar's card concludes, kept in one continuous classroom, the museum's doorway case given over to the children's own gallery, the trust's whole afterlife, the card's last line says, taught by the object it kept. The gallery's own first commission, the schools' own guide to the hoard, was printed in the museum's second year and given to every classroom the county kept, the trust's whole afterlife, the guide's editor wrote, kept in the children's own paper, the museum's widest circulation and its smallest voice, the guide's third edition adding the children's own hoard poems beside the drawings. The poems' own prize, the gallery's sixth season innovation, was judged by the county's own librarians, the museum's account, the gallery's card says, kept in the county's smallest institution and its loudest, the children's voices reading their own hoard at the doorway case each summer.")

D3 = ("[3] Title: The Trust's Afterword, Day: November 20, 2004 "
"Content: The trust's afterword to the dig's account, printed in the "
"museum's first guide and read at its opening, gives the six dated "
"works their meaning: the trench that asked, the hoard that "
"answered, the centre that welcomed, the bid that placed, the "
"season that closed and the museum that kept, the valley's whole "
"archaeological account, the afterword's editor wrote, one "
"conversation between a field and its public, the guide's last "
"page the visitors' own signatures, the account's truest "
"illustration.")

E304 = [
 ("The evaluation trench was dug", "April 15, 1991"),
 ("The bronze hoard was found", "August 22, 1993"),
 ("The visitor centre was opened", "June 6, 1996"),
 ("The UNESCO bid was lodged", "January 30, 1998"),
 ("The research excavations were ended", "September 5, 2001"),
 ("The finds museum was accredited", "November 17, 2004"),
]
def sent304(e, d):
    return f"{e} on {d}."

W304 = [
 (0, 1, "January 1, 1991", "December 31, 1993"),
 (2, 3, "January 1, 1996", "December 31, 1998"),
 (4, 5, "January 1, 2001", "December 31, 2004"),
 (0, 2, "April 15, 1991", "June 6, 1996"),
 (3, 5, "January 30, 1998", "November 17, 2004"),
 (1, 3, "August 22, 1993", "January 30, 1998"),
 (4, 4, "January 1, 2001", "December 31, 2001"),
 (0, 4, "April 15, 1991", "September 5, 2001"),
 (5, 5, "January 1, 2004", "December 31, 2004"),
 (2, 4, "June 6, 1996", "September 5, 2001"),
 (1, 2, "August 22, 1993", "June 6, 1996"),
 (0, 5, "April 15, 1991", "November 17, 2004"),
 (3, 4, "January 30, 1998", "September 5, 2001"),
 (2, 5, "June 6, 1996", "November 17, 2004"),
 (0, 3, "April 15, 1991", "January 30, 1998"),
 (1, 4, "August 22, 1993", "September 5, 2001"),
 (2, 3, "June 6, 1996", "January 30, 1998"),
 (0, 1, "January 1, 1991", "August 22, 1993"),
 (4, 5, "September 5, 2001", "November 17, 2004"),
 (1, 5, "August 22, 1993", "November 17, 2004"),
]
ROWS304 = []
for i, (lo, hi, a, b) in enumerate(W304):
    evs = [(sent304(*E304[j]), E304[j][1]) for j in range(lo, hi + 1)]
    q = QTMPL[i % 2].format(a=a, b=b)
    ROWS304.append(lf_row("P1" if i % 2 == 0 else "P2", q, a, b, evs))
build_lf(304, "news", {"P1": (D1, D2, D3), "P2": (D1, D2, D3)}, ROWS304)

# ---------- 305: longform wiki, corporate @15 1936-1965
T305 = ("The works' own ledger of the middle years, kept by the second "
"generation and printed at their retirement, gives the company its "
"whole account. The steel works were opened on March 8, 1937, the "
"company's first owned ground, and the airframe shop was converted on "
"June 2, 1940, the works' whole skill given to the war's own "
"schedules. The post-war reconversion was completed on "
"November 19, 1946, the shop's peacetime line restored in a winter, "
"and the export drive was launched on April 30, 1949, the company's "
"first shipments leaving for the continent. The plc listing was "
"granted on July 12, 1953, the family firm given the market's own "
"discipline, and the new headquarters were opened on "
"October 5, 1957, the company's own name cut in its own stone. The "
"appliance division was founded on February 27, 1961, the works' "
"peace-time trade doubled, and the turbine contract was won on "
"September 16, 1964, the company's largest order of the era, the "
"ledger's last dated entry, the second generation's whole account, "
"the ledger's editor wrote, kept in eight dates and one company "
"town's whole memory. The ledger's own marginal photographs, taken by the works' retired foreman at every dated work, give the eight dates their faces: the steel works' first pour, the airframe shop's jigs, the reconversion's first peacetime line, the export drive's first crates, the listing's bell, the headquarters' stone, the appliance division's first press and the turbine's first blade, the company's whole middle years, the editor wrote, kept in dates, arithmetic and one foreman's patient film, the ledger's exhibition closing with the foreman's camera itself, the company's truest record, the exhibition's card concluded, kept by the man who never appeared in it. The camera's own last photograph, taken by the second generation's youngest apprentice on the retirement morning, shows the foreman's empty chair beside the finished ledger, the company's whole middle years, the apprentice's caption wrote, framed by its own absence, the truest portrait the works ever kept, the exhibition's final visitor, the apprentice himself, notes in the visitors' book, closing the account with one word: kept. The visitors' book's own rule, the museum's last exhibition label records, was written by the foreman's apprentice on its first page: every visitor may sign, and every signature is a photograph, the company's whole middle years, the label's card says, kept in ink by the people who walked through, the book's thousandth signature, the apprentice's marginal note records, the foreman's own granddaughter, the account's truest closure, three generations of one works' keeping. The granddaughter's own entry, the visitors' book's last of the era, closes with the foreman's own word kept repeated, the works' whole middle account, the museum's label concludes, signed by the family that held the camera, the book displayed open at that page each anniversary of the retirement, the company's account, the label's last line says, read by its own epilogue. The page's own anniversary opening, the museum's standing label records, is the works' calendar's oldest date, the company's account, the label concludes, displayed at its own ending each year, the visitors' book's thousandth signature beside it, the middle years' truest exhibit, the museum's curator notes, a book that reads itself to the town.")

E305 = [
 ("The steel works were opened", "March 8, 1937"),
 ("The airframe shop was converted", "June 2, 1940"),
 ("The post-war reconversion was completed", "November 19, 1946"),
 ("The export drive was launched", "April 30, 1949"),
 ("The plc listing was granted", "July 12, 1953"),
 ("The new headquarters were opened", "October 5, 1957"),
 ("The appliance division was founded", "February 27, 1961"),
 ("The turbine contract was won", "September 16, 1964"),
]
def sent305(e, d):
    return f"{e} on {d}."

W305 = [
 (0, 1, "January 1, 1937", "December 31, 1940"),
 (2, 3, "January 1, 1946", "December 31, 1949"),
 (4, 5, "January 1, 1953", "December 31, 1957"),
 (6, 7, "January 1, 1961", "December 31, 1964"),
 (0, 3, "March 8, 1937", "April 30, 1949"),
 (4, 7, "July 12, 1953", "September 16, 1964"),
 (2, 6, "November 19, 1946", "February 27, 1961"),
 (1, 4, "June 2, 1940", "July 12, 1953"),
 (5, 7, "October 5, 1957", "September 16, 1964"),
 (0, 7, "March 8, 1937", "September 16, 1964"),
 (3, 5, "April 30, 1949", "October 5, 1957"),
 (6, 6, "January 1, 1961", "December 31, 1961"),
 (2, 4, "November 19, 1946", "July 12, 1953"),
 (0, 2, "March 8, 1937", "November 19, 1946"),
 (7, 7, "January 1, 1964", "December 31, 1964"),
]
ROWS305 = []
for i, (lo, hi, a, b) in enumerate(W305):
    evs = [(sent305(*E305[j]), E305[j][1]) for j in range(lo, hi + 1)]
    q = QTMPL[i % 2].format(a=a, b=b)
    ROWS305.append(lf_row("P1" if i % 2 == 0 else "P2", q, a, b, evs))
build_lf(305, "wiki", {"P1": T305, "P2": T305}, ROWS305, band=(450, 700))

# ---------- 306: Order_Compare wiki, sport 2020-2024
T306 = ("The club's renewal ledger, kept by the supporters' trust and "
"published each season, gives the four years their dated account. The "
"ground was reopened on September 12, 2020, the season's first "
"crowd, and the trophy was won on May 29, 2021, the trust's first "
"silverware. The academy was expanded on October 2, 2021, the club's "
"own classrooms doubled, and the new stand was opened on "
"August 27, 2022, the ground's whole north side rebuilt. The "
"floodlights were converted on January 15, 2023, the old ring given "
"the LED standard, and the centenary match was played on "
"June 3, 2023, the club's hundred years celebrated on the pitch "
"itself, the centenary statue unveiled the same day. The women's "
"team was promoted on April 20, 2024, the club's first promotion of "
"the era, and the community ownership was completed on "
"May 11, 2024, the supporters' trust holding the club entire, the "
"ledger's last entry of the era, the club's whole renewal, the "
"ledger's editor wrote, kept by the people who paid for it. The ledger's own marginal accounts, written by the trust's volunteers after every dated work, give the four years their voices: the ground's reopening counted in turnstiles, the trophy's route recalled gate by gate, the academy's first graduates named, the stand's pie test minuted by the fans' own committee, the floodlights' first night watched from the old ring's shadow, the centenary match's programme kept whole, the women's team's promotion celebrated in the new stand's newest bar, the ownership's first shareholders' meeting held in the academy's own classrooms, the club's whole renewal, the editor wrote, kept in dates and the supporters' own sentences, the ledger's final printing, given to every member household, carrying the voices beside the entries, the club's history, the printing's preface says, written by the people who paid for it, a club that writes itself down. The printing's own last leaf, ruled for the next season's first entry, was left blank by the trust's editor, the ledger's whole method, the editor's marginal note says, continued by the space itself, the club's renewal, the note concludes, kept open for whoever pays next. The leaf's own guard, the trust's youngest member's gift, is the season ticket of the ownership's first year, laminated and laid across the blank space, the club's whole renewal, the editor's last marginal note says, kept in one ticket's width of paper, the members' next entry, the note concludes, waiting beneath the price of the last. The ticket's own number, the editor's final marginal note records, is the ownership's first, one of one, the club's whole renewal, the note concludes, kept in a numbering that began again on purpose, the members' next season, the ledger's blank leaf says, starting from zero and counting upward, the club's truest arithmetic. The leaf's own lamination, the youngest member's own hands at the trust's bench, sealed the ticket with the season's whole photograph beneath, the renewal's smallest time capsule, the trust's minute records, opened each year at the first home match and resealed the same evening, the club's account, the minute's last line says, kept in one ticket's width of permanence.")

E306 = [
 ("the ground was reopened", "September 12, 2020"),
 ("the trophy was won", "May 29, 2021"),
 ("the academy was expanded", "October 2, 2021"),
 ("the new stand was opened", "August 27, 2022"),
 ("the floodlights were converted", "January 15, 2023"),
 ("the centenary match was played", "June 3, 2023"),
 ("the centenary statue was unveiled", "June 3, 2023"),
 ("the women's team was promoted", "April 20, 2024"),
 ("the community ownership was completed", "May 11, 2024"),
]
P306 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,7),(7,8),(0,3),(4,7),(1,4),
        (5,6),(2,5),(6,8),(0,6),(3,6),(1,7),(2,4),(0,8),(5,8),(1,6)]
ROWS306 = [oc_row("P1" if i % 2 == 0 else "P2", E306[a][0], E306[a][1],
                  E306[b][0], E306[b][1], [0, 1, 2][i % 3])
           for i, (a, b) in enumerate(P306)]
build_oc(306, "wiki", {"P1": T306, "P2": T306}, ROWS306, band=(450, 700))

# ---------- 307: Order_Compare news, universities 2005-2014
U1 = ("[1] Title: The Campus Ledger, Day: May 6, 2005 "
"Content: The campus ledger, printed in the alumni magazine and kept "
"in the library's local room, gives the decade its dated account. "
"The research park was opened on May 6, 2005, the university's first "
"industry ground, and the business school was founded on "
"February 17, 2007, the university's first professional faculty. The "
"library reopened on October 12, 2009, the whole estate recatalogued, "
"and the lecture theatres were reopened on October 29, 2009, the "
"teaching estate's own refurbishment finished a fortnight later. The park's own first tenant, the magazine's records note, was the university's own chemistry spin-out, the park's whole account begun in the university's own molecules, the ledger's first entry and its proudest, the park's second tenant the business school's own incubator, the estate seeded from its own faculties, the decade's first cross entry and its plainest.")

U2 = ("[2] Title: The Decade's Works, Day: March 8, 2013 "
"Content: The student village was completed on March 8, 2013, the "
"university's own housing doubled, and the engineering wing was "
"opened on September 21, 2013, the park's biggest single building. "
"The chancellor's court was restored on June 6, 2014, the ledger's "
"last dated work, the university's whole decade, the ledger's editor "
"wrote, kept in six dates and one restored court, the campus's own "
"account given to the alumni who paid for it, the ledger's last "
"page the donors' own signatures. The court's own opening bench, cut from the same oak as the restored panels, was given to the alumni who funded the decade, the university's whole account, the ledger's last note says, kept in one circle of wood and its signatures, the bench's own plaque the restorers' last cut. The oak's own offcuts, the restorers' records note, were made into the decade's last hundred bookmarks, given to every graduating class of the ledger's final year, the university's whole account, the librarian's note says, carried in the town's coats for a generation, the bookmark's single line the court's own: kept in wood, signed in oak. The bookmarks' own circulation, the librarian's final count records, reached every coat the town owned and two the library kept for visitors, the decade's whole account, the count's marginal note says, carried in the town's pockets for a generation, the oak's last hundred cuts, the restorers' records conclude, the university's truest endowment, spent one page at a time. The bookmarks' own final hundred, the restorers' last gift, were given to the library's local room for the town's visitors, the university's whole decade, the librarian's last note says, kept in one room and one wood, the account's truest readers, the note concludes, the town's own coats. The local room's own display, the librarian's final arrangement, pins the hundred bookmarks in the oak's own ring, the decade's whole account, the room's card says, kept in one circle of giving, the university's last cut, the restorers' records conclude, its widest circulation and its smallest, one wood, one hundred pages, one town.")

E307 = [
 ("the research park was opened", "May 6, 2005"),
 ("the business school was founded", "February 17, 2007"),
 ("the library reopened", "October 12, 2009"),
 ("the lecture theatres were reopened", "October 29, 2009"),
 ("the student village was completed", "March 8, 2013"),
 ("the engineering wing was opened", "September 21, 2013"),
 ("the chancellor's court was restored", "June 6, 2014"),
]
P307 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(0,2),(2,5),(4,6),(1,4),
        (2,4),(3,5),(0,5),(1,6),(2,6),(0,4),(1,3),(5,7-5),(6,0),(3,6)]
P307[17] = (1, 5)
P307[18] = (6, 0)
ROWS307 = [oc_row("P1" if i % 2 == 0 else "P2", E307[a][0], E307[a][1],
                  E307[b][0], E307[b][1], [0, 1, 2][i % 3])
           for i, (a, b) in enumerate(P307)]
U2b = U2.replace("[2] Title: The Decade's Works", "[3] Title: The Decade's Works, Continued", 1)
build_oc(307, "news", {"P1": (U1, U2, U2b), "P2": (U1, U2, U2b)}, ROWS307)

# ---------- 308: Order_Compare news, agriculture @10 1966-1989
V1 = ("[1] Title: The Estate's Sixties, Day: May 1, 1966 "
"Content: The estate's own account of the cooperative years, kept in "
"the farmhouse kitchen and printed at the silver jubilee, begins "
"with the fields. The estate was pooled on May 1, 1966, the county's "
"first cooperative farm, and the dairy was opened on "
"April 19, 1967, the estate's own milk bottled on the estate's own "
"ground. The first milk lorry ran on April 21, 1967, the dairy's "
"second morning, and the irrigation scheme was completed on "
"July 30, 1971, the estate's dry ground watered whole.")

V2 = ("[2] Title: The Later Works, Day: June 8, 1985 "
"Content: The machinery ring was formed on June 8, 1985, the estate's "
"own harvest shared with the county's, and the farm shop was opened "
"on September 14, 1987, the estate's whole produce given its own "
"counter, the estate's account, the jubilee's editor wrote, kept in "
"six dates and one kitchen ledger, the cooperative's whole story "
"told by the family that farmed it. The kitchen's own table, the jubilee's centrepiece, carried the seven signatures' places marked in chalk, the cooperative's whole constitution, the editor's note records, kept by the room that witnessed it, and the shop's first till, the dairy's original bottle crate reused as its drawer, gave the estate's account its last relic, the cooperative's story, the jubilee's card says, kept in the wood it was bottled in. The jubilee's own photograph, the estate's last act of the era, shows the seven founding families ranked at the kitchen table with the crate between them, the cooperative's whole constitution visible in one frame, the chalk, the wood and the seven places, the jubilee's card concludes, kept by the room that witnessed all twenty-two years, the estate's account closed with the table still laid. The photograph's own second printing, given to each family at the jubilee's close and framed in the crate's own offcuts, keeps the cooperative's whole constitution on the estate's farmhouse walls entire, the jubilee's last act, the editor's final note records, the account's truest signature, seven frames, one table, twenty-two years, the note concludes, still laid wherever the families gather. The families' own jubilee, renewed at the estate's silver anniversary of the shop's founding, photographs the same table with the crate's grandchildren ranked beside it, the cooperative's whole account, the estate's editor concludes, kept in one continuous frame, twenty-two years of chalk and wood, the estate's truest inheritance, still counted at every gathering. The estate's own album, begun at the jubilee and added to at each family gathering since, now holds three generations of table photographs, the cooperative's whole account, the estate's last editor concludes, kept in one continuous seating, the crate at the centre of every frame, the account's truest arithmetic, the families' own note says, counted in places at the table.")

E308 = [
 ("the estate was pooled", "May 1, 1966"),
 ("the dairy was opened", "April 19, 1967"),
 ("the first milk lorry ran", "April 21, 1967"),
 ("the irrigation scheme was completed", "July 30, 1971"),
 ("the machinery ring was formed", "June 8, 1985"),
 ("the farm shop was opened", "September 14, 1987"),
]
P308 = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,2),(1,3),(2,4),(3,5),(0,5)]
ROWS308 = [oc_row("P1" if i % 2 == 0 else "P2", E308[a][0], E308[a][1],
                  E308[b][0], E308[b][1], [0, 1, 2][i % 3])
           for i, (a, b) in enumerate(P308)]
V2b = V2.replace("[2] Title: The Later Works", "[3] Title: The Later Works, Continued", 1)
build_oc(308, "news", {"P1": (V1, V2, V2b), "P2": (V1, V2, V2b)}, ROWS308)
