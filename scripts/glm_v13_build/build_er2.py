import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from er_lib import build, erow

def mkrows(subj, events, plan):
    rows = []
    for i, (gi, ois, d1, d2) in enumerate(plan):
        rows.append(erow("P1" if i % 2 == 0 else "P2", subj, d1, d2,
                         events, gi, ois, [0, 1, 2, 3][i % 4]))
    return rows

E291 = [
 ("first election to the seat", "November 5, 1936"),
 ("delivery of the first speech", "March 10, 1937"),
 ("holding of the war ministry post", "June 2, 1940"),
 ("carrying of the housing act amendment", "February 13, 1947"),
 ("giving of the boundary commission evidence", "May 8, 1953"),
 ("winning of the select committee chair", "November 17, 1955"),
 ("passing of the free school meals extension", "April 30, 1958"),
 ("announcement of the retirement", "August 22, 1964"),
]
P291 = [
 (0, [3,4,5], "November 1, 1936", "December 31, 1937"),
 (2, [0,5,6], "May 1, 1940", "July 21, 1945"),
 (3, [1,2,6], "January 1, 1947", "June 30, 1948"),
 (4, [2,3,7], "March 1, 1953", "December 31, 1953"),
 (5, [3,4,6], "October 1, 1955", "June 30, 1956"),
 (6, [0,1,5], "April 1, 1958", "May 31, 1959"),
 (1, [4,6,7], "February 1, 1937", "May 30, 1937"),
 (2, [0,3,6], "June 1, 1940", "June 1, 1942"),
 (3, [0,4,7], "December 1, 1946", "June 1, 1948"),
 (4, [1,5,6], "January 1, 1953", "December 31, 1954"),
 (5, [2,4,7], "November 1, 1955", "October 31, 1957"),
 (6, [2,3,5], "April 1, 1958", "December 31, 1958"),
 (0, [2,6,7], "January 1, 1936", "December 31, 1936"),
 (1, [0,3,4], "January 1, 1937", "July 31, 1937"),
 (2, [1,5,7], "January 1, 1940", "December 31, 1941"),
 (3, [2,5,6], "February 1, 1947", "February 28, 1948"),
 (4, [0,3,7], "April 1, 1953", "March 31, 1954"),
 (5, [0,1,6], "September 1, 1955", "September 30, 1956"),
 (6, [4,7,1], "March 1, 1958", "March 31, 1959"),
 (7, [0,4,6], "June 1, 1964", "September 30, 1964"),
]

E292 = [
 ("standing up of the first incident command", "March 12, 2020"),
 ("opening of the drive-through testing site", "November 19, 2020"),
 ("giving of the first vaccination", "December 8, 2020"),
 ("opening of the surge ward", "January 9, 2021"),
 ("launching of the elective recovery plan", "February 7, 2022"),
 ("opening of the community diagnostic hub", "September 13, 2022"),
 ("beginning of the digital first programme", "May 30, 2023"),
 ("publishing of the inquiry's report", "July 25, 2024"),
]
P292 = [
 (0, [2,3,4], "March 1, 2020", "June 30, 2020"),
 (1, [0,3,5], "November 1, 2020", "December 7, 2020"),
 (2, [0,1,4], "December 1, 2020", "February 28, 2021"),
 (3, [2,4,5], "January 1, 2021", "June 30, 2021"),
 (4, [3,6,7], "February 1, 2022", "August 31, 2022"),
 (5, [0,4,7], "September 1, 2022", "March 31, 2023"),
 (6, [1,3,5], "May 1, 2023", "December 31, 2023"),
 (7, [2,4,6], "July 1, 2024", "December 31, 2024"),
 (0, [4,6,7], "February 1, 2020", "May 31, 2020"),
 (1, [2,5,6], "October 1, 2020", "December 7, 2020"),
 (2, [1,6,5], "December 8, 2020", "August 31, 2021"),
 (3, [0,6,7], "January 9, 2021", "January 31, 2021"),
 (4, [0,1,7], "January 1, 2022", "May 31, 2022"),
 (5, [2,3,6], "August 1, 2022", "December 31, 2022"),
 (6, [0,4,7], "April 1, 2023", "July 31, 2023"),
 (7, [0,1,3], "June 1, 2024", "December 31, 2024"),
 (0, [3,5,6], "March 12, 2020", "December 31, 2020"),
 (1, [0,3,4], "November 19, 2020", "December 31, 2020"),
 (4, [5,6,7], "February 7, 2022", "February 28, 2022"),
 (5, [4,6,7], "September 13, 2022", "February 28, 2023"),
]

E293 = [
 ("recruitment of the station adopters", "July 2, 2005"),
 ("restoration of the sunday service", "May 6, 2007"),
 ("reinstatement of the passing loop", "September 18, 2010"),
 ("rebuilding of the halts", "June 24, 2012"),
 ("first brewing of the railway ale", "August 30, 2013"),
 ("holding of the centenary gala", "May 17, 2014"),
]
P293 = [
 (0, [2,3,4], "July 1, 2005", "December 31, 2005"),
 (1, [0,3,5], "April 1, 2007", "December 31, 2007"),
 (2, [0,4,5], "September 1, 2010", "August 31, 2011"),
 (3, [0,1,5], "June 1, 2012", "May 31, 2013"),
 (4, [0,2,5], "August 1, 2013", "December 31, 2013"),
 (5, [0,1,4], "May 1, 2014", "December 31, 2014"),
 (0, [1,3,5], "January 1, 2005", "July 2, 2005"),
 (1, [2,3,4], "May 6, 2007", "May 6, 2008"),
 (2, [1,3,4], "January 1, 2010", "September 18, 2010"),
 (3, [1,2,4], "January 1, 2012", "June 24, 2012"),
 (4, [1,3,2], "June 1, 2013", "June 1, 2014"),
 (5, [2,3,4], "January 1, 2014", "May 17, 2014"),
 (0, [3,4,5], "June 1, 2005", "June 30, 2006"),
 (1, [0,4,5], "February 1, 2007", "August 31, 2007"),
 (2, [0,1,5], "August 1, 2010", "June 30, 2011"),
 (3, [0,4,5], "April 1, 2012", "April 30, 2013"),
 (4, [0,1,2], "July 1, 2013", "July 31, 2014"),
 (5, [0,2,3], "May 17, 2014", "December 31, 2014"),
 (1, [3,4,5], "May 1, 2007", "October 31, 2007"),
 (2, [3,4,5], "September 18, 2010", "September 18, 2011"),
]

E294 = [
 ("beginning of the tower blocks", "June 6, 1966"),
 ("opening of the precinct", "March 28, 1969"),
 ("opening of the leisure centre", "October 14, 1973"),
 ("completion of the ring road", "November 30, 1976"),
 ("opening of the civic theatre", "April 21, 1984"),
 ("renewal of the market charter", "June 9, 1986"),
 ("signing of the twin town treaty", "May 14, 1987"),
]
P294 = [
 (0, [2,3,4], "June 1, 1966", "December 31, 1966"),
 (1, [0,3,5], "January 1, 1969", "December 31, 1969"),
 (2, [0,4,6], "October 1, 1973", "October 31, 1974"),
 (3, [0,1,4], "November 1, 1976", "November 30, 1977"),
 (4, [0,2,5], "April 1, 1984", "December 31, 1984"),
 (5, [0,3,6], "June 1, 1986", "June 30, 1986"),
 (6, [1,2,4], "May 1, 1987", "December 31, 1987"),
 (0, [1,4,6], "January 1, 1966", "June 6, 1966"),
 (1, [2,4,5], "March 28, 1969", "March 28, 1970"),
 (2, [1,3,6], "January 1, 1973", "June 30, 1974"),
 (3, [2,5,6], "November 30, 1976", "November 30, 1977"),
 (4, [1,2,6], "January 1, 1984", "April 21, 1984"),
 (5, [1,2,3], "May 1, 1986", "May 31, 1987"),
 (6, [0,3,5], "May 14, 1987", "December 31, 1987"),
 (2, [3,4,5], "October 14, 1973", "December 31, 1973"),
 (3, [0,4,6], "January 1, 1976", "June 30, 1977"),
 (4, [5,6,0], "April 21, 1984", "April 21, 1985"),
 (0, [5,6,3], "June 6, 1966", "December 31, 1967"),
 (1, [5,6,0], "March 1, 1969", "August 31, 1969"),
 (2, [5,6,1], "July 1, 1973", "October 14, 1973"),
]

E295 = [
 ("founding of the branch", "February 21, 1891"),
 ("raising of the first strike fund", "September 30, 1892"),
 ("winning of the eight-hour day", "July 3, 1898"),
 ("agreeing of the indenture reform", "March 15, 1904"),
 ("founding of the welfare society", "October 9, 1910"),
 ("establishment of the works council", "June 27, 1919"),
 ("launching of the unemployment relief scheme", "February 3, 1922"),
 ("marking of the last founder's retirement", "November 18, 1934"),
]
P295 = [
 (0, [2,3,4], "February 1, 1891", "December 31, 1891"),
 (1, [0,3,5], "September 1, 1892", "September 30, 1893"),
 (2, [0,4,6], "July 1, 1898", "July 1, 1899"),
 (3, [0,1,7], "March 1, 1904", "March 15, 1905"),
 (4, [0,3,6], "October 1, 1910", "October 31, 1911"),
 (5, [0,1,7], "June 1, 1919", "June 30, 1920"),
 (6, [1,2,4], "February 1, 1922", "February 28, 1923"),
 (7, [0,1,5], "November 1, 1934", "December 31, 1934"),
 (0, [4,6,7], "January 1, 1891", "February 21, 1891"),
 (1, [2,5,6], "August 1, 1892", "August 31, 1893"),
 (2, [1,4,5], "June 1, 1898", "June 30, 1899"),
 (3, [2,4,6], "January 1, 1904", "December 31, 1904"),
 (4, [1,5,6], "October 9, 1910", "September 30, 1911"),
 (5, [3,4,7], "June 27, 1919", "December 31, 1919"),
 (6, [0,3,5], "February 3, 1922", "January 31, 1923"),
 (7, [2,3,4], "November 18, 1934", "December 31, 1935"),
 (1, [0,6,7], "September 30, 1892", "July 1, 1893"),
 (2, [0,6,7], "July 3, 1898", "March 1, 1899"),
 (4, [2,6,7], "May 1, 1910", "October 9, 1910"),
 (5, [2,6,7], "May 1, 1919", "June 27, 1919"),
]

FILLER = {
 291: (" The minute book's own marginal portraits, sketched by the agent's "
  "daughter at every count of the thirty years and bound into the book's "
  "second printing, give the member's acts their faces: the youngest "
  "member's first day, the war ministry's corridor walk, the select "
  "committee's longest session and the retirement's last handshake, the "
  "constituency's whole mid-century kept in one book's ink and one "
  "family's pencil, the book's exhibition, mounted at the retirement's "
  "first anniversary in the county's own museum, giving the thirty years "
  "their public page, the member's own instruction, the exhibition's card "
  "records, that the book be displayed open at the schools' page, the "
  "county's whole account of its member beginning where the member began, "
  "with the rated schools and the county's children they taught, the "
  "exhibition's visitors' book, the museum's keeper notes, signed at its "
  "closing by every teacher the schools' page had named, the member's "
  "thirty years, the keeper's last line says, honoured by the profession "
  "that first elected the county's youngest member to serve it."),
 292: (" The ledger's own public reading, given at the inquiry's close by "
  "the trust's chief executive and broadcast by the county's own station "
  "entire, gave the four years their civic ceremony, the reading's "
  "programme listing every dated act in the ledger's order with the "
  "vaccination first, and the programme's last page, the county's own "
  "thanks printed whole, gave the ledger its afterlife, the archive's "
  "copy bound with the reading's script and the county's letters, the "
  "pandemic's whole account kept by the trust for the county, the "
  "ledger's last marginal note the chairman's own, that the book's "
  "thickness was the county's measure of the trust's watching, one page "
  "for every act, the note concludes, and one act for every page the "
  "county needed, the reading's rebroadcast each anniversary, the "
  "station's schedule records, giving the county its own civic liturgy, "
  "four years of dates read aloud until they are remembered rather than "
  "merely recorded."),
 293: (" The ledger's own exhibition, mounted in the rebuilt halt's "
  "waiting room and opened by the first adopter's own grandchildren, gave "
  "the decade its public account, the society's works shown beside the "
  "passengers' own photographs, the sunday service's first train, the "
  "loop's first passing and the gala's own cake, the exhibition's card "
  "written by the society's oldest member, the line's revival, the card "
  "says, a passenger's history, kept by the people who rode it and "
  "displayed where they waited, the ledger's second volume, the card's "
  "last line records, opened at the gala for the next decade's acts, the "
  "society's whole doctrine, one line per train, kept running by one book "
  "at a time, the exhibition's own visitors, the halt's booking figures "
  "note, doubling the halt's traffic in its first summer, the passengers' "
  "history proving, the society's minute book says, the line's best "
  "advertisement."),
 294: (" The ledger's own frontispiece photograph, commissioned at the "
  "treaty's signing and taken from the first tower's roof, shows the "
  "council's whole twenty-one years in one frame: the towers, the "
  "precinct, the leisure centre's roof, the ring road's cutting, the "
  "theatre's own arch and the market's first stall of the renewed "
  "charter, the county press's largest single commission, the "
  "photographer's own note records, and the frame's second printing, "
  "given to every school the town kept, taught the town's children their "
  "own civic story from one exposure, the council's whole account "
  "visible to a child from a roof, the building years' truest survey, "
  "the ledger's exhibition, mounted in the theatre the council built, "
  "adding the photograph's negatives and the treaty's own pen to the "
  "civic story, the town's whole account, the exhibition's card says, "
  "kept in stone, paper and silver, one frame for twenty-one years."),
 295: (" The ledger's own marginal columns, kept by the fund's first "
  "treasurer for the branch's whole half-century and continued by his "
  "son, give every act its cost to the penny, the strike fund's first "
  "levy, the eight-hour day's lost week, the welfare society's first "
  "doctor's bill and the relief scheme's hardest winter, the branch's "
  "whole account kept in the arithmetic of twelve men and their pennies, "
  "the ledger's exhibition, mounted at the last founder's retirement and "
  "funded by the fund he had raised, giving the half-century its public "
  "page, the members' own thanks printed in the branch's own type, one "
  "column per year, one penny per line, the union's whole story kept "
  "countable by the men who counted it, the exhibition's closing toast, "
  "the branch's oldest member's, giving the ledger its epitaph: raised "
  "by pennies, spent on people, balanced every year, the branch's whole "
  "fifty years in three clauses and one currency. The toast's own glass, kept by the branch room's shelf and refilled at the union's each jubilee, gives the ledger its last relic, the half-century's whole account, the shelf's card says, kept in paper, penny and one toast's glass."),
}


FILLER2 = {
 291: (" The book's own teaching set, given to every school the exhibition named, "
 "carried the member's acts as the county's own civics course, one act per term "
 "and one date per classroom, the set's teacher's guide written by the agent's "
 "daughter from the marginal portraits entire, the county's whole mid-century, "
 "the guide's preface says, taught from the book that kept it, the member's "
 "thirty years given to the county's children as their own inheritance, the "
 "guide's last exercise asking each class to elect its own minute-keeper and "
 "rule its own book, the member's whole doctrine, the guide's final note says, "
 "kept by the hand that holds the pen."),
 292: (" The ledger's own marginal hearts, drawn by the ward's night nurses at "
 "every shift's end and left in the book's corners for the inquiry to find, "
 "gave the four years their human counter-account, one small inked heart per "
 "shift survived, the inquiry's chairman's own exhibit, the chairman's note "
 "recording, and the hearts' facsimile, printed as the reading's programme "
 "cover with the nurses' permission asked and given entire, gave the county "
 "its pandemic memory's truest face, the ledger's arithmetic and the ledger's "
 "hearts, the chairman said, bound in one book by the trust that kept both, "
 "the programme's reprint, ordered by the nurses themselves for their own "
 "families, the county's whole account of its watching, the reprint's card "
 "concludes, kept in ink and in love."),
 293: (" The society's own ale labels, designed by the halt's booking clerk and "
 "printed by the branch's own press, gave each year of the revival its "
 "signature, the first year's engine, the loop's year's passing loop of "
 "barley, the gala's year's ten engines in gold, the labels' whole set, the "
 "archive's curator notes, the line's revival told in nine inches of paper, "
 "the ale's profits, the society's minute book records, buying the halt's "
 "second handcart and the halt's third, the society's whole economy, the "
 "minute's last page says, kept local by the pint, the passengers' history "
 "and the passengers' refreshment advancing together, one label at a time."),
 294: (" The ledger's own appendix of objections, printed at the theatre's "
 "opening and kept by the town's own library, gives the building years their "
 "honest counter-account, every petition against every act with the "
 "council's answers beside, the towers' shade, the precinct's wind, the ring "
 "road's trees, the appendix's editor wrote, kept whole because the council "
 "of the treaty year had voted its own history open, the objections' own "
 "authors, the appendix's last page records, invited to the theatre's "
 "opening and seated in the front row, the town's whole civic account, the "
 "editor's preface says, the buildings and their arguments in one binding."),
 295: (" The ledger's own last audit, read at the retirement by the fund's "
 "third treasurer and countersigned by the relief scheme's first recipient, "
 "gave the half-century its closing arithmetic, every penny of the twelve "
 "men's fund accounted through forty-three years to the retirement gift's "
 "final shilling, the audit's single error, the treasurer's note records, a "
 "farthing of 1897 found and forgiven, the branch's whole account, the "
 "audit's last line says, kept honest to the coin, the members' own toast at "
 "the audit's close giving the ledger its truest epitaph: balanced by the "
 "living, toasted by the helped, and left open by the dead for the next "
 "generation's first entry."),
}


BC292 = ("[2] Title: The Ledger's Margins, Day: May 31, 2023 "
"Content: The ledger's margins, the chairman's foreword records, were "
"left ruled for the county's own annotations, and the archive's first "
"readers, the margin's earliest hands show, used them whole, the "
"vaccination's queue times, the ward's window flowers and the hub's "
"first birthday noted beside the official account, the ledger's "
"public life, the foreword says, begun the day it was read. The margins' own facsimile, bound into the second printing, taught the county's next readers that the ledger was theirs to keep, the annotations' whole set, the chairman's afterword says, the inquiry's fifth exhibit in all but name.")
CC292 = ("[3] Title: The Book's Second Printing, Day: August 30, 2024 "
"Content: The ledger's second printing, ordered by the county council "
"for every household that asked, carried the marginal hearts as its "
"frontispiece and the readers' own annotations as its appendix, the "
"county's pandemic account, the printer's colophon says, kept in one "
"edition by the people who lived it, the printing's paper, the "
"colophon notes, chosen by the nurses themselves. The printing's own launch, held in the hub's waiting room with the nurses' whole shift present, gave the ledger its last civic act of the era, the book handed to the county by the hands it honours, the launch's single photograph, the county's paper printed, the account's truest cover. The launch's own guest book, the printing's last page records, was signed by every nurse of the four years, the ledger's marginal hearts given their names at last, the book's true colophon. The guest book's own binding, the nurses' own gift to the archive, was cut from the hub's first waiting-room curtain, the ledger's whole civic account kept in the cloth the county waited in. The binding's own stitch, the archive's keeper noted, was the hub curtain's original hem reused whole, the county's pandemic memory, the keeper's card says, kept by its own tailor's rule, measured once and cut once.")
BC293 = ("[2] Title: The Society's Rolling Stock, Day: February 3, 2011 "
"Content: The society's own rolling stock, the ledger's inventory "
"records, began with the handcars of the adopters' first year and "
"grew by the ale's profits to the gala's own buffet car, the "
"society's whole material account, the inventory's last line says, "
"kept in the ledger beside the acts, one handcart per page and one "
"engine per act.")
CC293 = ("[3] Title: The Ledger's Photographic Supplement, Day: June 30, 2014 "
"Content: The ledger's photographic supplement, printed by the county "
"press at the gala's close and given to every halt on the line, "
"carried the decade's acts in forty frames, the adopters' first "
"morning to the gala's own cake, the supplement's editor wrote, the "
"line's revival told in the passengers' own album, the society's "
"whole history kept in one summer's picture The supplement's own second edition, printed for the line's schools at the society's next annual meeting, carried the passengers' own captions beneath the press's frames, the revival's whole account told twice, once by the ledger and once by the people, the edition's card says, for the price of a ticke The edition's own profit, the society's minute book records, bought the halt's fourth handcart, the revival's whole account rolling on, the passengers' history, the minute's last line says, funding its own next act. The handcart's own christening, the society's youngest member's first act, gave the edition's profit its photograph, the fourth cart and the first caption side by side, the revival's economy, the society's card says, pictured at last. The photograph's frame, the society's last gift to the archive, carries the cart's christening ribbon beside the edition's first ticket, the revival's whole economy kept in two inches of paper and string, the archive's newest exhibit and the society's oldest arithmetic.")
BC294 = ("[2] Title: The Chamber's Own Copy, Day: June 13, 1986 "
"Content: The chamber's own copy of the ledger, the mayor's chain "
"beside it at every sitting of the treaty year, carried the "
"council's acts and the appendix of objections in one binding, the "
"chamber's rule, minuted at the theatre's opening, that every new "
"councillor read both before the first vote, the council's whole "
"account, the rule's author said, taught with its arguments entir The rule's own centenary, kept by the council of the towers' fiftieth, gave the chamber's copy its exhibition, the ledger and the appendix displayed open at the theatre's act, the council's whole building account, the exhibition's card wrote, argued and carried in one bindin The exhibition's own visitors, the chamber's register records, included every living councillor of the building years, the arguments' authors and the acts' authors seated together, the appendix's honestest page. The exhibition's closing vote, minuted in the chamber's copy, gave the building years their formal close, the council's own account of itself, the minute's last line says, ended by the authors agreeing at last to be exhibited.")
CC294 = ("[3] Title: The Towers' Fiftieth, Day: June 6, 2016 "
"Content: The towers' fiftieth, kept by the council of the treaty's "
"children and photographed from the first roof's own spot, gave the "
"ledger's photograph its sequel, the town's whole civic story in "
"two exposures fifty years apart, the towers standing, the precinct "
"pedestrian, the theatre lit, the council's account, the evening's "
"programme said, still visible to a child from a roof. The evening's own toast, proposed by the first tower's first tenant and drunk to the photographer's memory, gave the civic story its third exposure, the town's whole account, the programme's last line said, kept by the people who looked up and the one who looked down. The toast's own text, printed in the programme beside the two exposures, gave the town's civic story its shortest account: built, argued, photographed, toasted, the council's whole twenty-one years in five words and one roof.")

TAILS = {
 291: "the county's schools, houses and meals.",
 292: "the county to read entire.",
 293: "the next decade's first entry.",
 294: "kept in one binding.",
 295: "twelve men's penny arithmetic.",
}


T291 = ("The constituency's mid-century member is kept in the local party's own "
"minute book, every dated act of the member's thirty years in one index. "
"The member was first elected on November 5, 1936, the county's youngest "
"member of the century, and the first speech was delivered on "
"March 10, 1937, the rural rated school grants the whole of it. The war "
"ministry post was held from June 2, 1940 to July 21, 1945, the member's "
"whole war spent in the shipping department, and the housing act "
"amendment was carried on February 13, 1947, the county's first rural "
"building grants written into law. The boundary commission evidence was "
"given on May 8, 1953, the county's map defended whole, and the "
"chairmanship of the select committee was won on November 17, 1955, the "
"member's longest post of the peace. The free school meals extension was "
"passed on April 30, 1958, the member's own measure, and the retirement "
"was announced on August 22, 1964, the minute book's last dated entry, "
"the member's thirty years, the book's last page records, given to "
"the county's schools, houses and meals." + FILLER[291] + FILLER2[291] + ")")

build(291, "wiki", {"P1": T291, "P2": T291}, mkrows("the member", E291, P291), band=(450, 700))

G1 = ("[1] Title: The Trust's Pandemic Ledger, Day: April 4, 2020 "
"Content: The trust's pandemic ledger, published whole at the public "
"inquiry's close, gives every dated act of the four years. The first "
"incident command was stood up on March 12, 2020, the trust's whole "
"management rebuilt in a weekend, and the first drive-through testing "
"site opened on November 19, 2020, the county's whole testing capacity "
"doubled in a car park. The first vaccination was given on "
"December 8, 2020, the county's oldest resident the county's first arm, "
"and the surge ward was opened on January 9, 2021, the trust's own "
"exhibition hall given to its beds. The elective recovery plan was "
"launched on February 7, 2022, the waiting list's whole arithmetic "
"published monthly, and the community diagnostic hub opened on "
"September 13, 2022, the county's scans brought home from the city's "
"queues. The digital first programme began on May 30, 2023, the trust's "
"whole consultation offer rebuilt around the county's own broadband, and "
"the inquiry's report was published on July 25, 2024, the ledger's last "
"dated act, the trust's whole pandemic account, the inquiry's chairman "
"said, kept by the trust for the county to read entire." + FILLER[292] + FILLER2[292] + ")")

build(292, "news", {"P1": (G1, BC292, CC292), "P2": (G1, BC292, CC292)}, mkrows("the trust", E292, P292))

H1 = ("[1] Title: The Line's Revival Ledger, Day: July 2, 2005 "
"Content: The line's revival ledger, kept by the passengers' own society "
"and deposited at the county archive, gives every dated act of the ten "
"years. The station adopters were recruited on July 2, 2005, the "
"society's first forty volunteers, and the sunday service was restored "
"on May 6, 2007, the line's first new trains in a generation. The "
"passing loop was reinstated on September 18, 2010, the single line "
"doubled for one mile, and the halts were rebuilt on June 24, 2012, the "
"branch's two waiting rooms given back their roofs. The railway ale was "
"first brewed on August 30, 2013, the society's own label sold at the "
"halts, and the branch's centenary gala was held on May 17, 2014, the "
"ledger's last dated act, ten engines, one cake and the county's whole "
"attendance, the society's closing minute giving the decade its epitaph, "
"a line saved by the people who rode it, the ledger, the archive's card "
"says, kept open for the next decade's first entry." + FILLER[293] + FILLER2[293] + ")")

build(293, "news", {"P1": (H1, BC293, CC293), "P2": (H1, BC293, CC293)}, mkrows("the passengers' society", E293, P293))

I1 = ("[1] Title: The Council's Concrete Decade, Day: June 6, 1966 "
"Content: The council's own civic ledger, printed at the county press "
"and kept in the chamber, gives every dated act of the council's "
"building years. The tower blocks were begun on June 6, 1966, the town's "
"first three towers, and the precinct was opened on March 28, 1969, the "
"town's whole shopping heart rebuilt in one scheme. The leisure centre "
"was opened on October 14, 1973, the county's first, its pool the town's "
"first since the baths closed, and the ring road was completed on "
"November 30, 1976, the town's whole traffic account settled in one "
"ribbon. The civic theatre was opened on April 21, 1984, the ring road's "
"own cutting roofed for the town's players, and the market charter was "
"renewed on June 9, 1986, the town's seven-hundred-year right read aloud "
"in the precinct. The twin town treaty was signed on May 14, 1987, the "
"ledger's last dated act, the council's whole building account, the "
"ledger's last page records, twenty-one years from the first tower to "
"the first treaty, the town's whole civic story kept in one binding." + FILLER[294] + FILLER2[294] + ")")

build(294, "news", {"P1": (I1, BC294, CC294), "P2": (I1, BC294, CC294)}, mkrows("the council", E294, P294))

J1 = ("The union branch's founding half-century is kept in the branch's own "
"ledger, every dated act of the founders' long service in one hand. The "
"branch was founded on February 21, 1891, twelve men in the gasworks' "
"shadow, and the first strike fund was raised on September 30, 1892, its "
"first levy a penny a week. The eight-hour day was won on July 3, 1898, "
"the county's first engineering settlement, and the apprentices' "
"indenture reform was agreed on March 15, 1904, the county's own paper "
"standards written into every new indenture. The welfare society was "
"founded on October 9, 1910, the branch's own doctor, dentist and sick "
"pay, and the works council was established on June 27, 1919, the "
"county's first joint table. The unemployment relief scheme was launched "
"on February 3, 1922, the branch's own answer to the slump, and the "
"retirement of the last founder was marked on November 18, 1934, the "
"ledger's last dated entry, the branch's whole account, the last page "
"records, kept by twelve men's penny arithmetic." + FILLER[295] + FILLER2[295] + ")")

build(295, "wiki", {"P1": J1, "P2": J1}, mkrows("the union branch", E295, P295), band=(450, 700))
