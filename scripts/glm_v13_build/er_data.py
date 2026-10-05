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
 (6, [2,3,5], "May 1, 1958", "December 31, 1958"),
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
 (2, [1,3,5], "December 8, 2020", "August 31, 2021"),
 (3, [0,6,7], "January 9, 2021", "January 31, 2021"),
 (4, [0,1,7], "January 1, 2022", "May 31, 2022"),
 (5, [2,3,6], "August 1, 2022", "December 31, 2022"),
 (6, [0,4,7], "April 1, 2023", "July 31, 2023"),
 (7, [0,1,3], "June 1, 2024", "December 31, 2024"),
 (0, [1,5,6], "March 12, 2020", "December 31, 2020"),
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
 (4, [1,3,5], "June 1, 2013", "June 1, 2014"),
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
  "fifty years in three clauses and one currency."),
}

TAILS = {
 291: "the county's schools, houses and meals.",
 292: "the county to read entire.",
 293: "the next decade's first entry.",
 294: "kept in one binding.",
 295: "twelve men's penny arithmetic.",
}

build(291, "wiki", {"P1": """T291_PLACEHOLDER""", "P2": """T291_PLACEHOLDER"""}, None, band=(450, 700))
