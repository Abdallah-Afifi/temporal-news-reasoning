import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from er_lib import build, erow

def mkrows(subj, events, plan):
    rows = []
    for i, (gi, ois, d1, d2) in enumerate(plan):
        rows.append(erow("P1" if i % 2 == 0 else "P2", subj, d1, d2,
                         events, gi, ois, [0, 1, 2, 3][i % 4]))
    return rows

# =================== 291: ER wiki, politics 1936-1965
T291 = ("The constituency's mid-century member is kept in the local "
"party's own minute book, every dated act of the member's thirty years "
"in one index. The member was first elected on November 5, 1936, the "
"county's youngest member of the century, and the first speech was "
"delivered on March 10, 1937, the rural rated school grants the whole "
"of it. The war ministry post was held from June 2, 1940 to "
"July 21, 1945, the member's whole war spent in the shipping "
"department, and the housing act amendment was carried on "
"February 13, 1947, the county's first rural building grants written "
"into law. The boundary commission evidence was given on "
"May 8, 1953, the county's map defended whole, and the "
"chairmanship of the select committee was won on "
"November 17, 1955, the member's longest post of the peace. The "
"free school meals extension was passed on April 30, 1958, the "
"member's own measure, and the retirement was announced on "
"August 22, 1964, the minute book's last dated entry, the member's "
"thirty years, the book's last page records, given to the county's "
"schools, houses and meals.")

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
build(291, "wiki", {"P1": T291, "P2": T291}, mkrows("the member", E291, P291), band=(450, 700))

# =================== 292: ER news, public health 2020-2024
G1 = ("[1] Title: The Trust's Pandemic Ledger, Day: April 4, 2020 "
"Content: The trust's pandemic ledger, published whole at the "
"public inquiry's close, gives every dated act of the four "
"years. The first incident command was stood up on "
"March 12, 2020, the trust's whole management rebuilt in a "
"weekend, and the first vaccination was given on "
"December 8, 2020, the county's oldest resident the county's "
"first arm. The surge ward was opened on January 9, 2021, "
"the trust's own exhibition hall given to its beds, and the "
"first drive-through testing site opened on "
"November 19, 2020, the county's whole testing capacity "
"doubled in a car park.")

G2 = ("[2] Title: The Recovery Years, Day: June 5, 2023 "
"Content: The recovery acts filled the ledger's later pages. "
"The elective recovery plan was launched on "
"February 7, 2022, the waiting list's whole arithmetic "
"published monthly, and the community diagnostic hub opened "
"on September 13, 2022, the county's scans brought home from "
"the city's queues. The digital first programme began on "
"May 30, 2023, the trust's whole consultation offer rebuilt "
"around the county's own broadband, and the inquiry's report "
"was published on July 25, 2024, the ledger's last dated "
"act, the trust's whole pandemic account, the inquiry's "
"chairman said, kept by the trust for the county to read "
"entire. The ledger's own public reading, given at the inquiry's close by the trust's chief executive and broadcast by the county's own station entire, gave the four years their civic ceremony, the reading's programme listing every dated act in the ledger's order with the vaccination first, and the programme's last page, the county's own thanks printed whole, gave the ledger its afterlife, the archive's copy bound with the reading's script and the county's letters, the pandemic's whole account kept by the trust for the county, the ledger's last marginal note the chairman's own, that the book's thickness was the county's measure of the trust's watching, one page for every act, the note concludes, and one act for every page the county needed.")
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
build(292, "news", {"P1": (G1, G1, G2), "P2": (G1, G1, G2)}, mkrows("the trust", E292, P292))

# =================== 293: ER news, transport 2005-2014
H1 = ("[1] Title: The Line's Revival Ledger, Day: July 2, 2005 "
"Content: The line's revival ledger, kept by the passengers' own "
"society and deposited at the county archive, gives every dated "
"act of the ten years. The station adopters were recruited on "
"July 2, 2005, the society's first forty volunteers, and the "
"sunday service was restored on May 6, 2007, the line's first "
"new trains in a generation. The passing loop was reinstated on "
"September 18, 2010, the single line doubled for one mile, and "
"the halts were rebuilt on June 24, 2012, the branch's two "
"waiting rooms given back their roofs.")

H2 = ("[2] Title: The Society's Decade, Day: May 17, 2014 "
"Content: The ledger's later pages carry the society's own "
"works. The railway ale was first brewed on "
"August 30, 2013, the society's own label sold at the halts, "
"and the branch's centenary gala was held on "
"May 17, 2014, the ledger's last dated act, ten engines, one "
"cake and the county's whole attendance, the society's "
"closing minute giving the decade its epitaph: a line saved "
"by the people who rode it, the ledger, the archive's card "
"says, kept open for the next decade's first entry.
 The ledger's own exhibition, mounted in the rebuilt halt's waiting room and
opened by the first adopter's own grandchildren, gave the decade its public
account, the society's three works shown beside the passengers' own
photographs, the sunday service's first train, the loop's first passing and
the gala's own cake, the exhibition's card written by the society's oldest
member, the line's revival, the card says, a passenger's history, kept by the
people who rode it and displayed where they waited, the ledger's second
volume, the card's last line records, opened at the gala for the next
decade's acts, the society's whole doctrine, one line per train, kept
running by one book at a time.")

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
build(293, "news", {"P1": (H1, H2, H2), "P2": (H1, H2, H2)}, mkrows("the passengers' society", E293, P293))

# =================== 294: ER news, municipal 1966-1989
I1 = ("[1] Title: The Council's Concrete Decade, Day: June 6, 1966 "
"Content: The council's own civic ledger, printed at the county "
"press and kept in the chamber, gives every dated act of the "
"council's building years. The tower blocks were begun on "
"June 6, 1966, the town's first three towers, and the precinct "
"was opened on March 28, 1969, the town's whole shopping heart "
"rebuilt in one scheme. The leisure centre was opened on "
"October 14, 1973, the county's first, its pool the town's "
"first since the baths closed, and the ring road was completed "
"on November 30, 1976, the town's whole traffic account "
"settled in one ribbon.")

I2 = ("[2] Title: The Ledger's Later Acts, Day: April 21, 1984 "
"Content: The ledger's later pages carry the council's softer "
"works. The civic theatre was opened on "
"April 21, 1984, the ring road's own cutting roofed for the "
"town's players, and the market charter was renewed on "
"June 9, 1986, the town's seven-hundred-year right read aloud "
"in the precinct. The twin town treaty was signed on "
"May 14, 1987, the ledger's last dated act, the council's "
"whole building account, the ledger's last page records, "
"twenty-one years from the first tower to the first treaty, "
"the town's whole civic story kept in one binding.
 The ledger's own frontispiece photograph, commissioned at the treaty's
signing and taken from the first tower's roof, shows the council's whole
twenty-one years in one frame: the towers, the precinct, the leisure
centre's roof, the ring road's cutting, the theatre's own arch and the
market's first stall of the renewed charter, the county press's largest
single commission, the photographer's own note records, and the frame's
second printing, given to every school the town kept, taught the town's
children their own civic story from one exposure, the council's whole
account, the ledger's last marginal note says, visible to a child from a
roof, the building years' truest survey.")

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
build(294, "news", {"P1": (I1, I2, I2), "P2": (I1, I2, I2)}, mkrows("the council", E294, P294))

# =================== 295: ER wiki, labour 1890-1935
T295 = ("The union branch's founding half-century is kept in the "
"branch's own ledger, every dated act of the founders' long "
"service in one hand. The branch was founded on "
"February 21, 1891, twelve men in the gasworks' shadow, and "
"the first strike fund was raised on "
"September 30, 1892, its first levy a penny a week. The "
"eight-hour day was won on July 3, 1898, the county's first "
"engineering settlement, and the apprentices' indenture reform "
"was agreed on March 15, 1904, the county's own paper "
"standards written into every new indenture. The welfare "
"society was founded on October 9, 1910, the branch's own "
"doctor, dentist and sick pay, and the works council was "
"established on June 27, 1919, the county's first joint "
"table. The unemployment relief scheme was launched on "
"February 3, 1922, the branch's own answer to the slump, and "
"the retirement of the last founder was marked on "
"November 18, 1934, the ledger's last dated entry, the "
"branch's whole account, the last page records, kept by "
"twelve men's penny arithmetic.
 The ledger's own marginal columns, kept by the fund's first treasurer for
the branch's whole half-century and continued by his son, give every act its
cost to the penny, the strike fund's first levy, the eight-hour day's lost
week, the welfare society's first doctor's bill and the relief scheme's
hardest winter, the branch's whole account, the columns' editor wrote, kept
in the arithmetic of twelve men and their pennies, the ledger's exhibition,
mounted at the last founder's retirement and funded by the fund he had
raised, giving the half-century its public page, the members' own thanks,
the exhibition's card records, printed in the branch's own type, one column
per year, one penny per line, the union's whole story kept countable by the
men who counted it.")

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
build(295, "wiki", {"P1": T295, "P2": T295}, mkrows("the union branch", E295, P295), band=(450, 700))
