import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lf_oc_lib import build_oc, oc_row
from collections import Counter
from dateutil import parser as dp

def assemble(specs, slots):
    rows = []
    for i, ((e1, d1, e2, d2), slot) in enumerate(zip(specs, slots)):
        r = oc_row("P1" if i % 2 == 0 else "P2", e1, d1, e2, d2, slot)
        a, b = dp.parse(d1).date(), dp.parse(d2).date()
        diff = abs((a - b).days)
        assert diff <= 45, ("gap too wide for near-tie", e1, diff)
        assert r["gi"] in (1, 2), ("Fact-1-earlier leaked", e1, e2)
        if r["gi"] == 1:
            assert diff > 14 and b < a
        else:
            assert diff <= 14
        rows.append(r)
    g = Counter(r["gi"] for r in rows)
    s = Counter(r["slot"] for r in rows)
    assert g[0] == 0, g
    return rows, g, s

# ============ 310: wiki, lighthouses and coastal safety, 1936-1965, 20 rows
T310 = ("The rock light's own ledger, kept by the keepers and copied into the "
"board's minute book each quarter, gives the station's middle years their "
"dated account, forty entries in twenty-nine years, every one kept below in "
"the board's own words. The station was registered on October 1, 1936, the "
"ledger's first entry and its plainest, and the lamp was first lit the same "
"evening, the rock's electric dawn watched from every boat in the bay. The "
"lamp was converted on March 4, 1937, the oil order cancelled at last, and "
"the keepers' cottages were rebuilt on April 9, 1937, the winter's storms "
"answered with new slate. The fog signal was installed on June 18, 1938, "
"the rock's first horn, and the radio room was commissioned on July 30, "
"1938, the keepers' voices carried to the shore without a wire. The tramway "
"was relaid on February 9, 1939, the landing's rails renewed, and the winch "
"house was enlarged on March 16, 1939, the coal skips doubled at a haul. "
"The blackout drill was held on March 11, 1941, the war's first practice "
"dark, and the signal was silenced for the drill on March 14, 1941, the "
"horn muffled in the keepers' own felt. The rock extension was completed on "
"May 27, 1946, the landing widened by four paces, and the helipad was "
"marked on July 1, 1946, the station's first cross in white paint. The "
"paraffin store was demolished on August 15, 1947, the last tank taken off "
"by barge, and the mains cable was landed on September 19, 1947, the rock "
"wired to the mainland for good. The relief keeper's log was begun on July "
"2, 1948, the station's second diary, and the barometer was installed on "
"July 12, 1948, the wall's newest brass. The lens was repolished on "
"October 8, 1949, the optic's first full cleaning, and the clockwork drive "
"was replaced on November 19, 1949, the weights retired to the museum "
"shelf. The boat slip was deepened on April 14, 1951, the ebb crossing "
"answered, and the crane rail was extended on May 18, 1951, the stores "
"landed two at a time. The storm damage was repaired on January 22, 1953, "
"the winter's glass paid back pane by pane, and the gallery rail was "
"renewed on March 6, 1953, the lamp room ring walked safely again. The "
"weather station was opened on June 10, 1954, the rock's own readings "
"begun, and the tide gauge was fitted on July 24, 1954, the flood watch "
"shared with the harbour. The tower was repainted on May 6, 1957, the "
"summer scheme's first coat, and the scaffolding was struck on June 13, "
"1957, the white shown clean to the steamers. The helicopter trial was "
"made on February 20, 1958, the relief flown in ten minutes, and the winch "
"was tested on February 27, 1958, the basket rated for two men and a "
"hamper. The bell was rehung on September 17, 1959, the fog's old answer "
"restored, and the automation switch was installed on October 31, 1959, "
"the keepers' last winter made lighter. The museum room was opened on June "
"6, 1960, the station's story shown to the public, and the first donation "
"was logged on June 9, 1960, a keeper's brass whistle given to the case. "
"The last keeper retired on November 5, 1962, thirty years counted in the "
"one log, and the plaque was unveiled on December 10, 1962, the board's "
"thanks cut in bronze. The visitors' gallery was opened on April 20, 1963, "
"the public's first landing by ticket, and the car park was laid on June "
"4, 1963, the cliff top given its gravel. The centenary exhibition was "
"held on August 8, 1965, the station's hundred years told in a single "
"room, and the memorial bench was placed on September 12, 1965, the "
"keepers' names cut in the oak. The final inspection was passed on "
"September 30, 1965, the board's own walk of the whole station, and the "
"flag was raised on October 5, 1965, the ledger closed at full light.")

SP310 = [
 ("the station was registered", "October 1, 1936", "the lamp was first lit", "October 1, 1936"),
 ("the keepers' cottages were rebuilt", "April 9, 1937", "the lamp was converted", "March 4, 1937"),
 ("the radio room was commissioned", "July 30, 1938", "the fog signal was installed", "June 18, 1938"),
 ("the winch house was enlarged", "March 16, 1939", "the tramway was relaid", "February 9, 1939"),
 ("the signal was silenced for the drill", "March 14, 1941", "the blackout drill was held", "March 11, 1941"),
 ("the helipad was marked", "July 1, 1946", "the rock extension was completed", "May 27, 1946"),
 ("the mains cable was landed", "September 19, 1947", "the paraffin store was demolished", "August 15, 1947"),
 ("the barometer was installed", "July 12, 1948", "the relief keeper's log was begun", "July 2, 1948"),
 ("the clockwork drive was replaced", "November 19, 1949", "the lens was repolished", "October 8, 1949"),
 ("the crane rail was extended", "May 18, 1951", "the boat slip was deepened", "April 14, 1951"),
 ("the gallery rail was renewed", "March 6, 1953", "the storm damage was repaired", "January 22, 1953"),
 ("the tide gauge was fitted", "July 24, 1954", "the weather station was opened", "June 10, 1954"),
 ("the scaffolding was struck", "June 13, 1957", "the tower was repainted", "May 6, 1957"),
 ("the winch was tested", "February 27, 1958", "the helicopter trial was made", "February 20, 1958"),
 ("the automation switch was installed", "October 31, 1959", "the bell was rehung", "September 17, 1959"),
 ("the first donation was logged", "June 9, 1960", "the museum room was opened", "June 6, 1960"),
 ("the plaque was unveiled", "December 10, 1962", "the last keeper retired", "November 5, 1962"),
 ("the car park was laid", "June 4, 1963", "the visitors' gallery was opened", "April 20, 1963"),
 ("the memorial bench was placed", "September 12, 1965", "the centenary exhibition was held", "August 8, 1965"),
 ("the flag was raised", "October 5, 1965", "the final inspection was passed", "September 30, 1965"),
]
SL310 = [0, 1, 2] * 6 + [0, 1]

# ============ 311: news, public libraries, 2005-2014, 20 rows
S311a = ("[1] Title: The Library Ledger, Day: April 4, 2005 "
"Content: The county library's renewal decade, kept in the chief "
"librarian's ledger and printed by the town's paper entire at its end, "
"runs as follows, in the ledger's own order. The reading room was "
"reopened on February 8, 2005, the scaffolding down and the radiators "
"bled, and the new boilers were lit on March 15, 2005, the first winter "
"without coal in the building's history. The library trust was founded on "
"April 4, 2005, the renewal's own footing, and the founding donors were "
"registered the same day, forty names on the ledger's first ruled page. "
"The mobile shelving was installed on August 19, 2005, the stacks rolled "
"for the first time on their new rails, and the returns chute was widened "
"on October 1, 2005, the autumn's overdue amnesty emptied through it in a "
"week. The book bus route was launched on May 10, 2006, the valley's "
"farms on the timetable at last, and the weekend loans were doubled on "
"June 14, 2006, the first summer the shelves were counted twice on "
"Mondays. The book-wash day was held on March 9, 2007, the volunteers' "
"sponges at work in the reading room, and the dust jackets were repaired "
"on April 15, 2007, the county's oldest novels given new sleeves. The "
"archive room was sealed on July 18, 2007, the deeds moved to the vault, "
"and the microfilm readers were installed on August 29, 2007, the parish "
"registers read without gloves at last. The children's wing was painted "
"on September 3, 2008, the mural's sea wall begun, and the story corner "
"was carpeted on October 8, 2008, the first story time sat cross-legged "
"on the new weave.")

S311b = ("[2] Title: The Ledger Continued, Day: March 3, 2009 "
"Content: The renewal's middle years, kept in the same hand, run on. The "
"lending fees were abolished on January 20, 2009, the ledger's shortest "
"invoice written and filed, and the self-service kiosks went live on "
"March 3, 2009, the first book issued by machine to the mayor, who tried "
"twice. The power cut struck on December 2, 2009, the town dark from the "
"substation out, and the candlelit story hour was held on December 5, "
"2009, forty children and the whole of the picture books read by flame. "
"The local history index was published on April 15, 2010, the parish's "
"four hundred years sorted to a card, and the photograph collection was "
"digitised on May 28, 2010, the flood of 1897 kept safe in pixels. The "
"mosaic was unveiled on June 16, 2010, the entrance's new floor, and the "
"artist's talk was hosted on June 26, 2010, every tessera explained to a "
"full room. The roof lantern was restored on June 7, 2011, the dome's "
"glass lifted and reset, and the reading garden was planted on July 21, "
"2011, the seed catalogues germinated in the library's own beds. The "
"small presses fair was hosted on July 6, 2012, the town's own pamphlets "
"on the long tables, and the zine shelf was mounted on August 12, 2012, "
"the fair's whole harvest shelved within the month. The night-study "
"scheme was piloted on September 12, 2012, the hall lit until ten for the "
"exam seasons, and the exam desk hire began on October 26, 2012, the "
"first hundred desks counted out and counted back. The flood alarm was "
"tested on November 8, 2012, the river's name read aloud from the gauge, "
"and the sandbags were stored on November 13, 2012, the volunteers' wall "
"ready in the corridor.")

S311c = ("[3] Title: The Ledger's Last Leaves, Day: September 2, 2014 "
"Content: The renewal's final years, the paper's editor wrote, were the "
"library's quietest and fullest. The music scores were catalogued on "
"February 11, 2013, the band parts united at last, and the listening "
"booth was fitted on March 22, 2013, the first score heard as well as "
"read. The reading room's centenary was marked on October 21, 2013, the "
"founding minutes read aloud at noon, and the centenary cake was cut on "
"October 25, 2013, the icing a sugar replica of the lantern. The code "
"club was founded on November 10, 2013, the first Saturday robots on the "
"story corner's carpet, and the robots were lent on December 15, 2013, "
"the club's waiting list given its own shelf. The phone-charging bench "
"was fitted on May 14, 2014, the reading room's quietest renovation, and "
"the study lamps were rewired on June 24, 2014, the last glare taken off "
"the local studies. The last card drawer was retired on August 30, 2014, "
"the oak frame emptied of its millionth card, and the drawer was given to "
"the museum on September 2, 2014, the ledger's last entry, the editor "
"concluding that the decade had been kept, as the library keeps "
"everything, in order and in the open.")

SP311 = [
 ("the new boilers were lit", "March 15, 2005", "the reading room was reopened", "February 8, 2005"),
 ("the founding donors were registered", "April 4, 2005", "the library trust was founded", "April 4, 2005"),
 ("the returns chute was widened", "October 1, 2005", "the mobile shelving was installed", "August 19, 2005"),
 ("the weekend loans were doubled", "June 14, 2006", "the book bus route was launched", "May 10, 2006"),
 ("the dust jackets were repaired", "April 15, 2007", "the book-wash day was held", "March 9, 2007"),
 ("the microfilm readers were installed", "August 29, 2007", "the archive room was sealed", "July 18, 2007"),
 ("the story corner was carpeted", "October 8, 2008", "the children's wing was painted", "September 3, 2008"),
 ("the self-service kiosks went live", "March 3, 2009", "the lending fees were abolished", "January 20, 2009"),
 ("the candlelit story hour was held", "December 5, 2009", "the power cut struck", "December 2, 2009"),
 ("the photograph collection was digitised", "May 28, 2010", "the local history index was published", "April 15, 2010"),
 ("the artist's talk was hosted", "June 26, 2010", "the mosaic was unveiled", "June 16, 2010"),
 ("the reading garden was planted", "July 21, 2011", "the roof lantern was restored", "June 7, 2011"),
 ("the zine shelf was mounted", "August 12, 2012", "the small presses fair was hosted", "July 6, 2012"),
 ("the exam desk hire began", "October 26, 2012", "the night-study scheme was piloted", "September 12, 2012"),
 ("the sandbags were stored", "November 13, 2012", "the flood alarm was tested", "November 8, 2012"),
 ("the listening booth was fitted", "March 22, 2013", "the music scores were catalogued", "February 11, 2013"),
 ("the centenary cake was cut", "October 25, 2013", "the reading room's centenary was marked", "October 21, 2013"),
 ("the robots were lent", "December 15, 2013", "the code club was founded", "November 10, 2013"),
 ("the study lamps were rewired", "June 24, 2014", "the phone-charging bench was fitted", "May 14, 2014"),
 ("the drawer was given to the museum", "September 2, 2014", "the last card drawer was retired", "August 30, 2014"),
]
SL311 = [0, 1, 2] * 6 + [0, 2]

# ============ 312: news, mining communities, 1966-1989, 20 rows
S312a = ("[1] Title: The Valley's Own Ledger, Day: March 22, 1966 "
"Content: The mining valley's regeneration, kept in the welfare fund's "
"ledger and printed by the colliery town's gazette entire, runs as "
"follows. The pit-head baths were rebuilt on March 22, 1966, the first "
"project of the fund's own choosing, and the new lockers were fitted on "
"May 5, 1966, every man's name stamped in brass. The rescue brigade was "
"founded on January 9, 1967, the valley's own team, and the first drill "
"was held the same day, the brigade's first crawl through the old "
"workings. The banner was repainted on June 14, 1967, the lodge's silk "
"given its colliers back, and the brass was restored on July 28, 1967, "
"the band's instruments burnished for the gala. The welfare hall was "
"re-roofed on September 6, 1968, the dances kept dry for another "
"generation, and the stage lights were rewired on October 18, 1968, the "
"pantomime seen plainly at last. The brickworks chimney was felled on May "
"13, 1969, the valley's dustiest landmark laid flat in a morning, and the "
"site was levelled on June 20, 1969, the bricks sorted for the hall's new "
"yard. The gala day was held on June 21, 1969, the whole valley walking "
"behind the banner, and the children's race was run on June 24, 1969, the "
"spare buns given as prizes. The coal board offices were closed on August "
"11, 1970, the ledgers moved to the county town, and the building was "
"sold to the trust on September 25, 1970, the valley buying its own "
"frontage for the first time. The library van was upgraded on February "
"24, 1971, the weekly shelves doubled, and the shelves were widened on "
"April 7, 1971, the westerns and the romances separated at last.")

S312b = ("[2] Title: The Ledger Continued, Day: July 8, 1972 "
"Content: The fund's middle years ran on in the same plain entries. The "
"children's playground was opened on July 8, 1972, the pit dirt turned to "
"sand and swings, and the paddling pool was filled on August 16, 1972, "
"the first summer's shrieks counted in the hundreds. The band hall was "
"repanelled on October 15, 1973, the practice room warmed through, and "
"the instruments were lacquered on November 27, 1973, the cornets shining "
"for the contests. The chapel's last service was held on November 30, "
"1974, the seam under the hill given its amen, and the keys were handed "
"over on December 3, 1974, the trustees passing them to the fund. The "
"railway spur was lifted on April 2, 1975, the last coal wagon gone "
"before the rails, and the ballast was recycled on May 9, 1975, the "
"spur's bed given to the allotment paths. The allotments were extended on "
"June 17, 1976, twenty new plots staked on the levelled ground, and the "
"rhubarb rows were planted on July 30, 1976, the fund's own crowns from "
"the champion's garden. The memorial gate was forged on September 19, "
"1978, the smith's heaviest work in forty years, and the dedication was "
"held on October 27, 1978, the valley's names read out entire. The office "
"clock was restored on May 21, 1981, the colliery's own time kept again, "
"and the chime was repaired on July 3, 1981, the quarter hours returned "
"to the square.")

S312c = ("[3] Title: The Ledger's Last Entries, Day: May 3, 1985 "
"Content: The ledger's final years, the gazette's editor wrote, kept the "
"valley's account through the closing and after. The winding house was "
"preserved on March 8, 1984, the headgear saved by the fund's petition, "
"and the visitor walkway was built on April 20, 1984, the public's first "
"safe climb of the tower. The deep mine was closed on May 3, 1985, the "
"last shift drawn at noon, and the final shift reached the surface on May "
"3, 1985, the whole town at the fence to count them up. The spoil heap "
"was grassed on August 25, 1986, the black hill given its green, and the "
"wildflower seed was sown on October 3, 1986, the schoolchildren's "
"afternoon on the slope. The heritage centre was opened on June 27, 1987, "
"the baths' old drying floor made into the gallery, and the first tour "
"was given on June 30, 1987, the guide a bath attendant of thirty years. "
"The last lamp was extinguished on October 10, 1989, the lamp room's "
"final flame out at the fund's own ceremony, and the switch was preserved "
"on October 13, 1989, mounted in the centre's doorway, the ledger closed "
"with the valley's own light kept, the editor concluded, in the place "
"that kept everything else.")

SP312 = [
 ("the new lockers were fitted", "May 5, 1966", "the pit-head baths were rebuilt", "March 22, 1966"),
 ("the first drill was held", "January 9, 1967", "the rescue brigade was founded", "January 9, 1967"),
 ("the brass was restored", "July 28, 1967", "the banner was repainted", "June 14, 1967"),
 ("the stage lights were rewired", "October 18, 1968", "the welfare hall was re-roofed", "September 6, 1968"),
 ("the site was levelled", "June 20, 1969", "the brickworks chimney was felled", "May 13, 1969"),
 ("the children's race was run", "June 24, 1969", "the gala day was held", "June 21, 1969"),
 ("the building was sold to the trust", "September 25, 1970", "the coal board offices were closed", "August 11, 1970"),
 ("the shelves were widened", "April 7, 1971", "the library van was upgraded", "February 24, 1971"),
 ("the paddling pool was filled", "August 16, 1972", "the children's playground was opened", "July 8, 1972"),
 ("the instruments were lacquered", "November 27, 1973", "the band hall was repanelled", "October 15, 1973"),
 ("the keys were handed over", "December 3, 1974", "the chapel's last service was held", "November 30, 1974"),
 ("the ballast was recycled", "May 9, 1975", "the railway spur was lifted", "April 2, 1975"),
 ("the rhubarb rows were planted", "July 30, 1976", "the allotments were extended", "June 17, 1976"),
 ("the dedication was held", "October 27, 1978", "the memorial gate was forged", "September 19, 1978"),
 ("the chime was repaired", "July 3, 1981", "the office clock was restored", "May 21, 1981"),
 ("the visitor walkway was built", "April 20, 1984", "the winding house was preserved", "March 8, 1984"),
 ("the final shift reached the surface", "May 3, 1985", "the deep mine was closed", "May 3, 1985"),
 ("the wildflower seed was sown", "October 3, 1986", "the spoil heap was grassed", "August 25, 1986"),
 ("the first tour was given", "June 30, 1987", "the heritage centre was opened", "June 27, 1987"),
 ("the switch was preserved", "October 13, 1989", "the last lamp was extinguished", "October 10, 1989"),
]
SL312 = [0, 1, 2] * 6 + [1, 2]

# ============ 313: news, wildlife and conservation, 2020-2024, 10 rows
S313a = ("[1] Title: The Reserve's Dotted Ledger, Day: March 16, 2020 "
"Content: The reserve's revival, kept in the warden's annual ledger and "
"printed by the county's paper at each season's turn, runs as follows. "
"The osprey platform was raised on March 16, 2020, the volunteers' "
"midweek crane watched by the whole heronry, and the nesting camera was "
"fitted on April 27, 2020, the first egg counted from the warden's bothy. "
"The hedgehog hospital was opened on April 8, 2020, the old cattle shed "
"given its weighing scales, and the first release was made on April 9, "
"2020, eleven convalescents walked back into the hedge line at dusk. The "
"hedgerow survey was completed on July 13, 2021, forty miles of the "
"parish's boundaries walked and noted, and the dormouse boxes were hung "
"on August 16, 2021, the survey's hundredth tube numbered in pencil. The "
"wetland scrapes were dug on October 5, 2021, the arable corner returned "
"to its old shallow water, and the wader counts were doubled on November "
"15, 2021, the lapwing flocks read in thousands for the first time since "
"the survey began. The scrape digging itself, the warden's ledger notes, was watched from the road by the whole parish in turns, the contractor's excavator given a round of applause at the last stone, the arable corner's return to water, the ledger's own footnote says, the reserve's plainest reversal and its loudest, the water's return celebrated by the count of wings that followed it in.")

S313b = ("[2] Title: The Beavers' Chapter, Day: May 11, 2022 "
"Content: The revival's second year, the warden's ledger noted, was "
"written mostly by the animals themselves. The beaver family was released "
"on May 11, 2022, four kits and their parents carried to the upper pool "
"in fishing boxes, and the first lodge was built on May 24, 2022, the "
"release family's own architecture counted stick by stick from the hide. "
"The sea-eagle chick fledged on July 19, 2022, the cliff nest empty at "
"the morning check, and the chick was ringed on July 23, 2022, the "
"warden's ladder borrowed from the fire brigade and returned the same "
"afternoon, the ring's number read to the visitors' applause. The ringing's own entry, continued in the same hand, records the bird's first circle of the cliff twice and its landing on the reserve's tallest ash, the visitors' applause given again from the hide, the season's second chapter, the warden's margin note says, written by the eagle itself and merely witnessed, the ledger's proudest page and its shortest, the wild's own signature kept in the warden's pencil.")

S313c = ("[3] Title: The Ledger's Last Seasons, Day: February 20, 2023 "
"Content: The revival's later seasons, the paper's correspondent wrote, "
"filled the ledger's remaining pages at the same patient pace. The "
"orchard was planted on February 20, 2023, sixty county varieties staked "
"and named, and the bee hotel was installed on April 3, 2023, the "
"orchard's first tenants counted by the junior wardens within the week. "
"The river weir was removed on September 7, 2023, the water's old wall "
"taken down stone by stone, and the eel pass was counted on October 15, "
"2023, the season's migration logged in the thousands for the first time "
"in the weir's century. The corncrake was recorded on June 2, 2024, the "
"rasping call heard from the meadow gate at dusk, and the meadow mowing "
"was delayed on June 13, 2024, the cut held back for the nesting season "
"at the farmers' own suggestion. Dark-sky status was granted on November "
"12, 2024, the parish's lamps dimmed to the reserve's advantage, and the "
"bat lanterns were dimmed on December 20, 2024, the ledger's last entry, "
"the warden concluding that the reserve had been kept, as the wild keeps "
"itself, by patience and by the counted year. The ledger's own epilogue, written by the warden for the paper's year-end edition and read aloud at the reserve's open morning, gives the revival its whole arithmetic: four years, sixty orchard trees, a thousand eels, one corncrake and a parish's worth of dimmed lamps, the epilogue's last line the warden's own, that the counting had been the keeping, and the keeping the revival, the reserve's account, the open morning's programme says, closed with the county's own dark sky overhead and opened again each spring by the same patient pen.")

SP313 = [
 ("the nesting camera was fitted", "April 27, 2020", "the osprey platform was raised", "March 16, 2020"),
 ("the first release was made", "April 9, 2020", "the hedgehog hospital was opened", "April 8, 2020"),
 ("the dormouse boxes were hung", "August 16, 2021", "the hedgerow survey was completed", "July 13, 2021"),
 ("the wader counts were doubled", "November 15, 2021", "the wetland scrapes were dug", "October 5, 2021"),
 ("the first lodge was built", "May 24, 2022", "the beaver family was released", "May 11, 2022"),
 ("the chick was ringed", "July 23, 2022", "the sea-eagle chick fledged", "July 19, 2022"),
 ("the bee hotel was installed", "April 3, 2023", "the orchard was planted", "February 20, 2023"),
 ("the eel pass was counted", "October 15, 2023", "the river weir was removed", "September 7, 2023"),
 ("the meadow mowing was delayed", "June 13, 2024", "the corncrake was recorded", "June 2, 2024"),
 ("the bat lanterns were dimmed", "December 20, 2024", "dark-sky status was granted", "November 12, 2024"),
]
SL313 = [0, 1, 2] * 3 + [0]

ROWS310, G, S = assemble(SP310, SL310)
print("310 golds F1/F2/same:", G[0], G[1], G[2], "letters:", dict(sorted(S.items())))
build_oc(310, "wiki", {"P1": T310, "P2": T310}, ROWS310, band=(450, 700))

ROWS311, G, S = assemble(SP311, SL311)
print("311 golds F1/F2/same:", G[0], G[1], G[2], "letters:", dict(sorted(S.items())))
build_oc(311, "news", {"P1": (S311a, S311b, S311c), "P2": (S311a, S311b, S311c)}, ROWS311)

ROWS312, G, S = assemble(SP312, SL312)
print("312 golds F1/F2/same:", G[0], G[1], G[2], "letters:", dict(sorted(S.items())))
build_oc(312, "news", {"P1": (S312a, S312b, S312c), "P2": (S312a, S312b, S312c)}, ROWS312)

ROWS313, G, S = assemble(SP313, SL313)
print("313 golds F1/F2/same:", G[0], G[1], G[2], "letters:", dict(sorted(S.items())))
build_oc(313, "news", {"P1": (S313a, S313b, S313c), "P2": (S313a, S313b, S313c)}, ROWS313)
