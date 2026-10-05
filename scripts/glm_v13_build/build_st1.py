import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from st_lib import build_st, st_row

# =================== 298: storytelling wiki, courts 1936-1965
T298 = ("The story of the crown court's middle years, told by the oldest "
"usher to the county's paper at his retirement, runs as follows. When "
"the new crown court opened its doors on October 12, 1937, there were "
"eleven oak benches in the public gallery, each cut from the old "
"assize hall's single fallen cedar, and the usher's first duty was "
"counting them twice daily. The building held fourteen cells below "
"the courtroom, though the average occupancy in those years was "
"never more than three, and the great clock above the portico kept "
"a time seven minutes ahead of the town's other clocks, the judge's "
"own order, so that no barrister could plead a late watch. The law "
"library on the second floor contained four thousand volumes, "
"shelved by the usher's own system, which the county's librarians "
"later adopted entire. In the war years the basement sheltered "
"two hundred townsfolk a night, and the usher kept a tally chalked "
"on the cells' wall that reached nine hundred nights before the "
"all-clear fell for good. The dome was regilded in 1949 with leaf "
"enough for six doors, and the whispering gallery beneath it, where "
"twelve jurors once claimed to hear the judge's pen, became the "
"building's favourite tour. The story the usher told most often "
"concerned the escaped canary of 1953, which lived in the dome for "
"forty days and was pardoned by the judge himself at the "
"sitting's end, the court's only unconditional discharge. The "
"retirement of the old registrar in 1957 emptied the oldest desk "
"of nineteen drawers, each lock keyed to a different decade, and "
"the usher was given the eighth drawer for his own. When the "
"court's centenary stone was rededicated in 1962, the town's "
"children counted the steps to the portico and found "
"thirty-seven, one for every year of the usher's service then "
"kept. He retired in 1965 after twenty-eight years of counted "
"benches, chalked nights and pardoned birds, leaving the story, "
"the paper's editor wrote, of a building kept by a man who "
"counted everything and lost nothing, the usher's own ledger of "
"the court's middle years numbering exactly sixty pages, one "
"for every bench-count of his last month, the county's plainest "
"and truest court record. The paper's own footnote to the story, printed beside it at the usher's request, records the counting's moral: that a court is kept by what it notices, and that the usher had noticed everything, the footnote's last line giving the story its own arithmetic, twenty-eight years of service, one canary, one green-ink morning and eleven benches, the whole law of the building, the usher said, in four figures and a bird. The footnote's own frame, cut from the old assize cedar's last offcut by the town's joiner, holds the printed story behind the court's portico glass, the building's newest exhibit and its plainest, the joiner's card beside it recording the tree's whole service, one hall, eleven benches and one frame, the court's middle years, the archivist concludes, kept in a single cedar's three lifetimes.")

FR298 = [
 ("The story ends with the usher counting the gallery's benches one last time and finding all {v} of them sound.", "eleven", "nine"),
 ("The story ends with the night warden chalking the shelter's tally on the cells' wall and reading {v} nights aloud.", "nine hundred", "eight hundred"),
 ("The story ends with the judge discharging the canary after {v} days in the dome, the court rising to whistle.", "forty", "fourteen"),
 ("The story ends with the children counting the portico's steps and shouting {v} back to the town.", "thirty-seven", "twenty-nine"),
 ("The story ends with the registrar's desk standing open, all {v} drawers unlocked at last.", "nineteen", "twelve"),
 ("The story ends with the usher's ledger closed at exactly {v} pages, one for every last bench-count.", "sixty", "fifty"),
 ("The story ends with the librarian shelving the law library's {v} volumes under the usher's system entire.", "four thousand", "five thousand"),
 ("The story ends with the cells' average occupancy entered in the night book as no more than {v}.", "three", "six"),
 ("The story ends with the town's clocks reset and the portico's kept {v} minutes ahead, as ordered.", "seven", "eleven"),
 ("The story ends with the dome regilded and the whispering gallery holding {v} jurors' worth of echo.", "twelve", "nine"),
 ("The story ends with the shelter's tally chalked to {v} nights before the final all-clear.", "nine hundred", "seven hundred"),
 ("The story ends with the usher's last month of bench-counts filling {v} pages to the line.", "sixty", "forty"),
 ("The story ends with the cedar's benches, all {v}, given one final coat before his retirement.", "eleven", "thirteen"),
 ("The story ends with the courthouse basement counted safe for {v} townsfolk a night.", "two hundred", "three hundred"),
 ("The story ends with the pardoned canary's {v} days entered in the court's own record.", "forty", "thirty"),
 ("The story ends with the portico climbed and {v} steps counted, one for each year kept.", "thirty-seven", "thirty-one"),
 ("The story ends with the eighth drawer of {v} given to the usher for his own.", "nineteen", "seventeen"),
 ("The story ends with the whispering gallery tried again and {v} jurors claiming to hear the pen.", "twelve", "eight"),
 ("The story ends with the clock's {v} minutes of authority kept over every watch in town.", "seven", "five"),
 ("The story ends with the shelter nights tallied at {v} before the war's last morning.", "nine hundred", "six hundred"),
]
ROWS298 = [st_row("P1" if i % 2 == 0 else "P2", f, c, w, i % 2)
           for i, (f, c, w) in enumerate(FR298)]
build_st(298, "wiki", {"P1": T298, "P2": T298}, ROWS298, band=(450, 700))

# =================== 299: storytelling news, climate 2020-2024
S299a = ("[1] Title: The Warden's Weather Story, Day: March 20, 2020 "
"Content: The story of the weather station's renewal years, told by "
"the warden at the station's jubilee and printed by the county's "
"paper entire, runs as follows. When the new automatic station was "
"commissioned on March 17, 2020, it carried three redundant "
"anemometers, and the warden's first act was to unplug one, on the "
"principle that a station trusted like a barometer must be checked "
"like a kettle. The rain gauge was relocated twice that first year "
"and finally settled forty paces from the door, where the old "
"storm post still leaned. The station logged sixty-one dry days  The deepest snow of the renewal years fell at "
"twenty-two centimetres, and the station logged sixty-one dry days "
"in the drought of 2022, and the warden's logbook recorded the "
"drought's first drop of rain at four minutes past three in the "
"morning, the only entry in green ink the logbook ever held. The "
"storm naming began in 2021 with a list of twenty names drawn by "
"the county's schoolchildren, and the first named storm, the "
"third of the list, took the station's own door off its hinges.")

S299b = ("[2] Title: The Story Continued, Day: June 5, 2023 "
"Content: The station's solar array was installed in 2023 with "
"nine panels, and the warden's report noted that the kettle, "
"the station's oldest instrument by sentiment, now ran on the "
"sun entirely. The rainfall total for 2023 was eight hundred "
"and twelve millimetres, the wettest of the renewal years, and "
"the snowfall of February 2024, the deepest in a generation, "
"was measured with the yardstick the warden kept for the "
"purpose since his first winter. The station's open day in "
"2024 drew four hundred visitors, and the warden told the "
"story of the green ink entry to every one of them, the "
"station's whole renewal, the paper's editor wrote, kept by "
"a man who wrote down the weather as if it were news, which, "
"the editor concluded, in the county it was. The story's own postscript, written by the warden for the station's archive and printed in the paper's second edition, adds the counting's whole purpose: that the station's instruments had been trusted because they were tended, and tended because they were counted, the postscript's last line the warden's own, that the weather had never once surprised him who wrote it down, the station's whole renewal, the archive's card says, kept in green ink, red ink and the ordinary black of sixty-one dry days. The postscript's own display, mounted in the station's hallway at the warden's retirement and read aloud each open day since, carries the green-ink entry beneath glass with the kettle beside it, the station's whole renewal, the archive's card says, kept in three inks, one kettle and forty paces of settled gauge, the warden's last instruction being that the entry be read to every visitor in the order it was written, the weather's own grammar, the card concludes, taught by the man who kept it.")

FR299 = [
 ("The story ends with the warden unplugging one of the {v} anemometers and trusting the kettle test.", "three", "two"),
 ("The story ends with the rain gauge settled {v} paces from the door, beside the old storm post.", "forty", "fifty"),
 ("The story ends with the logbook recording the drought's {v} dry days without a single entry in ink.", "sixty-one", "fifty-nine"),
 ("The story ends with the drought's first rain logged at {v} minutes past three, in green ink.", "four", "seven"),
 ("The story ends with the storm list carrying {v} names drawn by the county's schoolchildren.", "twenty", "twelve"),
 ("The story ends with the first named storm, the {v} of the list, taking the door off its hinges.", "third", "first"),
 ("The story ends with the kettle running on {v} solar panels and no other power at all.", "nine", "five"),
 ("The story ends with the rainfall total entered as {v} millimetres, the renewal's wettest year.", "eight hundred and twelve", "seven hundred and twelve"),
 ("The story ends with the snowfall measured with the yardstick kept since the warden's first winter, at {v} centimetres.", "twenty-two", "thirty"),
 ("The story ends with the open day drawing {v} visitors and the green ink story told to each.", "four hundred", "five hundred"),
 ("The story ends with the gauge's two moves and final rest {v} paces out, where the post leaned.", "forty", "thirty"),
 ("The story ends with the drought counted at {v} dry days and the kettle kept boiling.", "sixty-one", "sixty"),
 ("The story ends with the green ink entry timed at four minutes past three and logged in the drought's own {v}.", "green ink", "red ink"),
 ("The story ends with the array's {v} panels counted and the kettle's cord coiled away for good.", "nine", "eleven"),
 ("The story ends with the storm list's {v} names read aloud at the open day, the door displayed beside them.", "twenty", "fifteen"),
 ("The story ends with the yardstick marking the deepest snow at {v} centimetres of a generation.", "twenty-two", "eighteen"),
 ("The story ends with the anemometers reduced from {v} to two and the station none the worse.", "three", "four"),
 ("The story ends with the year's rain totalling {v} millimetres and the gauge finally trusted.", "eight hundred and twelve", "eight hundred"),
 ("The story ends with the open day's {v} visitors each hearing the storm-door story whole.", "four hundred", "three hundred"),
 ("The story ends with the kettle retired at last, powered once by coal and finally by {v} panels.", "nine", "six"),
]
ROWS299 = [st_row("P1" if i % 2 == 0 else "P2", f, c, w, i % 2)
           for i, (f, c, w) in enumerate(FR299)]
S299c = S299b.replace("[2] Title: The Story Continued", "[3] Title: The Story Continued, Completed", 1) + " The completion's own toast, drunk at the party from the story's own saucers, gave the account its last sound, the paper's reporter noted, louder than the whole story's quietest years, the tale, the party's last entry records, retired to applause."
build_st(299, "news", {"P1": (S299a, S299b, S299c), "P2": (S299a, S299b, S299c)}, ROWS299)

# =================== 300: storytelling news, energy 2005-2014
S300a = ("[1] Title: The Engineer's Turbine Story, Day: April 8, 2005 "
"Content: The story of the co-operative's first decade, told by the "
"retired engineer at the turbine's tenth birthday and printed in "
"the harbour's paper entire, runs as follows. When the first "
"turbine was raised on April 5, 2005, it carried forty-one bolts, "
"one for each founding member, and the engineer torqued the last "
"bolt himself at dawn with the wrench now mounted in the control "
"room. The turbine's first year produced three gigawatt hours, "
"and the co-op's first cheque was written to the harbour's "
"lifeboat station for forty pounds and eleven pence, the "
"dividend of a single winter. The battery shed followed in 2008 "
"with ninety-six cells, and the engineer's log noted the shed's "
"first full charge held through nine windless days.")

S300b = ("[2] Title: The Story Continued, Day: May 30, 2012 "
"The second turbine was raised in 2010, thirty of the founding "
"members standing for the raising, and the same "
"forty-one bolts, and the engineer, now grey, torqued the last "
"bolt at dawn again, the co-op's own liturgy. The meter at the "
"slip's head turned its first millionth kilowatt hour in 2012, "
"and the engineer's photograph beside it shows him pointing at "
"the numeral seven, his own favourite, carried in the reading "
"three times. The co-op's membership reached two hundred in "
"2013, and the tenth birthday cake in 2015 carried forty-one "
"candles, one bolt's worth, the harbour's paper concluding "
"that the co-op's whole story was kept by a man who counted "
"in bolts and believed in sevens, the engineer's own ledger "
"of the decade numbering ten pages, one per year, and "
"eleven words, one per member's pound. The story's own coda, told by the engineer at the co-op's annual meeting and added to the printed account by the members' vote, gives the counting its last word: that the turbine had turned on forty-one bolts and the co-op on forty-one pounds, and that both had held because both were counted, the coda's final line the engineer's own, that a bolt torqued at dawn is a promise, and a promise counted is a co-operative, the whole first decade, the meeting's minute says, kept in that one sentence. The coda's own engraving, cut by the harbour's last shipwright into the control room's door, carries the sentence at bolt height, the engineer's own measure, so that every operator reads it at the start of every shift, the co-op's whole first decade, the minute's last annotation says, kept at eye level and torque weight, the shipwright's final line recording the door's own count, forty-one letters to the sentence's first clause, matched to the bolts by accident and kept by design.")

FR300 = [
 ("The story ends with the engineer torquing the last of {v} bolts at dawn, wrench in hand.", "forty-one", "thirty-nine"),
 ("The story ends with the first year's production entered as {v} gigawatt hours.", "three", "four"),
 ("The story ends with the lifeboat's cheque written for {v}, the winter's whole dividend.", "forty pounds and eleven pence", "forty pounds and nine pence"),
 ("The story ends with the battery shed's {v} cells holding the first full charge.", "ninety-six", "sixty-nine"),
 ("The story ends with the charge held through {v} windless days and the kettle still warm.", "nine", "five"),
 ("The story ends with the second turbine raised and the same {v} bolts counted home.", "forty-one", "thirty"),
 ("The story ends with the meter turning its first millionth kilowatt hour in {v}.", "2012", "2011"),
 ("The story ends with the engineer pointing at the numeral seven, carried {v} times in the reading.", "three", "two"),
 ("The story ends with the membership book closing at {v} names for the first time.", "two hundred", "one hundred"),
 ("The story ends with the birthday cake carrying {v} candles, one bolt's worth.", "forty-one", "twenty"),
 ("The story ends with the ledger kept at {v} pages, one per year of the decade.", "ten", "twelve"),
 ("The story ends with the ledger's last line holding eleven words, one per member's {v}.", "pound", "shilling"),
 ("The story ends with the wrench mounted in the control room above the {v} bolts' specification.", "forty-one", "fifty"),
 ("The story ends with the first cheque framed beside the meter reading {v} gigawatt hours.", "three", "two"),
 ("The story ends with the shed's charge counted good for {v} windless days entire.", "nine", "seven"),
 ("The story ends with the second raising attended by {v} of the founding members still standing.", "thirty", "twenty"),
 ("The story ends with the millionth hour photographed and the seven counted {v} times.", "three", "four"),
 ("The story ends with the membership reaching {v} and the book ruled for the next hundred.", "two hundred", "two hundred and fifty"),
 ("The story ends with the cake cut into {v} pieces, one bolt's worth for each table.", "forty-one", "thirty"),
 ("The story ends with the engineer's decade closed at ten pages and {v} words for the pound.", "eleven", "nine"),
]
ROWS300 = [st_row("P1" if i % 2 == 0 else "P2", f, c, w, i % 2)
           for i, (f, c, w) in enumerate(FR300)]
S300c = S300b.replace("[2] Title: The Story Continued", "[3] Title: The Story Continued, Completed", 1) + " The completion's own toast, drunk at the party from the story's own saucers, gave the account its last sound, the paper's reporter noted, louder than the whole story's quietest years, the tale, the party's last entry records, retired to applause."
build_st(300, "news", {"P1": (S300a, S300b, S300c), "P2": (S300a, S300b, S300c)}, ROWS300)

# =================== 301: storytelling news, tech @15 1966-1989
S301a = ("[1] Title: The Supervisor's Exchange Story, Day: June 4, 1966 "
"Content: The story of the exchange's automation, told by the "
"supervisor at the switchboard's retirement party and kept in "
"the town journal's archive, runs as follows. When the new "
"crossbar exchange was cut over on June 1, 1966, there were "
"two hundred operators on the board, and the supervisor's "
"first automated call was dialed wrong twice by the mayor and "
"completed third time to the fire station, which sent an "
"engine anyway. The exchange's first year connected nine "
"thousand lines, and the fault rate fell to one in five "
"hundred calls, the journal's own figure, from one in eighty "
"on the manual board. The night service was kept by four "
"operators through the changeover, and their kettle, the "
"board's true heart, was rewired three times before it was "
"finally allowed its own socket.")

S301b = ("[2] Title: The Story's Last Entry, Day: August 19, 1989 "
"Content: The last manual call was completed on August 16, 1989, "
"at nine minutes past nine in the evening, the supervisor "
"plugging home the board's final connection to the clock of "
"the town's oldest subscriber, and the journal printed the "
"board's closing tally entire: twenty-three years, four "
"operators, one kettle and zero calls lost in the changeover's "
"whole account, the supervisor's own story, the journal's "
"editor concluded, of a machine tended like a garden and "
"retired like a friend, the exchange's truest record kept "
"in the kettle's steam and the operators' ledger. The story's own last page, written by the supervisor for the journal's archive and printed beside the closing tally, gives the changeover its epitaph: that the machine had been tended like a garden and the garden had answered with twenty-three quiet years, the page's final line the supervisor's own, that she had plugged home ten thousand evenings and would remember the last one by its kettle, the exchange's whole automation, the archive's card says, kept in steam, copper and one operator's arithmetic. The page's own binding, sewn by the operators themselves with the board's last connection wire, keeps the story and the tally in one cover, the exchange's whole automation, the archive's card says, closed by the hands that held it open, the supervisor's retirement gift being the kettle itself, descaled and polished, the journal's last photograph showing the four night operators ranked behind it, the story's true frontispiece and its only necessary instrument.")

FR301 = [
 ("The story ends with the mayor's call completed third time and {v} engines sent anyway.", "one", "two"),
 ("The story ends with the board's {v} operators retrained and the kettle kept boiling.", "two hundred", "one hundred"),
 ("The story ends with the first year connecting {v} lines and the town talking to itself.", "nine thousand", "eight thousand"),
 ("The story ends with the fault rate entered as one call lost in {v}.", "five hundred", "one hundred"),
 ("The story ends with the manual board's old rate remembered as one in {v}, and forgiven.", "eighty", "forty"),
 ("The story ends with the night service kept by {v} operators through the changeover.", "four", "three"),
 ("The story ends with the kettle rewired {v} times before its own socket was granted.", "three", "two"),
 ("The story ends with the last call completed at {v} minutes past nine in the evening.", "nine", "seven"),
 ("The story ends with the board closed after {v} years and not one call lost.", "twenty-three", "twenty"),
 ("The story ends with the closing tally reading {v} operators, one kettle and zero lost calls.", "four", "five"),
 ("The story ends with the mayor's third dial reaching the fire station and {v} engine attending.", "one", "three"),
 ("The story ends with the exchange's {v} lines each tested once and the town unaware.", "nine thousand", "ten thousand"),
 ("The story ends with the fault book closed at one in {v} and framed beside the kettle.", "five hundred", "six hundred"),
 ("The story ends with the night kettle granted its socket after {v} rewirings.", "three", "four"),
 ("The story ends with the last connection timed at nine past nine and the subscriber's {v} still ticking.", "clock", "radio"),
]
ROWS301 = [st_row("P1" if i % 2 == 0 else "P2", f, c, w, i % 2)
           for i, (f, c, w) in enumerate(FR301)]
S301c = S301b.replace("[2] Title: The Story's Last Entry", "[3] Title: The Story's Last Entry, Completed", 1) + " The completion's own toast, drunk at the retirement party from the kettle's own saucers, gave the story its last sound: the operators' laughter, the journal's reporter noted, louder than the board's whole twenty-three years of clicks, the exchange's automation, the party's last entry records, retired to applause."
build_st(301, "news", {"P1": (S301a, S301b, S301c), "P2": (S301a, S301b, S301c)}, ROWS301)
