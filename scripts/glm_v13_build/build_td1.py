import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from td_lib import build

OPT = ["Session 1", "Session 2", "Session 3", "Session 4"]
GREET = {
 1: ("Agent", "Good morning, the records line.", "The dates, straight from the register."),
 2: ("Agent", "Records line.", "The register is open."),
 3: ("Agent", "Good afternoon.", "The register is open."),
 4: ("Agent", "Records line.", "The last page, then."),
}

def gen(fno, sessions, n_mcq=11, n_ft=9):
    """sessions: list of (no, time, date, opener, [(fact_phrase, sent)...])"""
    P = []
    for no, time, sdate, opener, facts in sessions:
        who, g1, g2 = GREET[no]
        P.append(f"Session {no} happened at {time} on {sdate}.\\n")
        P.append(f"Speaker: {who}: {g1}\\n")
        P.append(f"Speaker: Caller: {opener}\\n")
        P.append(f"Speaker: {who}: {g2}\\n")
        for _, sent in facts:
            P.append(f"Speaker: {who}: {sent}\\n")
        P.append(f"Speaker: Caller: And the register keeps the rest?\\n")
        P.append(f"Speaker: {who}: The register's callers are the register's other half, the town asking and the book answering, since before this session's date of {sdate}.\\n")
        P.append(f"Speaker: {who}: Every page. This session was opened on {sdate}, and the register's own note for the day is filed with it.\\n")
    passage = (" ".join(x.strip('"') for x in []) ) # noop
    txt = ("\n".join(P)).replace("\\n", "\n")
    rows = []
    DRE = re.compile(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}")
    allf = [(no, fp, DRE.search(sent).group(0)) for no, _, _, _, facts in sessions for fp, sent in facts]
    # MCQ over first n_mcq facts
    for i, (no, fp, d) in enumerate(allf[:n_mcq]):
        rat = f"Session {no} records {fp} on {d}."
        rows.append(dict(kind="mcq", pid="P1" if i % 2 == 0 else "P2",
                         stem=f"Which session records {fp}?",
                         opts=OPT, gold_idx=no - 1, rat=rat,
                         slot=[0, 1, 2, 3][i % 4], guard=d))
    # FT over all facts with a second question template
    for j, (no, fp, d) in enumerate(allf[:n_ft]):
        rat = f"Session {no} states that {fp} is dated {d}."
        rows.append(dict(kind="ft", pid="P2" if j % 2 == 0 else "P1",
                         q=f"When did {fp} take place, according to the conversation?",
                         gold=d, rat=rat, guard=d))
    return {("P1"): txt, "P2": txt}, rows

FILES = {
 278: ("courts", [
   (1, "9:12 am", "January 20, 2021", "The centenary page needs the court's dates.", [
     ("the beginning of the virtual hearings", "The virtual hearings began on April 6, 2020, and the jury restart was managed on August 3, 2020"),
     ("the opening of the extended hours pilot", "The extended hours pilot opened on March 15, 2021"),
     ("the installation of the court's digital kiosks", "The digital kiosks were installed on September 2, 2021"),
   ]),
   (2, "10:40 am", "June 7, 2021", "When did the reparations panel first sit?", [
     ("the reparations panel's first sitting", "The panel's first sitting was on October 12, 2021, and its first orders were published on December 3, 2021"),
     ("the video links being made permanent", "The video links were made permanent on May 9, 2022"),
     ("the hiring of the panel's clerks", "The panel's clerks were hired on July 21, 2021"),
   ]),
   (3, "2:15 pm", "February 14, 2022", "Did the panel's report have a date?", [
     ("the publication of the panel's full report", "The full report came on September 6, 2022"),
     ("the holding of the court's open day", "The open day was held on June 21, 2023"),
     ("the retirement of the court's oldest ledger", "The oldest ledger was retired on March 9, 2023"),
   ]),
   (4, "11:05 am", "April 3, 2023", "The digital sentencing statements?", [
     ("the issuing of the first digital statements", "The first digital statements were issued on November 17, 2023, and the paper register closed on January 31, 2024"),
     ("the opening of the court's archive room", "The archive room opened on May 16, 2024"),
   ]),
 ]),
 279: ("climate", [
   (1, "9:05 am", "March 11, 2006", "The flood board wants the station's story with dates.", [
     ("the reopening of the weather station", "The station was reopened on May 4, 2005, and the automatic gauges were installed on September 19, 2005"),
     ("the logging of the January storm", "The January storm was logged on January 12, 2006"),
     ("the planting of the station's hedge", "The station's hedge was planted on April 3, 2006"),
   ]),
   (2, "10:30 am", "July 8, 2007", "The summer flood, when exactly?", [
     ("the summer flood's peak", "The flood peaked on July 20, 2007, and the warning siren was reinstated on August 15, 2007"),
     ("the commissioning of the river gauge", "The river gauge was commissioned on February 2, 2008"),
     ("the dredging of the lower river", "The lower river was dredged on October 19, 2007"),
   ]),
   (3, "2:20 pm", "November 5, 2009", "The drought years, which do I write down?", [
     ("the beginning of the 2006 dry spell", "The 2006 dry spell began on June 10, 2006"),
     ("the beginning of the second dry spell", "The second dry spell began on May 6, 2010"),
     ("the holding of the first schools open day", "The first schools open day was held on September 14, 2011"),
   ]),
   (4, "3:45 pm", "February 19, 2012", "The wind farm's met mast?", [
     ("the raising of the met mast", "The met mast went up on October 8, 2012, and the storm naming began on November 1, 2013"),
     ("the publication of the station's first almanac", "The station's first almanac was published on December 5, 2013"),
   ]),
 ]),
 280: ("energy", [
   (1, "9:20 am", "April 14, 1967", "The anniversary supplement wants the station's dates.", [
     ("the commissioning of the second turbine", "The second turbine was commissioned on October 2, 1967, and the cooling towers were finished on August 19, 1969"),
     ("the completion of the original turbine's overhaul", "The original turbine's overhaul was completed on June 30, 1970"),
     ("the building of the station's cricket pitch", "The station's cricket pitch was laid on June 5, 1968"),
   ]),
   (2, "10:55 am", "September 3, 1971", "The flue gas scrubbers?", [
     ("the fitting of the flue gas scrubbers", "The scrubbers were fitted on March 27, 1972, and the monitoring station opened on July 14, 1974"),
     ("the dedication of the miners' memorial", "The miners' memorial was dedicated on May 5, 1975"),
     ("the planting of the station's orchard", "The station's orchard was planted on November 2, 1972"),
   ]),
   (3, "2:10 pm", "January 18, 1978", "The control room refit?", [
     ("the commissioning of the new control room", "The new control room was commissioned on September 1, 1978, and the open days began on June 9, 1979"),
     ("the setting of the efficiency record", "The efficiency record was set on February 23, 1983"),
     ("the station's fiftieth anniversary dinner", "The fiftieth anniversary dinner was held on June 2, 1978"),
   ]),
   (4, "11:30 am", "May 22, 1986", "The district heating?", [
     ("the opening of the district heating scheme", "The district heating opened on October 9, 1986, and the visitor centre opened on April 17, 1987"),
     ("the station's last coal delivery", "The last coal delivery was received on March 20, 1988"),
   ]),
 ]),
 281: ("tech", [
   (1, "9:15 am", "May 6, 1892", "The town journal wants the exchange's dates.", [
     ("the opening of the first exchange", "The first exchange opened on March 3, 1892, and the trunk line was connected on November 21, 1893"),
     ("the beginning of the night switchboard", "The night switchboard began on February 7, 1895"),
     ("the exchange's first directory", "The first directory was printed on December 12, 1892"),
   ]),
   (2, "10:45 am", "September 12, 1899", "The underground cables, when?", [
     ("the laying of the underground cables", "The cables were laid on June 14, 1901, and the common battery system was installed on October 2, 1902"),
     ("the cutting over of the automatic exchange", "The automatic exchange was cut over on April 28, 1912"),
     ("the exchange's thousandth subscriber", "The thousandth subscriber was connected on March 8, 1903"),
   ]),
   (3, "2:30 pm", "January 9, 1913", "The coast radio station?", [
     ("the opening of the coast radio station", "The coast radio station opened on September 1, 1913, and the first radio broadcast was heard on June 16, 1923"),
     ("the completion of the universal dial service", "The universal dial service was completed on May 30, 1930"),
     ("the exchange's war-time operators' badge", "The operators' badge was struck on June 1, 1920"),
   ]),
   (4, "11:20 am", "October 17, 1931", "The speaking clock?", [
     ("the beginning of the speaking clock", "The speaking clock began on January 13, 1932, and the telex service followed on August 24, 1934"),
     ("the printing of the exchange's last crank directory", "The last crank directory was printed on June 9, 1933"),
   ]),
 ]),
}

for fno, (_, sessions) in sorted(FILES.items()):
    subs, rows = gen(fno, sessions)
    build(fno, subs, rows)
