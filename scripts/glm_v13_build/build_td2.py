import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from td_lib import build

OPT = ["Session 1", "Session 2", "Session 3", "Session 4"]
GREET = {
 1: ("Agent", "Good morning, the records line.", "The dates, straight from the register."),
 2: ("Agent", "Records line.", "The register is open."),
 3: ("Agent", "Good afternoon.", "The register is open."),
 4: ("Agent", "Records line.", "The last page, then."),
}
DRE = re.compile(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}")

def gen(fno, sessions, n_mcq=11, n_ft=9):
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
        P.append(f"Speaker: {who}: Every page. This session was opened on {sdate}, and the register's own note for the day is filed with it.\\n")
        P.append(f"Speaker: {who}: The register's callers are the register's other half, the town asking and the book answering, since before this session's date of {sdate}.\\n")
    txt = ("\n".join(P)).replace("\\n", "\n")
    rows = []
    allf = [(no, fp, DRE.search(sent).group(0)) for no, _, _, _, facts in sessions for fp, sent in facts]
    for i, (no, fp, d) in enumerate(allf[:n_mcq]):
        rat = f"Session {no} records {fp} on {d}."
        rows.append(dict(kind="mcq", pid="P1" if i % 2 == 0 else "P2",
                         stem=f"Which session records {fp}?",
                         opts=OPT, gold_idx=no - 1, rat=rat,
                         slot=[0, 1, 2, 3][i % 4], guard=d))
    for j, (no, fp, d) in enumerate(allf[:n_ft]):
        rat = f"Session {no} states that {fp} is dated {d}."
        rows.append(dict(kind="ft", pid="P2" if j % 2 == 0 else "P1",
                         q=f"When did {fp} take place, according to the conversation?",
                         gold=d, rat=rat, guard=d))
    return {"P1": txt, "P2": txt}, rows

FILES = {
 282: ("banking", [
   (1, "9:10 am", "June 8, 2015", "The branch's history board needs the dates.", [
     ("the launch of the mobile app", "The mobile app was launched on June 12, 2020, and the contactless limit was raised on September 30, 2020"),
     ("the start of the video adviser service", "The video adviser service started on January 25, 2021"),
     ("the opening of the branch's quiet room", "The quiet room opened on October 4, 2021"),
   ]),
   (2, "10:50 am", "March 3, 2016", "The rate saver account, when?", [
     ("the installation of the fraud block system", "The fraud block system was installed on October 8, 2021, and the branch refurbishment was finished on February 14, 2022"),
     ("the launch of the rate saver account", "The rate saver account was launched on November 21, 2022"),
     ("the branch's hundredth anniversary tea", "The hundredth anniversary tea was held on May 12, 2021"),
   ]),
   (3, "2:25 pm", "November 7, 2019", "The paperless switch, did it have a date?", [
     ("the adoption of paperless statements", "Paperless statements became the default on March 15, 2023"),
     ("the opening of the community banking hub", "The community banking hub opened on August 29, 2023"),
     ("the last chip-and-signature card", "The last chip-and-signature card was issued on May 5, 2023"),
   ]),
   (4, "11:15 am", "April 22, 2021", "The cash deposit machines?", [
     ("the installation of the cash deposit machines", "The cash deposit machines were installed on July 6, 2022, and the hub's opening hours were extended on January 9, 2024"),
     ("the branch's school savings scheme", "The school savings scheme began on September 27, 2022"),
   ]),
 ]),
 283: ("shipping", [
   (1, "9:05 am", "May 18, 1993", "The port's guidebook wants the dates.", [
     ("the opening of the roll-on berth", "The roll-on berth was opened on February 14, 1994, and the grain terminal followed on October 2, 1995"),
     ("the commissioning of the first tug", "The first tug, the Kittiwake, was commissioned on June 7, 1996"),
     ("the port's dredging of the outer channel", "The outer channel was dredged on March 19, 1994"),
   ]),
   (2, "10:35 am", "September 9, 1997", "The ferry service, when did it start?", [
     ("the launch of the island ferry service", "The island ferry service was launched on April 11, 1998, and the ferry terminal was opened on May 23, 1999"),
     ("the installation of the first gantry crane", "The first gantry crane, the Atlas, was installed on September 3, 2000"),
     ("the port's first cruise call", "The first cruise call was logged on July 8, 1998"),
   ]),
   (3, "2:20 pm", "January 14, 2001", "The rail spur, was that us or the county?", [
     ("the building of the rail spur", "The rail spur was built on June 20, 2002, and the intermodal yard opened on February 27, 2003"),
     ("the port's ISO certification", "The port's ISO certification was granted on November 16, 2001"),
     ("the second tug, the Fulmar", "The second tug, the Fulmar, was commissioned on August 30, 2002"),
   ]),
   (4, "11:40 am", "June 30, 2004", "The last page's dates now.", [
     ("the opening of the cold store", "The cold store was opened on October 12, 2004"),
     ("the port's new seal", "The port's new seal was adopted on January 5, 2004"),
   ]),
 ]),
 284: ("archaeology", [
   (1, "9:15 am", "July 4, 1938", "The museum's founder wants the dig dates.", [
     ("the opening of the first season's trench", "The first season's trench was opened on August 8, 1938, and the villa's mosaic was found on September 2, 1938"),
     ("the visit of the county's inspector", "The county's inspector visited on October 16, 1938"),
   ]),
   (2, "10:25 am", "June 12, 1952", "The war years, anything kept?", [
     ("the reinterment of the wartime stores", "The wartime stores were reinterred on March 5, 1952, and the dig's second season opened on June 20, 1952"),
     ("the finding of the hoard", "The hoard was found on July 9, 1952"),
   ]),
   (3, "2:10 pm", "May 6, 1961", "The museum building itself?", [
     ("the laying of the museum's foundation", "The museum's foundation was laid on June 18, 1962, and the museum opened on April 30, 1963"),
     ("the last season's trench", "The last season's trench closed on August 27, 1965"),
   ]),
   (4, "11:30 am", "September 2, 1965", "Anything after the last trench?", [
     ("the planting of the dig's memorial orchard", "The memorial orchard was planted on October 19, 1965"),
     ("the printing of the dig's full report", "The full report was printed on December 8, 1965"),
   ]),
 ]),
}

for fno, (_, sessions) in sorted(FILES.items()):
    subs, rows = gen(fno, sessions, n_mcq=11, n_ft=9)
    if len(rows) > 5 and fno == 284:
        rows = rows[:5]
    build(fno, subs, rows)
