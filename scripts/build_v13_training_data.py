"""Build the v13 mixture: v12 (minus wrong TimeQA golds) + AUG_TPL3 + AUG_PROG.

Provenance (docs/audit_2026_10_04.md §1.1, corrected 2026-10-05): the v13
wave in data/manual_aug_glm_v13/ was produced by the GLM model working
through the opencode agent, in two ways:

    AUG_GLM3, provenance "glm-agent"      -- rows written one by one by the
        model (the first-sitting packets and the parts/ / build_* files, which
        only format text the model wrote);
    AUG_TPL3, provenance "agent-template" -- rows from random slot-filling
        generators (gen_rel, gen_dialogue, gen_stories, gen_reasoning,
        build105/106).

AUG_PROG keeps "AUG_PROG", provenance "programmatic". The raw files keep
"AUG_GLM2" because ingest_glm_batch.check() requires it; the gate runs on the
raw rows before relabelling.

Quality filters (all deterministic, each counted in manifest.json):
  - storytelling (TPL3): gold-longer share forced to 50% -- keep every row
    whose gold ending is not the longer one, plus a seeded equal-size subset
    of gold-longer rows (audit §2.1: 83% gold-longer, a length shortcut);
  - relation (TPL3): BEFORE/AFTER/SIMULTANEOUS/INCLUDES/IS_INCLUDED
    recomputed from the parsed dates (event-event and event-to-time); rows
    whose label disagrees are DROPPED, not flipped -- their rationales are
    reversed too (audit §2.2);
  - duration (TPL3): drop the broken "How long did it take the X to
    take/last?" template and rows whose options share one number and differ
    only by unit (audit §2.3);
  - prog_timeline: drop rows where a year-only fact shares its year with
    another fact -- the order is decided by hidden day-level dates
    (audit §2.5);
  - prog_duration_compare: drop the event-named rows whose context dates
    only one event per pair (audit §2.5);
  - TimeQA (v12 base): drop the rows the CoT pass documented as SKIP
    (passage contradicts / does not support the gold, `# SKIP gNNNN:` lines
    in data/cot_raw/), matched by (question, context) via
    data/cot_packets/answer_key.json (audit §2.6). combined_80_20_v12
    itself is not modified.

Template-source rule (researcher decision 2026-10-05): template-generated
rows are REMOVED unless they are math (MATH_CATEGORIES). This applies to
AUG_TPL3 and to the v12-base rows from glm_raw 296/297/309 (syn_gen.py);
GLM-written rows are kept. Every row must also survive the independent math
re-verification (data/v13_verify/failed_questions.json) and the human-style
review (data/v13_verify/review_removed.json: WRONG / AMBIGUOUS / MALFORMED).

New rows split 80/20 by the same stable sha1(question) rule as before.

Manifest keys: glm3_rows / template_rows count the two v13 wave sources;
glm_rows mirrors glm3_rows for older readers.

Usage: venv/bin/python scripts/build_v13_training_data.py [--allow-no-template]
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "data" / "combined_80_20_v12"
TPL_NEW = ROOT / "data" / "manual_aug_glm_v13"
PROG = ROOT / "data" / "prog_aug_v13"
OUT = ROOT / "data" / "combined_80_20_v13"
COT_RAW = ROOT / "data" / "cot_raw"
ANSWER_KEY = ROOT / "data" / "cot_packets" / "answer_key.json"
STORY_SEED = 20261005
GLM_RAW = ROOT / "data" / "glm_raw"
# glm_raw files written by a random slot-filling generator (scripts/glm_v13_build/
# syn_gen.py). Every other file in 271-313 was written row by row by the GLM
# model (via the opencode agent) and only formatted by the build_*/ *_lib.py
# scripts, so it is GLM-authored data, not template output.
SCRIPT_BUILT_FILES = (296, 297, 309)
V13_RAW = ROOT / "data" / "glm_raw_v13"
V13_PLAN = ROOT / "data" / "glm_packets_v13" / "_plan.json"
REVIEW_REMOVED = ROOT / "data" / "v13_verify" / "review_removed.json"
# v13 packets written by random slot-filling generators (gen_rel / gen_dialogue /
# gen_stories / build105-106); gen_reasoning's packets are added from the plan.
TEMPLATE_V13_PACKETS = {"003", "004", "005", "006", "033", "034", "035",
                        "037", "038", "039", "105", "106"}
GEN_REASONING_SKIPPED = {"041", "057", "069", "083", "095"}   # GLM-written first sitting


def template_packets(plan: Path = V13_PLAN) -> set[str]:
    out = set(TEMPLATE_V13_PACKETS)
    if plan.exists():
        out |= {x["packet"][:3] for x in json.loads(plan.read_text())["plan"]
                if x["category"] in MATH_CATEGORIES and x["packet"][:3] not in GEN_REASONING_SKIPPED}
    return out


def v13_question_packets(raw: Path = V13_RAW) -> dict[str, str]:
    """question -> 3-digit packet id, from the raw v13 replies."""
    out = {}
    for p in sorted(raw.glob("*.txt")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "question" in r:
                    out[r["question"].strip()] = p.name[:3]
    return out


def review_removed(path: Path = REVIEW_REMOVED) -> set[str]:
    """Questions the human-style review judged WRONG / AMBIGUOUS / MALFORMED."""
    return set(json.loads(path.read_text())) if path.exists() else set()
VERIFY_FAILED = ROOT / "data" / "v13_verify" / "failed_questions.json"
MATH_CATEGORIES = {"Computation", "Timeline", "Duration_Compare", "Order_Compare",
                   "Relative_Reasoning"}


def script_built_questions(raw: Path = GLM_RAW, files=SCRIPT_BUILT_FILES) -> dict[str, str]:
    """question -> category for every row in the given glm_raw files
    (default: the template-generated ones)."""
    out = {}
    for n in files:
        p = raw / f"{n:03d}.txt"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "question" in r:
                    out[r["question"].strip()] = r.get("category")
    return out


def verify_failed(path: Path = VERIFY_FAILED) -> set[str]:
    return set(json.loads(path.read_text())) if path.exists() else set()

# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})\b")
_OPT = re.compile(r"^([A-E])\.\s+(.*)$", re.M)


def parse_dates(s: str) -> list[dt.date]:
    """Full dates (D Month YYYY / Month D, YYYY) in order of appearance."""
    found = []
    for m in _DMY.finditer(s):
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            try:
                found.append((m.start(), dt.date(int(m.group(3)), mo, int(m.group(1)))))
            except ValueError:
                pass
    for m in _MDY.finditer(s):
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            try:
                found.append((m.start(), dt.date(int(m.group(3)), mo, int(m.group(2)))))
            except ValueError:
                pass
    return [d for _, d in sorted(found)]


def options(q: str) -> list[str]:
    tail = q.split("Choices:")[-1] if "Choices:" in q else q
    return [m.group(2).strip() for m in _OPT.finditer(tail)]


def _time_span(t: str):
    t = t.strip()
    d = parse_dates(t)
    if d:
        return d[0], d[0]
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", t)
    if m and _MONTHS.get(m.group(1).lower()):
        y, mo = int(m.group(2)), _MONTHS[m.group(1).lower()]
        return dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])
    if re.fullmatch(r"\d{4}", t):
        return dt.date(int(t), 1, 1), dt.date(int(t), 12, 31)
    return None


def computed_relation(q: str) -> str | None:
    """Relation of the first-mentioned event to the second (or to the time),
    from parsed dates; None when the row's shape can't be parsed reliably."""
    body = q.split("What is the relationship")[0]
    m = re.search(r"and the time '([^']+)'", q)
    if m:                                             # event-to-time
        ev, sp = parse_dates(body), _time_span(m.group(1))
        if len(ev) != 1 or not sp:
            return None
        p, (lo, hi) = ev[0], sp
        if lo == hi:
            return "BEFORE" if p < lo else "AFTER" if p > lo else "SIMULTANEOUS"
        return "BEFORE" if p < lo else "AFTER" if p > hi else "IS_INCLUDED"
    sents = [s for s in re.split(r"(?<=\d)\.\s+", body.strip()) if s.strip()]
    if len(sents) != 2:
        return None
    e1, e2 = parse_dates(sents[0]), parse_dates(sents[1])
    if len(e1) == 1 and len(e2) == 1:
        a, b = e1[0], e2[0]
        return "BEFORE" if a < b else "AFTER" if a > b else "SIMULTANEOUS"
    if len(e1) == 2 and len(e2) == 1:
        lo, hi = sorted(e1)
        p = e2[0]
        return "INCLUDES" if lo <= p <= hi else ("AFTER" if p < lo else "BEFORE")
    if len(e1) == 1 and len(e2) == 2:
        lo, hi = sorted(e2)
        p = e1[0]
        return "IS_INCLUDED" if lo <= p <= hi else ("BEFORE" if p < lo else "AFTER")
    if len(e1) == 2 and len(e2) == 2:
        a, b = sorted(e1)
        c, d = sorted(e2)
        if (a, b) == (c, d):
            return None
        if c <= a and b <= d:
            return "IS_INCLUDED"
        if a <= c and d <= b:
            return "INCLUDES"
        if b < c:
            return "BEFORE"
        if a > d:
            return "AFTER"
    return None


_BROKEN_DURATION = re.compile(r"How long did it take the .+? to (take|last)\?")
_UNIT = r"(seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|centur(?:y|ies))"


def duration_unit_only(q: str) -> bool:
    """All options are the same number with only the unit changing."""
    opts = options(q)
    if len(opts) < 2:
        return False
    stems = set()
    for o in opts:
        m = re.fullmatch(r"(.+?)\s+" + _UNIT, o.strip().rstrip("."), re.I)
        if not m:
            return False
        stems.add(m.group(1).lower())
    return len(stems) == 1


def timeline_underdetermined(q: str) -> bool:
    """A year-only fact ('... in 1947.') shares its year with another fact."""
    facts = [m.group(2) for m in _OPT.finditer(q)]
    years, year_only = [], []
    for f in facts:
        ys = re.findall(r"\b(1[5-9]\d\d|20\d\d)\b", f)
        years.append(ys[-1] if ys else None)
        year_only.append(bool(re.search(r"\bin \d{4}\.$", f.strip())))
    return any(year_only[i] and years[i] is not None and
               any(years[j] == years[i] for j in range(len(facts)) if j != i)
               for i in range(len(facts)))


def duration_compare_incoherent(r: dict) -> bool:
    q = r["question"]
    return ("*Duration 1:* Between the " in q or "*Duration 2:* Between the " in q)


def story_gold_longer(r: dict) -> bool | None:
    opts, gold = options(r["question"]), r["targets"][0]
    others = [o for o in opts if o != gold]
    if len(opts) != 2 or len(others) != 1:
        return None
    return len(gold) > len(others[0])


def cot_skip_keys(cot_raw: Path = COT_RAW, answer_key: Path = ANSWER_KEY):
    """{(question, context)} of TimeQA rows the CoT pass documented as SKIP."""
    if not answer_key.exists():
        return set(), {}
    key = json.loads(answer_key.read_text(encoding="utf-8"))
    skips = {}
    for f in sorted(glob.glob(str(cot_raw / "*.txt"))):
        for line in open(f, encoding="utf-8"):
            m = re.match(r"\s*#\s*SKIP\s+(g\d+)\s*[:\-]?\s*(.*)", line)
            if m:
                skips.setdefault(m.group(1), m.group(2).strip())
    keys = {(key[g]["question"], key[g]["context"]) for g in skips if g in key}
    return keys, skips


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def filter_template_rows(rows: list[dict], counts: Counter,
                         failed: set[str] = frozenset()) -> list[dict]:
    keep, story_long, story_rest = [], [], []
    for r in rows:
        cat = r.get("category")
        if r.get("source_dataset") == "AUG_TPL3" and cat not in MATH_CATEGORIES:
            counts[f"tpl_non_math_template_removed_{cat}"] += 1
            continue
        if r["question"].strip() in failed:
            counts[f"tpl_math_failed_independent_verify_{cat}"] += 1
            continue
        if cat == "relation":
            c = computed_relation(r["question"])
            if c is not None and c != r["targets"][0]:
                counts["tpl_relation_label_disagrees_with_dates"] += 1
                continue
        elif cat == "duration":
            if _BROKEN_DURATION.search(r["question"]):
                counts["tpl_duration_broken_template"] += 1
                continue
            # easy is not wrong: the unit-only rule applies to template rows only
            if r.get("source_dataset") == "AUG_TPL3" and duration_unit_only(r["question"]):
                counts["tpl_duration_options_differ_only_by_unit"] += 1
                continue
        elif cat == "storytelling" and r.get("source_dataset") == "AUG_TPL3":
            # length-shortcut balancing is a template-data fix; GLM-written
            # stories are kept unless the review finds them wrong
            longer = story_gold_longer(r)
            (story_long if longer else story_rest).append(r)
            continue
        keep.append(r)
    rng = random.Random(STORY_SEED)
    k = min(len(story_rest), len(story_long))
    sampled = rng.sample(story_long, k) if k else []
    counts["tpl_storytelling_gold_longer_dropped_for_50pct_balance"] += len(story_long) - k
    if len(story_rest) > len(story_long):        # never the case today; keep balance anyway
        counts["tpl_storytelling_gold_shorter_dropped_for_balance"] += len(story_rest) - k
        story_rest = rng.sample(story_rest, k)
    keep.extend(story_rest + sampled)
    return keep


def filter_prog_rows(rows: list[dict], counts: Counter,
                     failed: set[str] = frozenset()) -> list[dict]:
    keep = []
    for r in rows:
        cat = r.get("category")
        if r["question"].strip() in failed:
            counts[f"prog_failed_independent_verify_{cat}"] += 1
            continue
        if cat == "prog_timeline" and timeline_underdetermined(r["question"]):
            counts["prog_timeline_year_only_collision"] += 1
            continue
        if cat == "prog_duration_compare" and duration_compare_incoherent(r):
            counts["prog_duration_compare_event_named_incoherent"] += 1
            continue
        keep.append(r)
    return keep


def filter_base_rows(rows: list[dict], skip_keys: set, counts: Counter, split: str,
                     scripted: dict[str, str] | None = None,
                     failed: set[str] = frozenset()) -> list[dict]:
    scripted = scripted or {}
    keep = []
    for r in rows:
        if r.get("source_dataset") == "TimeQA" and \
                (r.get("question"), r.get("context", "")) in skip_keys:
            counts[f"timeqa_cot_skip_wrong_gold_{split}"] += 1
            continue
        q = r.get("question", "").strip()
        if q in failed and q not in scripted:
            counts[f"v12_failed_verify_or_review_{r.get('source_dataset')}_{split}"] += 1
            continue
        if r.get("source_dataset") == "AUG_GLM2" and q in scripted:
            cat = r.get("category")
            if cat not in MATH_CATEGORIES:
                counts[f"v12_script_built_non_math_removed_{split}"] += 1
                continue
            if q in failed:
                counts[f"v12_script_built_math_failed_verify_{split}"] += 1
                continue
            r["provenance"] = "agent-template"
        keep.append(r)
    return keep


def split_of(r: dict) -> str:
    h = int(hashlib.sha1(r["question"].encode()).hexdigest(), 16)
    return "val" if h % 5 == 0 else "train"


def main() -> int:
    from build_v11_parity_data import bench_index, contaminated
    from ingest_glm_batch import check as gate_check

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-no-template", "--allow-no-glm", dest="allow_empty",
                    action="store_true",
                    help="build without AUG_TPL3 rows (pipeline validation only)")
    args = ap.parse_args()
    counts: Counter = Counter()

    # --- AUG_TPL3 (agent-written template rows) -------------------------
    tpl = []
    if TPL_NEW.exists():
        tpl = [json.loads(l) for f in sorted(TPL_NEW.glob("*.jsonl"))
               for l in open(f, encoding="utf-8") if l.strip()]
    if not tpl and not args.allow_empty and not os.environ.get("V13_ALLOW_NO_GLM"):
        raise SystemExit(f"FATAL: no rows in {TPL_NEW}, or pass --allow-no-template "
                         f"for a provisional pipeline-validation build.")
    bad = [r for r in tpl if gate_check(r)]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} banked rows fail the gate, e.g. {gate_check(bad[0])}")
    q2p, tpl_packets = v13_question_packets(), template_packets()
    for r in tpl:
        if q2p.get(r["question"].strip()) in tpl_packets:
            r["source_dataset"], r["provenance"] = "AUG_TPL3", "agent-template"
        else:   # written row by row by the GLM model (opencode), incl. first-sitting math
            r["source_dataset"], r["provenance"] = "AUG_GLM3", "glm-agent"
    tpl_in = len(tpl)
    removed_by_review = review_removed()
    failed = verify_failed() | removed_by_review
    scripted = script_built_questions()
    tpl = filter_template_rows(tpl, counts, failed)

    # --- AUG_PROG ---------------------------------------------------------
    audit = json.loads((PROG / "audit.json").read_text())
    if audit.get("errors"):
        raise SystemExit(f"FATAL: prog generator audit recorded {len(audit['errors'])} error(s)")
    prog = [json.loads(l) for f in sorted(PROG.glob("prog_*.jsonl"))
            for l in open(f, encoding="utf-8") if l.strip()]
    for r in prog:
        r["provenance"] = "programmatic"
    prog_in = len(prog)
    prog = filter_prog_rows(prog, counts, failed)

    new_rows = tpl + prog
    idx = bench_index()
    clean = [r for r in new_rows if not contaminated(r, *idx)]
    counts["contaminated_dropped"] = len(new_rows) - len(clean)
    skip_keys, skips = cot_skip_keys()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "data/combined_80_20_v12 (minus CoT-flagged wrong TimeQA golds, minus "
                  "the template-generated non-math rows of glm_raw 296/297/309) + "
                  "data/manual_aug_glm_v13 split by authorship: AUG_GLM3 (written row by "
                  "row by the GLM model via the opencode agent) and AUG_TPL3 (random "
                  "slot-filling generators; math only) + data/prog_aug_v13 (AUG_PROG)",
        "v13_wave_rows_in": tpl_in,
        "glm3_rows": sum(r["source_dataset"] == "AUG_GLM3" for r in clean),
        "template_rows": sum(r["source_dataset"] == "AUG_TPL3" for r in clean),
        "review_removed_questions": len(removed_by_review),
        "prog_rows_in": prog_in,
        "prog_rows": sum(r["source_dataset"] == "AUG_PROG" for r in clean),
        "cot_skip_ids_used": len(skips),
        "filters": {},
        "provisional": not tpl,
        "splits": {}}
    manifest["glm_rows"] = manifest["glm3_rows"]
    for split in ("train", "val"):
        base_all = [json.loads(l) for l in open(SRC / f"{split}.jsonl", encoding="utf-8")]
        base = filter_base_rows(base_all, skip_keys, counts, split, scripted, failed)
        added = [r for r in clean if split_of(r) == split]
        rows = base + added
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "from_v12": len(base_all), "v12_kept": len(base), "added": len(added),
            "out": len(rows),
            "added_by_source": dict(Counter(r.get("source_dataset") for r in added)),
            "added_by_category": dict(sorted(Counter(r.get("category") for r in added).items())),
            "out_by_source": dict(Counter(r.get("source_dataset") for r in rows))}
        print(split, manifest["splits"][split]["out"], manifest["splits"][split]["out_by_source"])
    manifest["filters"] = dict(sorted(counts.items()))
    print("filters:", manifest["filters"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_version_folder.py"),
                    "v13", "data/combined_80_20_v13", "--story",
                    "v12 (minus 207 TimeQA rows whose gold the CoT pass found contradicted "
                    "or unsupported by the passage, and minus the non-math rows that came "
                    "from the agent's script-built glm_raw files 271-313) + AUG_TPL3 MATH "
                    "ONLY: agent-written Python template rows (scripts/glm_v13_build/; NOT "
                    "GLM-5.2 chat -- audit_2026_10_04 §1.1) for Computation, Timeline, "
                    "Relative_Reasoning, Duration_Compare, Order_Compare; every non-math "
                    "template row removed by researcher decision + AUG_PROG programmatic "
                    "math/temporal data (deliberately non-LLM, kept). Every kept math row "
                    "passed an independent re-verification (scripts/verify_v13_math.py); "
                    "filter counts in data/manifest.json."],
                   check=True)
    if not tpl:
        print("\nNOTE: provisional build (no template rows). The schedule refuses it "
              "unless V13_ALLOW_NO_GLM=1 is exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
