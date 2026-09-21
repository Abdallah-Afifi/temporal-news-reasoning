"""Validate and ingest GLM chat replies into the AUG_GLM2 slice.

Replaces scripts/validate_glm_news.py for the v9 ruleset. That script is v8-era
and would reject nearly every correct row: it REQUIRES dual gold
`[text, LETTER]`, which §2 now forbids ("targets is single-element", removed in
v7c) and §8 lists as prohibited; it insists on exactly four options when §3
allows three; and it knows only 3 of the 18 categories.

Usage:
    venv/bin/python scripts/ingest_glm_batch.py data/glm_raw/*.txt
    venv/bin/python scripts/ingest_glm_batch.py --status

Accepted rows are appended to data/manual_aug_glm/<Category>_batchNN.jsonl.
Rejected rows are written to data/glm_raw/_rejected.jsonl with a reason, so a
bad batch can be re-prompted rather than silently lost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from glm_client import extract_jsonl  # noqa: E402

REQUIRED = {"source_dataset", "slice", "category", "provenance", "question",
            "context", "targets", "rationale", "source", "source_id"}
NO_CONTEXT = {"relation", "ordering", "duration"}
LABELS = {
    "nli_saq": {"entailment", "neutral", "contradiction"},
    "nli_mcq": {"entailment", "neutral", "contradiction"},
    "relation": {"IDENTITY", "BEFORE", "DURING"},
    "ordering": {"TRUE", "Undetermined", "FALSE"},
}
# Categories that must carry a Choices block, and how many options.
MCQ_OPTS = {"Timeline": None, "Duration_Compare": 3, "Order_Compare": 3,
            "nli_mcq": 3, "relation": 3, "ordering": 3, "duration": 4,
            "storytelling": 2}
NEVER_MCQ = {"Computation", "Localization", "nli_saq", "longform_free"}
OPT_RE = re.compile(r"^\s*([A-Z])\.\s+(.*\S)\s*$")
FURNITURE = re.compile(r"Related Stories|Post navigation|Read More|Share this|"
                       r"Sign up for|Getty Images|answer in digits|\| Published:",
                       re.I)
DANGLE = re.compile(r"\b(in|on|at|of|to|for|with|from|by|the|a|an|and|or|that|"
                    r"as|into|after|before|when|who|which|was|were|is|are|has|"
                    r"have|had|he|she|they|it)\s*$", re.I)



# --- §7.4 arithmetic verification -------------------------------------------
# MEASURED ON THE PILOT: GLM got 4 of 6 Computation golds exactly right. It
# produced "May 4, 2000" for Feb 8 2001 minus 9 months 5 days (correct: May 3)
# and "5 years 2 months" for a span of 5y 2m 11d. §7.4 requires >= 99%
# recomputed-correct, so the gold is RECOMPUTED here and the row is rejected on
# any mismatch. The model proposes; the arithmetic is checked.
_MONTH = (r"(?:Jan|Feb|Mar|Apr|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*")
_DATE_RE = re.compile(rf"\b({_MONTH}\s+\d{{1,2}},\s*\d{{4}}|\d{{1,2}}\s+{_MONTH}\s+\d{{4}})\b")
_OFFSET_RE = re.compile(
    r"(?:(\d+)\s*years?)?[\s,and]*(?:(\d+)\s*months?)?[\s,and]*(?:(\d+)\s*days?)?"
    r"\s*(before|after|prior|earlier|later)", re.I)


def _parse_date(s: str):
    try:
        from dateutil import parser as _dp
        return _dp.parse(s, fuzzy=False).date()
    except Exception:
        return None


def _ymd(gold: str):
    """(y, m, d) from a gold like '1 year 7 months 21 days', else None."""
    y = re.search(r"(\d+)\s*years?", gold, re.I)
    m = re.search(r"(\d+)\s*months?", gold, re.I)
    d = re.search(r"(\d+)\s*days?", gold, re.I)
    if not (y or m or d):
        return None
    return (int(y.group(1)) if y else 0,
            int(m.group(1)) if m else 0,
            int(d.group(1)) if d else 0)


def check_arithmetic(r: dict) -> list[str]:
    if r.get("category") != "Computation":
        return []
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        return []
    q, gold = str(r.get("question", "")), str(r["targets"][0]).strip()
    # Anchors come from the QUESTION ONLY. Taking them from the rationale too
    # let the answer date become its own anchor: "grant issued April 10 2018 /
    # report 11 months later" verified against March 10 2019 + 11m and failed a
    # row whose gold was right.
    q_wo_hint = q.split("(Hint:")[0]
    gold_date = _parse_date(gold)
    dates = [d for d in (_parse_date(x) for x in _DATE_RE.findall(q_wo_hint)) if d]
    uniq = sorted({d for d in dates if d != gold_date})
    # FALL BACK TO THE RATIONALE when the question does not name the dates.
    # Requiring them in the stem made the checker shape the data: a generation
    # run put a date in 100% of stems, against TIME's 36.7% that print NONE and
    # require finding them in the passage. That is L6 inverted -- maximising
    # availability instead of matching it. The rationale is REQUIRED by §5.1 to
    # state both dates, so it is the right place to verify from. Excluding the
    # gold date keeps the earlier false positive fixed: without it, an offset
    # row's own answer became its anchor.
    if len(uniq) < 2:
        rat = [d for d in (_parse_date(x) for x in
                           _DATE_RE.findall(str(r.get("rationale", "")))) if d]
        uniq = sorted({d for d in (dates + rat) if d != gold_date})

    if gold_date is not None:
        # date-offset form: anchor +/- (y, m, d)
        mo = _OFFSET_RE.search(q_wo_hint)
        if mo and uniq and any(mo.group(i) for i in (1, 2, 3)):
            yy, mm, dd, direction = mo.groups()
            off = relativedelta(years=int(yy or 0), months=int(mm or 0),
                                days=int(dd or 0))
            anchor = uniq[0] if len(uniq) == 1 else uniq[-1]
            back = direction.lower() in ("before", "prior", "earlier")
            want = anchor - off if back else anchor + off
            if want != gold_date:
                return [f"arithmetic: gold {gold!r} but {anchor} "
                        f"{'-' if back else '+'} {int(yy or 0)}y{int(mm or 0)}m"
                        f"{int(dd or 0)}d = {want} (§7.4)"]
        return []

    span = _ymd(gold)
    if span is None or len(uniq) < 2:
        return []
    a, b = uniq[0], uniq[-1]
    rd = relativedelta(b, a)
    if (rd.years, rd.months, rd.days) != span:
        return [f"arithmetic: gold {gold!r} but {a} to {b} is "
                f"{rd.years}y {rd.months}m {rd.days}d (§7.4)"]
    return []



_LETTER_DATE = re.compile(
    r"\b([A-E])\s*=\s*(\d{4}-\d{2}-\d{2})\b")


def check_timeline(r: dict) -> list[str]:
    """Recompute a Timeline gold from letter-keyed dates in the rationale.

    Timeline options are PARAPHRASES of passage events, not quotes, so they
    cannot be matched back to the passage to date them -- a fuzzy matcher tried
    on the first live batch resolved 1 row in 20 and false-flagged that one.
    The category that produced D51 (one distinct answer across 4,476 rows) had
    no arithmetic gate at all. Asking the rationale to carry `A=YYYY-MM-DD`
    makes the ordering checkable.

    Silent when the keys are absent, so batches written before this was
    introduced still ingest.
    """
    if r.get("category") != "Timeline":
        return []
    pairs = _LETTER_DATE.findall(str(r.get("rationale", "")))
    if not pairs:
        return []
    gold = str(r["targets"][0]).strip()
    letters = [x for x in gold.split(",") if x]
    dated = dict(pairs)
    if set(dated) != set(letters):
        return [f"Timeline rationale dates {sorted(dated)} do not cover the "
                f"gold sequence {letters} (§7.4)"]
    want = ",".join(sorted(letters, key=lambda L: dated[L]))
    if want != gold:
        return [f"Timeline order: gold {gold!r} but the stated dates sort to "
                f"{want!r} ({dated}) (§7.4)"]
    return []



_DUR_DT = (r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
           r"\s+\d{1,2},\s*\d{4})")
_DUR1 = re.compile(r"Duration\s*1[^.]*?from\s*" + _DUR_DT + r"\s*to\s*" + _DUR_DT, re.I)
_DUR2 = re.compile(r"Duration\s*2[^.]*?from\s*" + _DUR_DT + r"\s*to\s*" + _DUR_DT, re.I)


def check_duration_compare(r: dict) -> list[str]:
    """Recompute which of two durations is longer, from the rationale.

    v6 scored 33.3% here -- exactly the chance rate -- meaning its rows carried
    no signal at all. The comparison is only worth training on if it genuinely
    follows from the stated dates, so it is recomputed rather than trusted.
    Silent when the rationale does not state both spans in the expected form.
    """
    if r.get("category") != "Duration_Compare":
        return []
    try:
        from dateutil import parser as _dp
    except ImportError:
        return []
    rat = str(r.get("rationale", ""))
    m1, m2 = _DUR1.search(rat), _DUR2.search(rat)
    if not (m1 and m2):
        return []
    try:
        a1, a2 = _dp.parse(m1.group(1)).date(), _dp.parse(m1.group(2)).date()
        b1, b2 = _dp.parse(m2.group(1)).date(), _dp.parse(m2.group(2)).date()
    except Exception:
        return []
    d1, d2 = abs((a2 - a1).days), abs((b2 - b1).days)
    diff = abs(d1 - d2)
    rel = diff / max(d1, d2, 1)
    gold = str(r["targets"][0]).strip()
    close = rel <= 0.10 or diff <= 60
    if gold.startswith("The two"):
        if not close:
            return [f"Duration_Compare: gold says the spans are similar but "
                    f"they are {d1}d vs {d2}d ({100*rel:.0f}% apart) (§7.4)"]
    elif gold.startswith("Duration 1"):
        if not (d1 > d2 and not close):
            return [f"Duration_Compare: gold says Duration 1 is longer but "
                    f"d1={d1}d d2={d2}d (§7.4)"]
    elif gold.startswith("Duration 2"):
        if not (d2 > d1 and not close):
            return [f"Duration_Compare: gold says Duration 2 is longer but "
                    f"d1={d1}d d2={d2}d (§7.4)"]
    return []



def check_order_compare(r: dict) -> list[str]:
    """Recompute which of two facts happened earlier, from the rationale's
    stated dates.

    Live batch 147 caught FOUR rows by hand where the gold letter contradicted
    the dates before this gate existed -- exactly the failure class
    check_duration_compare and check_timeline already close for their
    categories. This closes the same gap here.
    """
    if r.get("category") != "Order_Compare":
        return []
    try:
        from dateutil import parser as _dp
    except ImportError:
        return []
    dates = _DATE_RE.findall(str(r.get("rationale", "")))
    if len(dates) < 2:
        return []
    try:
        a, b = _dp.parse(dates[0]).date(), _dp.parse(dates[1]).date()
    except Exception:
        return []
    diff = abs((b - a).days)
    gold = str(r["targets"][0]).strip()
    # A 1-2 day gap is a JUDGMENT CALL, not an arithmetic fact: live batch 147
    # correctly labelled a 1-day gap "Fact 2 earlier" ("published the next
    # day" -- narrative framing signals sequence) right alongside four rows
    # correctly labelling an identical 1-day gap "almost the same time"
    # (coincidental proximity). No date-only rule can tell those apart, so the
    # gate stays SILENT in that zone rather than guess, and only fires where
    # no reasonable reading disagrees.
    if diff <= 2:
        return []
    if "same time" in gold.lower():
        if diff > 14:
            return [f"Order_Compare: gold says almost the same time but the "
                    f"dates are {diff} days apart (§7.4)"]
    elif gold.startswith("Fact 1"):
        if not a < b:
            return [f"Order_Compare: gold says Fact 1 earlier but dates are "
                    f"{a} / {b} (§7.4)"]
    elif gold.startswith("Fact 2"):
        if not b < a:
            return [f"Order_Compare: gold says Fact 2 earlier but dates are "
                    f"{a} / {b} (§7.4)"]
    return []


def options(question: str) -> list[str]:
    if "Choices:" not in question:
        return []
    out = []
    for line in question.split("Choices:", 1)[1].splitlines():
        m = OPT_RE.match(line)
        if m:
            out.append(m.group(2))
    return out


def check(r: dict) -> list[str]:
    e: list[str] = []
    missing = REQUIRED - set(r)
    if missing:
        return [f"missing fields {sorted(missing)}"]
    cat = r.get("category")
    if cat not in ALLOC_NAMES:
        return [f"unknown category {cat!r}"]

    tg = r.get("targets")
    if not (isinstance(tg, list) and len(tg) == 1 and str(tg[0]).strip()):
        return [f"targets must be a single non-empty element (§2), got {tg!r}"]
    gold = str(tg[0]).strip()
    q = str(r.get("question", ""))
    ctx = str(r.get("context", ""))
    opts = options(q)

    # --- §3 / §2 MCQ structure ---
    if cat in NEVER_MCQ and opts:
        e.append(f"{cat} must be free text but carries a Choices block")
    want_n = MCQ_OPTS.get(cat, "any")
    if cat in MCQ_OPTS and not opts:
        e.append(f"{cat} must carry a Choices block")
    if opts:
        if isinstance(want_n, int) and len(opts) != want_n:
            e.append(f"{cat} expects {want_n} options, parsed {len(opts)}")
        if len(set(o.lower() for o in opts)) != len(opts):
            e.append("duplicate options")
        if cat == "Timeline":
            if not re.fullmatch(r"[A-Z](,[A-Z])+", gold):
                e.append(f"Timeline gold must be a letter sequence, got {gold!r}")
        elif cat in LABELS:
            if gold not in LABELS[cat]:
                e.append(f"{cat} gold must be one of {sorted(LABELS[cat])}, got {gold!r}")
        elif gold not in opts:
            e.append("gold is not CHARACTER-IDENTICAL to any option (§2)")
        if len(gold) == 1 and gold.isalpha():
            e.append("gold is a bare letter (§8)")
    elif cat in LABELS and gold not in LABELS[cat]:
        e.append(f"{cat} gold must be one of {sorted(LABELS[cat])}, got {gold!r}")

    # --- context ---
    if cat in NO_CONTEXT:
        if ctx.strip():
            e.append(f"{cat} is a no-context category but context is non-empty")
    else:
        if ctx.strip().lower() in ("", "none", "null"):
            e.append("empty context (§2 forbids the literal \"None\")")
        elif len(ctx.split()) < 60:
            e.append(f"context too short ({len(ctx.split())} words)")
    if r.get("provenance") == "news" and ctx and not re.search(r"\[1\]", ctx):
        e.append("news provenance must use the three-passage [1]/[2]/[3] shape (§6.1)")

    # --- §9 answer style, the gate the template build kept failing ---
    if not opts and cat not in ("longform_free",) and len(gold.split()) > 12:
        e.append(f"free-text gold too long: {len(gold.split())} words (§9 asks terse; TIME mean 2.35)")
    if FURNITURE.search(gold):
        e.append("gold contains page furniture")
    if len(gold) > 25 and DANGLE.search(gold.rstrip(".")):
        e.append("gold is truncated mid-clause")

    # --- §5.1 Computation must compute, not copy ---
    if cat == "Computation":
        if gold and ctx and gold.lower() in ctx.lower():
            e.append("Computation gold appears verbatim in context — copied, not computed (§5.1)")
        # An abstain row has nothing to compute, so demanding a digit in its
        # rationale forces a date to be invented purely to pass the check.
        _abstain = gold.strip().lower().rstrip(".") in (
            "there is no answer", "cannot be determined from the context",
            "the passage does not say",
            "none of the options is supported by the passage")
        if not _abstain and not re.search(r"\d", str(r.get("rationale", ""))):
            e.append("Computation rationale shows no arithmetic")
        if "Hint:" not in q:
            e.append("Computation question is missing the benchmark hint (§3)")
    if cat == "Localization" and re.search(r"\b\d{4}-01-01\b|January 1, \d{4}", gold):
        e.append("fabricated YYYY-01-01 gold (§5.3)")

    e += check_arithmetic(r)
    e += check_timeline(r)
    e += check_duration_compare(r)
    e += check_order_compare(r)
    # The pilot produced a "question" that stated two facts and stopped, with
    # no interrogative at all -- unanswerable as posed. Only enforced where
    # there is no Choices block: with options present the task is explicit, and
    # requiring "?" there rejected valid `relation` rows.
    # LABELS categories are exempt: an NLI premise/hypothesis is declarative by
    # design ("nli_saq carries no options -- the model must emit the bare
    # token"), so requiring "?" there would have rejected 100% of that
    # 400-row category. Caught by tests/test_glm_ingest.py.
    #
    # longform_free is also exempt. The category card gives TWO templates,
    # one interrogative ("What developments does the passage report...?"), one
    # imperative ("Describe what happened between X and Y."). Live batch
    # 133-136 hit this gate on the imperative form and had to rewrite 27
    # otherwise-correct rows into the interrogative template to pass -- a
    # false rejection that narrowed the category to only one of its two
    # specified templates. See tests/test_glm_ingest.py.
    _IMPERATIVE = {"longform_free"}
    if "?" not in q and not opts and cat not in LABELS and cat not in _IMPERATIVE:
        e.append("question contains no interrogative")
    if not str(r.get("rationale", "")).strip():
        e.append("empty rationale")
    if str(r.get("rationale", "")).strip() and str(r.get("rationale")) in q:
        e.append("rationale leaks into question (§8)")
    if r.get("source_dataset") != "AUG_GLM2":
        e.append(f'source_dataset must be "AUG_GLM2", got {r.get("source_dataset")!r}')
    return e


ALLOC_NAMES = {
    "Computation", "Timeline", "Localization", "Counterfactual",
    "Duration_Compare", "Relative_Reasoning", "Order_Reasoning",
    "Co_temporality", "Explicit_Reasoning", "Order_Compare", "nli_saq",
    "nli_mcq", "relation", "ordering", "temporal_dialogue", "duration",
    "storytelling", "longform_free",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="GLM reply files (data/glm_raw/*.txt)")
    ap.add_argument("--out", default="data/manual_aug_glm")
    ap.add_argument("--plan", default="data/glm_packets/_plan.json")
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    have: Counter = Counter()
    for f in out_dir.glob("*.jsonl"):
        for line in f.open():
            if line.strip():
                have[json.loads(line)["category"]] += 1

    if args.status or not args.files:
        plan_p = PROJECT_ROOT / args.plan
        want = json.loads(plan_p.read_text())["by_category"] if plan_p.exists() else {}
        print(f"{'category':22s}{'have':>7}{'want':>7}{'left':>7}")
        tot_h = tot_w = 0
        for c in sorted(want or have):
            h, w = have.get(c, 0), want.get(c, 0)
            tot_h += h; tot_w += w
            print(f"{c:22s}{h:7d}{w:7d}{max(0, w-h):7d}")
        print(f"{'TOTAL':22s}{tot_h:7d}{tot_w:7d}{max(0, tot_w-tot_h):7d}")
        return 0

    # Ledger of reply files already ingested, keyed by content hash. Without
    # it, re-running over data/glm_raw/*.txt reprocesses every reply, reports
    # every row as a duplicate, and writes those non-defects into the reject
    # log -- which is meant to hold rows worth re-prompting.
    ledger_p = out_dir / ".ingested.json"
    ledger = {}
    if ledger_p.exists():
        try:
            ledger = json.loads(ledger_p.read_text())
        except Exception:
            ledger = {}

    accepted: dict[str, list] = defaultdict(list)
    rejects = []
    dups_only = []
    skew: list[str] = []
    import collections as _c2
    seen_q: set[str] = set()
    for f in out_dir.glob("*.jsonl"):
        for line in f.open():
            if line.strip():
                seen_q.add(re.sub(r"\s+", " ", json.loads(line)["question"]).strip().lower())

    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"  !! {p} not found"); continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        # Chat packets write each passage once under a "=== PASSAGE Pn ==="
        # banner and have rows reference it, so a reply does not have to repeat
        # ~240 words of context per row and truncate.
        # A passage block ends at the next banner OR at the first JSONL row.
        # Without the `^\{` alternative the LAST block ran to end-of-file and
        # swallowed every JSON line into its own text, so each row referencing
        # the final passage got raw JSON as its context.
        passages = dict(re.findall(
            r"^=+\s*PASSAGE\s+(\S+?)\s*=+\s*$\n(.*?)(?=^=+\s*PASSAGE\s|^\{|\Z)",
            raw, re.S | re.M))
        passages = {k: v.strip() for k, v in passages.items()}
        body = re.split(r"^=+\s*PASSAGE\s+\S+?\s*=+\s*$", raw, flags=re.M)
        digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()
        if ledger.get(p.name, {}).get("sha1") == digest:
            print(f"  {p.name:34s} already ingested "
                  f"({ledger[p.name].get('rows', 0)} rows) - skipped")
            continue
        rows = extract_jsonl(raw if not passages else "\n".join(body))
        if passages:
            miss = 0
            for r in rows:
                ref = str(r.pop("passage", "")).strip()
                if ref and not str(r.get("context", "")).strip():
                    if ref in passages:
                        r["context"] = passages[ref]
                    else:
                        miss += 1
            if miss:
                print(f"     ({miss} rows referenced an unknown passage label)")
        ok = bad = dup = 0
        batch_rows: list[dict] = []
        for r in rows:
            errs = check(r)
            key = re.sub(r"\s+", " ", str(r.get("question", ""))).strip().lower()
            if not errs and key in seen_q:
                errs = ["duplicate question (already ingested)"]
                dup += 1
            if errs == ["duplicate question (already ingested)"]:
                bad += 1
                dups_only.append(r)
            elif errs:
                bad += 1
                rejects.append({"packet": p.name, "reasons": errs, "row": r})
            else:
                seen_q.add(key)
                accepted[r["category"]].append(r)
                batch_rows.append(r)
                ok += 1
        # BATCH-LEVEL L7 CHECK. Per-row validation cannot see this: a reply
        # whose every MCQ gold is option A passes every row check and is still
        # worthless ("pick A" beats the task). Batch 038-043 shipped 72/72 golds
        # at A and only surfaced when the whole category was audited, ten
        # batches later than it should have.
        _mcq = [r for r in batch_rows if "Choices:" in r.get("question", "")]
        if len(_mcq) >= 6:
            _pos: _c2.Counter = _c2.Counter()
            for r in _mcq:
                ch = options(r["question"])
                g = str(r["targets"][0]).strip()
                if g in ch:
                    _pos["ABCDEFG"[ch.index(g)]] += 1
            if _pos:
                top, n_top = _pos.most_common(1)[0]
                share = n_top / sum(_pos.values())
                if share > 0.60:
                    print(f"  {p.name:34s} !! GOLD POSITION SKEW: {dict(_pos)} "
                          f"-- {top} is {share*100:.0f}% of MCQ golds (L7 limit "
                          f"~30%). REGENERATE THIS BATCH.")
                    skew.append(p.name)
        print(f"  {p.name:34s} parsed={len(rows):3d} accepted={ok:3d} rejected={bad:3d}"
              + (f" (dup {dup})" if dup else ""))
        ledger[p.name] = {"sha1": digest, "rows": ok}

    for cat, rows in accepted.items():
        n = len(list(out_dir.glob(f"{cat}_batch*.jsonl"))) + 1
        with (out_dir / f"{cat}_batch{n:02d}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rejects:
        rp = PROJECT_ROOT / "data/glm_raw/_rejected.jsonl"
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("a") as fh:
            for r in rejects:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        reasons = Counter(x for r in rejects for x in r["reasons"])
        print(f"\n{len(rejects)} rejected -> {rp}")
        for why, n in reasons.most_common(12):
            print(f"  {n:4d}  {why[:100]}")
    ledger_p.write_text(json.dumps(ledger, indent=1))
    if dups_only:
        print(f"\n{len(dups_only)} rows skipped as already-ingested duplicates "
              f"(not defects, not written to the reject log)")
    if skew:
        print(f"\n!! {len(skew)} batch(es) have skewed MCQ gold positions and "
              f"should be regenerated: {', '.join(skew)}")
    print(f"\naccepted {sum(len(v) for v in accepted.values())} rows into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
