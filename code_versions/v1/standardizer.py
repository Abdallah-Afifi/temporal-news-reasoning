#!/usr/bin/env python3
"""Standardize heterogeneous fine-tuning data into JSONL and split train/val.

Usage:
  python scripts/standardize_finetuning_data.py --source data/fine_tuning_data --out data/fine_tuning_data/processed --ratio 0.8 --seed 42

It will heuristically extract text records from JSON, CSV, TXT, and XML files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import re
import xml.etree.ElementTree as ET

TEXT_FIELD_CANDIDATES = ("text", "content", "body", "prompt", "completion", "answer", "article")


def extract_from_txt(path: Path) -> Iterable[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    # split on double newlines as records, fallback to whole file
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        parts = [text]
    return ({"text": p, "meta": {"source": str(path)}} for p in parts)


def extract_from_json(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    records: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        # If it looks like a single record with a text field
        for key in TEXT_FIELD_CANDIDATES:
            if key in data and isinstance(data[key], str):
                records.append({"text": data[key], "meta": {"source": str(path)}})
                return records
        # else try to find nested list
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                records.append({"text": item, "meta": {"source": str(path)}})
            elif isinstance(item, dict):
                for key in TEXT_FIELD_CANDIDATES:
                    if key in item and isinstance(item[key], str):
                        records.append({"text": item[key], "meta": {"source": str(path)}})
                        break
                else:
                    # fall back to concatenating string values
                    vals = [str(v).strip() for v in item.values() if isinstance(v, (str, int, float))]
                    if vals:
                        records.append({"text": " \n ".join(vals), "meta": {"source": str(path)}})
    elif isinstance(data, str):
        records.append({"text": data, "meta": {"source": str(path)}})
    return records


def extract_from_csv(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    rows = []
    try:
        reader = csv.DictReader(text.splitlines())
        for r in reader:
            # pick best candidate column if present
            for key in TEXT_FIELD_CANDIDATES:
                if key in r and r[key].strip():
                    rows.append({"text": r[key].strip(), "meta": {"source": str(path)}})
                    break
            else:
                # fallback: join all values
                vals = [v.strip() for v in r.values() if v and v.strip()]
                if vals:
                    rows.append({"text": " || ".join(vals), "meta": {"source": str(path)}})
    except Exception:
        return []
    return rows


def extract_from_xml(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return []
    records = []
    # Heuristic: consider each child element of root as a record
    for child in list(root):
        texts = []
        if child.text and child.text.strip():
            texts.append(child.text.strip())
        for sub in list(child):
            if sub.text and sub.text.strip():
                texts.append(sub.text.strip())
        if texts:
            records.append({"text": " \n ".join(texts), "meta": {"source": str(path)}})
    # if none found, fallback to entire file text
    if not records:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            records.append({"text": txt, "meta": {"source": str(path)}})
    return records


def iter_records(source_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt",):
                yield from extract_from_txt(path)
            elif suffix in (".json",):
                yield from extract_from_json(path)
            elif suffix in (".csv",):
                yield from extract_from_csv(path)
            elif suffix in (".xml",):
                yield from extract_from_xml(path)
            else:
                # try to open small files as text
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    yield {"text": text, "meta": {"source": str(path)}}
        except Exception:
            continue


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/fine_tuning_data"))
    p.add_argument("--out", type=Path, default=Path("data/fine_tuning_data/processed"))
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-records", type=int, default=1, help="Ignore empty dataset if fewer records")
    p.add_argument("--dry-run", action="store_true", help="Don't write outputs; just report counts")
    # New flags for chain-of-thought and cleaning
    p.add_argument("--generate-cot", action="store_true", help="Generate simple chain-of-thought for each record")
    p.add_argument("--baseline-dir", type=Path, default=None, help="Path to baseline results to assist COT/difficulty estimation")
    p.add_argument("--min-tokens", type=int, default=3, help="Minimum token count to keep a record")
    args = p.parse_args()

    src = args.source
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for rec in iter_records(src):
        text = rec.get("text")
        if not text:
            continue
        # Normalize whitespace
        text = " ".join(text.split())
        records.append({"text": text, "meta": rec.get("meta", {})})

    # deduplicate by text
    unique = {}
    for r in records:
        unique.setdefault(r["text"], r)
    records = list(unique.values())

    # filter by token length
    def token_count(s: str) -> int:
        return max(0, len(s.split()))

    records = [r for r in records if token_count(r["text"]) >= args.min_tokens]

    random.Random(args.seed).shuffle(records)

    n = len(records)
    if n < args.min_records:
        print(f"Found {n} records (<{args.min_records}), aborting.")
        return
    split = int(n * args.ratio)
    train = records[:split]
    val = records[split:]

    print(f"Found {n} records -> train={len(train)}, val={len(val)}")

    if args.dry_run:
        return

    # Optionally generate simple chain-of-thought (heuristic)
    def generate_cot_for_text(text: str, baseline_dir: Optional[Path] = None) -> str:
        # Simple heuristic COT: split into sentences and produce numbered reasoning steps
        sents = re.split(r"(?<=[.!?])\\s+", text)
        sents = [s.strip() for s in sents if s.strip()]
        steps = []
        if "?" in text and len(text) < 300:
            # treat as question
            steps.append("Restate the question: " + (text if len(text) < 200 else text[:200] + "..."))
            # find numbers/dates/entities
            nums = re.findall(r"\\b\\d{4}\\b|\\b\\d+\\b", text)
            if nums:
                steps.append("Identify relevant numbers/dates: " + ", ".join(nums))
            if sents:
                steps.append("Consider: " + (sents[0]))
            steps.append("Combine facts and infer an answer.")
            steps.append("Answer: " + (sents[-1] if sents else "[short answer]"))
        else:
            # summarization-style COT: list key sentences as steps
            for i, sent in enumerate(sents[:4], start=1):
                steps.append(f"Step {i}: {sent}")
            if not steps:
                steps.append("Observation: " + (text[:200] + ("..." if len(text) > 200 else "")))
        return " \n ".join(steps)

    if args.generate_cot:
        baseline_dir = args.baseline_dir
        for r in records:
            try:
                cot = generate_cot_for_text(r["text"], baseline_dir)
                r["chain_of_thought"] = cot
            except Exception:
                r["chain_of_thought"] = ""

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote train -> {train_path} (lines={len(train)})")
    print(f"Wrote val   -> {val_path} (lines={len(val)})")


if __name__ == "__main__":
    main()
