"""
Download a filtered subset of CommonCrawl News (CCNews) for RAG-based temporal reasoning.
Source: stanford-oval/ccnews on HuggingFace
Target: ~100K recent English news articles from 2023-2024
"""
from datasets import load_dataset
import json
import os
import time

SAVE_DIR = "/home/g2/temporal-news-reasoning/data/corpus/ccnews"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_PER_YEAR = 100000  # 100K articles per year
YEARS = [str(y) for y in range(2025, 1999, -1)]

for year in YEARS:
    print(f"\n{'='*50}")
    print(f"Streaming CCNews {year} (English, target: {TARGET_PER_YEAR} articles)...")
    print(f"{'='*50}")


    try:
        dataset = load_dataset("stanford-oval/ccnews", name=year, streaming=True)
    except Exception as e:
        print(f"  Skipping {year}: {e}")
        continue



    print("Dataset loaded, starting to iterate...")
    print(f"Features: {dataset['train'].features}")

    articles = []
    skipped = 0
    start_time = time.time()
    for i, example in enumerate(dataset["train"]):
        if i == 0:
            elapsed = time.time() - start_time
            print(f"First row received after {elapsed:.1f}s")
            print(f"Sample keys: {list(example.keys())}")

        # Filter: English only, non-empty text
        lang = example.get("language", "")
        if lang and lang != "en":
            skipped += 1
            continue
        plain_text = example.get("plain_text", "")
        if not plain_text or len(plain_text) < 200:
            skipped += 1
            continue

        articles.append({
            "title": example.get("title", ""),
            "plain_text": plain_text,
            "published_date": str(example.get("published_date", "")),
        })

        if len(articles) % 5000 == 0:
            print(f"  Collected {len(articles):,} articles (skipped {skipped:,} non-English/short)...")

        if len(articles) >= TARGET_PER_YEAR:
            break

    outfile = os.path.join(SAVE_DIR, f"ccnews_{year}_en.jsonl")
    with open(outfile, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(articles):,} articles to {outfile}")
    print(f"  (skipped {skipped:,} non-English/short articles)")

print(f"\n{'='*50}")
print("CCNews download complete!")
print(f"{'='*50}")
