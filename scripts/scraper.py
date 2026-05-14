import json
import time
import os
import random
from datetime import datetime
from bs4 import BeautifulSoup
import cloudscraper

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_ARTICLES_TO_SCRAPE = 250000
INPUT_FILE = "/home/g2/temporal-news-reasoning/data/corpus/News_Category_Dataset_v3.json"

OUTPUT_FILE = "/home/g2/temporal-news-reasoning/data/corpus/scraped/scraped_articles.json"
FAILED_ARTICLES_FILE = "/home/g2/temporal-news-reasoning/data/corpus/scraped/failed_articles.json"
STATISTICS_FILE = "/home/g2/temporal-news-reasoning/data/corpus/scraped/scraping_statistics.json"

CHECKPOINT_EVERY = 50  # Save progress every N articles


# ============================================================================
# SCRAPER CLASS
# ============================================================================

class NewsScraper:
    def __init__(self, num_articles=100):
        self.num_articles = num_articles
        self.scraped_articles = []
        self.failed_articles = []

        # Browser-like scraper (better than requests)
        self.session = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "windows",
                "mobile": False
            }
        )

    # ----------------------------------------------------------------------

    def scrape_article(self, url, max_retries=5):
        """
        Archive-first strategy:
        1) Try archive.org
        2) Fallback to live HuffPost
        """

        archive_url = f"https://web.archive.org/{url}"
        urls_to_try = [archive_url, url]

        for target_url in urls_to_try:

            response = None

            for attempt in range(max_retries):
                try:
                    response = self.session.get(target_url, timeout=20)
                    response.raise_for_status()
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 2)
                        time.sleep(sleep_time)
                    else:
                        response = None

            if response is None:
                continue

            try:
                soup = BeautifulSoup(response.content, "html.parser")

                # ==============================
                # 1️⃣ Structured JSON extraction
                # ==============================
                script = soup.find("script", type="application/ld+json")
                if script and script.string:
                    try:
                        data = json.loads(script.string)

                        if isinstance(data, list):
                            for entry in data:
                                if isinstance(entry, dict) and "articleBody" in entry:
                                    return True, entry["articleBody"]
                        elif isinstance(data, dict) and "articleBody" in data:
                            return True, data["articleBody"]
                    except:
                        pass

                # ==============================
                # 2️⃣ Paragraph fallback
                # ==============================
                paragraphs = soup.find_all("p")
                text_blocks = []

                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        text_blocks.append(text)

                if text_blocks:
                    return True, "\n\n".join(text_blocks)

            except Exception:
                continue

        return False, "Archive and live scraping both failed"

    # ----------------------------------------------------------------------

    def load_articles_from_json(self):
        articles = []

        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= self.num_articles:
                        break
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue

            print(f"Loaded {len(articles)} articles.")
            return articles

        except FileNotFoundError:
            print("Input file not found.")
            return []

    # ----------------------------------------------------------------------

    def save_json(self, filepath, data):
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------------

    def run(self):

        print("=" * 80)
        print("Starting HuffPost Archive-First Scraper")
        print(f"Target: {self.num_articles} articles")
        print(f"Start time: {datetime.now()}")
        print("=" * 80)

        articles = self.load_articles_from_json()
        if not articles:
            return

        for idx, article in enumerate(articles, 1):

            link = article.get("link", "")
            headline = article.get("headline", "")
            date = article.get("date", "")

            print(f"\n[{idx}/{len(articles)}] {headline[:80]}")
            print(f"URL: {link}")

            success, result = self.scrape_article(link)

            if success:
                self.scraped_articles.append({
                    "title": headline,
                    "content": result,
                    "date": date,
                    "link": link
                })
                print(f"[SUCCESS] {len(result)} characters")

            else:
                self.failed_articles.append({
                    "title": headline,
                    "link": link,
                    "error": result
                })
                print(f"[FAILED] {result}")

            # Randomized delay (critical)
            time.sleep(random.uniform(3, 7))

            # Checkpoint saving
            if idx % CHECKPOINT_EVERY == 0:
                print("Saving checkpoint...")
                self.save_json(OUTPUT_FILE, self.scraped_articles)
                self.save_json(FAILED_ARTICLES_FILE, self.failed_articles)

        # Final save
        self.save_json(OUTPUT_FILE, self.scraped_articles)
        self.save_json(FAILED_ARTICLES_FILE, self.failed_articles)

        stats = {
            "timestamp": str(datetime.now()),
            "total_processed": len(articles),
            "success": len(self.scraped_articles),
            "failed": len(self.failed_articles),
            "success_rate": round(
                (len(self.scraped_articles) / len(articles)) * 100, 2
            ) if articles else 0
        }

        self.save_json(STATISTICS_FILE, stats)

        print("\n" + "=" * 80)
        print("SCRAPING COMPLETE")
        print("=" * 80)
        print(f"Success: {len(self.scraped_articles)}")
        print(f"Failed: {len(self.failed_articles)}")
        print(f"Success rate: {stats['success_rate']}%")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    scraper = NewsScraper(num_articles=NUM_ARTICLES_TO_SCRAPE)
    scraper.run()