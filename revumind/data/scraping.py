"""
Amazon-style Product Review Scraper
====================================
Scrapes product reviews (text, stars, image URLs) and loads them
into a clean, structured Pandas DataFrame.

Supports:
  - requests + BeautifulSoup for real HTML pages
  - Selenium fallback for JavaScript-rendered pages
  - Polite rate limiting + retry logic
  - Saves to CSV and Parquet

Usage:
  python review_scraper.py
"""

import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data Model ─────────────────────────────────────────────────────────────────
@dataclass
class Review:
    """One product review — all fields you care about."""
    product_id:    str
    product_name:  str
    reviewer_name: str
    star_rating:   Optional[float]   # 1.0 – 5.0
    review_title:  str
    review_text:   str
    review_date:   str
    verified:      bool
    helpful_votes: int
    image_urls:    str               # pipe-separated list  "url1|url2"
    source_url:    str


# ── HTTP Session ───────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_page(session: requests.Session, url: str, retries: int = 3) -> Optional[str]:
    """
    Fetch a URL with retry + exponential backoff.
    Returns HTML string or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            # Polite delay: 2–5 seconds between requests
            time.sleep(random.uniform(2, 5))
            response = session.get(url, timeout=15)
            response.raise_for_status()
            log.info(f"  ✓ fetched {url[:80]}  [{response.status_code}]")
            return response.text
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning(f"  Attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s…")
            time.sleep(wait)
    log.error(f"  ✗ gave up on {url}")
    return None


# ── Parser: Amazon-style HTML ──────────────────────────────────────────────────
def parse_star_rating(element) -> Optional[float]:
    """Extract numeric rating from aria-label like 'Rated 4.0 out of 5 stars'."""
    if element is None:
        return None
    label = element.get("aria-label", "") or element.get("title", "")
    for token in label.split():
        try:
            val = float(token)
            if 1.0 <= val <= 5.0:
                return val
        except ValueError:
            continue
    return None


def parse_helpful_votes(element) -> int:
    """Extract vote count from text like '47 people found this helpful'."""
    if element is None:
        return 0
    text = element.get_text(strip=True)
    for token in text.split():
        token = token.replace(",", "")
        if token.isdigit():
            return int(token)
    return 0


def parse_review_images(review_div) -> list[str]:
    """Collect all image URLs attached to a review."""
    urls = []
    # Review images are usually in <img> tags inside a dedicated image section
    img_section = review_div.select("div[data-hook='review-image-tile-section'] img, "
                                    "div.review-image-tile img, "
                                    "img[data-hook='review-image']")
    for img in img_section:
        src = img.get("src") or img.get("data-src") or ""
        # Amazon serves thumbnails at ._SY88. — swap for a larger size
        src = src.replace("._SY88.", "._SL500_.")
        if src.startswith("http"):
            urls.append(src)
    return urls


def parse_reviews_from_html(
    html: str,
    product_id: str,
    product_name: str,
    source_url: str,
) -> list[Review]:
    """
    Parse all reviews out of one page of HTML.
    Selector set matches Amazon's review page structure (2024).
    Adjust selectors for other sites.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    # Each review lives inside a div with data-hook="review"
    review_divs = soup.select("div[data-hook='review']")
    if not review_divs:
        # Fallback: older Amazon layout
        review_divs = soup.select("div.a-section.review")

    for div in review_divs:
        # ── Reviewer name ──────────────────────────────────────────────────────
        name_el = div.select_one("span.a-profile-name, [class*='reviewer-name']")
        reviewer_name = name_el.get_text(strip=True) if name_el else "Anonymous"

        # ── Star rating ────────────────────────────────────────────────────────
        star_el = div.select_one(
            "i[data-hook='review-star-rating'] span, "
            "span[data-hook='review-star-rating'], "
            "a[class*='star-rating']"
        )
        star_rating = parse_star_rating(star_el)

        # ── Review title ───────────────────────────────────────────────────────
        title_el = div.select_one(
            "a[data-hook='review-title'] span, "
            "span[data-hook='review-title']"
        )
        review_title = title_el.get_text(strip=True) if title_el else ""

        # ── Review body text ───────────────────────────────────────────────────
        body_el = div.select_one(
            "span[data-hook='review-body'] span, "
            "div[data-hook='review-collapsed'] span"
        )
        review_text = body_el.get_text(strip=True) if body_el else ""

        # ── Date ───────────────────────────────────────────────────────────────
        date_el = div.select_one("span[data-hook='review-date']")
        review_date = date_el.get_text(strip=True) if date_el else ""

        # ── Verified purchase badge ────────────────────────────────────────────
        verified_el = div.select_one(
            "span[data-hook='avp-badge'], "
            "span[class*='verified-purchase']"
        )
        verified = verified_el is not None

        # ── Helpful votes ──────────────────────────────────────────────────────
        helpful_el = div.select_one("span[data-hook='helpful-vote-statement']")
        helpful_votes = parse_helpful_votes(helpful_el)

        # ── Images attached to this review ────────────────────────────────────
        image_urls = parse_review_images(div)

        reviews.append(Review(
            product_id    = product_id,
            product_name  = product_name,
            reviewer_name = reviewer_name,
            star_rating   = star_rating,
            review_title  = review_title,
            review_text   = review_text,
            review_date   = review_date,
            verified      = verified,
            helpful_votes = helpful_votes,
            image_urls    = "|".join(image_urls),   # pipe-separated for CSV safety
            source_url    = source_url,
        ))

    return reviews


# ── Pagination helper ──────────────────────────────────────────────────────────
def get_next_page_url(html: str, base_url: str) -> Optional[str]:
    """Find the 'Next page' link in paginated review pages."""
    soup = BeautifulSoup(html, "html.parser")
    next_btn = soup.select_one(
        "li.a-last a, "
        "a[data-hook='pagination-bar-next-button'], "
        "a[aria-label='Next page']"
    )
    if next_btn and next_btn.get("href"):
        return urljoin(base_url, next_btn["href"])
    return None


# ── Main scraper ───────────────────────────────────────────────────────────────
def scrape_product_reviews(
    product_id:   str,
    product_name: str,
    start_url:    str,
    max_pages:    int = 5,
) -> list[Review]:
    """
    Scrape up to max_pages of reviews for one product.
    """
    session = make_session()
    all_reviews: list[Review] = []
    url = start_url

    for page_num in range(1, max_pages + 1):
        log.info(f"Scraping page {page_num}/{max_pages}  →  {url[:80]}")
        html = fetch_page(session, url)
        if not html:
            break

        page_reviews = parse_reviews_from_html(html, product_id, product_name, url)
        all_reviews.extend(page_reviews)
        log.info(f"  Found {len(page_reviews)} reviews on page {page_num}")

        next_url = get_next_page_url(html, url)
        if not next_url:
            log.info("  No next page — done with this product.")
            break
        url = next_url

    return all_reviews


# ── DataFrame builder ──────────────────────────────────────────────────────────
def reviews_to_dataframe(reviews: list[Review]) -> pd.DataFrame:
    """
    Convert a list of Review objects into a clean, typed DataFrame.
    """
    if not reviews:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in reviews])

    # ── Types ──────────────────────────────────────────────────────────────────
    df["star_rating"]   = pd.to_numeric(df["star_rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["verified"]      = df["verified"].astype(bool)

    # ── Parse date ────────────────────────────────────────────────────────────
    # Amazon date format: "Reviewed in India on 12 March 2024"
    df["review_date_parsed"] = pd.to_datetime(
        df["review_date"].str.extract(r"(\d{1,2}\s\w+\s\d{4})")[0],
        format="%d %B %Y",
        errors="coerce",
    )

    # ── Text cleaning ─────────────────────────────────────────────────────────
    df["review_text"]  = df["review_text"].str.strip().replace(r"\s+", " ", regex=True)
    df["review_title"] = df["review_title"].str.strip()
    df["review_text"]  = df["review_text"].fillna("")
    df["review_title"] = df["review_title"].fillna("")

    # ── Derived features (feature engineering) ────────────────────────────────
    df["review_length"]     = df["review_text"].str.len()
    df["word_count"]        = df["review_text"].str.split().str.len()
    df["has_images"]        = df["image_urls"].str.len() > 0
    df["image_count"]       = df["image_urls"].apply(
        lambda x: len(x.split("|")) if x else 0
    )
    df["sentiment_label"]   = pd.cut(
        df["star_rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        right=True,
    )
    df["is_long_review"]    = df["word_count"] > 100
    df["is_highly_helpful"] = df["helpful_votes"] >= 10

    # ── Deduplication ─────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["product_id", "reviewer_name", "review_text"])
    log.info(f"Deduplication: {before} → {len(df)} rows")

    # ── Column order ──────────────────────────────────────────────────────────
    cols = [
        "product_id", "product_name",
        "reviewer_name", "star_rating", "sentiment_label",
        "review_title", "review_text",
        "review_length", "word_count", "is_long_review",
        "review_date", "review_date_parsed",
        "verified", "helpful_votes", "is_highly_helpful",
        "has_images", "image_count", "image_urls",
        "source_url",
    ]
    df = df[[c for c in cols if c in df.columns]]

    return df.reset_index(drop=True)


# ── Alternative: load from a Kaggle CSV ───────────────────────────────────────
def load_from_kaggle_csv(filepath: str) -> pd.DataFrame:
    """
    Load and standardise the popular Amazon Reviews dataset from Kaggle.
    Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
    Columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
             HelpfulnessDenominator, Score, Time, Summary, Text
    """
    raw = pd.read_csv(filepath)
    log.info(f"Loaded {len(raw):,} rows from {filepath}")

    df = pd.DataFrame()
    df["product_id"]    = raw["ProductId"]
    df["product_name"]  = raw["ProductId"]          # name not in this dataset
    df["reviewer_name"] = raw["ProfileName"].fillna("Anonymous")
    df["star_rating"]   = pd.to_numeric(raw["Score"], errors="coerce")
    df["review_title"]  = raw["Summary"].fillna("").str.strip()
    df["review_text"]   = raw["Text"].fillna("").str.strip()
    df["review_date"]   = pd.to_datetime(raw["Time"], unit="s", errors="coerce")
    df["review_date_parsed"] = df["review_date"]
    df["verified"]      = False
    df["helpful_votes"] = raw["HelpfulnessNumerator"].fillna(0).astype(int)
    df["image_urls"]    = ""
    df["source_url"]    = "kaggle-amazon-fine-food-reviews"

    # ── Feature engineering ───────────────────────────────────────────────────
    df["review_length"]     = df["review_text"].str.len()
    df["word_count"]        = df["review_text"].str.split().str.len()
    df["has_images"]        = False
    df["image_count"]       = 0
    df["is_long_review"]    = df["word_count"] > 100
    df["is_highly_helpful"] = df["helpful_votes"] >= 10
    df["sentiment_label"]   = pd.cut(
        df["star_rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        right=True,
    )

    df = df.drop_duplicates(subset=["product_id", "reviewer_name", "review_text"])
    return df.reset_index(drop=True)


# ── Save helpers ───────────────────────────────────────────────────────────────
def save_dataframe(df: pd.DataFrame, name: str = "reviews") -> None:
    """Save to both CSV (human-readable) and Parquet (fast, typed)."""
    csv_path     = f"{name}.csv"
    parquet_path = f"{name}.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)

    log.info(f"Saved {len(df):,} reviews → {csv_path} and {parquet_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Quick sanity check of the loaded DataFrame."""
    print("\n" + "="*55)
    print("  DATAFRAME SUMMARY")
    print("="*55)
    print(f"  Shape          : {df.shape}")
    print(f"  Products       : {df['product_id'].nunique()}")
    print(f"  Avg star rating: {df['star_rating'].mean():.2f}")
    print(f"  Avg word count : {df['word_count'].mean():.0f}")
    print(f"  Reviews w/ imgs: {df['has_images'].sum()} ({df['has_images'].mean()*100:.1f}%)")
    print(f"  Verified       : {df['verified'].sum()} ({df['verified'].mean()*100:.1f}%)")
    print(f"\n  Sentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\n  Missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print("="*55 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Option A: scrape a real product page ──────────────────────────────────
    # Swap in any product's review URL below.
    # Note: Amazon aggressively blocks scrapers — use this on sites that
    # allow scraping, or on a demo site like books.toscrape.com
    #
    PRODUCTS = [
        {
            "product_id":   "B08N5WRWNW",
            "product_name": "Echo Dot 4th Gen",
            "url": (
                "https://www.amazon.in/Echo-Dot-4th-Gen/product-reviews/"
                "B08N5WRWNW/?pageNumber=1&sortBy=recent"
            ),
        },
    ]

    all_reviews = []
    for p in PRODUCTS:
        reviews = scrape_product_reviews(
            product_id   = p["product_id"],
            product_name = p["product_name"],
            start_url    = p["url"],
            max_pages    = 3,
        )
        all_reviews.extend(reviews)

    if all_reviews:
        df = reviews_to_dataframe(all_reviews)
        print_summary(df)
        save_dataframe(df, name="scraped_reviews")

    # ── Option B: load from Kaggle CSV (recommended for quick start) ──────────
    # Download from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
    # Then uncomment:
    #
    # df = load_from_kaggle_csv("Reviews.csv")
    # print_summary(df)
    # save_dataframe(df, name="kaggle_reviews")
 
    print("Done. Load the CSV with: df = pd.read_csv('scraped_reviews.csv')")