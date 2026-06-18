"""
Feature Engineering Pipeline for Product Review Text Data
==========================================================
Covers every category of features you need for:
  - Sentiment Analysis
  - Text Classification
  - ML model input (Scikit-learn / PyTorch / TensorFlow)

Run on your reviews DataFrame produced by review_scraper.py:
    df = pd.read_parquet("scraped_reviews.parquet")
    df_features = build_all_features(df)
"""

import re
import string
import pandas as pd
import numpy as np
from collections import Counter

# NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD   # LSA / topic features

# Download required NLTK data (run once)
for pkg in ["punkt", "stopwords", "wordnet", "vader_lexicon", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
VADER      = SentimentIntensityAnalyzer()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC TEXT STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def add_basic_text_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Fast, no-NLP features derived purely from raw text.
    Great as baseline features for any ML model.
    """
    t = df[text_col].fillna("")

    df["char_count"]          = t.str.len()
    df["word_count"]          = t.str.split().str.len().fillna(0).astype(int)
    df["sentence_count"]      = t.apply(lambda x: len(sent_tokenize(x)) if x else 0)
    df["avg_word_length"]     = t.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["avg_sentence_length"] = np.where(
        df["sentence_count"] > 0,
        df["word_count"] / df["sentence_count"],
        0,
    )

    # Punctuation & style signals
    df["exclamation_count"]   = t.str.count(r"!")
    df["question_count"]      = t.str.count(r"\?")
    df["caps_word_count"]     = t.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
    )
    df["caps_ratio"]          = df["caps_word_count"] / (df["word_count"] + 1)

    # Digit / price mentions
    df["has_number"]          = t.str.contains(r"\d").astype(int)
    df["number_count"]        = t.str.count(r"\b\d+\b")
    df["has_price"]           = t.str.contains(r"₹|\$|rs\.|price|cost", case=False).astype(int)

    # Review length tier — useful as a categorical feature
    df["length_tier"] = pd.cut(
        df["word_count"],
        bins=[0, 20, 75, 200, np.inf],
        labels=["very_short", "short", "medium", "long"],
    )

    print(f"[✓] Basic text features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Standard cleaning pipeline. Returns a clean string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"<.*?>", " ", text)                      # strip HTML tags
    text = re.sub(r"[^\w\s]", " ", text)                    # remove punctuation
    text = re.sub(r"\d+", " NUM ", text)                    # replace numbers
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Clean → tokenize → remove stopwords → lemmatize → rejoin."""
    text   = clean_text(text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def add_preprocessed_columns(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Add both a cleaned and a fully preprocessed text column.
    - clean_text   : lowercase, no URLs/HTML/punct — keep stopwords (for VADER)
    - processed_text: clean + no stopwords + lemmatized (for TF-IDF / ML)
    """
    print("Preprocessing text (this may take a minute)…")
    df["clean_text"]     = df[text_col].fillna("").apply(clean_text)
    df["processed_text"] = df[text_col].fillna("").apply(preprocess_text)
    print(f"[✓] Text preprocessing done  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. SENTIMENT FEATURES  (VADER)
# ══════════════════════════════════════════════════════════════════════════════

def add_sentiment_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    VADER gives 4 sentiment scores per review.
    compound score → overall polarity (-1 to +1)
    pos / neg / neu → proportion of positive / negative / neutral tokens
    """
    scores = df[text_col].fillna("").apply(VADER.polarity_scores)
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_positive"] = scores.apply(lambda s: s["pos"])
    df["vader_negative"] = scores.apply(lambda s: s["neg"])
    df["vader_neutral"]  = scores.apply(lambda s: s["neu"])

    # Categorical label from compound score
    df["vader_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )

    # Sentiment-star agreement  (useful feature for detecting fake reviews)
    if "star_rating" in df.columns:
        star_sentiment = pd.cut(
            df["star_rating"],
            bins=[0, 2, 3, 5],
            labels=["negative", "neutral", "positive"],
            right=True,
        )
        df["sentiment_star_agree"] = (df["vader_label"] == star_sentiment).astype(int)

    print(f"[✓] VADER sentiment features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEXICON-BASED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "excellent", "amazing", "fantastic", "love", "perfect", "great", "wonderful",
    "outstanding", "superb", "best", "awesome", "brilliant", "good", "happy",
    "satisfied", "recommend", "quality", "worth",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "worst", "hate", "horrible", "poor", "bad", "useless",
    "broken", "defective", "damaged", "waste", "disappointed", "refund", "return",
    "never", "problem", "issue", "fake", "cheap",
}
FEATURE_WORDS = {
    "battery", "screen", "camera", "sound", "design", "size", "weight",
    "performance", "speed", "quality", "price", "value", "delivery", "packaging",
    "build", "display", "charging",
}


def add_lexicon_features(df: pd.DataFrame, text_col: str = "processed_text") -> pd.DataFrame:
    """Count domain-specific word categories in each review."""

    def count_words(text, word_set):
        tokens = set(text.lower().split())
        return len(tokens & word_set)

    df["positive_word_count"] = df[text_col].apply(lambda t: count_words(t, POSITIVE_WORDS))
    df["negative_word_count"] = df[text_col].apply(lambda t: count_words(t, NEGATIVE_WORDS))
    df["feature_word_count"]  = df[text_col].apply(lambda t: count_words(t, FEATURE_WORDS))

    df["positive_word_ratio"] = df["positive_word_count"] / (df["word_count"] + 1)
    df["negative_word_ratio"] = df["negative_word_count"] / (df["word_count"] + 1)
    df["lexicon_net_score"]   = df["positive_word_count"] - df["negative_word_count"]

    print(f"[✓] Lexicon features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. TF-IDF FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_features(
    df: pd.DataFrame,
    text_col:    str = "processed_text",
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    n_svd_components: int = 50,
) -> tuple[pd.DataFrame, TfidfVectorizer, TruncatedSVD]:
    """
    Build TF-IDF matrix and reduce to dense LSA (SVD) features.

    Returns:
        df           : DataFrame with 50 lsa_* columns appended
        vectorizer   : fitted TfidfVectorizer (save for inference)
        svd          : fitted TruncatedSVD    (save for inference)

    Why SVD on top of TF-IDF?
      Raw TF-IDF → 5000 sparse columns → too many for tree models.
      SVD compresses them to 50 dense columns capturing topic structure.
    """
    print(f"Building TF-IDF ({max_features} features, ngrams {ngram_range})…")
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        min_df       = 2,
        max_df       = 0.95,
        sublinear_tf = True,      # log(1 + tf) — reduces impact of very common terms
    )
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(""))

    print(f"Reducing TF-IDF to {n_svd_components} LSA components…")
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    lsa = svd.fit_transform(tfidf_matrix)

    lsa_cols = [f"lsa_{i}" for i in range(n_svd_components)]
    df_lsa   = pd.DataFrame(lsa, columns=lsa_cols, index=df.index)
    df       = pd.concat([df, df_lsa], axis=1)

    explained = svd.explained_variance_ratio_.sum()
    print(f"[✓] TF-IDF + LSA done  →  {n_svd_components} components explain "
          f"{explained:.1%} of variance  |  {df.shape[1]} total columns")
    return df, vectorizer, svd


# ══════════════════════════════════════════════════════════════════════════════
# 6. STAR RATING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode star ratings in multiple ways for different model types."""
    if "star_rating" not in df.columns:
        return df

    # Numeric — for regression / gradient boosting
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

    # Bucketed — for classification
    df["rating_bucket"] = pd.cut(
        df["star_rating"],
        bins=[0, 1, 2, 3, 4, 5],
        labels=["1star", "2star", "3star", "4star", "5star"],
        right=True,
    )

    # Binary — positive (4–5) vs not
    df["is_positive_review"] = (df["star_rating"] >= 4).astype(int)

    # One-hot encode rating (useful for neural nets)
    rating_dummies = pd.get_dummies(df["rating_bucket"], prefix="rating")
    df = pd.concat([df, rating_dummies], axis=1)

    # Z-score normalised rating (useful for linear models)
    mean, std = df["star_rating"].mean(), df["star_rating"].std()
    df["star_rating_zscore"] = (df["star_rating"] - mean) / (std + 1e-9)

    print(f"[✓] Rating features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. METADATA FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features from review metadata — not text content.
    Often the highest-signal features for helpfulness / quality prediction.
    """
    # Verified purchase
    if "verified" in df.columns:
        df["is_verified"] = df["verified"].astype(int)

    # Helpful votes
    if "helpful_votes" in df.columns:
        df["helpful_votes"]     = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)
        df["log_helpful_votes"] = np.log1p(df["helpful_votes"])   # log(1+x) — handles zeros
        df["is_helpful"]        = (df["helpful_votes"] >= 5).astype(int)

    # Images attached
    if "image_count" in df.columns:
        df["has_images"]   = (df["image_count"] > 0).astype(int)
        df["log_img_count"] = np.log1p(df["image_count"])

    # Date-based features
    if "review_date_parsed" in df.columns:
        df["review_date_parsed"] = pd.to_datetime(df["review_date_parsed"], errors="coerce")
        df["review_year"]  = df["review_date_parsed"].dt.year
        df["review_month"] = df["review_date_parsed"].dt.month
        df["review_dow"]   = df["review_date_parsed"].dt.dayofweek   # 0=Mon … 6=Sun
        df["is_weekend"]   = (df["review_dow"] >= 5).astype(int)

    # Reviewer activity (requires multiple rows per reviewer)
    if "reviewer_name" in df.columns:
        review_counts = df.groupby("reviewer_name")["review_text"].transform("count")
        df["reviewer_review_count"] = review_counts
        df["is_power_reviewer"]     = (review_counts >= 5).astype(int)

    # Per-product stats (useful for anomaly / fake-review detection)
    if "product_id" in df.columns:
        prod_stats = df.groupby("product_id")["star_rating"].agg(["mean", "std", "count"])
        prod_stats.columns = ["product_avg_rating", "product_rating_std", "product_review_count"]
        df = df.merge(prod_stats, on="product_id", how="left")

        # How far this review's rating deviates from the product average
        df["rating_deviation"] = (df["star_rating"] - df["product_avg_rating"]).abs()

    print(f"[✓] Metadata features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. POS TAG FEATURES  (Part-of-Speech)
# ══════════════════════════════════════════════════════════════════════════════

def get_pos_counts(text: str) -> dict:
    """Count nouns, verbs, adjectives, adverbs in a text."""
    if not text:
        return {"noun_count": 0, "verb_count": 0, "adj_count": 0, "adv_count": 0}
    tokens = word_tokenize(text[:500])    # cap at 500 chars for speed
    tags   = nltk.pos_tag(tokens)
    return {
        "noun_count": sum(1 for _, t in tags if t.startswith("NN")),
        "verb_count": sum(1 for _, t in tags if t.startswith("VB")),
        "adj_count":  sum(1 for _, t in tags if t.startswith("JJ")),
        "adv_count":  sum(1 for _, t in tags if t.startswith("RB")),
    }


def add_pos_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    POS ratios are strong signals for review quality:
    - High adjective ratio → opinionated review (good for sentiment)
    - High noun ratio → factual / descriptive review
    """
    print("Computing POS tags (slow — skipping on large datasets > 50k rows)…")
    if len(df) > 50_000:
        print("  [!] Skipping POS features on large dataset. Sample first.")
        return df

    pos_df = df[text_col].fillna("").apply(get_pos_counts).apply(pd.Series)
    df = pd.concat([df, pos_df], axis=1)

    total = df["word_count"] + 1
    df["adj_ratio"]  = df["adj_count"]  / total
    df["noun_ratio"] = df["noun_count"] / total
    df["verb_ratio"] = df["verb_count"] / total
    df["adv_ratio"]  = df["adv_count"]  / total

    print(f"[✓] POS features added  →  {df.shape[1]} total columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. NORMALISATION & SCALING
# ══════════════════════════════════════════════════════════════════════════════

NUMERIC_COLS_TO_SCALE = [
    "char_count", "word_count", "sentence_count", "avg_word_length",
    "avg_sentence_length", "exclamation_count", "question_count",
    "caps_ratio", "number_count",
    "vader_compound", "vader_positive", "vader_negative",
    "positive_word_count", "negative_word_count", "lexicon_net_score",
    "helpful_votes", "log_helpful_votes", "reviewer_review_count",
    "product_avg_rating", "product_rating_std", "rating_deviation",
]


def add_scaled_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScaler on numeric columns.
    Scaled columns get a _scaled suffix so originals are preserved.
    Returns both the updated df and the fitted scaler (save for inference).
    """
    cols = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols].fillna(0))
    scaled_df = pd.DataFrame(
        scaled, columns=[f"{c}_scaled" for c in cols], index=df.index
    )
    df = pd.concat([df, scaled_df], axis=1)
    print(f"[✓] Scaling done on {len(cols)} columns  →  {df.shape[1]} total columns")
    return df, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 10. MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_all_features(
    df: pd.DataFrame,
    text_col:     str  = "review_text",
    tfidf_feats:  bool = True,
    pos_feats:    bool = True,
    scale:        bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete feature engineering pipeline.

    Returns:
        df_out    : enriched DataFrame
        artifacts : dict with fitted vectorizer, svd, scaler
                    → save these to disk for consistent inference
    """
    print("\n" + "="*60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("="*60)

    artifacts = {}

    df = add_basic_text_features(df, text_col)
    df = add_preprocessed_columns(df, text_col)
    df = add_sentiment_features(df, "clean_text")
    df = add_lexicon_features(df, "processed_text")
    df = add_rating_features(df)
    df = add_metadata_features(df)

    if pos_feats:
        df = add_pos_features(df, "clean_text")

    if tfidf_feats:
        df, vectorizer, svd = build_tfidf_features(df, "processed_text")
        artifacts["vectorizer"] = vectorizer
        artifacts["svd"]        = svd

    if scale:
        df, scaler = add_scaled_features(df)
        artifacts["scaler"] = scaler

    print("\n" + "="*60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Input rows    : {len(df):,}")
    print(f"  Total columns : {df.shape[1]}")
    print("="*60 + "\n")

    return df, artifacts


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SUMMARY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_groups(df: pd.DataFrame) -> None:
    """Print a summary of every feature group and their columns."""
    groups = {
        "Basic text stats":  [c for c in df.columns if c in [
            "char_count","word_count","sentence_count","avg_word_length",
            "avg_sentence_length","exclamation_count","question_count",
            "caps_ratio","number_count","has_price","length_tier"]],
        "VADER sentiment":   [c for c in df.columns if c.startswith("vader")],
        "Lexicon":           [c for c in df.columns if "word_count" in c or "lexicon" in c or "ratio" in c],
        "Rating":            [c for c in df.columns if "rating" in c or "star" in c],
        "Metadata":          [c for c in df.columns if c in [
            "is_verified","helpful_votes","log_helpful_votes","has_images",
            "review_year","review_month","is_weekend","reviewer_review_count",
            "product_avg_rating","product_rating_std","rating_deviation"]],
        "POS tags":          [c for c in df.columns if c.endswith("_count") and c[:3] in ["nou","ver","adj","adv"]],
        "LSA (TF-IDF)":      [c for c in df.columns if c.startswith("lsa_")],
        "Scaled":            [c for c in df.columns if c.endswith("_scaled")],
    }
    print("\nFEATURE GROUPS")
    print("-"*50)
    for group, cols in groups.items():
        if cols:
            print(f"  {group:<22} {len(cols):>3} features")
    print("-"*50)
    print(f"  {'TOTAL':<22} {df.shape[1]:>3} columns\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — runs when executed directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Build a small demo DataFrame ─────────────────────────────────────────
    demo_data = {
        "product_id":   ["P001", "P001", "P002", "P002", "P003"],
        "product_name": ["Echo Dot"] * 2 + ["Fire TV"] * 2 + ["Kindle"],
        "reviewer_name":["Riya", "Arjun", "Priya", "Karan", "Sneha"],
        "star_rating":  [5, 1, 4, 2, 5],
        "review_text":  [
            "Absolutely love this product! The sound quality is amazing and the battery lasts forever. "
            "Perfect for my home. Highly recommend to everyone!",

            "Terrible product. Broke after 2 days. Worst purchase ever. The screen is cracked "
            "and customer service refused to help. Total waste of money!",

            "Good value for the price. Camera quality is decent. Delivery was fast. "
            "The design is sleek and the performance is smooth. Happy with this purchase.",

            "Disappointed with the quality. Battery drains too fast and the sound is poor. "
            "Expected much better for this price. Would not recommend.",

            "This is the best Kindle I have ever owned! Reading experience is phenomenal. "
            "Very lightweight and the display is crystal clear. Worth every rupee!",
        ],
        "verified":      [True, True, False, True, True],
        "helpful_votes": [23, 5, 8, 2, 41],
        "image_count":   [2, 0, 1, 0, 3],
        "review_date_parsed": pd.to_datetime([
            "2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-01-28"
        ]),
    }
    df_demo = pd.DataFrame(demo_data)

    print("Input DataFrame:")
    print(df_demo[["reviewer_name", "star_rating", "review_text"]].to_string(index=False))

    # ── Run the full pipeline ─────────────────────────────────────────────────
    df_out, artifacts = build_all_features(
        df_demo,
        text_col    = "review_text",
        tfidf_feats = True,
        pos_feats   = True,
        scale       = True,
    )

    print_feature_groups(df_out)

    # ── Show key engineered features ──────────────────────────────────────────
    show_cols = [
        "reviewer_name", "star_rating",
        "word_count", "avg_word_length", "exclamation_count",
        "vader_compound", "vader_label",
        "positive_word_count", "negative_word_count",
        "is_verified", "has_images", "is_positive_review",
        "adj_ratio",
    ]
    show_cols = [c for c in show_cols if c in df_out.columns]
    print("Sample engineered features:")
    print(df_out[show_cols].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    df_out.to_parquet("reviews_engineered.parquet", index=False)
    print("\nSaved → reviews_engineered.parquet")
    print(f"Artifacts available: {list(artifacts.keys())}")

    # To save artifacts for later inference:
    # import joblib
    # joblib.dump(artifacts["vectorizer"], "tfidf_vectorizer.pkl")
    # joblib.dump(artifacts["svd"],        "lsa_svd.pkl")
    # joblib.dump(artifacts["scaler"],     "standard_scaler.pkl")