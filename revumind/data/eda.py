"""
Exploratory Data Analysis — Product Review Data
=================================================
Complete EDA pipeline covering:
  1.  Dataset overview & data types
  2.  Missing value analysis
  3.  Star rating distribution
  4.  Review length analysis
  5.  Sentiment distribution
  6.  Time-series trends
  7.  Verified vs unverified reviews
  8.  Helpful votes analysis
  9.  Top keywords (positive & negative)
  10. Correlation heatmap
  11. Product-level comparisons
  12. Outlier detection

Run after feature_engineering.py:
    df = pd.read_parquet("reviews_engineered.parquet")
    run_full_eda(df)
"""

import warnings

warnings.filterwarnings("ignore")

from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from revumind.utils.constants import PALETTE, STOP_WORDS, configure_plotting

# ── Global style ───────────────────────────────────────────────────────────────
POS_COLOR = PALETTE[0]
NEG_COLOR = PALETTE[3]
NEU_COLOR = PALETTE[2]

configure_plotting()

SAVE_FIGS = True  # set False to only show, not save


def savefig(name: str) -> None:
    if SAVE_FIGS:
        plt.savefig(f"eda_{name}.png", bbox_inches="tight")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════


def overview(df: pd.DataFrame) -> None:
    """Print a structured summary of the dataset."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {df.shape[1]}")

    if "product_id" in df.columns:
        print(f"  Unique products : {df['product_id'].nunique():,}")
    if "reviewer_name" in df.columns:
        print(f"  Unique reviewers: {df['reviewer_name'].nunique():,}")
    if "star_rating" in df.columns:
        r = df["star_rating"]
        print(f"  Avg star rating : {r.mean():.2f}  (median {r.median():.1f})")
    if "word_count" in df.columns:
        w = df["word_count"]
        print(f"  Avg word count  : {w.mean():.0f}  (median {w.median():.0f})")

    print("\n  Column dtypes:")
    for dtype, cols in df.dtypes.groupby(df.dtypes):
        print(f"    {str(dtype):<12} → {list(cols.index)[:6]}" + (" …" if len(cols) > 6 else ""))
    print("=" * 60 + "\n")

    print("  Sample rows:")
    text_cols = ["reviewer_name", "star_rating", "review_text", "vader_label", "word_count"]
    show = [c for c in text_cols if c in df.columns]
    print(df[show].head(3).to_string(index=False))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("[✓] No missing values found.")
        return

    pct = (missing / len(df) * 100).round(1)
    fig, ax = plt.subplots(figsize=(9, max(3, len(missing) * 0.45)))
    bars = ax.barh(missing.index, pct.values, color=PALETTE[3], alpha=0.8)
    ax.bar_label(bars, labels=[f"{v}%" for v in pct.values], padding=4, fontsize=10)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    savefig("missing_values")
    print(f"  Columns with missing data: {len(missing)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. STAR RATING DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════


def plot_star_distribution(df: pd.DataFrame) -> None:
    if "star_rating" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Count bar
    counts = df["star_rating"].value_counts().sort_index()
    colors = [NEG_COLOR, NEG_COLOR, NEU_COLOR, POS_COLOR, POS_COLOR]
    axes[0].bar(
        counts.index.astype(str),
        counts.values,
        color=colors[: len(counts)],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title("Review Count by Star Rating", fontweight="bold")
    axes[0].set_xlabel("Stars")
    axes[0].set_ylabel("Count")
    for i, v in zip(counts.index, counts.values):
        axes[0].text(i - 1, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=9)

    # Pie
    labels = [f"{'★' * int(s)}  {c:,}" for s, c in zip(counts.index, counts.values)]
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors[: len(counts)],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    axes[1].set_title("Rating Share", fontweight="bold")

    plt.suptitle("Star Rating Distribution", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("star_distribution")

    print("  Rating breakdown:")
    for s, c in counts.items():
        bar = "█" * int(c / counts.max() * 20)
        print(f"    {int(s)}★  {bar:<20}  {c:,}  ({c/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REVIEW LENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def plot_review_length(df: pd.DataFrame) -> None:
    if "word_count" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of word counts (cap at 500 for readability)
    data = df["word_count"].clip(upper=500)
    axes[0].hist(data, bins=40, color=PALETTE[1], alpha=0.8, edgecolor="white")
    axes[0].axvline(
        data.median(),
        color=NEG_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Median {data.median():.0f}",
    )
    axes[0].axvline(
        data.mean(),
        color=PALETTE[4],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {data.mean():.0f}",
    )
    axes[0].set_title("Word Count Distribution", fontweight="bold")
    axes[0].set_xlabel("Words (capped at 500)")
    axes[0].set_ylabel("Reviews")
    axes[0].legend(fontsize=9)

    # Word count by star rating — violin
    if "star_rating" in df.columns:
        plot_df = df[df["word_count"] < 500].copy()
        plot_df["star_rating"] = plot_df["star_rating"].astype(int).astype(str)
        sns.violinplot(
            data=plot_df,
            x="star_rating",
            y="word_count",
            palette=PALETTE,
            ax=axes[1],
            inner="quartile",
        )
        axes[1].set_title("Word Count by Star Rating", fontweight="bold")
        axes[1].set_xlabel("Stars")
        axes[1].set_ylabel("Word Count")

    # Length tier
    if "length_tier" in df.columns:
        tier_counts = df["length_tier"].value_counts()
        axes[2].bar(
            tier_counts.index.astype(str),
            tier_counts.values,
            color=PALETTE[: len(tier_counts)],
            alpha=0.85,
            edgecolor="white",
        )
        axes[2].set_title("Review Length Tiers", fontweight="bold")
        axes[2].set_xlabel("Length tier")
        axes[2].set_ylabel("Count")
        for x, v in zip(tier_counts.index.astype(str), tier_counts.values):
            axes[2].text(x, v + len(df) * 0.005, f"{v/len(df)*100:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Review Length Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("review_length")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SENTIMENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════


def plot_sentiment(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # VADER compound distribution
    if "vader_compound" in df.columns:
        axes[0].hist(df["vader_compound"], bins=40, color=PALETTE[0], alpha=0.8, edgecolor="white")
        axes[0].axvline(
            0.05, color=POS_COLOR, linestyle="--", linewidth=1.3, label="Positive threshold"
        )
        axes[0].axvline(
            -0.05, color=NEG_COLOR, linestyle="--", linewidth=1.3, label="Negative threshold"
        )
        axes[0].set_title("VADER Compound Score", fontweight="bold")
        axes[0].set_xlabel("Compound score (-1 to 1)")
        axes[0].set_ylabel("Reviews")
        axes[0].legend(fontsize=8)

    # VADER label counts
    if "vader_label" in df.columns:
        vc = df["vader_label"].value_counts()
        clr = {"positive": POS_COLOR, "neutral": NEU_COLOR, "negative": NEG_COLOR}
        axes[1].bar(
            vc.index.astype(str),
            vc.values,
            color=[clr.get(k, PALETTE[0]) for k in vc.index],
            alpha=0.85,
            edgecolor="white",
        )
        axes[1].set_title("Sentiment Label Counts", fontweight="bold")
        axes[1].set_xlabel("Sentiment")
        axes[1].set_ylabel("Count")
        for x, v in zip(vc.index.astype(str), vc.values):
            axes[1].text(x, v + len(df) * 0.005, f"{v/len(df)*100:.1f}%", ha="center", fontsize=9)

    # VADER compound vs star rating — scatter with regression line
    if "vader_compound" in df.columns and "star_rating" in df.columns:
        sample = df.sample(min(1000, len(df)), random_state=42)
        axes[2].scatter(
            sample["star_rating"], sample["vader_compound"], alpha=0.25, s=18, color=PALETTE[1]
        )
        # Compute and plot regression line
        m, b = np.polyfit(
            df["star_rating"].dropna(), df.loc[df["star_rating"].notna(), "vader_compound"], 1
        )
        xs = np.array([1, 5])
        axes[2].plot(
            xs,
            m * xs + b,
            color=NEG_COLOR,
            linewidth=2,
            label=f"r={df['star_rating'].corr(df['vader_compound']):.2f}",
        )
        axes[2].set_title("Stars vs VADER Score", fontweight="bold")
        axes[2].set_xlabel("Star rating")
        axes[2].set_ylabel("VADER compound")
        axes[2].legend(fontsize=9)

    plt.suptitle("Sentiment Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("sentiment")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TIME-SERIES TRENDS
# ══════════════════════════════════════════════════════════════════════════════


def plot_time_trends(df: pd.DataFrame) -> None:
    date_col = next((c for c in ["review_date_parsed", "review_date"] if c in df.columns), None)
    if date_col is None:
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if len(df) < 5:
        return

    df["year_month"] = df[date_col].dt.to_period("M")
    monthly = (
        df.groupby("year_month")
        .agg(
            review_count=("review_text", "count"),
            avg_rating=("star_rating", "mean"),
            avg_sentiment=("vader_compound", "mean"),
        )
        .reset_index()
    )
    monthly["year_month"] = monthly["year_month"].astype(str)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(
        monthly["year_month"],
        monthly["review_count"],
        color=PALETTE[1],
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    axes[0].fill_between(
        monthly["year_month"], monthly["review_count"], alpha=0.15, color=PALETTE[1]
    )
    axes[0].set_ylabel("Review count")
    axes[0].set_title("Monthly Review Volume", fontweight="bold")

    if "avg_rating" in monthly.columns and monthly["avg_rating"].notna().any():
        axes[1].plot(
            monthly["year_month"],
            monthly["avg_rating"],
            color=POS_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        axes[1].axhline(
            df["star_rating"].mean(),
            linestyle="--",
            color=NEG_COLOR,
            linewidth=1.2,
            alpha=0.7,
            label=f"Overall mean {df['star_rating'].mean():.2f}",
        )
        axes[1].set_ylabel("Avg star rating")
        axes[1].set_title("Monthly Average Rating", fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].set_ylim(1, 5)

    if "avg_sentiment" in monthly.columns and monthly["avg_sentiment"].notna().any():
        clrs = [POS_COLOR if v >= 0 else NEG_COLOR for v in monthly["avg_sentiment"]]
        axes[2].bar(
            monthly["year_month"],
            monthly["avg_sentiment"],
            color=clrs,
            alpha=0.8,
            edgecolor="white",
        )
        axes[2].axhline(0, color="grey", linewidth=0.8)
        axes[2].set_ylabel("Avg VADER compound")
        axes[2].set_title("Monthly Sentiment Trend", fontweight="bold")

    # Rotate x-tick labels
    tick_step = max(1, len(monthly) // 12)
    ticks = list(range(0, len(monthly), tick_step))
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(
        [monthly["year_month"].iloc[i] for i in ticks],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    plt.suptitle("Review Trends Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("time_trends")


# ══════════════════════════════════════════════════════════════════════════════
# 7. VERIFIED vs UNVERIFIED
# ══════════════════════════════════════════════════════════════════════════════


def plot_verified_analysis(df: pd.DataFrame) -> None:
    if "verified" not in df.columns:
        return
    if df["verified"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Count
    vc = df["verified"].value_counts()
    axes[0].bar(
        ["Unverified", "Verified"],
        vc.values,
        color=[PALETTE[3], PALETTE[0]],
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_title("Verified vs Unverified", fontweight="bold")
    axes[0].set_ylabel("Count")
    for x, v in enumerate(vc.values):
        axes[0].text(x, v + len(df) * 0.005, f"{v/len(df)*100:.1f}%", ha="center", fontsize=10)

    # Avg rating
    if "star_rating" in df.columns:
        avg = df.groupby("verified")["star_rating"].mean()
        axes[1].bar(
            ["Unverified", "Verified"],
            avg.values,
            color=[PALETTE[3], PALETTE[0]],
            alpha=0.85,
            edgecolor="white",
        )
        axes[1].set_title("Avg Rating by Verification", fontweight="bold")
        axes[1].set_ylabel("Avg star rating")
        axes[1].set_ylim(0, 5.5)
        for x, v in enumerate(avg.values):
            axes[1].text(x, v + 0.05, f"{v:.2f}★", ha="center", fontsize=10)

    # VADER compound
    if "vader_compound" in df.columns:
        sns.boxplot(
            data=df, x="verified", y="vader_compound", palette=[PALETTE[3], PALETTE[0]], ax=axes[2]
        )
        axes[2].set_title("Sentiment by Verification", fontweight="bold")
        axes[2].set_xticklabels(["Unverified", "Verified"])
        axes[2].set_ylabel("VADER compound score")

    plt.suptitle("Verified Purchase Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("verified_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# 8. HELPFUL VOTES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def plot_helpful_votes(df: pd.DataFrame) -> None:
    if "helpful_votes" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Distribution (log scale)
    data = df["helpful_votes"].clip(upper=100)
    axes[0].hist(data[data > 0], bins=30, color=PALETTE[4], alpha=0.8, edgecolor="white")
    axes[0].set_title("Helpful Votes Distribution", fontweight="bold")
    axes[0].set_xlabel("Votes (capped at 100)")
    axes[0].set_ylabel("Reviews")

    # Votes vs word count
    if "word_count" in df.columns:
        sample = df[df["helpful_votes"] < 100].sample(min(800, len(df)), random_state=42)
        axes[1].scatter(
            sample["word_count"], sample["helpful_votes"], alpha=0.3, s=15, color=PALETTE[1]
        )
        axes[1].set_title("Word Count vs Helpful Votes", fontweight="bold")
        axes[1].set_xlabel("Word count")
        axes[1].set_ylabel("Helpful votes")

    # Votes by sentiment
    if "vader_label" in df.columns:
        avg_votes = df.groupby("vader_label")["helpful_votes"].median()
        clr = {"positive": POS_COLOR, "neutral": NEU_COLOR, "negative": NEG_COLOR}
        axes[2].bar(
            avg_votes.index.astype(str),
            avg_votes.values,
            color=[clr.get(k, PALETTE[0]) for k in avg_votes.index],
            alpha=0.85,
            edgecolor="white",
        )
        axes[2].set_title("Median Helpful Votes by Sentiment", fontweight="bold")
        axes[2].set_xlabel("Sentiment")
        axes[2].set_ylabel("Median helpful votes")
        for x, v in zip(avg_votes.index.astype(str), avg_votes.values):
            axes[2].text(x, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)

    plt.suptitle("Helpful Votes Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("helpful_votes")


# ══════════════════════════════════════════════════════════════════════════════
# 9. TOP KEYWORDS — positive vs negative reviews
# ══════════════════════════════════════════════════════════════════════════════


def get_top_words(texts: pd.Series, n: int = 20) -> list[tuple[str, int]]:
    words = []
    for text in texts.fillna(""):
        for w in str(text).lower().split():
            w = w.strip(".,!?\"'()[]")
            if w not in STOP_WORDS and len(w) > 2 and w.isalpha():
                words.append(w)
    return Counter(words).most_common(n)


def plot_top_keywords(df: pd.DataFrame, text_col: str = "review_text") -> None:
    if text_col not in df.columns:
        return
    if "star_rating" not in df.columns and "vader_label" not in df.columns:
        return

    # Split into positive and negative
    if "star_rating" in df.columns:
        pos_texts = df[df["star_rating"] >= 4][text_col]
        neg_texts = df[df["star_rating"] <= 2][text_col]
    else:
        pos_texts = df[df["vader_label"] == "positive"][text_col]
        neg_texts = df[df["vader_label"] == "negative"][text_col]

    pos_words = get_top_words(pos_texts)
    neg_words = get_top_words(neg_texts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, words, color, title in [
        (axes[0], pos_words, POS_COLOR, "Top Words — Positive Reviews (4-5★)"),
        (axes[1], neg_words, NEG_COLOR, "Top Words — Negative Reviews (1-2★)"),
    ]:
        labels, counts = zip(*words) if words else ([], [])
        bars = ax.barh(
            list(labels)[::-1], list(counts)[::-1], color=color, alpha=0.8, edgecolor="white"
        )
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frequency")

    plt.suptitle("Most Frequent Keywords by Sentiment", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("top_keywords")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [
        "star_rating",
        "word_count",
        "sentence_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "vader_compound",
        "vader_positive",
        "vader_negative",
        "positive_word_count",
        "negative_word_count",
        "helpful_votes",
        "image_count",
        "adj_ratio",
        "noun_ratio",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 3:
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.75)))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    savefig("correlation_heatmap")

    # Print top correlations with star_rating
    if "star_rating" in corr.columns:
        top = corr["star_rating"].drop("star_rating").abs().sort_values(ascending=False).head(8)
        print("  Top features correlated with star_rating:")
        for feat, val in top.items():
            direction = "+" if corr.loc[feat, "star_rating"] > 0 else "-"
            print(f"    {direction}{val:.3f}  {feat}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. PRODUCT-LEVEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════


def plot_product_comparison(df: pd.DataFrame, top_n: int = 8) -> None:
    if "product_name" not in df.columns and "product_id" not in df.columns:
        return
    prod_col = "product_name" if "product_name" in df.columns else "product_id"

    prod = (
        df.groupby(prod_col)
        .agg(
            review_count=("review_text", "count"),
            avg_rating=("star_rating", "mean"),
            avg_sentiment=("vader_compound", "mean"),
            avg_words=("word_count", "mean"),
        )
        .reset_index()
    )
    prod = prod[prod["review_count"] >= 2].nlargest(top_n, "review_count")
    if len(prod) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(prod) * 0.55)))

    for ax, col, color, label in [
        (axes[0], "avg_rating", PALETTE[0], "Avg Star Rating"),
        (axes[1], "avg_sentiment", PALETTE[1], "Avg Sentiment (VADER)"),
        (axes[2], "review_count", PALETTE[4], "Review Count"),
    ]:
        if col not in prod.columns:
            continue
        ax.barh(prod[prod_col], prod[col], color=color, alpha=0.85, edgecolor="white")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(label)
        ax.invert_yaxis()

    plt.suptitle("Product Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("product_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# 12. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag statistical outliers using IQR method on key numeric columns.
    Prints a report and returns a DataFrame of flagged rows.
    """
    outlier_cols = ["word_count", "helpful_votes", "vader_compound"]
    outlier_cols = [c for c in outlier_cols if c in df.columns]

    flags = pd.Series(False, index=df.index)
    print("\n  OUTLIER REPORT (IQR method)")
    print("  " + "-" * 40)

    for col in outlier_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        flags |= mask
        print(f"    {col:<22}  {mask.sum():>4} outliers  " f"(range [{lo:.1f}, {hi:.1f}])")

    print(f"\n  Total flagged rows: {flags.sum()} ({flags.mean()*100:.1f}%)")

    if flags.sum() > 0 and "review_text" in df.columns:
        print("\n  Sample outlier reviews:")
        sample = df[flags][["review_text", "word_count", "star_rating", "helpful_votes"]].head(3)
        print(sample.to_string(index=False))

    return df[flags].copy()


# ══════════════════════════════════════════════════════════════════════════════
# MASTER EDA RUNNER
# ══════════════════════════════════════════════════════════════════════════════


def run_full_eda(df: pd.DataFrame) -> None:
    """Run every EDA section in sequence."""
    overview(df)
    plot_missing_values(df)
    plot_star_distribution(df)
    plot_review_length(df)
    plot_sentiment(df)
    plot_time_trends(df)
    plot_verified_analysis(df)
    plot_helpful_votes(df)
    plot_top_keywords(df)
    plot_correlation_heatmap(df)
    plot_product_comparison(df)
    outliers = detect_outliers(df)

    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print(f"  Plots saved as  eda_*.png")
    print(f"  Outlier rows    {len(outliers):,} returned for inspection")
    print("=" * 60 + "\n")

    return outliers


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Option A: run on your scraped / engineered data ───────────────────────
    # df = pd.read_parquet("reviews_engineered.parquet")

    # ── Option B: built-in demo dataset ──────────────────────────────────────
    import random

    random.seed(42)
    np.random.seed(42)
    n = 200
    products = ["Echo Dot", "Fire TV", "Kindle", "AirPods", "Phone Case"]

    df = pd.DataFrame(
        {
            "product_id": np.random.choice(["P001", "P002", "P003", "P004", "P005"], n),
            "product_name": np.random.choice(products, n),
            "reviewer_name": [f"User_{i}" for i in range(n)],
            "star_rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.08, 0.12, 0.25, 0.45]),
            "review_text": (
                [
                    "Great product, works perfectly! Amazing quality and fast delivery. Highly recommend!"
                ]
                * 60
                + ["Terrible. Broke after a week. Waste of money. Very disappointed with quality."]
                * 30
                + ["Decent product for the price. Nothing special but gets the job done okay."]
                * 50
                + ["Excellent! Best purchase I have made. Sound quality is outstanding."] * 40
                + ["Poor battery life and bad customer service. Would not buy again."] * 20
            )[:n],
            "verified": np.random.choice([True, False], n, p=[0.7, 0.3]),
            "helpful_votes": np.random.exponential(scale=5, size=n).astype(int),
            "image_count": np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.13, 0.07]),
            "review_date_parsed": pd.date_range("2022-01-01", periods=n, freq="2D"),
        }
    )

    # Add basic engineered columns so all plots work without the full pipeline
    df["word_count"] = df["review_text"].str.split().str.len()
    df["char_count"] = df["review_text"].str.len()
    df["sentence_count"] = df["review_text"].str.count(r"[.!?]").clip(lower=1)
    df["avg_word_length"] = df["review_text"].apply(lambda x: np.mean([len(w) for w in x.split()]))
    df["exclamation_count"] = df["review_text"].str.count("!")
    df["question_count"] = df["review_text"].str.count(r"\?")
    df["caps_ratio"] = df["review_text"].apply(
        lambda x: sum(1 for w in x.split() if w.isupper()) / (len(x.split()) + 1)
    )
    df["length_tier"] = pd.cut(
        df["word_count"],
        bins=[0, 20, 75, 200, np.inf],
        labels=["very_short", "short", "medium", "long"],
    )
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    scores = df["review_text"].apply(vader.polarity_scores)
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_positive"] = scores.apply(lambda s: s["pos"])
    df["vader_negative"] = scores.apply(lambda s: s["neg"])
    df["vader_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    df["positive_word_count"] = (
        df["review_text"]
        .str.lower()
        .str.count(r"great|excellent|amazing|love|perfect|best|recommend|outstanding")
    )
    df["negative_word_count"] = (
        df["review_text"]
        .str.lower()
        .str.count(r"terrible|awful|worst|hate|broken|waste|poor|disappointed")
    )
    df["adj_ratio"] = np.random.uniform(0.05, 0.3, n)
    df["noun_ratio"] = np.random.uniform(0.1, 0.4, n)

    run_full_eda(df)
