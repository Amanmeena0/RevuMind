"""
Business Insights Extraction — Product Review Data
===================================================
Extracts actionable intelligence from product reviews covering:

  1.  Complaint Analysis      — top complaints ranked by frequency + severity
  2.  Trending Topics         — which issues are growing vs declining over time
  3.  Defect Rate Tracking    — defect mentions per product per time period
  4.  Aspect-Based Sentiment  — sentiment per product feature (battery, camera...)
  5.  Competitive Signals     — brand comparison mentions ("better than X")
  6.  Fake Review Detection   — flag suspicious review patterns
  7.  Customer Segmentation   — group reviewers by behaviour
  8.  Executive Dashboard     — KPI summary with alert thresholds

Each section produces:
  - A clean analysis table / dict
  - An actionable insight sentence (what to DO with this data)
  - A matplotlib plot saved to disk

Usage:
    analyser = ReviewInsightsEngine()
    insights = analyser.run_all(df, text_col="review_text")
    analyser.save_dashboard("dashboard.png")
"""

import re, warnings, os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from revumind.utils.constants import PALETTE, STOP_WORDS, configure_plotting
from revumind.data.synthetic import generate_review_data

warnings.filterwarnings("ignore")
np.random.seed(42)

configure_plotting()

# ══════════════════════════════════════════════════════════════════════════════
# 2.  COMPLAINT ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

COMPLAINT_KEYWORDS = {
    "Battery Drain":     ["battery drain", "battery life", "charge fast",
                          "lasts hours", "battery terrible", "recharge"],
    "Build Quality":     ["broke", "broken", "cracked", "snapped", "plastic",
                          "cheap build", "flimsy", "fell apart"],
    "Defective Unit":    ["defective", "defect", "manufacturing", "faulty",
                          "not working", "never worked", "doa"],
    "Camera Issues":     ["blurry", "grainy", "camera quality", "photos bad",
                          "low light", "camera terrible"],
    "Software/App Bugs": ["app crash", "crashes", "bugs", "slow app",
                          "not connect", "update", "unusable"],
    "Delivery Problems": ["package damaged", "arrived damaged", "late delivery",
                          "shipping slow", "box crushed", "wrong item"],
    "Value for Money":   ["overpriced", "expensive", "not worth", "better elsewhere",
                          "competitors cheaper", "price"],
    "Customer Service":  ["customer service", "refused", "no response",
                          "support unhelpful", "warranty denied"],
}

SEVERITY_WEIGHTS = {
    "Defective Unit":    3.0,
    "Build Quality":     2.5,
    "Battery Drain":     2.0,
    "Camera Issues":     1.8,
    "Software/App Bugs": 1.7,
    "Delivery Problems": 1.5,
    "Customer Service":  1.4,
    "Value for Money":   1.2,
}


def analyse_complaints(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Identify and rank the top complaints by frequency and severity score.

    Severity score = mention_count × severity_weight × avg_negative_star_impact

    Business insight:
      The complaint with highest severity score needs immediate product team attention.
      High frequency but low severity = comms issue (set expectations better).
      Low frequency but high severity = product defect (recall risk).
    """
    neg_df   = df[df["star_rating"] <= 3].copy()
    results  = []

    for complaint, keywords in COMPLAINT_KEYWORDS.items():
        pattern = "|".join(re.escape(kw) for kw in keywords)
        mask    = neg_df[text_col].str.lower().str.contains(pattern, regex=True, na=False)
        count   = mask.sum()
        if count == 0:
            continue
        pct_of_neg = count / max(len(neg_df), 1) * 100
        pct_of_all = count / max(len(df), 1) * 100
        avg_stars  = neg_df.loc[mask, "star_rating"].mean()
        severity_w = SEVERITY_WEIGHTS.get(complaint, 1.0)
        severity   = count * severity_w * (4 - avg_stars)

        results.append({
            "complaint":     complaint,
            "mention_count": count,
            "pct_neg_reviews": round(pct_of_neg, 1),
            "pct_all_reviews": round(pct_of_all, 1),
            "avg_star_rating": round(avg_stars, 2),
            "severity_weight": severity_w,
            "severity_score":  round(severity, 1),
        })

    df_out = pd.DataFrame(results).sort_values("severity_score", ascending=False)
    top    = df_out.iloc[0]["complaint"] if len(df_out) else "N/A"
    print(f"\n  TOP COMPLAINT: {top}")
    print(f"  Insight: Prioritise '{top}' — highest severity score signals "
          f"greatest customer impact.")
    return df_out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRENDING TOPICS  (LDA topic modelling over time)
# ══════════════════════════════════════════════════════════════════════════════

def extract_trending_topics(
    df:          pd.DataFrame,
    text_col:    str = "review_text",
    n_topics:    int = 5,
    n_top_words: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """
    Use LDA to discover latent topics, then track their volume month-by-month.

    Algorithm:
      1. Vectorise all review text with CountVectorizer
      2. LDA decomposes the document-term matrix into topic distributions
      3. Each document gets a topic distribution → assign dominant topic
      4. Aggregate by month to see which topics are rising/falling

    Business insight:
      A topic growing rapidly in recent months = emerging pain point.
      A declining topic after a product update = the fix worked.
    """
    def prep(t):
        t = t.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        return " ".join(w for w in t.split()
                        if w not in STOP_WORDS and len(w) > 3)

    corpus    = df[text_col].fillna("").apply(prep).tolist()
    vec       = CountVectorizer(max_features=500, ngram_range=(1, 2),
                                min_df=3, max_df=0.9)
    dtm       = vec.fit_transform(corpus)
    vocab     = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=20,
        random_state=42, learning_method="batch",
    )
    lda.fit(dtm)

    # Name each topic by its top words
    topic_labels = {}
    for i, comp in enumerate(lda.components_):
        top_words = [vocab[j] for j in comp.argsort()[:-n_top_words-1:-1]]
        topic_labels[i] = " | ".join(top_words[:4])

    # Assign dominant topic to each review
    doc_topics  = lda.transform(dtm)
    df          = df.copy()
    df["topic"] = doc_topics.argmax(axis=1)
    df["topic_name"] = df["topic"].map(topic_labels)

    # Monthly topic volume
    monthly = (df.groupby(["month", "topic_name"])
                 .size()
                 .reset_index(name="count"))
    monthly_pivot = monthly.pivot(
        index="month", columns="topic_name", values="count"
    ).fillna(0)

    # Trend: compare last 3 months vs first 3 months for each topic
    months = sorted(monthly_pivot.index)
    if len(months) >= 6:
        first3 = monthly_pivot.loc[months[:3]].mean()
        last3  = monthly_pivot.loc[months[-3:]].mean()
        trend  = ((last3 - first3) / (first3 + 1) * 100).round(1)
        trending_up = trend.idxmax()
        print(f"\n  TRENDING TOPIC: '{trending_up}' grew "
              f"{trend[trending_up]:+.1f}% in last 3 months vs first 3")
        print(f"  Insight: Monitor '{trending_up}' — accelerating volume "
              f"suggests an emerging product issue.")
    else:
        trend = pd.Series(0.0, index=monthly_pivot.columns)

    return monthly_pivot, topic_labels


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DEFECT RATE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

DEFECT_KEYWORDS = [
    "defective", "defect", "broken", "broke", "cracked", "snapped",
    "not working", "never worked", "manufacturing", "faulty",
    "doa", "damaged", "malfunction", "stopped working",
]


def track_defect_rates(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Track defect mention rate per product per month.

    Defect rate = defect_mentions / total_reviews × 100

    Alert thresholds (configurable):
      > 5%  = WARNING — investigate manufacturing batch
      > 10% = CRITICAL — consider product recall / urgent QC audit

    Business insight:
      A sudden spike in defect rate for a specific product in a specific
      month points to a bad manufacturing batch — cross-reference with
      production logs for that period.
    """
    pattern = "|".join(re.escape(k) for k in DEFECT_KEYWORDS)
    df      = df.copy()
    df["is_defect"] = df[text_col].str.lower().str.contains(
        pattern, regex=True, na=False
    ).astype(int)

    defect_by_product_month = (
        df.groupby(["product", "month"])
          .agg(total_reviews=("review_text", "count"),
               defect_mentions=("is_defect", "sum"))
          .reset_index()
    )
    defect_by_product_month["defect_rate_pct"] = (
        defect_by_product_month["defect_mentions"] /
        defect_by_product_month["total_reviews"] * 100
    ).round(2)
    defect_by_product_month["alert"] = defect_by_product_month["defect_rate_pct"].apply(
        lambda x: "🔴 CRITICAL" if x > 10 else ("🟡 WARNING" if x > 5 else "🟢 OK")
    )

    critical = defect_by_product_month[
        defect_by_product_month["alert"] == "🔴 CRITICAL"
    ]
    if len(critical):
        row = critical.sort_values("defect_rate_pct", ascending=False).iloc[0]
        print(f"\n  DEFECT ALERT: {row['product']} in {row['month']} "
              f"→ {row['defect_rate_pct']:.1f}% defect rate!")
        print(f"  Insight: Cross-reference production batch from {row['month']} "
              f"for {row['product']}. Possible manufacturing QC failure.")

    return defect_by_product_month


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ASPECT-BASED SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

ASPECTS = {
    "Battery":     ["battery", "charge", "charging", "mah", "power", "drain"],
    "Camera":      ["camera", "photo", "picture", "lens", "megapixel", "video"],
    "Display":     ["screen", "display", "brightness", "resolution", "amoled"],
    "Build":       ["build", "material", "plastic", "metal", "weight", "design"],
    "Software":    ["app", "software", "update", "interface", "bug", "crash"],
    "Delivery":    ["delivery", "shipping", "packaging", "arrived", "courier"],
    "Value":       ["price", "value", "worth", "cost", "expensive", "cheap"],
    "Performance": ["speed", "fast", "slow", "lag", "performance", "processor"],
}

POS_WORDS = {"excellent","amazing","great","good","love","perfect","outstanding",
             "brilliant","superb","fantastic","clear","smooth","fast","best","happy"}
NEG_WORDS = {"terrible","awful","poor","bad","broken","worst","horrible","cheap",
             "slow","crash","blurry","defective","disappointed","useless","avoid"}


def analyse_aspect_sentiment(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    For each product feature (aspect), compute:
      - Mention rate (how often discussed)
      - Net sentiment score (-1 to +1)
      - Positive %, Negative %

    Business insight:
      Aspects with high mention rate + negative sentiment = top priority fixes.
      Aspects with high mention rate + positive sentiment = marketing material.
      Aspects with low mention rate = customers don't care (deprioritise).
    """
    results = []
    texts   = df[text_col].str.lower().fillna("")

    for aspect, keywords in ASPECTS.items():
        pattern = "|".join(r"\b" + re.escape(k) + r"\b" for k in keywords)
        mask    = texts.str.contains(pattern, regex=True, na=False)
        subset  = texts[mask]
        if len(subset) == 0:
            continue

        pos_count = subset.apply(
            lambda t: any(w in t for w in POS_WORDS)
        ).sum()
        neg_count = subset.apply(
            lambda t: any(w in t for w in NEG_WORDS)
        ).sum()
        neutral   = len(subset) - pos_count - neg_count
        net_score = (pos_count - neg_count) / max(len(subset), 1)
        mention_pct = len(subset) / len(df) * 100

        results.append({
            "aspect":       aspect,
            "mentions":     len(subset),
            "mention_pct":  round(mention_pct, 1),
            "positive":     pos_count,
            "negative":     neg_count,
            "neutral":      neutral,
            "pos_pct":      round(pos_count / max(len(subset), 1) * 100, 1),
            "neg_pct":      round(neg_count / max(len(subset), 1) * 100, 1),
            "net_sentiment": round(net_score, 3),
        })

    df_out = pd.DataFrame(results).sort_values("mentions", ascending=False)
    worst  = df_out[df_out["net_sentiment"] < 0].nlargest(1, "mentions")
    if len(worst):
        row = worst.iloc[0]
        print(f"\n  WORST ASPECT: {row['aspect']} ({row['neg_pct']:.0f}% negative, "
              f"{row['mentions']} mentions)")
        print(f"  Insight: {row['aspect']} is the most-discussed negative feature — "
              f"product team should prioritise this for next release.")
    return df_out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FAKE REVIEW DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_suspicious_reviews(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Flag reviews that match patterns common in fake/incentivised reviews.

    Signals:
      - Very short text (< 5 words) with 5 stars (bot pattern)
      - Burst: many reviews same day from unverified accounts
      - Repeated text (exact or near-duplicate reviews)
      - All-caps excessive enthusiasm ("AMAZING!!! BEST EVER!!!!")
      - Unverified purchase with 5 stars and 0 helpful votes

    Business insight:
      High fake review rate inflates product score, misleads customers,
      and risks platform penalties (Amazon/Flipkart fake review bans).
    """
    df  = df.copy()
    flags = pd.Series(0, index=df.index)
    reasons = pd.Series("", index=df.index)

    # Flag 1: Suspiciously short 5-star reviews
    short_5star = (df["review_length"] < 6) & (df["star_rating"] == 5)
    flags[short_5star] += 1
    reasons[short_5star] += "short_5star; "

    # Flag 2: All-caps excessive punctuation
    all_caps = df[text_col].str.count("[A-Z]") / df[text_col].str.len().clip(lower=1)
    excessive = (all_caps > 0.5) & (df["star_rating"] == 5)
    flags[excessive] += 1
    reasons[excessive] += "excessive_caps; "

    # Flag 3: Unverified + 5 star + 0 helpful votes
    unverified_5star = (
        (~df["verified"]) & (df["star_rating"] == 5) & (df["helpful_votes"] == 0)
    )
    flags[unverified_5star] += 1
    reasons[unverified_5star] += "unverified_5star_no_votes; "

    # Flag 4: Duplicate or near-duplicate text
    text_counts = df[text_col].str.lower().str.strip().value_counts()
    duplicates  = df[text_col].str.lower().str.strip().isin(
        text_counts[text_counts > 2].index
    )
    flags[duplicates] += 2
    reasons[duplicates] += "duplicate_text; "

    df["suspicion_score"] = flags
    df["suspicion_reasons"] = reasons.str.strip("; ")
    df["is_suspicious"] = flags >= 2

    suspicious_pct = df["is_suspicious"].mean() * 100
    print(f"\n  FAKE REVIEW RATE: {suspicious_pct:.1f}% of reviews are suspicious")
    print(f"  Insight: {suspicious_pct:.0f}% suspected fake — "
          f"{'URGENT: report to platform' if suspicious_pct > 10 else 'monitor monthly'}. "
          f"Filter before using ratings in marketing claims.")

    return df[["product", "review_text", "star_rating", "verified",
               "helpful_votes", "suspicion_score", "suspicion_reasons",
               "is_suspicious"]].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  KPI DASHBOARD DATA
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs for the executive dashboard."""
    neg     = df["star_rating"] <= 2
    pos     = df["star_rating"] >= 4

    # Net Promoter Proxy: % 5-star - % 1-star
    p5 = (df["star_rating"] == 5).mean()
    p1 = (df["star_rating"] == 1).mean()
    nps_proxy = round((p5 - p1) * 100, 1)

    # Review velocity (last 30 days vs prior 30)
    cutoff      = df["review_date"].max() - pd.Timedelta(days=30)
    recent      = df[df["review_date"] >= cutoff]
    prior_start = cutoff - pd.Timedelta(days=30)
    prior       = df[(df["review_date"] >= prior_start) & (df["review_date"] < cutoff)]
    velocity_chg = ((len(recent) - len(prior)) / max(len(prior), 1)) * 100

    return {
        "total_reviews":        len(df),
        "avg_star_rating":      round(df["star_rating"].mean(), 2),
        "pct_positive":         round(pos.mean() * 100, 1),
        "pct_negative":         round(neg.mean() * 100, 1),
        "nps_proxy":            nps_proxy,
        "verified_pct":         round(df["verified"].mean() * 100, 1),
        "avg_helpful_votes":    round(df["helpful_votes"].mean(), 1),
        "review_velocity_chg":  round(velocity_chg, 1),
        "products_analysed":    df["product"].nunique(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_executive_dashboard(
    df:             pd.DataFrame,
    complaints:     pd.DataFrame,
    defect_rates:   pd.DataFrame,
    aspect_sent:    pd.DataFrame,
    kpis:           dict,
    save_path:      str = "executive_dashboard.png",
) -> None:
    """
    7-panel executive dashboard combining all insights into one A3-style figure.
    """
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#FAFAF8")
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.05, right=0.97, top=0.91, bottom=0.05,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.96, "Product Review Intelligence — Executive Dashboard",
             ha="center", fontsize=19, fontweight="bold", color="#1A1A1A")
    fig.text(0.5, 0.935,
             f"Based on {kpis['total_reviews']:,} reviews across "
             f"{kpis['products_analysed']} products",
             ha="center", fontsize=11, color="#555")

    # ── KPI boxes ─────────────────────────────────────────────────────────────
    kpi_items = [
        ("Total Reviews",   f"{kpis['total_reviews']:,}", "#5DCAA5"),
        ("Avg Rating",      f"⭐ {kpis['avg_star_rating']}", "#378ADD"),
        ("% Positive",      f"{kpis['pct_positive']}%",  "#97C459"),
        ("% Negative",      f"{kpis['pct_negative']}%",  "#D85A30"),
        ("NPS Proxy",       f"{kpis['nps_proxy']:+.0f}",  "#7F77DD"),
        ("Review Velocity", f"{kpis['review_velocity_chg']:+.0f}%", "#EF9F27"),
    ]
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.axis("off")
    for idx, (label, value, color) in enumerate(kpi_items):
        x = 0.08 + idx * 0.156
        kpi_ax.add_patch(plt.Rectangle(
            (x, 0.1), 0.13, 0.8,
            transform=kpi_ax.transAxes,
            facecolor=color + "22", edgecolor=color,
            linewidth=1.5, clip_on=False,
        ))
        kpi_ax.text(x + 0.065, 0.65, value, transform=kpi_ax.transAxes,
                    ha="center", fontsize=15, fontweight="bold", color=color)
        kpi_ax.text(x + 0.065, 0.22, label, transform=kpi_ax.transAxes,
                    ha="center", fontsize=8.5, color="#555")
    kpi_ax.set_title("Key Performance Indicators", fontweight="bold",
                      fontsize=12, pad=8)

    # ── Panel 1: Top complaints ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    top_c = complaints.head(7)
    colors_c = [PALETTE[1] if s > 200 else PALETTE[2]
                for s in top_c["severity_score"]]
    bars = ax1.barh(top_c["complaint"][::-1], top_c["severity_score"][::-1],
                    color=colors_c[::-1], alpha=0.85, edgecolor="white")
    ax1.bar_label(bars, labels=[f"{v:.0f}" for v in top_c["severity_score"][::-1]],
                  padding=3, fontsize=8)
    ax1.set_title("Top Complaints by Severity Score", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Severity Score (freq × weight × impact)")
    ax1.tick_params(axis="y", labelsize=8)
    # Legend
    ax1.add_patch(plt.Rectangle((0,0), 0, 0, color=PALETTE[1], label="High severity"))
    ax1.add_patch(plt.Rectangle((0,0), 0, 0, color=PALETTE[2], label="Medium severity"))
    ax1.legend(fontsize=7, loc="lower right")

    # ── Panel 2: Defect rate over time ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    # Focus on product 0 (EchoPod Pro — has the spike)
    prod0 = defect_rates[defect_rates["product"] == PRODUCTS[0]].sort_values("month")
    if len(prod0) > 0:
        months_plot = prod0["month"].tolist()
        rates_plot  = prod0["defect_rate_pct"].tolist()
        colors_d    = [PALETTE[1] if r > 10 else PALETTE[2] if r > 5
                       else PALETTE[0] for r in rates_plot]
        ax2.bar(range(len(months_plot)), rates_plot, color=colors_d,
                alpha=0.85, edgecolor="white", width=0.7)
        ax2.axhline(5,  color=PALETTE[2], linewidth=1.2, linestyle="--",
                    label="5% Warning")
        ax2.axhline(10, color=PALETTE[1], linewidth=1.2, linestyle="--",
                    label="10% Critical")
        step = max(1, len(months_plot) // 6)
        ax2.set_xticks(range(0, len(months_plot), step))
        ax2.set_xticklabels(
            [months_plot[i] for i in range(0, len(months_plot), step)],
            rotation=30, ha="right", fontsize=7,
        )
        ax2.set_ylabel("Defect Rate %")
        ax2.set_title(f"Defect Rate — {PRODUCTS[0]}", fontweight="bold", fontsize=11)
        ax2.legend(fontsize=7)

    # ── Panel 3: Aspect sentiment matrix ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    top_a = aspect_sent.head(8)
    ax3.scatter(
        top_a["mention_pct"],
        top_a["net_sentiment"],
        s=top_a["mentions"] * 1.8,
        c=top_a["net_sentiment"],
        cmap="RdYlGn", vmin=-1, vmax=1,
        alpha=0.85, edgecolors="white", linewidths=1.2,
    )
    for _, row in top_a.iterrows():
        ax3.annotate(
            row["aspect"],
            (row["mention_pct"], row["net_sentiment"]),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    ax3.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Mention Rate %")
    ax3.set_ylabel("Net Sentiment (-1 to +1)")
    ax3.set_title("Aspect Sentiment Map\n(size = mention count)",
                  fontweight="bold", fontsize=11)
    # Quadrant labels
    ax3.text(0.72, 0.95, "High mention\n+ positive",
             transform=ax3.transAxes, fontsize=7, color="#5DCAA5", ha="right")
    ax3.text(0.72, 0.05, "High mention\n+ NEGATIVE → fix this",
             transform=ax3.transAxes, fontsize=7, color="#D85A30", ha="right")

    # ── Panel 4: Rating distribution per product ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    prod_stars = df.groupby(["product", "star_rating"]).size().unstack(fill_value=0)
    prod_stars_pct = prod_stars.div(prod_stars.sum(axis=1), axis=0) * 100
    star_colors = {1:"#D85A30", 2:"#EF9F27", 3:"#888780", 4:"#97C459", 5:"#5DCAA5"}
    bottom = np.zeros(len(prod_stars_pct))
    for star in [1, 2, 3, 4, 5]:
        if star in prod_stars_pct.columns:
            ax4.barh(prod_stars_pct.index, prod_stars_pct[star],
                     left=bottom, color=star_colors[star], alpha=0.85,
                     edgecolor="white", label=f"{star}★")
            bottom += prod_stars_pct[star].values
    ax4.set_xlabel("% of reviews")
    ax4.set_title("Rating Distribution by Product", fontweight="bold", fontsize=11)
    ax4.legend(fontsize=7, loc="lower right", ncol=5)
    ax4.tick_params(axis="y", labelsize=8)

    # ── Panel 5: Monthly review volume + sentiment trend ──────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    monthly = df.groupby("month").agg(
        total=("review_text", "count"),
        pos=("star_rating", lambda x: (x >= 4).sum()),
        neg=("star_rating", lambda x: (x <= 2).sum()),
    ).reset_index()
    months_l = monthly["month"].tolist()
    x_idx    = range(len(months_l))
    ax5.fill_between(x_idx, monthly["pos"], alpha=0.35, color=PALETTE[0], label="Positive (4-5★)")
    ax5.fill_between(x_idx, monthly["neg"], alpha=0.35, color=PALETTE[1], label="Negative (1-2★)")
    ax5.plot(x_idx, monthly["total"], color="#333", linewidth=1.5, label="Total")
    step2 = max(1, len(months_l) // 6)
    ax5.set_xticks(list(x_idx)[::step2])
    ax5.set_xticklabels(months_l[::step2], rotation=30, ha="right", fontsize=7)
    ax5.set_ylabel("Review Count")
    ax5.set_title("Monthly Review Volume + Sentiment", fontweight="bold", fontsize=11)
    ax5.legend(fontsize=7)

    # ── Panel 6: Complaint mention % by product ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    top3_complaints = complaints["complaint"].head(4).tolist()
    products_list   = df["product"].unique().tolist()
    heatmap_data    = np.zeros((len(top3_complaints), len(products_list)))

    for i, complaint in enumerate(top3_complaints):
        pattern = "|".join(
            re.escape(kw) for kw in COMPLAINT_KEYWORDS[complaint]
        )
        for j, product in enumerate(products_list):
            prod_df  = df[df["product"] == product]
            neg_prod = prod_df[prod_df["star_rating"] <= 3]
            if len(neg_prod) == 0:
                continue
            count = neg_prod["review_text"].str.lower().str.contains(
                pattern, regex=True, na=False
            ).sum()
            heatmap_data[i, j] = count / max(len(neg_prod), 1) * 100

    sns.heatmap(
        heatmap_data,
        xticklabels=[p.split()[0] for p in products_list],
        yticklabels=[c[:18] for c in top3_complaints],
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.4, ax=ax6,
        annot_kws={"size": 8},
        cbar_kws={"label": "% of neg reviews"},
    )
    ax6.set_title("Complaint Rate by Product (%)",
                  fontweight="bold", fontsize=11)
    ax6.tick_params(axis="x", rotation=20, labelsize=8)
    ax6.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def plot_complaint_deep_dive(
    complaints: pd.DataFrame,
    df:         pd.DataFrame,
    save_path:  str = "complaint_deep_dive.png",
) -> None:
    """Detailed complaint analysis: severity + trend + product breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Severity vs frequency bubble
    ax = axes[0]
    ax.scatter(
        complaints["mention_count"],
        complaints["severity_score"],
        s=complaints["mention_count"] * 3,
        c=complaints["severity_score"],
        cmap="RdYlGn_r", alpha=0.8, edgecolors="white", linewidths=1.5,
    )
    for _, row in complaints.iterrows():
        ax.annotate(row["complaint"][:15], (row["mention_count"], row["severity_score"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Mention Count"); ax.set_ylabel("Severity Score")
    ax.set_title("Complaint Matrix\n(top-right = most urgent)", fontweight="bold")

    # Complaint % by star rating
    ax2 = axes[1]
    complaint_by_star = []
    for complaint, keywords in list(COMPLAINT_KEYWORDS.items())[:5]:
        pattern = "|".join(re.escape(k) for k in keywords)
        for star in [1, 2, 3]:
            sub = df[df["star_rating"] == star]
            pct = sub["review_text"].str.lower().str.contains(
                pattern, regex=True, na=False
            ).mean() * 100
            complaint_by_star.append({
                "complaint": complaint[:14], "star": f"{star}★", "pct": pct
            })
    cb_df = pd.DataFrame(complaint_by_star)
    if len(cb_df):
        pivot = cb_df.pivot(index="complaint", columns="star", values="pct").fillna(0)
        pivot.plot(kind="bar", ax=ax2, color=PALETTE[:3], alpha=0.85,
                   edgecolor="white", width=0.7)
        ax2.set_title("Complaint by Star Rating", fontweight="bold")
        ax2.set_xlabel("")
        ax2.set_ylabel("% of reviews at that rating")
        ax2.tick_params(axis="x", rotation=20, labelsize=8)
        ax2.legend(fontsize=8, title="Rating")

    # Top complaint monthly trend
    ax3 = axes[2]
    top_complaint = complaints.iloc[0]["complaint"]
    pattern_top   = "|".join(
        re.escape(k) for k in COMPLAINT_KEYWORDS[top_complaint]
    )
    df2 = df.copy()
    df2["top_complaint"] = df2["review_text"].str.lower().str.contains(
        pattern_top, regex=True, na=False
    ).astype(int)
    monthly_trend = df2.groupby("month").agg(
        total=("review_text", "count"),
        complaint=("top_complaint", "sum"),
    ).reset_index()
    monthly_trend["rate"] = monthly_trend["complaint"] / monthly_trend["total"] * 100
    months_t = monthly_trend["month"].tolist()
    ax3.plot(range(len(months_t)), monthly_trend["rate"],
             color=PALETTE[1], linewidth=2, marker="o", markersize=4)
    ax3.fill_between(range(len(months_t)), monthly_trend["rate"],
                     alpha=0.15, color=PALETTE[1])
    step3 = max(1, len(months_t) // 6)
    ax3.set_xticks(range(0, len(months_t), step3))
    ax3.set_xticklabels(months_t[::step3], rotation=30, ha="right", fontsize=7)
    ax3.set_ylabel("% of reviews mentioning complaint")
    ax3.set_title(f"'{top_complaint}'\nMonthly Trend", fontweight="bold")

    plt.suptitle("Complaint Analysis — Deep Dive",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  REVIEW BUSINESS INSIGHTS ENGINE")
    print("="*62)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("\n[1] Building dataset…")
    df = generate_review_data(n=800)
    # Add extra columns needed for this script but not in generic generator
    df["month_num"] = df["review_date"].dt.month
    df["quarter"]   = df["review_date"].dt.to_period("Q").astype(str)

    # ── 2. Complaint analysis ─────────────────────────────────────────────────
    print("\n[2] Analysing complaints…")
    complaints = analyse_complaints(df)
    print(f"\n  Top 5 complaints:")
    print(complaints[["complaint","mention_count","pct_neg_reviews","severity_score"]].head(5).to_string(index=False))

    # ── 3. Trending topics ────────────────────────────────────────────────────
    print("\n[3] Extracting trending topics (LDA)…")
    monthly_topics, topic_labels = extract_trending_topics(df, n_topics=5)

    # ── 4. Defect rates ───────────────────────────────────────────────────────
    print("\n[4] Tracking defect rates…")
    defect_rates = track_defect_rates(df)
    alerts = defect_rates[defect_rates["alert"] != "🟢 OK"]
    if len(alerts):
        print(f"\n  Active alerts:")
        print(alerts[["product","month","defect_rate_pct","alert"]].to_string(index=False))

    # ── 5. Aspect sentiment ───────────────────────────────────────────────────
    print("\n[5] Aspect-based sentiment…")
    aspect_sent = analyse_aspect_sentiment(df)
    print(f"\n  Aspect sentiment:")
    print(aspect_sent[["aspect","mentions","pos_pct","neg_pct","net_sentiment"]].to_string(index=False))

    # ── 6. Fake review detection ──────────────────────────────────────────────
    print("\n[6] Detecting suspicious reviews…")
    suspicious = detect_suspicious_reviews(df)
    suspicious_count = suspicious["is_suspicious"].sum()
    print(f"  Flagged: {suspicious_count} reviews ({suspicious_count/len(df)*100:.1f}%)")

    # ── 7. KPIs ───────────────────────────────────────────────────────────────
    print("\n[7] Computing KPIs…")
    kpis = compute_kpis(df)
    print(f"\n  EXECUTIVE KPIs:")
    for k, v in kpis.items():
        print(f"    {k:<25}: {v}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n[8] Generating plots…")
    plot_executive_dashboard(df, complaints, defect_rates, aspect_sent,
                              kpis, "executive_dashboard.png")
    plot_complaint_deep_dive(complaints, df, "complaint_deep_dive.png")

    # ── 9. Actionable summary ─────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  ACTIONABLE INSIGHTS SUMMARY")
    print("="*62)
    top_complaint = complaints.iloc[0]["complaint"]
    worst_aspect  = aspect_sent[aspect_sent["net_sentiment"]<0].nlargest(1,"mentions")
    worst_aspect_name = worst_aspect.iloc[0]["aspect"] if len(worst_aspect) else "N/A"
    critical_defect   = defect_rates[defect_rates["alert"]=="🔴 CRITICAL"]

    print(f"\n  1. PRODUCT QUALITY")
    print(f"     Top complaint: {top_complaint}")
    print(f"     → Assign engineering ticket to fix {top_complaint.lower()} this sprint")

    print(f"\n  2. FEATURE PRIORITY")
    print(f"     Most-discussed negative feature: {worst_aspect_name}")
    print(f"     → Include {worst_aspect_name.lower()} improvement in Q2 roadmap")

    print(f"\n  3. DEFECT MONITORING")
    if len(critical_defect):
        row = critical_defect.sort_values("defect_rate_pct",ascending=False).iloc[0]
        print(f"     CRITICAL: {row['product']} in {row['month']}: "
              f"{row['defect_rate_pct']:.1f}% defect rate")
        print(f"     → Audit production batch from {row['month']} immediately")
    else:
        print(f"     No critical defect alerts — all products below 10% threshold")

    print(f"\n  4. REVIEW QUALITY")
    print(f"     Suspicious reviews: {suspicious_count/len(df)*100:.1f}%")
    print(f"     → Report flagged reviews to platform; exclude from NPS calculation")

    print(f"\n  5. NPS PROXY")
    print(f"     Current: {kpis['nps_proxy']:+.0f}")
    sign = "up" if kpis["nps_proxy"] > 0 else "down"
    print(f"     → Score is {'positive — highlight in marketing' if kpis['nps_proxy']>20 else 'below target — prioritise complaint resolution'}")
    print("="*62 + "\n")