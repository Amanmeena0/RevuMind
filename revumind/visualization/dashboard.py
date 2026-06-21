"""
Product Review Intelligence Dashboard
======================================
Streamlit + Plotly interactive dashboard showing:
  • Sentiment trends over time (line, area, heatmap)
  • Image defect rates by product and month
  • Aspect-based sentiment breakdown
  • Top complaints with severity scoring
  • Real-time KPI cards with delta indicators
  • Product-level drill-down with filters

Run with:
    pip install streamlit plotly pandas numpy
    streamlit run dashboard.py

Then open: http://localhost:8501
"""

from collections import Counter
from datetime import datetime, timedelta
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from revumind.utils.constants import PALETTE

# ── Design tokens ─────────────────────────────────────────────────────────────
C_POS = PALETTE[0]  # teal  — positive sentiment
C_NEG = PALETTE[1]  # coral — negative / defect
C_NEU = PALETTE[2]  # amber — neutral / warning
C_PRP = PALETTE[3]  # purple
C_BLU = PALETTE[4]  # blue  — informational
C_BG = "#0F1117"  # dark background
C_CARD = "#1A1D27"  # card background
C_BORDER = "#2A2D3A"  # subtle border

PLOTLY_TEMPLATE = "plotly_dark"
FONT_FAMILY = "IBM Plex Mono, monospace"


def hex_to_rgba(hex_str: str, alpha: float) -> str:
    if hex_str.startswith("rgb"):
        if hex_str.startswith("rgba"):
            return hex_str
        return hex_str.replace(")", f", {alpha})").replace("rgb", "rgba")
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0F1117;
    color: #E8E8E8;
}

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0F1117 0%, #1A1D27 50%, #0F1117 100%);
    border-bottom: 1px solid #2A2D3A;
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #5DCAA5;
    letter-spacing: -0.03em;
    margin: 0;
}
.dash-subtitle {
    font-size: 0.8rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.pos::before { background: #5DCAA5; }
.kpi-card.neg::before { background: #D85A30; }
.kpi-card.neu::before { background: #EF9F27; }
.kpi-card.blu::before { background: #378ADD; }
.kpi-card.prp::before { background: #7F77DD; }

.kpi-label {
    font-size: 0.68rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-delta {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-delta.up   { color: #5DCAA5; }
.kpi-delta.down { color: #D85A30; }
.kpi-delta.flat { color: #888; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-left: 3px solid #5DCAA5;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem;
}

/* Alert banner */
.alert-banner {
    background: rgba(216, 90, 48, 0.12);
    border: 1px solid rgba(216, 90, 48, 0.4);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #D85A30;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.8rem;
}
.alert-banner .alert-icon { margin-right: 0.5rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1D27;
    border-right: 1px solid #2A2D3A;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #AAA;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2A2D3A;
    border-radius: 6px;
}

/* Plotly charts */
.js-plotly-plot { border-radius: 8px; }

/* Divider */
.dash-divider {
    border: none;
    border-top: 1px solid #2A2D3A;
    margin: 1.2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

PRODUCTS = ["EchoPod Pro", "SnapCam X2", "ThermoKing", "FitBand Ultra", "DeskMate Lamp"]
COMPLAINT_TYPES = [
    "Battery Drain",
    "Build Quality",
    "Defective Unit",
    "Camera Issues",
    "Software Bugs",
    "Delivery Problems",
]
ASPECTS = ["Battery", "Camera", "Display", "Build", "Software", "Delivery", "Value"]
DEFECT_KW = [
    "defective",
    "defect",
    "broken",
    "broke",
    "cracked",
    "faulty",
    "not working",
    "damaged",
]

REVIEW_CORPUS = {
    "positive": [
        "Absolutely love this product! Amazing quality and fast delivery.",
        "Best purchase this year. Outstanding build and brilliant performance.",
        "Fantastic! Exceeded all expectations. Five stars. Perfect.",
        "Superb camera quality. Battery lasts all day. Excellent value!",
        "Outstanding performance. No lag. Brilliant display. Recommended!",
    ],
    "neutral": [
        "Decent product for the price. Nothing special but works.",
        "Average quality. Works fine but nothing impressive. Okay.",
        "Mixed feelings. Some good features but also some drawbacks.",
        "Not great not terrible. Does the job. Acceptable quality.",
    ],
    "negative": [
        "Terrible product. Broke after two days. Complete waste. Avoid!",
        "Awful quality. Nothing like pictures. Very disappointed.",
        "Worst purchase. Stopped working after one week. Poor build.",
        "Horrible. Product defective. Returning immediately. Never again.",
        "Cheap plastic. Battery drains in one hour. Terrible quality.",
        "Hinge snapped on day 5. Clearly a manufacturing defect.",
        "Screen cracked on its own. Defective batch. Returning.",
    ],
}


@st.cache_data
def generate_data(n: int = 1200) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    start = datetime(2023, 1, 1)
    for i in range(n):
        prod_idx = np.random.choice(len(PRODUCTS), p=[0.35, 0.25, 0.15, 0.15, 0.10])
        days_offset = int(i * 400 / n + np.random.randint(-8, 8))
        date = start + timedelta(days=max(0, days_offset))

        # Sentiment weighted by product
        if prod_idx in [0, 3]:  # EchoPod, FitBand — mixed
            sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.40, 0.20, 0.40])
        elif prod_idx == 4:  # DeskMate — mostly negative (defect spike)
            p_neg = 0.65 if 6 <= date.month <= 10 else 0.45
            p_pos = 0.85 - p_neg
            sentiment = np.random.choice(
                ["positive", "neutral", "negative"], p=[p_pos, 0.15, p_neg]
            )
        else:
            sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.55, 0.20, 0.25])

        text = np.random.choice(REVIEW_CORPUS[sentiment])
        stars = (
            5
            if sentiment == "positive"
            else 3 if sentiment == "neutral" else np.random.choice([1, 2])
        )

        # Defect flag
        is_defect = any(k in text.lower() for k in DEFECT_KW)

        # Aspect
        aspect = np.random.choice(ASPECTS)

        # Image defect rate: higher for DeskMate months 6-10
        img_defect = (prod_idx == 4 and 6 <= date.month <= 10 and np.random.random() < 0.55) or (
            prod_idx != 4 and np.random.random() < 0.12
        )

        rows.append(
            {
                "product": PRODUCTS[prod_idx],
                "review_text": text,
                "sentiment": sentiment,
                "star_rating": stars,
                "aspect": aspect,
                "is_defect": int(is_defect),
                "img_defect": int(img_defect),
                "helpful_votes": int(np.random.exponential(5)),
                "verified": np.random.random() < 0.72,
                "review_date": date,
            }
        )

    df = pd.DataFrame(rows)
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["month"] = df["review_date"].dt.to_period("M").astype(str)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["quarter"] = df["review_date"].dt.to_period("Q").astype(str)
    return df


@st.cache_data
def build_complaint_scores(filtered_complaints: pd.DataFrame) -> pd.DataFrame:
    """
    Builds complaint metrics from the pre-computed complaint_summary table.
    """
    if len(filtered_complaints) == 0:
        return pd.DataFrame(columns=["complaint", "count", "pct", "severity"])

    # Group by topic_name and aggregate
    comp_group = filtered_complaints.groupby("topic_name").agg({
        "complaint_count": "sum",
        "severity_score": "sum"
    }).reset_index()

    complaints = comp_group.rename(columns={
        "topic_name": "complaint",
        "complaint_count": "count",
        "severity_score": "severity"
    })
    total_neg = filtered_complaints["complaint_count"].sum()
    complaints["pct"] = (complaints["count"] / max(total_neg, 1) * 100).round(1)
    complaints["severity"] = complaints["severity"].round(1)
    return complaints.sort_values("severity", ascending=False).reset_index(drop=True)



def sentiment_color(sentiment: str) -> str:
    return {
        "positive": C_POS,
        "neutral": C_NEU,
        "negative": C_NEG,
    }.get(sentiment, C_BLU)


def plotly_layout(title: str = "", height: int = 340) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=dict(
            text=title,
            font=dict(size=13, family=FONT_FAMILY, color="#CCC"),
            x=0.01,
            xanchor="left",
        ),
        font=dict(family=FONT_FAMILY, color="#CCC", size=11),
        plot_bgcolor=C_CARD,
        paper_bgcolor=C_CARD,
        margin=dict(l=8, r=8, t=40, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        ),
        xaxis=dict(gridcolor="#2A2D3A", showgrid=True, zeroline=False, title_font_size=10),
        yaxis=dict(gridcolor="#2A2D3A", showgrid=True, zeroline=False, title_font_size=10),
    )


# Load database configurations dynamically
@st.cache_data
def load_real_data_from_db() -> tuple:
    """
    Attempts to load real analyzed review records from the SQLite database.
    Maps database schema to the columns expected by the Streamlit dashboard.
    """
    import os
    import sqlite3

    db_path = "revumind.db"
    if not os.path.exists(db_path):
        return None, 0

    try:
        conn = sqlite3.connect(db_path)
        # Get total reviews count (extremely fast query)
        total_count = int(pd.read_sql_query("SELECT count(*) FROM reviews", conn).iloc[0, 0])

        # Query reviews table with calculated sentiment and joined aspect mapping (limited to 20,000 for high performance)
        query = """
            SELECT 
                r.id as id,
                r.product_id as product,
                r.review_text as review_text,
                CASE 
                    WHEN r.score >= 4 THEN 'positive'
                    WHEN r.score = 3 THEN 'neutral'
                    ELSE 'negative'
                END as sentiment,
                r.score as star_rating,
                r.helpfulness_denominator as helpful_votes,
                r.review_time as review_date,
                a.aspect_term as aspect
            FROM reviews r
            LEFT JOIN (
                SELECT review_id, MIN(aspect_term) as aspect_term 
                FROM aspect_sentiments 
                GROUP BY review_id
            ) a ON r.id = a.review_id
            ORDER BY r.review_time DESC
            LIMIT 20000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            return None, 0

        # Format columns
        df["review_date"] = pd.to_datetime(df["review_date"])
        df["verified"] = True
        df["img_defect"] = 0
        df["is_defect"] = (df["sentiment"] == "negative").astype(int)
        df["aspect"] = df["aspect"].fillna("Value").str.capitalize()

        return df, total_count
    except Exception as e:
        return None, 0


def get_dashboard_data() -> pd.DataFrame:
    """
    Tries to retrieve real database records first;
    falls back to synthetic data generation if database is empty.
    """
    db_df, total_count = load_real_data_from_db()
    if db_df is not None and len(db_df) >= 5:
        # Store total reviews count in session state for displaying in KPI cards
        st.session_state["db_total_reviews"] = total_count

        # Calculate monthly period keys required by dashboard
        db_df["month"] = db_df["review_date"].dt.to_period("M").astype(str)
        db_df["month_dt"] = pd.to_datetime(db_df["month"])
        db_df["quarter"] = db_df["review_date"].dt.to_period("Q").astype(str)
        return db_df
    return generate_data(1200)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
    <div style="font-family:'IBM Plex Mono';font-size:1rem;
                font-weight:700;color:#5DCAA5;margin-bottom:1rem;">
        ⬡ VIEW MODE
    </div>
    """,
        unsafe_allow_html=True,
    )
    view_mode = st.radio(
        "Select Mode",
        options=["Dashboard Analytics", "Model Playground (Test Reviews)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="font-family:'IBM Plex Mono';font-size:1rem;
                font-weight:700;color:#5DCAA5;margin-bottom:1rem;">
        ⬡ FILTERS
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load pre-computed aggregates directly
    import sqlite3
    import os
    from datetime import datetime

    db_path = "revumind.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        prod_sum_df = pd.read_sql_query("SELECT * FROM product_summary", conn)
        monthly_sent_df = pd.read_sql_query("SELECT * FROM monthly_sentiment", conn)
        aspect_sum_df = pd.read_sql_query("SELECT * FROM aspect_summary", conn)
        complaint_sum_df = pd.read_sql_query("SELECT * FROM complaint_summary", conn)
        conn.close()
    else:
        prod_sum_df = pd.DataFrame(columns=["product_id", "total_reviews", "average_stars", "average_helpfulness", "positive_count", "neutral_count", "negative_count"])
        monthly_sent_df = pd.DataFrame(columns=["month", "product_id", "total_reviews", "positive_count", "neutral_count", "negative_count"])
        aspect_sum_df = pd.DataFrame(columns=["product_id", "aspect_term", "positive_count", "neutral_count", "negative_count", "avg_confidence"])
        complaint_sum_df = pd.DataFrame(columns=["product_id", "topic_name", "complaint_count", "severity_score"])

    unique_products = sorted(prod_sum_df["product_id"].unique().tolist()) if len(prod_sum_df) else []
    selected_products = st.multiselect(
        "Products",
        options=unique_products,
        default=unique_products,
    )

    if len(monthly_sent_df) > 0:
        monthly_sent_df["month_dt"] = pd.to_datetime(monthly_sent_df["month"] + "-01")
        date_min = monthly_sent_df["month_dt"].min().date()
        date_max = monthly_sent_df["month_dt"].max().date()
    else:
        date_min = datetime.now().date()
        date_max = datetime.now().date()

    date_range = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    verified_only = st.checkbox("Verified purchases only", value=False)
    min_stars = st.slider("Min star rating", 1, 5, 1)

    st.markdown("---")
    st.markdown(
        """
    <div style="font-family:'IBM Plex Mono';font-size:0.65rem;color:#555;
                text-transform:uppercase;letter-spacing:0.1em;">
        Alert Thresholds
    </div>
    """,
        unsafe_allow_html=True,
    )
    defect_warn = st.slider("Defect rate warning %", 1, 20, 5)
    defect_critical = st.slider("Defect rate critical %", 5, 40, 10)

    st.markdown("---")
    st.markdown(
        """
    <div style="font-family:'IBM Plex Mono';font-size:0.62rem;color:#444;
                text-transform:uppercase;letter-spacing:0.08em;margin-top:1rem;">
        Product Review Intelligence<br>v2.1 · Built with Streamlit + Plotly
    </div>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA FILTERING (Analytics Layer Summaries)
# ══════════════════════════════════════════════════════════════════════════════

if selected_products:
    filtered_prod_sum = prod_sum_df[prod_sum_df["product_id"].isin(selected_products)]
    filtered_monthly = monthly_sent_df[monthly_sent_df["product_id"].isin(selected_products)]
    filtered_aspects = aspect_sum_df[aspect_sum_df["product_id"].isin(selected_products)]
    filtered_complaints = complaint_sum_df[complaint_sum_df["product_id"].isin(selected_products)]
else:
    filtered_prod_sum = prod_sum_df
    filtered_monthly = monthly_sent_df
    filtered_aspects = aspect_sum_df
    filtered_complaints = complaint_sum_df

# Apply date range filtering to summaries
if len(date_range) == 2 and len(filtered_monthly) > 0:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    filtered_monthly = filtered_monthly[
        (filtered_monthly["month_dt"] >= start_dt) & (filtered_monthly["month_dt"] <= end_dt)
    ]

# Dummy df skeleton to prevent unmodified visualizer script errors
df = pd.DataFrame(columns=["id", "product", "review_text", "sentiment", "star_rating", "helpful_votes", "review_date", "verified", "img_defect", "is_defect", "aspect"])


@st.cache_resource
def load_inference_engine():
    from revumind.pipeline.inference import RevuMindInferenceEngine
    return RevuMindInferenceEngine()

if view_mode == "Model Playground (Test Reviews)":
    st.markdown(
        """
    <div class="dash-header">
        <div class="dash-title">⬡ Model Inference Playground</div>
        <div class="dash-subtitle">Real-time Aspect-Based Sentiment Analysis & Helpfulness Scoring</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Try a New Review</div>', unsafe_allow_html=True)
    st.write("Test the trained RevuMind NLP and Machine Learning models on custom review texts in real-time.")

    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        test_text = st.text_area("Review Text:", placeholder="Type or paste a product review here...", height=150)
    with col_input2:
        test_rating = st.slider("Star Rating:", 1, 5, 5)
        run_analysis = st.button("Run Real-time Analysis", use_container_width=True)

    if run_analysis and test_text:
        with st.spinner("Running 7-model inference cascade..."):
            try:
                engine = load_inference_engine()
                result = engine.analyze_single_review(test_text, test_rating)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown("#### Classification Insights")

                        sentiment = result["overall_sentiment"].upper()
                        conf = result["sentiment_confidence"]
                        sent_color = C_POS if sentiment == "POSITIVE" else C_NEG if sentiment == "NEGATIVE" else C_NEU

                        st.markdown(f"**Predicted Sentiment:** <span style='color:{sent_color};font-weight:bold;'>{sentiment}</span> (Confidence: {conf})", unsafe_allow_html=True)
                        st.metric("Predicted Helpfulness Score", f"{result['predicted_helpfulness']*100:.1f}%")
                        st.markdown(f"**Assigned Topic:** {result['topic_name']}")
                        st.markdown(f"**Topic Keywords:** {', '.join(result['topic_keywords'])}")

                    with col_res2:
                        st.markdown("#### Extracted Aspects & Sentiment")
                        aspects = result["aspect_sentiments"]
                        if aspects:
                            aspect_df = pd.DataFrame(aspects)
                            st.dataframe(
                                aspect_df,
                                column_config={
                                    "aspect_term": st.column_config.TextColumn("Aspect/Feature"),
                                    "sentiment_label": st.column_config.TextColumn("Sentiment"),
                                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("No specific features/aspects detected in the review text.")
            except Exception as e:
                st.error(f"Inference failed: {str(e)}")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div class="dash-header">
    <div class="dash-title">⬡ Product Review Intelligence</div>
    <div class="dash-subtitle">Sentiment Trends · Defect Rates · Aspect Analysis · Complaint Tracking</div>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

total = int(filtered_prod_sum["total_reviews"].sum()) if len(filtered_prod_sum) > 0 else 0
if total > 0:
    avg_star = (filtered_prod_sum["average_stars"] * filtered_prod_sum["total_reviews"]).sum() / total
    pct_pos = (filtered_prod_sum["positive_count"].sum() / total) * 100
    pct_neg = (filtered_prod_sum["negative_count"].sum() / total) * 100
    pct_neu = (filtered_prod_sum["neutral_count"].sum() / total) * 100
else:
    avg_star = 0.0
    pct_pos = 0.0
    pct_neg = 0.0
    pct_neu = 0.0
defect_r = 0.0  # Image defect rate placeholder

# Trend vs prior period using pre-computed monthly_sentiment
sorted_months = sorted(filtered_monthly["month"].unique().tolist()) if len(filtered_monthly) > 0 else []
if len(sorted_months) >= 2:
    recent_month = sorted_months[-1]
    prior_month = sorted_months[-2]

    recent_m_df = filtered_monthly[filtered_monthly["month"] == recent_month]
    prior_m_df = filtered_monthly[filtered_monthly["month"] == prior_month]

    recent_total = recent_m_df["total_reviews"].sum()
    prior_total = prior_m_df["total_reviews"].sum()

    recent_pos_pct = (recent_m_df["positive_count"].sum() / recent_total * 100) if recent_total > 0 else 0.0
    prior_pos_pct = (prior_m_df["positive_count"].sum() / prior_total * 100) if prior_total > 0 else 0.0

    chg = recent_pos_pct - prior_pos_pct
    d1 = "up" if chg > 0 else "down" if chg < 0 else "flat"
    d1v = f"{'+' if chg >= 0 else ''}{chg:.1f}"
else:
    d1 = "flat"
    d1v = "—"

d2 = "flat"
d2v = "—"


def delta_str(current, prior, fmt=".1f", reverse=False):
    c = current.mean() if hasattr(current, "mean") else current
    p = prior.mean() if hasattr(prior, "mean") else prior
    chg = c - p
    direction = "up" if (chg > 0) != reverse else "down"
    if abs(chg) < 0.01:
        direction = "flat"
    sign = "+" if chg >= 0 else ""
    return direction, f"{sign}{chg:{fmt}}"


st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi-card pos">
    <div class="kpi-label">Total Reviews</div>
    <div class="kpi-value" style="color:{C_POS}">{st.session_state.get("db_total_reviews", total):,}</div>
    <div class="kpi-delta flat">across {filtered_prod_sum['product_id'].nunique() if len(filtered_prod_sum) > 0 else 0} products</div>
  </div>
  <div class="kpi-card blu">
    <div class="kpi-label">Avg Star Rating</div>
    <div class="kpi-value" style="color:{C_BLU}">⭐ {avg_star:.2f}</div>
    <div class="kpi-delta flat">out of 5.0</div>
  </div>
  <div class="kpi-card pos">
    <div class="kpi-label">% Positive</div>
    <div class="kpi-value" style="color:{C_POS}">{pct_pos:.1f}%</div>
    <div class="kpi-delta {d1}">{'▲' if d1=='up' else '▼' if d1=='down' else '—'} {d1v}% vs prev period</div>
  </div>
  <div class="kpi-card neg">
    <div class="kpi-label">% Negative</div>
    <div class="kpi-value" style="color:{C_NEG}">{pct_neg:.1f}%</div>
    <div class="kpi-delta flat">{pct_neu:.1f}% neutral</div>
  </div>
  <div class="kpi-card pos">
    <div class="kpi-label">Image Defect Rate</div>
    <div class="kpi-value" style="color:#5DCAA5">{defect_r:.1f}%</div>
    <div class="kpi-delta {d2}">—</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════════════════

critical_prods = []  # Image defects are not present in this dataset partition

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: SENTIMENT TREND + DEFECT RATE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">Sentiment Trends & Defect Rates</div>', unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2])

with col1:
    # Reconstruct monthly_sent from pre-computed monthly_sentiment table
    if len(filtered_monthly) > 0:
        monthly_sent_melted = pd.melt(
            filtered_monthly,
            id_vars=["month_dt"],
            value_vars=["positive_count", "neutral_count", "negative_count"],
            var_name="sentiment",
            value_name="count"
        )
        monthly_sent_melted["sentiment"] = monthly_sent_melted["sentiment"].str.replace("_count", "")
        monthly_sent = monthly_sent_melted.groupby(["month_dt", "sentiment"])["count"].sum().reset_index()
    else:
        monthly_sent = pd.DataFrame(columns=["month_dt", "sentiment", "count"])

    fig_sent = go.Figure()
    for sent, color in [("positive", C_POS), ("neutral", C_NEU), ("negative", C_NEG)]:
        sub = monthly_sent[monthly_sent["sentiment"] == sent].sort_values("month_dt")
        fig_sent.add_trace(
            go.Scatter(
                x=sub["month_dt"],
                y=sub["count"],
                name=sent.capitalize(),
                mode="lines",
                line=dict(color=color, width=2),
                stackgroup="one",
                fillcolor=hex_to_rgba(color, 0.35),
                hovertemplate=f"<b>{sent}</b><br>%{{x|%b %Y}}: %{{y}} reviews<extra></extra>",
            )
        )
    fig_sent.update_layout(
        **plotly_layout("Monthly Sentiment Volume", 320),
        xaxis_title="",
        yaxis_title="Reviews",
        hovermode="x unified",
    )
    st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})

with col2:
    # Defect rate over time (pre-computed placeholder or simple flat line)
    if len(filtered_monthly) > 0:
        monthly_defect = filtered_monthly.groupby("month_dt").agg(
            total=("total_reviews", "sum")
        ).reset_index()
        monthly_defect["defect_rate"] = 0.0
    else:
        monthly_defect = pd.DataFrame(columns=["month_dt", "defect_rate"])

    fig_def = go.Figure()
    # Warning band
    fig_def.add_hrect(
        y0=defect_warn,
        y1=defect_critical,
        fillcolor=C_NEU,
        opacity=0.08,
        line_width=0,
        annotation_text="WARNING",
        annotation_position="right",
        annotation_font_size=9,
        annotation_font_color=C_NEU,
    )
    # Critical band
    fig_def.add_hrect(
        y0=defect_critical,
        y1=100,
        fillcolor=C_NEG,
        opacity=0.08,
        line_width=0,
        annotation_text="CRITICAL",
        annotation_position="right",
        annotation_font_size=9,
        annotation_font_color=C_NEG,
    )
    fig_def.add_trace(
        go.Scatter(
            x=monthly_defect["month_dt"],
            y=monthly_defect["defect_rate"],
            name="Defect Rate %",
            mode="lines+markers",
            line=dict(color=C_NEG, width=2.5),
            marker=dict(size=5, color=C_NEG),
            fill="tozeroy",
            fillcolor=hex_to_rgba(C_NEG, 0.12),
            hovertemplate="<b>%{x|%b %Y}</b><br>Defect rate: %{y:.1f}%<extra></extra>",
        )
    )
    fig_def.add_hline(y=defect_warn, line_dash="dot", line_color=C_NEU, line_width=1)
    fig_def.add_hline(y=defect_critical, line_dash="dot", line_color=C_NEG, line_width=1)
    fig_def.update_layout(
        **plotly_layout("Image Defect Rate Over Time", 320),
        xaxis_title="",
        yaxis_title="Defect Rate %",
        showlegend=False,
    )
    st.plotly_chart(fig_def, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: SENTIMENT HEATMAP + ASPECT SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">Sentiment Heatmap & Aspect Analysis</div>', unsafe_allow_html=True
)

col3, col4 = st.columns([2, 3])

with col3:
    # Heatmap: product × month avg rating
    if len(filtered_monthly) > 0:
        # Calculate estimated star_rating per month per product
        m_data = filtered_monthly.copy()
        m_data["star_rating"] = (
            m_data["positive_count"] * 4.5
            + m_data["neutral_count"] * 3.0
            + m_data["negative_count"] * 1.5
        ) / m_data["total_reviews"].clip(lower=1)
        heatmap_data = m_data.pivot(index="product_id", columns="month", values="star_rating")
        # Keep last 12 months
        heatmap_data = heatmap_data.iloc[:, -12:]
    else:
        heatmap_data = pd.DataFrame()

    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale=[
            [0, "#D85A30"],
            [0.4, "#EF9F27"],
            [0.7, "#5DCAA5"],
            [1, "#5DCAA5"],
        ],
        zmin=1,
        zmax=5,
        labels=dict(color="Avg ★"),
        aspect="auto",
    )
    fig_heat.update_traces(
        hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Avg rating: %{z:.2f}<extra></extra>",
        texttemplate="%{z:.1f}",
        textfont_size=9,
    )
    fig_heat.update_layout(
        **plotly_layout("Avg Star Rating — Product × Month", 340),
        xaxis_title="",
        yaxis_title="",
        coloraxis_colorbar=dict(
            title="Avg ★",
            thickness=10,
            len=0.6,
            tickfont=dict(size=9),
        ),
    )
    fig_heat.update_xaxes(tickangle=45, tickfont_size=8)
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

with col4:
    # Reconstruct aspect sentiments diverging bar from pre-computed summary table
    if len(filtered_aspects) > 0:
        aspect_group = filtered_aspects.groupby("aspect_term").agg({
            "positive_count": "sum",
            "neutral_count": "sum",
            "negative_count": "sum"
        }).reset_index()

        aspect_rows = []
        for idx, row in aspect_group.iterrows():
            pos_c = row["positive_count"]
            neg_c = row["negative_count"]
            neu_c = row["neutral_count"]
            total_c = pos_c + neg_c + neu_c
            if total_c == 0:
                continue
            aspect_rows.append({
                "aspect": row["aspect_term"].capitalize(),
                "mentions": total_c,
                "pos_pct": pos_c / total_c * 100,
                "neg_pct": neg_c / total_c * 100,
                "net": (pos_c - neg_c) / total_c
            })
        asp_df = pd.DataFrame(aspect_rows)
        if len(asp_df) > 0:
            asp_df = asp_df.sort_values("net")
        else:
            asp_df = pd.DataFrame(columns=["aspect", "mentions", "pos_pct", "neg_pct", "net"])
    else:
        asp_df = pd.DataFrame(columns=["aspect", "mentions", "pos_pct", "neg_pct", "net"])

    fig_asp = go.Figure()
    fig_asp.add_trace(
        go.Bar(
            y=asp_df["aspect"],
            x=asp_df["pos_pct"],
            name="Positive %",
            orientation="h",
            marker_color=C_POS,
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Positive: %{x:.1f}%<extra></extra>",
        )
    )
    fig_asp.add_trace(
        go.Bar(
            y=asp_df["aspect"],
            x=-asp_df["neg_pct"],
            name="Negative %",
            orientation="h",
            marker_color=C_NEG,
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Negative: %{customdata:.1f}%<extra></extra>",
            customdata=asp_df["neg_pct"],
        )
    )
    layout_dict = plotly_layout("Aspect Sentiment — Diverging View", 340)
    layout_dict["xaxis"].update(
        tickvals=[-60, -40, -20, 0, 20, 40, 60, 80],
        ticktext=["60", "40", "20", "0", "20", "40", "60", "80"],
    )
    fig_asp.update_layout(
        **layout_dict,
        barmode="relative",
        xaxis_title="← Negative %  |  Positive % →",
        yaxis_title="",
    )
    fig_asp.add_vline(x=0, line_color="#888", line_width=1)
    st.plotly_chart(fig_asp, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: DEFECT BY PRODUCT + COMPLAINT SEVERITY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">Defect Analysis & Complaint Severity</div>',
    unsafe_allow_html=True,
)

col5, col6 = st.columns([2, 3])

with col5:
    # Defect rate per product — horizontal bar with colour coding
    if len(filtered_prod_sum) > 0:
        prod_defect = filtered_prod_sum.rename(columns={"product_id": "product"})
        prod_defect["rate"] = 0.0
        prod_defect["defects"] = 0
        prod_defect = prod_defect.sort_values("rate", ascending=True)
    else:
        prod_defect = pd.DataFrame(columns=["product", "rate", "defects"])
    bar_colors = [
        C_NEG if r > defect_critical else C_NEU if r > defect_warn else C_POS
        for r in prod_defect["rate"]
    ]
    fig_prod = go.Figure(
        go.Bar(
            x=prod_defect["rate"],
            y=prod_defect["product"],
            orientation="h",
            marker_color=bar_colors,
            marker_opacity=0.85,
            text=[f"{r:.1f}%" for r in prod_defect["rate"]],
            textposition="outside",
            textfont=dict(size=10, family=FONT_FAMILY),
            hovertemplate="<b>%{y}</b><br>Defect rate: %{x:.1f}%<br>Defects: %{customdata}<extra></extra>",
            customdata=prod_defect["defects"],
        )
    )
    fig_prod.add_vline(
        x=defect_warn,
        line_dash="dot",
        line_color=C_NEU,
        line_width=1.5,
        annotation_text=f"Warn {defect_warn}%",
        annotation_font_size=8,
    )
    fig_prod.add_vline(
        x=defect_critical,
        line_dash="dot",
        line_color=C_NEG,
        line_width=1.5,
        annotation_text=f"Crit {defect_critical}%",
        annotation_font_size=8,
    )
    fig_prod.update_layout(
        **plotly_layout("Image Defect Rate by Product", 300),
        xaxis_title="Defect Rate %",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_prod, use_container_width=True, config={"displayModeBar": False})

with col6:
    # Complaint severity bubble chart
    complaints = build_complaint_scores(filtered_complaints)
    fig_bubble = px.scatter(
        complaints,
        x="count",
        y="severity",
        size="severity",
        color="complaint",
        color_discrete_sequence=[C_NEG, C_NEU, C_BLU, C_POS, C_PRP, "#D4537E"],
        text="complaint",
        hover_data={"count": True, "severity": True, "pct": True},
        size_max=45,
        labels={"count": "Mention Count", "severity": "Severity Score", "pct": "% of Neg Reviews"},
    )
    fig_bubble.update_traces(
        textposition="top center",
        textfont=dict(size=9, family=FONT_FAMILY),
        hovertemplate="<b>%{text}</b><br>Mentions: %{x}<br>Severity: %{y:.0f}"
        "<br>% neg reviews: %{customdata[2]:.1f}%<extra></extra>",
    )
    fig_bubble.update_layout(
        **plotly_layout("Top Complaints — Severity Matrix", 300),
        showlegend=False,
        xaxis_title="Mention Count →",
        yaxis_title="Severity Score →",
    )
    fig_bubble.add_annotation(
        text="↗ Highest priority",
        x=0.85,
        y=0.95,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=9, color="#555"),
    )
    st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# ROW 4: PRODUCT DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Product Drill-Down</div>', unsafe_allow_html=True)

selected_product = st.selectbox(
    "Select product to drill down:",
    options=unique_products,
    key="drill_down",
)

# 🤖 AI Summary block
st.markdown('<div class="section-header">🤖 AI Executive Summary</div>', unsafe_allow_html=True)
if selected_product:
    conn = sqlite3.connect("revumind.db")
    summary_record = conn.execute("SELECT summary_text FROM summaries WHERE product_id = ? AND cohort_type = 'all' LIMIT 1", (selected_product,)).fetchone()
    conn.close()

    if summary_record:
        ai_summary = summary_record[0]
        st.info(ai_summary)
    else:
        # Check if there are reviews to generate summary
        conn = sqlite3.connect("revumind.db")
        cursor = conn.cursor()
        cursor.execute("SELECT review_text FROM reviews WHERE product_id = ? LIMIT 30", (selected_product,))
        sample_texts = [r[0] for r in cursor.fetchall()]
        conn.close()

        if sample_texts:
            @st.cache_data
            def get_cached_summary(texts_tuple: tuple) -> str:
                from revumind.models.summarizer.model import ExecutiveSummarizer
                summarizer = ExecutiveSummarizer(use_bart=False)
                return summarizer.generate_summary(list(texts_tuple))

            with st.spinner("Analyzing and summarizing product reviews on-the-fly..."):
                ai_summary = get_cached_summary(tuple(sample_texts))
            st.info(ai_summary)
        else:
            st.info("No reviews available for this product to summarize.")
else:
    st.info("Please select a product to display summary.")

col7, col8, col9 = st.columns(3)

with col7:
    # Star rating distribution - queried dynamically using quick count (indexed scan)
    if selected_product:
        conn = sqlite3.connect("revumind.db")
        cursor = conn.cursor()
        cursor.execute("SELECT score, COUNT(*) FROM reviews WHERE product_id = ? GROUP BY score", (selected_product,))
        counts = dict(cursor.fetchall())
        conn.close()
        star_counts = pd.Series({i: counts.get(i, 0) for i in range(1, 6)})
    else:
        star_counts = pd.Series({1: 0, 2: 0, 3: 0, 4: 0, 5: 0})

    star_colors = {1: C_NEG, 2: "#E87040", 3: C_NEU, 4: "#7CC97A", 5: C_POS}
    fig_stars = go.Figure(
        go.Bar(
            x=star_counts.index.astype(str),
            y=star_counts.values,
            marker_color=[star_colors.get(s, C_BLU) for s in star_counts.index],
            marker_opacity=0.85,
            text=star_counts.values,
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}★</b><br>Count: %{y}<extra></extra>",
        )
    )
    fig_stars.update_layout(
        **plotly_layout(f"{selected_product} — Rating Distribution", 260),
        xaxis_title="Star Rating",
        yaxis_title="Count",
        showlegend=False,
    )
    st.plotly_chart(fig_stars, use_container_width=True, config={"displayModeBar": False})

with col8:
    # Monthly defect rate for selected product (pre-computed from monthly_sentiment)
    if selected_product and len(monthly_sent_df) > 0:
        prod_monthly = monthly_sent_df[monthly_sent_df["product_id"] == selected_product].copy()
        prod_monthly["total"] = prod_monthly["total_reviews"]
        prod_monthly["rate"] = 0.0
    else:
        prod_monthly = pd.DataFrame(columns=["month_dt", "total", "rate"])

    line_colors = [C_POS] * len(prod_monthly)
    fig_prod_def = go.Figure()
    fig_prod_def.add_trace(
        go.Scatter(
            x=prod_monthly["month_dt"] if len(prod_monthly) else [],
            y=prod_monthly["rate"] if len(prod_monthly) else [],
            mode="lines+markers",
            line=dict(color=C_BLU, width=2),
            marker=dict(size=6, color=line_colors),
            fill="tozeroy",
            fillcolor=hex_to_rgba(C_BLU, 0.13),
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        )
    )
    fig_prod_def.update_layout(
        **plotly_layout(f"{selected_product} — Monthly Defect Rate", 260),
        xaxis_title="",
        yaxis_title="Defect %",
        showlegend=False,
    )
    st.plotly_chart(fig_prod_def, use_container_width=True, config={"displayModeBar": False})

with col9:
    # Sentiment pie - pre-computed from product_summary
    if selected_product and len(prod_sum_df) > 0:
        prod_sum_rows = prod_sum_df[prod_sum_df["product_id"] == selected_product]
        if len(prod_sum_rows) > 0:
            prod_sum_row = prod_sum_rows.iloc[0]
            sent_counts = pd.Series({
                "positive": prod_sum_row["positive_count"],
                "neutral": prod_sum_row["neutral_count"],
                "negative": prod_sum_row["negative_count"]
            })
            avg_rating = prod_sum_row["average_stars"]
        else:
            sent_counts = pd.Series({"positive": 0, "neutral": 0, "negative": 0})
            avg_rating = 0.0
    else:
        sent_counts = pd.Series({"positive": 0, "neutral": 0, "negative": 0})
        avg_rating = 0.0

    fig_pie = go.Figure(
        go.Pie(
            labels=sent_counts.index.str.capitalize(),
            values=sent_counts.values,
            marker_colors=[sentiment_color(s) for s in sent_counts.index],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=10, family=FONT_FAMILY),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig_pie.add_annotation(
        text=f"{avg_rating:.1f}★",
        x=0.5,
        y=0.5,
        font_size=18,
        showarrow=False,
        font_color=C_POS,
        font_family=FONT_FAMILY,
    )
    layout_dict = plotly_layout(f"{selected_product} — Sentiment Split", 260)
    layout_dict["legend"].update(font_size=9, orientation="h", y=-0.05)
    fig_pie.update_layout(
        **layout_dict,
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# ROW 5: RECENT REVIEWS TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Recent Reviews</div>', unsafe_allow_html=True)

# Colour-map function for dataframe display
show_cols = [
    "review_date",
    "product",
    "star_rating",
    "sentiment",
    "review_text",
    "verified",
    "helpful_votes",
    "img_defect",
]

# Filters on the table (moved up to build query)
search_term = st.text_input(
    "🔍 Search reviews:",
    placeholder="Type keyword to filter…",
    label_visibility="collapsed",
)

# Fetch top 50 recent reviews directly from SQLite using fast query (on-demand lazy load)
conn = sqlite3.connect("revumind.db")
query_params = []
where_clauses = ["1=1"]

if selected_products:
    where_clauses.append("product_id IN ({})".format(",".join("?" for _ in selected_products)))
    query_params.extend(selected_products)

if search_term:
    where_clauses.append("review_text LIKE ?")
    query_params.append(f"%{search_term}%")

query = """
    SELECT 
        review_time as review_date,
        product_id as product,
        score as star_rating,
        CASE 
            WHEN score >= 4 THEN 'positive'
            WHEN score = 3 THEN 'neutral'
            ELSE 'negative'
        END as sentiment,
        review_text,
        1 as verified,
        helpfulness_denominator as helpful_votes,
        0 as img_defect
    FROM reviews
    WHERE {}
    ORDER BY review_time DESC
    LIMIT 50
""".format(" AND ".join(where_clauses))

recent_raw = pd.read_sql_query(query, conn, params=query_params)
conn.close()

if len(recent_raw) > 0:
    recent_raw["review_date"] = pd.to_datetime(recent_raw["review_date"])
    recent = recent_raw[show_cols].rename(
        columns={
            "review_date": "Date",
            "product": "Product",
            "star_rating": "★",
            "sentiment": "Sentiment",
            "review_text": "Review",
            "verified": "Verified",
            "helpful_votes": "Helpful",
            "img_defect": "ImgDefect",
        }
    )
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
else:
    recent = pd.DataFrame(columns=["Date", "Product", "★", "Sentiment", "Review", "Verified", "Helpful", "ImgDefect"])


st.dataframe(
    recent,
    use_container_width=True,
    height=240,
    column_config={
        "★": st.column_config.NumberColumn(format="%.0f ⭐"),
        "Verified": st.column_config.CheckboxColumn(),
        "ImgDefect": st.column_config.CheckboxColumn("🔴 Defect"),
        "Review": st.column_config.TextColumn(width="large"),
        "Helpful": st.column_config.NumberColumn(format="%d 👍"),
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<hr class="dash-divider">
<div style="font-family:'IBM Plex Mono';font-size:0.65rem;color:#444;
            text-align:center;padding:0.5rem 0;">
    Product Review Intelligence Dashboard · Built with Streamlit + Plotly ·
    Data refreshes on filter change · All insights are auto-computed
</div>
""",
    unsafe_allow_html=True,
)
