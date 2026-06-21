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
def build_complaint_scores(df: pd.DataFrame) -> pd.DataFrame:
    COMPLAINT_KW = {
        "Battery Drain": ["battery", "drain", "charge", "mah"],
        "Build Quality": ["broke", "broken", "cracked", "cheap", "plastic", "flimsy"],
        "Defective Unit": ["defective", "defect", "faulty", "not working"],
        "Camera Issues": ["blurry", "grainy", "camera", "low light"],
        "Software Bugs": ["crash", "bug", "slow app", "update", "unusable"],
        "Delivery Problems": ["damaged", "late", "shipping", "packaging", "wrong item"],
    }
    WEIGHTS = {
        "Defective Unit": 3,
        "Build Quality": 2.5,
        "Battery Drain": 2,
        "Camera Issues": 1.8,
        "Software Bugs": 1.7,
        "Delivery Problems": 1.5,
    }
    neg = df[df["star_rating"] <= 3]
    rows = []
    for complaint, kws in COMPLAINT_KW.items():
        pat = "|".join(re.escape(k) for k in kws)
        cnt = neg["review_text"].str.lower().str.contains(pat, regex=True, na=False).sum()
        if cnt == 0:
            continue
        avg_stars = neg.loc[
            neg["review_text"].str.lower().str.contains(pat, regex=True, na=False), "star_rating"
        ].mean()
        rows.append(
            {
                "complaint": complaint,
                "count": cnt,
                "pct": round(cnt / max(len(neg), 1) * 100, 1),
                "severity": round(cnt * WEIGHTS.get(complaint, 1) * (4 - avg_stars), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("severity", ascending=False).reset_index(drop=True)


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

    df_all = get_dashboard_data()

    unique_products = sorted(df_all["product"].unique().tolist())
    selected_products = st.multiselect(
        "Products",
        options=unique_products,
        default=unique_products,
    )

    date_min = df_all["review_date"].min().date()
    date_max = df_all["review_date"].max().date()
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
# DATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════

df = df_all.copy()

if selected_products:
    df = df[df["product"].isin(selected_products)]

if len(date_range) == 2:
    df = df[
        (df["review_date"].dt.date >= date_range[0]) & (df["review_date"].dt.date <= date_range[1])
    ]

if verified_only:
    df = df[df["verified"]]

df = df[df["star_rating"] >= min_stars]

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

total = len(df)
avg_star = df["star_rating"].mean()
pct_pos = (df["sentiment"] == "positive").mean() * 100
pct_neg = (df["sentiment"] == "negative").mean() * 100
defect_r = df["img_defect"].mean() * 100

# Trend vs prior period
cutoff = df["review_date"].max() - pd.Timedelta(days=30)
prev_c = df["review_date"].max() - pd.Timedelta(days=60)
recent_df = df[df["review_date"] >= cutoff]
prior_df = df[(df["review_date"] >= prev_c) & (df["review_date"] < cutoff)]


def delta_str(current, prior, fmt=".1f", reverse=False):
    if hasattr(prior, "__len__") and len(prior) == 0:
        return "flat", "—"
    c = current.mean() if hasattr(current, "mean") else current
    p = prior.mean() if hasattr(prior, "mean") else prior
    chg = c - p
    direction = "up" if (chg > 0) != reverse else "down"
    if abs(chg) < 0.01:
        direction = "flat"
    sign = "+" if chg >= 0 else ""
    return direction, f"{sign}{chg:{fmt}}"


d1, d1v = delta_str(
    (recent_df["sentiment"] == "positive").mean() * 100,
    (prior_df["sentiment"] == "positive").mean() * 100 if len(prior_df) else 0,
)
d2, d2v = delta_str(
    recent_df["img_defect"].mean() * 100 if len(recent_df) else 0,
    prior_df["img_defect"].mean() * 100 if len(prior_df) else 0,
    reverse=True,
)

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi-card pos">
    <div class="kpi-label">Total Reviews</div>
    <div class="kpi-value" style="color:{C_POS}">{st.session_state.get("db_total_reviews", total):,}</div>
    <div class="kpi-delta flat">across {df['product'].nunique()} products</div>
  </div>
  <div class="kpi-card blu">
    <div class="kpi-label">Avg Star Rating</div>
    <div class="kpi-value" style="color:{C_BLU}">⭐ {avg_star:.2f}</div>
    <div class="kpi-delta flat">out of 5.0</div>
  </div>
  <div class="kpi-card pos">
    <div class="kpi-label">% Positive</div>
    <div class="kpi-value" style="color:{C_POS}">{pct_pos:.1f}%</div>
    <div class="kpi-delta {d1}">{'▲' if d1=='up' else '▼' if d1=='down' else '—'} {d1v}% vs prev 30d</div>
  </div>
  <div class="kpi-card neg">
    <div class="kpi-label">% Negative</div>
    <div class="kpi-value" style="color:{C_NEG}">{pct_neg:.1f}%</div>
    <div class="kpi-delta flat">{(df['sentiment']=='neutral').mean()*100:.1f}% neutral</div>
  </div>
  <div class="kpi-card {'neg' if defect_r > defect_critical else 'neu' if defect_r > defect_warn else 'pos'}">
    <div class="kpi-label">Image Defect Rate</div>
    <div class="kpi-value" style="color:{'#D85A30' if defect_r > defect_critical else '#EF9F27' if defect_r > defect_warn else '#5DCAA5'}">{defect_r:.1f}%</div>
    <div class="kpi-delta {d2}">{'▲' if d2=='up' else '▼' if d2=='down' else '—'} {d2v}% vs prev 30d</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════════════════

defect_by_prod = df.groupby("product")["img_defect"].mean() * 100
critical_prods = defect_by_prod[defect_by_prod > defect_critical]
if len(critical_prods):
    for prod, rate in critical_prods.items():
        st.markdown(
            f"""
        <div class="alert-banner">
            <span class="alert-icon">🔴</span>
            CRITICAL DEFECT ALERT — <strong>{prod}</strong>:
            {rate:.1f}% defect rate exceeds {defect_critical}% threshold.
            Audit production batch immediately.
        </div>
        """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: SENTIMENT TREND + DEFECT RATE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">Sentiment Trends & Defect Rates</div>', unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2])

with col1:
    # Stacked area chart: monthly sentiment counts
    monthly_sent = df.groupby(["month_dt", "sentiment"]).size().reset_index(name="count")
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
    # Defect rate over time — line with threshold bands
    monthly_defect = (
        df.groupby("month_dt")
        .agg(
            total=("review_text", "count"),
            defects=("img_defect", "sum"),
        )
        .reset_index()
    )
    monthly_defect["defect_rate"] = monthly_defect["defects"] / monthly_defect["total"] * 100

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
    heatmap_data = (
        df.groupby(["product", "month"])["star_rating"]
        .mean()
        .reset_index()
        .pivot(index="product", columns="month", values="star_rating")
    )
    # Keep last 12 months
    heatmap_data = heatmap_data.iloc[:, -12:]

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
    # Aspect sentiment: diverging bar
    POS_KW = {
        "excellent",
        "amazing",
        "great",
        "good",
        "love",
        "perfect",
        "outstanding",
        "brilliant",
        "clear",
        "smooth",
        "fast",
        "best",
    }
    NEG_KW = {
        "terrible",
        "awful",
        "poor",
        "bad",
        "broken",
        "worst",
        "horrible",
        "cheap",
        "slow",
        "crash",
        "blurry",
        "defective",
        "disappointed",
    }
    ASPECT_KW = {
        "Battery": ["battery", "charge", "drain"],
        "Camera": ["camera", "photo", "lens", "picture"],
        "Display": ["screen", "display", "brightness"],
        "Build": ["build", "plastic", "metal", "design"],
        "Software": ["app", "software", "bug", "crash"],
        "Delivery": ["delivery", "shipping", "packaging"],
        "Value": ["price", "value", "worth", "expensive"],
        "Performance": ["speed", "fast", "slow", "lag", "performance"],
    }
    aspect_rows = []
    for aspect, kws in ASPECT_KW.items():
        pat = "|".join(r"\b" + k + r"\b" for k in kws)
        mask = df["review_text"].str.lower().str.contains(pat, regex=True, na=False)
        sub = df[mask]["review_text"].str.lower()
        if len(sub) == 0:
            continue
        pos_c = sub.apply(lambda t: any(w in t for w in POS_KW)).sum()
        neg_c = sub.apply(lambda t: any(w in t for w in NEG_KW)).sum()
        net = (pos_c - neg_c) / max(len(sub), 1)
        aspect_rows.append(
            {
                "aspect": aspect,
                "mentions": len(sub),
                "pos_pct": pos_c / max(len(sub), 1) * 100,
                "neg_pct": neg_c / max(len(sub), 1) * 100,
                "net": net,
            }
        )
    asp_df = pd.DataFrame(aspect_rows).sort_values("net")

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
    prod_defect = (
        df.groupby("product")
        .agg(total=("img_defect", "count"), defects=("img_defect", "sum"))
        .reset_index()
    )
    prod_defect["rate"] = prod_defect["defects"] / prod_defect["total"] * 100
    prod_defect = prod_defect.sort_values("rate", ascending=True)
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
    complaints = build_complaint_scores(df)
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
    options=df["product"].unique().tolist(),
    key="drill_down",
)

prod_df = df[df["product"] == selected_product]

# 🤖 AI Summary block
@st.cache_resource
def get_summarizer():
    from revumind.models.summarizer.model import ExecutiveSummarizer
    return ExecutiveSummarizer(use_bart=False)

st.markdown('<div class="section-header">🤖 AI Executive Summary</div>', unsafe_allow_html=True)
summary_texts = prod_df["review_text"].dropna().tolist()
if summary_texts:
    @st.cache_data
    def get_cached_summary(texts_tuple: tuple) -> str:
        summarizer = get_summarizer()
        return summarizer.generate_summary(list(texts_tuple))
    
    # Pass a representative sample of up to 30 reviews for quick summarization
    import random
    random.seed(42)
    sample_size = min(len(summary_texts), 30)
    sampled_texts = random.sample(summary_texts, sample_size)
    
    with st.spinner("Analyzing and summarizing product reviews..."):
        ai_summary = get_cached_summary(tuple(sampled_texts))
    st.info(ai_summary)
else:
    st.info("No reviews available for this product to summarize.")

col7, col8, col9 = st.columns(3)

with col7:
    # Star rating distribution
    star_counts = prod_df["star_rating"].value_counts().sort_index()
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
    # Monthly defect rate for selected product
    prod_monthly = (
        prod_df.groupby("month_dt")
        .agg(total=("img_defect", "count"), defects=("img_defect", "sum"))
        .reset_index()
    )
    prod_monthly["rate"] = prod_monthly["defects"] / prod_monthly["total"] * 100

    line_colors = [
        C_NEG if r > defect_critical else C_NEU if r > defect_warn else C_POS
        for r in prod_monthly["rate"]
    ]
    fig_prod_def = go.Figure()
    fig_prod_def.add_trace(
        go.Scatter(
            x=prod_monthly["month_dt"],
            y=prod_monthly["rate"],
            mode="lines+markers",
            line=dict(color=C_BLU, width=2),
            marker=dict(size=6, color=line_colors),
            fill="tozeroy",
            fillcolor=hex_to_rgba(C_BLU, 0.13),
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        )
    )
    fig_prod_def.add_hline(
        y=defect_critical,
        line_dash="dash",
        line_color=C_NEG,
        line_width=1.5,
        annotation_text=f"Critical ({defect_critical}%)",
        annotation_font_size=8,
        annotation_font_color=C_NEG,
    )
    fig_prod_def.update_layout(
        **plotly_layout(f"{selected_product} — Monthly Defect Rate", 260),
        xaxis_title="",
        yaxis_title="Defect %",
        showlegend=False,
    )
    st.plotly_chart(fig_prod_def, use_container_width=True, config={"displayModeBar": False})

with col9:
    # Sentiment pie
    sent_counts = prod_df["sentiment"].value_counts()
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
        text=f"{prod_df['star_rating'].mean():.1f}★",
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
recent = (
    df.sort_values("review_date", ascending=False)
    .head(50)[show_cols]
    .rename(
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
)
recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

# Filters on the table
search_term = st.text_input(
    "🔍 Search reviews:",
    placeholder="Type keyword to filter…",
    label_visibility="collapsed",
)
if search_term:
    recent = recent[recent["Review"].str.contains(search_term, case=False, na=False)]

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
