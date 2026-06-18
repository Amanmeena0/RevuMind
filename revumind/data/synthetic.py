"""
Synthetic dataset generation for product reviews.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PRODUCTS = ["EchoPod Pro", "SnapCam X2", "ThermoKing Bottle", "FitBand Ultra", "DeskMate Lamp"]
BRANDS   = ["TechNova", "PixelCraft", "CoolWare", "FitTech", "LumiCo"]

REVIEW_TEMPLATES = [
    # Battery complaints
    ("Battery drains so fast. Barely lasts {n} hours on a full charge. "
     "Very disappointed with the battery life for this price.", -1, "battery"),
    ("Battery life is terrible. Had to charge {n} times a day. "
     "Defective product or very poor quality cells used.", -1, "battery"),
    ("Excellent battery life! Lasts {n} days easily. Very impressed.", +1, "battery"),
    # Camera
    ("Camera quality is blurry in low light. Photos look grainy at night. "
     "Expected much better for the price paid.", -1, "camera"),
    ("Camera is absolutely stunning. Crystal clear photos even at night. "
     "Best camera I have used in this price range.", +1, "camera"),
    # Build/defect
    ("Product broke after {n} days. Cheap plastic build. Complete waste of money.", -1, "build"),
    ("Hinge snapped on day {n}. Clearly a manufacturing defect. "
     "Returning immediately. Poor quality control.", -1, "defect"),
    ("Screen cracked on its own within {n} weeks. Defective batch probably. "
     "Customer service refused to replace.", -1, "defect"),
    ("Build quality is excellent. Feels premium and solid. "
     "Has survived drops multiple times without damage.", +1, "build"),
    # Delivery
    ("Package arrived damaged. Box was crushed. "
     "Product inside also scratched. Terrible packaging.", -1, "delivery"),
    ("Delivery took {n} extra days. No tracking updates. "
     "Very poor logistics experience.", -1, "delivery"),
    ("Super fast delivery. Arrived in {n} days. "
     "Packaging was excellent. Very impressed.", +1, "delivery"),
    # Software/app
    ("App crashes constantly. Cannot connect to device. "
     "Software is full of bugs. Completely unusable.", -1, "software"),
    ("App is slow and outdated. No updates in {n} months. "
     "Competitors have much better apps.", -1, "software"),
    ("App works perfectly. Intuitive interface. "
     "Regular updates with new features. Love it.", +1, "software"),
    # Value
    ("Overpriced for what you get. Competitors offer better specs "
     "for less money. Not worth the premium price.", -1, "value"),
    ("Excellent value for money. Better than alternatives "
     "costing {n} percent more. Highly recommend.", +1, "value"),
    # Positive overall
    ("Amazing product! Exceeded all expectations. "
     "Will definitely buy again and recommend to friends.", +1, "general"),
    ("Perfect in every way. Five stars. "
     "Best purchase I have made this year.", +1, "general"),
    # Competitive
    ("Switched from CompetitorBrand and this is so much better. "
     "No comparison. This wins on every metric.", +1, "comparison"),
    ("Was using BrandX before this. Huge downgrade. "
     "Going back to my old device.", -1, "comparison"),
]

def generate_review_data(n: int = 800) -> pd.DataFrame:
    """Build a realistic product review dataset with time dimension."""
    rows = []
    start_date = datetime(2023, 1, 1)

    for i in range(n):
        product_idx = np.random.choice(len(PRODUCTS), p=[0.35,0.25,0.15,0.15,0.10])
        template, sentiment, aspect = REVIEW_TEMPLATES[i % len(REVIEW_TEMPLATES)]

        days_offset = int(i * 365 / n + np.random.randint(-10, 10))
        review_date = start_date + timedelta(days=max(0, days_offset))

        # Inject defect spike: product 0, months 8-10
        if product_idx == 0 and 7 <= review_date.month <= 10 and np.random.random() < 0.4:
            template, sentiment, aspect = REVIEW_TEMPLATES[6]   # hinge defect

        # Fill template placeholders
        text = template.format(n=np.random.randint(2, 20))

        # Add noise
        if np.random.random() < 0.15:
            text += " " + np.random.choice([
                "Would not buy again.", "Highly recommended!",
                "Save your money.", "Best purchase ever.", "Very average.",
            ])

        rows.append({
            "product":       PRODUCTS[product_idx],
            "brand":         BRANDS[product_idx],
            "review_text":   text,
            "star_rating":   5 if sentiment == 1 else np.random.choice([1, 2]),
            "aspect":        aspect,
            "sentiment":     sentiment,
            "helpful_votes": int(np.random.exponential(5)),
            "verified":      np.random.random() < 0.72,
            "review_date":   review_date,
            "review_length": len(text.split()),
        })

    df = pd.DataFrame(rows)
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["month"]       = df["review_date"].dt.to_period("M").astype(str)
    return df
