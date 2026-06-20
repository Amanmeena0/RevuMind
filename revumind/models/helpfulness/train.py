"""
XGBoost Helpfulness Predictor Training Pipeline
===============================================
Extracts features from preprocessed reviews (text statistics, readability index,
rating, metadata) and trains a regression model to predict the helpfulness score
(HelpfulnessNumerator / HelpfulnessDenominator).
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost, otherwise fall back to Scikit-Learn Gradient Boosting
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    XGB_AVAILABLE = False

# Import VADER for fallback sentiment feature
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import local utilities
from revumind.utils.readability import calculate_readability

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Common product aspects to find aspect mentions
ASPECT_KEYWORDS = [
    "battery",
    "charger",
    "display",
    "screen",
    "camera",
    "price",
    "sound",
    "audio",
    "shipping",
    "delivery",
    "service",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes tabular features for each review.
    If downstream transformer outputs (e.g. RoBERTa sentiment, spaCy entities, BERTopic ids)
    are missing, computes heuristic fallbacks.
    """
    logger.info("Extracting readability and text features...")

    # 1. Text statistics & readability
    readability_features = []
    for text in df["clean_review_text"].astype(str):
        metrics = calculate_readability(text)
        readability_features.append(metrics)

    df_readability = pd.DataFrame(readability_features)
    for col in df_readability.columns:
        df[col] = df_readability[col].values

    # 2. Score mapping
    df["rating_normalized"] = df["Score"] / 5.0

    # 3. Time features: Delta in days relative to the latest review in dataset
    # Handles both Unix timestamp integers and string dates
    if df["Time"].dtype in [np.int64, np.float64, np.int32]:
        review_dates = pd.to_datetime(df["Time"], unit="s")
    else:
        review_dates = pd.to_datetime(df["Time"])

    max_date = review_dates.max()
    df["time_delta_days"] = (max_date - review_dates).dt.total_seconds() / (24 * 3600)

    # 4. Fallback Sentiment feature (RoBERTa mock/fallback using VADER)
    if "sentiment_polarity" not in df.columns:
        logger.info("sentiment_polarity not found in columns. Computing VADER fallback...")
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        vader_scores = []
        for text in df["clean_review_text"].astype(str):
            score = sia.polarity_scores(text)
            # Polarity = positive score - negative score
            vader_scores.append(
                {
                    "sentiment_polarity": score["compound"],
                    "sentiment_confidence": max(score["pos"], score["neg"], score["neu"]),
                }
            )
        df_vader = pd.DataFrame(vader_scores)
        df["sentiment_polarity"] = df_vader["sentiment_polarity"].values
        df["sentiment_confidence"] = df_vader["sentiment_confidence"].values

    # 5. Fallback Entity density feature (spaCy mock/fallback)
    if "entity_density" not in df.columns:
        logger.info(
            "entity_density not found in columns. Computing capitalization-based fallback..."
        )

        # Simple heuristic: count capitalized words (excluding first words)
        def count_capitalized_entities(text):
            words = text.split()
            if len(words) <= 1:
                return 0
            # Count capitalized words that are not the first word of sentences
            capital_count = sum(1 for i, w in enumerate(words[1:]) if w and w[0].isupper())
            return capital_count / len(words)

        df["entity_density"] = (
            df["clean_review_text"].astype(str).apply(count_capitalized_entities)
        )

    # 6. Fallback Aspect density feature (DeBERTa ABSA mock/fallback)
    if "aspect_density" not in df.columns:
        logger.info("aspect_density not found. Computing aspect-keyword fallback...")

        def count_aspect_density(text):
            text_lower = text.lower()
            aspect_count = sum(1 for w in ASPECT_KEYWORDS if w in text_lower)
            words = text_lower.split()
            if len(words) == 0:
                return 0.0
            return aspect_count / len(words)

        df["aspect_density"] = df["clean_review_text"].astype(str).apply(count_aspect_density)

    # 7. Fallback Topic ID assignment (BERTopic mock/fallback)
    if "is_topic_assigned" not in df.columns:
        logger.info("is_topic_assigned not found. Assigning default binary cluster feature...")
        df["is_topic_assigned"] = (
            df["clean_review_text"].str.contains("|".join(ASPECT_KEYWORDS), case=False).astype(int)
        )

    return df


def train_helpfulness_model(
    data_path: str = "data/processed/clean_reviews.csv",
    output_dir: str = "models/helpfulness/weights",
    min_exposure_votes: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """
    Executes feature extraction and model training
    """
    logger.info(f"Loading dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cleaned reviews not found at {data_path}. Please run preprocess.py first."
        )

    df = pd.read_csv(data_path)

    # 1. Filter dataset for reviews with exposure
    # Helps reduce label noise by training only on reviews with enough votes (HelpfulnessDenominator >= threshold)
    logger.info(f"Filtering reviews with at least {min_exposure_votes} exposure votes...")
    df_filtered = df[df["HelpfulnessDenominator"] >= min_exposure_votes].copy()

    if len(df_filtered) < 100:
        logger.warning(
            f"Only {len(df_filtered)} reviews found with exposure >= {min_exposure_votes}."
        )
        logger.warning("Lowering exposure threshold to 1 vote to prevent undersampling.")
        df_filtered = df[df["HelpfulnessDenominator"] >= 1].copy()

    if len(df_filtered) == 0:
        logger.error("No reviews with helpfulness votes found! Cannot train model.")
        return

    logger.info(f"Dataset exposure slice size: {len(df_filtered)} reviews.")

    # Calculate target variable safely
    df_filtered["target_score"] = (
        df_filtered["HelpfulnessNumerator"] / df_filtered["HelpfulnessDenominator"]
    )

    # Extract features
    df_features = extract_features(df_filtered)

    # Define features to use
    feature_cols = [
        "word_count",
        "sentence_count",
        "char_count",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "rating_normalized",
        "time_delta_days",
        "sentiment_polarity",
        "sentiment_confidence",
        "entity_density",
        "aspect_density",
        "is_topic_assigned",
    ]

    X = df_features[feature_cols]
    y = df_features["target_score"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    logger.info(f"Features dimension: Train={X_train.shape}, Validation={X_val.shape}")

    # Scale variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Select regressor
    if XGB_AVAILABLE:
        logger.info("Training XGBoost Regressor model...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        logger.warning(
            "XGBoost package not available. Falling back to Scikit-Learn GradientBoostingRegressor..."
        )
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            random_state=random_seed,
        )

    # Fit model
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val_scaled)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    logger.info(f"Validation Metrics: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Save the pipeline components
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, "helpfulness_model.pkl")
    scaler_save_path = os.path.join(output_dir, "helpfulness_scaler.pkl")

    logger.info(f"Saving scaler to {scaler_save_path}...")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Saving model to {model_save_path}...")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    # Print feature importances
    if XGB_AVAILABLE:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    logger.info("Feature Importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        logger.info(f"  {col}: {imp:.4f}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Helpfulness Predictor")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/clean_reviews.csv",
        help="Path to clean CSV reviews file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/helpfulness/weights",
        help="Directory to save model weights",
    )
    parser.add_argument(
        "--min_votes",
        type=int,
        default=5,
        help="Minimum exposure votes required for target calculations",
    )

    args = parser.parse_args()

    train_helpfulness_model(
        data_path=args.data_path, output_dir=args.output_dir, min_exposure_votes=args.min_votes
    )
