"""
Database Ingestion Pipeline
===========================
Coordinates loading raw reviews from `archive/Reviews.csv`, running them through
the RevuMind V2 Inference Engine, and saving the structured predictions,
aspects, entities, and embeddings into the database.
"""

import argparse
import logging
import os

import pandas as pd
from sqlalchemy.orm import Session

# Import DB core & session
from revumind.core.database import SessionLocal, get_db
from revumind.db import models

# Import Inference Engine
from revumind.pipeline.inference import RevuMindInferenceEngine

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_reviews_to_db(
    raw_data_path: str = "archive/Reviews.csv",
    sample_size: int = 50,
    min_exposure_votes: int = 0,
    offset: int = 0,
):
    """
    Reads CSV reviews, analyzes them with the 7-model inference cascade,
    and populates relational database tables.
    """
    logger.info("Initializing Ingestion Database Pipeline...")

    # Initialize DB Session
    db: Session = SessionLocal()

    # Initialize Inference Engine
    engine = RevuMindInferenceEngine()

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw CSV not found at {raw_data_path}.")

    logger.info(
        f"Loading raw reviews from {raw_data_path} (offset={offset}, limit={sample_size})..."
    )
    # Load target columns
    target_cols = [
        "ProductId",
        "UserId",
        "ProfileName",
        "Score",
        "Summary",
        "Text",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "Time",
    ]
    if offset > 0:
        df = pd.read_csv(
            raw_data_path,
            usecols=lambda col: col in target_cols,
            skiprows=range(1, offset + 1),
            nrows=sample_size * 2,
        )
    else:
        df = pd.read_csv(
            raw_data_path,
            usecols=lambda col: col in target_cols,
            nrows=sample_size * 2,
        )

    # Filter nulls in essential columns
    df = df.dropna(subset=["ProductId", "Score", "Text"])

    # Filter by exposure if requested
    if min_exposure_votes > 0:
        df = df[df["HelpfulnessDenominator"] >= min_exposure_votes]

    # Trim to target sample size
    df = df.head(sample_size)
    logger.info(f"Processing slice size of {len(df)} reviews for DB ingestion...")

    count_reviews = 0
    count_entities = 0
    count_aspects = 0

    try:
        for idx, row in df.iterrows():
            product_id = str(row["ProductId"])

            # 1. Check or create Product record
            product = db.query(models.Product).filter(models.Product.id == product_id).first()
            if not product:
                product = models.Product(id=product_id, brand="Unknown", category="General")
                db.add(product)
                db.flush()  # Flush to lock ID mapping

            # 2. Run Inference Cascade
            raw_text = str(row["Text"])
            raw_summary = str(row["Summary"]) if pd.notna(row["Summary"]) else ""
            rating = int(row["Score"])
            time_unix = int(row["Time"])

            # Combine raw Summary + Text for full model inference
            full_raw_text = f"{raw_summary}. {raw_text}" if raw_summary else raw_text

            analysis = engine.analyze_single_review(full_raw_text, rating, time_unix)

            if "error" in analysis:
                logger.warning(f"Failed to process review {idx}: {analysis['error']}")
                continue

            # Convert timestamp
            review_time = pd.to_datetime(time_unix, unit="s")

            # 3. Create Review Record
            review = models.Review(
                product_id=product_id,
                user_id=str(row["UserId"]),
                profile_name=str(row["ProfileName"]) if pd.notna(row["ProfileName"]) else None,
                score=rating,
                helpfulness_numerator=int(row["HelpfulnessNumerator"]),
                helpfulness_denominator=int(row["HelpfulnessDenominator"]),
                predicted_helpfulness=analysis["predicted_helpfulness"],
                review_time=review_time,
                summary=raw_summary,
                review_text=raw_text,
                clean_review_text=analysis["clean_review_text"],
                topic_id=analysis["topic_id"],
            )
            db.add(review)
            db.flush()  # Flush to obtain auto-incremented review.id

            # 4. Create ReviewEmbedding Record
            embedding = models.ReviewEmbedding(
                review_id=review.id,
                embedding=analysis[
                    "embedding"
                ],  # stored as list (handles pgvector list or pickle serialize)
            )
            db.add(embedding)

            # 5. Create Entity Records
            for ent in analysis.get("entities", []):
                entity = models.Entity(
                    review_id=review.id,
                    entity_text=ent["text"],
                    entity_type=ent["label"],
                    confidence=ent["confidence"],
                    start_char=ent["start_char"],
                    end_char=ent["end_char"],
                )
                db.add(entity)
                count_entities += 1

            # 6. Create AspectSentiment Records
            for asp in analysis.get("aspect_sentiments", []):
                aspect = models.AspectSentiment(
                    review_id=review.id,
                    aspect_term=asp["aspect"],
                    sentiment_label=asp["sentiment_label"],
                    confidence=asp["confidence"],
                )
                db.add(aspect)
                count_aspects += 1

            count_reviews += 1
            if count_reviews % 10 == 0:
                logger.info(f"Ingested {count_reviews}/{len(df)} reviews to database...")

        # Save transactions
        db.commit()
        logger.info("\n=========================================")
        logger.info("       INGESTION PIPELINE COMPLETE        ")
        logger.info("=========================================")
        logger.info(
            f"Successfully Ingested: Reviews={count_reviews}, Entities={count_entities}, Aspect Sentiments={count_aspects}"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Inference Ingestion failed. Transaction rolled back: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest reviews and predictions to DB")
    parser.add_argument(
        "--raw_data", type=str, default="archive/Reviews.csv", help="Path to raw CSV reviews"
    )
    parser.add_argument("--sample_size", type=int, default=50, help="Number of reviews to process")
    parser.add_argument(
        "--min_votes", type=int, default=0, help="Minimum helpfulness votes filter"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Number of rows to skip before starting ingestion"
    )

    args = parser.parse_args()

    ingest_reviews_to_db(
        raw_data_path=args.raw_data,
        sample_size=args.sample_size,
        min_exposure_votes=args.min_votes,
        offset=args.offset,
    )
