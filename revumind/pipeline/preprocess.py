"""
RevuMind V2 Preprocessing Pipeline
==================================
Implements STEP 2 of the workflow.
Loads raw reviews from `./archive/Reviews.csv`, cleans and processes them,
and saves the output to `./data/processed/clean_reviews.csv`.
"""

import logging
import os
import re

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Heuristic English check: requires a minimum ratio of ASCII printable letters to total characters
def is_likely_english(text: str, ratio_threshold: float = 0.8) -> bool:
    """
    Fast heuristic to check if the review is mostly English.
    Avoids slow external API calls or large libraries for 500k reviews.
    """
    if not text:
        return False
    # Count alphabet characters and general English characters
    alpha_chars = len(re.findall(r'[a-zA-Z\s.,!?\'"-]', text))
    total_chars = len(text)
    if total_chars == 0:
        return False
    return (alpha_chars / total_chars) >= ratio_threshold


def remove_emojis(text: str) -> str:
    """
    Remove emoji and special Unicode symbols using Unicode ranges.
    """
    if not isinstance(text, str):
        return ""
    # Remove characters outside standard text/punctuation range (keep standard ASCII and basic Latin-1)
    # This effectively strips out most emojis and custom unicode graphics.
    return re.sub(r"[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F]", "", text)


def clean_review_text(text: str) -> str:
    """
    Applies the full cleaning suite to a single review string:
    - HTML removal
    - URL removal
    - Emoji removal
    - Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove Emojis and other non-standard Unicode symbols
    text = remove_emojis(text)

    # 4. Normalize spacing and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(
    input_path: str = "archive/Reviews.csv",
    output_path: str = "data/processed/clean_reviews.csv",
    sample_size: int = None,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """
    Executes the full preprocessing pipeline:
    - Loads dataset in chunks (for memory safety with 300MB+ CSV files)
    - Removes null reviews
    - Removes duplicates
    - Cleans and normalizes text
    - Performs language filtering
    - Merges Summary + Text
    - Saves clean reviews to output path
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Starting ingestion from: {input_path}")

    # Target columns for Step 1
    target_cols = [
        "ProductId",
        "UserId",
        "Score",
        "Summary",
        "Text",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "Time",
    ]

    processed_chunks = []
    total_raw_rows = 0

    # Read CSV in chunks
    for i, chunk in enumerate(
        pd.read_csv(input_path, usecols=lambda col: col in target_cols, chunksize=chunk_size)
    ):
        chunk_rows = len(chunk)
        total_raw_rows += chunk_rows

        # 1. Drop rows with null essentials
        chunk = chunk.dropna(subset=["ProductId", "Score", "Summary", "Text"])

        # 2. Text cleaning: Apply to Summary and Text separately first
        chunk["clean_summary"] = chunk["Summary"].astype(str).apply(clean_review_text)
        chunk["clean_text"] = chunk["Text"].astype(str).apply(clean_review_text)

        # 3. Merge Summary + Text to form clean_review_text
        # We append Summary with a period separator if it doesn't already have end punctuation
        def merge_fields(row):
            summary = row["clean_summary"]
            text = row["clean_text"]
            if not summary:
                return text
            if summary[-1] in [".", "!", "?"]:
                return f"{summary} {text}"
            return f"{summary}. {text}"

        chunk["clean_review_text"] = chunk.apply(merge_fields, axis=1)

        # 4. Filter empty entries resulting from cleaning
        chunk = chunk[chunk["clean_review_text"].str.strip() != ""]

        # 5. Language filtering (keep likely English reviews)
        chunk = chunk[chunk["clean_review_text"].apply(is_likely_english)]

        processed_chunks.append(chunk)
        logger.info(f"Processed chunk {i+1} (Rows read: {total_raw_rows})")

        if sample_size and sum(len(c) for c in processed_chunks) >= sample_size:
            logger.info(f"Reached specified sample limit: {sample_size}")
            break

    # Combine chunks
    df = pd.concat(processed_chunks, ignore_index=True)

    # 6. Remove Duplicates (based on ProductId, UserId, and the cleaned review text)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["ProductId", "UserId", "clean_review_text"])
    logger.info(f"Removed {initial_len - len(df)} duplicate reviews.")

    # Trim to sample size if specified
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size]

    # Save the processed dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Processed dataset saved to: {output_path} ({len(df)} reviews)")
    logger.info(f"Ingestion statistics: Raw Rows={total_raw_rows} -> Cleaned Rows={len(df)}")

    return df


if __name__ == "__main__":
    # For testing or fast prototyping, process a subset of 10,000 reviews
    # To run on the full 300MB dataset, set sample_size=None
    preprocess_dataset(sample_size=10000)
