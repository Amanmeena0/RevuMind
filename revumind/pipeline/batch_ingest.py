"""
Batch Ingestion Orchestrator
============================
Sequentially runs database ingestion in chunks of 50,000 reviews
to prevent database locks and memory overload.
"""

import logging
import subprocess

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import os

RAW_DATA_PATH = "data/raw/Reviews.csv"
CHUNK_SIZE = 50000
TOTAL_REVIEWS = 568454
CHECKPOINT_PATH = "data/processed/ingestion_checkpoint.txt"


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
    return 0


def save_checkpoint(offset: int):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(offset))


def run():
    start_offset = load_checkpoint()
    if start_offset > 0:
        logger.info(f"Resuming ingestion from checkpoint offset: {start_offset:,}")

    logger.info(
        f"Starting batch ingestion for reviews in chunks of {CHUNK_SIZE:,} (Total reviews: {TOTAL_REVIEWS:,})..."
    )
    for offset in range(start_offset, TOTAL_REVIEWS, CHUNK_SIZE):
        logger.info(f"\n=========================================")
        logger.info(f" PROCESSING CHUNK: Offset {offset:,} to {offset + CHUNK_SIZE:,}")
        logger.info(f"=========================================")

        cmd = [
            ".venv/bin/python",
            "revumind/pipeline/ingest_to_db.py",
            "--raw_data",
            RAW_DATA_PATH,
            "--sample_size",
            str(CHUNK_SIZE),
            "--offset",
            str(offset),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"Batch failed at offset {offset}. Stopping ingestion.")
            break

        # Save checkpoint for next run
        next_offset = offset + CHUNK_SIZE
        save_checkpoint(next_offset)
        logger.info(f"Checkpoint saved: offset {next_offset:,} is next.")

    if load_checkpoint() >= TOTAL_REVIEWS:
        logger.info("Batch ingestion process completed successfully.")
        # Reset checkpoint on clean completion
        save_checkpoint(0)


if __name__ == "__main__":
    run()
