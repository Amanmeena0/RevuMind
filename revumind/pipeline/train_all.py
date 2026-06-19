"""
Unified Model Training Orchestrator
==================================
Runs the complete training cascade:
1. Preprocess raw data from `archive/Reviews.csv` -> `data/processed/clean_reviews.csv`
2. Fit the Topic Modeler (BERTopic/KMeans fallback) and save weights
3. Train the Helpfulness Predictor (XGBoost/GradientBoosting fallback) and save weights
4. Output training reports
"""

import os
import argparse
import logging

# Import pipeline components
from revumind.pipeline.preprocess import preprocess_dataset
from revumind.models.topics.model import ReviewTopicModeler
from revumind.models.helpfulness.train import train_helpfulness_model

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_full_training_pipeline(
    raw_data_path: str = "archive/Reviews.csv",
    processed_data_path: str = "data/processed/clean_reviews.csv",
    sample_size: int = None,
    num_topics: int = 10,
    min_exposure_votes: int = 5,
    helpfulness_dir: str = "models/helpfulness/weights",
    topics_save_path: str = "models/topics/weights/topic_model.pkl"
):
    """
    Orchestrates the preprocessing and model fitting workflows
    """
    logger.info("=========================================")
    logger.info("   STARTING REVUMIND V2 TRAINING PIPELINE ")
    logger.info("=========================================")
    
    # Step 1: Preprocess dataset
    logger.info("\n--- STEP 1: Preprocessing Data ---")
    df_clean = preprocess_dataset(
        input_path=raw_data_path,
        output_path=processed_data_path,
        sample_size=sample_size
    )
    
    # Step 2: Fit Topic Modeling Pipeline
    logger.info("\n--- STEP 2: Fitting Topic Modeler ---")
    texts = df_clean["clean_review_text"].astype(str).tolist()
    
    modeler = ReviewTopicModeler(num_topics=num_topics)
    modeler.fit(texts)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(topics_save_path), exist_ok=True)
    modeler.save(topics_save_path)
    logger.info(f"Topic modeler saved to {topics_save_path}")
    
    # Step 3: Train Helpfulness Predictor (XGBoost / Gradient Boosting)
    logger.info("\n--- STEP 3: Training Helpfulness Model ---")
    # For helpfulness training, we pass the path of the clean reviews CSV
    train_helpfulness_model(
        data_path=processed_data_path,
        output_dir=helpfulness_dir,
        min_exposure_votes=min_exposure_votes
    )
    
    logger.info("\n=========================================")
    logger.info("   TRAINING PIPELINE FINISHED SUCCESSFULLY")
    logger.info("=========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuMind V2 Model Training Pipeline Orchestrator")
    parser.add_argument("--raw_data", type=str, default="archive/Reviews.csv", help="Path to raw CSV reviews")
    parser.add_argument("--processed_data", type=str, default="data/processed/clean_reviews.csv", help="Path to save processed data")
    parser.add_argument("--sample_size", type=int, default=10000, help="Subset size for quick training runs. Use 0 for full run.")
    parser.add_argument("--num_topics", type=int, default=10, help="Number of topics to extract")
    parser.add_argument("--min_votes", type=int, default=1, help="Minimum helpfulness votes required for regressor training")
    
    args = parser.parse_args()
    
    sample = None if args.sample_size == 0 else args.sample_size
    
    run_full_training_pipeline(
        raw_data_path=args.raw_data,
        processed_data_path=args.processed_data,
        sample_size=sample,
        num_topics=args.num_topics,
        min_exposure_votes=args.min_votes
    )
