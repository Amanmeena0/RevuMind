"""
Sentence Embeddings Extractor
=============================
Implements STEP 5 of the RevuMind V2 workflow.
Converts review texts into dense floating-point vector representations.
Default model is `all-MiniLM-L6-v2` (384-dimensional embeddings).
"""

import logging
import os
from typing import List, Union

import numpy as np

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to load SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer


class ReviewEmbeddingsExtractor:
    """
    Sentence embeddings generator wrapping SentenceTransformers,
    falling back to a 384-feature TF-IDF vectorizer to maintain exact pgvector schema alignments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None

        self._initialize_extractor()

    def _initialize_extractor(self):
        """
        Loads the SentenceTransformer weights or initializes fallback TF-IDF vectorizer.
        """
        if self.use_transformer:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded model. Embedding dimension: 384")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}. Falling back to TF-IDF.")
                self.use_transformer = False
                # Use exactly 384 features to match pgvector(384) schema constraints
                self.model = TfidfVectorizer(max_features=384, stop_words="english")
        else:
            logger.info("Initializing baseline 384-feature TF-IDF vectorizer...")
            self.model = TfidfVectorizer(max_features=384, stop_words="english")

    def extract_embeddings(
        self, texts: Union[str, List[str]], is_training: bool = False
    ) -> np.ndarray:
        """
        Generates embeddings for review texts.
        Returns a NumPy array of shape (num_texts, 384).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, 384))

        if self.use_transformer:
            # Generate dense embeddings
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            # Run TF-IDF fallback
            if is_training:
                logger.info("Fitting TF-IDF vectorizer on training text...")
                vectors = self.model.fit_transform(texts).toarray()
            else:
                try:
                    vectors = self.model.transform(texts).toarray()
                except Exception:
                    # In case vectorizer has not been fit, perform a fit-transform as safety
                    logger.warning(
                        "TF-IDF vectorizer was not fitted. Running fit_transform as fallback..."
                    )
                    vectors = self.model.fit_transform(texts).toarray()

            # Handle cases where vocabulary is smaller than 384 features
            if vectors.shape[1] < 384:
                # Pad with zeros to guarantee exactly 384 elements for pgvector
                padding = np.zeros((vectors.shape[0], 384 - vectors.shape[1]))
                vectors = np.hstack([vectors, padding])

            return vectors


if __name__ == "__main__":
    # Test script run
    logger.info("Running ReviewEmbeddingsExtractor self-test...")

    # Run with fallback TF-IDF mode first for fast test
    extractor = ReviewEmbeddingsExtractor(use_transformer=False)

    test_texts = [
        "The screen display is incredibly clear with gorgeous colors.",
        "However, the battery life is absolutely terrible, barely lasting 4 hours.",
    ]

    embeddings = extractor.extract_embeddings(test_texts, is_training=True)
    print(f"\nGenerated embeddings shape: {embeddings.shape} (Expected: (2, 384))")
    print(f"Vector sample (first 5 elements of review 1): {embeddings[0][:5]}")
