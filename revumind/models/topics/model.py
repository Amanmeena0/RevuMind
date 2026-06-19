"""
Topic Modeling Pipeline
=======================
Implements STEP 7 of the RevuMind V2 workflow.
Maps review embeddings to custom topics using a modular setup:
1. Sentence Embeddings (all-MiniLM-L6-v2)
2. Dimensionality Reduction (UMAP / PCA fallback)
3. Density Clustering (HDBSCAN / KMeans fallback)
4. Class-Based TF-IDF keyword extraction (c-TF-IDF)
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple

# Scikit-learn fallbacks
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to load SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to load UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to load HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class ReviewTopicModeler:
    """
    Topic Modeler supporting BERTopic style pipeline (MiniLM + UMAP + HDBSCAN + c-TF-IDF)
    with a robust fallback to (MiniLM/TF-IDF + PCA + KMeans + c-TF-IDF).
    """
    def __init__(
        self,
        num_topics: int = 10,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42
    ):
        self.num_topics = num_topics
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state
        
        self.encoder = None
        self.dim_reducer = None
        self.clusterer = None
        self.c_tfidf_vectorizer = None
        self.topic_keywords = {}
        self.is_fitted = False
        self.use_fallback = False

    def _initialize_models(self):
        """
        Loads encoder and initializes pipeline depending on library availability.
        """
        # 1. Embedding Model Setup
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading SentenceTransformer: {self.embedding_model_name}")
            self.encoder = SentenceTransformer(self.embedding_model_name)
        else:
            logger.warning("sentence-transformers package not found. Will use TF-IDF features as baseline embeddings.")
            self.encoder = TfidfVectorizer(max_features=512, stop_words="english")
            
        # 2. Dim Reducer & Clusterer Setup
        if UMAP_AVAILABLE and HDBSCAN_AVAILABLE:
            logger.info("Initializing UMAP and HDBSCAN pipeline...")
            self.dim_reducer = umap.UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.1,
                metric='cosine',
                random_state=self.random_state
            )
            # HDBSCAN clusters density. min_cluster_size determines granularity
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=15,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            self.use_fallback = False
        else:
            logger.warning("UMAP/HDBSCAN packages not available. Falling back to Scikit-Learn PCA + MiniBatchKMeans...")
            self.dim_reducer = PCA(n_components=5, random_state=self.random_state)
            self.clusterer = MiniBatchKMeans(n_clusters=self.num_topics, random_state=self.random_state)
            self.use_fallback = True

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense embeddings from texts.
        """
        if self.encoder is None:
            self._initialize_models()
            
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            return self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            # TF-IDF fit-transform (returns sparse matrix, convert to dense array)
            if not self.is_fitted:
                return self.encoder.fit_transform(texts).toarray()
            else:
                return self.encoder.transform(texts).toarray()

    def _calculate_c_tfidf(self, texts: List[str], labels: np.ndarray) -> Dict[int, List[Tuple[str, float]]]:
        """
        Calculates Class-Based TF-IDF (c-TF-IDF) keywords for each topic.
        Aggregates all texts belonging to a specific topic and runs TF-IDF vectorization.
        """
        logger.info("Extracting c-TF-IDF topic keywords...")
        
        # Create a document for each cluster label
        df_docs = pd.DataFrame({"text": texts, "topic": labels})
        grouped_docs = df_docs.groupby("topic")["text"].apply(lambda x: " ".join(x)).reset_index()
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        c_tfidf_matrix = vectorizer.fit_transform(grouped_docs["text"]).toarray()
        feature_names = vectorizer.get_feature_names_out()
        
        topic_keywords = {}
        for idx, row in grouped_docs.iterrows():
            topic_id = int(row["topic"])
            scores = c_tfidf_matrix[idx]
            
            # Sort keywords by TF-IDF scores
            sorted_indices = np.argsort(scores)[::-1][:8]
            keywords_with_scores = [(feature_names[i], float(scores[i])) for i in sorted_indices if scores[i] > 0]
            topic_keywords[topic_id] = keywords_with_scores
            
        return topic_keywords

    def fit(self, texts: List[str]):
        """
        Fits the topic modeling pipeline on a corpus of text documents.
        """
        logger.info("Initializing models...")
        self._initialize_models()
        
        logger.info("Extracting embeddings...")
        embeddings = self.get_embeddings(texts)
        
        logger.info("Reducing dimensionality...")
        embeddings_reduced = self.dim_reducer.fit_transform(embeddings)
        
        logger.info("Clustering reduced embeddings...")
        if self.use_fallback:
            self.clusterer.fit(embeddings_reduced)
            labels = self.clusterer.labels_
        else:
            self.clusterer.fit(embeddings_reduced)
            labels = self.clusterer.labels_
            
        # Extract c-TF-IDF keywords for clusters
        self.topic_keywords = self._calculate_c_tfidf(texts, labels)
        self.is_fitted = True
        logger.info("Topic modeling pipeline successfully fitted!")

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predicts topics and maps keywords for a list of reviews.
        """
        if not self.is_fitted:
            raise ValueError("Topic modeler is not fitted yet. Run fit() first.")
            
        embeddings = self.get_embeddings(texts)
        embeddings_reduced = self.dim_reducer.transform(embeddings)
        
        if self.use_fallback:
            labels = self.clusterer.predict(embeddings_reduced)
        else:
            # HDBSCAN prediction requires approximate mapping
            import hdbscan
            labels, strengths = hdbscan.approximate_predict(self.clusterer, embeddings_reduced)
            
        # Map labels to keyword details
        results = []
        for label in labels:
            label_int = int(label)
            keywords = self.topic_keywords.get(label_int, [])
            results.append({
                "topic_id": label_int,
                "topic_keywords": [k[0] for k in keywords],
                "topic_name": f"Topic {label_int}: " + ", ".join([k[0] for k in keywords[:3]]) if keywords else f"Topic {label_int}"
            })
            
        return labels, results

    def save(self, output_path: str):
        """
        Serializes pipeline components to disk
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving topic modeling pipeline to {output_path}...")
        
        # Save components using pickle
        state = {
            "num_topics": self.num_topics,
            "embedding_model_name": self.embedding_model_name,
            "random_state": self.random_state,
            "dim_reducer": self.dim_reducer,
            "clusterer": self.clusterer,
            "topic_keywords": self.topic_keywords,
            "is_fitted": self.is_fitted,
            "use_fallback": self.use_fallback,
            "encoder": self.encoder  # Save fitted encoder (TfidfVectorizer / SentenceTransformer)
        }
        with open(output_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, input_path: str):
        """
        Loads components from disk
        """
        logger.info(f"Loading topic modeling pipeline from {input_path}...")
        with open(input_path, "rb") as f:
            state = pickle.load(f)
            
        self.num_topics = state["num_topics"]
        self.embedding_model_name = state["embedding_model_name"]
        self.random_state = state["random_state"]
        self.dim_reducer = state["dim_reducer"]
        self.clusterer = state["clusterer"]
        self.topic_keywords = state["topic_keywords"]
        self.is_fitted = state["is_fitted"]
        self.use_fallback = state["use_fallback"]
        self.encoder = state.get("encoder", None)
        
        # Fallback if encoder is not in state (e.g. legacy model)
        if self.encoder is None:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.encoder = SentenceTransformer(self.embedding_model_name)
            else:
                self.encoder = TfidfVectorizer(max_features=512, stop_words="english")

if __name__ == "__main__":
    # Test script run
    logger.info("Running Topic Modeler self-test...")
    test_docs = [
        "The battery life is amazing and lasts two days.",
        "Charger is very slow, took three hours to charge.",
        "Camera resolution is crystal clear in low light.",
        "Photos taken with the night lens are beautiful.",
        "The screen display has vivid colors and 120Hz.",
        "It has a beautiful display and responsive touch screen.",
        "Price is too high for this cheap plastic frame.",
        "Very expensive product, not worth the budget.",
        "Customer service was terrible and slow to respond.",
        "Support team resolved my issue immediately, great delivery."
    ]
    
    modeler = ReviewTopicModeler(num_topics=4)
    modeler.fit(test_docs)
    
    labels, details = modeler.predict([
        "I love the battery length and fast charging screen.",
        "Customer support was very rude."
    ])
    
    for i, detail in enumerate(details):
        print(f"Review {i+1} assigned to: {detail['topic_name']}")
