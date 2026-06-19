"""
Review Inference Orchestrator & Business Insights Engine
========================================================
Implements STEP 10 of the RevuMind V2 workflow.
Loads all trained pipeline components and runs the full inference cascade
on single reviews or batch dataframes, producing structured insights.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

# Import pipeline components
from revumind.pipeline.preprocess import clean_review_text
from revumind.utils.readability import calculate_readability

# Import model wrappers
from revumind.models.embeddings.model import ReviewEmbeddingsExtractor
from revumind.models.ner.model import ProductNERPipeline
from revumind.models.absa.model import AspectSentimentAnalyzer
from revumind.models.topics.model import ReviewTopicModeler
from revumind.models.summarizer.model import ExecutiveSummarizer

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RevuMindInferenceEngine:
    """
    Unified Inference Engine coordinating the 7 NLP/ML models to generate business insights.
    """
    def __init__(
        self,
        helpfulness_dir: str = "models/helpfulness/weights",
        topics_model_path: str = "models/topics/weights/topic_model.pkl",
        use_deberta: bool = False,   # Set to True in GPU environments
        use_bart: bool = False       # Set to True in GPU environments
    ):
        self.helpfulness_dir = helpfulness_dir
        self.topics_model_path = topics_model_path
        
        # Initialize modules
        logger.info("Initializing inference wrappers...")
        self.embeddings_extractor = ReviewEmbeddingsExtractor(use_transformer=False)
        self.ner_pipeline = ProductNERPipeline(use_transformer=False)
        self.absa_analyzer = AspectSentimentAnalyzer(use_deberta=use_deberta)
        self.summarizer = ExecutiveSummarizer(use_bart=use_bart)
        
        # Load fitted Topic model
        self.topic_modeler = ReviewTopicModeler()
        if os.path.exists(topics_model_path):
            self.topic_modeler.load(topics_model_path)
        else:
            logger.warning(f"Fitted Topic Model not found at {topics_model_path}. Topic modeling will run in zero-shot dummy mode.")
            
        # Load fitted Helpfulness model & scaler
        self.helpfulness_model = None
        self.helpfulness_scaler = None
        self._load_helpfulness_model()

    def _load_helpfulness_model(self):
        """
        Loads the pickled scaler and model for helpfulness prediction.
        """
        model_path = os.path.join(self.helpfulness_dir, "helpfulness_model.pkl")
        scaler_path = os.path.join(self.helpfulness_dir, "helpfulness_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info(f"Loading helpfulness model from {model_path}...")
            with open(model_path, "rb") as f:
                self.helpfulness_model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self.helpfulness_scaler = pickle.load(f)
        else:
            logger.warning("Fitted Helpfulness model not found. Predictions will return a fallback placeholder.")

    def analyze_single_review(
        self,
        raw_text: str,
        rating: int,
        review_time_unix: int = None
    ) -> Dict[str, Any]:
        """
        Runs the full 7-model cascade on a single raw review.
        """
        # 1. Preprocess raw text
        clean_text = clean_review_text(raw_text)
        if not clean_text:
            return {"error": "Text cleaning returned an empty review."}
            
        # 2. Extract Embedding vector
        embedding = self.embeddings_extractor.extract_embeddings(clean_text)[0]
        
        # 3. Named Entity Recognition
        entities = self.ner_pipeline.extract_entities(clean_text)
        
        # Filter features/aspect candidates from NER outputs
        aspect_candidates = [ent["text"] for ent in entities if ent["label"] in ["FEATURE", "PRODUCT"]]
        
        # 4. Aspect-Based Sentiment Analysis
        aspect_sentiments = self.absa_analyzer.analyze_review_aspects(clean_text, aspects=aspect_candidates)
        
        # 5. Topic Model Assignment
        if self.topic_modeler.is_fitted:
            _, topic_details = self.topic_modeler.predict([clean_text])
            topic_id = topic_details[0]["topic_id"]
            topic_name = topic_details[0]["topic_name"]
            topic_keywords = topic_details[0]["topic_keywords"]
            is_topic_assigned = 1 if topic_id >= 0 else 0
        else:
            topic_id, topic_name, topic_keywords, is_topic_assigned = -1, "General", [], 0
            
        # 6. Readability & Text stats
        readability = calculate_readability(clean_text)
        
        # 7. Overall Sentiment (RoBERTa mock/fallback compounds)
        overall_pol = self.absa_analyzer.fallback_sia.polarity_scores(clean_text)
        sentiment_polarity = overall_pol["compound"]
        sentiment_confidence = max(overall_pol["pos"], overall_pol["neg"], overall_pol["neu"])
        
        if sentiment_polarity >= 0.05:
            overall_sentiment = "positive"
        elif sentiment_polarity <= -0.05:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
            
        # 8. Helpfulness Prediction via XGBoost
        predicted_helpfulness = 0.50 # default
        if self.helpfulness_model and self.helpfulness_scaler:
            # Build feature array
            rating_normalized = rating / 5.0
            time_delta_days = 0.0 # single prediction baseline
            
            word_count = readability["word_count"]
            entity_density = len(entities) / max(1, word_count)
            aspect_density = len(aspect_sentiments) / max(1, word_count)
            
            features = np.array([[
                word_count,
                readability["sentence_count"],
                readability["char_count"],
                readability["flesch_reading_ease"],
                readability["flesch_kincaid_grade"],
                rating_normalized,
                time_delta_days,
                sentiment_polarity,
                sentiment_confidence,
                entity_density,
                aspect_density,
                is_topic_assigned
            ]])
            
            features_scaled = self.helpfulness_scaler.transform(features)
            predicted_helpfulness = float(self.helpfulness_model.predict(features_scaled)[0])
            predicted_helpfulness = max(0.0, min(1.0, predicted_helpfulness)) # Clamp
            
        return {
            "clean_review_text": clean_text,
            "overall_sentiment": overall_sentiment,
            "sentiment_confidence": float(round(sentiment_confidence, 2)),
            "entities": entities,
            "aspect_sentiments": aspect_sentiments,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_keywords": topic_keywords,
            "predicted_helpfulness": float(round(predicted_helpfulness, 2)),
            "embedding": embedding.tolist()
        }

    def analyze_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs batch predictions on a DataFrame.
        Aggregates results to produce global business insights (Praises, Complaints, Summaries).
        """
        logger.info(f"Starting batch inference on {len(df)} records...")
        
        processed_records = []
        for idx, row in df.iterrows():
            # Support either 'clean_review_text' if preprocessed, or default text columns
            text = row.get("clean_review_text", row.get("Text", ""))
            rating = row.get("Score", 3)
            time_unix = row.get("Time", None)
            
            analysis = self.analyze_single_review(text, rating, time_unix)
            
            # Merge original columns with analysis outputs
            record = {
                "ProductId": row.get("ProductId", "Unknown"),
                "UserId": row.get("UserId", "Unknown"),
                "Score": rating,
                "clean_review_text": text,
                "overall_sentiment": analysis.get("overall_sentiment", "neutral"),
                "predicted_helpfulness": analysis.get("predicted_helpfulness", 0.5),
                "topic_id": analysis.get("topic_id", -1),
                "topic_name": analysis.get("topic_name", "General"),
                "entities_count": len(analysis.get("entities", [])),
                "aspects_count": len(analysis.get("aspect_sentiments", []))
            }
            
            # Flatten aspect sentiments for tabular analytics
            for asp in analysis.get("aspect_sentiments", []):
                record[f"aspect_{asp['aspect'].replace(' ', '_')}"] = asp["sentiment_label"]
                
            processed_records.append(record)
            
        df_results = pd.DataFrame(processed_records)
        
        # Generate Business Insights (Step 10 Aggregations)
        logger.info("Aggregating Business Insights...")
        
        # 1. Sentiment Distribution
        sentiment_counts = df_results["overall_sentiment"].value_counts().to_dict()
        
        # 2. Extract Top Praises and Complaints from Aspect Sentiments
        aspect_cols = [c for c in df_results.columns if c.startswith("aspect_")]
        praises = {}
        complaints = {}
        
        for col in aspect_cols:
            aspect_name = col.replace("aspect_", "").capitalize()
            val_counts = df_results[col].value_counts().to_dict()
            
            pos_count = val_counts.get("positive", 0)
            neg_count = val_counts.get("negative", 0)
            
            if pos_count > 0:
                praises[aspect_name] = pos_count
            if neg_count > 0:
                complaints[aspect_name] = neg_count
                
        sorted_praises = sorted(praises.items(), key=lambda x: x[1], reverse=True)[:5]
        sorted_complaints = sorted(complaints.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 3. Cohort summaries using BART / Extractive Summarizer
        logger.info("Generating Cohort Executive Summaries...")
        positive_reviews = df_results[df_results["overall_sentiment"] == "positive"]["clean_review_text"].tolist()[:10]
        negative_reviews = df_results[df_results["overall_sentiment"] == "negative"]["clean_review_text"].tolist()[:10]
        
        insights = {
            "total_analyzed": len(df_results),
            "sentiment_distribution": sentiment_counts,
            "top_praises": [{"aspect": p[0], "count": p[1]} for p in sorted_praises],
            "top_complaints": [{"aspect": c[0], "count": c[1]} for c in sorted_complaints],
            "summaries": {
                "positive_cohort_summary": self.summarizer.generate_summary(positive_reviews) if positive_reviews else "No positive reviews to summarize.",
                "negative_cohort_summary": self.summarizer.generate_summary(negative_reviews) if negative_reviews else "No negative reviews to summarize."
            },
            "business_recommendations": []
        }
        
        # 4. Synthesize Business Recommendations
        for complaint_aspect, count in sorted_complaints[:2]:
            insights["business_recommendations"].append(
                f"Improve {complaint_aspect.lower()} quality. It is a recurring pain point mentioned in {count} negative customer reviews."
            )
        for praise_aspect, count in sorted_praises[:2]:
            insights["business_recommendations"].append(
                f"Highlight {praise_aspect.lower()} performance in marketing campaigns, as it is praised in {count} positive reviews."
            )
            
        if not insights["business_recommendations"]:
            insights["business_recommendations"].append("Monitor product reception to gather performance points.")
            
        return df_results, insights

if __name__ == "__main__":
    logger.info("Running RevuMindInferenceEngine self-test...")
    
    engine = RevuMindInferenceEngine()
    
    # 1. Test single review analysis
    test_text = "I love this headphone! The sound quality is beautiful and the battery is exceptional. However, it is very expensive."
    result = engine.analyze_single_review(test_text, rating=4)
    
    print("\n--- SINGLE REVIEW ANALYSIS ---")
    print(f"Review: '{test_text}'")
    print(f"Overall Sentiment: {result['overall_sentiment'].upper()} (Confidence: {result['sentiment_confidence']})")
    print(f"Topic Name: {result['topic_name']}")
    print(f"Predicted Helpfulness Score: {result['predicted_helpfulness']:.2f}")
    print("Aspect Sentiment:")
    for asp in result["aspect_sentiments"]:
        print(f"  - {asp['aspect'].capitalize()}: {asp['sentiment_label'].upper()}")
        
    # 2. Test batch analysis
    test_batch = pd.DataFrame([
        {"Text": "The screen is amazing but shipping took a week.", "Score": 4},
        {"Text": "Worst phone charger ever, screen cracked on day one.", "Score": 1},
        {"Text": "Perfect audio and clear sound quality.", "Score": 5},
        {"Text": "Nice battery life, acceptable price.", "Score": 4}
    ])
    
    df_res, insights = engine.analyze_batch(test_batch)
    
    print("\n--- BATCH ANALYSIS INSIGHTS ---")
    print(f"Processed Rows: {insights['total_analyzed']}")
    print(f"Sentiment Distribution: {insights['sentiment_distribution']}")
    print(f"Top Praises: {insights['top_praises']}")
    print(f"Top Complaints: {insights['top_complaints']}")
    print("\nPositive Summary:")
    print(insights["summaries"]["positive_cohort_summary"])
    print("\nNegative Summary:")
    print(insights["summaries"]["negative_cohort_summary"])
    print("\nRecommendations:")
    for i, rec in enumerate(insights["business_recommendations"]):
        print(f"  {i+1}. {rec}")
