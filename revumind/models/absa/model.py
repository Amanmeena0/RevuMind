"""
Aspect-Based Sentiment Analysis (ABSA) Pipeline
===============================================
Implements STEP 6 of the RevuMind V2 workflow.
Determines sentiment polarity for specific product aspects/features 
(e.g., "Battery" -> Positive, "Price" -> Negative).
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Any

import numpy as np

# NLP imports for fallback
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data if needed
for pkg in ["punkt", "punkt_tab", "vader_lexicon"]:
    nltk.download(pkg, quiet=True)

# Try importing Hugging Face transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Default list of features to evaluate if no specific aspects are requested
DEFAULT_ASPECTS = ["battery", "screen", "display", "camera", "price", "sound", "audio", "software", "shipping", "durability"]

class AspectSentimentAnalyzer:
    """
    ABSA module that leverages a fine-tuned DeBERTa ABSA classifier,
    falling back to a sentence-level lexicon scoring pipeline when deep learning is unavailable.
    """
    def __init__(
        self,
        model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
        use_deberta: bool = True
    ):
        self.model_name = model_name
        self.use_deberta = use_deberta and TRANSFORMERS_AVAILABLE
        
        self.tokenizer = None
        self.model = None
        self.device = None
        self.fallback_sia = None
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """
        Loads the DeBERTa model or falls back to NLTK VADER.
        """
        if self.use_deberta:
            try:
                logger.info(f"Loading DeBERTa ABSA model: {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                logger.info(f"Loaded DeBERTa ABSA. Running on device: {self.device.upper()}")
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to load DeBERTa ABSA: {e}. Falling back to lexicon method.")
                self.use_deberta = False
                self.fallback_sia = SentimentIntensityAnalyzer()
        else:
            logger.info("Initializing baseline Lexicon-based ABSA analyzer...")
            self.fallback_sia = SentimentIntensityAnalyzer()

    def _vader_absa(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Fallback ABSA: Isolates sentences mentioning the aspect,
        scores them with VADER, and returns the aggregate score.
        """
        sentences = sent_tokenize(text)
        aspect_lower = aspect.lower()
        
        # Keep only sentences mentioning the aspect
        relevant_sentences = [s for s in sentences if aspect_lower in s.lower()]
        
        if not relevant_sentences:
            return {
                "aspect": aspect,
                "sentiment_label": "neutral",
                "confidence": 0.50,
                "score": 0.0
            }
            
        # Score matching sentences
        scores = []
        for s in relevant_sentences:
            pol = self.fallback_sia.polarity_scores(s)
            scores.append(pol["compound"])
            
        avg_score = np.mean(scores)
        
        # Map score to label
        if avg_score >= 0.05:
            label = "positive"
            confidence = min(0.95, 0.5 + abs(avg_score))
        elif avg_score <= -0.05:
            label = "negative"
            confidence = min(0.95, 0.5 + abs(avg_score))
        else:
            label = "neutral"
            confidence = 0.60
            
        return {
            "aspect": aspect,
            "sentiment_label": label,
            "confidence": float(round(confidence, 2)),
            "score": float(round(avg_score, 2))
        }

    def _deberta_absa(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Runs transformer inference with DeBERTa ABSA:
        Formulates input as "[CLS] text [SEP] aspect [SEP]"
        """
        # Clean inputs
        text = str(text)
        aspect = str(aspect)
        
        # Prepare inputs
        inputs = self.tokenizer(
            text,
            aspect,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move tensors to training device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        # Label indices: typically 0 -> Negative, 1 -> Neutral, 2 -> Positive
        # Depending on specific ABSA checkpoints, verify mapped index labels
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        labels_map = {0: "negative", 1: "neutral", 2: "positive"}
        label = labels_map.get(pred_idx, "neutral")
        
        return {
            "aspect": aspect,
            "sentiment_label": label,
            "confidence": float(round(confidence, 2)),
            "score": float(round(probs[2] - probs[0], 2)) # positive prob - negative prob
        }

    def analyze_review_aspects(self, text: str, aspects: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analyzes a single review text for a list of aspects.
        If no aspects are passed, dynamically extracts candidate terms from the text.
        """
        if not text or not text.strip():
            return []
            
        # If no aspects provided, find which default aspects are present in the text
        if not aspects:
            text_lower = text.lower()
            aspects = [a for a in DEFAULT_ASPECTS if a in text_lower]
            
        if not aspects:
            return []
            
        results = []
        for aspect in aspects:
            if self.use_deberta:
                res = self._deberta_absa(text, aspect)
            else:
                res = self._vader_absa(text, aspect)
            results.append(res)
            
        return results

if __name__ == "__main__":
    # Test script run
    logger.info("Running AspectSentimentAnalyzer self-test...")
    
    # Run with fallback lexicon first for fast validation
    analyzer = AspectSentimentAnalyzer(use_deberta=False)
    
    test_review = (
        "The screen display is incredibly clear with gorgeous colors. "
        "However, the battery life is absolutely terrible, barely lasting 4 hours. "
        "The pricing seems somewhat fair."
    )
    
    logger.info(f"Testing text: '{test_review}'")
    aspect_sentiments = analyzer.analyze_review_aspects(test_review)
    
    print("\nAspect Sentiments Result:")
    for item in aspect_sentiments:
        print(f"  {item['aspect'].capitalize()} -> {item['sentiment_label'].upper()} (Confidence: {item['confidence']:.2f}, Polarity Score: {item['score']})")
