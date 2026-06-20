"""
BART Summarization Pipeline
===========================
Implements STEP 8 of the RevuMind V2 workflow.
Summarizes grouped reviews (e.g., all reviews of a product, or all reviews
under a specific topic) into a concise executive summary.
"""

import logging
import os
import re
from typing import Any, Dict, List

# NLP imports for fallback
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data if needed
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

# Try importing Hugging Face transformers
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ExecutiveSummarizer:
    """
    Summarizer module that leverages a pre-trained BART seq2seq model,
    falling back to an extractive word-frequency scoring algorithm when deep learning is unavailable.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", use_bart: bool = True):
        self.model_name = model_name
        self.use_bart = use_bart and TRANSFORMERS_AVAILABLE

        self.tokenizer = None
        self.model = None
        self.device = None
        self.english_stopwords = set(stopwords.words("english"))

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """
        Loads the BART model or flags standard fallback.
        """
        if self.use_bart:
            try:
                logger.info(f"Loading BART model: {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

                self.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu")
                )
                logger.info(f"Loaded BART. Running on device: {self.device.upper()}")
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to load BART: {e}. Falling back to extractive method.")
                self.use_bart = False
        else:
            logger.info("Initializing baseline Extractive Summarizer...")

    def _extractive_summarize(self, texts: List[str], num_sentences: int = 3) -> str:
        """
        Fallback Extractive Summarizer:
        1. Tokenizes input text into sentences and words.
        2. Calculates word frequencies (ignoring stopwords/punctuation).
        3. Scores each sentence by summing word frequencies.
        4. Selects the top-scoring sentences and returns them in order.
        """
        # Combine texts into a single corpus document
        combined_text = " ".join([t.strip() for t in texts if t.strip()])
        if not combined_text:
            return ""

        sentences = sent_tokenize(combined_text)
        if len(sentences) <= num_sentences:
            return combined_text

        # Clean and count word frequencies
        words = word_tokenize(combined_text.lower())
        word_frequencies = {}
        for w in words:
            if w.isalnum() and w not in self.english_stopwords:
                word_frequencies[w] = word_frequencies.get(w, 0) + 1

        if not word_frequencies:
            return " ".join(sentences[:num_sentences])

        # Normalize frequencies
        max_freq = max(word_frequencies.values())
        for w in word_frequencies:
            word_frequencies[w] = word_frequencies[w] / max_freq

        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            score = 0.0
            for w in sentence_words:
                if w in word_frequencies:
                    score += word_frequencies[w]
            sentence_scores[i] = score

        # Get indices of top sentences
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[
            :num_sentences
        ]
        # Sort indices to keep chronological reading order
        top_indices.sort()

        summary_sentences = [sentences[idx] for idx in top_indices]
        return " ".join(summary_sentences)

    def _bart_summarize(self, texts: List[str]) -> str:
        """
        Concatenates text, runs tokenization, and uses BART Seq2Seq generation.
        """
        # Combine texts with space separator
        combined_text = " ".join([t.strip() for t in texts if t.strip()])

        # Ingest text, truncate to BART limit (1024 tokens)
        inputs = self.tokenizer(
            combined_text, max_length=1024, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    def generate_summary(self, texts: List[str]) -> str:
        """
        Synthesizes an executive summary from a group of review texts.
        """
        if not texts:
            return ""

        if self.use_bart:
            try:
                return self._bart_summarize(texts)
            except Exception as e:
                logger.error(f"BART execution failed: {e}. Falling back to extractive summary.")
                return self._extractive_summarize(texts)
        else:
            return self._extractive_summarize(texts)


if __name__ == "__main__":
    # Test script run
    logger.info("Running ExecutiveSummarizer self-test...")

    # Run with fallback extractive mode for fast test
    summarizer = ExecutiveSummarizer(use_bart=False)

    test_reviews = [
        "I bought this product last week and I must say the battery life is absolutely incredible. It lasted me two full days of heavy usage.",
        "The sound quality on these earphones is outstanding, with rich bass and crystal clear trebles. I am extremely satisfied.",
        "Shipping was incredibly fast. The package arrived in Seattle within 24 hours of placing the order, well protected.",
        "However, the charging speed is rather slow, taking almost three hours to fully charge from zero percent.",
        "The app is slightly bloated and crashes occasionally when connecting to bluetooth, which is annoying.",
    ]

    logger.info(f"Summarizing {len(test_reviews)} reviews...")
    summary = summarizer.generate_summary(test_reviews)

    print("\nGenerated Summary:")
    print(summary)
