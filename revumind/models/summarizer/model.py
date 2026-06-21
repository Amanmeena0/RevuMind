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

# Transformers availability check (lazy-loaded inside constructor)
TRANSFORMERS_AVAILABLE = None
torch = None
AutoModelForSeq2SeqLM = None
AutoTokenizer = None


class ExecutiveSummarizer:
    """
    Summarizer module that leverages a pre-trained BART seq2seq model,
    falling back to an extractive word-frequency scoring algorithm when deep learning is unavailable.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", use_bart: bool = True):
        global TRANSFORMERS_AVAILABLE, torch, AutoModelForSeq2SeqLM, AutoTokenizer
        if TRANSFORMERS_AVAILABLE is None:
            try:
                import torch as torch_lib
                from transformers import AutoModelForSeq2SeqLM as AutoModel_lib, AutoTokenizer as AutoTokenizer_lib
                torch = torch_lib
                AutoModelForSeq2SeqLM = AutoModel_lib
                AutoTokenizer = AutoTokenizer_lib
                TRANSFORMERS_AVAILABLE = True
            except ImportError:
                TRANSFORMERS_AVAILABLE = False

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

    def _extractive_summarize(self, texts: List[str], num_sentences: int = 5) -> str:
        """
        Diversity-aware Extractive Summarizer:
        1. Processes each review independently to extract its best sentence.
        2. Filters out anecdotal fragments and very short sentences.
        3. Scores sentences by evaluative language and word frequencies.
        4. Picks top diverse sentences (max one per review).
        """
        if not texts:
            return ""

        # Evaluative signal words that indicate product opinions
        eval_words = {
            "great", "excellent", "good", "best", "love", "amazing", "perfect",
            "terrible", "worst", "awful", "bad", "poor", "horrible", "broken",
            "recommend", "quality", "value", "price", "taste", "flavor",
            "delicious", "fresh", "stale", "disappointed", "satisfied",
            "worth", "cheap", "expensive", "favorite", "outstanding",
            "product", "item", "purchase", "order", "buy", "bought",
            "better", "worse", "compared", "alternative", "overall",
        }

        # Build global word frequencies across all reviews
        all_words = []
        for text in texts:
            if text and text.strip():
                all_words.extend(word_tokenize(text.lower()))

        word_frequencies = {}
        for w in all_words:
            if w.isalnum() and w not in self.english_stopwords and len(w) > 2:
                word_frequencies[w] = word_frequencies.get(w, 0) + 1

        if not word_frequencies:
            return " ".join(texts[:num_sentences])

        max_freq = max(word_frequencies.values())
        for w in word_frequencies:
            word_frequencies[w] = word_frequencies[w] / max_freq

        # Score each sentence from each review independently
        candidates = []  # (score, review_idx, sentence_text)
        for review_idx, text in enumerate(texts):
            if not text or not text.strip():
                continue
            sentences = sent_tokenize(text.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                # Filter: skip very short or very long sentences
                word_count = len(sentence.split())
                if word_count < 6 or word_count > 45:
                    continue

                # Filter: skip sentences starting with dashes (list fragments)
                if sentence.startswith("-") or sentence.startswith("--"):
                    continue

                words = word_tokenize(sentence.lower())

                # Base score from word frequencies
                freq_score = sum(word_frequencies.get(w, 0) for w in words if w.isalnum())
                # Normalize by length to avoid bias toward long sentences
                freq_score = freq_score / max(len(words), 1)

                # Bonus for evaluative/opinion language
                eval_count = sum(1 for w in words if w in eval_words)
                eval_bonus = eval_count * 0.3

                # Penalty for overly personal/anecdotal sentences
                personal_words = {"i", "my", "me", "we", "our"}
                personal_count = sum(1 for w in words if w in personal_words)
                personal_penalty = min(personal_count * 0.1, 0.4)

                total_score = freq_score + eval_bonus - personal_penalty
                candidates.append((total_score, review_idx, sentence))

        if not candidates:
            return " ".join(texts[:num_sentences])

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Pick top sentences ensuring diversity (max 1 per review)
        selected = []
        used_reviews = set()
        for score, review_idx, sentence in candidates:
            if review_idx in used_reviews:
                continue
            selected.append(sentence)
            used_reviews.add(review_idx)
            if len(selected) >= num_sentences:
                break

        return " | ".join(selected)

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
