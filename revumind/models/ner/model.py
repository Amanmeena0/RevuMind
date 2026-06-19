"""
Named Entity Recognition (NER) Pipeline
======================================
Implements STEP 4 of the RevuMind V2 workflow.
Extracts Brands, Products, Organizations, and Locations from review texts
using spaCy (leveraging Transformer models or standard web baseline).
"""

import os
import re
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
import logging
from typing import List, Dict, Any

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Known brand list for phrase matching
KNOWN_BRANDS = [
    "Samsung", "Apple", "Sony", "LG", "OnePlus", "Xiaomi", "Realme", "Oppo", "Vivo",
    "Motorola", "Nokia", "Huawei", "Google", "Microsoft", "Dell", "HP", "Lenovo", "Asus",
    "Acer", "Toshiba", "Panasonic", "Philips", "Bose", "JBL", "Sennheiser", "boAt",
    "Noise", "Boat", "Skullcandy", "Jabra", "Anker", "Portronics", "Ambrane", "Nike",
    "Adidas", "Puma", "Reebok", "Levi", "Zara", "IKEA"
]

# Common features to match as aspects/features
COMMON_FEATURES = [
    "battery", "screen", "display", "camera", "lens", "price", "charging", "charger",
    "sound", "audio", "speaker", "bass", "mic", "microphone", "software", "app",
    "shipping", "delivery", "frame", "body", "durability", "material", "cord", "wire"
]

class ProductNERPipeline:
    """
    Production-ready NER wrapper that loads either spaCy's transformer model (en_core_web_trf)
    or falls back to the lightweight standard pipeline (en_core_web_sm).
    """
    def __init__(self, use_transformer: bool = True):
        self.nlp = None
        self.use_transformer = use_transformer
        self._initialize_spacy()
        self._add_custom_matchers()

    def _initialize_spacy(self):
        """
        Attempts to load the requested model, with fallback capabilities.
        """
        model_name = "en_core_web_trf" if self.use_transformer else "en_core_web_sm"
        
        try:
            logger.info(f"Loading spaCy model: {model_name}...")
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found.")
            if self.use_transformer:
                logger.info("Attempting to load fallback model en_core_web_sm...")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.info("Downloading en_core_web_sm...")
                    import subprocess
                    import sys
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load("en_core_web_sm")
            else:
                logger.info("Downloading en_core_web_sm...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")

    def _add_custom_matchers(self):
        """
        Sets up PhraseMatcher and Matcher to tag custom entities like BRAND and FEATURE
        """
        # 1. Add PhraseMatcher for known brands
        self.brand_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        brand_docs = [self.nlp.make_doc(b) for b in KNOWN_BRANDS]
        self.brand_matcher.add("BRAND_PHRASES", brand_docs)
        
        # 2. Add PhraseMatcher for common features
        self.feature_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        feature_docs = [self.nlp.make_doc(f) for f in COMMON_FEATURES]
        self.feature_matcher.add("FEATURE_PHRASES", feature_docs)
        
        # 3. Add Regex Matcher for product model numbers (e.g. S24, Pixel 8, Gen 3)
        self.model_matcher = Matcher(self.nlp.vocab)
        model_patterns = [
            [{"TEXT": {"REGEX": r"\b[A-Za-z]+\d{1,4}[A-Za-z]?\b"}}],  # e.g., S24, RT68U
            [{"TEXT": {"REGEX": r"\b[A-Za-z]+[-]\d{1,4}\b"}}],         # e.g., WH-1000
        ]
        self.model_matcher.add("MODEL_NUMBERS", model_patterns)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs spaCy pipeline + custom matchers on review text.
        Extracts Brands, Products, Organizations, Locations, and Features.
        """
        if not text or not text.strip():
            return []
            
        doc = self.nlp(text)
        extracted = []
        seen = set() # Avoid duplicate entity listings for same character offsets
        
        # 1. Run custom PhraseMatchers
        # Brand Phrases
        brand_matches = self.brand_matcher(doc)
        for _, start, end in brand_matches:
            span = doc[start:end]
            offset = (span.start_char, span.end_char)
            if offset not in seen:
                extracted.append({
                    "text": span.text,
                    "label": "BRAND",
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "confidence": 0.95
                })
                seen.add(offset)
                
        # Feature Phrases
        feature_matches = self.feature_matcher(doc)
        for _, start, end in feature_matches:
            span = doc[start:end]
            offset = (span.start_char, span.end_char)
            if offset not in seen:
                extracted.append({
                    "text": span.text,
                    "label": "FEATURE",
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "confidence": 0.90
                })
                seen.add(offset)
                
        # Model Matcher
        model_matches = self.model_matcher(doc)
        for _, start, end in model_matches:
            span = doc[start:end]
            offset = (span.start_char, span.end_char)
            if offset not in seen:
                extracted.append({
                    "text": span.text,
                    "label": "PRODUCT_MODEL",
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "confidence": 0.85
                })
                seen.add(offset)

        # 2. Extract Standard spaCy entities (ORG, PRODUCT, GPE/Locations)
        # Note: we filter out entities that overlap with custom matches
        for ent in doc.ents:
            # Map labels to our requested schema
            mapped_label = None
            confidence = 0.80
            
            if ent.label_ in ["ORG"]:
                mapped_label = "ORG"
            elif ent.label_ in ["PRODUCT", "WORK_OF_ART"]:
                mapped_label = "PRODUCT"
            elif ent.label_ in ["GPE", "LOC"]:
                mapped_label = "LOC"
                
            if mapped_label:
                # Add if no overlap with our custom matchers
                overlap = False
                for start_char in range(ent.start_char, ent.end_char):
                    for ext in extracted:
                        if start_char in range(ext["start_char"], ext["end_char"]):
                            overlap = True
                            break
                            
                if not overlap:
                    offset = (ent.start_char, ent.end_char)
                    extracted.append({
                        "text": ent.text,
                        "label": mapped_label,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": confidence
                    })
                    seen.add(offset)
                    
        # Sort by starting character
        extracted = sorted(extracted, key=lambda x: x["start_char"])
        return extracted

if __name__ == "__main__":
    # Test script run
    logger.info("Running ProductNERPipeline self-test...")
    
    pipeline = ProductNERPipeline(use_transformer=False) # Use small model for fast test
    
    test_text = "I bought my Samsung Galaxy S24 from Amazon in Seattle. The battery screen is vivid but the charger frame is hot."
    entities = pipeline.extract_entities(test_text)
    
    print(f"Text: '{test_text}'\n")
    print("Extracted Entities:")
    for ent in entities:
        print(f"  [{ent['label']}] '{ent['text']}' (Index: {ent['start_char']}-{ent['end_char']}, Confidence: {ent['confidence']:.2f})")
