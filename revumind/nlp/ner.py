"""
Named Entity Recognition (NER) for Product Reviews
====================================================
Extracts brands, product names, organizations, and custom
product-specific entities from review text using spaCy.

Pipeline:
  1.  spaCy built-in NER  (ORG, PRODUCT, GPE, MONEY, etc.)
  2.  Rule-based matcher   (regex + phrase patterns for known brands)
  3.  Custom entity ruler  (add your own brand/product dictionary)
  4.  Pattern-based extractor (e.g. "Model X123", "Version 2.0")
  5.  Post-processing      (normalise, deduplicate, confidence score)
  6.  Batch processing     (run over an entire reviews DataFrame)
  7.  Visualisation        (entity frequency charts + displacy HTML)
  8.  Analysis helpers     (brand sentiment, entity co-occurrence)

Usage:
    from ner_pipeline import ProductNERPipeline
    ner = ProductNERPipeline()
    result = ner.extract("I love my Samsung Galaxy S24. The camera is amazing!")
"""

import re
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ── spaCy ─────────────────────────────────────────────────────────────────────
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language

# Download model if needed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True)
    nlp = spacy.load("en_core_web_sm")

print(f"spaCy version: {spacy.__version__} | Model: en_core_web_sm")

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE   = ["#5DCAA5","#378ADD","#EF9F27","#D85A30","#7F77DD",
             "#D4537E","#97C459","#888780"]
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})


# ══════════════════════════════════════════════════════════════════════════════
# 0.  ENTITY TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

# Known brands — used for PhraseMatcher and EntityRuler
KNOWN_BRANDS = [
    # Electronics
    "Samsung","Apple","Sony","LG","OnePlus","Xiaomi","Realme","Oppo","Vivo",
    "Motorola","Nokia","Huawei","Google","Microsoft","Dell","HP","Lenovo","Asus",
    "Acer","Toshiba","Panasonic","Philips","Bose","JBL","Sennheiser","boAt",
    "Noise","Boat","Skullcandy","Jabra","Anker","Portronics","Ambrane",
    # Appliances
    "Whirlpool","IFB","Bosch","Godrej","Haier","Voltas","Daikin","Blue Star",
    "Prestige","Butterfly","Havells","Bajaj","Usha","Orient","Crompton",
    # E-commerce
    "Amazon","Flipkart","Myntra","Meesho","Nykaa","Snapdeal",
    # General
    "Nike","Adidas","Puma","Reebok","Levi","H&M","Zara","IKEA",
]

# Product model number patterns (regex)
MODEL_PATTERNS = [
    r"\b[A-Z]{1,5}[\s\-]?\d{2,6}[A-Za-z]?\b",       # S24, Galaxy A54, RT-AC68U
    r"\b[A-Z][a-z]+\s+[A-Z]\d{1,3}\b",                # iPhone X, Pixel 8
    r"\b\d{1,2}th\s+[Gg]en(?:eration)?\b",             # 5th Generation
    r"\bGen\s*\d\b",                                    # Gen 2, Gen3
    r"\bv\d+(?:\.\d+)*\b",                             # v2.0, v3.1.2
    r"\b(?:Pro|Max|Ultra|Plus|Lite|Mini|SE|Air|Note)\b",# product tiers
]

# Entity label mapping for display
LABEL_COLORS = {
    "BRAND":    "#5DCAA5",
    "PRODUCT":  "#378ADD",
    "ORG":      "#7F77DD",
    "MONEY":    "#EF9F27",
    "QUANTITY": "#D85A30",
    "GPE":      "#D4537E",
    "MODEL":    "#97C459",
    "FEATURE":  "#888780",
}

# Product feature keywords (aspect extraction)
FEATURE_ASPECTS = {
    "battery":    ["battery","charge","charging","mah","life","drain","fast charge"],
    "camera":     ["camera","photo","picture","video","lens","megapixel","mp","zoom","selfie"],
    "display":    ["screen","display","amoled","lcd","resolution","refresh","brightness"],
    "sound":      ["sound","audio","speaker","bass","volume","noise","headphone","earphone"],
    "performance":["performance","speed","fast","lag","processor","ram","storage","hang"],
    "build":      ["build","quality","design","material","plastic","metal","glass","weight"],
    "price":      ["price","value","cost","worth","money","affordable","expensive","cheap"],
    "delivery":   ["delivery","shipping","packaging","arrived","box","damage","courier"],
    "software":   ["software","update","ui","ux","interface","app","bloatware","os"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENTITY RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text:       str
    label:      str        # BRAND, PRODUCT, ORG, MODEL, FEATURE, ...
    source:     str        # "spacy", "phrase_matcher", "entity_ruler", "regex"
    start_char: int = 0
    end_char:   int = 0
    confidence: float = 1.0
    context:    str = ""   # surrounding words for disambiguation


@dataclass
class NERResult:
    original_text:  str
    entities:       list[Entity] = field(default_factory=list)
    brands:         list[str]    = field(default_factory=list)
    products:       list[str]    = field(default_factory=list)
    model_numbers:  list[str]    = field(default_factory=list)
    orgs:           list[str]    = field(default_factory=list)
    money:          list[str]    = field(default_factory=list)
    aspects:        dict         = field(default_factory=dict)
    entity_count:   int          = 0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOM PIPELINE COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_phrase_matcher(nlp_model, brands: list[str]) -> PhraseMatcher:
    """
    PhraseMatcher: exact string matching for known brand names.
    Faster than regex for large dictionaries.
    Case-insensitive via LOWER attribute.
    """
    matcher = PhraseMatcher(nlp_model.vocab, attr="LOWER")
    patterns = [nlp_model.make_doc(b.lower()) for b in brands]
    matcher.add("BRAND", patterns)
    return matcher


def build_entity_ruler(nlp_model, brands: list[str]) -> "EntityRuler":
    """
    EntityRuler: adds patterns directly into spaCy's NER pipeline.
    Runs BEFORE the statistical NER so it takes priority for known brands.
    """
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    patterns = []
    for brand in brands:
        # Exact match
        patterns.append({"label": "ORG", "pattern": brand})
        # With common suffixes
        for suffix in ["Inc", "Ltd", "Corp", "Electronics", "India"]:
            patterns.append({"label": "ORG",
                             "pattern": [{"LOWER": brand.lower()},
                                         {"LOWER": suffix.lower()}]})
    ruler.add_patterns(patterns)
    return ruler


def extract_model_numbers(text: str) -> list[tuple[str, int, int]]:
    """
    Regex-based extraction of product model numbers.
    Returns list of (matched_text, start, end) tuples.
    """
    found = []
    for pattern in MODEL_PATTERNS:
        for m in re.finditer(pattern, text):
            found.append((m.group(), m.start(), m.end()))
    # Deduplicate overlapping matches
    found.sort(key=lambda x: x[1])
    deduped = []
    last_end = -1
    for text_m, start, end in found:
        if start >= last_end:
            deduped.append((text_m, start, end))
            last_end = end
    return deduped


def extract_aspects(text: str) -> dict[str, list[str]]:
    """
    Keyword-based aspect/feature extraction.
    Maps review text to product aspects (battery, camera, etc.)
    """
    text_lower = text.lower()
    found = {}
    for aspect, keywords in FEATURE_ASPECTS.items():
        matched = [kw for kw in keywords if re.search(r'\b'+re.escape(kw)+r'\b', text_lower)]
        if matched:
            found[aspect] = matched
    return found


def get_entity_context(text: str, start: int, end: int, window: int = 30) -> str:
    """Get surrounding words for context/disambiguation."""
    ctx_start = max(0, start - window)
    ctx_end   = min(len(text), end + window)
    ctx = text[ctx_start:ctx_end]
    return f"...{ctx}..."


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MAIN NER PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ProductNERPipeline:
    """
    Multi-layer NER system combining:
      Layer 1: spaCy statistical NER    → general entities (ORG, PRODUCT, MONEY)
      Layer 2: PhraseMatcher            → known brand dictionary
      Layer 3: Regex matcher            → model numbers, version strings
      Layer 4: Aspect extractor         → product features mentioned

    Why multiple layers?
      spaCy's statistical model is great for general entities but misses
      domain-specific brand names (especially Indian brands like boAt).
      The PhraseMatcher catches those with 100% precision.
      Regex catches structured patterns like "Galaxy S24" that neither catches.
    """

    def __init__(
        self,
        brands:            list[str] = KNOWN_BRANDS,
        min_entity_length: int       = 2,
        deduplicate:       bool      = True,
    ):
        self.brands            = brands
        self.min_len           = min_entity_length
        self.deduplicate       = deduplicate

        # Load fresh spaCy model (without entity ruler initially)
        self.nlp = spacy.load("en_core_web_sm")

        # Add entity ruler BEFORE statistical NER
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler_patterns = []
        for brand in brands:
            ruler_patterns.append({"label": "ORG", "pattern": brand})
            ruler_patterns.append({"label": "ORG",
                                   "pattern": [{"LOWER": brand.lower()}]})
        ruler.add_patterns(ruler_patterns)

        # Build phrase matcher for secondary matching
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        phrase_patterns = [self.nlp.make_doc(b.lower()) for b in brands]
        self.phrase_matcher.add("BRAND", phrase_patterns)

        print(f"  Pipeline components: {self.nlp.pipe_names}")
        print(f"  Loaded {len(brands)} brand patterns")

    # ── Core extraction ───────────────────────────────────────────────────────
    def extract(self, text: str) -> NERResult:
        """
        Run all NER layers on a single review text.
        Returns NERResult with entities grouped by type.
        """
        result = NERResult(original_text=text)
        all_entities: list[Entity] = []

        # ── Layer 1 & 2: spaCy NER + EntityRuler ─────────────────────────────
        doc = self.nlp(text)
        for ent in doc.ents:
            if len(ent.text.strip()) < self.min_len:
                continue
            # Map spaCy labels to our taxonomy
            label = self._map_label(ent.label_)
            context = get_entity_context(text, ent.start_char, ent.end_char)
            all_entities.append(Entity(
                text       = ent.text.strip(),
                label      = label,
                source     = "spacy+ruler",
                start_char = ent.start_char,
                end_char   = ent.end_char,
                confidence = 0.9,
                context    = context,
            ))

        # ── Layer 3: PhraseMatcher for additional brand hits ──────────────────
        matches = self.phrase_matcher(doc)
        existing_spans = {(e.start_char, e.end_char) for e in all_entities}
        for match_id, start, end in matches:
            span = doc[start:end]
            if (span.start_char, span.end_char) not in existing_spans:
                all_entities.append(Entity(
                    text       = span.text.strip(),
                    label      = "BRAND",
                    source     = "phrase_matcher",
                    start_char = span.start_char,
                    end_char   = span.end_char,
                    confidence = 1.0,
                    context    = get_entity_context(text, span.start_char, span.end_char),
                ))
                existing_spans.add((span.start_char, span.end_char))

        # ── Layer 4: Regex for model numbers ─────────────────────────────────
        model_hits = extract_model_numbers(text)
        for model_text, start, end in model_hits:
            if (start, end) not in existing_spans and len(model_text.strip()) > 1:
                all_entities.append(Entity(
                    text       = model_text.strip(),
                    label      = "MODEL",
                    source     = "regex",
                    start_char = start,
                    end_char   = end,
                    confidence = 0.85,
                    context    = get_entity_context(text, start, end),
                ))

        # ── Layer 5: Aspect extraction ────────────────────────────────────────
        result.aspects = extract_aspects(text)

        # ── Post-processing ───────────────────────────────────────────────────
        if self.deduplicate:
            all_entities = self._deduplicate(all_entities)

        result.entities = sorted(all_entities, key=lambda e: e.start_char)

        # ── Group by type ─────────────────────────────────────────────────────
        for e in result.entities:
            t = e.text
            if e.label in ("BRAND", "ORG"):
                if t not in result.brands:
                    result.brands.append(t)
            elif e.label == "PRODUCT":
                if t not in result.products:
                    result.products.append(t)
            elif e.label == "MODEL":
                if t not in result.model_numbers:
                    result.model_numbers.append(t)
            elif e.label == "MONEY":
                result.money.append(t)

        result.entity_count = len(result.entities)
        return result

    def _map_label(self, spacy_label: str) -> str:
        """Translate spaCy's generic labels to our product taxonomy."""
        mapping = {
            "ORG":      "BRAND",
            "PRODUCT":  "PRODUCT",
            "GPE":      "GPE",
            "LOC":      "GPE",
            "MONEY":    "MONEY",
            "QUANTITY": "QUANTITY",
            "CARDINAL": "QUANTITY",
            "PERSON":   "PERSON",
            "DATE":     "DATE",
            "NORP":     "ORG",
            "FAC":      "PRODUCT",
            "WORK_OF_ART": "PRODUCT",
        }
        return mapping.get(spacy_label, spacy_label)

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """
        Remove duplicate or overlapping entities.
        Priority: phrase_matcher > spacy+ruler > regex
        """
        # Sort by start_char, then by confidence descending
        entities.sort(key=lambda e: (e.start_char, -e.confidence))
        deduped = []
        seen_texts = set()
        last_end = -1

        for ent in entities:
            # Skip if overlapping with a previous entity
            if ent.start_char < last_end:
                continue
            # Skip exact text duplicates (case-insensitive)
            key = ent.text.lower().strip()
            if key in seen_texts:
                continue
            deduped.append(ent)
            seen_texts.add(key)
            last_end = ent.end_char

        return deduped

    # ── Batch processing ──────────────────────────────────────────────────────
    def process_dataframe(
        self,
        df:       pd.DataFrame,
        text_col: str = "review_text",
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Run NER over all reviews in a DataFrame.
        Adds columns: brands, products, model_numbers, aspects,
                      entity_count, has_brand, top_brand
        """
        print(f"  Running NER on {len(df):,} reviews…")
        rows = []
        for i, text in enumerate(df[text_col].fillna(""), 1):
            r = self.extract(str(text))
            rows.append({
                "brands":        " | ".join(r.brands),
                "products":      " | ".join(r.products),
                "model_numbers": " | ".join(r.model_numbers),
                "money_mentions":" | ".join(r.money),
                "aspects":       " | ".join(r.aspects.keys()),
                "entity_count":  r.entity_count,
                "has_brand":     len(r.brands) > 0,
                "brand_count":   len(r.brands),
                "top_brand":     r.brands[0] if r.brands else "",
            })
            if i % 100 == 0:
                print(f"    {i}/{len(df)} done…")

        result_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df, result_df], axis=1)

    # ── displacy-style HTML renderer ──────────────────────────────────────────
    def render_html(self, text: str, save_path: str = "ner_render.html") -> str:
        """
        Generate highlighted HTML showing entities inline in text.
        Color-coded by entity type. Save to file for viewing in browser.
        """
        result = self.extract(text)
        html = ["<div style='font-family:monospace;font-size:15px;line-height:2;padding:16px'>"]
        last = 0
        for ent in result.entities:
            # Text before this entity
            html.append(text[last:ent.start_char].replace("\n", "<br>"))
            color = LABEL_COLORS.get(ent.label, "#cccccc")
            html.append(
                f"<mark style='background:{color}22;border:1.5px solid {color};"
                f"border-radius:4px;padding:2px 6px;margin:0 2px;'>"
                f"<b>{ent.text}</b>"
                f"<sup style='color:{color};font-size:10px;margin-left:3px'>"
                f"{ent.label}</sup></mark>"
            )
            last = ent.end_char
        html.append(text[last:].replace("\n", "<br>"))
        html.append("</div>")
        html_str = "".join(html)
        if save_path:
            with open(save_path, "w") as f:
                f.write(f"<html><body bgcolor='#fafafa'>{html_str}</body></html>")
        return html_str


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def brand_sentiment_analysis(
    df: pd.DataFrame,
    brand_col:     str = "brands",
    sentiment_col: str = "vader_compound",
) -> pd.DataFrame:
    """
    Join extracted brands with sentiment scores.
    Returns a DataFrame showing avg sentiment per brand.
    """
    rows = []
    for _, row in df.iterrows():
        brands = [b.strip() for b in str(row[brand_col]).split("|") if b.strip()]
        sentiment = row.get(sentiment_col, 0)
        for brand in brands:
            rows.append({"brand": brand, "sentiment": sentiment})
    if not rows:
        return pd.DataFrame()
    brand_df = pd.DataFrame(rows)
    return (brand_df.groupby("brand")["sentiment"]
            .agg(["mean", "count", "std"])
            .rename(columns={"mean":"avg_sentiment","count":"review_count","std":"sentiment_std"})
            .sort_values("review_count", ascending=False)
            .round(3))


def entity_cooccurrence(
    df: pd.DataFrame,
    brand_col: str = "brands",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find which brands are mentioned together in the same review.
    Useful for identifying competitive comparisons.
    """
    pairs = Counter()
    for brands_str in df[brand_col].fillna(""):
        brands = [b.strip() for b in brands_str.split("|") if b.strip()]
        for i in range(len(brands)):
            for j in range(i + 1, len(brands)):
                pair = tuple(sorted([brands[i], brands[j]]))
                pairs[pair] += 1
    rows = [{"brand_1": a, "brand_2": b, "co_mentions": c}
            for (a, b), c in pairs.most_common(top_n)]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_entity_distribution(
    results: list[NERResult],
    save_path: str = "entity_distribution.png",
) -> None:
    """Bar chart of entity type frequencies across all reviews."""
    label_counts = Counter()
    for r in results:
        for e in r.entities:
            label_counts[e.label] += 1

    if not label_counts:
        return

    labels, counts = zip(*label_counts.most_common())
    colors = [LABEL_COLORS.get(l, "#888780") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Entity Type Distribution Across Reviews", fontweight="bold", fontsize=13)
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_top_brands(
    results: list[NERResult],
    top_n: int = 15,
    save_path: str = "top_brands.png",
) -> None:
    """Horizontal bar chart of most frequently mentioned brands."""
    brand_counter = Counter()
    for r in results:
        for brand in r.brands:
            brand_counter[brand.title()] += 1

    if not brand_counter:
        print("  No brands found to plot.")
        return

    top = brand_counter.most_common(top_n)
    labels, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    bars = ax.barh(list(labels)[::-1], list(counts)[::-1],
                   color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_title(f"Top {top_n} Most Mentioned Brands", fontweight="bold", fontsize=13)
    ax.set_xlabel("Mention Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_aspect_frequency(
    results: list[NERResult],
    save_path: str = "aspect_frequency.png",
) -> None:
    """Bar chart of how often each product aspect is mentioned."""
    aspect_counter = Counter()
    for r in results:
        for aspect in r.aspects:
            aspect_counter[aspect] += 1

    if not aspect_counter:
        return

    aspects, counts = zip(*aspect_counter.most_common())
    colors = PALETTE[:len(aspects)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(aspects, counts, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Product Aspect Mention Frequency", fontweight="bold", fontsize=13)
    ax.set_xlabel("Product Aspect")
    ax.set_ylabel("Reviews Mentioning Aspect")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_brand_sentiment(
    brand_sent_df: pd.DataFrame,
    top_n: int = 12,
    save_path: str = "brand_sentiment.png",
) -> None:
    """Horizontal bar chart: avg sentiment score per brand."""
    if brand_sent_df.empty:
        return

    top = brand_sent_df.head(top_n)
    colors = ["#5DCAA5" if s >= 0 else "#D85A30"
              for s in top["avg_sentiment"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(top) * 0.5)))

    # Sentiment
    bars = axes[0].barh(top.index[::-1], top["avg_sentiment"][::-1],
                        color=colors[::-1], alpha=0.85, edgecolor="white")
    axes[0].axvline(0, color="grey", linewidth=0.8)
    axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axes[0].set_title("Avg Sentiment Score by Brand", fontweight="bold")
    axes[0].set_xlabel("Avg VADER compound score")

    # Review count
    axes[1].barh(top.index[::-1], top["review_count"][::-1],
                 color=PALETTE[1], alpha=0.85, edgecolor="white")
    axes[1].set_title("Review Count by Brand", fontweight="bold")
    axes[1].set_xlabel("Number of reviews")

    plt.suptitle("Brand Sentiment Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_ner_result(result: NERResult) -> None:
    sep = "─" * 62
    label_icons = {
        "BRAND":"🏷 ","PRODUCT":"📦","MODEL":"🔢","MONEY":"💰",
        "GPE":"📍","PERSON":"👤","DATE":"📅","QUANTITY":"🔢","ORG":"🏢",
    }
    print(f"\n{'═'*62}")
    print(f"  NER RESULT")
    print(f"{'═'*62}")
    print(f"  Text: \"{result.original_text[:100]}"
          f"{'…' if len(result.original_text)>100 else ''}\"")
    print(f"  Entities found: {result.entity_count}")
    print(sep)

    if result.entities:
        print(f"  {'ENTITY':<28} {'TYPE':<12} {'SOURCE':<16} {'CONF'}")
        print(f"  {'─'*26} {'─'*10} {'─'*14} {'─'*4}")
        for e in result.entities:
            icon = label_icons.get(e.label, "  ")
            conf = f"{e.confidence:.0%}"
            print(f"  {icon}{e.text:<26} {e.label:<12} {e.source:<16} {conf}")
    else:
        print("  (no entities found)")

    if result.brands:
        print(f"\n  🏷  Brands    : {', '.join(result.brands)}")
    if result.products:
        print(f"  📦 Products   : {', '.join(result.products)}")
    if result.model_numbers:
        print(f"  🔢 Models     : {', '.join(result.model_numbers)}")
    if result.money:
        print(f"  💰 Prices     : {', '.join(result.money)}")
    if result.aspects:
        print(f"\n  🔍 Aspects mentioned:")
        for aspect, keywords in result.aspects.items():
            print(f"     {aspect:<15} → {', '.join(keywords)}")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_REVIEWS = [
    "I bought the Samsung Galaxy S24 Ultra last month and I'm blown away! "
    "The camera is simply the best I've ever used. The 200MP sensor captures "
    "incredible detail. Paid ₹1,24,999 for it and every rupee is worth it. "
    "Battery lasts all day easily. Much better than my old iPhone 14 Pro.",

    "Bought boAt Airdopes 141 for ₹999 during Amazon sale. Sound quality is "
    "surprisingly good for the price. Bass is decent and the connection to my "
    "OnePlus Nord CE 3 is instant. Battery life of 42 hours total is amazing. "
    "Build quality feels a bit plastic but that's expected at this price point.",

    "Sony WH-1000XM5 headphones are absolutely phenomenal. Noise cancellation "
    "is leagues ahead of Bose QC45. The Bluetooth 5.2 connects instantly. "
    "Bought from Flipkart for ₹26,990 with 2 year warranty. The 30-hour "
    "battery life and foldable design make this perfect for travel.",

    "Very disappointed with this Xiaomi Redmi Note 13 Pro. The display looks "
    "good but performance lags after a few weeks. Heating issue is real — "
    "even during basic tasks. For ₹24,999 I expected much better. "
    "My previous Realme GT Neo 5 was far superior in terms of speed and build.",

    "The Apple AirPods Pro 2nd Gen with USB-C are incredible. The Active Noise "
    "Cancellation has improved massively over Gen 1. Transparency mode sounds "
    "completely natural. Connecting to my MacBook Pro M3 and iPhone 15 Pro Max "
    "is seamless. At ₹24,900 it's expensive but worth every paisa.",
]

if __name__ == "__main__":
    # ── Initialise pipeline ───────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  INITIALISING NER PIPELINE")
    print("="*62)
    ner = ProductNERPipeline(brands=KNOWN_BRANDS)

    # ── Process individual reviews ────────────────────────────────────────────
    print("\n\nPROCESSING INDIVIDUAL REVIEWS")
    results = []
    for review in SAMPLE_REVIEWS:
        result = ner.extract(review)
        print_ner_result(result)
        results.append(result)

    # ── Batch DataFrame processing ────────────────────────────────────────────
    print("\nBATCH DATAFRAME PROCESSING")
    print("─" * 62)
    import numpy as np
    np.random.seed(42)
    n = len(SAMPLE_REVIEWS)
    df = pd.DataFrame({
        "review_id":      range(1, n + 1),
        "review_text":    SAMPLE_REVIEWS,
        "star_rating":    [5, 4, 5, 2, 5],
        "vader_compound": [0.92, 0.75, 0.88, -0.65, 0.95],
    })
    df_processed = ner.process_dataframe(df, text_col="review_text")
    show_cols = ["review_id","top_brand","brands","aspects","entity_count"]
    print("\n  Extracted DataFrame columns:")
    print(df_processed[show_cols].to_string(index=False))

    # ── Brand sentiment analysis ──────────────────────────────────────────────
    print("\n\nBRAND SENTIMENT ANALYSIS")
    print("─" * 62)
    brand_sent = brand_sentiment_analysis(
        df_processed, brand_col="brands", sentiment_col="vader_compound"
    )
    if not brand_sent.empty:
        print(brand_sent.to_string())

    # ── Entity co-occurrence ──────────────────────────────────────────────────
    print("\n\nBRAND CO-OCCURRENCE (mentioned in same review)")
    print("─" * 62)
    cooc = entity_cooccurrence(df_processed, brand_col="brands")
    if not cooc.empty:
        print(cooc.to_string(index=False))
    else:
        print("  No co-occurrences found in this sample.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n\nGENERATING PLOTS")
    print("─" * 62)
    plot_entity_distribution(results, "entity_distribution.png")
    plot_top_brands(results, top_n=10, save_path="top_brands.png")
    plot_aspect_frequency(results, "aspect_frequency.png")
    plot_brand_sentiment(brand_sent, save_path="brand_sentiment.png")

    # ── HTML render ───────────────────────────────────────────────────────────
    ner.render_html(SAMPLE_REVIEWS[0], save_path="ner_render.html")
    print("  Saved → ner_render.html  (open in browser to see highlighted entities)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_entities = sum(r.entity_count for r in results)
    all_brands = [b for r in results for b in r.brands]
    print(f"\n{'='*62}")
    print(f"  SUMMARY")
    print(f"  Reviews processed : {len(results)}")
    print(f"  Total entities    : {total_entities}")
    print(f"  Unique brands     : {len(set(b.lower() for b in all_brands))}")
    print(f"  Brands found      : {', '.join(sorted(set(all_brands)))}")
    print(f"{'='*62}")
    print("\n  Usage in your project:")
    print("    from ner_pipeline import ProductNERPipeline")
    print("    ner = ProductNERPipeline()")
    print("    result = ner.extract(review_text)")
    print("    df_out = ner.process_dataframe(df, text_col='review_text')")