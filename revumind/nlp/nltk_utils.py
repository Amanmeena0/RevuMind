"""
NLTK Pipeline for Job Description Text Preprocessing
=====================================================
Complete pipeline covering:
  1. Text cleaning
  2. Tokenization (word + sentence)
  3. Stopword removal (standard + custom job-specific)
  4. POS tagging
  5. Lemmatization
  6. Keyword extraction (TF-IDF + RAKE + frequency)
  7. Skill / tech stack extraction
  8. n-gram extraction (bigrams, trigrams)
  9. Batch processing a full DataFrame

Usage:
    from nltk_pipeline import JobDescriptionPipeline
    pipeline = JobDescriptionPipeline()
    result = pipeline.process("We are looking for a Python developer...")
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Scikit-learn for TF-IDF keyword extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Download all required NLTK data ───────────────────────────────────────────
NLTK_PACKAGES = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "omw-1.4", "maxent_ne_chunker", "words",
]
for pkg in NLTK_PACKAGES:
    nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize, MWETokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.util import ngrams as nltk_ngrams


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Standard English stopwords + job-description specific filler words
CUSTOM_STOPWORDS = {
    # Generic job posting filler
    "role", "position", "opportunity", "join", "team", "company", "looking",
    "seeking", "candidate", "ideal", "must", "will", "able", "work", "working",
    "experience", "knowledge", "strong", "good", "excellent", "great", "plus",
    "preferred", "required", "requirement", "skill", "skills", "ability",
    "responsibilities", "responsibility", "qualification", "qualifications",
    "apply", "applicant", "employer", "employee", "employment", "hiring",
    "year", "years", "month", "months", "day", "days",
    "etc", "eg", "ie", "also", "well", "like", "make", "get", "use",
    "include", "including", "within", "across", "along", "based",
    "provide", "support", "ensure", "help", "need", "want",
}

BASE_STOPWORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS

# ── Tech skills taxonomy ───────────────────────────────────────────────────────
TECH_SKILLS = {
    "languages": {
        "python", "java", "javascript", "typescript", "scala", "r", "go",
        "golang", "rust", "c++", "c#", "kotlin", "swift", "ruby", "php",
        "bash", "shell", "perl", "matlab", "julia",
    },
    "ml_ai": {
        "machine learning", "deep learning", "neural network", "nlp",
        "computer vision", "reinforcement learning", "transfer learning",
        "natural language processing", "large language model", "llm",
        "generative ai", "diffusion model", "transformer",
    },
    "frameworks": {
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "spacy",
        "nltk", "opencv", "fastapi", "flask", "django", "spark", "hadoop",
        "kafka", "airflow", "dbt", "react", "angular", "vue", "node",
    },
    "data": {
        "pandas", "numpy", "sql", "nosql", "mongodb", "postgresql", "mysql",
        "sqlite", "redis", "elasticsearch", "tableau", "power bi", "looker",
        "excel", "spark", "hive", "presto", "databricks", "snowflake",
    },
    "cloud_devops": {
        "aws", "gcp", "azure", "docker", "kubernetes", "ci/cd", "git",
        "github", "gitlab", "jenkins", "terraform", "ansible", "linux",
        "unix", "mlops", "devops", "microservices", "rest", "api",
    },
    "soft_skills": {
        "communication", "collaboration", "leadership", "problem solving",
        "analytical", "critical thinking", "teamwork", "agile", "scrum",
        "project management", "mentoring", "presentation",
    },
}

# Flatten for fast lookup
ALL_SKILLS_FLAT = {skill for category in TECH_SKILLS.values() for skill in category}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASS — output of one processed document
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessedDocument:
    original_text:     str
    clean_text:        str
    sentences:         list[str]          = field(default_factory=list)
    tokens_raw:        list[str]          = field(default_factory=list)
    tokens_clean:      list[str]          = field(default_factory=list)
    tokens_lemmatized: list[str]          = field(default_factory=list)
    tokens_stemmed:    list[str]          = field(default_factory=list)
    pos_tags:          list[tuple]        = field(default_factory=list)
    bigrams:           list[tuple]        = field(default_factory=list)
    trigrams:          list[tuple]        = field(default_factory=list)
    freq_keywords:     list[tuple]        = field(default_factory=list)
    rake_keywords:     list[tuple]        = field(default_factory=list)
    skills_found:      dict               = field(default_factory=dict)
    noun_phrases:      list[str]          = field(default_factory=list)
    processed_string:  str               = ""   # rejoined lemmatized tokens


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class JobDescriptionPipeline:
    """
    End-to-end NLTK preprocessing pipeline for job description text.

    Steps:
        1. clean()           → remove HTML, URLs, special chars
        2. sent_tokenize()   → split into sentences
        3. word_tokenize()   → split into words
        4. remove_stopwords()→ filter stopwords
        5. pos_tag()         → part-of-speech tagging
        6. lemmatize()       → root form (run/running → run)
        7. extract_ngrams()  → bigrams & trigrams
        8. extract_keywords()→ frequency + RAKE
        9. extract_skills()  → match against tech taxonomy
       10. extract_noun_phrases() → NP chunking
    """

    def __init__(
        self,
        stopwords:       set   = BASE_STOPWORDS,
        min_token_len:   int   = 2,
        max_token_len:   int   = 40,
        use_stemmer:     bool  = False,   # stemmer = faster but lossy; lemmatizer = better
    ):
        self.stopwords     = stopwords
        self.min_len       = min_token_len
        self.max_len       = max_token_len
        self.use_stemmer   = use_stemmer
        self.lemmatizer    = WordNetLemmatizer()
        self.stemmer       = PorterStemmer()

    # ── Step 1: Text Cleaning ─────────────────────────────────────────────────
    def clean(self, text: str) -> str:
        """
        Normalize raw job description text.
        Order matters — do URL removal before punctuation stripping.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
        text = re.sub(r"<[^>]+>", " ", text)                    # strip HTML tags
        text = re.sub(r"[^\w\s\.\,\+\#]", " ", text)           # keep . , + # (for C#, C++)
        text = re.sub(r"c\+\+", "cpp", text)                    # normalise C++
        text = re.sub(r"c#", "csharp", text)                    # normalise C#
        text = re.sub(r"\b\d+\+?\s*years?\b", " ", text)        # remove "5 years", "3+ years"
        text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
        return text

    # ── Step 2: Sentence Tokenization ────────────────────────────────────────
    def sent_tokenize(self, text: str) -> list[str]:
        """Split text into sentences. Useful for sentence-level analysis."""
        return sent_tokenize(text)

    # ── Step 3: Word Tokenization ─────────────────────────────────────────────
    def word_tokenize(self, text: str) -> list[str]:
        """
        Tokenize into words.
        NLTK's word_tokenize handles contractions, punctuation, and edge cases
        better than a simple split().
        """
        return word_tokenize(text)

    # ── Step 4: Stopword Removal ──────────────────────────────────────────────
    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Remove stopwords AND filter by length.
        Length filter removes single chars and very long tokens (likely noise).
        """
        return [
            t for t in tokens
            if t not in self.stopwords
            and t not in string.punctuation
            and self.min_len <= len(t) <= self.max_len
            and not t.isnumeric()
        ]

    # ── Step 5: POS Tagging ───────────────────────────────────────────────────
    def pos_tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        """
        Assign Part-of-Speech tags to each token.
        Returns list of (token, tag) tuples.
        Tags: NN=noun, VB=verb, JJ=adjective, RB=adverb, NNP=proper noun
        """
        return pos_tag(tokens)

    # ── Step 6: Lemmatization ─────────────────────────────────────────────────
    @staticmethod
    def _wordnet_pos(treebank_tag: str) -> str:
        """Convert Penn Treebank POS tag to WordNet POS for better lemmatization."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN   # default

    def lemmatize(self, pos_tagged: list[tuple[str, str]]) -> list[str]:
        """
        Lemmatize using POS context for accuracy.
        "running" + VBG → "run"   (verb)
        "better"  + JJR → "good"  (adjective)
        "studies" + NNS → "study" (noun)
        Without POS: lemmatizer defaults to noun and misses verb forms.
        """
        return [
            self.lemmatizer.lemmatize(token, self._wordnet_pos(tag))
            for token, tag in pos_tagged
        ]

    def stem(self, tokens: list[str]) -> list[str]:
        """
        Stemming: faster but cruder than lemmatization.
        'running' → 'run', 'studies' → 'studi' (not a real word)
        Use only when speed matters more than readability.
        """
        return [self.stemmer.stem(t) for t in tokens]

    # ── Step 7: n-gram Extraction ─────────────────────────────────────────────
    def extract_ngrams(
        self,
        tokens: list[str],
        n_values: list[int] = [2, 3],
    ) -> dict[int, list[tuple]]:
        """
        Extract bigrams and trigrams.
        Bigrams like ('machine', 'learning') are more meaningful than
        individual tokens for technical job descriptions.
        """
        result = {}
        for n in n_values:
            result[n] = list(nltk_ngrams(tokens, n))
        return result

    # ── Step 8a: Frequency-based Keywords ─────────────────────────────────────
    def extract_freq_keywords(
        self,
        tokens: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, int]]:
        """
        Simple term frequency — most common tokens after preprocessing.
        Fast and interpretable.
        """
        return Counter(tokens).most_common(top_n)

    # ── Step 8b: RAKE Keyword Extraction ─────────────────────────────────────
    def extract_rake_keywords(
        self,
        text: str,
        top_n: int = 15,
        min_words: int = 1,
        max_words: int = 4,
    ) -> list[tuple[str, float]]:
        """
        RAKE (Rapid Automatic Keyword Extraction) — implemented from scratch.

        How RAKE works:
          1. Split text into candidate phrases at stopword boundaries
          2. Score each word by: freq(word) / degree(word)
             where degree = number of co-occurrences in phrases
          3. Score each phrase = sum of word scores
          Phrases that co-occur with many other words get high scores.
        """
        # Split into candidate phrases at stopword/punctuation boundaries
        splitter  = re.compile(r"[\s\-\–\,\.\;\:\!\?\(\)\[\]\{\}\"\']+")
        stop_pat  = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self.stopwords) + r")\b",
            re.IGNORECASE,
        )
        sentences = sent_tokenize(text.lower())
        candidates = []
        for sent in sentences:
            # Split on stopwords
            phrases = stop_pat.split(sent)
            for phrase in phrases:
                phrase = phrase.strip()
                words  = [w for w in splitter.split(phrase)
                          if w and w not in string.punctuation]
                if min_words <= len(words) <= max_words:
                    candidates.append(words)

        if not candidates:
            return []

        # Build word frequency and degree matrices
        word_freq   = Counter()
        word_degree = Counter()
        for phrase in candidates:
            for word in phrase:
                word_freq[word]   += 1
                word_degree[word] += len(phrase) - 1   # co-occurrences

        # Word score = freq / (freq + degree)  (Brin & Page variant)
        word_score = {
            w: word_freq[w] / (word_freq[w] + word_degree[w] + 1e-9)
            for w in word_freq
        }

        # Phrase score = sum of word scores
        phrase_scores = {}
        for phrase in candidates:
            phrase_str   = " ".join(phrase)
            phrase_score = sum(word_score.get(w, 0) for w in phrase)
            if phrase_str not in phrase_scores:
                phrase_scores[phrase_str] = phrase_score

        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:top_n]

    # ── Step 9: Skill Extraction ──────────────────────────────────────────────
    def extract_skills(self, text: str) -> dict[str, list[str]]:
        """
        Match text against the tech skills taxonomy.
        Uses substring matching so 'pytorch framework' still matches 'pytorch'.
        Returns a dict mapping category → list of found skills.
        """
        text_lower = text.lower()
        found = defaultdict(list)
        for category, skills in TECH_SKILLS.items():
            for skill in sorted(skills):   # sorted for consistent output
                # Use word-boundary match to avoid partial matches (e.g. 'r' in 'learning')
                if len(skill) <= 2:
                    # Short skills: exact word match only
                    pattern = r"\b" + re.escape(skill) + r"\b"
                else:
                    pattern = re.escape(skill)
                if re.search(pattern, text_lower):
                    found[category].append(skill)
        return dict(found)

    # ── Step 10: Noun Phrase (NP) Chunking ────────────────────────────────────
    def extract_noun_phrases(self, pos_tagged: list[tuple]) -> list[str]:
        """
        Extract noun phrases using a simple regex grammar on POS tags.
        Grammar: optional adjective(s) followed by noun(s)
        e.g. 'machine learning engineer', 'senior data scientist'
        """
        grammar = r"""
            NP: {<DT>?<JJ.*>*<NN.*>+}   # Det? + Adj* + Noun+
                {<NNP>+}                  # Proper nouns
        """
        cp     = nltk.RegexpParser(grammar)
        tree   = cp.parse(pos_tagged)
        phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join(word for word, tag in subtree.leaves())
                if len(phrase.split()) >= 2:   # only multi-word phrases
                    phrases.append(phrase)
        return list(dict.fromkeys(phrases))   # deduplicate, preserve order

    # ── Master process() method ───────────────────────────────────────────────
    def process(self, text: str) -> ProcessedDocument:
        """
        Run the full pipeline on a single job description.
        Returns a ProcessedDocument with every intermediate step accessible.
        """
        doc = ProcessedDocument(original_text=text, clean_text="")

        # 1. Clean
        doc.clean_text = self.clean(text)

        # 2. Sentence tokenize
        doc.sentences  = self.sent_tokenize(doc.clean_text)

        # 3. Word tokenize
        doc.tokens_raw = self.word_tokenize(doc.clean_text)

        # 4. Remove stopwords
        doc.tokens_clean = self.remove_stopwords(doc.tokens_raw)

        # 5. POS tag
        doc.pos_tags = self.pos_tag(doc.tokens_clean)

        # 6. Lemmatize (with POS context)
        doc.tokens_lemmatized = self.lemmatize(doc.pos_tags)

        # Optional: stem
        if self.use_stemmer:
            doc.tokens_stemmed = self.stem(doc.tokens_clean)

        # 7. n-grams
        grams = self.extract_ngrams(doc.tokens_lemmatized)
        doc.bigrams  = grams.get(2, [])
        doc.trigrams = grams.get(3, [])

        # 8. Keywords
        doc.freq_keywords = self.extract_freq_keywords(doc.tokens_lemmatized)
        doc.rake_keywords = self.extract_rake_keywords(doc.clean_text)

        # 9. Skills
        doc.skills_found = self.extract_skills(doc.clean_text)

        # 10. Noun phrases
        doc.noun_phrases = self.extract_noun_phrases(doc.pos_tags)

        # Processed string (for TF-IDF / ML downstream)
        doc.processed_string = " ".join(doc.tokens_lemmatized)

        return doc


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSOR — run pipeline on a full DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def process_dataframe(
    df:       pd.DataFrame,
    text_col: str = "job_description",
    pipeline: JobDescriptionPipeline = None,
) -> pd.DataFrame:
    """
    Apply the pipeline to every row of a DataFrame.
    Adds columns:
        processed_text   → lemmatized string (for TF-IDF)
        top_keywords     → comma-separated top 10 keywords
        skills_languages → found programming languages
        skills_ml_ai     → found ML/AI skills
        skills_frameworks→ found frameworks
        skills_data      → found data tools
        skills_cloud     → found cloud/devops tools
        token_count      → number of clean tokens
        sentence_count   → number of sentences
        skill_count      → total unique skills found
    """
    if pipeline is None:
        pipeline = JobDescriptionPipeline()

    print(f"Processing {len(df):,} job descriptions…")
    results = []
    for i, text in enumerate(df[text_col].fillna("")):
        doc = pipeline.process(text)
        results.append({
            "processed_text":    doc.processed_string,
            "top_keywords":      ", ".join(kw for kw, _ in doc.freq_keywords[:10]),
            "skills_languages":  ", ".join(doc.skills_found.get("languages", [])),
            "skills_ml_ai":      ", ".join(doc.skills_found.get("ml_ai", [])),
            "skills_frameworks": ", ".join(doc.skills_found.get("frameworks", [])),
            "skills_data":       ", ".join(doc.skills_found.get("data", [])),
            "skills_cloud":      ", ".join(doc.skills_found.get("cloud_devops", [])),
            "soft_skills":       ", ".join(doc.skills_found.get("soft_skills", [])),
            "noun_phrases":      " | ".join(doc.noun_phrases[:8]),
            "token_count":       len(doc.tokens_clean),
            "sentence_count":    len(doc.sentences),
            "skill_count":       sum(len(v) for v in doc.skills_found.values()),
            "bigram_count":      len(doc.bigrams),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} done…")

    result_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, result_df], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF KEYWORD EXTRACTION — across a corpus of job descriptions
# ══════════════════════════════════════════════════════════════════════════════

def extract_tfidf_keywords(
    processed_texts: list[str],
    top_n_per_doc:   int = 10,
    max_features:    int = 3000,
) -> list[list[tuple[str, float]]]:
    """
    TF-IDF keyword extraction across the whole corpus.
    Unlike frequency counting, TF-IDF rewards words that are important
    in a specific document but rare across the corpus.
    Returns a list of (keyword, score) lists, one per document.
    """
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = (1, 2),
        min_df       = 2,
        max_df       = 0.85,
        sublinear_tf = True,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n_per_doc]
        keywords = [(feature_names[idx], round(row[idx], 4))
                    for idx in top_indices if row[idx] > 0]
        results.append(keywords)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_doc_results(doc: ProcessedDocument, max_items: int = 12) -> None:
    """Print every step of the pipeline output in a readable format."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  PIPELINE OUTPUT")
    print(f"{'═'*60}")

    print(f"\n📝 ORIGINAL TEXT ({len(doc.original_text)} chars):")
    print(f"   {doc.original_text[:200]}{'…' if len(doc.original_text)>200 else ''}")

    print(f"\n🧹 CLEANED TEXT:")
    print(f"   {doc.clean_text[:200]}{'…' if len(doc.clean_text)>200 else ''}")

    print(f"\n📄 SENTENCES  ({len(doc.sentences)}):")
    for i, s in enumerate(doc.sentences[:3], 1):
        print(f"   {i}. {s[:100]}")

    print(f"\n🔤 RAW TOKENS  ({len(doc.tokens_raw)}):")
    print(f"   {doc.tokens_raw[:20]}")

    print(f"\n✂️  CLEAN TOKENS (after stopword removal)  ({len(doc.tokens_clean)}):")
    print(f"   {doc.tokens_clean[:20]}")

    print(f"\n🏷️  POS TAGS  (first 15):")
    for token, tag in doc.pos_tags[:15]:
        print(f"   {token:<20} → {tag}")

    print(f"\n🌱 LEMMATIZED TOKENS  ({len(doc.tokens_lemmatized)}):")
    print(f"   {doc.tokens_lemmatized[:20]}")

    if doc.tokens_stemmed:
        print(f"\n✂️  STEMMED TOKENS (for comparison):")
        print(f"   {doc.tokens_stemmed[:20]}")

    print(f"\n🔗 TOP BIGRAMS  ({len(doc.bigrams)} total, top {min(max_items,8)}):")
    bigram_counts = Counter(doc.bigrams).most_common(8)
    for bg, count in bigram_counts:
        print(f"   {' '.join(bg):<30}  ×{count}")

    print(f"\n🔗 TOP TRIGRAMS  (top {min(max_items,6)}):")
    for tg, count in Counter(doc.trigrams).most_common(6):
        print(f"   {' '.join(tg):<40}  ×{count}")

    print(f"\n📊 FREQUENCY KEYWORDS  (top {min(max_items, len(doc.freq_keywords))}):")
    for kw, freq in doc.freq_keywords[:max_items]:
        bar = "█" * min(freq * 3, 25)
        print(f"   {kw:<25} {bar}  {freq}")

    print(f"\n🔑 RAKE KEYWORDS  (top {min(max_items, len(doc.rake_keywords))}):")
    for phrase, score in doc.rake_keywords[:max_items]:
        print(f"   {phrase:<35}  score: {score:.4f}")

    print(f"\n💡 NOUN PHRASES  ({len(doc.noun_phrases)}):")
    print(f"   {' | '.join(doc.noun_phrases[:10])}")

    print(f"\n🛠️  SKILLS FOUND:")
    if doc.skills_found:
        for category, skills in doc.skills_found.items():
            if skills:
                print(f"   {category:<20} → {', '.join(skills)}")
    else:
        print("   (none matched)")

    total_skills = sum(len(v) for v in doc.skills_found.values())
    print(f"\n   Total skills identified: {total_skills}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_JD_1 = """
Senior Data Scientist — Machine Learning

We are looking for an experienced Senior Data Scientist to join our growing
AI team in Bangalore. The ideal candidate will have 4+ years of experience
building and deploying machine learning models in production environments.

Responsibilities:
- Design and implement end-to-end ML pipelines using Python and TensorFlow
- Develop NLP models for text classification, named entity recognition, and
  sentiment analysis using transformers and BERT-based architectures
- Work with large-scale datasets using PySpark and SQL on AWS infrastructure
- Collaborate with cross-functional teams to translate business requirements
  into data science solutions
- Deploy models using Docker and Kubernetes with CI/CD pipelines on GCP
- Mentor junior data scientists and lead technical discussions

Requirements:
- Strong proficiency in Python, with experience in PyTorch or TensorFlow
- Deep understanding of machine learning algorithms including XGBoost,
  Random Forest, SVM, and neural networks
- Experience with NLP frameworks: Hugging Face, spaCy, or NLTK
- Proficiency in SQL and NoSQL databases (PostgreSQL, MongoDB)
- Familiarity with cloud platforms: AWS, GCP, or Azure
- Excellent communication and analytical skills
- Experience with MLOps tools: MLflow, Kubeflow, or similar

Nice to have: Kafka, Airflow, Tableau, Power BI, Scala
"""

SAMPLE_JD_2 = """
ML Engineer — Computer Vision

Join our computer vision team! We need someone skilled in OpenCV,
PyTorch, and deep learning frameworks to build real-time object detection
systems. You should know Python, C++, and have experience with
transfer learning using ResNet or YOLO architectures.

Requirements: Docker, Kubernetes, REST APIs, Git, Linux, AWS.
Strong problem solving and teamwork skills required.
Experience with Tableau or Power BI is a plus.
"""

if __name__ == "__main__":
    # ── Process a single job description ──────────────────────────────────────
    pipeline = JobDescriptionPipeline()
    doc = pipeline.process(SAMPLE_JD_1)
    print_doc_results(doc)

    # ── Lemmatization vs Stemming comparison ──────────────────────────────────
    print("\nLEMMATIZATION vs STEMMING comparison:")
    print("─" * 50)
    test_words = [
        ("running",   "VBG"), ("studies",  "NNS"), ("better",  "JJR"),
        ("developers","NNS"), ("building", "VBG"), ("analyses","NNS"),
        ("configured","VBD"), ("training", "VBG"), ("deployed","VBD"),
    ]
    lem = WordNetLemmatizer(); stem = PorterStemmer()
    print(f"  {'Original':<15} {'Lemmatized':<15} {'Stemmed':<15}")
    print("  " + "─"*42)
    for word, pos in test_words:
        wn_pos = JobDescriptionPipeline._wordnet_pos(pos)
        lemmed = lem.lemmatize(word, wn_pos)
        stemmed = stem.stem(word)
        print(f"  {word:<15} {lemmed:<15} {stemmed:<15}")

    # ── Batch process multiple job descriptions ────────────────────────────────
    print("\n\nBATCH PROCESSING DEMO:")
    print("─" * 50)
    df_jobs = pd.DataFrame({
        "job_id":          ["J001", "J002"],
        "job_title":       ["Senior Data Scientist", "ML Engineer"],
        "job_description": [SAMPLE_JD_1, SAMPLE_JD_2],
    })
    df_processed = process_dataframe(df_jobs, text_col="job_description", pipeline=pipeline)

    print("\nExtracted columns per job:")
    show_cols = ["job_title", "token_count", "sentence_count",
                 "skill_count", "skills_languages", "skills_ml_ai",
                 "skills_frameworks", "top_keywords"]
    print(df_processed[show_cols].to_string(index=False))

    # ── TF-IDF keywords across corpus ─────────────────────────────────────────
    print("\n\nTF-IDF KEYWORDS (corpus-level):")
    print("─" * 50)
    tfidf_kws = extract_tfidf_keywords(
        df_processed["processed_text"].tolist(), top_n_per_doc=8
    )
    for title, kws in zip(df_jobs["job_title"], tfidf_kws):
        print(f"\n  {title}:")
        for kw, score in kws:
            print(f"    {kw:<30}  tfidf: {score:.4f}")

    print("\n✓ Pipeline complete.")
    print("  Load in your project with:")
    print("    from nltk_pipeline import JobDescriptionPipeline, process_dataframe")
    print("    pipeline = JobDescriptionPipeline()")
    print("    doc = pipeline.process(your_text)")
    print("    df_out = process_dataframe(df, text_col='description')")