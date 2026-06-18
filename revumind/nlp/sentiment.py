"""
Sentiment Analyser — Product Reviews
=====================================
Combines VADER (rule-based) + ML models (TF-IDF + Logistic Regression,
Random Forest, XGBoost) into a final ensemble.

Pipeline:
  1. Text cleaning & preprocessing
  2. VADER sentiment scoring
  3. TF-IDF feature extraction
  4. ML model training (3 classifiers)
  5. Ensemble: VADER + best ML model
  6. Evaluation (classification report, confusion matrix, ROC-AUC)
  7. Inference function for new reviews

Usage:
    analyser = SentimentAnalyser()
    analyser.fit(df, text_col="review_text", label_col="star_rating")
    result = analyser.predict("This product is absolutely amazing!")
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
for pkg in ["vader_lexicon", "stopwords", "punkt", "wordnet",
            "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import scipy.sparse as sp

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
VADER_SIA  = SentimentIntensityAnalyzer()

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "positive": "#5DCAA5",
    "neutral":  "#EF9F27",
    "negative": "#D85A30",
    "blue":     "#378ADD",
    "purple":   "#7F77DD",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>",          " ", text)
    text = re.sub(r"[^\w\s]",          " ", text)
    text = re.sub(r"\d+",              " ", text)
    text = re.sub(r"\s+",              " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """clean → tokenise → remove stopwords → lemmatise → rejoin"""
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LABELLING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def make_sentiment_label(
    df: pd.DataFrame,
    star_col: str = "star_rating",
    mode: str = "binary",          # "binary"  → pos / neg
                                   # "ternary" → pos / neu / neg
) -> pd.Series:
    """
    Convert star rating to sentiment label.

    Binary  : 1-2★ → negative   4-5★ → positive   (3★ dropped)
    Ternary : 1-2★ → negative   3★  → neutral     4-5★ → positive
    """
    stars = pd.to_numeric(df[star_col], errors="coerce")
    if mode == "binary":
        mask = stars != 3
        labels = np.where(stars >= 4, "positive", "negative")
        return pd.Series(labels, index=df.index).where(mask)
    else:
        return pd.cut(
            stars,
            bins=[0, 2, 3, 5],
            labels=["negative", "neutral", "positive"],
            right=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VADER SCORER
# ══════════════════════════════════════════════════════════════════════════════

def vader_score(text: str) -> dict:
    """Return all 4 VADER scores for a raw (uncleaned) text."""
    return VADER_SIA.polarity_scores(text)


def vader_label(compound: float, mode: str = "binary") -> str:
    if mode == "binary":
        return "positive" if compound >= 0.05 else "negative"
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def add_vader_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Append vader_compound, vader_pos, vader_neg, vader_neu, vader_label."""
    scores = df[text_col].fillna("").apply(vader_score)
    df = df.copy()
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_label"]    = df["vader_compound"].apply(vader_label)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ML SENTIMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyser:
    """
    Three-stage sentiment system:
      Stage 1 → VADER (rule-based, no training needed)
      Stage 2 → TF-IDF + ML classifier (trained on labelled reviews)
      Stage 3 → Ensemble: weighted average of VADER + ML probabilities
    """

    def __init__(
        self,
        mode:           str   = "binary",    # "binary" | "ternary"
        tfidf_features: int   = 8000,
        ngram_range:    tuple = (1, 2),
        ensemble_weight: float = 0.4,        # weight of VADER in ensemble (0-1)
    ):
        self.mode            = mode
        self.tfidf_features  = tfidf_features
        self.ngram_range     = ngram_range
        self.ensemble_weight = ensemble_weight   # 1-weight goes to ML model
        self.classes_        = None
        self.best_model_name = None

        # TF-IDF vectoriser (shared across all models)
        self.vectorizer = TfidfVectorizer(
            max_features = tfidf_features,
            ngram_range  = ngram_range,
            min_df       = 2,
            max_df       = 0.95,
            sublinear_tf = True,
        )

        # Three classifiers to compare
        self.models = {
            "Logistic Regression": LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                solver="lbfgs", multi_class="auto",
            ),
            "Linear SVC": CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=20,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
        }
        self.best_model   = None
        self.label_encoder = LabelEncoder()
        self._is_fitted    = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        df:        pd.DataFrame,
        text_col:  str = "review_text",
        label_col: str = "star_rating",   # star_rating OR sentiment_label
    ) -> "SentimentAnalyser":
        print("\n" + "="*60)
        print("  TRAINING SENTIMENT ANALYSER")
        print("="*60)

        # ── Labels ────────────────────────────────────────────────────────────
        if df[label_col].isin([1, 2, 3, 4, 5]).all() or \
           pd.to_numeric(df[label_col], errors="coerce").between(1, 5).all():
            df = df.copy()
            df["_label"] = make_sentiment_label(df, label_col, self.mode)
        else:
            df = df.copy()
            df["_label"] = df[label_col]

        df = df.dropna(subset=["_label", text_col])
        print(f"  Samples         : {len(df):,}")
        print(f"  Label dist      : {dict(df['_label'].value_counts())}")

        # ── Preprocessing ─────────────────────────────────────────────────────
        print("\n  Preprocessing text…")
        df["_processed"] = df[text_col].apply(preprocess)

        # ── VADER features ────────────────────────────────────────────────────
        df = add_vader_features(df, text_col)
        print(f"  VADER accuracy  : "
              f"{(df['vader_label'] == df['_label']).mean():.1%}")

        # ── Train / test split ────────────────────────────────────────────────
        X_text  = df["_processed"].values
        y       = df["_label"].values
        X_vader = df[["vader_compound","vader_pos","vader_neg","vader_neu"]].values

        X_train_t, X_test_t, X_vader_train, X_vader_test, y_train, y_test = \
            train_test_split(X_text, X_vader, y,
                             test_size=0.2, random_state=42, stratify=y)

        # ── TF-IDF ────────────────────────────────────────────────────────────
        print("  Building TF-IDF matrix…")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_t)
        X_test_tfidf  = self.vectorizer.transform(X_test_t)

        # Append VADER features to TF-IDF matrix
        X_train = sp.hstack([X_train_tfidf,
                              sp.csr_matrix(X_vader_train)], format="csr")
        X_test  = sp.hstack([X_test_tfidf,
                              sp.csr_matrix(X_vader_test)],  format="csr")

        self.classes_ = sorted(np.unique(y))
        self.label_encoder.fit(self.classes_)

        # ── Train all models, pick best by CV F1 ─────────────────────────────
        print("\n  Cross-validation (3-fold) on training set:")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_score = -1
        results = {}

        for name, clf in self.models.items():
            cv_scores = cross_val_score(
                clf, X_train, y_train, cv=cv,
                scoring="f1_weighted", n_jobs=-1,
            )
            mean_f1 = cv_scores.mean()
            results[name] = mean_f1
            print(f"    {name:<25}  F1={mean_f1:.4f}  (±{cv_scores.std():.4f})")
            if mean_f1 > best_score:
                best_score      = mean_f1
                self.best_model = clf
                self.best_model_name = name

        print(f"\n  Best model: {self.best_model_name} (F1={best_score:.4f})")

        # ── Final fit on full training set ────────────────────────────────────
        self.best_model.fit(X_train, y_train)

        # ── Store test data for evaluation ────────────────────────────────────
        self._X_test        = X_test
        self._X_test_text   = X_test_t
        self._X_vader_test  = X_vader_test
        self._y_test        = y_test
        self._is_fitted     = True

        # ── Evaluate ──────────────────────────────────────────────────────────
        self._evaluate()
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _evaluate(self) -> None:
        y_pred_ml   = self.best_model.predict(self._X_test)
        y_pred_ens  = self._ensemble_predict_labels(
            self._X_test, self._X_vader_test
        )

        print("\n" + "─"*60)
        print(f"  TEST SET RESULTS  ({len(self._y_test):,} samples)")
        print("─"*60)

        for name, preds in [
            ("VADER alone",     [vader_label(v, self.mode)
                                 for v in self._X_vader_test[:, 0]]),
            (self.best_model_name, y_pred_ml),
            ("Ensemble",        y_pred_ens),
        ]:
            acc = accuracy_score(self._y_test, preds)
            f1  = f1_score(self._y_test, preds, average="weighted")
            print(f"  {name:<28}  Acc={acc:.4f}  F1={f1:.4f}")

        print("\n  Classification report (Ensemble):")
        print(classification_report(self._y_test, y_pred_ens))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    def plot_confusion_matrix(self, save_path: str = "confusion_matrix.png") -> None:
        y_pred = self._ensemble_predict_labels(self._X_test, self._X_vader_test)
        cm     = confusion_matrix(self._y_test, y_pred, labels=self.classes_)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=self.classes_, yticklabels=self.classes_,
            linewidths=0.5, ax=ax, annot_kws={"size": 13},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title(f"Confusion Matrix — Ensemble\n({self.best_model_name} + VADER)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    # ── Score distribution plot ───────────────────────────────────────────────
    def plot_score_distribution(self, df_sample: pd.DataFrame,
                                text_col: str = "review_text",
                                save_path: str = "score_dist.png") -> None:
        df_sample = add_vader_features(df_sample.copy(), text_col)
        processed = df_sample[text_col].apply(preprocess).values
        tfidf     = self.vectorizer.transform(processed)
        vader_arr = df_sample[["vader_compound","vader_pos",
                                "vader_neg","vader_neu"]].values
        X = sp.hstack([tfidf, sp.csr_matrix(vader_arr)], format="csr")
        proba = self.best_model.predict_proba(X)

        classes = self.best_model.classes_
        fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
        if len(classes) == 1:
            axes = [axes]
        for ax, cls, i in zip(axes, classes, range(len(classes))):
            color = PALETTE.get(cls, "#888780")
            ax.hist(proba[:, i], bins=25, color=color, alpha=0.8, edgecolor="white")
            ax.set_title(f'P("{cls}")', fontweight="bold")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Reviews")
        plt.suptitle("ML Model Probability Distributions", fontsize=13,
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    # ── Top features plot ─────────────────────────────────────────────────────
    def plot_top_features(self, top_n: int = 20,
                          save_path: str = "top_features.png") -> None:
        """Show the words with the highest LR coefficients per class."""
        if self.best_model_name != "Logistic Regression":
            print("  Feature plot only available for Logistic Regression.")
            return
        clf        = self.best_model
        feat_names = np.array(self.vectorizer.get_feature_names_out())
        # Vader features appended at the end
        vader_names = np.array(["vader_compound","vader_pos","vader_neg","vader_neu"])
        all_names   = np.concatenate([feat_names, vader_names])

        n_classes = len(clf.classes_)
        coef      = clf.coef_   # shape: (n_classes, n_features)

        fig, axes = plt.subplots(1, n_classes, figsize=(8 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        for ax, cls, c in zip(axes, clf.classes_, coef):
            top_idx   = c.argsort()[-top_n:][::-1]
            bot_idx   = c.argsort()[:top_n]
            idx       = np.concatenate([top_idx, bot_idx])
            vals      = c[idx]
            colors    = [PALETTE["positive"] if v > 0 else PALETTE["negative"]
                         for v in vals]
            ax.barh(all_names[idx], vals, color=colors, alpha=0.85, edgecolor="white")
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_title(f'Top features → "{cls}"', fontweight="bold")
            ax.set_xlabel("LR coefficient")
            ax.invert_yaxis()

        plt.suptitle("Most Important Words per Sentiment Class",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    # ── Internal: ensemble probabilities ─────────────────────────────────────
    def _ensemble_predict_proba(
        self,
        X_combined,
        vader_arr: np.ndarray,
    ) -> np.ndarray:
        """
        Weighted average of ML model probabilities + VADER-derived probabilities.
        VADER compound → soft probabilities via sigmoid-like mapping.
        """
        ml_proba = self.best_model.predict_proba(X_combined)   # (N, n_classes)

        # Map VADER compound to class probabilities
        compounds = vader_arr[:, 0]   # shape (N,)
        n = len(compounds)

        if self.mode == "binary":
            # classes = ["negative", "positive"]
            p_pos = (compounds + 1) / 2          # map [-1,1] → [0,1]
            p_neg = 1 - p_pos
            vader_proba = np.column_stack([p_neg, p_pos])
            # reorder to match self.classes_
            if self.classes_[0] == "positive":
                vader_proba = vader_proba[:, ::-1]
        else:
            # classes = ["negative","neutral","positive"]
            p_pos = np.clip((compounds - 0.05) / 0.95, 0, 1)
            p_neg = np.clip((-compounds - 0.05) / 0.95, 0, 1)
            p_neu = np.maximum(0, 1 - p_pos - p_neg)
            vader_proba = np.column_stack([p_neg, p_neu, p_pos])

        # Weighted average
        w_vader = self.ensemble_weight
        w_ml    = 1 - w_vader
        return w_ml * ml_proba + w_vader * vader_proba

    def _ensemble_predict_labels(self, X_combined, vader_arr) -> np.ndarray:
        proba = self._ensemble_predict_proba(X_combined, vader_arr)
        idx   = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]

    # ── Public: predict on new text ───────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Full pipeline inference on a single new review.
        Returns a dict with all scores and final label.
        """
        assert self._is_fitted, "Call fit() first."

        # VADER
        vs      = vader_score(text)
        v_label = vader_label(vs["compound"], self.mode)

        # ML model
        processed = preprocess(text)
        tfidf     = self.vectorizer.transform([processed])
        vader_arr = np.array([[vs["compound"], vs["pos"],
                               vs["neg"],      vs["neu"]]])
        X_combined = sp.hstack([tfidf, sp.csr_matrix(vader_arr)], format="csr")

        ml_label  = self.best_model.predict(X_combined)[0]
        ml_proba  = self.best_model.predict_proba(X_combined)[0]

        # Ensemble
        ens_proba = self._ensemble_predict_proba(X_combined, vader_arr)[0]
        ens_label = self.classes_[np.argmax(ens_proba)]

        return {
            "text":           text[:120] + ("…" if len(text) > 120 else ""),
            "vader": {
                "compound": round(vs["compound"], 4),
                "positive": round(vs["pos"], 4),
                "negative": round(vs["neg"], 4),
                "label":    v_label,
            },
            "ml_model": {
                "name":   self.best_model_name,
                "label":  ml_label,
                "probas": {cls: round(p, 4)
                           for cls, p in zip(self.best_model.classes_, ml_proba)},
            },
            "ensemble": {
                "label":  ens_label,
                "probas": {cls: round(p, 4)
                           for cls, p in zip(self.classes_, ens_proba)},
            },
        }

    def predict_batch(
        self,
        texts: list[str],
        return_df: bool = True,
    ) -> pd.DataFrame | list[dict]:
        results = [self.predict(t) for t in texts]
        if not return_df:
            return results
        rows = []
        for r in results:
            rows.append({
                "text":            r["text"],
                "vader_label":     r["vader"]["label"],
                "vader_compound":  r["vader"]["compound"],
                "ml_label":        r["ml_model"]["label"],
                "ensemble_label":  r["ensemble"]["label"],
                **{f"p_{cls}": p for cls, p in r["ensemble"]["probas"].items()},
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PRETTY PRINT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_prediction(result: dict) -> None:
    sep = "─" * 58
    bars = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
    print(f"\n{sep}")
    print(f"  📝 \"{result['text']}\"")
    print(sep)

    v = result["vader"]
    print(f"  VADER       → {bars.get(v['label'],'⚪')} {v['label'].upper():<10}"
          f"  compound: {v['compound']:+.4f}")

    m = result["ml_model"]
    p = m["probas"]
    print(f"  {m['name'][:20]:<20}→ {bars.get(m['label'],'⚪')} {m['label'].upper():<10}"
          f"  " + "  ".join(f"{cls}: {prob:.3f}" for cls, prob in p.items()))

    e = result["ensemble"]
    ep = e["probas"]
    print(f"  ENSEMBLE    → {bars.get(e['label'],'⚪')} {e['label'].upper():<10}"
          f"  " + "  ".join(f"{cls}: {prob:.3f}" for cls, prob in ep.items()))
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Build demo dataset ────────────────────────────────────────────────────
    np.random.seed(42)
    N = 600

    positive_reviews = [
        "Absolutely love this product! Works perfectly and looks great.",
        "Best purchase I have made this year. Highly recommend!",
        "Amazing quality. Fast delivery. Very happy with this.",
        "Five stars! Exactly as described. Brilliant product.",
        "Outstanding performance. Worth every rupee. Will buy again.",
        "Fantastic item. Exceeded all my expectations. Love it!",
        "Perfect size, great quality, excellent value for money.",
        "Superb! The build quality is top notch. Very satisfied.",
    ]
    negative_reviews = [
        "Terrible product. Broke after two days. Total waste of money.",
        "Awful quality. Nothing like the pictures. Very disappointed.",
        "Worst purchase ever. Stopped working after one week.",
        "Complete junk. Do not buy this. Returning immediately.",
        "Cheap and nasty. Falls apart easily. Very poor quality.",
        "Absolutely horrible. Does not work at all. Scam product.",
        "Disgusting quality. Battery drains in one hour. Terrible.",
        "Defective item received. Seller refused refund. Avoid!",
    ]
    neutral_reviews = [
        "It is okay. Does what it says. Nothing special.",
        "Average product. Works fine but nothing impressive.",
        "Decent quality for the price. Not great, not bad.",
        "So so. Some good points, some bad. Acceptable overall.",
    ]

    texts, stars = [], []
    for _ in range(N // 2):
        texts.append(np.random.choice(positive_reviews))
        stars.append(np.random.choice([4, 5]))
    for _ in range(N // 4):
        texts.append(np.random.choice(negative_reviews))
        stars.append(np.random.choice([1, 2]))
    for _ in range(N // 4):
        texts.append(np.random.choice(neutral_reviews))
        stars.append(3)

    df = pd.DataFrame({"review_text": texts, "star_rating": stars})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    analyser = SentimentAnalyser(mode="binary", ensemble_weight=0.35)
    analyser.fit(df, text_col="review_text", label_col="star_rating")

    # ── Plots ─────────────────────────────────────────────────────────────────
    analyser.plot_confusion_matrix("confusion_matrix.png")
    analyser.plot_top_features("top_features.png")

    # ── Single predictions ────────────────────────────────────────────────────
    print("\n\nPREDICTION EXAMPLES")
    test_reviews = [
        "This is the best product I have ever bought. Absolutely love it!",
        "Complete garbage. Stopped working after 2 days. Avoid at all costs.",
        "It is okay I guess. Works fine but nothing to write home about.",
        "Amazing build quality and superb performance. Five stars!",
        "Disappointed. Quality is poor and delivery was very late.",
        "Not bad. Average product. Does its job but could be better.",
    ]
    for review in test_reviews:
        result = analyser.predict(review)
        print_prediction(result)

    # ── Batch prediction ──────────────────────────────────────────────────────
    print("\n\nBATCH PREDICTION (DataFrame output):")
    batch_df = analyser.predict_batch(test_reviews)
    print(batch_df[["text","vader_label","ml_label",
                     "ensemble_label","p_positive","p_negative"]].to_string(index=False))

    print("\n\nHOW TO USE IN YOUR PROJECT:")
    print("─" * 50)
    print("  from sentiment_analyser import SentimentAnalyser")
    print("  analyser = SentimentAnalyser(mode='binary')")
    print("  analyser.fit(df, text_col='review_text', label_col='star_rating')")
    print("  result = analyser.predict('This product is amazing!')")
    print("  df_out = analyser.predict_batch(list_of_reviews)")