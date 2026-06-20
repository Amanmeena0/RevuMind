"""
Multi-Class Text Classifier — Product Reviews
=============================================
Categorises reviews into 5 classes:
  1. Positive Praise      (5★ — love it, amazing, perfect)
  2. Negative Complaint   (1★ — broken, terrible, refund)
  3. Feature Request      (suggestion, wish, would be better if)
  4. Quality Issue        (defective, poor build, cheap material)
  5. Delivery / Service   (shipping, packaging, customer service)

Pipeline:
  1.  Text cleaning & preprocessing
  2.  TF-IDF vectorisation  (unigrams + bigrams)
  3.  Logistic Regression   (one-vs-rest, class_weight=balanced)
  4.  Compare vs Naive Bayes, Linear SVC, Random Forest
  5.  Hyperparameter tuning (GridSearchCV)
  6.  Evaluation            (confusion matrix, classification report, ROC-AUC)
  7.  Explainability        (top features per class)
  8.  Inference             (predict single review or batch DataFrame)

Usage:
    clf = ReviewClassifier()
    clf.fit(df, text_col="review_text", label_col="category")
    result = clf.predict("The box arrived completely crushed!")
"""

import json
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from revumind.data.synthetic import generate_review_data
from revumind.utils.constants import PALETTE, STOP_WORDS, configure_plotting

warnings.filterwarnings("ignore")

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk

for p in ["stopwords", "wordnet", "punkt"]:
    nltk.download(p, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

LEMMATIZER = WordNetLemmatizer()

# ── Palette ───────────────────────────────────────────────────────────────────
CLASSES = [
    "positive_praise",
    "negative_complaint",
    "feature_request",
    "quality_issue",
    "delivery_service",
]
CLASS_LABELS = {
    "positive_praise": "✅ Positive Praise",
    "negative_complaint": "❌ Negative Complaint",
    "feature_request": "💡 Feature Request",
    "quality_issue": "⚠️  Quality Issue",
    "delivery_service": "🚚 Delivery / Service",
}

configure_plotting()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    tokens = clean(text).split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════


def build_dataset(n: int = 800) -> pd.DataFrame:
    """
    Build a labelled multi-class review dataset using shared generator.
    """
    df = generate_review_data(n)
    # Map aspects to categories if necessary, or just use as is
    df = df.rename(columns={"aspect": "category"})
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CLASSIFIER CLASS
# ══════════════════════════════════════════════════════════════════════════════


class ReviewClassifier:
    """
    Multi-class review classifier.

    Architecture:
      TF-IDF (unigram + bigram) → Logistic Regression (OvR)

    Why Logistic Regression for text?
      - Fast training even on large vocab
      - Gives calibrated probabilities (unlike SVM)
      - Coefficients are directly interpretable per class
      - class_weight='balanced' handles imbalanced categories
      - Regularisation (C param) prevents overfitting on rare words
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.classes_ = None
        self._fitted = False

        # Scikit-learn Pipeline: vectoriser + classifier in one object
        # This ensures the SAME vectoriser is used for train and predict
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,  # log(1 + tf) dampens high-freq terms
                        strip_accents="unicode",
                        analyzer="word",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight="balanced",  # upweight rare categories
                        solver="lbfgs",
                        multi_class="multinomial",
                        random_state=42,
                    ),
                ),
            ]
        )

        # Comparison models (raw, not pipeline — for CV comparison only)
        self.comparison_models = {
            "Logistic Regression": self.pipeline,
            "Naive Bayes": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=2,
                            sublinear_tf=True,
                        ),
                    ),
                    ("clf", MultinomialNB(alpha=0.1)),
                ]
            ),
            "Linear SVC": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=2,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, max_iter=3000, class_weight="balanced")
                        ),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=3000, ngram_range=(1, 1), min_df=2, sublinear_tf=True
                        ),
                    ),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200, class_weight="balanced", random_state=42
                        ),
                    ),
                ]
            ),
        }

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        label_col: str = "category",
        tune: bool = False,
    ) -> "ReviewClassifier":

        print("\n" + "=" * 62)
        print("  TRAINING MULTI-CLASS REVIEW CLASSIFIER")
        print("=" * 62)

        # ── Preprocess ────────────────────────────────────────────────────────
        print("  Preprocessing text…")
        X = df[text_col].fillna("").apply(preprocess).values
        y = df[label_col].values
        self.classes_ = sorted(np.unique(y))

        print(f"  Samples    : {len(X):,}")
        print(f"  Classes    : {self.classes_}")
        print(f"  Class dist :")
        for cls, cnt in Counter(y).most_common():
            bar = "█" * int(cnt / len(y) * 30)
            print(f"    {cls:<25} {bar} {cnt}")

        # ── Train / test split ────────────────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self._X_te = X_te
        self._y_te = y_te

        # ── Compare models via cross-validation ───────────────────────────────
        print("\n  Cross-validation (5-fold) comparison:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        for name, model in self.comparison_models.items():
            scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_weighted", n_jobs=-1)
            cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
            print(f"    {name:<22}  F1={scores.mean():.4f}  ±{scores.std():.4f}")
        self._cv_results = cv_results

        # ── Optional: GridSearch on LR ────────────────────────────────────────
        if tune:
            print("\n  Hyperparameter tuning (GridSearchCV)…")
            param_grid = {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 0.5, 1.0, 5.0],
            }
            grid = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_tr, y_tr)
            self.pipeline = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV F1 : {grid.best_score_:.4f}")
        else:
            # Standard fit on full training set
            self.pipeline.fit(X_tr, y_tr)

        # ── Evaluate on test set ──────────────────────────────────────────────
        self._evaluate(X_te, y_te)
        self._fitted = True
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _evaluate(self, X_te, y_te) -> None:
        y_pred = self.pipeline.predict(X_te)
        y_prob = self.pipeline.predict_proba(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="weighted")

        # ROC-AUC (one-vs-rest for multiclass)
        y_bin = label_binarize(y_te, classes=self.classes_)
        try:
            auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")

        print(f"\n{'─'*62}")
        print(f"  TEST SET RESULTS  ({len(y_te):,} samples)")
        print(f"{'─'*62}")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  F1 (weighted)    : {f1:.4f}")
        print(f"  ROC-AUC (OvR)   : {auc:.4f}")
        print(f"\n  Classification report:")
        print(classification_report(y_te, y_pred, target_names=self.classes_, zero_division=0))

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        y_pred = self.pipeline.predict(self._X_te)
        cm = confusion_matrix(self._y_te, y_pred, labels=self.classes_)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            linewidths=0.4,
            ax=axes[0],
            annot_kws={"size": 10},
        )
        axes[0].set_title("Confusion Matrix (counts)", fontweight="bold")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_xticklabels(self.classes_, rotation=30, ha="right", fontsize=8)
        axes[0].set_yticklabels(self.classes_, rotation=0, fontsize=8)

        # Normalised (row %)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            linewidths=0.4,
            ax=axes[1],
            annot_kws={"size": 10},
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Confusion Matrix (row %)", fontweight="bold")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_xticklabels(self.classes_, rotation=30, ha="right", fontsize=8)
        axes[1].set_yticklabels(self.classes_, rotation=0, fontsize=8)

        plt.suptitle(
            "Multi-Class Classification — Confusion Matrices",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    def plot_model_comparison(self, save_path="model_comparison.png"):
        names = list(self._cv_results.keys())
        means = [self._cv_results[n]["mean"] for n in names]
        stds = [self._cv_results[n]["std"] for n in names]

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [PALETTE[0] if n == "Logistic Regression" else "#B4B2A9" for n in names]
        bars = ax.bar(
            names,
            means,
            yerr=stds,
            color=colors,
            alpha=0.88,
            edgecolor="white",
            capsize=5,
            error_kw={"linewidth": 1.5},
        )
        ax.bar_label(
            bars, labels=[f"{v:.4f}" for v in means], padding=5, fontsize=10, fontweight="bold"
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("CV F1 Score (weighted)")
        ax.set_title("Model Comparison — 5-Fold Cross Validation F1", fontweight="bold")
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    def plot_top_features(self, top_n=15, save_path="top_features.png"):
        """
        Show the top TF-IDF words per class using LR coefficients.
        High coefficient → strongly predicts that class over others.
        """
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feat_names = np.array(tfidf.get_feature_names_out())
        classes = clf.classes_

        n_cls = len(classes)
        fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 6))
        if n_cls == 1:
            axes = [axes]

        for ax, cls, coef, color in zip(axes, classes, clf.coef_, PALETTE):
            top_idx = coef.argsort()[-top_n:]
            words = feat_names[top_idx]
            weights = coef[top_idx]
            bars = ax.barh(words, weights, color=color, alpha=0.82, edgecolor="white")
            ax.set_title(CLASS_LABELS.get(cls, cls), fontweight="bold", fontsize=9)
            ax.set_xlabel("LR coefficient")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=8)

        plt.suptitle(
            "Top Discriminative Words per Category", fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    def plot_class_probabilities(self, reviews: list[str], save_path="class_probas.png"):
        """
        Stacked bar showing P(class) for a set of example reviews.
        Great for demos — shows the model's confidence across all classes.
        """
        processed = [preprocess(r) for r in reviews]
        proba = self.pipeline.predict_proba(processed)
        classes = self.pipeline.classes_

        short_labels = [r[:45] + "…" if len(r) > 45 else r for r in reviews]

        fig, ax = plt.subplots(figsize=(12, max(4, len(reviews) * 0.7)))
        bar_bottoms = np.zeros(len(reviews))
        for i, (cls, color) in enumerate(zip(classes, PALETTE)):
            ax.barh(
                short_labels,
                proba[:, i],
                left=bar_bottoms,
                color=color,
                alpha=0.85,
                edgecolor="white",
                label=CLASS_LABELS.get(cls, cls),
            )
            bar_bottoms += proba[:, i]

        ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Probability")
        ax.set_title("Predicted Class Probabilities per Review", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.01, 0), borderaxespad=0)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    def plot_prediction_confidence(self, save_path="confidence_dist.png"):
        """Histogram of max predicted probability — shows model calibration."""
        y_prob = self.pipeline.predict_proba(self._X_te)
        max_proba = y_prob.max(axis=1)
        y_pred = self.pipeline.predict(self._X_te)
        correct = y_pred == self._y_te

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(
            max_proba[correct],
            bins=25,
            color=PALETTE[0],
            alpha=0.7,
            label="Correct",
            edgecolor="white",
        )
        ax.hist(
            max_proba[~correct],
            bins=25,
            color=PALETTE[1],
            alpha=0.7,
            label="Wrong",
            edgecolor="white",
        )
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Max predicted probability (confidence)")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Confidence Distribution", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        assert self._fitted, "Call fit() first."
        processed = preprocess(text)
        label = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        return {
            "text": text[:120] + ("…" if len(text) > 120 else ""),
            "prediction": label,
            "label": CLASS_LABELS.get(label, label),
            "confidence": float(proba.max()),
            "all_probs": {
                cls: round(float(p), 4) for cls, p in zip(self.pipeline.classes_, proba)
            },
        }

    def predict_batch(self, df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
        processed = df[text_col].fillna("").apply(preprocess).values
        preds = self.pipeline.predict(processed)
        probas = self.pipeline.predict_proba(processed)
        df = df.copy()
        df["predicted_category"] = preds
        df["predicted_label"] = [CLASS_LABELS.get(p, p) for p in preds]
        df["confidence"] = probas.max(axis=1).round(4)
        for cls, col_proba in zip(self.pipeline.classes_, probas.T):
            df[f"p_{cls}"] = col_proba.round(4)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════


def print_prediction(result: dict) -> None:
    sep = "─" * 60
    conf_bar = "█" * int(result["confidence"] * 20)
    print(f"\n{sep}")
    print(f"  📝 \"{result['text']}\"")
    print(f"{sep}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {conf_bar} {result['confidence']:.1%}")
    print(f"\n  All class probabilities:")
    for cls, p in sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 25)
        print(f"    {CLASS_LABELS.get(cls, cls):<28} {bar:<26} {p:.4f}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("Building dataset…")
    df = build_dataset(n_per_class=150)
    print(f"Dataset: {len(df):,} reviews × {df['category'].nunique()} classes")
    print(df["category"].value_counts().to_string())

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = ReviewClassifier(max_features=10_000, ngram_range=(1, 2), C=1.0)
    clf.fit(df, text_col="review_text", label_col="category", tune=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    clf.plot_confusion_matrix("confusion_matrix.png")
    clf.plot_model_comparison("model_comparison.png")
    clf.plot_top_features(top_n=12, save_path="top_features.png")
    clf.plot_prediction_confidence("confidence_dist.png")

    # Example reviews for probability chart
    demo_reviews = [
        "Absolutely amazing product! Best purchase ever. Love it!",
        "Broke after 2 days. Terrible quality. Want refund now.",
        "Good product but wish it had wireless charging feature.",
        "Very cheap plastic material. Feels flimsy and fragile.",
        "Package arrived damaged. Delivery took 10 extra days.",
    ]
    clf.plot_class_probabilities(demo_reviews, "class_probas.png")

    # ── Single predictions ────────────────────────────────────────────────────
    print("\n\nPREDICTION EXAMPLES")
    test_reviews = [
        "Absolutely amazing product! Best purchase I have made this year!",
        "Complete garbage. Stopped working after 3 days. Avoid at all costs!",
        "Nice product but I wish it had a longer cable. Please add this feature.",
        "Very cheap plastic. Feels brittle and scratched on first day of use.",
        "Box arrived crushed. Item was damaged inside. Delivery was very bad.",
        "Outstanding quality and brilliant performance. Totally recommend this!",
        "The build quality is terrible. Joints are wobbly and plastic creaks.",
        "Wrong item sent by seller. Customer service took 7 days to reply.",
    ]
    for review in test_reviews:
        result = clf.predict(review)
        print_prediction(result)

    # ── Batch prediction ──────────────────────────────────────────────────────
    print("\n\nBATCH PREDICTION:")
    batch_df = pd.DataFrame({"review_text": test_reviews})
    out_df = clf.predict_batch(batch_df)
    print(out_df[["review_text", "predicted_label", "confidence"]].to_string(index=False))

    # ── TF-IDF vocabulary stats ───────────────────────────────────────────────
    tfidf = clf.pipeline.named_steps["tfidf"]
    print(f"\n\nTF-IDF VECTORISER STATS:")
    print(f"  Vocabulary size : {len(tfidf.vocabulary_):,}")
    print(f"  Ngram range     : {tfidf.ngram_range}")
    print(f"  Sample terms    : {list(tfidf.vocabulary_.keys())[:15]}")

    print("\n\nHOW TO USE:")
    print("  clf = ReviewClassifier()")
    print("  clf.fit(df, text_col='review_text', label_col='category')")
    print("  clf.predict('The box arrived completely crushed!')")
    print("  clf.predict_batch(df)")
