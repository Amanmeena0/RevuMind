"""
SHAP Explainability — Product Review Sentiment Predictions
==========================================================
Uses SHAP (SHapley Additive exPlanations) to explain WHY a model
predicted a particular sentiment for each review.

What SHAP answers:
  "Which words/features pushed this prediction toward positive,
   and which pushed it toward negative?"

SHAP explainers covered:
  1. LinearExplainer   → for Logistic Regression (exact, fast)
  2. TreeExplainer     → for Random Forest / XGBoost (exact, fast)
  3. KernelExplainer   → for any black-box model (approximate, slow)

Plots generated:
  1. Summary plot      → global feature importance across all samples
  2. Bar plot          → mean |SHAP| per feature
  3. Waterfall plot    → single prediction breakdown (why THIS review)
  4. Force plot        → visual push/pull for one prediction
  5. Beeswarm plot     → feature impact distribution
  6. Dependence plot   → how one feature interacts with another
  7. Decision plot     → cumulative SHAP path for multiple reviews
  8. Text SHAP         → word-level importance highlighted in review text

Usage:
    explainer = SentimentExplainer()
    explainer.fit(df)
    explainer.explain_single("This product is absolutely amazing!")
    explainer.plot_all(save_dir="shap_plots/")
"""

import os
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from revumind.utils.constants import PALETTE, STOP_WORDS, configure_plotting

warnings.filterwarnings("ignore")
np.random.seed(42)

configure_plotting()

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_DATA = {
    "positive": [
        "Absolutely love this product! Amazing quality and fast delivery. Highly recommend to everyone!",
        "Best purchase I have made this year. Outstanding build quality and brilliant performance.",
        "Fantastic product. Exceeded all my expectations. Five stars! Perfect in every way.",
        "Incredible quality. The design is sleek and the performance is outstanding. Worth every rupee.",
        "Superb! The camera quality is phenomenal. Battery lasts all day. Excellent value for money.",
        "Love the premium feel. Audio quality is crystal clear. Absolutely fantastic experience overall.",
        "Perfect product. Works flawlessly from day one. Great packaging too. Very happy with purchase.",
        "Outstanding performance. No lag whatsoever. Brilliant display quality. Highly recommended!",
        "Amazing product. Delivery was super fast. Build quality is excellent. Will definitely buy again.",
        "Wonderful purchase. Everything works perfectly. Great customer service. Love this product!",
    ],
    "neutral": [
        "Decent product for the price. Nothing special but gets the job done okay.",
        "Average quality. Works fine but nothing impressive. Acceptable for daily use.",
        "Okay product. Some good points some bad. Would not strongly recommend.",
        "It is alright. Does what it says. Not great but not terrible either.",
        "Mixed feelings about this. Has some good features but also some drawbacks.",
        "Works as expected. Delivery was on time. Packaging was adequate. Nothing outstanding.",
        "Middle of the road product. Price is fair. Quality is acceptable. Delivery was normal.",
        "Not bad not great. Does the job. Might work for some people not for others.",
    ],
    "negative": [
        "Terrible product. Broke after two days. Complete waste of money. Avoid at all costs!",
        "Awful quality. Nothing like the pictures shown. Very disappointed with this purchase.",
        "Worst purchase ever. Stopped working after one week. Poor build quality. Do not buy.",
        "Horrible experience. Product is completely defective. Returning immediately. Never again.",
        "Complete garbage. The battery drains in one hour. Cheap plastic. Absolutely terrible.",
        "Very poor quality. Paint peeled off in two days. Feels extremely cheap and fragile.",
        "Disappointed with this product. Does not work as described. Waste of money. Bad quality.",
        "Broken on arrival. Packaging was damaged. Customer service unhelpful. Terrible experience.",
        "Not worth it at all. Quality is horrible. Returned this junk immediately. Zero stars.",
        "Cheap and nasty. Falls apart easily. Smells bad. Terrible value. Avoid this product!",
    ],
}


def build_dataset(n_per_class: int = 120) -> pd.DataFrame:
    rows = []
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    for sentiment, texts in REVIEW_DATA.items():
        label = label_map[sentiment]
        for i in range(n_per_class):
            base = texts[i % len(texts)]
            # Add slight variation
            if np.random.random() < 0.25:
                additions = {
                    "positive": [" Great product!", " Highly recommend!", " Love it!"],
                    "neutral": [" Okay overall.", " Not bad.", " Acceptable."],
                    "negative": [" Very bad!", " Terrible!", " Do not buy!"],
                }
                base += np.random.choice(additions[sentiment])
            rows.append(
                {
                    "review_text": base,
                    "sentiment": sentiment,
                    "label": label,
                    "star_rating": np.random.choice(
                        [4, 5]
                        if sentiment == "positive"
                        else [3] if sentiment == "neutral" else [1, 2]
                    ),
                    "helpful_votes": np.random.randint(5, 50 if sentiment == "positive" else 20),
                }
            )

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Dataset: {len(df):,} reviews | {df['label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION — with named features for SHAP
# ══════════════════════════════════════════════════════════════════════════════


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)


class FeatureBuilder:
    """
    Builds a named feature matrix combining TF-IDF and linguistic features.
    Named features are ESSENTIAL for SHAP — they make plots readable.

    Feature groups:
      tfidf_*     : top TF-IDF unigrams and bigrams (most words)
      ling_*      : hand-crafted linguistic quality signals
    """

    def __init__(self, max_tfidf: int = 200):
        self.max_tfidf = max_tfidf
        self.vectorizer = TfidfVectorizer(
            max_features=max_tfidf,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.feat_names = []
        self._fitted = False

    def _linguistic(self, texts: list) -> np.ndarray:
        """10 interpretable features that directly explain helpfulness."""
        feats = []
        for t in texts:
            words = t.split()
            sents = re.split(r"[.!?]+", t)
            unique = set(w.lower() for w in words)
            feats.append(
                [
                    len(words),
                    len(t),
                    np.mean([len(w) for w in words]) if words else 0,
                    len(unique) / max(len(words), 1),
                    t.count("!"),
                    t.count("?"),
                    sum(1 for w in words if w.isupper() and len(w) > 1),
                    int(bool(re.search(r"\d", t))),
                    int(
                        bool(re.search(r"\b(however|but|although|despite|downside)\b", t.lower()))
                    ),
                    len(sents),
                ]
            )
        return np.array(feats, dtype=np.float32)

    LING_NAMES = [
        "ling_word_count",
        "ling_char_count",
        "ling_avg_word_len",
        "ling_vocab_richness",
        "ling_exclamation_count",
        "ling_question_count",
        "ling_caps_word_count",
        "ling_has_numbers",
        "ling_has_concession",
        "ling_sentence_count",
    ]

    def fit_transform(self, texts: list) -> np.ndarray:
        processed = [preprocess(t) for t in texts]

        # TF-IDF
        X_tfidf = self.vectorizer.fit_transform(processed).toarray()
        tfidf_names = [f"tfidf_{n}" for n in self.vectorizer.get_feature_names_out()]

        # Linguistic
        L = self._linguistic(texts)
        L_scaled = self.scaler.fit_transform(L)

        self.feat_names = tfidf_names + self.LING_NAMES
        self._fitted = True

        X = np.hstack([X_tfidf, L_scaled])
        print(
            f"  Features: {len(tfidf_names)} TF-IDF + {len(self.LING_NAMES)} linguistic = {X.shape[1]} total"
        )
        return X

    def transform(self, texts: list) -> np.ndarray:
        assert self._fitted
        processed = [preprocess(t) for t in texts]
        X_tfidf = self.vectorizer.transform(processed).toarray()
        L = self._linguistic(texts)
        L_scaled = self.scaler.transform(L)
        return np.hstack([X_tfidf, L_scaled])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODELS
# ══════════════════════════════════════════════════════════════════════════════


def train_logistic_regression(X_tr, y_tr) -> LogisticRegression:
    """
    Logistic Regression — best model for LinearExplainer (exact SHAP values).
    LinearExplainer is O(features) — instant even with 10k features.
    """
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        multi_class="multinomial",
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_random_forest(X_tr, y_tr) -> RandomForestClassifier:
    """
    Random Forest — use TreeExplainer (exact, fast).
    TreeExplainer exploits tree structure → exact SHAP in O(T*L²) per sample.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SHAP EXPLAINERS
# ══════════════════════════════════════════════════════════════════════════════


class SentimentExplainer:
    """
    SHAP explainability wrapper for sentiment classifiers.

    SHAP theory (game theory):
      For each prediction, SHAP computes how much each feature CONTRIBUTED
      compared to the average prediction across all samples.

      shap_value[feature] = E[f(X)] contribution of that feature
                          = change in prediction when feature is included

      Sum of all SHAP values = prediction - baseline (mean prediction)

    Three key SHAP properties:
      1. Efficiency   : SHAP values sum to the prediction - baseline
      2. Symmetry     : equal contributors get equal SHAP values
      3. Dummy         : non-contributing features get SHAP=0
      4. Additivity   : SHAP values combine linearly across features

    Explainer types:
      LinearExplainer  → for linear models (LR, Ridge, SVM)
                         Computes exact SHAP using feature covariances
      TreeExplainer    → for tree ensembles (RF, XGB, LightGBM)
                         Exact SHAP using recursive tree structure
      KernelExplainer  → for ANY model (neural nets, SVMs, etc.)
                         Approximate SHAP — slow but universal
    """

    def __init__(self):
        self.feat_builder = FeatureBuilder(max_tfidf=200)
        self.lr_model = None
        self.rf_model = None
        self.lr_explainer = None
        self.rf_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.texts_test = None
        self.feat_names = None
        self.shap_values_lr = None
        self.shap_values_rf = None

    def fit(
        self, df: pd.DataFrame, text_col: str = "review_text", label_col: str = "label"
    ) -> "SentimentExplainer":

        texts = df[text_col].tolist()
        y = df[label_col].values

        # Split
        texts_tr, texts_te, y_tr, y_te = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )

        # Features
        print("\n  Extracting features…")
        self.X_train = self.feat_builder.fit_transform(texts_tr)
        self.X_test = self.feat_builder.transform(texts_te)
        self.y_test = y_te
        self.texts_test = texts_te
        self.feat_names = self.feat_builder.feat_names

        # Train models
        print("  Training Logistic Regression…")
        self.lr_model = train_logistic_regression(self.X_train, y_tr)
        lr_preds = self.lr_model.predict(self.X_test)
        print(
            f"  LR  → Acc={accuracy_score(y_te, lr_preds):.3f}  "
            f"F1={f1_score(y_te, lr_preds, average='weighted'):.3f}"
        )

        print("  Training Random Forest…")
        self.rf_model = train_random_forest(self.X_train, y_tr)
        rf_preds = self.rf_model.predict(self.X_test)
        print(
            f"  RF  → Acc={accuracy_score(y_te, rf_preds):.3f}  "
            f"F1={f1_score(y_te, rf_preds, average='weighted'):.3f}"
        )

        if not SHAP_OK:
            print("  SHAP not installed — skipping explainer creation.")
            return self

        # Build SHAP explainers
        print("\n  Building SHAP explainers…")

        # LinearExplainer for LR
        self.lr_explainer = shap.LinearExplainer(
            self.lr_model,
            self.X_train,
            feature_names=self.feat_names,
        )

        # TreeExplainer for RF
        self.rf_explainer = shap.TreeExplainer(
            self.rf_model,
            feature_names=self.feat_names,
        )

        # Compute SHAP values on test set
        # shape: (n_samples, n_features, n_classes) for multiclass
        print("  Computing SHAP values for test set…")
        bg = shap.sample(self.X_train, 50)  # background for LinearExplainer

        self.shap_values_lr = self.lr_explainer.shap_values(self.X_test)
        self.shap_values_rf = self.rf_explainer.shap_values(self.X_test)

        print(f"  LR SHAP shape : {np.array(self.shap_values_lr).shape}")
        print(f"  RF SHAP shape : {np.array(self.shap_values_rf).shape}")
        return self

    # ── Single review explanation ─────────────────────────────────────────────
    def explain_single(
        self,
        text: str,
        model: str = "lr",
        class_idx: int = 2,  # 0=negative, 1=neutral, 2=positive
        top_n: int = 15,
    ) -> dict:
        """
        Explain WHY the model gave a specific prediction to one review.
        Returns top features pushing toward and away from the predicted class.
        """
        X = self.feat_builder.transform([text])

        if model == "lr":
            clf = self.lr_model
            exp = self.lr_explainer
        else:
            clf = self.rf_model
            exp = self.rf_explainer

        pred_class = clf.predict(X)[0]
        pred_proba = clf.predict_proba(X)[0]

        if not SHAP_OK:
            return {"prediction": CLASS_NAMES[pred_class], "proba": pred_proba}

        shap_vals = exp.shap_values(X)

        # For multiclass: shap_vals[class_idx] → shape (1, n_features)
        if isinstance(shap_vals, list):
            sv_for_class = shap_vals[class_idx][0]
        else:
            sv_for_class = shap_vals[0, :, class_idx]

        # Pair feature name with SHAP value
        pairs = list(zip(self.feat_names, sv_for_class))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(n, v) for n, v in pairs_sorted if v > 0][:top_n]
        top_negative = [(n, v) for n, v in pairs_sorted if v < 0][:top_n]

        print(f'\n  Review: "{text[:80]}…"')
        print(
            f"  Prediction: {CLASS_NAMES[pred_class].upper()}  "
            f"(prob: "
            + "  ".join(f"{CLASS_NAMES[i]}={p:.3f}" for i, p in enumerate(pred_proba))
            + ")"
        )
        print(f"\n  Top features → {CLASS_NAMES[class_idx]} " f"(SHAP > 0, push prediction UP):")
        for name, val in top_positive[:8]:
            bar = "█" * int(abs(val) * 200)
            clean = name.replace("tfidf_", "").replace("ling_", "[ling] ")
            print(f"    +{val:>7.4f}  {clean:<30}  {bar}")

        print(
            f"\n  Top features → away from {CLASS_NAMES[class_idx]} "
            f"(SHAP < 0, push prediction DOWN):"
        )
        for name, val in top_negative[:8]:
            bar = "█" * int(abs(val) * 200)
            clean = name.replace("tfidf_", "").replace("ling_", "[ling] ")
            print(f"    {val:>8.4f}  {clean:<30}  {bar}")

        return {
            "text": text,
            "prediction": CLASS_NAMES[pred_class],
            "proba": pred_proba,
            "shap_values": sv_for_class,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CUSTOM SHAP PLOTS (no shap library needed for some)
# ══════════════════════════════════════════════════════════════════════════════


def plot_global_importance(
    shap_values: list,
    feat_names: list,
    class_names: list = CLASS_NAMES,
    top_n: int = 20,
    save_path: str = "shap_global_importance.png",
) -> None:
    """
    Global feature importance: mean |SHAP value| per feature per class.

    Interpretation:
      Higher mean |SHAP| = this feature moves predictions MORE on average.
      A high value for "positive" class means this feature strongly affects
      positive vs not-positive predictions.
    """
    # shap_values is a list of (n_samples, n_features) arrays, one per class
    if isinstance(shap_values, list):
        sv_arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
    else:
        sv_arr = shap_values

    fig, axes = plt.subplots(1, len(class_names), figsize=(7 * len(class_names), 8))

    for ax, cls_idx, cls_name in zip(axes, range(len(class_names)), class_names):
        if sv_arr.ndim == 3:
            mean_abs = np.abs(sv_arr[cls_idx]).mean(axis=0)  # (n_features,)
        else:
            mean_abs = np.abs(sv_arr[:, :, cls_idx]).mean(axis=0)

        # Top N features
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]
        top_vals = mean_abs[top_idx]
        top_names = [
            feat_names[i].replace("tfidf_", "").replace("ling_", "[ling] ") for i in top_idx
        ]

        color = PALETTE[cls_idx % len(PALETTE)]
        bars = ax.barh(top_names[::-1], top_vals[::-1], color=color, alpha=0.82, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
        ax.set_title(f"Class: {cls_name.upper()}\nmean |SHAP|", fontweight="bold", fontsize=11)
        ax.set_xlabel("Mean |SHAP value|")

    plt.suptitle(
        "Global Feature Importance (SHAP) — per Sentiment Class",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_shap_summary_custom(
    shap_values: list,
    X_test: np.ndarray,
    feat_names: list,
    class_idx: int = 2,
    top_n: int = 20,
    save_path: str = "shap_summary.png",
) -> None:
    """
    Beeswarm-style summary plot showing SHAP value distribution per feature.

    Each dot = one sample.
    Dot colour = original feature value (red=high, blue=low).
    X position = SHAP value (right=pushes positive, left=pushes negative).

    This tells you not just WHICH features matter, but HOW they affect predictions.
    """
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]  # (n_samples, n_features)
    else:
        sv = shap_values[:, :, class_idx]

    # Top N features by mean |SHAP|
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]
    sv_top = sv[:, top_idx]
    X_top = X_test[:, top_idx]
    names_top = [feat_names[i].replace("tfidf_", "").replace("ling_", "[ling] ") for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 9))

    for row, (feat_sv, feat_x, name) in enumerate(zip(sv_top.T, X_top.T, names_top)):
        # Jitter y positions for beeswarm effect
        y_jitter = row + np.random.uniform(-0.25, 0.25, len(feat_sv))

        # Colour by feature value
        feat_norm = (feat_x - feat_x.min()) / (feat_x.max() - feat_x.min() + 1e-9)
        colors = plt.cm.RdBu_r(feat_norm)

        ax.scatter(feat_sv, y_jitter, c=colors, s=12, alpha=0.6, linewidths=0)

    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(names_top, fontsize=8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(
        f"SHAP Summary — Class: {CLASS_NAMES[class_idx].upper()}\n"
        f"Red=high feature value  |  Blue=low feature value",
        fontweight="bold",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Feature value (normalised)", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_waterfall_single(
    shap_result: dict,
    class_idx: int = 2,
    top_n: int = 12,
    save_path: str = "shap_waterfall.png",
) -> None:
    """
    Waterfall plot for ONE review: shows cumulative SHAP contribution.

    Reading it:
      Start at baseline (mean prediction across all reviews).
      Each bar shows how much one feature moves the prediction up or down.
      End at the final predicted probability.

    This is the most interpretable plot for explaining a SINGLE prediction.
    """
    sv = shap_result["shap_values"]
    names = shap_result.get("feat_names", [f"F{i}" for i in range(len(sv))])

    # Sort by absolute SHAP
    pairs = sorted(zip(names, sv), key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_n]
    rest_sv = sum(v for _, v in pairs[top_n:])

    labels = [n.replace("tfidf_", "").replace("ling_", "[ling] ") for n, _ in top]
    values = [v for _, v in top]
    if rest_sv != 0:
        labels.append(f"… {len(pairs)-top_n} other features")
        values.append(rest_sv)

    # Build cumulative waterfall
    baseline = 0.0
    running = baseline
    lefts, widths, colors = [], [], []
    for v in values:
        if v >= 0:
            lefts.append(running)
            widths.append(v)
            colors.append(PALETTE[0])  # green = positive push
        else:
            lefts.append(running + v)
            widths.append(-v)
            colors.append(PALETTE[1])  # red = negative push
        running += v

    fig, ax = plt.subplots(figsize=(11, max(5, len(values) * 0.55)))
    y_pos = range(len(values))
    ax.barh(
        list(y_pos), widths, left=lefts, color=colors, alpha=0.85, edgecolor="white", height=0.65
    )

    # Value labels
    for i, (l, w, c) in enumerate(zip(lefts, widths, colors)):
        val = w if c == PALETTE[0] else -w
        ax.text(l + w + 0.001, i, f"{val:+.4f}", va="center", fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP value (contribution to prediction)")

    pred = shap_result.get("prediction", "")
    text_short = shap_result.get("text", "")[:60] + "…"
    ax.set_title(
        f'SHAP Waterfall — Why predicted: {pred.upper()}\n"{text_short}"',
        fontweight="bold",
    )
    ax.invert_yaxis()

    # Legend
    pos_patch = mpatches.Patch(color=PALETTE[0], label="Pushes toward positive")
    neg_patch = mpatches.Patch(color=PALETTE[1], label="Pushes toward negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_text_shap(
    text: str,
    shap_result: dict,
    save_path: str = "shap_text.png",
) -> None:
    """
    Word-level SHAP highlighting in the original review text.
    Green words push toward predicted class; red words push away.

    This is the most intuitive explanation for non-technical stakeholders.
    """
    sv = shap_result["shap_values"]
    names = shap_result.get("feat_names", [f"F{i}" for i in range(len(sv))])

    # Build word → SHAP mapping from TF-IDF features
    word_shap = {}
    for feat_name, shap_val in zip(names, sv):
        if feat_name.startswith("tfidf_"):
            word = feat_name.replace("tfidf_", "")
            if " " not in word:  # only unigrams for word highlighting
                word_shap[word] = word_shap.get(word, 0) + shap_val

    if not word_shap:
        print("  No TF-IDF word features found for text SHAP plot.")
        return

    # Tokenise original text preserving spaces
    tokens = re.findall(r"[\w']+|[^\w]", text)
    max_abs = max(abs(v) for v in word_shap.values()) if word_shap else 1

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    x, y = 0.01, 0.7
    line_width = 0
    max_line = 0.95

    for token in tokens:
        word_lower = token.lower().strip(".,!?\"'")
        shap_val = word_shap.get(word_lower, 0)

        # Colour: green if positive SHAP, red if negative, grey if neutral
        intensity = min(abs(shap_val) / (max_abs + 1e-9), 1.0)
        if shap_val > 0.001:
            color = (1 - intensity * 0.7, 1, 1 - intensity * 0.7)  # green
        elif shap_val < -0.001:
            color = (1, 1 - intensity * 0.7, 1 - intensity * 0.7)  # red
        else:
            color = (0.95, 0.95, 0.95)  # light grey

        # Estimate token width
        token_w = len(token) * 0.013 + 0.005

        # Wrap to next line if needed
        if line_width + token_w > max_line:
            x = 0.01
            y -= 0.3
            line_width = 0

        if token.strip():
            ax.text(
                x,
                y,
                token,
                ha="left",
                va="center",
                fontsize=11,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor=color, edgecolor="white", linewidth=0.5
                ),
            )
        else:
            ax.text(x, y, token, ha="left", va="center", fontsize=11, fontfamily="monospace")

        x += token_w
        line_width += token_w

    pred = shap_result.get("prediction", "").upper()
    ax.set_title(
        f"Word-level SHAP — Predicted: {pred}  "
        f"|  Green=pushes toward {pred}  |  Red=pushes away",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_class_comparison(
    shap_values: list,
    feat_names: list,
    top_n: int = 15,
    save_path: str = "shap_class_comparison.png",
) -> None:
    """
    Side-by-side diverging bars showing top features for each class.
    Positive bars = features that push TOWARD this class.
    Negative bars = features that push AWAY from this class.

    Reveals which words are discriminative vs shared across classes.
    """
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(7 * len(CLASS_NAMES), 7))

    for ax, cls_idx, cls_name in zip(axes, range(len(CLASS_NAMES)), CLASS_NAMES):
        if isinstance(shap_values, list):
            sv = shap_values[cls_idx]  # (n_samples, n_features)
        else:
            sv = shap_values[:, :, cls_idx]

        mean_shap = sv.mean(axis=0)  # signed mean — not |abs|
        top_idx = np.argsort(np.abs(mean_shap))[-top_n:][::-1]
        vals = mean_shap[top_idx]
        names_top = [
            feat_names[i].replace("tfidf_", "").replace("ling_", "[ling] ") for i in top_idx
        ]

        colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in vals]
        ax.barh(names_top[::-1], vals[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.axvline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(
            f"Class: {cls_name.upper()}\nMean SHAP (signed)", fontweight="bold", fontsize=11
        )
        ax.set_xlabel("Mean SHAP value")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(
        "SHAP Feature Impact per Class — Positive=pushes toward class | Negative=pushes away",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_decision_path(
    shap_values: list,
    feat_names: list,
    texts: list,
    y_true: np.ndarray,
    class_idx: int = 2,
    n_samples: int = 8,
    save_path: str = "shap_decision.png",
) -> None:
    """
    Decision plot: cumulative SHAP values for multiple reviews.
    Each line = one review; shows the path from baseline to final prediction.
    Lines converging to the right = model is confident.
    Lines crossing = similar final predictions via different features.
    """
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][:n_samples]
    else:
        sv = shap_values[:n_samples, :, class_idx]

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]
    sv_top = sv[:, top_idx]
    top_names = [feat_names[i].replace("tfidf_", "").replace("ling_", "[ling] ") for i in top_idx]

    # Cumulative sum left to right (features sorted by importance)
    cumsum = np.cumsum(sv_top, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    color_map = {0: PALETTE[1], 1: PALETTE[2], 2: PALETTE[0]}

    for i in range(min(n_samples, len(sv))):
        true_lbl = y_true[i] if i < len(y_true) else 0
        color = color_map.get(true_lbl, PALETTE[3])
        label = f"Review {i+1} (true={CLASS_NAMES[true_lbl]})"
        ax.plot(
            range(len(top_idx)),
            cumsum[i],
            color=color,
            alpha=0.7,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=label,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", label="Baseline")
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Cumulative SHAP → '{CLASS_NAMES[class_idx]}'")
    ax.set_title(
        f"SHAP Decision Plot — {n_samples} Reviews\n"
        f"Shows how features accumulate to push toward '{CLASS_NAMES[class_idx]}'",
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  SHAP EXPLAINABILITY — SENTIMENT PREDICTIONS")
    print("=" * 62)

    # ── 1. Build and train ────────────────────────────────────────────────────
    print("\n[1] Building dataset and training models…")
    df = build_dataset(n_per_class=100)
    exp = SentimentExplainer()
    exp.fit(df)

    # ── 2. Explain individual predictions ─────────────────────────────────────
    print("\n[2] Explaining individual predictions…")

    example_reviews = [
        "This product is absolutely amazing! Love the quality and fast delivery!",
        "Terrible product. Broke after two days. Complete waste of money. Avoid!",
        "Decent product. Nothing special but works fine for daily use okay.",
    ]

    single_results = []
    for review in example_reviews:
        result = exp.explain_single(review, model="lr", class_idx=2)
        result["feat_names"] = exp.feat_names
        single_results.append(result)

    # ── 3. Generate all plots ─────────────────────────────────────────────────
    print("\n[3] Generating SHAP plots…")
    os.makedirs("shap_plots", exist_ok=True)

    if SHAP_OK and exp.shap_values_lr is not None:
        # Global importance
        plot_global_importance(
            exp.shap_values_lr,
            exp.feat_names,
            save_path="shap_plots/shap_global_importance.png",
        )
        # Summary beeswarm for positive class
        plot_shap_summary_custom(
            exp.shap_values_lr,
            exp.X_test,
            exp.feat_names,
            class_idx=2,
            top_n=20,
            save_path="shap_plots/shap_summary.png",
        )
        # Class comparison
        plot_class_comparison(
            exp.shap_values_lr,
            exp.feat_names,
            save_path="shap_plots/shap_class_comparison.png",
        )
        # Decision plot
        plot_decision_path(
            exp.shap_values_lr,
            exp.feat_names,
            exp.texts_test,
            exp.y_test,
            class_idx=2,
            save_path="shap_plots/shap_decision.png",
        )

    # Waterfall for best positive example
    plot_waterfall_single(
        single_results[0],
        save_path="shap_plots/shap_waterfall_positive.png",
    )
    plot_waterfall_single(
        single_results[1],
        save_path="shap_plots/shap_waterfall_negative.png",
    )

    # Text SHAP
    plot_text_shap(
        example_reviews[0],
        single_results[0],
        save_path="shap_plots/shap_text_positive.png",
    )
    plot_text_shap(
        example_reviews[1],
        single_results[1],
        save_path="shap_plots/shap_text_negative.png",
    )

    print("\n" + "=" * 62)
    print("  DONE")
    print("  Install SHAP: pip install shap")
    print("  Key interview points:")
    print("  → SHAP values sum to: prediction - baseline")
    print("  → LinearExplainer: exact, fast for LR")
    print("  → TreeExplainer: exact, fast for RF/XGBoost")
    print("  → KernelExplainer: any model, slower (sampling)")
    print("  → Always explain on HELD-OUT test data")
    print("=" * 62 + "\n")
