"""
Class Imbalance Handling + Hyperparameter Tuning
=================================================
Predicting review quality (helpful / neutral / unhelpful) where
real-world data is always imbalanced — most reviews get 0 helpful votes.

Techniques covered:
  IMBALANCE HANDLING
  1.  Visualise the imbalance problem
  2.  Class weights          (built into sklearn, free fix)
  3.  SMOTE                  (synthetic minority oversampling)
  4.  ADASYN                 (adaptive synthetic sampling)
  5.  Random under-sampling  (majority class reduction)
  6.  Combined: SMOTE + Tomek links (hybrid)
  7.  Threshold tuning       (move decision boundary after training)

  HYPERPARAMETER TUNING
  8.  GridSearchCV           (exhaustive grid — best for small grids)
  9.  RandomizedSearchCV     (random sampling — best for large grids)
  10. Cross-validation strategy for imbalanced data (StratifiedKFold)
  11. Choosing the right scoring metric (F1 vs accuracy vs PR-AUC)
  12. Pipeline integration   (prevent data leakage during CV)
  13. Results analysis       (best params, CV curves, confusion matrix)

Key interview insight:
  Accuracy is MISLEADING for imbalanced data.
  A model that always predicts "unhelpful" gets 80% accuracy on an 80/20
  split. Always use F1 (macro or weighted), PR-AUC, or balanced accuracy.

Usage:
    handler = ImbalanceHandler()
    X_res, y_res = handler.apply(X_train, y_train, method="smote")
    tuner = HyperparamTuner()
    best_model = tuner.fit(X_res, y_res)
"""

import re, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from revumind.utils.constants import PALETTE, configure_plotting

warnings.filterwarnings("ignore")
np.random.seed(42)

configure_plotting()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC IMBALANCED DATASET
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_TEXTS = {
    0: [   # unhelpful — most common (80% of real data)
        "Good product.", "Nice.", "Works fine.", "Okay.", "Five stars!",
        "Fast delivery.", "As described.", "Happy with purchase.", "Good.",
        "Arrived quickly.", "Works as expected.", "Recommended.", "Fine.",
        "Bad product.", "Waste of money.", "Not good.", "Disappointed.",
        "Returned.", "Broken.", "Poor quality.", "Terrible.", "Avoid.",
    ],
    1: [   # neutral — moderately rare (15%)
        "Good product overall. Works as expected. Delivery was fast. Packaging was nice.",
        "Decent quality for the price. Some minor issues. Customer support was helpful.",
        "Works fine. Not the best not the worst. Does what it says. Acceptable quality.",
        "Okay product. Some good points some bad. Would not strongly recommend.",
        "Average quality. Works fine but nothing impressive. Acceptable overall product.",
        "Mixed feelings about this product. Has pros and cons. Delivery was on time.",
        "It is alright. Does what it says. Nothing special but meets basic requirements.",
    ],
    2: [   # helpful — rarest (5%)
        "I have been using this product for 3 months and here is my detailed assessment. "
        "Battery life is exceptional lasting 2 full days with heavy use. Build quality is "
        "premium metal. Camera captures stunning detail even in low light. However the "
        "software has occasional bugs. Overall worth every rupee at this price point.",

        "Detailed review after 6 weeks of daily use. Sound quality is outstanding with "
        "rich deep bass. Noise cancellation blocks 95 percent of ambient sound. Very "
        "comfortable for long sessions. Charging takes 2 hours and lasts 30 hours. "
        "Only downside is the companion app which feels bloated and slow.",

        "Bought this for my home office. Display colour accuracy is excellent for design "
        "work. 144Hz refresh makes everything silky smooth. Build is sturdy and premium. "
        "Arrived safely packaged. Setup took 10 minutes. Compared to my previous monitor "
        "this is a massive upgrade. The stand is adjustable and the cables are well managed.",

        "After extensive testing here is my comprehensive review. Performance is top notch "
        "with zero lag even during heavy multitasking with 20 browser tabs. Display is "
        "vibrant and colour accurate. Battery optimization has improved massively with the "
        "latest update. Customer service responded within 2 hours to my query. However the "
        "charger is underpowered at only 18W. Would recommend a 45W charger separately.",
    ],
}


def build_imbalanced_dataset(
    n_total:    int   = 600,
    ratios:     tuple = (0.80, 0.15, 0.05),  # unhelpful, neutral, helpful
) -> tuple[list, np.ndarray]:
    """
    Build a realistically imbalanced review dataset.

    Ratios reflect real e-commerce platforms:
      80% of reviews have 0 helpful votes (short, generic)
      15% have a few votes (decent reviews)
       5% are genuinely helpful (detailed, specific)
    """
    texts, labels = [], []
    for label, ratio in enumerate(ratios):
        n = int(n_total * ratio)
        pool = REVIEW_TEXTS[label]
        for i in range(n):
            base = pool[i % len(pool)]
            # Add variation
            if np.random.random() < 0.3:
                add = np.random.choice([
                    " Would recommend.", " Not recommended.",
                    " Good value.", " Poor value.", " Happy.", " Disappointed.",
                ])
                base = base + add
            texts.append(base)
            labels.append(label)

    # Shuffle
    idx = np.random.permutation(len(texts))
    return [texts[i] for i in idx], np.array(labels)[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

STOP_WORDS = {
    'i','me','my','we','our','you','your','it','its','am','is','are','was',
    'were','be','been','have','has','had','do','does','did','a','an','the',
    'and','but','if','or','as','of','at','by','for','with','to','from',
    'in','on','so','too','very','just','not','also','this','that',
}


def extract_features(
    texts_train: list,
    texts_test:  list,
    max_features: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TF-IDF + linguistic features.
    Fit on train, transform both — prevents data leakage.

    IMPORTANT: always fit the vectorizer ONLY on training data.
    If you fit on all data, test set vocabulary leaks into training,
    inflating performance estimates.
    """
    # TF-IDF
    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        min_df=1, sublinear_tf=True,
    )
    X_tr_tfidf = vec.fit_transform(texts_train)
    X_te_tfidf = vec.transform(texts_test)

    # Linguistic features
    def ling(texts):
        feats = []
        for t in texts:
            words = t.split()
            feats.append([
                len(words),
                len(t),
                np.mean([len(w) for w in words]) if words else 0,
                len(set(w.lower() for w in words)) / max(len(words), 1),
                t.count("!"),
                t.count("?"),
                int(bool(re.search(r"\d", t))),
                int(bool(re.search(r"\b(however|but|although|despite)\b", t.lower()))),
                int(bool(re.search(r"\b(vs|compared|versus|better|worse)\b", t.lower()))),
                len(re.split(r"[.!?]+", t)),
            ])
        return np.array(feats, dtype=np.float32)

    L_tr = ling(texts_train)
    L_te = ling(texts_test)

    sc = StandardScaler().fit(L_tr)
    L_tr_s = sc.transform(L_tr)
    L_te_s = sc.transform(L_te)

    import scipy.sparse as sp
    X_tr = sp.hstack([X_tr_tfidf, sp.csr_matrix(L_tr_s)]).toarray()
    X_te = sp.hstack([X_te_tfidf, sp.csr_matrix(L_te_s)]).toarray()

    print(f"  Feature matrix: train={X_tr.shape}  test={X_te.shape}")
    return X_tr, X_te


# ══════════════════════════════════════════════════════════════════════════════
# 3.  IMBALANCE HANDLING METHODS
# ══════════════════════════════════════════════════════════════════════════════

class ImbalanceHandler:
    """
    Collection of techniques for handling class imbalance.

    Which to use?
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ Method              │ Best when                                    │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ class_weight        │ First thing to try. Free, no data change.    │
    │ SMOTE               │ Feature space well-defined. Classic choice.  │
    │ ADASYN              │ Complex boundary. Focuses on hard samples.   │
    │ Under-sampling      │ Majority class has redundant examples.       │
    │ SMOTETomek          │ Best general-purpose hybrid approach.        │
    │ Threshold tuning    │ After training. Fine-grained control.        │
    └─────────────────────┴──────────────────────────────────────────────┘
    """

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> dict:
        """
        Compute balanced class weights for sklearn classifiers.

        Formula: weight[c] = n_samples / (n_classes * count[c])
        → Rare class gets higher weight → penalises misclassifying it more.

        Usage: LogisticRegression(class_weight=weights)
               RandomForestClassifier(class_weight=weights)
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        print(f"  Class weights: {weight_dict}")
        return weight_dict

    @staticmethod
    def apply_smote(
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SMOTE — Synthetic Minority Oversampling Technique.

        How it works:
          For each minority sample:
            1. Find k nearest neighbours in feature space
            2. Pick a random neighbour
            3. Create synthetic sample: original + random_ratio * (neighbour - original)

        This creates new points ALONG THE LINE between real minority samples,
        rather than just duplicating them (which just memorises training data).

        Limitation: works in feature space — noisy features create noisy synthetics.
        Better to apply AFTER TF-IDF projection (dense features).
        """
        if not IMBLEARN_OK:
            print("  SMOTE not available. Using class_weight instead.")
            return X, y

        smote = SMOTE(
            k_neighbors=min(k_neighbors, Counter(y).most_common()[-1][1] - 1),
            sampling_strategy=sampling_strategy,
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X, y)
        print(f"  SMOTE: {Counter(y)} → {Counter(y_res)}")
        return X_res, y_res

    @staticmethod
    def apply_adasyn(
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ADASYN — Adaptive Synthetic Sampling.

        Improvement over SMOTE: generates MORE samples near the decision
        boundary (hard-to-classify minority samples) and FEWER near easy ones.
        Focuses the classifier's attention where it matters most.
        """
        if not IMBLEARN_OK:
            return X, y

        try:
            adasyn = ADASYN(sampling_strategy="auto", random_state=42)
            X_res, y_res = adasyn.fit_resample(X, y)
            print(f"  ADASYN: {Counter(y)} → {Counter(y_res)}")
        except Exception as e:
            print(f"  ADASYN failed ({e}), falling back to SMOTE")
            X_res, y_res = ImbalanceHandler.apply_smote(X, y)
        return X_res, y_res

    @staticmethod
    def apply_random_undersample(
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Random under-sampling: remove majority class samples randomly.

        Pro: Fast, removes noise from majority class.
        Con: Discards potentially useful data.
        Use when: majority class has many redundant or noisy examples.
        """
        if not IMBLEARN_OK:
            return X, y

        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=42
        )
        X_res, y_res = rus.fit_resample(X, y)
        print(f"  Under-sample: {Counter(y)} → {Counter(y_res)}")
        return X_res, y_res

    @staticmethod
    def apply_smote_tomek(
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SMOTETomek — hybrid: SMOTE then Tomek links cleaning.

        1. SMOTE oversamples minority classes
        2. Tomek links: find pairs (majority sample, minority sample) that
           are each other's nearest neighbour → remove the majority one
           This cleans the boundary between classes.

        Best general-purpose technique for most imbalanced problems.
        """
        if not IMBLEARN_OK:
            return X, y

        try:
            st = SMOTETomek(random_state=42)
            X_res, y_res = st.fit_resample(X, y)
            print(f"  SMOTETomek: {Counter(y)} → {Counter(y_res)}")
        except Exception as e:
            print(f"  SMOTETomek failed ({e}), trying SMOTE only")
            X_res, y_res = ImbalanceHandler.apply_smote(X, y)
        return X_res, y_res

    @staticmethod
    def tune_threshold(
        model,
        X_val:    np.ndarray,
        y_val:    np.ndarray,
        target_class: int = 2,
        metric:   str = "f1",
    ) -> float:
        """
        Find the optimal decision threshold for a binary or multiclass model.

        After training, instead of using p >= 0.5 as the decision boundary,
        scan thresholds from 0.1 to 0.9 and find the one that maximises F1
        on the validation set.

        IMPORTANT: tune on validation set, evaluate on held-out test set.
        Never tune threshold on the test set — that's data leakage.

        For multiclass: treat target_class as positive, rest as negative.
        """
        if not hasattr(model, "predict_proba"):
            print("  Model has no predict_proba — threshold tuning skipped.")
            return 0.5

        proba  = model.predict_proba(X_val)
        scores = proba[:, target_class]
        y_bin  = (y_val == target_class).astype(int)

        best_thresh, best_score = 0.5, 0.0
        thresholds = np.arange(0.1, 0.9, 0.05)
        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            if preds.sum() == 0:
                continue
            score = f1_score(y_bin, preds, zero_division=0)
            if score > best_score:
                best_score  = score
                best_thresh = thresh

        print(f"  Optimal threshold (class {target_class}): "
              f"{best_thresh:.2f}  (F1={best_score:.4f})")
        return best_thresh


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GRIDSEARCHCV TUNER
# ══════════════════════════════════════════════════════════════════════════════

class HyperparamTuner:
    """
    GridSearchCV and RandomizedSearchCV for review quality classification.

    Key principles:
      1. Use StratifiedKFold → preserves class ratio in every fold
      2. Score on F1 macro (not accuracy) → fair for imbalanced classes
      3. Wrap in Pipeline → prevents data leakage between CV folds
      4. Always include class_weight in the grid → crucial for imbalance
    """

    def __init__(self, cv_folds: int = 5, n_jobs: int = -1):
        self.cv      = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )
        self.n_jobs  = n_jobs
        self.best_model   = None
        self.best_params  = None
        self.grid_results = {}

    def _f1_macro_scorer(self):
        """Custom scorer — F1 macro weighted by class frequency."""
        return make_scorer(f1_score, average="weighted", zero_division=0)

    # ── Logistic Regression grid ──────────────────────────────────────────────
    def tune_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        mode:    str = "grid",   # "grid" | "random"
    ) -> GridSearchCV:
        """
        Parameters to tune:
          C           : regularisation strength (lower = more regularised)
          penalty     : l1 (sparse, feature selection) vs l2 (smooth)
          class_weight: None vs "balanced" — crucial for imbalance
          solver      : algorithm for optimisation

        Why these matter:
          C too high → overfits to training data, memorises majority class
          C too low  → underfits, treats all classes as noise
          class_weight="balanced" is often the single biggest win for imbalance
        """
        if mode == "grid":
            param_grid = {
                "C":            [0.01, 0.1, 1.0, 5.0, 10.0],
                "class_weight": [None, "balanced"],
                "penalty":      ["l2"],
                "solver":       ["lbfgs"],
                "max_iter":     [500],
            }
        else:
            param_grid = {
                "C":            loguniform(0.001, 100),
                "class_weight": [None, "balanced"],
                "penalty":      ["l2"],
                "solver":       ["lbfgs"],
                "max_iter":     [500],
            }

        base = LogisticRegression(random_state=42, multi_class="auto")

        if mode == "grid":
            search = GridSearchCV(
                base, param_grid, cv=self.cv,
                scoring=self._f1_macro_scorer(),
                n_jobs=self.n_jobs, verbose=0,
                return_train_score=True,
            )
        else:
            search = RandomizedSearchCV(
                base, param_grid, n_iter=20, cv=self.cv,
                scoring=self._f1_macro_scorer(),
                n_jobs=self.n_jobs, verbose=0,
                random_state=42, return_train_score=True,
            )

        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        self.grid_results["LogisticRegression"] = search
        print(f"  LR [{mode}] best F1={search.best_score_:.4f}  "
              f"params={search.best_params_}  ({elapsed:.1f}s)")
        return search

    # ── Random Forest grid ────────────────────────────────────────────────────
    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        mode:    str = "random",
    ) -> RandomizedSearchCV:
        """
        Parameters to tune:
          n_estimators  : number of trees (more = better but slower)
          max_depth     : tree depth (None = full depth, risks overfitting)
          min_samples_leaf: minimum samples per leaf (higher = more generalised)
          max_features  : features per split ("sqrt" is standard)
          class_weight  : "balanced" or "balanced_subsample"
                          balanced_subsample rebalances each bootstrap sample
        """
        if mode == "grid":
            param_grid = {
                "n_estimators":     [100, 200],
                "max_depth":        [None, 10, 20],
                "min_samples_leaf": [1, 5, 10],
                "class_weight":     ["balanced", "balanced_subsample"],
            }
        else:
            param_grid = {
                "n_estimators":     randint(50, 300),
                "max_depth":        [None, 5, 10, 15, 20, 30],
                "min_samples_leaf": randint(1, 20),
                "max_features":     ["sqrt", "log2", 0.3, 0.5],
                "class_weight":     ["balanced", "balanced_subsample"],
            }

        base = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)

        search = RandomizedSearchCV(
            base, param_grid, n_iter=15, cv=self.cv,
            scoring=self._f1_macro_scorer(),
            n_jobs=1, verbose=0, random_state=42,
            return_train_score=True,
        )

        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        self.grid_results["RandomForest"] = search
        print(f"  RF [random] best F1={search.best_score_:.4f}  "
              f"params={search.best_params_}  ({elapsed:.1f}s)")
        return search

    # ── Full summary ──────────────────────────────────────────────────────────
    def compare_all(self) -> pd.DataFrame:
        rows = []
        for name, search in self.grid_results.items():
            rows.append({
                "model":       name,
                "best_cv_f1":  search.best_score_,
                "best_params": str(search.best_params_),
            })
        df = pd.DataFrame(rows).sort_values("best_cv_f1", ascending=False)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    model_name:  str = "Model",
    class_names: list = CLASS_NAMES,
) -> dict:
    """
    Comprehensive evaluation for imbalanced multiclass classification.

    Metrics used:
      accuracy          : misleading for imbalanced data
      balanced_accuracy : macro-average recall — better for imbalance
      f1_weighted       : F1 weighted by class frequency
      f1_macro          : F1 averaged across classes (treats all equally)

    For imbalanced data: report BOTH weighted and macro F1.
    Weighted favours majority class; macro gives equal weight to minority.
    The gap between them tells you how badly the model ignores rare classes.
    """
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc      = accuracy_score(y_test, preds)
    bal_acc  = balanced_accuracy_score(y_test, preds)
    f1_w     = f1_score(y_test, preds, average="weighted",  zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro",     zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"\n  {model_name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy          : {acc:.4f}  ← misleading for imbalanced")
    print(f"  Balanced Accuracy : {bal_acc:.4f}  ← better metric")
    print(f"  F1 weighted       : {f1_w:.4f}")
    print(f"  F1 macro          : {f1_macro:.4f}  ← penalises ignoring minorities")
    print(f"  Per-class F1      : " +
          "  ".join(f"{c}:{v:.3f}" for c, v in zip(class_names, f1_per)))
    print(f"\n{classification_report(y_test, preds, target_names=class_names, zero_division=0)}")

    return {
        "model_name":   model_name,
        "accuracy":     acc,
        "bal_accuracy": bal_acc,
        "f1_weighted":  f1_w,
        "f1_macro":     f1_macro,
        "f1_per_class": f1_per,
        "preds":        preds,
        "proba":        proba,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    y_original:  np.ndarray,
    y_resampled: dict,
    save_path:   str = "class_distribution.png",
) -> None:
    """
    Before/after plots for each imbalance technique.
    Shows how each method changes the class distribution.
    """
    methods = ["original"] + list(y_resampled.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))

    for ax, name in zip(axes, methods):
        y = y_original if name == "original" else y_resampled[name]
        counts = Counter(y)
        labels = [CLASS_NAMES[k] for k in sorted(counts.keys())]
        values = [counts[k] for k in sorted(counts.keys())]
        colors = PALETTE[:len(labels)]
        bars   = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel("Count")
        total = sum(values)
        ax.set_ylim(0, max(values) * 1.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 0.5,
                    f"{val/total:.0%}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    plt.suptitle("Class Distribution — Before and After Resampling",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_gridsearch_heatmap(
    search:    GridSearchCV,
    param_x:   str,
    param_y:   str,
    save_path: str = "gridsearch_heatmap.png",
) -> None:
    """
    Heatmap of CV F1 scores over a 2-param grid.
    Shows which combination of hyperparameters works best.
    """
    results_df = pd.DataFrame(search.cv_results_)
    pivot = results_df.pivot_table(
        values="mean_test_score",
        index=f"param_{param_y}",
        columns=f"param_{param_x}",
        aggfunc="mean",
    )

    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="YlOrRd",
        linewidths=0.4, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "CV F1 (weighted)"},
    )
    ax.set_title(f"GridSearchCV Heatmap — CV F1 by {param_x} vs {param_y}",
                 fontweight="bold")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrices(
    results_list: list[dict],
    y_test:       np.ndarray,
    class_names:  list = CLASS_NAMES,
    save_path:    str = "confusion_matrices.png",
) -> None:
    """Side-by-side normalised confusion matrices for all strategies."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm      = confusion_matrix(y_test, res["preds"], labels=range(len(class_names)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4, ax=ax, annot_kws={"size": 11},
            vmin=0, vmax=1,
        )
        f1 = res["f1_macro"]
        ax.set_title(f"{res['model_name']}\nMacro F1={f1:.3f}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_yticklabels(class_names, rotation=0)

    plt.suptitle("Confusion Matrices — Imbalance Strategies Compared",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_metrics_comparison(
    results_list: list[dict],
    save_path:    str = "metrics_comparison.png",
) -> None:
    """Bar chart comparing all strategies on key metrics."""
    names     = [r["model_name"] for r in results_list]
    metrics   = [
        ("accuracy",     "Accuracy\n(misleading!)"),
        ("bal_accuracy", "Balanced\nAccuracy"),
        ("f1_weighted",  "F1\n(weighted)"),
        ("f1_macro",     "F1\n(macro)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, (metric, label) in zip(axes, metrics):
        vals   = [r[metric] for r in results_list]
        best   = max(vals)
        colors = [PALETTE[0] if v == best else "#B4B2A9" for v in vals]
        bars   = ax.bar(names, vals, color=colors, alpha=0.87, edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals],
                     padding=3, fontsize=8, fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=30)

        # Shade accuracy plot to emphasise it's unreliable
        if metric == "accuracy":
            ax.set_facecolor("#fff0f0")
            ax.text(0.5, 0.05, "⚠ misleading for imbalanced data",
                    transform=ax.transAxes, ha="center", fontsize=7,
                    color="red", alpha=0.7)

    plt.suptitle("Imbalance Handling Strategies — Metric Comparison",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_threshold_curve(
    model,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    target_class: int = 2,
    class_names:  list = CLASS_NAMES,
    save_path:    str = "threshold_curve.png",
) -> None:
    """
    F1 vs threshold curve for the minority class.
    Shows how the decision boundary affects minority class detection.
    """
    if not hasattr(model, "predict_proba"):
        return

    proba   = model.predict_proba(X_val)[:, target_class]
    y_bin   = (y_val == target_class).astype(int)
    threshs = np.arange(0.05, 0.95, 0.025)
    f1s, precs, recs = [], [], []

    for t in threshs:
        preds = (proba >= t).astype(int)
        if preds.sum() == 0:
            f1s.append(0); precs.append(0); recs.append(0)
            continue
        f1s.append(f1_score(y_bin, preds, zero_division=0))
        precs.append(f1_score(y_bin, preds, average="binary",
                               pos_label=1, zero_division=0))
        from sklearn.metrics import recall_score
        recs.append(recall_score(y_bin, preds, zero_division=0))

    best_idx = np.argmax(f1s)
    fig, ax  = plt.subplots(figsize=(9, 4))
    ax.plot(threshs, f1s,   color=PALETTE[0], linewidth=2, label="F1")
    ax.plot(threshs, precs, color=PALETTE[1], linewidth=1.5,
            linestyle="--", label="Precision")
    ax.plot(threshs, recs,  color=PALETTE[2], linewidth=1.5,
            linestyle=":",  label="Recall")
    ax.axvline(threshs[best_idx], color="grey", linewidth=1.5, linestyle="-.",
               label=f"Best threshold={threshs[best_idx]:.2f} (F1={f1s[best_idx]:.3f})")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Default threshold=0.5")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning — Class: '{class_names[target_class]}'",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("\n" + "="*62)
    print("  CLASS IMBALANCE + HYPERPARAMETER TUNING")
    print("="*62)

    # ── 1. Build imbalanced dataset ───────────────────────────────────────────
    print("\n[1] Building imbalanced dataset…")
    texts, y = build_imbalanced_dataset(n_total=600, ratios=(0.80, 0.15, 0.05))
    print(f"  Total: {len(y):,} reviews")
    print(f"  Class counts: {dict(Counter(y))}")
    print(f"  Class ratios: " +
          "  ".join(f"{CLASS_NAMES[k]}={v/len(y):.1%}"
                    for k, v in sorted(Counter(y).items())))

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    print("\n[2] Extracting features…")
    texts_tr, texts_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    texts_tr, texts_val, y_tr, y_val = train_test_split(
        texts_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
    )
    X_tr, X_te = extract_features(texts_tr, texts_te)
    X_val, _   = extract_features(texts_val, texts_te)

    # ── 3. Apply imbalance techniques ─────────────────────────────────────────
    print("\n[3] Applying imbalance handling techniques…")
    handler  = ImbalanceHandler()
    weights  = handler.compute_class_weights(y_tr)

    y_resampled = {}

    if IMBLEARN_OK:
        X_smote, y_smote       = handler.apply_smote(X_tr, y_tr)
        X_adasyn, y_adasyn     = handler.apply_adasyn(X_tr, y_tr)
        X_under, y_under       = handler.apply_random_undersample(X_tr, y_tr)
        X_st, y_st             = handler.apply_smote_tomek(X_tr, y_tr)
        y_resampled = {
            "smote": y_smote, "adasyn": y_adasyn,
            "undersample": y_under, "smote_tomek": y_st,
        }

    # ── 4. GridSearchCV hyperparameter tuning ─────────────────────────────────
    print("\n[4] Hyperparameter tuning…")
    tuner = HyperparamTuner(cv_folds=5)

    print("\n  Logistic Regression — GridSearchCV:")
    lr_search = tuner.tune_logistic_regression(X_tr, y_tr, mode="grid")

    print("\n  Random Forest — RandomizedSearchCV:")
    rf_search = tuner.tune_random_forest(X_tr, y_tr, mode="random")

    print("\n  Summary:")
    print(tuner.compare_all().to_string(index=False))

    # ── 5. Train and compare strategies ──────────────────────────────────────
    print("\n[5] Evaluating all strategies on test set…")
    all_results = []

    # a) No handling (naive baseline)
    clf_naive = LogisticRegression(C=1, max_iter=500, random_state=42)
    clf_naive.fit(X_tr, y_tr)
    all_results.append(evaluate_model(
        clf_naive, X_te, y_te, "Naive (no handling)"
    ))

    # b) Class weights
    clf_weighted = LogisticRegression(
        C=lr_search.best_params_.get("C", 1.0),
        class_weight="balanced", max_iter=500, random_state=42,
    )
    clf_weighted.fit(X_tr, y_tr)
    all_results.append(evaluate_model(
        clf_weighted, X_te, y_te, "Class weights"
    ))

    # c) Best GridSearch model
    all_results.append(evaluate_model(
        lr_search.best_estimator_, X_te, y_te, "GridSearch LR"
    ))

    # d) Best RF model
    all_results.append(evaluate_model(
        rf_search.best_estimator_, X_te, y_te, "RandomSearch RF"
    ))

    # e) SMOTE + tuned LR
    if IMBLEARN_OK:
        clf_smote = LogisticRegression(
            C=lr_search.best_params_.get("C", 1.0),
            class_weight="balanced", max_iter=500, random_state=42,
        )
        clf_smote.fit(X_smote, y_smote)
        all_results.append(evaluate_model(
            clf_smote, X_te, y_te, "SMOTE + tuned LR"
        ))

    # ── 6. Threshold tuning ───────────────────────────────────────────────────
    print("\n[6] Threshold tuning for minority class…")
    best_thresh = handler.tune_threshold(
        clf_weighted, X_val, y_val, target_class=2
    )

    # ── 7. Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  FINAL RESULTS SUMMARY")
    print("="*62)
    print(f"\n  {'Strategy':<25} {'Accuracy':>10} {'Bal.Acc':>10} "
          f"{'F1-wt':>10} {'F1-mac':>10}")
    print("  " + "─"*60)
    for r in all_results:
        print(f"  {r['model_name']:<25} {r['accuracy']:>10.4f} "
              f"{r['bal_accuracy']:>10.4f} {r['f1_weighted']:>10.4f} "
              f"{r['f1_macro']:>10.4f}")

    best = max(all_results, key=lambda r: r["f1_macro"])
    print(f"\n  Best strategy (macro F1): {best['model_name']}")
    print(f"\n  KEY LESSON:")
    print(f"  Accuracy alone: {all_results[0]['accuracy']:.1%} (naive) vs "
          f"{best['accuracy']:.1%} (best)")
    print(f"  Macro F1:       {all_results[0]['f1_macro']:.3f} (naive) vs "
          f"{best['f1_macro']:.3f} (best) ← the real improvement")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n[8] Generating plots…")
    plot_class_distribution(y_tr, y_resampled, "class_distribution.png")
    plot_gridsearch_heatmap(
        lr_search, param_x="C", param_y="class_weight",
        save_path="gridsearch_heatmap.png",
    )
    plot_confusion_matrices(all_results[:4], y_te, save_path="confusion_matrices.png")
    plot_metrics_comparison(all_results, save_path="metrics_comparison.png")
    plot_threshold_curve(
        clf_weighted, X_val, y_val, target_class=2,
        save_path="threshold_curve.png",
    )

    print(f"\n{'='*62}")
    print("  DONE. Install imbalanced-learn for SMOTE/ADASYN:")
    print("  pip install imbalanced-learn")
    print(f"{'='*62}\n")