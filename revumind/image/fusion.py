"""
Multimodal Feature Fusion — CNN Image + TF-IDF Text
====================================================
Extracts visual features from product images using a CNN (ResNet50 backbone),
combines them with TF-IDF text features from review text, and trains a
fusion classifier to predict review sentiment / product quality.

Pipeline:
  1.  Image feature extraction   (CNN → 2048-dim embedding vector)
  2.  Text feature extraction    (TF-IDF + SVD → 100-dim LSA vector)
  3.  Feature fusion strategies  (early, late, attention-weighted)
  4.  Fusion classifier          (MLP trained on combined features)
  5.  Comparison                 (image-only vs text-only vs fusion)
  6.  Evaluation                 (accuracy, F1, per-modality ablation)
  7.  Visualisation              (UMAP, feature importance, results)

Key insight: Neither modality alone is as good as both together.
Text tells you sentiment. Image tells you visual quality.
Together they catch cases where text says "great!" but image shows damage.

Usage:
    extractor = MultimodalExtractor()
    features  = extractor.extract(image, text)
    model     = FusionClassifier()
    model.fit(X_image, X_text, y)
"""

import re, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
import random
from collections import Counter

warnings.filterwarnings("ignore")

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import label_binarize
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

from revumind.utils.constants import PALETTE, STOP_WORDS, configure_plotting

configure_plotting()

print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_TEMPLATES = {
    "positive": [
        "Absolutely love this product! Amazing quality and fast delivery. Highly recommend!",
        "Best purchase I have made this year. Superb build quality and great performance.",
        "Fantastic product. Exceeded all my expectations. Five stars from me!",
        "Outstanding quality. The design is sleek and performance is brilliant.",
        "Perfect in every way. Great value for money. Will definitely buy again.",
        "Excellent product. Works exactly as described. Very happy with this purchase.",
    ],
    "negative": [
        "Terrible product. Broke after two days. Complete waste of money. Avoid!",
        "Awful quality. Nothing like the pictures. Very disappointed overall.",
        "Worst purchase ever. Stopped working after one week. Do not buy this.",
        "Poor build quality. Cheap material. Feels flimsy and fragile. Bad.",
        "Defective item. Does not work at all. Returning immediately. Horrible.",
        "Disgusting quality. Fake product sent. Not what was advertised. Terrible.",
    ],
    "neutral": [
        "Decent product for the price. Nothing special but gets the job done.",
        "Average quality. Works fine but nothing impressive. Acceptable overall.",
        "Okay product. Some good points some bad. Would not strongly recommend.",
        "It is alright. Does what it says. Not great but not terrible either.",
    ],
}

def make_product_image(quality: str, size: int = 128, seed: int = 0) -> Image.Image:
    """
    Generate synthetic product image.
    good     → clean product on white background
    damaged  → product with visible defects / dark patches
    scratched→ product with scratch marks
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    img  = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.28)
    base = (rng.randint(40, 80), rng.randint(100, 160), rng.randint(180, 220))
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=base, outline=(30,80,140), width=2)
    draw.ellipse([cx-rx//2, cy-ry//2-5, cx+rx//4, cy],
                 fill=tuple(min(255, c+70) for c in base))
    if quality == "damaged":
        for _ in range(rng.randint(4, 7)):
            bx = rng.randint(cx-rx+8, cx+rx-8)
            by = rng.randint(cy-ry+8, cy+ry-8)
            br = rng.randint(5, 14)
            draw.ellipse([bx-br, by-br, bx+br, by+br], fill=(15, 15, 15))
    elif quality == "scratched":
        for _ in range(rng.randint(3, 5)):
            x0, y0 = rng.randint(cx-rx, cx), rng.randint(cy-ry, cy)
            x1, y1 = x0 + rng.randint(25, 55), y0 + rng.randint(20, 45)
            draw.line([(x0,y0),(x1,y1)], fill=(160,160,160), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def build_multimodal_dataset(n_per_class: int = 200) -> pd.DataFrame:
    """
    Build paired (image, text, label) dataset.
    Label = overall product quality (positive/negative/neutral).
    Images and text are correlated but not perfectly — this is realistic.
    """
    rows = []
    seed = 0
    quality_map = {
        "positive": ["good", "good", "scratched"],    # mostly good images
        "negative": ["damaged", "scratched", "damaged"],
        "neutral":  ["good", "scratched", "damaged"],
    }
    np.random.seed(42)

    for label, templates in REVIEW_TEMPLATES.items():
        for i in range(n_per_class):
            # Pick image quality (correlated with label but not 100%)
            img_quality = random.choice(quality_map[label])
            img = make_product_image(img_quality, size=128, seed=seed)

            # Pick review text
            text = random.choice(templates)

            # Add some noise to text
            if random.random() < 0.15:
                # Mismatch: good text with bad image or vice versa
                text = random.choice(REVIEW_TEMPLATES[
                    random.choice(list(REVIEW_TEMPLATES.keys()))
                ])

            rows.append({
                "label":       label,
                "review_text": text,
                "img_quality": img_quality,
                "_image":      img,   # PIL image stored in memory
                "seed":        seed,
            })
            seed += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Dataset: {len(df):,} samples × {df['label'].nunique()} classes")
    print(f"  Label dist: {dict(df['label'].value_counts())}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CNN IMAGE FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class CNNFeatureExtractor:
    """
    Extract a fixed-size embedding vector from each product image
    using a ResNet backbone with the final classification head removed.

    How it works:
      ResNet50: Input → Conv layers → AdaptiveAvgPool → [FC: 2048→1000]
      We remove the FC layer (model.fc = Identity) so the output is
      the 2048-dim pooled feature vector — a rich visual embedding.

    Why ResNet features?
      Even without fine-tuning, ImageNet pretrained features capture:
      - Textures, edges, colors (layers 1–2)
      - Shapes, patterns (layer 3)
      - Object-level semantics (layer 4)
      These generalise well to product images.

    In your real project:
      Use a ResNet50 fine-tuned on your product categories for best results.
      Here we use random weights to avoid internet download.
    """

    def __init__(
        self,
        model_name:  str = "resnet50",
        feature_dim: int = 2048,
        use_pretrained: bool = False,   # False = no download needed for demo
    ):
        self.feature_dim = feature_dim
        self.transform   = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Load backbone and strip the classification head
        if use_pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)

        # Remove FC layer → output is 2048-dim global average pool vector
        model.fc = nn.Identity()
        self.model = model.eval().to(DEVICE)

        # Freeze all parameters (we only use this for feature extraction)
        for p in self.model.parameters():
            p.requires_grad = False

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  CNN extractor: ResNet50 backbone ({total:,} params, frozen)")
        print(f"  Output dim   : {feature_dim}")

    def extract_one(self, image: Image.Image) -> np.ndarray:
        """Extract feature vector for a single PIL image."""
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze().cpu().numpy()   # shape: (2048,)

    def extract_batch(
        self,
        images: list,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract features for a list of PIL images.
        Returns numpy array of shape (N, feature_dim).
        """
        all_features = []
        n = len(images)
        for i in range(0, n, batch_size):
            batch_imgs  = images[i : i + batch_size]
            tensors     = torch.stack([
                self.transform(img) for img in batch_imgs
            ]).to(DEVICE)
            with torch.no_grad():
                feats = self.model(tensors)
            all_features.append(feats.cpu().numpy())
            if verbose and (i // batch_size) % 5 == 0:
                print(f"    CNN: {min(i+batch_size, n)}/{n} images processed…")

        result = np.vstack(all_features)
        print(f"  CNN features shape: {result.shape}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TEXT FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


class TextFeatureExtractor:
    """
    Extract dense text features using TF-IDF + Truncated SVD (LSA).

    Why SVD on top of TF-IDF?
      Raw TF-IDF is sparse (10k+ dimensions).
      SVD compresses it to 100 dense dimensions capturing semantic topics.
      Dense features work much better with neural fusion layers.

    Pipeline:
      text → preprocess → TF-IDF (10k sparse) → SVD (100 dense) → feature vector
    """

    def __init__(
        self,
        max_features:    int = 8000,
        ngram_range:     tuple = (1, 2),
        n_components:    int = 100,    # LSA dimensions
    ):
        self.n_components = n_components
        self.vectorizer   = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            min_df       = 1,
            sublinear_tf = True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit_transform(self, texts: list) -> np.ndarray:
        processed = [preprocess_text(t) for t in texts]
        tfidf     = self.vectorizer.fit_transform(processed)
        lsa       = self.svd.fit_transform(tfidf)
        self._fitted = True
        var_exp = self.svd.explained_variance_ratio_.sum()
        print(f"  TF-IDF vocab   : {len(self.vectorizer.vocabulary_):,}")
        print(f"  LSA components : {self.n_components} "
              f"(explains {var_exp:.1%} variance)")
        print(f"  Text features  : {lsa.shape}")
        return lsa

    def transform(self, texts: list) -> np.ndarray:
        assert self._fitted, "Call fit_transform first."
        processed = [preprocess_text(t) for t in texts]
        tfidf     = self.vectorizer.transform(processed)
        return self.svd.transform(tfidf)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE FUSION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def fuse_early(
    X_image: np.ndarray,
    X_text:  np.ndarray,
) -> np.ndarray:
    """
    Early fusion: concatenate image and text feature vectors.
    Simplest approach — lets the classifier learn cross-modal interactions.
    Shape: (N, image_dim + text_dim)

    Pros: Simple, no extra architecture needed.
    Cons: High-dim input; image features (2048) dominate over text (100).
    Solution: PCA/scaling to balance modality contributions.
    """
    return np.hstack([X_image, X_text])


def fuse_early_scaled(
    X_image: np.ndarray,
    X_text:  np.ndarray,
    scaler:  StandardScaler = None,
) -> tuple:
    """
    Early fusion with per-modality StandardScaling.
    Prevents the 2048-dim image features from swamping the 100-dim text features.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_combined = np.hstack([X_image, X_text])
        X_scaled   = scaler.fit_transform(X_combined)
        return X_scaled, scaler
    else:
        return scaler.transform(np.hstack([X_image, X_text])), scaler


def fuse_late(
    proba_image: np.ndarray,
    proba_text:  np.ndarray,
    w_image:     float = 0.4,
    w_text:      float = 0.6,
) -> np.ndarray:
    """
    Late fusion: weighted average of classifier output probabilities.
    Each modality trains its own model, then outputs are combined.

    w_text > w_image because text is usually more reliable for sentiment.
    Adjust weights based on validation performance.

    Pros: Each model optimises its own modality independently.
    Cons: Can't learn cross-modal interactions (e.g. text-image mismatches).
    """
    assert abs(w_image + w_text - 1.0) < 1e-6, "Weights must sum to 1"
    return w_image * proba_image + w_text * proba_text


class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion: learns HOW MUCH to trust each modality
    dynamically per sample using a small neural network.

    How it works:
      1. Project each modality to same hidden dim
      2. Compute attention scores (softmax over modalities)
      3. Weighted sum of projected features
      4. Final classifier head

    Why attention?
      For a review with a very short generic text ("nice product"),
      the model should learn to rely more on the image.
      For a detailed text review, text should dominate.
      Fixed weights (late fusion) can't adapt per sample — attention can.
    """

    def __init__(
        self,
        image_dim:  int = 2048,
        text_dim:   int = 100,
        hidden_dim: int = 256,
        n_classes:  int = 3,
        dropout:    float = 0.3,
    ):
        super().__init__()

        # Project each modality to same hidden_dim
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention gate: learns scalar weight per modality
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),        # 2 attention scores (image, text)
            nn.Softmax(dim=1),
        )

        # Final classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        x_image: torch.Tensor,   # (B, image_dim)
        x_text:  torch.Tensor,   # (B, text_dim)
    ) -> tuple:
        img_proj  = self.image_proj(x_image)   # (B, hidden_dim)
        txt_proj  = self.text_proj(x_text)     # (B, hidden_dim)

        # Compute attention weights
        combined  = torch.cat([img_proj, txt_proj], dim=1)  # (B, 2*hidden)
        attn_w    = self.attention(combined)                 # (B, 2)

        # Weighted fusion
        fused = (attn_w[:, 0:1] * img_proj +
                 attn_w[:, 1:2] * txt_proj)    # (B, hidden_dim)

        logits = self.classifier(fused)         # (B, n_classes)
        return logits, attn_w


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SKLEARN FUSION CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def train_sklearn_models(
    X_image_tr: np.ndarray, X_text_tr: np.ndarray, y_tr: np.ndarray,
    X_image_te: np.ndarray, X_text_te: np.ndarray, y_te: np.ndarray,
    class_names: list,
) -> dict:
    """
    Compare 5 classification strategies:
      1. Image only (CNN features)
      2. Text only  (TF-IDF/LSA features)
      3. Early fusion (concatenated features)
      4. Early fusion scaled
      5. Late fusion  (weighted probability average)
    """

    results = {}

    # Scale features individually for fair comparison
    img_scaler  = StandardScaler().fit(X_image_tr)
    txt_scaler  = StandardScaler().fit(X_text_tr)
    Xi_tr_s     = img_scaler.transform(X_image_tr)
    Xt_tr_s     = txt_scaler.transform(X_text_tr)
    Xi_te_s     = img_scaler.transform(X_image_te)
    Xt_te_s     = txt_scaler.transform(X_text_te)

    # Fused features
    X_fused_tr, fuse_scaler = fuse_early_scaled(X_image_tr, X_text_tr)
    X_fused_te, _           = fuse_early_scaled(X_image_te, X_text_te, fuse_scaler)

    configs = {
        "Image only (CNN)":      (Xi_tr_s, Xi_te_s),
        "Text only (TF-IDF)":   (Xt_tr_s, Xt_te_s),
        "Early fusion":          (X_fused_tr, X_fused_te),
    }

    clf_template = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=42
    )

    for name, (Xtr, Xte) in configs.items():
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        )
        clf.fit(Xtr, y_tr)
        preds = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        acc   = accuracy_score(y_te, preds)
        f1    = f1_score(y_te, preds, average="weighted")
        results[name] = {
            "clf":  clf, "preds": preds, "proba": proba,
            "acc":  acc, "f1":    f1,
        }
        print(f"  {name:<30} Acc={acc:.4f}  F1={f1:.4f}")

    # Late fusion: image model proba + text model proba
    img_proba  = results["Image only (CNN)"]["proba"]
    txt_proba  = results["Text only (TF-IDF)"]["proba"]
    late_proba = fuse_late(img_proba, txt_proba, w_image=0.35, w_text=0.65)
    late_preds = late_proba.argmax(axis=1)
    acc = accuracy_score(y_te, late_preds)
    f1  = f1_score(y_te, late_preds, average="weighted")
    results["Late fusion (0.35/0.65)"] = {
        "preds": late_preds, "proba": late_proba,
        "acc": acc, "f1": f1,
    }
    print(f"  {'Late fusion (0.35/0.65)':<30} Acc={acc:.4f}  F1={f1:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ATTENTION FUSION TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class FusionDataset(Dataset):
    def __init__(self, X_image, X_text, y):
        self.X_image = torch.FloatTensor(X_image)
        self.X_text  = torch.FloatTensor(X_text)
        self.y       = torch.LongTensor(y)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i):
        return self.X_image[i], self.X_text[i], self.y[i]


def train_attention_fusion(
    X_image_tr, X_text_tr, y_tr,
    X_image_te, X_text_te, y_te,
    n_classes: int = 3,
    epochs:    int = 30,
) -> tuple:
    """Train the attention fusion neural network."""

    ds_tr = FusionDataset(X_image_tr, X_text_tr, y_tr)
    ds_te = FusionDataset(X_image_te, X_text_te, y_te)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=64)

    model = AttentionFusion(
        image_dim  = X_image_tr.shape[1],
        text_dim   = X_text_tr.shape[1],
        hidden_dim = 256,
        n_classes  = n_classes,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0; best_weights = None

    print(f"\n  Training Attention Fusion ({epochs} epochs)…")
    for epoch in range(1, epochs + 1):
        model.train(); tot_loss = tot_corr = tot_n = 0
        for xi, xt, yb in dl_tr:
            xi, xt, yb = xi.to(DEVICE), xt.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(xi, xt)
            loss      = criterion(logits, yb)
            loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(yb)
            tot_corr += (logits.argmax(1) == yb).sum().item()
            tot_n    += len(yb)
        scheduler.step()

        # Val
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for xi, xt, yb in dl_te:
                logits, _ = model(xi.to(DEVICE), xt.to(DEVICE))
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_l.extend(yb.numpy())
        val_acc = accuracy_score(all_l, all_p)
        tr_acc  = tot_corr / tot_n
        tr_loss = tot_loss / tot_n
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            import copy
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  loss={tr_loss:.4f}  "
                  f"train_acc={tr_acc:.3%}  val_acc={val_acc:.3%}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Final eval
    model.eval()
    all_p, all_l, all_attn = [], [], []
    with torch.no_grad():
        for xi, xt, yb in dl_te:
            logits, attn = model(xi.to(DEVICE), xt.to(DEVICE))
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(yb.numpy())
            all_attn.extend(attn.cpu().numpy())

    acc = accuracy_score(all_l, all_p)
    f1  = f1_score(all_l, all_p, average="weighted")
    avg_attn = np.array(all_attn).mean(axis=0)
    print(f"\n  Attention Fusion → Acc={acc:.4f}  F1={f1:.4f}")
    print(f"  Avg attention weights: image={avg_attn[0]:.3f}  text={avg_attn[1]:.3f}")

    return model, acc, f1, history, avg_attn, np.array(all_p), np.array(all_l)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_fusion_comparison(
    results:    dict,
    attn_acc:   float,
    attn_f1:    float,
    save_path:  str = "fusion_comparison.png",
) -> None:
    """Bar chart comparing all fusion strategies."""
    all_results = {**results, "Attention fusion (neural)": {"acc": attn_acc, "f1": attn_f1}}
    names   = list(all_results.keys())
    accs    = [all_results[n]["acc"] for n in names]
    f1s     = [all_results[n]["f1"]  for n in names]

    # Highlight fusion methods
    colors = []
    for n in names:
        if "fusion" in n.lower():   colors.append(PALETTE[0])
        elif "image" in n.lower():  colors.append(PALETTE[1])
        else:                       colors.append(PALETTE[2])

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width/2, accs, width, color=colors, alpha=0.85,
                edgecolor="white", label="Accuracy")
    b2 = ax.bar(x + width/2, f1s,  width, color=colors, alpha=0.5,
                edgecolor="white", label="F1 (weighted)", hatch="//")
    ax.bar_label(b1, labels=[f"{v:.3f}" for v in accs], padding=3, fontsize=8)
    ax.bar_label(b2, labels=[f"{v:.3f}" for v in f1s],  padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fusion Strategy Comparison — Image vs Text vs Combined",
                 fontweight="bold")
    ax.legend(fontsize=9)
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[0], label="Fusion methods"),
        Patch(facecolor=PALETTE[1], label="Image only"),
        Patch(facecolor=PALETTE[2], label="Text only"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Rectangle((0,0),1,1, fc="white", ec="grey", label="Accuracy (solid)"),
        plt.Rectangle((0,0),1,1, fc="white", ec="grey", hatch="//", label="F1 (hatched)"),
    ], fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_attention_weights(
    avg_attn:   np.ndarray,
    attn_history: list,
    save_path:  str = "attention_weights.png",
) -> None:
    """Visualise what the attention fusion model relies on."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Average attention pie
    axes[0].pie(
        avg_attn, labels=["Image (CNN)", "Text (TF-IDF)"],
        colors=[PALETTE[1], PALETTE[2]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[0].set_title("Average Attention Weights\n(how much each modality contributes)",
                       fontweight="bold")

    # Val accuracy over training
    axes[1].plot(attn_history["val_acc"],   color=PALETTE[0], linewidth=2, label="Val Acc")
    axes[1].plot(attn_history["train_acc"], color=PALETTE[1], linewidth=2,
                 linestyle="--", label="Train Acc")
    axes[1].set_title("Attention Fusion Training", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.suptitle("Attention-Based Multimodal Fusion",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrices(
    results:     dict,
    y_te:        np.ndarray,
    class_names: list,
    save_path:   str = "confusion_matrices.png",
) -> None:
    """Side-by-side confusion matrices for Image, Text, Fusion."""
    keys = ["Image only (CNN)", "Text only (TF-IDF)", "Early fusion"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))

    for ax, key in zip(axes, keys):
        if key not in results: continue
        cm = confusion_matrix(y_te, results[key]["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.4, ax=ax, annot_kws={"size": 12})
        acc = results[key]["acc"]
        ax.set_title(f"{key}\nAcc={acc:.3f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_yticklabels(class_names, rotation=0)

    plt.suptitle("Confusion Matrices by Modality",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_feature_space(
    X_image: np.ndarray,
    X_text:  np.ndarray,
    y:       np.ndarray,
    class_names: list,
    save_path: str = "feature_space.png",
) -> None:
    """
    2D PCA projection of image and text feature spaces.
    Shows how well each modality separates the classes.
    """
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, X, title in [
        (axes[0], X_image, "CNN Image Features (PCA 2D)"),
        (axes[1], X_text,  "TF-IDF Text Features (PCA 2D)"),
    ]:
        pca  = PCA(n_components=2, random_state=42)
        X2d  = pca.fit_transform(X)
        for i, cls in enumerate(class_names):
            mask = y == i
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       color=PALETTE[i], alpha=0.6, s=18, label=cls)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)

    plt.suptitle("Feature Space Visualisation (PCA 2D Projection)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_sample_grid(df: pd.DataFrame, save_path: str = "sample_grid.png") -> None:
    """Show sample images + text per class."""
    classes = df["label"].unique()
    n_cols  = 3
    fig, axes = plt.subplots(len(classes), n_cols,
                              figsize=(n_cols * 3.5, len(classes) * 3))

    for row, cls in enumerate(sorted(classes)):
        subset = df[df["label"] == cls].head(n_cols)
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(subset):
                img  = subset.iloc[col]["_image"]
                text = subset.iloc[col]["review_text"]
                ax.imshow(img)
                ax.set_title(text[:45] + "…", fontsize=7, wrap=True)
            if col == 0:
                ax.set_ylabel(cls.upper(), fontsize=10,
                               fontweight="bold", rotation=0,
                               labelpad=60, va="center")
            ax.axis("off")

    plt.suptitle("Multimodal Dataset — Sample (Image + Text) per Class",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — full end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    print("\n" + "="*60)
    print("  MULTIMODAL FUSION PIPELINE")
    print("="*60)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("\n[1] Building multimodal dataset…")
    df = build_multimodal_dataset(n_per_class=150)
    plot_sample_grid(df, "sample_grid.png")

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)
    print(f"  Classes: {class_names}")

    # ── 2. Extract CNN image features ─────────────────────────────────────────
    print("\n[2] Extracting CNN image features…")
    cnn = CNNFeatureExtractor(use_pretrained=False)
    X_image = cnn.extract_batch(df["_image"].tolist(), batch_size=32)

    # ── 3. Extract TF-IDF text features ──────────────────────────────────────
    print("\n[3] Extracting TF-IDF text features…")
    text_ext = TextFeatureExtractor(max_features=5000, n_components=100)
    X_text   = text_ext.fit_transform(df["review_text"].tolist())

    # ── 4. Train / test split ─────────────────────────────────────────────────
    print("\n[4] Splitting data…")
    (Xi_tr, Xi_te, Xt_tr, Xt_te, y_tr, y_te) = train_test_split(
        X_image, X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_tr):,}  Test: {len(y_te):,}")

    # ── 5. sklearn fusion strategies ─────────────────────────────────────────
    print("\n[5] Comparing fusion strategies (sklearn)…")
    print(f"  {'Strategy':<30} Acc         F1")
    print(f"  {'─'*55}")
    results = train_sklearn_models(
        Xi_tr, Xt_tr, y_tr,
        Xi_te, Xt_te, y_te,
        class_names,
    )

    # ── 6. Attention fusion (neural) ─────────────────────────────────────────
    print("\n[6] Training Attention Fusion model…")
    # Scale features before neural training
    img_sc = StandardScaler().fit(Xi_tr)
    txt_sc = StandardScaler().fit(Xt_tr)
    Xi_tr_s = img_sc.transform(Xi_tr); Xi_te_s = img_sc.transform(Xi_te)
    Xt_tr_s = txt_sc.transform(Xt_tr); Xt_te_s = txt_sc.transform(Xt_te)

    attn_model, attn_acc, attn_f1, attn_history, avg_attn, attn_preds, _ = \
        train_attention_fusion(
            Xi_tr_s, Xt_tr_s, y_tr,
            Xi_te_s, Xt_te_s, y_te,
            n_classes=len(class_names), epochs=25,
        )

    # ── 7. Full results summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    all_accs = {k: v["acc"] for k, v in results.items()}
    all_accs["Attention fusion (neural)"] = attn_acc
    for name, acc in sorted(all_accs.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 30)
        print(f"  {name:<32} {bar:<32} {acc:.4f}")

    best_name = max(all_accs, key=all_accs.get)
    print(f"\n  Best strategy: {best_name} ({all_accs[best_name]:.4f})")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n[8] Generating plots…")
    plot_fusion_comparison(results, attn_acc, attn_f1, "fusion_comparison.png")
    plot_attention_weights(avg_attn, attn_history, "attention_weights.png")
    plot_confusion_matrices(results, y_te, class_names, "confusion_matrices.png")
    plot_feature_space(Xi_te, Xt_te, y_te, class_names, "feature_space.png")

    print(f"\n{'='*60}")
    print("  DONE. Your multimodal pipeline is ready.")
    print("  Key takeaway:")
    print("  → Fusion almost always beats either modality alone")
    print("  → Text dominates for sentiment, image for visual defects")
    print("  → Attention fusion adapts per sample — best for mismatches")
    print(f"{'='*60}\n")