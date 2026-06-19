"""
Transfer Learning with ResNet50 — Product Defect Detection
==========================================================
Classifies product images as: Good | Defective | Scratched | Broken

Pipeline:
  1.  Dataset setup         (synthetic images + ImageFolder structure)
  2.  Data augmentation     (torchvision transforms for train/val/test)
  3.  ResNet50 loading      (pretrained ImageNet weights)
  4.  Model surgery         (freeze backbone, replace classifier head)
  5.  Training loop         (loss, optimizer, LR scheduler, early stopping)
  6.  Fine-tuning           (unfreeze layers, lower LR, train again)
  7.  Evaluation            (accuracy, F1, confusion matrix, ROC-AUC)
  8.  Grad-CAM              (visualise WHAT the model looks at)
  9.  Inference             (predict single image or batch)
  10. Model saving          (state_dict + full export)

Key concepts explained inline with comments.

Usage:
    model = DefectClassifier()
    model.train_model(train_loader, val_loader, epochs=10)
    result = model.predict("path/to/product.jpg")
"""

import os, warnings, time, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random

warnings.filterwarnings("ignore")

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

# ── Scikit-learn metrics ──────────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
)
from sklearn.preprocessing import label_binarize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {DEVICE}")
print(f"Torchvision: {torchvision.__version__}")

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE   = ["#5DCAA5", "#D85A30", "#EF9F27", "#7F77DD"]
CLASS_EMO = {"good": "✅", "defective": "❌", "scratched": "⚠️ ", "broken": "💥"}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130,
                     "axes.spines.top": False,
                     "axes.spines.right": False})


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def make_product_image(
    label:    str,
    size:     int = 128,
    seed:     int = 0,
) -> Image.Image:
    """
    Generate a synthetic product image for each defect class.
    In a real project these would be your actual product photos.

    Good      → clean blue ellipse product on white background
    Defective → same but with random dark patches (defects)
    Scratched → diagonal scratch lines across the surface
    Broken    → product split into fragments with cracks
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    img  = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.28)

    # Base product body
    base_color = (
        rng.randint(40, 80),
        rng.randint(100, 160),
        rng.randint(180, 220),
    )
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=base_color, outline=(30, 80, 140), width=2,
    )

    # Highlight
    draw.ellipse(
        [cx - rx//2, cy - ry//2 - 5,
         cx + rx//4, cy],
        fill=tuple(min(255, c + 70) for c in base_color),
    )

    if label == "good":
        pass   # clean product — nothing extra

    elif label == "defective":
        # Random dark blotches
        for _ in range(rng.randint(3, 6)):
            bx = rng.randint(cx - rx + 10, cx + rx - 10)
            by = rng.randint(cy - ry + 10, cy + ry - 10)
            br = rng.randint(4, 12)
            draw.ellipse([bx-br, by-br, bx+br, by+br],
                         fill=(20, 20, 20))

    elif label == "scratched":
        # Diagonal scratch lines
        for _ in range(rng.randint(2, 4)):
            x0 = rng.randint(cx - rx, cx)
            y0 = rng.randint(cy - ry, cy)
            x1 = x0 + rng.randint(20, 50)
            y1 = y0 + rng.randint(15, 40)
            draw.line([(x0, y0), (x1, y1)], fill=(180, 180, 180), width=2)
            draw.line([(x0+2, y0), (x1+2, y1)], fill=(220, 220, 220), width=1)

    elif label == "broken":
        # Crack lines radiating from a point
        crack_x = rng.randint(cx - 15, cx + 15)
        crack_y = rng.randint(cy - 15, cy + 15)
        for angle in range(0, 360, rng.randint(45, 90)):
            rad   = np.radians(angle + rng.randint(-15, 15))
            length = rng.randint(20, 40)
            ex    = int(crack_x + length * np.cos(rad))
            ey    = int(crack_y + length * np.sin(rad))
            draw.line([(crack_x, crack_y), (ex, ey)],
                      fill=(60, 60, 60), width=2)
        draw.ellipse([crack_x-3, crack_y-3, crack_x+3, crack_y+3],
                     fill=(30, 30, 30))

    # Slight Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def build_synthetic_dataset(
    root_dir:       str = "synthetic_data",
    n_per_class:    int = 120,
    classes:        list = ["good", "defective", "scratched", "broken"],
    val_split:      float = 0.15,
    test_split:     float = 0.15,
) -> dict:
    """
    Create ImageFolder-compatible directory structure:
        synthetic_data/
            train/  good/ defective/ scratched/ broken/
            val/    good/ ...
            test/   good/ ...
    """
    splits     = ["train", "val", "test"]
    n_val      = int(n_per_class * val_split)
    n_test     = int(n_per_class * test_split)
    n_train    = n_per_class - n_val - n_test
    split_sizes = {"train": n_train, "val": n_val, "test": n_test}

    print(f"  Building synthetic dataset → {root_dir}/")
    print(f"  Classes: {classes}")
    print(f"  Split sizes per class: {split_sizes}")

    seed = 0
    for split in splits:
        for cls in classes:
            path = Path(root_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)
            for i in range(split_sizes[split]):
                img = make_product_image(cls, size=128, seed=seed)
                img.save(path / f"{cls}_{i:04d}.png")
                seed += 1

    total = sum(split_sizes.values()) * len(classes)
    print(f"  Saved {total} images across {len(splits)} splits")
    return {"root": root_dir, "classes": classes, "split_sizes": split_sizes}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# ImageNet mean/std — MUST use these when using pretrained models
# because ResNet50 weights were trained on data normalised with these values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(phase: str = "train") -> transforms.Compose:
    """
    Data transforms for each phase.

    Train → aggressive augmentation to prevent overfitting:
      RandomResizedCrop: crops random region then resizes → scale invariance
      RandomHorizontalFlip: mirror → orientation invariance
      ColorJitter: vary brightness/contrast/saturation → lighting invariance
      RandomRotation: ±15° → minor angle invariance
      Normalize: MUST match ImageNet stats for pretrained weights to work

    Val/Test → only resize + center crop + normalize (no randomness)

    WHY 224×224?
      ResNet50 was trained on 224×224 ImageNet images.
      Using the same size ensures the pretrained features align correctly.
    """
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1,
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:   # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(
    root_dir:   str = "synthetic_data",
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict:
    """Build DataLoaders for train / val / test splits."""
    datasets = {
        phase: ImageFolder(
            root      = os.path.join(root_dir, phase),
            transform = get_transforms(phase),
        )
        for phase in ["train", "val", "test"]
    }
    loaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size  = batch_size,
            shuffle     = (phase == "train"),
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
        for phase in ["train", "val", "test"]
    }
    class_names = datasets["train"].classes
    print(f"  Class names : {class_names}")
    for phase, ds in datasets.items():
        print(f"  {phase:<6}: {len(ds):>4} images  "
              f"({len(loaders[phase])} batches × {batch_size})")
    return loaders, class_names


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_resnet50(
    num_classes:    int   = 4,
    freeze_backbone: bool = True,
    dropout_rate:   float = 0.4,
) -> nn.Module:
    """
    Load pretrained ResNet50 and replace the classifier head.

    ResNet50 architecture:
      Input (3×224×224)
        → Conv1 (7×7, stride 2) → BatchNorm → ReLU → MaxPool
        → Layer1 (3 bottleneck blocks, 256 channels)
        → Layer2 (4 bottleneck blocks, 512 channels)
        → Layer3 (6 bottleneck blocks, 1024 channels)
        → Layer4 (3 bottleneck blocks, 2048 channels)
        → AdaptiveAvgPool → Flatten
        → FC (2048 → 1000 classes) ← WE REPLACE THIS

    Transfer learning strategy:
      Stage 1: Freeze ALL backbone layers, train ONLY the new head.
               Fast convergence, prevents destroying pretrained features.
               Use high LR for head (0.001).

      Stage 2: Unfreeze Layer4 (last ResNet block) + head, retrain.
               Allows fine-tuning of high-level features.
               Use very low LR (0.0001) to avoid catastrophic forgetting.

    Why pretrained? ResNet50 on ImageNet learned:
      - Layer1–2: edges, textures, colors (generic — always useful)
      - Layer3:   shapes, patterns (useful for defect detection)
      - Layer4:   object parts (most task-specific — fine-tune this)
    """
    # Load pretrained weights (ImageNet)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ── Freeze backbone ───────────────────────────────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("  Backbone frozen — only classifier head will train")

    # ── Replace final fully-connected layer ───────────────────────────────────
    # model.fc is currently: Linear(2048 → 1000)  [ImageNet classes]
    # We replace it with:    Linear(2048 → num_classes)
    in_features = model.fc.in_features   # 2048 for ResNet50

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),            # regularisation
        nn.Linear(in_features, 512),           # intermediate layer
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(512, num_classes),           # output layer
    )
    # Only the new head has requires_grad=True (if backbone is frozen)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}  "
          f"({trainable_params/total_params:.1%} of total)")

    return model.to(DEVICE)


def unfreeze_layer4(model: nn.Module, lr_backbone: float = 1e-4) -> optim.Optimizer:
    """
    Fine-tuning step: unfreeze ResNet Layer4 for domain adaptation.

    WHY only Layer4?
      Layer1–3 features (edges, textures, shapes) transfer well to any domain.
      Layer4 is most task-specific — unfreezing it lets the model adapt
      high-level features to your specific product defect patterns.

    WHY lower LR?
      Pretrained weights in Layer4 are already good. A high LR would
      destroy these features (catastrophic forgetting).
      Use 10× lower LR than the head.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Layer4 unfrozen | Trainable params: {trainable:,}")

    # Separate LR for backbone (lower) vs head (higher)
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": lr_backbone},
        {"params": model.fc.parameters(),     "lr": lr_backbone * 5},
    ])
    return optimizer


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    Saves the best model weights automatically.

    patience : number of epochs to wait before stopping
    delta    : minimum improvement to count as progress
    """
    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.counter  = 0
        self.best_loss = np.inf
        self.best_weights = None
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss    = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter      = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"  Early stopping triggered (patience={self.patience})")


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    phase:      str = "train",
) -> tuple[float, float]:
    """
    One forward + backward pass over the dataset.
    Returns (avg_loss, accuracy).

    Key PyTorch training steps:
      1. model.train() / model.eval() — controls BatchNorm and Dropout
      2. optimizer.zero_grad()        — clear gradients from last step
      3. forward pass                 — compute predictions
      4. criterion(outputs, labels)   — compute loss
      5. loss.backward()              — compute gradients (backprop)
      6. optimizer.step()             — update weights
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_correct = 0
    total_samples = 0

    # torch.no_grad() skips gradient computation during eval → saves memory/time
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)              # forward pass
            loss    = criterion(outputs, labels) # cross-entropy loss

            if is_train:
                loss.backward()                  # backpropagation
                # Gradient clipping: prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                 # weight update

            preds  = outputs.argmax(dim=1)
            total_loss    += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_model(
    model:        nn.Module,
    loaders:      dict,
    criterion:    nn.Module,
    optimizer:    optim.Optimizer,
    scheduler:    object,
    epochs:       int  = 15,
    patience:     int  = 5,
    stage_name:   str  = "Stage 1",
) -> dict:
    """Full training loop with early stopping and LR scheduling."""

    early_stop = EarlyStopping(patience=patience)
    history    = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    print(f"\n  {'─'*55}")
    print(f"  {stage_name}  (max {epochs} epochs, patience={patience})")
    print(f"  {'─'*55}")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14}"
          f"{'Val Loss':<14} {'Val Acc':<12} LR")
    print(f"  {'─'*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, "train")
        val_loss, val_acc = train_one_epoch(
            model, loaders["val"], criterion, optimizer, "val")

        # LR scheduler step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0
        marker     = " ← best" if val_loss <= min(history["val_loss"]) else ""

        print(f"  {epoch:<8} {tr_loss:<14.4f} {tr_acc:<14.3%}"
              f"{val_loss:<14.4f} {val_acc:<12.3%} {current_lr:.2e}"
              f"{marker}")

        early_stop(val_loss, model)
        if early_stop.stop:
            break

    # Restore best weights
    if early_stop.best_weights is not None:
        model.load_state_dict(early_stop.best_weights)
        print(f"  Restored best weights (val_loss={early_stop.best_loss:.4f})")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:       nn.Module,
    loader:      DataLoader,
    class_names: list,
) -> dict:
    """Full evaluation on test set: accuracy, F1, ROC-AUC, confusion matrix."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    # ROC-AUC (one-vs-rest for multiclass)
    y_bin = label_binarize(all_labels, classes=range(len(class_names)))
    try:
        auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    print(f"\n  TEST SET RESULTS")
    print(f"  {'─'*40}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(f"  ROC-AUC (OvR) : {auc:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0))

    return {
        "accuracy": acc, "f1": f1, "auc": auc,
        "preds": all_preds, "labels": all_labels, "probs": all_probs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    WHAT it shows: which spatial regions of the image the model focused on
                   when making its prediction.

    HOW it works:
      1. Forward pass → store feature maps from the target layer (layer4)
      2. Backprop     → compute gradient of predicted class score w.r.t. feature maps
      3. Global Average Pool the gradients → get importance weight per channel
      4. Weighted sum of feature maps → activation heatmap
      5. ReLU + upsample to input size → overlay on original image

    WHY layer4?
      It's the last convolutional layer before the global pool.
      Has the highest-level semantic features AND still has spatial resolution.

    Interview use: "I used Grad-CAM to verify the model was looking at
    actual defects (scratches, cracks) and not at background artifacts."
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, 3, 224, 224)
        class_idx:    int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        Returns numpy array (224, 224) with values in [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(DEVICE)
        image_tensor.requires_grad_(True)

        # Forward
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward on the target class score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Importance weights: global avg pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)                                          # ReLU: ignore negative

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Upsample to input resolution (224×224)
        import cv2
        cam = cv2.resize(cam, (224, 224))
        return cam


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(
    model:       nn.Module,
    image:       Image.Image,
    class_names: list,
    transform:   transforms.Compose = None,
) -> dict:
    """Predict class for a single PIL Image."""
    if transform is None:
        transform = get_transforms("val")

    model.eval()
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs":  {cls: round(float(p), 4)
                       for cls, p in zip(class_names, probs)},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history1: dict,
    history2: dict = None,
    save_path: str = "training_history.png",
) -> None:
    """Plot loss and accuracy curves for Stage 1 (and optional Stage 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, metric, ylabel in zip(
        axes,
        ["loss", "acc"],
        ["Loss", "Accuracy"],
    ):
        ax.plot(history1[f"train_{metric}"], color=PALETTE[0],
                linewidth=1.8, label="Stage1 Train")
        ax.plot(history1[f"val_{metric}"],   color=PALETTE[0],
                linewidth=1.8, linestyle="--", label="Stage1 Val")
        if history2:
            offset = len(history1[f"train_{metric}"])
            xs2 = range(offset, offset + len(history2[f"train_{metric}"]))
            ax.plot(xs2, history2[f"train_{metric}"], color=PALETTE[1],
                    linewidth=1.8, label="Stage2 Train (fine-tune)")
            ax.plot(xs2, history2[f"val_{metric}"],   color=PALETTE[1],
                    linewidth=1.8, linestyle="--", label="Stage2 Val (fine-tune)")
            ax.axvline(offset, color="grey", linestyle=":", linewidth=1.2,
                       label="Fine-tune start")
        ax.set_title(f"Training {ylabel}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle("Training History — Transfer Learning (ResNet50)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrix(
    eval_results: dict,
    class_names:  list,
    save_path:    str = "confusion_matrix.png",
) -> None:
    cm      = confusion_matrix(eval_results["labels"], eval_results["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Row %"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    cmap="Blues" if fmt == "d" else "YlOrRd",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.4, ax=ax, annot_kws={"size": 12})
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.set_yticklabels(class_names, rotation=0)

    plt.suptitle("ResNet50 Transfer Learning — Defect Classification",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_class_samples(
    class_names: list,
    save_path:   str = "class_samples.png",
) -> None:
    """Show sample images for each defect class."""
    n_samples = 4
    fig, axes = plt.subplots(len(class_names), n_samples,
                              figsize=(n_samples * 2.5, len(class_names) * 2.5))

    for row, cls in enumerate(class_names):
        for col in range(n_samples):
            img = make_product_image(cls, size=128, seed=row * 100 + col)
            axes[row, col].imshow(img)
            if col == 0:
                emoji = CLASS_EMO.get(cls, "")
                axes[row, col].set_ylabel(f"{emoji} {cls}", fontsize=10,
                                           fontweight="bold", rotation=0,
                                           labelpad=70, va="center")
            axes[row, col].axis("off")

    plt.suptitle("Product Defect Classes — Sample Images",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_prediction_confidence(
    eval_results: dict,
    class_names:  list,
    save_path:    str = "confidence_dist.png",
) -> None:
    """Confidence distribution for correct vs wrong predictions."""
    max_prob = eval_results["probs"].max(axis=1)
    correct  = eval_results["preds"] == eval_results["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(max_prob[correct],  bins=20, color=PALETTE[0],
                 alpha=0.75, label="Correct",   edgecolor="white")
    axes[0].hist(max_prob[~correct], bins=20, color=PALETTE[1],
                 alpha=0.75, label="Incorrect", edgecolor="white")
    axes[0].set_title("Confidence Distribution", fontweight="bold")
    axes[0].set_xlabel("Max predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Per-class accuracy
    per_class_acc = []
    for i, cls in enumerate(class_names):
        mask = eval_results["labels"] == i
        if mask.sum() > 0:
            acc = (eval_results["preds"][mask] == i).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)

    bars = axes[1].bar(class_names, per_class_acc,
                        color=PALETTE[:len(class_names)], alpha=0.85,
                        edgecolor="white")
    axes[1].bar_label(bars, labels=[f"{v:.1%}" for v in per_class_acc],
                       padding=4, fontsize=10)
    axes[1].set_title("Per-Class Accuracy", fontweight="bold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.15)
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Model Performance Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_gradcam(
    model:       nn.Module,
    class_names: list,
    save_path:   str = "gradcam.png",
) -> None:
    """Grad-CAM heatmaps for one sample per class."""
    import cv2
    gradcam   = GradCAM(model, model.layer4[-1])
    transform = get_transforms("val")

    fig, axes = plt.subplots(
        len(class_names), 3,
        figsize=(9, len(class_names) * 2.8),
    )

    for row, cls in enumerate(class_names):
        pil_img = make_product_image(cls, size=224, seed=row * 999)
        tensor  = transform(pil_img).unsqueeze(0)

        # Predict
        result  = predict_single(model, pil_img, class_names, transform)
        cam     = gradcam.generate(tensor)

        # Create heatmap overlay
        img_np  = np.array(pil_img.resize((224, 224)))
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (img_np * 0.55 + heatmap * 0.45).clip(0, 255).astype(np.uint8)

        axes[row, 0].imshow(pil_img.resize((224, 224)))
        axes[row, 0].set_title(f"Original: {cls}", fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(f"Grad-CAM heatmap", fontsize=9)
        axes[row, 1].axis("off")

        pred_lbl = result["prediction"]
        conf     = result["confidence"]
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Pred: {pred_lbl} ({conf:.0%})", fontsize=9,
            color="green" if pred_lbl == cls else "red",
        )
        axes[row, 2].axis("off")

    plt.suptitle("Grad-CAM — What ResNet50 Sees",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — full end-to-end run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    CLASS_NAMES = ["good", "defective", "scratched", "broken"]
    DATA_ROOT   = "/home/claude/product_data"
    BATCH_SIZE  = 16
    N_PER_CLASS = 80    # increase for better accuracy

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1: BUILD SYNTHETIC DATASET")
    print("="*60)
    build_synthetic_dataset(DATA_ROOT, n_per_class=N_PER_CLASS,
                             classes=CLASS_NAMES)

    # ── 2. Sample grid ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2: CLASS SAMPLE IMAGES")
    print("="*60)
    plot_class_samples(CLASS_NAMES, "class_samples.png")

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: BUILD DATALOADERS")
    print("="*60)
    loaders, class_names = get_dataloaders(DATA_ROOT, BATCH_SIZE)

    # ── 4. Build model (frozen backbone) ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4: BUILD RESNET50 MODEL")
    print("="*60)
    model = build_resnet50(num_classes=len(CLASS_NAMES), freeze_backbone=True)

    # ── 5. Stage 1 — train head only ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5: STAGE 1 — TRAIN HEAD ONLY")
    print("="*60)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=1e-3, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="min", factor=0.5, patience=3, verbose=False
    )
    history1 = train_model(model, loaders, criterion, optimizer1, scheduler1,
                            epochs=12, patience=5, stage_name="Stage 1 (head only)")

    # ── 6. Stage 2 — fine-tune Layer4 ────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 6: STAGE 2 — FINE-TUNE LAYER4")
    print("="*60)
    optimizer2 = unfreeze_layer4(model, lr_backbone=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=8)
    history2 = train_model(model, loaders, criterion, optimizer2, scheduler2,
                            epochs=8, patience=4, stage_name="Stage 2 (fine-tune)")

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 7: EVALUATE ON TEST SET")
    print("="*60)
    eval_results = evaluate_model(model, loaders["test"], class_names)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 8: GENERATE PLOTS")
    print("="*60)
    plot_training_history(history1, history2, "training_history.png")
    plot_confusion_matrix(eval_results, class_names, "confusion_matrix.png")
    plot_prediction_confidence(eval_results, class_names, "confidence_dist.png")

    # Grad-CAM (needs cv2)
    try:
        import cv2
        plot_gradcam(model, class_names, "gradcam.png")
    except ImportError:
        print("  [!] OpenCV not available — skipping Grad-CAM plot")

    # ── 9. Single image prediction ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 9: INFERENCE EXAMPLES")
    print("="*60)
    for cls in CLASS_NAMES:
        img    = make_product_image(cls, size=224, seed=9999)
        result = predict_single(model, img, class_names)
        emoji  = CLASS_EMO.get(cls, "")
        correct = "✓" if result["prediction"] == cls else "✗"
        print(f"  {correct} True: {emoji}{cls:<12} "
              f"Pred: {result['prediction']:<12} "
              f"Conf: {result['confidence']:.1%}")
        print(f"    Probs: " +
              "  ".join(f"{k}:{v:.3f}" for k, v in result["all_probs"].items()))

    # ── 10. Save model ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 10: SAVE MODEL")
    print("="*60)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names":      class_names,
        "architecture":     "resnet50",
        "num_classes":      len(class_names),
    }, "resnet50_defect_classifier.pth")
    print("  Saved → resnet50_defect_classifier.pth")

    # Load back example:
    print("\n  Load back with:")
    print("    checkpoint = torch.load('resnet50_defect_classifier.pth')")
    print("    model = build_resnet50(num_classes=4, freeze_backbone=False)")
    print("    model.load_state_dict(checkpoint['model_state_dict'])")
    print("    model.eval()")

    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}\n")