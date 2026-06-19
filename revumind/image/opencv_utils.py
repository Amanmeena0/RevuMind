"""
OpenCV Image Preprocessing Pipeline — Product Review Images
============================================================
Covers every operation you need for interview + real projects:

  1.  Load & validate images (file / URL / numpy array)
  2.  Resize with aspect-ratio preservation
  3.  Color space conversions  (BGR → RGB / HSV / LAB / Gray)
  4.  Denoising & smoothing    (Gaussian, Median, Bilateral)
  5.  Edge detection           (Canny, Sobel, Laplacian)
  6.  Color histogram extraction (BGR + HSV, normalised)
  7.  Thresholding             (Otsu, adaptive)
  8.  Morphological operations (dilate, erode, open, close)
  9.  Contour detection        (find & draw product contours)
  10. Feature extraction       (dominant colors, brightness, contrast)
  11. Image augmentation       (flip, rotate, brightness, noise)
  12. Batch pipeline           (process entire DataFrame of image URLs)
  13. Visualisation            (side-by-side comparison grids)

Usage:
    pipeline = ImagePreprocessor(target_size=(224, 224))
    result   = pipeline.process("path/to/image.jpg")
    features = result.features   # dict ready for ML model
"""

import os
import io
import warnings
import requests
import numpy as np
import pandas as pd
import cv2
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

from revumind.utils.constants import PALETTE

warnings.filterwarnings("ignore")

print(f"OpenCV version : {cv2.__version__}")
print(f"NumPy  version : {np.__version__}")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageResult:
    """Holds every intermediate image and extracted feature."""
    source:       str                      # path / URL / "array"
    original:     Optional[np.ndarray] = None   # BGR, original size
    resized:      Optional[np.ndarray] = None   # BGR, target_size
    gray:         Optional[np.ndarray] = None   # single channel
    blurred:      Optional[np.ndarray] = None   # denoised
    edges_canny:  Optional[np.ndarray] = None
    edges_sobel:  Optional[np.ndarray] = None
    thresh:       Optional[np.ndarray] = None   # binary mask
    morphed:      Optional[np.ndarray] = None   # after morph ops
    contours_img: Optional[np.ndarray] = None   # contours drawn
    features:     dict = field(default_factory=dict)
    success:      bool = True
    error:        str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_from_file(path: str) -> Optional[np.ndarray]:
    """Load image from disk. Returns BGR numpy array or None."""
    img = cv2.imread(path)
    if img is None:
        print(f"  [!] Cannot read: {path}")
    return img


def load_from_url(url: str, timeout: int = 10) -> Optional[np.ndarray]:
    """
    Download image from URL and decode to BGR numpy array.
    Used when review image URLs come from a scraper.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        arr  = np.frombuffer(resp.content, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"  [!] URL load failed: {e}")
        return None


def load_image(
    source: Union[str, np.ndarray],
) -> tuple[Optional[np.ndarray], str]:
    """
    Universal loader: accepts file path, URL, or existing numpy array.
    Always returns BGR (OpenCV default) + source label.
    """
    if isinstance(source, np.ndarray):
        return source.copy(), "array"
    if source.startswith("http://") or source.startswith("https://"):
        return load_from_url(source), source
    return load_from_file(source), source


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESIZE
# ══════════════════════════════════════════════════════════════════════════════

def resize_image(
    img:         np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    keep_aspect: bool            = True,
    pad_color:   tuple           = (255, 255, 255),  # white padding
) -> np.ndarray:
    """
    Resize image to target_size.

    keep_aspect=True  → letterbox: preserve ratio, pad with pad_color
    keep_aspect=False → stretch:   distort to exact target size

    Why 224×224?  → Standard input size for most CNN architectures
                    (VGG, ResNet, MobileNet, EfficientNet)
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]

    if not keep_aspect:
        return cv2.resize(img, (target_w, target_h),
                          interpolation=cv2.INTER_LINEAR)

    # ── Letterbox resize (preserves aspect ratio) ─────────────────────────────
    scale   = min(target_w / w, target_h / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to exact target size
    canvas  = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COLOR SPACE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_color_spaces(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """
    Convert BGR image into multiple color spaces.

    BGR  → OpenCV default (note: NOT RGB!)
    RGB  → matplotlib / PIL standard
    GRAY → single channel for edge detection
    HSV  → Hue-Saturation-Value: best for color segmentation
    LAB  → Perceptually uniform: good for similarity metrics
    YCrCb→ Separates luminance from chroma: used in skin detection

    Key interview point: OpenCV reads images as BGR, NOT RGB.
    Always convert before displaying with matplotlib (which expects RGB).
    """
    return {
        "BGR":   img_bgr,
        "RGB":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "GRAY":  cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        "HSV":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "LAB":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
        "YCrCb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DENOISING & SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def denoise_image(
    img:    np.ndarray,
    method: str = "bilateral",
) -> np.ndarray:
    """
    Remove noise while preserving edges.

    Gaussian  → fast, blurs edges too, kernel must be odd (e.g. 5×5)
    Median    → best for salt-and-pepper noise, preserves edges
    Bilateral → BEST for product images: smooths flat areas, keeps edges sharp
                Params: d=9 (filter size), sigmaColor=75, sigmaSpace=75

    Interview tip: always use Bilateral before Canny edge detection.
    Gaussian or Median lose edge sharpness which hurts Canny accuracy.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
    elif method == "median":
        return cv2.medianBlur(img, ksize=5)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "nlm":                     # Non-Local Means (slow but best)
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EDGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_edges_canny(
    gray:          np.ndarray,
    threshold1:    int = 50,
    threshold2:    int = 150,
    aperture_size: int = 3,
) -> np.ndarray:
    """
    Canny edge detector — the gold standard for edge detection.

    How it works (4 steps):
      1. Gaussian smoothing     → reduce noise
      2. Gradient computation   → find intensity changes (Sobel)
      3. Non-maximum suppression→ thin edges to 1 pixel wide
      4. Hysteresis thresholding→ keep strong edges, link weak ones

    threshold1 (low) : edges below this are rejected
    threshold2 (high): edges above this are always kept
    Edges between the two thresholds are kept only if connected to strong edges.

    Rule of thumb: threshold2 = 2× or 3× threshold1
    Lower thresholds → more edges (noisier)
    Higher thresholds → fewer, cleaner edges
    """
    return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)


def detect_edges_sobel(
    gray: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Sobel edge detector — computes gradient in X and Y directions.
    Combined gradient magnitude = sqrt(Gx² + Gy²)

    Less robust than Canny but useful for directional edge analysis.
    ksize must be 1, 3, 5, or 7.
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)   # horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)   # vertical edges
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_edges_laplacian(gray: np.ndarray) -> np.ndarray:
    """
    Laplacian edge detector — second derivative, isotropic (no direction).
    Good for detecting blurry vs sharp images (variance of Laplacian).
    Blurry image → low variance. Sharp image → high variance.
    """
    lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def blur_score(gray: np.ndarray) -> float:
    """
    Detect if a product image is blurry.
    Returns variance of Laplacian — threshold ~100 separates sharp vs blurry.
    Low score (< 100) → blurry image (bad product photo)
    High score (> 100)→ sharp image
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COLOR HISTOGRAM EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_color_histogram(
    img:        np.ndarray,
    color_space: str = "BGR",
    bins:        int = 32,
    normalize:   bool = True,
) -> dict[str, np.ndarray]:
    """
    Extract color histogram — distribution of color intensities per channel.

    Why histograms?
      → Compact image representation (3 × bins values vs H×W×3 pixels)
      → Invariant to rotation and small spatial changes
      → Fast to compute and compare (chi-square, correlation, intersection)
      → Used as features for product categorisation, duplicate detection

    BGR  space → channel order: Blue, Green, Red
    HSV  space → Hue channel most useful (pure color ignoring brightness)
    GRAY space → single luminance distribution
    """
    if color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        channel_names = ["Hue", "Saturation", "Value"]
        ranges        = [(0, 180), (0, 256), (0, 256)]   # Hue range is 0-179 in OpenCV
    elif color_space == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        channel_names = ["Gray"]
        ranges        = [(0, 256)]
    else:   # BGR
        channel_names = ["Blue", "Green", "Red"]
        ranges        = [(0, 256)] * 3

    histograms = {}
    if len(img.shape) == 2:   # grayscale
        channels = [img]
    else:
        channels = [img[:, :, i] for i in range(img.shape[2])]

    for name, channel, (lo, hi) in zip(channel_names, channels, ranges):
        hist = cv2.calcHist([channel], [0], None, [bins], [lo, hi])
        hist = hist.flatten()
        if normalize:
            hist = cv2.normalize(hist, hist, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX).flatten()
        histograms[name] = hist

    return histograms


def histogram_to_feature_vector(histograms: dict) -> np.ndarray:
    """Flatten all channel histograms into a single 1-D feature vector."""
    return np.concatenate(list(histograms.values()))


def compare_histograms(
    hist1: np.ndarray,
    hist2: np.ndarray,
    method: str = "correlation",
) -> float:
    """
    Compare two histogram feature vectors.
    Used for product duplicate detection or similarity search.

    correlation → 1.0 = identical,  0.0 = no relation, -1.0 = inverse
    chi_square  → 0.0 = identical,  higher = more different
    intersection→ higher = more similar
    """
    h1 = hist1.astype(np.float32).reshape(-1, 1)
    h2 = hist2.astype(np.float32).reshape(-1, 1)
    methods = {
        "correlation":  cv2.HISTCMP_CORREL,
        "chi_square":   cv2.HISTCMP_CHISQR,
        "intersection": cv2.HISTCMP_INTERSECT,
        "bhattacharyya":cv2.HISTCMP_BHATTACHARYYA,
    }
    return float(cv2.compareHist(h1, h2, methods[method]))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  THRESHOLDING
# ══════════════════════════════════════════════════════════════════════════════

def threshold_image(
    gray:   np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """
    Convert grayscale to binary (black/white) image.

    Simple    → fixed threshold value (needs manual tuning)
    Otsu      → automatically finds optimal threshold using histogram valleys
                Best for images with bimodal histogram (product vs background)
    Adaptive  → different threshold for each local region
                Best for images with varying lighting (shadows, reflections)

    Use case: segment product from white background in review photos.
    """
    if method == "simple":
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        # Otsu's method: finds threshold that minimises intra-class variance
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2
        )
    elif method == "adaptive_gaussian":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2
        )
    else:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MORPHOLOGICAL OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def morphological_ops(
    binary:    np.ndarray,
    operation: str = "close",
    ksize:     int = 5,
) -> np.ndarray:
    """
    Morphological operations on binary images.

    Erosion  → shrink white regions, remove small noise pixels
    Dilation → expand white regions, fill small holes
    Opening  → erosion then dilation: remove small noise blobs
    Closing  → dilation then erosion: fill holes inside objects
    Gradient → dilation - erosion: gives edge outline

    Use case for reviews:
      Closing → fill gaps in product boundary before contour detection
      Opening → remove specks and noise from thresholded product images
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize)
    )
    ops = {
        "erode":    cv2.MORPH_ERODE,
        "dilate":   cv2.MORPH_DILATE,
        "open":     cv2.MORPH_OPEN,
        "close":    cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat":   cv2.MORPH_TOPHAT,
    }
    return cv2.morphologyEx(binary, ops.get(operation, cv2.MORPH_CLOSE), kernel)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CONTOUR DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_contours(
    binary: np.ndarray,
    original_bgr: np.ndarray,
    min_area: float = 0.005,   # min contour area as fraction of image
) -> tuple[np.ndarray, list, dict]:
    """
    Find and draw product contours.

    cv2.RETR_EXTERNAL  → only outermost contours (product boundary)
    cv2.RETR_TREE      → full hierarchy (nested contours)
    cv2.CHAIN_APPROX_SIMPLE → compress straight lines to endpoints

    Returns:
        drawn_img    : BGR image with contours overlaid
        contours     : list of contour arrays
        stats        : bounding box, area, perimeter of largest contour
    """
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area    = binary.shape[0] * binary.shape[1]
    min_px_area = min_area * img_area

    # Filter small contours (noise)
    valid = [c for c in contours if cv2.contourArea(c) > min_px_area]

    # Draw on a copy of original
    drawn = original_bgr.copy()
    cv2.drawContours(drawn, valid, -1, (0, 255, 0), 2)   # green lines

    stats = {}
    if valid:
        # Largest contour = likely the main product
        largest = max(valid, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        peri    = cv2.arcLength(largest, True)
        x, y, w, h = cv2.boundingRect(largest)
        # Approximate polygon
        approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)
        # Draw bounding box in blue
        cv2.rectangle(drawn, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stats = {
            "contour_count":   len(valid),
            "largest_area":    float(area),
            "largest_area_pct":round(area / img_area * 100, 2),
            "bounding_box":    (x, y, w, h),
            "perimeter":       float(peri),
            "approx_vertices": len(approx),
            "aspect_ratio":    round(w / max(h, 1), 3),
            "extent":          round(area / max(w * h, 1), 3),   # fill ratio
        }

    return drawn, valid, stats


# ══════════════════════════════════════════════════════════════════════════════
# 10. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_image_features(
    img_bgr: np.ndarray,
    gray:    np.ndarray,
) -> dict:
    """
    Extract a comprehensive set of numeric features from one image.
    These become columns in your ML model's feature matrix.
    """
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    features = {}

    # ── Dimensions ────────────────────────────────────────────────────────────
    features["img_width"]   = w
    features["img_height"]  = h
    features["aspect_ratio"]= round(w / max(h, 1), 3)
    features["total_pixels"]= w * h

    # ── Brightness & contrast ─────────────────────────────────────────────────
    features["brightness_mean"] = float(gray.mean())
    features["brightness_std"]  = float(gray.std())
    features["contrast"]        = float(gray.std() / max(gray.mean(), 1))
    features["blur_score"]      = round(blur_score(gray), 2)

    # ── Per-channel BGR stats ─────────────────────────────────────────────────
    for i, ch in enumerate(["blue", "green", "red"]):
        channel = img_bgr[:, :, i]
        features[f"{ch}_mean"] = float(channel.mean())
        features[f"{ch}_std"]  = float(channel.std())

    # ── HSV stats ─────────────────────────────────────────────────────────────
    features["hue_mean"]        = float(hsv[:, :, 0].mean())
    features["saturation_mean"] = float(hsv[:, :, 1].mean())
    features["value_mean"]      = float(hsv[:, :, 2].mean())
    features["is_colorful"]     = features["saturation_mean"] > 50

    # ── Edge density ──────────────────────────────────────────────────────────
    edges = detect_edges_canny(gray)
    features["edge_density"]    = float(edges.sum()) / (w * h * 255)

    # ── Histogram feature vector (96-dim for 32 bins × 3 BGR channels) ────────
    bgr_hist = extract_color_histogram(img_bgr, color_space="BGR", bins=32)
    hsv_hist = extract_color_histogram(img_bgr, color_space="HSV", bins=32)
    features["bgr_hist_vector"] = histogram_to_feature_vector(bgr_hist)  # shape (96,)
    features["hsv_hist_vector"] = histogram_to_feature_vector(hsv_hist)  # shape (96,)

    # ── Dominant colors (K-Means on pixels) ───────────────────────────────────
    features["dominant_colors"] = get_dominant_colors(img_bgr, k=3)

    # ── White background detection (useful for product listing images) ─────────
    white_mask = (img_bgr[:, :, 0] > 200) & \
                 (img_bgr[:, :, 1] > 200) & \
                 (img_bgr[:, :, 2] > 200)
    features["white_bg_ratio"]  = float(white_mask.sum()) / (w * h)
    features["has_white_bg"]    = features["white_bg_ratio"] > 0.4

    return features


def get_dominant_colors(img_bgr: np.ndarray, k: int = 3) -> list[tuple]:
    """
    K-Means clustering on pixel colors → find k dominant colors.
    Returns list of (R, G, B) tuples for the k cluster centers.
    """
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    # Subsample for speed on large images
    if len(pixels) > 5000:
        idx    = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.astype(int)
    # Count pixels per cluster (sort by frequency)
    counts  = Counter(labels.flatten())
    order   = sorted(counts.keys(), key=lambda x: -counts[x])
    # Return as RGB tuples
    return [(int(centers[i][2]), int(centers[i][1]), int(centers[i][0]))
            for i in order]


# ══════════════════════════════════════════════════════════════════════════════
# 11. AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_image(
    img:        np.ndarray,
    flip_h:     bool  = True,
    rotate_deg: float = 15.0,
    brightness: float = 1.2,     # multiplier, 1.0 = no change
    add_noise:  bool  = True,
    noise_std:  float = 10.0,
) -> dict[str, np.ndarray]:
    """
    Common augmentations used during CNN training.
    Augmentation artificially increases dataset size and prevents overfitting.

    flip_h     → horizontal flip (mirror image)
    rotate_deg → random rotation within ±rotate_deg degrees
    brightness → multiply pixel values (>1 brighter, <1 darker)
    add_noise  → Gaussian noise simulates camera sensor noise
    """
    augmented = {}

    # Horizontal flip
    if flip_h:
        augmented["flip_h"] = cv2.flip(img, 1)

    # Rotation
    h, w = img.shape[:2]
    angle = np.random.uniform(-rotate_deg, rotate_deg)
    M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    augmented["rotated"] = cv2.warpAffine(img, M, (w, h),
                                           borderMode=cv2.BORDER_REFLECT)

    # Brightness adjustment
    bright = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    augmented["bright"] = bright

    # Gaussian noise
    if add_noise:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        augmented["noisy"] = noisy

    return augmented


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """
    End-to-end image preprocessing pipeline for product review images.
    Designed to feed into CNN classifiers or feature-based ML models.
    """

    def __init__(
        self,
        target_size:   tuple = (224, 224),
        denoise_method: str  = "bilateral",
        edge_thresh:   tuple = (50, 150),
        thresh_method: str   = "otsu",
        morph_op:      str   = "close",
        extract_feats: bool  = True,
    ):
        self.target_size    = target_size
        self.denoise_method = denoise_method
        self.edge_thresh    = edge_thresh
        self.thresh_method  = thresh_method
        self.morph_op       = morph_op
        self.extract_feats  = extract_feats

    def process(
        self,
        source: Union[str, np.ndarray],
    ) -> ImageResult:
        """
        Run the full preprocessing pipeline on one image.
        Returns ImageResult with every intermediate stage.
        """
        result = ImageResult(source=str(source)[:80])

        # 1. Load
        img, lbl = load_image(source)
        if img is None:
            result.success = False
            result.error   = f"Failed to load: {source}"
            return result

        result.original = img

        # 2. Resize (letterbox to target_size)
        result.resized  = resize_image(img, self.target_size, keep_aspect=True)

        # 3. Grayscale
        result.gray     = cv2.cvtColor(result.resized, cv2.COLOR_BGR2GRAY)

        # 4. Denoise
        result.blurred  = denoise_image(result.resized, self.denoise_method)
        gray_blur        = cv2.cvtColor(result.blurred, cv2.COLOR_BGR2GRAY)

        # 5. Edge detection
        result.edges_canny = detect_edges_canny(gray_blur, *self.edge_thresh)
        result.edges_sobel = detect_edges_sobel(gray_blur)

        # 6. Threshold
        result.thresh   = threshold_image(result.gray, self.thresh_method)

        # 7. Morphological cleanup
        result.morphed  = morphological_ops(result.thresh, self.morph_op)

        # 8. Contours
        contour_img, contours, contour_stats = detect_contours(
            result.morphed, result.resized
        )
        result.contours_img = contour_img

        # 9. Feature extraction
        if self.extract_feats:
            feats = extract_image_features(result.resized, result.gray)
            feats.update(contour_stats)
            bgr_hist = extract_color_histogram(result.resized, "BGR", bins=32)
            hsv_hist = extract_color_histogram(result.resized, "HSV", bins=32)
            feats["bgr_histograms"] = bgr_hist
            feats["hsv_histograms"] = hsv_hist
            result.features = feats

        return result

    # ── Batch over DataFrame ──────────────────────────────────────────────────
    def process_dataframe(
        self,
        df:        pd.DataFrame,
        img_col:   str = "image_url",
        save_dir:  str = "processed_images",
    ) -> pd.DataFrame:
        """
        Run pipeline on every image URL / path in a DataFrame.
        Adds scalar feature columns. Histogram vectors saved separately.
        """
        os.makedirs(save_dir, exist_ok=True)
        scalar_rows = []
        print(f"  Processing {len(df):,} images…")
        for i, source in enumerate(df[img_col].fillna(""), 1):
            r = self.process(source)
            row = {
                "success":          r.success,
                "error":            r.error,
            }
            if r.success and r.features:
                # Only scalar features go into the DataFrame
                for k, v in r.features.items():
                    if isinstance(v, (int, float, bool)):
                        row[k] = v
            scalar_rows.append(row)
            if i % 20 == 0:
                print(f"    {i}/{len(df)} done…")

        feat_df = pd.DataFrame(scalar_rows, index=df.index)
        return pd.concat([df, feat_df], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 13. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_stages(
    result:    ImageResult,
    save_path: str = "pipeline_stages.png",
) -> None:
    """Side-by-side grid showing every preprocessing stage."""
    stages = [
        ("Original (BGR→RGB)",  cv2.cvtColor(result.original,  cv2.COLOR_BGR2RGB)  if result.original  is not None else None),
        ("Resized 224×224",     cv2.cvtColor(result.resized,   cv2.COLOR_BGR2RGB)  if result.resized   is not None else None),
        ("Grayscale",           result.gray),
        ("Denoised (Bilateral)",cv2.cvtColor(result.blurred,   cv2.COLOR_BGR2RGB)  if result.blurred   is not None else None),
        ("Canny Edges",         result.edges_canny),
        ("Sobel Edges",         result.edges_sobel),
        ("Otsu Threshold",      result.thresh),
        ("Morphed (Close)",     result.morphed),
        ("Contours",            cv2.cvtColor(result.contours_img, cv2.COLOR_BGR2RGB) if result.contours_img is not None else None),
    ]
    stages = [(t, i) for t, i in stages if i is not None]
    n = len(stages)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, stages):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Image Preprocessing Pipeline Stages",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_color_histograms(
    result:    ImageResult,
    save_path: str = "color_histograms.png",
) -> None:
    """Plot BGR and HSV histograms side by side."""
    if not result.features or "bgr_histograms" not in result.features:
        return

    bgr_hist = result.features["bgr_histograms"]
    hsv_hist = result.features["hsv_histograms"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    x = np.arange(32)

    # BGR histograms
    for ax, (ch, color, mpl_color) in zip(
        axes[0],
        [("Blue","#378ADD","b"), ("Green","#5DCAA5","g"), ("Red","#D85A30","r")]
    ):
        if ch in bgr_hist:
            ax.bar(x, bgr_hist[ch], color=mpl_color, alpha=0.75, width=0.8)
            ax.set_title(f"BGR — {ch} channel", fontweight="bold")
            ax.set_xlabel("Bin"); ax.set_ylabel("Normalised frequency")

    # HSV histograms
    for ax, (ch, color) in zip(
        axes[1],
        [("Hue","#7F77DD"), ("Saturation","#EF9F27"), ("Value","#888780")]
    ):
        if ch in hsv_hist:
            ax.bar(x, hsv_hist[ch], color=color, alpha=0.75, width=0.8)
            ax.set_title(f"HSV — {ch} channel", fontweight="bold")
            ax.set_xlabel("Bin"); ax.set_ylabel("Normalised frequency")

    plt.suptitle("Color Histograms (BGR + HSV)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_dominant_colors(
    result:    ImageResult,
    save_path: str = "dominant_colors.png",
) -> None:
    """Swatch grid of K dominant colors extracted via K-Means."""
    colors = result.features.get("dominant_colors", [])
    if not colors:
        return

    fig, axes = plt.subplots(1, len(colors), figsize=(3 * len(colors), 2.5))
    if len(colors) == 1:
        axes = [axes]

    for ax, (r, g, b) in zip(axes, colors):
        swatch = np.full((80, 80, 3), [r, g, b], dtype=np.uint8)
        ax.imshow(swatch)
        ax.set_title(f"RGB\n({r},{g},{b})", fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Dominant Colors (K-Means, k=3)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_feature_summary(
    result:    ImageResult,
    save_path: str = "feature_summary.png",
) -> None:
    """Bar chart of scalar feature values extracted from the image."""
    scalar = {k: v for k, v in result.features.items()
              if isinstance(v, (int, float)) and k not in
              ["img_width","img_height","total_pixels"]}

    if not scalar:
        return

    keys   = list(scalar.keys())
    values = [scalar[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, max(4, len(keys) * 0.4)))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(keys))]
    bars    = ax.barh(keys[::-1], values[::-1], color=colors[::-1],
                      alpha=0.82, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_title("Extracted Image Features", fontweight="bold")
    ax.set_xlabel("Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — runs without any real image file
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_product_image(w: int = 400, h: int = 400) -> np.ndarray:
    """
    Generate a realistic synthetic product image for demo purposes.
    Simulates a coloured product on a white background.
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 248   # near-white background

    # Product body (rounded rectangle simulation with ellipse)
    center = (w // 2, h // 2)
    cv2.ellipse(img, center, (130, 90), 0, 0, 360, (52, 152, 219), -1)   # blue body
    cv2.ellipse(img, center, (110, 72), 0, 0, 360, (41, 128, 185), -1)   # darker inner

    # Product highlight (glossy effect)
    cv2.ellipse(img, (w//2 - 35, h//2 - 25), (55, 30), -30, 0, 360,
                (255, 255, 255), -1)
    # Fade highlight
    highlight_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(highlight_mask, (w//2-35, h//2-25), (55,30), -30, 0, 360, 255, -1)
    img[highlight_mask > 0] = (
        img[highlight_mask > 0].astype(float) * 0.4 + 255 * 0.6
    ).astype(np.uint8)

    # Product button / detail
    cv2.circle(img, (w//2 + 70, h//2), 18, (231, 76, 60), -1)   # red button
    cv2.circle(img, (w//2 + 70, h//2), 12, (192, 57, 43), -1)

    # Add subtle noise (realistic)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    np.random.seed(42)
    print("\n" + "="*62)
    print("  OPENCV IMAGE PREPROCESSING PIPELINE — DEMO")
    print("="*62)

    # ── Create synthetic product image ────────────────────────────────────────
    print("\n  Creating synthetic product image…")
    demo_img = make_synthetic_product_image(400, 400)

    # ── Run the full pipeline ─────────────────────────────────────────────────
    pipeline = ImagePreprocessor(
        target_size    = (224, 224),
        denoise_method = "bilateral",
        edge_thresh    = (50, 150),
        thresh_method  = "otsu",
        morph_op       = "close",
        extract_feats  = True,
    )

    print("  Running pipeline…")
    result = pipeline.process(demo_img)

    # ── Print extracted features ──────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  EXTRACTED FEATURES")
    print(f"{'─'*62}")
    scalar_feats = {k: v for k, v in result.features.items()
                    if isinstance(v, (int, float, bool))}
    for k, v in scalar_feats.items():
        if isinstance(v, float):
            print(f"  {k:<28} : {v:.4f}")
        else:
            print(f"  {k:<28} : {v}")

    print(f"\n  Dominant colors (RGB):")
    for i, (r, g, b) in enumerate(result.features.get("dominant_colors", []), 1):
        print(f"    Color {i}: RGB({r:3d}, {g:3d}, {b:3d})")

    print(f"\n  BGR histogram vector shape : "
          f"{result.features['bgr_hist_vector'].shape}")
    print(f"  HSV histogram vector shape : "
          f"{result.features['hsv_hist_vector'].shape}")
    print(f"  Combined feature vector    : "
          f"({result.features['bgr_hist_vector'].shape[0] + result.features['hsv_hist_vector'].shape[0]},)")

    # ── Histogram comparison demo ─────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  HISTOGRAM COMPARISON DEMO")
    print(f"{'─'*62}")
    img2    = make_synthetic_product_image(400, 400)
    img2[:, :, 0] = np.clip(img2[:, :, 0].astype(int) + 30, 0, 255)  # shift hue
    result2 = pipeline.process(img2)
    h1 = result.features["bgr_hist_vector"]
    h2 = result2.features["bgr_hist_vector"]
    print(f"  Same image correlation     : {compare_histograms(h1, h1, 'correlation'):.4f}")
    print(f"  Different image correlation: {compare_histograms(h1, h2, 'correlation'):.4f}")

    # ── Individual operations demo ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  INDIVIDUAL OPERATION OUTPUTS")
    print(f"{'─'*62}")

    gray   = cv2.cvtColor(result.resized, cv2.COLOR_BGR2GRAY)
    edges  = detect_edges_canny(gray)
    sobel  = detect_edges_sobel(gray)
    thresh = threshold_image(gray, "otsu")
    morphed= morphological_ops(thresh, "close")
    _, _, cstats = detect_contours(morphed, result.resized)

    print(f"  Resized shape      : {result.resized.shape}")
    print(f"  Edges (Canny)      : {edges.shape}, "
          f"non-zero px: {np.count_nonzero(edges)}")
    print(f"  Edges (Sobel)      : {sobel.shape}")
    print(f"  Threshold          : {thresh.shape}, "
          f"white px: {(thresh > 0).sum()}")
    print(f"  Contour stats      : {cstats}")
    print(f"  Blur score         : {blur_score(gray):.2f} "
          f"({'sharp' if blur_score(gray) > 100 else 'blurry'})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  GENERATING PLOTS")
    print(f"{'─'*62}")
    plot_pipeline_stages(result,     "pipeline_stages.png")
    plot_color_histograms(result,    "color_histograms.png")
    plot_dominant_colors(result,     "dominant_colors.png")
    plot_feature_summary(result,     "feature_summary.png")

    print(f"\n{'='*62}")
    print("  DONE — use this in your project:")
    print("    pipeline = ImagePreprocessor(target_size=(224, 224))")
    print("    result   = pipeline.process('path/or/url/to/image.jpg')")
    print("    features = result.features  # dict → plug into ML model")
    print("    df_out   = pipeline.process_dataframe(df, img_col='image_url')")
    print(f"{'='*62}\n")