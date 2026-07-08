"""
data_augmentation.py
====================
Stage 2 — Data Augmentation Pipeline for Custom Jewelry VTO Model

Expands a small jewelry image dataset (91 PNG images) into a training-ready
corpus of 500–1000 samples per category using photorealistic augmentations
while preserving jewellery visibility and alpha transparency.

Usage:
    python data_augmentation.py --input_dir ./static --output_dir ./dataset/augmented

Dependencies:
    pip install albumentations opencv-python-headless numpy tqdm Pillow
"""

import os
import cv2
import numpy as np
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARNING] albumentations not installed. Using manual augmentations only.")
    print("          Run: pip install albumentations")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "chain":       {"folder": "chain",       "target_count": 600, "keypoint_type": "necklace"},
    "jhumka":      {"folder": "jhumka",      "target_count": 700, "keypoint_type": "earring"},
    "mangalsutra": {"folder": "mangalsutra", "target_count": 600, "keypoint_type": "necklace"},
    "necklace":    {"folder": "necklace",    "target_count": 600, "keypoint_type": "necklace"},
    "ring":        {"folder": "ring",        "target_count": 800, "keypoint_type": "ring"},
}

# Augmentation intensity levels
AUGMENTATION_LEVELS = {
    "light":    {"p": 0.3, "rotate_limit": 10,  "scale_limit": 0.1},
    "moderate": {"p": 0.5, "rotate_limit": 20,  "scale_limit": 0.2},
    "heavy":    {"p": 0.7, "rotate_limit": 35,  "scale_limit": 0.35},
}

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# CORE AUGMENTATION TRANSFORMS
# ---------------------------------------------------------------------------

def build_albumentations_pipeline(level: str = "moderate") -> "A.Compose":
    """
    Constructs an albumentations augmentation pipeline tuned for jewelry images.

    Jewellery-specific choices:
    - No harsh distortions that warp the jewellery shape beyond recognition
    - Brightness/contrast shifts simulate different lighting conditions in-store
    - Perspective transforms simulate the jewellery being viewed at slight angles
    - HueSaturation is kept subtle to preserve gold/silver tones
    - CoarseDropout (cutout) forces the model to focus on shape not texture

    Args:
        level: One of 'light', 'moderate', 'heavy'

    Returns:
        albumentations Compose pipeline
    """
    cfg = AUGMENTATION_LEVELS[level]
    p = cfg["p"]

    return A.Compose([
        # ── Geometric transforms ──────────────────────────────────────────
        A.HorizontalFlip(p=0.5),

        A.Rotate(
            limit=cfg["rotate_limit"],
            border_mode=cv2.BORDER_CONSTANT,
            value=0,            # fill with transparent (0) for BGRA images
            p=p
        ),

        A.Affine(
            scale=(1.0 - cfg["scale_limit"], 1.0 + cfg["scale_limit"]),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            shear=(-8, 8),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            p=p
        ),

        A.Perspective(
            scale=(0.02, 0.06),
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=0,
            p=p * 0.6          # perspective is strong — apply less often
        ),

        # ── Photometric transforms ────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=p
        ),

        A.HueSaturationValue(
            hue_shift_limit=8,          # small: preserve gold/silver tones
            sat_shift_limit=25,
            val_shift_limit=25,
            p=p * 0.7
        ),

        A.GaussNoise(
            var_limit=(5.0, 30.0),      # simulate camera sensor noise
            p=p * 0.4
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=p * 0.3
        ),

        A.ImageCompression(
            quality_lower=70,
            quality_upper=100,
            p=p * 0.3
        ),

        # ── Cutout / occlusion simulation ─────────────────────────────────
        # Simulates partial occlusion (hair, clothing) during try-on
        A.CoarseDropout(
            max_holes=4,
            max_height=20,
            max_width=20,
            min_holes=1,
            fill_value=0,
            p=p * 0.3
        ),

        # ── Normalisation-ready channel swap ──────────────────────────────
        # We keep this OFF here — normalisation happens in the DataLoader
    ])


def augment_manual(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Manual augmentation fallback when albumentations is not installed.
    Supports BGRA (4-channel) images natively.

    Args:
        image: Input image as numpy array (BGR or BGRA)
        seed:  Random seed for reproducibility

    Returns:
        Augmented image as numpy array
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h, w = image.shape[:2]
    aug = image.copy()

    # Horizontal flip
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)

    # Rotation
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Scale + translate
    scale = random.uniform(0.8, 1.2)
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    M2 = np.float32([[scale, 0, tx], [0, scale, ty]])
    aug = cv2.warpAffine(aug, M2, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Brightness (apply only to BGR channels, not alpha)
    if aug.shape[2] == 4:
        bgr, alpha = aug[:, :, :3], aug[:, :, 3:]
    else:
        bgr, alpha = aug, None

    beta = random.uniform(-40, 40)
    bgr = np.clip(bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if alpha is not None:
        aug = np.concatenate([bgr, alpha], axis=2)
    else:
        aug = bgr

    return aug


# ---------------------------------------------------------------------------
# BACKGROUND HELPERS
# ---------------------------------------------------------------------------

def remove_white_background(image: np.ndarray, threshold: int = 230) -> np.ndarray:
    """
    Converts white/near-white pixels to transparent (alpha = 0).
    Required for PNG jewelry images shot on white studio backgrounds.

    Args:
        image:     Input image (BGR or BGRA)
        threshold: Pixel value above which all channels must be to be masked

    Returns:
        BGRA image with white background removed
    """
    if image is None:
        raise ValueError("Input image is None")

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create mask where ALL channels exceed threshold
    white_mask = (
        (image[:, :, 0] > threshold) &
        (image[:, :, 1] > threshold) &
        (image[:, :, 2] > threshold)
    )
    image[white_mask, 3] = 0
    return image


def composite_on_random_background(
    jewelry_bgra: np.ndarray,
    bg_size: tuple = (512, 512)
) -> np.ndarray:
    """
    Composites the jewelry BGRA image onto a randomly-coloured background.
    This is important for training — the model must work on any background,
    not just white studio shots.

    Background types:
    - Solid neutral colours (mimicking skin tones or clothing)
    - Simple gradient backgrounds
    - Uniform noise (stress test)

    Args:
        jewelry_bgra: Jewelry image with transparent background (H, W, 4)
        bg_size:      (width, height) of output background canvas

    Returns:
        BGR image (3 channels) with jewelry composited on random background
    """
    bg_w, bg_h = bg_size

    bg_type = random.choice(["solid", "gradient", "skin"])

    if bg_type == "solid":
        # Neutral grey/beige tones
        colour = [random.randint(160, 240)] * 3
        background = np.full((bg_h, bg_w, 3), colour, dtype=np.uint8)

    elif bg_type == "gradient":
        # Vertical gradient from light to slightly darker
        top_val = random.randint(180, 240)
        bot_val = random.randint(130, top_val)
        background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        for row in range(bg_h):
            val = int(top_val - (top_val - bot_val) * row / bg_h)
            background[row, :] = [val, val, val]

    else:  # skin tones
        # Approximate human skin-tone palette (BGR)
        skin_tones = [
            (175, 190, 220),  # fair
            (140, 165, 195),  # medium
            (100, 120, 155),  # tan
            (75,  95,  125),  # dark
        ]
        base = random.choice(skin_tones)
        noise = np.random.randint(-15, 15, (bg_h, bg_w, 3))
        background = np.clip(
            np.full((bg_h, bg_w, 3), base, dtype=np.int16) + noise,
            0, 255
        ).astype(np.uint8)

    # Resize jewelry to fit within background (centred)
    jh, jw = jewelry_bgra.shape[:2]
    scale = min(bg_w / jw, bg_h / jh) * random.uniform(0.6, 0.9)
    new_w = max(1, int(jw * scale))
    new_h = max(1, int(jh * scale))
    jewelry_resized = cv2.resize(jewelry_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place at random position (ensuring it fits)
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    ox = random.randint(0, max_x)
    oy = random.randint(0, max_y)

    # Alpha blend
    alpha = jewelry_resized[:, :, 3].astype(np.float32) / 255.0
    alpha_3d = alpha[:, :, np.newaxis]
    fg = jewelry_resized[:, :, :3].astype(np.float32)
    bg_region = background[oy:oy+new_h, ox:ox+new_w].astype(np.float32)
    blended = alpha_3d * fg + (1 - alpha_3d) * bg_region
    background[oy:oy+new_h, ox:ox+new_w] = np.clip(blended, 0, 255).astype(np.uint8)

    return background


# ---------------------------------------------------------------------------
# DATASET BUILDER
# ---------------------------------------------------------------------------

class JewelryDatasetBuilder:
    """
    Builds a structured, augmented dataset from raw jewelry PNG images.

    Directory layout produced:
        output_dir/
            chain/
                original/
                augmented/
            jhumka/
                original/
                augmented/
            ... (one folder per category)
            dataset_manifest.json   ← lists all files with category + split labels

    The manifest is used by the PyTorch DataLoader in train.py.
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.manifest = []
        self.stats = {}

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image, converting to BGRA and removing white background."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")

        # Ensure 4 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        img = remove_white_background(img)
        return img

    def _save_image(self, image: np.ndarray, path: Path, as_bgr: bool = False):
        """Save image; use PNG for BGRA, JPG for composited BGR."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_bgr:
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(path), image)

    def _get_source_images(self, category: str, folder: str) -> list:
        """Collect all valid source images for a category."""
        folder_path = self.input_dir / folder
        if not folder_path.exists():
            print(f"[SKIP] Folder not found: {folder_path}")
            return []

        images = []
        for f in folder_path.iterdir():
            if f.suffix.lower() in SUPPORTED_EXTS and f.name.lower() != "cover.jpeg":
                images.append(f)

        print(f"  Found {len(images)} source images in '{folder}'")
        return images

    def build_category(self, category: str, cfg: dict):
        """
        Augment a single category to reach target_count samples.

        Strategy:
        1. Copy originals as-is (both BGRA and composited on random bg)
        2. Repeatedly apply augmentation until target_count is reached
        3. Alternate between albumentations and manual augmentation
        4. Mix: 50% BGRA (for overlay training) + 50% composited (for detector training)
        """
        print(f"\n[{category.upper()}] Building augmented dataset...")

        folder = cfg["folder"]
        target = cfg["target_count"]
        sources = self._get_source_images(category, folder)

        if not sources:
            return

        orig_dir = self.output_dir / category / "original"
        aug_dir  = self.output_dir / category / "augmented"
        orig_dir.mkdir(parents=True, exist_ok=True)
        aug_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Copy originals ────────────────────────────────────────
        original_records = []
        for src in sources:
            img_bgra = self._load_image(src)
            dest = orig_dir / (src.stem + ".png")
            self._save_image(img_bgra, dest)

            # Also save a composited version for detection training
            comp = composite_on_random_background(img_bgra)
            comp_dest = orig_dir / (src.stem + "_comp.jpg")
            self._save_image(comp, comp_dest, as_bgr=True)

            original_records.append({
                "path": str(dest.relative_to(self.output_dir)),
                "category": category,
                "keypoint_type": cfg["keypoint_type"],
                "split": "train",
                "augmented": False,
            })

        self.manifest.extend(original_records)

        # ── Step 2: Augment until target ──────────────────────────────────
        augmented_count = len(sources)  # we already have originals
        aug_index = 0

        if HAS_ALBUMENTATIONS:
            pipeline_mod  = build_albumentations_pipeline("moderate")
            pipeline_heavy = build_albumentations_pipeline("heavy")
        else:
            pipeline_mod = pipeline_heavy = None

        pbar = tqdm(total=target, initial=augmented_count,
                    desc=f"  Augmenting {category}")

        while augmented_count < target:
            src = random.choice(sources)
            img_bgra = self._load_image(src)

            # Choose augmentation approach
            use_heavy = (random.random() < 0.4)

            if HAS_ALBUMENTATIONS:
                # albumentations works on uint8 numpy arrays
                # Apply to BGR channels; reattach alpha afterwards
                bgr  = img_bgra[:, :, :3]
                alpha = img_bgra[:, :, 3]

                pipeline = pipeline_heavy if use_heavy else pipeline_mod
                result  = pipeline(image=bgr)
                aug_bgr = result["image"]

                # Augment alpha separately with same geometric params
                # (simplified: use same flip/rotate as bgr)
                aug_alpha = pipeline(image=np.stack([alpha, alpha, alpha], axis=2))["image"][:, :, 0]
                aug_bgra  = np.dstack([aug_bgr, aug_alpha])
            else:
                aug_bgra = augment_manual(img_bgra, seed=aug_index)

            # Save as BGRA PNG (used for overlay placement)
            fname_bgra = aug_dir / f"{category}_{aug_index:05d}.png"
            self._save_image(aug_bgra, fname_bgra)

            # Save composited version (used for category classification / detector)
            aug_comp = composite_on_random_background(aug_bgra)
            fname_comp = aug_dir / f"{category}_{aug_index:05d}_comp.jpg"
            self._save_image(aug_comp, fname_comp, as_bgr=True)

            # Determine split (80/10/10 train/val/test)
            r = random.random()
            split = "train" if r < 0.80 else ("val" if r < 0.90 else "test")

            self.manifest.append({
                "path": str(fname_comp.relative_to(self.output_dir)),
                "category": category,
                "keypoint_type": cfg["keypoint_type"],
                "split": split,
                "augmented": True,
                "source": src.name,
            })

            aug_index += 1
            augmented_count += 1
            pbar.update(1)

        pbar.close()

        self.stats[category] = {
            "source_count": len(sources),
            "total_generated": augmented_count,
            "target": target,
        }
        print(f"  Done: {augmented_count} samples generated (target: {target})")

    def build_all(self):
        """Run augmentation for all configured categories."""
        print("=" * 60)
        print("  Jewelry VTO — Data Augmentation Pipeline")
        print("=" * 60)

        for category, cfg in CATEGORY_CONFIG.items():
            self.build_category(category, cfg)

        # Save manifest
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "categories": list(CATEGORY_CONFIG.keys()),
                "category_config": CATEGORY_CONFIG,
                "total_samples": len(self.manifest),
                "stats": self.stats,
                "samples": self.manifest,
            }, f, indent=2)

        print("\n" + "=" * 60)
        print(f"  Manifest saved: {manifest_path}")
        print(f"  Total samples:  {len(self.manifest)}")
        for cat, st in self.stats.items():
            print(f"    {cat:15s}: {st['source_count']:3d} → {st['total_generated']:4d} samples")
        print("=" * 60)


# ---------------------------------------------------------------------------
# KEYPOINT ANNOTATION HELPER (for Stage 3 training)
# ---------------------------------------------------------------------------

def generate_pseudo_keypoints(
    image_bgra: np.ndarray,
    keypoint_type: str
) -> dict:
    """
    Generates approximate keypoint annotations from the alpha-channel mask.
    These are used as weak supervision for training the keypoint detection head.

    For real training, you should annotate 30–50 images per category manually
    using LabelMe or CVAT, then use these pseudo-labels for the rest.

    Keypoint definitions:
        necklace / chain / mangalsutra:
            - 'top_left':  top-left attachment point
            - 'top_right': top-right attachment point
            - 'bottom':    pendant / lowest point

        earring / jhumka:
            - 'hook_left':  left earring hook point
            - 'hook_right': right earring hook point

        ring:
            - 'center':    ring centre
            - 'top':       top of ring (finger insertion)
            - 'bottom':    bottom of ring

    Args:
        image_bgra:    Jewelry image with transparent background
        keypoint_type: One of 'necklace', 'earring', 'ring'

    Returns:
        Dict mapping keypoint name → (x, y) in pixel coordinates, normalised [0,1]
    """
    alpha = image_bgra[:, :, 3]
    h, w = alpha.shape

    # Find bounding box of non-transparent region
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        return {}

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    if keypoint_type == "necklace":
        return {
            "top_left":  (x_min / w,        y_min / h),
            "top_right": (x_max / w,        y_min / h),
            "bottom":    (cx / w,           y_max / h),
            "center":    (cx / w,           cy / h),
        }
    elif keypoint_type == "earring":
        return {
            "hook_left":  (x_min / w + 0.25 * (x_max - x_min) / w, y_min / h),
            "hook_right": (x_min / w + 0.75 * (x_max - x_min) / w, y_min / h),
            "bottom":     (cx / w,                                   y_max / h),
        }
    elif keypoint_type == "ring":
        return {
            "center": (cx / w,  cy / h),
            "top":    (cx / w,  y_min / h),
            "bottom": (cx / w,  y_max / h),
            "left":   (x_min / w, cy / h),
            "right":  (x_max / w, cy / h),
        }
    else:
        return {"center": (cx / w, cy / h)}


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jewelry VTO — Data Augmentation Pipeline (Stage 2)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./static",
        help="Root static folder containing chain/, jhumka/, etc."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset/augmented",
        help="Output directory for augmented dataset"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process only this category (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    builder = JewelryDatasetBuilder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    if args.category:
        if args.category not in CATEGORY_CONFIG:
            print(f"[ERROR] Unknown category '{args.category}'")
            print(f"        Valid: {list(CATEGORY_CONFIG.keys())}")
        else:
            cfg = CATEGORY_CONFIG[args.category]
            builder.build_category(args.category, cfg)
            builder._save_manifest()
    else:
        builder.build_all()
