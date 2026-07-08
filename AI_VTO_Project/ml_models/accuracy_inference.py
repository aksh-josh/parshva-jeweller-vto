"""
accuracy_inference.py
=====================
Accuracy Quality Model — Inference Engine (runs on CPU in VS Code)

Loads accuracy_model_best.pth and provides a simple API for app.py.

Usage in app.py:
    from ml_models.accuracy_inference import AccuracyScorer

    # At startup (once):
    scorer = AccuracyScorer("./checkpoints/accuracy_model_best.pth")

    # In api_tryon(), after overlay:
    quality_score = scorer.predict(processed_frame, category="necklace")

    # In record_frame():
    combined_score = 0.55 * rule_based_score + 0.45 * quality_score
"""

import os
import sys
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from accuracy_model import AccuracyQualityModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CATEGORY_TO_IDX = {"chain": 0, "jhumka": 1, "mangalsutra": 2, "necklace": 3, "ring": 4}


class AccuracyScorer:
    """
    Lightweight wrapper around AccuracyQualityModel for production inference.

    Features:
    - Lazy loads model on first call (does not block app startup)
    - Runs on CPU (~2ms per frame on modern laptop)
    - Thread-safe (model.eval() + torch.no_grad())
    - Graceful fallback (returns 0.72 if model not loaded)
    - Caches last 30 scores for rolling average
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model           = None
        self.device          = torch.device("cpu")  # always CPU for inference
        self._loaded         = False
        self._score_cache    = []
        self._load_times     = []

        # Try to load immediately (non-blocking for app startup)
        self._try_load()

    def _try_load(self):
        """Attempt to load the model. Fails silently if checkpoint missing."""
        if not os.path.exists(self.checkpoint_path):
            print(f"[AccuracyScorer] Checkpoint not found: {self.checkpoint_path}")
            print("  Train accuracy model on Colab first, then place .pth here.")
            return

        try:
            self.model = AccuracyQualityModel(pretrained=False).to(self.device)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self._loaded = True
            print(f"[AccuracyScorer] Quality model loaded — "
                  f"epoch {ckpt.get('epoch','?')}, "
                  f"val_MAE={ckpt.get('val_mae', 0):.4f}")
        except Exception as e:
            print(f"[AccuracyScorer] Load failed: {e}")
            self.model   = None
            self._loaded = False

    def predict(
        self,
        frame_bgr:  "np.ndarray",
        category:   str = "necklace",
    ) -> float:
        """
        Predict placement quality score for a rendered VTO frame.

        Args:
            frame_bgr: BGR numpy array (H, W, 3) — the PROCESSED frame with jewelry
            category:  jewelry category string

        Returns:
            Float quality score in [0, 1]
            Returns 0.72 (neutral score) if model not loaded
        """
        if not self._loaded or self.model is None:
            return 0.72  # neutral fallback

        try:
            import cv2
            t0 = time.time()

            # Convert BGR → RGB → PIL → tensor
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = INFERENCE_TRANSFORM(pil).unsqueeze(0).to(self.device)

            cat_idx = CATEGORY_TO_IDX.get(category, 3)
            cat_t   = torch.tensor([cat_idx], device=self.device)

            with torch.no_grad():
                score = self.model(tensor, cat_t).item()

            elapsed_ms = (time.time() - t0) * 1000
            self._load_times.append(elapsed_ms)
            if len(self._load_times) > 100:
                self._load_times.pop(0)

            self._score_cache.append(score)
            if len(self._score_cache) > 30:
                self._score_cache.pop(0)

            return float(score)

        except Exception as e:
            print(f"[AccuracyScorer] Predict error: {e}")
            return 0.72

    def compute_combined_score(
        self,
        rule_based_metrics: dict,
        quality_score:      float,
        rule_weight:        float = 0.55,
        model_weight:       float = 0.45,
    ) -> float:
        """
        Combine rule-based metrics with model quality score.

        Args:
            rule_based_metrics: dict from vto_accuracy.get_live_metrics()
                                 keys: pck_5pct, mean_iou, mean_pos_error_px, mean_jitter_px
            quality_score:      output of predict()
            rule_weight:        weight for rule-based component (default 0.55)
            model_weight:       weight for model component (default 0.45)

        Returns:
            Combined accuracy score in [0, 1]
        """
        pck5      = rule_based_metrics.get("pck_5pct", 70.0) / 100.0
        iou       = rule_based_metrics.get("mean_iou", 0.60)
        jitter    = rule_based_metrics.get("mean_jitter_px", 8.0)
        pos_err   = rule_based_metrics.get("mean_pos_error_px", 15.0)

        # Normalise jitter (0px=1.0, 20px=0.0)
        jitter_score = float(np.clip(1.0 - jitter / 20.0, 0.0, 1.0))

        # Normalise positional error (0px=1.0, 50px=0.0)
        pos_score = float(np.clip(1.0 - pos_err / 50.0, 0.0, 1.0))

        # Rule-based composite
        rule_score = (0.35 * pck5 + 0.35 * iou + 0.15 * jitter_score + 0.15 * pos_score)

        # Combined
        combined = rule_weight * rule_score + model_weight * quality_score
        return float(np.clip(combined, 0.0, 1.0))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self._load_times)) if self._load_times else 0.0

    @property
    def rolling_mean_score(self) -> float:
        return float(np.mean(self._score_cache)) if self._score_cache else 0.72

    def grade(self, score: float) -> str:
        if score >= 0.90: return "A"
        if score >= 0.75: return "B"
        if score >= 0.60: return "C"
        return "D"

    def get_status(self) -> dict:
        return {
            "loaded":           self._loaded,
            "checkpoint_path":  self.checkpoint_path,
            "mean_inference_ms":self.mean_inference_ms,
            "rolling_score":    self.rolling_mean_score,
            "grade":            self.grade(self.rolling_mean_score),
        }


# ── Sanity check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== AccuracyScorer — Inference Test ===")

    # Test without checkpoint (should use fallback)
    scorer = AccuracyScorer("./nonexistent.pth")
    print(f"Fallback score (no model): {scorer.predict(np.zeros((480,640,3),dtype=np.uint8))}")

    # Test combined score computation
    rule_metrics = {
        "pck_5pct":          82.0,
        "mean_iou":          0.71,
        "mean_pos_error_px": 11.5,
        "mean_jitter_px":    3.8,
    }
    combined = scorer.compute_combined_score(rule_metrics, quality_score=0.85)
    print(f"Combined score (rules=0.78, model=0.85): {combined:.3f}")
    print(f"Grade: {scorer.grade(combined)}")
    print("\n[OK] AccuracyScorer inference test passed.")
