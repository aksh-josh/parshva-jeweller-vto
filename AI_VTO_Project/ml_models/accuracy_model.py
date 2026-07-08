"""
accuracy_model.py
=================
Trained Accuracy Quality Model — EfficientNet-B0 + Quality Head

What this model does:
    Takes a rendered VTO frame (with jewelry overlaid) and predicts
    a placement quality score from 0.0 to 1.0.

    0.0 = very poor  (jewelry completely wrong position)
    0.5 = acceptable (jewelry roughly in the right area)
    1.0 = excellent  (jewelry perfectly placed, visually natural)

Why a learned model on top of the rule-based system:
    The rule-based system measures geometric accuracy (PCK, IoU, positional error).
    This model measures PERCEPTUAL quality — does it LOOK correct? It catches:
      • Perspective mismatch (jewelry flat on tilted face)
      • Lighting mismatch (jewelry too bright/dark vs scene)
      • Earring asymmetry (left and right earrings at different heights)
      • Ring perspective (flat ring on curved finger)
    These factors cannot be captured by landmark math alone.

Architecture:
    EfficientNet-B0 (4.0M params, pre-trained ImageNet)
        → Global Average Pool → 1280-d feature vector
        → Linear(1280 → 256) → BatchNorm → GELU → Dropout(0.3)
        → Linear(256 → 64)   → GELU
        → Linear(64  → 1)    → Sigmoid
    Output: scalar in [0, 1]

Why EfficientNet-B0 (not B2):
    The quality model only needs to judge overall visual coherence,
    not detect precise keypoints. B0 (4M params) runs in ~2ms on CPU —
    fast enough for 30fps live use. B2 would be 40ms/frame.

Training:
    See accuracy_train.py — trains on Colab T4 GPU, ~15 minutes.
    Produces accuracy_model_best.pth (~17 MB).

Inference:
    See accuracy_inference.py — loads checkpoint, runs on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class AccuracyQualityModel(nn.Module):
    """
    EfficientNet-B0 based placement quality predictor.

    Input:  (B, 3, 256, 256) normalised RGB frame WITH jewelry rendered
    Output: (B, 1) quality score in [0, 1]
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # EfficientNet-B0 backbone — features_only=False to get 1280-d vector
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool="avg",      # global average pooling → 1280-d
        )
        backbone_out = 1280

        # Freeze all backbone layers initially — unfreeze in Phase 2
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Quality prediction head
        self.head = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Category embedding (optional conditioning)
        # Allows the model to apply different quality standards per category
        self.CATEGORIES = ["chain", "jhumka", "mangalsutra", "necklace", "ring"]
        self.cat_embed = nn.Embedding(len(self.CATEGORIES), 32)
        self.cat_fc    = nn.Linear(1280 + 32, 1280)

        self.cat_to_idx = {c: i for i, c in enumerate(self.CATEGORIES)}

    def unfreeze_last_n(self, n: int = 3):
        """Unfreeze last n blocks of EfficientNet backbone for fine-tuning."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AccuracyModel] Unfroze last {n} backbone blocks. Trainable: {n_train:,}")

    def forward(
        self,
        frame:    torch.Tensor,
        category: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame:    (B, 3, 256, 256) rendered VTO frame (with jewelry)
            category: (B,) LongTensor of category indices (optional)

        Returns:
            (B, 1) quality score in [0, 1]
        """
        features = self.backbone(frame)   # (B, 1280)

        # Optionally condition on jewelry category
        if category is not None:
            cat_emb = self.cat_embed(category)          # (B, 32)
            fused   = torch.cat([features, cat_emb], dim=1)  # (B, 1312)
            features = F.gelu(self.cat_fc(fused))            # (B, 1280)

        score = self.head(features)       # (B, 1)
        return score

    def predict_score(self, frame: torch.Tensor, category_name: str = "necklace") -> float:
        """
        Convenience method for single-frame inference.

        Args:
            frame:         (1, 3, 256, 256) or (3, 256, 256) tensor
            category_name: jewelry category string

        Returns:
            Float quality score in [0, 1]
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        cat_idx = self.cat_to_idx.get(category_name, 3)  # default to necklace
        cat_t   = torch.tensor([cat_idx], device=frame.device)
        with torch.no_grad():
            score = self.forward(frame, cat_t)
        return float(score.item())

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class CombinedAccuracyLoss(nn.Module):
    """
    Hybrid loss: 70% MSE (continuous score) + 30% BCE (good/bad boundary).

    The MSE component trains the model to predict continuous quality scores.
    The BCE component adds stronger gradient signal at the 0.5 decision boundary,
    helping the model learn to distinguish "acceptable" from "unacceptable" placement.

    Labels:
        quality_score: float in [0, 1] — rule-based score Q_rule
        binary_label:  1 if quality_score > 0.65, else 0
    """

    def __init__(self, mse_weight: float = 0.7, bce_weight: float = 0.3):
        super().__init__()
        self.mse_w = mse_weight
        self.bce_w = bce_weight
        self.mse   = nn.MSELoss()
        self.bce   = nn.BCELoss()

    def forward(
        self,
        predictions: torch.Tensor,   # (B, 1) model output [0, 1]
        quality_scores: torch.Tensor, # (B,) continuous labels [0, 1]
        binary_labels: torch.Tensor,  # (B,) binary labels {0, 1}
    ) -> dict:
        pred_flat   = predictions.squeeze(1)   # (B,)
        binary_flat = binary_labels.float()

        mse_loss = self.mse(pred_flat, quality_scores)
        bce_loss = self.bce(pred_flat, binary_flat)
        total    = self.mse_w * mse_loss + self.bce_w * bce_loss

        return {
            "total": total,
            "mse":   mse_loss,
            "bce":   bce_loss,
        }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== AccuracyQualityModel — Sanity Check ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = AccuracyQualityModel(pretrained=False).to(device)
    params = model.count_parameters()
    print(f"Total params:     {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}  (backbone frozen)")

    model.unfreeze_last_n(3)
    params = model.count_parameters()
    print(f"Trainable params: {params['trainable']:,}  (after unfreeze)")

    # Forward pass
    frames = torch.randn(4, 3, 256, 256).to(device)
    cats   = torch.randint(0, 5, (4,)).to(device)
    scores = model(frames, cats)
    print(f"\nOutput shape: {tuple(scores.shape)}")
    print(f"Score range:  {scores.min().item():.3f} – {scores.max().item():.3f}")
    assert scores.shape == (4, 1), "Wrong output shape"
    assert 0.0 <= scores.min().item() <= 1.0, "Score out of range"

    # Loss check
    criterion     = CombinedAccuracyLoss()
    q_labels      = torch.rand(4).to(device)
    b_labels      = (q_labels > 0.65).float().to(device)
    losses        = criterion(scores, q_labels, b_labels)
    print(f"\nLoss — total: {losses['total'].item():.4f}  "
          f"mse: {losses['mse'].item():.4f}  bce: {losses['bce'].item():.4f}")

    # Single-frame predict
    single = torch.randn(1, 3, 256, 256).to(device)
    s      = model.predict_score(single, "necklace")
    print(f"\nSingle-frame score: {s:.3f}")

    print("\n[OK] AccuracyQualityModel verified.")
