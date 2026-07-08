"""
advanced_train.py
=================
Advanced Training Script — EfficientNet-B2 + HRNet

Advanced training features used and WHY:

┌──────────────────────────┬─────────────────────────────────────────────┐
│ Technique                │ Why                                         │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Mixed Precision (AMP)    │ FP16 forward/backward → 2× faster, 2× less │
│ torch.cuda.amp           │ VRAM. Critical for Colab T4 GPU (15GB)      │
├──────────────────────────┼─────────────────────────────────────────────┤
│ EMA (Exponential Moving  │ Smoothed model weights → more stable, often │
│ Average) of weights      │ 0.5–1% better than checkpoint weights       │
├──────────────────────────┼─────────────────────────────────────────────┤
│ OneCycleLR scheduler     │ Fast warmup → peak LR → cosine decay.       │
│                          │ Trains to peak accuracy in fewer epochs      │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Gradient clipping        │ Prevents exploding gradients during          │
│ (max_norm=5.0)           │ backbone unfreeze phase                      │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Warmup (5 epochs)        │ Prevents early large gradient updates from  │
│                          │ destroying pre-trained backbone weights      │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 3-Phase training         │ Phase1: heads only (frozen backbone)        │
│                          │ Phase2: last 3 stages unfrozen              │
│                          │ Phase3: full model with low LR              │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Heatmap-based PCK        │ PCK computed from heatmap peaks not from    │
│ evaluation               │ regression output → accurate metric          │
└──────────────────────────┴─────────────────────────────────────────────┘

Colab usage:
    !python advanced_train.py \\
        --data_dir ./dataset/augmented \\
        --output_dir ./checkpoints_v2 \\
        --epochs 40 \\
        --batch_size 16 \\
        --use_amp
"""

import os
import json
import time
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from advanced_model import (
    AdvancedJewelryVTOModel, AdvancedMultiTaskLoss,
    CATEGORIES, N_CLASSES, MAX_KEYPOINTS, INPUT_SIZE,
)

CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class AdvancedJewelryDataset(Dataset):
    """
    Dataset for advanced model training.
    Reads dataset_manifest.json from data_augmentation.py.
    """

    def __init__(self, manifest_path: str, split: str, transform=None):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.root      = Path(manifest_path).parent
        self.transform = transform
        self.samples   = [s for s in manifest["samples"] if s["split"] == split]
        from model import KEYPOINTS_PER_TYPE
        self.kp_per_type = KEYPOINTS_PER_TYPE
        print(f"  [{split}] {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.root / s["path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self)-1))
        if self.transform:
            img = self.transform(img)

        cat     = s["category"]
        kp_type = s.get("keypoint_type", "necklace")
        n_active = self.kp_per_type.get(kp_type, 1)

        kp_coords  = torch.zeros(MAX_KEYPOINTS, 2)
        kp_validity = torch.zeros(MAX_KEYPOINTS)
        kp_validity[:n_active] = 1.0

        return img, {
            "class_labels": torch.tensor(CATEGORY_TO_IDX[cat], dtype=torch.long),
            "kp_coords":    kp_coords,
            "kp_validity":  kp_validity,
        }


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),  # cutout
        ])
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = {
        "class_labels": torch.stack([b[1]["class_labels"] for b in batch]),
        "kp_coords":    torch.stack([b[1]["kp_coords"]    for b in batch]),
        "kp_validity":  torch.stack([b[1]["kp_validity"]  for b in batch]),
    }
    return imgs, targets


# ---------------------------------------------------------------------------
# EXPONENTIAL MOVING AVERAGE
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.

    EMA_weight = decay × EMA_weight + (1 - decay) × current_weight

    At evaluation time, swap to EMA weights for inference.
    This typically gives 0.5–1% better accuracy than raw checkpoint weights.

    decay=0.9999 is standard for image classification.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )

    def apply(self, model: nn.Module):
        """Apply EMA weights to model for evaluation."""
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module, original_state: dict):
        """Restore original weights after evaluation."""
        model.load_state_dict(original_state)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_pck_from_heatmaps(
    pred_heatmaps: torch.Tensor,   # (B, N_kp, H, W)
    gt_coords:     torch.Tensor,   # (B, N_kp, 2) normalised [0,1]
    validity:      torch.Tensor,   # (B, N_kp)
    threshold:     float = 0.05,
) -> float:
    """
    PCK computed from heatmap argmax peaks.
    More accurate than soft-argmax for evaluation.
    """
    B, N, H, W = pred_heatmaps.shape
    flat_idx   = pred_heatmaps.view(B, N, -1).argmax(dim=-1)   # (B, N)
    pred_y     = (flat_idx // W).float() / H                    # normalised y
    pred_x     = (flat_idx %  W).float() / W                    # normalised x
    pred_coords = torch.stack([pred_x, pred_y], dim=-1)         # (B, N, 2)

    dist    = torch.norm(pred_coords.cpu() - gt_coords.cpu(), dim=-1)  # (B, N)
    correct = (dist < threshold).float() * validity.cpu()
    n_valid = validity.sum().item()
    return float(correct.sum() / max(n_valid, 1))


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# ADVANCED TRAINER
# ---------------------------------------------------------------------------

class AdvancedTrainer:
    """
    3-Phase training engine:

    Phase 1 (epochs 1–10):
        - Backbone fully frozen
        - Train only HRNet head + CosFace + embedding proj
        - High LR for heads (1e-3), warmup 2 epochs

    Phase 2 (epochs 11–25):
        - Unfreeze last 3 EfficientNet stages
        - Differential LRs: backbone 1e-5, heads 1e-4
        - Mixed precision (AMP) for speed

    Phase 3 (epochs 26–40):
        - Unfreeze entire backbone
        - Very low uniform LR (5e-6)
        - EMA active
    """

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"\n[AdvancedTrainer] Device: {self.device}")
        print(f"  Mixed precision (AMP): {args.use_amp and self.device.type=='cuda'}")

        self.use_amp = args.use_amp and (self.device.type == "cuda")
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._build_dataloaders()
        self._build_model()

        self.history   = {k: [] for k in [
            "train_loss", "val_loss", "train_acc", "val_acc",
            "train_pck",  "val_pck",
        ]}
        self.best_val_acc = 0.0
        self.phase        = 1

    def _build_dataloaders(self):
        manifest = os.path.join(self.args.data_dir, "dataset_manifest.json")
        assert os.path.exists(manifest), (
            f"dataset_manifest.json not found at {manifest}\n"
            "Run data_augmentation.py first."
        )
        train_ds = AdvancedJewelryDataset(manifest, "train", get_transforms("train"))
        val_ds   = AdvancedJewelryDataset(manifest, "val",   get_transforms("val"))

        self.train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=collate_fn,
            pin_memory=(self.device.type == "cuda"), drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=collate_fn,
        )

    def _build_model(self):
        self.model = AdvancedJewelryVTOModel(
            pretrained=True, freeze_stages=6,    # freeze all 6 stages initially
        ).to(self.device)
        self.criterion = AdvancedMultiTaskLoss(
            lambda_hm=1.0, lambda_cls=1.0, lambda_coord=0.3, lambda_val=0.5
        )
        self.ema = EMA(self.model, decay=0.9999)
        self._setup_optimizer_phase1()

    def _setup_optimizer_phase1(self):
        """Phase 1: train heads only."""
        head_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(head_params, lr=self.args.lr_head, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr_head,
            steps_per_epoch=len(self.train_loader),
            epochs=self.args.phase1_epochs,
            pct_start=0.2,    # 20% warmup
        )
        print("[Phase 1] Heads-only training with OneCycleLR.")

    def _setup_optimizer_phase2(self):
        """Phase 2: differential LRs for backbone vs heads."""
        self.model.backbone.unfreeze_all()
        # Refreeze early stages (keep stem + first 2 stages frozen)
        children = list(self.model.backbone.backbone.children())
        for child in children[:3]:
            for p in child.parameters():
                p.requires_grad = False

        backbone_params = [p for n, p in self.model.named_parameters()
                           if "backbone" in n and p.requires_grad]
        head_params     = [p for n, p in self.model.named_parameters()
                           if "backbone" not in n and p.requires_grad]
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.args.lr_backbone},
            {"params": head_params,     "lr": self.args.lr_head},
        ], weight_decay=1e-4)
        remaining = self.args.phase2_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=remaining * len(self.train_loader), eta_min=1e-7
        )
        print("[Phase 2] Differential LRs: backbone=1e-5, heads=1e-4.")

    def _setup_optimizer_phase3(self):
        """Phase 3: full fine-tune at very low LR."""
        self.model.backbone.unfreeze_all()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=5e-6, weight_decay=1e-4
        )
        remaining = self.args.epochs - self.args.phase1_epochs - self.args.phase2_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, remaining) * len(self.train_loader), eta_min=1e-8
        )
        print("[Phase 3] Full fine-tune at LR=5e-6.")

    def _run_epoch(self, loader, phase: str) -> dict:
        is_train = (phase == "train")
        self.model.train(is_train)

        total_loss = total_acc = total_pck = n = 0

        with torch.set_grad_enabled(is_train):
            for imgs, targets in tqdm(loader, desc=f"  {phase}", leave=False):
                imgs    = imgs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                labels  = targets["class_labels"]

                # Mixed precision forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        preds  = self.model(imgs, labels if is_train else None)
                        losses = self.criterion(preds, targets)
                else:
                    preds  = self.model(imgs, labels if is_train else None)
                    losses = self.criterion(preds, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(losses["total"]).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        losses["total"].backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.ema.update(self.model)

                acc = compute_accuracy(preds["class_logits"], labels)
                pck = compute_pck_from_heatmaps(
                    preds["heatmaps"], targets["kp_coords"], targets["kp_validity"]
                )
                total_loss += losses["total"].item()
                total_acc  += acc
                total_pck  += pck
                n          += 1

        return {"loss": total_loss/n, "acc": total_acc/n, "pck": total_pck/n}

    def _save(self, epoch: int, val_acc: float, is_best: bool):
        state = {
            "epoch": epoch, "phase": self.phase,
            "model_state": self.model.state_dict(),
            "ema_state":   self.ema.shadow,
            "val_acc": val_acc, "history": self.history,
        }
        torch.save(state, self.output_dir / "latest.pth")
        if is_best:
            torch.save(state, self.output_dir / "best.pth")
            print(f"    [*] Best model saved (val_acc={val_acc:.4f})")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        print(f"\n[AdvancedTrainer] {self.args.epochs} total epochs")
        p1_end = self.args.phase1_epochs
        p2_end = p1_end + self.args.phase2_epochs

        for epoch in range(1, self.args.epochs + 1):
            # Phase transitions
            if epoch == p1_end + 1 and self.phase == 1:
                self.phase = 2
                self._setup_optimizer_phase2()
            elif epoch == p2_end + 1 and self.phase == 2:
                self.phase = 3
                self._setup_optimizer_phase3()

            t0         = time.time()
            tr         = self._run_epoch(self.train_loader, "train")

            # EMA evaluation
            orig_state = copy.deepcopy(self.model.state_dict())
            self.ema.apply(self.model)
            va         = self._run_epoch(self.val_loader, "val")
            self.ema.restore(self.model, orig_state)

            for k in ["loss", "acc", "pck"]:
                self.history[f"train_{k}"].append(tr[k])
                self.history[f"val_{k}"].append(va[k])

            is_best = va["acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = va["acc"]
            self._save(epoch, va["acc"], is_best)

            elapsed = time.time() - t0
            print(
                f"  Ep {epoch:3d}/{self.args.epochs} [P{self.phase}] | "
                f"Loss {tr['loss']:.3f}/{va['loss']:.3f} | "
                f"Acc {tr['acc']:.3f}/{va['acc']:.3f} | "
                f"PCK {tr['pck']:.3f}/{va['pck']:.3f} | "
                f"{elapsed:.0f}s"
            )

        print(f"\n[AdvancedTrainer] Done. Best val_acc: {self.best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="./dataset/augmented")
    p.add_argument("--output_dir",     default="./checkpoints_v2")
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--phase1_epochs",  type=int,   default=10)
    p.add_argument("--phase2_epochs",  type=int,   default=15)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr_head",        type=float, default=1e-3)
    p.add_argument("--lr_backbone",    type=float, default=1e-5)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--use_amp",        action="store_true",
                   help="Use mixed precision (GPU only)")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = AdvancedTrainer(args)
    trainer.train()
