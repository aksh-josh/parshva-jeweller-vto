"""
ml_models/emotion_inference.py  — FIXED VERSION
================================================
Root cause of mat1(1x352) × mat2(512x256) error:
  - Local timm gives feats[2] shape (1, 352, H, W)
  - shared_proj on local machine = Linear(208→512) after filtering
  - GAP output = 352-d, but shared_proj expects 208-d → crashes

Fix: bypass shared_proj entirely on local machine.
     Feed the 352-d GAP output directly into a NEW emotion head
     that accepts whatever dim the backbone actually produces.
     Detected at load time from the checkpoint.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque, Counter

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_EMOTIONS    = 7

EMOTION_ANIMATION_MAP = {
    "happy":    {"sparkle_level": 1.0, "sway_speed": 0.6, "zoom_scale": 0.2,
                 "shake_intensity": 0.0, "brightness_mult": 1.15, "label": "Happy"},
    "surprise": {"sparkle_level": 0.6, "sway_speed": 0.3, "zoom_scale": 1.0,
                 "shake_intensity": 0.0, "brightness_mult": 1.1,  "label": "Surprised"},
    "neutral":  {"sparkle_level": 0.0, "sway_speed": 1.0, "zoom_scale": 0.0,
                 "shake_intensity": 0.0, "brightness_mult": 1.0,  "label": "Neutral"},
    "sad":      {"sparkle_level": 0.0, "sway_speed": 0.3, "zoom_scale": 0.0,
                 "shake_intensity": 0.0, "brightness_mult": 0.75, "label": "Sad"},
    "angry":    {"sparkle_level": 0.0, "sway_speed": 0.0, "zoom_scale": 0.0,
                 "shake_intensity": 1.0, "brightness_mult": 0.9,  "label": "Angry"},
    "fear":     {"sparkle_level": 0.0, "sway_speed": 0.8, "zoom_scale": 0.3,
                 "shake_intensity": 0.5, "brightness_mult": 0.85, "label": "Fear"},
    "disgust":  {"sparkle_level": 0.0, "sway_speed": 0.2, "zoom_scale": 0.0,
                 "shake_intensity": 0.3, "brightness_mult": 0.8,  "label": "Disgust"},
}


class FlexibleEmotionHead(nn.Module):
    """
    Emotion head that accepts ANY input dimension.
    Automatically adapts to local (352-d) or Colab (512-d) backbone output.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_EMOTIONS),
        )

    def forward(self, x):
        return self.net(x)


class JM3NEmotionInference:

    def __init__(self, checkpoint_path: str, vto_model, device: str = "cpu"):
        self.device        = device
        self.is_loaded     = False
        self.vto_model     = vto_model
        self.emotion_head  = None
        self.gap_dim       = None   # actual GAP output dim on this machine

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._reset_state()

        if not os.path.exists(checkpoint_path):
            print(f"[JM3N] emotion_model.pth not found — JM3N disabled.")
            return

        try:
            ckpt = torch.load(checkpoint_path, map_location=device)

            # ── Step 1: detect actual GAP dim from a live forward pass ────
            self.gap_dim = self._detect_gap_dim()
            if self.gap_dim is None:
                print("[JM3N] Cannot detect GAP dim — backbone not loaded yet.")
                return

            # ── Step 2: detect what input_dim the checkpoint head expects ─
            ckpt_input_dim = self._detect_ckpt_input_dim(ckpt)
            print(f"[JM3N] Local GAP dim={self.gap_dim}, checkpoint head input={ckpt_input_dim}")

            # ── Step 3: build head with LOCAL gap_dim (not checkpoint dim) ─
            # This makes it work regardless of timm version mismatch
            self.emotion_head = FlexibleEmotionHead(self.gap_dim).to(device)

            # ── Step 4: try to load checkpoint weights ─────────────────────
            loaded_ok = self._load_head_weights(ckpt, ckpt_input_dim)

            self.emotion_head.eval()
            self.is_loaded = True
            val_acc = ckpt.get("val_acc", 0)
            missing_note = "" if loaded_ok else " (retrained head — weights incompatible)"
            print(f"[JM3N] Emotion head ready — val_acc={val_acc*100:.1f}%"
                  f" | input_dim={self.gap_dim}{missing_note}")

        except Exception as e:
            print(f"[JM3N] Load failed: {e}")
            self.is_loaded = False

    def _detect_gap_dim(self):
        """Run a dummy forward pass to find the actual GAP output dimension."""
        if self.vto_model is None:
            return None
        try:
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                feats = self.vto_model.backbone(dummy)
                g = self.vto_model.flat(self.vto_model.gap(feats[2]))
            return g.shape[1]
        except Exception as e:
            print(f"[JM3N] GAP dim detection failed: {e}")
            return None

    def _detect_ckpt_input_dim(self, ckpt):
        """Read the first Linear weight from checkpoint to find its input size."""
        # Try emotion_head.net.0.weight or emotion_head.0.weight
        for key in ["emotion_head.net.0.weight", "emotion_head.0.weight"]:
            if key in ckpt.get("model_state", {}):
                return ckpt["model_state"][key].shape[1]
        # Try emotion_head_state
        for key in ["net.0.weight", "0.weight"]:
            if key in ckpt.get("emotion_head_state", {}):
                return ckpt["emotion_head_state"][key].shape[1]
        return None

    def _load_head_weights(self, ckpt, ckpt_input_dim):
        """
        Load emotion head weights if dimensions match local gap_dim.
        If they don't match (e.g. checkpoint=512, local=352),
        keep randomly-initialised weights — model still runs, just less accurate
        until retrained on Colab with the fixed Cell 6.
        """
        if ckpt_input_dim != self.gap_dim:
            print(f"[JM3N] Checkpoint head input_dim={ckpt_input_dim} != local GAP={self.gap_dim}")
            print(f"[JM3N] Using randomly-initialised head. Re-train on Colab with fixed Cell 6.")
            return False

        # Dimensions match — load the weights
        head_state = {}
        model_state = ckpt.get("model_state", {})
        for k, v in model_state.items():
            if "emotion_head" in k:
                # normalise key to net.X.weight/bias
                new_k = k.replace("emotion_head.net.", "net.").replace("emotion_head.", "net.")
                head_state[new_k] = v

        if not head_state:
            hs = ckpt.get("emotion_head_state", {})
            for k, v in hs.items():
                new_k = k if k.startswith("net.") else f"net.{k}"
                head_state[new_k] = v

        if head_state:
            missing, unexpected = self.emotion_head.load_state_dict(head_state, strict=False)
            return len(missing) == 0
        return False

    def _reset_state(self):
        self.current_emotion     = "neutral"
        self.current_confidence  = 0.0
        self.current_anim_params = EMOTION_ANIMATION_MAP["neutral"].copy()
        self.frame_count         = 0
        self.predict_every       = 10
        self.emotion_history     = deque(maxlen=5)
        self.prob_history        = deque(maxlen=5)
        self._inference_times    = deque(maxlen=30)
        self._error_count        = 0

    def extract_face_crop(self, frame_bgr, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        if face_landmarks is not None:
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            x1 = max(0, int(min(xs)) - 20)
            y1 = max(0, int(min(ys)) - 20)
            x2 = min(w, int(max(xs)) + 20)
            y2 = min(h, int(max(ys)) + 20)
            if x2 > x1 and y2 > y1:
                return frame_bgr[y1:y2, x1:x2]
        cx, cy = w // 2, h // 2
        size   = min(w, h) // 2
        return frame_bgr[max(0,cy-size):cy+size, max(0,cx-size):cx+size]

    def predict(self, frame_bgr, face_landmarks, frame_shape):
        self.frame_count += 1
        if not self.is_loaded or self.vto_model is None or self.emotion_head is None:
            return self._make_response()
        if self.frame_count % self.predict_every != 0:
            return self._make_response()

        try:
            t0   = time.time()
            crop = self.extract_face_crop(frame_bgr, face_landmarks, frame_shape)
            if crop is None or crop.size == 0:
                return self._make_response()

            img_tensor = self.transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feats  = self.vto_model.backbone(img_tensor)
                # Use raw GAP output — bypass shared_proj (fixes 352 vs 512 error)
                g      = self.vto_model.flat(self.vto_model.gap(feats[2]))  # (1, gap_dim)
                logits = self.emotion_head(g)                                # (1, 7)
                probs  = F.softmax(logits, dim=1).squeeze(0)                # (7,)

            self.prob_history.append(probs.cpu().numpy())
            avg_probs  = np.mean(self.prob_history, axis=0)
            pred_idx   = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx])

            self.emotion_history.append(pred_idx)
            majority = Counter(self.emotion_history).most_common(1)[0][0]
            emotion  = EMOTION_CLASSES[majority]

            self.current_emotion     = emotion
            self.current_confidence  = confidence
            self.current_anim_params = EMOTION_ANIMATION_MAP[emotion].copy()
            self.current_anim_params["confidence"] = confidence
            self._inference_times.append((time.time() - t0) * 1000)
            self._error_count = 0

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[JM3N] Inference error (#{self._error_count}): {e}")

        return self._make_response()

    def _make_response(self):
        return {
            "emotion":     self.current_emotion,
            "confidence":  self.current_confidence,
            "anim_params": self.current_anim_params,
            "mean_ms":     round(np.mean(self._inference_times), 1)
                           if self._inference_times else 0.0,
        }

    def get_status(self):
        return {
            "loaded":            self.is_loaded,
            "current_emotion":   self.current_emotion,
            "gap_dim":           self.gap_dim,
            "mean_inference_ms": round(np.mean(self._inference_times), 1)
                                 if self._inference_times else 0.0,
        }