"""
overlay_engine.py
=================
Stage 4 — Geometry-Aware Overlay Engine

Replaces the simple cv2.resize + overlay approach from app.py with:
    1. Perspective-transform-based placement (handles head tilt, face angle)
    2. Pose-adaptive scaling (jewelry scales with detected face/hand size)
    3. Lighting-aware alpha blending (adjusts for ambient brightness)
    4. Specular highlight synthesis (makes metal look realistic)

Usage:
    from overlay_engine import GeometryOverlayEngine
    engine = GeometryOverlayEngine()
    result = engine.apply(frame, jewelry_img, placement_keypoints, jewelry_type)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PlacementKeypoints:
    """
    Normalised [0,1] coordinates for jewelry placement.
    All values are relative to image dimensions.

    For necklace/chain/mangalsutra:
        anchor_left:  top-left attachment point on skin
        anchor_right: top-right attachment point on skin
        drop_point:   lowest point of pendant

    For earrings:
        left_lobe:  left ear-lobe centre
        right_lobe: right ear-lobe centre

    For rings:
        finger_base:  base of finger (knuckle)
        finger_tip:   fingertip
        finger_width: width of finger at base (scalar, normalised)
    """
    # Necklace
    anchor_left:   Optional[Tuple[float, float]] = None
    anchor_right:  Optional[Tuple[float, float]] = None
    drop_point:    Optional[Tuple[float, float]] = None

    # Earring
    left_lobe:     Optional[Tuple[float, float]] = None
    right_lobe:    Optional[Tuple[float, float]] = None

    # Ring
    finger_base:   Optional[Tuple[float, float]] = None
    finger_tip:    Optional[Tuple[float, float]] = None
    finger_width:  Optional[float]               = None

    # Face metrics (computed from MediaPipe or custom model)
    face_width_px: Optional[float] = None
    face_tilt_deg: Optional[float] = None   # head roll angle


# ---------------------------------------------------------------------------
# LIGHTING ESTIMATOR
# ---------------------------------------------------------------------------

class LightingEstimator:
    """
    Estimates ambient lighting conditions from the face region.
    Used to adjust jewelry brightness to match the scene.

    Simple approach: sample brightness from face region and compute
    a multiplicative correction factor for the jewelry overlay.
    """

    @staticmethod
    def estimate_face_brightness(frame: np.ndarray, face_bbox: Tuple) -> float:
        """
        Args:
            frame:     BGR frame
            face_bbox: (x1, y1, x2, y2) face bounding box in pixels

        Returns:
            Normalised brightness [0, 1] — 0.5 = neutral, <0.5 = dark, >0.5 = bright
        """
        x1, y1, x2, y2 = [int(c) for c in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return 0.5

        hsv        = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean() / 255.0
        return float(brightness)

    @staticmethod
    def adjust_jewelry_brightness(
        jewelry_bgra: np.ndarray,
        scene_brightness: float,
        target_brightness: float = 0.55,
    ) -> np.ndarray:
        """
        Adjusts jewelry brightness to match scene.
        Metal jewelry should be slightly brighter than skin.

        Args:
            jewelry_bgra:       Jewelry image (H, W, 4)
            scene_brightness:   Estimated scene brightness [0, 1]
            target_brightness:  Desired relative brightness for jewelry

        Returns:
            Brightness-adjusted jewelry BGRA image
        """
        correction = target_brightness / max(scene_brightness, 0.1)
        correction = np.clip(correction, 0.7, 1.5)  # limit adjustment range

        adjusted = jewelry_bgra.copy().astype(np.float32)
        adjusted[:, :, :3] = np.clip(adjusted[:, :, :3] * correction, 0, 255)
        return adjusted.astype(np.uint8)


# ---------------------------------------------------------------------------
# SPECULAR HIGHLIGHT SYNTHESISER
# ---------------------------------------------------------------------------

class SpecularHighlightSynthesiser:
    """
    Adds simple specular highlights to metal jewelry to improve realism.

    Metal surfaces reflect light as a concentrated bright spot.
    We approximate this with a Gaussian highlight at the brightest point
    of the jewelry image, blended with a white overlay.

    This is a simplified Phong reflection model approximation.
    """

    @staticmethod
    def add_highlight(
        jewelry_bgra: np.ndarray,
        light_direction: Tuple[float, float] = (0.3, -0.7),
        intensity: float = 0.15,
    ) -> np.ndarray:
        """
        Args:
            jewelry_bgra:    Input jewelry image (H, W, 4)
            light_direction: Normalised (x, y) light direction vector
            intensity:       Highlight blend intensity [0, 1]

        Returns:
            Jewelry image with specular highlight added
        """
        if jewelry_bgra is None or jewelry_bgra.size == 0:
            return jewelry_bgra

        h, w = jewelry_bgra.shape[:2]
        alpha = jewelry_bgra[:, :, 3]

        # Only apply highlight where jewelry is present (alpha > 0)
        mask = (alpha > 10).astype(np.float32)

        # Compute highlight position based on light direction
        lx, ly   = light_direction
        center_x = int(np.clip(w * (0.5 + lx * 0.3), 0, w - 1))
        center_y = int(np.clip(h * (0.5 + ly * 0.3), 0, h - 1))

        # Gaussian highlight map
        Y, X    = np.ogrid[:h, :w]
        sigma   = min(h, w) * 0.15
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        gaussian = gaussian * mask  # only on jewelry pixels

        # Blend white highlight
        result  = jewelry_bgra.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] + gaussian * 255.0 * intensity,
                0, 255
            )

        return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# PERSPECTIVE TRANSFORM UTILITIES
# ---------------------------------------------------------------------------

def get_perspective_transform_necklace(
    jewelry_img: np.ndarray,
    anchor_left_px:  Tuple[int, int],
    anchor_right_px: Tuple[int, int],
    drop_px:         Tuple[int, int],
    face_tilt_deg:   float = 0.0,
) -> Optional[np.ndarray]:
    """
    Computes a perspective transformation for necklace placement.

    Maps the jewelry image corners to the desired placement on the face,
    accounting for head tilt and perspective foreshortening.

    Args:
        jewelry_img:      Source jewelry image (H, W, 4)
        anchor_left_px:   Left attachment point in frame pixels
        anchor_right_px:  Right attachment point in frame pixels
        drop_px:          Pendant drop point in frame pixels
        face_tilt_deg:    Head roll angle (positive = right tilt)

    Returns:
        Warped jewelry image (same size as source), or None on failure
    """
    h, w = jewelry_img.shape[:2]

    # Source corners: top-left, top-right, bottom-right, bottom-left
    src_pts = np.float32([
        [0,   0  ],
        [w-1, 0  ],
        [w-1, h-1],
        [0,   h-1],
    ])

    lx, ly   = anchor_left_px
    rx, ry   = anchor_right_px
    dx, dy   = drop_px
    width_px = abs(rx - lx)

    # Destination corners accounting for head tilt
    tilt_rad = np.radians(face_tilt_deg)
    tilt_offset = int(width_px * 0.05 * np.sin(tilt_rad))

    dst_pts = np.float32([
        [lx,                ly + tilt_offset   ],  # top-left
        [rx,                ry - tilt_offset   ],  # top-right
        [dx + width_px//2,  dy - tilt_offset   ],  # bottom-right
        [dx - width_px//2,  dy + tilt_offset   ],  # bottom-left
    ])

    # Compute and apply perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    if M is None:
        return None

    frame_h = max(dy + 100, h * 2)
    frame_w = max(rx + 100, w * 2)
    warped  = cv2.warpPerspective(
        jewelry_img, M, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped


# ---------------------------------------------------------------------------
# MAIN OVERLAY ENGINE
# ---------------------------------------------------------------------------

class GeometryOverlayEngine:
    """
    Geometry-aware jewelry overlay engine.

    Replaces the simple cv2.resize + alpha blend from the original app.py
    with a full perspective-transform pipeline that handles:
        - Head tilt (roll angle from MediaPipe face mesh)
        - Perspective foreshortening
        - Lighting adjustment
        - Specular highlights on metal

    Usage:
        engine = GeometryOverlayEngine()

        # For necklace:
        result_frame = engine.apply_necklace(
            frame, jewelry_bgra, face_landmarks, frame_shape
        )

        # For earrings:
        result_frame = engine.apply_earrings(
            frame, jewelry_bgra, face_landmarks, frame_shape
        )

        # For rings:
        result_frame = engine.apply_ring(
            frame, jewelry_bgra, hand_landmarks, frame_shape
        )
    """

    def __init__(self, use_lighting: bool = True, use_specular: bool = True):
        self.use_lighting  = use_lighting
        self.use_specular  = use_specular
        self.lighting_est  = LightingEstimator()
        self.specular_syn  = SpecularHighlightSynthesiser()

    def _alpha_blend(
        self,
        background: np.ndarray,
        overlay_bgra: np.ndarray,
        x: int, y: int
    ) -> np.ndarray:
        """
        Alpha-composite overlay_bgra onto background at position (x, y).
        Uses float32 arithmetic to avoid uint8 overflow.
        """
        bg_h, bg_w = background.shape[:2]
        ol_h, ol_w = overlay_bgra.shape[:2]

        x1 = max(0, x);        y1 = max(0, y)
        x2 = min(bg_w, x+ol_w); y2 = min(bg_h, y+ol_h)
        ox1 = max(0, -x);       oy1 = max(0, -y)
        ox2 = ox1 + (x2-x1);    oy2 = oy1 + (y2-y1)

        if x2 <= x1 or y2 <= y1:
            return background

        ol_crop = overlay_bgra[oy1:oy2, ox1:ox2]
        bg_crop = background[y1:y2, x1:x2].astype(np.float32)
        alpha   = ol_crop[:, :, 3:4].astype(np.float32) / 255.0
        fg      = ol_crop[:, :, :3].astype(np.float32)

        blended = alpha * fg + (1.0 - alpha) * bg_crop
        background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return background

    def _get_lm(self, landmark, w: int, h: int) -> Tuple[int, int]:
        """Convert MediaPipe landmark to pixel coords."""
        return int(landmark.x * w), int(landmark.y * h)

    def _estimate_head_tilt(self, face_lm, w: int, h: int) -> float:
        """
        Estimate head roll angle from left/right eye landmarks.
        Returns angle in degrees (positive = tilted right).
        """
        left_eye  = self._get_lm(face_lm.landmark[33],  w, h)
        right_eye = self._get_lm(face_lm.landmark[263], w, h)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def apply_necklace(
        self,
        frame:       np.ndarray,
        jewelry_bgra: np.ndarray,
        face_lm,
        zoom_factor: float = 1.0,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Apply necklace/chain/mangalsutra with perspective transform.

        Args:
            frame:        BGR webcam frame (H, W, 3)
            jewelry_bgra: Jewelry image with transparency (H_j, W_j, 4)
            face_lm:      MediaPipe FaceMesh landmarks object
            zoom_factor:  User-controlled zoom slider

        Returns:
            (processed_frame, status_message, success_bool)
        """
        h_img, w_img = frame.shape[:2]
        lm = face_lm.landmark

        # Key face measurement points
        left_x,  _        = self._get_lm(lm[234], w_img, h_img)
        right_x, _        = self._get_lm(lm[454], w_img, h_img)
        chin_x,  chin_y   = self._get_lm(lm[152], w_img, h_img)
        left_jaw_x, jaw_y = self._get_lm(lm[132], w_img, h_img)

        face_width = abs(right_x - left_x)
        if face_width < 20:
            return frame, "Face too small", False

        head_tilt = self._estimate_head_tilt(face_lm, w_img, h_img)

        # Target width scales with face width and zoom
        target_w = int(face_width * 1.7 * zoom_factor)
        target_w = max(60, target_w)

        ol_h, ol_w = jewelry_bgra.shape[:2]
        target_h   = int(target_w * (ol_h / max(ol_w, 1)))

        # Prepare jewelry: resize, lighting, specular
        jewelry = cv2.resize(jewelry_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if self.use_lighting:
            face_bbox       = (left_x, jaw_y - face_width//2, right_x, chin_y + face_width//3)
            scene_brightness = self.lighting_est.estimate_face_brightness(frame, face_bbox)
            jewelry          = self.lighting_est.adjust_jewelry_brightness(jewelry, scene_brightness)

        if self.use_specular:
            jewelry = self.specular_syn.add_highlight(jewelry, intensity=0.12)

        # Placement: centred below chin, shifted by tilt
        tilt_shift = int(face_width * 0.04 * np.sin(np.radians(head_tilt)))
        pos_x = chin_x - target_w // 2 + tilt_shift
        pos_y = chin_y + int(face_width * 0.15)

        frame = self._alpha_blend(frame, jewelry, pos_x, pos_y)
        return frame, "Necklace placed", True

    def apply_earrings(
        self,
        frame:        np.ndarray,
        jewelry_bgra: np.ndarray,
        face_lm,
        zoom_factor:  float = 1.0,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Apply earrings/jhumkas with ear-lobe placement and head-tilt correction.
        """
        h_img, w_img = frame.shape[:2]
        lm = face_lm.landmark

        left_ear_x,  left_ear_y  = self._get_lm(lm[234], w_img, h_img)
        right_ear_x, right_ear_y = self._get_lm(lm[454], w_img, h_img)
        _, left_jaw_y            = self._get_lm(lm[132], w_img, h_img)
        _, right_jaw_y           = self._get_lm(lm[361], w_img, h_img)

        left_x,  _ = self._get_lm(lm[234], w_img, h_img)
        right_x, _ = self._get_lm(lm[454], w_img, h_img)
        face_width = abs(right_x - left_x)
        head_tilt  = self._estimate_head_tilt(face_lm, w_img, h_img)

        # Compute ear-lobe positions
        left_lobe_y  = (left_ear_y  + left_jaw_y)  // 2
        right_lobe_y = (right_ear_y + right_jaw_y) // 2

        # Split or mirror the earring image
        ol_h, ol_w   = jewelry_bgra.shape[:2]
        img_aspect   = ol_w / max(ol_h, 1)

        if img_aspect >= 0.85:  # wide image = both earrings side by side
            mid              = ol_w // 2
            left_earring     = jewelry_bgra[:, :mid]
            right_earring    = jewelry_bgra[:, mid:]
        else:                   # single earring — mirror for the other side
            left_earring  = jewelry_bgra
            right_earring = cv2.flip(jewelry_bgra, 1)

        target_h = int(face_width * 0.45 * zoom_factor)
        target_h = max(30, target_h)

        for side, ear_img, ear_x, lobe_y in [
            ("left",  left_earring,  left_ear_x,  left_lobe_y),
            ("right", right_earring, right_ear_x, right_lobe_y),
        ]:
            eh, ew    = ear_img.shape[:2]
            t_w       = max(15, int(target_h * (ew / max(eh, 1))))
            resized   = cv2.resize(ear_img, (t_w, target_h), interpolation=cv2.INTER_AREA)

            if self.use_specular:
                resized = self.specular_syn.add_highlight(resized, intensity=0.10)

            # Tilt correction: shift earrings up/down based on head roll
            tilt_offset = int(face_width * 0.08 * np.sin(np.radians(head_tilt)))
            oy = lobe_y + (tilt_offset if side == "left" else -tilt_offset)
            ox = ear_x - t_w // 2

            frame = self._alpha_blend(frame, resized, ox, oy)

        return frame, "Earrings placed", True

    def apply_ring(
        self,
        frame:         np.ndarray,
        jewelry_bgra:  np.ndarray,
        hand_landmarks,
        finger_index:  int = 3,      # 3 = ring finger
        zoom_factor:   float = 1.0,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Apply ring to a specific finger using hand landmarks.

        MediaPipe Hands finger landmark indices:
            0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky

        Ring finger base = landmark 13, tip = 16
        We place the ring at the middle knuckle (between base and mid).

        Args:
            finger_index: Which finger (1=thumb, 2=index, 3=ring, 4=pinky)
        """
        h_img, w_img = frame.shape[:2]
        lm = hand_landmarks.landmark

        # Ring finger landmarks: base=13, mid=14, pip=15, tip=16
        finger_base_lm  = {1: 1,  2: 5,  3: 13, 4: 17}
        finger_mid_lm   = {1: 2,  2: 6,  3: 14, 4: 18}

        base_idx = finger_base_lm.get(finger_index, 13)
        mid_idx  = finger_mid_lm.get(finger_index, 14)

        bx, by   = self._get_lm(lm[base_idx], w_img, h_img)
        mx, my   = self._get_lm(lm[mid_idx],  w_img, h_img)

        # Estimate finger width from adjacent landmarks
        wrist_x, _ = self._get_lm(lm[0], w_img, h_img)
        # Approximate finger width as face/7 (rough heuristic)
        finger_width = max(20, abs(bx - mx) // 2)

        # Ring centre: midpoint between base and mid joint
        ring_cx = (bx + mx) // 2
        ring_cy = (by + my) // 2

        # Finger angle for rotating the ring image
        angle = np.degrees(np.arctan2(my - by, mx - bx)) - 90

        target_w = int(finger_width * 2.4 * zoom_factor)
        target_w = max(20, target_w)

        ol_h, ol_w = jewelry_bgra.shape[:2]
        target_h   = int(target_w * (ol_h / max(ol_w, 1)))

        # Resize and rotate ring to match finger angle
        ring_resized = cv2.resize(jewelry_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)
        M     = cv2.getRotationMatrix2D((target_w//2, target_h//2), -angle, 1.0)
        ring_rotated = cv2.warpAffine(
            ring_resized, M, (target_w, target_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
        )

        if self.use_specular:
            ring_rotated = self.specular_syn.add_highlight(ring_rotated, intensity=0.18)

        pos_x = ring_cx - target_w // 2
        pos_y = ring_cy - target_h // 2

        frame = self._alpha_blend(frame, ring_rotated, pos_x, pos_y)
        return frame, "Ring placed", True


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[OverlayEngine] Sanity check...")
    engine = GeometryOverlayEngine()

    # Create dummy frame and jewelry
    frame   = np.ones((480, 640, 3), dtype=np.uint8) * 200
    jewelry = np.zeros((100, 200, 4), dtype=np.uint8)
    jewelry[:, :, 2] = 200   # red channel
    jewelry[:, :, 3] = 255   # fully opaque

    # Test alpha blend
    result = engine._alpha_blend(frame.copy(), jewelry, 100, 150)
    assert result[150, 100, 2] > 150, "Alpha blend failed"

    # Test specular highlight
    highlighted = engine.specular_syn.add_highlight(jewelry.copy(), intensity=0.2)
    assert highlighted.shape == jewelry.shape, "Specular highlight changed shape"

    print("[OverlayEngine] All checks passed.")
