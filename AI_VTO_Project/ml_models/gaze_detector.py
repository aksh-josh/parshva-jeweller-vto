"""
gaze_detector.py — Enhanced Gaze Detection with Y-axis Tracking
================================================================
VERSION: v7 — Per-card gaze detection with vertical position tracking

FEATURES:
- Head-pose-based gaze detection (primary method)
- X-axis: Left/Right detection for recommendation panel
- Y-axis: Vertical position to identify specific card being looked at
- Iris landmark validation for accuracy
- Smoothing buffer to reduce jitter
- Dwell time tracking per card

USAGE:
    detector = GazeDetector(dwell_threshold_seconds=2.0)
    result = detector.update(face_landmarks, frame_width, frame_height)
    
    if result['looking_right'] and result['looking_at_y'] is not None:
        card_idx = map_gaze_to_card(result['looking_at_y'])
"""

import time
import numpy as np


class GazeDetector:
    def __init__(self,
                 dwell_threshold_seconds=2.0,
                 right_gaze_threshold=-0.015,
                 smoothing_frames=8):
        """
        Initialize gaze detector with Y-axis tracking
        
        Args:
            dwell_threshold_seconds: Time required to trigger dwell event
            right_gaze_threshold: X offset threshold for "looking right" detection
            smoothing_frames: Number of frames for smoothing buffer
        """
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.right_gaze_threshold = right_gaze_threshold
        self.smoothing_frames = smoothing_frames
        
        # Smoothing buffers
        self.smoothing_buffer_x = []
        self.smoothing_buffer_y = []  # NEW: Y-axis smoothing
        
        # Dwell tracking
        self.dwell_start_time = None
        self.dwell_triggered = False
        
        # State
        self.is_active = False
        
        # Statistics
        self.total_frames = 0
        self.valid_frames = 0
        self.dwell_events_triggered = 0
        
    def update(self, face_landmarks, frame_width, frame_height):
        """
        Process face landmarks and return gaze state with Y-axis position
        
        Args:
            face_landmarks: MediaPipe face landmarks object
            frame_width: Width of video frame
            frame_height: Height of video frame
            
        Returns:
            dict: {
                'gaze_x': float,           # Horizontal offset (-1 to 1)
                'gaze_y': float,           # Vertical position (0=top, 1=bottom)
                'looking_right': bool,     # True if looking at right side
                'looking_at_y': float,     # Y position when looking right, else None
                'dwell_pct': float,        # Dwell progress (0.0 to 1.0)
                'dwell_event': bool,       # True when dwell completes
                'gaze_valid': bool,        # True if iris landmarks detected
                'method': str              # Detection method used
            }
        """
        self.total_frames += 1
        
        # Return invalid state if not active or no landmarks
        if not self.is_active or face_landmarks is None:
            return self._invalid_state()
        
        lm = face_landmarks.landmark
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1: Head Pose Calculation (X-axis + Y-axis)
        # ═══════════════════════════════════════════════════════════════════
        
        # Key landmarks for head pose
        nose_tip = lm[1]      # Nose tip
        left_eye = lm[33]     # Left eye inner corner
        right_eye = lm[263]   # Right eye inner corner
        chin = lm[152]        # Chin
        forehead = lm[10]     # Forehead center
        
        # Horizontal offset: compare nose to eye center
        head_center_x = (left_eye.x + right_eye.x) / 2
        gaze_offset_x = head_center_x - nose_tip.x
        
        # ── NEW: Vertical offset calculation ──────────────────────────────
        # When looking up: nose moves up relative to eyes (negative offset)
        # When looking down: nose moves down relative to eyes (positive offset)
        eye_center_y = (left_eye.y + right_eye.y) / 2
        gaze_offset_y = nose_tip.y - eye_center_y
        
        # Normalize Y to 0-1 range for easier card mapping
        # Typical range: -0.05 (looking up) to +0.05 (looking down)
        # Scale factor: multiply by 10 to expand range, then add 0.5 to center
        gaze_y_normalized = 0.5 + (gaze_offset_y * 10)
        gaze_y_normalized = max(0.0, min(1.0, gaze_y_normalized))
        
        # Alternative: Use chin-to-forehead ratio for more stable Y tracking
        face_height = abs(chin.y - forehead.y)
        nose_relative_y = (nose_tip.y - forehead.y) / max(face_height, 0.01)
        # Blend both methods for robustness
        gaze_y_normalized = (gaze_y_normalized * 0.6) + (nose_relative_y * 0.4)
        gaze_y_normalized = max(0.0, min(1.0, gaze_y_normalized))
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2: Smoothing
        # ═══════════════════════════════════════════════════════════════════
        
        self.smoothing_buffer_x.append(gaze_offset_x)
        self.smoothing_buffer_y.append(gaze_y_normalized)
        
        if len(self.smoothing_buffer_x) > self.smoothing_frames:
            self.smoothing_buffer_x.pop(0)
        if len(self.smoothing_buffer_y) > self.smoothing_frames:
            self.smoothing_buffer_y.pop(0)
        
        smoothed_x = sum(self.smoothing_buffer_x) / len(self.smoothing_buffer_x)
        smoothed_y = sum(self.smoothing_buffer_y) / len(self.smoothing_buffer_y)
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3: Direction Detection
        # ═══════════════════════════════════════════════════════════════════
        
        looking_right = smoothed_x < self.right_gaze_threshold
        looking_at_y = smoothed_y if looking_right else None
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 4: Dwell Time Tracking
        # ═══════════════════════════════════════════════════════════════════
        
        if looking_right:
            if self.dwell_start_time is None:
                self.dwell_start_time = time.time()
            
            elapsed = time.time() - self.dwell_start_time
            dwell_pct = min(1.0, elapsed / self.dwell_threshold_seconds)
            dwell_event = (dwell_pct >= 1.0 and not self.dwell_triggered)
            
            if dwell_event:
                self.dwell_triggered = True
                self.dwell_events_triggered += 1
        else:
            # Reset dwell when not looking right
            self.dwell_start_time = None
            self.dwell_triggered = False
            dwell_pct = 0.0
            dwell_event = False
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 5: Iris Validation
        # ═══════════════════════════════════════════════════════════════════
        
        # Check if iris landmarks are detected (468-477 for both eyes)
        left_iris = lm[468:473]
        right_iris = lm[473:478]
        
        iris_detected = all(0 < p.x < 1 and 0 < p.y < 1 for p in left_iris) and \
                        all(0 < p.x < 1 and 0 < p.y < 1 for p in right_iris)
        
        if iris_detected:
            self.valid_frames += 1
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 6: Return Result
        # ═══════════════════════════════════════════════════════════════════
        
        return {
            "gaze_x": round(smoothed_x, 4),
            "gaze_y": round(smoothed_y, 4),
            "looking_right": looking_right,
            "looking_at_y": looking_at_y,
            "dwell_pct": round(dwell_pct, 3),
            "dwell_event": dwell_event,
            "gaze_valid": iris_detected,
            "method": "head_pose_xy",
        }
    
    def _invalid_state(self):
        """Return default state when gaze tracking is not active/valid"""
        return {
            "gaze_x": 0.0,
            "gaze_y": 0.5,
            "looking_right": False,
            "looking_at_y": None,
            "dwell_pct": 0.0,
            "dwell_event": False,
            "gaze_valid": False,
            "method": "none",
        }
    
    def reset_dwell(self):
        """Manually reset dwell timer (useful after triggering an action)"""
        self.dwell_start_time = None
        self.dwell_triggered = False
    
    def get_stats(self):
        """Get detector statistics"""
        return {
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "valid_rate": round(self.valid_frames / max(self.total_frames, 1), 3),
            "dwell_events_triggered": self.dwell_events_triggered,
            "threshold": self.right_gaze_threshold,
            "dwell_threshold_sec": self.dwell_threshold_seconds,
            "current_offset": round(sum(self.smoothing_buffer_x) / max(len(self.smoothing_buffer_x), 1), 4),
            "current_y": round(sum(self.smoothing_buffer_y) / max(len(self.smoothing_buffer_y), 1), 4),
            "iris_valid": self.valid_frames > 0,
            "method": "head_pose_xy",
        }