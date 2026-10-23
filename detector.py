"""
detector.py - MediaPipe FaceLandmarker Wrapper & Feature Extraction
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from metrics import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS
)


@dataclass
class DetectionResult:
    """Structure holding all analyzed metrics for a single video frame."""
    face_detected: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    blendshape_blink_left: float = 0.0
    blendshape_blink_right: float = 0.0
    blendshape_jaw_open: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    landmarks_px: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    rotation_vec: Optional[np.ndarray] = None
    translation_vec: Optional[np.ndarray] = None


class FaceMeshDetector:
    """Encapsulates MediaPipe FaceLandmarker with metric calculations."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.MODEL_PATH
        self._ensure_model_exists()
        self._init_detector()

    def _ensure_model_exists(self):
        """Download face_landmarker.task if not already cached locally."""
        if not os.path.exists(self.model_path):
            print(f"[FaceMeshDetector] Downloading model to {self.model_path}...")
            urllib.request.urlretrieve(config.MODEL_URL, self.model_path)
            print("[FaceMeshDetector] Model download complete.")

    def _init_detector(self):
        """Initialize MediaPipe FaceLandmarker."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        """
        Analyze a single frame for facial landmarks, EAR, MAR, blendshapes, and head pose.
        """
        result = DetectionResult()
        if frame_bgr is None or frame_bgr.size == 0:
            return result

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection = self.detector.detect(mp_image)

        if not detection.face_landmarks:
            return result

        result.face_detected = True
        landmarks = detection.face_landmarks[0]

        # Convert landmarks to pixel coordinates
        landmarks_px: Dict[int, Tuple[int, int]] = {}
        for idx, lm in enumerate(landmarks):
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks_px[idx] = (px, py)
        result.landmarks_px = landmarks_px

        # Extract blendshapes if available
        if detection.face_blendshapes:
            for cat in detection.face_blendshapes[0]:
                name = cat.category_name
                score = cat.score
                if name == "eyeBlinkLeft":
                    result.blendshape_blink_left = float(score)
                elif name == "eyeBlinkRight":
                    result.blendshape_blink_right = float(score)
                elif name == "jawOpen":
                    result.blendshape_jaw_open = float(score)

        # 1. Calculate Left & Right EAR
        left_pts = [landmarks_px[idx] for idx in LEFT_EYE_LANDMARKS if idx in landmarks_px]
        right_pts = [landmarks_px[idx] for idx in RIGHT_EYE_LANDMARKS if idx in landmarks_px]

        if len(left_pts) == 6:
            result.ear_left = calculate_ear(left_pts)
        if len(right_pts) == 6:
            result.ear_right = calculate_ear(right_pts)

        if result.ear_left > 0 and result.ear_right > 0:
            result.ear = (result.ear_left + result.ear_right) / 2.0
        else:
            result.ear = max(result.ear_left, result.ear_right)

        # 2. Calculate MAR (Mouth Aspect Ratio)
        result.mar = calculate_mar(landmarks_px)

        # 3. Calculate Head Pose (Pitch, Yaw, Roll)
        pitch, yaw, roll, rvec, tvec = estimate_head_pose(landmarks_px, w, h)
        result.head_pitch = pitch
        result.head_yaw = yaw
        result.head_roll = roll
        result.rotation_vec = rvec
        result.translation_vec = tvec

        return result

    def close(self):
        """Release underlying resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
