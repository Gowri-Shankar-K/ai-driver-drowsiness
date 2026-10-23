"""
metrics.py - Computer Vision & Biometric Metrics for Drowsiness Detection

Implements:
1. Eye Aspect Ratio (EAR) based on Soukupova & Cech (2016)
2. Mouth Aspect Ratio (MAR) for Yawn Detection
3. Head Pose Estimation (Pitch, Yaw, Roll) via cv2.solvePnP (Nodding / Distraction)
4. PERCLOS (Percentage of Eye Closure) over a moving temporal window
"""

import collections
import math
import numpy as np
import cv2

# --- MediaPipe Landmark Indices ---
# Left eye landmarks: [outer_corner, top1, top2, inner_corner, bottom2, bottom1]
LEFT_EYE_LANDMARKS = [263, 385, 387, 362, 373, 380]
# Right eye landmarks: [outer_corner, top1, top2, inner_corner, bottom2, bottom1]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Additional comprehensive contours for visualization
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Mouth landmarks for MAR calculation (inner / outer lips)
# Outer horizontal corners: 61, 291
# Vertical pairs: (81, 178), (13, 14), (311, 402)
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_VERTICAL_PAIRS = [(81, 178), (13, 14), (311, 402)]

MOUTH_CONTOUR = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

# 3D Model Points for Head Pose Estimation (in mm)
HEAD_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),         # Chin (landmark 152 or 199)
    (-225.0, 170.0, -135.0),      # Left eye outer corner (landmark 263 / 33)
    (225.0, 170.0, -135.0),       # Right eye outer corner (landmark 33 / 263)
    (-150.0, -150.0, -125.0),     # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)       # Right mouth corner (landmark 291)
], dtype=np.float64)

HEAD_POSE_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]


def euclidean_dist(pt1, pt2):
    """Calculate Euclidean distance between two 2D/3D points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def calculate_ear(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR) given 6 landmark points:
    P1 (outer), P2 (upper_outer), P3 (upper_inner), P4 (inner), P5 (lower_inner), P6 (lower_outer).
    
    Formula: EAR = (|P2 - P6| + |P3 - P5|) / (2.0 * |P1 - P4|)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def calculate_mar(landmarks_dict):
    """
    Calculate Mouth Aspect Ratio (MAR) to detect yawning.
    
    Formula: MAR = sum(vertical_pairs) / (2.0 * horizontal_width)
    """
    if MOUTH_LEFT_CORNER not in landmarks_dict or MOUTH_RIGHT_CORNER not in landmarks_dict:
        return 0.0

    p_left = landmarks_dict[MOUTH_LEFT_CORNER]
    p_right = landmarks_dict[MOUTH_RIGHT_CORNER]
    horizontal = euclidean_dist(p_left, p_right)

    if horizontal == 0:
        return 0.0

    vertical_sum = 0.0
    for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
        if top_idx in landmarks_dict and bot_idx in landmarks_dict:
            vertical_sum += euclidean_dist(landmarks_dict[top_idx], landmarks_dict[bot_idx])

    mar = vertical_sum / (2.0 * horizontal)
    return float(mar)


def estimate_head_pose(landmarks_dict, img_w, img_h):
    """
    Estimate head orientation (pitch, yaw, roll) using cv2.solvePnP.
    Returns:
        pitch: degrees (> 0 looking up, < -15 nodding down / drowsy)
        yaw: degrees (> 0 turned left, < 0 turned right, |yaw| > 25 distracted)
        roll: degrees (tilt sideways)
        rotation_vector, translation_vector
    """
    for idx in HEAD_POSE_LANDMARK_IDS:
        if idx not in landmarks_dict:
            return 0.0, 0.0, 0.0, None, None

    image_points = np.array([
        landmarks_dict[1],    # Nose tip
        landmarks_dict[199],  # Chin
        landmarks_dict[33],   # Right eye right corner
        landmarks_dict[263],  # Left eye left corner
        landmarks_dict[61],   # Left mouth corner
        landmarks_dict[291]   # Right mouth corner
    ], dtype=np.float64)

    # Approximate camera matrix assuming principal point at center
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        HEAD_3D_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, None, None

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles (pitch, yaw, roll)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    pitch = float(math.degrees(x))
    yaw = float(math.degrees(y))
    roll = float(math.degrees(z))

    return pitch, yaw, roll, rotation_vector, translation_vector


class PerclosTracker:
    """
    PERCLOS Tracker: Calculates Percentage of Eye Closure over a rolling window.
    PERCLOS is the NHTSA standard fatigue metric. A value > 15-20% indicates significant fatigue.
    """
    def __init__(self, window_size=90):
        self.window_size = window_size
        self.eye_states = collections.deque(maxlen=window_size)

    def update(self, is_eye_closed: bool) -> float:
        """Add current frame eye status and return PERCLOS percentage (0.0 to 100.0)."""
        self.eye_states.append(1 if is_eye_closed else 0)
        if not self.eye_states:
            return 0.0
        return (sum(self.eye_states) / len(self.eye_states)) * 100.0

    def reset(self):
        self.eye_states.clear()
