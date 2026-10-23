"""
Configuration settings for AI Driver Drowsiness Detection System.
Based on research papers by Soukupova & Cech (2016) and Ersoy et al. (2026).
"""

import os

# --- Camera & Video Settings ---
CAMERA_INDEX = 0             # 0 is usually the built-in laptop webcam
FRAME_WIDTH = 1280           # Target capture width (falls back to device capabilities)
FRAME_HEIGHT = 720           # Target capture height
FPS_TARGET = 30

# --- Eye Aspect Ratio (EAR) Parameters ---
# Standard EAR threshold for open vs closed eyes
DEFAULT_EAR_THRESHOLD = 0.25
# Number of consecutive frames eyes must be closed to trigger drowsiness alarm
EAR_CONSEC_FRAMES = 20

# --- Mouth Aspect Ratio (MAR) Parameters ---
# Threshold above which mouth is considered yawning
DEFAULT_MAR_THRESHOLD = 0.65
# Number of consecutive frames mouth must be open to trigger yawn alarm
MAR_CONSEC_FRAMES = 25

# --- Calibration Settings ---
# Number of initial frames used to auto-calibrate personalized EAR/MAR thresholds
CALIBRATION_FRAMES = 60
CALIBRATION_EAR_FACTOR = 0.75  # EAR threshold = 75% of baseline open eye EAR
CALIBRATION_MAR_FACTOR = 1.60  # MAR threshold = 160% of baseline resting mouth MAR

# --- Audio Alert Settings ---
AUDIO_ALERT_ENABLED = True
AUDIO_FREQUENCY = 2500       # Frequency in Hz (loud high-pitch warning beep)
AUDIO_DURATION_MS = 250      # Duration of beep in milliseconds
AUDIO_COOLDOWN_SEC = 0.3     # Minimum interval between alerts to avoid audio queue bloat

# --- Logging & Data Storage ---
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness_events.csv")

# --- MediaPipe Model Settings ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# --- PERCLOS (Percentage of Eye Closure) Parameters ---
PERCLOS_WINDOW_FRAMES = 90   # Sliding window (~3 seconds at 30 fps)
PERCLOS_THRESHOLD = 0.20     # > 20% eyes closed within window triggers drowsiness alert

# --- Head Pose (Nodding / Distraction) Parameters ---
HEAD_PITCH_NOD_THRESHOLD = -14.0       # Head tilting down in degrees (nodding off)
HEAD_YAW_DISTRACTION_THRESHOLD = 28.0  # Head looking left/right in degrees (distracted)
HEAD_CONSEC_FRAMES = 25                # Consecutive frames for head posture alert

# --- UI & Display Settings ---
COLOR_SAFE = (0, 200, 0)        # Green (BGR)
COLOR_WARNING = (0, 165, 255)   # Amber/Orange (BGR)
COLOR_DANGER = (0, 0, 255)      # Red (BGR)
COLOR_TEXT = (255, 255, 255)    # White (BGR)
COLOR_ACCENT = (255, 191, 0)    # Deep Sky Cyan/Gold
COLOR_BG_PANEL = (25, 25, 25)   # Dark slate

