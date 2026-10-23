"""
calibration.py - Dynamic Personalized Driver Calibration Engine

Establishes driver-specific baseline metrics (open-eye EAR, resting mouth MAR, neutral head pose)
over an initial calibration phase to minimize false alarms and adapt to individual eye shapes.
"""

from typing import Optional, Tuple
import numpy as np
import config


class DriverCalibrator:
    """Collects baseline frames to compute personalized detection thresholds."""

    STATUS_PENDING = "PENDING"
    STATUS_CALIBRATING = "CALIBRATING"
    STATUS_COMPLETE = "COMPLETE"

    def __init__(self, target_frames: int = config.CALIBRATION_FRAMES):
        self.target_frames = target_frames
        self.ear_factor = config.CALIBRATION_EAR_FACTOR
        self.mar_factor = config.CALIBRATION_MAR_FACTOR

        self.ear_samples = []
        self.mar_samples = []
        self.pitch_samples = []
        self.yaw_samples = []

        self.status = self.STATUS_PENDING
        self.baseline_ear = config.DEFAULT_EAR_THRESHOLD / self.ear_factor
        self.baseline_mar = config.DEFAULT_MAR_THRESHOLD / self.mar_factor
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

        self.ear_threshold = config.DEFAULT_EAR_THRESHOLD
        self.mar_threshold = config.DEFAULT_MAR_THRESHOLD

    def start(self):
        """Begin or restart calibration."""
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.pitch_samples.clear()
        self.yaw_samples.clear()
        self.status = self.STATUS_CALIBRATING

    @property
    def is_calibrated(self) -> bool:
        return self.status == self.STATUS_COMPLETE

    @property
    def progress_pct(self) -> float:
        if self.status == self.STATUS_COMPLETE:
            return 100.0
        if self.status != self.STATUS_CALIBRATING or not self.target_frames:
            return 0.0
        return min(100.0, (len(self.ear_samples) / self.target_frames) * 100.0)

    def update(self, ear: float, mar: float, pitch: float, yaw: float) -> bool:
        """
        Record a sample frame. Returns True if calibration just finished on this frame.
        """
        if self.status != self.STATUS_CALIBRATING:
            return False

        # Ignore obvious blinks or yawns during calibration to get pure baseline
        if 0.15 < ear < 0.50:
            self.ear_samples.append(ear)
        if 0.10 < mar < 0.80:
            self.mar_samples.append(mar)
        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)

        if len(self.ear_samples) >= self.target_frames:
            self._finalize()
            return True

        return False

    def _finalize(self):
        """Compute final calibrated baseline and operational thresholds."""
        if self.ear_samples:
            # Use median to be resilient against occasional blinks
            self.baseline_ear = float(np.median(self.ear_samples))
            calibrated_ear_thresh = self.baseline_ear * self.ear_factor
            # Clamp to sensible anatomical bounds
            self.ear_threshold = float(np.clip(calibrated_ear_thresh, 0.18, 0.32))
        else:
            self.ear_threshold = config.DEFAULT_EAR_THRESHOLD

        if self.mar_samples:
            self.baseline_mar = float(np.median(self.mar_samples))
            calibrated_mar_thresh = self.baseline_mar * self.mar_factor
            # Clamp to sensible bounds
            self.mar_threshold = float(np.clip(calibrated_mar_thresh, 0.50, 0.85))
        else:
            self.mar_threshold = config.DEFAULT_MAR_THRESHOLD

        if self.pitch_samples:
            self.baseline_pitch = float(np.median(self.pitch_samples))
        if self.yaw_samples:
            self.baseline_yaw = float(np.median(self.yaw_samples))

        self.status = self.STATUS_COMPLETE
        print(f"[DriverCalibrator] Calibrated successfully!")
        print(f"  Baseline EAR: {self.baseline_ear:.3f} -> Threshold: {self.ear_threshold:.3f}")
        print(f"  Baseline MAR: {self.baseline_mar:.3f} -> Threshold: {self.mar_threshold:.3f}")
