"""
test_system.py - Comprehensive Unit & Integration Tests for Drowsiness Detection System
"""

import os
import sys
import unittest
import numpy as np
import cv2

import config
from metrics import calculate_ear, calculate_mar, estimate_head_pose, PerclosTracker
from alerts import AudioAlertManager
from calibration import DriverCalibrator
from logger import EventLogger
from detector import FaceMeshDetector, DetectionResult
from ui_overlay import HUDOverlay


class TestDrowsinessSystem(unittest.TestCase):

    def test_01_ear_calculation(self):
        """Test Eye Aspect Ratio calculation on open vs closed eyes."""
        # Simulated open eye: outer=(10, 50), upper1=(25, 40), upper2=(35, 40), inner=(50, 50), lower2=(35, 60), lower1=(25, 60)
        # horizontal width = 40, vertical heights = 20, 20 -> EAR = (20 + 20) / (2 * 40) = 0.50
        open_eye = [(10, 50), (25, 40), (35, 40), (50, 50), (35, 60), (25, 60)]
        ear_open = calculate_ear(open_eye)
        self.assertAlmostEqual(ear_open, 0.50, places=2)

        # Simulated closed eye: vertical heights are nearly 0
        closed_eye = [(10, 50), (25, 50), (35, 50), (50, 50), (35, 51), (25, 51)]
        ear_closed = calculate_ear(closed_eye)
        self.assertLess(ear_closed, 0.10)
        print(f"[TEST 1] EAR Test Passed: Open={ear_open:.2f}, Closed={ear_closed:.2f}")

    def test_02_mar_calculation(self):
        """Test Mouth Aspect Ratio calculation."""
        # Simulated closed resting mouth
        resting_mouth = {
            61: (100, 200),
            291: (200, 200),
            81: (130, 195),
            178: (130, 205),
            13: (150, 195),
            14: (150, 205),
            311: (170, 195),
            402: (170, 205),
        }
        mar_resting = calculate_mar(resting_mouth)
        self.assertLess(mar_resting, 0.30)

        # Simulated yawning wide open mouth
        yawn_mouth = {
            61: (100, 200),
            291: (200, 200),
            81: (130, 160),
            178: (130, 240),
            13: (150, 150),
            14: (150, 250),
            311: (170, 160),
            402: (170, 240),
        }
        mar_yawn = calculate_mar(yawn_mouth)
        self.assertGreater(mar_yawn, 0.70)
        print(f"[TEST 2] MAR Test Passed: Resting={mar_resting:.2f}, Yawn={mar_yawn:.2f}")

    def test_03_perclos_tracker(self):
        """Test PERCLOS sliding window calculation."""
        tracker = PerclosTracker(window_size=10)
        for _ in range(5):
            tracker.update(is_eye_closed=False)
        for _ in range(5):
            perclos = tracker.update(is_eye_closed=True)
        # 5 out of 10 closed = 50%
        self.assertEqual(perclos, 50.0)
        print(f"[TEST 3] PERCLOS Test Passed: Window percentage={perclos:.1f}%")

    def test_04_driver_calibration(self):
        """Test personalized calibration logic and threshold bounds."""
        calibrator = DriverCalibrator(target_frames=10)
        calibrator.start()
        self.assertEqual(calibrator.status, DriverCalibrator.STATUS_CALIBRATING)

        for _ in range(10):
            finished = calibrator.update(ear=0.32, mar=0.25, pitch=0.0, yaw=0.0)

        self.assertTrue(finished)
        self.assertTrue(calibrator.is_calibrated)
        # Check calibrated threshold matches factor
        expected_ear = 0.32 * config.CALIBRATION_EAR_FACTOR
        self.assertAlmostEqual(calibrator.ear_threshold, expected_ear, places=2)
        print(f"[TEST 4] Calibration Test Passed: Calibrated EAR threshold={calibrator.ear_threshold:.3f}")

    def test_05_audio_alerts(self):
        """Test audio alert trigger without errors."""
        audio = AudioAlertManager(enabled=True)
        # Trigger alert (non-blocking)
        audio.trigger_alert(AudioAlertManager.ALERT_DROWSY)
        audio.trigger_alert(AudioAlertManager.ALERT_YAWN)
        print("[TEST 5] Audio Alert Test Passed.")

    def test_06_event_logger(self):
        """Test logging drowsiness incidents to CSV."""
        test_csv = os.path.join(os.path.dirname(__file__), "test_events.csv")
        if os.path.exists(test_csv):
            os.remove(test_csv)

        logger = EventLogger(log_path=test_csv)
        logger.log_event(
            event_type="DROWSY - EYES CLOSED",
            duration_sec=1.5,
            duration_frames=30,
            ear=0.18,
            ear_thresh=0.25,
            mar=0.30,
            mar_thresh=0.65,
            perclos=35.0,
            pitch=-16.5,
            yaw=2.1,
            notes="Automated Unit Test Event"
        )

        self.assertTrue(os.path.exists(test_csv))
        with open(test_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # Header + 1 event
        self.assertIn("DROWSY - EYES CLOSED", lines[1])

        # Clean up test file
        if os.path.exists(test_csv):
            os.remove(test_csv)
        print("[TEST 6] Event Logger Test Passed.")

    def test_07_face_mesh_detector(self):
        """Test FaceMeshDetector initialization and frame processing."""
        detector = FaceMeshDetector()
        # Test with blank frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.process_frame(dummy)
        self.assertIsInstance(res, DetectionResult)
        self.assertFalse(res.face_detected)
        detector.close()
        print("[TEST 7] Detector Initialization & Inference Test Passed.")

    def test_08_hud_overlay_render(self):
        """Test HUD Overlay rendering across all alert states without crashing."""
        hud = HUDOverlay()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Test rendering safe state
        hud.draw_hud(
            img=frame,
            ear=0.31,
            ear_thresh=0.25,
            mar=0.28,
            mar_thresh=0.65,
            perclos=5.0,
            head_pitch=0.0,
            head_yaw=0.0,
            status_text="DRIVER ALERT",
            status_level="SAFE",
            ear_consec=0,
            mar_consec=0,
            fps=30.0,
            calib_pct=100.0,
            session_time_str="00:05:22",
            recent_events=["12:00:00 DROWSY (1.2s)"],
            audio_enabled=True,
            face_detected=True
        )

        # Test rendering danger flashing border
        hud.draw_hud(
            img=frame,
            ear=0.15,
            ear_thresh=0.25,
            mar=0.75,
            mar_thresh=0.65,
            perclos=45.0,
            head_pitch=-18.0,
            head_yaw=5.0,
            status_text="DROWSY ALARM - WAKE UP!",
            status_level="DANGER",
            ear_consec=25,
            mar_consec=30,
            fps=28.5,
            calib_pct=100.0,
            session_time_str="00:08:15",
            recent_events=["12:05:00 DROWSY (2.5s)"],
            audio_enabled=True,
            face_detected=True
        )

        self.assertEqual(frame.shape, (720, 1280, 3))
        print("[TEST 8] HUD Overlay Rendering Test Passed.")


if __name__ == "__main__":
    unittest.main()
