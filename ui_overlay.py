"""
ui_overlay.py - High-Tech Automotive HUD Overlay for Driver Monitoring

Renders an automotive telemetry display, glassmorphism panels, real-time gauges,
alert banners, PERCLOS meters, and facial landmark wireframes on the OpenCV video frame.
"""

import math
import time
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

import config
from metrics import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, MOUTH_CONTOUR


class HUDOverlay:
    """Renders real-time HUD telemetry, biometric gauges, and warning alerts."""

    def __init__(self):
        self.show_mesh = True
        self.flash_counter = 0

    def draw_glass_panel(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (20, 24, 28),
        alpha: float = 0.72,
        border_color: Optional[Tuple[int, int, int]] = (60, 70, 80)
    ):
        """Draw a semi-transparent glassmorphic panel with subtle border."""
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])

        if x2 <= x1 or y2 <= y1:
            return

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    def draw_progress_bar(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        val: float,
        min_val: float,
        max_val: float,
        threshold: float,
        invert_thresh: bool = False,
        label: str = "",
        val_format: str = "{:.2f}"
    ):
        """
        Draw a horizontal progress gauge bar with a threshold indicator line.
        invert_thresh: True if higher than threshold is danger (e.g., MAR, PERCLOS).
                       False if lower than threshold is danger (e.g., EAR).
        """
        # Clamp normalized fill
        clamped_val = np.clip(val, min_val, max_val)
        fill_ratio = (clamped_val - min_val) / (max_val - min_val + 1e-6)
        fill_w = int(w * fill_ratio)

        # Threshold position
        clamped_thresh = np.clip(threshold, min_val, max_val)
        thresh_ratio = (clamped_thresh - min_val) / (max_val - min_val + 1e-6)
        thresh_x = int(x + w * thresh_ratio)

        # Danger check
        if invert_thresh:
            is_danger = val >= threshold
        else:
            is_danger = val <= threshold

        bar_color = config.COLOR_DANGER if is_danger else config.COLOR_SAFE

        # Background track
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 45, 52), -1)
        # Filled bar
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)

        # Threshold line marker
        cv2.line(img, (thresh_x, y - 3), (thresh_x, y + h + 3), (0, 240, 255), 2, cv2.LINE_AA)

        # Border
        cv2.rectangle(img, (x, y), (x + w, y + h), (85, 95, 110), 1, cv2.LINE_AA)

        # Text Label and Numeric Value
        if label:
            cv2.putText(
                img,
                label,
                (x, y - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.42,
                (200, 210, 220),
                1,
                cv2.LINE_AA
            )
            val_text = val_format.format(val)
            t_size = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)[0]
            cv2.putText(
                img,
                val_text,
                (x + w - t_size[0], y - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.42,
                bar_color,
                1,
                cv2.LINE_AA
            )

    def draw_pulsing_danger_border(self, img: np.ndarray, thickness: int = 14):
        """Draw an aggressive flashing red emergency border and centered alarm banner."""
        self.flash_counter += 1
        alpha = 0.5 + 0.5 * math.sin(self.flash_counter * 0.45)
        color = (0, 0, int(180 + 75 * alpha))

        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), color, thickness)

        # Centered emergency banner
        banner_w, banner_h = 560, 70
        bx1 = (w - banner_w) // 2
        by1 = h // 2 - 35
        self.draw_glass_panel(img, (bx1, by1), (bx1 + banner_w, by1 + banner_h), color=(20, 10, 10), alpha=0.90, border_color=(0, 0, 255))

        alarm_msg = "CRITICAL ALARM: WAKE UP!"
        t_sz = cv2.getTextSize(alarm_msg, cv2.FONT_HERSHEY_DUPLEX, 1.15, 2)[0]
        tx = (w - t_sz[0]) // 2
        ty = by1 + (banner_h + t_sz[1]) // 2

        cv2.putText(img, alarm_msg, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.15, (0, 0, 255), 2, cv2.LINE_AA)

    def draw_landmarks_mesh(
        self,
        img: np.ndarray,
        landmarks_px: Dict[int, Tuple[int, int]],
        is_eyes_closed: bool,
        is_yawning: bool
    ):
        """Draw sleek, futuristic eye and mouth contours."""
        if not self.show_mesh or not landmarks_px:
            return

        eye_color = (0, 70, 255) if is_eyes_closed else (255, 200, 50)
        mouth_color = (0, 100, 255) if is_yawning else (50, 220, 150)

        # Draw left eye
        left_eye_pts = [landmarks_px[i] for i in LEFT_EYE_CONTOUR if i in landmarks_px]
        if len(left_eye_pts) > 2:
            cv2.polylines(img, [np.array(left_eye_pts, dtype=np.int32)], True, eye_color, 1, cv2.LINE_AA)

        # Draw right eye
        right_eye_pts = [landmarks_px[i] for i in RIGHT_EYE_CONTOUR if i in landmarks_px]
        if len(right_eye_pts) > 2:
            cv2.polylines(img, [np.array(right_eye_pts, dtype=np.int32)], True, eye_color, 1, cv2.LINE_AA)

        # Draw mouth
        mouth_pts = [landmarks_px[i] for i in MOUTH_CONTOUR if i in landmarks_px]
        if len(mouth_pts) > 2:
            cv2.polylines(img, [np.array(mouth_pts, dtype=np.int32)], True, mouth_color, 1, cv2.LINE_AA)

    def draw_hud(
        self,
        img: np.ndarray,
        ear: float,
        ear_thresh: float,
        mar: float,
        mar_thresh: float,
        perclos: float,
        head_pitch: float,
        head_yaw: float,
        status_text: str,
        status_level: str,  # 'SAFE', 'WARNING', 'DANGER', 'CALIBRATING'
        ear_consec: int,
        mar_consec: int,
        fps: float,
        calib_pct: float,
        session_time_str: str,
        recent_events: List[str],
        audio_enabled: bool,
        face_detected: bool
    ):
        """Assemble and render the complete driver monitoring dashboard."""
        h, w = img.shape[:2]

        # 1. Top Navigation & Status Bar
        self.draw_glass_panel(img, (15, 12), (w - 15, 76), color=(18, 22, 28), alpha=0.82)

        # Brand / System Title
        cv2.putText(
            img,
            "AI DRIVER GUARDIAN",
            (32, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 215, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            "MULTI-MODAL DROWSINESS & FATIGUE SYSTEM",
            (32, 63),
            cv2.FONT_HERSHEY_DUPLEX,
            0.36,
            (140, 160, 180),
            1,
            cv2.LINE_AA
        )

        # Center Status Pill / Badge
        status_colors = {
            "SAFE": config.COLOR_SAFE,
            "CALIBRATING": (255, 191, 0),
            "WARNING": config.COLOR_WARNING,
            "DANGER": config.COLOR_DANGER
        }
        badge_color = status_colors.get(status_level, config.COLOR_SAFE)

        pill_w = 340
        pill_x1 = (w - pill_w) // 2
        pill_x2 = pill_x1 + pill_w
        pill_y1 = 20
        pill_y2 = 68

        # Draw glowing pill outline & background
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), (30, 35, 42), -1)
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), badge_color, 2, cv2.LINE_AA)

        # Center status text
        st_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
        cv2.putText(
            img,
            status_text,
            (pill_x1 + (pill_w - st_size[0]) // 2, pill_y1 + 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            badge_color,
            2,
            cv2.LINE_AA
        )

        # Top Right Info (FPS & Audio Status)
        audio_txt = "[AUDIO ON]" if audio_enabled else "[AUDIO MUTED]"
        audio_color = (0, 220, 120) if audio_enabled else (120, 120, 120)
        cv2.putText(
            img,
            f"FPS: {fps:4.1f}",
            (w - 170, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            audio_txt,
            (w - 170, 62),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            audio_color,
            1,
            cv2.LINE_AA
        )

        # 2. Right Telemetry & Biometrics Side Panel
        panel_w = 320
        px1 = w - panel_w - 15
        py1 = 90
        px2 = w - 15
        py2 = h - 50
        self.draw_glass_panel(img, (px1, py1), (px2, py2), color=(18, 22, 28), alpha=0.82)

        # Panel Header
        cv2.putText(
            img,
            "BIOMETRIC TELEMETRY",
            (px1 + 18, py1 + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )
        cv2.line(img, (px1 + 15, py1 + 38), (px2 - 15, py1 + 38), (50, 60, 72), 1)

        # Gauges
        gauge_y = py1 + 65
        gauge_w = panel_w - 36
        gauge_h = 14
        gx = px1 + 18

        # EAR Gauge
        self.draw_progress_bar(
            img, gx, gauge_y, gauge_w, gauge_h,
            val=ear, min_val=0.08, max_val=0.42,
            threshold=ear_thresh, invert_thresh=False,
            label=f"EAR  [{ear_consec:02d}/{config.EAR_CONSEC_FRAMES:02d}]",
            val_format="{:.3f}"
        )

        # MAR Gauge
        gauge_y += 55
        self.draw_progress_bar(
            img, gx, gauge_y, gauge_w, gauge_h,
            val=mar, min_val=0.10, max_val=1.20,
            threshold=mar_thresh, invert_thresh=True,
            label=f"MAR  [{mar_consec:02d}/{config.MAR_CONSEC_FRAMES:02d}]",
            val_format="{:.3f}"
        )

        # PERCLOS Gauge
        gauge_y += 55
        self.draw_progress_bar(
            img, gx, gauge_y, gauge_w, gauge_h,
            val=perclos, min_val=0.0, max_val=100.0,
            threshold=config.PERCLOS_THRESHOLD * 100.0, invert_thresh=True,
            label="PERCLOS FATIGUE INDEX",
            val_format="{:.1f}%"
        )

        # Head Orientation Angles
        gauge_y += 45
        cv2.putText(
            img,
            f"HEAD PITCH: {head_pitch:+5.1f} deg",
            (gx, gauge_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            config.COLOR_WARNING if head_pitch < config.HEAD_PITCH_NOD_THRESHOLD else (200, 210, 220),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            f"HEAD YAW:   {head_yaw:+5.1f} deg",
            (gx, gauge_y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            config.COLOR_WARNING if abs(head_yaw) > config.HEAD_YAW_DISTRACTION_THRESHOLD else (200, 210, 220),
            1,
            cv2.LINE_AA
        )

        # Calibration Progress if active
        if status_level == "CALIBRATING":
            gauge_y += 55
            self.draw_progress_bar(
                img, gx, gauge_y, gauge_w, 12,
                val=calib_pct, min_val=0.0, max_val=100.0,
                threshold=100.0, invert_thresh=False,
                label="CALIBRATION BASELINE PROGRESS",
                val_format="{:.0f}%"
            )

        # Recent Incidents Log Header
        gauge_y += 55
        cv2.line(img, (px1 + 15, gauge_y), (px2 - 15, gauge_y), (50, 60, 72), 1)
        cv2.putText(
            img,
            "RECENT EVENTS LOG",
            (gx, gauge_y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.44,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )

        log_y = gauge_y + 44
        if not recent_events:
            cv2.putText(
                img,
                "No fatigue events recorded",
                (gx, log_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.36,
                (120, 130, 140),
                1,
                cv2.LINE_AA
            )
        else:
            for ev in recent_events[-4:]:
                cv2.putText(
                    img,
                    ev,
                    (gx, log_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.35,
                    (255, 180, 80),
                    1,
                    cv2.LINE_AA
                )
                log_y += 18

        # Face Detection Lost Warning
        if not face_detected:
            warn_w, warn_h = 360, 60
            wx1 = (w - warn_w) // 2
            wy1 = h // 2 - 30
            self.draw_glass_panel(img, (wx1, wy1), (wx1 + warn_w, wy1 + warn_h), color=(30, 20, 20), alpha=0.88, border_color=(0, 0, 255))
            cv2.putText(
                img,
                "NO FACE DETECTED - POSITION CAMERA",
                (wx1 + 22, wy1 + 38),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (0, 100, 255),
                1,
                cv2.LINE_AA
            )

        # 3. Bottom Hotkeys Help Bar
        self.draw_glass_panel(img, (15, h - 42), (w - 15, h - 10), color=(18, 22, 28), alpha=0.82)
        hotkeys_text = "[Q] Quit   [C] Recalibrate   [M] Toggle Mesh   [S] Sound Toggle   [R] Reset Stats"
        cv2.putText(
            img,
            hotkeys_text,
            (32, h - 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            (170, 185, 200),
            1,
            cv2.LINE_AA
        )

        time_text = f"UPTIME: {session_time_str}"
        tt_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)[0]
        cv2.putText(
            img,
            time_text,
            (w - 32 - tt_size[0], h - 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )

        # 4. Critical Danger Flashing Border
        if status_level == "DANGER":
            self.draw_pulsing_danger_border(img)
