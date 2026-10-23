"""
simulate.py - Simulated Driver Demonstration & HUD Showcase

Runs the AI Driver Drowsiness system in simulation mode to showcase:
- Real-time HUD and glassmorphic telemetry panels
- Baseline calibration sequence
- Drowsiness alarms (Eye Aspect Ratio drop)
- Yawn alerts (Mouth Aspect Ratio spike)
- Head nodding and distraction warnings
- Non-blocking audio warning tones and CSV incident logging

Useful for verifying the full visual and audio pipeline without requiring a physical camera.
"""

import sys
import time
import cv2
import numpy as np

import config
from alerts import AudioAlertManager
from calibration import DriverCalibrator
from logger import EventLogger
from metrics import PerclosTracker
from ui_overlay import HUDOverlay


def run_simulation(duration_sec: int = 25, save_screenshot: bool = True):
    print("=" * 65)
    print("  AI Driver Drowsiness System - Live Simulation Showcase")
    print("=" * 65)

    audio = AudioAlertManager(enabled=True)
    logger = EventLogger()
    calibrator = DriverCalibrator(target_frames=45)
    calibrator.start()
    perclos_tracker = PerclosTracker(window_size=config.PERCLOS_WINDOW_FRAMES)
    hud = HUDOverlay()

    w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    window_name = "AI Driver Drowsiness System - Simulation Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    start_time = time.time()
    frame_idx = 0
    fps = 30.0

    ear_consec_frames = 0
    mar_consec_frames = 0
    perclos_val = 0.0

    drowsy_event_active = False
    drowsy_start = 0.0

    yawn_event_active = False
    yawn_start = 0.0

    print("Running simulated driver scenarios:")
    print("  0s - 3s:  Baseline Personalized Calibration")
    print("  3s - 7s:  Normal Alert Driving")
    print("  7s - 12s: Drowsiness & Prolonged Eye Closure (Alarm Trigger)")
    print(" 12s - 16s: Yawning Episode (Amber Warning)")
    print(" 16s - 20s: Head Nodding / Microsleep")
    print(" 20s - 25s: Safe Recovery & Telemetry Logging")
    print("\nPress 'q' in the window at any time to exit.\n")

    screenshot_taken = False

    while True:
        elapsed = time.time() - start_time
        if elapsed > duration_sec:
            break

        frame_idx += 1

        # Background simulator visual: sleek dark automotive cockpit backdrop
        frame = np.full((h, w, 3), (24, 28, 34), dtype=np.uint8)

        # Draw a simulated horizon / road grid
        cv2.line(frame, (0, h // 2 + 50), (w, h // 2 + 50), (45, 55, 65), 2)
        cv2.line(frame, (w // 2, h // 2 + 50), (w // 4, h), (40, 50, 60), 2)
        cv2.line(frame, (w // 2, h // 2 + 50), (3 * w // 4, h), (40, 50, 60), 2)

        # Simulation States
        sim_ear = 0.32
        sim_mar = 0.22
        sim_pitch = 0.0
        sim_yaw = 0.0
        scenario_desc = "NORMAL DRIVING"

        if elapsed < 3.0:
            scenario_desc = "CALIBRATING BASELINE"
            sim_ear = 0.31 + 0.02 * np.sin(frame_idx * 0.2)
            sim_mar = 0.23 + 0.01 * np.cos(frame_idx * 0.2)
        elif 3.0 <= elapsed < 7.0:
            scenario_desc = "ALERT DRIVING"
            sim_ear = 0.32 + 0.02 * np.sin(frame_idx * 0.3)
            sim_mar = 0.24 + 0.01 * np.sin(frame_idx * 0.1)
            # Occasional natural blink
            if 4.8 <= elapsed <= 5.0:
                sim_ear = 0.08
        elif 7.0 <= elapsed < 12.0:
            scenario_desc = "DROWSY - EYES CLOSING"
            # Progressive eye closure dropping below threshold
            sim_ear = 0.12 + 0.02 * np.sin(frame_idx * 0.1)
            sim_mar = 0.22
        elif 12.0 <= elapsed < 16.0:
            scenario_desc = "YAWNING"
            sim_ear = 0.28
            # Wide mouth opening
            sim_mar = 0.78 + 0.05 * np.sin(frame_idx * 0.2)
        elif 16.0 <= elapsed < 20.0:
            scenario_desc = "MICROSLEEP / HEAD NOD"
            sim_ear = 0.14
            sim_pitch = -19.5
        else:
            scenario_desc = "RECOVERED / NORMAL"
            sim_ear = 0.33
            sim_mar = 0.22
            sim_pitch = -2.0

        status_text = "DRIVER ALERT"
        status_level = "SAFE"

        # Calibration
        if calibrator.status == DriverCalibrator.STATUS_CALIBRATING:
            status_text = f"CALIBRATING ({calibrator.progress_pct:.0f}%)"
            status_level = "CALIBRATING"
            calibrator.update(sim_ear, sim_mar, sim_pitch, sim_yaw)
        elif calibrator.is_calibrated:
            ear_thresh = calibrator.ear_threshold
            mar_thresh = calibrator.mar_threshold
            is_closed = sim_ear < ear_thresh
            perclos_val = perclos_tracker.update(is_closed)

            # EAR Check
            if is_closed:
                ear_consec_frames += 1
                if ear_consec_frames >= config.EAR_CONSEC_FRAMES:
                    status_level = "DANGER"
                    status_text = "DROWSY ALARM - WAKE UP!"
                    audio.trigger_alert(AudioAlertManager.ALERT_DROWSY)
                    if not drowsy_event_active:
                        drowsy_event_active = True
                        drowsy_start = time.time()
            else:
                if drowsy_event_active:
                    dur = time.time() - drowsy_start
                    logger.log_event(
                        "DROWSY - EYES CLOSED", dur, ear_consec_frames,
                        sim_ear, ear_thresh, sim_mar, mar_thresh, perclos_val,
                        sim_pitch, sim_yaw, "Simulated fatigue"
                    )
                    drowsy_event_active = False
                ear_consec_frames = 0

            # MAR Check
            if sim_mar >= mar_thresh:
                mar_consec_frames += 1
                if mar_consec_frames >= config.MAR_CONSEC_FRAMES:
                    if status_level != "DANGER":
                        status_level = "WARNING"
                        status_text = "WARNING: YAWN DETECTED"
                    audio.trigger_alert(AudioAlertManager.ALERT_YAWN)
                    if not yawn_event_active:
                        yawn_event_active = True
                        yawn_start = time.time()
            else:
                if yawn_event_active:
                    dur = time.time() - yawn_start
                    logger.log_event(
                        "YAWN DETECTED", dur, mar_consec_frames,
                        sim_ear, ear_thresh, sim_mar, mar_thresh, perclos_val,
                        sim_pitch, sim_yaw, "Simulated yawn"
                    )
                    yawn_event_active = False
                mar_consec_frames = 0

            # Pitch Check
            if sim_pitch < config.HEAD_PITCH_NOD_THRESHOLD:
                if status_level != "DANGER":
                    status_level = "DANGER"
                    status_text = "DANGER: HEAD NOD / MICROSLEEP"
                audio.trigger_alert(AudioAlertManager.ALERT_HEAD_NOD)

        # Draw scenario label in center
        cv2.putText(
            frame,
            f"SCENARIO: {scenario_desc}",
            (w // 2 - 190, h // 2 + 160),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )

        # Render HUD Overlay
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"00:{mins:02d}:{secs:02d}"
        hud.draw_hud(
            img=frame,
            ear=sim_ear,
            ear_thresh=calibrator.ear_threshold,
            mar=sim_mar,
            mar_thresh=calibrator.mar_threshold,
            perclos=perclos_val if calibrator.is_calibrated else 0.0,
            head_pitch=sim_pitch,
            head_yaw=sim_yaw,
            status_text=status_text,
            status_level=status_level,
            ear_consec=ear_consec_frames,
            mar_consec=mar_consec_frames,
            fps=fps,
            calib_pct=calibrator.progress_pct,
            session_time_str=time_str,
            recent_events=logger.recent_events,
            audio_enabled=audio.enabled,
            face_detected=True
        )

        # Save screenshot during the critical alarm phase
        if save_screenshot and status_level == "DANGER" and not screenshot_taken:
            screenshot_path = "simulation_alarm_hud.png"
            cv2.imwrite(screenshot_path, frame)
            screenshot_taken = True
            print(f"[Showcase] Saved alarm HUD screenshot to {screenshot_path}")

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(33) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    # Flush any trailing active events
    if drowsy_event_active:
        dur = time.time() - drowsy_start
        logger.log_event(
            "DROWSY - EYES CLOSED", dur, ear_consec_frames,
            sim_ear, calibrator.ear_threshold, sim_mar, calibrator.mar_threshold,
            perclos_val, sim_pitch, sim_yaw, "Simulated fatigue"
        )
    if yawn_event_active:
        dur = time.time() - yawn_start
        logger.log_event(
            "YAWN DETECTED", dur, mar_consec_frames,
            sim_ear, calibrator.ear_threshold, sim_mar, calibrator.mar_threshold,
            perclos_val, sim_pitch, sim_yaw, "Simulated yawn"
        )

    cv2.destroyAllWindows()
    print("\n[Showcase] Simulation completed successfully.")
    print(f"Logged events recorded to: {config.LOG_FILE_PATH}")


if __name__ == "__main__":
    run_simulation(duration_sec=12)
