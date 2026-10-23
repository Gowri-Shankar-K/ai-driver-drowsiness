"""
main.py - AI Driver Drowsiness & Fatigue Detection System
Main execution entry point.
"""

import argparse
import sys
import time
import cv2

import config
from alerts import AudioAlertManager
from calibration import DriverCalibrator
from detector import FaceMeshDetector
from logger import EventLogger
from metrics import PerclosTracker
from ui_overlay import HUDOverlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Driver Drowsiness & Fatigue Monitoring System"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to pre-recorded video file for offline testing"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.FRAME_WIDTH,
        help=f"Target frame width (default: {config.FRAME_WIDTH})"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.FRAME_HEIGHT,
        help=f"Target frame height (default: {config.FRAME_HEIGHT})"
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audio buzzer alerts"
    )
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=None,
        help="Manual EAR threshold override (e.g., 0.25)"
    )
    parser.add_argument(
        "--mar-thresh",
        type=float,
        default=None,
        help="Manual MAR threshold override (e.g., 0.65)"
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip personalized initial baseline calibration"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  AI Driver Drowsiness & Fatigue Detection System")
    print("  Based on Soukupova & Cech (2016), NHTSA PERCLOS, and MediaPipe")
    print("=" * 65)

    # 1. Initialize Subsystems
    print("[1/5] Initializing MediaPipe Face Landmark Detector...")
    detector = FaceMeshDetector()

    print("[2/5] Initializing Audio Alert Engine...")
    audio = AudioAlertManager(enabled=not args.no_sound)

    print("[3/5] Initializing Event Logger...")
    logger = EventLogger()

    print("[4/5] Initializing Calibration & Metrics...")
    calibrator = DriverCalibrator()
    if args.skip_calibration:
        calibrator.status = DriverCalibrator.STATUS_COMPLETE
        if args.ear_thresh:
            calibrator.ear_threshold = args.ear_thresh
        if args.mar_thresh:
            calibrator.mar_threshold = args.mar_thresh
    else:
        calibrator.start()

    perclos_tracker = PerclosTracker(window_size=config.PERCLOS_WINDOW_FRAMES)
    hud = HUDOverlay()

    # 2. Setup Video Capture
    video_source = args.video if args.video else args.camera
    print(f"[5/5] Opening video source: {video_source}...")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        print("Please verify your webcam connection or input video file path.")
        return 1

    # Attempt to set desired capture resolution (hardware-permitting)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution established: {actual_w}x{actual_h}")
    print("\nStarting video stream. Press 'q' in the video window to exit.\n")

    # State Tracking Counters
    ear_consec_frames = 0
    mar_consec_frames = 0
    pitch_consec_frames = 0
    yaw_consec_frames = 0
    perclos_val = 0.0

    drowsy_event_active = False
    drowsy_start_time = 0.0

    yawn_event_active = False
    yawn_start_time = 0.0

    # FPS Calculation
    prev_frame_time = time.time()
    fps = 0.0
    start_session_time = time.time()

    window_name = "AI Driver Drowsiness Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    # Loop video file for continuous testing
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARNING] Frame capture dropped or camera disconnected.")
                    break

            # Calculate FPS
            curr_frame_time = time.time()
            time_delta = curr_frame_time - prev_frame_time
            if time_delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_delta)
            prev_frame_time = curr_frame_time

            # Format Uptime
            session_elapsed = int(curr_frame_time - start_session_time)
            mins, secs = divmod(session_elapsed, 60)
            hours, mins = divmod(mins, 60)
            session_time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

            # Process frame for landmarks and metrics
            detection = detector.process_frame(frame)

            status_text = "DRIVER ALERT"
            status_level = "SAFE"

            # Check Calibration Phase
            if calibrator.status == DriverCalibrator.STATUS_CALIBRATING:
                status_text = f"CALIBRATING ({calibrator.progress_pct:.0f}%)"
                status_level = "CALIBRATING"
                if detection.face_detected:
                    calibrated_now = calibrator.update(
                        detection.ear,
                        detection.mar,
                        detection.head_pitch,
                        detection.head_yaw
                    )
                    if calibrated_now:
                        if args.ear_thresh:
                            calibrator.ear_threshold = args.ear_thresh
                        if args.mar_thresh:
                            calibrator.mar_threshold = args.mar_thresh

            # Operational Monitoring Phase
            elif calibrator.is_calibrated:
                ear_thresh = calibrator.ear_threshold
                mar_thresh = calibrator.mar_threshold

                is_eye_closed = False
                if detection.face_detected:
                    # Blendshape assisted eye-closure validation
                    eye_closed_ear = detection.ear < ear_thresh
                    eye_closed_blendshape = (
                        detection.blendshape_blink_left > 0.60 and
                        detection.blendshape_blink_right > 0.60
                    )
                    # Either EAR is below threshold or blendshape confirms closed eyes
                    is_eye_closed = eye_closed_ear or eye_closed_blendshape

                perclos_val = perclos_tracker.update(is_eye_closed)

                # --- 1. Drowsiness / Eye Closure Evaluation ---
                if is_eye_closed:
                    ear_consec_frames += 1
                    if ear_consec_frames >= config.EAR_CONSEC_FRAMES:
                        status_level = "DANGER"
                        status_text = "DROWSY ALARM - WAKE UP!"
                        audio.trigger_alert(AudioAlertManager.ALERT_DROWSY)

                        if not drowsy_event_active:
                            drowsy_event_active = True
                            drowsy_start_time = time.time()
                else:
                    if drowsy_event_active:
                        dur = time.time() - drowsy_start_time
                        logger.log_event(
                            event_type="DROWSY - EYES CLOSED",
                            duration_sec=dur,
                            duration_frames=ear_consec_frames,
                            ear=detection.ear,
                            ear_thresh=ear_thresh,
                            mar=detection.mar,
                            mar_thresh=mar_thresh,
                            perclos=perclos_val,
                            pitch=detection.head_pitch,
                            yaw=detection.head_yaw,
                            notes="Extended eye closure detected"
                        )
                        drowsy_event_active = False
                    ear_consec_frames = 0

                # --- 2. Yawning Evaluation ---
                is_yawning = False
                if detection.face_detected:
                    is_yawning = (detection.mar >= mar_thresh) or (detection.blendshape_jaw_open > 0.65)

                if is_yawning:
                    mar_consec_frames += 1
                    if mar_consec_frames >= config.MAR_CONSEC_FRAMES:
                        if status_level != "DANGER":
                            status_level = "WARNING"
                            status_text = "WARNING: YAWN DETECTED"
                        audio.trigger_alert(AudioAlertManager.ALERT_YAWN)

                        if not yawn_event_active:
                            yawn_event_active = True
                            yawn_start_time = time.time()
                else:
                    if yawn_event_active:
                        dur = time.time() - yawn_start_time
                        logger.log_event(
                            event_type="YAWN DETECTED",
                            duration_sec=dur,
                            duration_frames=mar_consec_frames,
                            ear=detection.ear,
                            ear_thresh=ear_thresh,
                            mar=detection.mar,
                            mar_thresh=mar_thresh,
                            perclos=perclos_val,
                            pitch=detection.head_pitch,
                            yaw=detection.head_yaw,
                            notes="Frequent or continuous yawn"
                        )
                        yawn_event_active = False
                    mar_consec_frames = 0

                # --- 3. Head Nodding (Microsleep) Evaluation ---
                if detection.face_detected and detection.head_pitch < config.HEAD_PITCH_NOD_THRESHOLD:
                    pitch_consec_frames += 1
                    if pitch_consec_frames >= config.HEAD_CONSEC_FRAMES:
                        if status_level != "DANGER":
                            status_level = "DANGER"
                            status_text = "DANGER: HEAD NOD / MICROSLEEP"
                        audio.trigger_alert(AudioAlertManager.ALERT_HEAD_NOD)
                else:
                    pitch_consec_frames = 0

                # --- 4. Distraction (Looking Away) Evaluation ---
                if detection.face_detected and abs(detection.head_yaw) > config.HEAD_YAW_DISTRACTION_THRESHOLD:
                    yaw_consec_frames += 1
                    if yaw_consec_frames >= config.HEAD_CONSEC_FRAMES:
                        if status_level == "SAFE":
                            status_level = "WARNING"
                            status_text = "DISTRACTED: LOOKING AWAY"
                        audio.trigger_alert(AudioAlertManager.ALERT_DISTRACTION)
                else:
                    yaw_consec_frames = 0

                # --- 5. PERCLOS Chronic Fatigue Check ---
                if perclos_val > (config.PERCLOS_THRESHOLD * 100.0) and status_level == "SAFE":
                    status_level = "WARNING"
                    status_text = f"FATIGUE RISK (PERCLOS {perclos_val:.0f}%)"

            # Draw facial mesh contours
            hud.draw_landmarks_mesh(
                frame,
                detection.landmarks_px,
                is_eyes_closed=(ear_consec_frames > 5),
                is_yawning=(mar_consec_frames > 5)
            )

            # Draw full telemetry HUD
            hud.draw_hud(
                img=frame,
                ear=detection.ear,
                ear_thresh=calibrator.ear_threshold,
                mar=detection.mar,
                mar_thresh=calibrator.mar_threshold,
                perclos=perclos_tracker.update(False) if not calibrator.is_calibrated else perclos_val,
                head_pitch=detection.head_pitch,
                head_yaw=detection.head_yaw,
                status_text=status_text,
                status_level=status_level,
                ear_consec=ear_consec_frames,
                mar_consec=mar_consec_frames,
                fps=fps,
                calib_pct=calibrator.progress_pct,
                session_time_str=session_time_str,
                recent_events=logger.recent_events,
                audio_enabled=audio.enabled,
                face_detected=detection.face_detected
            )

            cv2.imshow(window_name, frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("\n[INFO] Exit command received. Shutting down...")
                break
            elif key in (ord('c'), ord('C')):
                print("[INFO] Initiating driver recalibration...")
                calibrator.start()
            elif key in (ord('m'), ord('M')):
                hud.show_mesh = not hud.show_mesh
                print(f"[INFO] Landmark mesh overlay: {'ON' if hud.show_mesh else 'OFF'}")
            elif key in (ord('s'), ord('S')):
                state = audio.toggle()
                print(f"[INFO] Audio alerts: {'ENABLED' if state else 'MUTED'}")
            elif key in (ord('r'), ord('R')):
                perclos_tracker.reset()
                ear_consec_frames = 0
                mar_consec_frames = 0
                pitch_consec_frames = 0
                yaw_consec_frames = 0
                print("[INFO] Session fatigue metrics reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user interruption.")
    finally:
        # Flush any in-progress active event to CSV
        if drowsy_event_active:
            dur = time.time() - drowsy_start_time
            logger.log_event(
                event_type="DROWSY - EYES CLOSED",
                duration_sec=dur,
                duration_frames=ear_consec_frames,
                ear=detection.ear,
                ear_thresh=calibrator.ear_threshold,
                mar=detection.mar,
                mar_thresh=calibrator.mar_threshold,
                perclos=perclos_val,
                pitch=detection.head_pitch,
                yaw=detection.head_yaw,
                notes="Session ended during event"
            )
        if yawn_event_active:
            dur = time.time() - yawn_start_time
            logger.log_event(
                event_type="YAWN DETECTED",
                duration_sec=dur,
                duration_frames=mar_consec_frames,
                ear=detection.ear,
                ear_thresh=calibrator.ear_threshold,
                mar=detection.mar,
                mar_thresh=calibrator.mar_threshold,
                perclos=perclos_val,
                pitch=detection.head_pitch,
                yaw=detection.head_yaw,
                notes="Session ended during event"
            )

        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[INFO] System shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
