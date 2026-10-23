"""
logger.py - Event Logging for AI Driver Drowsiness System
Records all fatigue, micro-sleep, yawn, and distraction incidents to CSV.
"""

import csv
import datetime
import os
from typing import Optional

import config


class EventLogger:
    """Logs drowsiness events to a CSV file and maintains session event counts."""

    HEADERS = [
        "Timestamp",
        "Event_Type",
        "Duration_Sec",
        "Duration_Frames",
        "EAR",
        "EAR_Threshold",
        "MAR",
        "MAR_Threshold",
        "PERCLOS_Pct",
        "Head_Pitch_Deg",
        "Head_Yaw_Deg",
        "Notes"
    ]

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or config.LOG_FILE_PATH
        self.session_counts = {
            "DROWSY": 0,
            "YAWN": 0,
            "HEAD_NOD": 0,
            "DISTRACTION": 0
        }
        self.recent_events = []  # Last 5 events for UI display
        self._init_csv()

    def _init_csv(self):
        """Create file and write header if not present."""
        try:
            folder = os.path.dirname(self.log_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
                with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADERS)
        except Exception as e:
            print(f"[EventLogger] Error initializing CSV: {e}")

    def log_event(
        self,
        event_type: str,
        duration_sec: float,
        duration_frames: int,
        ear: float,
        ear_thresh: float,
        mar: float,
        mar_thresh: float,
        perclos: float,
        pitch: float,
        yaw: float,
        notes: str = ""
    ):
        """Append an incident record to CSV and in-memory log."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update counter
        key = event_type.split()[0].upper()
        if key in self.session_counts:
            self.session_counts[key] += 1
        elif "YAWN" in event_type:
            self.session_counts["YAWN"] += 1
        elif "DROWS" in event_type:
            self.session_counts["DROWSY"] += 1
        elif "NOD" in event_type:
            self.session_counts["HEAD_NOD"] += 1

        row = [
            timestamp,
            event_type,
            f"{duration_sec:.2f}",
            duration_frames,
            f"{ear:.3f}",
            f"{ear_thresh:.3f}",
            f"{mar:.3f}",
            f"{mar_thresh:.3f}",
            f"{perclos:.1f}",
            f"{pitch:.1f}",
            f"{yaw:.1f}",
            notes
        ]

        # Add to recent events memory for UI
        self.recent_events.append(f"{timestamp[-8:]} {event_type} ({duration_sec:.1f}s)")
        if len(self.recent_events) > 6:
            self.recent_events.pop(0)

        # Write to disk
        try:
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[EventLogger] Failed to write event to CSV: {e}")
