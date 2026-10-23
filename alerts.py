"""
alerts.py - Audio Alert System with Non-Blocking Synthesized Sound Generator
"""

import math
import sys
import threading
import time
import numpy as np

import config

# Try initializing pygame mixer; if not available, will use winsound (Windows)
PYGAME_AVAILABLE = False
try:
    import pygame
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

WINSOUND_AVAILABLE = False
if sys.platform == "win32":
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except ImportError:
        pass


class AudioAlertManager:
    """Non-blocking, low-latency audio alert synthesizer and player."""

    ALERT_DROWSY = "DROWSY"
    ALERT_YAWN = "YAWN"
    ALERT_HEAD_NOD = "HEAD_NOD"
    ALERT_DISTRACTION = "DISTRACTION"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and config.AUDIO_ALERT_ENABLED
        self.last_alert_time = 0.0
        self.cooldown = config.AUDIO_COOLDOWN_SEC
        self._sound_cache = {}
        self._lock = threading.Lock()

        if PYGAME_AVAILABLE:
            self._pregenerate_sounds()

    def _generate_tone_sound(self, frequency: int, duration_ms: int, volume: float = 0.8):
        """Synthesize a pure sine wave tone into a pygame Sound object."""
        sample_rate = 44100
        n_samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
        # Apply slight Hann fade in/out to avoid speaker pops
        fade_len = min(int(sample_rate * 0.01), n_samples // 4)
        envelope = np.ones(n_samples)
        if fade_len > 0:
            envelope[:fade_len] = np.sin(np.linspace(0, np.pi / 2, fade_len))
            envelope[-fade_len:] = np.sin(np.linspace(np.pi / 2, 0, fade_len))

        waveform = np.sin(2.0 * np.pi * frequency * t) * envelope
        waveform = (waveform * volume * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(waveform)
        return sound

    def _pregenerate_sounds(self):
        """Pre-render synthesized sound waves to eliminate playback latency."""
        try:
            # Urgent high-pitch alarm for eyes closed (2600Hz)
            self._sound_cache[self.ALERT_DROWSY] = self._generate_tone_sound(2600, 300, 0.9)
            # Amber alert for yawn (1800Hz)
            self._sound_cache[self.ALERT_YAWN] = self._generate_tone_sound(1800, 220, 0.7)
            # Rapid chirp for head nodding down (2200Hz)
            self._sound_cache[self.ALERT_HEAD_NOD] = self._generate_tone_sound(2200, 250, 0.8)
            # Dual alert for distraction (1400Hz)
            self._sound_cache[self.ALERT_DISTRACTION] = self._generate_tone_sound(1400, 200, 0.6)
        except Exception as e:
            # Fallback will use winsound if pygame sound generation fails
            pass

    def trigger_alert(self, alert_type: str = ALERT_DROWSY):
        """
        Trigger an audio alert if cooldown has elapsed.
        Always runs asynchronously in non-blocking fashion.
        """
        if not self.enabled:
            return

        current_time = time.time()
        with self._lock:
            if current_time - self.last_alert_time < self.cooldown:
                return
            self.last_alert_time = current_time

        # Play sound asynchronously
        threading.Thread(target=self._play_sound_worker, args=(alert_type,), daemon=True).start()

    def _play_sound_worker(self, alert_type: str):
        """Asynchronous worker for sound playback."""
        try:
            if PYGAME_AVAILABLE and alert_type in self._sound_cache:
                self._sound_cache[alert_type].play()
                return

            if WINSOUND_AVAILABLE:
                freq = config.AUDIO_FREQUENCY
                dur = config.AUDIO_DURATION_MS
                if alert_type == self.ALERT_YAWN:
                    freq = 1800
                    dur = 180
                elif alert_type == self.ALERT_DISTRACTION:
                    freq = 1500
                    dur = 150
                winsound.Beep(freq, dur)
        except Exception:
            pass

    def toggle(self) -> bool:
        """Toggle audio alerts on/off."""
        self.enabled = not self.enabled
        return self.enabled
