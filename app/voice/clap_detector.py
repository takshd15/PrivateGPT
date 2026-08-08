"""Double-clap wake detector for Jarvix v2.

Designed to be robust on laptop mics whose noise suppression weakens transients.
Two ideas from clap/onset-detection research:

1. ADAPTIVE threshold - a clap is loud *relative to the recent ambient level*,
   not at some fixed absolute volume. We track a running ambient estimate and
   fire when a block jumps well above it (with an absolute floor as a backstop).

2. SHORT duration - a clap is a brief impulse (tens of ms); a spoken word
   sustains for 100-300 ms. So a loud burst only counts as a clap if it ends
   quickly. This is what rejects "hello" even though it's loud.

Two qualifying claps within 0.12-0.8 s trigger the wake.
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from app.config import CLAP_THRESHOLD
from app.voice.recorder import MicUnavailable

SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.02      # 20 ms blocks
MAX_CLAP_SECONDS = 0.18    # loud burst must be brief; speech sustains longer
SPIKE_RATIO = 6.0          # clap = this many times the ambient level
AMBIENT_ALPHA = 0.95       # smoothing for the running ambient estimate


def brightness(samples: np.ndarray) -> float:
    """High-frequency / total energy ratio (for the calibrate display only)."""
    if samples.size < 2:
        return 0.0
    total = float(np.mean(samples**2)) + 1e-9
    return float(np.mean(np.diff(samples) ** 2)) / total


class _ClapTracker:
    """Streaming event detector. ``push`` is called per block with the block's
    peak and the current dynamic threshold; it returns a result dict when a loud
    burst ends (classified clap or not)."""

    def __init__(self, block_s: float = _BLOCK_SECONDS):
        self.block_s = block_s
        self.in_event = False
        self.run = 0
        self.max_peak = 0.0
        self.low = 0.0

    def push(self, peak: float, threshold: float) -> dict | None:
        if not self.in_event:
            if peak >= threshold:          # loud onset starts an event
                self.in_event = True
                self.run = 1
                self.max_peak = peak
                self.low = threshold * 0.5
            return None

        if peak >= self.low:               # event continues while still loud
            self.run += 1
            self.max_peak = max(self.max_peak, peak)
            return None

        duration = self.run * self.block_s  # burst ended -> classify by length
        is_clap = duration <= MAX_CLAP_SECONDS
        result = {
            "is_clap": is_clap,
            "peak": self.max_peak,
            "duration_ms": duration * 1000.0,
            "reason": "" if is_clap else "too long (speech?)",
        }
        self.in_event = False
        self.run = 0
        return result


def _dynamic_threshold(ambient: float) -> float:
    """Loud-enough cutoff: the larger of the absolute floor and ambient*ratio."""
    return max(CLAP_THRESHOLD, ambient * SPIKE_RATIO)


def wait_for_double_clap(
    threshold: float | None = None,  # kept for API compat; ignored (adaptive now)
    min_gap: float = 0.12,
    max_gap: float = 0.8,
    timeout: float | None = None,
    verbose: bool = False,
) -> bool:
    """Block until a double clap is heard. Returns True on detection, or False
    if ``timeout`` seconds pass first (None = wait forever)."""
    block = int(SAMPLE_RATE * _BLOCK_SECONDS)
    tracker = _ClapTracker()
    ambient = 0.005
    last_clap_t: float | None = None
    start = time.time()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block
        ) as stream:
            while True:
                if timeout is not None and (time.time() - start) > timeout:
                    return False

                data, _ = stream.read(block)
                samples = data[:, 0]
                if not samples.size:
                    continue
                rms = float(np.sqrt(np.mean(samples**2)))
                peak = float(np.max(np.abs(samples)))

                result = tracker.push(peak, _dynamic_threshold(ambient))
                if not tracker.in_event:  # only learn ambient when it's quiet
                    ambient = AMBIENT_ALPHA * ambient + (1 - AMBIENT_ALPHA) * rms

                if result and result["is_clap"]:
                    now = time.time()
                    if verbose:
                        print(f"  clap! (peak {result['peak']:.3f}, "
                              f"{result['duration_ms']:.0f}ms)")
                    if last_clap_t is not None and min_gap <= (now - last_clap_t) <= max_gap:
                        return True
                    last_clap_t = now
    except Exception as exc:  # PortAudioError, no device, etc.
        raise MicUnavailable(str(exc)) from exc
