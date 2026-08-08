"""Microphone capture for Jarvix v2.

Records mono 16 kHz float32 audio (the format faster-whisper expects) and stops
automatically after a short trailing silence, so the user does not have to time
their command. Falls back to a hard max duration so it can never hang forever.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from app.config import MIC_SILENCE_THRESHOLD, VOICE_RECORD_SECONDS

SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.1


class MicUnavailable(Exception):
    """Raised when the microphone can't be opened or read."""


def record(
    max_seconds: float | None = None,
    silence_threshold: float | None = None,
    trailing_silence: float = 1.0,
    start_timeout: float = 5.0,
) -> np.ndarray:
    """Record from the default mic until silence or ``max_seconds``.

    Returns a 1-D float32 array at 16 kHz. Returns an empty array if the user
    never started speaking within ``start_timeout``. Raises ``MicUnavailable``
    if there is no usable microphone, so callers can fail gracefully.
    """
    if max_seconds is None:
        max_seconds = float(VOICE_RECORD_SECONDS)
    if silence_threshold is None:
        silence_threshold = MIC_SILENCE_THRESHOLD

    block = int(SAMPLE_RATE * _BLOCK_SECONDS)
    max_blocks = int(max_seconds / _BLOCK_SECONDS)
    trailing_blocks = int(trailing_silence / _BLOCK_SECONDS)
    start_blocks = int(start_timeout / _BLOCK_SECONDS)

    frames: list[np.ndarray] = []
    started = False
    silent_run = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block
        ) as stream:
            for i in range(max_blocks):
                data, _ = stream.read(block)
                samples = data[:, 0]
                rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0

                if rms >= silence_threshold:
                    started = True
                    silent_run = 0
                    frames.append(samples.copy())
                else:
                    silent_run += 1
                    if started:
                        frames.append(samples.copy())
                        if silent_run >= trailing_blocks:
                            break
                    elif i >= start_blocks:
                        break
    except Exception as exc:  # PortAudioError, no device, etc.
        raise MicUnavailable(str(exc)) from exc

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)
