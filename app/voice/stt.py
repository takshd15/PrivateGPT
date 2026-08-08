"""Speech-to-text for Jarvix v2.

Voice commands (transcribe(), below) go to AssemblyAI's Sync API: fast,
accurate, and each call is one bounded ~4s clip triggered by a confirmed wake
event. The always-on wake-word listener (transcribe_wake_word()) stays on
local faster-whisper instead: it fires on any speech energy near the mic, so
routing it to a paid cloud API would mean streaming ambient room noise
off-device continuously rather than only on confirmed commands.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

from app.config import ASSEMBLYAI_API_KEY, STT_MODEL, STT_LANGUAGE

_SYNC_URL = "https://sync.assemblyai.com/transcribe"
_WARM_URL = "https://sync.assemblyai.com/warm"
_MODEL = "universal-3-5-pro"
_KEYTERMS_MAX_CHARS = 2048  # AssemblyAI Sync API limit across all terms combined

_session = requests.Session()


@lru_cache(maxsize=1)
def _model():
    # int8 keeps memory + CPU cost low on the 7.6 GB laptop.
    from faster_whisper import WhisperModel

    return WhisperModel(STT_MODEL, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def _keyterms() -> list[str]:
    """Domain vocabulary to bias AssemblyAI's decoder: app/folder aliases plus
    a few names from config. Trimmed to the API's combined character cap."""
    from app.config import DEFAULT_WEATHER_LOCATION, USER_DISPLAY_NAME, WAKE_WORD

    terms = {WAKE_WORD, "Spotify", USER_DISPLAY_NAME, DEFAULT_WEATHER_LOCATION}

    aliases_file = Path(__file__).resolve().parents[1] / "memory" / "aliases.json"
    try:
        aliases = json.loads(aliases_file.read_text(encoding="utf-8"))
        terms.update(aliases.get("folders", {}).keys())
        terms.update(aliases.get("apps", {}).keys())
    except (OSError, json.JSONDecodeError):
        pass

    result: list[str] = []
    total = 0
    for term in sorted({t.strip() for t in terms if t and t.strip()}, key=str.lower):
        added = len(term) + (1 if result else 0)
        if total + added > _KEYTERMS_MAX_CHARS:
            break
        result.append(term)
        total += added
    return result


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _do_warm() -> None:
    try:
        _session.get(_WARM_URL, headers={"X-AAI-Model": _MODEL}, timeout=5)
    except requests.RequestException:
        pass


def warm() -> None:
    """Pre-open the AssemblyAI connection so the TLS/TCP handshake overlaps
    with recording instead of adding latency after the user finishes talking.
    Fire-and-forget: call right before record() starts. Never raises."""
    if not ASSEMBLYAI_API_KEY:
        return
    threading.Thread(target=_do_warm, daemon=True).start()


def transcribe(audio: np.ndarray, language: str | None = None) -> str:
    """Transcribe a 16 kHz mono voice-command clip via AssemblyAI's Sync API."""
    if audio is None or audio.size == 0:
        return ""
    if not ASSEMBLYAI_API_KEY:
        raise ValueError("AssemblyAI API key is not configured in .env.")

    config = {
        "sample_rate": 16000,
        "channels": 1,
        "language_code": language or STT_LANGUAGE,
        "keyterms_prompt": _keyterms(),
    }
    try:
        response = _session.post(
            _SYNC_URL,
            headers={"Authorization": ASSEMBLYAI_API_KEY, "X-AAI-Model": _MODEL},
            files={
                "audio": ("command.pcm", _pcm16_bytes(audio), "audio/pcm"),
                "config": (None, json.dumps(config), "application/json"),
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    return response.json().get("text", "").strip()


def transcribe_wake_word(audio: np.ndarray, language: str | None = None) -> str:
    """Accuracy-biased transcription for the short wake-listening window.

    Wake recognition is worth a little extra CPU: a wider beam recovers muffled
    speech, while ``hotwords`` teaches Whisper the likely spellings without
    changing how ordinary commands are transcribed.
    """
    if audio is None or audio.size == 0:
        return ""

    # Bring quiet speech into a useful range without aggressively amplifying
    # room noise. Whisper accepts float32 mono audio in [-1, 1].
    samples = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if 0 < rms < 0.06:
        gain = min(4.0, 0.06 / rms)
        samples = np.clip(samples * gain, -1.0, 1.0)

    segments, _ = _model().transcribe(
        samples,
        language=language or STT_LANGUAGE,
        beam_size=5,
        hotwords=(
            "Jarvis Jarvix Jervis Jorvis Jorvix Garvis Zarvis Charvis "
            "Joris Jorwes Doris Chavez"
        ),
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()
