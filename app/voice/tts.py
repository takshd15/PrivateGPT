"""Text-to-speech for Jarvix v2.

OpenAI's TTS API is the primary voice. Local Windows TTS is a soft fallback so
Jarvix doesn't die just because the network, quota, or account has a bad moment.
"""

import re

import numpy as np
import pyttsx3
import requests
import sounddevice as sd

from app.config import (
    OPENAI_API_KEY,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_SPEED,
    OPENAI_TTS_TIMEOUT_SECONDS,
    OPENAI_TTS_VOICE,
    TTS_RATE,
    TTS_VOICE_HINTS,
    TTS_VOLUME,
)

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
OPENAI_TTS_SAMPLE_RATE = 24000  # OpenAI's "pcm" response_format: 24kHz mono s16le


def _voice_score(voice) -> int:
    haystack = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
    return sum(1 for hint in TTS_VOICE_HINTS if hint in haystack)


def _engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)

    voices = engine.getProperty("voices") or []
    best = max(voices, key=_voice_score, default=None)
    if best and _voice_score(best) > 0:
        engine.setProperty("voice", best.id)

    return engine


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _speak_local(text: str) -> None:
    engine = _engine()
    engine.say(text)
    engine.runAndWait()


def _speak_openai(text: str) -> None:
    """Stream OpenAI's TTS response straight to the speaker as it arrives,
    instead of waiting for the full clip - cuts time-to-first-sound noticeably
    on longer replies."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    with requests.post(
        OPENAI_TTS_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_TTS_MODEL,
            "voice": OPENAI_TTS_VOICE,
            "input": text,
            "response_format": "pcm",
            "speed": OPENAI_TTS_SPEED,
        },
        timeout=OPENAI_TTS_TIMEOUT_SECONDS,
        stream=True,
    ) as response:
        response.raise_for_status()
        leftover = b""
        with sd.OutputStream(samplerate=OPENAI_TTS_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            for chunk in response.iter_content(chunk_size=4800):  # ~100ms @ 24kHz s16
                if not chunk:
                    continue
                data = leftover + chunk
                usable_len = len(data) - (len(data) % 2)  # keep whole int16 samples
                leftover = data[usable_len:]
                if usable_len:
                    stream.write(np.frombuffer(data[:usable_len], dtype=np.int16))


def synthesize(text: str) -> bytes:
    """Return raw 16-bit PCM audio bytes (OPENAI_TTS_SAMPLE_RATE, mono) for
    ``text`` via OpenAI's TTS API, instead of playing it through the local
    speakers like speak()/_speak_openai() do. For HTTP callers (app/server.py)
    that need to send audio back to a browser rather than play it on the
    machine running the backend. Raises on failure - unlike speak(), there is
    no local-speaker fallback to fall back to for a remote caller; the caller
    decides how to surface the failure."""
    text = _clean_for_speech(text)
    if not text:
        return b""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    response = requests.post(
        OPENAI_TTS_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_TTS_MODEL,
            "voice": OPENAI_TTS_VOICE,
            "input": text,
            "response_format": "pcm",
            "speed": OPENAI_TTS_SPEED,
        },
        timeout=OPENAI_TTS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.content


def speak(text: str) -> None:
    text = _clean_for_speech(text)
    if not text:
        return
    try:
        _speak_openai(text)
    except Exception as exc:
        try:
            print(f"[openai tts unavailable, using local tts] {exc}")
            _speak_local(text)
        except Exception as local_exc:  # engine init / driver / device failure
            print(f"[tts unavailable, printing instead] {text}  ({local_exc})")
