"""Local HTTP bridge for the JARVIX web frontend.

Not a deployed service - this is a localhost dev server exposing the exact
same dialogue/orchestrator stack the `route` and `wake` Typer commands in
main.py already drive, over HTTP/SSE instead of a terminal or microphone.
Single-user, no auth, CORS wide open - matches the rest of this codebase's
existing single-tenant posture (USER_ID is one hardcoded env var already).

Run from the `jarvix/` directory:

    ./.venv/Scripts/python.exe -m uvicorn app.server:app --reload --port 8000

Endpoints:
    GET  /api/health      -> {"status": "ok"} - frontend connection indicator.
    POST /api/command     -> text/event-stream of live orchestrator progress,
                              ending with a {"type": "final", "text": ...} event.
                              Body: {"session_id": str, "text": str}.
    POST /api/transcribe  -> {"text": str} - real speech-to-text on a browser
                              mic recording (multipart file upload).
    POST /api/speak       -> audio/wav bytes - real text-to-speech.
                              Body: {"text": str}.
    GET  /api/briefing    -> {"text": str} - the same daily briefing text the
                              terminal's `brief`/`wake` welcome routine speaks.
"""

from __future__ import annotations

import io
import json
import queue
import struct
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.brain.dialogue import VoiceDialogue
from app.main import _brief_text, known_project_names, run_intent
from app.voice import stt, tts

app = FastAPI(title="Jarvix Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One VoiceDialogue per browser session, keyed by a client-generated id - the
# same per-session ownership model `wake`/`route` already use, just fanned
# out to multiple concurrent sessions instead of one process-lifetime instance.
_SESSIONS: dict[str, VoiceDialogue] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(session_id: str) -> VoiceDialogue:
    with _SESSIONS_LOCK:
        dialogue = _SESSIONS.get(session_id)
        if dialogue is None:
            dialogue = VoiceDialogue(known_entities=known_project_names)
            _SESSIONS[session_id] = dialogue
        return dialogue


class CommandRequest(BaseModel):
    session_id: str
    text: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post("/api/command")
def command(req: CommandRequest) -> StreamingResponse:
    dialogue = _get_session(req.session_id)
    events: queue.Queue = queue.Queue()
    SENTINEL = object()

    def on_event(event: dict) -> None:
        events.put(event)

    def worker() -> None:
        try:
            final_text = dialogue.handle(req.text, run_intent, on_event=on_event)
            events.put({"type": "final", "text": final_text})
        except Exception as exc:
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is SENTINEL:
                break
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Voice: real speech-to-text and text-to-speech for the browser client.
# app/voice/stt.py and app/voice/tts.py were built around a local mic
# (numpy array already in hand) and local speakers (sounddevice playback) -
# these endpoints are the adapter layer that lets a browser use the exact
# same underlying AssemblyAI/OpenAI calls over HTTP instead.
# --------------------------------------------------------------------------- #
_TARGET_SAMPLE_RATE = 16000  # matches app/voice/recorder.py's SAMPLE_RATE


def _decode_browser_audio(raw: bytes) -> np.ndarray:
    """Decode a browser MediaRecorder clip (webm/opus, or whatever the
    browser sent) into 16kHz mono float32 PCM - the exact shape
    app/voice/recorder.py already produces for the local mic path, so
    stt.transcribe() needs no changes. Raises on unreadable/empty audio."""
    import av
    from av.audio.resampler import AudioResampler

    container = av.open(io.BytesIO(raw))
    try:
        stream = container.streams.audio[0]
        resampler = AudioResampler(format="flt", layout="mono", rate=_TARGET_SAMPLE_RATE)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):  # flush
            chunks.append(resampled.to_ndarray())
    finally:
        container.close()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=1).flatten()


@app.post("/api/transcribe")
async def transcribe(file: UploadFile) -> JSONResponse:
    raw = await file.read()
    try:
        audio = _decode_browser_audio(raw)
    except Exception as exc:
        return JSONResponse({"text": "", "error": f"couldn't decode audio: {exc}"})

    try:
        text = stt.transcribe(audio)
    except Exception as exc:
        return JSONResponse({"text": "", "error": f"transcription failed: {exc}"})

    return JSONResponse({"text": text})


def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Minimal 44-byte WAV header around raw PCM16 bytes, so the browser's
    native <audio>/Audio() playback can use it directly with no separate
    decoder - OpenAI's TTS response_format="pcm" returns headerless PCM."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", len(pcm),
    )
    return header + pcm


class SpeakRequest(BaseModel):
    text: str


@app.post("/api/speak")
def speak_endpoint(req: SpeakRequest) -> Response:
    try:
        pcm = tts.synthesize(req.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"speech synthesis failed: {exc}")
    wav_bytes = _wrap_pcm_as_wav(pcm, sample_rate=tts.OPENAI_TTS_SAMPLE_RATE)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/briefing")
def briefing() -> dict:
    return {"text": _brief_text()}
