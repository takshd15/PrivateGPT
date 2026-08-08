"""Wake loop for Jarvix v2: wake trigger -> listen -> transcribe -> act -> speak.

Decoupled from the rest of the app: the caller injects ``handle_text`` (which
parses + executes an intent and returns a response string) and ``speak``. This
keeps the voice plumbing here and the routing logic in main.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from app.voice.wakeword import command_after_wake_word


_MUSIC_WORDS = (
    "music",
    "song",
    "spotify",
    "play",
    "pause",
    "resume",
    "skip",
    "next",
    "previous",
    "volume",
    "louder",
    "quieter",
)

# After a command, Jarvix keeps listening without repeating the wake word for
# this long since the *last* thing it heard (a rolling window, not one-shot).
_AWAKE_IDLE_SECONDS = 45.0
_AWAKE_ATTEMPT_SECONDS = 6.0  # how long each listening attempt waits for speech to start

_SLEEP_PATTERN = re.compile(r"\bsleep\b", re.IGNORECASE)


def _looks_like_music_command(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in _MUSIC_WORDS)


def _looks_like_sleep_command(text: str) -> bool:
    return bool(_SLEEP_PATTERN.search(text))


def _command_from_wake_text(text: str | None) -> str:
    return command_after_wake_word(text)


def wake_loop(
    handle_text: Callable[[str], str],
    speak: Callable[[str], None],
    wait_for_wake: Callable[[], str | None] | None = None,
    announce: str = "Yes?",
    once: bool = False,
) -> None:
    """Run the wake loop. ``wait_for_wake`` blocks until the user triggers Jarvix
    (defaults to the double-clap detector). Set ``once=True`` for a single command.

    After a command, Jarvix stays awake and keeps listening without the wake
    word for ``_AWAKE_IDLE_SECONDS`` since the last thing it heard. Say "sleep"
    to end that window immediately instead of waiting it out.
    """
    # Imported lazily so non-voice commands don't pay the import cost.
    from app.voice.recorder import record, MicUnavailable
    from app.voice.stt import transcribe, warm
    from app.tools import music as music_tool
    from app.config import VOICE_RECORD_SECONDS

    if wait_for_wake is None:
        from app.voice.clap_detector import wait_for_double_clap
        wait_for_wake = lambda: wait_for_double_clap(verbose=True)

    def _capture_command(start_timeout: float) -> str | None:
        """One record+transcribe attempt. None = no speech started (silence);
        "" = audio captured but transcription failed; else the heard text."""
        warm()
        audio = record(
            max_seconds=float(VOICE_RECORD_SECONDS),
            trailing_silence=1.1,
            start_timeout=start_timeout,
        )
        if audio.size == 0:
            return None
        return transcribe(audio)

    def _run_command(text: str) -> None:
        print(f"Heard: {text}")
        response = handle_text(text)
        if response:
            print(f"Jarvix: {response}")
            speak(response)

    while True:
        print("Waiting for wake trigger...")
        try:
            wake_text = wait_for_wake()
        except MicUnavailable as exc:
            print(f"Microphone unavailable: {exc}")
            speak("My microphone isn't available, so I can't listen right now.")
            return

        wake_command = _command_from_wake_text(wake_text)
        if announce:
            speak(announce)

        if wake_command:
            text = wake_command
        else:
            paused_music = music_tool.pause_if_playing()
            print("Listening...")
            try:
                text = _capture_command(start_timeout=5.0)
            except MicUnavailable as exc:
                print(f"Microphone unavailable: {exc}")
                speak("My microphone isn't available, so I can't listen right now.")
                return
            if text is None:
                print("Heard no command audio.")
                speak("I didn't hear a command.")
                if paused_music:
                    music_tool.resume_playback()
                if once:
                    return
                continue
            if not text:
                print("Command audio was captured, but transcription was empty.")
                speak("I couldn't make that out.")
                if paused_music:
                    music_tool.resume_playback()
                if once:
                    return
                continue
            if paused_music and not _looks_like_music_command(text):
                music_tool.resume_playback()

        _run_command(text)
        if once:
            return

        # Stay awake: keep listening without the wake word for a rolling
        # window since the last thing heard, until "sleep" or the idle timeout.
        # Music is paused once for the whole window rather than per-attempt,
        # so it doesn't stutter on/off during silent stretches.
        paused_music = music_tool.pause_if_playing()
        try:
            last_activity = time.monotonic()
            while time.monotonic() - last_activity < _AWAKE_IDLE_SECONDS:
                try:
                    heard = _capture_command(start_timeout=_AWAKE_ATTEMPT_SECONDS)
                except MicUnavailable as exc:
                    print(f"Microphone unavailable: {exc}")
                    speak("My microphone isn't available, so I can't listen right now.")
                    return
                if not heard:
                    continue  # silence or empty transcription - keep the window open
                if _looks_like_sleep_command(heard):
                    print("Heard: sleep")
                    speak("Going to sleep.")
                    break
                if paused_music and not _looks_like_music_command(heard):
                    music_tool.resume_playback()
                    paused_music = False
                _run_command(heard)
                last_activity = time.monotonic()
        finally:
            if paused_music:
                music_tool.resume_playback()
