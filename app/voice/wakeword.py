"""Voice wake-word detector for Jarvix v2: listen for "hey jarvis".

Gates on the offline STT (faster-whisper). It is energy-gated by the
recorder: it only transcribes when there's actually speech, so the CPU is idle
when the room is quiet. Matching is deliberately loose to absorb the small STT
model's mishears of the name (jarvis / jarvix / jervis ...). When the same
utterance also carries a command ("hey jarvis, play some music"), that clip is
re-transcribed via AssemblyAI for accuracy before the command is extracted.
"""

from __future__ import annotations

import re

from app.config import MIC_SILENCE_THRESHOLD, WAKE_DEBUG, WAKE_WORD
from app.voice.recorder import record
from app.voice.stt import transcribe_wake_word


# Only close, name-like mishears are accepted by name. Tokens that collide with
# ordinary words or other names ("doris", "harvest", "chavez", "joris") were
# removed: they woke Jarvix on nonsense. Genuine STT slips ("garvis", "zarvis",
# "charvis", "jarviss") are still caught by the fuzzy edit-distance fallback in
# ``_is_wake_token``, so they don't need to be listed here.
_KNOWN_VARIANTS = {
    "jarvis",
    "jarvix",
    "jervis",
    "jorvis",
    "jorvix",
    "jarves",
}

# Real names/words the loose matcher would otherwise wake on. "joris" in
# particular slips through the fuzzy fallback (edit distance 2 from "jarvis"),
# so collisions are rejected explicitly regardless of distance.
_NON_WAKE_WORDS = {
    "joris",
    "doris",
    "boris",
    "morris",
    "chavez",
    "harvest",
    "travis",
}


def _edit_distance(left: str, right: str) -> int:
    """Small dependency-free Levenshtein distance for one spoken word."""
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def _is_wake_token(token: str) -> bool:
    token = token.lower()
    if token in _NON_WAKE_WORDS:
        return False
    configured = re.sub(r"[^a-z]", "", WAKE_WORD.lower())
    if token == configured or token in _KNOWN_VARIANTS:
        return True
    # Fuzzy matching is deliberately narrow. Broad fragments such as "jar" and
    # "travis" caused music and ordinary speech to look like wake phrases.
    if not 5 <= len(token) <= 7 or token[0] not in "jgzcs":
        return False
    return min(_edit_distance(token, "jarvis"), _edit_distance(token, "jarvix")) <= 2


def _wake_span(text: str) -> tuple[int, int] | None:
    """Return a wake-token span only near the beginning of the utterance."""
    words = list(re.finditer(r"[A-Za-z]+", text))
    for index, match in enumerate(words[:3]):
        if match.group(0).lower() == "hey":
            continue
        if _is_wake_token(match.group(0)):
            return match.span()
    return None


def _matches(text: str) -> bool:
    return _wake_span(text) is not None


def command_after_wake_word(text: str | None) -> str:
    """Extract a same-utterance command after any accepted wake variant."""
    if not text:
        return ""
    span = _wake_span(text)
    if span is None:
        return ""
    command = text[span[1]:].strip(" ,.?!:-")
    if command.lower() in {"", "can you", "can you please", "please"}:
        return ""
    # If the "command" is just repeated wake-name garbage ("Jarvis Jorvis",
    # "Garvis Garvis"), there is no real instruction. Ignore it.
    tokens = re.findall(r"[A-Za-z]+", command)
    if tokens and all(_is_wake_token(tok) for tok in tokens):
        return ""
    return command


def wait_for_wake_word(verbose: bool = False) -> str:
    """Block until the wake word is heard. Returns the matching transcript.

    Raises ``MicUnavailable`` (from ``record``) if there's no usable mic.
    """
    from app.voice.stt import transcribe, warm

    while True:
        warm()  # pre-open the AssemblyAI connection while this window records
        audio = record(
            max_seconds=5.0,
            silence_threshold=MIC_SILENCE_THRESHOLD,
            trailing_silence=1.0,
            start_timeout=3.0,
        )
        if audio.size == 0:
            continue  # no speech this window; keep listening
        text = transcribe_wake_word(audio)
        if not text:
            continue
        if not _matches(text):
            if verbose and WAKE_DEBUG:
                print(f"  ignored: {text!r}")
            continue

        final_text = text
        if command_after_wake_word(text):
            # Same-breath command ("hey jarvis, play some music"): the small
            # local model is fast but accent-sensitive. Re-transcribe this
            # exact clip with AssemblyAI for the trailing command - only pay
            # for the extra call when there's actually a command to get right.
            try:
                better_text = transcribe(audio)
            except Exception:
                better_text = ""
            if better_text and _matches(better_text):
                final_text = better_text

        if verbose:
            print(f"  heard: {final_text!r}")
        return final_text
