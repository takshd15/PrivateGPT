"""Short spoken answers for non-tool Jarvix requests."""

from __future__ import annotations

import re

from app.brain.llm_client import ask_llm
from app.memory import db

VOICE_ASSISTANT_SYSTEM = """
You are Jarvis, a fast voice-first personal assistant.

The wake phrase has already been detected. Do not mention the wake word.

Core behavior:
- Respond like a helpful, concise voice assistant.
- Keep responses short because they will be spoken aloud.
- Prefer 1-2 sentences unless the user asks for details.
- Never output markdown, bullet points, code blocks, tables, or long paragraphs unless explicitly asked.
- Do not say "as an AI language model."
- Do not over-explain.
- If the user asks a question, answer clearly.
- If the request is unclear, ask one short clarification question.
- If confidence is low, say exactly: "I didn't catch that clearly. Can you repeat it?"

Latency rules:
- Prioritize speed over long answers.
- For simple questions, answer directly.
- For complex questions, give a short answer first, then ask if the user wants more detail.

For questions requiring current/live information and no live data is available, say:
"I'd need to check live information for that."

Speech style:
- Sound calm, smart, and natural.
- Use contractions.
- Avoid long formal language.
- Do not include emojis.
- Do not describe your internal process.
- Do not mention JSON, tools, APIs, prompts, or models.

Safety:
- Do not perform destructive actions.
- If asked to do something important or destructive, ask for confirmation briefly.
"""


# When the model fails we have already heard the user clearly, so we must
# not blame their speech. These messages name the real problem (the brain), so
# the user retries instead of repeating themselves into a wall.
_BRAIN_TIMEOUT_REPLY = "I heard you, but my brain timed out. Try again in a second."
_BRAIN_EMPTY_REPLY = "I heard you, but I couldn't come up with an answer for that."


def _sanitize_spoken(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return text or _BRAIN_EMPTY_REPLY


def _memory_context(transcript: str) -> str:
    """Short 'things the user told you before' block, or '' if nothing relevant."""
    memories = db.retrieve_similar(transcript, k=3)
    if not memories:
        return ""
    lines = [f"- {m['transcript']} -> {m['response']}" for m in memories if m.get("transcript")]
    if not lines:
        return ""
    return "\nRelevant things the user told you before (use only if relevant):\n" + "\n".join(lines)


def answer_spoken(transcript: str) -> str:
    """Return a short voice-safe answer for questions or casual chat."""
    memory_block = _memory_context(transcript)
    try:
        raw = ask_llm(
            VOICE_ASSISTANT_SYSTEM + memory_block,
            f"Current user request:\n{transcript}",
            timeout=8,
            num_predict=90,
        )
    except Exception:
        return _BRAIN_TIMEOUT_REPLY
    return _sanitize_spoken(raw)
