"""One-line LLM summaries for list-shaped tool output (emails, news, ...).

Each item is summarized independently: smaller prompts stay fast and one
failed item never takes the rest of the list down with it.
"""

from __future__ import annotations

from app.brain.llm_client import ask_llm

_SUMMARY_SYSTEM = """You compress one {kind} into a single short spoken sentence.

Rules:
- Output ONLY the summary sentence. No preamble, no labels, no markdown, no quotes.
- State the single main point only - the one thing the user actually needs to know.
- Do not repeat the sender/title verbatim if it adds nothing.
- Keep it under 20 words.
- If the content is spam/marketing with no real point, say so plainly in a few words.
"""


def summarize_item(text: str, kind: str = "email", fallback: str = "") -> str:
    """Summarize one item's text into a single sentence. Falls back on failure."""
    text = (text or "").strip()
    if not text:
        return fallback
    try:
        raw = ask_llm(
            _SUMMARY_SYSTEM.format(kind=kind),
            f"Content:\n{text[:2000]}",
            timeout=8,
            num_predict=60,
        )
        summary = " ".join(raw.strip().split())
        return summary or fallback
    except Exception:
        return fallback


def summarize_items(items: list[str], kind: str = "email", fallbacks: list[str] | None = None) -> list[str]:
    """Summarize each item in ``items`` independently. Same length as input."""
    fallbacks = fallbacks or ["" for _ in items]
    return [
        summarize_item(text, kind=kind, fallback=fallback)
        for text, fallback in zip(items, fallbacks)
    ]
