"""Command parsing seam for Jarvix v2.

``intent_router.parse`` uses fast rules first, then an LLM fallback for
unfamiliar wording. This module is the stable entry point the rest of the app
can import without caring how parsing is implemented.
"""

from __future__ import annotations

from app.brain.intent_router import Intent, parse as _parse


def parse_command(text: str) -> Intent:
    """Turn raw (typed or transcribed) text into a structured Intent."""
    return _parse(text)
