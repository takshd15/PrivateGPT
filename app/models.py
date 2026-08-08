"""Typed schemas for Jarvix v0.

The model proposes event/deadline candidates as raw JSON. We parse them
into EventCandidate so the rest of the app works with validated, typed data
instead of free-form dicts. The model never writes to Calendar directly.
"""
from typing import Optional
from pydantic import BaseModel, Field

# Allowed categories the extractor is told to use. Anything else is coerced
# to "other" in normalize_category().
CATEGORIES = {
    "hackathon",
    "meeting",
    "interview",
    "booking",
    "appointment",
    "exam",
    "deadline",
    "travel",
    "payment",
    "reminder",
    "other",
}

# Categories that represent a point-in-time commitment (default to a 1h block
# when a start time exists but no end time was given).
TIMED_CATEGORIES = {"meeting", "interview", "appointment"}


def normalize_category(value: Optional[str]) -> str:
    if not value:
        return "other"
    v = value.strip().lower()
    return v if v in CATEGORIES else "other"


class EventCandidate(BaseModel):
    """A single calendar/deadline candidate extracted from one email."""

    title: str = ""
    category: str = "other"
    date: Optional[str] = None          # YYYY-MM-DD or None
    start_time: Optional[str] = None    # HH:MM or None
    end_time: Optional[str] = None      # HH:MM or None
    all_day: bool = False
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    source_email_id: str = ""
    source_subject: str = ""
    confidence: float = 0.0
    reason: str = ""
    missing_fields: list[str] = Field(default_factory=list)

    def is_timed(self) -> bool:
        """True when this candidate should become a timed Calendar event."""
        return bool(self.date and self.start_time and not self.all_day)
