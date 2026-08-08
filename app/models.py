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


class TaskCandidate(BaseModel):
    """A single task extracted from a REMEMBER/REMIND utterance."""

    title: str = ""
    details: str = ""
    due_date_phrase: Optional[str] = None   # raw spoken phrase; app/tools/live_info.py resolves it
    priority: int = 0                        # 0-3
    entity_name: Optional[str] = None
    domain: Optional[str] = None
    confidence: float = 0.0


class ProjectCandidate(BaseModel):
    """A new project extracted from a NEW_PROJECT description."""

    name: str = ""
    description: str = ""
    status: str = "active"
    confidence: float = 0.0


PROGRESS_EVENT_TYPES = {"update", "blocker", "milestone", "decision"}


class ProgressEventCandidate(BaseModel):
    """A single progress-log entry extracted from a LOG_PROGRESS utterance."""

    entity_name: str = ""
    domain: Optional[str] = None
    event_text: str = ""
    event_type: str = "update"               # one of PROGRESS_EVENT_TYPES
    event_date_phrase: Optional[str] = None  # raw spoken phrase; live_info.py resolves it
    confidence: float = 0.0


class OpportunityCandidate(BaseModel):
    """A single opportunity extracted from web search results (FIND_OPPORTUNITIES)."""

    name: str = ""
    type: str = ""                       # job/grant/competition/scholarship/hackathon/etc.
    url: str = ""
    deadline: Optional[str] = None       # free text - matches the real opportunities.deadline column type
    fit_score: float = 0.0               # 0.0-1.0
    reason: str = ""
    next_action: str = ""


class EmailSignupCandidate(BaseModel):
    """An event/webinar/course signup detected in one email. Null fields when absent."""

    name: str = ""
    event_type: str = ""                 # webinar/course/meetup/event/etc.
    date_phrase: Optional[str] = None
    location: Optional[str] = None
    confirmation_details: str = ""


class EmailApplicationCandidate(BaseModel):
    """A job/program application signal detected in one email. Status stays
    free text to match the varied vocabulary already in the applications
    table (e.g. 'aspirational', 'needs_context') rather than a fixed enum."""

    company_name: str = ""
    program: str = ""
    status: str = ""
    deadline_phrase: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None


class EmailSignalCandidate(BaseModel):
    """Combined single-LLM-call extraction over one email: both a possible
    signup and a possible application signal, either or both null when the
    email doesn't indicate that kind of thing (most emails should be null -
    precision over recall)."""

    signup: Optional[EmailSignupCandidate] = None
    application: Optional[EmailApplicationCandidate] = None


class MeetingFollowupCandidate(BaseModel):
    """A concise summary of a spoken meeting-discussion recap, for
    meetings.notes and a linked memories row."""

    summary: str = ""
    confidence: float = 0.0


class MemoryDetectionCandidate(BaseModel):
    """A durable fact/preference/plan passively detected in ordinary
    conversation. Empty content means nothing worth remembering was said."""

    content: str = ""
    domain: Optional[str] = None
    importance: float = 0.0   # 0.0-1.0; caller only writes when this clears a threshold


class GoalDetectionCandidate(BaseModel):
    """A goal/objective passively detected in ordinary conversation, separate
    from a plain fact. Empty title means no goal was stated."""

    title: str = ""
    description: str = ""
    priority: int = 0   # 0 default, higher only if urgency is explicit


class ConvoDetectionCandidate(BaseModel):
    """Combined single-LLM-call passive detection over one conversational
    exchange: both a possible durable memory and a possible goal, either or
    both empty when nothing worth capturing was said (most exchanges should
    be empty - this must not fire on every message)."""

    memory: MemoryDetectionCandidate = Field(default_factory=MemoryDetectionCandidate)
    goal: GoalDetectionCandidate = Field(default_factory=GoalDetectionCandidate)
