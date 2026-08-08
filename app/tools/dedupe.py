"""Duplicate detection between extracted candidates and existing Calendar events.

We never block a write automatically; we surface "possible duplicate" so the user
decides. Comparison is title similarity + same date + overlapping time (if timed).
"""
from difflib import SequenceMatcher

from app.models import EventCandidate

TITLE_SIMILARITY_THRESHOLD = 0.6


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _event_date(event_dt: str) -> str:
    """Return the YYYY-MM-DD part of a Calendar start/end value.

    Calendar gives either '2026-06-21' (all-day) or '2026-06-21T14:00:00+02:00'.
    """
    if not event_dt:
        return ""
    return event_dt[:10]


def _event_time(event_dt: str) -> str:
    """Return HH:MM from a dateTime value, or '' for all-day."""
    if event_dt and "T" in event_dt:
        return event_dt[11:16]
    return ""


def _times_overlap(c: EventCandidate, ev_start: str, ev_end: str) -> bool:
    """Loose overlap check when both sides have a time; otherwise treat as match."""
    c_start = c.start_time
    e_start = _event_time(ev_start)
    if not c_start or not e_start:
        # One side is all-day -> same-day collision is enough to flag.
        return True
    c_end = c.end_time or c_start
    e_end = _event_time(ev_end) or e_start
    return c_start < e_end and e_start < c_end


def find_duplicates(candidate: EventCandidate, events: list[dict]) -> list[dict]:
    """Return existing events that look like duplicates of the candidate."""
    matches: list[dict] = []
    for ev in events:
        ev_date = _event_date(ev.get("start", ""))
        if not candidate.date or ev_date != candidate.date:
            continue
        if _title_similarity(candidate.title, ev.get("summary", "")) < TITLE_SIMILARITY_THRESHOLD:
            continue
        if not _times_overlap(candidate, ev.get("start", ""), ev.get("end", "")):
            continue
        matches.append(ev)
    return matches
