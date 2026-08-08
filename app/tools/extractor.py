"""Extract structured event/deadline candidates from emails using OpenAI.

Email content (snippet/body) is sent to OpenAI to extract candidates.

Safety: the model only *proposes* candidates as strict JSON. It never writes to
Calendar. App code (here + main.py) validates and the user confirms before any
write happens.
"""
import json
from datetime import datetime, timedelta

from app.brain.llm_client import ask_llm
from app.memory.cache import content_hash, get_cached, set_cached
from app.models import EventCandidate, TIMED_CATEGORIES, normalize_category

MIN_CONFIDENCE = 0.70

EXTRACTION_SYSTEM = """You are a strict information-extraction engine for a personal assistant.

From a SINGLE email, extract real-world commitments worth putting on a calendar:
hackathons, meetings, interviews, bookings, appointments, exams, application or
submission deadlines, travel, payment due dates, reminders.

Hard rules:
- Do NOT invent dates. If a date is not explicitly stated or unambiguously implied
  by the email, set "date" to null.
- Do NOT invent meetings, times, locations, or links.
- If a field is unclear or absent, leave it null and add its name to "missing_fields".
- Never claim anything was scheduled or added.
- "confidence" is your certainty (0.0-1.0) that this is a real, actionable commitment.
- Output STRICT JSON only. No prose, no markdown.

Return a JSON object of this exact shape:
{"candidates": [
  {
    "title": "string",
    "category": "hackathon|meeting|interview|booking|appointment|exam|deadline|travel|payment|reminder|other",
    "date": "YYYY-MM-DD or null",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "all_day": true,
    "location": "string or null",
    "meeting_link": "string or null",
    "confidence": 0.0,
    "reason": "string",
    "missing_fields": ["string"]
  }
]}

If the email contains no real commitment, return {"candidates": []}.
"""


def _build_user_prompt(email: dict, today: str) -> str:
    body = (email.get("body") or "")[:1500]
    return (
        f"Today's date is {today}. Resolve relative dates (e.g. 'next Monday') "
        f"against it, but only when the email clearly implies them.\n\n"
        f"EMAIL\n"
        f"From: {email.get('from', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Date: {email.get('date', '')}\n"
        f"Snippet: {email.get('snippet', '')}\n"
        f"Body:\n{body}\n"
    )


def _parse_candidates(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("candidates", [])
        return items if isinstance(items, list) else []
    return []


def extract_from_email(email: dict, today: str, use_cache: bool = True) -> list[EventCandidate]:
    message_id = email.get("id", "")
    chash = content_hash((email.get("snippet") or "") + (email.get("body") or ""))

    # Cache hit: same message, unchanged content -> skip the model entirely.
    if use_cache:
        cached = get_cached(message_id, chash)
        if cached is not None:
            return [EventCandidate(**d) for d in cached]

    try:
        raw = ask_llm(
            EXTRACTION_SYSTEM,
            _build_user_prompt(email, today),
            json_mode=True,
            timeout=5,
            num_predict=256,
        )
    except Exception:
        return []

    candidates: list[EventCandidate] = []
    for item in _parse_candidates(raw):
        if not isinstance(item, dict):
            continue
        # Stamp the source from trusted email data, not the model's guess.
        item["source_email_id"] = message_id
        item["source_subject"] = email.get("subject", "")
        item["category"] = normalize_category(item.get("category"))
        try:
            candidates.append(EventCandidate(**item))
        except Exception:
            # Skip malformed items rather than crash the whole scan.
            continue

    if use_cache:
        set_cached(
            message_id,
            email.get("subject", ""),
            email.get("date", ""),
            chash,
            [c.model_dump() for c in candidates],
        )
    return candidates


def extract_candidates(emails: list[dict], use_cache: bool = True) -> list[EventCandidate]:
    """Run extraction per-email (small prompts keep the 3B model reliable)."""
    today = datetime.now().strftime("%Y-%m-%d")
    out: list[EventCandidate] = []
    for email in emails:
        out.extend(extract_from_email(email, today, use_cache=use_cache))
    return out


def _plus_one_hour(hhmm: str) -> str:
    try:
        t = datetime.strptime(hhmm, "%H:%M")
        return (t + timedelta(hours=1)).strftime("%H:%M")
    except ValueError:
        return hhmm


def validate_candidates(
    candidates: list[EventCandidate],
) -> tuple[list[EventCandidate], list[tuple[EventCandidate, str]]]:
    """Apply v0 validation rules.

    Returns (accepted, rejected) where rejected items carry a reason string.
    """
    accepted: list[EventCandidate] = []
    rejected: list[tuple[EventCandidate, str]] = []

    for c in candidates:
        if not c.title.strip():
            rejected.append((c, "no title"))
            continue
        if not c.date:
            rejected.append((c, "no date"))
            continue
        if c.confidence < MIN_CONFIDENCE:
            rejected.append((c, f"low confidence ({c.confidence:.2f} < {MIN_CONFIDENCE})"))
            continue
        # Vague "other" candidates without strong certainty are not real commitments.
        if c.category == "other" and c.confidence < 0.80:
            rejected.append((c, "vague / unclear commitment"))
            continue

        # Defaults.
        if not c.start_time:
            # No time -> treat as an all-day deadline/event.
            c.all_day = True
            c.end_time = None
        elif not c.end_time and c.category in TIMED_CATEGORIES:
            # Timed meeting/interview with start but no end -> default 1 hour.
            c.end_time = _plus_one_hour(c.start_time)

        accepted.append(c)

    return accepted, rejected
