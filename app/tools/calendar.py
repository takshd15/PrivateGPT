from datetime import datetime, timedelta, timezone, date as date_cls, time as time_cls
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from app.config import TIMEZONE
from app.models import EventCandidate
from app.tools.google_auth import calendar_service


def get_upcoming_events(days: int = 2, limit: int = 20) -> list[dict]:
    service = calendar_service()

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=limit,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = []
    for event in events_result.get("items", []):
        start = event["start"].get("dateTime", event["start"].get("date"))
        end_time = event["end"].get("dateTime", event["end"].get("date"))

        events.append(
            {
                "id": event.get("id"),
                "summary": event.get("summary", "Untitled"),
                "start": start,
                "end": end_time,
                "location": event.get("location", ""),
            }
        )

    return events


def get_events_for_date(day: date_cls, limit: int = 20) -> list[dict]:
    """Fetch events occurring on one local calendar day."""
    service = calendar_service()
    try:
        tz = ZoneInfo(TIMEZONE)
    except ZoneInfoNotFoundError:
        tz = datetime.now().astimezone().tzinfo or timezone.utc
    start = datetime.combine(day, time_cls.min, tzinfo=tz)
    end = start + timedelta(days=1)
    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            maxResults=limit,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = []
    for event in result.get("items", []):
        events.append(
            {
                "id": event.get("id"),
                "summary": event.get("summary", "Untitled"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "location": event.get("location", ""),
            }
        )
    return events


def get_events_window(days_ahead: int = 120, limit: int = 250) -> list[dict]:
    """Fetch events from now through `days_ahead` days for duplicate checking."""
    return get_upcoming_events(days=days_ahead, limit=limit)


def create_calendar_event(summary: str, start_iso: str, end_iso: str, description: str = "", location: str = ""):
    service = calendar_service()

    event = {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": TIMEZONE},
        "end": {"dateTime": end_iso, "timeZone": TIMEZONE},
    }

    created = service.events().insert(calendarId="primary", body=event).execute()
    return created


def _candidate_description(candidate: EventCandidate) -> str:
    parts = []
    if candidate.reason:
        parts.append(candidate.reason)
    if candidate.meeting_link:
        parts.append(f"Link: {candidate.meeting_link}")
    parts.append("")
    parts.append("Added by Jarvix from email:")
    parts.append(f"Subject: {candidate.source_subject}")
    parts.append(f"Email id: {candidate.source_email_id}")
    return "\n".join(parts)


def create_event_from_candidate(candidate: EventCandidate) -> dict:
    """Create a Calendar event from a validated candidate.

    Timed event when a start time exists; otherwise an all-day event. The source
    email subject/id is always stored in the description for traceability.
    Caller MUST have obtained user confirmation before calling this.
    """
    service = calendar_service()

    event = {
        "summary": candidate.title,
        "description": _candidate_description(candidate),
    }
    if candidate.location:
        event["location"] = candidate.location

    if candidate.is_timed():
        start_iso = f"{candidate.date}T{candidate.start_time}:00"
        end_time = candidate.end_time or candidate.start_time
        end_iso = f"{candidate.date}T{end_time}:00"
        event["start"] = {"dateTime": start_iso, "timeZone": TIMEZONE}
        event["end"] = {"dateTime": end_iso, "timeZone": TIMEZONE}
    else:
        # All-day event. Google treats end.date as exclusive, so add one day.
        start_date = date_cls.fromisoformat(candidate.date)
        end_date = start_date + timedelta(days=1)
        event["start"] = {"date": start_date.isoformat()}
        event["end"] = {"date": end_date.isoformat()}

    return service.events().insert(calendarId="primary", body=event).execute()
