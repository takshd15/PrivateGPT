import html
import json
import re
import threading

import typer
from rich.console import Console
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.brain.llm_client import ask_llm, warmup as _warmup
from app.tools.gmail import get_recent_emails
from app.tools.calendar import (
    get_upcoming_events,
    get_events_window,
    get_events_for_date,
    create_event_from_candidate,
    create_calendar_event,
)
from app.tools.extractor import extract_candidates, validate_candidates
from app.tools.dedupe import find_duplicates
from app.tools.prefilter import prefilter, is_worthy
from app.safety.permissions import needs_confirmation, confirm
from app.models import EventCandidate
from app.voice.tts import speak
from app.tools.desktop import (
    open_app as desktop_open_app,
    open_folder as desktop_open_folder,
    open_workspace as desktop_open_workspace,
    DesktopError,
)
from app.tools import music as music_tool
from app.tools import email_actions
from app.tools import live_info
from app.tools import summarize
from app.tools import news as news_tool
from app.memory import db
from app.memory.db import progress_events as db_progress_events
from app.memory.db import projects as db_projects
from app.memory.db import tasks as db_tasks
from app.memory.db import opportunities as db_opportunities
from app.memory.db import goals as db_goals
from app.memory.db import memories as db_memories
from app.memory.db import meetings as db_meetings
from app.memory.db import applications as db_applications
from app.memory.db import events as db_events
from app.brain import intent_router, structuring
from app.brain.voice_assistant import answer_spoken
from app.tools import websearch
from app.runtime import scheduler
from app.config import (
    USER_DISPLAY_NAME,
    JARVIX_GREETING,
    AUTO_OPEN_APP_ON_WAKE,
    AUTO_OPEN_FOLDER_ON_WAKE,
    AUTO_START_MUSIC_ON_WAKE,
    AUTO_MUSIC_QUERY_ON_WAKE,
    AUTO_MUSIC_URI_ON_WAKE,
    TIMEZONE,
    USER_ID,
    ROOT,
)

app = typer.Typer(help="Jarvix v0 - local AI assistant for Gmail + Calendar.")
console = Console()


# --------------------------------------------------------------------------- #
# Rendering / confirmation helpers
# --------------------------------------------------------------------------- #
def _format_candidate(index: int, c: EventCandidate, dupes: list[dict]) -> str:
    when = c.date or "?"
    if c.start_time:
        when += f" {c.start_time}"
        if c.end_time:
            when += f"-{c.end_time}"
    elif c.all_day:
        when += " (all day)"

    lines = [
        f"[bold]{index}.[/bold] {c.title}  [dim]({c.category}, conf {c.confidence:.2f})[/dim]",
        f"    When: {when}",
    ]
    if c.location:
        lines.append(f"    Where: {c.location}")
    if c.meeting_link:
        lines.append(f"    Link: {c.meeting_link}")
    lines.append(f"    Why: {c.reason}")
    lines.append(f"    Source: {c.source_subject}")
    if c.missing_fields:
        lines.append(f"    [yellow]Missing: {', '.join(c.missing_fields)}[/yellow]")
    if dupes:
        names = ", ".join(d.get("summary", "?") for d in dupes)
        lines.append(f"    [magenta]Possible duplicate of: {names}[/magenta]")
    return "\n".join(lines)


def _select_candidates(candidates: list[EventCandidate]) -> list[EventCandidate]:
    """Ask the user which candidates to add. Returns the approved subset."""
    answer = input(
        "\nApprove which? Type 'all', 'none', or numbers (e.g. 1,3): "
    ).strip().lower()

    if answer in ("", "none", "no", "n"):
        return []
    if answer in ("all", "a", "yes", "y"):
        return list(candidates)

    chosen: list[EventCandidate] = []
    for token in answer.replace(" ", "").split(","):
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(candidates):
                chosen.append(candidates[idx])
    return chosen


# --------------------------------------------------------------------------- #
# Intent dispatch (used by `route` and the wake loop)
# --------------------------------------------------------------------------- #
def _looks_like_clear_question(text: str) -> bool:
    """True when an unrouted sentence is clearly a real question, not a mishear.

    Used so a clean, understandable sentence the router couldn't classify is
    answered instead of bouncing back a "I didn't catch that" prompt.
    """
    t = (text or "").lower().strip()
    if len(t.split()) < 3:
        return False
    starters = (
        "best ", "top ", "should ", "can ", "could ", "would ",
        "is ", "are ", "do ", "does ", "did ",
        "what ", "why ", "who ", "when ", "where ", "how ",
        "explain ", "tell me ", "give me ",
    )
    return t.startswith(starters) or "?" in t


_last_background_threads: list[threading.Thread] = []


def _detect_memory_and_goal(transcript: str, response: str) -> None:
    """Passive, silent detection of a durable memory and/or goal from one
    conversational exchange. Never speaks, never confirms - matches the
    spec's explicit 'passive capture is silent by design.' Wrapped broadly so
    a detection failure never surfaces anywhere but the console.
    """
    if not USER_ID:
        return
    try:
        detection = structuring.detect_memory_and_goal(transcript, response)
        if detection.memory.content.strip() and detection.memory.importance > 0.4:
            db_memories.insert_memory(
                USER_ID,
                content=detection.memory.content,
                domain=detection.memory.domain,
                importance=detection.memory.importance,
                source="auto-convo",
            )
        if detection.goal.title.strip():
            existing = db_goals.find_similar_goal(USER_ID, detection.goal.title)
            if not existing:
                db_goals.insert_goal(
                    USER_ID,
                    title=detection.goal.title,
                    description=detection.goal.description,
                    priority=detection.goal.priority,
                    source="auto-goal",
                )
    except Exception as exc:
        console.print(f"[yellow]Passive detection skipped: {exc}[/yellow]")


def run_intent(intent: intent_router.Intent) -> str:
    """Execute a parsed intent, log it to RAG memory, and return the response.

    Logging happens on a background thread so a slow/unavailable DB or
    embedding call never delays the spoken reply (same pattern as the
    welcome-routine music thread below). The wake loop stays alive long
    enough for this to finish naturally; short-lived CLI commands (route)
    join it briefly before the process exits so logging isn't silently lost.

    Passive memory/goal detection runs on a SEPARATE background thread, only
    for QUESTION/CONVERSATION intents - kept decoupled from RAG interaction
    logging (which fires for every intent) since they're different features
    over different tables.
    """
    global _last_background_threads
    response = _dispatch_intent(intent)
    _last_background_threads = []
    if (intent.raw or "").strip():
        thread = threading.Thread(
            target=db.log_interaction,
            args=(intent.raw, intent.name, response),
            name="jarvix-memory-log",
            daemon=True,
        )
        thread.start()
        _last_background_threads.append(thread)
        if intent.name in (intent_router.QUESTION, intent_router.CONVERSATION):
            detect_thread = threading.Thread(
                target=_detect_memory_and_goal,
                args=(intent.raw, response),
                name="jarvix-passive-detect",
                daemon=True,
            )
            detect_thread.start()
            _last_background_threads.append(detect_thread)
    return response


def _dispatch_intent(intent: intent_router.Intent) -> str:
    """Execute a parsed intent and return a short response to speak/print.

    Safe device/music intents run directly. Higher-level intents are handled
    here. Calendar writes (scan_mail) are intentionally NOT auto-run by voice -
    they require the terminal confirmation gate.
    """
    simple = intent_router.execute(intent)
    if simple is not None:
        return simple

    if intent.name == intent_router.BRIEF:
        return _brief_text()
    if intent.name == intent_router.TODAY:
        return _today_text()
    if intent.name == intent_router.CALENDAR_DATE:
        return _calendar_date_text(intent.arg)
    if intent.name == intent_router.ADD_EVENT:
        return _add_event_text(intent.values or {})
    if intent.name == intent_router.REMEMBER:
        return _remember_text(intent.values or {})
    if intent.name == intent_router.NEW_PROJECT:
        return _new_project_text(intent.values or {})
    if intent.name == intent_router.LOG_PROGRESS:
        return _log_progress_text(intent.values or {})
    if intent.name == intent_router.FIND_OPPORTUNITIES:
        return _find_opportunities_text(intent.arg)
    if intent.name == intent_router.MEETING_FOLLOWUP:
        return _meeting_followup_text(intent.values or {})
    if intent.name == intent_router.BROWSER_RESULT:
        # Both a fresh browser_task's finished/failed/needs-human result and
        # a resumed one (after browser_confirm's yes/no) arrive here as a
        # ready-to-speak string - see VoiceDialogue._continue's
        # "browser_confirm" branch and _resume_browser.
        return (intent.values or {}).get("answer", "")
    if intent.name == intent_router.TIME:
        return live_info.spoken_time()
    if intent.name == intent_router.WEATHER:
        return live_info.weather(intent.arg)
    if intent.name == intent_router.NEWS:
        return _news_text()
    if intent.name == intent_router.READ_EMAILS:
        return _read_emails_text(intent.raw)
    if intent.name == intent_router.SCAN_MAIL:
        # Reads aloud + asks for typed confirmation; never auto-writes the calendar.
        _run_scan(limit=4, days=3, use_prefilter=True, speak_summary=True)
        return ""  # all speaking already happened inside the scan
    if intent.name == intent_router.DRAFT_EMAIL:
        return _voice_email(intent.recipient, intent.message, send=False)
    if intent.name == intent_router.SEND_EMAIL:
        return _voice_email(intent.recipient, intent.message, send=True)
    if intent.name == intent_router.QUESTION:
        # The orchestrator (app/brain/orchestrator.py) already produced a
        # tool-grounded answer when it resolved this intent via the LLM
        # fallback path - reuse it instead of asking answer_spoken() to
        # re-derive an answer with no tools and no memory of what was found.
        answer = (intent.values or {}).get("answer")
        return answer if answer else answer_spoken(intent.raw)
    if intent.name == intent_router.CONVERSATION:
        return answer_spoken(intent.raw)
    if intent.name == intent_router.CLARIFICATION_NEEDED:
        # A context-aware question from the orchestrator beats the generic
        # fallback - it demonstrates Jarvix understood enough to ask something
        # specific (e.g. "Do you mean which of the four opportunities...?").
        question = (intent.values or {}).get("question")
        return question if question else "I didn't catch that clearly. Can you repeat it?"
    # An UNKNOWN intent that still reads as a clear question is a routing miss,
    # not a bad mic. Answer it instead of asking the user to repeat themselves.
    if intent.name == intent_router.UNKNOWN and _looks_like_clear_question(intent.raw):
        return answer_spoken(intent.raw)
    return "I didn't catch that clearly. Can you repeat it?"


def _voice_email(recipient_name: str | None, message: str | None, send: bool) -> str:
    """Draft an email (read it aloud), and optionally send it after confirmation.

    Drafting never sends. Sending requires an explicit typed 'yes' - no
    confirmation means no send. Unknown contacts are handled by asking the user
    to type the address.
    """
    if not recipient_name:
        return "I'm not sure who to email. Please try again and name the recipient."
    if not (message or "").strip():
        return "What should the email say?"

    to_email = email_actions.resolve_recipient(recipient_name)
    if not to_email:
        speak(f"I don't know {recipient_name}'s email. Please type it.")
        to_email = input(f"Email address for {recipient_name} (blank to cancel): ").strip()
        if not to_email:
            return "Okay, cancelled. No email drafted."
        try:
            email_actions.save_contact(recipient_name, to_email)
        except ValueError as exc:
            return str(exc)
        console.print(f"[dim]Saved {recipient_name} -> {to_email} to contacts.[/dim]")

    console.print("[bold cyan]Drafting...[/bold cyan]")
    draft = email_actions.build_draft(recipient_name, to_email, message or "")

    console.print(
        f"\n[bold green]Draft email[/bold green]\n"
        f"[bold]To:[/bold] {draft.to_name} <{draft.to_email}>\n"
        f"[bold]Subject:[/bold] {draft.subject}\n\n{draft.body}\n"
    )
    speak(f"Here is the draft for {recipient_name}. Subject: {draft.subject}. {draft.body}")

    if not send:
        return "I've drafted it but not sent it. Say send email when you want it out."

    # Phase 2.10 safety gate: routed through permissions.py. No "yes" = no send.
    assert needs_confirmation("send_email")
    speak("Should I send this?")
    if not confirm(f"Send email to {draft.to_email}\nSubject: {draft.subject}"):
        return "Okay, I did not send it."

    try:
        email_actions.send_email(draft)
    except Exception as exc:
        console.print(f"[red]Send failed: {exc}[/red]")
        return f"Sending failed: {exc}"

    return f"Sent to {recipient_name}."


def _humanize_actions(actions: list[str]) -> str:
    """['opening cursor', 'starting music'] -> 'Opening cursor and starting music'."""
    if len(actions) == 1:
        joined = actions[0]
    elif len(actions) == 2:
        joined = f"{actions[0]} and {actions[1]}"
    else:
        joined = ", ".join(actions[:-1]) + f", and {actions[-1]}"
    return joined[0].upper() + joined[1:]


def _parse_calendar_start(value: str | None) -> datetime | None:
    if not value or "T" not in value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    try:
        return parsed.astimezone(ZoneInfo(TIMEZONE))
    except ZoneInfoNotFoundError:
        return parsed.astimezone()


def _local_now() -> datetime:
    try:
        return datetime.now(ZoneInfo(TIMEZONE))
    except ZoneInfoNotFoundError:
        return datetime.now().astimezone()


def _event_local_date(event: dict):
    start = event.get("start") or ""
    if "T" in start:
        local = _parse_calendar_start(start)
        return local.date() if local else None
    try:
        return datetime.fromisoformat(start).date()
    except ValueError:
        return None


def _spoken_event_time(event: dict) -> str:
    start = event.get("start") or ""
    local = _parse_calendar_start(start)
    if not local:
        return "all day"
    return local.strftime("%I:%M %p").lstrip("0")


def _todays_events(limit: int = 12) -> list[dict]:
    today = _local_now().date()
    events = get_upcoming_events(days=2, limit=limit)
    _sync_meetings(events)
    return [event for event in events if _event_local_date(event) == today]


def _sync_meetings(events: list[dict]) -> None:
    """Passively upsert timed calendar events into meetings. No confirmation
    needed - this is a read-through sync, not a write the user asked for.
    All-day events are skipped (no meaningful starts_at/ends_at for the
    follow-up cadence). Wrapped so a sync failure never breaks calendar
    reading - the existing spoken-text logic must run regardless."""
    if not USER_ID:
        return
    try:
        for event in events:
            start = event.get("start") or ""
            end = event.get("end") or ""
            if "T" not in start:  # all-day event, no start/end time to sync
                continue
            gcal_id = event.get("id") or ""
            if not gcal_id:
                continue
            db_meetings.upsert_meeting(
                USER_ID,
                title=event.get("summary") or "Untitled",
                location=event.get("location") or None,
                starts_at=start,
                ends_at=end or start,
                gcal_event_id=gcal_id,
            )
    except Exception as exc:
        console.print(f"[yellow]Meeting sync skipped: {exc}[/yellow]")


def _calendar_brief_text(include_startup_actions: bool = False) -> str:
    """Fast deterministic calendar-only briefing for wake mode."""
    details = _calendar_tasks_text(include_startup_actions=include_startup_actions)
    return f"{JARVIX_GREETING}, {USER_DISPLAY_NAME}. {details}"


def _calendar_tasks_text(include_startup_actions: bool = False) -> str:
    """Fast deterministic calendar-only task summary without the greeting."""
    console.print("[bold cyan]Reading calendar...[/bold cyan]")
    events = get_upcoming_events(days=1, limit=8)
    _sync_meetings(events)

    if events:
        items = []
        for event in events[:4]:
            summary = event.get("summary", "Untitled")
            items.append(f"{_spoken_event_time(event)} {summary}")
        if len(items) == 1:
            task_text = items[0]
        elif len(items) == 2:
            task_text = f"{items[0]} and {items[1]}"
        else:
            task_text = ", ".join(items[:-1]) + f", and {items[-1]}"
        text = f"Your tasks for the day are {task_text}."
    else:
        text = "Your calendar is clear for the rest of today."

    if include_startup_actions:
        track = AUTO_MUSIC_QUERY_ON_WAKE.strip() or AUTO_MUSIC_URI_ON_WAKE.strip()
        music_text = f' and play "{track}"' if track else " and start Spotify"
        text += f" I'll open VS Code{music_text} for you."

    return text


def _calendar_date_text(date_phrase: str | None) -> str:
    day = live_info.resolve_date_phrase(date_phrase or "")
    if day is None:
        return "Which date should I check?"

    console.print(f"[bold cyan]Reading calendar for {day.isoformat()}...[/bold cyan]")
    events = get_events_for_date(day, limit=12)
    _sync_meetings(events)
    label = day.strftime("%A, %B %d").replace(" 0", " ")
    if not events:
        return f"Your calendar is clear on {label}."

    items = [
        f"{_spoken_event_time(event)} {event.get('summary', 'Untitled')}"
        for event in events[:5]
    ]
    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} and {items[1]}"
    else:
        joined = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"On {label}, you have {joined}."


def _add_event_text(values: dict) -> str:
    """Create a calendar event from the details collected by VoiceDialogue.

    The spoken 'yes' that reaches this function (event_confirm state in
    dialogue.py) IS the confirmation gate for this dangerous action - there is
    no second, blocking terminal prompt in the voice path.
    """
    title = (values.get("title") or "").strip()
    date_str = values.get("date") or ""
    start_time = values.get("start_time")
    if not title or not date_str:
        return "I didn't get enough details to add that event."

    try:
        if start_time:
            end_time = (datetime.strptime(start_time, "%H:%M") + timedelta(hours=1)).strftime("%H:%M")
            start_iso = f"{date_str}T{start_time}:00"
            end_iso = f"{date_str}T{end_time}:00"
            create_calendar_event(title, start_iso, end_iso)
        else:
            candidate = EventCandidate(title=title, date=date_str, all_day=True, confidence=1.0)
            create_event_from_candidate(candidate)
        console.print(f"[green]Added event:[/green] {title} on {date_str}" + (f" at {start_time}" if start_time else ""))
        return f"Done. I added '{title}' to your calendar."
    except Exception as exc:
        console.print(f"[red]Failed to add event:[/red] {exc}")
        return f"Sorry, I couldn't add that event: {exc}"


def known_project_names() -> list[str]:
    """Project names for the current user, used by LOG_PROGRESS entity matching.

    Passed into VoiceDialogue as a callback (see `wake`/`route`) so app/brain/
    never imports app.memory.db directly - main.py owns that boundary.
    """
    if not USER_ID:
        return []
    return [p["name"] for p in db_projects.list_projects(USER_ID)]


def _remember_text(values: dict) -> str:
    """Insert a task from the details collected by VoiceDialogue.

    The spoken/typed 'yes' that reaches this function (remember_confirm state
    in dialogue.py) IS the confirmation gate - there is no second, blocking
    terminal prompt.
    """
    title = (values.get("title") or "").strip()
    if not title:
        return "I didn't get enough details to remember that."
    if not USER_ID:
        return "I can't save that - USER_ID isn't configured."

    try:
        task_id = db_tasks.insert_task(
            USER_ID,
            title=title,
            details=values.get("details") or "",
            due_date=values.get("due_date"),
            priority=int(values.get("priority") or 0),
            entity_name=values.get("entity_name"),
            domain=values.get("domain"),
        )
        if task_id is None:
            return "Sorry, I couldn't save that."
        console.print(f"[green]Remembered:[/green] {title}")
        return f"Got it, I'll remember: {title}."
    except Exception as exc:
        console.print(f"[red]Failed to remember task:[/red] {exc}")
        return f"Sorry, I couldn't save that: {exc}"


def _new_project_text(values: dict) -> str:
    """Insert a project from the details collected by VoiceDialogue.

    The spoken/typed 'yes' that reaches this function (project_confirm state
    in dialogue.py) IS the confirmation gate - there is no second, blocking
    terminal prompt.
    """
    name = (values.get("name") or "").strip()
    if not name:
        return "I didn't get a project name to save."
    if not USER_ID:
        return "I can't save that - USER_ID isn't configured."

    try:
        project_id = db_projects.insert_project(
            USER_ID,
            name=name,
            description=values.get("description") or "",
            status=values.get("status") or "active",
        )
        if project_id is None:
            return "Sorry, I couldn't save that project."
        console.print(f"[green]Saved project:[/green] {name}")
        return f"Saved the project: {name}."
    except Exception as exc:
        console.print(f"[red]Failed to save project:[/red] {exc}")
        return f"Sorry, I couldn't save that project: {exc}"


def _log_progress_text(values: dict) -> str:
    """Insert a progress event from the details collected by VoiceDialogue.

    The spoken/typed 'yes' that reaches this function (progress_confirm state
    in dialogue.py) IS the confirmation gate - there is no second, blocking
    terminal prompt.
    """
    entity_name = (values.get("entity_name") or "").strip()
    event_text = (values.get("event_text") or "").strip()
    if not entity_name or not event_text:
        return "I didn't get enough details to log that."
    if not USER_ID:
        return "I can't save that - USER_ID isn't configured."

    try:
        event_id = db_progress_events.insert_progress_event(
            USER_ID,
            entity_name=entity_name,
            event_text=event_text,
            event_type=values.get("event_type") or "update",
            event_date=values.get("event_date"),
        )
        if event_id is None:
            return "Sorry, I couldn't log that."
        console.print(f"[green]Logged progress:[/green] {entity_name}: {event_text}")
        return f"Logged progress on {entity_name}."
    except Exception as exc:
        console.print(f"[red]Failed to log progress:[/red] {exc}")
        return f"Sorry, I couldn't log that: {exc}"


def _meeting_followup_text(values: dict) -> str:
    """Handle a meeting-followup dialogue outcome: permanent decline, or
    capture + save the discussion recap. The spoken yes/no that reaches this
    function (dialogue.py's meeting_followup_offer/capture states) IS the
    confirmation gate - there is no second, blocking terminal prompt.
    """
    meeting_id = values.get("meeting_id")
    if not meeting_id:
        return ""

    if values.get("declined"):
        db_meetings.mark_followup_status(meeting_id, "declined")
        return "Okay, I won't ask about that one again."

    title = values.get("title") or "that meeting"
    summary = (values.get("summary") or "").strip()
    if not summary:
        return "I didn't catch enough to log that."
    if not USER_ID:
        return "I can't save that - USER_ID isn't configured."

    try:
        db_meetings.append_meeting_notes(meeting_id, summary)
        db_memories.insert_memory(
            USER_ID,
            content=summary,
            kind="progress",
            entity_name=title,
            source="meeting-followup",
        )
        db_meetings.mark_followup_status(meeting_id, "done")
        console.print(f"[green]Logged meeting follow-up:[/green] {title}: {summary}")
        return "Got it. Want me to email them, or should I remind you to follow up again later?"
    except Exception as exc:
        console.print(f"[red]Failed to log meeting follow-up:[/red] {exc}")
        return f"Sorry, I couldn't save that: {exc}"


def _profile_snippet() -> str:
    """Short goals+memories context for opportunity-search personalization.
    Empty string if the user has neither yet - structuring.py's prompt
    already falls back to general relevance scoring in that case."""
    if not USER_ID:
        return ""
    lines = []
    for g in db_goals.list_goals(USER_ID, limit=8):
        lines.append(f"Goal: {g['title']} (priority {g['priority']})")
    for m in db_memories.list_recent_memories(USER_ID, limit=8):
        lines.append(f"{m['kind']}: {m['content']}")
    return "\n".join(lines)


def _find_opportunities_text(location: str | None) -> str:
    """Search the web for opportunities, extract structured candidates, and
    insert ALL of them (status='found') - no confirmation needed, this is
    read-only relative to the user's other data (per the spec)."""
    if not USER_ID:
        return "I can't search for opportunities - USER_ID isn't configured."

    console.print("[bold cyan]Searching for opportunities...[/bold cyan]")
    profile = _profile_snippet()
    today = live_info.local_now().strftime("%Y-%m-%d")
    location_text = location.strip() if location else ""
    query_parts = ["opportunities hackathons grants scholarships accelerators jobs"]
    if location_text:
        query_parts.append(f"in {location_text}")
    if profile:
        goal_titles = [line.split(":", 1)[-1].strip() for line in profile.splitlines() if line.startswith("Goal:")]
        if goal_titles:
            query_parts.append("relevant to: " + "; ".join(goal_titles[:3]))
    query = " ".join(query_parts)

    try:
        results = websearch.search(query, num_results=8)
    except websearch.WebSearchError as exc:
        return str(exc)
    except Exception as exc:
        return f"Sorry, I couldn't reach the search service: {exc}"

    if not results:
        return "I didn't find any opportunities right now."

    console.print("[bold cyan]Extracting candidates with OpenAI...[/bold cyan]")
    candidates = structuring.structure_opportunities(results, profile, today)
    if not candidates:
        return "I didn't find any concrete opportunities worth adding right now."

    inserted = 0
    for c in candidates:
        opp_id = db_opportunities.insert_opportunity(
            USER_ID,
            name=c.name,
            type_=c.type,
            url=c.url,
            deadline=c.deadline,
            fit_score=c.fit_score,
            reason=c.reason,
            next_action=c.next_action,
            source="exa",
            raw=c.model_dump(),
        )
        if opp_id:
            inserted += 1

    best = max(candidates, key=lambda c: c.fit_score)
    console.print(f"[green]Found {len(candidates)}, saved {inserted}.[/green]")
    plural = "opportunity" if len(candidates) == 1 else "opportunities"
    return (
        f"I found {len(candidates)} {plural} and saved them, sir. "
        f"{best.name}'s the standout - {best.reason} Want the rest of the list?"
    )


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _email_limit_from_text(text: str | None, default: int = 5, max_limit: int = 10) -> int:
    if not text:
        return default

    lowered = text.lower()
    token = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    patterns = [
        rf"\b(?:last|latest|recent|newest|first)\s+{token}\b",
        rf"\b{token}\s+(?:email|emails|mail|messages)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = match.group(1)
        count = int(value) if value.isdigit() else _NUMBER_WORDS.get(value, default)
        return max(1, min(count, max_limit))

    return default


def _clean_email_text(value: str | None, max_chars: int = 260) -> str:
    text = html.unescape(value or "")
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -:\t\r\n")
    for marker in (
        " MEGATHON ",
        " View Event ",
        " Download the ",
        " Reply above this line ",
        " Click the link below ",
    ):
        if marker.lower() in text.lower():
            text = re.split(re.escape(marker), text, maxsplit=1, flags=re.I)[0].strip(" -:\t\r\n")
    if len(text) <= max_chars:
        return text

    shortened = text[:max_chars].rsplit(" ", 1)[0].strip(" ,.;:")
    return f"{shortened}..."


def _sender_name(value: str | None) -> str:
    sender = html.unescape(value or "").strip()
    if "<" in sender:
        sender = sender.split("<", 1)[0].strip().strip('"')
    return sender or "unknown sender"


def _email_fallback_summary(email: dict) -> str:
    """Deterministic one-liner used only if the LLM call for this email fails."""
    subject = _clean_email_text(email.get("subject") or "(no subject)", max_chars=55)
    body = _clean_email_text(email.get("snippet") or email.get("body") or "", max_chars=85)
    if body.lower().startswith(subject.lower()):
        body = body[len(subject):].strip(" .:-")
    return f"{subject}. {body}" if body else subject


def _email_llm_text(email: dict) -> str:
    """Pre-cleaned text handed to the LLM: strips HTML/tracking noise first,
    both to reduce token count and to shrink prompt-injection surface from
    marketing email bodies."""
    subject = _clean_email_text(email.get("subject") or "(no subject)", max_chars=200)
    body = _clean_email_text(email.get("snippet") or email.get("body") or "", max_chars=1500)
    return f"Subject: {subject}\nFrom: {_sender_name(email.get('from'))}\nBody: {body}"


def _email_summary(email: dict) -> str:
    one_liner = summarize.summarize_item(
        _email_llm_text(email), kind="email", fallback=_email_fallback_summary(email)
    )
    sender = _sender_name(email.get("from"))
    return f"{sender}: {one_liner}"


def _read_emails_text(command_text: str | None = None, days: int = 1) -> str:
    """Read a short Gmail summary only after the user asks for it."""
    limit = min(_email_limit_from_text(command_text), 5)
    console.print("[bold cyan]Reading today's Gmail...[/bold cyan]")
    emails = get_recent_emails(limit=limit, days=days)
    if not emails:
        return "I don't see any recent Gmail from today."

    console.print("[bold cyan]Summarizing with OpenAI...[/bold cyan]")
    return "\n".join(
        f"{index}. {_email_summary(email)}"
        for index, email in enumerate(emails[:limit], start=1)
    )


def _article_llm_text(article: dict) -> str:
    return f"Title: {article.get('title', '')}\nSource: {article.get('source', '')}\nDescription: {article.get('description', '')}"


def _article_fallback_summary(article: dict) -> str:
    return article.get("title") or "(no title)"


def _news_text() -> str:
    """Read top headlines, summarized to one line each via OpenAI."""
    console.print("[bold cyan]Fetching news...[/bold cyan]")
    try:
        articles = news_tool.get_top_headlines(limit=5)
    except news_tool.NewsError as exc:
        return str(exc)
    except Exception as exc:
        return f"Sorry, I couldn't reach the news service: {exc}"

    if not articles:
        return "I couldn't find any headlines right now."

    console.print("[bold cyan]Summarizing with OpenAI...[/bold cyan]")
    lines = []
    for index, article in enumerate(articles, start=1):
        one_liner = summarize.summarize_item(
            _article_llm_text(article), kind="news article", fallback=_article_fallback_summary(article)
        )
        source = article.get("source") or ""
        lines.append(f"{index}. {source + ': ' if source else ''}{one_liner}")
    return "\n".join(lines)


def _open_welcome_workspace() -> None:
    if AUTO_OPEN_APP_ON_WAKE and AUTO_OPEN_FOLDER_ON_WAKE:
        try:
            msg = desktop_open_workspace(AUTO_OPEN_APP_ON_WAKE, AUTO_OPEN_FOLDER_ON_WAKE)
            console.print(f"[dim]{msg}[/dim]")
        except DesktopError as exc:
            console.print(f"[yellow]Welcome: {exc}[/yellow]")
    elif AUTO_OPEN_APP_ON_WAKE:
        try:
            msg = desktop_open_app(AUTO_OPEN_APP_ON_WAKE)
            console.print(f"[dim]{msg}[/dim]")
        except DesktopError as exc:
            console.print(f"[yellow]Welcome: {exc}[/yellow]")


def _start_welcome_music() -> None:
    if not AUTO_START_MUSIC_ON_WAKE:
        return
    try:
        track = AUTO_MUSIC_URI_ON_WAKE.strip() or AUTO_MUSIC_QUERY_ON_WAKE.strip()
        if track:
            msg = music_tool.play(track)
        else:
            music_tool.open_spotify()
            msg = music_tool.play_pause()
        console.print(f"[dim]{msg}[/dim]")
    except Exception as exc:  # media key should never crash the routine
        console.print(f"[yellow]Welcome: music failed: {exc}[/yellow]")


def run_welcome() -> str:
    """The Jarvis-style wake routine: brief + safe auto-actions.

    Only SAFE actions auto-run (calendar read, open VS Code workspace, Spotify).
    No Gmail or calendar writes ever happen here.
    """
    _open_welcome_workspace()
    return _calendar_brief_text(include_startup_actions=True)


def _speak_welcome() -> dict | None:
    """Speak greeting first, then start Spotify while reading the calendar.

    Appends the "since last brief" rollup and a one-at-a-time meeting
    follow-up offer to the spoken calendar details, matching `brief`'s
    output. Returns the pending-followup candidate dict (or None) so the
    caller can arm dialogue.pending for the next turn - this function only
    speaks, it doesn't touch dialogue state (main.py owns that boundary).
    """
    _open_welcome_workspace()
    greeting = f"{JARVIX_GREETING}, {USER_DISPLAY_NAME}."
    console.print(f"\n[bold green]Jarvix:[/bold green]\n{greeting}")
    speak(greeting)

    music_thread = threading.Thread(target=_start_welcome_music, name="jarvix-welcome-music")
    music_thread.start()

    details = _calendar_tasks_text(include_startup_actions=True)
    details += _briefing_rollup_text()
    meeting = _pending_meeting_followup()
    if meeting:
        details += f" Also, want a follow-up on {meeting['title']} from last week?"
    console.print(details)
    speak(details)
    return meeting


@app.command()
def welcome(silent: bool = typer.Option(False, "--silent", help="Print only, do not speak.")):
    """Run the welcome routine once: briefing + safe auto-actions."""
    if silent:
        text = run_welcome()
        _start_welcome_music()
        console.print(f"\n[bold green]Jarvix:[/bold green]\n{text}")
        return
    _speak_welcome()


@app.command()
def draft_email(
    to: str = typer.Argument(..., help="Recipient contact name, e.g. alex."),
    message: str = typer.Argument(..., help="What to say, e.g. \"I'll be late\"."),
):
    """Draft an email (does NOT send). Reads it aloud."""
    console.print(_voice_email(to, message, send=False))


@app.command()
def send_email(
    to: str = typer.Argument(..., help="Recipient contact name, e.g. alex."),
    message: str = typer.Argument(..., help="What to say, e.g. \"I'll be late\"."),
):
    """Draft an email, read it aloud, and send it after a typed confirmation."""
    console.print(_voice_email(to, message, send=True))


@app.command()
def reauth():
    """Delete the saved Google token and re-consent (needed after scope changes)."""
    from app.config import GOOGLE_TOKEN_FILE
    from app.tools.google_auth import get_google_credentials

    if GOOGLE_TOKEN_FILE.exists():
        GOOGLE_TOKEN_FILE.unlink()
        console.print(f"[yellow]Deleted {GOOGLE_TOKEN_FILE}[/yellow]")
    console.print("[bold cyan]Opening browser for Google consent (incl. Gmail send)...[/bold cyan]")
    get_google_credentials()
    console.print("[green]Re-authenticated. New scopes are active.[/green]")


@app.command()
def spotify_auth():
    """Authorize Spotify Web API playback and save a refresh token."""
    music_tool.get_spotify_access_token()
    console.print("[green]Spotify is authorized for playback.[/green]")


@app.command()
def say(text: str = typer.Argument("Jarvix voice test.")):
    """Speak text aloud to test local TTS."""
    speak(text)


@app.command()
def route(text: str = typer.Argument(..., help="Command text to route, e.g. 'open cursor'.")):
    """Parse a text command and drive it through VoiceDialogue, typing y/n for any follow-ups.

    Single-shot intents (already-complete commands) resolve immediately just
    like before. Multi-turn intents (add_event, remember, new_project,
    log_progress, email drafting, weather-with-no-city, ...) now actually work
    here too, instead of only under `wake` - this is the offline/no-mic test
    path for every dialogue.py flow.
    """
    from app.brain.dialogue import VoiceDialogue

    dialogue = VoiceDialogue(known_entities=known_project_names)

    def execute(intent: intent_router.Intent) -> str:
        console.print(f"[dim]intent={intent.name} arg={intent.arg}[/dim]")
        return run_intent(intent)

    reply = dialogue.handle(text, execute)
    while dialogue.pending is not None:
        console.print(f"[green]Jarvix:[/green] {reply}")
        line = input("> ").strip()
        reply = dialogue.handle(line, execute)
    console.print(f"[green]{reply}[/green]")
    # This one-shot CLI command exits right after printing; give the
    # background memory-log/passive-detection threads a moment to finish
    # instead of losing them.
    for bg_thread in _last_background_threads:
        bg_thread.join(timeout=10)
    _shutdown_browser_if_running()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command()
def warmup():
    """No-op now that the brain is OpenAI's API (no local model to pre-load)."""
    _warmup()
    console.print("[green]Nothing to warm up - the brain runs on OpenAI's API now.[/green]")


@app.command()
def test_brain():
    """Test the OpenAI-backed brain."""
    answer = ask_llm(
        system_prompt="You are Jarvix, a personal voice assistant.",
        user_prompt="Say hello in one sentence and explain what you are.",
    )
    console.print(answer)


def _spoken_candidate(c: EventCandidate) -> str:
    """A short, speakable description of one candidate, e.g. 'Interview on 2026-06-21 at 15:00'."""
    when = ""
    if c.date:
        when = f" on {c.date}"
        if c.start_time:
            when += f" at {c.start_time}"
    return f"{c.title}{when}"


def _run_scan(limit: int, days: int, use_prefilter: bool, speak_summary: bool = False):
    """Shared scan pipeline: read -> (prefilter) -> extract -> validate -> dedupe -> confirm -> write.

    When ``speak_summary`` is on (voice flow), Jarvix reads the candidates aloud
    before the confirmation prompt and speaks the final result. The calendar
    write still requires the same typed confirmation - voice never auto-writes.
    """
    console.print(f"[bold cyan]Reading last {days} days of Gmail (up to {limit})...[/bold cyan]")
    emails = get_recent_emails(limit=limit, days=days)
    console.print(f"  {len(emails)} emails fetched.")

    if use_prefilter:
        before = len(emails)
        emails = prefilter(emails)
        console.print(
            f"  Prefilter kept {len(emails)}/{before} likely-relevant email(s); "
            f"the model only sees those."
        )

    console.print("[bold cyan]Extracting candidates with OpenAI (cached when unchanged)...[/bold cyan]")
    raw_candidates = extract_candidates(emails)
    accepted, rejected = validate_candidates(raw_candidates)

    console.print(
        f"  {len(accepted)} valid candidate(s), {len(rejected)} rejected."
    )
    if rejected:
        for c, why in rejected:
            console.print(f"    [dim]rejected: {c.title or '(untitled)'} - {why}[/dim]")

    if not accepted:
        console.print("[yellow]No calendar-worthy candidates found. Nothing to add.[/yellow]")
        if speak_summary:
            speak("I didn't find any calendar-worthy items in your recent email.")
        return

    console.print("[bold cyan]Checking your calendar for duplicates...[/bold cyan]")
    existing = get_events_window()

    console.print("\n[bold green]Candidates:[/bold green]\n")
    dupes_by_index: list[list[dict]] = []
    for i, c in enumerate(accepted, start=1):
        dupes = find_duplicates(c, existing)
        dupes_by_index.append(dupes)
        console.print(_format_candidate(i, c, dupes))
        console.print("")

    if speak_summary:
        n = len(accepted)
        items = " ".join(
            f"{i}. {_spoken_candidate(c)}"
            + (" (possible duplicate)" if dupes_by_index[i - 1] else "")
            + "."
            for i, c in enumerate(accepted, start=1)
        )
        speak(
            f"I found {n} calendar-worthy item{'s' if n != 1 else ''}: {items} "
            f"Should I add {'them' if n != 1 else 'it'}?"
        )

    # Safety gate: writing to the calendar always requires explicit confirmation.
    assert needs_confirmation("create_calendar_event")
    approved = _select_candidates(accepted)

    if not approved:
        console.print("[yellow]Nothing approved. No events created.[/yellow]")
        if speak_summary:
            speak("Okay, I didn't add anything.")
        return

    console.print(f"\n[bold cyan]Creating {len(approved)} event(s)...[/bold cyan]")
    created_count = 0
    for c in approved:
        try:
            created = create_event_from_candidate(c)
            created_count += 1
            console.print(f"  [green]Added:[/green] {c.title}  {created.get('htmlLink', '')}")
        except Exception as exc:  # surface the failure, keep going
            console.print(f"  [red]Failed:[/red] {c.title} - {exc}")

    if speak_summary:
        speak(f"Done. I added {created_count} to your calendar.")


@app.command()
def scan_mail(
    limit: int = typer.Option(15, help="Max emails to scan."),
    days: int = typer.Option(10, help="How many days back to read (7-14)."),
):
    """Scan Gmail (prefiltered + cached), extract candidates, and add approved ones to Calendar."""
    _run_scan(limit=limit, days=days, use_prefilter=True)


@app.command()
def scan_mail_deep(
    limit: int = typer.Option(12, help="Max emails to scan."),
    days: int = typer.Option(14, help="How many days back to read."),
):
    """Thorough scan: no keyword prefilter, the model sees every fetched email."""
    _run_scan(limit=limit, days=days, use_prefilter=False)


@app.command()
def open_app(name: str = typer.Argument(..., help="App alias, e.g. cursor / chrome / vscode.")):
    """Open an allowlisted desktop app by alias."""
    try:
        msg = desktop_open_app(name)
        console.print(f"[green]{msg}[/green]")
    except DesktopError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def open_folder(name: str = typer.Argument(..., help="Folder alias, e.g. jarvix / downloads.")):
    """Open an allowlisted folder by alias in Explorer."""
    try:
        msg = desktop_open_folder(name)
        console.print(f"[green]{msg}[/green]")
    except DesktopError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def wake(
    with_welcome: bool = typer.Option(
        True, "--welcome/--no-welcome", help="Run the welcome routine on the first trigger."
    ),
    mode: str = typer.Option(
        None, "--mode", help="Override WAKE_MODE: 'wakeword', 'clap', or 'enter'."
    ),
):
    """Stay awake: first trigger runs the welcome routine, then trigger to talk. Ctrl+C to stop."""
    from app.config import WAKE_MODE, WAKE_WORD
    from app.voice.wake_loop import wake_loop
    from app.voice.recorder import MicUnavailable

    wake_mode = (mode or WAKE_MODE).strip().lower()

    if wake_mode == "enter":
        def wait_for_wake() -> str | None:
            input("  [press Enter to talk] ")
            return None
        trigger_hint = "Press Enter to talk."
    elif wake_mode == "clap":
        from app.voice.clap_detector import wait_for_double_clap
        def wait_for_wake() -> str | None:
            wait_for_double_clap(verbose=True)
            return None
        trigger_hint = "Double-clap to talk."
    else:  # wakeword (default)
        from app.voice.wakeword import wait_for_wake_word
        def wait_for_wake() -> str | None:
            return wait_for_wake_word(verbose=True)
        trigger_hint = f'Say "hey {WAKE_WORD}" to talk.'

    from app.brain.dialogue import VoiceDialogue
    dialogue = VoiceDialogue(known_entities=known_project_names)

    def execute_voice_intent(intent: intent_router.Intent) -> str:
        console.print(f"[dim]intent={intent.name} arg={intent.arg}[/dim]")
        return run_intent(intent)

    def handle(text: str) -> str:
        return dialogue.handle(text, execute_voice_intent)

    def next_turn_max_seconds() -> float | None:
        # The new_project flow's "describe it" step needs a long-form answer,
        # not a short command clip - everything else uses the normal length.
        if dialogue.pending is not None and dialogue.pending.kind == "project_describe":
            return 20.0
        return None

    try:
        scheduler.start(USER_ID)
    except Exception as exc:
        console.print(f"[yellow]Autopilot scheduler failed to start: {exc}[/yellow]")

    console.print(f"[bold cyan]Jarvix is awake ({wake_mode} mode). {trigger_hint} Ctrl+C to stop.[/bold cyan]")
    try:
        if with_welcome:
            console.print(f"[cyan]{trigger_hint} to start your day...[/cyan]")
            wait_for_wake()
            pending_meeting = _speak_welcome()
            if pending_meeting:
                from app.brain.dialogue import Pending

                dialogue.pending = Pending(
                    "meeting_followup_offer",
                    {"meeting_id": pending_meeting["id"], "title": pending_meeting["title"]},
                )
        wake_loop(
            handle,
            speak,
            wait_for_wake=wait_for_wake,
            announce="",
            max_seconds_for_next_turn=next_turn_max_seconds,
        )
    except MicUnavailable as exc:
        console.print(f"[red]Microphone unavailable: {exc}. Jarvix can't run the wake loop.[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Jarvix going to sleep.[/yellow]")
    finally:
        _shutdown_browser_if_running()


def _shutdown_browser_if_running() -> None:
    """Close the Playwright Chrome context (if browser_task ever started one)
    so no orphaned Chrome process/profile lock survives the wake loop."""
    from app.browser.manager import get_manager

    manager = get_manager()
    if manager.is_running:
        console.print("[dim]Closing Jarvix's browser...[/dim]")
        manager.shutdown()


@app.command()
def autopilot_status():
    """Print last-run time and item counts for each background autopilot job."""
    state = scheduler.read_state()
    if not state:
        console.print("[yellow]No autopilot jobs have run yet.[/yellow]")
        return
    for job_name, info in state.items():
        console.print(f"\n[bold green]{job_name}[/bold green]")
        for key, value in info.items():
            console.print(f"  {key}: {value}")


@app.command()
def autopilot_run(
    job: str = typer.Argument(..., help="Job to run now: email-scan."),
):
    """Manually run one autopilot job immediately (bypasses its interval) - for testing."""
    if not USER_ID:
        console.print("[red]USER_ID isn't configured.[/red]")
        raise typer.Exit(code=1)
    try:
        result = scheduler.run_job_now(job, USER_ID)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{job} finished:[/green] {result}")


@app.command()
def clap_calibrate(seconds: int = typer.Option(15, help="How long to listen.")):
    """Live mic monitor for tuning clap detection.

    Clap a few times AND say a word or two. Claps should show as 'CLAP'; speech
    should be 'rejected'. Use the reported clap peak/brightness to tune.
    """
    import time
    import numpy as np
    import sounddevice as sd
    from app.config import CLAP_THRESHOLD
    from app.voice.clap_detector import _ClapTracker, _dynamic_threshold, brightness
    from app.voice.recorder import MicUnavailable

    sr, blk = 16000, int(16000 * 0.02)
    console.print(
        f"[bold cyan]Listening {seconds}s (adaptive, floor={CLAP_THRESHOLD}).\n"
        f"Clap a few times, then say 'hello' a few times...[/bold cyan]"
    )

    tracker = _ClapTracker()
    ambient = 0.005
    clap_peaks: list[float] = []
    last_clap_t = None
    doubles = 0
    start = time.time()
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blk) as s:
            while time.time() - start < seconds:
                d, _ = s.read(blk)
                samples = d[:, 0]
                rms = float(np.sqrt(np.mean(samples**2)))
                peak = float(np.max(np.abs(samples)))
                res = tracker.push(peak, _dynamic_threshold(ambient))
                if not tracker.in_event:
                    ambient = 0.95 * ambient + 0.05 * rms
                if not res:
                    continue
                br = brightness(samples)
                if res["is_clap"]:
                    clap_peaks.append(res["peak"])
                    now = time.time()
                    dbl = last_clap_t is not None and 0.12 <= (now - last_clap_t) <= 0.8
                    if dbl:
                        doubles += 1
                    tag = " [green]<- DOUBLE![/green]" if dbl else ""
                    console.print(
                        f"  [green]CLAP[/green]  peak {res['peak']:.3f}  "
                        f"bright {br:.2f}  {res['duration_ms']:.0f}ms{tag}"
                    )
                    last_clap_t = now
                else:
                    console.print(
                        f"  [dim]rejected  peak {res['peak']:.3f}  "
                        f"bright {br:.2f}  {res['duration_ms']:.0f}ms ({res['reason']})[/dim]"
                    )
    except Exception as exc:
        raise MicUnavailable(str(exc)) from exc

    console.print(f"\n[bold]Claps detected:[/bold] {len(clap_peaks)}   "
                  f"[bold]Double-claps:[/bold] {doubles}")
    if clap_peaks:
        suggested = round(min(clap_peaks) * 0.6, 3)
        console.print(
            f"[green]Quietest clap {min(clap_peaks):.3f}. "
            f"If claps were missed, set CLAP_THRESHOLD={suggested} in .env.[/green]"
        )
    else:
        console.print(
            "[yellow]No claps classified. If sounds showed as 'too long', clap sharper. "
            "If nothing showed at all, your mic is suppressing transients - disable mic "
            "enhancements in Windows Sound settings, or use WAKE_MODE=enter.[/yellow]"
        )


@app.command()
def listen():
    """Record one spoken command from the mic and transcribe it (offline)."""
    from app.voice.recorder import record, MicUnavailable
    from app.voice.stt import transcribe, warm

    console.print("[bold cyan]Listening... speak now.[/bold cyan]")
    warm()
    try:
        audio = record()
    except MicUnavailable as exc:
        console.print(f"[red]Microphone unavailable: {exc}[/red]")
        raise typer.Exit(code=1)
    if audio.size == 0:
        console.print("[yellow]Heard nothing. Try again.[/yellow]")
        raise typer.Exit(code=1)

    console.print("[bold cyan]Transcribing...[/bold cyan]")
    text = transcribe(audio)
    if not text:
        console.print("[yellow]Could not make out any words.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]You said:[/bold green] {text}")


@app.command()
def mic_debug(seconds: float = typer.Option(5.0, help="Seconds to record.")):
    """Record once, show mic levels, and transcribe the result."""
    import numpy as np
    from app.voice.recorder import record, MicUnavailable
    from app.voice.stt import transcribe, warm

    console.print(f"[bold cyan]Recording {seconds:.1f}s. Say: Jarvis read my calendar.[/bold cyan]")
    warm()
    try:
        audio = record(
            max_seconds=seconds,
            silence_threshold=0.003,
            trailing_silence=1.2,
            start_timeout=seconds,
        )
    except MicUnavailable as exc:
        console.print(f"[red]Microphone unavailable: {exc}[/red]")
        raise typer.Exit(code=1)

    if audio.size == 0:
        console.print("[yellow]No audio crossed the speech threshold.[/yellow]")
        return

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    console.print(f"[dim]samples={audio.size} rms={rms:.4f} peak={peak:.4f}[/dim]")
    text = transcribe(audio)
    console.print(f"[bold green]Transcribed:[/bold green] {text or '(empty)'}")


@app.command()
def music(
    action: str = typer.Argument(
        ..., help="playpause | play | pause | next | prev | volume-up | volume-down | mute"
    )
):
    """Control playback with media keys: playpause, next, prev, volume-up/down, mute."""
    act = action.strip().lower()
    if act in ("playpause", "play", "pause", "toggle", "play-pause"):
        msg = music_tool.play_pause()
    elif act in ("next", "skip"):
        msg = music_tool.next_track()
    elif act in ("prev", "previous", "back"):
        msg = music_tool.previous_track()
    elif act in ("volume-up", "volumeup", "vol-up", "louder"):
        msg = music_tool.volume_up()
    elif act in ("volume-down", "volumedown", "vol-down", "quieter"):
        msg = music_tool.volume_down()
    elif act in ("mute", "unmute"):
        msg = music_tool.mute()
    else:
        console.print(
            f"[red]Unknown music action '{action}'. Use: playpause, next, prev, "
            f"volume-up, volume-down, mute.[/red]"
        )
        raise typer.Exit(code=1)
    console.print(f"[green]{msg}[/green]")


@app.command()
def play(song: str = typer.Argument(..., help="Song or artist to search and play on Spotify.")):
    """Open Spotify, search for a song, and start playback."""
    try:
        msg = music_tool.play(song)
        console.print(f"[green]{msg}[/green]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


_BRIEF_STATE_FILE = ROOT / "app" / "memory" / "brief_state.json"


def _read_last_brief_at() -> str:
    """ISO timestamp of the last briefing, or 7 days ago on first run/failure."""
    try:
        if _BRIEF_STATE_FILE.exists():
            state = json.loads(_BRIEF_STATE_FILE.read_text(encoding="utf-8"))
            if state.get("last_brief_at"):
                return state["last_brief_at"]
    except Exception:
        pass
    return (datetime.now(ZoneInfo("UTC")) - timedelta(days=7)).isoformat()


def _write_last_brief_at() -> None:
    try:
        _BRIEF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BRIEF_STATE_FILE.write_text(
            json.dumps({"last_brief_at": datetime.now(ZoneInfo("UTC")).isoformat()}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        console.print(f"[yellow]Couldn't save brief state: {exc}[/yellow]")


def _briefing_rollup_text() -> str:
    """A short ' Since we last talked, ...' sentence covering what's new
    since the last briefing, or '' if there's nothing worth mentioning.
    Writes the new last-brief-time as a side effect - callers should only
    call this once per actual briefing (not for exploratory/test calls).
    """
    if not USER_ID:
        return ""
    since = _read_last_brief_at()
    try:
        new_opportunities = [
            {"name": o["name"], "fit_score": o["fit_score"]}
            for o in db_opportunities.list_recent_opportunities(USER_ID, limit=20)
            if str(o.get("created_at") or "") >= since
        ]
        new_applications = db_applications.list_updated_since(USER_ID, since)
        new_events = db_events.count_events_since(USER_ID, since)
        pending_followups = len(db_meetings.list_followup_candidates(USER_ID))
        high_importance_memories = [
            m["content"] for m in db_memories.list_recent_memories(USER_ID, limit=20)
            if m.get("importance", 0) > 0.6 and str(m.get("created_at") or "") >= since
        ]
        high_importance_goals = [
            g["title"] for g in db_goals.list_goals(USER_ID, limit=20)
            if str(g.get("created_at") or "") >= since
        ]
        sources = {
            "new_opportunities": new_opportunities,
            "new_applications_updated": new_applications,
            "new_autopilot_events": new_events,
            "pending_meeting_followups": pending_followups,
            "high_importance_memories": high_importance_memories,
            "high_importance_goals": high_importance_goals,
        }
        has_anything = any(
            [new_opportunities, new_applications, new_events, pending_followups,
             high_importance_memories, high_importance_goals]
        )
        summary = structuring.summarize_briefing_rollup(sources) if has_anything else ""
    except Exception as exc:
        console.print(f"[yellow]Briefing rollup skipped: {exc}[/yellow]")
        summary = ""
    finally:
        _write_last_brief_at()
    return f" {summary}" if summary else ""


def _pending_meeting_followup() -> dict | None:
    """The single oldest meeting due for a follow-up ask, or None. Asked
    about one at a time, not all at once."""
    if not USER_ID:
        return None
    candidates = db_meetings.list_followup_candidates(USER_ID)
    return candidates[0] if candidates else None


def _brief_text() -> str:
    """Build the short spoken daily briefing text without reading Gmail.

    Appends, when applicable: a "since last brief" rollup and a one-at-a-time
    meeting follow-up offer. The caller (wake command) is responsible for
    arming dialogue.pending so the next spoken turn catches the follow-up
    answer - this function only builds text, it doesn't touch dialogue state.
    """
    text = _calendar_brief_text()
    text += _briefing_rollup_text()
    meeting = _pending_meeting_followup()
    if meeting:
        text += f" Also, want a follow-up on {meeting['title']} from last week?"
    return text


@app.command()
def brief(
    silent: bool = typer.Option(False, "--silent", help="Print only, do not speak."),
):
    """Speak a short spoken daily briefing: greeting + today's events + key emails."""
    briefing = _brief_text()
    console.print("\n[bold green]Jarvix:[/bold green]\n")
    console.print(briefing)
    if not silent:
        speak(briefing)


def _today_text() -> str:
    """Build a fast spoken schedule summary without Gmail or the LLM."""
    console.print("[bold cyan]Reading calendar...[/bold cyan]")
    events = _todays_events(limit=12)
    if not events:
        return "Your calendar is clear for the rest of today."

    spoken = []
    for event in events[:6]:
        summary = event.get("summary") or "Untitled"
        spoken.append(f"{_spoken_event_time(event)}, {summary}")

    if len(spoken) == 1:
        items = spoken[0]
    elif len(spoken) == 2:
        items = f"{spoken[0]} and {spoken[1]}"
    else:
        items = ", ".join(spoken[:-1]) + f", and {spoken[-1]}"

    extra = len(events) - len(spoken)
    suffix = f" You also have {extra} more item{'s' if extra != 1 else ''}." if extra > 0 else ""
    return f"Today you have {len(events)} item{'s' if len(events) != 1 else ''}: {items}.{suffix}"


@app.command()
def today():
    """Read today's Calendar and print a short spoken schedule summary."""
    plan = _today_text()
    console.print("\n[bold green]Jarvix plan:[/bold green]\n")
    console.print(plan)


@app.command()
def today_fast():
    """Instant deterministic plan (no LLM): calendar + deadlines + top emails."""
    soon = get_upcoming_events(days=2, limit=20)
    window = get_events_window(days_ahead=14, limit=50)
    emails = get_recent_emails(limit=5, days=7)

    timed = [e for e in soon if "T" in (e.get("start") or "")]
    deadlines = [e for e in window if "T" not in (e.get("start") or "")]
    important = [e for e in emails if is_worthy(e)] or emails

    console.print("\n[bold green]Today (fast):[/bold green]\n")

    console.print("[bold]Calendar:[/bold]")
    if timed:
        for e in timed:
            when = e["start"][11:16] if "T" in e["start"] else e["start"]
            console.print(f"- {when} {e['summary']}")
    else:
        console.print("- (nothing scheduled today/tomorrow)")

    console.print("\n[bold]Deadlines:[/bold]")
    if deadlines:
        for e in deadlines[:5]:
            console.print(f"- {e['start']} {e['summary']}")
    else:
        console.print("- (none in the next 14 days)")

    console.print(f"\n[bold]Important emails:[/bold] ({len(important)})")
    for e in important[:3]:
        console.print(f"- {e['subject']}")

    console.print("\n[bold]Suggested:[/bold]")
    if timed:
        console.print(f"1. Prepare for: {timed[0]['summary']}.")
    else:
        console.print("1. No fixed events - use the time for deep work.")
    console.print("2. Triage the important emails above.")
    console.print("3. Re-check the calendar tonight.")
    console.print("\n[dim]Run 'today' for the full LLM plan.[/dim]")


@app.command()
def full_run(
    limit: int = typer.Option(12, help="Max emails to scan."),
    days: int = typer.Option(10, help="How many days back to read (7-14)."),
):
    """Run scan-mail first, then generate the today/tomorrow plan."""
    scan_mail(limit=limit, days=days)
    console.print("\n[bold]--- Daily plan ---[/bold]\n")
    today()


if __name__ == "__main__":
    app()
