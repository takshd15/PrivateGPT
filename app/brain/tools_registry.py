"""Structured tool registry exposed to the LLM orchestrator (app/brain/orchestrator.py).

Every entry wraps an EXISTING function in app/tools/* or app/memory/db/* - no
new business logic lives here. USER_ID is bound inside each handler (never
accepted as an LLM-supplied argument), matching the existing pattern in
app/main.py (e.g. known_project_names()).

Tools with mutates=True and requires_confirmation=True must not be executed
directly by the orchestrator loop - it stops and hands control back to the
existing dialogue.py pending-confirmation flow instead (see orchestrator.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import USER_ID
from app.memory import db
from app.memory.db import (
    applications as db_applications,
    goals as db_goals,
    meetings as db_meetings,
    memories as db_memories,
    opportunities as db_opportunities,
    people as db_people,
    projects as db_projects,
    tasks as db_tasks,
)
from app.tools import gmail, live_info, news as news_tool, websearch
from app.tools.calendar import get_events_for_date, get_upcoming_events


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    mutates: bool = False
    requires_confirmation: bool = False
    returns: str = ""


def _empty_params() -> dict:
    return {"type": "object", "properties": {}, "required": []}


# --------------------------------------------------------------------------- #
# Handlers - thin wrappers binding USER_ID and shaping args from the LLM.
# --------------------------------------------------------------------------- #
def _get_upcoming_events(days: int = 2) -> list[dict]:
    return get_upcoming_events(days=days, limit=20)


def _get_events_for_date(date: str) -> list[dict]:
    from datetime import date as date_cls

    day = live_info.resolve_date_phrase(date) or date_cls.fromisoformat(date)
    return get_events_for_date(day, limit=20)


def _get_recent_emails(limit: int = 10, days: int = 7) -> list[dict]:
    return gmail.get_recent_emails(limit=limit, days=days)


def _get_weather(location: str | None = None) -> str:
    return live_info.weather(location)


def _get_time() -> str:
    return live_info.spoken_time()


def _get_news(country: str = "us", limit: int = 5) -> list[dict]:
    return news_tool.get_top_headlines(country=country, limit=limit)


def _search_memories(query: str, limit: int = 5) -> list[dict]:
    return db.retrieve_similar(query, k=limit)


def _get_recent_memories(limit: int = 10) -> list[dict]:
    if not USER_ID:
        return []
    return db_memories.list_recent_memories(USER_ID, limit=limit)


def _get_goals(limit: int = 10) -> list[dict]:
    if not USER_ID:
        return []
    return db_goals.list_goals(USER_ID, limit=limit)


def _list_projects(limit: int = 50) -> list[dict]:
    if not USER_ID:
        return []
    return db_projects.list_projects(USER_ID, limit=limit)


def _get_project(name: str) -> list[dict]:
    if not USER_ID:
        return []
    needle = (name or "").strip().lower()
    return [p for p in db_projects.list_projects(USER_ID, limit=100) if needle in (p["name"] or "").lower()]


def _get_tasks(limit: int = 20) -> list[dict]:
    if not USER_ID:
        return []
    return db_tasks.list_open_tasks(USER_ID, limit=limit)


def _get_opportunities(limit: int = 10) -> list[dict]:
    if not USER_ID:
        return []
    return db_opportunities.list_recent_opportunities(USER_ID, limit=limit)


def _search_opportunities(query: str, num_results: int = 8) -> list[dict]:
    """Raw web-search results for opportunities, NOT inserted into the DB.
    (Saving found opportunities stays the job of the existing FIND_OPPORTUNITIES
    intent/_find_opportunities_text - this tool only gives the orchestrator
    fresh data to reason over when what's already saved isn't enough.)
    """
    return websearch.search(query, num_results=num_results)


def _get_applications(limit: int = 20) -> list[dict]:
    if not USER_ID:
        return []
    return db_applications.list_applications(USER_ID, limit=limit)


def _search_people(name: str) -> list[dict]:
    if not USER_ID:
        return []
    return db_people.find_person_by_name(USER_ID, name)


def _get_meetings(limit: int = 10) -> list[dict]:
    if not USER_ID:
        return []
    return db_meetings.list_recent(USER_ID, limit=limit)


def _web_search(query: str, num_results: int = 5) -> list[dict]:
    return websearch.search(query, num_results=num_results)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_upcoming_events",
        description="Get the user's upcoming calendar events for the next N days (default 2).",
        parameters={
            "type": "object",
            "properties": {"days": {"type": "integer", "description": "How many days ahead to look."}},
            "required": [],
        },
        handler=_get_upcoming_events,
        returns="List of events with summary, start, end, location.",
    ),
    ToolSpec(
        name="get_events_for_date",
        description="Get the user's calendar events for one specific day (today, tomorrow, a weekday name, or YYYY-MM-DD).",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string", "description": "e.g. 'today', 'tomorrow', 'next friday', or '2026-08-10'."}},
            "required": ["date"],
        },
        handler=_get_events_for_date,
        returns="List of events with summary, start, end, location.",
    ),
    ToolSpec(
        name="create_calendar_event",
        description="Create a new calendar event. MUTATES the calendar - requires user confirmation before running.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM 24h, or null for an all-day event."},
            },
            "required": ["title", "date"],
        },
        handler=None,  # routed through dialogue.py's existing ADD_EVENT confirmation flow, never called directly
        mutates=True,
        requires_confirmation=True,
        returns="Confirmation that the event was created.",
    ),
    ToolSpec(
        name="remember_task",
        description="Save a task/reminder the user wants remembered, e.g. 'remember to email my advisor tomorrow'. MUTATES - requires confirmation.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "due_date_phrase": {"type": "string", "description": "Spoken due date phrase, e.g. 'tomorrow', or omit if none."},
            },
            "required": ["title"],
        },
        handler=None,  # routed through dialogue.py's existing REMEMBER confirmation flow
        mutates=True,
        requires_confirmation=True,
        returns="Confirmation the task was saved.",
    ),
    ToolSpec(
        name="create_project",
        description="Record a new project the user says they're starting or working on. MUTATES - requires confirmation.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name"],
        },
        handler=None,  # routed through dialogue.py's existing NEW_PROJECT confirmation flow
        mutates=True,
        requires_confirmation=True,
        returns="Confirmation the project was saved.",
    ),
    ToolSpec(
        name="log_progress",
        description="Log a progress update, blocker, milestone, or decision against an existing project/application/opportunity. MUTATES - requires confirmation.",
        parameters={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "The project/application/opportunity this progress is about."},
                "event_text": {"type": "string"},
            },
            "required": ["entity_name", "event_text"],
        },
        handler=None,  # routed through dialogue.py's existing LOG_PROGRESS confirmation flow
        mutates=True,
        requires_confirmation=True,
        returns="Confirmation the progress was logged.",
    ),
    ToolSpec(
        name="get_recent_emails",
        description="Get the user's recent Gmail messages (subject, sender, snippet).",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "days": {"type": "integer", "description": "How many days back to look."},
            },
            "required": [],
        },
        handler=_get_recent_emails,
        returns="List of emails with subject, from, snippet.",
    ),
    ToolSpec(
        name="send_email",
        description="Draft and send an email to a named contact. MUTATES - sends real email, requires confirmation.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string", "description": "What the email should say."},
            },
            "required": ["recipient", "message"],
        },
        handler=None,  # routed through the existing SEND_EMAIL/DRAFT_EMAIL dialogue flow
        mutates=True,
        requires_confirmation=True,
        returns="Confirmation the email was sent.",
    ),
    ToolSpec(
        name="get_weather",
        description="Get the current weather/forecast for a city. Ask for a city if the user hasn't named one and none is on file.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name, or omit to use the user's default."}},
            "required": [],
        },
        handler=_get_weather,
        returns="Short spoken weather description.",
    ),
    ToolSpec(
        name="get_time",
        description="Get the current local time.",
        parameters=_empty_params(),
        handler=_get_time,
        returns="Short spoken time string.",
    ),
    ToolSpec(
        name="get_news",
        description="Get current top news headlines.",
        parameters={
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Two-letter country code, default 'us'."},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        handler=_get_news,
        returns="List of headline articles with title, source, description.",
    ),
    ToolSpec(
        name="search_memories",
        description="Semantic search over past conversations/interactions for relevant context (RAG). Use when you need to recall something the user said before that isn't in the current conversation.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
        handler=_search_memories,
        returns="List of past {transcript, response} pairs relevant to the query.",
    ),
    ToolSpec(
        name="get_recent_memories",
        description="Get the user's most important/recent stored memories (durable facts, preferences, plans).",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_recent_memories,
        returns="List of memories with kind, content, domain, importance.",
    ),
    ToolSpec(
        name="get_goals",
        description="Get the user's active goals, e.g. what they're trying to achieve (career, projects, startup, etc). Use this before ranking opportunities or advising on fit.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_goals,
        returns="List of goals with title, description, priority.",
    ),
    ToolSpec(
        name="list_projects",
        description="List the projects the user is currently working on or has recorded.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_list_projects,
        returns="List of projects with name, description, status.",
    ),
    ToolSpec(
        name="get_project",
        description="Look up a specific project by name (partial match), e.g. to answer 'tell me about GDCN'.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        handler=_get_project,
        returns="List of matching projects with name, description, status.",
    ),
    ToolSpec(
        name="get_tasks",
        description="Get the user's open tasks/reminders.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_tasks,
        returns="List of tasks with title, details, due_date, priority, entity_name.",
    ),
    ToolSpec(
        name="get_opportunities",
        description="Get opportunities (hackathons, grants, accelerators, jobs, scholarships) already found and saved for the user.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_opportunities,
        returns="List of opportunities with name, type, url, deadline, fit_score, status.",
    ),
    ToolSpec(
        name="search_opportunities",
        description="Search the web for NEW opportunities when the saved list is stale, empty, or the user asks for a fresh/different search. Does not save results automatically.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
            "required": ["query"],
        },
        handler=_search_opportunities,
        returns="List of raw search results with title, url, snippet.",
    ),
    ToolSpec(
        name="get_applications",
        description="Get the user's tracked job/program applications and their status.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_applications,
        returns="List of applications with name, program, status, deadline, notes.",
    ),
    ToolSpec(
        name="search_people",
        description="Look up a contact/person by name, e.g. to answer 'what did Sarah say' or find someone's role/email.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        handler=_search_people,
        returns="List of matching people with name, email, role, notes.",
    ),
    ToolSpec(
        name="get_meetings",
        description="Get the user's recent meetings, including notes/summaries from any that were discussed afterward.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        handler=_get_meetings,
        returns="List of meetings with title, notes, location, starts_at, ends_at.",
    ),
    ToolSpec(
        name="web_search",
        description="General-purpose web search for anything not covered by another tool (facts, current events beyond news headlines, lookups).",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
            "required": ["query"],
        },
        handler=_web_search,
        returns="List of search results with title, url, snippet.",
    ),
    ToolSpec(
        name="browser_task",
        description=(
            "Control a real Chrome browser to complete a task on any website: open pages, search, "
            "read content, click, type, fill forms, log into services, compare options, download or "
            "upload files, etc. Use this for anything that requires actually operating a website rather "
            "than a canned data lookup (e.g. 'find my latest email from X in Gmail', 'find the cheapest "
            "flight from A to B', 'check my GitHub notifications'). "
            "Also use this for 'open <name>' when the user names a site by a short familiar name rather "
            "than a URL (e.g. 'open github', 'open my bank') - it looks up the user's own browsing history "
            "to find the exact site/account they mean. "
            "Also use this for job/internship applications, applying to part-time jobs, and messaging "
            "leads/recruiters - it can fill out an entire application (name, email, resume upload, a drafted "
            "cover-letter/outreach message) using the user's own saved applicant details, but ALWAYS pauses "
            "for confirmation right before the final Submit/Apply/Send click, no matter how much of the form "
            "it already filled in. "
            "Runs its own step-by-step browser agent loop and pauses for your confirmation before any "
            "high-impact action (sending, purchasing, deleting, publishing, submitting an application, "
            "changing account/security settings, etc). Never buys, sends, or submits anything without that "
            "confirmation, and never invents personal details it wasn't given."
        ),
        parameters={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The full task to perform in the browser, in plain language, including any details already known (site, dates, destination, recipient, search terms, etc).",
                }
            },
            "required": ["instruction"],
        },
        handler=None,  # dispatched by app/brain/orchestrator.py directly to app/browser/tools.py's agent loop, not run as a plain data-lookup tool
        returns="A spoken-style summary of what was found/done, or a request to confirm a high-impact step, or that human help is needed (CAPTCHA/2FA/etc).",
    ),
]

_BY_NAME = {t.name: t for t in TOOLS}


def get(name: str) -> ToolSpec | None:
    return _BY_NAME.get(name)


def openai_schemas() -> list[dict]:
    """Tool list in the shape OpenAI's `tools` chat-completions param expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description + (f" Returns: {t.returns}" if t.returns else ""),
                "parameters": t.parameters,
            },
        }
        for t in TOOLS
    ]
