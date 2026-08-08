"""Raw speech -> typed structured-capture candidates, via OpenAI.

One function per entity type (task, project, progress event). Each is a pure
function: text in, a validated Pydantic model out - no DB access here (that
stays in app/main.py's handlers, which fetch any needed context first and
pass it in). Dates/times are never parsed by the LLM: it extracts the raw
spoken phrase, and the caller resolves it with app/tools/live_info.py.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from app.brain.llm_client import ask_llm
from app.models import (
    ConvoDetectionCandidate,
    EmailSignalCandidate,
    MeetingFollowupCandidate,
    OpportunityCandidate,
    ProgressEventCandidate,
    ProjectCandidate,
    TaskCandidate,
)

_TASK_SYSTEM = """You extract a single task from one spoken/typed request to a
personal assistant ("remember...", "remind me...", "don't forget...").

Return ONLY valid JSON with keys: title, details, due_date_phrase, priority,
entity_name, domain, confidence.

- title: short imperative phrase, e.g. "Email advisor" (not "remember to email advisor").
- details: any extra context, or "" if none.
- due_date_phrase: the raw spoken date phrase (e.g. "tomorrow", "next Friday",
  "August 20th") EXACTLY as implied by the text, or null if no date was mentioned.
  Do NOT resolve it to a real date yourself - copy the phrase.
- priority: 0-3 integer, 0 default, higher only if urgency is explicit ("urgent", "asap").
- entity_name: a project/topic name if one is clearly implied, else null.
- domain: a short category word (e.g. "school", "work", "personal"), or null.
- confidence: 0.0-1.0, your certainty that title captures a real, actionable task.

If no real task/reminder is present, return title as "" and confidence 0.0.
"""

_PROJECT_SYSTEM = """You write a clean project description from a spoken/typed
description a user gave about a project they're working on.

Return ONLY valid JSON with keys: description, status, confidence.

- description: a rewritten, clear 1-3 sentence description (not verbatim
  transcript) capturing what the project actually is.
- status: almost always "active" unless the text clearly implies otherwise
  (e.g. "on hold", "just an idea for later" -> "planned").
- confidence: 0.0-1.0, your certainty the input described a real project.
"""

_PROGRESS_SYSTEM = """You extract a progress-log entry from a spoken/typed
update about a project, application, or opportunity the user is tracking.

Known existing entities (projects/applications/etc.) the user might mean:
{entities}

Return ONLY valid JSON with keys: entity_name, event_text, event_type,
event_date_phrase, confidence.

- entity_name: pick the closest matching name from the known entities list
  above if one is clearly meant, copied EXACTLY as it appears in that list.
  If no known entity matches or none was implied, return null - do not invent
  a new name.
- event_text: a concise, rewritten summary of the update (not verbatim).
- event_type: exactly one of "update", "blocker", "milestone", "decision".
- event_date_phrase: the raw spoken date phrase if a date was mentioned
  (e.g. "yesterday", "last week"), else null (caller defaults to today).
- confidence: 0.0-1.0, your certainty this is a real, loggable progress update.
"""


_OPPORTUNITIES_SYSTEM = """You extract candidate opportunities (jobs, grants,
competitions, scholarships, hackathons, accelerators, etc.) from raw web
search results, for a personal assistant helping the user find things worth
pursuing.

Today's date: {today}

The user's stated goals and recent context (use this to judge fit - if this
is empty, score fit_score based on general relevance/quality instead):
{profile}

Return ONLY valid JSON with key "opportunities": a list of objects, each with
keys: name, type, url, deadline, fit_score, reason, next_action.

- name: a short, clear name/title for the opportunity.
- type: one of job/grant/competition/scholarship/hackathon/accelerator/other.
- url: the URL from the search result this came from. Never invent a URL.
- deadline: any deadline mentioned, as free text exactly as stated (e.g.
  "March 15, 2026", "rolling"), or null if none is stated. Do not resolve or
  guess a date - copy what's stated, or leave null.
- fit_score: 0.0-1.0, how well this matches the user's goals/context above
  (or general relevance/quality if no goals were given).
- reason: one short sentence explaining the fit_score.
- next_action: one short, concrete suggested next step (e.g. "Read the
  eligibility page and note the deadline").

Only include results that describe a genuine, specific opportunity - skip
generic pages, navigation results, or results with no real opportunity
described. If nothing in the results qualifies, return an empty list.
"""

_EMAIL_SIGNALS_SYSTEM = """You analyze ONE email for two SEPARATE signals. Most
emails should produce NULL for both - be strict, this must not flag ordinary
newsletters, marketing, or routine notifications.

Return ONLY valid JSON with keys: signup, application.

1. "signup": null unless this email clearly confirms the user signed up,
   enrolled, registered, or confirmed attendance for a SPECIFIC event, webinar,
   course, or meetup. If it does, an object with keys: name, event_type
   (webinar/course/meetup/event/etc.), date_phrase (raw phrase as stated, or
   null), location (or null), confirmation_details (one short sentence).

2. "application": null unless this email is clearly about a job/program
   application the user submitted (application confirmation, interview
   invite, rejection, offer, or status update). If it is, an object with
   keys: company_name, program (role/program name, or ""), status (describe
   the status in 1-3 words, e.g. "applied", "interviewing", "rejected",
   "offer", "waitlisted" - your own words, not a fixed list), deadline_phrase
   (raw phrase, or null), contact_name, contact_email.
   - Only set contact_name/contact_email when a SPECIFIC NAMED PERSON (not a
     no-reply/system/ATS sender) is the point of contact for this email.
     Otherwise leave both null.

Bias toward null for both. A promotional newsletter, a receipt, a generic
"thanks for subscribing" - all null.
"""


_MEETING_FOLLOWUP_SYSTEM = """You write a clean, concise summary of what the
user says was discussed in a meeting they're recapping by voice.

Return ONLY valid JSON with keys: summary, confidence.

- summary: a rewritten, clear 1-3 sentence summary of the discussion (not
  verbatim transcript) - what was actually decided or covered.
- confidence: 0.0-1.0, your certainty this is a real recap (not silence,
  refusal, or an unrelated answer).
"""

_CONVO_DETECTION_SYSTEM = """You review one exchange between a user and their
voice assistant for two SEPARATE signals. Most exchanges should produce EMPTY
for both - be strict, this must not fire on ordinary factual questions or
small talk.

Return ONLY valid JSON with keys: memory, goal.

1. "memory": an object with keys content, domain, importance.
   - content: "" unless the user stated a DURABLE fact about themselves, a
     preference, or a plan (e.g. "I prefer working late", "I'm allergic to
     peanuts", "I'm moving to Berlin in March"). If so, a short third-person
     sentence capturing it (e.g. "Prefers working late at night."). Do NOT
     record ordinary questions, requests, or facts about the world that
     aren't about the user.
   - domain: a short category word (e.g. "work", "health", "personal"), or
     null if content is "".
   - importance: 0.0-1.0, how durable/significant this fact is. 0.0 if
     content is "".

2. "goal": an object with keys title, description, priority.
   - title: "" unless the user stated a goal or objective they're pursuing
     (e.g. "I want to get into grad school", "trying to land an internship at
     Google") - a short imperative/noun phrase for the goal, distinct from a
     plain fact. If a task/reminder was stated instead (a concrete one-off
     action), that is NOT a goal - leave title "".
   - description: one sentence of context, or "" if title is "".
   - priority: 0-3 integer, 0 default, higher only if urgency/importance is
     explicit in the phrasing.

Current user request: {transcript}
Assistant's response: {response}
"""

_BRIEFING_ROLLUP_SYSTEM = """You write ONE short spoken sentence (STRICTLY
30-40 words, no more) summarizing what's new for the user since their last
briefing, from the structured data below. This will be spoken aloud as part
of a daily briefing, appended after the calendar summary.

Prioritize by recency and importance/fit_score - do not try to mention
everything, pick the 2-4 most notable items. Do not just list counts; give a
genuinely useful, natural-sounding summary a person would actually want to
hear.

Data since last briefing:
{sources}

If the data above is empty or has nothing worth mentioning, return an empty
string for "summary".

Return ONLY valid JSON with key: summary (a single string, 30-40 words or "").
"""


def _ask_json(system_prompt: str, user_prompt: str, num_predict: int = 250) -> dict:
    try:
        raw = ask_llm(system_prompt, user_prompt, json_mode=True, timeout=15, num_predict=num_predict)
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def structure_task(raw_text: str) -> TaskCandidate:
    data = _ask_json(_TASK_SYSTEM, f"Request: {raw_text}")
    try:
        return TaskCandidate(**data)
    except Exception:
        return TaskCandidate()


def structure_project(raw_text: str, name: str) -> ProjectCandidate:
    data = _ask_json(_PROJECT_SYSTEM, f"Project name: {name}\nDescription given by user: {raw_text}")
    data["name"] = name
    try:
        return ProjectCandidate(**data)
    except Exception:
        return ProjectCandidate(name=name, description=raw_text.strip())


def structure_progress_event(raw_text: str, known_entities: list[str]) -> ProgressEventCandidate:
    entities_list = ", ".join(known_entities) if known_entities else "(none known yet)"
    system_prompt = _PROGRESS_SYSTEM.format(entities=entities_list)
    data = _ask_json(system_prompt, f"Update: {raw_text}")
    try:
        return ProgressEventCandidate(**data)
    except Exception:
        return ProgressEventCandidate()


def _has_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def structure_opportunities(
    search_results: list[dict], profile_snippet: str, today: str
) -> list[OpportunityCandidate]:
    """One LLM call over all search results together. Filters out any
    candidate with no syntactically-valid url before returning."""
    if not search_results:
        return []
    results_text = "\n\n".join(
        f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('snippet', '')}"
        for r in search_results
    )
    system_prompt = _OPPORTUNITIES_SYSTEM.format(
        today=today, profile=profile_snippet or "(no stated goals yet)"
    )
    # Up to 8 opportunities, each a small object - needs a larger budget than
    # the other structuring calls (which return one small object) or the
    # response gets cut off mid-JSON and silently fails to parse.
    data = _ask_json(system_prompt, f"Search results:\n\n{results_text}", num_predict=1500)
    raw_candidates = data.get("opportunities", []) if isinstance(data, dict) else []
    if not isinstance(raw_candidates, list):
        return []

    candidates: list[OpportunityCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        try:
            candidate = OpportunityCandidate(**item)
        except Exception:
            continue
        if _has_url(candidate.url):
            candidates.append(candidate)
    return candidates


def structure_email_signals(email: dict) -> EmailSignalCandidate:
    """Combined single-call extraction: both a possible signup and a possible
    application signal from one email. Both null by default."""
    body = (email.get("body") or email.get("snippet") or "")[:2000]
    user_prompt = (
        f"From: {email.get('from', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Date: {email.get('date', '')}\n"
        f"Body:\n{body}"
    )
    data = _ask_json(_EMAIL_SIGNALS_SYSTEM, user_prompt)
    try:
        return EmailSignalCandidate(**data)
    except Exception:
        return EmailSignalCandidate()


def structure_meeting_followup(raw_text: str) -> MeetingFollowupCandidate:
    data = _ask_json(_MEETING_FOLLOWUP_SYSTEM, f"Recap: {raw_text}")
    try:
        return MeetingFollowupCandidate(**data)
    except Exception:
        return MeetingFollowupCandidate(summary=raw_text.strip())


def detect_memory_and_goal(transcript: str, response: str) -> ConvoDetectionCandidate:
    """One combined LLM call: passively detect a durable memory and/or a
    goal in one conversational exchange. Both empty by default - this must
    not fire on every message, only genuinely durable statements."""
    system_prompt = _CONVO_DETECTION_SYSTEM.format(transcript=transcript, response=response)
    data = _ask_json(system_prompt, "Analyze the exchange above.", num_predict=300)
    try:
        return ConvoDetectionCandidate(**data)
    except Exception:
        return ConvoDetectionCandidate()


def summarize_briefing_rollup(sources: dict) -> str:
    """One short (~30-40 word) spoken sentence summarizing what's new since
    the last briefing. "" if nothing worth mentioning or on failure - caller
    should fall back to a short deterministic line in that case."""
    if not sources:
        return ""
    sources_text = json.dumps(sources, indent=2, default=str)
    system_prompt = _BRIEFING_ROLLUP_SYSTEM.format(sources=sources_text)
    data = _ask_json(system_prompt, "Write the rollup sentence.", num_predict=150)
    summary = data.get("summary", "") if isinstance(data, dict) else ""
    return summary.strip() if isinstance(summary, str) else ""
