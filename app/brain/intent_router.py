"""Hybrid intent router for Jarvix v2.

Transcribed voice text -> a structured Intent -> a tool call. Deterministic
rules run first and are instant/predictable, so the frequent stuff (open
app/folder, music, briefing) never needs a network round trip. Anything the
rules don't recognize falls through to app/brain/orchestrator.py, the LLM-
first semantic router that has the full tool registry and recent conversation
context available and can chain multiple tool calls before answering.

``execute`` only runs the safe, deterministic intents (apps, folders, music).
Higher-level intents (brief / today / scan_mail) are returned to the caller
(main) which owns their orchestration and any confirmation gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from app.brain.orchestrator import (
    AnswerResult,
    BrowserAskUserResult,
    BrowserConfirmationResult,
    BrowserRealChromeGateResult,
    BrowserResultWrapper,
    ClarificationResult,
    ConfirmationResult,
    Turn,
    orchestrate,
)
from app.tools import desktop, music

# Intent names
OPEN_APP = "open_app"
OPEN_FOLDER = "open_folder"
MUSIC_PLAY_PAUSE = "music_play_pause"
MUSIC_NEXT = "music_next"
MUSIC_PREVIOUS = "music_previous"
MUSIC_VOLUME_UP = "music_volume_up"
MUSIC_VOLUME_DOWN = "music_volume_down"
MUSIC_PLAY_QUERY = "music_play_query"
BRIEF = "brief"
TODAY = "today"
READ_EMAILS = "read_emails"
SCAN_MAIL = "scan_mail"
DRAFT_EMAIL = "draft_email"
SEND_EMAIL = "send_email"
QUESTION = "question"
CONVERSATION = "conversation"
WEATHER = "weather"
TIME = "time"
CALENDAR_DATE = "calendar_date"
ADD_EVENT = "add_event"
NEWS = "news"
REMEMBER = "remember"
NEW_PROJECT = "new_project"
LOG_PROGRESS = "log_progress"
FIND_OPPORTUNITIES = "find_opportunities"
# browser_task's own result is spoken as-is (BROWSER_RESULT); a pause for a
# high-impact in-page action becomes BROWSER_CONFIRM, routed through
# dialogue.py's browser_confirm Pending state (see _confirmation_to_intent).
# A genuine information gap (e.g. an application field with nothing on file)
# becomes BROWSER_ASK, routed through dialogue.py's browser_answer Pending
# state - same resume pattern, different question (open-ended, not yes/no).
BROWSER_RESULT = "browser_result"
BROWSER_CONFIRM = "browser_confirm"
BROWSER_ASK = "browser_ask"
# Real-Chrome mode only: the per-task gate before Jarvix drives the user's own
# live browser. Routed through dialogue.py's browser_real_chrome_gate Pending
# state (yes = run the task, no = leave the browser alone).
BROWSER_GATE = "browser_gate"
# Never parsed from user speech - only ever proactively armed by main.py's
# briefing flow (dialogue.pending set directly), so it's intentionally absent
# from _parse_rules and the orchestrator's tool-to-intent mapping.
MEETING_FOLLOWUP = "meeting_followup"
CLARIFICATION_NEEDED = "clarification_needed"
UNKNOWN = "unknown"

# Intents that execute() handles directly (safe + deterministic).
_SIMPLE = {
    OPEN_APP,
    OPEN_FOLDER,
    MUSIC_PLAY_PAUSE,
    MUSIC_NEXT,
    MUSIC_PREVIOUS,
    MUSIC_VOLUME_UP,
    MUSIC_VOLUME_DOWN,
    MUSIC_PLAY_QUERY,
}

# Phrases that separate the recipient from the message ("...saying I'll be late").
_SAY_MARKERS = ("saying", "telling them", "telling him", "telling her", "to say", "that", "about")

# Common spoken-music mishears -> what the user almost certainly meant. STT
# routinely mangles artist names ("Travis code" for "Travis Scott"), so a small
# correction table makes the Spotify search land on the right thing.
_MUSIC_CORRECTIONS = {
    "travis code": "Travis Scott",
    "travis court": "Travis Scott",
    "travis cott": "Travis Scott",
    "tailor swift": "Taylor Swift",
    "taylor swiss": "Taylor Swift",
    "the weekend": "The Weeknd",
    "weekend": "The Weeknd",
}


@dataclass
class Intent:
    name: str
    arg: str | None = None
    raw: str = ""
    recipient: str | None = None
    message: str | None = None
    values: dict | None = None
    # Only set for BROWSER_CONFIRM - the paused browser_task loop state
    # (app.browser.tools.BrowserPaused) so dialogue.py can resume the SAME
    # in-progress browser task once the user answers yes/no.
    browser_paused: Any | None = None
    # Only set for BROWSER_ASK - the browser_task loop state waiting on an
    # open-ended answer (app.browser.tools.BrowserAskingUser), so dialogue.py
    # can resume the SAME in-progress browser task with whatever the user says.
    browser_asking: Any | None = None
    # Only set for BROWSER_GATE - the real-Chrome per-task gate state
    # (app.browser.tools.BrowserRealChromeGate) so dialogue.py can run the task
    # on approval or abandon it on decline.
    browser_gate: Any | None = None


def _extract_recipient(text: str) -> str | None:
    """Recipient phrase after ``to`` and before the message instruction."""
    # A literal email address is matched first and wins outright. The general
    # name pattern below stops at the first [,.!?], which silently truncates
    # any real address at its domain dot - "to someone@gmail.com" came back
    # as "someone@gmail" (live bug, 2026-08-11), i.e. every dotted domain,
    # which is all of them.
    m = re.search(
        r"\bto\s+([A-Za-z0-9][A-Za-z0-9._%+\-']*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,})",
        text,
        re.I,
    )
    if m:
        return m.group(1).strip(".,!?")
    stop = r"(?=\s+(?:saying|telling\s+(?:them|him|her)|to\s+say|that|about)\b|[,.!?]|$)"
    m = re.search(r"\bto\s+([A-Za-z][A-Za-z0-9@+ ._\-']*?)" + stop, text, re.I)
    return " ".join(m.group(1).split()).strip() if m else None


def _extract_message(text: str) -> str:
    """The instruction after a say-marker, e.g. '...saying I'll be late' -> "I'll be late"."""
    # The separator accepts a colon ("saying: do the thing") as well as plain
    # whitespace - dictating with a colon is natural, but a bare \s+ can't
    # match ':' so the whole message came back empty and the caller fell into
    # the "What should the email say?" slot-fill (live bug, 2026-08-11).
    # Deliberately NOT a loose [:,\-]? - that would make the fuzzy markers
    # ("that", "about") swallow hyphenated words like "about-face".
    pattern = (
        r"\b(?:" + "|".join(re.escape(m) for m in _SAY_MARKERS) + r")\b(?:\s*:\s*|\s+)(.+)"
    )
    m = re.search(pattern, text, re.I)
    return m.group(1).strip() if m else ""


def _clean(text: str) -> str:
    t = " ".join(text.lower().strip().strip(".!?,").split())
    t = re.sub(r"^(hey\s+)?jarvis\b[:,]?\s*", "", t)
    for filler in (
        "can you please ",
        "could you please ",
        "would you please ",
        "please ",
        "can you ",
        "could you ",
        "would you ",
    ):
        if t.startswith(filler):
            t = t[len(filler):]
            break
    return t.strip()


def _match_alias(text: str, names: list[str]) -> str | None:
    """Return the longest allowlisted alias that appears in the text, if any."""
    found = [
        name
        for name in names
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)
    ]
    return max(found, key=len) if found else None


def _extract_music_query(text: str) -> str | None:
    t = _clean(text)
    quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
    if quoted:
        return quoted.group(1).strip()

    m = re.search(r"\b(?:play|search for|find)\s+(.+)", t)
    if not m:
        return None

    query = m.group(1).strip()
    query = re.sub(r"\b(on|in)\s+spotify\b", "", query).strip()
    query = re.sub(r"\b(song|track|music)\b", "", query).strip()
    query = re.sub(r"^(a|an|some|the)\s+", "", query).strip()
    query = re.sub(r"\s+", " ", query)
    if query in {"", "a", "an", "some", "the", "spotify"}:
        return None
    return _MUSIC_CORRECTIONS.get(query.lower(), query) or None


def _extract_location_phrase(text: str) -> str | None:
    """Best-effort location from phrases such as 'weather in Enschede today'
    or 'opportunities in Berlin'. Shared by WEATHER and FIND_OPPORTUNITIES."""
    m = re.search(r"\b(?:in|for|at)\s+(.+)", text, re.I)
    if not m:
        return None
    location = re.sub(
        r"\b(?:right\s+now|now|today|tomorrow|this\s+(?:morning|afternoon|evening|week))\b.*$",
        "",
        m.group(1),
        flags=re.I,
    )
    return location.strip(" ,.?!") or None


def _extract_opportunity_location(text: str) -> str | None:
    """Location from phrases like 'opportunities in Berlin' - anchored right
    after 'opportunit(y/ies)' specifically, unlike the generic 'in/for/at X'
    pattern, since "search FOR opportunities" and "opportunities... for me"
    both contain a "for" that isn't introducing a location."""
    m = re.search(r"\bopportunit\w*\s+(?:in|for|at|near)\s+(.+)", text, re.I)
    if not m:
        return None
    location = m.group(1).strip(" ,.?!")
    if not location or location.lower() in {"me", "myself", "us"}:
        return None
    return location


def _extract_calendar_date(text: str) -> str | None:
    t = _clean(text)
    patterns = (
        r"\b(next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        r"\b(tomorrow)\b",
        r"\b(?:on|for)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:on|for)\s+(\d{4}-\d{2}-\d{2})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            return match.group(1)
    return None


def _parse_rules(text: str) -> Intent:
    """Fast deterministic parser for common spoken commands."""
    t = _clean(text)
    if not t:
        return Intent(UNKNOWN, raw=text)
    if t in {
        "can you",
        "can you please",
        "please",
        "and you",
        "by calendar",
        "bye",
        "you",
    }:
        return Intent(CLARIFICATION_NEEDED, raw=text)
    if "by calendar" in t:
        return Intent(CLARIFICATION_NEEDED, raw=text)

    folders = desktop.list_folders()
    apps = desktop.list_apps()

    # 0. Opportunity search - must precede block 1's music "find "/"search for "
    #    prefix match, which would otherwise swallow "find opportunities for me".
    if "opportunit" in t and any(k in t for k in ("find", "search", "look")):
        return Intent(FIND_OPPORTUNITIES, arg=_extract_opportunity_location(text), raw=text)

    # 1. Music commands are common in speech and should tolerate polite phrasing.
    if any(k in t for k in ("volume up", "louder", "turn it up", "turn up")):
        return Intent(MUSIC_VOLUME_UP, raw=text)
    if any(k in t for k in ("volume down", "quieter", "turn it down", "turn down")):
        return Intent(MUSIC_VOLUME_DOWN, raw=text)
    if any(
        k in t
        for k in (
            "change song",
            "change the song",
            "change track",
            "change the track",
            "next song",
            "next the song",
            "next track",
            "skip",
            "skip song",
            "skip this song",
            "skip track",
            "skip this track",
            "something else",
            "another song",
        )
    ):
        return Intent(MUSIC_NEXT, raw=text)
    if any(k in t for k in ("previous", "last song", "go back", "prev")):
        return Intent(MUSIC_PREVIOUS, raw=text)
    if "spotify" in t and "open" in t.split():
        return Intent(OPEN_APP, "spotify", text)
    if "spotify" in t and any(k in t.split() for k in ("start", "play")):
        query = _extract_music_query(text)
        return Intent(MUSIC_PLAY_QUERY, query, text)
    if any(k in t for k in ("play music", "play a music", "start music")):
        return Intent(MUSIC_PLAY_QUERY, raw=text)
    if any(k in t for k in ("pause", "resume", "play music", "stop music", "play pause")):
        return Intent(MUSIC_PLAY_PAUSE, raw=text)
    if t.startswith(("play ", "search for ", "find ")):
        query = _extract_music_query(text)
        return Intent(MUSIC_PLAY_QUERY, query, text) if query else Intent(MUSIC_PLAY_PAUSE, raw=text)

    # 2. Explicit "open ..." commands.
    if "open" in t.split() or t.startswith("open"):
        if "folder" in t or "directory" in t:
            fname = _match_alias(t, folders)
            return Intent(OPEN_FOLDER, fname, text) if fname else Intent(CLARIFICATION_NEEDED, arg="folder", raw=text)
        aname = _match_alias(t, apps)
        if aname:
            return Intent(OPEN_APP, aname, text)
        fname = _match_alias(t, folders)
        if fname:
            return Intent(OPEN_FOLDER, fname, text)
        return Intent(UNKNOWN, raw=text)

    # 3. Add/create a new calendar event (must come before the calendar/today
    #    catch-alls in blocks 6-7, which are read-only). Excludes email/mail
    #    mentions - "add calendar events from my email" is SCAN_MAIL (block 5),
    #    not a directly-described new event.
    if (
        ("email" not in t and "mail" not in t)
        and any(k in t for k in ("add", "create", "schedule", "put", "book", "new"))
        and any(k in t for k in ("event", "calendar", "meeting", "appointment", "reminder"))
    ):
        return Intent(ADD_EVENT, raw=text)

    # 3a. Remember / remind / don't-forget - must precede email drafting (block 4)
    #     so "remind me to email my advisor" becomes a task, not a drafted email.
    if any(k in t for k in ("remember", "remind me", "don't forget", "dont forget")):
        return Intent(REMEMBER, raw=text)

    # 3b. New project - declaring one is starting, not asking to list what
    #     already exists. "I'm working on/starting a NEW project" (or an
    #     explicit first-person announcement) creates one; a bare question
    #     like "what projects am I working on" must fall through instead (no
    #     rule here reads projects back - that's the orchestrator's job via
    #     the list_projects tool) rather than being misread as project #3.
    if "new project" in t or (("starting" in t or "started" in t) and "project" in t):
        return Intent(NEW_PROJECT, raw=text)
    if re.match(r"^(i'?m|i am|i've|i have|just)\s+(working on|building|starting)\b", t) and "project" in t:
        return Intent(NEW_PROJECT, raw=text)

    # 3c. Progress/update logging.
    if any(k in t for k in ("log progress", "progress on", "update on")):
        return Intent(LOG_PROGRESS, raw=text)

    # 4. Email drafting / sending (must come before plain email-reading).
    if ("email" in t or "mail" in t) and any(
        k in t for k in ("draft", "write", "compose", "send")
    ):
        recipient = _extract_recipient(text)
        message = _extract_message(text)
        name = SEND_EMAIL if ("send" in t.split()) else DRAFT_EMAIL
        return Intent(name, raw=text, recipient=recipient, message=message)

    # 5. Gmail reading vs calendar extraction.
    if any(k in t for k in ("email", "emails", "mail", "inbox")):
        if any(k in t for k in ("calendar", "calender", "event", "events", "schedule", "add")):
            return Intent(SCAN_MAIL, raw=text)
        if any(k in t for k in ("read", "tell", "summarize", "summary", "today", "latest", "recent", "check")):
            return Intent(READ_EMAILS, raw=text)
        return Intent(SCAN_MAIL, raw=text)

    # 6. Live information. These must precede the broad "today" calendar rule.
    if "weather" in t or "forecast" in t or "temperature" in t:
        return Intent(WEATHER, arg=_extract_location_phrase(text), raw=text)
    if any(
        phrase in t
        for phrase in ("what time", "current time", "time is it", "tell me the time")
    ):
        return Intent(TIME, raw=text)
    if any(k in t for k in ("news", "headlines", "top stories", "what's happening")):
        return Intent(NEWS, raw=text)

    # 7. Calendar with an explicit relative/absolute day.
    if any(k in t for k in ("calendar", "calender", "schedule", "events", "plan")):
        date_phrase = _extract_calendar_date(text)
        if date_phrase:
            return Intent(CALENDAR_DATE, arg=date_phrase, raw=text)

    # 8. Day / schedule / plan.
    if any(
        k in t
        for k in (
            "my day",
            "today",
            "calendar",
            "schedule",
            "my plan",
            "plan for",
            "to do",
            "to-do",
            "supposed to do",
        )
    ):
        return Intent(TODAY, raw=text)

    # 9. Briefing.
    if any(k in t for k in ("brief", "briefing", "welcome", "catch me up", "good morning")):
        return Intent(BRIEF, raw=text)

    # 10. Obvious questions/chit-chat skip the classifier and go straight to a spoken answer.
    if re.search(r"\bdifference between\s*$", t):
        return Intent(CLARIFICATION_NEEDED, arg="comparison", raw=text)
    if t.startswith(
        (
            "what ",
            "what's ",
            "why ",
            "who ",
            "who's ",
            "when ",
            "where ",
            "how ",
            "explain ",
            "tell me ",
            "give me ",
            "define ",
            "best ",
            "top ",
            "should ",
            "can ",
            "could ",
            "would ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
        )
    ):
        return Intent(QUESTION, raw=text)
    if any(k in t for k in ("how are you", "thank you", "thanks", "nice", "cool")):
        return Intent(CONVERSATION, raw=text)

    return Intent(UNKNOWN, raw=text)


# Intents the orchestrator's tool calls map back onto, so a mutating tool
# selection (e.g. create_calendar_event) flows through the SAME dialogue.py
# pending-confirmation states the deterministic ADD_EVENT/SEND_EMAIL rules
# already use, instead of inventing a second confirmation path.
_TOOL_TO_INTENT = {
    "create_calendar_event": ADD_EVENT,
    "send_email": SEND_EMAIL,
    "remember_task": REMEMBER,
    "create_project": NEW_PROJECT,
    "log_progress": LOG_PROGRESS,
}


def _confirmation_to_intent(result: ConfirmationResult, text: str) -> Intent:
    """A mutating tool the orchestrator selected, re-expressed as the same
    Intent the deterministic rules would have produced for it - so it flows
    through dialogue.py's EXISTING pending-confirmation/slot-filling states
    (which still ask for any details the orchestrator didn't already have)
    instead of a second, parallel confirmation system."""
    mapped = _TOOL_TO_INTENT.get(result.tool_name)
    args = result.arguments
    if mapped == ADD_EVENT:
        return Intent(
            ADD_EVENT,
            raw=text,
            values={"title": args.get("title", ""), "date": args.get("date", ""), "start_time": args.get("start_time")},
        )
    if mapped == SEND_EMAIL:
        return Intent(SEND_EMAIL, raw=text, recipient=args.get("recipient"), message=args.get("message"))
    if mapped == NEW_PROJECT:
        # Unlike REMEMBER/LOG_PROGRESS below, NEW_PROJECT's slot-filling in
        # dialogue.py is multi-turn (project_name -> project_describe ->
        # project_confirm), so a bare Intent(NEW_PROJECT, raw=text) with no
        # values would re-arm project_name and wait for the NEXT user turn to
        # fill in the name - and that next turn is the user's confirmation
        # reply ("yes"/"exactly, save this"), not a project name, so it got
        # captured as the name instead and the project never actually saved
        # (see jarvix.log 2026-08-10: "Trend Analyzer" stuck re-confirming
        # forever). When the orchestrator already extracted a name, pass it
        # straight through so dialogue.py can go directly to project_confirm
        # instead of re-asking for information it already has.
        name = (args.get("name") or "").strip()
        if name:
            return Intent(NEW_PROJECT, raw=text, values={"name": name, "description": args.get("description") or ""})
        return Intent(NEW_PROJECT, raw=text)
    if mapped in (REMEMBER, LOG_PROGRESS):
        # These two are handled by dialogue.py re-running its own
        # structuring.py-backed slot filling from `raw` (see _handle_inner) -
        # both do so IMMEDIATELY within the same turn (no follow-up round
        # trip), so returning the bare intent with no values is safe here,
        # unlike NEW_PROJECT above.
        return Intent(mapped, raw=text)
    return Intent(UNKNOWN, raw=text)


def _parse_with_orchestrator(
    text: str, history: list[Turn] | None, on_event: Callable[[dict], None] | None = None
) -> Intent:
    """Fallback for anything the deterministic rules didn't match: hands the
    request to the LLM orchestrator (app/brain/orchestrator.py), which has the
    full tool registry and recent conversation context available - unlike the
    old single-shot classify-and-return call this replaces, it can call
    multiple tools before deciding on a final answer or a clarification.

    ``on_event`` is passed straight through to orchestrate() - see its
    docstring in app/brain/orchestrator.py for the event shapes. Optional;
    None (the default) changes nothing about existing behavior.
    """
    try:
        result = orchestrate(text, history=history or [], on_event=on_event)
    except Exception:
        return Intent(UNKNOWN, raw=text)

    if isinstance(result, AnswerResult):
        return Intent(QUESTION, raw=text, values={"answer": result.text})
    if isinstance(result, ClarificationResult):
        return Intent(CLARIFICATION_NEEDED, raw=text, values={"question": result.question})
    if isinstance(result, ConfirmationResult):
        return _confirmation_to_intent(result, text)
    if isinstance(result, BrowserResultWrapper):
        return Intent(BROWSER_RESULT, raw=text, values={"answer": result.text})
    if isinstance(result, BrowserConfirmationResult):
        return Intent(BROWSER_CONFIRM, raw=text, browser_paused=result.paused, values={"description": result.description})
    if isinstance(result, BrowserAskUserResult):
        return Intent(BROWSER_ASK, raw=text, browser_asking=result.asking, values={"question": result.question})
    if isinstance(result, BrowserRealChromeGateResult):
        return Intent(BROWSER_GATE, raw=text, browser_gate=result.gate, values={"instruction": result.instruction})
    return Intent(UNKNOWN, raw=text)


def parse(
    text: str,
    use_llm: bool = True,
    history: list[Turn] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Intent:
    """Map raw transcribed text to an Intent. Never raises.

    Rules handle common commands (open app/folder, music, briefing, and every
    other deterministic pattern) instantly with no network round trip. When
    rules can't classify the request (UNKNOWN) OR only get as far as
    "this is some kind of question/chat" (QUESTION/CONVERSATION - rules have
    no tool registry or conversation memory of their own), it's handed to the
    LLM orchestrator instead, so an open-ended question is answered with real
    tool-grounded context rather than a plain, memory-less LLM call. Only
    when ``use_llm`` is true (the deterministic-rules test suite calls this
    with use_llm=False to test _parse_rules in isolation).

    ``on_event`` is an optional live-progress callback forwarded to the
    orchestrator when it's actually reached; ignored (and never called) when
    a rule resolves the request instead, since there's nothing to stream.
    """
    intent = _parse_rules(text)
    if not use_llm or intent.name not in (UNKNOWN, QUESTION, CONVERSATION):
        return intent
    return _parse_with_orchestrator(text, history, on_event=on_event)


def execute(intent: Intent) -> str | None:
    """Run a safe/deterministic intent and return a spoken response.

    Returns None for higher-level intents (brief/today/scan_mail/unknown) so the
    caller can handle them. Catches tool errors and returns a safe message.
    """
    if intent.name not in _SIMPLE:
        return None

    try:
        if intent.name == OPEN_APP:
            if (intent.arg or "").lower() == "spotify":
                return music.open_spotify()
            return desktop.open_app(intent.arg or "")
        if intent.name == OPEN_FOLDER:
            return desktop.open_folder(intent.arg or "")
        if intent.name == MUSIC_PLAY_PAUSE:
            return music.play_pause()
        if intent.name == MUSIC_NEXT:
            return music.next_track()
        if intent.name == MUSIC_PREVIOUS:
            return music.previous_track()
        if intent.name == MUSIC_VOLUME_UP:
            return music.volume_up()
        if intent.name == MUSIC_VOLUME_DOWN:
            return music.volume_down()
        if intent.name == MUSIC_PLAY_QUERY:
            if intent.arg:
                return music.play(intent.arg)
            music.open_spotify()
            return music.play_pause()
    except desktop.DesktopError as exc:
        return str(exc)
    except Exception as exc:  # never let a tool crash the voice loop
        return f"Sorry, that failed: {exc}"

    return None
