"""Rule-based intent router for Jarvix v2.

Transcribed voice text -> a structured Intent -> a tool call. Deterministic on
purpose: rules are instant and predictable, so the frequent stuff (open
app/folder, music, briefing) is matched by rules instead of a network round
trip. Anything unmatched becomes ``unknown`` and is answered safely instead of
guessed at.

``execute`` only runs the safe, deterministic intents (apps, folders, music).
Higher-level intents (brief / today / scan_mail) are returned to the caller
(main) which owns their orchestration and any confirmation gates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.brain.llm_client import ask_llm
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
# Never parsed from user speech - only ever proactively armed by main.py's
# briefing flow (dialogue.pending set directly), so it's intentionally absent
# from _parse_rules/_coerce_llm_intent/the LLM prompt's intent list.
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


def _extract_recipient(text: str) -> str | None:
    """Recipient phrase after ``to`` and before the message instruction."""
    stop = r"(?=\s+(?:saying|telling\s+(?:them|him|her)|to\s+say|that|about)\b|[,.!?]|$)"
    m = re.search(r"\bto\s+([A-Za-z][A-Za-z0-9@+ ._\-']*?)" + stop, text, re.I)
    return " ".join(m.group(1).split()).strip() if m else None


def _extract_message(text: str) -> str:
    """The instruction after a say-marker, e.g. '...saying I'll be late' -> "I'll be late"."""
    pattern = r"\b(?:" + "|".join(re.escape(m) for m in _SAY_MARKERS) + r")\b\s+(.+)"
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

    # 3b. New project.
    if "new project" in t or ("working on" in t and "project" in t):
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


def _coerce_llm_intent(data: dict, raw: str) -> Intent:
    name = str(data.get("intent") or UNKNOWN).strip().lower()
    allowed = {
        OPEN_APP,
        OPEN_FOLDER,
        MUSIC_PLAY_PAUSE,
        MUSIC_NEXT,
        MUSIC_PREVIOUS,
        MUSIC_VOLUME_UP,
        MUSIC_VOLUME_DOWN,
        MUSIC_PLAY_QUERY,
        BRIEF,
        TODAY,
        READ_EMAILS,
        SCAN_MAIL,
        DRAFT_EMAIL,
        SEND_EMAIL,
        QUESTION,
        CONVERSATION,
        WEATHER,
        TIME,
        CALENDAR_DATE,
        ADD_EVENT,
        NEWS,
        REMEMBER,
        NEW_PROJECT,
        LOG_PROGRESS,
        FIND_OPPORTUNITIES,
        CLARIFICATION_NEEDED,
        UNKNOWN,
    }
    if name not in allowed:
        return Intent(UNKNOWN, raw=raw)

    arg_value = data.get("arg")
    arg = str(arg_value).strip() if arg_value is not None else None
    recipient = data.get("recipient")
    recipient = str(recipient).strip() if recipient is not None else None
    message = data.get("message")
    message = str(message).strip() if message is not None else None

    if name == OPEN_APP and arg:
        arg = _match_alias(arg.lower(), desktop.list_apps())
        if not arg:
            return Intent(UNKNOWN, raw=raw)
    if name == OPEN_FOLDER and arg:
        arg = _match_alias(arg.lower(), desktop.list_folders())
        if not arg:
            return Intent(UNKNOWN, raw=raw)

    return Intent(name, arg=arg, raw=raw, recipient=recipient, message=message)


def _parse_with_llm(text: str) -> Intent:
    apps = ", ".join(desktop.list_apps())
    folders = ", ".join(desktop.list_folders())
    system_prompt = f"""
You classify one spoken command for Jarvix, a voice assistant that calls tools.

Decision procedure - follow in order:
1. Identify what the user is actually asking for.
2. Check whether exactly one of the intents listed below is the correct tool for it.
   Read each intent's description carefully; do not pick an intent whose
   description doesn't match just because a keyword overlaps.
3. If exactly one intent fits, return it with its required fields filled in.
4. If NO listed intent is the right tool for this request (it needs general
   knowledge, explanation, opinion, or reasoning instead), return {QUESTION} so
   the assistant answers directly instead of forcing a bad tool match.
5. Only use {CLARIFICATION_NEEDED} when the text itself is too broken/vague to
   even identify what is being asked (not merely because no tool fits it).

Return ONLY valid JSON with keys: intent, arg, recipient, message.
Use null when a field is not needed.

Allowed intents:
- {OPEN_APP}: open a configured app. arg must be one of: {apps}
- {OPEN_FOLDER}: open a configured folder. arg must be one of: {folders}
- {MUSIC_PLAY_QUERY}: play a requested song. arg is the song/artist/Spotify URL, or null for generic music.
- {MUSIC_PLAY_PAUSE}: pause, resume, or toggle current playback.
- {MUSIC_NEXT}: next/change/skip song.
- {MUSIC_PREVIOUS}: previous/back/last song.
- {MUSIC_VOLUME_UP}: louder/volume up.
- {MUSIC_VOLUME_DOWN}: quieter/volume down.
- {READ_EMAILS}: read or summarize Gmail/inbox messages.
- {SCAN_MAIL}: check Gmail for calendar-worthy events/deadlines and add approved items to Calendar.
- {ADD_EVENT}: create/add/schedule a brand-new calendar event the user is describing now (not extracted from email).
- {REMEMBER}: remember/remind/don't-forget a task or reminder. Takes priority over drafting/sending an email even if the task mentions emailing someone - "remind me to email X" is a task, not an email to send.
- {NEW_PROJECT}: the user says they're starting/working on a new project.
- {LOG_PROGRESS}: log/give a progress update on an existing project, application, or opportunity.
- {FIND_OPPORTUNITIES}: an explicit REQUEST to search/find opportunities right now (e.g. "find opportunities for me", "search for grants", "look for hackathons"). arg is the requested location, or null if not stated. A statement of wanting/trying to achieve something (e.g. "I want to get into grad school", "I'm trying to land an internship at X") is {CONVERSATION} or {QUESTION}, NOT this - the assistant should respond conversationally, not launch a search uninvited.
- {TODAY}: summarize today's schedule/plan/tasks.
- {CALENDAR_DATE}: calendar/schedule for a day other than today. arg is the spoken date phrase.
- {BRIEF}: brief/catch up/good morning.
- {DRAFT_EMAIL}: draft an email. recipient/message when present.
- {SEND_EMAIL}: send an email. recipient/message when present.
- {NEWS}: current news headlines/top stories.
- {WEATHER}: current weather or forecast. arg is the requested location or null.
- {TIME}: current local time.
- {QUESTION}: knowledge, explanation, advice, reasoning, or anything no other intent covers.
- {CONVERSATION}: casual chat that does not need a tool.
- {CLARIFICATION_NEEDED}: fragment/nonsense/likely speech-recognition failure.
- {UNKNOWN}: reserved; prefer {QUESTION} instead when in doubt.

Safety:
- Never invent app or folder aliases.
- Use {SCAN_MAIL} only when the command asks to find/add calendar events from email.
- Use {READ_EMAILS} when the command only asks to read/check/summarize email.
- Use {ADD_EVENT} only when the user is directly describing a new event to create, not asking to read/check the calendar.
- Use {REMEMBER} whenever the trigger word is "remember"/"remind me"/"don't forget", regardless of what the reminder is about.
- Use {CLARIFICATION_NEEDED} for fragments like "can you please", "by calendar", "and you", or nonsense - not for legitimate questions that simply have no dedicated tool.
"""
    user_prompt = f"Command: {text}"
    try:
        raw = ask_llm(
            system_prompt,
            user_prompt,
            json_mode=True,
            timeout=4,
            num_predict=80,
        )
        data = json.loads(raw)
    except Exception:
        return Intent(UNKNOWN, raw=text)
    return _coerce_llm_intent(data, text)


def parse(text: str, use_llm: bool = True) -> Intent:
    """Map raw transcribed text to an Intent. Never raises.

    Rules handle common commands instantly. The local LLM is used only as a
    fallback for unfamiliar wording.
    """
    intent = _parse_rules(text)
    if intent.name != UNKNOWN or not use_llm:
        return intent
    return _parse_with_llm(text)


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
