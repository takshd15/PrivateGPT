"""Small in-memory dialogue state for filling missing voice-command details."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime

from app.brain import intent_router, structuring
from app.tools import desktop, live_info


@dataclass
class Pending:
    kind: str
    values: dict[str, str] = field(default_factory=dict)


# Accepted affirmative answers at any confirmation prompt (add_event, remember,
# new_project, log_progress). Anything else cancels that flow.
_CONFIRM_WORDS = {"yes", "yeah", "yep", "confirm", "sure", "correct", "do it", "add it", "save it", "log it"}

# Answers that mean "no specific location" for the opportunity-search flow -
# opportunities may be remote/global, so this must be offered as a valid
# answer, not treated as a location the user forgot to clean up.
_ANY_LOCATION_WORDS = {"any location", "anywhere", "any", "remote", "no location", "doesn't matter", "does not matter"}

# Common spoken-city mishears, mapped to the place actually meant. Voice STT
# mangles names constantly; a small correction table beats failing the lookup.
_CITY_CORRECTIONS = {
    "parris": "Paris",
    "ensch cannon": "Enschede",
    "ench cannon": "Enschede",
    "enchede": "Enschede",
    "enskede": "Enschede",
}


class VoiceDialogue:
    """Remember exactly one incomplete command between wake interactions."""

    def __init__(self, known_entities: Callable[[], list[str]] | None = None) -> None:
        self.pending: Pending | None = None
        # Injected by main.py (which owns DB access) so this module never
        # imports app.memory.db directly - keeps brain/ independent of memory/.
        self._known_entities = known_entities or (lambda: [])

    def known_entities(self) -> list[str]:
        try:
            return self._known_entities()
        except Exception:
            return []

    @staticmethod
    def _resolve_event_date(phrase: str | None) -> str | None:
        if not phrase:
            return None
        day = live_info.resolve_date_phrase(phrase)
        return day.isoformat() if day else None

    @staticmethod
    def _value(text: str) -> str:
        # Strip a stray BOM (U+FEFF) some terminals/pipes prepend to piped or
        # redirected stdin (e.g. PowerShell `Get-Content x | python ...`) -
        # otherwise it silently survives .strip() and breaks exact-match
        # checks like the confirm-word set below.
        text = text.lstrip("﻿")
        value = " ".join(text.strip(" ,.?!").split())
        lowered = value.lower()
        for prefix in ("it is ", "it's ", "the city is ", "the folder is "):
            if lowered.startswith(prefix):
                return value[len(prefix):].strip()
        return value

    @staticmethod
    def _city(text: str) -> str:
        """Clean a spoken weather follow-up into a bare city name.

        Handles answers like "check for Paris" or "weather in Paris" that carry
        the question's verbs along with the city, plus common name mishears.
        """
        value = VoiceDialogue._value(text)
        lowered = value.lower()
        for prefix in (
            "check for ",
            "check ",
            "weather in ",
            "weather for ",
            "the weather in ",
            "for ",
            "in ",
        ):
            if lowered.startswith(prefix):
                value = value[len(prefix):].strip()
                lowered = value.lower()
                break
        return _CITY_CORRECTIONS.get(lowered, value)

    def _continue(self, text: str) -> tuple[intent_router.Intent | None, str | None]:
        assert self.pending is not None
        pending = self.pending
        value = self._value(text)
        if value.lower() in {"cancel", "never mind", "nevermind", "stop"}:
            self.pending = None
            return None, "Okay, cancelled."

        if pending.kind == "email_recipient":
            pending.values["recipient"] = value
            pending.kind = "email_message"
            return None, "What should the email say?"
        if pending.kind == "email_message":
            self.pending = None
            name = pending.values["intent"]
            return intent_router.Intent(
                name,
                raw=text,
                recipient=pending.values["recipient"],
                message=value,
            ), None
        if pending.kind == "weather_location":
            self.pending = None
            return intent_router.Intent(intent_router.WEATHER, arg=self._city(text), raw=text), None
        if pending.kind == "folder":
            aliases = desktop.list_folders()
            lowered = value.lower()
            match = next((name for name in aliases if name == lowered or name in lowered), None)
            if not match:
                return None, f"I don't know that folder. Try {', '.join(aliases)}."
            self.pending = None
            return intent_router.Intent(intent_router.OPEN_FOLDER, arg=match, raw=text), None
        if pending.kind == "comparison":
            self.pending = None
            original = pending.values["original"]
            return intent_router.Intent(intent_router.QUESTION, raw=f"{original} {value}"), None

        if pending.kind == "event_title":
            pending.values["title"] = value
            pending.kind = "event_date"
            return None, "When is it?"
        if pending.kind == "event_date":
            day = live_info.resolve_date_phrase(value)
            if day is None:
                return None, "Sorry, what date is that? You can say something like tomorrow or next Friday."
            pending.values["date"] = day.isoformat()
            pending.kind = "event_time"
            return None, "What time? Say 'all day' if there's no specific time."
        if pending.kind == "event_time":
            lowered = value.lower()
            if lowered in {"all day", "no specific time", "any time", "anytime", "no time"}:
                pending.values["start_time"] = None
            else:
                start_time = live_info.parse_spoken_time(value)
                if start_time is None:
                    return None, "Sorry, what time? For example, 3 PM, or say all day."
                pending.values["start_time"] = start_time
            pending.kind = "event_confirm"
            return None, self._event_confirmation(pending.values)
        if pending.kind == "event_confirm":
            if value.lower() not in {"yes", "yeah", "yep", "confirm", "sure", "correct", "do it", "add it"}:
                self.pending = None
                return None, "Okay, I didn't add anything."
            self.pending = None
            return intent_router.Intent(intent_router.ADD_EVENT, raw=text, values=dict(pending.values)), None

        if pending.kind == "remember_title":
            self.pending = None
            candidate = structuring.structure_task(value)
            return self._finish_remember(text, candidate)
        if pending.kind == "remember_confirm":
            if value.lower() not in _CONFIRM_WORDS:
                self.pending = None
                return None, "Okay, I didn't add anything."
            self.pending = None
            return intent_router.Intent(intent_router.REMEMBER, raw=text, values=dict(pending.values)), None

        if pending.kind == "project_name":
            pending.values["name"] = value
            pending.kind = "project_describe"
            return None, "Go ahead and describe it, I'm listening."
        if pending.kind == "project_describe":
            candidate = structuring.structure_project(text, pending.values["name"])
            pending.values["description"] = candidate.description
            pending.values["status"] = candidate.status
            pending.kind = "project_confirm"
            return None, f"Got it — {candidate.name}: {candidate.description} Save this?"
        if pending.kind == "project_confirm":
            if value.lower() not in _CONFIRM_WORDS:
                self.pending = None
                return None, "Okay, I didn't save anything."
            self.pending = None
            return intent_router.Intent(intent_router.NEW_PROJECT, raw=text, values=dict(pending.values)), None

        if pending.kind == "progress_entity":
            pending.values["entity_name"] = value
            pending.kind = "progress_confirm"
            return None, self._progress_confirmation(pending.values)
        if pending.kind == "progress_confirm":
            if value.lower() not in _CONFIRM_WORDS:
                self.pending = None
                return None, "Okay, I didn't log anything."
            self.pending = None
            return intent_router.Intent(intent_router.LOG_PROGRESS, raw=text, values=dict(pending.values)), None

        if pending.kind == "opportunity_location":
            self.pending = None
            lowered = value.lower()
            location = None if lowered in _ANY_LOCATION_WORDS else value
            return intent_router.Intent(intent_router.FIND_OPPORTUNITIES, arg=location, raw=text), None

        if pending.kind == "meeting_followup_offer":
            if value.lower() not in _CONFIRM_WORDS:
                self.pending = None
                return intent_router.Intent(
                    intent_router.MEETING_FOLLOWUP,
                    raw=text,
                    values={"meeting_id": pending.values["meeting_id"], "declined": True},
                ), None
            pending.kind = "meeting_followup_capture"
            return None, "What was discussed?"
        if pending.kind == "meeting_followup_capture":
            self.pending = None
            candidate = structuring.structure_meeting_followup(value)
            return intent_router.Intent(
                intent_router.MEETING_FOLLOWUP,
                raw=text,
                values={
                    "meeting_id": pending.values["meeting_id"],
                    "title": pending.values["title"],
                    "summary": candidate.summary or value,
                    "declined": False,
                },
            ), None

        self.pending = None
        return None, "I didn't catch that clearly. Can you repeat it?"

    def _finish_remember(
        self, text: str, candidate
    ) -> tuple[intent_router.Intent | None, str | None]:
        """Shared tail for the remember flow: resolve the date, ask to confirm."""
        due_date = None
        if candidate.due_date_phrase:
            day = live_info.resolve_date_phrase(candidate.due_date_phrase)
            due_date = day.isoformat() if day else None
        values = {
            "title": candidate.title,
            "details": candidate.details,
            "due_date": due_date,
            "priority": candidate.priority,
            "entity_name": candidate.entity_name,
            "domain": candidate.domain,
        }
        self.pending = Pending("remember_confirm", values)
        when = f", due {due_date}" if due_date else ""
        return None, f"I'll add: {candidate.title}{when}. Confirm?"

    @staticmethod
    def _progress_confirmation(values: dict) -> str:
        entity = values.get("entity_name") or "that"
        event_text = values.get("event_text", "")
        event_type = values.get("event_type", "update")
        return f"Log on {entity}: {event_text} ({event_type}). Confirm?"

    @staticmethod
    def _event_confirmation(values: dict[str, str]) -> str:
        title = values.get("title", "the event")
        date_str = values.get("date", "")
        try:
            day = date_cls.fromisoformat(date_str)
            label = day.strftime("%A, %B %d").replace(" 0", " ")
        except ValueError:
            label = date_str
        start_time = values.get("start_time")
        if start_time:
            spoken = datetime.strptime(start_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
            when = f"{label} at {spoken}"
        else:
            when = f"{label} (all day)"
        return f"Add '{title}' on {when} to your calendar?"

    def handle(self, text: str, execute: Callable[[intent_router.Intent], str]) -> str:
        if self.pending is not None:
            intent, response = self._continue(text)
            if response is not None:
                return response
            if intent is None:
                return "I didn't catch that clearly. Can you repeat it?"
            return execute(intent)

        intent = intent_router.parse(text)
        if intent.name in {intent_router.SEND_EMAIL, intent_router.DRAFT_EMAIL}:
            values = {"intent": intent.name}
            if not intent.recipient:
                self.pending = Pending("email_recipient", values)
                return "Who should I email?"
            values["recipient"] = intent.recipient
            if not (intent.message or "").strip():
                self.pending = Pending("email_message", values)
                return "What should the email say?"
        if intent.name == intent_router.WEATHER and not intent.arg:
            self.pending = Pending("weather_location")
            return "Which city should I check?"
        if intent.name == intent_router.ADD_EVENT:
            self.pending = Pending("event_title")
            return "What's the event called?"
        if intent.name == intent_router.REMEMBER:
            candidate = structuring.structure_task(text)
            if not candidate.title.strip():
                self.pending = Pending("remember_title")
                return "What should I remember, exactly?"
            _, response = self._finish_remember(text, candidate)
            return response
        if intent.name == intent_router.NEW_PROJECT:
            self.pending = Pending("project_name")
            return "What's the project called?"
        if intent.name == intent_router.LOG_PROGRESS:
            candidate = structuring.structure_progress_event(text, self.known_entities())
            if not candidate.entity_name.strip():
                self.pending = Pending(
                    "progress_entity",
                    {
                        "event_text": candidate.event_text or text,
                        "event_type": candidate.event_type,
                        "event_date": self._resolve_event_date(candidate.event_date_phrase),
                    },
                )
                return "Which project or application is this progress on?"
            self.pending = Pending(
                "progress_confirm",
                {
                    "entity_name": candidate.entity_name,
                    "event_text": candidate.event_text,
                    "event_type": candidate.event_type,
                    "event_date": self._resolve_event_date(candidate.event_date_phrase),
                },
            )
            return self._progress_confirmation(self.pending.values)
        if intent.name == intent_router.FIND_OPPORTUNITIES and not intent.arg:
            self.pending = Pending("opportunity_location")
            return "What location? You can also say any location."
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "folder":
            self.pending = Pending("folder")
            return "Which folder should I open?"
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "comparison":
            self.pending = Pending("comparison", {"original": intent.raw})
            return "What two things should I compare?"
        return execute(intent)
