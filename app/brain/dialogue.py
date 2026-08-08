"""Small in-memory dialogue state for filling missing voice-command details."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime

from app.brain import intent_router
from app.tools import desktop, live_info


@dataclass
class Pending:
    kind: str
    values: dict[str, str] = field(default_factory=dict)


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

    def __init__(self) -> None:
        self.pending: Pending | None = None

    @staticmethod
    def _value(text: str) -> str:
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

        self.pending = None
        return None, "I didn't catch that clearly. Can you repeat it?"

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
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "folder":
            self.pending = Pending("folder")
            return "Which folder should I open?"
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "comparison":
            self.pending = Pending("comparison", {"original": intent.raw})
            return "What two things should I compare?"
        return execute(intent)
