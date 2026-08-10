"""Small in-memory dialogue state for filling missing voice-command details."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime

from app.brain import intent_router, structuring
from app.brain.orchestrator import Turn, orchestrate
from app.browser.tools import (
    BrowserAskingUser,
    BrowserDone,
    BrowserFailed,
    BrowserNeedsHuman,
    BrowserPaused,
    resume_browser_task,
    resume_browser_task_with_answer,
    resume_real_chrome_task,
)
from app.tools import desktop, live_info


@dataclass
class Pending:
    kind: str
    values: dict[str, str] = field(default_factory=dict)


# Word-boundary affirmative/negative markers for confirmation prompts
# (add_event, remember, new_project, log_progress, meeting_followup). Matched
# anywhere in the utterance rather than requiring an exact whole-string match,
# so "Yeah, save this." and "yes please" both count as a clean yes instead of
# silently falling through to "I didn't save anything" (see _classify_confirmation).
# "save/add/log THIS" (echoing the assistant's own "Save this?" question back)
# and "exactly"/"that's right"/"that's correct" were added after a live bug
# (2026-08-10): "Exactly. Save this." fell through to "ambiguous" because
# neither word was recognized, silently failing to save a described project
# even though it's an unambiguous confirmation to a human ear.
_YES_WORDS = re.compile(
    r"\b(yes|yeah|yep|yup|exactly|confirm(?:ed)?|sure|correct|that'?s right|that'?s correct|"
    r"do it|add it|add this|save it|save this|log it|log this|go ahead)\b",
    re.I,
)
_NO_WORDS = re.compile(r"\b(no|nope|nah|don't|do not|cancel|never ?mind|stop)\b", re.I)
# A yes-word followed by a modification clause ("yeah, but call it X instead",
# "sure, except change the date") is NOT a clean confirmation of the payload
# as-is - it must be treated as ambiguous so the orchestrator can see the
# requested change, never silently confirmed with the old details.
_MODIFICATION_MARKERS = re.compile(r"\b(but|except|instead|actually|change|make it|not)\b", re.I)

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
        # Rolling window of the last 5 user/assistant turns, given to the
        # orchestrator so it can resolve references like "the best one" back
        # to whatever was just discussed. Bounded so this never grows unbounded
        # across a long wake session.
        self.history: deque[Turn] = deque(maxlen=5)

    def known_entities(self) -> list[str]:
        try:
            return self._known_entities()
        except Exception:
            return []

    def _record_turn(self, user_text: str, response: str) -> None:
        self.history.append(Turn("user", user_text))
        if response:
            self.history.append(Turn("assistant", response))

    @staticmethod
    def _classify_confirmation(value: str) -> str:
        """'yes' / 'no' / 'ambiguous' for a reply to an active confirmation
        prompt. Matches affirmative/negative markers anywhere in the
        utterance (word-boundary, not exact-string) so "Yeah, save this."
        counts as a clean yes instead of falling through to a silent no-op.
        A yes-word paired with a modification marker ("yeah, but call it X
        instead") is ambiguous, not a clean yes - the payload isn't being
        confirmed as-is."""
        has_yes = bool(_YES_WORDS.search(value))
        has_no = bool(_NO_WORDS.search(value))
        has_modification = bool(_MODIFICATION_MARKERS.search(value))
        if has_yes and has_modification:
            return "ambiguous"
        if has_yes and not has_no:
            return "yes"
        if has_no and not has_yes:
            return "no"
        return "ambiguous"

    def _resolve_ambiguous_confirmation(
        self, text: str, prompt_kind: str, values: dict, on_event: Callable[[dict], None] | None = None
    ) -> str:
        """A pending confirmation reply that's neither a clean yes nor no
        (e.g. "yeah, but change the name to X"). Ask the orchestrator to
        decide whether this modifies the pending payload and should be
        re-confirmed, or should be treated as a cancel - never silently
        discards the pending action."""
        summary = ", ".join(f"{k}={v}" for k, v in values.items() if v)
        note = (
            f"There is a pending '{prompt_kind}' confirmation with details: {summary}. "
            f"The user's reply was ambiguous. If they're modifying a detail, describe the "
            f"updated action briefly and ask them to confirm again. If they're declining, say so plainly."
        )
        result = orchestrate(text, history=list(self.history), pending_note=note, on_event=on_event)
        from app.brain.orchestrator import AnswerResult, ClarificationResult

        if isinstance(result, ClarificationResult):
            return result.question
        if isinstance(result, AnswerResult):
            return result.text
        return "Okay, I didn't change anything."

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

    def _continue(
        self, text: str, on_event: Callable[[dict], None] | None = None
    ) -> tuple[intent_router.Intent | None, str | None]:
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
            verdict = self._classify_confirmation(value)
            if verdict == "no":
                self.pending = None
                return None, "Okay, I didn't add anything."
            if verdict == "ambiguous":
                return None, self._resolve_ambiguous_confirmation(text, "add_event", pending.values, on_event=on_event)
            self.pending = None
            return intent_router.Intent(intent_router.ADD_EVENT, raw=text, values=dict(pending.values)), None

        if pending.kind == "remember_title":
            self.pending = None
            candidate = structuring.structure_task(value)
            return self._finish_remember(text, candidate)
        if pending.kind == "remember_confirm":
            verdict = self._classify_confirmation(value)
            if verdict == "no":
                self.pending = None
                return None, "Okay, I didn't add anything."
            if verdict == "ambiguous":
                return None, self._resolve_ambiguous_confirmation(text, "remember", pending.values, on_event=on_event)
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
            verdict = self._classify_confirmation(value)
            if verdict == "no":
                self.pending = None
                return None, "Okay, I didn't save anything."
            if verdict == "ambiguous":
                return None, self._resolve_ambiguous_confirmation(text, "new_project", pending.values, on_event=on_event)
            self.pending = None
            return intent_router.Intent(intent_router.NEW_PROJECT, raw=text, values=dict(pending.values)), None

        if pending.kind == "progress_entity":
            pending.values["entity_name"] = value
            pending.kind = "progress_confirm"
            return None, self._progress_confirmation(pending.values)
        if pending.kind == "progress_confirm":
            verdict = self._classify_confirmation(value)
            if verdict == "no":
                self.pending = None
                return None, "Okay, I didn't log anything."
            if verdict == "ambiguous":
                return None, self._resolve_ambiguous_confirmation(text, "log_progress", pending.values, on_event=on_event)
            self.pending = None
            return intent_router.Intent(intent_router.LOG_PROGRESS, raw=text, values=dict(pending.values)), None

        if pending.kind == "opportunity_location":
            self.pending = None
            lowered = value.lower()
            location = None if lowered in _ANY_LOCATION_WORDS else value
            return intent_router.Intent(intent_router.FIND_OPPORTUNITIES, arg=location, raw=text), None

        if pending.kind == "browser_confirm":
            self.pending = None
            verdict = self._classify_confirmation(value)
            if verdict == "ambiguous":
                # Treat anything that isn't a clean yes as a decline for a
                # live in-page action (unlike the DB-write confirmations
                # above) - resuming a stale browser task on a misread "yes"
                # risks clicking something like Send/Buy for real.
                verdict = "no"
            approved = verdict == "yes"
            return intent_router.Intent(
                intent_router.BROWSER_RESULT,
                raw=text,
                values={"answer": self._resume_browser(pending.values["paused"], approved)},
            ), None

        if pending.kind == "browser_answer":
            self.pending = None
            answer = self._value(text)
            return intent_router.Intent(
                intent_router.BROWSER_RESULT,
                raw=text,
                values={"answer": self._resume_browser_answer(pending.values["asking"], answer)},
            ), None

        if pending.kind == "browser_real_chrome_gate":
            self.pending = None
            verdict = self._classify_confirmation(value)
            # Ambiguous = decline: this gate authorizes driving the user's OWN
            # live browser (all their logins), so only a clean yes proceeds.
            approved = verdict == "yes"
            return intent_router.Intent(
                intent_router.BROWSER_RESULT,
                raw=text,
                values={"answer": self._resume_real_chrome(pending.values["gate"], approved)},
            ), None

        if pending.kind == "meeting_followup_offer":
            if self._classify_confirmation(value) != "yes":
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

    def _land_browser_result(self, result) -> str:
        """Shared tail for every browser_task resume path (confirm, answer):
        terminal results are spoken as-is; a fresh pause of EITHER kind
        re-arms `self.pending` so the task's next yes/no or open-ended
        question chains through instead of being dropped - a single
        multi-step task (e.g. an application) can hit several of these in a
        row before task_done.

        resume_browser_task()/resume_browser_task_with_answer() call
        app/browser/tools.py's _drive_loop directly and so return ITS raw
        result types (BrowserPaused/BrowserAskingUser) - NOT the
        app/brain/orchestrator.py wrapper types (BrowserConfirmationResult/
        BrowserAskUserResult) that the FIRST call into a fresh task goes
        through. Both shapes carry the same paused-loop state, just under
        different field names, so both are handled here."""
        if isinstance(result, BrowserDone):
            return result.summary
        if isinstance(result, BrowserNeedsHuman):
            return result.reason
        if isinstance(result, BrowserFailed):
            return result.reason
        if isinstance(result, BrowserPaused):
            self.pending = Pending("browser_confirm", {"paused": result})
            return result.description
        if isinstance(result, BrowserAskingUser):
            self.pending = Pending("browser_answer", {"asking": result})
            return result.question
        return "Something went wrong controlling the browser."

    def _resume_browser(self, paused, approved: bool) -> str:
        """Resume a paused browser_task after the user answered the
        confirmation prompt, returning the spoken-style result. The paused
        loop's page/tab was left untouched, so 'yes' executes exactly the one
        pending action and continues; 'no' tells the model it was declined
        and lets it wrap up instead of retrying."""
        return self._land_browser_result(resume_browser_task(paused, approved))

    def _resume_browser_answer(self, asking, answer: str) -> str:
        """Resume a browser_task that asked an open-ended question (a missing
        form/profile detail, not a yes/no) with the user's actual answer."""
        return self._land_browser_result(resume_browser_task_with_answer(asking, answer))

    def _resume_real_chrome(self, gate, approved: bool) -> str:
        """Run (approved) or abandon (declined) a real-Chrome task after the
        per-task gate. Approval runs the actual agent loop, whose result lands
        the same way as any other browser task (done / pause / ask)."""
        return self._land_browser_result(resume_real_chrome_task(gate, approved))

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

    def handle(
        self,
        text: str,
        execute: Callable[[intent_router.Intent], str],
        on_event: Callable[[dict], None] | None = None,
    ) -> str:
        response = self._handle_inner(text, execute, on_event=on_event)
        self._record_turn(text, response)
        return response

    def _handle_inner(
        self,
        text: str,
        execute: Callable[[intent_router.Intent], str],
        on_event: Callable[[dict], None] | None = None,
    ) -> str:
        if self.pending is not None:
            intent, response = self._continue(text, on_event=on_event)
            if response is not None:
                return response
            if intent is None:
                return "I didn't catch that clearly. Can you repeat it?"
            return execute(intent)

        intent = intent_router.parse(text, history=list(self.history), on_event=on_event)
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
            name = (intent.values or {}).get("name", "").strip() if intent.values else ""
            if name:
                # The orchestrator already extracted name+description in one
                # shot (see intent_router._confirmation_to_intent) - go
                # straight to confirmation instead of re-asking for the name,
                # which would otherwise capture the user's NEXT reply (their
                # actual confirmation word) as the project name instead.
                description = (intent.values or {}).get("description", "").strip()
                self.pending = Pending("project_confirm", {"name": name, "description": description, "status": "active"})
                return f"Got it — {name}: {description} Save this?" if description else f"Got it — {name}. Save this?"
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
        if intent.name == intent_router.BROWSER_CONFIRM:
            description = (intent.values or {}).get("description") or "do this"
            self.pending = Pending("browser_confirm", {"paused": intent.browser_paused})
            return f"I'm about to {description}, sir. Shall I proceed?"
        if intent.name == intent_router.BROWSER_ASK:
            question = (intent.values or {}).get("question") or "I need a bit more information - what should I do?"
            self.pending = Pending("browser_answer", {"asking": intent.browser_asking})
            return question
        if intent.name == intent_router.BROWSER_GATE:
            instruction = (intent.values or {}).get("instruction") or "that"
            self.pending = Pending("browser_real_chrome_gate", {"gate": intent.browser_gate})
            return (
                f"Heads up, sir - this will use your real Chrome, with everything you're signed "
                f"into. Shall I go ahead and {instruction}?"
            )
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "folder":
            self.pending = Pending("folder")
            return "Which folder should I open?"
        if intent.name == intent_router.CLARIFICATION_NEEDED and intent.arg == "comparison":
            self.pending = Pending("comparison", {"original": intent.raw})
            return "What two things should I compare?"
        return execute(intent)
