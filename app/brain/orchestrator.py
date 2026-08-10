"""LLM-first orchestrator: iterative tool-calling loop over app/brain/tools_registry.py.

This is the semantic router/planner. It decides WHAT the user means and WHICH
tools answer it, calling more than one in sequence when the first result
isn't enough. All deterministic execution, DB writes, and safety boundaries
stay where they already lived (app/tools/*, app/memory/db/*, app/safety/
permissions.py, app/brain/dialogue.py's confirmation states) - this module
only reads data and proposes actions; it never performs a mutating action
itself (see ToolSpec.requires_confirmation in tools_registry.py).

Reached from two places:
- app/brain/intent_router.py, as the fallback when the deterministic rules in
  _parse_rules() return UNKNOWN.
- app/brain/dialogue.py, when a pending confirmation reply is ambiguous
  (neither a clean yes nor a clean no).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.brain.llm_client import ask_llm_message
from app.brain.tools_registry import get as get_tool, openai_schemas
from app.browser.tools import (
    BrowserAskingUser,
    BrowserDone,
    BrowserFailed,
    BrowserNeedsHuman,
    BrowserPaused,
    BrowserRealChromeGate,
    run_browser_task,
)
from app.config import JARVIX_DEBUG_ORCHESTRATOR
from app.runtime.log import log as _file_log

MAX_TOOL_CALLS = 6
_TOOL_TIMEOUT_MESSAGE = "the request timed out"


@dataclass
class Turn:
    """One user/assistant exchange, for the rolling conversation window kept
    by app/brain/dialogue.py's VoiceDialogue.history (maxlen=5)."""

    role: Literal["user", "assistant"]
    text: str


@dataclass
class AnswerResult:
    text: str


@dataclass
class ClarificationResult:
    question: str


@dataclass
class ConfirmationResult:
    """A mutating tool was selected but not executed - the caller (dialogue.py
    /intent_router.py) must route this through the EXISTING pending-
    confirmation state machine, not a new one."""

    tool_name: str
    arguments: dict


@dataclass
class BrowserResultWrapper:
    """The browser_task tool ran to completion (or a dead end) without
    needing a mid-task confirmation - a spoken-style result either way
    (finished, failed, or needs human help with something like a CAPTCHA)."""

    text: str


@dataclass
class BrowserConfirmationResult:
    """The browser agent loop paused right before a high-impact action (see
    app/browser/safety.py) - the browser tab is left open and untouched.
    Carries the paused loop state so dialogue.py's browser_confirm Pending
    state can resume the SAME task instead of restarting it, once the user
    answers yes/no."""

    description: str
    paused: BrowserPaused


@dataclass
class BrowserAskUserResult:
    """The browser agent loop needs an answer to a genuine information gap
    (not an auth/CAPTCHA block - that's a plain BrowserResultWrapper via
    BrowserNeedsHuman) to keep going, e.g. filling in an application form
    field with nothing in the applicant profile. Carries the loop state so
    dialogue.py's browser_answer Pending state can resume the SAME task with
    whatever the user says next, instead of restarting it."""

    question: str
    asking: BrowserAskingUser


@dataclass
class BrowserRealChromeGateResult:
    """Real-Chrome mode only: the agent selected browser_task but hasn't
    touched anything yet - the user must first approve driving their OWN live
    Chrome (with all their logins) for this specific task. dialogue.py's
    browser_real_chrome_gate Pending state runs the task on 'yes' and abandons
    it on 'no'."""

    instruction: str
    gate: BrowserRealChromeGate


_SYSTEM_PROMPT = """You are JARVIS - Tony Stark's JARVIS, reimagined as Jarvix, a voice
assistant. Adopt his personality fully in how you phrase every reply: an
impeccably polite, dry-witted British butler-engineer. Calm, unflappable,
quietly confident, and a little sardonic. You address the user respectfully
(e.g. "sir") without overdoing it, understate rather than gush, and allow
yourself the occasional deadpan or wry aside - never slapstick, never
rambling. Under it all you are precise, competent, and genuinely looking
out for the user's interests, the way JARVIS looked out for Stark.

Your job: understand what the user actually wants, then use the available
tools to gather whatever information you need before answering. You are not
limited to one tool call - if the first result isn't enough, call another
tool. Use the recent conversation turns to resolve references like "the best
one", "that project", or "what we just found" to what was actually discussed.

Rules:
- Prefer calling a tool over asking the user for information a tool can find
  (goals, projects, tasks, opportunities, applications, people, meetings,
  memories, calendar, email, weather, news are all queryable).
- If the user is describing something to remember, save, schedule, or log
  (a task, a new project, a progress update, a calendar event, an email to
  send), call the matching mutating tool (remember_task, create_project,
  log_progress, create_calendar_event, send_email) even if some details are
  still missing - pass whatever you already have. A follow-up conversation
  will collect any missing detail; do not try to ask for it yourself in a
  plain-text answer, and do not skip calling the tool just because the
  request is short or vague.
- Only ask a clarifying question (via ask_clarifying_question) when the
  request is genuinely ambiguous about WHICH action/entity is meant, even
  after checking tools and conversation context - and when you do, make the
  question specific to what you already know (e.g. name the options), never
  a generic "I didn't catch that." Stay in character while you ask it.
- Tools marked as requiring confirmation (create_calendar_event, send_email,
  remember_task, create_project, log_progress) mutate data - selecting one of
  those ends your turn immediately; you do not receive its result because it
  has not run yet, so don't call any other tool in the same turn as one of these.
- You DO have a real browser at your disposal via browser_task - you can
  navigate, search, click, type, log in, fill out forms, and interact with
  any website exactly like a person would. Never tell the user you can't
  browse, open a site, click something, subscribe, apply, or otherwise
  interact with a webpage - call browser_task instead of claiming that
  limitation. This includes "open <name>" for a site named informally rather
  than by URL, applying to jobs/internships, and messaging people through a
  website. browser_task pauses on its own for anything high-impact (sending,
  buying, submitting, subscribing, deleting, publishing, changing account/
  security settings, etc), so it's always safe to call - you don't need to
  ask permission yourself before calling it.
- If the user asks to see/open/show something "in the browser" that a data
  tool you already called this turn (get_opportunities, get_applications,
  get_tasks, search_opportunities, web_search, etc.) actually returned real
  URLs for, put those SPECIFIC URLs in browser_task's instruction (e.g. "open
  https://techstars.com/... in a new tab") instead of writing a generic
  instruction that makes browser_task search the web again from scratch -
  the user wants the thing you just found, not a fresh, possibly different
  search. Only fall back to a generic search instruction when the data tool
  genuinely returned nothing usable (no URL) for what they're asking about.
- When the user's request has a SECOND half after the site/navigation part
  ("open X and tell me Y", "open X and check whether Z", "open X and click
  on Y", "go to X and read me the top result"), browser_task's instruction
  must carry that second half verbatim - never collapse it down to just
  "open X". The browser agent only knows what's in the instruction text; if
  you drop the "and tell me Y" part, it stops after loading the page and the
  user never gets an answer to what they actually asked for.
- HARD LIMIT: every final answer is at most 2 short spoken sentences (about
  40 words total), unless the user's own wording explicitly asked for detail
  (e.g. "list them all", "give me details", "tell me more", "the full
  list", "explain"). This is not a soft guideline - a 4-sentence answer is a
  failure even if each sentence is short. No markdown, no bullet points, no
  numbered lists, no links, no code blocks, ever - this is spoken aloud.
  Wit and warmth come through in word choice and tone, not length.
- Default to a verdict, not a report. When a tool returns several items
  (search results, opportunities, emails, schools, headlines, ways you could
  help), pick the ONE best/most relevant one and say only that, plus at most
  a half-sentence reason. Do not name a second or third option "for
  completeness," and do not summarize the full category of things you could
  do - name the one thing, then stop and ask if they want more. E.g. "The
  Doon School's your best bet, sir - it's the standout. Want the rest of the
  list?" not a survey of five schools; "I'd start by drafting the business
  plan, sir. Shall I set that up?" not a numbered menu of every capability.
- This limit applies no matter how many tools you called or how much
  research you did - more digging should make the one sentence you give
  back better-informed, not longer. If you're tempted to write "several
  ways" or "a few options," stop and pick just one instead.
- If a tool fails, don't just say you didn't understand - explain briefly
  what went wrong, in character, or try a different approach if one is
  available.
"""

_CLARIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_clarifying_question",
        "description": "Use this ONLY when the request is genuinely ambiguous and no tool call would resolve it. The question must be specific to what you already know from context.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
}


def _debug(msg: str) -> None:
    if JARVIX_DEBUG_ORCHESTRATOR:
        _file_log(f"[orchestrator] {msg}")


def _emit(on_event: Callable[[dict], None] | None, event: dict) -> None:
    """Fire an optional caller-supplied event callback (e.g. app/server.py's
    SSE bridge). Never lets a broken callback break orchestration - same
    never-raise contract this whole module already holds itself to."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        pass


def _truncate(value: Any, max_chars: int = 1200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"


def _capped_result(value: Any, max_chars: int = 1200) -> Any:
    """A JSON-safe version of a tool result, size-capped so one oversized
    result can't blow up an SSE frame for callers like app/server.py that
    need the actual structured data (not just the log-friendly summary
    string _truncate produces). Falls back to the truncated string itself if
    the value doesn't round-trip through JSON cleanly."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return _truncate(value, max_chars)
    if len(text) <= max_chars:
        return json.loads(text)
    return _truncate(value, max_chars)


def _build_messages(text: str, history: list[Turn], pending_note: str | None) -> list[dict]:
    system = _SYSTEM_PROMPT
    if pending_note:
        system += f"\n\nContext: {pending_note}"
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.text})
    messages.append({"role": "user", "content": text})
    return messages


def _run_tool(name: str, arguments: dict) -> Any:
    spec = get_tool(name)
    if spec is None:
        return {"error": f"unknown tool: {name}"}
    if spec.handler is None:
        return {"error": f"{name} cannot be executed directly here"}
    try:
        return spec.handler(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": f"{name} failed: {exc}"}


def orchestrate(
    text: str,
    history: list[Turn] | None = None,
    pending_note: str | None = None,
    timeout: int = 15,
    on_event: Callable[[dict], None] | None = None,
) -> (
    AnswerResult
    | ClarificationResult
    | ConfirmationResult
    | BrowserResultWrapper
    | BrowserConfirmationResult
    | BrowserAskUserResult
    | BrowserRealChromeGateResult
):
    """Run the iterative tool-calling loop for one user request. Never raises -
    any failure becomes an AnswerResult explaining what went wrong.

    ``on_event`` is an optional callback fired at each decision point
    (thinking / tool_call / tool_result / clarification / confirmation /
    answer), so a caller like app/server.py can stream live progress to a
    frontend. Purely additive - omitting it changes nothing about how this
    function behaves, and every event is also still mirrored into jarvix.log
    via _debug() when JARVIX_DEBUG_ORCHESTRATOR is set."""
    history = history or []
    messages = _build_messages(text, history, pending_note)
    tools = openai_schemas() + [_CLARIFY_TOOL]
    seen_calls: set[tuple[str, str]] = set()

    # Applies to EITHER a tool_call or a direct spoken answer, since OpenAI's
    # max_tokens caps the whole response either way - tool_call arguments are
    # a handful of short JSON keys (plenty of room below 100 tokens), and
    # this doubles as the same-turn answer cap so a request the model can
    # answer without any tool still lands within the ~40-word spoken limit.
    _TOOL_TURN_TOKENS = 100
    _FINAL_ANSWER_TOKENS = 80

    _debug(f"start text={text!r} history_turns={len(history)}")

    for step in range(MAX_TOOL_CALLS):
        _emit(on_event, {"type": "thinking"})
        try:
            message = ask_llm_message(
                messages, tools=tools, tool_choice="auto", timeout=timeout, num_predict=_TOOL_TURN_TOKENS
            )
        except Exception as exc:
            _debug(f"llm call failed: {exc}")
            fallback = "Sorry, I couldn't reach the assistant service right now."
            _emit(on_event, {"type": "answer", "text": fallback})
            return AnswerResult(fallback)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = (message.get("content") or "").strip()
            _debug(f"final answer after {step} tool call(s)")
            reply = content or "I don't have a good answer for that right now."
            _emit(on_event, {"type": "answer", "text": reply})
            return AnswerResult(reply)

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            }
        )

        # A model can request several calls in one turn; handle each, but a
        # confirmation-required tool always ends the whole request immediately
        # (its result doesn't exist yet - there's nothing further to feed back).
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if name == "ask_clarifying_question":
                question = arguments.get("question") or "Could you clarify what you mean?"
                _debug(f"clarification: {question!r}")
                _emit(on_event, {"type": "clarification", "question": question})
                return ClarificationResult(question)

            if name == "browser_task":
                instruction = arguments.get("instruction") or text
                _debug(f"browser_task instruction={instruction!r}")
                _emit(on_event, {"type": "browser_task_start", "instruction": instruction})
                result = run_browser_task(instruction, on_event=on_event, timeout=timeout)
                if isinstance(result, BrowserDone):
                    _emit(on_event, {"type": "answer", "text": result.summary})
                    return BrowserResultWrapper(result.summary)
                if isinstance(result, BrowserNeedsHuman):
                    _emit(on_event, {"type": "answer", "text": result.reason})
                    return BrowserResultWrapper(result.reason)
                if isinstance(result, BrowserFailed):
                    _emit(on_event, {"type": "answer", "text": result.reason})
                    return BrowserResultWrapper(result.reason)
                # BrowserPaused/BrowserAskingUser already emitted their own
                # browser_confirmation/browser_ask_user event from inside
                # _drive_loop at the moment they paused - re-emitting here
                # would duplicate it, so this just wraps the result.
                if isinstance(result, BrowserPaused):
                    return BrowserConfirmationResult(result.description, result)
                if isinstance(result, BrowserAskingUser):
                    return BrowserAskUserResult(result.question, result)
                # Real-Chrome mode: the per-task gate fired before anything ran
                # (its own browser_real_chrome_gate event was already emitted
                # inside run_browser_task).
                if isinstance(result, BrowserRealChromeGate):
                    return BrowserRealChromeGateResult(result.instruction, result)
                return BrowserResultWrapper("Something went wrong controlling the browser.")

            spec = get_tool(name)
            if spec is not None and spec.requires_confirmation:
                _debug(f"confirmation required for tool={name} args={arguments}")
                _emit(on_event, {"type": "confirmation", "tool": name, "arguments": arguments})
                return ConfirmationResult(name, arguments)

            call_key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if call_key in seen_calls:
                _debug(f"loop guard: repeated call to {name}, forcing synthesis")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps({"error": "already called with these arguments"}),
                    }
                )
                continue
            seen_calls.add(call_key)

            _debug(f"tool_call name={name} args={arguments}")
            _emit(on_event, {"type": "tool_call", "name": name, "arguments": arguments})
            result = _run_tool(name, arguments)
            summary = _truncate(result, 200)
            _debug(f"tool_result name={name} summary={summary}")
            _emit(
                on_event,
                {"type": "tool_result", "name": name, "summary": summary, "result": _capped_result(result)},
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": _truncate(result),
                }
            )

    # Loop-safety cutoff hit: ask once more with no tools so the model must
    # synthesize a final answer from whatever's already in the transcript.
    _debug(f"max tool calls ({MAX_TOOL_CALLS}) reached, forcing synthesis")
    _emit(on_event, {"type": "thinking"})
    try:
        message = ask_llm_message(messages, tools=None, timeout=timeout, num_predict=_FINAL_ANSWER_TOKENS)
        content = (message.get("content") or "").strip()
        reply = content or "I gathered some information but couldn't finish putting it together."
    except Exception:
        reply = "I gathered some information but couldn't finish putting it together."
    _emit(on_event, {"type": "answer", "text": reply})
    return AnswerResult(reply)
