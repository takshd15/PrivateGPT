"""The browser agent loop: a SEPARATE, inner iterative tool-calling loop
(mirroring app/brain/orchestrator.py's pattern, over its own small schema of
physical browser actions instead of the main data-lookup tool registry).

Reached from app/brain/tools_registry.py's `browser_task` tool. Runs
independently of the outer orchestrator loop - once the outer loop selects
browser_task, THIS loop takes over turn-by-turn control until the page task
finishes, needs the user (confirmation / human-required auth), or fails.

Pause/resume: when the loop is about to run a high-impact action (per
app/browser/safety.py), it stops immediately WITHOUT executing that action,
and returns a BrowserPaused state that captures enough to resume later
(the instruction, conversation-so-far, and the pending action itself) - the
browser tab itself is left open and untouched, so resuming just executes the
one pending action and continues the loop rather than restarting the task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.browser import actions, history
from app.browser.actions import ActionError
from app.browser.manager import BrowserUnavailable, get_manager
from app.browser.safety import (
    blocked_domain_for,
    describe_high_impact_action,
    label_is_high_impact,
    page_has_auth_challenge,
)
from app.browser.state import find_element, snapshot
from app.brain.llm_client import ask_llm_message
from app.config import (
    APPLICANT_EMAIL,
    APPLICANT_LINKEDIN_URL,
    APPLICANT_NAME,
    APPLICANT_PHONE,
    APPLICANT_PORTFOLIO_URL,
    APPLICANT_RESUME_PATH,
    JARVIX_DEBUG_ORCHESTRATOR,
    MAX_BROWSER_AGENT_STEPS,
)
from app.runtime.log import log as _file_log


def _log(msg: str) -> None:
    _file_log(f"[JARVIS] {msg}")


def _debug(msg: str) -> None:
    if JARVIX_DEBUG_ORCHESTRATOR:
        _file_log(f"[browser_agent] {msg}")


@dataclass
class BrowserDone:
    summary: str


@dataclass
class BrowserNeedsHuman:
    reason: str


@dataclass
class BrowserPaused:
    """A high-impact action is ready to run but hasn't - resume_browser_task()
    picks this back up and, if approved, executes exactly this one action
    before continuing the loop."""

    description: str
    action_name: str
    action_args: dict
    messages: list[dict] = field(default_factory=list)


@dataclass
class BrowserFailed:
    reason: str


@dataclass
class BrowserAskingUser:
    """The agent needs an answer from the user to continue (a missing detail
    it has no source of truth for - NOT an auth/CAPTCHA block, that's
    BrowserNeedsHuman). resume_browser_task_with_answer() feeds the reply
    back as this tool call's result and continues the SAME loop, the same
    way BrowserPaused resumes after a yes/no."""

    question: str
    tool_call_id: str
    messages: list[dict] = field(default_factory=list)


@dataclass
class BrowserRealChromeGate:
    """Real-Chrome mode only: the per-task gate. Before the agent loop is
    allowed to touch the user's OWN Chrome (with all their live logins) for a
    task, everything pauses here and asks the user to approve THIS specific
    task. resume_real_chrome_task() runs the actual loop on approval, or
    abandons it on decline. This is the upfront wall on top of the per-action
    high-impact gate and the domain blocklist."""

    instruction: str
    timeout: int = 20


BrowserResult = (
    BrowserDone | BrowserNeedsHuman | BrowserPaused | BrowserFailed | BrowserAskingUser | BrowserRealChromeGate
)


_SYSTEM_PROMPT = """You are the browser-operating sub-system of JARVIS, a
voice assistant. You control a real Chrome browser via a small set of tools
to complete ONE task on behalf of the user. You do not talk to the user
directly - you only call tools and, when finished, call `task_done` with a
short factual summary the main assistant will speak aloud.

Rules:
- Work step by step. After every navigation or click that changes the page,
  call `read_page` again before deciding what to do next - never assume you
  know what's on the page from a previous read.
- Use the numeric [id] shown by read_page to click/type/select - those IDs
  are only valid until the next read_page call, so always read again after
  the page changes.
- If you can't find what you need with read_page's text/elements, call
  `take_screenshot` so a vision-capable pass can inspect the page visually -
  use this as a fallback, not your first move.
- If the page shows a CAPTCHA, two-factor prompt, password re-entry, or any
  other identity/security check, call `needs_human` immediately and explain
  what's blocking you. NEVER attempt to solve, guess, or click through a
  CAPTCHA or authentication challenge.
- Before any action with a real-world consequence outside just browsing -
  sending a message, submitting a form with personal/financial/legal effect,
  purchasing anything, deleting or publishing something, changing an account
  or security setting, agreeing to terms - set `high_impact: true` on that
  tool call so the user is asked to confirm first. When in doubt, mark it
  high_impact. This ALWAYS includes the final Submit/Apply/Send step of any
  job/internship application or outreach message, no exceptions, even if you
  filled the rest of the form without asking - filling is fine to do freely,
  the final submission is not.
- If an action fails, read the page again and try a sensible alternative
  (different element, scroll first, wait and retry) before giving up. If you
  are well and truly stuck, call `task_failed` with a clear reason.
- Call `task_done` as soon as the user's request is satisfied. Its summary
  should be the actual answer/result (e.g. the email's contents, the
  cheapest flight found), not a description of what you clicked.
- Never invent data you didn't actually read from the page.

Opening a site by a short spoken name (e.g. "open github", "open my bank",
just a bare site/company name with no URL and no obvious single canonical
domain): call `resolve_site_from_history` FIRST to find the site the user
actually visits most under that name, instead of guessing a domain or
searching Google for it. If it returns no match, fall back to a normal
`goto` best-guess or a search. Skip this and `goto` straight to the domain
when the user already gave you an unambiguous URL or a globally-obvious
single-domain name (e.g. "google.com").

Job/internship applications, "apply to X", messaging a lead/recruiter, or any
form asking for the user's own name/email/phone/resume/LinkedIn/portfolio:
call `get_applicant_profile` once near the start to get the user's real
contact details and resume file path, and fill form fields with that data
verbatim - never invent a name, email, phone number, or work history that
wasn't given to you. If a required field has no value in the profile (e.g. a
portfolio URL, or a free-text question like "why do you want to work here"),
draft a reasonable short answer for free-text questions using only
information you actually have (the job posting's own text, the user's
resume path/title), or call `ask_user` if you genuinely cannot proceed
without more information. Use `upload_file` with the resume path from
`get_applicant_profile` for any resume/CV upload field.
"""

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "Navigate the current tab to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Go back to the previous page in this tab's history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_forward",
            "description": "Go forward in this tab's history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reload",
            "description": "Reload the current page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the current page: title, URL, numbered interactive elements, and visible text. Call this after every navigation or click.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an interactive element by its [id] from the last read_page call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "integer"},
                    "high_impact": {"type": "boolean", "description": "True if this click has a real-world consequence (send, submit, buy, delete, publish, agree, etc)."},
                    "reason": {"type": "string", "description": "One short sentence on what this click does, used in the confirmation prompt if high_impact."},
                },
                "required": ["element_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Click an input/textarea by [id] and type text into it, replacing any existing value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "integer"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "description": "Press Enter after typing (e.g. to submit a search box)."},
                    "high_impact": {"type": "boolean", "description": "True if submitting this has a real-world consequence."},
                    "reason": {"type": "string"},
                },
                "required": ["element_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select an option in a <select> element by [id]. Value can be the visible label or the option value.",
            "parameters": {
                "type": "object",
                "properties": {"element_id": {"type": "integer"}, "value": {"type": "string"}},
                "required": ["element_id", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a keyboard key, e.g. 'Enter', 'Escape', 'PageDown'.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount_px": {"type": "integer"},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_load",
            "description": "Wait briefly for the page to finish loading/settling (e.g. after a click that triggers an async update).",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the current page for visual inspection. Use only when read_page's text/elements aren't enough to understand the page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_file",
            "description": "Upload a local file into a file-input element by [id].",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "integer"},
                    "file_path": {"type": "string"},
                    "high_impact": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["element_id", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_via_click",
            "description": "Click a download link/button by [id] and save the resulting file.",
            "parameters": {
                "type": "object",
                "properties": {"element_id": {"type": "integer"}, "save_as": {"type": "string"}},
                "required": ["element_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tabs",
            "description": "List open browser tabs.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "new_tab",
            "description": "Open a new tab, optionally navigating it to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_tab",
            "description": "Switch the active tab by its index from list_tabs.",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_tab",
            "description": "Close a tab by index (defaults to the current tab).",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_site_from_history",
            "description": "Find the URL the user actually visits most under a short spoken/typed name (e.g. 'github', 'my bank', 'that internship board'), ranked by their real Chrome browsing history (visit count + recency). Use this before goto when the user named a site rather than giving a URL.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The short name the user said, e.g. 'github' or 'linkedin'."}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_applicant_profile",
            "description": "Get the user's real name, email, phone, resume file path, LinkedIn, and portfolio URL for filling in application/contact forms. Fields not on file come back empty - never invent a value for one of those, ask the user instead via ask_user.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a short question when you're missing information you have no other source for (not a CAPTCHA/2FA - use needs_human for those) and can't reasonably proceed without it, e.g. a form field with no sensible default and nothing in get_applicant_profile. Use sparingly - prefer a reasonable default over interrupting when one exists.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "needs_human",
            "description": "Call this when the page requires a CAPTCHA, 2FA, password re-entry, biometric check, or other human-only verification. Never try to solve these yourself.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Call when the user's request has been fully completed. Give the actual result, not a description of your clicks.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_failed",
            "description": "Call if the task genuinely can't be completed after reasonable attempts.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

# Actions with an optional high_impact flag - these are the ones the
# confirmation gate can intercept.
_GATEABLE_ACTIONS = {"click", "type_text", "upload_file"}


def _truncate(value: Any, max_chars: int = 1500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"


def _target_label(page, args: dict) -> str:
    element_id = args.get("element_id")
    if element_id is None:
        return ""
    try:
        locator = find_element(page, element_id)
        if locator is None:
            return ""
        return (locator.inner_text(timeout=1000) or locator.get_attribute("value") or "").strip()[:80]
    except Exception:
        return ""


def _execute_action(name: str, args: dict, on_event: Callable[[dict], None] | None) -> str:
    """Run one browser action, returning its result text or raising ActionError."""
    if name == "goto":
        return actions.goto(args.get("url", ""))
    if name == "go_back":
        return actions.go_back()
    if name == "go_forward":
        return actions.go_forward()
    if name == "reload":
        return actions.reload()
    if name == "click":
        return actions.click(int(args["element_id"]))
    if name == "type_text":
        return actions.type_text(int(args["element_id"]), args.get("text", ""), submit=bool(args.get("submit")))
    if name == "select_option":
        return actions.select_option(int(args["element_id"]), args.get("value", ""))
    if name == "press_key":
        return actions.press_key(args.get("key", "Enter"))
    if name == "scroll":
        return actions.scroll(args.get("direction", "down"), int(args.get("amount_px", 800)))
    if name == "wait_for_load":
        return actions.wait_for_load(float(args.get("seconds", 3.0)))
    if name == "upload_file":
        return actions.upload_file(int(args["element_id"]), args.get("file_path", ""))
    if name == "download_via_click":
        return actions.download_via_click(int(args["element_id"]), args.get("save_as"))
    if name == "list_tabs":
        return json.dumps(actions.list_tabs())
    if name == "new_tab":
        return actions.new_tab(args.get("url"))
    if name == "switch_tab":
        return actions.switch_tab(int(args["index"]))
    if name == "close_tab":
        idx = args.get("index")
        return actions.close_tab(int(idx) if idx is not None else None)
    raise ActionError(f"Unknown browser action: {name}")


def _take_screenshot_result() -> str:
    path = actions.screenshot()
    return f"Screenshot saved to {path}. (Vision inspection of screenshots is not wired up in this build - rely on read_page's text/elements instead.)"


def _would_use_real_chrome() -> bool:
    """Whether THIS task would actually end up driving the user's real
    Chrome, without starting anything - used to decide whether to show the
    per-task gate. Mirrors app/browser/manager.py's own _start_impl
    dispatch logic exactly (mode "dedicated" -> never, "real" -> always,
    "auto" -> only if the cheap CDP probe succeeds right now), so the gate
    only ever appears when real Chrome will genuinely be used - "auto" mode
    stays silent and un-gated on machines that never launch Chrome with
    --remote-debugging-port, which is the common case."""
    from app.browser.manager import _probe_cdp_available
    from app.config import (
        JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS,
        JARVIX_BROWSER_CDP_URL,
        JARVIX_BROWSER_MODE,
    )

    manager = get_manager()
    if manager.is_running:
        # Already decided for this process - don't re-probe or gate again
        # every single task once a browser is up and running.
        return manager.is_real_chrome
    if JARVIX_BROWSER_MODE == "dedicated":
        return False
    if JARVIX_BROWSER_MODE == "real":
        return True
    return _probe_cdp_available(JARVIX_BROWSER_CDP_URL, JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS)


def run_browser_task(
    instruction: str,
    on_event: Callable[[dict], None] | None = None,
    timeout: int = 20,
) -> BrowserResult:
    """Entry point from app/brain/tools_registry.py's browser_task tool.

    If this task would drive the user's REAL Chrome (JARVIX_BROWSER_MODE
    "real", or "auto" when a real Chrome is actually reachable), it does NOT
    run yet - it returns a BrowserRealChromeGate so the user approves driving
    their own live browser for THIS task first. resume_real_chrome_task runs
    the loop only after they say yes. In the default dedicated-profile case
    there's nothing to gate, so the loop starts immediately."""
    if _would_use_real_chrome():
        _log(f"Real-Chrome task awaiting approval: {instruction}")
        if on_event:
            try:
                on_event({"type": "browser_real_chrome_gate", "instruction": instruction})
            except Exception:
                pass
        return BrowserRealChromeGate(instruction=instruction, timeout=timeout)

    return _begin_browser_task(instruction, on_event=on_event, timeout=timeout)


def resume_real_chrome_task(
    gate: BrowserRealChromeGate,
    approved: bool,
    on_event: Callable[[dict], None] | None = None,
) -> BrowserResult:
    """Run (or abandon) a real-Chrome task after the user answered the
    per-task gate. On decline nothing touches their browser at all."""
    if not approved:
        _log(f"Real-Chrome task declined: {gate.instruction}")
        return BrowserFailed("Understood, sir - I left your browser alone.")
    _log(f"Real-Chrome task approved: {gate.instruction}")
    return _begin_browser_task(gate.instruction, on_event=on_event, timeout=gate.timeout)


def _begin_browser_task(
    instruction: str,
    on_event: Callable[[dict], None] | None = None,
    timeout: int = 20,
) -> BrowserResult:
    """Start the browser and drive the agent loop for one task - the shared
    tail of both run_browser_task (dedicated mode) and resume_real_chrome_task
    (real-Chrome mode, post-approval)."""
    try:
        get_manager().ensure_started()
    except BrowserUnavailable as exc:
        return BrowserFailed(str(exc))

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {instruction}"},
    ]
    _log(f"Task started: {instruction}")
    return _drive_loop(messages, timeout=timeout, on_event=on_event)


def resume_browser_task(
    paused: BrowserPaused,
    approved: bool,
    on_event: Callable[[dict], None] | None = None,
    timeout: int = 20,
) -> BrowserResult:
    """Continue a paused task after the user answered the confirmation
    prompt. On approval, executes exactly the one pending action (the page
    was left untouched while paused) and resumes the loop from there; on
    decline, tells the model the action was declined and lets it decide how
    to wrap up (usually task_done/task_failed) rather than silently exiting."""
    messages = list(paused.messages)
    if approved:
        _log(f"Confirmed: {paused.description}")
        try:
            result_text = _execute_action(paused.action_name, paused.action_args, on_event)
            messages.append(
                {"role": "tool", "tool_call_id": paused.action_args.get("_tool_call_id", ""), "content": _truncate(result_text)}
            )
        except ActionError as exc:
            messages.append(
                {"role": "tool", "tool_call_id": paused.action_args.get("_tool_call_id", ""), "content": _truncate(f"error: {exc}")}
            )
    else:
        _log(f"Declined: {paused.description}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": paused.action_args.get("_tool_call_id", ""),
                "content": "The user declined this action. Do not retry it. Either find another way to help, or call task_failed.",
            }
        )
    return _drive_loop(messages, timeout=timeout, on_event=on_event)


def resume_browser_task_with_answer(
    asking: BrowserAskingUser,
    answer: str,
    on_event: Callable[[dict], None] | None = None,
    timeout: int = 20,
) -> BrowserResult:
    """Continue a task after the user answered an ask_user question, feeding
    the reply back as that tool call's result - same resume pattern as
    resume_browser_task, just for a plain info gap instead of a high-impact
    action needing yes/no."""
    messages = list(asking.messages)
    _log(f"User answered: {answer}")
    messages.append({"role": "tool", "tool_call_id": asking.tool_call_id, "content": _truncate(answer)})
    return _drive_loop(messages, timeout=timeout, on_event=on_event)


def _drive_loop(
    messages: list[dict], timeout: int, on_event: Callable[[dict], None] | None
) -> BrowserResult:
    for step in range(MAX_BROWSER_AGENT_STEPS):
        if on_event:
            try:
                on_event({"type": "browser_thinking"})
            except Exception:
                pass
        try:
            message = ask_llm_message(messages, tools=_TOOLS, tool_choice="auto", timeout=timeout, num_predict=200)
        except Exception as exc:
            _debug(f"llm call failed: {exc}")
            return BrowserFailed(f"Lost contact with the assistant service: {exc}")

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = (message.get("content") or "").strip()
            return BrowserFailed(content or "The browser agent stopped without finishing the task.")

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})

        # Only handle the first tool call per turn - browser actions are
        # inherently sequential (each depends on the last one's page state),
        # unlike the outer orchestrator's read-only data lookups which are
        # safe to batch.
        call = tool_calls[0]
        for extra_call in tool_calls[1:]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": extra_call.get("id", ""),
                    "content": "skipped: only one browser action runs per turn, act on this result first",
                }
            )

        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_call_id = call.get("id", "")

        _debug(f"action={name} args={args}")

        if name == "task_done":
            summary = args.get("summary") or "Done."
            _log(f"Task completed: {summary}")
            if on_event:
                try:
                    on_event({"type": "browser_done", "summary": summary})
                except Exception:
                    pass
            return BrowserDone(summary)

        if name == "task_failed":
            reason = args.get("reason") or "The task could not be completed."
            _log(f"Task failed: {reason}")
            return BrowserFailed(reason)

        if name == "needs_human":
            reason = args.get("reason") or "This page needs you to verify something (CAPTCHA, 2FA, or similar)."
            _log(f"Needs human: {reason}")
            return BrowserNeedsHuman(reason)

        if name == "ask_user":
            question = args.get("question") or "I need a bit more information to continue - what should I do?"
            _log(f"Asking user: {question}")
            if on_event:
                try:
                    on_event({"type": "browser_ask_user", "question": question})
                except Exception:
                    pass
            return BrowserAskingUser(question=question, tool_call_id=tool_call_id, messages=messages)

        if name == "resolve_site_from_history":
            site_name = args.get("name") or ""
            try:
                match = history.best_site_url(site_name)
            except Exception as exc:
                _debug(f"history lookup failed: {exc}")
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": f"error: couldn't read browsing history: {exc}"}
                )
                continue
            if match is None:
                result_text = f"No match found in browsing history for '{site_name}'. Use your best guess for the domain, or search for it."
            else:
                result_text = json.dumps(
                    {"url": history.canonical_url(match), "visits": match.visit_count, "matched_title": match.title}
                )
            _log(f"Resolved '{site_name}' from history: {result_text}")
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _truncate(result_text)})
            continue

        if name == "get_applicant_profile":
            profile = {
                "name": APPLICANT_NAME,
                "email": APPLICANT_EMAIL,
                "phone": APPLICANT_PHONE,
                "resume_path": APPLICANT_RESUME_PATH,
                "linkedin_url": APPLICANT_LINKEDIN_URL,
                "portfolio_url": APPLICANT_PORTFOLIO_URL,
            }
            missing = [k for k, v in profile.items() if not v]
            if missing:
                profile["_missing"] = missing
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _truncate(json.dumps(profile))})
            continue

        if name == "read_page":
            try:
                manager = get_manager()
                snap = manager.run(lambda: snapshot(manager.page))
            except Exception as exc:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": f"error reading page: {exc}"})
                continue
            _log(f"Reading page: {snap.title} ({snap.url})")
            # Blocklist catch for real-Chrome mode: a click (not just goto)
            # can land on a protected site. If we're now on one, stop the task
            # rather than read/act on it.
            if get_manager().is_real_chrome:
                blocked = blocked_domain_for(snap.url)
                if blocked:
                    _log(f"Stopping: landed on blocked site {snap.url} ({blocked})")
                    return BrowserFailed(
                        f"That led to a protected site ({blocked}), sir - I don't operate on banking, "
                        "brokerage, or password-manager pages in your real Chrome. Stopping there."
                    )
            if page_has_auth_challenge(snap.text, snap.title):
                return BrowserNeedsHuman(
                    f"The page '{snap.title}' appears to need verification (CAPTCHA/2FA/security check). Please handle it, then tell me to continue."
                )
            if on_event:
                try:
                    on_event({"type": "browser_read", "title": snap.title, "url": snap.url})
                except Exception:
                    pass
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _truncate(snap.as_prompt_text())})
            continue

        if name == "take_screenshot":
            try:
                result_text = _take_screenshot_result()
            except ActionError as exc:
                result_text = f"error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _truncate(result_text)})
            continue

        # Gateable physical actions: decide whether this needs confirmation
        # BEFORE touching the page.
        if name in _GATEABLE_ACTIONS:
            try:
                manager = get_manager()
                label = manager.run(lambda: _target_label(manager.page, args))
            except Exception:
                label = ""
            flagged = bool(args.get("high_impact"))
            if flagged or label_is_high_impact(label) or label_is_high_impact(args.get("text", "")):
                description = describe_high_impact_action(
                    name, label or args.get("text", ""), {"reason": args.get("reason", "")}
                )
                _log(f"Pausing for confirmation: {description}")
                args_with_call_id = dict(args)
                args_with_call_id["_tool_call_id"] = tool_call_id
                if on_event:
                    try:
                        on_event({"type": "browser_confirmation", "description": description})
                    except Exception:
                        pass
                return BrowserPaused(
                    description=description,
                    action_name=name,
                    action_args=args_with_call_id,
                    messages=messages,
                )

        try:
            result_text = _execute_action(name, args, on_event)
            if on_event:
                try:
                    on_event({"type": "browser_action", "name": name, "result": result_text})
                except Exception:
                    pass
        except ActionError as exc:
            _debug(f"action failed: {exc}")
            result_text = f"error: {exc}"

        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _truncate(result_text)})

    _log("Max browser agent steps reached")
    return BrowserFailed("I've taken a lot of steps without finishing - stopping to avoid getting stuck in a loop.")
