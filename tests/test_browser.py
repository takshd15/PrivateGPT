"""Tests for the browser-control system (app/browser/*) and its wiring into
the orchestrator/intent_router/dialogue confirmation flow. All LLM calls are
mocked via app.browser.tools.ask_llm_message (same pattern as
tests/test_orchestrator.py); Playwright itself is mocked out entirely so
these run offline with no real Chrome."""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from app.brain import intent_router
from app.brain.dialogue import Pending, VoiceDialogue
from app.brain.orchestrator import (
    BrowserAskUserResult,
    BrowserConfirmationResult,
    BrowserRealChromeGateResult,
    BrowserResultWrapper,
    orchestrate,
)
from app.browser import history, safety
from app.browser.tools import (
    BrowserAskingUser,
    BrowserDone,
    BrowserFailed,
    BrowserNeedsHuman,
    BrowserPaused,
    BrowserRealChromeGate,
    resume_browser_task_with_answer,
    resume_real_chrome_task,
    run_browser_task,
)


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _assistant_message(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


# Most of this file tests the INNER agent loop, not the real-Chrome per-task
# gate - those tests never configure is_running/is_real_chrome on their
# MagicMock() manager, so app/browser/tools.py's _would_use_real_chrome()
# would otherwise see truthy-by-default mock attributes and wrongly show a
# gate. Force "dedicated" mode for the whole module so run_browser_task always
# runs the loop immediately by default; the real-Chrome-specific test classes
# below override this per-test via their own `patch("app.config.JARVIX_BROWSER_MODE", ...)`.
_mode_patch = patch("app.config.JARVIX_BROWSER_MODE", "dedicated")


def setUpModule():
    _mode_patch.start()


def tearDownModule():
    _mode_patch.stop()


class SafetyClassifierTests(unittest.TestCase):
    """The keyword backstop that catches high-impact actions even when the
    model's own tool call didn't set high_impact=true."""

    def test_send_button_label_is_high_impact(self):
        self.assertTrue(safety.label_is_high_impact("Send"))

    def test_buy_now_label_is_high_impact(self):
        self.assertTrue(safety.label_is_high_impact("Buy Now"))

    def test_delete_account_label_is_high_impact(self):
        self.assertTrue(safety.label_is_high_impact("Delete Account"))

    def test_plain_navigation_label_is_not_high_impact(self):
        self.assertFalse(safety.label_is_high_impact("Search"))
        self.assertFalse(safety.label_is_high_impact("Inbox"))
        self.assertFalse(safety.label_is_high_impact("Next page"))

    def test_captcha_page_text_is_detected_as_auth_challenge(self):
        self.assertTrue(safety.page_has_auth_challenge("Please verify you're a human before continuing."))

    def test_two_factor_prompt_is_detected(self):
        self.assertTrue(safety.page_has_auth_challenge("", page_title="Two-factor verification required"))

    def test_ordinary_page_is_not_an_auth_challenge(self):
        self.assertFalse(safety.page_has_auth_challenge("Your latest emails are listed below.", page_title="Inbox"))


class _FakeManagerContext:
    """Patches app.browser.tools.get_manager()/find_element()/snapshot() so
    the agent loop can run without any real Playwright objects. Individual
    tests configure `.page_text_by_call` etc. as needed via patch targets."""


class BrowserAgentLoopTests(unittest.TestCase):
    """The inner browser agent loop's own tool-calling cycle (goto/read_page/
    click/...), independent of the outer orchestrator."""

    def setUp(self):
        self.manager_patch = patch("app.browser.tools.get_manager")
        mock_get_manager = self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.mock_manager = MagicMock()
        mock_get_manager.return_value = self.mock_manager
        self.mock_manager.page = MagicMock()
        # is_running must be explicitly False, not a truthy-by-default
        # MagicMock - _would_use_real_chrome() checks this FIRST, before the
        # module's "dedicated" mode patch even matters, and a stray truthy
        # mock here makes every task wrongly show the real-Chrome gate.
        self.mock_manager.is_running = False
        # Real BrowserManager.run() marshals the callable onto the browser
        # thread and returns its result; the mock must actually call it too,
        # or every manager.run(lambda: ...) call site in tools.py silently
        # gets a MagicMock back instead of the real return value.
        self.mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

    def test_task_done_returns_browser_done(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "Found it."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses):
            result = run_browser_task("do something")
        self.assertIsInstance(result, BrowserDone)
        self.assertEqual(result.summary, "Found it.")

    def test_needs_human_returns_browser_needs_human(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "needs_human", '{"reason": "CAPTCHA on screen."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses):
            result = run_browser_task("log into example.com")
        self.assertIsInstance(result, BrowserNeedsHuman)
        self.assertIn("CAPTCHA", result.reason)

    def test_task_failed_returns_browser_failed(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "task_failed", '{"reason": "Could not find the button."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses):
            result = run_browser_task("click a button that does not exist")
        self.assertIsInstance(result, BrowserFailed)

    def test_no_tool_call_at_all_is_treated_as_failure_not_crash(self):
        responses = [_assistant_message(content="I'm not sure what to do.")]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses):
            result = run_browser_task("do something vague")
        self.assertIsInstance(result, BrowserFailed)

    def test_read_page_feeds_snapshot_back_and_loop_continues(self):
        from app.browser.state import ElementRef, PageSnapshot

        snap = PageSnapshot(title="Inbox", url="https://mail.example.com", elements=[ElementRef(0, "link", "Compose", False)], text="Welcome", truncated=False)
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "read_page", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_done", '{"summary": "Read the inbox."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools.snapshot", return_value=snap
        ), patch("app.browser.tools.page_has_auth_challenge", return_value=False):
            result = run_browser_task("read my inbox")
        self.assertIsInstance(result, BrowserDone)

    def test_captcha_detected_during_read_page_pauses_for_human(self):
        from app.browser.state import PageSnapshot

        snap = PageSnapshot(title="Verify you're human", url="https://example.com", elements=[], text="captcha challenge", truncated=False)
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "read_page", "{}")])]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools.snapshot", return_value=snap
        ):
            result = run_browser_task("log in")
        self.assertIsInstance(result, BrowserNeedsHuman)

    def test_flagged_high_impact_click_pauses_instead_of_executing(self):
        responses = [
            _assistant_message(
                tool_calls=[
                    _tool_call(
                        "c1", "click", '{"element_id": 3, "high_impact": true, "reason": "sends the email"}'
                    )
                ]
            ),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools._target_label", return_value="Send"
        ), patch("app.browser.actions.click") as mock_click:
            result = run_browser_task("send the email")
        self.assertIsInstance(result, BrowserPaused)
        self.assertEqual(result.action_name, "click")
        mock_click.assert_not_called()  # the action must NOT run before confirmation

    def test_unflagged_but_dangerous_label_is_still_caught_by_keyword_backstop(self):
        """Even if the model forgets to set high_impact, a label like 'Delete'
        must still pause - this is the safety net, not just the model's word."""
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "click", '{"element_id": 7}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools._target_label", return_value="Delete Account"
        ), patch("app.browser.actions.click") as mock_click:
            result = run_browser_task("clean up my account")
        self.assertIsInstance(result, BrowserPaused)
        mock_click.assert_not_called()

    def test_ordinary_click_executes_without_pausing(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "click", '{"element_id": 2}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_done", '{"summary": "Clicked search."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools._target_label", return_value="Search"
        ), patch("app.browser.actions.click", return_value="Clicked [2]") as mock_click:
            result = run_browser_task("search for something")
        self.assertIsInstance(result, BrowserDone)
        mock_click.assert_called_once_with(2)

    def test_action_error_is_fed_back_to_model_instead_of_crashing(self):
        from app.browser.actions import ActionError

        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "click", '{"element_id": 9}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_failed", '{"reason": "Element gone."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.tools._target_label", return_value=""
        ), patch("app.browser.actions.click", side_effect=ActionError("Element [9] no longer exists")):
            result = run_browser_task("click something stale")
        self.assertIsInstance(result, BrowserFailed)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("no longer exists" in (m.get("content") or "") for m in tool_messages))

    def test_max_steps_reached_returns_failed_not_infinite_loop(self):
        from app.config import MAX_BROWSER_AGENT_STEPS

        looping = [
            _assistant_message(tool_calls=[_tool_call(f"c{i}", "scroll", '{"direction": "down"}')])
            for i in range(MAX_BROWSER_AGENT_STEPS + 2)
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=looping), patch(
            "app.browser.actions.scroll", return_value="Scrolled down"
        ):
            result = run_browser_task("keep scrolling forever")
        self.assertIsInstance(result, BrowserFailed)

    def test_browser_unavailable_short_circuits_before_any_llm_call(self):
        from app.browser.manager import BrowserUnavailable

        self.mock_manager.ensure_started.side_effect = BrowserUnavailable("Chrome not installed")
        with patch("app.browser.tools.ask_llm_message") as mock_llm:
            result = run_browser_task("open gmail")
        self.assertIsInstance(result, BrowserFailed)
        mock_llm.assert_not_called()


class OrchestratorBrowserWiringTests(unittest.TestCase):
    """browser_task selected by the OUTER orchestrator dispatches into the
    inner browser loop and translates its result into the right wrapper type."""

    def test_browser_task_selection_runs_inner_loop_and_wraps_done_result(self):
        outer_responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "browser_task", '{"instruction": "check gmail"}')]),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=outer_responses), patch(
            "app.brain.orchestrator.run_browser_task", return_value=BrowserDone("You have 3 new emails.")
        ) as mock_run:
            result = orchestrate("check my gmail")
        self.assertIsInstance(result, BrowserResultWrapper)
        self.assertEqual(result.text, "You have 3 new emails.")
        mock_run.assert_called_once()

    def test_browser_task_pause_becomes_browser_confirmation_result(self):
        outer_responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "browser_task", '{"instruction": "email John"}')]),
        ]
        paused = BrowserPaused(description="send on 'Send'", action_name="click", action_args={"element_id": 4}, messages=[])
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=outer_responses), patch(
            "app.brain.orchestrator.run_browser_task", return_value=paused
        ):
            result = orchestrate("email John saying I'll be late")
        self.assertIsInstance(result, BrowserConfirmationResult)
        self.assertEqual(result.paused, paused)


class IntentRouterBrowserWiringTests(unittest.TestCase):
    """The two new orchestrator result types must come back as Intents that
    dialogue.py/main.py can consume, matching the existing pattern for
    AnswerResult/ClarificationResult/ConfirmationResult."""

    def test_browser_result_wrapper_becomes_browser_result_intent(self):
        with patch.object(intent_router, "orchestrate", return_value=BrowserResultWrapper("Found the email from Twente.")):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.BROWSER_RESULT)
        self.assertEqual(intent.values["answer"], "Found the email from Twente.")

    def test_browser_confirmation_result_becomes_browser_confirm_intent(self):
        paused = BrowserPaused(description="send the email", action_name="click", action_args={}, messages=[])
        with patch.object(intent_router, "orchestrate", return_value=BrowserConfirmationResult("send the email", paused)):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.BROWSER_CONFIRM)
        self.assertEqual(intent.browser_paused, paused)
        self.assertEqual(intent.values["description"], "send the email")


class DialogueBrowserConfirmTests(unittest.TestCase):
    """The mid-task pause/resume flow through VoiceDialogue's Pending state
    machine - the part unique to browser tasks vs. every other confirmation."""

    def test_browser_confirm_intent_arms_pending_and_asks_to_proceed(self):
        dialogue = VoiceDialogue()
        paused = BrowserPaused(description="click 'Send'", action_name="click", action_args={}, messages=[])
        intent = intent_router.Intent(intent_router.BROWSER_CONFIRM, browser_paused=paused, values={"description": "click 'Send'"})
        with patch.object(intent_router, "parse", return_value=intent):
            response = dialogue.handle("email John saying I'll be late", lambda i: "unexpected")
        self.assertIn("click 'Send'", response)
        self.assertIsNotNone(dialogue.pending)
        self.assertEqual(dialogue.pending.kind, "browser_confirm")
        self.assertIs(dialogue.pending.values["paused"], paused)

    def test_yes_resumes_and_speaks_final_summary(self):
        dialogue = VoiceDialogue()
        paused = BrowserPaused(description="click 'Send'", action_name="click", action_args={}, messages=[])
        dialogue.pending = Pending("browser_confirm", {"paused": paused})
        with patch("app.brain.dialogue.resume_browser_task", return_value=BrowserDone("Email sent to John.")) as mock_resume:
            captured = []
            response = dialogue.handle("yes, go ahead", lambda i: captured.append(i) or "done")
        mock_resume.assert_called_once_with(paused, True)
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].values["answer"], "Email sent to John.")
        self.assertIsNone(dialogue.pending)

    def test_no_resumes_with_decline_and_clears_pending(self):
        dialogue = VoiceDialogue()
        paused = BrowserPaused(description="click 'Send'", action_name="click", action_args={}, messages=[])
        dialogue.pending = Pending("browser_confirm", {"paused": paused})
        with patch("app.brain.dialogue.resume_browser_task", return_value=BrowserFailed("Okay, I didn't send it.")) as mock_resume:
            captured = []
            response = dialogue.handle("no, don't send it", lambda i: captured.append(i) or "done")
        mock_resume.assert_called_once_with(paused, False)
        self.assertEqual(captured[0].values["answer"], "Okay, I didn't send it.")
        self.assertIsNone(dialogue.pending)

    def test_ambiguous_reply_is_treated_as_decline_not_a_silent_resume(self):
        """Unlike the DB-write confirmations, an ambiguous reply here must
        NOT resume the browser task - resuming on a misread 'yes' could
        really click Send/Buy, which is worse than the DB-write case."""
        dialogue = VoiceDialogue()
        paused = BrowserPaused(description="click 'Send'", action_name="click", action_args={}, messages=[])
        dialogue.pending = Pending("browser_confirm", {"paused": paused})
        with patch("app.brain.dialogue.resume_browser_task", return_value=BrowserFailed("Declined.")) as mock_resume:
            dialogue.handle("hmm, actually let me think about it", lambda i: "done")
        mock_resume.assert_called_once_with(paused, False)

    def test_chained_second_confirmation_rearms_pending_instead_of_dropping_it(self):
        """If resuming immediately hits ANOTHER high-impact step, the next
        yes/no must still be caught, not fall through to a generic reply.
        resume_browser_task returns app/browser/tools.py's OWN BrowserPaused
        directly (not the orchestrator's BrowserConfirmationResult wrapper,
        which only wraps the FIRST pause of a fresh task) - this is the
        actual shape a second pause takes in production."""
        dialogue = VoiceDialogue()
        paused = BrowserPaused(description="click 'Send'", action_name="click", action_args={}, messages=[])
        second_paused = BrowserPaused(description="click 'Confirm purchase'", action_name="click", action_args={}, messages=[])
        dialogue.pending = Pending("browser_confirm", {"paused": paused})
        with patch("app.brain.dialogue.resume_browser_task", return_value=second_paused):
            captured = []
            dialogue.handle("yes", lambda i: captured.append(i) or "done")
        self.assertIsNotNone(dialogue.pending)
        self.assertEqual(dialogue.pending.kind, "browser_confirm")
        self.assertIs(dialogue.pending.values["paused"], second_paused)


class ChromeHistoryMatcherTests(unittest.TestCase):
    """app/browser/history.py - name -> best-visited URL resolution. Chrome
    itself is mocked out entirely (sqlite reads are patched at the
    _load_all_history seam) so these run offline with no real Chrome/profile."""

    def _matches(self, *rows):
        return [
            history.SiteMatch(url=url, title=title, visit_count=visits, last_visit_unix=last_visit, profile="Default")
            for url, title, visits, last_visit in rows
        ]

    def test_domain_match_beats_title_only_match(self):
        fake = self._matches(
            ("https://github.com/", "GitHub", 45, 100),
            ("https://example.com/about-github-mirrors", "About our GitHub mirrors", 500, 50),
        )
        with patch("app.browser.history._load_all_history", return_value=fake):
            result = history.find_site("github", top_n=1)
        self.assertEqual(result[0].domain, "github.com")

    def test_higher_visit_count_wins_among_truly_equal_domain_matches(self):
        # Two hostnames that BOTH contain "docs" as a plain substring (no
        # exact-domain-name tiebreak advantage either way) - visit count
        # alone should decide.
        fake = self._matches(
            ("https://docs.example.com/", "Example Docs", 200, 100),
            ("https://mydocs.example.org/", "My Docs", 500, 90),
        )
        with patch("app.browser.history._load_all_history", return_value=fake):
            result = history.find_site("docs", top_n=1)
        self.assertEqual(result[0].domain, "mydocs.example.org")

    def test_exact_domain_name_match_beats_higher_visit_count_elsewhere(self):
        # "gmail" is literally IN "gmail.com" as a clean domain-name match,
        # which should win over a higher-visit-count page that only matches
        # via its title/a longer hostname.
        fake = self._matches(
            ("https://mail.google.com/mail/u/0/#inbox", "Inbox - Gmail", 919, 200),
            ("https://gmail.com/", "Gmail", 531, 150),
        )
        with patch("app.browser.history._load_all_history", return_value=fake):
            result = history.find_site("gmail", top_n=1)
        self.assertEqual(result[0].domain, "gmail.com")

    def test_no_history_match_returns_empty_list(self):
        with patch("app.browser.history._load_all_history", return_value=[]):
            result = history.find_site("some obscure site nobody visits")
        self.assertEqual(result, [])

    def test_best_site_url_returns_none_when_nothing_matches(self):
        with patch("app.browser.history._load_all_history", return_value=[]):
            self.assertIsNone(history.best_site_url("nope"))

    def test_canonical_url_uses_domain_root_not_deep_link(self):
        match = history.SiteMatch(url="https://mail.google.com/mail/u/0/#inbox", title="Inbox", visit_count=1, last_visit_unix=0, profile="Default")
        self.assertEqual(history.canonical_url(match), "https://mail.google.com/")

    def test_multiple_urls_on_same_domain_collapse_to_most_visited(self):
        fake = self._matches(
            ("https://github.com/", "GitHub", 10, 100),
            ("https://github.com/notifications", "Notifications", 40, 90),
        )
        with patch("app.browser.history._load_all_history", return_value=fake):
            result = history.find_site("github", top_n=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].visit_count, 40)


class BrowserOpenByNameToolTests(unittest.TestCase):
    """The inner agent loop's resolve_site_from_history tool call."""

    def setUp(self):
        self.manager_patch = patch("app.browser.tools.get_manager")
        mock_get_manager = self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        mock_manager = MagicMock()
        # Explicitly False, not a truthy-by-default MagicMock -
        # _would_use_real_chrome() checks this before the module's
        # "dedicated" mode patch even applies.
        mock_manager.is_running = False
        mock_get_manager.return_value = mock_manager

    def test_resolved_site_is_fed_back_and_loop_continues_to_goto(self):
        match = history.SiteMatch(url="https://github.com/", title="GitHub", visit_count=45, last_visit_unix=100, profile="Default")
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "resolve_site_from_history", '{"name": "github"}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "goto", '{"url": "https://github.com/"}')]),
            _assistant_message(tool_calls=[_tool_call("c3", "task_done", '{"summary": "Opened GitHub."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.history.best_site_url", return_value=match
        ), patch("app.browser.actions.goto", return_value="Opened https://github.com/"):
            result = run_browser_task("open github")
        self.assertIsInstance(result, BrowserDone)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("github.com" in (m.get("content") or "") for m in tool_messages))

    def test_no_history_match_is_reported_as_a_tool_result_not_a_crash(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "resolve_site_from_history", '{"name": "some totally obscure site"}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_failed", '{"reason": "No match and no obvious domain."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.history.best_site_url", return_value=None
        ):
            result = run_browser_task("open some totally obscure site")
        self.assertIsInstance(result, BrowserFailed)

    def test_history_read_failure_is_fed_back_not_raised(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "resolve_site_from_history", '{"name": "github"}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_failed", '{"reason": "Could not check history."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.history.best_site_url", side_effect=RuntimeError("Chrome history locked")
        ):
            result = run_browser_task("open github")
        self.assertIsInstance(result, BrowserFailed)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("couldn't read browsing history" in (m.get("content") or "") for m in tool_messages))


class ApplicantProfileToolTests(unittest.TestCase):
    """get_applicant_profile - the browser agent's only source of the user's
    real contact/resume details, never invented."""

    def setUp(self):
        self.manager_patch = patch("app.browser.tools.get_manager")
        mock_get_manager = self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        mock_manager = MagicMock()
        # Explicitly False, not a truthy-by-default MagicMock -
        # _would_use_real_chrome() checks this before the module's
        # "dedicated" mode patch even applies.
        mock_manager.is_running = False
        mock_get_manager.return_value = mock_manager

    def test_configured_fields_are_returned_verbatim(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_applicant_profile", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_done", '{"summary": "Got the profile."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.tools.APPLICANT_NAME", "Taksh Dange"
        ), patch("app.browser.tools.APPLICANT_EMAIL", "takshdange@gmail.com"), patch(
            "app.browser.tools.APPLICANT_RESUME_PATH", "C:\\resume.pdf"
        ):
            run_browser_task("apply to this internship")
        first_call_messages = mock_llm.call_args_list[0].args[0]
        # the profile tool result only appears in the SECOND call's messages
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        content = " ".join(m.get("content") or "" for m in tool_messages)
        self.assertIn("Taksh Dange", content)
        self.assertIn("takshdange@gmail.com", content)
        self.assertIn("resume.pdf", content)

    def test_missing_fields_are_flagged_not_fabricated(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_applicant_profile", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_done", '{"summary": "ok"}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.tools.APPLICANT_PORTFOLIO_URL", ""
        ):
            run_browser_task("apply to this internship")
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        content = " ".join(m.get("content") or "" for m in tool_messages)
        self.assertIn("_missing", content)
        self.assertIn("portfolio_url", content)


class AskUserFlowTests(unittest.TestCase):
    """ask_user pauses the loop for an open-ended answer (not yes/no) and
    resumes with whatever the user actually says - the info-gap counterpart
    to the high-impact-action confirm/resume flow."""

    def setUp(self):
        self.manager_patch = patch("app.browser.tools.get_manager")
        mock_get_manager = self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        mock_manager = MagicMock()
        # Explicitly False, not a truthy-by-default MagicMock -
        # _would_use_real_chrome() checks this before the module's
        # "dedicated" mode patch even applies.
        mock_manager.is_running = False
        mock_get_manager.return_value = mock_manager

    def test_ask_user_returns_browser_asking_user_without_finishing(self):
        responses = [
            _assistant_message(
                tool_calls=[_tool_call("c1", "ask_user", '{"question": "What role are you applying for?"}')]
            ),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses):
            result = run_browser_task("apply to this company")
        self.assertIsInstance(result, BrowserAskingUser)
        self.assertIn("role", result.question)

    def test_resume_with_answer_feeds_it_back_and_continues(self):
        asking = BrowserAskingUser(question="What role?", tool_call_id="c1", messages=[{"role": "system", "content": "x"}])
        responses = [
            _assistant_message(tool_calls=[_tool_call("c2", "task_done", '{"summary": "Applied for Backend Intern."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm:
            result = resume_browser_task_with_answer(asking, "Backend Intern")
        self.assertIsInstance(result, BrowserDone)
        first_call_messages = mock_llm.call_args_list[0].args[0]
        tool_messages = [m for m in first_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("Backend Intern" in (m.get("content") or "") for m in tool_messages))


class OrchestratorBrowserAskWiringTests(unittest.TestCase):
    def test_browser_task_ask_user_becomes_browser_ask_user_result(self):
        outer_responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "browser_task", '{"instruction": "apply to internship"}')]),
        ]
        asking = BrowserAskingUser(question="What role?", tool_call_id="c1", messages=[])
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=outer_responses), patch(
            "app.brain.orchestrator.run_browser_task", return_value=asking
        ):
            result = orchestrate("apply to the internship")
        self.assertIsInstance(result, BrowserAskUserResult)
        self.assertEqual(result.question, "What role?")
        self.assertEqual(result.asking, asking)


class IntentRouterBrowserAskWiringTests(unittest.TestCase):
    def test_browser_ask_user_result_becomes_browser_ask_intent(self):
        asking = BrowserAskingUser(question="What role?", tool_call_id="c1", messages=[])
        with patch.object(intent_router, "orchestrate", return_value=BrowserAskUserResult("What role?", asking)):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.BROWSER_ASK)
        self.assertEqual(intent.browser_asking, asking)
        self.assertEqual(intent.values["question"], "What role?")


class DialogueBrowserAnswerTests(unittest.TestCase):
    """The browser_answer Pending state - mirrors browser_confirm but for an
    open-ended reply instead of yes/no."""

    def test_browser_ask_intent_arms_pending_and_speaks_the_question(self):
        dialogue = VoiceDialogue()
        asking = BrowserAskingUser(question="What role are you applying for?", tool_call_id="c1", messages=[])
        intent = intent_router.Intent(intent_router.BROWSER_ASK, browser_asking=asking, values={"question": "What role are you applying for?"})
        with patch.object(intent_router, "parse", return_value=intent):
            response = dialogue.handle("apply to this internship", lambda i: "unexpected")
        self.assertEqual(response, "What role are you applying for?")
        self.assertIsNotNone(dialogue.pending)
        self.assertEqual(dialogue.pending.kind, "browser_answer")
        self.assertIs(dialogue.pending.values["asking"], asking)

    def test_answer_resumes_and_speaks_final_summary(self):
        dialogue = VoiceDialogue()
        asking = BrowserAskingUser(question="What role?", tool_call_id="c1", messages=[])
        dialogue.pending = Pending("browser_answer", {"asking": asking})
        with patch("app.brain.dialogue.resume_browser_task_with_answer", return_value=BrowserDone("Applied for Backend Intern.")) as mock_resume:
            captured = []
            response = dialogue.handle("Backend Intern", lambda i: captured.append(i) or "done")
        mock_resume.assert_called_once_with(asking, "Backend Intern")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].values["answer"], "Applied for Backend Intern.")
        self.assertIsNone(dialogue.pending)

    def test_chained_confirm_after_answer_rearms_pending_as_browser_confirm(self):
        """Answering an info-gap question can immediately run into a
        high-impact step (e.g. now that the role is known, the agent is
        ready to click Submit) - the NEXT pending kind must switch to
        browser_confirm, not stay stuck as browser_answer.
        resume_browser_task_with_answer returns app/browser/tools.py's OWN
        BrowserPaused directly, same as resume_browser_task - see the
        chained-confirmation test above for why the orchestrator's wrapper
        type is the wrong shape to mock here."""
        dialogue = VoiceDialogue()
        asking = BrowserAskingUser(question="What role?", tool_call_id="c1", messages=[])
        paused = BrowserPaused(description="click 'Submit application'", action_name="click", action_args={}, messages=[])
        dialogue.pending = Pending("browser_answer", {"asking": asking})
        with patch("app.brain.dialogue.resume_browser_task_with_answer", return_value=paused):
            dialogue.handle("Backend Intern", lambda i: "done")
        self.assertIsNotNone(dialogue.pending)
        self.assertEqual(dialogue.pending.kind, "browser_confirm")
        self.assertIs(dialogue.pending.values["paused"], paused)


class DomainBlocklistTests(unittest.TestCase):
    """The code-enforced protected-site list for real-Chrome mode - a hard
    wall independent of the LLM's judgement."""

    def test_default_banking_domain_is_blocked(self):
        self.assertTrue(safety.is_blocked_url("https://www.chase.com/login"))
        self.assertEqual(safety.blocked_domain_for("https://secure.chase.com/x"), "chase.com")

    def test_password_manager_is_blocked(self):
        self.assertTrue(safety.is_blocked_url("https://vault.bitwarden.com/"))

    def test_generic_bank_substring_is_blocked(self):
        # The bare "bank" entry catches arbitrary banking hostnames.
        self.assertTrue(safety.is_blocked_url("https://onlinebanking.example.com/"))

    def test_ordinary_site_is_not_blocked(self):
        self.assertFalse(safety.is_blocked_url("https://github.com/"))
        self.assertFalse(safety.is_blocked_url("https://www.youtube.com/"))
        self.assertIsNone(safety.blocked_domain_for("https://linkedin.com/jobs"))

    def test_goto_raises_actionerror_on_blocked_url_in_real_chrome_mode(self):
        from app.browser import actions

        mock_manager = MagicMock()
        mock_manager.is_real_chrome = True
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        with patch("app.browser.actions.get_manager", return_value=mock_manager):
            with self.assertRaises(actions.ActionError) as ctx:
                actions.goto("https://chase.com/login")
        self.assertIn("protected-site", str(ctx.exception).lower())

    def test_goto_allows_blocked_url_when_NOT_real_chrome_mode(self):
        """The dedicated isolated profile has no real logins to protect, so the
        blocklist doesn't apply there - only in real-Chrome mode."""
        from app.browser import actions

        mock_manager = MagicMock()
        mock_manager.is_real_chrome = False
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        mock_page = MagicMock()
        mock_page.url = "https://chase.com/login"
        mock_manager.page = mock_page
        with patch("app.browser.actions.get_manager", return_value=mock_manager):
            result = actions.goto("https://chase.com/login")
        self.assertIn("chase.com", result)
        mock_page.goto.assert_called_once()


class RealChromeTaskGateTests(unittest.TestCase):
    """The per-task gate: in real-Chrome mode a task must be approved before
    the agent loop touches the user's live browser at all."""

    def test_real_mode_returns_gate_without_running_loop(self):
        mock_manager = MagicMock()
        mock_manager.is_running = False
        with patch("app.config.JARVIX_BROWSER_MODE", "real"), patch(
            "app.browser.tools.ask_llm_message"
        ) as mock_llm, patch("app.browser.tools.get_manager", return_value=mock_manager) as mock_gm:
            result = run_browser_task("open my email")
        self.assertIsInstance(result, BrowserRealChromeGate)
        self.assertEqual(result.instruction, "open my email")
        # Nothing touched the browser or the LLM yet - it's purely a gate.
        mock_llm.assert_not_called()
        mock_gm.return_value.ensure_started.assert_not_called()

    def test_dedicated_mode_runs_loop_immediately_no_gate(self):
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "Done."}')])]
        mock_manager = MagicMock()
        mock_manager.is_running = False
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        with patch("app.config.JARVIX_BROWSER_MODE", "dedicated"), patch(
            "app.browser.tools.ask_llm_message", side_effect=responses
        ), patch("app.browser.tools.get_manager", return_value=mock_manager):
            result = run_browser_task("open my email")
        self.assertIsInstance(result, BrowserDone)

    def test_auto_mode_gates_when_cdp_is_reachable(self):
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "Done."}')])]
        mock_manager = MagicMock()
        mock_manager.is_running = False
        with patch("app.config.JARVIX_BROWSER_MODE", "auto"), patch(
            "app.browser.manager._probe_cdp_available", return_value=True
        ), patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.tools.get_manager", return_value=mock_manager
        ):
            result = run_browser_task("open my email")
        self.assertIsInstance(result, BrowserRealChromeGate)
        mock_llm.assert_not_called()

    def test_auto_mode_skips_gate_when_cdp_unreachable(self):
        """The common case: no real Chrome launched with the debug flag -
        auto mode must behave exactly like dedicated mode, no gate at all."""
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "Done."}')])]
        mock_manager = MagicMock()
        mock_manager.is_running = False
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        with patch("app.config.JARVIX_BROWSER_MODE", "auto"), patch(
            "app.browser.manager._probe_cdp_available", return_value=False
        ), patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools.get_manager", return_value=mock_manager
        ):
            result = run_browser_task("open my email")
        self.assertIsInstance(result, BrowserDone)

    def test_already_running_browser_is_not_reprobed_or_regated(self):
        """Once a browser is up for this process, later tasks in the SAME
        session must not re-probe or re-gate every single time - only the
        mode the process actually landed on (is_real_chrome) matters."""
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "Done."}')])]
        mock_manager = MagicMock()
        mock_manager.is_running = True
        mock_manager.is_real_chrome = False
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        with patch("app.config.JARVIX_BROWSER_MODE", "auto"), patch(
            "app.browser.manager._probe_cdp_available"
        ) as mock_probe, patch("app.browser.tools.ask_llm_message", side_effect=responses), patch(
            "app.browser.tools.get_manager", return_value=mock_manager
        ):
            result = run_browser_task("open my email")
        self.assertIsInstance(result, BrowserDone)
        mock_probe.assert_not_called()

    def test_gate_decline_leaves_browser_untouched(self):
        gate = BrowserRealChromeGate(instruction="open my email")
        with patch("app.browser.tools.get_manager") as mock_gm, patch(
            "app.browser.tools.ask_llm_message"
        ) as mock_llm:
            result = resume_real_chrome_task(gate, approved=False)
        self.assertIsInstance(result, BrowserFailed)
        mock_gm.return_value.ensure_started.assert_not_called()
        mock_llm.assert_not_called()

    def test_gate_approval_runs_the_task(self):
        gate = BrowserRealChromeGate(instruction="check my notifications")
        responses = [_assistant_message(tool_calls=[_tool_call("c1", "task_done", '{"summary": "All clear."}')])]
        mock_manager = MagicMock()
        mock_manager.run.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        with patch("app.browser.tools.get_manager", return_value=mock_manager), patch(
            "app.browser.tools.ask_llm_message", side_effect=responses
        ):
            result = resume_real_chrome_task(gate, approved=True)
        self.assertIsInstance(result, BrowserDone)
        self.assertEqual(result.summary, "All clear.")


class OrchestratorRealChromeGateWiringTests(unittest.TestCase):
    def test_browser_task_gate_becomes_gate_result(self):
        outer = [_assistant_message(tool_calls=[_tool_call("c1", "browser_task", '{"instruction": "open email"}')])]
        gate = BrowserRealChromeGate(instruction="open email")
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=outer), patch(
            "app.brain.orchestrator.run_browser_task", return_value=gate
        ):
            result = orchestrate("open my email")
        self.assertIsInstance(result, BrowserRealChromeGateResult)
        self.assertEqual(result.gate, gate)


class IntentRouterRealChromeGateWiringTests(unittest.TestCase):
    def test_gate_result_becomes_browser_gate_intent(self):
        gate = BrowserRealChromeGate(instruction="open email")
        with patch.object(intent_router, "orchestrate", return_value=BrowserRealChromeGateResult("open email", gate)):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.BROWSER_GATE)
        self.assertEqual(intent.browser_gate, gate)
        self.assertEqual(intent.values["instruction"], "open email")


class DialogueRealChromeGateTests(unittest.TestCase):
    def test_gate_intent_arms_pending_and_warns_about_real_chrome(self):
        dialogue = VoiceDialogue()
        gate = BrowserRealChromeGate(instruction="open my email")
        intent = intent_router.Intent(intent_router.BROWSER_GATE, browser_gate=gate, values={"instruction": "open my email"})
        with patch.object(intent_router, "parse", return_value=intent):
            response = dialogue.handle("open my email in my real chrome", lambda i: "unexpected")
        self.assertIn("real Chrome", response)
        self.assertEqual(dialogue.pending.kind, "browser_real_chrome_gate")
        self.assertIs(dialogue.pending.values["gate"], gate)

    def test_yes_runs_the_task(self):
        dialogue = VoiceDialogue()
        gate = BrowserRealChromeGate(instruction="open my email")
        dialogue.pending = Pending("browser_real_chrome_gate", {"gate": gate})
        with patch("app.brain.dialogue.resume_real_chrome_task", return_value=BrowserDone("Opened your email.")) as mock_run:
            captured = []
            dialogue.handle("yes go ahead", lambda i: captured.append(i) or "done")
        mock_run.assert_called_once_with(gate, True)
        self.assertEqual(captured[0].values["answer"], "Opened your email.")
        self.assertIsNone(dialogue.pending)

    def test_no_abandons_the_task(self):
        dialogue = VoiceDialogue()
        gate = BrowserRealChromeGate(instruction="open my email")
        dialogue.pending = Pending("browser_real_chrome_gate", {"gate": gate})
        with patch("app.brain.dialogue.resume_real_chrome_task", return_value=BrowserFailed("Left it alone.")) as mock_run:
            dialogue.handle("no, don't", lambda i: "done")
        mock_run.assert_called_once_with(gate, False)
        self.assertIsNone(dialogue.pending)

    def test_ambiguous_reply_is_treated_as_decline(self):
        """Approving use of the user's OWN live browser must require a clean
        yes - anything ambiguous declines."""
        dialogue = VoiceDialogue()
        gate = BrowserRealChromeGate(instruction="open my email")
        dialogue.pending = Pending("browser_real_chrome_gate", {"gate": gate})
        with patch("app.brain.dialogue.resume_real_chrome_task", return_value=BrowserFailed("Left it alone.")) as mock_run:
            dialogue.handle("hmm maybe later", lambda i: "done")
        mock_run.assert_called_once_with(gate, False)


class RealChromeManagerTests(unittest.TestCase):
    """The manager's CDP-attach vs launch decision and its no-close-on-detach
    teardown (so shutdown never closes the user's own Chrome)."""

    def test_shutdown_in_real_chrome_mode_does_not_close_context(self):
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        mock_ctx = MagicMock()
        mgr._context = mock_ctx
        mgr._attached_to_real_chrome = True
        mgr._playwright = MagicMock()
        mgr._shutdown_impl()
        mock_ctx.close.assert_not_called()  # must NOT close the user's Chrome
        mgr._playwright  # detached

    def test_shutdown_in_dedicated_mode_does_close_context(self):
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        mock_ctx = MagicMock()
        mgr._context = mock_ctx
        mgr._attached_to_real_chrome = False
        mgr._playwright = MagicMock()
        mgr._shutdown_impl()
        mock_ctx.close.assert_called_once()  # our own launched context - safe to close

    def test_close_tab_refuses_in_real_chrome_mode(self):
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        mgr._attached_to_real_chrome = True
        with self.assertRaises(RuntimeError):
            mgr._close_tab_impl(MagicMock())


class ExternallyClosedBrowserRelaunchTests(unittest.TestCase):
    """Regression: if the Chrome window Jarvix opened gets closed (by the
    user, or a crash) between two separate tasks, the manager must detect the
    dead context and relaunch - not keep handing back a reference that fails
    every call forever after ('Target page, context or browser has been
    closed'), which is exactly what happened live: task 1 opened GitHub fine,
    the window was closed, and task 2's goto failed silently in a loop."""

    def test_start_impl_relaunches_after_close_event_fires(self):
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        first_context = MagicMock()
        first_context.pages = []
        first_context.new_page.return_value = MagicMock(is_closed=lambda: False)

        second_context = MagicMock()
        second_context.pages = []
        second_context.new_page.return_value = MagicMock(is_closed=lambda: False)

        with patch("app.browser.manager.sync_playwright") as mock_sync_pw:
            mock_pw_instance = MagicMock()
            mock_sync_pw.return_value.start.return_value = mock_pw_instance
            mock_pw_instance.chromium.launch_persistent_context.side_effect = [first_context, second_context]

            mgr._start_dedicated_impl()
            self.assertIs(mgr._context, first_context)
            self.assertFalse(mgr._context_closed)

            # Simulate the window being closed externally - Playwright fires
            # this event on the context itself, which _on_context_closed
            # (registered via first_context.on("close", ...)) picks up.
            close_handler = next(
                call.args[1] for call in first_context.on.call_args_list if call.args[0] == "close"
            )
            close_handler()
            self.assertTrue(mgr._context_closed)

            # The next start() call must detect this and relaunch, not hand
            # back the dead first_context.
            result = mgr._start_impl()
            self.assertIs(result, second_context)
            self.assertIs(mgr._context, second_context)
            self.assertFalse(mgr._context_closed)

    def test_live_context_is_reused_without_relaunching(self):
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        existing_context = MagicMock()
        mgr._context = existing_context
        mgr._context_closed = False

        with patch("app.browser.manager.sync_playwright") as mock_sync_pw:
            result = mgr._start_impl()

        self.assertIs(result, existing_context)
        mock_sync_pw.assert_not_called()  # no relaunch when the context is still alive


class CdpProbeTests(unittest.TestCase):
    """The fast reachability check 'auto' mode uses to decide, on every fresh
    browser start, whether the user's real Chrome is actually attachable -
    without paying for a full connect_over_cdp() attempt."""

    def test_probe_true_on_valid_cdp_response(self):
        from app.browser.manager import _probe_cdp_available

        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.read.return_value = json.dumps({"Browser": "Chrome/120.0", "webSocketDebuggerUrl": "ws://x"}).encode()
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with patch("app.browser.manager.urllib.request.urlopen", return_value=fake_response):
            self.assertTrue(_probe_cdp_available("http://127.0.0.1:9222", 1.0))

    def test_probe_false_on_connection_refused(self):
        from app.browser.manager import _probe_cdp_available

        with patch("app.browser.manager.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            self.assertFalse(_probe_cdp_available("http://127.0.0.1:9222", 1.0))

    def test_probe_false_on_timeout(self):
        from app.browser.manager import _probe_cdp_available

        with patch("app.browser.manager.urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertFalse(_probe_cdp_available("http://127.0.0.1:9222", 1.0))

    def test_probe_false_on_404_like_the_devtools_mcp_toggle(self):
        """Regression for the exact failure mode hit live: Chrome's in-browser
        'Allow remote debugging' toggle answers on the port but 404s every
        classic CDP endpoint - the probe must treat that as unavailable, not
        crash trying to parse a non-200 body as CDP JSON."""
        from app.browser.manager import _probe_cdp_available

        fake_response = MagicMock()
        fake_response.status = 404
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with patch("app.browser.manager.urllib.request.urlopen", return_value=fake_response):
            self.assertFalse(_probe_cdp_available("http://127.0.0.1:9222", 1.0))

    def test_probe_false_on_200_but_not_actually_cdp_json(self):
        """Some other, unrelated local service answering on that port with a
        200 must not be mistaken for a real CDP endpoint."""
        from app.browser.manager import _probe_cdp_available

        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        with patch("app.browser.manager.urllib.request.urlopen", return_value=fake_response):
            self.assertFalse(_probe_cdp_available("http://127.0.0.1:9222", 1.0))


class BrowserModeResolutionTests(unittest.TestCase):
    """app/config.py's JARVIX_BROWSER_MODE resolution, including back-compat
    with the old JARVIX_BROWSER_USE_REAL_CHROME on/off flag."""

    def _reload_config(self, env: dict):
        import importlib
        import os

        import app.config as config_module

        old_environ = dict(os.environ)
        try:
            # Set both to "" (present-but-empty) rather than popping them -
            # config.py's load_dotenv(override=False) call only fills a var
            # in from the real .env file when it's ABSENT from os.environ, so
            # popping would let this machine's own .env (which may set
            # JARVIX_BROWSER_MODE) leak into the test instead of the
            # scenario's env dict below.
            for k in ("JARVIX_BROWSER_MODE", "JARVIX_BROWSER_USE_REAL_CHROME"):
                os.environ[k] = ""
            os.environ.update(env)
            importlib.reload(config_module)
            return config_module.JARVIX_BROWSER_MODE, config_module.JARVIX_BROWSER_USE_REAL_CHROME
        finally:
            os.environ.clear()
            os.environ.update(old_environ)
            importlib.reload(config_module)

    def test_defaults_to_auto_with_no_env_set(self):
        mode, use_real = self._reload_config({})
        self.assertEqual(mode, "auto")
        self.assertFalse(use_real)

    def test_explicit_mode_is_respected(self):
        mode, use_real = self._reload_config({"JARVIX_BROWSER_MODE": "dedicated"})
        self.assertEqual(mode, "dedicated")
        self.assertFalse(use_real)

    def test_explicit_real_mode(self):
        mode, use_real = self._reload_config({"JARVIX_BROWSER_MODE": "real"})
        self.assertEqual(mode, "real")
        self.assertTrue(use_real)

    def test_old_flag_true_maps_to_real_for_backcompat(self):
        mode, use_real = self._reload_config({"JARVIX_BROWSER_USE_REAL_CHROME": "true"})
        self.assertEqual(mode, "real")
        self.assertTrue(use_real)

    def test_old_flag_false_maps_to_auto_not_dedicated(self):
        """The pre-existing default behavior (dedicated-only) becomes 'auto'
        under the new scheme, not literally 'dedicated' - auto is a strict
        superset of the old default (same behavior when no real Chrome is
        reachable, better behavior when one is)."""
        mode, use_real = self._reload_config({"JARVIX_BROWSER_USE_REAL_CHROME": "false"})
        self.assertEqual(mode, "auto")
        self.assertFalse(use_real)

    def test_invalid_mode_value_falls_back_to_auto(self):
        mode, use_real = self._reload_config({"JARVIX_BROWSER_MODE": "not-a-real-mode"})
        self.assertEqual(mode, "auto")
        self.assertFalse(use_real)

    def test_explicit_mode_takes_priority_over_old_flag(self):
        mode, use_real = self._reload_config({"JARVIX_BROWSER_MODE": "dedicated", "JARVIX_BROWSER_USE_REAL_CHROME": "true"})
        self.assertEqual(mode, "dedicated")
        self.assertFalse(use_real)


class HungBrowserConnectionTests(unittest.TestCase):
    """Regression (live bug, 2026-08-10): a mid-task Playwright call that
    hangs forever (connection to Chrome silently died) used to wedge the
    single dedicated browser thread permanently - every future browser task
    would then hang too, with ZERO log output, which is exactly what the user
    saw ("the browser part crashed and didn't do anything"). Fixed with a
    hard timeout on _BrowserThread.run() that abandons the stuck worker
    thread, starts a fresh one, and marks the BrowserManager's cached
    Playwright/context state dead so the next call relaunches instead of
    reusing broken references."""

    def test_hung_call_raises_and_does_not_block_forever(self):
        from app.browser.manager import _BrowserThread, _BrowserThreadHung

        thread = _BrowserThread()
        self.addCleanup(lambda: None)  # daemon thread, no explicit teardown needed

        def _never_returns():
            import time

            time.sleep(999)

        with patch("app.browser.manager._HARD_CALL_TIMEOUT_SECONDS", 0.2):
            start = __import__("time").time()
            with self.assertRaises(_BrowserThreadHung):
                thread.run(_never_returns)
            elapsed = __import__("time").time() - start
        # Must return promptly (bounded by the patched timeout), not hang
        # for the full 999s the stuck call would otherwise take.
        self.assertLess(elapsed, 5.0)

    def test_hung_call_gets_a_fresh_thread_for_the_next_call(self):
        from app.browser.manager import _BrowserThread

        thread = _BrowserThread()
        old_worker = thread._thread

        with patch("app.browser.manager._HARD_CALL_TIMEOUT_SECONDS", 0.2):
            try:
                thread.run(lambda: __import__("time").sleep(999))
            except Exception:
                pass

        self.assertIsNot(thread._thread, old_worker)
        # The new thread must actually work for a normal, fast call.
        self.assertEqual(thread.run(lambda: 42), 42)

    def test_manager_marks_dead_on_hang_and_relaunches_without_calling_stop_on_old_playwright(self):
        """The old self._playwright belongs to the abandoned thread's dead
        connection - _teardown_partial must NOT call .stop() on it from the
        new thread (that would violate Playwright's single-thread rule and
        could itself hang)."""
        from app.browser.manager import BrowserManager

        mgr = BrowserManager()
        old_playwright = MagicMock()
        mgr._playwright = old_playwright
        mgr._context = MagicMock()
        mgr._context_closed = False

        mgr._mark_dead()
        self.assertTrue(mgr._context_closed)
        self.assertTrue(mgr._dead_from_hang)

        new_context = MagicMock()
        new_context.pages = []
        new_context.new_page.return_value = MagicMock(is_closed=lambda: False)
        with patch("app.browser.manager.sync_playwright") as mock_sync_pw:
            mock_pw_instance = MagicMock()
            mock_sync_pw.return_value.start.return_value = mock_pw_instance
            mock_pw_instance.chromium.launch_persistent_context.return_value = new_context
            result = mgr._start_impl()

        self.assertIs(result, new_context)
        old_playwright.stop.assert_not_called()  # never touched from the new thread
        self.assertFalse(mgr._context_closed)
        self.assertFalse(mgr._dead_from_hang)

    def test_actions_convert_hung_connection_into_actionerror(self):
        """app/browser/actions.py's _run() must convert a hung/dead connection
        into ActionError so app/browser/tools.py's existing `except
        ActionError` handling in the agent loop catches it - same as any
        other action failure - instead of an unhandled exception silently
        killing the whole task."""
        from app.browser import actions
        from app.browser.manager import _BrowserThreadHung

        mock_manager = MagicMock()
        mock_manager.run.side_effect = _BrowserThreadHung("connection dead")
        with patch("app.browser.actions.get_manager", return_value=mock_manager):
            with self.assertRaises(actions.ActionError):
                actions.goto("https://example.com")

    def test_hang_during_task_is_reported_not_silently_swallowed(self):
        """Full agent-loop-level regression: a hung goto() must surface as a
        BrowserFailed (spoken to the user) via task_failed, or at minimum
        never leave the loop silently stuck - matching the exact failure
        shape from jarvix.log (goto hung, zero further output, no result
        ever returned to the user)."""
        from app.browser.actions import ActionError

        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "goto", '{"url": "https://example.com"}')]),
            _assistant_message(tool_calls=[_tool_call("c2", "task_failed", '{"reason": "Lost connection to the browser."}')]),
        ]
        with patch("app.browser.tools.ask_llm_message", side_effect=responses) as mock_llm, patch(
            "app.browser.actions.goto", side_effect=ActionError("A browser action didn't respond within 45s.")
        ):
            result = run_browser_task("open a page")
        self.assertIsInstance(result, BrowserFailed)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("didn't respond" in (m.get("content") or "") for m in tool_messages))


if __name__ == "__main__":
    unittest.main()
