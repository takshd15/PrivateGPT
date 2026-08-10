"""Tests for the LLM-first orchestrator (app/brain/orchestrator.py) and the
confirmation-word fix in app/brain/dialogue.py. All LLM calls are mocked via
app.brain.orchestrator.ask_llm_message so this stays offline/deterministic,
matching the pattern in tests/test_voice_regressions.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.brain import intent_router, tools_registry
from app.brain.dialogue import Pending, VoiceDialogue
from app.brain.orchestrator import (
    MAX_TOOL_CALLS,
    AnswerResult,
    ClarificationResult,
    ConfirmationResult,
    Turn,
    orchestrate,
)


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _assistant_message(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


def _mock_tool(name: str, **kwargs):
    """patch.object on the live ToolSpec.handler for ``name`` in the
    registry. Patching the module-level _get_x function instead would miss -
    ToolSpec.handler captured that function object at import time, so the
    registry entry itself is the only thing _run_tool actually calls through."""
    return patch.object(tools_registry.get(name), "handler", **kwargs)


class OrchestratorToolRoutingTests(unittest.TestCase):
    """Spec case 1/2/3/4/8: a single tool call resolves the request instead of
    falling through to a generic clarification or the wrong tool."""

    def test_news_request_calls_news_tool(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_news", "{}")]),
            _assistant_message(content="Top story: markets rallied today."),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses), _mock_tool(
            "get_news", return_value=[{"title": "Markets rallied", "source": "AP"}]
        ):
            result = orchestrate("What's the news today?")
        self.assertIsInstance(result, AnswerResult)
        self.assertIn("markets", result.text.lower())

    def test_projects_query_calls_list_projects_not_new_project_flow(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "list_projects", "{}")]),
            _assistant_message(content="You're working on GDCN and Jarvix."),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses), _mock_tool(
            "list_projects", return_value=[{"name": "GDCN"}, {"name": "Jarvix"}]
        ):
            result = orchestrate("What are the projects I'm working on?")
        self.assertIsInstance(result, AnswerResult)
        self.assertIn("GDCN", result.text)

    def test_add_event_request_yields_confirmation_result(self):
        responses = [
            _assistant_message(
                tool_calls=[_tool_call("c1", "create_calendar_event", '{"title": "Dentist", "date": "2026-08-10"}')]
            ),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses):
            result = orchestrate("Can you add an event to my calendar?")
        self.assertIsInstance(result, ConfirmationResult)
        self.assertEqual(result.tool_name, "create_calendar_event")
        self.assertEqual(result.arguments["title"], "Dentist")


class OrchestratorContextTests(unittest.TestCase):
    """Spec case 5/6/13: resolving 'the best one' against recent history
    instead of asking a generic clarification."""

    def test_best_opportunity_uses_history_and_chains_tools(self):
        history = [
            Turn("user", "Find opportunities for me."),
            Turn("assistant", "I found four: Y Combinator, Flagship, Berkeley SkyDeck and Doriot."),
        ]
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_goals", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "get_opportunities", "{}")]),
            _assistant_message(content="Y Combinator looks like the strongest fit given your startup goals."),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses) as mock_llm, _mock_tool(
            "get_goals", return_value=[{"title": "Build a startup", "priority": 3}]
        ), _mock_tool(
            "get_opportunities",
            return_value=[{"name": "Y Combinator", "fit_score": 0.9}, {"name": "Flagship", "fit_score": 0.5}],
        ):
            result = orchestrate("What's the best for me?", history=history)

        self.assertIsInstance(result, AnswerResult)
        self.assertIn("Y Combinator", result.text)
        # The conversation history must actually reach the model, or "the
        # best one" has nothing to resolve against.
        first_call_messages = mock_llm.call_args_list[0].args[0]
        joined = " ".join(m.get("content") or "" for m in first_call_messages if isinstance(m.get("content"), str))
        self.assertIn("Y Combinator", joined)

    def test_genuinely_ambiguous_request_with_no_context_asks_useful_clarification(self):
        responses = [
            _assistant_message(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "ask_clarifying_question",
                        '{"question": "What would you like me to find the best option for - opportunities, projects, or something else?"}',
                    )
                ]
            ),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses):
            result = orchestrate("What's the best one?", history=[])
        self.assertIsInstance(result, ClarificationResult)
        self.assertNotEqual(result.question, "I didn't catch that clearly. Can you repeat it?")


class OrchestratorMultiToolTests(unittest.TestCase):
    """Spec case 15: iterative multi-tool sequences within one request."""

    def test_three_round_tool_chain_synthesizes_final_answer(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_applications", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "get_upcoming_events", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c3", "get_tasks", "{}")]),
            _assistant_message(content="Your ASML application needs a follow-up email this week."),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses) as mock_llm, _mock_tool(
            "get_applications", return_value=[{"name": "ASML", "status": "interview"}]
        ), _mock_tool("get_upcoming_events", return_value=[]), _mock_tool("get_tasks", return_value=[]):
            result = orchestrate("Which of my applications needs attention this week?")
        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(mock_llm.call_count, 4)


class OrchestratorLoopSafetyTests(unittest.TestCase):
    """Spec case 16: loop protection against runaway/repeated tool calls."""

    def test_max_tool_calls_forces_synthesis_without_crashing(self):
        # The model keeps asking for a new (non-repeating) tool every round -
        # exercise the MAX_TOOL_CALLS cutoff itself, not the repeat-guard.
        looping = [
            _assistant_message(tool_calls=[_tool_call(f"c{i}", "get_weather", f'{{"location": "City{i}"}}')])
            for i in range(MAX_TOOL_CALLS + 2)
        ]
        final = _assistant_message(content="Here's what I found so far.")
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=looping + [final]) as mock_llm, _mock_tool(
            "get_weather", side_effect=lambda location=None: f"Sunny in {location}."
        ):
            result = orchestrate("keep going")
        self.assertIsInstance(result, AnswerResult)
        # MAX_TOOL_CALLS loop iterations + 1 final no-tools synthesis call.
        self.assertEqual(mock_llm.call_count, MAX_TOOL_CALLS + 1)

    def test_repeated_identical_tool_call_does_not_loop_forever(self):
        repeated = [
            _assistant_message(tool_calls=[_tool_call(f"c{i}", "get_weather", '{"location": "Paris"}')])
            for i in range(MAX_TOOL_CALLS + 2)
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=repeated), _mock_tool(
            "get_weather", return_value="Sunny in Paris."
        ):
            result = orchestrate("weather in Paris please")
        self.assertIsInstance(result, AnswerResult)


class OrchestratorFailureHandlingTests(unittest.TestCase):
    """Spec case 14/19: tool failures produce a graceful spoken explanation,
    never crash, never fall back to the generic 'I didn't catch that'."""

    def test_llm_call_failure_returns_safe_answer_not_clarification(self):
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=RuntimeError("network down")):
            result = orchestrate("What's the weather?")
        self.assertIsInstance(result, AnswerResult)
        self.assertNotIn("didn't catch that", result.text.lower())

    def test_tool_failure_is_fed_back_to_llm_instead_of_crashing(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_weather", "{}")]),
            _assistant_message(content="I couldn't reach the weather service, but I can try again shortly."),
        ]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses) as mock_llm, _mock_tool(
            "get_weather", side_effect=RuntimeError("weather API down")
        ):
            result = orchestrate("What's the weather?")
        self.assertIsInstance(result, AnswerResult)
        self.assertIn("weather service", result.text)
        # The failure must have been reported back as a tool result, not raised.
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(any("failed" in (m.get("content") or "") for m in tool_messages))


class IntentRouterOrchestratorWiringTests(unittest.TestCase):
    """The orchestrator's results must come back out as Intents that main.py's
    existing dispatch (checking intent.values['answer']/['question']) and
    dialogue.py's existing confirmation states can consume unchanged."""

    def test_answer_result_becomes_question_intent_with_answer_value(self):
        with patch.object(intent_router, "orchestrate", return_value=AnswerResult("It's sunny.")):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.QUESTION)
        self.assertEqual(intent.values["answer"], "It's sunny.")

    def test_clarification_result_becomes_clarification_intent_with_question_value(self):
        with patch.object(intent_router, "orchestrate", return_value=ClarificationResult("Which project?")):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.CLARIFICATION_NEEDED)
        self.assertEqual(intent.values["question"], "Which project?")

    def test_create_calendar_event_confirmation_maps_to_add_event_intent(self):
        confirmation = ConfirmationResult("create_calendar_event", {"title": "Dentist", "date": "2026-08-10", "start_time": None})
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.ADD_EVENT)
        self.assertEqual(intent.values["title"], "Dentist")

    def test_remember_confirmation_maps_to_bare_remember_intent(self):
        confirmation = ConfirmationResult("remember_task", {"title": "Email advisor"})
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.REMEMBER)

    def test_create_project_confirmation_with_name_carries_values_through(self):
        """Regression: when the orchestrator already extracted name+description
        in one shot (e.g. 'the name is Trend Analyzer, it analyzes trends for
        textiles...'), the mapped intent must carry them - a bare
        Intent(NEW_PROJECT, raw=text) would re-arm dialogue.py's multi-turn
        project_name slot and swallow the user's NEXT reply (their actual
        confirmation word) as if it were the project name, so the project
        never actually saves. See jarvix.log 2026-08-10 for the live failure."""
        confirmation = ConfirmationResult(
            "create_project",
            {"name": "Trend Analyzer", "description": "Analyzes textile trends before they happen."},
        )
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.NEW_PROJECT)
        self.assertEqual(intent.values["name"], "Trend Analyzer")
        self.assertEqual(intent.values["description"], "Analyzes textile trends before they happen.")

    def test_create_project_confirmation_without_name_falls_back_to_bare_intent(self):
        """If the orchestrator somehow didn't extract a name, the old
        re-ask-from-scratch behavior is the correct fallback."""
        confirmation = ConfirmationResult("create_project", {"description": "Something, not sure what yet."})
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            intent = intent_router.parse("some unmatched phrase", use_llm=True)
        self.assertEqual(intent.name, intent_router.NEW_PROJECT)
        self.assertIsNone(intent.values)

    def test_use_llm_false_never_touches_orchestrator(self):
        with patch.object(intent_router, "orchestrate") as mock_orchestrate:
            intent_router.parse("totally unmatched gibberish text", use_llm=False)
        mock_orchestrate.assert_not_called()


class ConfirmationWordFixTests(unittest.TestCase):
    """Spec case 4/10/11/12: the exact bug reported - 'Yeah, save this.'
    silently discarding a pending project. Regression-tests the fix in
    VoiceDialogue._classify_confirmation."""

    def test_yeah_save_this_confirms_pending_project(self):
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Jarvix", "description": "A voice assistant.", "status": "active"})
        captured = []
        response = dialogue.handle("Yeah, save this.", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.NEW_PROJECT)
        self.assertIsNone(dialogue.pending)

    def test_yes_please_confirms_pending_reminder(self):
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("remember_confirm", {"title": "Email advisor"})
        captured = []
        response = dialogue.handle("yes please", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.REMEMBER)

    def test_exactly_save_this_confirms_pending_project(self):
        """Regression (2026-08-10 live bug): 'Exactly. Save this.' fell
        through to 'ambiguous' because neither word was in _YES_WORDS,
        silently failing to save a fully-described project."""
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Trend Analyzer", "description": "Textile trend analysis.", "status": "active"})
        captured = []
        response = dialogue.handle("Exactly. Save this.", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.NEW_PROJECT)
        self.assertIsNone(dialogue.pending)

    def test_thats_correct_confirms_pending_project(self):
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Jarvix", "description": "A voice assistant.", "status": "active"})
        captured = []
        response = dialogue.handle("Yes, that's correct.", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.NEW_PROJECT)

    def test_clean_no_still_discards_pending_project(self):
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Jarvix"})
        captured = []
        response = dialogue.handle("no, don't", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "Okay, I didn't save anything.")
        self.assertEqual(captured, [])
        self.assertIsNone(dialogue.pending)

    def test_ambiguous_reply_routes_through_orchestrator_and_keeps_pending(self):
        from app.brain import dialogue as dialogue_module

        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Jarvix", "description": "A voice assistant.", "status": "active"})
        with patch.object(
            dialogue_module,
            "orchestrate",
            return_value=AnswerResult("Got it - I'll call it JarvixOS instead. Save this?"),
        ):
            response = dialogue.handle("yeah, but call it JarvixOS instead", lambda i: "unexpected")
        self.assertIn("JarvixOS", response)
        # Ambiguous replies never silently discard the pending action - it's
        # either confirmed, explicitly declined, or kept alive for another try.
        self.assertIsNotNone(dialogue.pending)


class NewProjectOneShotConfirmationTests(unittest.TestCase):
    """Regression for a live bug (2026-08-10): describing a new project in
    ONE utterance with enough detail for the orchestrator to call
    create_project directly (name + description already known) must land
    straight on the project_confirm state - not re-arm project_name and wait
    for another turn, which swallows the user's actual confirmation reply
    ("Exactly. Save this.") as if it were the project name, so the project
    never saves and the orchestrator just re-asks to confirm forever."""

    def test_one_shot_description_then_yes_actually_saves(self):
        confirmation = ConfirmationResult(
            "create_project",
            {
                "name": "Trend Analyzer",
                "description": "Analyzes textile trends in Surat sarees before they happen and generates designs.",
            },
        )
        dialogue = VoiceDialogue()
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            prompt = dialogue.handle(
                "So the name is Trend Analyzer, for the textile industry it analyzes the trend before it happens.",
                lambda i: "unexpected",
            )
        # Must go straight to confirmation, not "What's the project called?"
        self.assertIn("Trend Analyzer", prompt)
        self.assertIn("Save this?", prompt)
        self.assertEqual(dialogue.pending.kind, "project_confirm")
        self.assertEqual(dialogue.pending.values["name"], "Trend Analyzer")

        captured = []
        response = dialogue.handle("Exactly. Save this.", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.NEW_PROJECT)
        self.assertEqual(captured[0].values["name"], "Trend Analyzer")
        self.assertIsNone(dialogue.pending)

    def test_one_shot_description_without_name_still_asks(self):
        """Fallback safety net: if the orchestrator's args are missing a
        name for some reason, the old ask-first behavior must still apply."""
        confirmation = ConfirmationResult("create_project", {"description": "Some project, unclear what yet."})
        dialogue = VoiceDialogue()
        with patch.object(intent_router, "orchestrate", return_value=confirmation):
            prompt = dialogue.handle("I want to start a new project", lambda i: "unexpected")
        self.assertEqual(prompt, "What's the project called?")
        self.assertEqual(dialogue.pending.kind, "project_name")


class RollingHistoryTests(unittest.TestCase):
    """Spec section 6: the last 5 conversational turns are tracked and handed
    to the orchestrator, bounded so it never grows unbounded."""

    def test_history_is_recorded_and_bounded_to_five_turns(self):
        dialogue = VoiceDialogue()
        for i in range(4):
            dialogue.handle(f"open cursor", lambda i: "Opening Cursor.")
        self.assertLessEqual(len(dialogue.history), 5)
        self.assertTrue(any(t.role == "user" for t in dialogue.history))
        self.assertTrue(any(t.role == "assistant" for t in dialogue.history))


if __name__ == "__main__":
    unittest.main()
