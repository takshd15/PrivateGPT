"""Tests for the on_event streaming callback added to orchestrate() (and
threaded through intent_router.parse()/dialogue.VoiceDialogue.handle()) so
app/server.py can push live progress to the frontend. Same offline/mocked
pattern as tests/test_orchestrator.py - no real network/DB calls."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.brain import tools_registry
from app.brain.orchestrator import orchestrate


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _assistant_message(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


def _mock_tool(name: str, **kwargs):
    return patch.object(tools_registry.get(name), "handler", **kwargs)


class OrchestratorEventSequenceTests(unittest.TestCase):
    def test_single_tool_call_emits_thinking_tool_call_tool_result_answer(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_news", "{}")]),
            _assistant_message(content="Markets rallied today."),
        ]
        events = []
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses), _mock_tool(
            "get_news", return_value=[{"title": "Markets rallied"}]
        ):
            result = orchestrate("What's the news?", on_event=events.append)

        types = [e["type"] for e in events]
        self.assertEqual(types, ["thinking", "tool_call", "tool_result", "thinking", "answer"])
        self.assertEqual(events[1]["name"], "get_news")
        self.assertEqual(events[2]["name"], "get_news")
        self.assertEqual(events[2]["result"], [{"title": "Markets rallied"}])
        self.assertEqual(result.text, events[-1]["text"])

    def test_multi_tool_chain_emits_one_pair_per_tool_call(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_goals", "{}")]),
            _assistant_message(tool_calls=[_tool_call("c2", "get_opportunities", "{}")]),
            _assistant_message(content="Y Combinator is the best fit."),
        ]
        events = []
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses), _mock_tool(
            "get_goals", return_value=[{"title": "Build a startup"}]
        ), _mock_tool("get_opportunities", return_value=[{"name": "Y Combinator"}]):
            orchestrate("What's the best opportunity?", on_event=events.append)

        tool_call_names = [e["name"] for e in events if e["type"] == "tool_call"]
        tool_result_names = [e["name"] for e in events if e["type"] == "tool_result"]
        self.assertEqual(tool_call_names, ["get_goals", "get_opportunities"])
        self.assertEqual(tool_result_names, ["get_goals", "get_opportunities"])
        self.assertEqual(events[-1]["type"], "answer")

    def test_clarification_emits_clarification_event_not_answer(self):
        responses = [
            _assistant_message(
                tool_calls=[_tool_call("c1", "ask_clarifying_question", '{"question": "Which project?"}')]
            ),
        ]
        events = []
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses):
            orchestrate("log some progress", on_event=events.append)

        self.assertEqual(events[-1], {"type": "clarification", "question": "Which project?"})

    def test_confirmation_required_tool_emits_confirmation_event(self):
        responses = [
            _assistant_message(
                tool_calls=[_tool_call("c1", "create_calendar_event", '{"title": "Dentist", "date": "2026-08-10"}')]
            ),
        ]
        events = []
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses):
            orchestrate("add an event", on_event=events.append)

        self.assertEqual(events[-1]["type"], "confirmation")
        self.assertEqual(events[-1]["tool"], "create_calendar_event")

    def test_on_event_omitted_changes_nothing(self):
        """The default (no on_event) must still work exactly as before - this
        is the existing test_orchestrator.py contract, re-asserted here to
        pin that adding the parameter didn't change default behavior."""
        responses = [_assistant_message(content="Fine either way.")]
        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses):
            result = orchestrate("hello")
        self.assertEqual(result.text, "Fine either way.")

    def test_raising_on_event_callback_does_not_break_the_loop(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "get_time", "{}")]),
            _assistant_message(content="It's noon."),
        ]

        def bad_callback(event):
            raise RuntimeError("frontend disconnected")

        with patch("app.brain.orchestrator.ask_llm_message", side_effect=responses), _mock_tool(
            "get_time", return_value="12:00"
        ):
            result = orchestrate("what time is it", on_event=bad_callback)
        self.assertEqual(result.text, "It's noon.")


class IntentRouterAndDialogueEventForwardingTests(unittest.TestCase):
    """The callback must actually reach orchestrate() through both entry
    points the frontend/backend use - intent_router.parse() directly, and
    VoiceDialogue.handle() (including its ambiguous-confirmation path)."""

    def test_parse_forwards_on_event_to_orchestrator(self):
        from app.brain import intent_router
        from app.brain.orchestrator import AnswerResult

        captured = {}

        def fake_orchestrate(text, history=None, pending_note=None, timeout=15, on_event=None):
            captured["on_event"] = on_event
            if on_event:
                on_event({"type": "thinking"})
            return AnswerResult("ok")

        with patch.object(intent_router, "orchestrate", side_effect=fake_orchestrate):
            events = []
            intent_router.parse("unmatched text", use_llm=True, on_event=events.append)

        self.assertIsNotNone(captured["on_event"])
        self.assertEqual(events, [{"type": "thinking"}])

    def test_dialogue_handle_forwards_on_event_for_fresh_intent(self):
        from app.brain.dialogue import VoiceDialogue

        dialogue = VoiceDialogue()
        events = []

        def fake_parse(text, use_llm=True, history=None, on_event=None):
            from app.brain import intent_router as ir

            if on_event:
                on_event({"type": "thinking"})
            return ir.Intent(ir.TIME, raw=text)

        with patch("app.brain.dialogue.intent_router.parse", side_effect=fake_parse):
            dialogue.handle("what time is it", lambda i: "It's noon.", on_event=events.append)

        self.assertIn({"type": "thinking"}, events)

    def test_dialogue_handle_forwards_on_event_through_ambiguous_confirmation(self):
        from app.brain.dialogue import Pending, VoiceDialogue
        from app.brain.orchestrator import AnswerResult

        dialogue = VoiceDialogue()
        dialogue.pending = Pending("project_confirm", {"name": "Jarvix"})
        events = []

        with patch("app.brain.dialogue.orchestrate", return_value=AnswerResult("Confirm again?")) as mock_orch:
            dialogue.handle("yeah, but call it JarvixOS", lambda i: "unexpected", on_event=events.append)

        self.assertEqual(mock_orch.call_args.kwargs["on_event"], events.append)


if __name__ == "__main__":
    unittest.main()
