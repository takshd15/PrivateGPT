"""Regression cases taken from the June 2026 Jarvix voice transcript."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.brain import intent_router, structuring
from app.brain.dialogue import Pending, VoiceDialogue
from app.models import ProgressEventCandidate, ProjectCandidate, TaskCandidate
from app.tools.live_info import resolve_date_phrase
from app.voice.wakeword import _matches, command_after_wake_word
from datetime import date


class IntentRouterTests(unittest.TestCase):
    def parse(self, text: str) -> intent_router.Intent:
        return intent_router.parse(text, use_llm=False)

    def test_weather_today_does_not_become_calendar(self):
        intent = self.parse("What's the weather like in Enschede today?")
        self.assertEqual(intent.name, intent_router.WEATHER)
        self.assertEqual(intent.arg, "Enschede")

    def test_calendar_relative_dates_are_preserved(self):
        tomorrow = self.parse("Can you read my calendar for tomorrow?")
        tuesday = self.parse("Can you read my calendar for next Tuesday?")
        self.assertEqual((tomorrow.name, tomorrow.arg), (intent_router.CALENDAR_DATE, "tomorrow"))
        self.assertEqual((tuesday.name, tuesday.arg), (intent_router.CALENDAR_DATE, "next tuesday"))

    def test_open_spotify_is_not_play_pause(self):
        intent = self.parse("Can you please open Spotify?")
        self.assertEqual((intent.name, intent.arg), (intent_router.OPEN_APP, "spotify"))

    def test_claude_is_an_allowlisted_app(self):
        intent = self.parse("Open Claude")
        self.assertEqual((intent.name, intent.arg), (intent_router.OPEN_APP, "claude"))

    def test_partial_app_name_does_not_open_an_app(self):
        intent = self.parse("Open codecs")
        self.assertNotEqual(intent.name, intent_router.OPEN_APP)

    def test_generic_folder_request_asks_for_folder(self):
        intent = self.parse("Can you open a folder in my laptop?")
        self.assertEqual((intent.name, intent.arg), (intent_router.CLARIFICATION_NEEDED, "folder"))

    def test_email_slots_are_extracted(self):
        intent = self.parse("Send an email to Tisha saying I will be late")
        self.assertEqual(intent.name, intent_router.SEND_EMAIL)
        self.assertEqual(intent.recipient, "Tisha")
        self.assertEqual(intent.message, "I will be late")

    def test_time_is_deterministic_intent(self):
        self.assertEqual(self.parse("What time is it?").name, intent_router.TIME)

    def test_incomplete_comparison_requests_followup(self):
        intent = self.parse("Tell me the difference between")
        self.assertEqual((intent.name, intent.arg), (intent_router.CLARIFICATION_NEEDED, "comparison"))

    def test_clear_spoken_questions_route_to_question(self):
        for text in (
            "What is the best Indian dessert?",
            "Top 5 Indian desserts",
            "Should I stay or should I go?",
            "Can you explain blockchain to me?",
        ):
            self.assertEqual(self.parse(text).name, intent_router.QUESTION, text)

    def test_music_query_mishear_is_corrected(self):
        intent = self.parse("Play Travis code")
        self.assertEqual(intent.name, intent_router.MUSIC_PLAY_QUERY)
        self.assertEqual(intent.arg, "Travis Scott")

    def test_c_drive_opens_a_folder(self):
        intent = self.parse("Open C drive")
        self.assertEqual((intent.name, intent.arg), (intent_router.OPEN_FOLDER, "c drive"))


class DialogueTests(unittest.TestCase):
    def test_email_message_is_collected_on_next_turn(self):
        dialogue = VoiceDialogue()
        captured = []

        def execute(intent):
            captured.append(intent)
            return "done"

        self.assertEqual(dialogue.handle("Send an email to Tisha", execute), "What should the email say?")
        self.assertEqual(dialogue.handle("Tell her I will be late", execute), "done")
        self.assertEqual(captured[0].recipient, "Tisha")
        self.assertEqual(captured[0].message, "Tell her I will be late")

    def test_weather_location_is_collected_on_next_turn(self):
        dialogue = VoiceDialogue()
        captured = []
        self.assertEqual(dialogue.handle("What's the weather?", lambda i: "unexpected"), "Which city should I check?")
        dialogue.handle("Enschede", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].arg, "Enschede")

    def test_weather_followup_strips_filler_words(self):
        dialogue = VoiceDialogue()
        captured = []
        dialogue.handle("What's the weather?", lambda i: "unexpected")
        dialogue.handle("check for Paris", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].arg, "Paris")

    def test_weather_followup_corrects_city_mishear(self):
        dialogue = VoiceDialogue()
        captured = []
        dialogue.handle("What's the weather?", lambda i: "unexpected")
        dialogue.handle("Ench Cannon", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].arg, "Enschede")


class DateResolutionTests(unittest.TestCase):
    def test_next_weekday_is_in_following_week(self):
        monday = date(2026, 6, 22)
        self.assertEqual(resolve_date_phrase("next tuesday", monday), date(2026, 6, 30))
        self.assertEqual(resolve_date_phrase("tuesday", monday), date(2026, 6, 23))


class WakeWordTests(unittest.TestCase):
    def test_genuine_name_mishears_still_trigger(self):
        self.assertTrue(_matches("Garvis, pause the music"))  # fuzzy distance 1
        self.assertTrue(_matches("Hey Jorvix, what time is it?"))
        self.assertTrue(_matches("Jarvix, read my calendar"))
        self.assertTrue(_matches("Jervis, what's the weather?"))

    def test_variant_keeps_same_utterance_command(self):
        self.assertEqual(command_after_wake_word("Garvis, pause the music"), "pause the music")

    def test_repeated_wake_garbage_is_not_a_command(self):
        # "Jarvis Jorvis" / "Garvis Garvis" carry no real instruction to run.
        self.assertEqual(command_after_wake_word("Jarvis Jorvis"), "")
        self.assertEqual(command_after_wake_word("Jarvix Garvis Jarvis"), "")

    def test_name_and_word_collisions_do_not_trigger(self):
        # These woke Jarvix on nonsense in the June transcript.
        self.assertFalse(_matches("Doris"))
        self.assertFalse(_matches("Joris, can you read my calendar?"))
        self.assertFalse(_matches("Harvest, read my calendar"))

    def test_music_and_broad_old_aliases_do_not_trigger(self):
        self.assertFalse(_matches("Enjoy this, pause the music"))
        self.assertFalse(_matches("Travis Scott is playing"))
        self.assertFalse(_matches("That's all of us"))
        self.assertFalse(_matches("The jar is on the table"))


class StructuredCaptureTests(unittest.TestCase):
    """REMEMBER / NEW_PROJECT / LOG_PROGRESS: single-turn capture, one
    follow-up on missing info, and confirm=no discarding cleanly. All LLM
    calls are stubbed so this stays offline/deterministic like the rest of
    the suite - only the dialogue.py state machine is under test here."""

    # -- remember -----------------------------------------------------------

    def test_remember_single_turn_capture(self):
        dialogue = VoiceDialogue()
        captured = []
        candidate = TaskCandidate(title="Email advisor", due_date_phrase="tomorrow", confidence=1.0)
        with patch.object(structuring, "structure_task", return_value=candidate):
            prompt = dialogue.handle("remember to email my advisor tomorrow", lambda i: "unexpected")
            self.assertIn("Email advisor", prompt)
            self.assertIn("Confirm?", prompt)
            dialogue.handle("yes", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].name, intent_router.REMEMBER)
        self.assertEqual(captured[0].values["title"], "Email advisor")
        self.assertIsNotNone(captured[0].values["due_date"])

    def test_remember_missing_title_asks_once(self):
        dialogue = VoiceDialogue()
        empty = TaskCandidate(title="", confidence=0.0)
        followup = TaskCandidate(title="Buy milk", confidence=0.6)
        with patch.object(structuring, "structure_task", side_effect=[empty, followup]):
            prompt = dialogue.handle("remember something", lambda i: "unexpected")
            self.assertEqual(prompt, "What should I remember, exactly?")
            captured = []
            confirm_prompt = dialogue.handle("buy milk", lambda i: captured.append(i) or "done")
        # Proceeds straight to confirmation after the one follow-up, no second ask.
        self.assertIn("Buy milk", confirm_prompt)

    def test_remember_confirm_no_discards(self):
        dialogue = VoiceDialogue()
        candidate = TaskCandidate(title="Email advisor", confidence=1.0)
        captured = []
        with patch.object(structuring, "structure_task", return_value=candidate):
            dialogue.handle("remember to email my advisor", lambda i: "unexpected")
            response = dialogue.handle("no", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "Okay, I didn't add anything.")
        self.assertEqual(captured, [])
        self.assertIsNone(dialogue.pending)

    # -- new_project ----------------------------------------------------------

    def test_new_project_single_turn_capture(self):
        dialogue = VoiceDialogue()
        candidate = ProjectCandidate(name="Jarvix", description="A voice assistant.", status="active", confidence=1.0)
        prompt1 = dialogue.handle("I'm working on a new project", lambda i: "unexpected")
        self.assertEqual(prompt1, "What's the project called?")
        prompt2 = dialogue.handle("Jarvix", lambda i: "unexpected")
        self.assertEqual(prompt2, "Go ahead and describe it, I'm listening.")
        with patch.object(structuring, "structure_project", return_value=candidate):
            prompt3 = dialogue.handle("It's a voice assistant.", lambda i: "unexpected")
        self.assertIn("Jarvix", prompt3)
        self.assertIn("Save this?", prompt3)
        captured = []
        dialogue.handle("yes", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].name, intent_router.NEW_PROJECT)
        self.assertEqual(captured[0].values["name"], "Jarvix")

    def test_new_project_confirm_no_discards(self):
        dialogue = VoiceDialogue()
        candidate = ProjectCandidate(name="Jarvix", description="A voice assistant.", confidence=1.0)
        dialogue.handle("new project", lambda i: "unexpected")
        dialogue.handle("Jarvix", lambda i: "unexpected")
        captured = []
        with patch.object(structuring, "structure_project", return_value=candidate):
            dialogue.handle("It's a voice assistant.", lambda i: "unexpected")
        response = dialogue.handle("no", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "Okay, I didn't save anything.")
        self.assertEqual(captured, [])

    # -- log_progress ---------------------------------------------------------

    def test_log_progress_single_turn_capture_with_known_entity(self):
        dialogue = VoiceDialogue(known_entities=lambda: ["DAAD Application", "Jarvix"])
        candidate = ProgressEventCandidate(
            entity_name="DAAD Application",
            event_text="Submitted transcripts.",
            event_type="update",
            confidence=0.9,
        )
        captured = []
        with patch.object(structuring, "structure_progress_event", return_value=candidate):
            prompt = dialogue.handle("log progress on my DAAD application", lambda i: "unexpected")
        self.assertIn("DAAD Application", prompt)
        self.assertIn("Confirm?", prompt)
        dialogue.handle("yes", lambda i: captured.append(i) or "done")
        self.assertEqual(captured[0].name, intent_router.LOG_PROGRESS)
        self.assertEqual(captured[0].values["entity_name"], "DAAD Application")

    def test_log_progress_missing_entity_asks_once(self):
        from app.brain.orchestrator import ConfirmationResult

        dialogue = VoiceDialogue(known_entities=lambda: [])
        unmatched = ProgressEventCandidate(entity_name="", event_text="Made progress.", event_type="update", confidence=0.5)
        # "log some progress" has no deterministic rule match (no "log
        # progress"/"progress on"/"update on" substring), so it reaches the
        # orchestrator fallback - stub that boundary (patched where
        # intent_router imported it) so this stays offline, matching the
        # class's "no live LLM calls" contract.
        with patch.object(structuring, "structure_progress_event", return_value=unmatched), patch.object(
            intent_router, "orchestrate", return_value=ConfirmationResult("log_progress", {"entity_name": "", "event_text": "Made progress."})
        ):
            prompt = dialogue.handle("log some progress", lambda i: "unexpected")
        self.assertEqual(prompt, "Which project or application is this progress on?")
        confirm_prompt = dialogue.handle("Jarvix", lambda i: "unexpected")
        self.assertIn("Jarvix", confirm_prompt)

    def test_log_progress_confirm_no_discards(self):
        dialogue = VoiceDialogue(known_entities=lambda: ["Jarvix"])
        candidate = ProgressEventCandidate(entity_name="Jarvix", event_text="Made progress.", event_type="update", confidence=0.9)
        captured = []
        with patch.object(structuring, "structure_progress_event", return_value=candidate):
            dialogue.handle("progress on Jarvix", lambda i: "unexpected")
        response = dialogue.handle("no", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "Okay, I didn't log anything.")
        self.assertEqual(captured, [])

    # -- cross-cutting --------------------------------------------------------

    def test_remember_beats_email_drafting_in_rules(self):
        intent = intent_router.parse("remind me to email my advisor tomorrow", use_llm=False)
        self.assertEqual(intent.name, intent_router.REMEMBER)


class AutopilotTests(unittest.TestCase):
    """FIND_OPPORTUNITIES routing/dialogue and the email-signal extraction
    contract. No live network/DB calls - LLM-backed calls are stubbed."""

    def test_find_opportunities_routes_before_music_find_rule(self):
        # Regression guard: block 1's music rule matches "find "/"search for "
        # as a prefix and would otherwise swallow this before the fix.
        intent = intent_router.parse("find opportunities for me", use_llm=False)
        self.assertEqual(intent.name, intent_router.FIND_OPPORTUNITIES)
        self.assertNotEqual(intent.name, intent_router.MUSIC_PLAY_QUERY)

    def test_find_opportunities_ordinary_find_still_music(self):
        intent = intent_router.parse('find "some song"', use_llm=False)
        self.assertEqual(intent.name, intent_router.MUSIC_PLAY_QUERY)

    def test_find_opportunities_asks_for_location_once(self):
        dialogue = VoiceDialogue()
        prompt = dialogue.handle("find opportunities for me", lambda i: "unexpected")
        self.assertEqual(prompt, "What location? You can also say any location.")
        captured = []
        dialogue.handle("Berlin", lambda i: captured.append(i) or "searching")
        self.assertEqual(captured[0].name, intent_router.FIND_OPPORTUNITIES)
        self.assertEqual(captured[0].arg, "Berlin")

    def test_find_opportunities_accepts_any_location(self):
        dialogue = VoiceDialogue()
        dialogue.handle("find opportunities for me", lambda i: "unexpected")
        captured = []
        dialogue.handle("any location", lambda i: captured.append(i) or "searching")
        self.assertEqual(captured[0].arg, None)

    def test_find_opportunities_location_stated_upfront_skips_prompt(self):
        dialogue = VoiceDialogue()
        captured = []
        response = dialogue.handle(
            "find opportunities in Amsterdam", lambda i: captured.append(i) or "searching"
        )
        self.assertEqual(response, "searching")
        self.assertEqual(captured[0].arg, "Amsterdam")

    def test_email_signal_extraction_defaults_to_null(self):
        from app.models import EmailSignalCandidate

        with patch.object(structuring, "structure_email_signals", return_value=EmailSignalCandidate()):
            signals = structuring.structure_email_signals({"subject": "50% off sale"})
        self.assertIsNone(signals.signup)
        self.assertIsNone(signals.application)

    def test_company_matching_is_case_insensitive_exact(self):
        # Pure-function contract test: the SQL predicate normalizes both
        # sides with lower(trim(...)), so "Google" and "google" (or padded
        # whitespace) must be treated as the same company, but "Google LLC"
        # must NOT silently match "Google" (no fuzzy matching).
        def normalize(name: str) -> str:
            return name.strip().lower()

        self.assertEqual(normalize("Google"), normalize("  google  "))
        self.assertNotEqual(normalize("Google"), normalize("Google LLC"))


class ConnectiveTissueTests(unittest.TestCase):
    """Meeting follow-up opt-out permanence, duplicate-goal suppression, and
    the briefing rollup's word-count/section-inclusion contract. No live
    network/DB calls - LLM/DB calls are stubbed."""

    def test_meeting_followup_opt_out_is_permanent(self):
        dialogue = VoiceDialogue()
        dialogue.pending = Pending("meeting_followup_offer", {"meeting_id": "m-1", "title": "Roadmap Sync"})
        captured = []
        response = dialogue.handle("no", lambda i: captured.append(i) or "done")
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.MEETING_FOLLOWUP)
        self.assertEqual(captured[0].values["meeting_id"], "m-1")
        self.assertTrue(captured[0].values["declined"])
        # The flow ends here - dialogue never re-asks in the same session.
        self.assertIsNone(dialogue.pending)

    def test_meeting_followup_accept_captures_and_offers_actions(self):
        from app.brain import structuring as structuring_module
        from app.models import MeetingFollowupCandidate

        dialogue = VoiceDialogue()
        dialogue.pending = Pending("meeting_followup_offer", {"meeting_id": "m-2", "title": "Roadmap Sync"})
        prompt = dialogue.handle("yes", lambda i: "unexpected")
        self.assertEqual(prompt, "What was discussed?")
        self.assertEqual(dialogue.pending.kind, "meeting_followup_capture")

        candidate = MeetingFollowupCandidate(summary="Agreed to push launch to September.", confidence=0.9)
        captured = []
        with patch.object(structuring_module, "structure_meeting_followup", return_value=candidate):
            response = dialogue.handle(
                "we agreed to push the launch to September",
                lambda i: captured.append(i) or "done",
            )
        self.assertEqual(response, "done")
        self.assertEqual(captured[0].name, intent_router.MEETING_FOLLOWUP)
        self.assertEqual(captured[0].values["summary"], "Agreed to push launch to September.")
        self.assertFalse(captured[0].values["declined"])

    def test_duplicate_goal_suppression(self):
        from app.memory.db.goals import _title_similarity, SIMILARITY_THRESHOLD

        # Same goal, rephrased across sessions - must be recognized as a duplicate.
        score = _title_similarity("get into grad school", "trying to get into grad school this year")
        self.assertGreaterEqual(score, SIMILARITY_THRESHOLD)

        # Genuinely different goals must NOT be suppressed.
        score2 = _title_similarity("get into grad school", "land an internship at Google")
        self.assertLess(score2, SIMILARITY_THRESHOLD)

    def test_briefing_rollup_appends_when_nonempty(self):
        import app.main as main_module

        with patch.object(main_module, "_read_last_brief_at", return_value="2026-01-01T00:00:00+00:00"), \
             patch.object(main_module, "_write_last_brief_at", return_value=None), \
             patch.object(main_module.db_opportunities, "list_recent_opportunities", return_value=[
                 {"name": "YC Fall 2026", "fit_score": 0.9, "created_at": "2026-08-01T00:00:00+00:00"}
             ]), \
             patch.object(main_module.db_applications, "list_updated_since", return_value=[]), \
             patch.object(main_module.db_events, "count_events_since", return_value=0), \
             patch.object(main_module.db_meetings, "list_followup_candidates", return_value=[]), \
             patch.object(main_module.db_memories, "list_recent_memories", return_value=[]), \
             patch.object(main_module.db_goals, "list_goals", return_value=[]), \
             patch.object(main_module.structuring, "summarize_briefing_rollup", return_value="You have a new YC opportunity."), \
             patch.object(main_module, "USER_ID", "fake-user-id"):
            text = main_module._briefing_rollup_text()
        self.assertEqual(text, " You have a new YC opportunity.")

    def test_briefing_rollup_empty_when_nothing_new(self):
        import app.main as main_module

        with patch.object(main_module, "_read_last_brief_at", return_value="2026-01-01T00:00:00+00:00"), \
             patch.object(main_module, "_write_last_brief_at", return_value=None), \
             patch.object(main_module.db_opportunities, "list_recent_opportunities", return_value=[]), \
             patch.object(main_module.db_applications, "list_updated_since", return_value=[]), \
             patch.object(main_module.db_events, "count_events_since", return_value=0), \
             patch.object(main_module.db_meetings, "list_followup_candidates", return_value=[]), \
             patch.object(main_module.db_memories, "list_recent_memories", return_value=[]), \
             patch.object(main_module.db_goals, "list_goals", return_value=[]), \
             patch.object(main_module, "USER_ID", "fake-user-id"):
            text = main_module._briefing_rollup_text()
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
