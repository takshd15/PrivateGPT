"""Regression cases taken from the June 2026 Jarvix voice transcript."""

from __future__ import annotations

import unittest

from app.brain import intent_router
from app.brain.dialogue import VoiceDialogue
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


if __name__ == "__main__":
    unittest.main()
