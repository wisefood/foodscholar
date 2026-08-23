"""Tests for the thread memory: sliding verbatim window + compacted summary."""

import unittest
from unittest.mock import patch

from services.qa_service import QAService


class _FakeCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ttl=None):
        self.store[key] = value
        return True

    def delete(self, key):
        self.store.pop(key, None)
        return True


class _FakeSummaryLLM:
    def invoke(self, _prompt, config=None):
        class _Response:
            content = "- User asked about fiber.\n- Answer: 25-30 g/day."
        return _Response()


class ComposePriorConversationTests(unittest.TestCase):
    def test_summary_and_turns_compose_in_order(self):
        text = QAService.compose_prior_conversation(
            {
                "summary": "- Earlier: fiber basics.",
                "turns": [
                    {"question": "How much fiber?", "answer": "About 25-30 g."}
                ],
            }
        )
        self.assertIn("- Earlier: fiber basics.", text)
        self.assertIn("MOST RECENT EXCHANGES (verbatim):", text)
        self.assertIn("User: How much fiber?", text)
        self.assertIn("FoodScholar: About 25-30 g.", text)
        # Summary precedes the verbatim window.
        self.assertLess(
            text.index("fiber basics"), text.index("MOST RECENT")
        )

    def test_empty_state_composes_to_none(self):
        self.assertIsNone(
            QAService.compose_prior_conversation({"summary": None, "turns": []})
        )

    def test_summary_only_and_turns_only_both_work(self):
        self.assertEqual(
            QAService.compose_prior_conversation(
                {"summary": "- S.", "turns": []}
            ),
            "- S.",
        )
        turns_only = QAService.compose_prior_conversation(
            {"summary": None, "turns": [{"question": "Q?", "answer": "A."}]}
        )
        self.assertTrue(turns_only.startswith("MOST RECENT EXCHANGES"))


class ConversationTurnStorageTests(unittest.TestCase):
    def setUp(self):
        self.service = QAService(cache_enabled=False)
        self.service.cache_manager = _FakeCache()
        self.service._conversation_summary_llm = _FakeSummaryLLM()

    def _update(self, question, answer):
        self.service._update_conversation_summary(
            thread_id="t1",
            previous_summary=None,
            question=question,
            answer_text=answer,
            language="en",
            trace_context={"user_id": None},
        )

    def test_turns_slide_keeping_the_most_recent_two(self):
        self._update("Q1?", "A1.")
        self._update("Q2?", "A2.")
        self._update("Q3?", "A3.")

        state = self.service._load_conversation_state("t1")
        self.assertEqual(
            [turn["question"] for turn in state["turns"]], ["Q2?", "Q3?"]
        )
        self.assertTrue(state["summary"].startswith("- User asked"))

    def test_long_answers_are_truncated_in_the_window(self):
        self._update("Q?", "x" * 5000)
        state = self.service._load_conversation_state("t1")
        self.assertEqual(len(state["turns"][0]["answer"]), 1500)

    def test_no_thread_id_loads_empty_state(self):
        state = self.service._load_conversation_state(None)
        self.assertEqual(state, {"summary": None, "turns": []})


if __name__ == "__main__":
    unittest.main()
