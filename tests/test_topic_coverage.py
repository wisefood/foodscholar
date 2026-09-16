"""Tests for the topic-coverage checklist.

The two Round 2 cholesterol findings are pinned: the answer should separate
dietary from blood cholesterol and explain HDL/LDL, and it should not stop at
dietary cholesterol when saturated fat, refined carbohydrate, fibre, plant
sterols and physical activity all move the number.
"""
import unittest

from services.topic_coverage import (
    TOPICS,
    coverage_for,
    coverage_prompt_lines,
    missing_coverage,
)


class TestTopicMatching(unittest.TestCase):
    def test_cholesterol_questions_match(self):
        for question in (
            "Does eating eggs raise cholesterol?",
            "how do I lower my LDL",
            "what is a good lipid profile",
            "should I take statins",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(coverage_for(question))

    def test_unrelated_questions_do_not(self):
        for question in ("How much fibre should I eat?", "is coffee bad for me", ""):
            with self.subTest(question=question):
                self.assertIsNone(coverage_for(question))

    def test_the_registry_stays_small(self):
        """Each entry claims we know what a complete answer looks like.

        A failure here is a prompt to check the new entry had domain sign-off,
        not to raise the number.
        """
        self.assertLessEqual(len(TOPICS), 5)


class TestPromptLines(unittest.TestCase):
    def test_both_reported_gaps_are_named(self):
        text = " ".join(coverage_prompt_lines("how do I lower cholesterol")).lower()
        # The distinction the evaluator asked for.
        self.assertIn("dietary cholesterol", text)
        self.assertIn("blood cholesterol", text)
        self.assertIn("ldl", text)
        # The factors the answer stopped short of.
        for factor in ("saturated fat", "refined carbohydrate", "fibre",
                       "sterols", "physical activity"):
            self.assertIn(factor, text)

    def test_nothing_is_injected_for_other_questions(self):
        self.assertEqual(coverage_prompt_lines("how much protein per day"), [])

    def test_the_checklist_never_demands_unsupported_claims(self):
        """It steers what the answer reaches for, not what it asserts."""
        text = " ".join(coverage_prompt_lines("cholesterol")).lower()
        self.assertIn("where the retrieved sources support it", text)


class TestMissingCoverage(unittest.TestCase):
    def test_a_partial_answer_reports_what_it_skipped(self):
        answer = (
            "Dietary cholesterol has a modest effect on blood cholesterol. "
            "Saturated fat matters more, and soluble fibre helps."
        )
        missing = missing_coverage("cholesterol advice", answer)
        self.assertIn("refined carbohydrate and free sugars", missing)
        self.assertIn("physical activity", missing)

    def test_soluble_fibre_counts_as_fibre(self):
        """Matching is on explicit terms, not the first words of the label."""
        answer = "soluble fibre helps"
        self.assertNotIn(
            "dietary fibre, especially soluble fibre",
            missing_coverage("cholesterol", answer),
        )

    def test_exercise_counts_as_physical_activity(self):
        self.assertNotIn(
            "physical activity", missing_coverage("cholesterol", "take regular exercise")
        )

    def test_a_complete_answer_reports_nothing(self):
        answer = (
            "Dietary cholesterol differs from blood cholesterol. Saturated fat, "
            "replaced with unsaturated fat, matters most. Cut free sugars, eat "
            "soluble fibre, try plant stanols, and take regular exercise."
        )
        self.assertEqual(missing_coverage("cholesterol", answer), [])

    def test_off_topic_answers_are_not_scored(self):
        self.assertEqual(missing_coverage("how much protein", "anything"), [])

    def test_empty_answer_is_not_scored(self):
        self.assertEqual(missing_coverage("cholesterol", ""), [])


if __name__ == "__main__":
    unittest.main()
