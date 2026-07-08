"""MemoryService — nudge policy on QA turns (no LLM, no platform calls)."""

import unittest
from unittest.mock import patch

from services.memory_service import MemoryService


PROFILE = {
    "nutritional_preferences": {
        "food_likes": ["chicken"],
        "food_dislikes": ["olives"],
    },
    "allergies": ["peanuts"],
    "dietary_groups": ["vegetarian"],
    "properties": {
        "memory_optouts": ["cilantro"],
        "dietary_goals": [{"slug": "reduce_sugar", "label": "cut back on sugar"}],
    },
}


def _suggest(service, candidates):
    """Run suggest() with the extractor and profile lookup stubbed out."""
    with patch.object(service, "_extract", return_value=candidates), patch.object(
        service, "_fetch_profile", return_value=PROFILE
    ):
        return service.suggest("member-1", "irrelevant — extractor is stubbed")


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.service = MemoryService()

    def test_high_confidence_like_nudges(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": "lentils", "confidence": "high",
             "statement": "It seems you love lentils — remember this?"},
        ])
        self.assertEqual([(s["kind"], s["value"]) for s in out], [("like", "lentils")])
        self.assertTrue(out[0]["id"])

    def test_low_confidence_dropped_except_allergy_hints(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": "kale", "confidence": "medium"},
            {"kind": "allergy_hint", "value": "shellfish", "confidence": "low"},
        ])
        self.assertEqual([s["kind"] for s in out], ["allergy_hint"])

    def test_known_same_kind_and_optouts_filtered(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": "chicken", "confidence": "high"},      # already liked
            {"kind": "allergy_hint", "value": "peanuts", "confidence": "high"},  # known allergy
            {"kind": "like", "value": "cilantro", "confidence": "high"},     # opted out
        ])
        self.assertEqual(out, [])

    def test_contradiction_still_nudges_with_callout(self):
        """A dislike of something in LIKES must nudge (same-kind dedupe only)."""
        out = _suggest(self.service, [
            {"kind": "dislike", "value": "chicken", "confidence": "high"},
        ])
        self.assertEqual(len(out), 1)
        self.assertIn("currently in your likes", out[0]["statement"])

    def test_capped_at_two_per_turn(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": f"item-{i}", "confidence": "high"}
            for i in range(5)
        ])
        self.assertEqual(len(out), 2)

    def test_invalid_kinds_and_failures_degrade_to_empty(self):
        out = _suggest(self.service, [
            {"kind": "standing_seed", "value": "pastitsio", "confidence": "high"},
            {"kind": "like", "value": "", "confidence": "high"},
        ])
        self.assertEqual(out, [])
        with patch.object(self.service, "_extract", side_effect=RuntimeError("boom")):
            self.assertEqual(self.service.suggest("member-1", "q"), [])

    def test_decide_rejects_invalid_payloads(self):
        with self.assertRaises(ValueError):
            self.service.decide("member-1", "standing_seed", "pastitsio", "accept")
        with self.assertRaises(ValueError):
            self.service.decide("member-1", "like", "  ", "accept")

    # --- goal kind -------------------------------------------------------- #

    def test_high_confidence_goal_nudges(self):
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_fat", "confidence": "high",
             "statement": "It sounds like you want to reduce fat — track this goal?"},
        ])
        self.assertEqual([(s["kind"], s["value"]) for s in out],
                         [("goal", "reduce_fat")])

    def test_known_goal_deduped(self):
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_sugar", "confidence": "high"},  # already a goal
        ])
        self.assertEqual(out, [])

    def test_low_confidence_goal_dropped(self):
        out = _suggest(self.service, [
            {"kind": "goal", "value": "lose_weight", "confidence": "medium"},
        ])
        self.assertEqual(out, [])

    # --- dietary_pattern kind -------------------------------------------- #

    def test_high_confidence_pattern_nudges(self):
        out = _suggest(self.service, [
            {"kind": "dietary_pattern", "value": "keto", "confidence": "high"},
        ])
        self.assertEqual([(s["kind"], s["value"]) for s in out],
                         [("dietary_pattern", "keto")])

    def test_known_pattern_deduped(self):
        out = _suggest(self.service, [
            {"kind": "dietary_pattern", "value": "vegetarian", "confidence": "high"},  # in dietary_groups
        ])
        self.assertEqual(out, [])

    def test_new_kinds_accepted_by_decide(self):
        # decide() must not reject the new kinds as invalid payloads.
        with patch.object(self.service, "_apply", return_value=True) as ap:
            self.assertTrue(
                self.service.decide("m", "goal", "reduce_fat", "accept"))
            ap.assert_called_once()
        with patch.object(self.service, "_apply", return_value=True):
            self.assertTrue(
                self.service.decide("m", "dietary_pattern", "keto", "accept"))


if __name__ == "__main__":
    unittest.main()
