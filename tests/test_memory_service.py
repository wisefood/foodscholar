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
    "properties": {"memory_optouts": ["cilantro"]},
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


if __name__ == "__main__":
    unittest.main()
