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

    def test_known_same_kind_comes_back_preselected_and_optouts_are_dropped(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": "chicken", "confidence": "high"},      # already liked
            {"kind": "allergy_hint", "value": "peanuts", "confidence": "high"},  # known allergy
            {"kind": "like", "value": "cilantro", "confidence": "high"},     # opted out
        ])
        self.assertEqual(
            [(s["kind"], s["value"], s["already_known"]) for s in out],
            [("like", "chicken", True), ("allergy_hint", "peanuts", True)],
        )
        self.assertIn("already in your likes", out[0]["statement"])
        self.assertIn("already listed among your allergies", out[1]["statement"])

    def test_fresh_nudges_are_not_marked_known(self):
        out = _suggest(self.service, [
            {"kind": "like", "value": "lentils", "confidence": "high"},
        ])
        self.assertFalse(out[0]["already_known"])

    def test_known_value_below_high_confidence_is_not_shown(self):
        """Pre-selected chips obey the same confidence gate as real nudges."""
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_sugar", "confidence": "medium"},  # known goal
            {"kind": "like", "value": "chicken", "confidence": "low"},          # known like
        ])
        self.assertEqual(out, [])

    def test_new_nudges_take_the_slots_before_known_ones(self):
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_sugar", "confidence": "high"},  # known
            {"kind": "like", "value": "lentils", "confidence": "high"},       # new
            {"kind": "like", "value": "tofu", "confidence": "high"},          # new
        ])
        self.assertEqual([s["value"] for s in out], ["lentils", "tofu"])
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_sugar", "confidence": "high"},  # known
            {"kind": "like", "value": "lentils", "confidence": "high"},       # new
        ])
        self.assertEqual(
            [(s["value"], s["already_known"]) for s in out],
            [("lentils", False), ("reduce_sugar", True)],
        )

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

    def test_suggestions_carry_the_question_as_provenance(self):
        """R1-D3/A2: the panel must be able to say why a goal was inferred."""
        with patch.object(
            self.service, "_extract",
            return_value=[{"kind": "goal", "value": "reduce_fat", "confidence": "high"}],
        ), patch.object(self.service, "_fetch_profile", return_value=PROFILE):
            out = self.service.suggest(
                "member-1",
                "We're worried about our family's heart health — is red meat harmful?",
            )
        self.assertEqual(len(out), 1)
        self.assertIn("heart health", out[0]["source_text"])

    def test_accepted_source_text_reaches_the_write(self):
        with patch.object(self.service, "_apply", return_value=True) as apply_mock:
            self.service.decide("m", "goal", "reduce_fat", "accept", "because I asked")
        apply_mock.assert_called_once_with("m", "goal", "reduce_fat", "because I asked")

    # --- goal kind -------------------------------------------------------- #

    def test_high_confidence_goal_nudges(self):
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_fat", "confidence": "high",
             "statement": "It sounds like you want to reduce fat — track this goal?"},
        ])
        self.assertEqual([(s["kind"], s["value"]) for s in out],
                         [("goal", "reduce_fat")])

    def test_known_goal_comes_back_preselected(self):
        """An existing goal is shown as already tracked, never inferred twice."""
        out = _suggest(self.service, [
            {"kind": "goal", "value": "reduce_sugar", "confidence": "high",
             "statement": "It sounds like you want to cut sugar — track this goal?"},
        ])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["already_known"])
        self.assertEqual(out[0]["value"], "reduce_sugar")
        # The model's question is replaced by an observation.
        self.assertEqual(out[0]["statement"], "“reduce sugar” is already one of your goals.")

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

    def test_known_pattern_comes_back_preselected(self):
        out = _suggest(self.service, [
            {"kind": "dietary_pattern", "value": "vegetarian", "confidence": "high"},  # in dietary_groups
        ])
        self.assertEqual(
            [(s["value"], s["already_known"]) for s in out], [("vegetarian", True)]
        )
        self.assertIn("already part of your dietary profile", out[0]["statement"])

    def test_new_kinds_accepted_by_decide(self):
        # decide() must not reject the new kinds as invalid payloads.
        with patch.object(self.service, "_apply", return_value=True) as ap:
            self.assertTrue(
                self.service.decide("m", "goal", "reduce_fat", "accept"))
            ap.assert_called_once()
        with patch.object(self.service, "_apply", return_value=True):
            self.assertTrue(
                self.service.decide("m", "dietary_pattern", "keto", "accept"))

    def test_vegetarian_pattern_writes_dietary_groups_not_food_likes(self):
        """Regression: accepting dietary_pattern 'vegetarian' must update
        dietary_groups, NEVER food_likes (the reported bug)."""
        import sys
        from unittest.mock import MagicMock

        class _Prof:
            def __init__(self):
                self.nutritional_preferences = {}
                self.allergies = []
                self.dietary_groups = ["omnivore"]
                self.properties = {}

        prof = _Prof()
        member = MagicMock()
        member.profile = prof
        client = MagicMock()
        client.members.get.return_value = member
        plat = MagicMock()
        plat.WISEFOOD_PLATFORM.get_client.return_value = client

        saved = sys.modules.get("backend.platform")
        sys.modules["backend.platform"] = plat
        try:
            from services.memory_service import MemoryService
            MemoryService().decide("m", "dietary_pattern", "vegetarian", "accept")
        finally:
            if saved is not None:
                sys.modules["backend.platform"] = saved
            else:
                sys.modules.pop("backend.platform", None)

        self.assertIn("vegetarian", prof.dietary_groups)
        # The bug: it must NOT have gone into food_likes.
        self.assertNotIn(
            "vegetarian",
            [str(v).lower() for v in prof.nutritional_preferences.get("food_likes", [])],
        )


class MemoryWriteTests(unittest.TestCase):
    """_apply against a fake platform profile (no network)."""

    class _Prof:
        def __init__(self, **fields):
            self.nutritional_preferences = fields.get("nutritional_preferences", {})
            self.allergies = fields.get("allergies", [])
            self.dietary_groups = fields.get("dietary_groups", [])
            self.properties = fields.get("properties", {})
            self.writes = 0

        def __setattr__(self, name, value):
            if name in ("nutritional_preferences", "allergies",
                        "dietary_groups", "properties") and "writes" in self.__dict__:
                self.__dict__["writes"] += 1
            object.__setattr__(self, name, value)

    def _decide(self, prof, kind, value, source_text=""):
        import sys
        from unittest.mock import MagicMock

        member = MagicMock()
        member.profile = prof
        client = MagicMock()
        client.members.get.return_value = member
        plat = MagicMock()
        plat.WISEFOOD_PLATFORM.get_client.return_value = client

        saved = sys.modules.get("backend.platform")
        sys.modules["backend.platform"] = plat
        try:
            return MemoryService().decide("m", kind, value, "accept", source_text)
        finally:
            if saved is not None:
                sys.modules["backend.platform"] = saved
            else:
                sys.modules.pop("backend.platform", None)

    def test_new_goal_is_written_once_with_a_log_entry(self):
        prof = self._Prof(properties={"memory_log": []})
        self.assertTrue(self._decide(prof, "goal", "increase_protein", "more protein please"))
        self.assertEqual(
            [g["slug"] for g in prof.properties["dietary_goals"]], ["increase_protein"]
        )
        self.assertEqual(len(prof.properties["memory_log"]), 1)
        self.assertEqual(prof.properties["memory_log"][0]["source_text"], "more protein please")
        self.assertEqual(prof.writes, 1)  # properties only

    def test_reaccepting_a_known_goal_writes_nothing(self):
        prof = self._Prof(properties={
            "dietary_goals": [{"slug": "increase_protein", "label": "increase protein"}],
            "memory_log": [{"kind": "goal", "value": "increase_protein",
                            "source": "foodscholar", "recorded_at": "2026-07-01T00:00:00+00:00"}],
        })
        self.assertFalse(self._decide(prof, "goal", "increase_protein"))
        self.assertEqual(len(prof.properties["dietary_goals"]), 1)
        self.assertEqual(len(prof.properties["memory_log"]), 1)
        self.assertEqual(prof.writes, 0)

    def test_reaccepting_a_known_like_writes_nothing(self):
        prof = self._Prof(nutritional_preferences={"food_likes": ["lentils"]},
                          properties={"memory_log": []})
        self.assertFalse(self._decide(prof, "like", "lentils"))
        self.assertEqual(prof.writes, 0)

    def test_like_that_resolves_a_contradiction_is_still_a_write(self):
        prof = self._Prof(nutritional_preferences={"food_likes": ["lentils"],
                                                   "food_dislikes": ["lentils"]},
                          properties={})
        self.assertTrue(self._decide(prof, "like", "lentils"))
        self.assertEqual(prof.nutritional_preferences["food_dislikes"], [])
        self.assertEqual(prof.writes, 2)  # nutritional_preferences + properties


if __name__ == "__main__":
    unittest.main()
