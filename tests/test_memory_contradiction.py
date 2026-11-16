"""A nudge that contradicts the profile is asked again, and says so.

Live: "i dont eat meat, how can i get more protein?" from a profile that says
"omnivore" and once declined "vegetarian": no nudge, and the profile stayed
wrong for FoodChat.
"""

import sys
import unittest
from unittest.mock import MagicMock

from services.memory_service import MemoryService

PROFILE = {
    "dietary_groups": ["omnivore"],
    "nutritional_preferences": {"food_likes": ["pastitsio"], "food_dislikes": []},
    "properties": {"memory_optouts": ["vegetarian", "increase_protein", "pastitsio"]},
}


def _cand(kind, value):
    return {"kind": kind, "value": value, "confidence": "high", "statement": f"You {value}?"}


class PolicyTests(unittest.TestCase):
    def test_declined_but_contradicting_diet_is_asked_again(self):
        out = MemoryService()._apply_policy([_cand("dietary_pattern", "vegetarian")], PROFILE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["contradicts"], ["omnivore"])

    def test_declined_and_not_contradicting_stays_quiet(self):
        out = MemoryService()._apply_policy([_cand("goal", "increase_protein")], PROFILE)
        self.assertEqual(out, [])

    def test_same_rung_diets_do_not_conflict(self):
        profile = {**PROFILE, "dietary_groups": ["vegan"], "properties": {}}
        out = MemoryService()._apply_policy([_cand("dietary_pattern", "plant_based")], profile)
        self.assertNotIn("contradicts", out[0])

    def test_restrictions_never_conflict_with_diets(self):
        profile = {**PROFILE, "properties": {}}
        out = MemoryService()._apply_policy([_cand("dietary_pattern", "halal")], profile)
        self.assertNotIn("contradicts", out[0])


class AcceptTests(unittest.TestCase):
    def test_accepting_replaces_the_conflicting_diet_and_lifts_the_opt_out(self):
        class _Prof:
            nutritional_preferences = {}
            allergies = []
            dietary_groups = ["omnivore", "halal"]
            properties = {"memory_optouts": ["vegetarian", "greek"]}

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
            MemoryService().decide("m", "dietary_pattern", "vegetarian", "accept")
        finally:
            if saved is not None:
                sys.modules["backend.platform"] = saved
            else:
                sys.modules.pop("backend.platform", None)
        self.assertEqual(prof.dietary_groups, ["halal", "vegetarian"])
        self.assertEqual(prof.properties["memory_optouts"], ["greek"])


if __name__ == "__main__":
    unittest.main()
