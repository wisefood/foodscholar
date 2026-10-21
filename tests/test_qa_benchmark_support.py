"""Backend support for the QA benchmark: persona context, cache key, runtime info."""

import unittest
from unittest.mock import patch

from config import config
from models.qa import QARequest
from services.qa_service import QAService


class RequestContextTests(unittest.TestCase):
    def setUp(self):
        self.service = QAService(cache_enabled=False)

    def test_request_fields_reach_the_context_without_a_member(self):
        context = self.service._resolve_user_context(
            QARequest(
                question="How much calcium do I need?",
                expertise_level="beginner",
                country="Ireland",
                region="Dublin, Ireland",
                member_age_group="older adult",
            )
        )
        self.assertEqual(context.country, "Ireland")
        self.assertEqual(context.region, "Dublin, Ireland")
        self.assertEqual(context.member_age_group, "older adult")
        self.assertEqual(context.experience_group, "beginner")

    def test_region_alone_derives_the_country(self):
        context = self.service._resolve_user_context(
            QARequest(question="How much calcium do I need?", region="IE-D")
        )
        self.assertEqual(context.country, "IE")

    def test_member_household_wins_over_request_fields(self):
        member = {
            "country": "SI",
            "region": "SI-071",
            "experience_group": None,
            "member_age_group": None,
            "profile": {},
        }
        with patch.object(
            QAService, "_lookup_member_context", return_value=member
        ):
            context = self.service._resolve_user_context(
                QARequest(
                    question="How much calcium do I need?",
                    member_id="m-1",
                    country="Ireland",
                    region="Dublin, Ireland",
                    member_age_group="adult",
                )
            )
        self.assertEqual((context.country, context.region), ("SI", "SI-071"))
        # The member left it unset, so the request's value fills the gap.
        self.assertEqual(context.member_age_group, "adult")


class CacheKeyTests(unittest.TestCase):
    def test_every_ranking_setting_is_part_of_the_key(self):
        from services.qa_pipeline.ranking import RANKING_SETTINGS

        service = QAService(cache_enabled=False)
        request = QARequest(question="Do whole grains lower cholesterol?")
        base = service._build_cache_key(request)
        for name in RANKING_SETTINGS:
            original = config.settings.get(name)
            with patch.dict(config.settings, {name: f"changed-{original}"}):
                self.assertNotEqual(
                    service._build_cache_key(request), base, name
                )

    def test_bypass_flag_is_not_part_of_the_key(self):
        service = QAService(cache_enabled=False)
        question = "Do whole grains lower cholesterol?"
        self.assertEqual(
            service._build_cache_key(QARequest(question=question)),
            service._build_cache_key(QARequest(question=question, bypass_cache=True)),
        )


class RuntimeInfoTests(unittest.TestCase):
    def test_snapshot_reports_commit_models_prompts_and_settings(self):
        from services.qa_pipeline.ranking import RANKING_SETTINGS

        service = QAService(cache_enabled=False)
        with patch.dict(config.settings, {"GIT_COMMIT": "abc123"}), patch(
            "services.qa_service.served_prompt_versions",
            return_value={"foodscholar/qa-planner-system": 4},
        ):
            info = service.runtime_info()

        self.assertEqual(info.git_commit, "abc123")
        self.assertEqual(info.models["answer"], config.settings["QA_DEFAULT_MODEL"])
        self.assertEqual(info.models["planner"], config.settings["QA_PLANNER_MODEL"])
        self.assertIn("embedding", info.models)
        self.assertEqual(info.prompt_versions, {"foodscholar/qa-planner-system": 4})
        self.assertEqual(set(info.ranking), set(RANKING_SETTINGS))
        self.assertIn("QA_MAX_REPAIR_ROUNDS", info.retrieval)
        self.assertFalse(info.answer_cache_enabled)


if __name__ == "__main__":
    unittest.main()
