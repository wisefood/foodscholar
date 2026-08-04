import tempfile
import unittest
from pathlib import Path


class GuidelineExtractorServiceTests(unittest.TestCase):
    def test_dedupe_guidelines_collapses_near_duplicates(self):
        from services.guideline_extractor import dedupe_guidelines

        items = [
            "- Offer vegetables daily.",
            "Offer vegetables daily",
            "Offer vegetables daily!",
            "Serve pulses regularly.",
        ]

        self.assertEqual(
            dedupe_guidelines(items),
            ["Offer vegetables daily.", "Serve pulses regularly."],
        )

    def test_artifact_workspace_uses_uuid_folder_and_source_pdf(self):
        from services.guideline_extractor import GuidelineExtractorService

        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GuidelineExtractorService(workspace_root=tmpdir)
            info = service.get_artifact_workspace(artifact_uuid)

            self.assertTrue(Path(info.artifact_dir).is_dir())
            self.assertEqual(info.workspace_root, str(Path(tmpdir).resolve()))
            self.assertEqual(
                info.pdf_path,
                str(Path(tmpdir).resolve() / artifact_uuid / "source.pdf"),
            )
            self.assertFalse(info.pdf_exists)

    def test_invalid_artifact_uuid_is_rejected(self):
        from services.guideline_extractor import (
            GuidelineExtractionError,
            GuidelineExtractorService,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GuidelineExtractorService(workspace_root=tmpdir)
            with self.assertRaises(GuidelineExtractionError):
                service.get_artifact_workspace("not-a-uuid")


class GuideContextTests(unittest.TestCase):
    """
    The guide context is what lets a rule sentence carry its population. It is
    assembled from the catalog record and, where that is silent, from the guide
    document itself — the catalog must always win.
    """

    def test_catalog_record_maps_onto_context(self):
        from services.guideline_extractor import GuideContext

        context = GuideContext.from_guide(
            {
                "urn": "urn:guide:ie-1-4",
                "title": "Eating guidelines for 1-4 year olds",
                "region": "IE",
                "target_audiences": ["parents", "carers"],
                "publication_year": 2020,
            }
        )

        self.assertEqual(context.title, "Eating guidelines for 1-4 year olds")
        self.assertEqual(context.region, "IE")
        self.assertFalse(context.is_empty())

    def test_empty_context_tells_the_model_not_to_assume(self):
        from services.guideline_extractor import GuideContext

        block = GuideContext().as_prompt_block()

        self.assertIn("not available", block)
        self.assertIn("do not assume a population", block)

    def test_population_and_age_range_reach_the_prompt(self):
        from services.guideline_extractor import GuideContext

        block = GuideContext(
            title="Eating guidelines for 1-4 year olds",
            population_note="children aged 1 to 4 years",
            age_min_months=12,
            age_max_months=48,
        ).as_prompt_block()

        self.assertIn("children aged 1 to 4 years", block)
        self.assertIn("12 to 48", block)

    def test_document_profile_fills_gaps_without_overwriting_the_catalog(self):
        from services.guideline_extractor import GuideContext, GuideDocumentProfile

        catalog = GuideContext(title="Curated title", region="IE")
        profile = GuideDocumentProfile(
            title="Title printed on the cover",
            region="GB",
            population_note="children aged 1 to 4 years",
            age_min_months=12,
            age_max_months=48,
            evidence=["for children aged 1-4 years"],
        )

        merged = catalog.merge_document_profile(profile)

        # Curated values survive; only the gaps are filled.
        self.assertEqual(merged.title, "Curated title")
        self.assertEqual(merged.region, "IE")
        self.assertEqual(merged.population_note, "children aged 1 to 4 years")
        self.assertEqual(merged.age_min_months, 12)
        self.assertIn("population_note", merged.derived_fields)
        self.assertNotIn("title", merged.derived_fields)
        self.assertEqual(merged.evidence, ["for children aged 1-4 years"])
        # The receiver is left untouched.
        self.assertIsNone(catalog.population_note)

    def test_thin_metadata_still_requests_a_document_profile(self):
        from services.guideline_extractor import GuideContext

        # A title and region alone leave every rule population-less.
        self.assertTrue(
            GuideContext(title="A guide", region="IE").needs_document_profile()
        )
        self.assertTrue(GuideContext().needs_document_profile())
        self.assertFalse(
            GuideContext(
                title="A guide",
                region="IE",
                population_note="children aged 1 to 4 years",
            ).needs_document_profile()
        )


class ExtractionPayloadTests(unittest.TestCase):
    def test_v2_rule_payload_is_mapped(self):
        from services.guideline_extractor import _rule_from_payload

        rule = _rule_from_payload(
            {
                "text": "  Provide portions of red meat twice a week.  ",
                "section_label": "Protein foods",
                "source_snippet": "red meat twice a week",
                "target_population_hint": "children aged 1-4 years",
                "age_min_months": 12,
                "age_max_months": 48,
                "life_stage": ["early_childhood"],
                "setting": ["home"],
                "health_conditions": [],
                "nutrients": ["iron"],
                "guideline_type": "food_based",
                "topic": ["protein"],
                "action_type_hint": "eat",
                "confidence": 0.9,
            }
        )

        self.assertEqual(rule.text, "Provide portions of red meat twice a week.")
        self.assertEqual(rule.life_stage, ["early_childhood"])
        self.assertEqual(rule.age_min_months, 12)
        self.assertEqual(rule.action_type_hint, "eat")

    def test_not_stated_sentinels_become_none(self):
        from services.guideline_extractor import _rule_from_payload

        rule = _rule_from_payload(
            {
                "text": "Eat vegetables every day.",
                "section_label": "",
                "source_snippet": "",
                "target_population_hint": "",
                "age_min_months": -1,
                "age_max_months": -1,
                "life_stage": [],
                "setting": [],
                "health_conditions": [],
                "nutrients": [],
                "guideline_type": "",
                "topic": [],
                "action_type_hint": "",
                "confidence": 0.5,
            }
        )

        self.assertIsNone(rule.age_min_months)
        self.assertIsNone(rule.section_label)
        self.assertIsNone(rule.guideline_type)
        self.assertEqual(rule.life_stage, [])

    def test_invented_vocabulary_is_dropped(self):
        from services.guideline_extractor import _rule_from_payload

        rule = _rule_from_payload(
            {
                "text": "Eat vegetables every day.",
                "life_stage": ["early_childhood", "middle_age"],
                "setting": ["home", "spaceship"],
                "guideline_type": "vibes_based",
                "action_type_hint": "vibe",
            }
        )

        self.assertEqual(rule.life_stage, ["early_childhood"])
        self.assertEqual(rule.setting, ["home"])
        self.assertIsNone(rule.guideline_type)
        self.assertIsNone(rule.action_type_hint)

    def test_v1_string_payload_still_yields_a_rule(self):
        from services.guideline_extractor import _rule_from_payload

        rule = _rule_from_payload("- Offer vegetables daily.")

        self.assertEqual(rule.text, "Offer vegetables daily.")
        self.assertEqual(rule.life_stage, [])

    def test_reversed_age_range_is_discarded(self):
        from services.guideline_extractor import _rule_from_payload

        rule = _rule_from_payload(
            {"text": "Eat well.", "age_min_months": 48, "age_max_months": 12}
        )

        self.assertIsNone(rule.age_min_months)
        self.assertIsNone(rule.age_max_months)

    def test_triage_schema_asks_about_continuation(self):
        from services.guideline_extractor import schema_for_triage

        schema = schema_for_triage()

        self.assertIn("continues_from_previous", schema["properties"])
        self.assertIn("continues_from_previous", schema["required"])


if __name__ == "__main__":
    unittest.main()
