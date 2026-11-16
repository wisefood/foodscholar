"""A rule is read in its guide's context (services/qa_pipeline/guides.py).

Live: "Count 30g cooked beef as 1 serving" is tagged general_population with
no life stage and no age, in "Healthy Ireland Food Pyramid: a guide for
children aged 1-4". 21 of that guide's 26 rules carry no life stage.
"""

import time
import unittest

from models.qa import SubQuestionFilters
from services.qa_pipeline import audience, guides, ranking
from services.qa_pipeline.guides import GuideContext


def _seed(test, value):
    saved = dict(guides._cache)
    guides._cache.update(at=time.monotonic(), value=value)
    test.addCleanup(lambda: guides._cache.update(saved))


class GuideTextTests(unittest.TestCase):
    def test_what_the_guide_says_of_itself(self):
        said = guides._from_guide_text("Meat Shelf - Healthy Ireland Food Pyramid", "Healthy Ireland Food Pyramid\nA guide for children aged 1–4")
        self.assertEqual((said.min_months, said.max_months), (12, 59))
        self.assertIn("early_childhood", said.life_stages)
        self.assertIsNone(guides._from_guide_text("Dietary guidelines for Hungary", "A plate model"))


class AttachTests(unittest.TestCase):
    def setUp(self):
        _seed(self, {
            "urn:guide:toddlers": GuideContext(
                title="Meat Shelf", description="A guide for children aged 1–4",
                life_stages=["early_childhood"], populations=["under_5_years"],
                min_months=12, max_months=59, source="guide",
            ),
            "urn:guide:plate": GuideContext(title="Diet plate", life_stages=["adulthood"], source="rules"),
            "urn:guide:mixed": GuideContext(title="Family mealtimes"),
        })

    def test_a_silent_rule_inherits_its_guides_audience(self):
        rule = {"source_type": "guideline", "guide_urn": "urn:guide:toddlers",
                "rule_text": "Count 30g cooked beef as 1 serving.", "target_populations": ["general_population"]}
        guides.attach([rule])
        self.assertEqual(rule["life_stage"], ["early_childhood"])
        self.assertEqual((rule["age_min_months"], rule["age_max_months"]), (12, 59))
        self.assertEqual(rule["audience_from_guide"], "guide")
        self.assertIn("children aged 1–4", rule["guide_description"])
        # So, with nobody named, the toddler portion steps back for the adult's.
        nobody = {"sq1": SubQuestionFilters()}
        self.assertEqual(ranking.source_fit(rule, ["sq1"], nobody)[0], audience.UNSTATED_FACTOR)
        adult = {"sq1": SubQuestionFilters(facets={"life_stage": ["adulthood"]}, must=["life_stage"])}
        self.assertLess(ranking.source_fit(rule, ["sq1"], adult)[0], 0.5)

    def test_a_rule_that_states_its_own_audience_keeps_it(self):
        rule = {"source_type": "guideline", "guide_urn": "urn:guide:toddlers",
                "life_stage": ["pregnancy"], "target_populations": ["pregnant_people"]}
        guides.attach([rule])
        self.assertEqual(rule["life_stage"], ["pregnancy"])
        self.assertNotIn("audience_from_guide", rule)
        self.assertEqual(rule["guide_title"], "Meat Shelf")

    def test_what_the_rules_of_a_guide_agree_on(self):
        rule = {"source_type": "guideline", "guide_urn": "urn:guide:plate"}
        guides.attach([rule])
        self.assertEqual((rule["life_stage"], rule["audience_from_guide"]), (["adulthood"], "rules"))

    def test_a_guide_for_nobody_in_particular_and_articles_are_left_alone(self):
        rule = {"source_type": "guideline", "guide_urn": "urn:guide:mixed"}
        article = {"source_type": "article", "guide_urn": "urn:guide:toddlers"}
        guides.attach([rule, article])
        self.assertNotIn("life_stage", rule)
        self.assertNotIn("life_stage", article)

    def test_unavailable_index_changes_nothing(self):
        guides._cache.update(at=0.0, value=None)
        from unittest.mock import patch

        rule = {"source_type": "guideline", "guide_urn": "urn:guide:toddlers"}
        with patch.object(guides, "_load", side_effect=RuntimeError("es down")):
            guides.attach([rule])
        self.assertNotIn("life_stage", rule)


if __name__ == "__main__":
    unittest.main()
