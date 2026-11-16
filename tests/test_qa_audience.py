"""The corpus's tags and ages, as the agent uses them (facets.py, audience.py).

Production: 208 of 531 searchable guideline rules are for children, and every
rule and enriched article is tagged (who, food group, nutrient, condition,
setting, study type, …). The planner wrote free text against those keyword
fields, as boosts only, so nothing matched and nothing was ever ruled out: an
adult asking about meat portions got the 30 g serving for 1-4 year olds.
"""

import time
import unittest
from unittest.mock import patch

from agents.qa_planner import _coerce_filters
from models.qa import PlannedSubQuestion, SubQuestionFilters
from services.qa_pipeline import audience, facets, ranking, retrieval

CATALOG = {
    "guidelines": {
        "target_populations": ["under_5_years", "general_population", "adults", "ages_5_to_18", "elderly", "pregnant_people"],
        "life_stage": ["early_childhood", "adulthood", "older_adulthood", "adolescence", "school_age", "pregnancy"],
        "food_groups": ["grains", "protein_foods", "dairy", "vegetables"],
        "nutrients": ["fat", "sugar", "protein", "salt", "calcium"],
        "health_conditions": ["constipation"],
        "topic": ["serving_size", "hydration"],
        "setting": ["general", "home"],
        "audience": ["individual", "caregiver"],
        "frequency": ["daily", "weekly"],
        "action_type": ["eat", "limit"],
    },
    "articles": {
        "study_type": ["Randomized Controlled Trial", "Meta-analysis", "Observational (Cohort)"],
        "biological_model": ["Human", "Animal", "In vitro"],
        "population_group": ["Adults (18-64)", "Adults", "Children (3-12)", "Infants (0-2)"],
        "age_group": ["Adults (18-64)", "Adults", "Children (3-12)", "Infants (0-2)"],
        "topics": ["Cardiovascular health"],
    },
}

TODDLER_RULE = {"source_type": "guideline", "age_min_months": 12, "age_max_months": 48,
                "life_stage": ["early_childhood"], "target_populations": ["under_5_years"],
                "food_groups": ["protein_foods"]}
ADULT_RULE = {"source_type": "guideline", "age_min_months": 216, "age_max_months": -1,
              "life_stage": ["adulthood"], "target_populations": ["adults"], "food_groups": ["protein_foods"]}
GENERAL_RULE = {"source_type": "guideline", "age_min_months": -1, "age_max_months": -1,
                "target_populations": ["general_population"], "food_groups": ["dairy"]}


class _WithCatalog(unittest.TestCase):
    def setUp(self):
        saved = dict(facets._cache)
        facets._cache.update(at=time.monotonic(), value=CATALOG)
        self.addCleanup(lambda: facets._cache.update(saved))


class CatalogTests(_WithCatalog):
    def test_the_planner_is_given_fields_and_exact_values(self):
        card = facets.for_planner()
        self.assertIn("under_5_years", card["guidelines"]["target_populations"]["values"])
        self.assertIn("Human", card["articles"]["biological_model"]["values"])
        self.assertIn("how_to_use", card)

    def test_what_the_agent_wrote_becomes_the_index_spelling(self):
        self.assertEqual(facets.canonical("target_populations", ["Under 5 years"]), ["under_5_years"])
        self.assertEqual(facets.canonical("food_groups", ["protein foods", "unicorn"]), ["protein_foods"])
        self.assertEqual(facets.canonical("biological_model", ["human"]), ["Human"])
        # Every spelling of the adult group in the article tags.
        self.assertEqual(facets.canonical("age_group", ["Adults (18-64)"]), ["Adults (18-64)", "Adults"])
        self.assertEqual(facets.canonical("not_a_field", ["x"]), [])

    def test_unavailable_index_falls_back(self):
        facets._cache.update(at=0.0, value=None)
        with patch.object(facets, "_load", side_effect=RuntimeError("es down")):
            self.assertIn("guidelines", facets.catalog())


class PlannerOutputTests(_WithCatalog):
    def test_facets_must_and_age_are_read(self):
        f = _coerce_filters({
            "facets": {"life_stage": ["early_childhood"], "food_groups": ["Protein Foods"], "made_up": ["x"]},
            "must": ["life_stage", "made_up"],
            "age_months": [36, 47],
        })
        self.assertEqual(f.facets, {"life_stage": ["early_childhood"], "food_groups": ["protein_foods"]})
        self.assertEqual(f.must, ["life_stage"])
        self.assertEqual((f.age_min_months, f.age_max_months), (36, 47))

    def test_older_per_aspect_lists_are_folded_in(self):
        f = _coerce_filters({"nutrients": ["calcium"], "study_types": ["meta-analysis"]})
        self.assertEqual(f.facets, {"nutrients": ["calcium"], "study_type": ["Meta-analysis"]})


class FitTests(_WithCatalog):
    def test_any_tag_not_only_who_it_is_for(self):
        wants = SubQuestionFilters(facets={"food_groups": ["dairy"], "nutrients": ["calcium"]})
        dairy, detail = facets.fit(GENERAL_RULE, wants)
        meat, _ = facets.fit(ADULT_RULE, wants)
        self.assertGreater(dairy, 1.0)
        self.assertLess(meat, 1.0)
        self.assertEqual(detail, {"food_groups": "match"})

    def test_a_requirement_rules_out_what_a_preference_only_lowers(self):
        prefer = SubQuestionFilters(facets={"life_stage": ["adulthood"]})
        require = prefer.model_copy(update={"must": ["life_stage"]})
        self.assertEqual(facets.fit(TODDLER_RULE, prefer)[0], facets.PREFER_MISS)
        self.assertEqual(facets.fit(TODDLER_RULE, require)[0], facets.MUST_MISS)

    def test_an_untagged_source_is_not_a_mismatch(self):
        wants = SubQuestionFilters(facets={"biological_model": ["Human"]}, must=["biological_model"])
        self.assertEqual(facets.fit({"source_type": "article", "biological_model": "Not stated"}, wants)[0], 1.0)
        self.assertEqual(facets.fit({"source_type": "article"}, wants)[0], 1.0)
        self.assertEqual(facets.fit({"source_type": "article", "biological_model": "Animal"}, wants)[0], facets.MUST_MISS)

    def test_ranking_takes_the_best_fit_of_the_searches_that_found_it(self):
        by_sq = {
            "adult": SubQuestionFilters(facets={"life_stage": ["adulthood"]}, must=["life_stage"]),
            "child": SubQuestionFilters(facets={"life_stage": ["early_childhood"]}, must=["life_stage"]),
        }
        self.assertLess(ranking.source_fit(TODDLER_RULE, ["adult"], by_sq)[0], 0.5)
        self.assertGreater(ranking.source_fit(TODDLER_RULE, ["adult", "child"], by_sq)[0], 1.0)


class ClauseTests(_WithCatalog):
    def test_a_requirement_filters_but_lets_the_untagged_through(self):
        wants = SubQuestionFilters(facets={"life_stage": ["adulthood"], "food_groups": ["dairy"]}, must=["life_stage"])
        hard, soft = facets.clauses(wants, "guidelines")
        self.assertEqual(len(soft), 2)
        self.assertEqual(len(hard), 1)
        allowed = hard[0]["bool"]["should"]
        self.assertEqual(allowed[0], {"terms": {"life_stage": ["adulthood"]}})
        self.assertNotIn("adulthood", allowed[1]["bool"]["must_not"]["terms"]["life_stage"])
        # Article tags do not leak into a guideline search.
        self.assertEqual(facets.clauses(SubQuestionFilters(facets={"study_type": ["Meta-analysis"]}), "guidelines"), ([], []))

    def test_too_few_hits_relax_the_requirement(self):
        seen = []

        def lexical(_query, *, size, filters=None, **_kw):
            seen.append(list(filters.must))
            return []

        sq = PlannedSubQuestion(
            id="sq1", text="q", branch="guidelines", lexical_query="q",
            filters=SubQuestionFilters(facets={"life_stage": ["pregnancy"]}, must=["life_stage"]),
        )
        with patch.object(retrieval, "search_guidelines_lexical", lexical):
            out = retrieval.run_branch(sq, branch="guidelines", vector=None, user_context=None, expertise_level=None)
        self.assertEqual(seen, [["life_stage"], []])
        self.assertEqual(out.status["relaxed"], ["life_stage"])


class AgeTests(_WithCatalog):
    def test_who_the_question_is_about(self):
        a = audience.resolve("How much meat should my 3-year-old eat?")
        self.assertEqual((a.min_months, a.max_months, a.source), (36, 36, "question"))
        self.assertEqual(audience.resolve("Is fish safe when pregnant?").life_stages, ["pregnancy"])
        self.assertEqual(audience.resolve("How much protein?", member_age_group="adult").source, "profile")
        self.assertFalse(audience.resolve("How much protein do I need?").known)

    def test_the_safety_net_fills_only_what_the_plan_left_out(self):
        child = audience.resolve("How much meat should my 3-year-old eat?")
        filled = audience.apply(SubQuestionFilters(facets={"food_groups": ["protein_foods"]}), child)
        self.assertEqual(filled.facets["life_stage"], ["early_childhood"])
        self.assertEqual((filled.age_min_months, sorted(filled.must)), (36, ["age", "life_stage"]))
        planned = SubQuestionFilters(facets={"life_stage": ["adulthood"]})
        self.assertIs(audience.apply(planned, child), planned)
        # From the profile it is a preference: they may be asking for someone else.
        profile = audience.apply(SubQuestionFilters(), audience.resolve("How much protein?", member_age_group="adult"))
        self.assertEqual(profile.must, [])

    def test_age_window(self):
        adult = SubQuestionFilters(age_min_months=216, age_max_months=780, must=["age"])
        self.assertEqual(audience.age_fit(TODDLER_RULE, adult), audience.MISMATCH_FACTOR)
        self.assertEqual(audience.age_fit(ADULT_RULE, adult), audience.MATCH_FACTOR)
        self.assertEqual(audience.age_fit(GENERAL_RULE, adult), 1.0)
        self.assertEqual(audience.age_fit({"source_type": "article", "age_group": "Infants (0-2)"}, adult), 0.5)

    def test_nobody_named_group_only_rules_step_back(self):
        nobody = SubQuestionFilters(facets={"food_groups": ["protein_foods"]})
        self.assertEqual(audience.unstated_fit(TODDLER_RULE, nobody), audience.UNSTATED_FACTOR)
        self.assertEqual(audience.unstated_fit(ADULT_RULE, nobody), 1.0)
        self.assertEqual(audience.unstated_fit(TODDLER_RULE, SubQuestionFilters(facets={"life_stage": ["early_childhood"]})), 1.0)
        # The serving-size case: an adult's rule now outranks the toddler's.
        by_sq = {"sq1": nobody}
        self.assertGreater(
            ranking.source_fit(ADULT_RULE, ["sq1"], by_sq)[0],
            ranking.source_fit(TODDLER_RULE, ["sq1"], by_sq)[0],
        )


if __name__ == "__main__":
    unittest.main()
