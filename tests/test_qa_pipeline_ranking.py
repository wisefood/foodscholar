"""Tests for the deterministic evidence ranking of the agentic QA pipeline."""

import math
import unittest

from config import config
from models.qa import RetrievedSource
from services.qa_pipeline import ranking
from services.qa_pipeline.retrieval import rrf_fuse
from services.qa_pipeline.state import EvidenceItem


def _source(urn: str, source_type: str = "article") -> RetrievedSource:
    return RetrievedSource(
        source_type=source_type,
        urn=urn,
        title=f"Title {urn}",
        similarity_score=1.0,
    )


def _item(payload, rrf_norm=1.0) -> EvidenceItem:
    urn = payload.get("urn", "u")
    source_type = payload.setdefault("source_type", "article")
    item = EvidenceItem(payload=payload, source=_source(urn, source_type))
    item.rrf_norm = rrf_norm
    return item


class RecencyFactorTests(unittest.TestCase):
    def test_half_life_math(self):
        # A paper exactly one half-life old scores 0.5.
        factor = ranking.recency_factor(
            {"publication_year": "2020"}, now_year=2026
        )
        self.assertAlmostEqual(factor, 0.5, places=6)

    def test_floor_applies_to_old_papers(self):
        factor = ranking.recency_factor(
            {"publication_year": "1995"}, now_year=2026
        )
        self.assertEqual(factor, config.settings["QA_RECENCY_FLOOR"])

    def test_current_year_is_neutral(self):
        self.assertEqual(
            ranking.recency_factor({"publication_year": 2026}, now_year=2026), 1.0
        )

    def test_iso_date_strings_parse(self):
        factor = ranking.recency_factor(
            {"publication_year": "2020-06-01T00:00:00Z"}, now_year=2026
        )
        self.assertAlmostEqual(factor, 0.5, places=6)

    def test_unparseable_year_is_neutral(self):
        for value in (None, "", "unknown", "n/a", 12):
            self.assertEqual(
                ranking.recency_factor({"publication_year": value}), 1.0
            )


class InfluenceFactorTests(unittest.TestCase):
    def test_no_citations_is_neutral(self):
        self.assertEqual(ranking.influence_factor({}), 1.0)
        self.assertEqual(ranking.influence_factor({"citationCount": 0}), 1.0)
        self.assertEqual(ranking.influence_factor({"citationCount": None}), 1.0)

    def test_cap_reaches_full_weight(self):
        factor = ranking.influence_factor({"citationCount": 1000})
        self.assertAlmostEqual(
            factor, 1.0 + config.settings["QA_INFLUENCE_WEIGHT"], places=6
        )

    def test_over_cap_is_clamped(self):
        capped = ranking.influence_factor({"citationCount": 1000})
        over = ranking.influence_factor({"citationCount": 50000})
        self.assertEqual(over, capped)

    def test_both_field_spellings_are_read(self):
        camel = ranking.influence_factor({"citationCount": 100})
        snake = ranking.influence_factor({"citation_count": 100})
        self.assertEqual(camel, snake)
        self.assertGreater(camel, 1.0)

    def test_string_numbers_are_read(self):
        self.assertEqual(
            ranking.influence_factor({"citationCount": "100"}),
            ranking.influence_factor({"citationCount": 100}),
        )

    def test_influential_citations_count_double(self):
        plain = ranking.influence_factor({"citationCount": 100})
        influential = ranking.influence_factor(
            {"citationCount": 0, "influentialCitationCount": 50}
        )
        self.assertEqual(plain, influential)

    def test_cold_start_recent_paper_not_penalized(self):
        # A 2026 paper with no citations ranks purely on relevance/recency.
        payload = {"publication_year": 2026, "citationCount": 0}
        self.assertEqual(ranking.influence_factor(payload), 1.0)
        self.assertEqual(ranking.recency_factor(payload, now_year=2026), 1.0)


class StudyDesignFactorTests(unittest.TestCase):
    def test_hierarchy(self):
        meta = ranking.study_design_factor({"ai_category": "Meta-analysis"})
        rct = ranking.study_design_factor({"ai_category": "Randomized Controlled Trial"})
        cohort = ranking.study_design_factor({"ai_category": "Prospective cohort study"})
        animal = ranking.study_design_factor({"ai_category": "Animal study"})
        self.assertGreater(meta, rct)
        self.assertGreater(rct, cohort)
        self.assertGreater(cohort, 1.0)
        self.assertLess(animal, 1.0)

    def test_first_match_wins(self):
        # A systematic review of RCTs is scored as a review.
        factor = ranking.study_design_factor(
            {"ai_category": "Systematic review of randomized trials"}
        )
        self.assertEqual(factor, 1.3)

    def test_unknown_or_absent_is_neutral(self):
        self.assertEqual(ranking.study_design_factor({}), 1.0)
        self.assertEqual(
            ranking.study_design_factor({"ai_category": "editorial"}), 1.0
        )

    def test_dedicated_study_type_field_is_read(self):
        # Production carries the design on `study_type` for the annotated
        # cohort; ai_category is the older spelling.
        self.assertEqual(
            ranking.study_design_factor({"study_type": "Systematic Review"}), 1.3
        )

    def test_biological_model_discount_overrides_design(self):
        # An animal RCT must not outrank human evidence.
        factor = ranking.study_design_factor(
            {"study_type": "Randomized Controlled Trial", "biological_model": "Animal"}
        )
        self.assertEqual(factor, 0.85)
        self.assertEqual(
            ranking.study_design_factor({"biological_model": "In vitro"}), 0.85
        )
        # A human model changes nothing.
        self.assertEqual(
            ranking.study_design_factor(
                {"study_type": "Cohort", "biological_model": "Human"}
            ),
            1.1,
        )


class EarnedTierTests(unittest.TestCase):
    """Untiered but heavily-cited work earns tier standing on its own;
    an explicit tier — promotion or demotion — always wins."""

    def test_untiered_highly_influential_earns_prime_standing(self):
        boost, label = ranking.tier_factor(
            {"citation_count": 600, "influential_citation_count": 40}
        )
        self.assertEqual(label, "earned_prime")
        self.assertEqual(boost, config.settings["QA_EARNED_PRIME_BOOST"])
        # Earned standing stays below the curated equivalent.
        self.assertLess(boost, 1.6)

    def test_citations_without_influential_earn_core_not_prime(self):
        # 600 raw citations clear the prime citation bar but nothing builds
        # on the work — that is core standing, not prime.
        boost, label = ranking.tier_factor({"citationCount": 600})
        self.assertEqual(label, "earned_core")
        self.assertEqual(boost, config.settings["QA_EARNED_CORE_BOOST"])
        self.assertLess(boost, 1.25)

    def test_moderate_record_earns_core(self):
        boost, label = ranking.tier_factor({"citation_count": 200})
        self.assertEqual(label, "earned_core")

    def test_thin_record_earns_nothing(self):
        boost, label = ranking.tier_factor({"citation_count": 40})
        self.assertEqual((boost, label), (1.0, ""))

    def test_explicit_tier_wins_over_bibliometrics(self):
        # A curator's demotion is not overridden by a citation record.
        boost, label = ranking.tier_factor(
            {"indexing_tier": "archive_only", "citation_count": 5000,
             "influential_citation_count": 400}
        )
        self.assertEqual((boost, label), (0.6, "archive_only"))

    def test_influential_citations_count_double_toward_thresholds(self):
        # 100 citations + 200 doubled influential = 500 effective.
        boost, label = ranking.tier_factor(
            {"citation_count": 100, "influential_citation_count": 200}
        )
        self.assertEqual(label, "earned_prime")

    def test_kill_switch_disables_earned_tiers(self):
        original = config.settings.get("QA_EARNED_TIER_ENABLED", True)
        config.settings["QA_EARNED_TIER_ENABLED"] = False
        try:
            boost, label = ranking.tier_factor(
                {"citation_count": 600, "influential_citation_count": 40}
            )
            self.assertEqual((boost, label), (1.0, ""))
        finally:
            config.settings["QA_EARNED_TIER_ENABLED"] = original

    def test_score_parts_carry_the_tier_label(self):
        item = _item(
            {"urn": "a1", "citation_count": 600,
             "influential_citation_count": 40},
            rrf_norm=0.5,
        )
        adjusted = ranking.adjust_evidence([item], now_year=2026)
        self.assertEqual(adjusted[0].score_parts["tier_label"], "earned_prime")


class AdjustEvidenceTests(unittest.TestCase):
    def test_multiplicative_formula(self):
        item = _item(
            {
                "urn": "a1",
                "publication_year": "2020",
                "citationCount": 1000,
                "ai_category": "meta-analysis",
                "indexing_tier": "prime",
            },
            rrf_norm=0.8,
        )
        adjusted = ranking.adjust_evidence([item], now_year=2026)
        self.assertEqual(len(adjusted), 1)
        expected = 0.8 * 1.6 * 0.5 * (1.0 + config.settings["QA_INFLUENCE_WEIGHT"]) * 1.3
        self.assertAlmostEqual(adjusted[0].adjusted_score, expected, places=6)
        parts = adjusted[0].score_parts
        self.assertEqual(parts["tier"], 1.6)
        self.assertEqual(parts["recency"], 0.5)
        self.assertEqual(parts["study_design"], 1.3)

    def test_guidelines_only_use_rrf(self):
        item = _item(
            {"urn": "g1", "source_type": "guideline", "publication_year": "1999"},
            rrf_norm=0.7,
        )
        adjusted = ranking.adjust_evidence([item], now_year=2026)
        self.assertAlmostEqual(adjusted[0].adjusted_score, 0.7, places=6)

    def test_expert_only_articles_dropped_for_general_audience(self):
        visible = _item({"urn": "a1"}, rrf_norm=0.9)
        restricted = _item(
            {"urn": "a2", "reader_visibility": "expert_only"}, rrf_norm=0.9
        )
        adjusted = ranking.adjust_evidence(
            [visible, restricted], expertise_level="beginner"
        )
        self.assertEqual([i.payload["urn"] for i in adjusted], ["a1"])
        adjusted_expert = ranking.adjust_evidence(
            [visible, restricted], expertise_level="expert"
        )
        self.assertEqual(len(adjusted_expert), 2)


class SelectEvidenceTests(unittest.TestCase):
    def test_min_score_threshold_can_empty_the_pool(self):
        weak = _item({"urn": "a1"}, rrf_norm=0.01)
        weak.adjusted_score = 0.01
        selected, dropped = ranking.select_evidence([weak], top_k=5)
        self.assertEqual(selected, [])
        self.assertEqual(dropped["below_threshold"], 1)

    def test_per_document_diversity_cap(self):
        items = []
        for index in range(4):
            item = _item({"urn": f"rule-{index}", "source_type": "guideline",
                          "guide_urn": "guide-1"}, rrf_norm=1.0)
            item.adjusted_score = 1.0 - index * 0.1
            items.append(item)
        selected, dropped = ranking.select_evidence(items, top_k=5)
        self.assertEqual(len(selected), config.settings["QA_PER_DOC_CAP"])
        self.assertEqual(dropped["over_doc_cap"], 4 - config.settings["QA_PER_DOC_CAP"])

    def test_articles_capped_at_top_k_guidelines_at_five(self):
        items = []
        for index in range(10):
            item = _item({"urn": f"a{index}", "doi": f"doi-{index}"}, rrf_norm=1.0)
            item.adjusted_score = 1.0 - index * 0.01
            items.append(item)
        for index in range(8):
            item = _item(
                {"urn": f"g{index}", "source_type": "guideline",
                 "guide_urn": f"guide-{index}"},
                rrf_norm=1.0,
            )
            item.adjusted_score = 0.5
            items.append(item)
        selected, _ = ranking.select_evidence(items, top_k=3)
        articles = [i for i in selected if i.payload["source_type"] == "article"]
        guidelines = [i for i in selected if i.payload["source_type"] == "guideline"]
        self.assertEqual(len(articles), 3)
        self.assertEqual(len(guidelines), 5)
        # Block order: articles first, then guidelines (G-labels depend on it).
        self.assertEqual(selected[:3], articles)


class GuidelineAffinityTests(unittest.TestCase):
    """Enrichment facets reorder guidelines toward the asker and the question."""

    def _affinity(self, payload, *, user_context=None, facets=None):
        from models.qa import QAUserContext, SubQuestionFilters

        return ranking.guideline_affinity_factor(
            payload,
            user_context=user_context or QAUserContext(),
            facets=facets or SubQuestionFilters(),
        )

    def test_no_facets_is_neutral(self):
        self.assertEqual(
            self._affinity({"source_type": "guideline", "rule_text": "Eat well."}),
            1.0,
        )

    def test_region_match_boosts(self):
        from models.qa import QAUserContext

        factor = self._affinity(
            {"guide_region": "EU"},
            user_context=QAUserContext(country="EU"),
        )
        self.assertAlmostEqual(factor, 1.15, places=6)

    def test_age_group_matches_life_stage_facet(self):
        from models.qa import QAUserContext

        factor = self._affinity(
            {"life_stage": ["early_childhood"]},
            user_context=QAUserContext(member_age_group="toddler"),
        )
        self.assertAlmostEqual(factor, 1.15, places=6)

    def test_question_population_matches_target_populations(self):
        from models.qa import SubQuestionFilters

        factor = self._affinity(
            {"target_populations": ["pregnant_people"]},
            facets=SubQuestionFilters(target_populations=["pregnancy"]),
        )
        self.assertAlmostEqual(factor, 1.15, places=6)

    def test_topical_overlap_boosts(self):
        from models.qa import SubQuestionFilters

        factor = self._affinity(
            {"nutrients": ["fiber"], "health_conditions": ["diabetes"]},
            facets=SubQuestionFilters(health_conditions=["diabetes"]),
        )
        self.assertAlmostEqual(factor, 1.1, places=6)

    def test_full_match_is_bounded(self):
        from models.qa import QAUserContext, SubQuestionFilters

        factor = self._affinity(
            {
                "guide_region": "EU",
                "life_stage": ["adulthood"],
                "food_groups": ["vegetables"],
            },
            user_context=QAUserContext(country="EU", member_age_group="adult"),
            facets=SubQuestionFilters(food_groups=["vegetables"]),
        )
        self.assertAlmostEqual(factor, 1.15 * 1.15 * 1.1, places=6)
        self.assertLess(factor, 1.5)

    def test_affinity_lands_in_score_parts(self):
        from models.qa import QAUserContext

        item = _item(
            {"urn": "g1", "source_type": "guideline", "guide_region": "EU"},
            rrf_norm=0.5,
        )
        adjusted = ranking.adjust_evidence(
            [item], user_context=QAUserContext(country="EU")
        )
        self.assertAlmostEqual(adjusted[0].score_parts["affinity"], 1.15)
        self.assertAlmostEqual(adjusted[0].adjusted_score, 0.5 * 1.15, places=6)

    def test_articles_never_get_affinity(self):
        from models.qa import QAUserContext

        item = _item({"urn": "a1", "guide_region": "EU"}, rrf_norm=0.5)
        adjusted = ranking.adjust_evidence(
            [item], user_context=QAUserContext(country="EU")
        )
        self.assertEqual(adjusted[0].score_parts["affinity"], 1.0)


class RrfFusionTests(unittest.TestCase):
    def _leg(self, *urns):
        return [
            (f"article:{urn}", {"urn": urn, "source_type": "article"}, _source(urn))
            for urn in urns
        ]

    def test_document_in_both_legs_outranks_single_leg(self):
        fused = rrf_fuse([self._leg("a", "b"), self._leg("b", "c")])
        order = [item.payload["urn"] for item in fused]
        self.assertEqual(order[0], "b")
        self.assertEqual(len(fused), 3)

    def test_top_of_every_leg_normalizes_to_one(self):
        fused = rrf_fuse([self._leg("a"), self._leg("a")])
        self.assertAlmostEqual(fused[0].rrf_norm, 1.0, places=6)

    def test_empty_legs_do_not_dilute(self):
        fused = rrf_fuse([self._leg("a"), []])
        self.assertAlmostEqual(fused[0].rrf_norm, 1.0, places=6)

    def test_no_legs_returns_empty(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[], []]), [])


if __name__ == "__main__":
    unittest.main()
