"""
Tests for editorial policy enforcement during retrieval.

The policy itself lives in the catalog (wisefood-data-api ArticleSchema:
`reader_visibility`, `indexing_tier`, `ai_indexing_tier`). FoodScholar only reads
those fields off retrieval payloads, so the cases that matter are: an article
predating the fields must behave exactly as before, a restricted article must not
reach the wrong audience, and the tier must move ranking without reordering
guidelines.
"""

import unittest


class _Source:
    """Stand-in for models.qa.RetrievedSource."""

    def __init__(self, urn, *, source_type="article", similarity_score=1.0):
        self.urn = urn
        self.source_type = source_type
        self.similarity_score = similarity_score


def _article(urn, *, score=1.0, **fields):
    return {"urn": urn, "source_type": "article", "_score": score, **fields}


def _guideline(urn, *, score=1.0):
    return {"urn": urn, "source_type": "guideline", "_score": score}


class TestVocabulary(unittest.TestCase):
    def test_model_tier_spellings_normalize_to_catalog_slugs(self):
        from services.article_policy import normalize_tier

        # The enrichment prompt emits title case; the catalog stores slugs.
        self.assertEqual(normalize_tier("Archive-only"), "archive_only")
        self.assertEqual(normalize_tier("Do not index"), "do_not_index")
        self.assertEqual(normalize_tier("Core"), "core")
        self.assertEqual(normalize_tier("prime"), "prime")

    def test_unknown_tier_is_none(self):
        from services.article_policy import normalize_tier

        self.assertIsNone(normalize_tier("platinum"))
        self.assertIsNone(normalize_tier(""))
        self.assertIsNone(normalize_tier(None))

    def test_absent_tier_is_neutral(self):
        from services.article_policy import tier_boost

        self.assertEqual(tier_boost(None), 1.0)
        self.assertEqual(tier_boost("supportive"), 1.0)

    def test_prime_outranks_core(self):
        from services.article_policy import tier_boost

        self.assertGreater(tier_boost("prime"), tier_boost("core"))
        self.assertGreater(tier_boost("core"), tier_boost("supportive"))
        self.assertLess(tier_boost("archive_only"), 1.0)

    def test_only_expert_is_an_expert_audience(self):
        from services.article_policy import is_expert_audience

        self.assertTrue(is_expert_audience("expert"))
        self.assertTrue(is_expert_audience("Expert"))
        self.assertFalse(is_expert_audience("intermediate"))
        self.assertFalse(is_expert_audience("beginner"))
        self.assertFalse(is_expert_audience(None))


class TestLegacyArticles(unittest.TestCase):
    """Articles indexed before the fields existed must be unaffected."""

    def test_article_without_policy_fields_is_public(self):
        from services.article_policy import is_visible_to

        legacy = _article("urn:article:old")

        self.assertTrue(is_visible_to(legacy, expert=False))
        self.assertTrue(is_visible_to(legacy, expert=True))

    def test_article_without_tier_has_no_effective_tier(self):
        from services.article_policy import effective_tier

        self.assertIsNone(effective_tier(_article("urn:article:old")))

    def test_legacy_payloads_keep_their_order(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:a", score=0.9),
            _article("urn:article:b", score=0.8),
            _article("urn:article:c", score=0.7),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="beginner")

        self.assertEqual([p["urn"] for p in ranked], [
            "urn:article:a",
            "urn:article:b",
            "urn:article:c",
        ])

    def test_es_filter_excludes_rather_than_requires(self):
        from services.article_policy import article_filter_query

        # A `must` on reader_visibility:public would drop every legacy article.
        query = article_filter_query("beginner")

        self.assertNotIn("must", query["bool"])
        self.assertIn("must_not", query["bool"])


class TestReaderVisibility(unittest.TestCase):
    def test_expert_only_is_dropped_for_non_experts(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:open"),
            _article("urn:article:gated", reader_visibility="expert_only"),
        ]

        for level in ("beginner", "intermediate", None):
            ranked, _ = filter_and_rank(payloads, [], expertise_level=level)
            self.assertEqual(
                [p["urn"] for p in ranked], ["urn:article:open"], f"level={level}"
            )

    def test_expert_only_is_kept_for_experts(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:open"),
            _article("urn:article:gated", reader_visibility="expert_only"),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="expert")

        self.assertEqual(len(ranked), 2)

    def test_hidden_is_dropped_for_everyone(self):
        from services.article_policy import filter_and_rank

        payloads = [_article("urn:article:dead", reader_visibility="hidden")]

        for level in ("beginner", "expert"):
            ranked, _ = filter_and_rank(payloads, [], expertise_level=level)
            self.assertEqual(ranked, [], f"level={level}")

    def test_restricted_sources_are_dropped_alongside_payloads(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:open"),
            _article("urn:article:gated", reader_visibility="expert_only"),
        ]
        sources = [_Source("urn:article:open"), _Source("urn:article:gated")]

        _, ranked_sources = filter_and_rank(
            payloads, sources, expertise_level="beginner"
        )

        self.assertEqual([s.urn for s in ranked_sources], ["urn:article:open"])

    def test_guidelines_are_never_filtered(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:gated", reader_visibility="expert_only"),
            _guideline("guideline:1"),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="beginner")

        self.assertEqual([p["urn"] for p in ranked], ["guideline:1"])

    def test_unknown_visibility_value_is_treated_as_public(self):
        from services.article_policy import is_visible_to

        article = _article("urn:article:x", reader_visibility="somethingelse")

        self.assertTrue(is_visible_to(article, expert=False))


class TestElasticFilter(unittest.TestCase):
    def test_non_expert_filter_excludes_both_restricted_states(self):
        from services.article_policy import reader_visibility_filter

        excluded = reader_visibility_filter("beginner")["terms"]["reader_visibility"]

        self.assertIn("expert_only", excluded)
        self.assertIn("hidden", excluded)

    def test_expert_filter_excludes_only_hidden(self):
        from services.article_policy import reader_visibility_filter

        excluded = reader_visibility_filter("expert")["terms"]["reader_visibility"]

        self.assertEqual(excluded, ["hidden"])

    def test_deleted_articles_stay_excluded(self):
        from services.article_policy import article_filter_query

        must_not = article_filter_query("expert")["bool"]["must_not"]

        self.assertIn({"term": {"status": "deleted"}}, must_not)

    def test_do_not_index_tier_is_excluded_in_the_query(self):
        from services.article_policy import article_filter_query

        must_not = article_filter_query("expert")["bool"]["must_not"]

        self.assertIn({"terms": {"indexing_tier": ["do_not_index"]}}, must_not)


class TestTierRanking(unittest.TestCase):
    def test_prime_article_is_promoted_above_a_better_scoring_one(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:good", score=1.0),
            _article("urn:article:influential", score=0.8, indexing_tier="prime"),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="beginner")

        # 0.8 * 1.6 = 1.28 beats 1.0 * 1.0
        self.assertEqual(ranked[0]["urn"], "urn:article:influential")

    def test_archive_only_article_sinks(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:archived", score=1.0, indexing_tier="archive_only"),
            _article("urn:article:normal", score=0.8),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="beginner")

        self.assertEqual(ranked[0]["urn"], "urn:article:normal")

    def test_editor_tier_overrides_the_agent_proposal(self):
        from services.article_policy import effective_tier

        payload = _article(
            "urn:article:x", indexing_tier="prime", ai_indexing_tier="archive_only"
        )

        self.assertEqual(effective_tier(payload), "prime")

    def test_agent_proposal_applies_when_no_editor_tier(self):
        from services.article_policy import effective_tier

        payload = _article("urn:article:x", ai_indexing_tier="core")

        self.assertEqual(effective_tier(payload), "core")

    def test_tier_from_pre_field_extras_is_honoured(self):
        from services.article_policy import effective_tier

        # Articles enriched before ai_indexing_tier became a catalog field.
        payload = _article(
            "urn:article:x", extras={"evaluation": {"indexing_tier": "Core"}}
        )

        self.assertEqual(effective_tier(payload), "core")

    def test_guidelines_keep_their_block_position(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:a", score=0.5),
            _article("urn:article:b", score=0.9, indexing_tier="prime"),
            _guideline("guideline:1", score=0.1),
        ]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="beginner")

        # Articles reorder among themselves; the guideline stays last.
        self.assertEqual(
            [p["urn"] for p in ranked],
            ["urn:article:b", "urn:article:a", "guideline:1"],
        )

    def test_ranking_annotates_why(self):
        from services.article_policy import filter_and_rank

        payloads = [_article("urn:article:x", score=2.0, indexing_tier="prime")]
        ranked, _ = filter_and_rank(payloads, [], expertise_level="expert")

        self.assertEqual(ranked[0]["effective_indexing_tier"], "prime")
        self.assertEqual(ranked[0]["editorial_boost"], 1.6)
        self.assertAlmostEqual(ranked[0]["editorial_score"], 3.2)

    def test_sources_are_ranked_using_their_payload_policy(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:a", score=1.0),
            _article("urn:article:b", score=0.8, indexing_tier="prime"),
        ]
        sources = [
            _Source("urn:article:a", similarity_score=1.0),
            _Source("urn:article:b", similarity_score=0.8),
        ]
        _, ranked_sources = filter_and_rank(
            payloads, sources, expertise_level="expert"
        )

        self.assertEqual([s.urn for s in ranked_sources], [
            "urn:article:b",
            "urn:article:a",
        ])

    def test_limit_truncates_articles_but_not_guidelines(self):
        from services.article_policy import filter_and_rank

        payloads = [
            _article("urn:article:a", score=0.9),
            _article("urn:article:b", score=0.8),
            _article("urn:article:c", score=0.7),
            _guideline("guideline:1"),
        ]
        ranked, _ = filter_and_rank(
            payloads, [], expertise_level="beginner", limit=2
        )

        urns = [p["urn"] for p in ranked]
        self.assertEqual(urns, ["urn:article:a", "urn:article:b", "guideline:1"])


class TestEnrichmentTierProposal(unittest.TestCase):
    def test_model_tier_is_written_to_the_ai_field(self):
        from services.enrichment_jobs import extract_enrichment_fields

        enhance, _, _ = extract_enrichment_fields(
            {"evaluation": {"indexing_tier": "Archive-only"}}
        )

        self.assertEqual(enhance["ai_indexing_tier"], "archive_only")

    def test_enrichment_never_writes_the_editorial_field(self):
        from services.enrichment_jobs import extract_enrichment_fields

        enhance, article_fields, extras = extract_enrichment_fields(
            {"evaluation": {"indexing_tier": "Core"}}
        )

        self.assertNotIn("indexing_tier", enhance)
        self.assertNotIn("indexing_tier", article_fields)
        self.assertNotIn("reader_visibility", article_fields)

    def test_unknown_model_tier_is_dropped(self):
        from services.enrichment_jobs import extract_enrichment_fields

        enhance, _, _ = extract_enrichment_fields(
            {"evaluation": {"indexing_tier": "Platinum"}}
        )

        self.assertNotIn("ai_indexing_tier", enhance)


if __name__ == "__main__":
    unittest.main()
