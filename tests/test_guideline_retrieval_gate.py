"""
Tests for the editorial gate on guideline retrieval.

Guideline retrieval previously excluded only deleted rules, so drafts and
unreviewed rules could be cited in an answer or served as a daily tip. Every
user-facing path now goes through the shared helpers in ``qa_retrievers``, and
these tests exist to keep it that way: a path that builds its own query body is
a path that will drift out of the gate again.
"""

import inspect
import unittest
from unittest.mock import patch


class GuidelineRetrievalFilterTests(unittest.TestCase):
    def test_only_active_guidelines_are_retrievable(self):
        from services.qa_retrievers import guideline_retrieval_filter

        self.assertEqual(
            guideline_retrieval_filter(), [{"term": {"status": "active"}}]
        )

    def test_base_query_carries_the_filter(self):
        from services.qa_retrievers import guideline_base_query

        query = guideline_base_query("whole grains")

        self.assertEqual(query["bool"]["filter"], [{"term": {"status": "active"}}])
        self.assertEqual(
            query["bool"]["must"][0]["multi_match"]["query"], "whole grains"
        )

    def test_tip_pool_is_gated_with_and_without_a_query(self):
        from services.qa_retrievers import guideline_tip_pool_query

        for query in (None, "iron"):
            pool = guideline_tip_pool_query(query)
            self.assertEqual(
                pool["bool"]["filter"],
                [{"term": {"status": "active"}}],
                msg=f"tip pool query={query!r} is not gated",
            )
            # rule_text is what a tip is generated from; a rule without one is
            # unusable regardless of status.
            self.assertIn({"exists": {"field": "rule_text"}}, pool["bool"]["must"])

    def test_tip_pool_query_terms_are_searched(self):
        from services.qa_retrievers import guideline_tip_pool_query

        pool = guideline_tip_pool_query("iron")
        matchers = [
            clause for clause in pool["bool"]["must"] if "multi_match" in clause
        ]
        self.assertEqual(len(matchers), 1)
        self.assertEqual(matchers[0]["multi_match"]["query"], "iron")


class GuidelineSearchFieldTests(unittest.TestCase):
    # Fields that were being searched and boosted but do not exist in the
    # guideline mapping, so they never matched anything.
    PHANTOM_FIELDS = ("country", "population", "reader_group", "source")

    def _field_names(self, fields):
        return {field.split("^", 1)[0] for field in fields}

    def test_no_phantom_fields_are_searched(self):
        from services.qa_retrievers import GUIDELINE_SEARCH_FIELDS

        names = self._field_names(GUIDELINE_SEARCH_FIELDS)
        for phantom in self.PHANTOM_FIELDS:
            self.assertNotIn(phantom, names)

    def test_enrichment_facets_are_searched(self):
        from services.qa_retrievers import GUIDELINE_SEARCH_FIELDS

        names = self._field_names(GUIDELINE_SEARCH_FIELDS)
        for facet in ("life_stage", "nutrients", "health_conditions", "topic"):
            self.assertIn(facet, names)

    def test_rule_text_outranks_everything(self):
        from services.qa_retrievers import GUIDELINE_SEARCH_FIELDS

        boosts = {}
        for field in GUIDELINE_SEARCH_FIELDS:
            name, _, boost = field.partition("^")
            boosts[name] = float(boost) if boost else 1.0

        self.assertEqual(boosts["rule_text"], max(boosts.values()))


class GuidelineContextClauseTests(unittest.TestCase):
    def test_no_user_context_yields_no_clauses(self):
        from services.qa_retrievers import guideline_context_should_clauses

        self.assertEqual(guideline_context_should_clauses(None), [])

    AGE_WINDOW_FIELDS = {"age_min_months", "age_max_months"}

    def _referenced_fields(self, clauses):
        referenced = set()
        for clause in clauses:
            if "multi_match" in clause:
                for field in clause["multi_match"]["fields"]:
                    referenced.add(field.split("^", 1)[0])
            elif "bool" in clause:
                # The age-window boost: ranges over the mapped month fields.
                for sub in clause["bool"].get("must", []):
                    referenced.update(sub.get("range", {}).keys())
        return referenced

    def test_context_clauses_only_reference_mapped_fields(self):
        from models.qa import QAUserContext
        from services.qa_retrievers import guideline_context_should_clauses

        clauses = guideline_context_should_clauses(
            QAUserContext(country="IE", member_age_group="toddler")
        )

        self.assertTrue(clauses)
        referenced = self._referenced_fields(clauses)
        for phantom in GuidelineSearchFieldTests.PHANTOM_FIELDS:
            self.assertNotIn(phantom, referenced)

    def test_context_clauses_are_boosts_not_filters(self):
        from models.qa import QAUserContext
        from services.qa_retrievers import guideline_context_should_clauses

        clauses = guideline_context_should_clauses(
            QAUserContext(country="IE", member_age_group="toddler")
        )

        # A `should` clause reorders; anything else would silently hide the
        # guidelines of every other country from a user who set one. The
        # age-window clause is a bool-with-boost over the month ranges — a
        # scoring clause, never a top-level filter.
        for clause in clauses:
            self.assertNotIn("filter", clause)
            if "bool" in clause:
                self.assertIn("boost", clause["bool"])
                self.assertTrue(
                    self.AGE_WINDOW_FIELDS.issuperset(
                        self._referenced_fields([clause])
                    )
                )
            else:
                self.assertIn("multi_match", clause)

    def test_age_group_maps_to_life_stage_and_month_window(self):
        from services.qa_retrievers import guideline_age_should_clauses

        clauses = guideline_age_should_clauses("toddler")
        self.assertEqual(len(clauses), 2)
        life_stage_clause, window_clause = clauses
        self.assertIn(
            "early_childhood", life_stage_clause["multi_match"]["query"]
        )
        musts = window_clause["bool"]["must"]
        self.assertEqual(
            musts[0]["range"]["age_min_months"], {"gte": 0, "lte": 48}
        )
        self.assertEqual(musts[1]["range"]["age_max_months"], {"gte": 12})

        # Pregnancy has a life stage but no month window.
        pregnancy = guideline_age_should_clauses("pregnant")
        self.assertEqual(len(pregnancy), 1)
        self.assertIn("pregnancy", pregnancy[0]["multi_match"]["query"])

        # Unknown groups add nothing rather than guessing.
        self.assertEqual(guideline_age_should_clauses("martian"), [])
        self.assertEqual(guideline_age_should_clauses(None), [])


class HybridRetrievalTests(unittest.TestCase):
    """
    Hybrid retrieval is the default now that guidelines carry embeddings, and
    it must gate the vector leg exactly as it gates the keyword leg — a kNN
    query with no filter would surface draft rules the BM25 leg correctly
    hides. bm25-only remains the opt-out for un-embedded deployments.
    """

    def test_hybrid_is_on_by_default(self):
        from config import config
        from services.qa_retrievers import guideline_hybrid_enabled

        self.assertEqual(config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE"), "hybrid")
        self.assertTrue(guideline_hybrid_enabled())

    def test_flag_toggles_hybrid(self):
        from config import config
        from services.qa_retrievers import guideline_hybrid_enabled

        original = config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE")
        try:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "hybrid"
            self.assertTrue(guideline_hybrid_enabled())
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "HYBRID"
            self.assertTrue(guideline_hybrid_enabled())
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "bm25"
            self.assertFalse(guideline_hybrid_enabled())
        finally:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = original

    def test_hybrid_query_gates_the_vector_leg(self):
        from config import config
        from services.qa_retrievers import ElasticRagRetrieverAdapter

        captured = {}

        class _Client:
            def search(self, index, body):
                captured["body"] = body
                return {"hits": {"hits": []}}

        adapter = ElasticRagRetrieverAdapter(
            embed_query=lambda _: [0.1, 0.2, 0.3],
            articles_index="articles",
            guidelines_index="guidelines",
        )

        original = config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE")
        try:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "hybrid"
            with patch("services.qa_retrievers.ELASTIC_CLIENT") as elastic:
                elastic.client = _Client()
                _, _, status = adapter._retrieve_guidelines("whole grains", 5, None)
        finally:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = original

        body = captured["body"]
        self.assertEqual(status["mode"], "hybrid")
        self.assertEqual(body["knn"]["filter"], [{"term": {"status": "active"}}])
        self.assertEqual(body["query"]["bool"]["filter"], [{"term": {"status": "active"}}])
        self.assertEqual(body["_source"], {"excludes": ["embedding"]})

    def test_embedding_failure_degrades_to_keyword(self):
        from config import config
        from services.qa_retrievers import ElasticRagRetrieverAdapter

        captured = {}

        class _Client:
            def search(self, index, body):
                captured["body"] = body
                return {"hits": {"hits": []}}

        def _broken_embed(_):
            raise RuntimeError("embedding service down")

        adapter = ElasticRagRetrieverAdapter(
            embed_query=_broken_embed,
            articles_index="articles",
            guidelines_index="guidelines",
        )

        original = config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE")
        try:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "hybrid"
            with patch("services.qa_retrievers.ELASTIC_CLIENT") as elastic:
                elastic.client = _Client()
                _, _, status = adapter._retrieve_guidelines("whole grains", 5, None)
        finally:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = original

        # The answer still gets guideline evidence; it is just keyword-ranked.
        self.assertEqual(status["mode"], "keyword")
        self.assertNotIn("knn", captured["body"])
        self.assertEqual(
            captured["body"]["query"]["bool"]["filter"], [{"term": {"status": "active"}}]
        )


class SourceExcludeTests(unittest.TestCase):
    """
    The embedding vector is never read by any consumer, but once guidelines are
    embedded it dominates every hit — 384 floats per result, deserialized and
    thrown away. Every guideline read path must exclude it.
    """

    def test_bm25_query_excludes_the_vector(self):
        from services.qa_retrievers import ElasticRagRetrieverAdapter

        captured = {}

        class _Client:
            def search(self, index, body):
                captured["body"] = body
                return {"hits": {"hits": []}}

        adapter = ElasticRagRetrieverAdapter(
            embed_query=lambda _: [0.1],
            articles_index="articles",
            guidelines_index="guidelines",
        )
        with patch("services.qa_retrievers.ELASTIC_CLIENT") as elastic:
            elastic.client = _Client()
            adapter._retrieve_guidelines("whole grains", 5, None)

        self.assertEqual(captured["body"]["_source"]["excludes"], ["embedding"])

    def test_tip_pool_paths_exclude_the_vector(self):
        import inspect
        from services.qa_service import QAService

        for method in (
            QAService._search_tip_source_guidelines,
            QAService._get_random_tip_source_guidelines,
        ):
            with self.subTest(method=method.__qualname__):
                self.assertIn(
                    "GUIDELINE_SOURCE_EXCLUDES", inspect.getsource(method)
                )


class NoDuplicateQueryBodyTests(unittest.TestCase):
    """
    The gate is only as good as its weakest call site. These tests read the
    source of every guideline-retrieving function and fail if one reintroduces
    a hand-built query body.
    """

    def _source_of(self, obj) -> str:
        return inspect.getsource(obj)

    def test_adapter_uses_the_shared_query_builder(self):
        from services.qa_retrievers import ElasticRagRetrieverAdapter

        source = self._source_of(ElasticRagRetrieverAdapter._retrieve_guidelines)

        self.assertIn("guideline_base_query", source)
        self.assertNotIn("must_not", source)

    def test_pipeline_guideline_leg_uses_the_shared_builders(self):
        from services.qa_pipeline import retrieval as pipeline_retrieval

        source = self._source_of(pipeline_retrieval.search_guidelines_lexical)
        self.assertIn("guideline_base_query", source)
        self.assertNotIn("must_not", source)
        self.assertNotIn(
            "status",
            source.replace("guideline_base_query", ""),
            msg="pipeline guideline leg appears to filter on status itself",
        )

    def test_pipeline_guideline_knn_leg_is_gated(self):
        from services.qa_pipeline import retrieval as pipeline_retrieval

        source = self._source_of(pipeline_retrieval.search_guidelines_knn)
        self.assertIn("guideline_retrieval_filter", source)
        self.assertNotIn(
            "status",
            source.replace("guideline_retrieval_filter", ""),
            msg="pipeline guideline kNN leg appears to filter on status itself",
        )

    def test_qa_service_guideline_paths_use_the_shared_builders(self):
        from services.qa_service import QAService

        for method, builder in (
            (QAService._search_tip_source_guidelines, "guideline_tip_pool_query"),
            (QAService._get_random_tip_source_guidelines, "guideline_tip_pool_query"),
        ):
            source = self._source_of(method)
            self.assertIn(
                builder,
                source,
                msg=f"{method.__qualname__} does not use {builder}",
            )
            self.assertNotIn(
                "status",
                source.replace("guideline_tip_pool_query", "").replace(
                    "guideline_base_query", ""
                ),
                msg=f"{method.__qualname__} appears to filter on status itself",
            )


if __name__ == "__main__":
    unittest.main()
