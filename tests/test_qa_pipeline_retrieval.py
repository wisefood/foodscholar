"""Tests for the agentic pipeline's hybrid retrieval legs."""

import unittest
from unittest.mock import patch

from models.qa import PlannedSubQuestion
from services.qa_pipeline import retrieval


class _RecordingClient:
    """Fake ES `.client` that records bodies and returns canned hits."""

    def __init__(self, hits_by_index=None):
        self.calls = []
        self.hits_by_index = hits_by_index or {}

    def search(self, index, body):
        self.calls.append({"index": index, "body": body})
        return {"hits": {"hits": self.hits_by_index.get(index, [])}}


def _article_hit(urn, score=1.0, **fields):
    return {
        "_id": urn,
        "_score": score,
        "_source": {"urn": urn, "title": f"Title {urn}", **fields},
    }


def _guideline_hit(urn, score=1.0):
    return {
        "_id": urn,
        "_score": score,
        "_source": {
            "id": urn,
            "title": "Healthy eating guide",
            "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.",
            "guide_region": "EU",
        },
    }


def _sq(branch="both"):
    return PlannedSubQuestion(
        id="sq1",
        text="Should people choose whole grains?",
        why="Testing",
        branch=branch,
        lexical_query="whole grains health",
        dense_query="Are whole grains healthier than refined grains?",
    )


class ArticleLexicalLegTests(unittest.TestCase):
    def test_carries_editorial_prefilter_and_excludes_embedding(self):
        client = _RecordingClient({"articles": [_article_hit("a1")]})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client):
            leg = retrieval.search_articles_lexical(
                "whole grains", size=10, expertise_level="beginner"
            )

        body = client.calls[0]["body"]
        must_not = body["query"]["bool"]["must_not"]
        self.assertIn({"term": {"status": "deleted"}}, must_not)
        # A non-expert audience must not see expert_only articles.
        visibility = [c for c in must_not if "terms" in c and "reader_visibility" in c["terms"]]
        self.assertTrue(visibility)
        self.assertIn("expert_only", visibility[0]["terms"]["reader_visibility"])
        self.assertEqual(body["_source"], {"excludes": ["embedding"]})
        self.assertEqual(len(leg), 1)
        key, payload, source = leg[0]
        self.assertEqual(payload["source_type"], "article")
        self.assertEqual(source.urn, "a1")


class GuidelineLegTests(unittest.TestCase):
    def test_lexical_leg_uses_shared_gated_builder(self):
        client = _RecordingClient({"guidelines": [_guideline_hit("g1")]})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client):
            leg = retrieval.search_guidelines_lexical(
                "whole grains", size=5, user_context=None
            )

        body = client.calls[0]["body"]
        self.assertEqual(
            body["query"]["bool"]["filter"], [{"term": {"status": "active"}}]
        )
        self.assertEqual(body["_source"], {"excludes": ["embedding"]})
        self.assertEqual(len(leg), 1)
        self.assertEqual(leg[0][1]["source_type"], "guideline")

    def test_knn_leg_is_gated(self):
        captured = {}

        def fake_knn(**kwargs):
            captured.update(kwargs)
            return []

        with patch.object(retrieval.ELASTIC_CLIENT, "knn_search", side_effect=lambda **kw: fake_knn(**kw)):
            retrieval.search_guidelines_knn([0.1, 0.2], size=5)

        self.assertEqual(
            captured["filter_query"], [{"term": {"status": "active"}}]
        )
        self.assertEqual(captured["source_excludes"], ["embedding"])


class AttributeFilterTests(unittest.TestCase):
    """Metadata/attribute-aware retrieval: both legs are informed."""

    def _filters(self, **kwargs):
        from models.qa import SubQuestionFilters

        return SubQuestionFilters(**kwargs)

    def test_year_window_hard_filters_the_bm25_leg(self):
        client = _RecordingClient({"articles": []})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client):
            retrieval.search_articles_lexical(
                "omega-3 ldl",
                size=10,
                expertise_level=None,
                filters=self._filters(year_min=2020, study_types=["rct"]),
            )

        bool_query = client.calls[0]["body"]["query"]["bool"]
        self.assertEqual(
            bool_query["filter"],
            [{"range": {"publication_year": {"format": "yyyy", "gte": "2020"}}}],
        )
        # Study design boosts, never gates (ai_category coverage is partial).
        self.assertTrue(
            any("ai_category" in str(clause) for clause in bool_query["should"])
        )

    def test_knn_leg_is_informed_by_the_same_hard_filters(self):
        captured = {}

        with patch.object(
            retrieval.ELASTIC_CLIENT,
            "knn_search",
            side_effect=lambda **kw: captured.update(kw) or [],
        ):
            retrieval.search_articles_knn(
                [0.1] * 4,
                size=10,
                expertise_level=None,
                filters=self._filters(year_min=2020, open_access=True),
            )

        knn_filter = captured["filter_query"]
        self.assertIsInstance(knn_filter, list)
        self.assertIn(
            {"range": {"publication_year": {"format": "yyyy", "gte": "2020"}}},
            knn_filter,
        )
        self.assertIn({"term": {"open_access": True}}, knn_filter)
        # The editorial pre-filter is still there.
        self.assertTrue(any("must_not" in str(clause) for clause in knn_filter))

    def test_empty_filters_change_nothing(self):
        client = _RecordingClient({"articles": []})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client):
            retrieval.search_articles_lexical(
                "omega-3", size=10, expertise_level=None, filters=self._filters()
            )
        bool_query = client.calls[0]["body"]["query"]["bool"]
        self.assertNotIn("filter", bool_query)
        self.assertNotIn("should", bool_query)

    def test_guideline_attributes_are_boosts_never_filters(self):
        client = _RecordingClient({"guidelines": []})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client):
            retrieval.search_guidelines_lexical(
                "fiber intake",
                size=5,
                user_context=None,
                filters=self._filters(
                    regions=["EU"], target_populations=["pregnant_people"]
                ),
            )

        bool_query = client.calls[0]["body"]["query"]["bool"]
        # The editorial gate remains the only hard filter.
        self.assertEqual(bool_query["filter"], [{"term": {"status": "active"}}])
        self.assertTrue(bool_query["should"])
        for clause in bool_query["should"]:
            self.assertIn("multi_match", clause)


class RunBranchTests(unittest.TestCase):
    def test_article_branch_fuses_lexical_and_knn(self):
        client = _RecordingClient(
            {"articles": [_article_hit("a1"), _article_hit("a2", 0.5)]}
        )
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client), patch.object(
            retrieval.ELASTIC_CLIENT,
            "knn_search",
            return_value=[{"_id": "a2", "_score": 0.9, "urn": "a2", "title": "T"}],
        ):
            outcome = retrieval.run_branch(
                _sq(),
                branch="articles",
                vector=[0.1] * 4,
                user_context=None,
                expertise_level=None,
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.status["legs"]["lexical"], 2)
        self.assertEqual(outcome.status["legs"]["knn"], 1)
        # a2 appears in both legs and outranks a1.
        self.assertEqual(outcome.items[0].payload["urn"], "a2")
        self.assertEqual(outcome.items[0].sub_question_ids, ["sq1"])

    def test_leg_failure_degrades_not_fails(self):
        client = _RecordingClient({"articles": [_article_hit("a1")]})

        def broken_knn(**_kwargs):
            raise RuntimeError("es down")

        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client), patch.object(
            retrieval.ELASTIC_CLIENT, "knn_search", side_effect=broken_knn
        ):
            outcome = retrieval.run_branch(
                _sq(),
                branch="articles",
                vector=[0.1] * 4,
                user_context=None,
                expertise_level=None,
            )

        self.assertTrue(outcome.ok)
        self.assertIn("knn_error", outcome.status["legs"])
        self.assertEqual(len(outcome.items), 1)

    def test_all_legs_failing_reports_not_ok(self):
        class _Broken:
            def search(self, index, body):
                raise RuntimeError("es down")

        def broken_knn(**_kwargs):
            raise RuntimeError("es down")

        with patch.object(retrieval.ELASTIC_CLIENT, "_client", _Broken()), patch.object(
            retrieval.ELASTIC_CLIENT, "knn_search", side_effect=broken_knn
        ):
            outcome = retrieval.run_branch(
                _sq(),
                branch="articles",
                vector=[0.1] * 4,
                user_context=None,
                expertise_level=None,
            )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.items, [])

    def test_guideline_branch_runs_gated_knn_by_default(self):
        from config import config

        client = _RecordingClient({"guidelines": [_guideline_hit("g1")]})
        with patch.object(retrieval.ELASTIC_CLIENT, "_client", client), patch.object(
            retrieval.ELASTIC_CLIENT, "knn_search", return_value=[]
        ) as knn:
            outcome = retrieval.run_branch(
                _sq(branch="guidelines"),
                branch="guidelines",
                vector=[0.1] * 4,
                user_context=None,
                expertise_level=None,
            )

        # Guidelines are embedded, so hybrid is the default — and the vector
        # leg carries the editorial gate.
        knn.assert_called_once()
        self.assertEqual(
            knn.call_args.kwargs["filter_query"],
            [{"term": {"status": "active"}}],
        )
        self.assertTrue(outcome.ok)

        # bm25 mode (un-embedded deployments) skips the vector leg entirely.
        original = config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE")
        try:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "bm25"
            with patch.object(
                retrieval.ELASTIC_CLIENT, "_client", client
            ), patch.object(retrieval.ELASTIC_CLIENT, "knn_search") as knn_off:
                retrieval.run_branch(
                    _sq(branch="guidelines"),
                    branch="guidelines",
                    vector=[0.1] * 4,
                    user_context=None,
                    expertise_level=None,
                )
            knn_off.assert_not_called()
        finally:
            config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = original


class MergeEvidenceTests(unittest.TestCase):
    def test_rediscovered_document_accumulates_sub_questions(self):
        from services.qa_pipeline.state import EvidenceItem
        from models.qa import RetrievedSource

        def item(urn, sq_ids, rrf_norm):
            it = EvidenceItem(
                payload={"urn": urn, "source_type": "article"},
                source=RetrievedSource(urn=urn, title=urn, similarity_score=1.0),
            )
            it.sub_question_ids = sq_ids
            it.rrf_norm = rrf_norm
            return it

        merged = retrieval.merge_evidence(
            [item("a1", ["sq1"], 0.4)], [item("a1", ["sq2"], 0.7), item("a2", ["sq2"], 0.5)]
        )
        by_urn = {i.payload["urn"]: i for i in merged}
        self.assertEqual(len(merged), 2)
        self.assertEqual(by_urn["a1"].sub_question_ids, ["sq1", "sq2"])
        self.assertEqual(by_urn["a1"].rrf_norm, 0.7)


if __name__ == "__main__":
    unittest.main()
