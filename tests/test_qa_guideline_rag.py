import unittest
from unittest.mock import patch


class _FakeGuidelineSearchClient:
    def search(self, index, body):
        assert index == "guidelines"
        # This test pins QA_GUIDELINE_RETRIEVAL_MODE=bm25, so no vector leg.
        # Checked structurally rather than by searching the serialized body
        # for "embedding", because the query legitimately names that field in
        # `_source.excludes`.
        assert "knn" not in body
        # The vector is never returned; once the corpus is embedded it would
        # otherwise be the bulk of every hit.
        assert body["_source"]["excludes"] == ["embedding"]
        fields = body["query"]["bool"]["must"][0]["multi_match"]["fields"]
        assert "rule_text^4" in fields
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "guideline-es-1",
                        "_score": 7.5,
                        "_source": {
                            "id": "guideline-1",
                            "title": "Healthy eating guide",
                            "rule_text": (
                                "Choose whole-grain cereals, bread, rice, "
                                "or pasta more often."
                            ),
                            "food_groups": ["whole grains"],
                            "guide_region": "EU",
                        },
                    }
                ]
            }
        }


class QAGuidelineRagTests(unittest.TestCase):
    def test_default_rag_retrieves_articles_and_guidelines(self):
        import services.qa_retrievers as qa_retrievers_module
        from config import config
        from services.qa_service import QAService

        original_mode = config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE")
        config.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = "bm25"
        self.addCleanup(
            config.settings.__setitem__,
            "QA_GUIDELINE_RETRIEVAL_MODE",
            original_mode,
        )

        service = QAService(cache_enabled=False)
        service._embed_query = lambda _question: [0.1, 0.2]  # type: ignore[assignment]

        article = {
            "urn": "urn:article:whole-grains",
            "title": "Whole grains and diet quality",
            "authors": ["Doe"],
            "publication_year": "2021-01-01",
            "abstract": "Whole-grain intake is associated with higher diet quality in adults.",
            "_score": 0.91,
        }

        with patch.object(
            qa_retrievers_module.ELASTIC_CLIENT,
            "knn_search",
            return_value=[article],
        ), patch.object(
            qa_retrievers_module.ELASTIC_CLIENT,
            "_client",
            _FakeGuidelineSearchClient(),
        ):
            sources, retrieved = service._retrieve_articles(
                "Should people choose whole grains?",
                top_k=2,
                retriever="rag",
            )

        self.assertEqual(
            [s["source_type"] for s in sources],
            ["article", "guideline"],
        )
        self.assertEqual(sources[1]["urn"], "guideline-1")
        self.assertEqual(
            sources[1]["abstract"],
            "Choose whole-grain cereals, bread, rice, or pasta more often.",
        )
        self.assertEqual(
            [r.source_type for r in retrieved],
            ["article", "guideline"],
        )
        self.assertEqual(retrieved[1].category, "guideline")

    def test_qa_agent_accepts_guideline_citations(self):
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent.model = "test-model"

        guideline = {
            "source_type": "guideline",
            "urn": "guideline-1",
            "title": "Healthy eating guide",
            "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.",
            "publication_year": "2024-01-01",
            "guide_urn": "urn:guide:eu-healthy-eating",
            "guide_region": "EU",
            "page_no": 12,
        }
        parsed = {
            "answer": (
                "Choose whole grains more often "
                "[Dietary guideline: Healthy eating guide](/guidelines/guideline-1)."
            ),
            "cited_sources": [
                {
                    "urn": "guideline-1",
                    "section": "rule_text",
                    "quote": guideline["rule_text"],
                    "confidence": "high",
                }
            ],
            "overall_confidence": "high",
        }

        answer = agent._build_qa_answer(
            parsed,
            question="Should people choose whole grains?",
            articles=[guideline],
            rag_used=True,
        )

        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].source_type, "guideline")
        self.assertEqual(answer.citations[0].source_id, "guideline-1")
        self.assertEqual(answer.citations[0].source_title, "Healthy eating guide")
        self.assertEqual(answer.citations[0].section, "rule_text")
        self.assertEqual(answer.citations[0].quote, guideline["rule_text"])
        # Guide-routing hints ride the citation so the UI can land on the
        # guide page (rule highlighted, PDF page open) even when the rule
        # itself is not publicly readable.
        self.assertEqual(
            answer.citations[0].guide_urn, "urn:guide:eu-healthy-eating"
        )
        self.assertEqual(answer.citations[0].region, "EU")
        self.assertEqual(answer.citations[0].page_no, 12)


class QuoteContextTests(unittest.TestCase):
    """Citations carry the source text around the quote, for hover previews."""

    def test_citation_carries_surrounding_fragments(self):
        from agents.qa_agent import build_qa_answer

        abstract = (
            "Background sentence setting the scene for the trial. "
            "Whole-grain intake reduced LDL cholesterol in adults. "
            "Effects were consistent across all measured cohorts."
        )
        article = {
            "source_type": "article",
            "urn": "urn:article:a1",
            "title": "Whole grains and lipids",
            "abstract": abstract,
        }
        parsed = {
            "answer": "Whole grains help [X](/articles/urn:article:a1).",
            "cited_sources": [
                {
                    "urn": "urn:article:a1",
                    "section": "abstract",
                    "quote": "Whole-grain intake reduced LDL cholesterol in adults.",
                    "confidence": "high",
                }
            ],
            "overall_confidence": "high",
        }
        answer = build_qa_answer(
            parsed,
            question="Do whole grains lower cholesterol?",
            articles=[article],
            rag_used=True,
            model_used="test-model",
        )
        citation = answer.citations[0]
        self.assertEqual(
            citation.quote_context_before,
            "Background sentence setting the scene for the trial.",
        )
        self.assertEqual(
            citation.quote_context_after,
            "Effects were consistent across all measured cohorts.",
        )

    def test_quote_at_source_start_has_no_before_fragment(self):
        from agents.qa_agent import quote_context

        before, after = quote_context("Quote here. Then more text.", "Quote here.")
        self.assertEqual(before, "")
        self.assertEqual(after, "Then more text.")

    def test_long_context_is_trimmed_to_word_boundaries(self):
        from agents.qa_agent import quote_context

        source = ("word " * 100).strip() + " THE QUOTE. " + ("tail " * 100).strip()
        before, after = quote_context(source, "THE QUOTE.")
        self.assertLessEqual(len(before), 180)
        self.assertLessEqual(len(after), 180)
        self.assertFalse(before.startswith("ord"))  # no mid-word cut
        self.assertTrue(after.endswith("tail"))

    def test_missing_quote_or_text_is_empty(self):
        from agents.qa_agent import quote_context

        self.assertEqual(quote_context("", "q"), ("", ""))
        self.assertEqual(quote_context("text", None), ("", ""))
        self.assertEqual(quote_context("text", "absent"), ("", ""))


class SourceContextFacetTests(unittest.TestCase):
    """The answer prompt must surface applicability facets and bibliometrics."""

    def test_guideline_block_carries_enrichment_facets(self):
        from agents.qa_agent import prepare_source_context

        block = prepare_source_context(
            [
                {
                    "source_type": "guideline",
                    "urn": "guideline-1",
                    "title": "Toddler eating guide",
                    "rule_text": "Provide portions of red meat twice a week.",
                    "guide_region": "IE",
                    "life_stage": ["early_childhood"],
                    "age_min_months": 12,
                    "age_max_months": 48,
                    "nutrients": ["iron"],
                    "health_conditions": ["anemia"],
                    "action_type": "eat",
                    "frequency": "weekly",
                }
            ]
        )
        self.assertIn("Life Stage: early_childhood", block)
        self.assertIn("Applies To Ages: 1 year to 4 years", block)
        self.assertIn("Nutrients: iron", block)
        self.assertIn("Health Conditions: anemia", block)
        self.assertIn("Action/Frequency: eat / weekly", block)

    def test_guideline_block_omits_absent_facets(self):
        from agents.qa_agent import prepare_source_context

        block = prepare_source_context(
            [
                {
                    "source_type": "guideline",
                    "urn": "guideline-2",
                    "title": "General guide",
                    "rule_text": "Eat a variety of vegetables every day.",
                    "age_min_months": -1,
                    "age_max_months": -1,
                }
            ]
        )
        self.assertNotIn("Life Stage:", block)
        self.assertNotIn("Applies To Ages:", block)
        self.assertNotIn("Health Conditions:", block)

    def test_article_block_carries_citation_counts(self):
        from agents.qa_agent import prepare_source_context

        block = prepare_source_context(
            [
                {
                    "source_type": "article",
                    "urn": "urn:article:a1",
                    "title": "Whole grains and lipids",
                    "abstract": "Whole grains lower LDL.",
                    "citationCount": 240,
                    "influentialCitationCount": 12,
                }
            ]
        )
        self.assertIn("Citations: 240 (influential: 12)", block)

    def test_article_block_omits_unstored_bibliometrics(self):
        from agents.qa_agent import prepare_source_context

        block = prepare_source_context(
            [
                {
                    "source_type": "article",
                    "urn": "urn:article:a2",
                    "title": "Uncounted",
                    "abstract": "No metrics yet.",
                }
            ]
        )
        self.assertNotIn("Citations:", block)


if __name__ == "__main__":
    unittest.main()
