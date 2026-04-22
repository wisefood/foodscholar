import unittest
from unittest.mock import patch


class _FakeGuidelineSearchClient:
    def search(self, index, body):
        assert index == "guidelines"
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
        from services.qa_service import QAService

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


if __name__ == "__main__":
    unittest.main()
