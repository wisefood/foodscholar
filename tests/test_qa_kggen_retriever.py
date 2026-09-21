"""The `kggen` retriever adapter: Extended KG-Gen hybrid retrieval.

Covers the seam between the library's chunk-shaped results and the QA answer
layer's source vocabulary. The scoring itself is the library's and is tested
there (`tests/unit/test_retrieval_kggen.py` in foodscholar-lib).
"""

import unittest
from unittest.mock import patch

from models.qa import QAClarifierSafetyPlan
from services.qa_retrievers import (
    KGGenRetrieverAdapter,
    QARetrieverAdapters,
    kggen_source_type,
)


def _plan(article_query="salt and blood pressure"):
    return QAClarifierSafetyPlan(
        original_question="Does salt raise blood pressure?",
        canonical_question="Does salt raise blood pressure?",
        article_query=article_query,
        guideline_query="salt intake guideline",
    )


def _hit(chunk_id, *, chunk_source_type="abstract", score=0.9, **fields):
    base = {
        "chunk_id": chunk_id,
        "text": f"Passage {chunk_id} about sodium intake.",
        "score": score,
        "text_sim": 0.8,
        "triplet_sim": 0.6,
        "ppr_score": 0.4,
        "chunk_source_type": chunk_source_type,
        "source_doc_id": f"doc-{chunk_id}",
    }
    base.update(fields)
    return base


def _trace(**overrides):
    base = {
        "candidates": 100,
        "relations": 40,
        "relations_scored": 40,
        "subgraph_nodes": 25,
        "subgraph_edges": 30,
        "expansion_calls": 10,
        "seed_entities": ["salt"],
        "branches": ["text", "triplet", "ppr"],
        "embedder": "BAAI/bge-base-en-v1.5",
        "degraded": [],
    }
    base.update(overrides)
    return base


class TestSourceTypeMapping(unittest.TestCase):
    """The library types a corpus by document shape; QA types it by citability."""

    def test_guide_chunks_become_guidelines(self):
        self.assertEqual(kggen_source_type({"chunk_source_type": "guide"}), "guideline")

    def test_abstracts_and_textbooks_become_articles(self):
        self.assertEqual(kggen_source_type({"chunk_source_type": "abstract"}), "article")
        self.assertEqual(kggen_source_type({"chunk_source_type": "textbook"}), "article")

    def test_an_unknown_or_missing_type_falls_back_to_article(self):
        # Articles are the conservative default: a guideline is cited as a rule
        # to follow, so mislabelling an article as one overstates its authority.
        self.assertEqual(kggen_source_type({}), "article")
        self.assertEqual(kggen_source_type({"chunk_source_type": "novel"}), "article")


class TestKGGenAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = KGGenRetrieverAdapter()

    def _retrieve(self, hits, trace=None):
        with patch(
            "services.kggen_service.retrieve",
            return_value=(hits, trace or _trace()),
        ) as mock:
            result = self.adapter.retrieve(
                question="Does salt raise blood pressure?",
                plan=_plan(),
                top_k=5,
                user_context=None,
            )
        return result, mock

    def test_it_is_registered_under_its_own_name(self):
        registry = QARetrieverAdapters(
            embed_query=lambda _: [0.0],
            articles_index="articles",
            guidelines_index="guidelines",
        )
        self.assertIsInstance(registry.get("kggen"), KGGenRetrieverAdapter)

    def test_an_unknown_retriever_still_falls_back_to_rag(self):
        registry = QARetrieverAdapters(
            embed_query=lambda _: [0.0],
            articles_index="articles",
            guidelines_index="guidelines",
        )
        self.assertEqual(registry.get("linearrag").retriever_name, "rag")

    def test_article_and_guideline_hits_are_counted_separately(self):
        result, _ = self._retrieve(
            [
                _hit("c1", chunk_source_type="abstract"),
                _hit("c2", chunk_source_type="guide"),
                _hit("c3", chunk_source_type="textbook"),
            ]
        )

        self.assertTrue(result.status["ok"])
        self.assertEqual(result.status["article_hits"], 2)
        self.assertEqual(result.status["guideline_hits"], 1)

    def test_passage_text_reaches_the_answer_prompt_fields(self):
        """The prompt reads `abstract`; a guideline is quoted from `rule_text`."""
        result, _ = self._retrieve(
            [_hit("c1", chunk_source_type="guide", text="Limit salt to 5g a day.")]
        )
        payload = result.source_payloads[0]

        self.assertEqual(payload["abstract"], "Limit salt to 5g a day.")
        self.assertEqual(payload["description"], "Limit salt to 5g a day.")
        self.assertEqual(payload["rule_text"], "Limit salt to 5g a day.")
        self.assertEqual(payload["source_type"], "guideline")

    def test_urn_prefers_provenance_then_document_then_chunk(self):
        result, _ = self._retrieve(
            [
                _hit("c1", urn="urn:wf:article:1"),
                _hit("c2", source_doc_id="doc-42"),
                _hit("c3", source_doc_id=None),
            ]
        )
        urns = [s.urn for s in result.retrieved_sources]

        self.assertEqual(urns[0], "urn:wf:article:1")
        self.assertEqual(urns[1], "doc-42")
        # Never empty: a source with no urn cannot be cited or de-duplicated,
        # so the chunk id is the last resort rather than a blank.
        self.assertEqual(urns[2], "c3")

    def test_the_score_is_exposed_where_ranking_reads_it(self):
        """The orchestrator normalizes on `_score`; an absent one ranks as 0."""
        result, _ = self._retrieve([_hit("c1", score=0.77)])
        payload = result.source_payloads[0]

        self.assertEqual(payload["_score"], 0.77)
        self.assertEqual(payload["relevance_score"], 0.77)
        self.assertEqual(result.retrieved_sources[0].similarity_score, 0.77)

    def test_branch_scores_survive_into_the_payload(self):
        """Why a passage ranked where it did — unreviewable without them."""
        result, _ = self._retrieve([_hit("c1")])
        payload = result.source_payloads[0]

        self.assertEqual(payload["text_sim"], 0.8)
        self.assertEqual(payload["triplet_sim"], 0.6)
        self.assertEqual(payload["ppr_score"], 0.4)

    def test_the_graph_trace_is_reported_in_status(self):
        result, _ = self._retrieve([_hit("c1")])

        self.assertEqual(result.status["graph"]["subgraph_nodes"], 25)
        self.assertEqual(result.status["graph"]["branches"], ["text", "triplet", "ppr"])

    def test_a_degraded_ranking_is_reported_rather_than_hidden(self):
        """A hash-embedder fallback ranks confidently and means nothing."""
        result, _ = self._retrieve(
            [_hit("c1")],
            trace=_trace(embedder="mock-embedder-v0", degraded=["mock embedder"]),
        )

        self.assertEqual(result.status["graph"]["degraded"], ["mock embedder"])

    def test_the_query_is_contextualized_with_the_user_context(self):
        from models.qa import QAUserContext

        with patch(
            "services.kggen_service.retrieve", return_value=([], _trace())
        ) as mock:
            result = self.adapter.retrieve(
                question="Does salt raise blood pressure?",
                plan=_plan(),
                top_k=5,
                user_context=QAUserContext(country="Greece"),
            )

        sent = mock.call_args.args[0]
        self.assertIn("Greece", sent)
        self.assertIn("Greece", result.status["used_query"])

    def test_a_missing_graph_is_reported_not_raised(self):
        """KG_ENABLED=false raises a 503 inside the service.

        The pipeline must keep going and answer from what it has, exactly as
        it did when the retriever it replaced could not find its index file.
        """
        with patch(
            "services.kggen_service.retrieve",
            side_effect=RuntimeError("The knowledge graph is not enabled"),
        ):
            result = self.adapter.retrieve(
                question="Does salt raise blood pressure?",
                plan=_plan(),
                top_k=5,
                user_context=None,
            )

        self.assertFalse(result.status["ok"])
        self.assertIn("not enabled", result.status["error"])
        self.assertEqual(result.source_payloads, [])
        self.assertEqual(result.retrieved_sources, [])
        self.assertEqual(result.status["article_hits"], 0)


if __name__ == "__main__":
    unittest.main()
