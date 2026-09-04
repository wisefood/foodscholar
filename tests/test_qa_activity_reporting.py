"""What FoodScholar reports back to the gateway.

Two gaps this covers. The evidence retrieval behind every answer is a search,
and it was reported nowhere — so "the answer was thin" and "retrieval found
four documents and editorial policy dropped three of them" looked identical
from outside the service. And `qa.persist_failed` carried a cause but no id,
which counts failures without letting anyone find one.

The standing rule for both: ids, counts, durations and short enum-ish strings.
Never the asker's country, allergies or dietary profile — `user_context` holds
all three, and it goes nowhere near a reported row.
"""
import asyncio
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, "src")

import wf_telemetry  # noqa: E402
from models.qa import QAClarifierSafetyPlan  # noqa: E402
from services.qa_retrievers import RetrievalResult  # noqa: E402
from services.qa_service import QAService  # noqa: E402


class _FakeAdapter:
    """A retriever that returns exactly what the test tells it to."""

    def __init__(self, payloads, status):
        self._payloads = payloads
        self._status = status
        self.seen = None

    def retrieve(self, **kwargs):
        self.seen = kwargs
        return RetrievalResult(
            source_payloads=list(self._payloads),
            retrieved_sources=[None] * len(self._payloads),
            status=dict(self._status),
        )


def _article(urn, **extra):
    payload = {"urn": urn, "title": "t", "abstract": "a", "type": "article"}
    payload.update(extra)
    return payload


class QASearchReportingTests(unittest.TestCase):
    def setUp(self):
        self.reported = []
        patcher = mock.patch.object(
            wf_telemetry.TELEMETRY,
            "search",
            side_effect=lambda **kw: self.reported.append(kw),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _retrieve(self, adapter, *, expertise_level=None, user_context=None):
        service = QAService.__new__(QAService)
        service._retriever_adapters = {"rag": adapter}
        return QAService._retrieve_sources(
            service,
            question="how much fibre per day",
            plan=QAClarifierSafetyPlan(
                original_question="how much fibre per day",
                canonical_question="how much fibre per day",
                article_query="how much fibre per day",
                guideline_query="fibre dietary guideline",
            ),
            top_k=3,
            retriever="rag",
            user_context=user_context,
            expertise_level=expertise_level,
        )

    def test_a_retrieval_is_reported_as_a_qa_search(self):
        adapter = _FakeAdapter(
            [_article("urn:article:1"), _article("urn:article:2")],
            {"retriever": "rag", "ok": True, "article_hits": 2, "guideline_hits": 0},
        )
        self._retrieve(adapter)

        (row,) = self.reported
        self.assertEqual(row["surface"], "qa")
        self.assertEqual(row["raw_query"], "how much fibre per day")
        self.assertEqual(row["app"], "foodscholar")
        self.assertEqual(row["result_count_first_pass"], 2)
        self.assertEqual(row["result_count_final"], 2)
        self.assertEqual(row["filters"]["retriever"], "rag")
        self.assertEqual(row["filters"]["top_k"], 3)
        self.assertIsNotNone(row["latency_ms"])

    def test_what_editorial_policy_dropped_is_visible(self):
        """The whole point of the two counts: a retrieval that found plenty and
        returned nothing is not the same problem as one that found nothing."""
        adapter = _FakeAdapter(
            [
                _article("urn:article:1", reader_visibility="expert_only"),
                _article("urn:article:2", reader_visibility="expert_only"),
            ],
            {"retriever": "rag", "ok": True, "article_hits": 2, "guideline_hits": 0},
        )
        result = self._retrieve(adapter, expertise_level="beginner")

        (row,) = self.reported
        self.assertEqual(row["result_count_first_pass"], 2)
        self.assertEqual(row["result_count_final"], len(result.source_payloads))
        self.assertLess(row["result_count_final"], row["result_count_first_pass"])

    def test_a_broken_retriever_says_so(self):
        adapter = _FakeAdapter(
            [],
            {"retriever": "rag", "ok": False, "error": "repr(Exception())"},
        )
        self._retrieve(adapter)

        (row,) = self.reported
        self.assertEqual(row["filters"]["error"], "retrieval_failed")
        self.assertEqual(row["result_count_final"], 0)

    def test_the_askers_health_profile_is_never_reported(self):
        from models.qa import QAUserContext

        context = QAUserContext(country="Greece", allergies=["peanut"])
        adapter = _FakeAdapter(
            [_article("urn:article:1")],
            {"retriever": "rag", "ok": True},
        )
        self._retrieve(adapter, user_context=context)

        (row,) = self.reported
        rendered = repr(row).lower()
        self.assertNotIn("peanut", rendered)
        self.assertNotIn("greece", rendered)

    def test_a_broken_reporter_does_not_cost_an_answer(self):
        adapter = _FakeAdapter([_article("urn:article:1")], {"ok": True})
        with mock.patch.object(
            wf_telemetry.TELEMETRY, "search", side_effect=RuntimeError("broken")
        ):
            result = self._retrieve(adapter)
        self.assertEqual(len(result.source_payloads), 1)


class QAPersistFailureReportingTests(unittest.TestCase):
    """A persist failure that cannot be traced to a question is a counter.

    It used to carry only the exception class name, so an operator could see
    that writes were failing and had no way to find one of them.
    """

    def _persist_with_a_broken_database(self):
        reported = []
        service = QAService.__new__(QAService)
        service._persist_failures = 2

        def explode():
            raise TimeoutError("connection pool exhausted")

        fake_postgres = types.SimpleNamespace(
            POSTGRES_ASYNC_SESSION_FACTORY=explode
        )
        fake_models_db = types.SimpleNamespace(QARequestRecord=object)
        request = types.SimpleNamespace(
            question="q", mode="dual", top_k=3, expertise_level=None,
            language="en", user_id="user-1", member_id="member-1",
        )
        response = types.SimpleNamespace(request_id="qa-request-42")

        with mock.patch.dict(
            sys.modules,
            {"backend.postgres": fake_postgres, "models.db": fake_models_db},
        ), mock.patch.object(
            wf_telemetry.TELEMETRY,
            "event",
            side_effect=lambda name, **kw: reported.append((name, kw)),
        ):
            asyncio.run(
                QAService._persist_request(service, request, response, "a-model", True)
            )
        return reported

    def test_the_failure_carries_an_id_and_a_reason_category(self):
        (name, row), = self._persist_with_a_broken_database()
        self.assertEqual(name, "qa.persist_failed")
        self.assertEqual(row["props"]["qa_request_id"], "qa-request-42")
        self.assertEqual(row["props"]["cause"], "TimeoutError")
        self.assertEqual(row["props"]["consecutive_failures"], 3)
        self.assertEqual(row["app"], "foodscholar")

    def test_no_free_text_and_no_stack_are_reported(self):
        """The message and the traceback stay in the log, where they are not
        keyed to a user."""
        (_, row), = self._persist_with_a_broken_database()
        rendered = repr(row)
        self.assertNotIn("connection pool exhausted", rendered)
        self.assertNotIn("Traceback", rendered)


class GroqUsageCallbackWiringTests(unittest.TestCase):
    """FoodScholar produced no cost rows at all until the pool attached this.

    Attached at the pool rather than at the call sites for the same reason the
    Langfuse handler is: an answer is several model calls across the clarifier,
    the retrieval scout and the two answer legs, and costing them one call site
    at a time guarantees the next one added is missed.
    """

    def _pooled_client(self):
        import backend.groq as groq_module

        with mock.patch.object(
            groq_module, "ChatGroq", lambda **kw: types.SimpleNamespace(**kw)
        ):
            return groq_module.GroqConnectionPool().get_client(
                model="openai/gpt-oss-120b", temperature=0.0
            )

    def _usage_handler(self):
        client = self._pooled_client()
        handlers = [
            handler
            for handler in (getattr(client, "callbacks", None) or [])
            if type(handler).__name__ == "_UsageReporter"
        ]
        self.assertEqual(len(handlers), 1, "the pool attaches exactly one reporter")
        return handlers[0]

    def test_every_pooled_client_costs_its_calls(self):
        self.assertIsNotNone(self._usage_handler())

    def test_the_row_names_the_provider_and_the_app(self):
        """`provider` is what the service knows about its own client — these
        are ChatGroq clients, and a row without it cannot be priced."""
        handler = self._usage_handler()
        rows = []
        with mock.patch.object(
            wf_telemetry.TELEMETRY, "llm_usage", side_effect=lambda **kw: rows.append(kw)
        ):
            handler.on_chat_model_start({}, [[]], run_id="r")
            handler.on_llm_end(
                types.SimpleNamespace(
                    llm_output={
                        "token_usage": {"prompt_tokens": 8, "completion_tokens": 2},
                        "model_name": "openai/gpt-oss-120b",
                    },
                    generations=[],
                ),
                run_id="r",
            )
        (row,) = rows
        self.assertEqual(row["provider"], "groq")
        self.assertEqual(row["app"], "foodscholar")
        self.assertEqual(row["feature"], "foodscholar_llm")
        self.assertEqual(row["input_tokens"], 8)
        self.assertEqual(row["output_tokens"], 2)
        self.assertIsNotNone(row["latency_ms"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
