"""End-to-end tests of the agentic pipeline loop with faked stages."""

import unittest
from unittest.mock import AsyncMock, patch

from models.qa import (
    ClarificationOption,
    ClarificationRequest,
    PlannedSubQuestion,
    QAAnswer,
    QAPipelinePlan,
    QARequest,
    QAResponse,
    ResearchNote,
)
from services.qa_pipeline.evaluator import EvaluationResult
from services.qa_pipeline.retrieval import BranchOutcome
from services.qa_pipeline.state import EvidenceItem
from models.qa import RetrievedSource


def _plan():
    return QAPipelinePlan(
        original_question="Do whole grains lower cholesterol?",
        canonical_question="Do whole grains lower cholesterol?",
        article_query="whole grains cholesterol",
        guideline_query="whole grains dietary guideline",
        sub_questions=[
            PlannedSubQuestion(
                id="sq1",
                text="Effect of whole grains on cholesterol",
                why="Looking for trial evidence.",
                qtype="mechanism",
                branch="articles",
                lexical_query="whole grains cholesterol",
                dense_query="Do whole grains lower cholesterol?",
            ),
            PlannedSubQuestion(
                id="sq2",
                text="Recommended whole grain intake",
                why="Checking dietary guidelines.",
                qtype="recommendation",
                branch="guidelines",
                lexical_query="whole grain intake guideline",
                dense_query="How much whole grain should adults eat?",
            ),
        ],
    )


class _FakePlanner:
    plan_to_return = None

    def __init__(self, *args, **kwargs):
        pass

    async def plan(self, **_kwargs):
        return type(self).plan_to_return or _plan()


class _FakeVector:
    def tolist(self):
        return [0.1, 0.2]


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return [_FakeVector() for _ in texts]


def _outcome(sq_id, branch, urns):
    items = []
    for rank, urn in enumerate(urns, start=1):
        source_type = "guideline" if branch == "guidelines" else "article"
        payload = {
            "urn": urn,
            "source_type": source_type,
            "title": f"Title {urn}",
            "relevance_score": 1.0,
        }
        if source_type == "guideline":
            payload["rule_text"] = "Choose whole grains more often for health."
            payload["guide_urn"] = f"guide-{urn}"
        else:
            payload["abstract"] = "Whole grains lower LDL."
        item = EvidenceItem(
            payload=payload,
            source=RetrievedSource(
                source_type=source_type, urn=urn, title=urn, similarity_score=1.0
            ),
        )
        item.rrf_norm = 1.0 / rank
        item.sub_question_ids = [sq_id]
        items.append(item)
    return BranchOutcome(
        sub_question_id=sq_id,
        branch=branch,
        items=items,
        status={"ok": True, "hit_count": len(items), "legs": {"lexical": len(items)}},
    )


async def _fake_stream_answer(**kwargs):
    yield {"kind": "delta", "text": "Whole grains "}
    yield {"kind": "delta", "text": "help."}
    yield {
        "kind": "final",
        "answer": QAAnswer(
            answer="Whole grains help.",
            citations=[],
            confidence="high",
            model_used=kwargs.get("model", "test"),
            rag_used=bool(kwargs.get("payloads")),
            sources_consulted=len(kwargs.get("payloads") or []),
        ),
        "follow_ups": ["What about oats?"],
        "parsed_trailer": True,
    }


SUFFICIENT = EvaluationResult(verdict="sufficient", reason="Good coverage.")


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from models.qa import QAUserContext
        from services.qa_service import QAService

        self.service = QAService(cache_enabled=False)
        self.service._embedder = _FakeEmbedder()

        patchers = [
            patch("agents.qa_planner.QAPlannerAgent", _FakePlanner),
            patch.object(
                type(self.service),
                "_resolve_user_context",
                lambda _self, _request: QAUserContext(),
            ),
            patch.object(
                self.service, "_persist_request", new=AsyncMock()
            ),
            patch.object(
                type(self.service),
                "_update_conversation_summary",
                lambda _self, **_kwargs: None,
            ),
            patch(
                "services.qa_pipeline.orchestrator.stream_answer",
                _fake_stream_answer,
            ),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        _FakePlanner.plan_to_return = None

    async def _run(self, request=None, evaluations=None, branch_urns=None):
        request = request or QARequest(question="Do whole grains lower cholesterol?")
        evaluations = list(evaluations or [SUFFICIENT])
        calls = []

        def fake_run_branch(sq, *, branch, **_kwargs):
            calls.append((sq.id, branch, sq.lexical_query))
            urns = (branch_urns or {}).get(
                (sq.id, branch), [f"{sq.id}-{branch}-1", f"{sq.id}-{branch}-2"]
            )
            return _outcome(sq.id, branch, urns)

        async def fake_evaluate(state, **_kwargs):
            return evaluations.pop(0) if evaluations else SUFFICIENT

        events = []
        with patch(
            "services.qa_pipeline.retrieval.run_branch", side_effect=fake_run_branch
        ), patch("services.qa_pipeline.orchestrator.evaluate", fake_evaluate):
            async for event in self.service.run_pipeline(request):
                events.append(event)
        return events, calls

    async def test_sufficient_round_produces_full_event_sequence(self):
        events, calls = await self._run()
        names = [e.name for e in events]

        self.assertEqual(names[0], "stage.start")
        self.assertIn("stage.plan", names)
        self.assertEqual(names.count("stage.search_started"), 2)
        self.assertEqual(names.count("stage.search_results"), 2)
        self.assertIn("stage.rerank", names)
        self.assertIn("stage.evaluate", names)
        self.assertNotIn("stage.repair", names)
        self.assertIn("answer_started", names)
        self.assertGreaterEqual(names.count("answer_delta"), 2)
        self.assertIn("citations", names)
        self.assertEqual(names[-1], "done")

        # seq is monotonic and every event carries the request_id.
        seqs = [e.data["seq"] for e in events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len({e.data["request_id"] for e in events}), 1)

        done = events[-1].data
        response = QAResponse(**{k: v for k, v in done.items() if k != "seq"})
        self.assertEqual(response.primary_answer.answer, "Whole grains help.")
        self.assertIsNone(response.secondary_answer)
        self.assertTrue(response.retrieved_sources)
        self.assertTrue(response.qa_thread_id)

        # Collapsible-step transparency: paired running→done step events and
        # the full timeline preserved on the final response.
        step_events = [e.data for e in events if e.name == "step"]
        self.assertTrue(step_events)
        by_id = {}
        for step in step_events:
            by_id.setdefault(step["id"], []).append(step["status"])
        plan_steps = [s for s in step_events if s["kind"] == "plan"]
        self.assertEqual([s["status"] for s in plan_steps], ["running", "done"])
        search_steps = [s for s in step_events if s["kind"] == "search"]
        self.assertEqual(len(search_steps), 4)  # 2 searches × running+done
        answer_steps = [s for s in step_events if s["kind"] == "answer"]
        self.assertEqual([s["status"] for s in answer_steps], ["running", "done"])

        self.assertTrue(response.reasoning_steps)
        self.assertTrue(
            all(step.status == "done" for step in response.reasoning_steps)
        )
        kinds = [step.kind for step in response.reasoning_steps]
        for expected in ("plan", "search", "rank", "evaluate", "answer"):
            self.assertIn(expected, kinds)

    async def test_vocabulary_mismatch_repairs_only_flagged_sub_question(self):
        mismatch = EvaluationResult(
            verdict="vocabulary_mismatch",
            reason="sq1 wording missed.",
            per_sub_question=[
                {"id": "sq1", "covered": False, "gap": "no trial evidence"},
                {"id": "sq2", "covered": True, "gap": None},
            ],
            reformulated_queries=[
                {
                    "id": "sq1",
                    "lexical_query": "wholegrain LDL randomized",
                    "dense_query": "Do wholegrain diets reduce LDL cholesterol?",
                }
            ],
            used_llm=True,
        )
        events, calls = await self._run(evaluations=[mismatch, SUFFICIENT])
        names = [e.name for e in events]

        self.assertEqual(names.count("stage.repair"), 1)
        self.assertEqual(names.count("stage.evaluate"), 2)
        # Round 1: sq1+sq2. Round 2: only sq1, with the reformulated query.
        round_two = calls[2:]
        self.assertEqual(len(round_two), 1)
        self.assertEqual(round_two[0][0], "sq1")
        self.assertEqual(round_two[0][2], "wholegrain LDL randomized")
        self.assertEqual(names[-1], "done")

    async def test_corpus_gap_answers_without_repair(self):
        gap = EvaluationResult(
            verdict="corpus_gap",
            reason="Not covered by the corpus.",
            notes=[ResearchNote(text="Corpus lacks infant fiber data.", kind="gap")],
            used_llm=True,
        )
        events, calls = await self._run(evaluations=[gap])
        names = [e.name for e in events]

        self.assertNotIn("stage.repair", names)
        self.assertIn("stage.notes", names)
        self.assertEqual(len(calls), 2)  # single round only
        self.assertEqual(names[-1], "done")

    async def test_clarification_verdict_stores_thread_and_ends_stream(self):
        clarification = EvaluationResult(
            verdict="needs_user_clarification",
            reason="Regional guidance differs.",
            clarification=ClarificationRequest(
                id="country_or_region",
                question="Which region?",
                options=[ClarificationOption(label="EU", value="EU")],
            ),
            used_llm=True,
        )
        events, _calls = await self._run(evaluations=[clarification])
        names = [e.name for e in events]

        self.assertEqual(names[-1], "clarification")
        self.assertNotIn("done", names)
        payload = events[-1].data
        self.assertTrue(payload["needs_clarification"])
        thread_id = payload["qa_thread_id"]
        self.assertIn(thread_id, self.service._qa_threads)
        # Steps so far ride the clarification response too.
        self.assertTrue(payload["reasoning_steps"])
        self.assertEqual(payload["reasoning_steps"][-1]["kind"], "clarification")

    async def test_notes_are_emitted_and_carried_in_pipeline_meta(self):
        noted = EvaluationResult(
            verdict="sufficient",
            reason="Covered.",
            notes=[
                ResearchNote(
                    text="Two RCTs support whole grains lowering LDL.",
                    kind="finding",
                    source_urns=["sq1-articles-1"],
                ),
                ResearchNote(text="Check oat beta-glucan next.", kind="lead"),
            ],
            used_llm=True,
        )
        persist = AsyncMock()
        with patch.object(self.service, "_persist_request", new=persist):
            events, _ = await self._run(evaluations=[noted])
        names = [e.name for e in events]
        self.assertIn("stage.notes", names)
        notes_event = next(e for e in events if e.name == "stage.notes")
        self.assertEqual(len(notes_event.data["notes"]), 2)

        meta = persist.call_args.kwargs["pipeline_meta"]
        self.assertEqual(meta["mode"], "agentic")
        self.assertEqual(len(meta["notes"]), 2)
        self.assertEqual(len(meta["sub_questions"]), 2)

    async def test_answer_question_wrapper_returns_final_response(self):
        request = QARequest(question="Do whole grains lower cholesterol?")

        def fake_run_branch(sq, *, branch, **_kwargs):
            return _outcome(sq.id, branch, [f"{sq.id}-{branch}-1"])

        async def fake_evaluate(state, **_kwargs):
            return SUFFICIENT

        with patch(
            "services.qa_pipeline.retrieval.run_branch", side_effect=fake_run_branch
        ), patch("services.qa_pipeline.orchestrator.evaluate", fake_evaluate):
            response = await self.service.answer_question(request)

        self.assertIsInstance(response, QAResponse)
        self.assertEqual(response.primary_answer.answer, "Whole grains help.")
        self.assertFalse(response.needs_clarification)


if __name__ == "__main__":
    unittest.main()
