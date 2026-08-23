"""Tests for the sufficiency evaluator's gates, coercion, and note-taking."""

import unittest
from unittest.mock import AsyncMock, patch

from models.qa import QARequest, QAUserContext, RetrievedSource
from services.qa_pipeline import evaluator
from services.qa_pipeline.state import EvidenceItem, PipelineState


def _state(round_index=0, evidence=None, branch_statuses=None, country=None):
    from agents.qa_planner import build_fallback_pipeline_plan

    request = QARequest(question="How much fiber per day?")
    plan = build_fallback_pipeline_plan(
        question=request.question,
        request=request,
        user_context=QAUserContext(),
    )
    state = PipelineState(
        request=request,
        request_id="r1",
        effective_question=request.question,
        user_context=QAUserContext(country=country),
        effective_model="test-model",
        effective_retriever="rag",
        plan=plan,
    )
    state.round = round_index
    state.evidence = evidence or []
    state.branch_statuses = branch_statuses or []
    return state


def _guideline_item(urn="g1", region="EU"):
    item = EvidenceItem(
        payload={
            "urn": urn,
            "source_type": "guideline",
            "guide_region": region,
            "rule_text": "Eat 25-30 g of fiber per day.",
            "title": "Fiber guide",
        },
        source=RetrievedSource(
            source_type="guideline", urn=urn, title="Fiber guide", similarity_score=1.0
        ),
    )
    item.sub_question_ids = ["sq2"]
    item.adjusted_score = 0.8
    return item


class DeterministicGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_branches_failed_is_corpus_gap_without_llm(self):
        state = _state(
            branch_statuses=[{"ok": False}, {"ok": False}],
        )
        with patch.object(evaluator.GROQ_CHAT, "get_client") as llm:
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        llm.assert_not_called()
        self.assertEqual(result.verdict, "corpus_gap")
        self.assertTrue(any(n.kind == "gap" for n in result.notes))

    async def test_exhausted_repair_budget_is_sufficient_without_llm(self):
        state = _state(round_index=1, evidence=[_guideline_item()],
                       branch_statuses=[{"ok": True}])
        with patch.object(evaluator.GROQ_CHAT, "get_client") as llm:
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        llm.assert_not_called()
        self.assertEqual(result.verdict, "sufficient")

    async def test_llm_failure_never_blocks_the_answer(self):
        state = _state(evidence=[_guideline_item()], branch_statuses=[{"ok": True}])
        broken = type(
            "_L", (), {"ainvoke": AsyncMock(side_effect=RuntimeError("down"))}
        )()
        with patch.object(evaluator.GROQ_CHAT, "get_client", return_value=broken):
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        self.assertEqual(result.verdict, "sufficient")


class LlmVerdictTests(unittest.IsolatedAsyncioTestCase):
    def _llm(self, content):
        class _Response:
            pass

        response = _Response()
        response.content = content
        return type("_L", (), {"ainvoke": AsyncMock(return_value=response)})()

    async def test_notes_and_reformulations_are_coerced(self):
        content = """
        {"verdict": "vocabulary_mismatch", "reason": "Wording missed.",
         "per_sub_question": [{"id": "sq1", "covered": false, "gap": "nothing"}],
         "reformulated_queries": [{"id": "sq1", "lexical_query": "dietary fibre intake"}],
         "new_sub_questions": [],
         "clarification": null,
         "notes": [
           {"text": "EU guideline covers adult fiber intake.", "kind": "finding",
            "sub_question_id": "sq2", "source_urns": ["g1"]},
           {"text": "Try the British spelling fibre.", "kind": "lead"},
           {"text": "", "kind": "finding"},
           {"text": "Bad kind falls back.", "kind": "hunch"}
         ]}
        """
        state = _state(evidence=[_guideline_item()], branch_statuses=[{"ok": True}])
        with patch.object(
            evaluator.GROQ_CHAT, "get_client", return_value=self._llm(content)
        ):
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        self.assertEqual(result.verdict, "vocabulary_mismatch")
        self.assertEqual(len(result.notes), 3)
        self.assertEqual(result.notes[0].source_urns, ["g1"])
        self.assertEqual(result.notes[2].kind, "finding")
        self.assertEqual(
            result.reformulated_queries[0]["lexical_query"], "dietary fibre intake"
        )

    async def test_clarification_verdict_downgraded_when_not_allowed(self):
        content = '{"verdict": "needs_user_clarification", "reason": "Region?"}'
        state = _state(evidence=[_guideline_item()], branch_statuses=[{"ok": True}])
        with patch.object(
            evaluator.GROQ_CHAT, "get_client", return_value=self._llm(content)
        ):
            result = await evaluator.evaluate(
                state, clarification_allowed=False, max_repair_rounds=1
            )
        self.assertEqual(result.verdict, "sufficient")
        self.assertIsNone(result.clarification)

    async def test_clarification_verdict_builds_region_options_from_evidence(self):
        content = '{"verdict": "needs_user_clarification", "reason": "Region matters."}'
        state = _state(
            evidence=[_guideline_item("g1", "EU"), _guideline_item("g2", "US")],
            branch_statuses=[{"ok": True}],
        )
        with patch.object(
            evaluator.GROQ_CHAT, "get_client", return_value=self._llm(content)
        ):
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        self.assertEqual(result.verdict, "needs_user_clarification")
        values = [o.value for o in result.clarification.options]
        self.assertEqual(values[:2], ["EU", "US"])
        self.assertIn("general", values)

    async def test_unknown_verdict_defaults_to_sufficient(self):
        content = '{"verdict": "shrug", "reason": "?"}'
        state = _state(evidence=[_guideline_item()], branch_statuses=[{"ok": True}])
        with patch.object(
            evaluator.GROQ_CHAT, "get_client", return_value=self._llm(content)
        ):
            result = await evaluator.evaluate(
                state, clarification_allowed=True, max_repair_rounds=1
            )
        self.assertEqual(result.verdict, "sufficient")


if __name__ == "__main__":
    unittest.main()
