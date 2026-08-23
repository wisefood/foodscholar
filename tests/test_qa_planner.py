"""Tests for the research planner's fallback, coercion, and note seeding."""

import unittest
from unittest.mock import AsyncMock, patch

from agents.qa_planner import (
    QAPlannerAgent,
    _coerce_filters,
    _coerce_sub_questions,
    build_fallback_pipeline_plan,
)
from models.qa import QARequest, QAUserContext, ResearchNote


def _request(**kwargs):
    return QARequest(question="Do whole grains lower cholesterol?", **kwargs)


class FallbackPlanTests(unittest.TestCase):
    def test_fallback_covers_both_branches(self):
        plan = build_fallback_pipeline_plan(
            question="Do whole grains lower cholesterol?",
            request=_request(),
            user_context=QAUserContext(),
        )
        branches = [sq.branch for sq in plan.sub_questions]
        self.assertEqual(branches, ["articles", "guidelines"])
        for sq in plan.sub_questions:
            self.assertTrue(sq.why)
            self.assertTrue(sq.lexical_query)
            self.assertTrue(sq.dense_query)


class CoercionTests(unittest.TestCase):
    def test_malformed_entries_are_dropped(self):
        raw = [
            "not a dict",
            {"why": "missing text"},
            {"text": "Real one", "qtype": "invented", "branch": "everywhere"},
        ]
        result = _coerce_sub_questions(raw, question="q")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].qtype, "general")
        self.assertEqual(result[0].branch, "both")
        # Empty queries backfill from the sub-question text.
        self.assertEqual(result[0].lexical_query, "Real one")

    def test_duplicate_ids_are_renumbered(self):
        raw = [
            {"id": "sq1", "text": "First"},
            {"id": "sq1", "text": "Second"},
        ]
        result = _coerce_sub_questions(raw, question="q")
        self.assertEqual(len({sq.id for sq in result}), 2)

    def test_filters_coercion_is_defensive(self):
        filters = _coerce_filters(
            {
                "year_min": "2020",
                "year_max": "not a year",
                "open_access": "yes",
                "study_types": ["rct", "", None],
                "regions": "EU",
            }
        )
        self.assertEqual(filters.year_min, 2020)
        self.assertIsNone(filters.year_max)
        self.assertIsNone(filters.open_access)
        self.assertEqual(filters.study_types, ["rct"])
        self.assertEqual(filters.regions, [])

        self.assertTrue(_coerce_filters(None).is_empty())
        self.assertTrue(_coerce_filters("junk").is_empty())


class PlannerAgentTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, response_content):
        agent = QAPlannerAgent.__new__(QAPlannerAgent)
        agent.model = "test-model"
        agent.temperature = 0.0

        class _Response:
            content = response_content

        agent.llm = type(
            "_FakeLLM", (), {"ainvoke": AsyncMock(return_value=_Response())}
        )()
        return agent

    async def test_plan_parses_sub_questions_and_filters(self):
        agent = self._agent(
            """
            {"original_question": "q", "canonical_question": "q",
             "article_query": "a", "guideline_query": "g",
             "risk_level": "low", "safety_flags": [], "answer_guardrails": [],
             "needs_clarification": false, "clarification": null,
             "sub_questions": [
               {"id": "sq1", "text": "Recent RCTs on whole grains",
                "why": "Trials first.", "qtype": "mechanism",
                "branch": "articles", "lexical_query": "whole grains rct",
                "dense_query": "Do whole grains lower LDL?",
                "filters": {"year_min": 2020, "study_types": ["rct"]}}
             ]}
            """
        )
        plan = await agent.plan(
            question="q", request=_request(), user_context=QAUserContext()
        )
        self.assertEqual(len(plan.sub_questions), 1)
        sq = plan.sub_questions[0]
        self.assertEqual(sq.filters.year_min, 2020)
        self.assertEqual(sq.filters.study_types, ["rct"])

    async def test_prior_notes_reach_the_model_input(self):
        agent = self._agent('{"canonical_question": "q"}')
        await agent.plan(
            question="q",
            request=_request(),
            user_context=QAUserContext(),
            prior_notes=[
                ResearchNote(text="Corpus lacks infant fiber data.", kind="gap")
            ],
        )
        human_message = agent.llm.ainvoke.call_args.args[0][1]
        self.assertIn("Corpus lacks infant fiber data.", human_message.content)
        self.assertIn("research_notes", human_message.content)

    async def test_llm_failure_falls_back_deterministically(self):
        agent = QAPlannerAgent.__new__(QAPlannerAgent)
        agent.model = "test-model"
        agent.temperature = 0.0
        agent.llm = type(
            "_FakeLLM", (), {"ainvoke": AsyncMock(side_effect=RuntimeError("down"))}
        )()
        plan = await agent.plan(
            question="q", request=_request(), user_context=QAUserContext()
        )
        self.assertEqual(len(plan.sub_questions), 2)
        self.assertEqual(plan.reasoning_summary, "Deterministic fallback plan.")

    async def test_empty_model_plan_backfills_from_fallback(self):
        agent = self._agent('{"canonical_question": "", "sub_questions": []}')
        plan = await agent.plan(
            question="q", request=_request(), user_context=QAUserContext()
        )
        self.assertTrue(plan.canonical_question)
        self.assertEqual(len(plan.sub_questions), 2)


if __name__ == "__main__":
    unittest.main()
