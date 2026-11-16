"""A first-round "corpus gap" is checked once more before the answer says so.

Live (oat milk and calcium): one search, a few snippets, and the evaluator
called a corpus gap while the catalog held three surveys of exactly that. It
even wrote the better search into a note, which was never run, and the gap
note went into the thread's memory as fact.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.qa_agent import repair_citation_links
from models.qa import PlannedSubQuestion, QARequest, ResearchNote
from services.qa_pipeline import ablation, repair
from services.qa_pipeline.evaluator import EvaluationResult

import test_qa_pipeline_orchestrator as harness


def _state(round_index=0, tried=()):
    sq = PlannedSubQuestion(
        id="sq1", text="What is the calcium content of oat milk compared to dairy milk?",
        branch="articles", lexical_query="oat milk calcium dairy",
    )
    return SimpleNamespace(
        round=round_index,
        sub_questions=lambda: [sq],
        queries_tried=set(tried),
        plan=SimpleNamespace(sub_questions=[sq]),
    ), sq


GAP = EvaluationResult(
    verdict="corpus_gap",
    reason="No evidence on oat milk calcium.",
    per_sub_question=[{"id": "sq1", "covered": False}],
    notes=[
        ResearchNote(text="No evidence found on oat milk calcium content.", kind="gap"),
        ResearchNote(
            text='Potential future search: "oat milk calcium content" or "oat milk fortified calcium".',
            kind="lead",
        ),
    ],
    used_llm=True,
)


class RecheckPlanTests(unittest.TestCase):
    def test_first_round_gap_runs_the_evaluators_own_search(self):
        state, sq = _state()
        plan = repair.build_repair_plan(state, GAP, max_total_sub_questions=5)
        self.assertEqual([a["action"] for a in plan.actions], ["recheck_gap"])
        self.assertEqual(sq.lexical_query, "oat milk calcium content")

    def test_a_search_already_tried_is_skipped(self):
        state, sq = _state(tried={("articles", "oat milk calcium content")})
        repair.build_repair_plan(state, GAP, max_total_sub_questions=5)
        self.assertEqual(sq.lexical_query, "oat milk fortified calcium")

    def test_no_recheck_after_the_first_round(self):
        state, _ = _state(round_index=1)
        self.assertFalse(repair.build_repair_plan(state, GAP, max_total_sub_questions=5).has_work)

    def test_switch_off_keeps_the_old_behaviour(self):
        state, _ = _state()
        with ablation.active({"QA_GAP_RECHECK_ENABLED": False}):
            self.assertFalse(repair.build_repair_plan(state, GAP, max_total_sub_questions=5).has_work)

    def test_without_leads_the_question_in_plain_words(self):
        state, sq = _state()
        bare = EvaluationResult(verdict="corpus_gap", per_sub_question=[{"id": "sq1", "covered": False}])
        repair.build_repair_plan(state, bare, max_total_sub_questions=5)
        self.assertEqual(sq.lexical_query, sq.text)


class RecheckPipelineTests(unittest.IsolatedAsyncioTestCase):
    setUp = harness.OrchestratorTests.setUp
    _run = harness.OrchestratorTests._run

    async def test_recheck_that_finds_evidence_drops_the_gap_note(self):
        plan = harness._plan()
        harness._FakePlanner.plan_to_return = plan
        first_sq = plan.sub_questions[0].id
        gap = EvaluationResult(
            verdict="corpus_gap",
            per_sub_question=[{"id": first_sq, "covered": False}],
            notes=[
                ResearchNote(text="No evidence found on oat milk calcium content.", kind="gap"),
                ResearchNote(text='Try "oat milk fortified calcium".', kind="lead"),
            ],
            used_llm=True,
        )
        calls = []

        def fake_run_branch(sq, *, branch, **_kwargs):
            # The recheck finds sources the first search did not.
            calls.append((sq.id, branch, sq.lexical_query))
            n = sum(1 for c in calls if c[:2] == (sq.id, branch))
            return harness._outcome(sq.id, branch, [f"{sq.id}-{branch}-r{n}-1", f"{sq.id}-{branch}-r{n}-2"])

        evaluations = [gap]

        async def fake_evaluate(state, **_kwargs):
            return evaluations.pop(0) if evaluations else harness.SUFFICIENT

        events = []
        with patch.dict("config.config.settings", {"QA_MAX_REPAIR_ROUNDS": 1}), patch(
            "services.qa_pipeline.retrieval.run_branch", side_effect=fake_run_branch
        ), patch("services.qa_pipeline.orchestrator.evaluate", fake_evaluate):
            async for event in self.service.run_pipeline(
                QARequest(question="Is oat milk a good alternative to dairy milk for calcium?")
            ):
                events.append(event)
        self.assertIn((first_sq, plan.sub_questions[0].branch, "oat milk fortified calcium"), calls)
        done = events[-1].data
        notes = [n["text"] for n in (done.get("conversation_context") or {}).get("notes") or []]
        self.assertNotIn("No evidence found on oat milk calcium content.", notes)

    async def test_recheck_that_finds_nothing_new_keeps_the_gap_note(self):
        plan = harness._plan()
        harness._FakePlanner.plan_to_return = plan
        gap = EvaluationResult(
            verdict="corpus_gap",
            per_sub_question=[{"id": plan.sub_questions[0].id, "covered": False}],
            notes=[ResearchNote(text="Nothing on oat milk.", kind="gap"),
                   ResearchNote(text='Try "oat drink calcium".', kind="lead")],
            used_llm=True,
        )
        with patch.dict("config.config.settings", {"QA_MAX_REPAIR_ROUNDS": 1}):
            events, _ = await self._run(
                request=QARequest(question="Is oat milk a good alternative to dairy milk for calcium?"),
                evaluations=[gap],
            )
        notes = [n["text"] for n in (events[-1].data.get("conversation_context") or {}).get("notes") or []]
        self.assertIn("Nothing on oat milk.", notes)


class LenticularCitationTests(unittest.TestCase):
    SOURCES = [
        {"urn": "urn:article:assessing-plant-milks", "authors": ["A Johnson", "B Lee"], "publication_year": 2024},
    ]

    def test_label_of_a_retrieved_source_becomes_its_link(self):
        out = repair_citation_links("Not equivalent to dairy【Johnson et al. (2024)】.", self.SOURCES)
        self.assertIn("[Johnson et al. (2024)](/articles/urn:article:assessing-plant-milks)", out)

    def test_general_knowledge_and_unknown_labels_keep_closed_brackets(self):
        out = repair_citation_links("Check the label【general knowledge】. See【G455】.", self.SOURCES)
        self.assertIn("[general knowledge].", out)
        self.assertIn("[G455]", out)
        self.assertNotIn("【", out)

    def test_bracketed_id_as_a_link_target_is_not_linked_twice(self):
        out = repair_citation_links(
            "Drink milk [Johnson et al. (2024)]([urn:article:assessing-plant-milks]).", self.SOURCES
        )
        self.assertIn("[Johnson et al. (2024)](/articles/urn:article:assessing-plant-milks)", out)
        self.assertNotIn("([", out)


if __name__ == "__main__":
    unittest.main()
