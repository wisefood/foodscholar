"""The research log: how each retrieval round went (observe-only).

A repair round that only re-finds what the first round had, or that repeats
its searches, is "stalled"; one whose new evidence no longer talks about the
question is "drifting"; the end-of-budget "sufficient" is not mistaken for a
real judgement. None of it changes the run yet: it is recorded with the
answer (pipeline_meta) so the thresholds can be calibrated first.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from models.qa import QARequest, RetrievedSource
from services.qa_pipeline import progress
from services.qa_pipeline.evaluator import EvaluationResult
from services.qa_pipeline.retrieval import BranchOutcome
from services.qa_pipeline.state import EvidenceItem

import test_qa_pipeline_orchestrator as harness


def _item(urn, title="", abstract=""):
    return SimpleNamespace(key=f"article:{urn}", payload={"urn": urn, "title": title, "abstract": abstract})


class KeyTermTests(unittest.TestCase):
    def test_terms_skip_short_and_common_words_and_fold_plurals(self):
        self.assertEqual(progress.key_terms("Do whole grains lower cholesterol?"), {"whole", "grain", "lower", "cholesterol"})
        self.assertEqual(progress.key_terms("What is the best one?"), set())

    def test_query_key_ignores_case_and_spacing(self):
        self.assertEqual(progress.query_key("articles", "Whole  Grains LDL"), progress.query_key("articles", "whole grains ldl"))
        self.assertNotEqual(progress.query_key("articles", "x"), progress.query_key("guidelines", "x"))

    def test_overlap(self):
        q = progress.key_terms("Do whole grains lower cholesterol?")
        self.assertEqual(progress.overlap_with_question(q, [{"title": "Whole grains lower LDL cholesterol"}]), 1.0)
        self.assertEqual(progress.overlap_with_question(q, [{"title": "Sleep and screen time in teenagers"}]), 0.0)
        self.assertIsNone(progress.overlap_with_question(q, []))
        self.assertIsNone(progress.overlap_with_question(set(), [{"title": "x"}]))


class SignalsAndStatusTests(unittest.TestCase):
    Q = progress.key_terms("Do whole grains lower cholesterol?")

    def _signals(self, round_index=1, searched=(("sq1", "articles", "oat ldl"),), tried=frozenset(),
                 statuses=({"ok": True},), before=(), selected=(), per_sq=()):
        return progress.round_signals(
            round_index=round_index,
            searched=list(searched),
            queries_tried_before=set(tried),
            branch_statuses=list(statuses),
            pool_keys_before={f"article:{u}" for u in before},
            pool_keys={f"article:{u}" for u in before} | {i.key for i in selected},
            selected_keys_before={f"article:{u}" for u in before},
            selected=list(selected),
            question_terms=self.Q,
            per_sub_question=list(per_sq),
        )

    def test_counts_new_repeated_and_uncovered(self):
        s = self._signals(
            searched=[("sq1", "articles", "Oat LDL"), ("sq2", "guidelines", "fibre")],
            tried={progress.query_key("articles", "oat ldl")},
            before=["a"],
            selected=[_item("a", "Whole grains lower cholesterol"), _item("b", "Whole grain and cholesterol trial")],
            per_sq=[{"id": "sq1", "covered": True, "gap": None}, {"id": "sq2", "covered": False, "gap": "no guideline"}],
        )
        self.assertEqual(s["repeated_queries"], ["oat ldl"])
        self.assertEqual(s["new_in_selected"], 1)
        self.assertEqual(s["uncovered_sub_questions"], ["sq2"])
        self.assertGreater(s["question_overlap_new"], 0.5)

    def test_blocked_when_every_search_failed(self):
        s = self._signals(statuses=[{"ok": False}])
        self.assertEqual(progress.assess_round(s, verdict="corpus_gap", used_llm=False, baseline_overlap=None)[0], "blocked")

    def test_stalled_when_a_repair_round_adds_nothing_new(self):
        s = self._signals(before=["a"], selected=[_item("a", "Whole grains")])
        status, reason = progress.assess_round(s, verdict="sufficient", used_llm=False, baseline_overlap=0.8)
        self.assertEqual(status, "stalled")
        self.assertIn("nothing new", reason)

    def test_stalled_when_it_only_repeats_searches(self):
        s = self._signals(tried={progress.query_key("articles", "oat ldl")}, selected=[_item("b", "Whole grains lower cholesterol")])
        self.assertEqual(progress.assess_round(s, verdict="vocabulary_mismatch", used_llm=True, baseline_overlap=0.8)[0], "stalled")

    def test_drifting_when_new_evidence_leaves_the_question(self):
        s = self._signals(before=["a"], selected=[_item("a", "Whole grains"), _item("z", "Sleep and screen time in teenagers")])
        self.assertEqual(progress.assess_round(s, verdict="sufficient", used_llm=True, baseline_overlap=0.75)[0], "drifting")

    def test_drifting_from_the_first_round(self):
        s = self._signals(round_index=0, selected=[_item("z", "Sleep and screen time in teenagers")])
        self.assertEqual(progress.assess_round(s, verdict="sufficient", used_llm=True, baseline_overlap=None)[0], "drifting")

    def test_sufficient_only_when_actually_judged(self):
        s = self._signals(round_index=0, selected=[_item("b", "Whole grains lower cholesterol")])
        self.assertEqual(progress.assess_round(s, verdict="sufficient", used_llm=True, baseline_overlap=None)[0], "sufficient")
        status, reason = progress.assess_round(s, verdict="sufficient", used_llm=False, baseline_overlap=None)
        self.assertEqual(status, "on_track")
        self.assertIn("budget", reason)

    def test_log_entry(self):
        s = self._signals(selected=[_item("b", "Whole grains lower cholesterol")])
        entry = progress.log_entry(s, "on_track", "ok", searched=[("sq1", "articles", "oat ldl")], verdict="sufficient", gaps=["no RCTs", None])
        self.assertEqual(entry["tried"], ["articles: oat ldl"])
        self.assertEqual(entry["missing"], ["no RCTs"])
        self.assertEqual(entry["kept_new"], 1)
        self.assertIsNone(entry["next"])


MISMATCH = EvaluationResult(
    verdict="vocabulary_mismatch",
    reason="sq1 wording missed.",
    per_sub_question=[{"id": "sq1", "covered": False, "gap": "no trial evidence"}, {"id": "sq2", "covered": True, "gap": None}],
    reformulated_queries=[{"id": "sq1", "lexical_query": "wholegrain LDL randomized", "dense_query": "Do wholegrain diets reduce LDL?"}],
    used_llm=True,
)
BUDGET_SUFFICIENT = EvaluationResult(verdict="sufficient", reason="Answering with the best available evidence.", used_llm=False)


def _outcome(sq_id, branch, docs):
    items = []
    for rank, (urn, title) in enumerate(docs, start=1):
        item = EvidenceItem(
            payload={"urn": urn, "source_type": "article", "title": title, "abstract": title, "relevance_score": 1.0},
            source=RetrievedSource(source_type="article", urn=urn, title=title, similarity_score=1.0),
        )
        item.rrf_norm = 1.0 / rank
        item.sub_question_ids = [sq_id]
        items.append(item)
    return BranchOutcome(sub_question_id=sq_id, branch=branch, items=items, status={"ok": True, "hit_count": len(items), "legs": {"lexical": len(items)}})


class PipelineResearchLogTests(unittest.IsolatedAsyncioTestCase):
    setUp = harness.OrchestratorTests.setUp

    async def _run(self, evaluations, docs_by_round):
        """docs_by_round[r] -> list of (urn, title) every branch returns in round r."""
        rounds_seen = {}
        persist = AsyncMock()

        def fake_run_branch(sq, *, branch, **_kwargs):
            r = rounds_seen.setdefault((sq.id, branch), -1) + 1
            rounds_seen[(sq.id, branch)] = r
            docs = docs_by_round[min(r, len(docs_by_round) - 1)]
            return _outcome(sq.id, branch, [(f"{sq.id}-{branch}-{u}", t) for u, t in docs])

        evaluations = list(evaluations)

        async def fake_evaluate(state, **_kwargs):
            return evaluations.pop(0) if evaluations else BUDGET_SUFFICIENT

        with patch("services.qa_pipeline.retrieval.run_branch", side_effect=fake_run_branch), \
                patch("services.qa_pipeline.orchestrator.evaluate", fake_evaluate), \
                patch.object(self.service, "_persist_request", new=persist):
            events = [e async for e in self.service.run_pipeline(QARequest(question="Do whole grains lower cholesterol?"))]
        return events, persist.call_args.kwargs["pipeline_meta"]

    async def test_a_repair_round_that_refinds_the_same_sources_is_stalled(self):
        same = [("1", "Whole grains lower cholesterol"), ("2", "Whole grain intake and LDL")]
        events, meta = await self._run([MISMATCH, BUDGET_SUFFICIENT], [same, same])
        self.assertEqual(events[-1].name, "done")
        statuses = [p["status"] for p in meta["progress"]]
        self.assertEqual(statuses, ["on_track", "stalled"])
        log = meta["research_log"]
        self.assertEqual(log[0]["missing"], ["no trial evidence"])
        self.assertTrue(log[0]["next"])  # the repair that followed
        self.assertEqual(log[1]["kept_new"], 0)
        self.assertIn("articles: wholegrain LDL randomized", " | ".join(log[1]["tried"]))

    async def test_a_repair_round_with_new_on_topic_evidence_is_on_track_at_budget(self):
        first = [("1", "Whole grains lower cholesterol")]
        second = [("1", "Whole grains lower cholesterol"), ("9", "Oat whole grain trial lowers cholesterol")]
        _, meta = await self._run([MISMATCH, BUDGET_SUFFICIENT], [first, second])
        last = meta["progress"][-1]
        self.assertEqual(last["status"], "on_track")
        self.assertIn("budget", last["reason"])
        self.assertGreater(last["new_in_selected"], 0)

    async def test_a_repair_round_that_wanders_off_is_drifting(self):
        first = [("1", "Whole grains lower cholesterol")]
        second = [("1", "Whole grains lower cholesterol"), ("7", "Sleep duration and screen time in teenagers")]
        _, meta = await self._run([MISMATCH, BUDGET_SUFFICIENT], [first, second])
        self.assertEqual(meta["progress"][-1]["status"], "drifting")

    async def test_the_log_never_changes_the_run(self):
        """Same events with the research log broken: observe-only."""
        same = [("1", "Whole grains lower cholesterol")]
        events, _ = await self._run([MISMATCH, BUDGET_SUFFICIENT], [same, same])
        with patch.object(progress, "round_signals", side_effect=RuntimeError("boom")):
            broken, meta = await self._run([MISMATCH, BUDGET_SUFFICIENT], [same, same])
        self.assertEqual([e.name for e in events], [e.name for e in broken])
        self.assertEqual(meta["progress"], [])


if __name__ == "__main__":
    unittest.main()
