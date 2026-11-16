"""Relevance grading before the cut (services/qa_pipeline/relevance.py).

Live (oat milk and calcium): plant-milk calcium surveys ranked 12th-25th
behind a human-milk paper and a calcium-absorption trial, and only five
articles reach the answer.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models.qa import PlannedSubQuestion, QARequest
from services.qa_pipeline import ablation, ranking, relevance
from services.qa_pipeline.state import EvidenceItem, PipelineState


def _item(urn, title, score, sq="sq1", category=None):
    payload = {"urn": urn, "source_type": "article", "title": title, "abstract": title}
    if category:
        payload["ai_category"] = category
    item = EvidenceItem(payload=payload, source=SimpleNamespace(), sub_question_ids=[sq])
    item.adjusted_score = score
    return item


def _state(qtype="quantity"):
    sq = PlannedSubQuestion(id="sq1", text="What is the calcium content of oat milk compared to dairy milk?",
                            branch="articles", qtype=qtype)
    plan = SimpleNamespace(sub_questions=[sq])
    state = PipelineState.__new__(PipelineState)
    state.__dict__.update(
        request=QARequest(question="Is oat milk a good alternative to dairy milk for calcium?"),
        request_id="r1", effective_question="Is oat milk a good alternative to dairy milk for calcium?",
        plan=plan, round=0, relevance={},
    )
    state.sub_questions = lambda: [sq]
    return state


class _FakeLLM:
    calls = []

    def __init__(self, grades):
        self.grades = grades

    async def ainvoke(self, messages, config=None):
        payload = json.loads(messages[1].content)
        type(self).calls.append(payload)
        titles = {s["id"]: s["title"] for s in payload["sources"]}
        return SimpleNamespace(content=json.dumps({"grades": {sid: self.grades.get(t, 1) for sid, t in titles.items()}}))


class GradingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeLLM.calls = []

    async def _grade(self, state, items, grades):
        client = SimpleNamespace(get_client=lambda **_kw: _FakeLLM(grades))
        with patch.object(relevance, "GROQ_CHAT", client):
            await relevance.grade_relevance(state, items)

    async def test_the_survey_that_answers_overtakes_the_off_topic_paper(self):
        off = _item("a1", "Nutritional value of human milk versus other mammals' milk", 0.60)
        trial = _item("a2", "Postprandial absorption of a milk calcium permeate", 0.43)
        survey = _item("a3", "Plant-based milks vs cow's milk: an audit of calcium content", 0.37)
        state = _state()
        await self._grade(state, [off, trial, survey], {survey.payload["title"]: 3, off.payload["title"]: 0})
        order = sorted([off, trial, survey], key=lambda i: i.adjusted_score, reverse=True)
        self.assertEqual(order[0], survey)
        self.assertEqual(order[-1], off)
        self.assertEqual(survey.score_parts["relevance"], 3)

    async def test_a_repair_round_grades_only_what_is_new(self):
        state = _state()
        first = _item("a1", "Oat milk calcium survey", 0.5)
        await self._grade(state, [first], {})
        first.adjusted_score = 0.5
        second = _item("a2", "Another survey", 0.4)
        await self._grade(state, [first, second], {})
        self.assertEqual([len(c["sources"]) for c in _FakeLLM.calls], [1, 1])

    async def test_failure_leaves_the_ranking_as_it_was(self):
        class Broken:
            async def ainvoke(self, *_a, **_k):
                raise RuntimeError("provider down")

        items = [_item("a1", "x", 0.5), _item("a2", "y", 0.3)]
        with patch.object(relevance, "GROQ_CHAT", SimpleNamespace(get_client=lambda **_kw: Broken())):
            await relevance.grade_relevance(_state(), items)
        self.assertEqual([i.adjusted_score for i in items], [0.5, 0.3])

    async def test_a_reply_missing_commas_is_still_read(self):
        class Sloppy:
            async def ainvoke(self, *_a, **_k):
                return SimpleNamespace(content='{"grades":{"s1":3"s2":0,"s3": "2"}}')

        items = [_item("a1", "x", 0.5), _item("a2", "y", 0.4), _item("a3", "z", 0.3)]
        with patch.object(relevance, "GROQ_CHAT", SimpleNamespace(get_client=lambda **_kw: Sloppy())):
            await relevance.grade_relevance(_state(), items)
        self.assertEqual([i.score_parts.get("relevance") for i in items], [3, 0, 2])

    async def test_switch_off(self):
        items = [_item("a1", "x", 0.5)]
        with ablation.active({"QA_RERANK_ENABLED": False}):
            await self._grade(_state(), items, {"x": 3})
        self.assertEqual(_FakeLLM.calls, [])
        self.assertEqual(items[0].adjusted_score, 0.5)


class QuantityDesignTests(unittest.TestCase):
    def test_a_trial_gets_no_design_boost_for_a_quantity_search(self):
        def fresh():
            item = _item("a1", "RCT of calcium absorption", 0.0, category="Randomized Controlled Trial")
            item.rrf_norm = 0.8
            return item

        effect = ranking.adjust_evidence([fresh()])[0].score_parts["study_design"]
        quantity = ranking.adjust_evidence([fresh()], design_neutral_sub_questions={"sq1"})[0].score_parts["study_design"]
        self.assertGreater(effect, 1.0)
        self.assertEqual(quantity, 1.0)


if __name__ == "__main__":
    unittest.main()
