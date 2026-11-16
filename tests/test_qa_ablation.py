"""Ablation switches: one request answered with a part of the pipeline off.

The overlay must follow the request it was asked for and nothing else, be
refused unless the deployment allows it, never share a cache entry with a
normal answer, and actually switch the part off.
"""

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import config
from exceptions import InvalidError
from models.qa import PlannedSubQuestion, QARequest
from services.qa_pipeline import ablation, ranking, retrieval
from services.qa_pipeline.state import EvidenceItem

import test_qa_pipeline_orchestrator as harness


def _load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_script("run_ablation_study")
runner = _load_script("run_qa_benchmark")


class _Enabled:
    """Turn ablation requests on for one test (base settings, not overlay)."""

    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        self.before = dict.get(config.settings, "QA_ABLATION_ENABLED")
        dict.__setitem__(config.settings, "QA_ABLATION_ENABLED", self.on)

    def __exit__(self, *exc):
        dict.__setitem__(config.settings, "QA_ABLATION_ENABLED", self.before)


class OverlayTests(unittest.IsolatedAsyncioTestCase):
    def test_overlay_is_seen_inside_and_gone_after(self):
        before = config.settings.get("QA_MAX_REPAIR_ROUNDS")
        with ablation.active({"QA_MAX_REPAIR_ROUNDS": 0}):
            self.assertEqual(config.settings.get("QA_MAX_REPAIR_ROUNDS"), 0)
            self.assertEqual(config.settings["QA_MAX_REPAIR_ROUNDS"], 0)
            self.assertEqual(ablation.current(), {"QA_MAX_REPAIR_ROUNDS": 0})
        self.assertEqual(config.settings.get("QA_MAX_REPAIR_ROUNDS"), before)
        self.assertEqual(ablation.current(), {})

    async def test_concurrent_requests_do_not_see_each_others_overlay(self):
        seen = {}

        async def request(name, overlay):
            with ablation.active(overlay):
                await asyncio.sleep(0.01)
                # A worker thread, as the pipeline's blocking calls use.
                seen[name] = await asyncio.to_thread(
                    config.settings.get, "QA_RETRIEVAL_LEGS"
                )

        await asyncio.gather(
            request("lexical", {"QA_RETRIEVAL_LEGS": "lexical"}),
            request("vector", {"QA_RETRIEVAL_LEGS": "vector"}),
            request("normal", {}),
        )
        self.assertEqual(seen, {"lexical": "lexical", "vector": "vector", "normal": "hybrid"})


class ValidationTests(unittest.TestCase):
    def test_refused_unless_the_deployment_allows_it(self):
        with _Enabled(False):
            with self.assertRaises(ablation.AblationError):
                ablation.validate({"QA_PLANNER_ENABLED": False})
            # No ablation is always fine.
            self.assertEqual(ablation.validate(None), {})
            self.assertEqual(ablation.validate({}), {})

    def test_service_turns_a_refusal_into_a_400(self):
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        request = QARequest(question="Is salt bad for me?", ablation={"QA_PLANNER_ENABLED": False})
        with _Enabled(False), self.assertRaises(InvalidError):
            service._ablation_overlay(request)

    def test_unknown_and_malformed_settings_are_refused(self):
        with _Enabled():
            for bad in (
                {"QA_DEFAULT_MODEL": "x"},          # not a switch
                {"QA_PLANNER_ENABLED": "maybe"},
                {"QA_MAX_REPAIR_ROUNDS": 1.5},
                {"QA_MAX_REPAIR_ROUNDS": 9},
                {"QA_MAX_REPAIR_ROUNDS": True},
                {"QA_RETRIEVAL_LEGS": "both"},
            ):
                with self.subTest(bad=bad), self.assertRaises(ablation.AblationError):
                    ablation.validate(bad)

    def test_values_are_typed(self):
        with _Enabled():
            self.assertEqual(
                ablation.validate({
                    "QA_PLANNER_ENABLED": "false",
                    "QA_MAX_REPAIR_ROUNDS": "0",
                    "QA_RECENCY_FLOOR": 1,
                    "QA_RETRIEVAL_LEGS": "Lexical",
                }),
                {
                    "QA_MAX_REPAIR_ROUNDS": 0,
                    "QA_PLANNER_ENABLED": False,
                    "QA_RECENCY_FLOOR": 1.0,
                    "QA_RETRIEVAL_LEGS": "lexical",
                },
            )

    def test_every_switch_has_a_default_setting(self):
        for name in ablation.SWITCHES:
            self.assertIn(name, config.settings, name)

    def test_runtime_info_lists_the_switches(self):
        catalog = ablation.catalog()
        self.assertIn("QA_PLANNER_ENABLED", catalog["switches"])
        self.assertIs(catalog["switches"]["QA_PLANNER_ENABLED"]["default"], True)


class CacheKeyTests(unittest.TestCase):
    def test_an_ablated_answer_has_its_own_cache_key(self):
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        request = QARequest(question="Do whole grains lower cholesterol?")
        normal = service._build_cache_key(request)
        with ablation.active({"QA_PLANNER_ENABLED": False}):
            ablated = service._build_cache_key(request)
        self.assertNotEqual(normal, ablated)
        self.assertEqual(normal, service._build_cache_key(request))


class RetrievalSwitchTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def leg(name):
            def fake(*_args, size, **_kwargs):
                self.calls.append((name, size))
                return []
            return fake

        patchers = [
            patch.object(retrieval, "search_articles_lexical", leg("articles-bm25")),
            patch.object(retrieval, "search_articles_knn", leg("articles-knn")),
            patch.object(retrieval, "search_guidelines_lexical", leg("guidelines-bm25")),
            patch.object(retrieval, "search_guidelines_knn", leg("guidelines-knn")),
            patch.object(retrieval, "guideline_hybrid_enabled", lambda: False),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.sq = PlannedSubQuestion(id="sq1", text="q", branch="articles", lexical_query="q")

    def _branches(self):
        for branch in ("articles", "guidelines"):
            retrieval.run_branch(self.sq, branch=branch, vector=[0.1], user_context=None, expertise_level=None)
        return self.calls

    def test_default_runs_both_article_legs_and_keyword_guidelines(self):
        self.assertEqual(
            self._branches(),
            [("articles-bm25", 30), ("articles-knn", 30), ("guidelines-bm25", 30)],
        )

    def test_lexical_only(self):
        with ablation.active({"QA_RETRIEVAL_LEGS": "lexical"}):
            self.assertEqual(self._branches(), [("articles-bm25", 30), ("guidelines-bm25", 30)])

    def test_vector_only_still_searches_guidelines(self):
        with ablation.active({"QA_RETRIEVAL_LEGS": "vector"}):
            self.assertEqual(self._branches(), [("articles-knn", 30), ("guidelines-knn", 30)])

    def test_knn_size_before_the_fix(self):
        with ablation.active({"QA_KNN_CANDIDATES": 10}):
            self.assertEqual(
                self._branches(),
                [("articles-bm25", 30), ("articles-knn", 10), ("guidelines-bm25", 30)],
            )


class RankingSwitchTests(unittest.TestCase):
    def test_without_priors_the_score_is_relevance_alone(self):
        payload = {
            "urn": "a1", "source_type": "article", "title": "t",
            "publication_year": 1990, "citation_count": 900, "ai_category": "meta-analysis",
        }
        item = EvidenceItem(payload=dict(payload), source=SimpleNamespace())
        item.rrf_norm = 0.8
        with_priors = ranking.adjust_evidence([item])[0].adjusted_score

        item = EvidenceItem(payload=dict(payload), source=SimpleNamespace())
        item.rrf_norm = 0.8
        with ablation.active({"QA_RANKING_PRIORS_ENABLED": False}):
            without = ranking.adjust_evidence([item])[0].adjusted_score
        self.assertAlmostEqual(without, 0.8)
        self.assertNotAlmostEqual(with_priors, 0.8)


class MoreSwitchTests(unittest.TestCase):
    def _item(self, payload, rrf=0.8):
        item = EvidenceItem(payload=dict(payload), source=SimpleNamespace())
        item.rrf_norm = rrf
        return item

    def test_one_ranking_signal_off_keeps_the_others(self):
        payload = {"urn": "a1", "source_type": "article", "title": "t",
                   "publication_year": 1990, "citation_count": 900}
        full = ranking.adjust_evidence([self._item(payload)])[0].score_parts
        with ablation.active({"QA_RANKING_PRIORS_OFF": "recency"}):
            parts = ranking.adjust_evidence([self._item(payload)])[0].score_parts
        self.assertEqual(parts["recency"], 1.0)
        self.assertNotEqual(full["recency"], 1.0)
        self.assertEqual(parts["influence"], full["influence"])

    def test_prior_list_is_checked_and_normalised(self):
        with _Enabled():
            self.assertEqual(
                ablation.validate({"QA_RANKING_PRIORS_OFF": "Influence, recency"}),
                {"QA_RANKING_PRIORS_OFF": "recency,influence"},
            )
            with self.assertRaises(ablation.AblationError):
                ablation.validate({"QA_RANKING_PRIORS_OFF": "recency,colour"})

    def test_evidence_budget(self):
        items = [self._item({"urn": f"a{i}", "source_type": "article"}, rrf=1 - i / 100) for i in range(10)]
        for item in items:
            item.adjusted_score = item.rrf_norm
        self.assertEqual(len(ranking.select_evidence(items, top_k=5)[0]), 5)
        with ablation.active({"QA_ARTICLE_TOP_K": 8}):
            self.assertEqual(len(ranking.select_evidence(items, top_k=5)[0]), 8)

    def test_models_are_limited_to_the_deployments(self):
        with _Enabled():
            model = config.settings["QA_PLANNER_MODEL"]
            self.assertEqual(ablation.validate({"QA_EVALUATOR_MODEL": model}), {"QA_EVALUATOR_MODEL": model})
            with self.assertRaises(ablation.AblationError):
                ablation.validate({"QA_DEFAULT_MODEL": "some/unknown-model"})

    def test_answer_model_follows_the_overlay(self):
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        request = QARequest(question="Is salt bad for me?")
        model = config.settings["QA_PLANNER_MODEL"]
        with ablation.active({"QA_DEFAULT_MODEL": model}):
            self.assertEqual(service._resolve_model(request), model)

    def test_search_filters_off(self):
        from models.qa import SubQuestionFilters

        seen = []

        def fake(_query, *, size, filters=None, **_kwargs):
            seen.append(filters)
            return []

        sq = PlannedSubQuestion(id="sq1", text="q", branch="articles", lexical_query="q",
                                filters=SubQuestionFilters(year_min=2015))
        with patch.object(retrieval, "search_articles_lexical", fake):
            retrieval.run_branch(sq, branch="articles", vector=None, user_context=None, expertise_level=None)
            with ablation.active({"QA_SEARCH_FILTERS_ENABLED": False}):
                retrieval.run_branch(sq, branch="articles", vector=None, user_context=None, expertise_level=None)
        self.assertEqual(seen[0].year_min, 2015)
        self.assertIsNone(seen[1])


class PipelineSwitchTests(unittest.IsolatedAsyncioTestCase):
    # The orchestrator harness's fakes, without re-running its own tests.
    setUp = harness.OrchestratorTests.setUp
    _run = harness.OrchestratorTests._run

    async def test_planner_off_uses_the_question_as_asked(self):
        planned = []

        async def spy(_self, **kwargs):
            planned.append(kwargs)
            return harness._plan()

        request = QARequest(
            question="Do whole grains lower cholesterol?",
            ablation={"QA_PLANNER_ENABLED": False},
        )
        with _Enabled(), patch.object(harness._FakePlanner, "plan", spy):
            events, calls = await self._run(request=request)
        self.assertEqual(planned, [])
        self.assertEqual(sorted(branch for _, branch, _ in calls), ["articles", "guidelines"])
        self.assertEqual(events[-1].name, "done")

    async def test_planner_on_by_default(self):
        planned = []

        async def spy(_self, **kwargs):
            planned.append(kwargs)
            return harness._plan()

        with patch.object(harness._FakePlanner, "plan", spy):
            await self._run()
        self.assertEqual(len(planned), 1)

    async def test_thread_memory_off_reads_no_earlier_turns(self):
        loaded = []
        original = type(self.service)._load_conversation_state

        def spy(_self, thread_id):
            loaded.append(thread_id)
            return original(_self, thread_id)

        request = QARequest(
            question="And for children?", qa_thread_id="thread-1",
            ablation={"QA_THREAD_MEMORY_ENABLED": False},
        )
        with _Enabled(), patch.object(type(self.service), "_load_conversation_state", spy):
            await self._run(request=request)
        self.assertEqual(loaded, [None])

    async def test_refused_ablation_never_runs(self):
        request = QARequest(question="Is salt bad for me?", ablation={"QA_PLANNER_ENABLED": False})
        with _Enabled(False), self.assertRaises(InvalidError):
            await self._run(request=request)


class RunnerTests(unittest.TestCase):
    def test_parse_ablation(self):
        self.assertEqual(
            runner.parse_ablation(["QA_PLANNER_ENABLED=false", "QA_RETRIEVAL_LEGS=lexical"], '{"QA_MAX_REPAIR_ROUNDS": 0}'),
            {"QA_MAX_REPAIR_ROUNDS": 0, "QA_PLANNER_ENABLED": False, "QA_RETRIEVAL_LEGS": "lexical"},
        )
        with self.assertRaises(ValueError):
            runner.parse_ablation(["QA_PLANNER_ENABLED"], None)

    def test_payload_carries_the_ablation_only_when_set(self):
        q = {"question_id": "q1", "persona_id": "P", "foodscholar_question": "q?"}
        persona = {"api_mapping": {}}
        self.assertNotIn("ablation", runner.request_for(q, persona, run_id="r"))
        payload = runner.request_for(q, persona, run_id="r", ablation={"QA_PLANNER_ENABLED": False})
        self.assertEqual(payload["ablation"], {"QA_PLANNER_ENABLED": False})

    def test_refuses_an_api_that_would_refuse_or_ignore_it(self):
        overlay = {"QA_PLANNER_ENABLED": False}
        runner.check_ablation({}, {})  # no ablation: anything goes
        with self.assertRaises(RuntimeError):
            runner.check_ablation({}, overlay)  # an API from before ablation
        with self.assertRaises(RuntimeError):
            runner.check_ablation({"ablation": {"enabled": False, "switches": {}}}, overlay)
        with self.assertRaises(RuntimeError):
            runner.check_ablation({"ablation": {"enabled": True, "switches": {"QA_RRF_K": {}}}}, overlay)
        runner.check_ablation({"ablation": {"enabled": True, "switches": {"QA_PLANNER_ENABLED": {}}}}, overlay)


class RunnerMemoryTests(unittest.TestCase):
    def test_memory_off_follow_ups_are_not_asked_again(self):
        record = {"follow_up_of": "q1", "status": "ok",
                  "response": {"conversation_context": {"turn_count": 1}}}
        self.assertTrue(runner.lacks_thread_context(record))
        record["request"] = {"ablation": {"QA_THREAD_MEMORY_ENABLED": False}}
        self.assertFalse(runner.lacks_thread_context(record))


class StatisticsTests(unittest.TestCase):
    def test_exact_binomial(self):
        self.assertAlmostEqual(study.binomial_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(study.binomial_two_sided(5, 10), 1.0)
        # 0 of 10: 2 * 0.5^10
        self.assertAlmostEqual(study.binomial_two_sided(0, 10), 2 / 1024)
        self.assertAlmostEqual(study.binomial_two_sided(10, 10), 2 / 1024)
        # 2 of 10 (scipy.stats.binomtest gives 0.109375)
        self.assertAlmostEqual(study.binomial_two_sided(2, 10), 0.109375)

    def test_holm(self):
        adjusted = study.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)  # never below the step before
        self.assertIsNone(adjusted["d"])

    def test_paired_counts_only_shared_questions(self):
        result = study.paired({"q1": 1, "q2": 0, "q3": 1}, {"q1": 1, "q2": 1, "q4": 0})
        self.assertEqual(result["paired"], 2)
        self.assertEqual((result["better"], result["worse"]), (0, 1))
        self.assertAlmostEqual(result["delta"], -0.5)

    def test_verdicts(self):
        significant = {"paired": 300, "delta": -0.05, "p_holm": 0.001}
        self.assertEqual(study.verdict("ablation", significant, noise=0.01), "this part helps")
        self.assertEqual(study.verdict("baseline", significant, noise=0.01), "full system is better")
        self.assertEqual(study.verdict("ablation", significant, noise=0.06), "no detectable difference")
        self.assertEqual(study.verdict("ablation", {**significant, "p_holm": 0.2}, noise=None), "no detectable difference")


class StudyFileTests(unittest.TestCase):
    def _study(self, root, spec):
        path = os.path.join(root, "study.json")
        with open(path, "w") as handle:
            json.dump(spec, handle)
        return study.load_study(path)

    def test_sweeps_expand_and_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            loaded = self._study(root, {
                "name": "s", "out": root, "reference": "full",
                "datasets": {"m": {"personas": "p.json", "questions": "q.json"}},
                "configs": [
                    {"name": "full", "role": "system"},
                    {"name": "no_check", "ablation": {"QA_MAX_REPAIR_ROUNDS": 0}},
                ],
                "sweeps": [{"name": "rounds", "setting": "QA_MAX_REPAIR_ROUNDS", "values": [0, 1, 2], "default": 1}],
            })
        names = [c["name"] for c in loaded["configs"]]
        self.assertEqual(names, ["full", "no_check", "rounds=2"])
        self.assertEqual(loaded["_sweeps"][0]["labels"], [(0, "no_check"), (1, "full"), (2, "rounds=2")])
        added = loaded["configs"][-1]
        self.assertEqual((added["role"], added["stage"], added["ablation"]), ("sweep", 2, {"QA_MAX_REPAIR_ROUNDS": 2}))

    def test_by_type(self):
        result = study.by_type({"a": 1, "b": 0, "c": 1}, {"a": 1, "b": 1, "c": 0}, {"a": "x", "b": "x", "c": "y"})
        self.assertEqual(result, {"x": {"paired": 2, "delta": -0.5}, "y": {"paired": 1, "delta": 1.0}})


class ReportTests(unittest.TestCase):
    def _write_run(self, root, run_id, correct, latency=1000):
        run_dir = os.path.join(root, run_id)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "responses.jsonl"), "w") as handle:
            for i, _ in enumerate(correct):
                response = {
                    "primary_answer": {"answer": "B. Fine. Water helps [general knowledge].", "citations": [{}]},
                    "retrieved_sources": [{}, {}],
                    "reasoning_steps": [{"kind": "search"}, {"kind": "search"}, {"kind": "evaluate"}],
                }
                handle.write(json.dumps({"question_id": f"q{i}", "status": "ok", "latency_ms": latency, "response": response}) + "\n")
        with open(os.path.join(run_dir, "scores.jsonl"), "w") as handle:
            for i, ok in enumerate(correct):
                handle.write(json.dumps({"question_id": f"q{i}", "correct": ok}) + "\n")

    def test_report_pairs_every_configuration_with_the_full_system(self):
        with tempfile.TemporaryDirectory() as root:
            n = 40
            self._write_run(root, "s-m-full", [True] * n)
            self._write_run(root, "s-m-full-r2", [True] * (n - 1) + [False])
            self._write_run(root, "s-m-no_planner", [True] * 20 + [False] * 20)
            self._write_run(root, "s-m-no_cap", [True] * (n - 1) + [False])
            spec = {
                "name": "s", "out": root, "reference": "full", "repeats": {"full": 2},
                "datasets": {"m": {"personas": "p.json", "questions": "q.json", "scorer": "choice"}},
                "configs": [
                    {"name": "full", "role": "system"},
                    {"name": "no_planner", "role": "ablation"},
                    {"name": "no_cap", "role": "ablation"},
                ],
            }
            path = os.path.join(root, "study.json")
            with open(path, "w") as handle:
                json.dump(spec, handle)
            report = study.build_report(study.load_study(path))

        block = report["datasets"]["m"]
        self.assertEqual(block["metric"], "accuracy")
        self.assertAlmostEqual(block["noise_floor"], 1 / 40)
        rows = block["rows"]
        self.assertEqual(rows["no_planner"]["vs_reference"]["verdict"], "this part helps")
        self.assertEqual(rows["no_cap"]["vs_reference"]["verdict"], "no detectable difference")
        self.assertAlmostEqual(rows["full"]["searches"], 2)
        self.assertAlmostEqual(rows["full"]["general_knowledge_rate"], 1.0)
        markdown = study.render_markdown(report)
        self.assertIn("this part helps", markdown)
        self.assertIn("`full#2`", markdown)


if __name__ == "__main__":
    unittest.main()
