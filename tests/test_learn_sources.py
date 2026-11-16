"""Learn source sampling: who a rule is for, and a pool that does not repeat.

The daily quiz once asked "How much cooked meat counts as one serving?" with
"30 g" marked right: true only for children aged 1 to 4, from a toddler guide,
asked as if it were everyone's portion. And the quiz and tips kept returning to
the same few rules. These tests pin the two fixes: a rule carries its group
into the prompt and its age window into the evidence; and the pool is spread
across guides and journals, blends in studies, and puts recent sources last.
"""

import json
import unittest
from unittest.mock import patch

from services.daily_quiz import build_quiz_source_context, validate_quiz_items
from services.learn_sources import (
    APPLIES_TO_NOTE,
    balanced_order,
    blend,
    dedupe,
    format_age_window,
    guideline_age_window,
    guideline_applies_to,
    interleave_by_group,
    prefer_unseen,
    read_recent_sources,
    remember_sources,
)


class AgeWindowTests(unittest.TestCase):
    def test_toddler_window_is_a_group(self):
        self.assertEqual(guideline_age_window({"age_min_months": 12, "age_max_months": 48}), (12, 48))

    def test_window_reaching_adulthood_is_everyone(self):
        self.assertEqual(guideline_age_window({"age_min_months": 12, "age_max_months": None}), (None, None))
        self.assertEqual(guideline_age_window({"age_min_months": 216}), (None, None))
        self.assertEqual(guideline_age_window({"age_min_months": 0, "age_max_months": 1200}), (None, None))

    def test_older_adults_are_a_group(self):
        self.assertEqual(guideline_age_window({"age_min_months": 780}), (780, None))

    def test_not_stated_is_minus_one_or_missing(self):
        self.assertEqual(guideline_age_window({"age_min_months": -1, "age_max_months": -1}), (None, None))
        self.assertEqual(guideline_age_window({}), (None, None))
        self.assertEqual(guideline_age_window({"age_min_months": True}), (None, None))

    def test_format(self):
        self.assertEqual(format_age_window(12, 48), "ages 1 to 4 years")
        self.assertEqual(format_age_window(6, 12), "ages 6 to 12 months")
        self.assertEqual(format_age_window(18, 36), "ages 18 to 36 months")
        self.assertEqual(format_age_window(780, None), "ages 65 years and over")
        self.assertEqual(format_age_window(None, 60), "ages up to 5 years")
        self.assertEqual(format_age_window(None, None), "")
        self.assertEqual(format_age_window(36, 36), "age 3 years")

    def test_applies_to_names_population_life_stage_and_ages(self):
        line = guideline_applies_to(
            {
                "target_populations": ["toddlers"],
                "life_stage": ["early_childhood"],
                "age_min_months": 12,
                "age_max_months": 48,
            }
        )
        self.assertEqual(line, "toddlers; early childhood; ages 1 to 4 years")
        self.assertEqual(guideline_applies_to({"rule_text": "Eat vegetables."}), "")


class PoolOrderTests(unittest.TestCase):
    def test_interleave_takes_groups_in_turn_and_drops_nothing(self):
        items = ["a1", "a2", "a3", "a4", "b1", "c1", "c2"]
        out = interleave_by_group(items, group=lambda s: s[0])
        self.assertEqual(out, ["a1", "b1", "c1", "a2", "c2", "a3", "a4"])

    def test_interleave_cap_and_ungrouped(self):
        out = interleave_by_group(["a1", "a2", "a3", "x", "y"], group=lambda s: s[0] if s[0] == "a" else None, per_group=2)
        self.assertEqual(out, ["a1", "x", "y", "a2"])

    def test_balanced_order_puts_recent_last_and_limits(self):
        items = ["a1", "a2", "b1", "b2", "c1"]
        out = balanced_order(items, key=lambda s: s, group=lambda s: s[0], recent={"a1", "b1"})
        # Unseen first, guide by guide; then the recently shown ones.
        self.assertEqual(out, ["a2", "b2", "c1", "a1", "b1"])
        self.assertEqual(balanced_order(items, key=lambda s: s, group=lambda s: s[0], recent=set(), limit=3), ["a1", "b1", "c1"])

    def test_prefer_unseen(self):
        self.assertEqual(prefer_unseen([1, 2, 3], key=lambda n: str(n), recent={"1"}), [2, 3, 1])
        self.assertEqual(prefer_unseen([1, 2], key=str, recent=set()), [1, 2])

    def test_blend_two_rules_then_a_study(self):
        self.assertEqual(blend(["r1", "r2", "r3"], ["s1", "s2", "s3"]), ["r1", "r2", "s1", "r3", "s2", "s3"])
        self.assertEqual(blend([], ["s1"]), ["s1"])

    def test_dedupe_keeps_first_and_drops_keyless(self):
        self.assertEqual(dedupe([{"u": "1"}, {"u": "1"}, {"u": ""}, {"u": "2"}], key=lambda d: d["u"]), [{"u": "1"}, {"u": "2"}])


class _SortedSetRedis:
    """Just enough of redis-py's sorted sets."""

    def __init__(self, fail=False):
        self.fail = fail
        self.sets = {}
        self.expiry = {}

    def _check(self):
        if self.fail:
            raise ConnectionError("redis down")

    def zadd(self, key, mapping):
        self._check()
        self.sets.setdefault(key, {}).update(mapping)

    def zrangebyscore(self, key, low, high):
        self._check()
        hi = float("inf") if high == "+inf" else high
        return [m.encode() for m, s in self.sets.get(key, {}).items() if low <= s <= hi]

    def zremrangebyscore(self, key, low, high):
        self._check()
        lo = float("-inf") if low == "-inf" else low
        self.sets[key] = {m: s for m, s in self.sets.get(key, {}).items() if not (lo <= s <= high)}

    def expire(self, key, seconds):
        self.expiry[key] = seconds

    # The quiz cache, as the existing tests' fake has it.
    def get(self, key):
        self._check()
        return None

    def setex(self, key, _ttl, value):
        self._check()


class RecentSourcesTests(unittest.TestCase):
    def test_remember_then_read_within_window(self):
        redis = _SortedSetRedis()
        remember_sources(redis, "quiz", ["urn:1", None, "urn:2"], today=1000)
        self.assertEqual(read_recent_sources(redis, "quiz", today=1005), {"urn:1", "urn:2"})
        # Two weeks later they are forgotten.
        self.assertEqual(read_recent_sources(redis, "quiz", today=1015), set())

    def test_old_entries_are_pruned_on_write(self):
        redis = _SortedSetRedis()
        remember_sources(redis, "tips", ["old"], today=1000)
        remember_sources(redis, "tips", ["new"], today=1020)
        self.assertEqual(set(redis.sets["qa:learn:recent:tips"]), {"new"})

    def test_redis_down_or_missing_never_raises(self):
        self.assertEqual(read_recent_sources(_SortedSetRedis(fail=True), "quiz", today=1), set())
        self.assertEqual(read_recent_sources(None, "quiz", today=1), set())
        remember_sources(_SortedSetRedis(fail=True), "quiz", ["x"], today=1)
        remember_sources(None, "quiz", ["x"], today=1)


class QuizContextTests(unittest.TestCase):
    SOURCES = [
        {
            "text": "One serving of cooked meat is 30 g.",
            "guide": "Eating guidelines for 1-4 year olds",
            "applies_to": "toddlers; ages 1 to 4 years",
            "evidence": {
                "urn": "urn:guideline:meat",
                "passage": "One serving of cooked meat is 30 g.",
                "title": "Meat portions",
                "source_type": "guideline",
                "age_min_months": 12,
                "age_max_months": 48,
            },
        },
        {
            "text": "Higher fibre intake was associated with lower risk.",
            "evidence": {"urn": "urn:article:fibre", "passage": "Higher fibre ...", "title": "Fibre study", "source_type": "article"},
        },
    ]

    def test_rules_carry_guide_and_group_behind_the_note(self):
        context = build_quiz_source_context(self.SOURCES)
        self.assertTrue(context.startswith(APPLIES_TO_NOTE))
        self.assertIn("- Guide: Eating guidelines for 1-4 year olds", context)
        self.assertIn("- Applies to: toddlers; ages 1 to 4 years", context)
        study_block = context.split("Source 2")[1]
        self.assertNotIn("Applies to", study_block)

    def test_validated_item_keeps_the_age_window(self):
        items = validate_quiz_items(
            [{"question": "For children aged 1 to 4, how much cooked meat is one serving?", "options": ["30 g", "50 g", "100 g"], "answer": 0, "why": "The toddler guide says 30 g.", "source": 1}],
            self.SOURCES,
            count=3,
        )
        self.assertEqual(items[0]["evidence"]["age_min_months"], 12)
        self.assertEqual(items[0]["evidence"]["age_max_months"], 48)


def _rule(i, guide, **extra):
    return {
        "id": f"rule-{guide}-{i}",
        "guide_urn": f"urn:guide:{guide}",
        "title": f"Rule {i}",
        "rule_text": f"Rule number {i} from guide {guide} about everyday eating.",
        "guide_region": "EU",
        **extra,
    }


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)

        class _R:
            pass

        r = _R()
        r.content = self.content
        return r


class ServicePoolTests(unittest.TestCase):
    """The pools as QAService builds them, with Elasticsearch faked."""

    BIG_GUIDE = [_rule(i, "toddler", age_min_months=12, age_max_months=48) for i in range(8)]
    OTHER = [_rule(0, "adults"), _rule(1, "adults"), _rule(0, "pregnancy")]
    ARTICLES = [
        {"urn": f"urn:article:{j}-{i}", "title": f"Study {j}{i}", "venue": j, "abstract": f"A long enough abstract about diet and health number {i} in {j}."}
        for j in ("lancet", "bmj") for i in range(3)
    ]

    def _service(self, redis=None, llm=None):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        service._simple_question_redis = redis or _SortedSetRedis()
        service._simple_question_llm = llm or _FakeLLM("{}")

        def fake_random_search(index_name, size=10, *, seed=None, filter_query=None, source_excludes=None):
            if index_name == "guidelines":
                # The big guide dominates the head of the raw search result.
                return (self.BIG_GUIDE + self.OTHER)[:size]
            return self.ARTICLES[:size]

        for target, kwargs in (
            ("random_search", {"side_effect": fake_random_search}),
            ("get_entity", {"side_effect": lambda index, urn: {"title": {"urn:guide:toddler": "Eating guidelines for 1-4 year olds"}.get(urn, "Adult guide")}}),
        ):
            p = patch.object(qa_service_module.ELASTIC_CLIENT, target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        return service

    def test_quiz_sources_spread_guides_blend_studies_and_name_the_group(self):
        service = self._service()
        sources = service._quiz_sources(count=3, seed=1, member_topics=[])
        guides = [s["evidence"].get("guide_urn") for s in sources if s["evidence"].get("source_type") == "guideline"]
        # The first three rules come from three different guides.
        self.assertEqual(len(set(guides[:3])), 3)
        # Studies take part: every third source is a study.
        self.assertEqual(sources[2]["evidence"]["source_type"], "article")
        toddler = next(s for s in sources if s["evidence"].get("guide_urn") == "urn:guide:toddler")
        self.assertEqual(toddler["guide"], "Eating guidelines for 1-4 year olds")
        self.assertIn("ages 1 to 4 years", toddler["applies_to"])
        self.assertEqual((toddler["evidence"]["age_min_months"], toddler["evidence"]["age_max_months"]), (12, 48))

    def test_recently_shown_sources_go_last(self):
        redis = _SortedSetRedis()
        service = self._service(redis)
        first = service._quiz_sources(count=3, seed=1, member_topics=[])
        shown = [s["evidence"]["urn"] for s in first[:4]]
        remember_sources(redis, "quiz", shown, today=service._learn_today())
        second = service._quiz_sources(count=3, seed=1, member_topics=[])
        self.assertTrue(set(s["evidence"]["urn"] for s in second[:4]).isdisjoint(shown))

    def test_daily_quiz_remembers_its_sources(self):
        redis = _SortedSetRedis()
        llm = _FakeLLM(json.dumps({"items": [{"question": "For children aged 1 to 4, how much cooked meat is one serving?", "options": ["30 g", "50 g", "100 g"], "answer": 0, "why": "The toddler guide says 30 g.", "source": 1}]}))
        service = self._service(redis, llm)
        quiz = service.get_daily_quiz(count=1)
        self.assertEqual(len(quiz.questions), 1)
        self.assertIn(APPLIES_TO_NOTE, llm.prompts[0])
        self.assertIn(quiz.questions[0].evidence.urn, read_recent_sources(redis, "quiz", service._learn_today()))

    def test_tip_context_carries_guide_title_and_group(self):
        service = self._service()
        context = service._prepare_tip_guideline_context(self.BIG_GUIDE[:1] + self.OTHER[:1])
        self.assertTrue(context.startswith(APPLIES_TO_NOTE))
        self.assertIn("- Guide: Eating guidelines for 1-4 year olds", context)
        self.assertIn("- Applies to: ages 1 to 4 years", context)
        self.assertIn("- Applies to: not stated", context)


if __name__ == "__main__":
    unittest.main()
