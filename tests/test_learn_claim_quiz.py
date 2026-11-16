"""Claim-check intent and the daily quiz (FoodScholar Learn)."""

import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from models.qa import QAAnswer, QARequest, QAResponse
from services.daily_quiz import clamp_quiz_count, quiz_item_id, validate_quiz_items
from services.qa_pipeline import answering
from services.qa_pipeline.answering import normalize_claim_verdict, split_prose_verdict

from test_qa_answer_stream import PAYLOADS, _FakeStreamingLLM
import test_qa_pipeline_orchestrator as _orchestrator_tests


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ClaimIntentModelTests(unittest.TestCase):
    def test_intent_defaults_to_answer(self):
        self.assertEqual(QARequest(question="Are eggs healthy?").intent, "answer")

    def test_claim_check_is_accepted(self):
        request = QARequest(question="Eggs raise cholesterol.", intent="claim_check")
        self.assertEqual(request.intent, "claim_check")

    def test_unknown_intent_is_rejected(self):
        with self.assertRaises(ValidationError):
            QARequest(question="Eggs raise cholesterol.", intent="myth")

    def test_response_claim_verdict_defaults_to_none(self):
        response = QAResponse(
            question="q",
            mode="simple",
            primary_answer=QAAnswer(answer="a", citations=[], confidence="low", model_used="m", rag_used=False),
            generated_at="2026-01-01T00:00:00",
            request_id="r",
        )
        self.assertIsNone(response.claim_verdict)
        self.assertIn("claim_verdict", response.model_dump())


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


class VerdictParsingTests(unittest.TestCase):
    def test_canonical_values(self):
        for value in ("supported", "partly", "not_supported", "unclear"):
            self.assertEqual(normalize_claim_verdict(value), value)

    def test_variants_are_mapped(self):
        cases = {
            "Partly true": "partly",
            "PARTIALLY": "partly",
            "not supported": "not_supported",
            "Not-Supported": "not_supported",
            "false": "not_supported",
            "Myth": "not_supported",
            "insufficient evidence": "unclear",
            "Not enough evidence": "unclear",
            "True": "supported",
            "Supported.": "supported",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_claim_verdict(raw), expected, raw)

    def test_missing_or_invalid_is_none(self):
        for raw in (None, "", "   ", "maybe?", 3, ["supported"], {"verdict": "supported"}):
            self.assertIsNone(normalize_claim_verdict(raw), raw)

    def test_prose_verdict_line_is_split_off(self):
        text, verdict = split_prose_verdict("Verdict: Partly true\n\nEggs contain cholesterol.")
        self.assertEqual(verdict, "partly")
        self.assertEqual(text, "Eggs contain cholesterol.")

    def test_bold_and_comma_forms(self):
        self.assertEqual(split_prose_verdict("**Verdict:** Myth\nBody")[1], "not_supported")
        self.assertEqual(split_prose_verdict("Verdict, supported\nBody")[1], "supported")

    def test_verdict_word_later_in_prose_is_left_alone(self):
        text = "Eggs contain cholesterol.\nThe verdict: it depends."
        self.assertEqual(split_prose_verdict(text), (text, None))


# ---------------------------------------------------------------------------
# Streaming answer with the claim intent
# ---------------------------------------------------------------------------

CLAIM_TRAILER = (
    '{"cited_sources": [{"urn": "urn:article:a1", "section": "abstract", '
    '"quote": "Whole-grain intake reduced LDL cholesterol in adults.", '
    '"confidence": "high"}], "overall_confidence": "medium", '
    '"follow_ups": ["What about oats?"], "verdict": "partly true"}'
)


class _CapturingLLM(_FakeStreamingLLM):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.messages = None

    async def astream(self, messages, config=None):
        self.messages = messages
        async for chunk in super().astream(messages, config=config):
            yield chunk


async def _stream(chunks, intent):
    llm = _CapturingLLM(chunks)
    events = []
    with patch.object(answering.GROQ_CHAT, "get_client", return_value=llm):
        async for event in answering.stream_answer(
            question="Whole grains lower cholesterol.",
            payloads=PAYLOADS,
            expertise_level="beginner",
            language="en",
            model="test-model",
            intent=intent,
        ):
            events.append(event)
    final = [e for e in events if e["kind"] == "final"]
    assert len(final) == 1
    return final[0], llm.messages


class ClaimStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_verdict_comes_from_the_trailer(self):
        final, messages = await _stream(
            [
                "Whole grains lower LDL a little [Doe et al. (2021)](/articles/urn:article:a1).",
                "\n<<<END_ANSWER>>>\n",
                CLAIM_TRAILER,
            ],
            intent="claim_check",
        )
        self.assertEqual(final["claim_verdict"], "partly")
        self.assertIn("CLAIM CHECK MODE", messages[0].content)
        self.assertNotIn("verdict", final["answer"].answer.lower())

    async def test_ordinary_question_never_gets_a_verdict(self):
        final, messages = await _stream(
            ["Whole grains help.", "\n<<<END_ANSWER>>>\n", CLAIM_TRAILER],
            intent="answer",
        )
        self.assertIsNone(final["claim_verdict"])
        self.assertNotIn("CLAIM CHECK MODE", messages[0].content)

    async def test_missing_trailer_verdict_falls_back_to_a_prose_line(self):
        trailer_without_verdict = CLAIM_TRAILER.replace(', "verdict": "partly true"', "")
        final, _ = await _stream(
            ["Verdict: Myth\n\nEggs are fine for most people.", "\n<<<END_ANSWER>>>\n", trailer_without_verdict],
            intent="claim_check",
        )
        self.assertEqual(final["claim_verdict"], "not_supported")
        self.assertTrue(final["answer"].answer.startswith("Eggs are fine"))

    async def test_unusable_verdict_is_none(self):
        final, _ = await _stream(
            ["Eggs are fine.", "\n<<<END_ANSWER>>>\n", CLAIM_TRAILER.replace("partly true", "it depends")],
            intent="claim_check",
        )
        self.assertIsNone(final["claim_verdict"])


# ---------------------------------------------------------------------------
# Orchestrator: verdict on the done payload, intent reaches the answer step
# ---------------------------------------------------------------------------


def _fake_claim_stream(captured):
    async def fake(**kwargs):
        captured.append(kwargs)
        yield {"kind": "delta", "text": "Eggs are fine."}
        yield {
            "kind": "final",
            "answer": QAAnswer(
                answer="Eggs are fine.", citations=[], confidence="medium",
                model_used=kwargs.get("model", "test"), rag_used=True,
            ),
            "follow_ups": [],
            "parsed_trailer": True,
            "claim_verdict": "partly",
        }

    return fake


class ClaimOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    # The orchestrator test harness, borrowed (not inherited, so its own
    # tests do not run twice).
    setUp = _orchestrator_tests.OrchestratorTests.setUp
    _run = _orchestrator_tests.OrchestratorTests._run

    async def _done(self, intent):
        captured = []
        with patch(
            "services.qa_pipeline.orchestrator.stream_answer", _fake_claim_stream(captured)
        ):
            events, _ = await self._run(
                request=QARequest(question="Eggs raise your cholesterol.", intent=intent)
            )
        self.assertEqual(events[-1].name, "done")
        return events[-1].data, captured

    async def test_claim_check_puts_verdict_on_done(self):
        done, captured = await self._done("claim_check")
        self.assertEqual(done["claim_verdict"], {"verdict": "partly"})
        self.assertEqual(captured[0]["intent"], "claim_check")
        self.assertEqual(done["question"], "Eggs raise your cholesterol.")

    async def test_answer_intent_keeps_verdict_null(self):
        done, captured = await self._done("answer")
        self.assertIsNone(done["claim_verdict"])
        self.assertEqual(captured[0]["intent"], "answer")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class CacheKeyIntentTests(unittest.TestCase):
    def setUp(self):
        from services.qa_service import QAService

        self.service = QAService(cache_enabled=False)

    def _data_for(self, request):
        seen = {}

        def capture(prefix, data):
            seen.update(data)
            return "key-" + json.dumps(data, sort_keys=True, default=str)

        with patch.object(self.service.cache_manager, "generate_cache_key", side_effect=capture):
            key = self.service._build_cache_key(request)
        return key, seen

    def test_claim_and_question_with_same_words_get_different_keys(self):
        answer_key, answer_data = self._data_for(QARequest(question="Eggs raise cholesterol."))
        claim_key, claim_data = self._data_for(
            QARequest(question="Eggs raise cholesterol.", intent="claim_check")
        )
        self.assertNotEqual(answer_key, claim_key)
        self.assertEqual(claim_data["intent"], "claim_check")

    def test_default_intent_leaves_existing_keys_unchanged(self):
        _, data = self._data_for(QARequest(question="Are eggs healthy?"))
        self.assertNotIn("intent", data)


# ---------------------------------------------------------------------------
# Daily quiz
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "text": "Eat at least five portions of fruit and vegetables each day.",
        "evidence": {
            "urn": "guideline-1",
            "passage": "Eat at least five portions of fruit and vegetables each day.",
            "title": "Healthy eating guide",
            "source_type": "guideline",
            "guide_urn": "urn:guide:healthy-eating",
            "page_no": 4,
            "region": "EU",
        },
    },
    {
        "text": "Choose whole-grain cereals more often.",
        "evidence": {
            "urn": "guideline-2",
            "passage": "Choose whole-grain cereals more often.",
            "title": "Healthy eating guide",
            "source_type": "guideline",
        },
    },
    {
        "text": "Limit salt to 5 g a day.",
        "evidence": {"urn": "guideline-3", "passage": "Limit salt to 5 g a day.", "source_type": "guideline"},
    },
]


def _item(**overrides):
    item = {
        "question": "How many portions of fruit and vegetables a day?",
        "options": ["At least five", "Two", "One"],
        "answer": 0,
        "why": "Guides advise at least five portions a day.",
        "source": 1,
    }
    item.update(overrides)
    return item


class QuizValidationTests(unittest.TestCase):
    def test_count_is_clamped(self):
        self.assertEqual(clamp_quiz_count(0), 1)
        self.assertEqual(clamp_quiz_count(-3), 1)
        self.assertEqual(clamp_quiz_count(9), 5)
        self.assertEqual(clamp_quiz_count("4"), 4)
        self.assertEqual(clamp_quiz_count("x"), 3)
        self.assertEqual(clamp_quiz_count(None), 3)

    def test_valid_item_takes_evidence_from_its_source(self):
        out = validate_quiz_items([_item(evidence={"urn": "made-up"})], SOURCES, count=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["evidence"]["urn"], "guideline-1")
        self.assertEqual(out[0]["evidence"]["guide_urn"], "urn:guide:healthy-eating")
        self.assertEqual(out[0]["id"], quiz_item_id(_item()["question"]))

    def test_malformed_items_are_dropped(self):
        bad = [
            "not a dict",
            _item(question=""),
            _item(why=""),
            _item(options=["Only", "Two"]),
            _item(options=["a", "b", "c", "d", "e"]),
            _item(options=["Five", "five", "Two"]),
            _item(options=["Five", "", "Two"]),
            _item(options="Five, Two, One"),
            _item(answer=3),
            _item(answer=-1),
            _item(answer=True),
            _item(answer="first"),
            _item(source=0),
            _item(source=9),
            _item(source="two"),
        ]
        self.assertEqual(validate_quiz_items(bad, SOURCES, count=5), [])

    def test_answer_index_and_source_may_be_numeric_strings(self):
        out = validate_quiz_items([_item(answer="2", source="3")], SOURCES, count=3)
        self.assertEqual(out[0]["answer"], 2)
        self.assertEqual(out[0]["evidence"]["urn"], "guideline-3")

    def test_each_source_backs_one_question_and_count_caps(self):
        items = [
            _item(question="Q one?", source=1),
            _item(question="Q two?", source=1),
            _item(question="Q three?", source=2),
            _item(question="Q four?", source=3),
        ]
        out = validate_quiz_items(items, SOURCES, count=2)
        self.assertEqual([o["question"] for o in out], ["Q one?", "Q three?"])

    def test_unsafe_and_avoided_items_are_dropped(self):
        items = [
            _item(question="Which medication helps?", source=1),
            _item(question="Is peanut butter a good snack?", options=["Yes", "No", "Only at night"], source=2),
            _item(question="How much salt a day?", options=["5 g", "15 g", "25 g"], source=3),
        ]
        out = validate_quiz_items(
            items,
            SOURCES,
            count=3,
            is_unsafe=lambda text: "medication" in text.lower(),
            avoid_terms=["peanut"],
        )
        self.assertEqual([o["question"] for o in out], ["How much salt a day?"])

    def test_non_list_input_gives_empty(self):
        self.assertEqual(validate_quiz_items({"items": []}, SOURCES, count=3), [])
        self.assertEqual(validate_quiz_items(None, SOURCES, count=3), [])


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return _FakeLLMResponse(self.content)


class _FakeRedis:
    def __init__(self, fail=False):
        self.store = {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def setex(self, key, _ttl, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value


GUIDELINE_DOCS = [
    {"id": "guideline-1", "guide_urn": "urn:guide:g", "title": "Guide", "rule_text": "Eat at least five portions of fruit and vegetables each day.", "guide_region": "EU"},
    {"id": "guideline-2", "guide_urn": "urn:guide:g", "title": "Guide", "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.", "guide_region": "EU"},
    {"id": "guideline-3", "guide_urn": "urn:guide:g", "title": "Guide", "rule_text": "Limit salt to less than 5 g a day, about one teaspoon.", "guide_region": "EU"},
]


class DailyQuizServiceTests(unittest.TestCase):
    def _service(self, llm_content, redis=None):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        service._simple_question_llm = _FakeLLM(llm_content)
        service._simple_question_redis = redis or _FakeRedis()
        search = patch.object(
            qa_service_module.ELASTIC_CLIENT, "random_search", return_value=GUIDELINE_DOCS
        )
        search.start()
        self.addCleanup(search.stop)
        # Guide titles are looked up per guide; no Elasticsearch in tests.
        titles = patch.object(qa_service_module.ELASTIC_CLIENT, "get_entity", return_value=None)
        titles.start()
        self.addCleanup(titles.stop)
        return service

    LLM_JSON = json.dumps(
        {
            "items": [
                {"question": "How many portions of fruit and veg a day?", "options": ["At least five", "One", "Ten"], "answer": 0, "why": "Guides say at least five.", "source": 1},
                {"question": "Which bread is the better everyday choice?", "options": ["White", "Whole-grain", "Brioche"], "answer": 1, "why": "Whole grain is advised.", "source": 2},
                {"question": "Bad one", "options": ["A", "B"], "answer": 0, "why": "x", "source": 3},
            ]
        }
    )

    def test_generates_validates_and_caches(self):
        redis = _FakeRedis()
        service = self._service(self.LLM_JSON, redis)
        first = service.get_daily_quiz(language="hu", count=3)
        self.assertEqual(len(first.questions), 2)
        self.assertFalse(first.cache_hit)
        self.assertEqual(first.questions[0].evidence.urn, "guideline-1")
        self.assertEqual(first.questions[0].evidence.source_type, "guideline")
        self.assertIn("hu", service._simple_question_llm.prompts[0])
        second = service.get_daily_quiz(language="hu", count=3)
        self.assertTrue(second.cache_hit)
        self.assertEqual([q.id for q in second.questions], [q.id for q in first.questions])

    def test_count_is_clamped_by_the_service(self):
        service = self._service(self.LLM_JSON)
        self.assertLessEqual(len(service.get_daily_quiz(count=1).questions), 1)
        self.assertLessEqual(len(service.get_daily_quiz(count=50).questions), 5)

    def test_garbage_output_gives_empty_list_not_error(self):
        service = self._service("not json at all")
        response = service.get_daily_quiz()
        self.assertEqual(response.questions, [])

    def test_redis_down_still_answers(self):
        service = self._service(self.LLM_JSON, _FakeRedis(fail=True))
        self.assertEqual(len(service.get_daily_quiz().questions), 2)


if __name__ == "__main__":
    unittest.main()
