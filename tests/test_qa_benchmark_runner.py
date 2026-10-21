"""The QA benchmark runner (scripts/run_qa_benchmark.py) against a fake API."""

import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import httpx

_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "run_qa_benchmark.py"
)
_spec = importlib.util.spec_from_file_location("run_qa_benchmark", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


PERSONAS = {
    "P01": {
        "persona_id": "P01",
        "api_mapping": {
            "expertise_level": "beginner",
            "language": "en",
            "country_or_region": "Dublin, Ireland",
            "member_age_group": "adult",
        },
    },
    "P02": {
        "persona_id": "P02",
        "api_mapping": {
            "expertise_level": "expert",
            "language": "hu",
            "country_or_region": "Budapest, Hungary",
            "member_age_group": "older adult",
        },
    },
}

QUESTIONS = [
    {"question_id": "P01-Guideline-01", "persona_id": "P01", "question_type": "Guideline",
     "language": "en", "foodscholar_question": "How much sugar may a child eat?"},
    {"question_id": "P01-Clarify-01", "persona_id": "P01", "question_type": "Clarification behavior",
     "language": "en", "foodscholar_question": "How much milk should my child drink?"},
    {"question_id": "P02-Factual-01", "persona_id": "P02", "question_type": "Factual",
     "language": "en", "foodscholar_question": "What does creatine do?"},
    {"question_id": "P01-CitationVerification-01", "persona_id": "P01",
     "question_type": "Citation verification", "language": "en", "turn": 2,
     "follow_up_of": "P01-Guideline-01",
     "foodscholar_question": "Which guideline sets that limit?"},
    {"question_id": "P01-CitationVerification-02", "persona_id": "P01",
     "question_type": "Citation verification", "language": "en", "turn": 2,
     "follow_up_of": "P01-Clarify-01",
     "foodscholar_question": "Which study says so?"},
]


CHILD_AGE = {
    "id": "child_age_group",
    "question": "What age group of children are you referring to?",
    "input_type": "single_choice",
    "options": [
        {"label": "0–5 years", "value": "0-5"},
        {"label": "6–12 years", "value": "6-12"},
        {"label": "13–17 years", "value": "13-17"},
    ],
    "allow_free_text": False,
    "reason": "Needs differ by age.",
}


class FakeGroq:
    """Just enough of AsyncGroq: chat.completions.create returning fixed content."""

    def __init__(self, content, finish_reason="stop"):
        self.calls = []
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                message = SimpleNamespace(content=content)
                return SimpleNamespace(
                    model=kwargs["model"],
                    choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                )

        self.chat = SimpleNamespace(completions=Completions())


class FakeAPI:
    """Answers /qa/ask, asking for clarification on the milk question."""

    def __init__(self):
        self.asked = []
        # Questions whose first search errors, as an Elasticsearch timeout does.
        self.search_fails_for = set()
        # False: threads forget, as production did with CACHE_ENABLED=false.
        self.thread_memory = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == runner.RUNTIME_INFO_PATH:
            return httpx.Response(200, json={"git_commit": "abc", "generated_at": "t"})
        body = json.loads(request.content)
        self.asked.append(body)
        thread = body.get("qa_thread_id") or f"thread-{len(self.asked)}"
        clarify = "milk" in body["question"] and not body.get("clarification_response")
        failed = body["question"] in self.search_fails_for
        return httpx.Response(200, json={
            "question": body["question"],
            "qa_thread_id": thread,
            "needs_clarification": clarify,
            "clarification": CHILD_AGE if clarify else None,
            "cache_hit": False,
            "conversation_context": {
                "turn_count": 2 if body.get("qa_thread_id") and self.thread_memory else 1
            },
            "reasoning_steps": [
                {"kind": "search", "title": ("Search unavailable" if failed else "Found 3 sources") + ' — "q"',
                 "data": {"hit_count": 0 if failed else 3}},
            ],
        })


class SelectionTests(unittest.TestCase):
    def test_filter_pulls_in_the_parent_of_a_follow_up(self):
        chosen = runner.select_questions(QUESTIONS, types=["Citation verification"])
        self.assertEqual(
            [q["question_id"] for q in chosen],
            ["P01-Guideline-01", "P01-Clarify-01",
             "P01-CitationVerification-01", "P01-CitationVerification-02"],
        )

    def test_validate_rejects_unknown_persona_and_parent(self):
        bad = QUESTIONS + [
            {"question_id": "X", "persona_id": "P99", "foodscholar_question": "?"},
            {"question_id": "Y", "persona_id": "P01", "follow_up_of": "nope",
             "foodscholar_question": "?"},
        ]
        with self.assertRaises(ValueError) as ctx:
            runner.validate(bad, PERSONAS)
        self.assertIn("unknown persona P99", str(ctx.exception))
        self.assertIn("follow_up_of unknown question nope", str(ctx.exception))


class RequestTests(unittest.TestCase):
    def test_persona_mapping_becomes_the_request(self):
        payload = runner.request_for(QUESTIONS[2], PERSONAS["P02"], run_id="r1")
        self.assertEqual(payload["expertise_level"], "expert")
        self.assertEqual(payload["language"], "en")  # the question row's
        self.assertEqual(payload["region"], "Budapest, Hungary")
        self.assertEqual(payload["country"], "Hungary")
        self.assertEqual(payload["member_age_group"], "older adult")
        self.assertEqual(payload["user_id"], "benchmark:r1")
        self.assertTrue(payload["bypass_cache"])
        self.assertNotIn("qa_thread_id", payload)
        self.assertNotIn("retriever", payload)

    def test_language_can_come_from_the_persona(self):
        payload = runner.request_for(
            QUESTIONS[2], PERSONAS["P02"], run_id="r1", language_from="persona"
        )
        self.assertEqual(payload["language"], "hu")


class DriftTests(unittest.TestCase):
    def test_new_prompt_is_not_drift_but_a_changed_one_is(self):
        start = {"git_commit": "a", "generated_at": "t0",
                 "prompt_versions": {"p1": 3}, "ranking": {"QA_MIN_SCORE": 0.05}}
        end = {"git_commit": "a", "generated_at": "t1",
               "prompt_versions": {"p1": 4, "p2": "fallback"},
               "ranking": {"QA_MIN_SCORE": 0.1}}
        self.assertEqual(
            runner.runtime_drift(start, end),
            ["prompt_versions.p1", "ranking.QA_MIN_SCORE"],
        )


class ClarifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_constrains_the_choice_to_the_offered_options(self):
        groq = FakeGroq('{"choice": "6-12", "reason": "School-age kids."}')
        clarifier = runner.PersonaClarifier(client=groq)
        answer = await clarifier.answer(
            "P01-Clarify-01", PERSONAS["P01"], "How much milk?", CHILD_AGE
        )
        self.assertEqual(answer["clarification_response"]["selected_values"], ["6-12"])
        self.assertEqual(answer["model"], runner.CLARIFIER_MODEL)
        call = groq.calls[0]
        self.assertEqual(call["model"], "openai/gpt-oss-20b")
        schema = call["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["choice"]["enum"], ["0-5", "6-12", "13-17"])
        self.assertEqual((call["reasoning_format"], call["reasoning_effort"]), ("hidden", "low"))
        # The persona's profile is in the prompt; the API plumbing is not.
        prompt = call["messages"][1]["content"]
        self.assertIn("How much milk?", prompt)
        self.assertNotIn("api_mapping", prompt)

    async def test_scripted_answer_wins_without_calling_the_model(self):
        groq = FakeGroq("unused")
        clarifier = runner.PersonaClarifier(client=groq, scripted={"P01-Clarify-01": "0-5"})
        answer = await clarifier.answer("P01-Clarify-01", PERSONAS["P01"], "q", CHILD_AGE)
        self.assertEqual(answer["clarification_response"]["selected_values"], ["0-5"])
        self.assertEqual(answer["source"], "scripted")
        self.assertEqual(groq.calls, [])

    async def test_truncated_output_is_an_error_not_a_guess(self):
        clarifier = runner.PersonaClarifier(client=FakeGroq('{"choi', finish_reason="length"))
        answer = await clarifier.answer("q", PERSONAS["P01"], "q", CHILD_AGE)
        self.assertIn("ran out of tokens", answer["error"])


class RunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.api = FakeAPI()

    async def _run(self, questions, clarifier=None):
        transport = httpx.MockTransport(self.api)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            run = runner.BenchmarkRun(
                client, run_id="r1", run_dir=self.tmp.name, questions=questions,
                personas=PERSONAS, follow_up_delay=0, retries=0, clarifier=clarifier,
            )
            return await run.run()

    async def test_clarification_is_answered_and_both_turns_recorded(self):
        clarifier = runner.PersonaClarifier(
            client=FakeGroq('{"choice": "6-12", "reason": "Two school-age children."}')
        )
        records = await self._run(QUESTIONS, clarifier=clarifier)

        record = records["P01-Clarify-01"]
        self.assertEqual(record["status"], "ok")
        self.assertFalse(record["response"]["needs_clarification"])
        turn = record["clarification_turns"][0]
        self.assertTrue(turn["response"]["needs_clarification"])
        self.assertEqual(turn["answer"]["source"], "persona-llm")
        self.assertEqual(turn["answer"]["reason"], "Two school-age children.")
        # The second call answers in the clarification's own thread.
        self.assertEqual(
            record["request"]["clarification_response"],
            {"question_id": "child_age_group", "selected_values": ["6-12"], "free_text": None},
        )
        self.assertEqual(record["request"]["qa_thread_id"], turn["response"]["qa_thread_id"])
        # Its follow-up is no longer skipped.
        follow_up = records["P01-CitationVerification-02"]
        self.assertEqual(follow_up["status"], "ok")
        self.assertEqual(
            follow_up["request"]["qa_thread_id"], record["response"]["qa_thread_id"]
        )
        counts = runner.summarize(records)
        self.assertEqual((counts["clarifications_asked"], counts["needs_clarification"]), (1, 0))

    async def test_resume_with_a_clarifier_answers_what_was_left_unanswered(self):
        await self._run(QUESTIONS)  # no clarifier: the milk question stays a clarification
        clarifier = runner.PersonaClarifier(client=FakeGroq('{"choice": "6-12", "reason": "r"}'))
        records = await self._run(QUESTIONS, clarifier=clarifier)
        self.assertFalse(records["P01-Clarify-01"]["response"]["needs_clarification"])
        self.assertEqual(records["P01-CitationVerification-02"]["status"], "ok")

    async def test_unanswerable_clarification_stays_the_result(self):
        clarifier = runner.PersonaClarifier(
            client=FakeGroq('{"choice": "toddler", "reason": "?"}')
        )
        records = await self._run(QUESTIONS, clarifier=clarifier)
        record = records["P01-Clarify-01"]
        self.assertTrue(record["response"]["needs_clarification"])
        self.assertIn("error", record["clarification_turns"][0]["answer"])
        self.assertEqual(records["P01-CitationVerification-02"]["status"], "skipped")

    async def test_follow_up_reuses_the_parent_thread(self):
        records = await self._run(QUESTIONS)
        follow_up = records["P01-CitationVerification-01"]
        parent = records["P01-Guideline-01"]
        self.assertEqual(follow_up["status"], "ok")
        self.assertEqual(
            follow_up["request"]["qa_thread_id"], parent["response"]["qa_thread_id"]
        )
        self.assertFalse(runner.lacks_thread_context(follow_up))

    async def test_follow_up_of_a_clarification_is_skipped(self):
        records = await self._run(QUESTIONS)
        self.assertTrue(records["P01-Clarify-01"]["response"]["needs_clarification"])
        skipped = records["P01-CitationVerification-02"]
        self.assertEqual(skipped["status"], "skipped")
        self.assertNotIn(
            "Which study says so?", [b["question"] for b in self.api.asked]
        )

    async def test_every_record_is_written_and_resume_skips_answered(self):
        await self._run(QUESTIONS)
        path = os.path.join(self.tmp.name, "responses.jsonl")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.readlines()), len(QUESTIONS))

        asked_before = len(self.api.asked)
        records = await self._run(QUESTIONS)
        # Only the skipped follow-up is attempted again (and skipped again).
        self.assertEqual(len(self.api.asked), asked_before)
        self.assertEqual(records["P01-CitationVerification-02"]["status"], "skipped")
        counts = runner.summarize(records)
        self.assertEqual((counts["ok"], counts["skipped"]), (4, 1))
        self.assertEqual(counts["follow_up_without_context"], 0)

    async def test_failed_search_is_counted_and_re_asked_on_resume(self):
        parent_question = QUESTIONS[0]["foodscholar_question"]
        self.api.search_fails_for = {parent_question}
        records = await self._run(QUESTIONS)
        self.assertEqual(runner.search_failures(records["P01-Guideline-01"]), 1)
        self.assertEqual(runner.summarize(records)["degraded_retrieval"], 1)
        old_thread = records["P01-Guideline-01"]["response"]["qa_thread_id"]

        self.api.search_fails_for = set()
        asked_before = len(self.api.asked)
        records = await self._run(QUESTIONS)
        re_asked = [b["question"] for b in self.api.asked[asked_before:]]
        # The degraded parent, and its follow-up in the parent's new thread.
        self.assertEqual(
            re_asked, [parent_question, "Which guideline sets that limit?"]
        )
        new_thread = records["P01-Guideline-01"]["response"]["qa_thread_id"]
        self.assertNotEqual(new_thread, old_thread)
        self.assertEqual(
            records["P01-CitationVerification-01"]["request"]["qa_thread_id"], new_thread
        )
        self.assertEqual(runner.summarize(records)["degraded_retrieval"], 0)

    async def test_follow_up_without_context_is_re_asked_with_its_parent(self):
        self.api.thread_memory = False
        records = await self._run(QUESTIONS)
        self.assertEqual(runner.summarize(records)["follow_up_without_context"], 1)

        self.api.thread_memory = True
        asked_before = len(self.api.asked)
        records = await self._run(QUESTIONS)
        self.assertEqual(
            [b["question"] for b in self.api.asked[asked_before:]],
            [QUESTIONS[0]["foodscholar_question"], "Which guideline sets that limit?"],
        )
        self.assertEqual(runner.summarize(records)["follow_up_without_context"], 0)

    async def test_server_error_is_recorded_not_raised(self):
        def broken(request):
            return httpx.Response(500, text="boom")

        self.api = broken
        records = await self._run(QUESTIONS[:1])
        record = records["P01-Guideline-01"]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["http_status"], 500)
        self.assertIn("boom", record["error"])


if __name__ == "__main__":
    unittest.main()
