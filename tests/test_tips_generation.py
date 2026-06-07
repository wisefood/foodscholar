import unittest
from unittest.mock import patch


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _prompt, *args, **kwargs):
        # Accept (and ignore) a LangChain ``config=`` kwarg so the fake matches
        # the real client signature used for Langfuse trace attribution.
        return _FakeLLMResponse(self._content)


class TipsGenerationTests(unittest.TestCase):
    def test_generate_tips_prefers_guideline_rules(self):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        guidelines = [
            {
                "id": "guideline-1",
                "guide_urn": "urn:guide:healthy-eating",
                "title": "Healthy eating guide",
                "rule_text": "Eat at least five portions of fruit and vegetables each day.",
                "guide_region": "EU",
            },
            {
                "id": "guideline-2",
                "guide_urn": "urn:guide:healthy-eating",
                "title": "Healthy eating guide",
                "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.",
                "guide_region": "EU",
            },
        ]

        def _random_search(index_name: str, **_kwargs):
            if index_name == QAService.GUIDELINES_INDEX_NAME:
                return guidelines
            raise AssertionError("articles should not be queried when guidelines work")

        with patch.object(
            qa_service_module.ELASTIC_CLIENT,
            "random_search",
            side_effect=_random_search,
        ):
            llm_content = """
            {
              "items": [
                {"kind": "tip", "text": "Eat fruit and vegetables daily.", "guideline": 1},
                {"kind": "did_you_know", "text": "Dietary guides often recommend daily fruit and vegetables.", "guideline": 1},
                {"kind": "tip", "text": "Choose whole grains more often.", "guideline": 2},
                {"kind": "did_you_know", "text": "Whole-grain options are part of dietary guide advice.", "guideline": 2}
              ]
            }
            """.strip()

            service = QAService(cache_enabled=False)
            service._simple_question_llm = _FakeLLM(llm_content)

            payload = service._generate_tips_payload_once(
                tips_count=2,
                did_you_know_count=2,
            )
            self.assertEqual(len(payload["tips"]), 2)
            self.assertEqual(len(payload["did_you_know"]), 2)
            self.assertEqual(
                payload["tips_detail"][0]["evidence"]["urn"],
                "urn:guide:healthy-eating",
            )
            self.assertEqual(
                payload["tips_detail"][0]["evidence"]["passage"],
                guidelines[0]["rule_text"],
            )

    def test_guideline_rule_text_fallback_handles_invalid_json(self):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        guidelines = [
            {
                "id": "guideline-1",
                "title": "Food guide",
                "rule_text": "Eat at least five portions of fruit and vegetables each day.",
            },
            {
                "id": "guideline-2",
                "title": "Food guide",
                "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.",
            },
        ]

        with patch.object(
            qa_service_module.ELASTIC_CLIENT,
            "random_search",
            return_value=guidelines,
        ):
            service = QAService(cache_enabled=False)
            service._simple_question_llm = _FakeLLM('{"items": [')

            payload = service._generate_tips_payload_once(
                tips_count=2,
                did_you_know_count=2,
            )
            self.assertEqual(len(payload["tips"]), 2)
            self.assertEqual(len(payload["did_you_know"]), 2)
            self.assertEqual(
                payload["tips_detail"][0]["evidence"]["urn"],
                "guideline-1",
            )
            self.assertEqual(
                payload["did_you_know_detail"][0]["evidence"]["passage"],
                guidelines[0]["rule_text"],
            )
            all_text = " ".join(payload["tips"] + payload["did_you_know"])
            self.assertNotIn("Use dietary guide advice", all_text)
            self.assertNotIn("Dietary guides advise", all_text)

    def test_cached_tip_details_are_naturalized_without_changing_evidence(self):
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)
        evidence = {
            "urn": "urn:guide:smoothies",
            "passage": (
                "Smoothies can count towards your fruit and vegetable intake, "
                "but choose only fruit- and/or vegetable-based smoothies and "
                "check the label for sugar and fat."
            ),
            "title": "Smoothies can count",
        }
        payload = {
            "did_you_know_detail": [
                {
                    "text": "Dietary guides advise: Smoothies can count towards your fruit and vegetable intake.",
                    "evidence": evidence,
                }
            ],
            "tips_detail": [
                {
                    "text": "Use dietary guide advice: Smoothies can count towards your fruit and vegetable intake, but choose only fruit- and/or.",
                    "evidence": evidence,
                }
            ],
        }

        normalized = service._normalize_tips_payload(
            payload,
            tips_count=1,
            did_you_know_count=1,
        )

        self.assertNotIn("Dietary guides advise", normalized["did_you_know"][0])
        self.assertNotIn("Use dietary guide advice", normalized["tips"][0])
        self.assertIs(normalized["tips_detail"][0]["evidence"], evidence)
        self.assertIs(normalized["did_you_know_detail"][0]["evidence"], evidence)

    def test_tip_source_queries_are_topical(self):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        calls = []

        def _random_search(index_name: str, **kwargs):
            calls.append((index_name, kwargs))
            return []

        with patch.object(
            qa_service_module.ELASTIC_CLIENT,
            "random_search",
            side_effect=_random_search,
        ):
            service = QAService(cache_enabled=False)
            service._get_random_tip_source_guidelines(size=3, seed=20260422, query="whole grains")
            service._get_random_tip_source_articles(size=3, seed=20260422, query="fiber")

        self.assertEqual(len(calls), 2)
        self.assertIn("multi_match", str(calls[0][1]["filter_query"]))
        self.assertIn("whole grains", str(calls[0][1]["filter_query"]))
        self.assertIn("fiber", str(calls[1][1]["filter_query"]))

    def test_generate_tips_uses_generated_items(self):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        with patch.object(
            qa_service_module.ELASTIC_CLIENT,
            "random_search",
            return_value=[
                {
                    "urn": "urn:article:fiber-1",
                    "title": "Dietary fiber and health outcomes",
                    "publication_year": "2020-01-01",
                    "abstract": (
                        "A randomized trial in adults found higher fiber intake improved satiety "
                        "and supported healthy dietary patterns."
                    ),
                }
            ],
        ):
            llm_content = """
            {
              "items": [
                {"kind": "tip", "text": "Build meals around fiber-rich foods for fullness.", "article": 1},
                {"kind": "did_you_know", "text": "Higher fiber intake is linked with better diet quality.", "article": 1},
                {"kind": "tip", "text": "Choose whole grains more often than refined grains.", "article": 1},
                {"kind": "did_you_know", "text": "Fiber can support regular digestion and gut comfort.", "article": 1}
              ]
            }
            """.strip()

            service = QAService(cache_enabled=False)
            service._simple_question_llm = _FakeLLM(llm_content)

            payload = service._generate_tips_payload_once(tips_count=2, did_you_know_count=2)
            self.assertEqual(len(payload["tips"]), 2)
            self.assertEqual(len(payload["did_you_know"]), 2)
            self.assertIn("Choose whole grains more often than refined grains.", payload["tips"])
            self.assertEqual(len(payload.get("tips_detail", [])), 2)
            self.assertEqual(payload["tips_detail"][0]["evidence"]["urn"], "urn:article:fiber-1")
            self.assertTrue(payload["tips_detail"][0]["evidence"]["passage"])

    def test_tip_candidate_parser_accepts_json5_trailing_commas(self):
        import services.qa_service as qa_service_module
        from services.qa_service import QAService

        with patch.object(
            qa_service_module.ELASTIC_CLIENT,
            "random_search",
            return_value=[
                {
                    "urn": "urn:article:protein-1",
                    "title": "Protein and satiety",
                    "publication_year": "2019-01-01",
                    "abstract": (
                        "In humans, higher protein meals can increase satiety in controlled studies."
                    ),
                }
            ],
        ):
            llm_content = """
            {
              "items": [
                {"kind": "tip", "text": "Include protein with meals to support satiety.", "article": 1,},
                {"kind": "did_you_know", "text": "Protein can influence appetite hormones.", "article": 1,},
                {"kind": "tip", "text": "Combine protein and fiber for lasting fullness.", "article": 1,},
                {"kind": "did_you_know", "text": "Balanced meals can help steady energy levels.", "article": 1,},
              ],
            }
            """.strip()

            service = QAService(cache_enabled=False)
            service._simple_question_llm = _FakeLLM(llm_content)

            payload = service._generate_tips_payload_once(tips_count=2, did_you_know_count=2)
            self.assertEqual(len(payload["tips"]), 2)
            self.assertEqual(len(payload["did_you_know"]), 2)
            self.assertEqual(payload["tips_detail"][0]["evidence"]["urn"], "urn:article:protein-1")

    def test_generate_tips_of_the_day_falls_back_only_after_attempts(self):
        from services.qa_service import QAService

        service = QAService(cache_enabled=False)

        # Guardrail should not false-positive on substrings like "hydrat(ion)".
        self.assertFalse(service._mentions_animal_testing("hydration"))

        def _empty_once(*_args, **_kwargs):
            return {"did_you_know": [], "tips": []}

        service._generate_tips_payload_once = _empty_once  # type: ignore[assignment]

        payload = service._generate_tips_of_the_day(tips_count=2, did_you_know_count=2)
        self.assertEqual(len(payload["tips"]), 2)
        self.assertEqual(len(payload["did_you_know"]), 2)
        self.assertTrue(
            service._is_tips_payload_appropriate(payload, tips_count=2, did_you_know_count=2)
        )


class TipShorteningTests(unittest.TestCase):
    def test_shorten_keeps_short_sentence_whole(self):
        from services.qa_service import QAService
        service = QAService(cache_enabled=False)
        out = service._shorten_tip_sentence(
            "Eat five portions of fruit and vegetables each day.", max_words=18)
        self.assertEqual(out, "Eat five portions of fruit and vegetables each day.")

    def test_shorten_drops_overlong_sentence_instead_of_mangling(self):
        """A first sentence longer than the cap must NOT be cut mid-clause and
        emitted with a trailing dangling word (e.g. '... cereals and.')."""
        from services.qa_service import QAService
        service = QAService(cache_enabled=False)
        long_rule = (
            "Inactive boys aged 13 to 18 should have four to five daily "
            "servings from the wholemeal cereals and breads, potatoes, pasta "
            "and rice shelf of the food pyramid every single day."
        )
        out = service._shorten_tip_sentence(long_rule, max_words=18)
        # Must not produce a mangled fragment ending in a dangling conjunction.
        self.assertFalse(
            out.rstrip(".").endswith(" and"),
            f"mangled mid-clause output: {out!r}",
        )
        self.assertFalse(
            out.rstrip(".").endswith(" the"),
            f"mangled mid-clause output: {out!r}",
        )
        # Drop-if-too-long policy: an over-cap single sentence yields "".
        self.assertEqual(out, "")

    def test_shorten_takes_first_sentence_when_within_cap(self):
        from services.qa_service import QAService
        service = QAService(cache_enabled=False)
        out = service._shorten_tip_sentence(
            "Drink water regularly. Avoid sugary drinks throughout the day.",
            max_words=18)
        self.assertEqual(out, "Drink water regularly.")


class TipsLLMConfigTests(unittest.TestCase):
    def test_simple_question_llm_has_token_budget_and_low_reasoning(self):
        """The tips/starter LLM (a reasoning model) must set an explicit token
        budget and minimal reasoning so JSON output is not truncated."""
        import os
        from services.qa_service import QAService
        from backend.groq import GROQ_CHAT
        # Constructing a real ChatGroq needs a key; the pool reads it from env.
        os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
        GROQ_CHAT._api_key = GROQ_CHAT._api_key or "test-key-not-used"
        service = QAService(cache_enabled=False)
        llm = service.simple_question_llm
        # max_tokens must be set and generous enough for ~12 JSON items.
        self.assertIsNotNone(getattr(llm, "max_tokens", None))
        self.assertGreaterEqual(llm.max_tokens, 1024)
        # reasoning_effort should be minimized for this structured task.
        self.assertEqual(getattr(llm, "reasoning_effort", None), "low")


if __name__ == "__main__":
    unittest.main()
