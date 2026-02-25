import unittest
from unittest.mock import patch


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _prompt: str):
        return _FakeLLMResponse(self._content)


class TipsGenerationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
