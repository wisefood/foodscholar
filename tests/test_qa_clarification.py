import unittest

from models.qa import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationRequest,
    QAClarifierSafetyPlan,
    QARequest,
    QAUserContext,
)
from services.qa_retrievers import RetrievalResult
from services.qa_service import QAService


class QAClarificationTests(unittest.TestCase):
    def test_child_question_gets_structured_age_clarification(self):
        service = QAService(cache_enabled=False)
        request = QARequest(
            question="How often should children eat red meat?",
            retriever="rag",
        )
        context = QAUserContext(experience_group="beginner")

        clarification = service._build_material_clarification(
            question=request.question,
            request=request,
            user_context=context,
        )

        self.assertIsNotNone(clarification)
        self.assertEqual(clarification.id, "target_age_group")
        self.assertEqual(clarification.input_type, "single_choice")
        self.assertIn(
            "child",
            [option.value for option in clarification.options],
        )

    def test_clarification_answer_is_folded_into_effective_question(self):
        service = QAService(cache_enabled=False)
        clarification = ClarificationRequest(
            id="country_or_region",
            question="Which country or guideline region should the answer use?",
            input_type="single_choice",
            options=[
                ClarificationOption(label="Greece", value="GR"),
                ClarificationOption(label="No preference", value="general"),
            ],
        )
        request = QARequest(
            question="How often should I eat legumes?",
            qa_thread_id="thread-1",
            clarification_response=ClarificationAnswer(
                question_id="country_or_region",
                selected_values=["GR"],
                free_text="for a beginner audience",
            ),
        )

        effective_question = service._compose_effective_question(
            request=request,
            thread_context={
                "question": "How often should I eat legumes?",
                "clarification": clarification.model_dump(),
            },
        )

        self.assertIn("How often should I eat legumes?", effective_question)
        self.assertIn("Greece", effective_question)
        self.assertIn("beginner audience", effective_question)

    def test_guideline_context_boosts_include_region_and_experience(self):
        service = QAService(cache_enabled=False)
        context = QAUserContext(
            country="GR",
            region="GR",
            experience_group="beginner",
            member_age_group="adult",
        )

        clauses = service._guideline_context_should_clauses(context)

        self.assertGreaterEqual(len(clauses), 2)
        serialized = str(clauses)
        self.assertIn("guide_region", serialized)
        self.assertIn("target_populations", serialized)

    def test_region_clarification_options_use_scouted_guidelines(self):
        service = QAService(cache_enabled=False)
        request = QARequest(
            question="How much red meat should my child eat weekly?",
            qa_thread_id="thread-1",
        )
        plan = QAClarifierSafetyPlan(
            original_question=request.question,
            canonical_question=request.question,
            article_query="red meat child weekly intake",
            guideline_query="red meat child weekly dietary guideline",
            needs_clarification=True,
            clarification=ClarificationRequest(
                id="country_or_region",
                question="Which country or guideline region should the answer use?",
                options=[
                    ClarificationOption(label="United States", value="US"),
                    ClarificationOption(label="No preference", value="general"),
                ],
            ),
        )

        clarification = service._build_scout_clarification(
            question=request.question,
            plan=plan,
            pending_clarification=plan.clarification,
            request=request,
            user_context=QAUserContext(experience_group="beginner"),
            answered_ids={"target_age_group"},
            source_payloads=[
                {
                    "source_type": "guideline",
                    "guide_region": "HU",
                    "title": "Hungarian dietary guide",
                    "rule_text": "Limit red meat frequency for children.",
                },
                {
                    "source_type": "guideline",
                    "guide_region": "SI",
                    "title": "Slovenian dietary guide",
                    "rule_text": "Prefer poultry and legumes more often.",
                },
            ],
            retrieved_sources=[],
            retrieval_result=RetrievalResult(
                status={"guideline_hits": 2, "article_hits": 0}
            ),
            effective_rag=True,
        )

        self.assertIsNotNone(clarification)
        self.assertEqual(clarification.id, "country_or_region")
        self.assertEqual(
            [option.value for option in clarification.options],
            ["HU", "SI", "general"],
        )
        self.assertEqual(clarification.options[0].label, "Hungary")


if __name__ == "__main__":
    unittest.main()
