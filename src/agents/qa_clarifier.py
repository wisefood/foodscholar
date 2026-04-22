"""Structured clarification and safety planning for nutrition QA."""
import json
import logging
import re
from typing import Any, Dict, Optional, Set

try:
    from langchain.prompts import ChatPromptTemplate
except Exception as exc:  # pragma: no cover
    ChatPromptTemplate = None
    _CHAT_PROMPT_IMPORT_ERROR = exc

from backend.groq import GROQ_CHAT
from models.qa import (
    ClarificationOption,
    ClarificationRequest,
    QAClarifierSafetyPlan,
    QARequest,
    QAUserContext,
    DEFAULT_GROQ_MODEL,
)

logger = logging.getLogger(__name__)


class QAClarifierSafetyAgent:
    """One-call planner for safety, material clarification, and retrieval queries."""

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0.0):
        if ChatPromptTemplate is None:  # pragma: no cover
            raise RuntimeError(
                "langchain is required for QAClarifierSafetyAgent."
            ) from _CHAT_PROMPT_IMPORT_ERROR
        self.model = model
        self.temperature = temperature
        self.llm = GROQ_CHAT.get_client(model=model, temperature=temperature)

    def plan(
        self,
        *,
        question: str,
        request: QARequest,
        user_context: QAUserContext,
        answered_ids: Optional[Set[str]] = None,
    ) -> QAClarifierSafetyPlan:
        answered_ids = answered_ids or set()
        fallback = build_fallback_plan(
            question=question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are FoodScholar's combined Clarifier and Safety planner.

Return ONLY valid JSON matching this schema:
{
  "original_question": "string",
  "canonical_question": "string",
  "article_query": "string",
  "guideline_query": "string",
  "output_language": "ISO 639-1 code or null",
  "risk_level": "low | medium | high",
  "safety_flags": ["string"],
  "answer_guardrails": ["string"],
  "needs_clarification": true,
  "clarification": {
    "id": "stable_snake_case_id",
    "question": "one short question",
    "input_type": "single_choice | multiple_choice | free_text | number | boolean",
    "options": [{"label": "short label", "value": "stable_value", "description": null}],
    "allow_free_text": true,
    "reason": "why this materially changes the answer"
  },
  "reasoning_summary": "short operational note"
}

Responsibilities:
- Ask clarification only when the missing detail materially changes safety, retrieval, or practical advice.
- Prefer one short clarification with structured options.
- Do not ask conversational follow-up questions for curiosity.
- Create article_query for scientific articles and guideline_query for food-based dietary guidance.
- Consider user country, region, age group, and experience group when present.
- Flag safety-sensitive cases: infants/children, pregnancy/breastfeeding, chronic disease, kidney/liver disease, diabetes medication, eating disorders, allergies, medication/supplement interactions, severe symptoms.
- If no clarification is needed, set needs_clarification=false and clarification=null.""",
                ),
                (
                    "human",
                    json.dumps(
                        {
                            "question": question,
                            "request_language": request.language,
                            "expertise_level": request.expertise_level,
                            "retriever": request.retriever,
                            "user_context": user_context.model_dump(exclude_none=True),
                            "answered_clarification_ids": sorted(answered_ids),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )

        try:
            response = self.llm.invoke(prompt.format_messages())
            parsed = _parse_json_object(response.content)
            plan = QAClarifierSafetyPlan(**parsed)
            return _merge_with_fallback(plan, fallback, answered_ids)
        except Exception as exc:
            logger.warning(
                "Clarifier/safety planning failed; using deterministic fallback: %s",
                exc,
                exc_info=True,
            )
            return fallback


def build_fallback_plan(
    *,
    question: str,
    request: QARequest,
    user_context: QAUserContext,
    answered_ids: Optional[Set[str]] = None,
) -> QAClarifierSafetyPlan:
    """Deterministic planning fallback used when the LLM is unavailable."""
    answered_ids = answered_ids or set()
    question_lower = question.lower()
    clarification = _fallback_clarification(
        question_lower=question_lower,
        request=request,
        user_context=user_context,
        answered_ids=answered_ids,
    )
    safety_flags = _safety_flags(question_lower)
    risk_level = "medium" if safety_flags else "low"
    if any(flag in safety_flags for flag in ("severe_symptom", "eating_disorder")):
        risk_level = "high"

    query_context = _query_context(user_context)
    article_query = f"{question} {query_context}".strip()
    guideline_query = f"{question} dietary guideline recommendation {query_context}".strip()

    return QAClarifierSafetyPlan(
        original_question=question,
        canonical_question=question,
        article_query=article_query,
        guideline_query=guideline_query,
        output_language=request.language,
        risk_level=risk_level,
        safety_flags=safety_flags,
        answer_guardrails=_answer_guardrails(safety_flags),
        needs_clarification=clarification is not None,
        clarification=clarification,
        reasoning_summary="Deterministic fallback plan.",
    )


def _parse_json_object(content: str) -> Dict[str, Any]:
    content = content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(content)


def _merge_with_fallback(
    plan: QAClarifierSafetyPlan,
    fallback: QAClarifierSafetyPlan,
    answered_ids: Set[str],
) -> QAClarifierSafetyPlan:
    if not plan.article_query.strip():
        plan.article_query = fallback.article_query
    if not plan.guideline_query.strip():
        plan.guideline_query = fallback.guideline_query
    if not plan.canonical_question.strip():
        plan.canonical_question = fallback.canonical_question
    if plan.clarification and plan.clarification.id in answered_ids:
        plan.needs_clarification = False
        plan.clarification = None
    return plan


def _fallback_clarification(
    *,
    question_lower: str,
    request: QARequest,
    user_context: QAUserContext,
    answered_ids: Set[str],
) -> Optional[ClarificationRequest]:
    if (
        "target_age_group" not in answered_ids
        and _mentions_minor_without_age(question_lower)
        and not user_context.member_age_group
    ):
        return ClarificationRequest(
            id="target_age_group",
            question="Who is the nutrition guidance for?",
            input_type="single_choice",
            options=[
                ClarificationOption(label="Infant", value="infant"),
                ClarificationOption(label="Child", value="child"),
                ClarificationOption(label="Teen", value="teen"),
                ClarificationOption(label="Adult", value="adult"),
            ],
            allow_free_text=True,
            reason="Age materially changes safe nutrition advice.",
        )

    if (
        "country_or_region" not in answered_ids
        and _looks_like_practical_recommendation(question_lower)
        and not user_context.country
        and not request.member_id
    ):
        return ClarificationRequest(
            id="country_or_region",
            question="Which country or guideline region should the answer use?",
            input_type="single_choice",
            options=[
                ClarificationOption(label="United States", value="US"),
                ClarificationOption(label="European Union", value="EU"),
                ClarificationOption(label="United Kingdom", value="UK"),
                ClarificationOption(label="Greece", value="GR"),
                ClarificationOption(label="No preference", value="general"),
            ],
            allow_free_text=True,
            reason="Food-based recommendations can differ by country or guideline region.",
        )

    if (
        "safety_context" not in answered_ids
        and _looks_like_supplement_safety_question(question_lower)
    ):
        return ClarificationRequest(
            id="safety_context",
            question="Is there any relevant safety context?",
            input_type="single_choice",
            options=[
                ClarificationOption(label="None", value="none"),
                ClarificationOption(
                    label="Pregnant or breastfeeding",
                    value="pregnancy_or_breastfeeding",
                ),
                ClarificationOption(
                    label="Chronic condition",
                    value="chronic_condition",
                ),
                ClarificationOption(
                    label="Medication or supplement use",
                    value="medication_or_supplement",
                ),
            ],
            allow_free_text=True,
            reason="Conditions and medications can change supplement safety advice.",
        )

    return None


def _mentions_minor_without_age(question_lower: str) -> bool:
    minor_terms = [
        "child",
        "children",
        "kid",
        "kids",
        "toddler",
        "infant",
        "baby",
        "teen",
        "teenager",
    ]
    if not any(term in question_lower for term in minor_terms):
        return False
    return not bool(re.search(r"\b\d{1,2}\s*(-|to)?\s*(year|yr|yo)", question_lower))


def _looks_like_practical_recommendation(question_lower: str) -> bool:
    practical_terms = [
        "how much",
        "how many",
        "how often",
        "per week",
        "per day",
        "should",
        "recommend",
        "guideline",
    ]
    return any(term in question_lower for term in practical_terms)


def _looks_like_supplement_safety_question(question_lower: str) -> bool:
    supplement_terms = ["supplement", "dose", "dosage", "take", "tablet"]
    safety_terms = ["safe", "risk", "interact", "pregnan", "kidney", "liver"]
    return any(t in question_lower for t in supplement_terms) and any(
        t in question_lower for t in safety_terms
    )


def _safety_flags(question_lower: str) -> list[str]:
    flags = []
    checks = {
        "infant_or_child": ["infant", "baby", "child", "children", "kid", "toddler"],
        "pregnancy_or_breastfeeding": ["pregnan", "breastfeed", "lactat"],
        "kidney_disease": ["kidney", "ckd", "renal"],
        "liver_disease": ["liver", "hepatic"],
        "diabetes_medication": ["diabetes", "insulin", "metformin"],
        "medication_interaction": ["medication", "medicine", "drug", "interact"],
        "eating_disorder": ["eating disorder", "anorexia", "bulimia"],
        "allergy": ["allergy", "allergic", "anaphylaxis"],
        "severe_symptom": ["chest pain", "faint", "vomiting blood", "severe pain"],
    }
    for flag, terms in checks.items():
        if any(term in question_lower for term in terms):
            flags.append(flag)
    return flags


def _answer_guardrails(safety_flags: list[str]) -> list[str]:
    guardrails = []
    if safety_flags:
        guardrails.append("Do not diagnose or prescribe treatment.")
        guardrails.append("Recommend a clinician for personalized medical nutrition care.")
    if "infant_or_child" in safety_flags:
        guardrails.append("Avoid adult nutrition assumptions for pediatric questions.")
    if "medication_interaction" in safety_flags:
        guardrails.append("Mention medication/supplement interactions may need professional review.")
    return guardrails


def _query_context(user_context: QAUserContext) -> str:
    hints = []
    if user_context.country:
        hints.append(user_context.country)
    if user_context.region and user_context.region != user_context.country:
        hints.append(user_context.region)
    if user_context.member_age_group:
        hints.append(user_context.member_age_group)
    if user_context.experience_group:
        hints.append(user_context.experience_group)
    return " ".join(hints)
