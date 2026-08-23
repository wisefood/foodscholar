"""Research planner for the agentic QA pipeline.

One async LLM call that does everything the legacy clarifier/safety step did
(safety flags, guardrails, material clarification, canonical question) plus
the retrieval decomposition: typed sub-questions, each with a lexical and a
dense query and a user-visible one-line rationale. Prior research notes from
the same conversation thread are part of the input so follow-up questions
search from what earlier turns already established.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage

from agents.json_output import parse_json_object
from agents.qa_clarifier import build_fallback_plan
from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import QA_PLANNER_SYSTEM
from config import config
from models.qa import (
    PlannedSubQuestion,
    QAClarifierSafetyPlan,
    QAPipelinePlan,
    QARequest,
    QAUserContext,
    ResearchNote,
    SubQuestionFilters,
)

logger = logging.getLogger(__name__)


def _max_subquestions() -> int:
    try:
        return max(int(config.settings.get("QA_MAX_SUBQUESTIONS", 3)), 1)
    except (TypeError, ValueError):
        return 3


def build_fallback_pipeline_plan(
    *,
    question: str,
    request: QARequest,
    user_context: QAUserContext,
    answered_ids: Optional[Set[str]] = None,
) -> QAPipelinePlan:
    """Deterministic plan when the LLM is unavailable: one search per branch."""
    base = build_fallback_plan(
        question=question,
        request=request,
        user_context=user_context,
        answered_ids=answered_ids,
    )
    sub_questions = [
        PlannedSubQuestion(
            id="sq1",
            text=question,
            why="Direct search of the scientific literature for the question as asked.",
            qtype="general",
            branch="articles",
            lexical_query=base.article_query,
            dense_query=question,
        ),
        PlannedSubQuestion(
            id="sq2",
            text=question,
            why="Checking dietary guidelines for practical recommendations.",
            qtype="recommendation",
            branch="guidelines",
            lexical_query=base.guideline_query,
            dense_query=question,
        ),
    ]
    return QAPipelinePlan(**base.model_dump(), sub_questions=sub_questions)


def _coerce_filters(raw: Any) -> SubQuestionFilters:
    """Validate the planner's attribute constraints; empty on anything odd."""
    if not isinstance(raw, dict):
        return SubQuestionFilters()

    def _year(value: Any) -> Any:
        try:
            year = int(value)
        except (TypeError, ValueError):
            return None
        return year if 1800 <= year <= 2200 else None

    def _terms(value: Any) -> list:
        if not isinstance(value, list):
            return []
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ][:6]

    try:
        return SubQuestionFilters(
            year_min=_year(raw.get("year_min")),
            year_max=_year(raw.get("year_max")),
            open_access=raw.get("open_access")
            if isinstance(raw.get("open_access"), bool)
            else None,
            study_types=_terms(raw.get("study_types")),
            regions=_terms(raw.get("regions")),
            target_populations=_terms(raw.get("target_populations")),
            food_groups=_terms(raw.get("food_groups")),
            nutrients=_terms(raw.get("nutrients")),
            health_conditions=_terms(raw.get("health_conditions")),
        )
    except Exception:  # pragma: no cover - fully guarded above
        return SubQuestionFilters()


def _coerce_sub_questions(
    raw: Any, *, question: str, round_added: int = 0
) -> List[PlannedSubQuestion]:
    """Validate the planner's sub-question list, dropping malformed entries."""
    result: List[PlannedSubQuestion] = []
    if not isinstance(raw, list):
        return result
    for index, entry in enumerate(raw[: _max_subquestions()], start=1):
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        try:
            sq = PlannedSubQuestion(
                filters=_coerce_filters(entry.get("filters")),
                id=str(entry.get("id") or f"sq{index}"),
                text=text,
                why=str(entry.get("why") or "").strip(),
                qtype=entry.get("qtype")
                if entry.get("qtype")
                in {
                    "quantity",
                    "mechanism",
                    "safety",
                    "recommendation",
                    "comparison",
                    "general",
                }
                else "general",
                branch=entry.get("branch")
                if entry.get("branch") in {"articles", "guidelines", "both"}
                else "both",
                lexical_query=str(entry.get("lexical_query") or "").strip() or text,
                dense_query=str(entry.get("dense_query") or "").strip() or text,
                round_added=round_added,
            )
        except Exception:  # pragma: no cover - pydantic guards above
            continue
        result.append(sq)
    # Planner ids must be unique; collisions get positional ids.
    seen: Set[str] = set()
    for index, sq in enumerate(result, start=1):
        if sq.id in seen:
            sq.id = f"sq{index}"
        seen.add(sq.id)
    return result


class QAPlannerAgent:
    """One-call planner: safety + clarification + reasoned search decomposition."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or config.settings["QA_PLANNER_MODEL"]
        self.temperature = temperature
        self.llm = GROQ_CHAT.get_client(model=self.model, temperature=temperature)

    async def plan(
        self,
        *,
        question: str,
        request: QARequest,
        user_context: QAUserContext,
        answered_ids: Optional[Set[str]] = None,
        prior_notes: Optional[List[ResearchNote]] = None,
    ) -> QAPipelinePlan:
        answered_ids = answered_ids or set()
        fallback = build_fallback_pipeline_plan(
            question=question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
        )

        system_text = QA_PLANNER_SYSTEM.compile(
            max_subquestions=_max_subquestions()
        )
        payload: Dict[str, Any] = {
            "question": question,
            "request_language": request.language,
            "expertise_level": request.expertise_level,
            "user_context": user_context.model_dump(exclude_none=True),
            "answered_clarification_ids": sorted(answered_ids),
        }
        if prior_notes:
            payload["research_notes"] = [
                {"kind": note.kind, "text": note.text}
                for note in prior_notes[-12:]
            ]
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

        # user_context (which may include allergies) is model input by
        # necessity but is never copied into trace metadata.
        trace = build_trace_config(
            run_name="qa-planner",
            session_id=request.qa_thread_id,
            user_id=request.user_id or request.member_id,
            tags=["qa", "pipeline", "planner"],
        )

        try:
            response = await self.llm.ainvoke(messages, config=trace)
            parsed = parse_json_object(response.content)
            sub_questions = _coerce_sub_questions(
                parsed.pop("sub_questions", None), question=question
            )
            base = QAClarifierSafetyPlan(**{
                key: value
                for key, value in parsed.items()
                if key in QAClarifierSafetyPlan.model_fields
            })
            plan = QAPipelinePlan(
                **base.model_dump(), sub_questions=sub_questions
            )
            return _merge_plan_with_fallback(plan, fallback, answered_ids)
        except Exception as exc:
            logger.warning(
                "QA planner failed; using deterministic fallback plan: %s",
                exc,
                exc_info=True,
            )
            return fallback


def _merge_plan_with_fallback(
    plan: QAPipelinePlan,
    fallback: QAPipelinePlan,
    answered_ids: Set[str],
) -> QAPipelinePlan:
    """Backfill anything the model left empty; suppress answered clarifications."""
    if not plan.article_query.strip():
        plan.article_query = fallback.article_query
    if not plan.guideline_query.strip():
        plan.guideline_query = fallback.guideline_query
    if not plan.canonical_question.strip():
        plan.canonical_question = fallback.canonical_question
    if not plan.original_question.strip():
        plan.original_question = fallback.original_question
    if not plan.sub_questions:
        plan.sub_questions = fallback.sub_questions
    if plan.clarification and plan.clarification.id in answered_ids:
        plan.needs_clarification = False
        plan.clarification = None
    return plan
