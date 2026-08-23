"""Evidence sufficiency evaluation with diagnosed failure causes and notes.

Deterministic gates run first so the obvious cases (every branch failed, the
repair budget is spent) never cost an LLM call. Otherwise a small model judges
coverage per sub-question, diagnoses the cause of any failure — vocabulary
mismatch, wrong branch, missing sub-question, genuine corpus gap, or a missing
user detail — and keeps the pipeline's research notes: findings, gaps, and
leads that steer repair rounds now and seed the planner on follow-up turns.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.clarifier_fallback_i18n import localize as _clarify_i18n
from agents.json_output import parse_json_object
from agents.qa_planner import _coerce_sub_questions
from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import QA_EVALUATOR_SYSTEM
from config import config
from models.qa import (
    ClarificationOption,
    ClarificationRequest,
    PlannedSubQuestion,
    ResearchNote,
)
from services.qa_pipeline.state import EvidenceItem, PipelineState

logger = logging.getLogger(__name__)

VERDICTS = {
    "sufficient",
    "vocabulary_mismatch",
    "wrong_granularity",
    "decomposable_residue",
    "corpus_gap",
    "needs_user_clarification",
}

_SNIPPET_CHARS = 280
_TOP_PER_SUB_QUESTION = 5


@dataclass
class EvaluationResult:
    verdict: str
    reason: str = ""
    per_sub_question: List[Dict[str, Any]] = field(default_factory=list)
    reformulated_queries: List[Dict[str, str]] = field(default_factory=list)
    new_sub_questions: List[PlannedSubQuestion] = field(default_factory=list)
    clarification: Optional[ClarificationRequest] = None
    notes: List[ResearchNote] = field(default_factory=list)
    used_llm: bool = False


def guideline_regions(evidence: List[EvidenceItem]) -> List[str]:
    """Distinct guideline regions present in the evidence, in rank order."""
    regions: List[str] = []
    for item in evidence:
        payload = item.payload
        if payload.get("source_type") != "guideline":
            continue
        for key in ("guide_region", "venue", "country"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                region = value.strip().upper()
                if region not in regions:
                    regions.append(region)
                break
    return regions


def region_clarification(
    regions: List[str],
    reason: Optional[str],
    *,
    language: Optional[str] = None,
) -> ClarificationRequest:
    """An evidence-backed country/region clarification, general option included.

    User-facing strings come from the clarifier i18n table, so a Slovene
    question never gets an English clarification just because the evaluator
    (rather than the planner) raised it. Region values stay canonical codes.
    """
    strings = _clarify_i18n("country_or_region", language)
    labels = strings.get("labels", {}) if isinstance(strings, dict) else {}
    options = [
        ClarificationOption(
            label=str(labels.get(region, region)),
            value=region,
        )
        for region in regions[:4]
    ]
    options.append(
        ClarificationOption(
            label=str(labels.get("general", "No preference")),
            value="general",
        )
    )
    return ClarificationRequest(
        id="country_or_region",
        question=str(strings.get("question", "")),
        input_type="single_choice",
        options=options,
        allow_free_text=True,
        reason=reason or str(strings.get("reason", "")) or None,
    )


def _snippet(payload: Dict[str, Any]) -> str:
    text = (
        payload.get("rule_text")
        or payload.get("abstract")
        or payload.get("description")
        or ""
    )
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    return text[:_SNIPPET_CHARS]


def _evidence_digest(state: PipelineState) -> List[Dict[str, Any]]:
    """Per-sub-question view of the best evidence, compact enough to judge."""
    digest = []
    for sq in state.sub_questions():
        items = [
            item for item in state.evidence if sq.id in item.sub_question_ids
        ]
        items.sort(key=lambda i: i.adjusted_score, reverse=True)
        digest.append(
            {
                "id": sq.id,
                "text": sq.text,
                "why": sq.why,
                "qtype": sq.qtype,
                "branch": sq.branch,
                "hits": [
                    {
                        "urn": item.payload.get("urn"),
                        "source_type": item.payload.get("source_type"),
                        "title": item.payload.get("title"),
                        "year": str(item.payload.get("publication_year") or "")[:4]
                        or None,
                        # Enrichment metadata so the judge weighs evidence
                        # QUALITY, not just presence: study design and reach.
                        "study_type": item.payload.get("ai_category"),
                        "citations": item.payload.get("citationCount")
                        or item.payload.get("citation_count"),
                        "score": round(item.adjusted_score, 4),
                        "snippet": _snippet(item.payload),
                    }
                    for item in items[:_TOP_PER_SUB_QUESTION]
                ],
            }
        )
    return digest


def _coerce_notes(raw: Any) -> List[ResearchNote]:
    notes: List[ResearchNote] = []
    if not isinstance(raw, list):
        return notes
    for entry in raw[:8]:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        kind = entry.get("kind")
        if kind not in ("finding", "gap", "lead"):
            kind = "finding"
        source_urns = entry.get("source_urns")
        if not isinstance(source_urns, list):
            source_urns = []
        notes.append(
            ResearchNote(
                text=text,
                kind=kind,
                sub_question_id=(
                    str(entry["sub_question_id"])
                    if entry.get("sub_question_id")
                    else None
                ),
                source_urns=[str(urn) for urn in source_urns if urn][:6],
            )
        )
    return notes


def _coerce_clarification(raw: Any) -> Optional[ClarificationRequest]:
    if not isinstance(raw, dict):
        return None
    try:
        return ClarificationRequest(**raw)
    except Exception:
        return None


async def evaluate(
    state: PipelineState,
    *,
    clarification_allowed: bool,
    max_repair_rounds: int,
) -> EvaluationResult:
    """Judge the current evidence pool; deterministic gates before the LLM."""
    branch_ok = [status.get("ok", False) for status in state.branch_statuses]
    if branch_ok and not any(branch_ok):
        return EvaluationResult(
            verdict="corpus_gap",
            reason="Evidence retrieval is unavailable right now.",
            notes=[
                ResearchNote(
                    text="All retrieval branches failed; answered without corpus evidence.",
                    kind="gap",
                )
            ],
        )

    if state.round >= max_repair_rounds:
        return EvaluationResult(
            verdict="sufficient",
            reason=(
                "Answering with the best available evidence."
                if state.evidence
                else "No further searches available; answering from general knowledge."
            ),
        )

    payload: Dict[str, Any] = {
        "question": state.effective_question,
        "request_language": state.request.language,
        "round": state.round,
        "clarification_allowed": clarification_allowed,
        "user_country_known": bool(state.user_context.country),
        "guideline_regions_found": guideline_regions(state.evidence),
        "sub_questions": _evidence_digest(state),
        "notes_so_far": [
            {"kind": note.kind, "text": note.text} for note in state.notes[-10:]
        ],
    }

    trace = build_trace_config(
        run_name="qa-evaluator",
        session_id=state.request.qa_thread_id,
        user_id=state.request.user_id or state.request.member_id,
        tags=["qa", "pipeline", "evaluator"],
        extra_metadata={"request_id": state.request_id, "round": str(state.round)},
    )

    try:
        llm = GROQ_CHAT.get_client(
            model=config.settings["QA_EVALUATOR_MODEL"], temperature=0.0
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=QA_EVALUATOR_SYSTEM.compile()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            config=trace,
        )
        parsed = parse_json_object(response.content)
    except Exception as exc:
        # The judge must never stand between the user and an answer.
        logger.warning("QA evaluator failed; treating evidence as sufficient: %s", exc)
        return EvaluationResult(
            verdict="sufficient",
            reason="Proceeding with the retrieved evidence.",
        )

    verdict = str(parsed.get("verdict") or "").strip()
    if verdict not in VERDICTS:
        verdict = "sufficient"

    reformulated = []
    for entry in parsed.get("reformulated_queries") or []:
        if isinstance(entry, dict) and entry.get("id"):
            reformulated.append(
                {
                    "id": str(entry["id"]),
                    "lexical_query": str(entry.get("lexical_query") or "").strip(),
                    "dense_query": str(entry.get("dense_query") or "").strip(),
                }
            )

    result = EvaluationResult(
        verdict=verdict,
        reason=str(parsed.get("reason") or "").strip(),
        per_sub_question=[
            entry
            for entry in (parsed.get("per_sub_question") or [])
            if isinstance(entry, dict)
        ],
        reformulated_queries=reformulated,
        new_sub_questions=_coerce_sub_questions(
            parsed.get("new_sub_questions"),
            question=state.effective_question,
            round_added=state.round + 1,
        ),
        clarification=_coerce_clarification(parsed.get("clarification")),
        notes=_coerce_notes(parsed.get("notes")),
        used_llm=True,
    )

    if result.verdict == "needs_user_clarification":
        if not clarification_allowed:
            # The one-clarification-per-thread rule is enforced here, not
            # trusted to the model: downgrade to answering with what we have.
            result.verdict = "sufficient"
            result.clarification = None
        elif result.clarification is None:
            regions = guideline_regions(state.evidence)
            if regions:
                result.clarification = region_clarification(
                    regions,
                    result.reason or None,
                    language=state.request.language,
                )
            else:
                result.verdict = "sufficient"

    return result
