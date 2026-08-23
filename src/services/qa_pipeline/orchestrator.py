"""The agentic QA loop: plan → search → rank → evaluate → repair → answer.

``run_pipeline`` is an async generator of :class:`PipelineEvent`. The SSE
endpoint forwards events as frames; the classic ``/qa/ask`` endpoint drains
the generator and returns the terminal payload. Every event carries the
``request_id`` and a monotonic ``seq``.

Blocking work (Elasticsearch, Redis, the embedder, platform lookups) runs in
worker threads; LLM calls use the async client APIs; sub-question branches
run concurrently.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Set, TYPE_CHECKING

from config import config
from models.qa import (
    ClarificationRequest,
    PlannedSubQuestion,
    QARequest,
    QAResponse,
    ResearchNote,
    SubQuestionFilters,
)
from services.qa_pipeline import ranking, retrieval
from services.qa_pipeline.answering import stream_answer
from services.qa_pipeline.evaluator import EvaluationResult, evaluate
from services.qa_pipeline.events import PipelineEvent
from services.qa_pipeline.repair import build_repair_plan
from services.qa_pipeline.state import EvidenceItem, PipelineState
from services.qa_pipeline.steps import StepTracker

if TYPE_CHECKING:  # pragma: no cover
    from services.qa_service import QAService

logger = logging.getLogger(__name__)

TTL_QA_NOTES = 3600  # matches the conversation-summary horizon
MAX_THREAD_NOTES = 20
CACHE_REPLAY_CHUNK_CHARS = 400

# Strong references to fire-and-forget post-answer tasks (conversation
# summary), so the event loop does not garbage-collect them mid-flight.
_background_tasks: Set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class _Emitter:
    """Stamps request_id and a monotonic seq into every event."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.seq = 0

    def __call__(self, name: str, data: Optional[Dict[str, Any]] = None) -> PipelineEvent:
        payload = dict(data or {})
        payload["request_id"] = self.request_id
        payload["seq"] = self.seq
        self.seq += 1
        return PipelineEvent(name=name, data=payload)


def _notes_cache_key(thread_id: str) -> str:
    return f"qa_notes:{thread_id}"


def _load_prior_notes(service: "QAService", thread_id: Optional[str]) -> List[ResearchNote]:
    if not thread_id:
        return []
    cached = service.cache_manager.get(_notes_cache_key(thread_id))
    if not isinstance(cached, dict):
        return []
    notes = []
    for entry in cached.get("notes", []):
        if isinstance(entry, dict) and entry.get("text"):
            try:
                notes.append(ResearchNote(**entry))
            except Exception:
                continue
    return notes


def _store_notes(
    service: "QAService",
    thread_id: str,
    prior: List[ResearchNote],
    new: List[ResearchNote],
) -> None:
    merged: List[ResearchNote] = []
    seen: Set[str] = set()
    for note in [*prior, *new]:
        key = note.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(note)
    service.cache_manager.set(
        _notes_cache_key(thread_id),
        {"notes": [note.model_dump() for note in merged[-MAX_THREAD_NOTES:]]},
        ttl=TTL_QA_NOTES,
    )


def _repair_action_phrase(action: Dict[str, Any]) -> str:
    """A repair action as a sentence a reader understands, not an enum leak."""
    kind = action.get("action")
    value = str(action.get("new_query") or "").strip()
    if kind == "reformulate":
        return f'Reworded a search: "{value}"' if value else "Reworded a search"
    if kind == "switch_branch":
        target = "guidelines" if value == "guidelines" else "the literature"
        return f"Switched a search to {target}"
    if kind == "add_sub_question":
        return f'Added a search: "{value}"' if value else "Added a search"
    return str(kind or "Adjusted a search")


def _merged_facets(
    sub_questions: List[PlannedSubQuestion],
) -> SubQuestionFilters:
    """The union of every sub-question's attribute constraints, for ranking."""
    merged = SubQuestionFilters()
    for sq in sub_questions:
        f = sq.filters
        for field in (
            "study_types",
            "regions",
            "target_populations",
            "food_groups",
            "nutrients",
            "health_conditions",
        ):
            for term in getattr(f, field):
                if term not in getattr(merged, field):
                    getattr(merged, field).append(term)
    return merged


def _expand_branches(sub_questions: List[PlannedSubQuestion]) -> List[tuple]:
    tasks = []
    for sq in sub_questions:
        branches = ["articles", "guidelines"] if sq.branch == "both" else [sq.branch]
        for branch in branches:
            tasks.append((sq, branch))
    return tasks


def _sq_view(sq: PlannedSubQuestion) -> Dict[str, Any]:
    view = {
        "id": sq.id,
        "text": sq.text,
        "why": sq.why,
        "qtype": sq.qtype,
        "branch": sq.branch,
    }
    if not sq.filters.is_empty():
        view["filters"] = sq.filters.model_dump(exclude_none=True)
    return view


def _top_view(items: List[EvidenceItem], limit: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "urn": item.payload.get("urn"),
            "title": item.payload.get("title"),
            "source_type": item.payload.get("source_type"),
            "year": str(item.payload.get("publication_year") or "")[:4] or None,
            "adjusted_score": round(item.adjusted_score, 4),
            "score_parts": item.score_parts,
        }
        for item in items[:limit]
    ]


async def run_pipeline(
    service: "QAService",
    request: QARequest,
) -> AsyncIterator[PipelineEvent]:
    """Run the agentic QA pipeline, yielding events; terminal event ends it."""
    from services.qa_service import _request_trace_context

    request_id = str(uuid.uuid4())
    emit = _Emitter(request_id)
    steps = StepTracker()
    started = time.monotonic()

    service._validate_request(request)
    effective_model = service._resolve_model(request)
    effective_retriever = service._resolve_retriever(request)
    effective_rag = effective_retriever != "no_rag"

    yield emit(
        "stage.start",
        {
            "question": request.question,
            "mode": request.mode,
            "model": effective_model,
            "retriever": effective_retriever,
        },
    )

    user_context = await asyncio.to_thread(service._resolve_user_context, request)
    thread_context = await asyncio.to_thread(
        service._load_qa_thread, request.qa_thread_id
    )
    user_context = service._merge_thread_user_context(user_context, thread_context)
    conversation_summary = await asyncio.to_thread(
        service._load_conversation_summary, request.qa_thread_id
    )
    prior_notes = await asyncio.to_thread(
        _load_prior_notes, service, request.qa_thread_id
    )

    # A pending clarification the client did not answer is re-served verbatim.
    if thread_context and request.clarification_response is None:
        clarification = ClarificationRequest(**thread_context["clarification"])
        response = service._build_clarification_response(
            request=request,
            request_id=request_id,
            thread_id=thread_context["thread_id"],
            clarification=clarification,
            model_used=effective_model,
            user_context=user_context,
        )
        yield emit(
            "step",
            steps.add(
                "clarification",
                "Waiting for one detail",
                detail=clarification.question,
            ).model_dump(),
        )
        response.reasoning_steps = steps.snapshot()
        yield emit("clarification", response.model_dump())
        return

    effective_question = service._compose_effective_question(
        request=request, thread_context=thread_context
    )
    user_context = service._apply_clarification_to_user_context(
        user_context, request.clarification_response
    )
    answered_ids = service._answered_clarification_ids(thread_context, request)

    # ---- Plan -------------------------------------------------------------
    from agents.qa_planner import QAPlannerAgent, build_fallback_pipeline_plan

    plan_step = steps.start("plan", "Planning the research")
    yield emit("step", plan_step.model_dump())

    plan_started = time.monotonic()
    try:
        plan = await QAPlannerAgent().plan(
            question=effective_question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
            prior_notes=prior_notes,
        )
    except Exception as exc:  # pragma: no cover - planner already guards
        logger.warning("Planner crashed; deterministic fallback: %s", exc)
        plan = build_fallback_pipeline_plan(
            question=effective_question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
        )

    state = PipelineState(
        request=request,
        request_id=request_id,
        effective_question=effective_question,
        user_context=user_context,
        effective_model=effective_model,
        effective_retriever=effective_retriever,
        plan=plan,
    )
    state.prior_notes = prior_notes
    state.timings_ms["plan"] = int((time.monotonic() - plan_started) * 1000)

    yield emit(
        "stage.plan",
        {
            "canonical_question": plan.canonical_question,
            "risk_level": plan.risk_level,
            "safety_flags": plan.safety_flags,
            "sub_questions": [_sq_view(sq) for sq in plan.sub_questions],
            "prior_notes": [
                {"kind": n.kind, "text": n.text} for n in prior_notes
            ],
        },
    )
    search_count = len(plan.sub_questions)
    yield emit(
        "step",
        steps.finish(
            plan_step.id,
            title=(
                "Planned 1 search"
                if search_count == 1
                else f"Planned {search_count} searches"
            ),
            detail="; ".join(
                sq.why or sq.text for sq in plan.sub_questions[:4]
            ) or None,
            data={"sub_questions": [_sq_view(sq) for sq in plan.sub_questions]},
        ).model_dump(),
    )

    # ONE clarification per thread (see legacy rationale): once the user has
    # answered or declined anything, answer with reasonable assumptions.
    clarification_exhausted = bool(request.clarification_response) or bool(
        answered_ids
    )
    clarification = (
        plan.clarification
        if plan.needs_clarification and not clarification_exhausted
        else None
    )
    ask_before_search = clarification is not None and (
        service._should_ask_clarification_before_scout(
            clarification, effective_rag=effective_rag
        )
    )
    if clarification and ask_before_search:
        thread_id = request.qa_thread_id or str(uuid.uuid4())
        await asyncio.to_thread(
            service._store_qa_thread,
            thread_id=thread_id,
            question=effective_question,
            request=request,
            clarification=clarification,
            user_context=user_context,
            answered_ids=answered_ids,
        )
        response = service._build_clarification_response(
            request=request,
            request_id=request_id,
            thread_id=thread_id,
            clarification=clarification,
            model_used=effective_model,
            user_context=user_context,
        )
        yield emit(
            "step",
            steps.add(
                "clarification",
                "Asking for one detail first",
                detail=clarification.question,
            ).model_dump(),
        )
        response.reasoning_steps = steps.snapshot()
        yield emit("clarification", response.model_dump())
        return

    # ---- Cache ------------------------------------------------------------
    cache_key = service._build_cache_key(
        request,
        question=effective_question,
        effective_retriever=effective_retriever,
        user_context=user_context,
    )
    if clarification is None:
        cached = await asyncio.to_thread(service.cache_manager.get, cache_key)
        if cached:
            cached["cache_hit"] = True
            cached["request_id"] = request_id
            if request.qa_thread_id and request.clarification_response:
                await asyncio.to_thread(
                    service._clear_qa_thread, request.qa_thread_id
                )
            yield emit("stage.cache", {"hit": True})
            yield emit(
                "step",
                steps.add(
                    "cache",
                    "Answer retrieved from cache",
                    detail="This question was answered recently; replaying it.",
                ).model_dump(),
            )
            answer_payload = cached.get("primary_answer") or {}
            answer_text = str(answer_payload.get("answer") or "")
            yield emit("answer_started", {"model": cached.get("primary_answer", {}).get("model_used", effective_model), "cache_hit": True})
            for offset in range(0, len(answer_text), CACHE_REPLAY_CHUNK_CHARS):
                yield emit(
                    "answer_delta",
                    {"text": answer_text[offset: offset + CACHE_REPLAY_CHUNK_CHARS]},
                )
            yield emit(
                "citations",
                {
                    "citations": answer_payload.get("citations", []),
                    "confidence": answer_payload.get("confidence", "medium"),
                    "follow_up_suggestions": cached.get("follow_up_suggestions"),
                },
            )
            yield emit("done", cached)
            return

    # ---- Retrieval rounds ---------------------------------------------------
    max_repair_rounds = max(int(config.settings.get("QA_MAX_REPAIR_ROUNDS", 1)), 0)
    max_total_sub_questions = (
        max(int(config.settings.get("QA_MAX_SUBQUESTIONS", 3)), 1) + 2
    )
    evaluation: Optional[EvaluationResult] = None
    pool: List[EvidenceItem] = []

    if effective_rag and effective_retriever == "rag":
        to_search = list(plan.sub_questions)
        for round_index in range(max_repair_rounds + 1):
            state.round = round_index
            retrieve_started = time.monotonic()

            branch_tasks = _expand_branches(to_search)
            dense_queries = [sq.dense_query or sq.text for sq, _ in branch_tasks]
            vectors: List[Optional[List[float]]] = [None] * len(branch_tasks)
            if dense_queries:
                try:
                    embedded = await asyncio.to_thread(
                        service.embedder.encode,
                        dense_queries,
                        normalize_embeddings=True,
                    )
                    vectors = [vector.tolist() for vector in embedded]
                except Exception as exc:
                    logger.warning(
                        "Query embedding failed; lexical-only round: %s", exc
                    )

            search_steps: Dict[tuple, str] = {}
            for sq, branch in branch_tasks:
                yield emit(
                    "stage.search_started",
                    {
                        "sub_question_id": sq.id,
                        "branch": branch,
                        "why": sq.why,
                        "lexical_query": sq.lexical_query or sq.text,
                        "round": round_index,
                    },
                )
                branch_label = (
                    "guidelines" if branch == "guidelines" else "the literature"
                )
                step = steps.start(
                    "search",
                    f'Searching {branch_label}: "{sq.lexical_query or sq.text}"',
                    detail=sq.why or None,
                    data={
                        "sub_question_id": sq.id,
                        "branch": branch,
                        "lexical_query": sq.lexical_query or sq.text,
                    },
                    round=round_index,
                )
                search_steps[(sq.id, branch)] = step.id
                yield emit("step", step.model_dump())

            outcomes = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        retrieval.run_branch,
                        sq,
                        branch=branch,
                        vector=vectors[index],
                        user_context=user_context,
                        expertise_level=request.expertise_level,
                    )
                    for index, (sq, branch) in enumerate(branch_tasks)
                ],
                return_exceptions=True,
            )

            new_items: List[EvidenceItem] = []
            for (sq, branch), outcome in zip(branch_tasks, outcomes):
                if isinstance(outcome, BaseException):
                    logger.error(
                        "Branch %s/%s failed: %r", sq.id, branch, outcome
                    )
                    status = {"ok": False, "error": repr(outcome)}
                    hits: List[EvidenceItem] = []
                else:
                    status = outcome.status
                    hits = outcome.items
                state.branch_statuses.append(
                    {"sub_question_id": sq.id, "branch": branch, **status}
                )
                new_items.extend(hits)
                top_titles = [
                    {
                        "title": item.payload.get("title"),
                        "year": str(
                            item.payload.get("publication_year") or ""
                        )[:4]
                        or None,
                        "source_type": item.payload.get("source_type"),
                    }
                    for item in hits[:3]
                ]
                yield emit(
                    "stage.search_results",
                    {
                        "sub_question_id": sq.id,
                        "branch": branch,
                        "hit_count": len(hits),
                        "ok": status.get("ok", False),
                        "top": top_titles,
                        "round": round_index,
                    },
                )
                step_id = search_steps.get((sq.id, branch))
                if step_id:
                    if not status.get("ok", False):
                        outcome_title = "Search unavailable"
                    elif not hits:
                        outcome_title = "No sources found"
                    elif len(hits) == 1:
                        outcome_title = "Found 1 source"
                    else:
                        outcome_title = f"Found {len(hits)} sources"
                    query_text = (sq.lexical_query or sq.text)
                    yield emit(
                        "step",
                        steps.finish(
                            step_id,
                            title=f'{outcome_title} — "{query_text}"',
                            data={"hit_count": len(hits), "top": top_titles},
                        ).model_dump(),
                    )

            pool = retrieval.merge_evidence(pool, new_items)
            state.timings_ms["retrieve"] = state.timings_ms.get(
                "retrieve", 0
            ) + int((time.monotonic() - retrieve_started) * 1000)

            # ---- Rank -------------------------------------------------------
            rank_started = time.monotonic()
            adjusted = ranking.adjust_evidence(
                list(pool),
                expertise_level=request.expertise_level,
                user_context=user_context,
                question_facets=_merged_facets(state.sub_questions()),
            )
            selected, dropped = ranking.select_evidence(
                adjusted, top_k=request.top_k
            )
            state.evidence = selected
            state.timings_ms["rank"] = state.timings_ms.get("rank", 0) + int(
                (time.monotonic() - rank_started) * 1000
            )
            yield emit(
                "stage.rerank",
                {
                    "kept": len(selected),
                    "dropped": dropped,
                    "top": _top_view(selected, limit=min(request.top_k, 5)),
                    "round": round_index,
                },
            )
            yield emit(
                "step",
                steps.add(
                    "rank",
                    f"Prioritized evidence: kept {len(selected)} of {len(pool)}",
                    detail=(
                        "Ranked by relevance, recency, citation influence, "
                        "study design, and editorial tier."
                    ),
                    data={
                        "kept": len(selected),
                        "pool": len(pool),
                        "dropped": dropped,
                        "top": _top_view(selected, limit=3),
                    },
                    round=round_index,
                ).model_dump(),
            )

            # ---- Evaluate ----------------------------------------------------
            evaluate_started = time.monotonic()
            clarification_allowed = (
                not clarification_exhausted and not state.user_context.country
            )
            evaluation = await evaluate(
                state,
                clarification_allowed=clarification_allowed,
                max_repair_rounds=max_repair_rounds,
            )
            state.timings_ms["evaluate"] = state.timings_ms.get(
                "evaluate", 0
            ) + int((time.monotonic() - evaluate_started) * 1000)
            state.verdicts.append(
                {
                    "round": round_index,
                    "verdict": evaluation.verdict,
                    "reason": evaluation.reason,
                    "used_llm": evaluation.used_llm,
                }
            )

            added_notes = state.add_notes(evaluation.notes)
            if added_notes:
                yield emit(
                    "stage.notes",
                    {
                        "notes": [note.model_dump() for note in added_notes],
                        "round": round_index,
                    },
                )
                yield emit(
                    "step",
                    steps.add(
                        "notes",
                        (
                            "Kept 1 research note"
                            if len(added_notes) == 1
                            else f"Kept {len(added_notes)} research notes"
                        ),
                        detail=added_notes[0].text,
                        data={
                            "notes": [note.model_dump() for note in added_notes]
                        },
                        round=round_index,
                    ).model_dump(),
                )

            yield emit(
                "stage.evaluate",
                {
                    "round": round_index,
                    "verdict": evaluation.verdict,
                    "reason": evaluation.reason,
                    "gaps": [
                        entry.get("gap")
                        for entry in evaluation.per_sub_question
                        if entry.get("gap")
                    ],
                },
            )
            verdict_titles = {
                "sufficient": "Evidence check: enough to answer",
                "vocabulary_mismatch": "Evidence check: rewording a search",
                "wrong_granularity": "Evidence check: switching source type",
                "decomposable_residue": "Evidence check: one more angle to search",
                "corpus_gap": "Evidence check: the corpus has a gap",
                "needs_user_clarification": "Evidence check: need one detail from you",
            }
            yield emit(
                "step",
                steps.add(
                    "evaluate",
                    verdict_titles.get(evaluation.verdict, "Evidence check"),
                    detail=evaluation.reason or None,
                    data={"verdict": evaluation.verdict},
                    round=round_index,
                ).model_dump(),
            )

            if evaluation.verdict == "needs_user_clarification" and (
                evaluation.clarification is not None
            ):
                thread_id = request.qa_thread_id or str(uuid.uuid4())
                await asyncio.to_thread(
                    service._store_qa_thread,
                    thread_id=thread_id,
                    question=effective_question,
                    request=request,
                    clarification=evaluation.clarification,
                    user_context=user_context,
                    answered_ids=answered_ids,
                )
                if state.notes:
                    await asyncio.to_thread(
                        _store_notes, service, thread_id, prior_notes, state.notes
                    )
                response = service._build_clarification_response(
                    request=request,
                    request_id=request_id,
                    thread_id=thread_id,
                    clarification=evaluation.clarification,
                    model_used=effective_model,
                    user_context=user_context,
                )
                yield emit(
                    "step",
                    steps.add(
                        "clarification",
                        "Asking for one detail",
                        detail=evaluation.clarification.question,
                        round=round_index,
                    ).model_dump(),
                )
                response.reasoning_steps = steps.snapshot()
                yield emit("clarification", response.model_dump())
                return

            if evaluation.verdict in ("sufficient", "corpus_gap"):
                break

            repair_plan = build_repair_plan(
                state,
                evaluation,
                max_total_sub_questions=max_total_sub_questions,
            )
            if not repair_plan.has_work:
                break
            state.repairs.append(
                {"round": round_index, "actions": repair_plan.actions}
            )
            yield emit(
                "stage.repair",
                {"round": round_index, "actions": repair_plan.actions},
            )
            yield emit(
                "step",
                steps.add(
                    "repair",
                    "Refining the search",
                    detail="; ".join(
                        _repair_action_phrase(action)
                        for action in repair_plan.actions[:3]
                    )
                    or None,
                    data={"actions": repair_plan.actions},
                    round=round_index,
                ).model_dump(),
            )
            to_search = repair_plan.to_search

    elif effective_rag and effective_retriever == "linearrag":
        # Advanced/debug retriever: single pass through the legacy adapter,
        # then the same ranking and answer stages.
        retrieve_started = time.monotonic()
        yield emit(
            "stage.search_started",
            {
                "sub_question_id": "sq1",
                "branch": "linearrag",
                "why": "Graph-based passage retrieval over the article corpus.",
                "lexical_query": plan.article_query,
                "round": 0,
            },
        )
        linearrag_step = steps.start(
            "search",
            f'Searching the knowledge graph: "{plan.article_query}"',
            detail="Graph-based passage retrieval over the article corpus.",
            data={"branch": "linearrag"},
        )
        yield emit("step", linearrag_step.model_dump())
        result = await asyncio.to_thread(
            service._retrieve_sources,
            question=effective_question,
            plan=plan,
            top_k=request.top_k,
            retriever=effective_retriever,
            user_context=user_context,
            expertise_level=request.expertise_level,
        )
        state.branch_statuses.append(
            {"sub_question_id": "sq1", "branch": "linearrag", **result.status}
        )
        max_score = max(
            [p.get("_score", 0.0) or 0.0 for p in result.source_payloads] or [1.0]
        ) or 1.0
        pool = []
        for payload, source in zip(
            result.source_payloads, result.retrieved_sources
        ):
            item = EvidenceItem(payload=payload, source=source)
            item.rrf_norm = float(payload.get("_score", 0.0) or 0.0) / max_score
            item.sub_question_ids = ["sq1"]
            pool.append(item)
        adjusted = ranking.adjust_evidence(
            pool,
            expertise_level=request.expertise_level,
            user_context=user_context,
        )
        selected, dropped = ranking.select_evidence(adjusted, top_k=request.top_k)
        state.evidence = selected
        state.timings_ms["retrieve"] = int(
            (time.monotonic() - retrieve_started) * 1000
        )
        yield emit(
            "stage.search_results",
            {
                "sub_question_id": "sq1",
                "branch": "linearrag",
                "hit_count": len(selected),
                "ok": result.status.get("ok", False),
                "top": [],
                "round": 0,
            },
        )
        yield emit(
            "stage.rerank",
            {
                "kept": len(selected),
                "dropped": dropped,
                "top": _top_view(selected, limit=min(request.top_k, 5)),
                "round": 0,
            },
        )
        yield emit(
            "step",
            steps.finish(
                linearrag_step.id,
                title=f"Found {len(selected)} sources in the knowledge graph",
                data={"hit_count": len(selected)},
            ).model_dump(),
        )

    # ---- Answer -------------------------------------------------------------
    trace_context = _request_trace_context(request)
    context_payload = user_context.model_dump(exclude_none=True)
    context_payload["safety"] = {
        "risk_level": plan.risk_level,
        "flags": plan.safety_flags,
        "guardrails": plan.answer_guardrails,
    }
    context_payload["retrieval_reasoning"] = {
        "sub_questions": [_sq_view(sq) for sq in state.sub_questions()],
        "verdict": evaluation.verdict if evaluation else None,
        "gaps": [
            note.text for note in state.notes if note.kind == "gap"
        ][:4],
    }

    payloads = state.evidence_payloads() if effective_rag else []
    yield emit("answer_started", {"model": effective_model})
    answer_step = steps.start(
        "answer",
        "Writing the answer",
        detail=(
            f"Grounding on {len(payloads)} sources."
            if payloads
            else "Answering from general knowledge — no sources to cite."
        ),
        data={"model": effective_model, "sources": len(payloads)},
    )
    yield emit("step", answer_step.model_dump())

    answer_started = time.monotonic()
    final_answer = None
    follow_ups: List[str] = []
    async for event in stream_answer(
        question=effective_question,
        payloads=payloads,
        expertise_level=request.expertise_level,
        language=request.language,
        model=effective_model,
        user_context=context_payload,
        prior_conversation=conversation_summary,
        trace_context=trace_context,
    ):
        if event["kind"] == "delta":
            yield emit("answer_delta", {"text": event["text"]})
        elif event["kind"] == "final":
            final_answer = event["answer"]
            follow_ups = event["follow_ups"]
    state.timings_ms["answer"] = int((time.monotonic() - answer_started) * 1000)
    state.timings_ms["total"] = int((time.monotonic() - started) * 1000)

    yield emit(
        "citations",
        {
            "citations": [c.model_dump() for c in final_answer.citations],
            "confidence": final_answer.confidence,
            "follow_up_suggestions": follow_ups or None,
        },
    )
    yield emit(
        "step",
        steps.finish(
            answer_step.id,
            title="Answer written",
            detail=(
                f"{len(final_answer.citations)} citations"
                if final_answer.citations
                else None
            ),
            data={"citations": len(final_answer.citations)},
        ).model_dump(),
    )

    conversation_thread_id = request.qa_thread_id or str(uuid.uuid4())

    response = QAResponse(
        question=effective_question,
        mode=request.mode,
        primary_answer=final_answer,
        secondary_answer=None,
        dual_answer_feedback=None,
        retrieved_sources=state.retrieved_sources() if effective_rag else [],
        follow_up_suggestions=follow_ups or None,
        generated_at=datetime.now().isoformat(),
        cache_hit=False,
        request_id=request_id,
        qa_thread_id=conversation_thread_id,
        needs_clarification=False,
        clarification=None,
        user_context=user_context,
        reasoning_steps=steps.snapshot(),
    )

    # Memory nudges ride the done payload; best-effort, never blocks.
    if request.member_id:
        try:
            from services.memory_service import MEMORY_SERVICE
            from models.qa import MemorySuggestion

            suggestions = await asyncio.to_thread(
                MEMORY_SERVICE.suggest, request.member_id, request.question
            )
            if suggestions:
                response.memory_suggestions = [
                    MemorySuggestion(**s) for s in suggestions
                ]
        except Exception as exc:
            logger.warning("Memory suggestion skipped: %s", exc)

    # Research notes persist on the thread so a follow-up question's planner
    # starts from what this run already found.
    if state.notes:
        await asyncio.to_thread(
            _store_notes, service, conversation_thread_id, prior_notes, state.notes
        )

    from services.qa_service import TTL_QA_RESPONSE

    await asyncio.to_thread(
        service.cache_manager.set, cache_key, response.model_dump(), TTL_QA_RESPONSE
    )

    _spawn_background(
        asyncio.to_thread(
            service._update_conversation_summary,
            thread_id=conversation_thread_id,
            previous_summary=conversation_summary,
            question=effective_question,
            answer_text=final_answer.answer or "",
            language=request.language,
            trace_context=trace_context,
        )
    )

    persist_request = request.model_copy(
        update={
            "question": effective_question,
            "retriever": effective_retriever,
        }
    )
    await service._persist_request(
        persist_request,
        response,
        effective_model,
        effective_rag,
        None,
        pipeline_meta=state.pipeline_meta(),
    )
    if request.qa_thread_id and request.clarification_response:
        await asyncio.to_thread(service._clear_qa_thread, request.qa_thread_id)

    yield emit("done", response.model_dump())
