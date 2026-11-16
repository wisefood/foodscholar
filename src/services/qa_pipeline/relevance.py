"""Relevance grading: does this source answer THIS search?

Fused search scores only say a source shares words or meaning with the query,
and the ranking priors (tier, recency, citations, study design) know nothing
about the question. Live (oat milk and calcium): the pool held four surveys
of plant-milk calcium content, ranked 12th to 25th, while a paper comparing
human milk with other mammals' milk, an IBD lactose paper and a calcium
absorption trial took the five places the answer reads. The evidence check
then saw no oat-milk calcium data and called a corpus gap.

So before the cut, a fast model grades the head of each search's candidates
0-3 against that search's own question, and the grade weighs on the score:
directly answering sources rise, off-topic ones sink, and the priors still
order sources of the same grade. One call per search, run in parallel; a
failure leaves the ranking as it was. ``QA_RERANK_ENABLED`` switches it off.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import QA_RELEVANCE_SYSTEM
from config import config
from services.qa_pipeline import facets as corpus_facets
from services.qa_pipeline.audience import age_of, describe_in_words
from services.qa_pipeline.state import EvidenceItem, PipelineState

logger = logging.getLogger(__name__)

# Grade -> score multiplier. 3 must beat a 2 that has every prior in its
# favour (tier 1.6 x design 1.3 is about 2x), so the gap is wide on purpose.
RELEVANCE_FACTORS = {3: 2.0, 2: 1.0, 1: 0.5, 0: 0.15}
# Candidates beyond the graded head were not looked at. They must not take
# the places of sources that were (seen: every graded article off-topic, and
# five unread ones filled the answer's slots), so they rank with the 0s; a
# search whose whole head is off-topic then contributes nothing, which is
# the honest outcome.
UNGRADED_FACTOR = RELEVANCE_FACTORS[0]
_TEXT_CHARS = 450
_GRADE = re.compile(r'"(s\d+)"\s*:\s*"?([0-3])\b')


def _enabled() -> bool:
    return str(config.settings.get("QA_RERANK_ENABLED", True)).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _head_size() -> int:
    try:
        return max(int(config.settings.get("QA_RERANK_CANDIDATES", 30)), 1)
    except (TypeError, ValueError):
        return 30


def _source_text(payload: Dict[str, Any]) -> str:
    parts = [
        payload.get("rule_text") or payload.get("abstract") or payload.get("description") or "",
    ]
    takeaways = payload.get("ai_key_takeaways")
    if isinstance(takeaways, list):
        parts.append(" ".join(str(t) for t in takeaways[:3]))
    elif isinstance(takeaways, str):
        parts.append(takeaways)
    text = " ".join(" ".join(str(p) for p in parts if p).split())
    return text[:_TEXT_CHARS]


def _guide(payload: Dict[str, Any]) -> Optional[str]:
    """The guide a rule comes from, in the guide's own words."""
    title = str(payload.get("guide_title") or "").strip()
    description = " ".join(str(payload.get("guide_description") or "").split())[:160]
    return " — ".join(p for p in (title, description) if p) or None


def _source_tags(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the source says about itself: its tags and its age window."""
    tags = corpus_facets.tags(payload)
    age = age_of(payload)
    if age and "age_group" not in tags:
        tags["age"] = age
    return tags or None


def _candidates(state: PipelineState, items: List[EvidenceItem]) -> List[Tuple[Any, List[EvidenceItem]]]:
    """Each search's head of candidates still to grade."""
    head = _head_size()
    out = []
    for sq in state.sub_questions():
        mine = [i for i in items if sq.id in i.sub_question_ids]
        mine.sort(key=lambda i: i.adjusted_score, reverse=True)
        todo = [i for i in mine[:head] if (sq.id, i.key) not in state.relevance]
        if todo:
            out.append((sq, todo))
    return out


async def _grade_one(state: PipelineState, sq: Any, items: List[EvidenceItem]) -> Dict[str, int]:
    labels = {f"s{n}": item for n, item in enumerate(items, start=1)}
    payload = {
        "question": state.effective_question,
        "search": sq.text,
        "looking_for": sq.qtype,
        "asked_about": describe_in_words(state.audience),
        # What the plan said a source should be tagged with for this search.
        "wanted": corpus_facets.describe(sq.filters),
        "sources": [
            {
                "id": label,
                "type": item.payload.get("source_type"),
                "title": item.payload.get("title") or item.payload.get("guide_title"),
                "year": str(item.payload.get("publication_year") or "")[:4] or None,
                "guide": _guide(item.payload),
                "tags": _source_tags(item.payload),
                "text": _source_text(item.payload),
            }
            for label, item in labels.items()
        ],
    }
    trace = build_trace_config(
        run_name="qa-relevance",
        session_id=state.request.qa_thread_id,
        user_id=state.request.user_id or state.request.member_id,
        tags=["qa", "pipeline", "relevance"],
        extra_metadata={"request_id": state.request_id, "round": str(state.round)},
    )
    # The grade decides what the answer reads, so it is made by the model
    # that writes the answer unless QA_RERANK_MODEL names another: on live
    # questions the small model graded on-topic rules a step lower, and a
    # step is a factor of two here.
    model = (
        config.settings.get("QA_RERANK_MODEL")
        or config.settings.get("QA_DEFAULT_MODEL")
        or config.settings["QA_EVALUATOR_MODEL"]
    )
    llm = GROQ_CHAT.get_client(model=model, temperature=0.0)
    messages = [
        SystemMessage(content=QA_RELEVANCE_SYSTEM.compile()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    # Read leniently: the small model sometimes drops a comma between grades
    # ('"s4":0"s5":2'), the same way every time at temperature 0, so strict
    # JSON failed the whole pool live. Every "sN": digit pair is a grade.
    response = await llm.ainvoke(messages, config=trace)
    grades = {label: int(value) for label, value in _GRADE.findall(str(response.content or ""))}
    if not grades:
        raise ValueError(f"no grades in model output: {str(response.content)[:200]!r}")
    out: Dict[str, int] = {}
    for label, item in labels.items():
        try:
            grade = int(grades.get(label))
        except (TypeError, ValueError):
            continue
        out[item.key] = min(max(grade, 0), 3)
    return out


async def grade_relevance(state: PipelineState, items: List[EvidenceItem]) -> None:
    """Grade each search's head of candidates and weigh the grade into the score.

    Grades are kept on the state, so a repair round grades only what is new.
    """
    if not items or not _enabled():
        return
    work = _candidates(state, items)
    if work:
        results = await asyncio.gather(
            *(_grade_one(state, sq, todo) for sq, todo in work), return_exceptions=True
        )
        for (sq, _todo), result in zip(work, results):
            if isinstance(result, Exception):
                logger.warning("Relevance grading failed for %s; ranking unchanged: %s", sq.id, result)
                continue
            for key, grade in result.items():
                state.relevance[(sq.id, key)] = grade

    for item in items:
        grades = [g for (sq_id, key), g in state.relevance.items() if key == item.key and sq_id in item.sub_question_ids]
        if not grades:
            continue
        grade = max(grades)
        item.adjusted_score *= RELEVANCE_FACTORS[grade]
        item.score_parts["relevance"] = grade
        item.payload["adjusted_score"] = item.adjusted_score
        item.payload["score_parts"] = item.score_parts
    # A source no search looked at ranks with the 0s (only when something
    # was graded: a failed call leaves every score as it was).
    if not any("relevance" in item.score_parts for item in items):
        return
    for item in items:
        if "relevance" not in item.score_parts:
            item.adjusted_score *= UNGRADED_FACTOR
            item.payload["adjusted_score"] = item.adjusted_score
