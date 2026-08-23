"""Token-streamed answer generation with a citation trailer.

The model writes the markdown answer first (streamed to the user as deltas),
then a sentinel line, then a JSON trailer with verbatim citations. Citation
building reuses the same machinery as the legacy path (quote coercion to an
exact source span, G-labels), so streamed answers carry identical citations.

Defensive paths:
- A model that ignores the protocol and emits a JSON object anyway is detected
  by its first character and buffered instead of streamed, then parsed as the
  legacy JSON answer.
- A missing/unparseable trailer falls back to recovering citations from the
  inline markdown links in the streamed answer.
- A mid-stream failure still produces an answer from whatever was streamed.
"""
from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate

from agents.json_output import parse_json_object
from agents.qa_agent import (
    COMPLEXITY_INSTRUCTIONS,
    build_qa_answer,
    format_answer_context,
    format_prior_conversation,
    prepare_source_context,
)
from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import QA_ANSWER_STREAM_SYSTEM, QA_ANSWER_STREAM_USER
from models.qa import QAAnswer

logger = logging.getLogger(__name__)

ANSWER_SENTINEL = "<<<END_ANSWER>>>"

_INLINE_CITATION_RE = re.compile(r"\]\(/(articles|guidelines)/([^)\s]+)\)")


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def citations_from_inline_links(
    answer_text: str, payloads: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Recover cited_sources entries from the answer's markdown links."""
    known_urns = set()
    for payload in payloads:
        for key in ("urn", "id", "_id", "guide_urn"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                known_urns.add(value.strip())

    cited: List[Dict[str, Any]] = []
    seen = set()
    for match in _INLINE_CITATION_RE.finditer(answer_text):
        urn = match.group(2).strip()
        if urn in seen or urn not in known_urns:
            continue
        seen.add(urn)
        cited.append({"urn": urn, "confidence": "medium"})
    return cited


def _error_answer(model: str, partial_text: str, rag_used: bool) -> QAAnswer:
    text = partial_text.strip() or (
        "Unable to generate an answer at this time. Please try again."
    )
    return QAAnswer(
        answer=text,
        citations=[],
        confidence="low",
        model_used=model,
        rag_used=rag_used,
        sources_consulted=0,
        articles_consulted=0,
    )


async def stream_answer(
    *,
    question: str,
    payloads: List[Dict[str, Any]],
    expertise_level: str,
    language: str,
    model: str,
    temperature: float = 0.3,
    user_context: Optional[Dict[str, Any]] = None,
    prior_conversation: Optional[str] = None,
    trace_context: Optional[Dict[str, Optional[str]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream the answer; yields delta events, then one final event.

    Yields ``{"kind": "delta", "text": str}`` for each streamed answer chunk
    and finally ``{"kind": "final", "answer": QAAnswer, "follow_ups": [...],
    "parsed_trailer": bool}``.
    """
    rag_used = bool(payloads)
    retriever_shape = "rag" if rag_used else "no_rag"
    source_context = (
        prepare_source_context(payloads, retriever="rag")
        if payloads
        else "(no sources retrieved — answer from general knowledge, no citations)"
    )
    variables = {
        "expertise_level": expertise_level,
        "complexity": COMPLEXITY_INSTRUCTIONS.get(
            expertise_level, COMPLEXITY_INSTRUCTIONS["intermediate"]
        ),
        "language": language,
        "answer_context": format_answer_context(
            retriever=retriever_shape, user_context=user_context
        ),
        "prior_conversation": format_prior_conversation(prior_conversation),
        "question": question,
        "source_context": source_context,
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_ANSWER_STREAM_SYSTEM.langchain()),
            ("human", QA_ANSWER_STREAM_USER.langchain()),
        ]
    )
    messages = prompt.format_messages(**variables)

    trace_context = trace_context or {}
    trace = build_trace_config(
        run_name="qa-answer-stream",
        session_id=trace_context.get("session_id"),
        user_id=trace_context.get("user_id"),
        tags=["qa", "pipeline", "answer"],
    )

    llm = GROQ_CHAT.get_client(model=model, temperature=temperature)

    emitted = ""  # answer text already yielded as deltas
    pending = ""  # answer text held back (sentinel window / JSON detection)
    trailer = ""
    in_trailer = False
    json_mode: Optional[bool] = None  # None until the first non-space char
    failed = False

    try:
        async for chunk in llm.astream(messages, config=trace):
            text = _chunk_text(chunk)
            if not text:
                continue

            if in_trailer:
                trailer += text
                continue

            pending += text

            if json_mode is None:
                stripped = pending.lstrip()
                if not stripped:
                    continue
                # A model that ignored the protocol and answered in raw JSON:
                # buffer silently and parse at the end instead of streaming
                # JSON syntax at the user.
                json_mode = stripped.startswith("{")

            if json_mode:
                continue

            sentinel_at = pending.find(ANSWER_SENTINEL)
            if sentinel_at >= 0:
                to_emit = pending[:sentinel_at]
                trailer = pending[sentinel_at + len(ANSWER_SENTINEL):]
                pending = ""
                in_trailer = True
                if to_emit:
                    emitted += to_emit
                    yield {"kind": "delta", "text": to_emit}
                continue

            # Hold back a sentinel-sized tail so a sentinel split across
            # chunks is never emitted to the user.
            holdback = len(ANSWER_SENTINEL) - 1
            if len(pending) > holdback:
                to_emit = pending[:-holdback]
                pending = pending[-holdback:]
                emitted += to_emit
                yield {"kind": "delta", "text": to_emit}
    except Exception as exc:
        logger.error("Streaming answer failed: %s", exc, exc_info=True)
        failed = True

    parsed: Dict[str, Any] = {}
    parsed_trailer = False
    answer_text = emitted

    if failed and not (emitted or pending or trailer):
        yield {
            "kind": "final",
            "answer": _error_answer(model, "", rag_used),
            "follow_ups": [],
            "parsed_trailer": False,
        }
        return

    if json_mode:
        # Whole output was a JSON object (legacy shape): parse it in full.
        try:
            parsed = parse_json_object(pending + trailer)
            parsed_trailer = True
        except ValueError:
            parsed = {}
        answer_text = str(parsed.get("answer") or "").strip()
        if answer_text:
            emitted = answer_text
            yield {"kind": "delta", "text": answer_text}
    else:
        # Flush whatever tail was held back (no sentinel arrived, or a short
        # final chunk).
        if pending and not in_trailer:
            emitted += pending
            yield {"kind": "delta", "text": pending}
            pending = ""
        answer_text = emitted.strip()
        if trailer.strip():
            try:
                parsed = parse_json_object(trailer)
                parsed_trailer = True
            except ValueError:
                logger.warning("Citation trailer unparseable; recovering from links")
                parsed = {}

    if not answer_text:
        yield {
            "kind": "final",
            "answer": _error_answer(model, "", rag_used),
            "follow_ups": [],
            "parsed_trailer": parsed_trailer,
        }
        return

    cited_sources = parsed.get("cited_sources")
    if not isinstance(cited_sources, list) or not cited_sources:
        cited_sources = citations_from_inline_links(answer_text, payloads)

    answer = build_qa_answer(
        {
            "answer": answer_text,
            "cited_sources": cited_sources,
            "overall_confidence": parsed.get(
                "overall_confidence", "medium" if parsed_trailer else "medium"
            ),
        },
        question=question,
        articles=payloads if rag_used else None,
        rag_used=rag_used,
        model_used=model,
    )
    follow_ups = parsed.get("follow_ups")
    if not isinstance(follow_ups, list):
        follow_ups = []

    yield {
        "kind": "final",
        "answer": answer,
        "follow_ups": [str(f) for f in follow_ups if f][:5],
        "parsed_trailer": parsed_trailer,
    }
