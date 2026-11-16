"""The daily quiz: grounded multiple-choice questions for curious readers.

Generation lives on QAService (``get_daily_quiz``), next to the tips of the
day whose source sampling and caching it shares. This module holds the parts
that need no services: count clamping, the source context shown to the model,
and the strict validation of what the model returns. A malformed item is
dropped, never repaired: a quiz question with the wrong answer marked right
teaches the opposite of the truth.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional

from services.learn_sources import APPLIES_TO_NOTE

QUIZ_COUNT_DEFAULT = 3
QUIZ_COUNT_MIN = 1
QUIZ_COUNT_MAX = 5
QUIZ_MIN_OPTIONS = 3
QUIZ_MAX_OPTIONS = 4
QUIZ_QUESTION_MAX_CHARS = 300
QUIZ_OPTION_MAX_CHARS = 140
QUIZ_WHY_MAX_CHARS = 600
QUIZ_CONTEXT_PASSAGE_CHARS = 600


def clamp_quiz_count(count: Any) -> int:
    """1..5, default 3 for anything that is not a number."""
    try:
        value = int(count)
    except (TypeError, ValueError):
        return QUIZ_COUNT_DEFAULT
    return max(QUIZ_COUNT_MIN, min(QUIZ_COUNT_MAX, value))


def quiz_item_id(question: str) -> str:
    """Stable id for a question: the same text gives the same id."""
    normalized = " ".join(str(question).lower().split())
    return "q-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_quiz_source_context(sources: List[Dict[str, Any]]) -> str:
    """Numbered source blocks for the prompt (1-based, as the model cites)."""
    # Who each rule is for travels with it: a portion from a guide for 1-4
    # year olds must not come back as a question about everyone's portion.
    blocks = [APPLIES_TO_NOTE]
    for index, source in enumerate(sources, 1):
        evidence = source.get("evidence") or {}
        is_rule = evidence.get("source_type") == "guideline"
        kind = "Dietary guideline" if is_rule else "Study"
        text = str(source.get("text") or evidence.get("passage") or "")
        lines = [f"Source {index} ({kind}):", f"- Title: {evidence.get('title') or 'N/A'}"]
        if is_rule:
            lines.append(f"- Guide: {source.get('guide') or 'N/A'}")
            lines.append(f"- Applies to: {source.get('applies_to') or 'not stated'}")
        lines.append(f"- Text: {text[:QUIZ_CONTEXT_PASSAGE_CHARS]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _coerce_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value.strip())
    return None


def _clean(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _mentions_any(text: str, terms: List[str]) -> bool:
    haystack = text.lower()
    for term in terms:
        cleaned = str(term).strip().lower()
        if cleaned and re.search(r"\b" + re.escape(cleaned) + r"s?\b", haystack):
            return True
    return False


def validate_quiz_items(
    items: Any,
    sources: List[Dict[str, Any]],
    *,
    count: int,
    is_unsafe: Optional[Callable[[str], bool]] = None,
    avoid_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Keep only well-formed, grounded, safe items; at most ``count``.

    An item must have a question, 3-4 distinct non-empty options, an integer
    ``answer`` indexing them, a ``why``, and a ``source`` naming one of the
    numbered sources (1-based). Each source backs at most one question, and a
    question text appears once. ``evidence`` comes from the source, never from
    the model.
    """
    if not isinstance(items, list):
        return []
    avoid = [t for t in (avoid_terms or []) if str(t).strip()]
    valid: List[Dict[str, Any]] = []
    used_sources = set()
    seen_ids = set()

    for item in items:
        if len(valid) >= count:
            break
        if not isinstance(item, dict):
            continue

        question = _clean(item.get("question"))
        why = _clean(item.get("why"))
        raw_options = item.get("options")
        if not question or not why or len(question) > QUIZ_QUESTION_MAX_CHARS:
            continue
        if len(why) > QUIZ_WHY_MAX_CHARS:
            continue
        if not isinstance(raw_options, list):
            continue
        options = [_clean(o) for o in raw_options]
        if not (QUIZ_MIN_OPTIONS <= len(options) <= QUIZ_MAX_OPTIONS):
            continue
        if any(not o or len(o) > QUIZ_OPTION_MAX_CHARS for o in options):
            continue
        if len({o.lower() for o in options}) != len(options):
            continue

        answer = _coerce_index(item.get("answer"))
        if answer is None or not (0 <= answer < len(options)):
            continue

        source_index = _coerce_index(item.get("source", item.get("guideline")))
        if source_index is None or not (1 <= source_index <= len(sources)):
            continue
        if source_index in used_sources:
            continue
        evidence = (sources[source_index - 1] or {}).get("evidence") or {}
        if not evidence.get("urn") or not evidence.get("passage"):
            continue

        visible = " ".join([question, why, *options])
        if is_unsafe is not None and is_unsafe(" ".join([question, why])):
            continue
        if avoid and _mentions_any(visible, avoid):
            continue

        item_id = quiz_item_id(question)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        used_sources.add(source_index)

        valid.append(
            {
                "id": item_id,
                "question": question,
                "options": options,
                "answer": answer,
                "why": why,
                "evidence": {
                    "urn": evidence.get("urn"),
                    "passage": evidence.get("passage"),
                    "title": evidence.get("title") or None,
                    "publication_year": evidence.get("publication_year") or None,
                    "source_type": evidence.get("source_type") or None,
                    "guide_urn": evidence.get("guide_urn") or None,
                    "page_no": evidence.get("page_no") or None,
                    "region": evidence.get("region") or None,
                    "age_min_months": evidence.get("age_min_months"),
                    "age_max_months": evidence.get("age_max_months"),
                },
            }
        )
    return valid
