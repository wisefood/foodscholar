"""Deterministic evidence ranking: fusion normalization + score adjustment.

``adjusted = rrf_norm × tier × recency × influence × study_design``

Every factor is multiplicative around 1.0 and defensive about missing data:
an article with no citation metadata, no year, and no tier ranks purely on
retrieval relevance — signals boost or discount, never gate.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config
from models.qa import QAUserContext, SubQuestionFilters
from services.article_policy import (
    effective_tier,
    is_expert_audience,
    is_visible_to,
    tier_boost,
)
from services.qa_pipeline.state import EvidenceItem
from services.qa_retrievers import QA_GUIDELINE_RAG_TOP_K_MAX, age_group_facets

logger = logging.getLogger(__name__)

# Study-design weights matched as substrings of the (LLM-assigned) ai_category.
# Ordered: the first match wins, so "systematic review of RCTs" scores as a
# review, not as an RCT.
STUDY_DESIGN_WEIGHTS = [
    ("meta-analysis", 1.3),
    ("meta analysis", 1.3),
    ("systematic review", 1.3),
    ("umbrella review", 1.3),
    ("randomized", 1.2),
    ("randomised", 1.2),
    ("rct", 1.2),
    ("clinical trial", 1.2),
    ("cohort", 1.1),
    ("prospective", 1.1),
    ("longitudinal", 1.1),
    ("case report", 0.85),
    ("case study", 0.85),
    ("in vitro", 0.85),
    ("animal", 0.85),
    ("preclinical", 0.85),
    ("in vivo", 0.85),
]


def _setting(name: str, default: float) -> float:
    try:
        return float(config.settings.get(name, default))
    except (TypeError, ValueError):
        return default


def int_field(payload: Dict[str, Any], *keys: str) -> int:
    """Read an integer field defensively across camelCase/snake_case spellings."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(int(value), 0)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def parse_year(value: Any) -> Optional[int]:
    """Extract a 4-digit year from ints, 'YYYY', or ISO date strings."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4:
        return None
    head = text[:4]
    if not head.isdigit():
        return None
    year = int(head)
    if 1800 <= year <= 2200:
        return year
    return None


def recency_factor(
    payload: Dict[str, Any],
    *,
    now_year: Optional[int] = None,
) -> float:
    """Exponential decay by age with a floor; neutral when the year is unknown."""
    year = parse_year(payload.get("publication_year") or payload.get("year"))
    if year is None:
        return 1.0
    current = now_year or datetime.now(timezone.utc).year
    age = max(current - year, 0)
    half_life = max(_setting("QA_RECENCY_HALF_LIFE_YEARS", 6.0), 0.1)
    floor = _setting("QA_RECENCY_FLOOR", 0.35)
    return max(floor, math.exp(-math.log(2) * age / half_life))


def influence_factor(payload: Dict[str, Any]) -> float:
    """Log-scaled citation boost; 1.0 for missing/zero bibliometrics.

    Influential citations (Semantic Scholar) count double: a paper other work
    builds on matters more than one that gets mentioned in passing. Boost-only
    so recent papers with no citation history are never penalized.
    """
    citations = int_field(payload, "citationCount", "citation_count")
    influential = int_field(
        payload, "influentialCitationCount", "influential_citation_count"
    )
    effective = citations + 2 * influential
    if effective <= 0:
        return 1.0
    weight = _setting("QA_INFLUENCE_WEIGHT", 0.3)
    cap = max(_setting("QA_INFLUENCE_CITATION_CAP", 1000.0), 1.0)
    scaled = min(math.log1p(effective) / math.log1p(cap), 1.0)
    return 1.0 + weight * scaled


def study_design_factor(payload: Dict[str, Any]) -> float:
    """Weight by study design from the enrichment metadata.

    Reads every field the enrichment writes design information to —
    ``study_type`` and ``biological_model`` are the dedicated fields on the
    annotated cohort (production carries them on ~8k articles), ``ai_category``
    the older spelling. ``biological_model: In vitro / Animal`` is what
    reliably marks preclinical work even when the category string does not.
    """
    model = payload.get("biological_model")
    if isinstance(model, str) and any(
        term in model.lower() for term in ("in vitro", "animal", "preclinical")
    ):
        # Preclinical is preclinical whatever the study design says: an animal
        # RCT must not outrank human evidence for a human nutrition question.
        return 0.85

    parts = [
        payload.get("study_type"),
        payload.get("ai_category"),
        payload.get("category"),
    ]
    lowered = " ".join(str(p) for p in parts if isinstance(p, str)).lower()
    if not lowered:
        return 1.0
    for term, weight in STUDY_DESIGN_WEIGHTS:
        if term in lowered:
            return weight
    return 1.0


def _facet_terms(value: Any) -> set:
    """Normalize a keyword facet (list or string) for overlap comparison."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return set()
    terms = set()
    for item in value:
        text = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text:
            terms.add(text)
    return terms


def _term_words(terms: set) -> set:
    words = set()
    for term in terms:
        words.update(w for w in term.split("_") if len(w) > 3)
    return words


def _terms_overlap(a: set, b: set) -> bool:
    if not a or not b:
        return False
    if a & b:
        return True
    # "pregnant_people" should meet "pregnancy", "diabetes" should meet
    # "diabetic": compare word tokens, matching on containment or a shared
    # 6-character stem — long enough that unrelated facet words don't collide.
    words_a, words_b = _term_words(a), _term_words(b)
    for x in words_a:
        for y in words_b:
            if x in y or y in x:
                return True
            if len(x) >= 6 and len(y) >= 6 and x[:6] == y[:6]:
                return True
    return False


def guideline_affinity_factor(
    payload: Dict[str, Any],
    *,
    user_context: Optional[QAUserContext] = None,
    facets: Optional[SubQuestionFilters] = None,
) -> float:
    """How well a rule's enrichment facets fit the asker and the question.

    Boost-only, mirroring the retrieval-side philosophy: matching region,
    matching population/life stage, and topical facet overlap each earn a
    bounded multiplier; a rule with no facets ranks purely on relevance.
    """
    factor = 1.0

    wanted_regions = _facet_terms(facets.regions if facets else [])
    if user_context:
        wanted_regions |= _facet_terms(
            [user_context.country, user_context.region]
        )
    rule_regions = _facet_terms(payload.get("guide_region")) | _facet_terms(
        payload.get("applicable_regions")
    )
    if _terms_overlap(wanted_regions, rule_regions):
        factor *= 1.15

    wanted_populations = _facet_terms(
        facets.target_populations if facets else []
    )
    if user_context:
        resolved = age_group_facets(user_context.member_age_group)
        if resolved:
            wanted_populations |= _facet_terms(resolved[0])
    rule_populations = _facet_terms(payload.get("life_stage")) | _facet_terms(
        payload.get("target_populations")
    )
    if _terms_overlap(wanted_populations, rule_populations):
        factor *= 1.15

    wanted_topics = _facet_terms(
        [
            *(facets.food_groups if facets else []),
            *(facets.nutrients if facets else []),
            *(facets.health_conditions if facets else []),
        ]
    )
    rule_topics = (
        _facet_terms(payload.get("food_groups"))
        | _facet_terms(payload.get("nutrients"))
        | _facet_terms(payload.get("health_conditions"))
    )
    if _terms_overlap(wanted_topics, rule_topics):
        factor *= 1.1

    return factor


def adjust_evidence(
    items: List[EvidenceItem],
    *,
    expertise_level: Optional[str] = None,
    now_year: Optional[int] = None,
    user_context: Optional[QAUserContext] = None,
    question_facets: Optional[SubQuestionFilters] = None,
) -> List[EvidenceItem]:
    """Compute adjusted scores in place and drop audience-restricted items."""
    expert = is_expert_audience(expertise_level)
    adjusted: List[EvidenceItem] = []
    for item in items:
        payload = item.payload
        is_guideline = payload.get("source_type") == "guideline"

        if not is_guideline and not is_visible_to(payload, expert=expert):
            continue

        if is_guideline:
            tier = 1.0
            recency = 1.0
            influence = 1.0
            design = 1.0
            affinity = guideline_affinity_factor(
                payload, user_context=user_context, facets=question_facets
            )
        else:
            tier = tier_boost(effective_tier(payload))
            recency = recency_factor(payload, now_year=now_year)
            influence = influence_factor(payload)
            design = study_design_factor(payload)
            affinity = 1.0

        item.score_parts = {
            "rrf_norm": round(item.rrf_norm, 6),
            "tier": round(tier, 4),
            "recency": round(recency, 4),
            "influence": round(influence, 4),
            "study_design": round(design, 4),
            "affinity": round(affinity, 4),
        }
        item.adjusted_score = (
            item.rrf_norm * tier * recency * influence * design * affinity
        )
        payload["adjusted_score"] = item.adjusted_score
        payload["score_parts"] = item.score_parts
        adjusted.append(item)
    return adjusted


def select_evidence(
    items: List[EvidenceItem],
    *,
    top_k: int,
) -> tuple[List[EvidenceItem], Dict[str, int]]:
    """Order by adjusted score, apply diversity cap, threshold, and budgets.

    Returns the selected items (articles first, then guidelines — the block
    order the answer prompt's G-labels rely on) and drop counters. An empty
    selection is a legitimate outcome the evaluator must see.
    """
    min_score = _setting("QA_MIN_SCORE", 0.05)
    per_doc_cap = max(int(_setting("QA_PER_DOC_CAP", 2)), 1)

    below_threshold = 0
    over_doc_cap = 0
    doc_counts: Dict[str, int] = {}
    articles: List[EvidenceItem] = []
    guidelines: List[EvidenceItem] = []

    for item in sorted(items, key=lambda i: i.adjusted_score, reverse=True):
        if item.adjusted_score < min_score:
            below_threshold += 1
            continue
        doc_key = item.parent_doc_key
        if doc_counts.get(doc_key, 0) >= per_doc_cap:
            over_doc_cap += 1
            continue
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
        if item.payload.get("source_type") == "guideline":
            guidelines.append(item)
        else:
            articles.append(item)

    over_budget = max(len(articles) - top_k, 0) + max(
        len(guidelines) - QA_GUIDELINE_RAG_TOP_K_MAX, 0
    )
    selected = articles[:top_k] + guidelines[:QA_GUIDELINE_RAG_TOP_K_MAX]
    dropped = {
        "below_threshold": below_threshold,
        "over_doc_cap": over_doc_cap,
        "over_budget": over_budget,
    }
    return selected, dropped
