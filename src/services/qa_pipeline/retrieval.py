"""Per-sub-question hybrid retrieval with client-side reciprocal rank fusion.

Articles run a BM25 leg and a kNN leg; guidelines run the gated BM25 builder
(and a gated kNN leg when hybrid guideline retrieval is enabled). Legs are
fused client-side with RRF — the server-side ES ``rrf`` retriever is a
licensed feature, and fusion has to happen in code anyway so the ranking
module can adjust the fused score.

Guideline queries come exclusively from the shared builders in
``services.qa_retrievers`` (``guideline_base_query`` and friends); this module
never builds its own status filter. The gate tests scan this file's source to
keep it that way.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.elastic import ELASTIC_CLIENT
from config import config
from models.qa import (
    PlannedSubQuestion,
    QAUserContext,
    RetrievedSource,
    SubQuestionFilters,
)
from services.article_policy import article_filter_query
from services.qa_pipeline.state import EvidenceItem
from services.qa_retrievers import (
    GUIDELINE_SOURCE_EXCLUDES,
    guideline_base_query,
    guideline_context_should_clauses,
    guideline_hybrid_enabled,
    guideline_retrieval_filter,
    normalize_article_hit,
    normalize_guideline_hit,
)

logger = logging.getLogger(__name__)

RETRIEVER_NAME = "agentic"

# Verified against the production mapping: keywords (~105k docs) and tags
# (~105k) are the broadly populated lexical enrichments; ai_* and topics come
# from the annotation cohort. key_takeaways is mapped but empty (0 docs) and
# deliberately not searched — the guideline phantom-field lesson.
ARTICLE_LEXICAL_FIELDS = [
    "title^3",
    "abstract^2",
    "keywords^2",
    "ai_key_takeaways^2",
    "tags^2",
    "ai_tags^2",
    "topics^2",
    "venue",
]


@dataclass
class BranchOutcome:
    """Everything one (sub-question, branch) search produced."""

    sub_question_id: str
    branch: str
    items: List[EvidenceItem] = field(default_factory=list)
    status: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.status.get("ok"))


def _rrf_k() -> int:
    try:
        return max(int(config.settings.get("QA_RRF_K", 60)), 1)
    except (TypeError, ValueError):
        return 60


def _candidates() -> int:
    try:
        return max(int(config.settings.get("QA_RRF_CANDIDATES", 30)), 5)
    except (TypeError, ValueError):
        return 30


def rrf_fuse(
    legs: List[List[Tuple[str, Dict[str, Any], RetrievedSource]]],
    *,
    k: Optional[int] = None,
) -> List[EvidenceItem]:
    """Fuse ranked legs of (key, payload, source) into deduplicated items.

    ``score(d) = Σ_legs 1 / (k + rank(d))`` with rank starting at 1, then
    normalized by the best possible score so ``rrf_norm`` lands in (0, 1].
    Legs that returned nothing do not dilute the normalization.
    """
    k = k or _rrf_k()
    non_empty = [leg for leg in legs if leg]
    if not non_empty:
        return []

    fused: Dict[str, EvidenceItem] = {}
    for leg in non_empty:
        for rank, (key, payload, source) in enumerate(leg, start=1):
            contribution = 1.0 / (k + rank)
            item = fused.get(key)
            if item is None:
                item = EvidenceItem(payload=payload, source=source)
                fused[key] = item
            item.rrf_score += contribution

    best_possible = len(non_empty) / (k + 1)
    items = list(fused.values())
    for item in items:
        item.rrf_norm = min(item.rrf_score / best_possible, 1.0)
    items.sort(key=lambda i: i.rrf_norm, reverse=True)
    return items


def article_attribute_clauses(
    filters: Optional[SubQuestionFilters],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attribute constraints for the article index: (hard filters, boosts).

    Hard filters are limited to deterministic mapping-backed fields — a
    publication-year window and the open_access flag. Study designs are
    LLM-assigned text (``ai_category``) with partial coverage, so they boost
    rather than gate.
    """
    if filters is None or filters.is_empty():
        return [], []

    filter_clauses: List[Dict[str, Any]] = []
    should_clauses: List[Dict[str, Any]] = []

    if filters.year_min or filters.year_max:
        year_range: Dict[str, Any] = {"format": "yyyy"}
        if filters.year_min:
            year_range["gte"] = str(filters.year_min)
        if filters.year_max:
            year_range["lte"] = str(filters.year_max)
        filter_clauses.append({"range": {"publication_year": year_range}})

    if filters.open_access is True:
        filter_clauses.append({"term": {"open_access": True}})

    for study_type in filters.study_types[:4]:
        should_clauses.append(
            {
                "multi_match": {
                    "query": study_type,
                    "fields": ["ai_category^2", "study_type^2"],
                    "type": "best_fields",
                }
            }
        )
    # Population terms boost the enrichment's dedicated population fields on
    # top of the general text fields.
    for term in filters.target_populations[:4]:
        should_clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": [
                        "population_group^3",
                        "age_group^2",
                        "title",
                        "abstract",
                    ],
                    "type": "best_fields",
                }
            }
        )
    facet_terms = [
        *filters.food_groups[:4],
        *filters.nutrients[:4],
        *filters.health_conditions[:4],
    ]
    for term in facet_terms:
        should_clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": [
                        "title^2",
                        "abstract",
                        "keywords^2",
                        "tags^2",
                        "ai_tags^2",
                        "topics^2",
                        "ai_key_takeaways",
                    ],
                    "type": "best_fields",
                }
            }
        )
    return filter_clauses, should_clauses


def guideline_attribute_should_clauses(
    filters: Optional[SubQuestionFilters],
) -> List[Dict[str, Any]]:
    """Attribute boosts for the guideline index — boosts only, never filters.

    The editorial gate stays the single hard filter on guidelines; question
    attributes reorder within it. Facet fields carry values only after
    enrichment, so a hard filter here would silently hide the unenriched
    majority.
    """
    if filters is None or filters.is_empty():
        return []

    clauses: List[Dict[str, Any]] = []
    for region in filters.regions[:3]:
        clauses.append(
            {
                "multi_match": {
                    "query": region,
                    "fields": ["guide_region^5", "applicable_regions^5", "title"],
                    "type": "best_fields",
                }
            }
        )
    for population in filters.target_populations[:3]:
        clauses.append(
            {
                "multi_match": {
                    "query": population,
                    "fields": ["target_populations^3", "life_stage^3", "audience^2"],
                    "type": "best_fields",
                }
            }
        )
    for term in [*filters.food_groups[:4], *filters.nutrients[:4]]:
        clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": ["food_groups^3", "nutrients^3", "rule_text^2", "topic"],
                    "type": "best_fields",
                }
            }
        )
    for condition in filters.health_conditions[:3]:
        clauses.append(
            {
                "multi_match": {
                    "query": condition,
                    "fields": ["health_conditions^4", "rule_text^2", "notes"],
                    "type": "best_fields",
                }
            }
        )
    return clauses


def _identity_key(payload: Dict[str, Any]) -> str:
    prefix = "guideline" if payload.get("source_type") == "guideline" else "article"
    for k in ("urn", "doi", "id", "_id"):
        value = payload.get(k)
        if isinstance(value, str) and value.strip():
            return f"{prefix}:{value.strip()}"
    return f"{prefix}:{id(payload)}"


def search_articles_lexical(
    query: str,
    *,
    size: int,
    expertise_level: Optional[str],
    filters: Optional[SubQuestionFilters] = None,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    """BM25 leg over articles: editorial pre-filter + question attributes."""
    filter_clauses, should_clauses = article_attribute_clauses(filters)
    bool_query: Dict[str, Any] = {
        "must": [
            {
                "multi_match": {
                    "query": query,
                    "fields": ARTICLE_LEXICAL_FIELDS,
                    "type": "best_fields",
                }
            }
        ],
        "must_not": article_filter_query(expertise_level)["bool"]["must_not"],
    }
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    if should_clauses:
        bool_query["should"] = should_clauses
    body = {
        "size": size,
        "query": {"bool": bool_query},
        "_source": {"excludes": ["embedding"]},
    }
    response = ELASTIC_CLIENT.client.search(index="articles", body=body)
    leg = []
    for hit in response.get("hits", {}).get("hits", []):
        raw = {"_id": hit.get("_id"), "_score": hit.get("_score", 0.0)}
        source_doc = hit.get("_source") or {}
        if isinstance(source_doc, dict):
            raw.update(source_doc)
        payload, source = normalize_article_hit(raw, retriever=RETRIEVER_NAME)
        leg.append((_identity_key(payload), payload, source))
    return leg


def search_articles_knn(
    vector: List[float],
    *,
    size: int,
    expertise_level: Optional[str],
    filters: Optional[SubQuestionFilters] = None,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    """Informed vector leg over articles: semantic similarity within the
    question's hard attribute constraints (year window, open access), not a
    blind nearest-neighbour sweep."""
    filter_clauses, _ = article_attribute_clauses(filters)
    knn_filter: Any = article_filter_query(expertise_level)
    if filter_clauses:
        knn_filter = [knn_filter, *filter_clauses]
    raw_results = ELASTIC_CLIENT.knn_search(
        index_name="articles",
        query_vector=vector,
        k=size,
        num_candidates=max(size * 4, 100),
        field="embedding",
        filter_query=knn_filter,
        source_excludes=["embedding"],
    )
    leg = []
    for raw in raw_results:
        payload, source = normalize_article_hit(raw, retriever=RETRIEVER_NAME)
        leg.append((_identity_key(payload), payload, source))
    return leg


def search_guidelines_lexical(
    query: str,
    *,
    size: int,
    user_context: Optional[QAUserContext],
    filters: Optional[SubQuestionFilters] = None,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    """Gated BM25 leg over guidelines, built ONLY from the shared builders.

    Question attributes and user context both contribute boosts; the shared
    builder's editorial gate stays the only hard filter.
    """
    body = {
        "size": size,
        "query": guideline_base_query(query),
        "_source": {"excludes": GUIDELINE_SOURCE_EXCLUDES},
    }
    context_should = [
        *guideline_context_should_clauses(user_context),
        *guideline_attribute_should_clauses(filters),
    ]
    if context_should:
        body["query"]["bool"]["should"] = context_should

    response = ELASTIC_CLIENT.client.search(index="guidelines", body=body)
    leg = []
    for hit in response.get("hits", {}).get("hits", []):
        normalized = normalize_guideline_hit(hit, retriever=RETRIEVER_NAME)
        if normalized is None:
            continue
        payload, source = normalized
        leg.append((_identity_key(payload), payload, source))
    return leg


def search_guidelines_knn(
    vector: List[float],
    *,
    size: int,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    """Gated vector leg over guidelines; only used in hybrid guideline mode."""
    raw_results = ELASTIC_CLIENT.knn_search(
        index_name="guidelines",
        query_vector=vector,
        k=size,
        num_candidates=max(size * 4, 100),
        field="embedding",
        filter_query=guideline_retrieval_filter(),
        source_excludes=GUIDELINE_SOURCE_EXCLUDES,
    )
    leg = []
    for raw in raw_results:
        hit = {"_id": raw.pop("_id", None), "_score": raw.pop("_score", 0.0)}
        hit["_source"] = raw
        normalized = normalize_guideline_hit(hit, retriever=RETRIEVER_NAME)
        if normalized is None:
            continue
        payload, source = normalized
        leg.append((_identity_key(payload), payload, source))
    return leg


def run_branch(
    sub_question: PlannedSubQuestion,
    *,
    branch: str,
    vector: Optional[List[float]],
    user_context: Optional[QAUserContext],
    expertise_level: Optional[str],
) -> BranchOutcome:
    """Run all legs of one (sub-question, branch) search and fuse them.

    A failed leg is recorded and skipped rather than failing the branch; a
    failed branch is recorded and never fails the pipeline.
    """
    size = _candidates()
    lexical_query = sub_question.lexical_query or sub_question.text
    filters = sub_question.filters
    legs: List[List[Tuple[str, Dict[str, Any], RetrievedSource]]] = []
    leg_status: Dict[str, Any] = {}

    if branch == "articles":
        try:
            legs.append(
                search_articles_lexical(
                    lexical_query,
                    size=size,
                    expertise_level=expertise_level,
                    filters=filters,
                )
            )
            leg_status["lexical"] = len(legs[-1])
        except Exception as exc:
            logger.error("Article BM25 leg failed: %s", exc, exc_info=True)
            leg_status["lexical_error"] = repr(exc)
        if vector is not None:
            try:
                legs.append(
                    search_articles_knn(
                        vector,
                        size=size,
                        expertise_level=expertise_level,
                        filters=filters,
                    )
                )
                leg_status["knn"] = len(legs[-1])
            except Exception as exc:
                logger.error("Article kNN leg failed: %s", exc, exc_info=True)
                leg_status["knn_error"] = repr(exc)
    elif branch == "guidelines":
        try:
            legs.append(
                search_guidelines_lexical(
                    lexical_query,
                    size=size,
                    user_context=user_context,
                    filters=filters,
                )
            )
            leg_status["lexical"] = len(legs[-1])
        except Exception as exc:
            logger.error("Guideline BM25 leg failed: %s", exc, exc_info=True)
            leg_status["lexical_error"] = repr(exc)
        if vector is not None and guideline_hybrid_enabled():
            try:
                legs.append(search_guidelines_knn(vector, size=size))
                leg_status["knn"] = len(legs[-1])
            except Exception as exc:
                # Same degradation contract as the legacy adapter: a broken
                # vector leg costs recall, never the answer.
                logger.warning(
                    "Guideline kNN leg unavailable, keyword only: %s", exc
                )
                leg_status["knn_error"] = repr(exc)
    else:  # pragma: no cover - orchestrator only dispatches known branches
        raise ValueError(f"Unknown branch: {branch}")

    items = rrf_fuse(legs)
    for item in items:
        item.sub_question_ids = [sub_question.id]

    any_leg_ok = any(key in leg_status for key in ("lexical", "knn"))
    return BranchOutcome(
        sub_question_id=sub_question.id,
        branch=branch,
        items=items,
        status={
            "ok": any_leg_ok,
            "hit_count": len(items),
            "legs": leg_status,
            "used_query": lexical_query,
        },
    )


def merge_evidence(
    existing: List[EvidenceItem],
    new_items: List[EvidenceItem],
) -> List[EvidenceItem]:
    """Merge a new round's items into the pool, deduplicating by identity.

    A document found again keeps its best fused score and accumulates the
    sub-questions that surfaced it — corroboration across sub-questions is a
    signal the evaluator gets to see.
    """
    pool: Dict[str, EvidenceItem] = {item.key: item for item in existing}
    for item in new_items:
        current = pool.get(item.key)
        if current is None:
            pool[item.key] = item
            continue
        current.rrf_score = max(current.rrf_score, item.rrf_score)
        current.rrf_norm = max(current.rrf_norm, item.rrf_norm)
        for sq_id in item.sub_question_ids:
            if sq_id not in current.sub_question_ids:
                current.sub_question_ids.append(sq_id)
    return list(pool.values())
