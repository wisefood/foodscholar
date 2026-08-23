"""Retriever adapters for QA evidence normalization."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from backend.elastic import ELASTIC_CLIENT
from config import config
from models.qa import QAClarifierSafetyPlan, QAUserContext, RetrievedSource
from services.article_policy import article_filter_query

logger = logging.getLogger(__name__)

TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS = 12
QA_GUIDELINE_RAG_TOP_K_MAX = 5


@dataclass
class RetrievalResult:
    """Normalized retrieval output for answer formulation and API display."""

    source_payloads: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_sources: List[RetrievedSource] = field(default_factory=list)
    status: Dict[str, Any] = field(default_factory=dict)


class RetrieverAdapter(Protocol):
    """Retriever adapter protocol."""

    retriever_name: str

    def retrieve(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        top_k: int,
        user_context: Optional[QAUserContext],
        expertise_level: Optional[str] = None,
    ) -> RetrievalResult:
        ...


class NoRagRetrieverAdapter:
    """Adapter for no-retrieval QA."""

    retriever_name = "no_rag"

    def retrieve(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        top_k: int,
        user_context: Optional[QAUserContext],
        expertise_level: Optional[str] = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            status={
                "retriever": self.retriever_name,
                "ok": True,
                "article_hits": 0,
                "guideline_hits": 0,
            }
        )


class LinearragRetrieverAdapter:
    """Adapter for LinearRAG passage retrieval."""

    retriever_name = "linearrag"

    def retrieve(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        top_k: int,
        user_context: Optional[QAUserContext],
        expertise_level: Optional[str] = None,
    ) -> RetrievalResult:
        # LinearRAG has no queryable policy fields, so reader visibility and
        # tiers are enforced downstream by article_policy.filter_and_rank.
        try:
            from services.linearrag_service import retrieve as linearrag_retrieve

            query = contextualize_query(
                plan.article_query or plan.canonical_question or question,
                user_context,
            )
            raw_results = linearrag_retrieve(query, top_k=top_k)
        except Exception as exc:
            logger.error("LinearRAG retrieval failed: %s", exc, exc_info=True)
            return RetrievalResult(
                status={
                    "retriever": self.retriever_name,
                    "ok": False,
                    "error": repr(exc),
                    "article_hits": 0,
                    "guideline_hits": 0,
                }
            )

        payloads: List[Dict[str, Any]] = []
        retrieved: List[RetrievedSource] = []
        article_hits = 0
        guideline_hits = 0
        for lr in raw_results:
            source = lr.get("source") or {}
            source_type = infer_source_type(source)
            text = lr.get("text", "")
            payload = {
                "abstract": text,
                "description": text,
                "_score": lr.get("score", 0.0),
                "source_type": source_type,
                "retriever": self.retriever_name,
                **source,
            }
            if source_type == "guideline":
                guideline_hits += 1
                payload["rule_text"] = (
                    payload.get("rule_text")
                    or payload.get("description")
                    or payload.get("abstract")
                )
            else:
                article_hits += 1

            payloads.append(payload)
            retrieved.append(
                RetrievedSource(
                    source_type=source_type,
                    urn=text_value(
                        source.get("urn")
                        or source.get("id")
                        or source.get("_id")
                    ),
                    title=text_value(source.get("title")),
                    authors=(
                        normalize_string_list(source.get("authors"))
                        if source_type == "article"
                        else None
                    ),
                    venue=text_value(
                        source.get("venue") or source.get("guide_region"),
                        default=None,
                    ),
                    publication_year=text_value(
                        source.get("publication_year"),
                        default=None,
                    ),
                    category=text_value(source.get("category"), default=None),
                    tags=normalize_string_list(source.get("tags")),
                    similarity_score=score_value(lr.get("score")),
                )
            )

        return RetrievalResult(
            source_payloads=payloads,
            retrieved_sources=retrieved,
            status={
                "retriever": self.retriever_name,
                "ok": True,
                "article_hits": article_hits,
                "guideline_hits": guideline_hits,
                "used_query": query,
            },
        )


def normalize_article_hit(
    result: Dict[str, Any],
    *,
    retriever: str = "rag",
) -> tuple[Dict[str, Any], RetrievedSource]:
    """Shape a raw article hit into the payload + display source pair.

    Shared by the legacy adapter and the agentic pipeline so both feed the
    answer prompt and the UI the exact same article shape.
    """
    result["source_type"] = "article"
    result["retriever"] = retriever
    result["relevance_score"] = result.get("_score", 0.0)

    def _count(*keys: str) -> Optional[int]:
        for key in keys:
            value = result.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    retrieved = RetrievedSource(
        source_type="article",
        urn=text_value(result.get("urn") or result.get("_id")),
        title=text_value(result.get("title")),
        authors=normalize_string_list(result.get("authors")),
        venue=text_value(result.get("venue"), default=None),
        publication_year=text_value(
            result.get("publication_year"),
            default=None,
        ),
        category=text_value(result.get("category"), default=None),
        tags=normalize_string_list(result.get("tags")),
        similarity_score=score_value(result.get("_score")),
        # Prioritization signals, surfaced so the UI can show WHY a source
        # ranked where it did (date, reach, study design).
        citation_count=_count("citationCount", "citation_count"),
        influential_citation_count=_count(
            "influentialCitationCount", "influential_citation_count"
        ),
        study_type=text_value(result.get("ai_category"), default=None),
    )
    return result, retrieved


def normalize_guideline_hit(
    hit: Dict[str, Any],
    *,
    retriever: str = "rag",
) -> Optional[tuple[Dict[str, Any], RetrievedSource]]:
    """Shape a raw guideline ES hit; None when the rule text is unusable."""
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        return None

    guideline = {
        **source,
        "_id": hit.get("_id"),
        "_score": hit.get("_score", 0.0),
        "source_type": "guideline",
        "retriever": retriever,
    }
    rule_text = extract_guideline_rule_text(guideline)
    if len(rule_text) < TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS:
        return None

    urn = guideline_document_urn(guideline)
    guideline["urn"] = urn
    guideline["abstract"] = rule_text
    guideline["description"] = rule_text
    guideline["venue"] = guideline.get("guide_region")
    guideline["relevance_score"] = guideline.get("_score", 0.0)
    guideline["publication_year"] = guideline_publication_year(guideline)

    raw_page = guideline.get("page_no")
    page_no = (
        int(raw_page)
        if isinstance(raw_page, (int, float)) and not isinstance(raw_page, bool)
        else None
    )
    retrieved = RetrievedSource(
        source_type="guideline",
        urn=urn,
        title=text_value(
            guideline.get("title"),
            default="Dietary guideline",
        ),
        authors=None,
        venue=text_value(guideline.get("guide_region"), default=None),
        publication_year=text_value(
            guideline.get("publication_year"),
            default=None,
        ),
        category="guideline",
        tags=guideline_tags(guideline),
        similarity_score=score_value(guideline.get("_score")),
        guide_urn=text_value(guideline.get("guide_urn"), default=None),
        page_no=page_no,
    )
    return guideline, retrieved


class ElasticRagRetrieverAdapter:
    """Adapter for default Elastic article + guideline retrieval."""

    retriever_name = "rag"

    def __init__(
        self,
        *,
        embed_query: Callable[[str], List[float]],
        articles_index: str = "articles",
        guidelines_index: str = "guidelines",
    ):
        self.embed_query = embed_query
        self.articles_index = articles_index
        self.guidelines_index = guidelines_index

    def retrieve(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        top_k: int,
        user_context: Optional[QAUserContext],
        expertise_level: Optional[str] = None,
    ) -> RetrievalResult:
        article_query = contextualize_query(
            plan.article_query or plan.canonical_question or question,
            user_context,
        )
        guideline_query = contextualize_query(
            plan.guideline_query or plan.canonical_question or question,
            user_context,
        )
        article_payloads, article_sources, article_status = self._retrieve_articles(
            article_query,
            top_k,
            expertise_level=expertise_level,
        )
        guideline_top_k = min(max(top_k, 1), QA_GUIDELINE_RAG_TOP_K_MAX)
        guideline_payloads, guideline_sources, guideline_status = (
            self._retrieve_guidelines(
                guideline_query,
                guideline_top_k,
                user_context,
            )
        )

        return RetrievalResult(
            source_payloads=article_payloads + guideline_payloads,
            retrieved_sources=article_sources + guideline_sources,
            status={
                "retriever": self.retriever_name,
                "ok": article_status.get("ok", False)
                or guideline_status.get("ok", False),
                "articles": article_status,
                "guidelines": guideline_status,
                "article_hits": len(article_payloads),
                "guideline_hits": len(guideline_payloads),
            },
        )

    def _retrieve_articles(
        self,
        query: str,
        top_k: int,
        *,
        expertise_level: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], List[RetrievedSource], Dict[str, Any]]:
        try:
            query_vector = self.embed_query(query)
            raw_results = ELASTIC_CLIENT.knn_search(
                index_name=self.articles_index,
                query_vector=query_vector,
                k=top_k,
                num_candidates=max(top_k * 20, 100),
                field="embedding",
                # Editorially restricted articles are excluded in the query
                # rather than afterwards, so they never consume a top_k slot.
                filter_query=article_filter_query(expertise_level),
                source_excludes=["embedding"],
            )
        except Exception as exc:
            logger.error("Article RAG retrieval failed: %s", exc, exc_info=True)
            return [], [], {
                "ok": False,
                "error": repr(exc),
                "used_query": query,
                "mode": "vector",
            }

        payloads: List[Dict[str, Any]] = []
        retrieved: List[RetrievedSource] = []
        for result in raw_results:
            payload, source = normalize_article_hit(
                result, retriever=self.retriever_name
            )
            payloads.append(payload)
            retrieved.append(source)

        return (
            payloads,
            retrieved,
            {
                "ok": True,
                "hit_count": len(payloads),
                "used_query": query,
                "mode": "vector",
            },
        )

    def _retrieve_guidelines(
        self,
        query: str,
        top_k: int,
        user_context: Optional[QAUserContext],
    ) -> tuple[List[Dict[str, Any]], List[RetrievedSource], Dict[str, Any]]:
        if top_k <= 0:
            return [], [], {
                "ok": True,
                "hit_count": 0,
                "used_query": query,
                "mode": "keyword",
            }

        body = {
            "size": top_k,
            "query": guideline_base_query(query),
            "_source": {"excludes": GUIDELINE_SOURCE_EXCLUDES},
        }
        context_should = guideline_context_should_clauses(user_context)
        if context_should:
            body["query"]["bool"]["should"] = context_should

        mode = "keyword"
        if guideline_hybrid_enabled():
            # Rule sentences are short and paraphrase heavily, which is exactly
            # where BM25 alone misses; the vector leg catches semantic matches
            # the wording does not share. The gate applies to both legs.
            try:
                body["knn"] = {
                    "field": "embedding",
                    "query_vector": self.embed_query(query),
                    "k": top_k,
                    "num_candidates": max(top_k * 20, 100),
                    "filter": guideline_retrieval_filter(),
                    "boost": guideline_hybrid_knn_boost(),
                }
                mode = "hybrid"
            except Exception as exc:
                # A failed embedding call degrades to keyword search rather than
                # failing the answer outright.
                logger.warning(
                    "Guideline hybrid retrieval unavailable, using keyword only: %s",
                    exc,
                )

        try:
            response = ELASTIC_CLIENT.client.search(
                index=self.guidelines_index,
                body=body,
            )
        except Exception as exc:
            logger.error("Guideline RAG retrieval failed: %s", exc, exc_info=True)
            return [], [], {
                "ok": False,
                "error": repr(exc),
                "used_query": query,
                "mode": mode,
            }

        payloads: List[Dict[str, Any]] = []
        retrieved: List[RetrievedSource] = []
        for hit in response.get("hits", {}).get("hits", []):
            normalized = normalize_guideline_hit(
                hit, retriever=self.retriever_name
            )
            if normalized is None:
                continue
            guideline, source = normalized
            payloads.append(guideline)
            retrieved.append(source)

        return (
            payloads,
            retrieved,
            {
                "ok": True,
                "hit_count": len(payloads),
                "used_query": query,
                "mode": mode,
            },
        )


class QARetrieverAdapters:
    """Small registry for retriever adapters."""

    def __init__(
        self,
        *,
        embed_query: Callable[[str], List[float]],
        articles_index: str,
        guidelines_index: str,
    ):
        self.adapters: Dict[str, RetrieverAdapter] = {
            "rag": ElasticRagRetrieverAdapter(
                embed_query=embed_query,
                articles_index=articles_index,
                guidelines_index=guidelines_index,
            ),
            "linearrag": LinearragRetrieverAdapter(),
            "no_rag": NoRagRetrieverAdapter(),
        }

    def get(self, retriever: str) -> RetrieverAdapter:
        return self.adapters.get(retriever, self.adapters["rag"])


def contextualize_query(
    query: str,
    user_context: Optional[QAUserContext],
) -> str:
    if not user_context:
        return query

    hints = []
    if user_context.country:
        hints.append(f"country {user_context.country}")
    if user_context.region and user_context.region != user_context.country:
        hints.append(f"region {user_context.region}")
    if user_context.member_age_group:
        hints.append(f"age group {user_context.member_age_group}")
    if user_context.experience_group:
        hints.append(f"audience {user_context.experience_group}")

    if not hints:
        return query
    return f"{query}\nContext: {', '.join(hints)}"


def infer_source_type(source: Dict[str, Any]) -> str:
    source_type = source.get("source_type")
    if isinstance(source_type, str):
        normalized = source_type.strip().lower()
        if normalized in {"guideline", "dietary_guideline"}:
            return "guideline"
        if normalized in {"article", "paper", "publication"}:
            return "article"
    if source.get("rule_text") or source.get("guide_region") or source.get("guide_urn"):
        return "guideline"
    return "article"


def text_value(value: Any, default: Optional[str] = "") -> Optional[str]:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def normalize_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, list) and not value:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, (tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Guideline retrieval: the single source of truth
#
# Every path that surfaces a guideline to a user — QA answers, daily tips —
# must build its query from the helpers below. Duplicating a query body is how
# the editorial gate came to be enforced in one place and not the others.
# ---------------------------------------------------------------------------

# Only active guidelines are retrievable. Draft, archived, deprecated and
# deleted rules stay in the catalog for editors but are never cited to a user
# and never quoted in an answer. Activating a reviewed guide's rules is a
# deliberate editorial act (POST /guidelines/editorial-policy in the catalog).
GUIDELINE_ACTIVE_STATUS = "active"

# Fields a guideline query searches. The facet fields carry real values only
# once enrichment has run; before that they simply never match, which costs
# nothing. Fields absent from the mapping are excluded on purpose — they match
# nothing and quietly mislead anyone reading the query.
# Never returned to a caller. Once guidelines carry vectors this is the bulk of
# every hit's payload — 384 floats per result, parsed and discarded.
GUIDELINE_SOURCE_EXCLUDES = ["embedding"]

GUIDELINE_SEARCH_FIELDS = [
    "rule_text^4",
    "title^2",
    "notes",
    "topic^3",
    "food_groups^2",
    "nutrients^2",
    "health_conditions^2",
    "target_populations",
    "life_stage^2",
    "setting",
    "guide_region",
    "applicable_regions",
]


def guideline_retrieval_filter() -> List[Dict[str, Any]]:
    """The editorial gate on every user-facing guideline query."""
    return [{"term": {"status": GUIDELINE_ACTIVE_STATUS}}]


def guideline_hybrid_enabled() -> bool:
    """
    Whether guideline retrieval combines BM25 with vector search.

    On by default now that the guideline embedding backfill has run. Set
    QA_GUIDELINE_RETRIEVAL_MODE=bm25 on a deployment whose guidelines are not
    embedded yet — there the vector leg would rank the embedded minority
    above everything else.
    """
    return str(
        config.settings.get("QA_GUIDELINE_RETRIEVAL_MODE", "hybrid")
    ).strip().lower() == "hybrid"


def guideline_hybrid_knn_boost() -> float:
    """Relative weight of the vector leg against BM25 in hybrid mode."""
    try:
        return float(config.settings.get("QA_GUIDELINE_KNN_BOOST", 1.0))
    except (TypeError, ValueError):
        return 1.0


def guideline_base_query(query: str) -> Dict[str, Any]:
    """A gated BM25 query over the guideline index."""
    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": GUIDELINE_SEARCH_FIELDS,
                        "type": "best_fields",
                    }
                }
            ],
            "filter": guideline_retrieval_filter(),
        }
    }


def guideline_tip_pool_query(query: Optional[str] = None) -> Dict[str, Any]:
    """
    A gated query for the daily-tip guideline pool.

    Tips are shown unprompted to every reader, so they run through the same
    editorial gate as cited answers — arguably a stronger requirement, since
    nobody asked for them.
    """
    must: List[Dict[str, Any]] = [{"exists": {"field": "rule_text"}}]
    if query:
        must.append(
            {
                "multi_match": {
                    "query": query,
                    "fields": GUIDELINE_SEARCH_FIELDS,
                    "type": "best_fields",
                }
            }
        )
    return {"bool": {"must": must, "filter": guideline_retrieval_filter()}}


# Age groups → the enrichment facets they correspond to. ``member_age_group``
# arrives as free-ish text ("toddler", "adult", "0-3") while ``life_stage`` is
# a closed vocabulary and the age window is integer months — a lexical match
# between them never fires, so the mapping is explicit.
AGE_GROUP_FACETS: List[tuple] = [
    # (tokens that identify the group, life_stage terms, months_min, months_max)
    (("infant", "baby", "0-1", "0-12"), ["infancy"], 0, 12),
    (("toddler", "1-3", "1-4", "0-3"), ["early_childhood"], 12, 48),
    (("child", "kid", "4-8", "school", "5-12"), ["early_childhood", "school_age"], 48, 144),
    (("teen", "adolescent", "13-18"), ["adolescence"], 144, 216),
    (("adult", "18-64", "19-64"), ["adulthood"], 216, 780),
    (("elder", "older", "senior", "65"), ["older_adulthood"], 780, 1560),
    (("pregnan",), ["pregnancy"], None, None),
    (("lactat", "breastfeed"), ["lactation"], None, None),
]


def age_group_facets(age_group: Optional[str]) -> Optional[tuple]:
    """Resolve an age-group string to (life_stage terms, months_min, months_max)."""
    if not isinstance(age_group, str) or not age_group.strip():
        return None
    lowered = age_group.strip().lower()
    for tokens, life_stages, months_min, months_max in AGE_GROUP_FACETS:
        if any(token in lowered for token in tokens):
            return life_stages, months_min, months_max
    return None


def guideline_age_should_clauses(age_group: Optional[str]) -> List[Dict[str, Any]]:
    """Boosts for rules whose life_stage / age window matches the age group.

    Boost-only, like every context clause. The age-window clause requires the
    rule's window to OVERLAP the group's window; rules with the "not stated"
    sentinel (-1) simply never earn this particular boost.
    """
    resolved = age_group_facets(age_group)
    if not resolved:
        return []
    life_stages, months_min, months_max = resolved

    clauses: List[Dict[str, Any]] = [
        {
            "multi_match": {
                "query": " ".join(life_stages),
                "fields": ["life_stage^4", "target_populations^2"],
                "type": "best_fields",
            }
        }
    ]
    if months_min is not None and months_max is not None:
        clauses.append(
            {
                "bool": {
                    "must": [
                        {"range": {"age_min_months": {"gte": 0, "lte": months_max}}},
                        {"range": {"age_max_months": {"gte": months_min}}},
                    ],
                    "boost": 3.0,
                }
            }
        )
    return clauses


def guideline_context_should_clauses(
    user_context: Optional[QAUserContext],
) -> List[Dict[str, Any]]:
    """
    Soft preferences for guidelines matching the asker's country and age group.

    These only reorder results; nothing here excludes a guideline, so a user
    with no context set still gets the full corpus ranked by relevance.
    """
    if not user_context:
        return []

    clauses: List[Dict[str, Any]] = []
    clauses.extend(guideline_age_should_clauses(user_context.member_age_group))
    geography_terms = [
        term
        for term in [user_context.country, user_context.region]
        if isinstance(term, str) and term.strip()
    ]
    for term in geography_terms:
        clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": [
                        "guide_region^5",
                        "applicable_regions^5",
                        "title",
                    ],
                    "type": "best_fields",
                }
            }
        )

    audience_terms = [
        term
        for term in [
            user_context.member_age_group,
            user_context.experience_group,
        ]
        if isinstance(term, str) and term.strip()
    ]
    for term in audience_terms:
        clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": [
                        "target_populations^3",
                        "life_stage^3",
                        "audience^2",
                        "setting",
                        "notes",
                    ],
                    "type": "best_fields",
                }
            }
        )
    return clauses


def extract_guideline_rule_text(guideline: Dict[str, Any]) -> str:
    rule_text = guideline.get("rule_text") or ""
    if not isinstance(rule_text, str):
        rule_text = str(rule_text)
    return " ".join(rule_text.split()).strip()


def guideline_document_urn(guideline: Dict[str, Any]) -> str:
    for key in ("id", "_id", "urn", "guide_urn"):
        value = guideline.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    rule_text = extract_guideline_rule_text(guideline)
    if rule_text:
        return f"guideline:{uuid.uuid5(uuid.NAMESPACE_URL, rule_text)}"
    return "guideline"


def guideline_tags(guideline: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    for key in ("food_groups", "target_populations"):
        value = guideline.get(key)
        if isinstance(value, list):
            tags.extend(str(item) for item in value if item)
        elif isinstance(value, str) and value.strip():
            tags.append(value.strip())
    region = guideline.get("guide_region")
    if isinstance(region, str) and region.strip():
        tags.append(region.strip())
    return tags


def guideline_publication_year(guideline: Dict[str, Any]) -> Optional[str]:
    for key in ("applicability_start_date", "created_at", "updated_at"):
        value = guideline.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
