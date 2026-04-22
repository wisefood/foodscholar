"""Retriever adapters for QA evidence normalization."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from backend.elastic import ELASTIC_CLIENT
from models.qa import QAClarifierSafetyPlan, QAUserContext, RetrievedSource

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
    ) -> RetrievalResult:
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
    ) -> tuple[List[Dict[str, Any]], List[RetrievedSource], Dict[str, Any]]:
        try:
            query_vector = self.embed_query(query)
            raw_results = ELASTIC_CLIENT.knn_search(
                index_name=self.articles_index,
                query_vector=query_vector,
                k=top_k,
                num_candidates=max(top_k * 20, 100),
                field="embedding",
                filter_query={
                    "bool": {"must_not": {"term": {"status": "deleted"}}}
                },
                source_excludes=["embedding"],
            )
        except Exception as exc:
            logger.error("Article RAG retrieval failed: %s", exc, exc_info=True)
            return [], [], {"ok": False, "error": repr(exc), "used_query": query}

        payloads: List[Dict[str, Any]] = []
        retrieved: List[RetrievedSource] = []
        for result in raw_results:
            result["source_type"] = "article"
            result["retriever"] = self.retriever_name
            result["relevance_score"] = result.get("_score", 0.0)
            payloads.append(result)
            retrieved.append(
                RetrievedSource(
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
                )
            )

        return (
            payloads,
            retrieved,
            {"ok": True, "hit_count": len(payloads), "used_query": query},
        )

    def _retrieve_guidelines(
        self,
        query: str,
        top_k: int,
        user_context: Optional[QAUserContext],
    ) -> tuple[List[Dict[str, Any]], List[RetrievedSource], Dict[str, Any]]:
        if top_k <= 0:
            return [], [], {"ok": True, "hit_count": 0, "used_query": query}

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "rule_text^4",
                                    "title^2",
                                    "notes",
                                    "food_groups^2",
                                    "target_populations",
                                    "guide_region",
                                    "country",
                                    "population",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "must_not": [{"term": {"status": "deleted"}}],
                }
            },
        }
        context_should = guideline_context_should_clauses(user_context)
        if context_should:
            body["query"]["bool"]["should"] = context_should

        try:
            response = ELASTIC_CLIENT.client.search(
                index=self.guidelines_index,
                body=body,
            )
        except Exception as exc:
            logger.error("Guideline RAG retrieval failed: %s", exc, exc_info=True)
            return [], [], {"ok": False, "error": repr(exc), "used_query": query}

        payloads: List[Dict[str, Any]] = []
        retrieved: List[RetrievedSource] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            if not isinstance(source, dict):
                continue

            guideline = {
                **source,
                "_id": hit.get("_id"),
                "_score": hit.get("_score", 0.0),
                "source_type": "guideline",
                "retriever": self.retriever_name,
            }
            rule_text = extract_guideline_rule_text(guideline)
            if len(rule_text) < TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS:
                continue

            urn = guideline_document_urn(guideline)
            guideline["urn"] = urn
            guideline["abstract"] = rule_text
            guideline["description"] = rule_text
            guideline["venue"] = guideline.get("guide_region")
            guideline["relevance_score"] = guideline.get("_score", 0.0)
            guideline["publication_year"] = guideline_publication_year(guideline)

            payloads.append(guideline)
            retrieved.append(
                RetrievedSource(
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
                )
            )

        return (
            payloads,
            retrieved,
            {"ok": True, "hit_count": len(payloads), "used_query": query},
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


def guideline_context_should_clauses(
    user_context: Optional[QAUserContext],
) -> List[Dict[str, Any]]:
    if not user_context:
        return []

    clauses: List[Dict[str, Any]] = []
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
                        "country^5",
                        "source^2",
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
                        "population^3",
                        "reader_group^2",
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
