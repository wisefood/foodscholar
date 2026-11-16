"""Source coverage assessment for the agentic QA retrieval.

Answers one question honestly: can the QA pipeline reach every source in the
corpus (the ``articles`` and ``guidelines`` indexes), and if not, which ones,
and why? Three layers, each broken down by source type, by guide (rules) or
journal (studies), and by publication decade:

1. **Census of gates.** Every document is checked against the conditions that
   make it unreachable whatever the question: the editorial filters the legs
   apply in Elasticsearch, the rule-text floor the guideline normaliser
   applies, the ranking multipliers that zero a score, and the missing vector
   that hides a document from the kNN leg. The checks call the production
   policy functions (``article_policy``, ``qa_retrievers``) on each document,
   so they cannot drift from what the pipeline does.
2. **Reachability probe.** Each eligible document is searched for with
   queries built from itself (its title, its text, its keywords), through the
   real legs, the real reciprocal rank fusion and the real ranking and
   selection (``retrieval.run_branch``, ``ranking.adjust_evidence``,
   ``ranking.select_evidence``). A document that does not come back when asked
   about this directly is, for practical purposes, unreachable; the probe
   says why (no match, outranked, crowded by rules of its own guide, under the
   score floor, over the per-guide cap or the evidence budget).
3. **Concentration.** Real questions are run through both branches, counting
   which documents are ever retrieved and selected: how much of the eligible
   corpus answers actually draw on, and how unevenly (Gini).

Nothing here writes to Elasticsearch. The runner takes the ES client and the
embedding function as arguments, so the tests drive it with fakes.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from models.qa import PlannedSubQuestion
from services.article_policy import (
    TIER_DO_NOT_INDEX,
    VISIBILITY_EXPERT_ONLY,
    VISIBILITY_HIDDEN,
    effective_tier,
    reader_visibility_of,
)
from services.qa_pipeline import ranking, retrieval
from services.qa_pipeline.state import EvidenceItem
from services.qa_retrievers import (
    GUIDELINE_ACTIVE_STATUS,
    QA_GUIDELINE_RAG_TOP_K_MAX,
    TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS,
    extract_guideline_rule_text,
    guideline_hybrid_enabled,
    guideline_publication_year,
    normalize_article_hit,
    normalize_guideline_hit,
)

logger = logging.getLogger(__name__)

ARTICLE = "article"
GUIDELINE = "guideline"
ARTICLES_INDEX = "articles"
GUIDELINES_INDEX = "guidelines"

# Fields the census reads from an article. Abstracts are the bulk; the full
# document (and its vector) is never fetched.
ARTICLE_CENSUS_FIELDS = sorted(
    {
        "urn", "doi", "id", "title", "abstract", "keywords", "tags", "ai_tags",
        "topics", "ai_key_takeaways", "venue", "journal", "publication_year",
        "status", "reader_visibility", "indexing_tier", "ai_indexing_tier",
        "extras.evaluation.indexing_tier", "evaluation.indexing_tier",
        "citationCount", "citation_count", "influentialCitationCount",
        "influential_citation_count", "study_type", "ai_category", "category",
        "open_access",
    }
)

# --------------------------------------------------------------------------- #
# Gates: conditions that make a document unreachable whatever the question.
# ``hard`` gates exclude the document; ``audience`` ones exclude it for some
# readers; ``soft`` ones remove one retrieval leg but leave the other.
# --------------------------------------------------------------------------- #

GATES: Dict[str, Dict[str, str]] = {
    "article.deleted": {
        "kind": "hard",
        "plain": "deleted from the catalogue",
        "code": "src/services/article_policy.py:175-190 (must_not status=deleted, both article legs)",
    },
    "article.hidden": {
        "kind": "hard",
        "plain": "hidden from every reader by an editor",
        "code": "src/services/article_policy.py:156-167 (ES must_not) and src/services/qa_pipeline/ranking.py:323 (is_visible_to)",
    },
    "article.do_not_index": {
        "kind": "hard",
        "plain": "marked 'do not index' by an editor",
        "code": "src/services/article_policy.py:170-172 (ES must_not indexing_tier=do_not_index)",
    },
    "article.do_not_index.scored_zero": {
        "kind": "hard",
        "plain": "marked 'do not index' by the enrichment agent (searched, then scored to zero)",
        "code": "src/services/article_policy.py:45-52,112-138 (tier boost 0.0) and src/services/qa_pipeline/ranking.py:151-162,371,381-383 (below QA_MIN_SCORE)",
    },
    "article.no_text_no_vector": {
        "kind": "hard",
        "plain": "no searchable text and no vector",
        "code": "src/services/qa_pipeline/retrieval.py:48-57 (ARTICLE_LEXICAL_FIELDS) and 310-337 (kNN leg on `embedding`)",
    },
    "article.expert_only": {
        "kind": "audience",
        "plain": "shown to expert readers only",
        "code": "src/services/article_policy.py:156-167 (expert_only excluded unless expertise_level=expert)",
    },
    "article.no_embedding": {
        "kind": "soft",
        "plain": "no vector, so only keyword search can find it",
        "code": "src/services/qa_pipeline/retrieval.py:310-337 (kNN leg on `embedding`)",
    },
    "article.no_lexical_text": {
        "kind": "soft",
        "plain": "no searchable text, so only vector search can find it",
        "code": "src/services/qa_pipeline/retrieval.py:48-57,268-307",
    },
    "guideline.not_active": {
        "kind": "hard",
        "plain": "not active (draft, archived, deprecated or never activated)",
        "code": "src/services/qa_retrievers.py:617,643-645 (filter status=active on both guideline legs)",
    },
    "guideline.short_rule": {
        "kind": "hard",
        "plain": f"rule text under {TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS} characters (dropped by the normaliser)",
        "code": "src/services/qa_retrievers.py:17,259-278 (normalize_guideline_hit returns None at 277)",
    },
    "guideline.no_embedding": {
        "kind": "soft",
        "plain": "no vector, so only keyword search can find it",
        "code": "src/services/qa_pipeline/retrieval.py:375-399,463-465 (kNN leg on `embedding`, hybrid mode only; src/services/qa_retrievers.py:648-659)",
    },
}

# Limits applied per question, not per document: they decide which of the
# matching documents a single answer can use. The probe measures their effect.
PER_QUESTION_LIMITS: Dict[str, str] = {
    "QA_RRF_CANDIDATES": "hits taken from each leg (src/services/qa_pipeline/retrieval.py:81-85)",
    "QA_MIN_SCORE": "score floor after ranking multipliers (src/services/qa_pipeline/ranking.py:371,381-383)",
    "QA_PER_DOC_CAP": "rules per guide / chunks per article in one answer (src/services/qa_pipeline/ranking.py:372,384-388; src/services/qa_pipeline/state.py:40-52)",
    "QA_GUIDELINE_RAG_TOP_K_MAX": "rules per answer (src/services/qa_retrievers.py:18; src/services/qa_pipeline/ranking.py:393-397)",
    "top_k": "studies per answer, request default 5 (src/models/qa.py:345-350; src/services/qa_pipeline/ranking.py:393-397)",
    "knn_response_size": (
        "the kNN call passes k but no `size`, and Elasticsearch returns `size` hits "
        "(default 10), so a vector leg may return fewer than QA_RRF_CANDIDATES "
        "(src/backend/elastic.py:204-223, the search call at 219); the probe records the hits each kNN leg returned"
    ),
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(v) for v in value if v)
    if isinstance(value, dict):
        return " ".join(_text(v) for v in value.values() if v)
    return _WS_RE.sub(" ", str(value)).strip()


def _terms(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [_text(v) for v in value if _text(v)]
    text = _text(value)
    return [text] if text else []


def _nested(doc: Dict[str, Any], dotted: str) -> Any:
    current: Any = doc
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def year_bucket(value: Any) -> str:
    year = ranking.parse_year(value)
    if year is None:
        return "unknown"
    if year < 1990:
        return "before 1990"
    return f"{year // 10 * 10}s"


# --------------------------------------------------------------------------- #
# Document records
# --------------------------------------------------------------------------- #


@dataclass
class DocRecord:
    """One document as the assessment sees it."""

    doc_id: str
    key: str  # the EvidenceItem identity the pipeline deduplicates on
    source_type: str
    stratum: str  # guide URN for a rule, journal for a study
    year_bucket: str
    title: str
    status: str  # eligible | lexical_only | vector_only | audience_limited | unreachable
    gates: List[str] = field(default_factory=list)
    queries: List[Tuple[str, str]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "DocRecord":
        data = dict(data)
        data["queries"] = [tuple(q) for q in data.get("queries") or []]
        return cls(**data)


def _status_from_gates(gates: Sequence[str]) -> str:
    kinds = {GATES[g]["kind"] for g in gates if g in GATES}
    if "hard" in kinds:
        return "unreachable"
    if "audience" in kinds:
        return "audience_limited"
    if "article.no_embedding" in gates or "guideline.no_embedding" in gates:
        return "lexical_only"
    if "article.no_lexical_text" in gates:
        return "vector_only"
    return "eligible"


def article_queries(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Self-queries for a study: its title, its first sentence, its keywords."""
    queries: List[Tuple[str, str]] = []
    title = _text(doc.get("title"))
    if title:
        queries.append(("title", title[:300]))
    abstract = _text(doc.get("abstract") or doc.get("description"))
    if abstract:
        first = _SENTENCE_RE.split(abstract, maxsplit=1)[0]
        queries.append(("text", first[:300]))
    keywords: List[str] = []
    for key in ("keywords", "tags", "ai_tags", "topics"):
        for term in _terms(doc.get(key)):
            if term.lower() not in {k.lower() for k in keywords}:
                keywords.append(term)
    if keywords:
        queries.append(("keywords", " ".join(keywords[:6])[:300]))
    return queries


def guideline_queries(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Self-queries for a rule: its text, its title, its enrichment facets."""
    queries: List[Tuple[str, str]] = []
    rule = extract_guideline_rule_text(doc)
    if rule:
        queries.append(("text", rule[:300]))
    title = _text(doc.get("title"))
    if title and title.lower() != rule.lower():
        queries.append(("title", title[:300]))
    facets: List[str] = []
    for key in ("topic", "food_groups", "nutrients", "health_conditions", "target_populations", "life_stage"):
        for term in _terms(doc.get(key)):
            term = term.replace("_", " ")
            if term.lower() not in {f.lower() for f in facets}:
                facets.append(term)
    if facets:
        queries.append(("keywords", " ".join(facets[:6])[:300]))
    return queries


def article_record(hit: Dict[str, Any], *, has_embedding: Optional[bool]) -> DocRecord:
    """Classify one article hit (``_id`` + ``_source``) against every gate."""
    source = dict(hit.get("_source") or {})
    doc_id = str(hit.get("_id") or source.get("urn") or "")
    raw = {"_id": doc_id, "_score": 0.0, **source}
    payload, retrieved = normalize_article_hit(dict(raw), retriever=retrieval.RETRIEVER_NAME)
    key = EvidenceItem(payload=payload, source=retrieved).key

    gates: List[str] = []
    if source.get("status") == "deleted":
        gates.append("article.deleted")
    visibility = reader_visibility_of(source)
    if visibility == VISIBILITY_HIDDEN:
        gates.append("article.hidden")
    elif visibility == VISIBILITY_EXPERT_ONLY:
        gates.append("article.expert_only")
    tier = effective_tier(source)
    if tier == TIER_DO_NOT_INDEX:
        # The ES filter matches the curated slug exactly; any other spelling,
        # and the agent's proposal, pass the filter and are scored to zero.
        if source.get("indexing_tier") == TIER_DO_NOT_INDEX:
            gates.append("article.do_not_index")
        else:
            gates.append("article.do_not_index.scored_zero")
    has_text = any(_text(_nested(source, f.split("^")[0])) for f in retrieval.ARTICLE_LEXICAL_FIELDS)
    if has_embedding is False:
        gates.append("article.no_embedding")
    if not has_text:
        gates.append("article.no_lexical_text")
        if has_embedding is False:
            gates.append("article.no_text_no_vector")

    venue = _text(source.get("venue") or source.get("journal")) or "(no journal)"
    return DocRecord(
        doc_id=doc_id,
        key=key,
        source_type=ARTICLE,
        stratum=venue,
        year_bucket=year_bucket(source.get("publication_year")),
        title=_text(source.get("title"))[:200],
        status=_status_from_gates(gates),
        gates=gates,
        queries=article_queries(source),
        detail={"tier": tier, "visibility": visibility, "has_embedding": has_embedding},
    )


def guideline_record(hit: Dict[str, Any], *, has_embedding: Optional[bool]) -> DocRecord:
    """Classify one guideline hit against every gate."""
    source = dict(hit.get("_source") or {})
    doc_id = str(hit.get("_id") or source.get("id") or "")
    gates: List[str] = []
    status = _text(source.get("status")) or "(none)"
    if source.get("status") != GUIDELINE_ACTIVE_STATUS:
        gates.append("guideline.not_active")
    normalized = normalize_guideline_hit(
        {"_id": doc_id, "_score": 0.0, "_source": dict(source)},
        retriever=retrieval.RETRIEVER_NAME,
    )
    if normalized is None:
        gates.append("guideline.short_rule")
        key = f"{GUIDELINE}:{source.get('id') or doc_id}"
    else:
        payload, retrieved = normalized
        key = EvidenceItem(payload=payload, source=retrieved).key
    if has_embedding is False:
        gates.append("guideline.no_embedding")

    return DocRecord(
        doc_id=doc_id,
        key=key,
        source_type=GUIDELINE,
        stratum=_text(source.get("guide_urn")) or "(no guide)",
        year_bucket=year_bucket(guideline_publication_year(source)),
        title=extract_guideline_rule_text(source)[:200],
        status=_status_from_gates(gates),
        gates=gates,
        queries=guideline_queries(source),
        detail={"rule_status": status, "has_embedding": has_embedding},
    )


# --------------------------------------------------------------------------- #
# Census
# --------------------------------------------------------------------------- #


def scan_index(
    client: Any,
    index: str,
    *,
    query: Optional[Dict[str, Any]] = None,
    source: Any = True,
    page_size: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """Read every hit of a query with search_after on ``_doc`` (read-only)."""
    body: Dict[str, Any] = {
        "size": page_size,
        "query": query or {"match_all": {}},
        "sort": ["_doc"],
        "_source": source,
    }
    try:
        pit = client.open_point_in_time(index=index, keep_alive="2m")["id"]
    except Exception as exc:
        # No point-in-time (an old cluster): a scroll reads the same documents.
        logger.info("Point-in-time unavailable on %s (%s); scrolling instead", index, exc)
        from elasticsearch.helpers import scan as es_scan

        yield from es_scan(
            client,
            index=index,
            query={"query": body["query"], "_source": source},
            size=page_size,
            preserve_order=False,
        )
        return
    search_after = None
    try:
        while True:
            page = dict(body, pit={"id": pit, "keep_alive": "2m"})
            if search_after is not None:
                page["search_after"] = search_after
            response = client.search(body=page)
            pit = response.get("pit_id", pit)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                yield hit
            search_after = hits[-1].get("sort")
            if search_after is None or len(hits) < page_size:
                break
    finally:
        try:
            client.close_point_in_time(id=pit)
        except Exception:
            pass


def ids_without_embedding(client: Any, index: str) -> Optional[Set[str]]:
    """Ids with no ``embedding`` field; None when the cluster cannot say."""
    try:
        return {
            str(hit.get("_id"))
            for hit in scan_index(
                client,
                index,
                query={"bool": {"must_not": [{"exists": {"field": "embedding"}}]}},
                source=False,
            )
        }
    except Exception as exc:
        logger.warning("Cannot list %s documents without an embedding: %s", index, exc)
        return None


def run_census(client: Any, *, limit: Optional[int] = None) -> List[DocRecord]:
    """Every document of both indexes, classified against every gate."""
    records: List[DocRecord] = []
    for index, build, source in (
        (ARTICLES_INDEX, article_record, {"includes": ARTICLE_CENSUS_FIELDS}),
        (GUIDELINES_INDEX, guideline_record, {"excludes": ["embedding"]}),
    ):
        missing = ids_without_embedding(client, index)
        count = 0
        for hit in scan_index(client, index, source=source):
            has_embedding = None if missing is None else str(hit.get("_id")) not in missing
            records.append(build(hit, has_embedding=has_embedding))
            count += 1
            if limit and count >= limit:
                break
    return records


def summarize_census(records: Sequence[DocRecord]) -> Dict[str, Any]:
    """Totals per type, per gate (with example ids), per stratum and decade."""
    out: Dict[str, Any] = {}
    for source_type in (ARTICLE, GUIDELINE):
        docs = [r for r in records if r.source_type == source_type]
        statuses = Counter(r.status for r in docs)
        gate_counts: Dict[str, Dict[str, Any]] = {}
        for gate, meta in GATES.items():
            if not gate.startswith(source_type):
                continue
            hit = [r for r in docs if gate in r.gates]
            gate_counts[gate] = {
                "count": len(hit),
                "kind": meta["kind"],
                "plain": meta["plain"],
                "code": meta["code"],
                "examples": [r.doc_id for r in hit[:5]],
            }
        # The first hard gate of each unreachable document, so the reasons add up.
        first_hard = Counter(
            next(g for g in r.gates if GATES.get(g, {}).get("kind") == "hard")
            for r in docs
            if r.status == "unreachable"
        )
        by_stratum: Dict[str, Counter] = defaultdict(Counter)
        by_year: Dict[str, Counter] = defaultdict(Counter)
        for r in docs:
            by_stratum[r.stratum][r.status] += 1
            by_year[r.year_bucket][r.status] += 1
        extra: Dict[str, Any] = {}
        if source_type == GUIDELINE:
            extra["rule_status"] = dict(Counter(r.detail.get("rule_status") for r in docs))
            extra["guides_with_no_reachable_rule"] = sorted(
                s for s, c in by_stratum.items() if c.get("unreachable", 0) == sum(c.values())
            )
        out[source_type] = {
            "total": len(docs),
            "status": dict(statuses),
            "reachable": len(docs) - statuses.get("unreachable", 0),
            "gates": gate_counts,
            "unreachable_by_first_gate": dict(first_hard),
            "by_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
            "by_year": {k: dict(v) for k, v in sorted(by_year.items())},
            "embedding_known": all(r.detail.get("has_embedding") is not None for r in docs),
            **extra,
        }
    return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def stratified_sample(records: Sequence[DocRecord], n: Optional[int], *, seed: int = 0) -> List[DocRecord]:
    """At least one document from every stratum, the rest proportionally.

    With ``n`` unset (or not smaller than the population) everything is kept.
    Small guides and journals are never sampled away: each stratum keeps one
    document before any stratum gets a second.
    """
    population = list(records)
    if not n or n >= len(population):
        return population
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str], List[DocRecord]] = defaultdict(list)
    for record in population:
        strata[(record.source_type, record.stratum)].append(record)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = sorted(strata)
    rng.shuffle(keys)
    chosen: List[DocRecord] = []
    # One per stratum first, as far as n allows: every guide before any
    # journal (there are few guides and thousands of journals), then the
    # largest journals.
    for key in sorted(keys, key=lambda k: (k[0] != GUIDELINE, -len(strata[k]))):
        if len(chosen) >= n:
            break
        chosen.append(strata[key].pop())
    remaining = n - len(chosen)
    if remaining > 0:
        pool_total = sum(len(b) for b in strata.values())
        quotas = {k: int(remaining * len(b) / pool_total) if pool_total else 0 for k, b in strata.items()}
        for key, quota in quotas.items():
            chosen.extend(strata[key][:quota])
            strata[key] = strata[key][quota:]
        leftovers = [r for b in strata.values() for r in b]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n - len(chosen)])
    return chosen


# --------------------------------------------------------------------------- #
# Reachability probe
# --------------------------------------------------------------------------- #


def classify_drop(
    target_key: str,
    adjusted: Sequence[EvidenceItem],
    selected: Sequence[EvidenceItem],
    *,
    min_score: float,
    per_doc_cap: int,
) -> Optional[str]:
    """Why a retrieved document did not make the selection (None if it did).

    Walks the adjusted items in the order ``ranking.select_evidence`` does; the
    selection itself always comes from the real function.
    """
    if any(item.key == target_key for item in selected):
        return None
    counts: Dict[str, int] = {}
    for item in sorted(adjusted, key=lambda i: i.adjusted_score, reverse=True):
        if item.adjusted_score < min_score:
            if item.key == target_key:
                return "below_threshold"
            continue
        parent = item.parent_doc_key
        if counts.get(parent, 0) >= per_doc_cap:
            if item.key == target_key:
                return "over_doc_cap"
            continue
        counts[parent] = counts.get(parent, 0) + 1
        if item.key == target_key:
            return "over_budget"
    return "not_retrieved"


def _parent_of(item: EvidenceItem) -> str:
    return item.parent_doc_key


@dataclass
class ProbeContext:
    """What the probe calls: production functions unless a test swaps them."""

    run_branch: Callable[..., Any] = retrieval.run_branch
    adjust: Callable[..., List[EvidenceItem]] = ranking.adjust_evidence
    select: Callable[..., Tuple[List[EvidenceItem], Dict[str, int]]] = ranking.select_evidence
    embed: Optional[Callable[[List[str]], List[Optional[List[float]]]]] = None
    expertise_level: str = "intermediate"
    top_k: int = 5

    def min_score(self) -> float:
        return ranking._setting("QA_MIN_SCORE", 0.05)

    def per_doc_cap(self) -> int:
        return max(int(ranking._setting("QA_PER_DOC_CAP", 2)), 1)


def probe_document(record: DocRecord, ctx: ProbeContext) -> Dict[str, Any]:
    """Search for one document with its own queries; report what came back."""
    branch = "guidelines" if record.source_type == GUIDELINE else "articles"
    expertise = "expert" if record.status == "audience_limited" else ctx.expertise_level
    texts = [text for _, text in record.queries]
    vectors: List[Optional[List[float]]] = [None] * len(texts)
    if ctx.embed and texts:
        try:
            vectors = list(ctx.embed(texts))
        except Exception as exc:
            logger.warning("Embedding failed for %s; lexical only: %s", record.doc_id, exc)
    candidates = retrieval._candidates()
    # Whether the vector leg could have found this document at all.
    vector_visible = (
        record.detail.get("has_embedding") is not False
        and any(v is not None for v in vectors)
        and (record.source_type == ARTICLE or guideline_hybrid_enabled())
    )
    results: List[Dict[str, Any]] = []
    for (kind, text), vector in zip(record.queries, vectors):
        sq = PlannedSubQuestion(
            id=f"probe-{kind}",
            text=text,
            why="coverage probe",
            branch=branch,
            lexical_query=text,
            dense_query=text,
        )
        outcome = ctx.run_branch(
            sq, branch=branch, vector=vector, user_context=None, expertise_level=expertise
        )
        fused = list(outcome.items)
        rank = next((i for i, item in enumerate(fused, 1) if item.key == record.key), None)
        adjusted = ctx.adjust(list(fused), expertise_level=expertise)
        selected, _dropped = ctx.select(adjusted, top_k=ctx.top_k)
        drop = classify_drop(
            record.key, adjusted, selected, min_score=ctx.min_score(), per_doc_cap=ctx.per_doc_cap()
        )
        legs = dict(outcome.status.get("legs") or {})
        same_parent = 0.0
        if record.source_type == GUIDELINE and fused:
            top = fused[:candidates]
            same_parent = sum(
                1 for item in top if item.payload.get("guide_urn") == record.stratum and item.key != record.key
            ) / len(top)
        results.append(
            {
                "kind": kind,
                "fused_rank": rank,
                "fused_size": len(fused),
                "lexical_hits": legs.get("lexical"),
                "knn_hits": legs.get("knn"),
                "leg_errors": {k: v for k, v in legs.items() if k.endswith("_error")},
                "selected": drop is None,
                "drop_reason": drop,
                "same_guide_share": round(same_parent, 3),
            }
        )
    ranks = [r["fused_rank"] for r in results if r["fused_rank"]]
    reachable = any(r["selected"] for r in results)
    return {
        "id": record.doc_id,
        "key": record.key,
        "source_type": record.source_type,
        "stratum": record.stratum,
        "year_bucket": record.year_bucket,
        "status": record.status,
        "queries": results,
        "best_rank": min(ranks) if ranks else None,
        "reachable": reachable,
        "reason": None if reachable else unreachable_reason(
            results, candidates=candidates, vector_visible=vector_visible
        ),
    }


def unreachable_reason(
    results: Sequence[Dict[str, Any]], *, candidates: int, vector_visible: bool = True
) -> str:
    """The most telling reason across a document's queries.

    A document missing from a full leg was outranked: the leg had more matches
    than it keeps. Missing from a short keyword leg, with no vector leg that
    could have seen it, its own words did not match it (an indexing problem).
    """
    if not results:
        return "no_query"  # nothing to search with: no title, text or keywords
    drops = [r["drop_reason"] for r in results if r["fused_rank"]]
    for reason in ("over_doc_cap", "below_threshold", "over_budget"):
        if reason in drops:
            return reason
    if any(r["leg_errors"] for r in results) and not any(r["fused_size"] for r in results):
        return "search_error"
    crowded = [
        r
        for r in results
        if (r["lexical_hits"] or 0) >= candidates or (vector_visible and (r["knn_hits"] or 0) >= 1)
    ]
    if crowded:
        if any(r["same_guide_share"] >= 0.5 for r in crowded):
            return "outranked_by_same_guide"
        return "outranked"
    return "no_match"


def _load_done(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by an interrupted run
            done[f"{row.get('source_type')}:{row.get('id')}"] = row
    return done


def run_probe(
    records: Sequence[DocRecord],
    ctx: ProbeContext,
    *,
    results_path: str,
    workers: int = 4,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Probe every record not already in ``results_path``; append as it goes."""
    done = _load_done(results_path)
    todo = [r for r in records if f"{r.source_type}:{r.doc_id}" not in done]
    lock = threading.Lock()
    finished = [0]
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as out:

        def work(record: DocRecord) -> None:
            try:
                row = probe_document(record, ctx)
            except Exception as exc:  # one bad document never stops the run
                logger.exception("Probe failed for %s", record.doc_id)
                row = {
                    "id": record.doc_id,
                    "key": record.key,
                    "source_type": record.source_type,
                    "stratum": record.stratum,
                    "year_bucket": record.year_bucket,
                    "status": record.status,
                    "queries": [],
                    "best_rank": None,
                    "reachable": False,
                    "reason": "probe_error",
                    "error": repr(exc),
                }
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done[f"{row['source_type']}:{row['id']}"] = row
                finished[0] += 1
                if progress:
                    progress(finished[0], len(todo))

        with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as pool:
            list(pool.map(work, todo))
    wanted = {f"{r.source_type}:{r.doc_id}" for r in records}
    return [row for k, row in done.items() if k in wanted]


def vector_population(records: Sequence[DocRecord]) -> Dict[str, Optional[int]]:
    """Documents a kNN leg could return to the default reader, per type.

    Expert-only studies are left out (a conservative count: the cap check
    must not fire just because the reader's filter hides some vectors).
    None when the census could not tell which documents have vectors.
    """
    out: Dict[str, Optional[int]] = {}
    for source_type in (ARTICLE, GUIDELINE):
        docs = [
            r
            for r in records
            if r.source_type == source_type and r.status not in ("unreachable", "audience_limited")
        ]
        if any(r.detail.get("has_embedding") is None for r in docs):
            out[source_type] = None
        else:
            out[source_type] = sum(1 for r in docs if r.detail.get("has_embedding"))
    return out


def summarize_probe(
    rows: Sequence[Dict[str, Any]],
    *,
    candidates: int,
    vectors: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Recall at k, reachability and reasons, per type, stratum and decade.

    ``vectors`` (from ``vector_population``) lets the kNN check tell a leg cut
    short by the response size from a corpus that simply has fewer vectors.
    """
    ks = sorted({1, 5, 10, candidates})
    out: Dict[str, Any] = {}
    knn_response: Dict[str, Any] = {}
    for source_type in (ARTICLE, GUIDELINE):
        counts = [
            q["knn_hits"]
            for r in rows
            if r["source_type"] == source_type
            for q in r.get("queries", [])
            if isinstance(q.get("knn_hits"), int)
        ]
        if not counts:
            continue
        available = (vectors or {}).get(source_type)
        expected = candidates if available is None else min(candidates, available)
        knn_response[source_type] = {
            "requested_k": candidates,
            "max_hits_seen": max(counts),
            "documents_with_vectors": available,
            "capped_below_k": max(counts) < expected,
            "certain": available is not None,
        }
    for source_type in (ARTICLE, GUIDELINE):
        docs = [r for r in rows if r["source_type"] == source_type]
        n = len(docs)

        def share(pred: Callable[[Dict[str, Any]], bool], items: Sequence[Dict[str, Any]]) -> float:
            return round(sum(1 for r in items if pred(r)) / len(items), 4) if items else 0.0

        by_stratum: Dict[str, Dict[str, Any]] = {}
        for stratum, items in _group(docs, "stratum").items():
            by_stratum[stratum] = {
                "probed": len(items),
                "reachable": share(lambda r: r["reachable"], items),
                f"recall@{candidates}": share(lambda r: bool(r["best_rank"]) and r["best_rank"] <= candidates, items),
            }
        by_year = {
            bucket: {"probed": len(items), "reachable": share(lambda r: r["reachable"], items)}
            for bucket, items in _group(docs, "year_bucket").items()
        }
        unreachable = [r for r in docs if not r["reachable"]]
        out[source_type] = {
            "probed": n,
            "reachable": share(lambda r: r["reachable"], docs),
            "recall": {
                f"@{k}": share(lambda r, k=k: bool(r["best_rank"]) and r["best_rank"] <= k, docs) for k in ks
            },
            "reasons": dict(Counter(r["reason"] for r in unreachable)),
            "unreachable": [
                {"id": r["id"], "stratum": r["stratum"], "reason": r["reason"], "best_rank": r["best_rank"]}
                for r in unreachable
            ],
            "by_stratum": by_stratum,
            "strata_fully_unreachable": sorted(s for s, v in by_stratum.items() if v["reachable"] == 0.0),
            "by_year": by_year,
        }
    out["candidates"] = candidates
    out["knn_response"] = knn_response
    return out


def _group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------- #
# Concentration on real questions
# --------------------------------------------------------------------------- #


def load_questions(path: str) -> List[str]:
    """Plain lines, JSONL, or a JSON list; objects use question/text/prompt."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    stripped = raw.strip()
    items: List[Any] = []
    if stripped.startswith("["):
        items = json.loads(stripped)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    items.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
            items.append(line)
    questions: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("question") or item.get("text") or item.get("prompt")
        else:
            text = item
        text = _text(text)
        if text:
            questions.append(text)
    return questions


def gini(values: Sequence[float]) -> float:
    """0 when every document is used equally, towards 1 when a few take all."""
    data = sorted(float(v) for v in values if v is not None)
    n = len(data)
    total = sum(data)
    if n == 0 or total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(data))
    return round((2 * cumulative) / (n * total) - (n + 1) / n, 4)


def run_concentration(
    questions: Sequence[str],
    records: Sequence[DocRecord],
    ctx: ProbeContext,
    *,
    workers: int = 4,
) -> Dict[str, Any]:
    """Run each question through both branches; count what answers can use."""
    eligible = {r.key: r for r in records if r.status != "unreachable"}
    retrieved: Counter = Counter()
    selected: Counter = Counter()
    lock = threading.Lock()

    def work(question: str) -> None:
        vector = None
        if ctx.embed:
            try:
                vector = ctx.embed([question])[0]
            except Exception as exc:
                logger.warning("Embedding failed for a question; lexical only: %s", exc)
        pool: List[EvidenceItem] = []
        for branch in ("articles", "guidelines"):
            sq = PlannedSubQuestion(
                id=f"q-{branch}", text=question, why="coverage", branch=branch,
                lexical_query=question, dense_query=question,
            )
            outcome = ctx.run_branch(
                sq, branch=branch, vector=vector, user_context=None, expertise_level=ctx.expertise_level
            )
            pool = retrieval.merge_evidence(pool, list(outcome.items))
        adjusted = ctx.adjust(list(pool), expertise_level=ctx.expertise_level)
        chosen, _ = ctx.select(adjusted, top_k=ctx.top_k)
        with lock:
            retrieved.update({item.key for item in pool})
            selected.update({item.key for item in chosen})

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        list(executor.map(work, questions))

    out: Dict[str, Any] = {"questions": len(questions)}
    for source_type in (ARTICLE, GUIDELINE):
        universe = [r for r in eligible.values() if r.source_type == source_type]
        keys = {r.key for r in universe}
        strata = {r.stratum for r in universe}
        used_strata = {eligible[k].stratum for k in selected if k in keys}
        frequency = [selected.get(r.key, 0) for r in universe]
        top = [
            {"key": k, "title": eligible[k].title, "stratum": eligible[k].stratum, "times_selected": c}
            for k, c in selected.most_common()
            if k in keys
        ][:20]
        out[source_type] = {
            "eligible": len(universe),
            "distinct_retrieved": len([k for k in retrieved if k in keys]),
            "distinct_selected": len([k for k in selected if k in keys]),
            "gini_selected": gini(frequency),
            "top_selected": top,
            "strata": len(strata),
            "strata_never_selected": len(strata - used_strata),
            "strata_never_selected_share": round(len(strata - used_strata) / len(strata), 4) if strata else 0.0,
            "unknown_keys_selected": len([k for k in selected if k.startswith(source_type) and k not in eligible]),
        }
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _pct(part: float, whole: float) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "n/a"


_SINGULAR = {"studies": "study", "guide rules": "guide rule", "documents": "document", "guides": "guide", "journals": "journal"}


def _n(count: int, plural: str) -> str:
    return f"{count:,} {_SINGULAR.get(plural, plural) if count == 1 else plural}"


def render_markdown(report: Dict[str, Any]) -> str:
    """Plain words first, then the tables an engineer needs."""
    census = report.get("census") or {}
    probe = report.get("probe") or {}
    conc = report.get("concentration") or {}
    lines: List[str] = ["# Can FoodScholar's answers reach every source?", ""]

    a = census.get(ARTICLE, {})
    g = census.get(GUIDELINE, {})
    if census:
        lines.append(
            f"FoodScholar's answers can draw on **{a.get('reachable', 0):,} of {a.get('total', 0):,} studies** "
            f"({_pct(a.get('reachable', 0), a.get('total', 0))}) and **{g.get('reachable', 0):,} of "
            f"{g.get('total', 0):,} guide rules** ({_pct(g.get('reachable', 0), g.get('total', 0))})."
        )
        never = []
        for source_type, label in ((ARTICLE, "studies"), (GUIDELINE, "guide rules")):
            reasons = census.get(source_type, {}).get("unreachable_by_first_gate") or {}
            if reasons:
                parts = ", ".join(
                    f"{count:,} {GATES[gate]['plain']}" for gate, count in sorted(reasons.items(), key=lambda x: -x[1])
                )
                never.append(f"{_n(sum(reasons.values()), label)} can never be found, whatever the question: {parts}.")
        lines.extend(never or ["Nothing is excluded outright."])
        limited = (a.get("status") or {}).get("audience_limited", 0)
        if limited:
            lines.append(f"{_n(limited, 'studies')} {'is' if limited == 1 else 'are'} shown to expert readers only.")
        lexical = (a.get("status") or {}).get("lexical_only", 0) + (g.get("status") or {}).get("lexical_only", 0)
        if lexical:
            lines.append(
                f"{_n(lexical, 'documents')} {'has' if lexical == 1 else 'have'} no vector, so only a keyword "
                "match can find them, never a question worded differently."
            )
        if not report.get("config", {}).get("guideline_hybrid", True):
            lines.append("Guideline vector search is switched off: every rule is found by keywords only.")
        lines.append("")

    if probe:
        cand = probe.get("candidates")
        for source_type, label in ((ARTICLE, "studies"), (GUIDELINE, "guide rules")):
            p = probe.get(source_type) or {}
            if not p.get("probed"):
                continue
            lines.append(
                f"Asked about directly (by their own title, text or keywords), {_pct(p['reachable'], 1)} of the "
                f"{_n(p['probed'], label)} tested come back into an answer's sources; "
                f"{_pct(p['recall'].get(f'@{cand}', 0), 1)} are at least among the {cand} candidates searched."
            )
            reasons = p.get("reasons") or {}
            if reasons:
                lines.append(
                    "  The rest: " + ", ".join(f"{c:,} {REASON_PLAIN.get(r, r)}" for r, c in sorted(reasons.items(), key=lambda x: -x[1])) + "."
                )
            if p.get("strata_fully_unreachable"):
                what = "guides" if source_type == GUIDELINE else "journals"
                lines.append(f"  {_n(len(p['strata_fully_unreachable']), what)} had nothing come back at all.")
        for source_type, knn in (probe.get("knn_response") or {}).items():
            if knn.get("capped_below_k"):
                label = "studies" if source_type == ARTICLE else "guide rules"
                lines.append(
                    f"The vector search over {label} never returned more than {knn['max_hits_seen']} hits although "
                    f"{knn['requested_k']} were asked for"
                    + (f" and {knn['documents_with_vectors']:,} have vectors" if knn.get("certain") else "")
                    + (
                        ": Elasticsearch's default page size of 10 is likely capping it"
                        if knn.get("certain")
                        else ": Elasticsearch's default page size of 10 may be capping it (the census could "
                        "not count vectors to confirm)"
                    )
                    + " (src/backend/elastic.py:219 sends no `size`)."
                )
        lines.append("")

    if conc:
        for source_type, label in ((ARTICLE, "studies"), (GUIDELINE, "guide rules")):
            c = conc.get(source_type) or {}
            if not c:
                continue
            what = "guides" if source_type == GUIDELINE else "journals"
            lines.append(
                f"Across {conc.get('questions', 0):,} real questions, answers used {c['distinct_selected']:,} distinct "
                f"{label} ({_pct(c['distinct_selected'], c['eligible'])} of those available); "
                f"{c['strata_never_selected']:,} of {c['strata']:,} {what} were never used "
                f"(Gini {c['gini_selected']}: 0 is even use, 1 is a few sources taking everything)."
            )
        lines.append("")

    lines.extend(["## Gates", "", "| Gate | Kind | Documents | Where |", "|---|---|---|---|"])
    for source_type in (ARTICLE, GUIDELINE):
        for gate, data in (census.get(source_type, {}).get("gates") or {}).items():
            lines.append(f"| {data['plain']} | {data['kind']} | {data['count']:,} | `{data['code']}` |")
    lines.extend(["", "## Per-question limits", ""])
    for name, text in PER_QUESTION_LIMITS.items():
        value = (report.get("config", {}).get("ranking") or {}).get(name)
        lines.append(f"- **{name}**{f' = {value}' if value is not None else ''}: {text}")
    return "\n".join(lines) + "\n"


REASON_PLAIN = {
    "no_match": "matched none of their own words",
    "outranked": "were pushed out by other sources",
    "outranked_by_same_guide": "were pushed out by other rules of the same guide",
    "below_threshold": "scored under the minimum after ranking",
    "over_doc_cap": "lost to two other rules of the same guide (one answer takes at most two per guide)",
    "over_budget": "were found but fell outside an answer's source budget",
    "search_error": "hit a search error",
    "probe_error": "could not be tested",
    "no_query": "had no title, text or keywords to search with",
}


def runtime_config() -> Dict[str, Any]:
    """The settings that decide reach, recorded with every report."""
    settings = dict(ranking.ranking_config())
    settings["QA_GUIDELINE_RAG_TOP_K_MAX"] = QA_GUIDELINE_RAG_TOP_K_MAX
    return {
        "ranking": settings,
        "candidates": retrieval._candidates(),
        "guideline_hybrid": guideline_hybrid_enabled(),
    }
