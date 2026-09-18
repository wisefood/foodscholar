"""The knowledge graph browse index: mapping, lifecycle, and every read query.

One Elasticsearch index holds one document per shelf, theme and card. It is a
denormalized projection of the graph in Neo4j, written by services/kg_projector
once per rebuild and never by a request.

One index rather than three because a single search box should rank shelves,
themes and cards together, and narrowing to one kind is then a term filter. The
aggregations that drive the filter panel come out of the same request as the
hits.

Two fields in the mapping do most of the work:

  ancestor_ids  The full chain from the root, materialized at projection time.
                "Everything under this shelf" becomes one term filter instead
                of a variable-length graph traversal, and it makes breadcrumbs,
                scoped search and scoped facet counts free.

  label.autocomplete
                A search_as_you_type field, so typing "olive" matches "extra
                virgin olive oil". The completion suggester is faster but
                prefix-only, which is wrong for labels whose distinguishing
                word sits in the middle — which is most FoodOn labels.

This module is the only place that knows the mapping or writes a query against
it. Callers pass filters and get documents back.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import config
from exceptions import InvalidError, ServiceUnavailableError

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
except Exception as exc:  # pragma: no cover - deployment shape
    Elasticsearch = None
    _ES_IMPORT_ERROR = exc


KINDS = ("shelf", "theme", "card")

FACETS = (
    "foods",
    "health",
    "sustainability",
    "dietary_patterns",
    "allergies",
    "nutrients",
)

#: Fields returned for list/stream responses. Deliberately narrow: the detail
#: routes ask for the whole document, everything else does not need to.
SUMMARY_SOURCE = [
    "kind",
    "node_id",
    "label",
    "facet",
    "depth",
    "chunk_count",
    "has_card",
    "child_count",
    "theme_count",
    "status",
    "x",
    "y",
    "degree",
]

#: Sort for cursor pagination. The node_id tiebreak is what makes search_after
#: correct — without a unique final key, a page boundary can drop or repeat a
#: document.
LIST_SORT: List[Dict[str, str]] = [
    {"chunk_count": "desc"},
    {"node_id": "asc"},
]

RELEVANCE_SORT: List[Any] = ["_score", {"chunk_count": "desc"}, {"node_id": "asc"}]


BROWSE_MAPPING: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                # Folds accents, so "acai" finds "açaí" and "puree" finds
                # "purée". A food corpus is full of both spellings.
                "label_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    },
    # strict, not false: an unexpected field should fail the projection loudly
    # rather than be auto-inferred as text and quietly break sorts and aggs.
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "kind": {"type": "keyword"},
            "node_id": {"type": "keyword"},
            "label": {
                "type": "text",
                "analyzer": "label_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 512},
                    "autocomplete": {
                        "type": "search_as_you_type",
                        "analyzer": "label_analyzer",
                    },
                },
            },
            "description": {"type": "text", "analyzer": "label_analyzer"},
            "facet": {"type": "keyword"},
            "status": {"type": "keyword"},
            "depth": {"type": "integer"},
            "parent_id": {"type": "keyword"},
            "ancestor_ids": {"type": "keyword"},
            "shelf_ids": {"type": "keyword"},
            "chunk_count": {"type": "integer"},
            "support_direct": {"type": "integer"},
            "support_lifted": {"type": "integer"},
            "foodon_id": {"type": "keyword"},
            "see_also": {"type": "keyword"},
            "keyword_terms": {"type": "keyword"},
            "discovered_by": {"type": "keyword"},
            "discovery_pass": {"type": "keyword"},
            "target_type": {"type": "keyword"},
            "target_id": {"type": "keyword"},
            "evidence_quality": {"type": "keyword"},
            "safety_flagged": {"type": "boolean"},
            "controversy_note": {"type": "text", "index": False},
            "confidence_note": {"type": "text", "index": False},
            "tip": {"type": "text", "index": False},
            "cited_chunk_ids": {"type": "keyword", "index": False},
            "has_card": {"type": "boolean"},
            "child_count": {"type": "integer"},
            "theme_count": {"type": "integer"},
            # Layout computed once by the projector, so a browser never runs a
            # force simulation and the map is the same every visit.
            "x": {"type": "float"},
            "y": {"type": "float"},
            "degree": {"type": "integer"},
            "indexed_at": {"type": "date"},
            "graph_version": {"type": "keyword"},
        },
    },
}


# ---------------------------------------------------------------- client


class _BrowseClient:
    """Thread-safe singleton client for the knowledge graph cluster.

    Separate from backend.elastic's singleton because the graph may live on a
    different cluster and carries its own credentials. Same cluster in the
    default deployment; the separation costs one connection pool and saves a
    migration later.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._client = None
                    cls._instance = instance
        return cls._instance

    @property
    def client(self):
        if self._client is None:
            if Elasticsearch is None:  # pragma: no cover
                raise ServiceUnavailableError(
                    detail="elasticsearch is required for knowledge graph browsing.",
                    extra={"title": "KnowledgeGraphUnavailable"},
                ) from _ES_IMPORT_ERROR
            s = config.settings
            kwargs: Dict[str, Any] = {}
            if s["KG_ES_API_KEY"]:
                kwargs["api_key"] = s["KG_ES_API_KEY"]
            elif s["KG_ES_USERNAME"]:
                kwargs["basic_auth"] = (s["KG_ES_USERNAME"], s["KG_ES_PASSWORD"])
            self._client = Elasticsearch(hosts=s["KG_ES_URL"], **kwargs)
            logger.info("Knowledge graph browse client at %s", s["KG_ES_URL"])
        return self._client


def client():
    return _BrowseClient().client


def alias() -> str:
    return config.settings["KG_BROWSE_ALIAS"]


def versioned_index(graph_version: str) -> str:
    """Physical index name for one projection run."""
    safe = "".join(c for c in graph_version.lower() if c.isalnum() or c in "-_")
    return f"{alias()}_{safe or 'unversioned'}"


# ---------------------------------------------------------------- lifecycle


def alias_exists() -> bool:
    try:
        return bool(client().indices.exists_alias(name=alias()))
    except Exception:
        return False


def create_index(index_name: str) -> None:
    """Create one physical index with the browse mapping. Idempotent."""
    es = client()
    if es.indices.exists(index=index_name):
        logger.info("Browse index %s already exists", index_name)
        return
    es.indices.create(index=index_name, body=BROWSE_MAPPING)
    logger.info("Created browse index %s", index_name)


def swap_alias(index_name: str) -> List[str]:
    """Point the read alias at `index_name`, atomically.

    Returns the indices that were detached, so the caller can decide whether to
    delete them. The swap is one ES action: a reader is never left pointing at
    nothing, and rolling back is running this again with the old name.
    """
    es = client()
    actions: List[Dict[str, Any]] = []
    detached: List[str] = []
    try:
        current = es.indices.get_alias(name=alias())
        for existing in current:
            if existing != index_name:
                actions.append({"remove": {"index": existing, "alias": alias()}})
                detached.append(existing)
    except Exception:
        # No alias yet — the first projection run.
        pass
    actions.append({"add": {"index": index_name, "alias": alias()}})
    es.indices.update_aliases(body={"actions": actions})
    logger.info("Browse alias %s now points at %s", alias(), index_name)
    return detached


def delete_index(index_name: str) -> None:
    client().indices.delete(index=index_name, ignore_unavailable=True)
    logger.info("Deleted browse index %s", index_name)


def refresh() -> None:
    client().indices.refresh(index=alias())


def count(filters: Optional["BrowseFilters"] = None) -> int:
    body = {"query": _query(filters or BrowseFilters())}
    return int(client().count(index=alias(), body=body)["count"])


def indexed_graph_version() -> Optional[str]:
    """Which build the browse index currently holds.

    Read off one document rather than tracked separately, so it cannot drift
    from the data it describes.
    """
    try:
        resp = client().search(
            index=alias(),
            body={"size": 1, "_source": ["graph_version", "indexed_at"], "query": {"match_all": {}}},
        )
        hits = resp["hits"]["hits"]
        return hits[0]["_source"].get("graph_version") if hits else None
    except Exception:
        return None


# ---------------------------------------------------------------- filters


@dataclass
class BrowseFilters:
    """The filter panel, as a value.

    Every field narrows; none widens. An empty instance matches the whole
    graph, which is why the stream endpoints impose their own defaults.
    """

    q: Optional[str] = None
    kind: Sequence[str] = field(default_factory=tuple)
    facet: Sequence[str] = field(default_factory=tuple)
    status: Sequence[str] = field(default_factory=tuple)
    under: Optional[str] = None
    depth_max: Optional[int] = None
    depth: Optional[int] = None
    min_chunks: Optional[int] = None
    discovered_by: Sequence[str] = field(default_factory=tuple)
    evidence_quality: Sequence[str] = field(default_factory=tuple)
    has_card: Optional[bool] = None
    roots_only: bool = False
    parent_id: Optional[str] = None
    shelf_id: Optional[str] = None
    target_id: Optional[str] = None

    def validate(self) -> "BrowseFilters":
        for value in self.kind:
            if value not in KINDS:
                raise InvalidError(
                    detail=f"Unknown kind {value!r}; expected one of {', '.join(KINDS)}."
                )
        for value in self.facet:
            if value not in FACETS:
                raise InvalidError(
                    detail=f"Unknown facet {value!r}; expected one of {', '.join(FACETS)}."
                )
        return self


def _must_not_clauses(f: BrowseFilters) -> List[Dict[str, Any]]:
    """Negative clauses. Only roots need one so far."""
    if not f.roots_only:
        return []
    # The library calls a shelf a root when it has no parent, not when its
    # depth is zero. Those usually agree, and when they disagree the parent
    # pointer is the one that decides what the tree looks like.
    return [{"exists": {"field": "parent_id"}}]


def _filter_clauses(f: BrowseFilters) -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if f.kind:
        clauses.append({"terms": {"kind": list(f.kind)}})
    if f.facet:
        clauses.append({"terms": {"facet": list(f.facet)}})
    if f.status:
        clauses.append({"terms": {"status": list(f.status)}})
    if f.under:
        # The materialized path. One term lookup for a whole subtree.
        clauses.append({"term": {"ancestor_ids": f.under}})
    if f.parent_id:
        clauses.append({"term": {"parent_id": f.parent_id}})
    if f.shelf_id:
        clauses.append({"term": {"shelf_ids": f.shelf_id}})
    if f.target_id:
        clauses.append({"term": {"target_id": f.target_id}})
    if f.depth is not None:
        clauses.append({"term": {"depth": f.depth}})
    if f.depth_max is not None:
        clauses.append({"range": {"depth": {"lte": f.depth_max}}})
    if f.min_chunks is not None:
        clauses.append({"range": {"chunk_count": {"gte": f.min_chunks}}})
    if f.discovered_by:
        clauses.append({"terms": {"discovered_by": list(f.discovered_by)}})
    if f.evidence_quality:
        clauses.append({"terms": {"evidence_quality": list(f.evidence_quality)}})
    if f.has_card is not None:
        clauses.append({"term": {"has_card": f.has_card}})
    return clauses


def _query(f: BrowseFilters) -> Dict[str, Any]:
    clauses = _filter_clauses(f)
    must_not = _must_not_clauses(f)
    if not f.q:
        if not clauses and not must_not:
            return {"match_all": {}}
        return {"bool": {"filter": clauses, "must_not": must_not}}
    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": f.q,
                        "fields": ["label^3", "description"],
                        "fuzziness": "AUTO",
                    }
                }
            ],
            "filter": clauses,
            "must_not": must_not,
        }
    }


AGGREGATIONS: Dict[str, Any] = {
    "by_facet": {"terms": {"field": "facet", "size": 10}},
    "by_kind": {"terms": {"field": "kind", "size": 5}},
    "by_status": {"terms": {"field": "status", "size": 5}},
    "by_evidence_quality": {"terms": {"field": "evidence_quality", "size": 10}},
    "by_discovered_by": {"terms": {"field": "discovered_by", "size": 10}},
    "by_depth": {"histogram": {"field": "depth", "interval": 1, "min_doc_count": 1}},
}


def _buckets(aggs: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for name, agg in (aggs or {}).items():
        out[name] = {
            str(b["key"]): int(b["doc_count"]) for b in agg.get("buckets", [])
        }
    return out


# ---------------------------------------------------------------- cursors


def encode_cursor(sort_values: Sequence[Any]) -> str:
    raw = json.dumps(list(sort_values), separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: Optional[str]) -> Optional[List[Any]]:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        values = json.loads(raw)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise InvalidError(detail="Malformed pagination cursor.") from exc
    if not isinstance(values, list):
        raise InvalidError(detail="Malformed pagination cursor.")
    return values


# ---------------------------------------------------------------- reads


def _search(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return client().search(index=alias(), body=body)
    except Exception as exc:
        # A missing alias is the common case and deserves its own message: it
        # means the projector has never run, not that browsing is broken.
        if not alias_exists():
            raise ServiceUnavailableError(
                detail=(
                    "The knowledge graph browse index has not been built yet. "
                    "Run POST /api/v1/graph/reindex."
                ),
                extra={"title": "BrowseIndexMissing"},
            ) from exc
        raise


def get_node(node_id: str) -> Optional[Dict[str, Any]]:
    """One document by its graph id, whatever kind it is."""
    resp = _search(
        {"size": 1, "query": {"bool": {"filter": [{"term": {"node_id": node_id}}]}}}
    )
    hits = resp["hits"]["hits"]
    return hits[0]["_source"] if hits else None


def get_nodes(node_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Several documents by id, in one round-trip, keyed by node_id."""
    ids = [i for i in node_ids if i]
    if not ids:
        return {}
    resp = _search(
        {
            "size": len(ids),
            "query": {"bool": {"filter": [{"terms": {"node_id": ids}}]}},
        }
    )
    return {h["_source"]["node_id"]: h["_source"] for h in resp["hits"]["hits"]}


def page(
    filters: BrowseFilters,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    source: Optional[Sequence[str]] = None,
    with_aggs: bool = False,
) -> Dict[str, Any]:
    """One page of documents, with a cursor for the next.

    Sorted by relevance when there is a query and by size otherwise, always
    with a unique tiebreak so the cursor is stable.
    """
    filters.validate()
    sort = RELEVANCE_SORT if filters.q else LIST_SORT
    body: Dict[str, Any] = {
        "size": limit,
        "query": _query(filters),
        "sort": sort,
        "_source": list(source) if source else list(SUMMARY_SOURCE),
        "track_total_hits": True,
    }
    after = decode_cursor(cursor)
    if after is not None:
        body["search_after"] = after
    if with_aggs:
        body["aggs"] = AGGREGATIONS

    resp = _search(body)
    hits = resp["hits"]["hits"]
    next_cursor = (
        encode_cursor(hits[-1]["sort"]) if len(hits) == limit and hits else None
    )
    return {
        "items": [h["_source"] for h in hits],
        "total": int(resp["hits"]["total"]["value"]),
        "next_cursor": next_cursor,
        "facets": _buckets(resp.get("aggregations", {})) if with_aggs else {},
    }


def suggest(
    q: str,
    *,
    limit: int = 10,
    kind: Sequence[str] = (),
    facet: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Autocomplete over labels.

    bool_prefix across the shingled subfields search_as_you_type generates, so
    the last word is treated as a prefix and the earlier ones as whole terms.
    Ranked by relevance, then by how much corpus sits behind the node, so the
    first suggestion is the one people usually mean.
    """
    if not q or not q.strip():
        return []
    clauses = _filter_clauses(BrowseFilters(kind=kind, facet=facet).validate())
    body = {
        "size": limit,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": q,
                            "type": "bool_prefix",
                            "fields": [
                                "label.autocomplete",
                                "label.autocomplete._2gram",
                                "label.autocomplete._3gram",
                            ],
                        }
                    }
                ],
                "filter": clauses,
            }
        },
        "sort": ["_score", {"chunk_count": "desc"}],
        "_source": ["node_id", "kind", "label", "facet", "chunk_count"],
    }
    return [h["_source"] for h in _search(body)["hits"]["hits"]]


def facet_counts(filters: BrowseFilters) -> Dict[str, Dict[str, int]]:
    """Counts for the filter panel, scoped to the filters already applied."""
    filters.validate()
    body = {"size": 0, "query": _query(filters), "aggs": AGGREGATIONS}
    return _buckets(_search(body).get("aggregations", {}))


def children(node_id: str, *, limit: int = 200, cursor: Optional[str] = None):
    return page(BrowseFilters(parent_id=node_id), limit=limit, cursor=cursor)


def themes_for(shelf_id: str, *, limit: int = 200, cursor: Optional[str] = None):
    return page(
        BrowseFilters(kind=("theme",), shelf_id=shelf_id), limit=limit, cursor=cursor
    )


def card_for(target_id: str) -> Optional[Dict[str, Any]]:
    resp = _search(
        {
            "size": 1,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"kind": "card"}},
                        {"term": {"target_id": target_id}},
                    ]
                }
            },
        }
    )
    hits = resp["hits"]["hits"]
    return hits[0]["_source"] if hits else None


def breadcrumb(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ancestors of a node, root first.

    One round-trip for the whole chain, because the projector already stored
    it. The UI should never walk parents a request at a time.
    """
    ancestors = list(node.get("ancestor_ids") or [])
    if not ancestors:
        return []
    by_id = get_nodes(ancestors)
    return [by_id[a] for a in ancestors if a in by_id]


def level_page(
    filters: BrowseFilters,
    *,
    size: int,
    cursor: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    """One page of one depth level, for the graph stream.

    Returns the documents and the raw sort values to resume from, rather than
    an encoded cursor: the stream keeps them in memory and never hands them to
    a client.
    """
    body: Dict[str, Any] = {
        "size": size,
        "query": _query(filters),
        "sort": LIST_SORT,
        "_source": list(SUMMARY_SOURCE) + ["parent_id", "shelf_ids", "target_id", "target_type"],
    }
    if cursor is not None:
        body["search_after"] = cursor
    resp = _search(body)
    hits = resp["hits"]["hits"]
    next_after = hits[-1]["sort"] if len(hits) == size and hits else None
    return [h["_source"] for h in hits], next_after


def max_depth(filters: BrowseFilters) -> int:
    """Deepest level matching the filters, so the stream knows when to stop."""
    body = {"size": 0, "query": _query(filters), "aggs": {"m": {"max": {"field": "depth"}}}}
    value = _search(body).get("aggregations", {}).get("m", {}).get("value")
    return int(value) if value is not None else 0
