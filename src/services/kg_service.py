"""Access to the FoodScholar knowledge graph library.

The graph is built offline: a job runs the library's phases (ingest, annotate,
embed, layer A/B/C) and writes shelves, themes and cards into Neo4j and chunks
and cards into Elasticsearch. This service never builds it. It opens those
stores once per process and reads.

Three callers use the facade:

  - the projector (services/kg_projector), which reads the whole graph once per
    rebuild and writes the browse index;
  - the detail routes, which read individual chunks and entities;
  - the `kggen` QA retriever (services/kggen_service), which scores passages
    over the chunk and relation stores.

Everything else — search, autocomplete, filtering, the tree, the graph slices —
is served from the browse index and never touches this module.

Every call here is blocking: the library is synchronous throughout, the
Elasticsearch client is the sync one, and the Neo4j driver opens a session per
call. Callers on the request path run it in a threadpool.
"""
import logging
import threading
from typing import Any, Dict

from config import config
from exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

_fs = None
_lock = threading.Lock()


def kg_config() -> Dict[str, Any]:
    """The library config, assembled from environment settings.

    A dict rather than a YAML file in the image: it keeps configuration in
    config.py with everything else, and a file would drift from the deployment
    manifest that actually sets these values.
    """
    s = config.settings
    es_auth = {
        key: value
        for key, value in (
            ("api_key", s["KG_ES_API_KEY"] or None),
            ("username", s["KG_ES_USERNAME"] or None),
            ("password", s["KG_ES_PASSWORD"] or None),
        )
        if value
    }
    return {
        # Required by the library's schema even though a read-only service
        # never loads a corpus. It is not opened at construction time.
        "corpus": {"chunks_path": "data/chunks.parquet"},
        # Left unset on purpose. Loading FoodOn parses a 40 MB OWL file, and
        # nothing in browsing needs it: shelves already carry foodon_id, label
        # and see_also.
        "ontology": None,
        "storage": {
            "chunk_store": {
                "backend": "elastic",
                "url": s["KG_ES_URL"],
                "index": s["KG_CHUNK_INDEX"],
                **es_auth,
            },
            "card_store": {
                "backend": "elastic",
                "url": s["KG_ES_URL"],
                "index": s["KG_CARD_INDEX"],
                **es_auth,
            },
            "graph_store": {
                "backend": "neo4j",
                "url": s["KG_NEO4J_URL"],
                "user": s["KG_NEO4J_USER"],
                "password": s["KG_NEO4J_PASSWORD"] or None,
            },
        },
        # Layer 0 relations, on the same cluster as the chunks. This is what
        # the `kggen` QA retriever's triplet and PageRank branches read; the
        # browse routes never touch it. `backend` is a separate switch from
        # the chunk store in the library, so it has to be said explicitly —
        # left at its "memory" default the relation store would open empty
        # and both graph branches would silently score nothing.
        "relations": {
            "store": {
                "backend": "elastic",
                "es_index": s["KG_RELATION_INDEX"],
            },
        },
        # Retrieval scoring. Weights must sum to 1.0 or the library refuses
        # the config, which surfaces as a 503 on the first graph question
        # rather than a quietly skewed ranking.
        "retrieval": {
            "candidate_k": s["KG_RETRIEVAL_CANDIDATE_K"],
            "w_text": s["KG_RETRIEVAL_W_TEXT"],
            "w_triplet": s["KG_RETRIEVAL_W_TRIPLET"],
            "w_ppr": s["KG_RETRIEVAL_W_PPR"],
            "subgraph_depth": s["KG_RETRIEVAL_SUBGRAPH_DEPTH"],
            "max_expansion_calls": s["KG_RETRIEVAL_MAX_EXPANSION_CALLS"],
            "max_relations": s["KG_RETRIEVAL_MAX_RELATIONS"],
            "embed_cache_size": s["KG_RETRIEVAL_EMBED_CACHE_SIZE"],
        },
        # Only the embedder from the annotate section, and no "llm" key.
        #
        # It is named rather than left to the library default because it has
        # to match the model the graph's chunks were embedded with: kNN
        # compares the query vector against the stored ones, so a mismatch is
        # a dimension error or silently meaningless distances, not a small
        # quality loss. Naming it here also means a graph built with a
        # different encoder is one env var away from working.
        #
        # The model still loads lazily — it is a property on the facade — so
        # browse-only deployments never pay for it and browse search stays
        # lexical. A deployment serving `kggen` pays on the first graph
        # question rather than at boot.
        "annotate": {"embedder": s["KG_EMBED_MODEL"]},
    }


def get_graph():
    """The shared facade, or a readable 503.

    Built once and shared across threadpool workers. That is safe for reads:
    the Neo4j driver is thread-safe and opens a session per call, and the
    Elasticsearch client is thread-safe. It would not be safe for writes, and
    nothing here writes.
    """
    global _fs
    if not config.settings["KG_ENABLED"]:
        raise ServiceUnavailableError(
            detail="The knowledge graph is not enabled in this deployment.",
            extra={"title": "KnowledgeGraphDisabled"},
        )
    if _fs is None:
        with _lock:
            if _fs is None:
                _fs = _build()
    return _fs


def _build():
    try:
        from foodscholar import FoodScholar
    except ImportError as exc:  # pragma: no cover - deployment shape
        raise ServiceUnavailableError(
            detail=(
                "The knowledge graph needs the foodscholar library, which is "
                f"not importable: {exc}"
            ),
            extra={"title": "KnowledgeGraphUnavailable"},
        ) from exc

    try:
        fs = FoodScholar.from_config(kg_config())
    except Exception as exc:
        raise ServiceUnavailableError(
            detail=f"Could not open the knowledge graph stores: {exc}",
            extra={"title": "KnowledgeGraphUnavailable"},
        ) from exc

    logger.info("Knowledge graph stores opened: %s", fs.info())
    return fs


def summary() -> Dict[str, int]:
    """Counts straight from the source stores.

    Three store calls, no scan. Used to check that a build actually ran before
    the projector reads an empty graph and indexes nothing.
    """
    return get_graph().graph.summary()


def info() -> Dict[str, str]:
    """The library's own view of how it is configured."""
    return get_graph().info()


def graph_version() -> str:
    """Identifier for the graph currently in the source stores.

    The library hashes its own config, so this changes when a rebuild ran with
    different settings. The projector stamps it onto every browse document, and
    /health surfaces it, so "which build am I looking at" has an answer.
    """
    return get_graph().config_hash


def shutdown() -> None:
    """Close the Neo4j driver. The facade never does this itself."""
    global _fs
    if _fs is None:
        return
    close = getattr(_fs.graph_store, "close", None)
    if close is not None:
        try:
            close()
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.warning("Error closing the knowledge graph store: %s", exc)
    _fs = None
    logger.info("Knowledge graph stores closed")


def health() -> Dict[str, Any]:
    """Never raises: /health reports status, it does not fail on one."""
    if not config.settings["KG_ENABLED"]:
        return {"enabled": False}
    try:
        return {"enabled": True, "graph_version": graph_version(), **summary()}
    except Exception as exc:
        return {"enabled": True, "error": str(exc)}
