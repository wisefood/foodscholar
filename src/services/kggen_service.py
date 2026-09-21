"""Extended KG-Gen retrieval, through the FoodScholar library.

The retriever this replaced (LinearRAG) shipped its own 617MB index: a
GraphML file plus parquet embeddings that had to be mounted next to the
service, were untracked in git, and drifted from whatever the graph build
last produced. Nothing here reads a file. `fs.retrieve()` scores over the same
Elasticsearch and Neo4j stores the offline build writes and the browse routes
already read, so a graph rebuild is visible to QA with nothing to re-sync and
no volume to provision.

The facade is the shared one from `services.kg_service`: one set of store
connections per process, not a second pool opened for retrieval. That also
means retrieval honours `KG_ENABLED` — a deployment without the graph gets a
readable 503 rather than a stack trace.

Every call is blocking (the library is synchronous, the Elasticsearch client
is the sync one, Neo4j opens a session per call). Callers on the request path
run it in a threadpool, which is what `services.qa_service` does.
"""
import logging
from typing import Any, Dict, List, Tuple

from services.kg_service import get_graph

logger = logging.getLogger(__name__)


def retrieve(question: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rank passages for `question`.

    Returns `(hits, trace)`. Each hit is a flat dict — chunk text, the
    provenance the chunk carries, and the three branch scores that ranked it —
    so the adapter above can shape it without importing library types. The
    trace is the retriever's own account of what it did, which the QA status
    payload surfaces.
    """
    fs = get_graph()
    hits, trace = fs.retrieve(question, k=top_k)

    results: List[Dict[str, Any]] = []
    for hit in hits:
        chunk = hit.chunk
        # `source_metadata` is free-form on the model; `provenance` is the
        # typed view that folds the alias spellings (`DOI`/`doi`, page as str
        # or float) onto canonical keys. Read that rather than the raw dict,
        # or half the citations lose their page number and year.
        #
        # `model_dump` rather than `citation_fields()`: the latter is scoped to
        # what a citation renderer prints and drops `urn` and `audience`, and
        # `urn` is what the whole answer layer keys a source on.
        provenance = {
            key: value
            for key, value in chunk.provenance.model_dump().items()
            if value not in (None, "")
        }
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": hit.score,
                "text_sim": hit.text_sim,
                "triplet_sim": hit.triplet_sim,
                "ppr_score": hit.ppr_score,
                "chunk_source_type": chunk.source_type,
                "source_doc_id": chunk.source_doc_id,
                "year": chunk.year,
                "shelf_ids": list(chunk.shelf_ids or []),
                "theme_ids": list(chunk.theme_ids or []),
                **provenance,
            }
        )

    return results, {
        "candidates": trace.candidates,
        "relations": trace.relations,
        "relations_scored": trace.relations_scored,
        "subgraph_nodes": trace.subgraph_nodes,
        "subgraph_edges": trace.subgraph_edges,
        "expansion_calls": trace.expansion_calls,
        "seed_entities": trace.seed_entities,
        "branches": trace.branches_used,
        "embedder": trace.embedder_model,
        # Non-empty when the ranking is less trustworthy than a healthy one —
        # most importantly when the library fell back to its hash embedder
        # because the real one could not load, which otherwise produces
        # confidently-ranked nonsense with nothing to show for it.
        "degraded": trace.degraded,
    }
