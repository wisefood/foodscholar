"""Knowledge graph browsing endpoints.

Three surfaces over one graph:

  hierarchy   facet -> shelf -> theme -> card -> evidence chunks, with
              breadcrumbs built server-side so the interface never walks
              ancestors one request at a time;
  search      full-text and autocomplete over labels, with filter counts that
              come back in the same response as the hits;
  exploration Server-Sent Events carrying nodes and edges for a visual graph,
              so drawing it costs one connection instead of a request storm.

Everything here reads the browse index (services/kg_browse_index), which the
projector builds from Neo4j. The only calls that reach the library are the ones
for evidence chunks and ontology entities, which live outside that index.

Read handlers are plain `def`: render() runs them in a threadpool, which is
where the synchronous Elasticsearch client belongs. The stream handlers are
`async def` and push their own blocking work to the threadpool themselves.

No route declares `response_model`. render() wraps every result in the
{help, success, result} envelope, so a declared model would be validated
against the envelope and reject it. The models in models/kg are still the
contract — they are what these handlers build and what lands in `result`.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from config import config
from exceptions import NotFoundError
from models.kg import (
    CardView,
    ChunkView,
    EntitySummary,
    FacetSummary,
    GraphSummary,
    NodeSummary,
    ReindexResult,
    SearchPage,
    ShelfDetail,
    SuggestItem,
    ThemeDetail,
)
from routers.generic import render
from services import kg_browse_index as browse
from services import kg_projector, kg_service, kg_stream
from services.kg_events import SSE_HEADERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


def _csv(value: Optional[str]) -> tuple:
    """Parse a comma-separated filter value, dropping blanks."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _filters(
    q: Optional[str] = None,
    kind: Optional[str] = None,
    facet: Optional[str] = None,
    status: Optional[str] = None,
    under: Optional[str] = None,
    depth_max: Optional[int] = None,
    min_chunks: Optional[int] = None,
    discovered_by: Optional[str] = None,
    evidence_quality: Optional[str] = None,
    has_card: Optional[bool] = None,
) -> browse.BrowseFilters:
    """The filter panel as a value object. Every field narrows; none widens."""
    return browse.BrowseFilters(
        q=q,
        kind=_csv(kind),
        facet=_csv(facet),
        status=_csv(status),
        under=under,
        depth_max=depth_max,
        min_chunks=min_chunks,
        discovered_by=_csv(discovered_by),
        evidence_quality=_csv(evidence_quality),
        has_card=has_card,
    ).validate()


# ------------------------------------------------------------------ overview


@router.get("/summary")
@render()
def graph_summary(request: Request):
    """What the browse index holds: size, facets, and which build it came from.

    `graph_version` is the library's own config hash, stamped onto every
    document at projection time, so a stale index is visible rather than
    merely suspected.
    """
    # Whether the feature is on is answered before whether it has been built,
    # and separately, because the two are different situations. Only the routes
    # that reach the source stores check KG_ENABLED — the browse routes read an
    # Elasticsearch index and would happily serve one left behind by a
    # deployment that has since switched the graph off. Without this an
    # interface cannot tell "not enabled here" from "nobody has run the
    # projector", and it will tell the user to look for a button that does not
    # exist.
    if not config.settings["KG_ENABLED"]:
        return GraphSummary(enabled=False, built=False)

    status = kg_projector.status()
    if not status.get("built"):
        return GraphSummary(built=False)

    counts = browse.facet_counts(browse.BrowseFilters())
    by_kind = counts.get("by_kind", {})
    facets = [
        FacetSummary(
            facet=name,
            shelf_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("shelf",))),
            theme_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("theme",))),
            card_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("card",))),
        )
        for name in sorted(counts.get("by_facet", {}))
    ]
    return GraphSummary(
        built=True,
        alias=status.get("alias"),
        documents=status.get("documents", 0),
        graph_version=status.get("graph_version"),
        counts={k: int(v) for k, v in by_kind.items()},
        facets=facets,
    )


@router.get("/facets")
@render()
def list_facets(request: Request):
    """The six Layer A facets, with how much of the graph sits in each."""
    counts = browse.facet_counts(browse.BrowseFilters())
    return [
        FacetSummary(
            facet=name,
            shelf_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("shelf",))),
            theme_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("theme",))),
            card_count=browse.count(browse.BrowseFilters(facet=(name,), kind=("card",))),
        )
        for name in sorted(counts.get("by_facet", {}))
    ]


@router.get("/facets/{facet}/roots")
@render()
def facet_roots(
    request: Request,
    facet: str,
    status: str = Query("active", description="Shelf statuses to include"),
    limit: int = Query(100, ge=1, le=500),
):
    """Top-level shelves of one facet — where a browse session starts.

    Defaults to active shelves: a folded shelf absorbed an intermediary and is
    kept in the data so no FoodOn id is lost, not because anyone wants to
    navigate it. Pass `status=active,folded` to see them.
    """
    filters = browse.BrowseFilters(
        facet=(facet,), kind=("shelf",), status=_csv(status), roots_only=True
    ).validate()
    page = browse.page(filters, limit=limit)
    return [NodeSummary.from_doc(doc) for doc in page["items"]]


# ------------------------------------------------------------------ nodes
#
# Node ids take the `path` converter because theme ids are slash-separated
# (`foods/olive_oil/monounsaturated_fat_r1`) and card ids embed them. The
# server decodes `%2F` before routing, so with a plain `{node_id}` every theme
# and card was a 404 that never reached a handler, while shelves (`foodon:…`)
# worked — the tree opened, and nothing inside it did.
#
# `path` matches slashes, so order is now load-bearing: the bare
# `/nodes/{node_id:path}` comes last, or it takes `x/children` as an id. A
# sub-route cannot claim a real id in return — theme ids end in `_r1`, `_m2`
# or `_g3`, never in `/children`, `/themes`, `/breadcrumb` or `/chunks`.


@router.get("/nodes/{node_id:path}/children")
@render()
def node_children(
    request: Request,
    node_id: str,
    limit: int = Query(100, ge=1, le=500),
    cursor: Optional[str] = Query(None, description="Cursor from a previous page"),
):
    """Child shelves of a shelf."""
    page = browse.children(node_id, limit=limit, cursor=cursor)
    return SearchPage(
        items=[NodeSummary.from_doc(d) for d in page["items"]],
        total=page["total"],
        next_cursor=page["next_cursor"],
    )


@router.get("/nodes/{node_id:path}/themes")
@render()
def node_themes(
    request: Request,
    node_id: str,
    limit: int = Query(100, ge=1, le=500),
    cursor: Optional[str] = Query(None),
):
    """Themes discovered on a shelf."""
    page = browse.themes_for(node_id, limit=limit, cursor=cursor)
    return SearchPage(
        items=[NodeSummary.from_doc(d) for d in page["items"]],
        total=page["total"],
        next_cursor=page["next_cursor"],
    )


@router.get("/nodes/{node_id:path}/breadcrumb")
@render()
def node_breadcrumb(request: Request, node_id: str):
    """Ancestors of a node, root first, in one round-trip."""
    doc = browse.get_node(node_id)
    if doc is None:
        raise NotFoundError(detail=f"No graph node with id {node_id!r}.")
    return [NodeSummary.from_doc(d) for d in browse.breadcrumb(doc)]


@router.get("/nodes/{node_id:path}/chunks")
@render()
def node_chunks(
    request: Request,
    node_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Evidence passages attached to a shelf or theme.

    Chunks live in the library's own index, not the browse index, so this is
    the one browse route that reads through the facade. It queries the chunk
    index by attachment id: the library's shelf.chunks() would read the entire
    corpus and filter in Python, which is fine in a notebook and not here.
    """
    doc = browse.get_node(node_id)
    if doc is None:
        raise NotFoundError(detail=f"No graph node with id {node_id!r}.")

    field = "theme_ids" if doc["kind"] == "theme" else "shelf_ids"
    fs = kg_service.get_graph()
    store = fs.chunk_store
    resp = store._es.search(
        index=store.index,
        body={
            "from": offset,
            "size": limit,
            "query": {"bool": {"filter": [{"term": {field: node_id}}]}},
            "sort": [{"chunk_id": "asc"}],
        },
        source_excludes=["embedding"],
    )
    from foodscholar.io.chunk import Chunk

    return [
        ChunkView.from_chunk(Chunk.model_validate(hit["_source"]))
        for hit in resp["hits"]["hits"]
    ]


@router.get("/nodes/{node_id:path}")
@render()
def get_node(request: Request, node_id: str):
    """One node, with everything its detail page needs.

    A shelf comes back with breadcrumb, children, themes and card; a theme with
    its shelves and card; a card on its own. One response rather than five.
    """
    doc = browse.get_node(node_id)
    if doc is None:
        raise NotFoundError(detail=f"No graph node with id {node_id!r}.")

    if doc["kind"] == "card":
        return CardView.from_doc(doc)

    breadcrumb = [NodeSummary.from_doc(d) for d in browse.breadcrumb(doc)]
    card_doc = browse.card_for(node_id)
    card = CardView.from_doc(card_doc) if card_doc else None

    if doc["kind"] == "theme":
        return ThemeDetail(
            **NodeSummary.from_doc(doc).model_dump(),
            shelf_ids=list(doc.get("shelf_ids") or []),
            keyword_terms=list(doc.get("keyword_terms") or []),
            discovered_by=doc.get("discovered_by"),
            discovery_pass=doc.get("discovery_pass"),
            breadcrumb=breadcrumb,
            card=card,
        )

    return ShelfDetail(
        **NodeSummary.from_doc(doc).model_dump(),
        foodon_id=doc.get("foodon_id"),
        see_also=list(doc.get("see_also") or []),
        support_direct=doc.get("support_direct") or 0,
        support_lifted=doc.get("support_lifted") or 0,
        breadcrumb=breadcrumb,
        children=[NodeSummary.from_doc(d) for d in browse.children(node_id)["items"]],
        themes=[NodeSummary.from_doc(d) for d in browse.themes_for(node_id)["items"]],
        card=card,
    )


@router.get("/cards/{target_id:path}")
@render()
def card_for_target(request: Request, target_id: str):
    """The Layer C card describing a shelf or theme."""
    doc = browse.card_for(target_id)
    if doc is None:
        raise NotFoundError(detail=f"No card describes {target_id!r}.")
    return CardView.from_doc(doc)


# ------------------------------------------------------------------ search


@router.get("/suggest")
@render()
def suggest(
    request: Request,
    q: str = Query(..., min_length=1, description="What the user has typed so far"),
    kind: Optional[str] = Query(None, description="Comma-separated: shelf, theme, card"),
    facet: Optional[str] = Query(None, description="Comma-separated facet names"),
    limit: int = Query(10, ge=1, le=25),
):
    """Autocomplete over node labels.

    Matches inside a label, not only at the start, so typing "olive" surfaces
    "extra virgin olive oil". Ranked by relevance and then by how much corpus
    sits behind the node, so the first suggestion is usually the intended one.
    """
    return [
        SuggestItem(**item)
        for item in browse.suggest(q, limit=limit, kind=_csv(kind), facet=_csv(facet))
    ]


@router.get("/search")
@render()
def search(
    request: Request,
    q: Optional[str] = Query(None, description="Full-text query over labels and descriptions"),
    kind: Optional[str] = Query(None),
    facet: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    under: Optional[str] = Query(None, description="Restrict to the subtree below this node"),
    depth_max: Optional[int] = Query(None, ge=0),
    min_chunks: Optional[int] = Query(None, ge=0),
    discovered_by: Optional[str] = Query(None),
    evidence_quality: Optional[str] = Query(None),
    has_card: Optional[bool] = Query(None),
    limit: int = Query(25, ge=1, le=100),
    cursor: Optional[str] = Query(None),
):
    """Search and filter the graph, with the filter panel's counts included.

    `under` is the scoping filter: it matches the materialized ancestor chain,
    so one term lookup restricts everything to a subtree — shelves, the themes
    on them, and the cards describing both.
    """
    filters = _filters(
        q=q, kind=kind, facet=facet, status=status, under=under,
        depth_max=depth_max, min_chunks=min_chunks,
        discovered_by=discovered_by, evidence_quality=evidence_quality,
        has_card=has_card,
    )
    page = browse.page(filters, limit=limit, cursor=cursor, with_aggs=True)
    return SearchPage(
        items=[NodeSummary.from_doc(d) for d in page["items"]],
        total=page["total"],
        next_cursor=page["next_cursor"],
        facets=page["facets"],
    )


@router.get("/filters")
@render()
def filter_counts(
    request: Request,
    q: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    facet: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    under: Optional[str] = Query(None),
    depth_max: Optional[int] = Query(None, ge=0),
    min_chunks: Optional[int] = Query(None, ge=0),
    discovered_by: Optional[str] = Query(None),
    evidence_quality: Optional[str] = Query(None),
    has_card: Optional[bool] = Query(None),
):
    """Counts for the filter panel, scoped to the filters already applied."""
    return browse.facet_counts(
        _filters(
            q=q, kind=kind, facet=facet, status=status, under=under,
            depth_max=depth_max, min_chunks=min_chunks,
            discovered_by=discovered_by, evidence_quality=evidence_quality,
            has_card=has_card,
        )
    )


# ------------------------------------------------------------------ streaming


@router.get("/stream")
async def stream_graph(
    request: Request,
    q: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    facet: Optional[str] = Query(None),
    status: Optional[str] = Query("active"),
    under: Optional[str] = Query(None),
    depth_max: Optional[int] = Query(None, ge=0),
    min_chunks: Optional[int] = Query(None, ge=0),
    discovered_by: Optional[str] = Query(None),
    evidence_quality: Optional[str] = Query(None),
    has_card: Optional[bool] = Query(None),
    max_nodes: Optional[int] = Query(None, ge=1),
):
    """Stream the filtered graph as Server-Sent Events, for drawing.

    Frames arrive breadth-first by depth, so the roots render immediately and
    the graph grows outward:

    - `meta` — totals, deepest level, graph version, whether the ceiling will bite
    - `nodes` — a batch of VizNodes, each carrying precomputed `x`/`y`
    - `edges` — a batch of VizEdges, never sent before both endpoints
    - `progress` — running counts, once per level
    - terminal: `done` (with `dropped_edges` and `truncated`) or `error`

    Comment frames (`: keep-alive`) fill quiet stretches. Every frame carries a
    monotonic `seq`.

    Unfiltered, this walks the whole graph, so `depth_max` defaults to a
    shallow level of detail — open a node with `/graph/stream/expand` rather
    than asking for everything at once.
    """
    if depth_max is None:
        depth_max = config.settings["KG_STREAM_DEFAULT_DEPTH"]

    filters = _filters(
        q=q, kind=kind, facet=facet, status=status, under=under,
        depth_max=depth_max, min_chunks=min_chunks,
        discovered_by=discovered_by, evidence_quality=evidence_quality,
        has_card=has_card,
    )

    events = kg_stream.stream_levels(
        filters,
        is_disconnected=request.is_disconnected,
        max_nodes=max_nodes,
        title=f"Knowledge graph ({facet})" if facet else "Knowledge graph",
    )
    return StreamingResponse(
        kg_stream.as_sse(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/stream/expand")
async def stream_expand(
    request: Request,
    node: str = Query(..., description="Node to expand"),
):
    """Stream one node's neighborhood — parent, children, themes, card.

    Expand-on-click. A fresh short stream rather than a message on the open
    one, because SSE only runs server to client: a client cannot ask an
    existing stream for more.
    """
    events = kg_stream.stream_neighborhood(node, is_disconnected=request.is_disconnected)
    return StreamingResponse(
        kg_stream.as_sse(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ------------------------------------------------------------------ entities


@router.get("/entities")
@render()
def list_entities(
    request: Request,
    q: Optional[str] = Query(None, description="Lexical query over label and synonyms"),
    prefix: Optional[str] = Query(None, description="OBO source, e.g. FOODON"),
    limit: int = Query(25, ge=1, le=100),
):
    """Browse the linked ontology entities behind the corpus."""
    entities = kg_service.get_graph().entities
    found = (
        entities.search(q, prefix=prefix, k=limit)
        if q
        else entities.list(prefix=prefix, k=limit)
    )
    return [EntitySummary.from_entity(e) for e in found]


@router.get("/entities/{ontology_id}")
@render()
def get_entity(request: Request, ontology_id: str):
    """One ontology entity."""
    entity = kg_service.get_graph().entities.get(ontology_id)
    if entity is None:
        raise NotFoundError(detail=f"No entity with id {ontology_id!r}.")
    return EntitySummary.from_entity(entity)


@router.get("/entities/{ontology_id}/chunks")
@render()
def entity_chunks(
    request: Request,
    ontology_id: str,
    limit: int = Query(25, ge=1, le=100),
):
    """Passages that mention an entity.

    For FOODON ids this is a filtered query over the chunk index. For other
    prefixes the library falls back to the capped sample stored on the entity,
    so a short result here is not necessarily the whole story.
    """
    chunks = kg_service.get_graph().entities.chunks_for(ontology_id, k=limit)
    return [ChunkView.from_chunk(c) for c in chunks]


# ------------------------------------------------------------------ operations


@router.post("/reindex", status_code=201)
@render()
def reindex(
    request: Request,
    drop_old: bool = Query(True, description="Delete the index the alias used to point at"),
):
    """Rebuild the browse index from the graph.

    Run this after the offline build produces a new graph; until it runs, the
    browse routes keep serving the previous projection. Writes a fresh index
    and repoints the alias only on success, so a failed run leaves the last
    good one in place.

    Held by a lock, so a second call while one is running is a conflict rather
    than a duplicate pass over Neo4j. Blocking and slow — it reads the whole
    graph — so callers should expect to wait.
    """
    return ReindexResult(**kg_projector.reindex(drop_old=drop_old))
