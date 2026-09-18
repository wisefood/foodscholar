"""Stream graph slices to a client over SSE.

A graph view needs thousands of nodes. Fetching them page by page is a request
storm, and the client cannot draw until the storm stops. One stream lets the
server push batches while the browser draws each as it lands, so the graph
grows in front of the user instead of appearing all at once.

The walk is breadth-first by depth, which is both the cheapest thing to query
(depth is indexed) and the nicest thing to watch: the roots draw immediately
and the graph grows outward.

Two invariants hold the whole design together:

  An edge is never sent before both of its endpoints. A client that receives a
  dangling edge has to buffer or drop it, and both are bugs. Edges are held
  back until the level that completes them.

  The event loop is never blocked. The browse index client is synchronous, so
  every page fetch goes through a threadpool. A bare search() in an async
  generator would stall every other request in the process.

Node shape matches the library's VizNode/VizEdge, so a slice streamed to a
browser and a slice rendered in a notebook describe the same graph.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from starlette.concurrency import run_in_threadpool

from config import config
from services import kg_browse_index as browse
from services.kg_events import SSE_HEARTBEAT_FRAME, GraphEvent, sse_format

logger = logging.getLogger(__name__)

#: How many documents to pull from Elasticsearch per round-trip. Independent of
#: the frame size below: one page may produce several frames.
_PAGE_SIZE = 500


def _node(doc: Dict[str, Any]) -> Dict[str, Any]:
    """One browse document as a VizNode."""
    return {
        "id": doc["node_id"],
        "label": doc.get("label") or doc["node_id"],
        "kind": doc["kind"],
        # Size hint. Corpus behind the node is the honest measure of how much
        # it matters, and it is what the library's own views use.
        "weight": float(doc.get("chunk_count") or 0),
        "facet": doc.get("facet"),
        "attrs": {
            "x": doc.get("x"),
            "y": doc.get("y"),
            "depth": doc.get("depth"),
            "chunk_count": doc.get("chunk_count"),
            "degree": doc.get("degree"),
            "has_card": doc.get("has_card"),
            "status": doc.get("status"),
            "child_count": doc.get("child_count"),
            "theme_count": doc.get("theme_count"),
        },
    }


def _candidate_edges(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Edges implied by one document.

    Every edge the browse view needs was denormalized onto the nodes by the
    projector, so the stream reads Elasticsearch pages and never touches Neo4j.
    """
    node_id = doc["node_id"]
    kind = doc["kind"]
    edges: List[Dict[str, Any]] = []

    if kind == "shelf":
        parent = doc.get("parent_id")
        if parent:
            edges.append({"source": parent, "target": node_id, "kind": "parent_of"})
    elif kind == "theme":
        # A theme can sit on several shelves; each attachment is an edge.
        for shelf_id in doc.get("shelf_ids") or []:
            edges.append({"source": shelf_id, "target": node_id, "kind": "has_theme"})
    elif kind == "card":
        target = doc.get("target_id")
        if target:
            edges.append({"source": node_id, "target": target, "kind": "describes"})

    for edge in edges:
        edge["weight"] = 1.0
        edge["attrs"] = {}
    return edges


def _batched(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _fetch_level(
    filters: browse.BrowseFilters,
    after: Optional[List[Any]],
) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    """One page of one level, off the event loop."""
    return await run_in_threadpool(
        browse.level_page, filters, size=_PAGE_SIZE, cursor=after
    )


async def stream_levels(
    filters: browse.BrowseFilters,
    *,
    is_disconnected: Optional[Callable[[], Any]] = None,
    max_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
    title: str = "Knowledge graph",
) -> AsyncIterator[GraphEvent]:
    """Walk the filtered graph breadth-first, yielding frames.

    The caller turns these into SSE frames; keeping them as events means the
    same generator can be tested without parsing a wire format.
    """
    settings = config.settings
    max_nodes = max_nodes or settings["KG_STREAM_MAX_NODES"]
    batch_size = batch_size or settings["KG_STREAM_BATCH_SIZE"]
    started = time.perf_counter()

    filters.validate()
    total = await run_in_threadpool(browse.count, filters)
    deepest = await run_in_threadpool(browse.max_depth, filters)
    graph_version = await run_in_threadpool(browse.indexed_graph_version)

    seq = 0

    def event(name: str, data: Dict[str, Any]) -> GraphEvent:
        nonlocal seq
        seq += 1
        return GraphEvent(name=name, data={"seq": seq, **data})

    yield event(
        "meta",
        {
            "title": title,
            "level": "L3",
            "total_nodes": total,
            "max_depth": deepest,
            "graph_version": graph_version,
            "truncated": total > max_nodes,
            "batch_size": batch_size,
        },
    )

    sent: set = set()
    pending: List[Dict[str, Any]] = []
    sent_nodes = 0
    sent_edges = 0
    truncated = False

    for level in range(0, deepest + 1):
        if truncated:
            break
        level_filters = _with_depth(filters, level)
        after: Optional[List[Any]] = None

        while True:
            if is_disconnected is not None and await is_disconnected():
                logger.info("Graph stream abandoned by the client at level %d", level)
                return

            docs, after = await _fetch_level(level_filters, after)
            if not docs:
                break

            if sent_nodes + len(docs) > max_nodes:
                docs = docs[: max(0, max_nodes - sent_nodes)]
                truncated = True

            for doc in docs:
                sent.add(doc["node_id"])
                pending.extend(_candidate_edges(doc))

            for batch in _batched([_node(d) for d in docs], batch_size):
                sent_nodes += len(batch)
                yield event("nodes", {"level": level, "nodes": list(batch)})

            if truncated or after is None:
                break

        # Flush every edge whose endpoints have now both been sent. The rest
        # waits for a deeper level, or is dropped at the end.
        ready = [e for e in pending if e["source"] in sent and e["target"] in sent]
        if ready:
            pending = [e for e in pending if not (e["source"] in sent and e["target"] in sent)]
            for batch in _batched(ready, batch_size):
                sent_edges += len(batch)
                yield event("edges", {"level": level, "edges": list(batch)})

        yield event("progress", {"sent_nodes": sent_nodes, "sent_edges": sent_edges})

    yield event(
        "done",
        {
            "nodes": sent_nodes,
            "edges": sent_edges,
            # Edges whose other endpoint was filtered out or fell past the
            # ceiling. Reported rather than dropped silently, so a client can
            # tell a sparse view from a broken one.
            "dropped_edges": len(pending),
            "truncated": truncated,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    )


def _with_depth(filters: browse.BrowseFilters, depth: int) -> browse.BrowseFilters:
    """The same filters, pinned to one level."""
    import dataclasses

    return dataclasses.replace(filters, depth=depth)


async def stream_neighborhood(
    node_id: str,
    *,
    is_disconnected: Optional[Callable[[], Any]] = None,
    batch_size: Optional[int] = None,
) -> AsyncIterator[GraphEvent]:
    """One node plus what touches it: parent, children, themes, card.

    Expand-on-click. Short enough that a fresh connection costs nothing, which
    matters because SSE is one-way — a client cannot ask an open stream for
    more, it opens another.
    """
    batch_size = batch_size or config.settings["KG_STREAM_BATCH_SIZE"]
    started = time.perf_counter()
    seq = 0

    def event(name: str, data: Dict[str, Any]) -> GraphEvent:
        nonlocal seq
        seq += 1
        return GraphEvent(name=name, data={"seq": seq, **data})

    anchor = await run_in_threadpool(browse.get_node, node_id)
    if anchor is None:
        yield event("error", {
            "title": "Node not found",
            "detail": f"No graph node with id {node_id!r}.",
            "cause": "NotFound",
        })
        return

    yield event("meta", {
        "title": anchor.get("label") or node_id,
        "level": "L2",
        "anchor": node_id,
        "batch_size": batch_size,
    })

    docs: List[Dict[str, Any]] = [anchor]
    seen = {node_id}

    def collect(new_docs: List[Dict[str, Any]]) -> None:
        for doc in new_docs:
            if doc["node_id"] not in seen:
                seen.add(doc["node_id"])
                docs.append(doc)

    parent_id = anchor.get("parent_id")
    if parent_id:
        parent = await run_in_threadpool(browse.get_node, parent_id)
        if parent:
            collect([parent])

    if anchor["kind"] == "shelf":
        collect((await run_in_threadpool(browse.children, node_id))["items"])
        collect((await run_in_threadpool(browse.themes_for, node_id))["items"])

    card = await run_in_threadpool(browse.card_for, node_id)
    if card:
        collect([card])

    if is_disconnected is not None and await is_disconnected():
        return

    for batch in _batched([_node(d) for d in docs], batch_size):
        yield event("nodes", {"level": 0, "nodes": list(batch)})

    # Every endpoint is in `docs`, so the invariant holds by construction here.
    edges = [
        edge
        for doc in docs
        for edge in _candidate_edges(doc)
        if edge["source"] in seen and edge["target"] in seen
    ]
    for batch in _batched(edges, batch_size):
        yield event("edges", {"level": 0, "edges": list(batch)})

    yield event("done", {
        "nodes": len(docs),
        "edges": len(edges),
        "dropped_edges": 0,
        "truncated": False,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    })


async def as_sse(events: AsyncIterator[GraphEvent]) -> AsyncIterator[str]:
    """Wrap an event stream in SSE frames, with a heartbeat during quiet spells.

    The heartbeat is not decoration: an idle proxy will close a stream that
    says nothing, and a slow Elasticsearch page is long enough to trigger that.
    """
    heartbeat = config.settings["KG_STREAM_HEARTBEAT_SECONDS"]
    iterator = events.__aiter__()
    try:
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield SSE_HEARTBEAT_FRAME
                continue
            except StopAsyncIteration:
                break
            yield sse_format(event)
            if event.is_terminal:
                break
    except Exception as exc:
        # The response already returned 200, so a failure can only be an event.
        logger.error("Graph stream failed: %s", exc, exc_info=True)
        yield sse_format(
            GraphEvent(
                name="error",
                data={
                    "title": "Error streaming the graph",
                    "detail": "The graph stream ended early. Please retry.",
                    "cause": exc.__class__.__name__,
                },
            )
        )
