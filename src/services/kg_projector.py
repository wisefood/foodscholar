"""Project the knowledge graph into the browse index.

Reads the graph out of Neo4j once, denormalizes it, lays it out, and bulk
writes one document per shelf, theme and card into a fresh index — then points
the read alias at it.

This is the expensive half of browsing, and it is deliberately not on the
request path. It runs once per graph rebuild, behind a lock, and every reader
afterwards pays one inverted-index lookup instead of a graph traversal.

Three things are derived here that the graph does not store:

  ancestor_ids   The chain from the root down to each node, so a subtree is a
                 term filter. Themes and cards inherit their shelf's chain, so
                 "everything under X" catches all three kinds.

  depth          Shelves have one; themes and cards do not. A theme sits one
                 level below its shallowest shelf and a card one below its
                 target, which gives the graph stream a coherent breadth-first
                 order across kinds.

  x, y           Positions, computed once with a fixed seed. A browser never
                 runs a force simulation, and the map a user learns is the map
                 they get next week.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import config
from exceptions import ConflictError, ServiceUnavailableError
from services import kg_browse_index as browse
from services import kg_service

logger = logging.getLogger(__name__)

#: Facet blocks are laid out on a grid so the six of them do not overlap.
_FACET_GRID_COLUMNS = 3
_FACET_BLOCK_SPAN = 2.4
#: Cards hang just below whatever they describe rather than joining the layout.
_CARD_OFFSET = 0.06


# ---------------------------------------------------------------- locking


def _lock_client():
    from backend.redis import RedisClientSingleton

    return RedisClientSingleton().client


def _acquire_lock(token: str) -> bool:
    """Only one replica may project at a time.

    Two runs racing would read the same graph and rewrite the same index for no
    extra freshness, while doubling the load on Neo4j. The lock expires on its
    own so a crashed run does not block the next one forever.
    """
    try:
        return bool(
            _lock_client().set(
                config.settings["KG_REINDEX_LOCK_KEY"],
                token,
                nx=True,
                ex=config.settings["KG_REINDEX_LOCK_TIMEOUT"],
            )
        )
    except Exception as exc:
        raise ServiceUnavailableError(
            detail=f"Could not reach Redis to take the reindex lock: {exc}",
            extra={"title": "ReindexLockUnavailable"},
        ) from exc


def _release_lock(token: str) -> None:
    """Release only if we still hold it — never free someone else's lock."""
    key = config.settings["KG_REINDEX_LOCK_KEY"]
    try:
        client = _lock_client()
        if client.get(key) == token:
            client.delete(key)
    except Exception as exc:  # pragma: no cover - best-effort release
        logger.warning("Could not release the reindex lock: %s", exc)


# ---------------------------------------------------------------- reading


def _read_graph(fs) -> Tuple[List[Any], List[Any]]:
    """Shelves and themes, in two round-trips rather than N.

    Theme.shelf_ids carries the back-reference, so the shelf-to-theme mapping
    is a join here instead of one get_themes_for_shelf call per shelf. On a
    graph with thousands of shelves that is the difference between seconds and
    minutes.
    """
    shelves = [h.model for h in fs.graph.shelves()]
    themes = [h.model for h in fs.graph.themes()]
    logger.info("Read %d shelves and %d themes", len(shelves), len(themes))
    return shelves, themes


def _read_cards(fs, shelves: List[Any], themes: List[Any]) -> List[Any]:
    """Every card, one round-trip at a time.

    The library exposes get_card(target_id, target_type) and nothing bulkier —
    the card store offers only get_many(ids) and a vector search. So this is
    N+M calls, which is why it happens once per rebuild and never in a request.
    A list_cards() on the library's GraphStore would be the clean fix.
    """
    cards: List[Any] = []
    targets = [(s.shelf_id, "shelf") for s in shelves]
    targets += [(t.theme_id, "theme") for t in themes]
    started = time.perf_counter()
    for index, (target_id, target_type) in enumerate(targets, start=1):
        try:
            handle = fs.graph.card(target_id, target_type)
        except Exception as exc:
            logger.warning("Card lookup failed for %s %s: %s", target_type, target_id, exc)
            continue
        if handle is not None:
            cards.append(handle.model)
        if index % 500 == 0:
            logger.info("Card sweep: %d/%d targets", index, len(targets))
    logger.info(
        "Read %d cards from %d targets in %.1fs",
        len(cards), len(targets), time.perf_counter() - started,
    )
    return cards


# ---------------------------------------------------------------- derivation


def _ancestor_chains(shelves: List[Any]) -> Dict[str, List[str]]:
    """Root-first ancestor chain per shelf, iteratively and cycle-safe.

    The graph should be a forest, but a malformed parent pointer would turn a
    naive walk into an infinite loop, and a projection is not the place to
    discover that the hard way.
    """
    parent_of = {s.shelf_id: s.parent_shelf_id for s in shelves}
    chains: Dict[str, List[str]] = {}

    def chain_for(shelf_id: str) -> List[str]:
        if shelf_id in chains:
            return chains[shelf_id]
        # Walked child-first, reversed at the end. A memo hit contributes a
        # chain that is ALREADY root-first, so it becomes a prefix rather than
        # being appended — appending it and reversing the whole thing would
        # interleave the two orders and scramble the breadcrumb.
        upward: List[str] = []
        prefix: List[str] = []
        seen = {shelf_id}
        current = parent_of.get(shelf_id)
        while current is not None and current in parent_of:
            if current in seen:
                logger.warning("Cycle in the shelf tree at %s; truncating chain", current)
                break
            if current in chains:
                prefix = chains[current] + [current]
                break
            upward.append(current)
            seen.add(current)
            current = parent_of.get(current)
        upward.reverse()
        chain = prefix + upward
        chains[shelf_id] = chain
        return chain

    for shelf in shelves:
        chain_for(shelf.shelf_id)
    return chains


def _layout(
    shelves: List[Any], themes: List[Any]
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    """Positions and degrees for shelves and themes.

    One spring layout per facet, each dropped into its own cell of a grid, so
    facets read as distinct regions instead of one hairball. The seed is fixed,
    so the same graph always produces the same picture.

    networkx ships with the library, and spring_layout needs only numpy — the
    layouts that would look nicer (kamada_kawai, spectral) want scipy, which is
    not installed here.
    """
    import networkx as nx

    seed = config.settings["KG_LAYOUT_SEED"]
    graph = nx.Graph()
    facet_of: Dict[str, str] = {}

    for shelf in shelves:
        graph.add_node(shelf.shelf_id)
        facet_of[shelf.shelf_id] = shelf.facet
        if shelf.parent_shelf_id:
            graph.add_edge(shelf.parent_shelf_id, shelf.shelf_id)
    for theme in themes:
        graph.add_node(theme.theme_id)
        facet_of[theme.theme_id] = theme.facet
        for shelf_id in theme.shelf_ids:
            graph.add_edge(shelf_id, theme.theme_id)

    degrees = {node: int(deg) for node, deg in graph.degree()}

    positions: Dict[str, Tuple[float, float]] = {}
    facets = sorted({f for f in facet_of.values() if f})
    for index, facet in enumerate(facets):
        nodes = [n for n, f in facet_of.items() if f == facet and n in graph]
        if not nodes:
            continue
        subgraph = graph.subgraph(nodes)
        try:
            local = nx.spring_layout(subgraph, seed=seed, iterations=50)
        except Exception as exc:  # pragma: no cover - degenerate graphs
            logger.warning("Spring layout failed for facet %s (%s); using circular", facet, exc)
            local = nx.circular_layout(subgraph)

        column = index % _FACET_GRID_COLUMNS
        row = index // _FACET_GRID_COLUMNS
        offset_x = column * _FACET_BLOCK_SPAN
        offset_y = -row * _FACET_BLOCK_SPAN
        for node, (x, y) in local.items():
            positions[node] = (float(x) + offset_x, float(y) + offset_y)

    # Nodes with no facet (or isolated from every facet block) still need a
    # position, or the client has to invent one.
    for node in graph.nodes:
        positions.setdefault(node, (0.0, 0.0))
    return positions, degrees


# ---------------------------------------------------------------- documents


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _documents(
    shelves: List[Any],
    themes: List[Any],
    cards: List[Any],
    *,
    graph_version: str,
) -> Iterable[Dict[str, Any]]:
    chains = _ancestor_chains(shelves)
    positions, degrees = _layout(shelves, themes)
    indexed_at = _now()

    depth_of = {s.shelf_id: int(s.depth) for s in shelves}
    child_counts: Dict[str, int] = {}
    for shelf in shelves:
        if shelf.parent_shelf_id:
            child_counts[shelf.parent_shelf_id] = child_counts.get(shelf.parent_shelf_id, 0) + 1

    theme_counts: Dict[str, int] = {}
    for theme in themes:
        for shelf_id in theme.shelf_ids:
            theme_counts[shelf_id] = theme_counts.get(shelf_id, 0) + 1

    carded = {c.target_id for c in cards}

    def base(node_id: str, kind: str) -> Dict[str, Any]:
        x, y = positions.get(node_id, (0.0, 0.0))
        return {
            "kind": kind,
            "node_id": node_id,
            "x": round(x, 5),
            "y": round(y, 5),
            "degree": degrees.get(node_id, 0),
            "has_card": node_id in carded,
            "indexed_at": indexed_at,
            "graph_version": graph_version,
        }

    for shelf in shelves:
        doc = base(shelf.shelf_id, "shelf")
        doc.update(
            {
                # display_label is the human-facing name for a grouped shelf
                # and is None otherwise. Resolving it once here means no client
                # ever has to know that rule.
                "label": shelf.display_label or shelf.label,
                "description": "",
                "facet": shelf.facet,
                "status": shelf.status,
                "depth": int(shelf.depth),
                "parent_id": shelf.parent_shelf_id,
                "ancestor_ids": chains.get(shelf.shelf_id, []),
                "chunk_count": int(shelf.chunk_count),
                "support_direct": int(shelf.support_direct),
                "support_lifted": int(shelf.support_lifted),
                "foodon_id": shelf.foodon_id,
                "see_also": list(shelf.see_also or []),
                "child_count": child_counts.get(shelf.shelf_id, 0),
                "theme_count": theme_counts.get(shelf.shelf_id, 0),
            }
        )
        yield doc

    # A theme has no depth of its own, so it sits one level below its
    # shallowest shelf, and inherits that shelf's ancestor chain so a subtree
    # filter catches it. Cards read these back, so they are built once here
    # rather than recomputed per card.
    theme_depth: Dict[str, int] = {}
    theme_facet: Dict[str, Optional[str]] = {}
    theme_chain: Dict[str, List[str]] = {}
    theme_anchor: Dict[str, Optional[str]] = {}
    for theme in themes:
        shelf_ids = list(theme.shelf_ids or [])
        parent_depths = [depth_of[s] for s in shelf_ids if s in depth_of]
        anchor = shelf_ids[0] if shelf_ids else None
        theme_depth[theme.theme_id] = (min(parent_depths) + 1) if parent_depths else 1
        theme_facet[theme.theme_id] = theme.facet
        theme_anchor[theme.theme_id] = anchor
        theme_chain[theme.theme_id] = (
            (list(chains.get(anchor, [])) + [anchor]) if anchor else []
        )

    shelf_facet = {s.shelf_id: s.facet for s in shelves}

    for theme in themes:
        doc = base(theme.theme_id, "theme")
        doc.update(
            {
                "label": theme.label,
                "description": " ".join(theme.keyword_terms or []),
                "facet": theme.facet,
                "depth": theme_depth[theme.theme_id],
                "parent_id": theme_anchor[theme.theme_id],
                # The chain stored on the theme stops at its anchor shelf: the
                # theme is not its own ancestor, but everything above it is.
                "ancestor_ids": theme_chain[theme.theme_id],
                "shelf_ids": list(theme.shelf_ids or []),
                "chunk_count": int(theme.chunk_count),
                "keyword_terms": list(theme.keyword_terms or []),
                "discovered_by": theme.discovered_by,
                "discovery_pass": theme.discovery_pass,
                "child_count": 0,
                "theme_count": 0,
            }
        )
        yield doc

    for card in cards:
        target = card.target_id
        if card.target_type == "shelf":
            depth = depth_of.get(target, 0) + 1
            facet = shelf_facet.get(target)
            ancestors = list(chains.get(target, [])) + [target]
        else:
            depth = theme_depth.get(target, 1) + 1
            facet = theme_facet.get(target)
            ancestors = list(theme_chain.get(target, [])) + [target]

        x, y = positions.get(target, (0.0, 0.0))
        doc = base(card.card_id, "card")
        doc.update(
            {
                "label": card.title,
                "description": card.summary,
                "facet": facet,
                "depth": depth,
                "parent_id": target,
                "ancestor_ids": ancestors,
                "target_id": target,
                "target_type": card.target_type,
                "evidence_quality": card.evidence_quality,
                "safety_flagged": bool(card.safety_flagged),
                "controversy_note": card.controversy_note,
                "confidence_note": card.confidence_note,
                "tip": card.tip,
                "cited_chunk_ids": list(card.cited_chunk_ids or []),
                "chunk_count": len(card.cited_chunk_ids or []),
                "child_count": 0,
                "theme_count": 0,
                # A card is an annotation on its target, not a place of its
                # own, so it sits just below it rather than joining the layout.
                "x": round(x, 5),
                "y": round(y - _CARD_OFFSET, 5),
            }
        )
        yield doc


# ---------------------------------------------------------------- the run


def reindex(*, drop_old: bool = True) -> Dict[str, Any]:
    """Rebuild the browse index from the graph and swap the alias.

    Writes to a fresh physical index and repoints the alias only once the whole
    projection succeeded, so a reader never sees a half-built graph and a bad
    run leaves the previous one serving.
    """
    token = f"{time.time():.6f}"
    if not _acquire_lock(token):
        raise ConflictError(
            detail="A knowledge graph reindex is already running.",
            extra={"title": "ReindexInProgress"},
        )

    started = time.perf_counter()
    try:
        fs = kg_service.get_graph()
        graph_version = fs.config_hash
        counts = fs.graph.summary()
        if not counts.get("shelves"):
            raise ServiceUnavailableError(
                detail=(
                    "The knowledge graph is empty — the offline build has not run, "
                    "or the configured stores point somewhere else."
                ),
                extra={"title": "KnowledgeGraphEmpty"},
            )

        shelves, themes = _read_graph(fs)
        cards = _read_cards(fs, shelves, themes)

        index_name = browse.versioned_index(graph_version)
        # A retried run must not merge into a half-written index from the run
        # that failed. The alias still points at the previous good one.
        browse.delete_index(index_name)
        browse.create_index(index_name)

        written = _bulk_write(index_name, _documents(
            shelves, themes, cards, graph_version=graph_version
        ))

        browse.client().indices.refresh(index=index_name)
        detached = browse.swap_alias(index_name)
        if drop_old:
            for old in detached:
                browse.delete_index(old)

        elapsed = time.perf_counter() - started
        logger.info(
            "Reindexed the knowledge graph: %d documents into %s in %.1fs",
            written, index_name, elapsed,
        )
        return {
            "index": index_name,
            "alias": browse.alias(),
            "graph_version": graph_version,
            "documents": written,
            "shelves": len(shelves),
            "themes": len(themes),
            "cards": len(cards),
            "replaced": detached,
            "elapsed_seconds": round(elapsed, 2),
        }
    finally:
        _release_lock(token)


def _bulk_write(index_name: str, documents: Iterable[Dict[str, Any]]) -> int:
    from elasticsearch.helpers import bulk

    chunk_size = config.settings["KG_BROWSE_BULK_SIZE"]

    def actions():
        for doc in documents:
            yield {
                "_op_type": "index",
                "_index": index_name,
                # kind-prefixed so a shelf and a card can never collide on an
                # id, and so a rerun overwrites rather than duplicating.
                "_id": f"{doc['kind']}:{doc['node_id']}",
                "_source": doc,
            }

    success, errors = bulk(
        browse.client(), actions(), chunk_size=chunk_size, raise_on_error=True
    )
    if errors:  # pragma: no cover - raise_on_error makes this unreachable
        logger.error("Bulk indexing reported %d errors", len(errors))
    return int(success)


def status() -> Dict[str, Any]:
    """What the browse index currently holds. Never raises."""
    try:
        if not browse.alias_exists():
            return {"built": False}
        return {
            "built": True,
            "alias": browse.alias(),
            "documents": browse.count(),
            "graph_version": browse.indexed_graph_version(),
        }
    except Exception as exc:
        return {"built": False, "error": str(exc)}
