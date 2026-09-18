"""Response models for knowledge graph browsing.

The library's own Pydantic models are not returned directly. Two reasons, and
the first is not cosmetic: Chunk and Card carry a 768-float `embedding`, which
is 10-15 KB of JSON per record and turns a page of evidence into megabytes.
The second is that a wire contract should not change because an internal model
grew a field.

Documents arrive from the browse index as plain dicts; the `from_doc` helpers
are the single place that knows their shape.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

NodeKind = Literal["shelf", "theme", "card"]
TargetType = Literal["shelf", "theme"]


class NodeSummary(BaseModel):
    """A node as it appears in a list, a search result or a tree row."""

    node_id: str = Field(description="Graph id of the shelf, theme or card")
    kind: NodeKind = Field(description="Which layer this node belongs to")
    label: str = Field(description="Display name, already resolved for grouped shelves")
    facet: Optional[str] = Field(default=None, description="Layer A facet")
    depth: Optional[int] = Field(default=None, description="Distance from the facet root")
    chunk_count: int = Field(default=0, description="Corpus chunks behind this node")
    has_card: bool = Field(default=False, description="Whether a Layer C card describes it")
    child_count: int = Field(default=0, description="Number of child shelves")
    theme_count: int = Field(default=0, description="Number of themes on this shelf")
    status: Optional[str] = Field(
        default=None,
        description="active, folded (absorbed an intermediary) or absent; shelves only",
    )

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "NodeSummary":
        return cls(
            node_id=doc["node_id"],
            kind=doc["kind"],
            label=doc.get("label") or doc["node_id"],
            facet=doc.get("facet"),
            depth=doc.get("depth"),
            chunk_count=doc.get("chunk_count") or 0,
            has_card=bool(doc.get("has_card")),
            child_count=doc.get("child_count") or 0,
            theme_count=doc.get("theme_count") or 0,
            status=doc.get("status"),
        )


class CardView(BaseModel):
    """A Layer C write-up. Every claim in `summary` cites `cited_chunk_ids`."""

    card_id: str
    target_id: str
    target_type: TargetType
    title: str
    summary: str
    tip: Optional[str] = None
    evidence_quality: Optional[str] = Field(
        default=None, description="high, medium, low, debated or unclear"
    )
    controversy_note: Optional[str] = None
    confidence_note: Optional[str] = None
    cited_chunk_ids: List[str] = Field(default_factory=list)
    safety_flagged: bool = Field(
        default=False,
        description=(
            "Set on safety-sensitive facets such as allergies. Returned rather "
            "than hidden, so the interface decides how to present it"
        ),
    )

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "CardView":
        return cls(
            card_id=doc["node_id"],
            target_id=doc.get("target_id") or "",
            target_type=doc.get("target_type") or "shelf",
            title=doc.get("label") or "",
            summary=doc.get("description") or "",
            tip=doc.get("tip"),
            evidence_quality=doc.get("evidence_quality"),
            controversy_note=doc.get("controversy_note"),
            confidence_note=doc.get("confidence_note"),
            cited_chunk_ids=list(doc.get("cited_chunk_ids") or []),
            safety_flagged=bool(doc.get("safety_flagged")),
        )


class ShelfDetail(NodeSummary):
    """One shelf, with everything a detail page needs in a single response."""

    foodon_id: Optional[str] = None
    see_also: List[str] = Field(
        default_factory=list,
        description="FoodOn ids absorbed into this shelf when it folded an intermediary",
    )
    support_direct: int = 0
    support_lifted: int = Field(
        default=0, description="Support inherited from descendant terms"
    )
    breadcrumb: List[NodeSummary] = Field(
        default_factory=list, description="Ancestors, root first"
    )
    children: List[NodeSummary] = Field(default_factory=list)
    themes: List[NodeSummary] = Field(default_factory=list)
    card: Optional[CardView] = None


class ThemeDetail(NodeSummary):
    """One theme, with the shelves it spans and how it was discovered."""

    shelf_ids: List[str] = Field(default_factory=list)
    keyword_terms: List[str] = Field(default_factory=list)
    discovered_by: Optional[str] = Field(
        default=None, description="leiden, hdbscan or bertopic"
    )
    discovery_pass: Optional[str] = Field(
        default=None, description="relatedness, merged or global_similarity"
    )
    breadcrumb: List[NodeSummary] = Field(default_factory=list)
    card: Optional[CardView] = None


class ChunkView(BaseModel):
    """A source passage. The embedding and per-mention annotations are omitted."""

    chunk_id: str
    text: str
    source_doc_id: str
    source_type: Optional[str] = Field(
        default=None, description="abstract, textbook or guide"
    )
    section_type: Optional[str] = None
    year: Optional[int] = None
    shelf_ids: List[str] = Field(default_factory=list)
    theme_ids: List[str] = Field(default_factory=list)
    foodon_ids: List[str] = Field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: Any) -> "ChunkView":
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_doc_id=chunk.source_doc_id,
            source_type=chunk.source_type,
            section_type=chunk.section_type,
            year=chunk.year,
            shelf_ids=list(chunk.shelf_ids or []),
            theme_ids=list(chunk.theme_ids or []),
            foodon_ids=list(chunk.foodon_ids or []),
        )


class EntitySummary(BaseModel):
    """A linked ontology entity as it appears in the entity browser."""

    ontology_id: str
    prefix: str
    label: str
    synonyms: List[str] = Field(default_factory=list)
    facet_hint: Optional[str] = None
    mention_count: int = 0
    chunk_count: int = 0

    @classmethod
    def from_entity(cls, entity: Any) -> "EntitySummary":
        return cls(
            ontology_id=entity.ontology_id,
            prefix=entity.prefix,
            label=entity.label,
            synonyms=list(entity.synonyms or []),
            facet_hint=entity.facet_hint,
            mention_count=entity.mention_count,
            chunk_count=entity.chunk_count,
        )


class SuggestItem(BaseModel):
    """One autocomplete suggestion."""

    node_id: str
    kind: NodeKind
    label: str
    facet: Optional[str] = None
    chunk_count: int = 0


class SearchPage(BaseModel):
    """A page of results, plus the counts that drive the filter panel.

    `facets` is scoped to the filters already applied, so the numbers describe
    what narrowing further would actually yield.
    """

    items: List[NodeSummary] = Field(default_factory=list)
    total: int = 0
    next_cursor: Optional[str] = Field(
        default=None, description="Opaque cursor; absent on the last page"
    )
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class FacetSummary(BaseModel):
    """One Layer A facet and how much of the graph sits in it."""

    facet: str
    shelf_count: int = 0
    theme_count: int = 0
    card_count: int = 0


class GraphSummary(BaseModel):
    """What the browse index currently holds."""

    built: bool
    alias: Optional[str] = None
    documents: int = 0
    graph_version: Optional[str] = Field(
        default=None, description="Identifies which build is being served"
    )
    counts: Dict[str, int] = Field(default_factory=dict)
    facets: List[FacetSummary] = Field(default_factory=list)


class ReindexResult(BaseModel):
    """The outcome of one projection run."""

    index: str
    alias: str
    graph_version: str
    documents: int
    shelves: int
    themes: int
    cards: int
    replaced: List[str] = Field(default_factory=list)
    elapsed_seconds: float
