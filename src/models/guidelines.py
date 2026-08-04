"""Request and response models for guideline extraction endpoints."""

from typing import List, Literal, Optional, cast

from pydantic import BaseModel, Field, field_validator

from services.guideline_extractor import (
    DEFAULT_PROFILE_PAGE_COUNT,
    get_default_dpi,
    get_default_model,
)

GuidelineActionType = Literal[
    "eat",
    "drink",
    "use",
    "do",
    "avoid",
    "prepare",
    "limit",
    "choose",
    "increase",
    "reduce",
]

GUIDELINE_ACTION_TYPES: tuple[str, ...] = (
    "eat",
    "drink",
    "use",
    "do",
    "avoid",
    "prepare",
    "limit",
    "choose",
    "increase",
    "reduce",
)
GUIDELINE_ACTION_TYPE_ALIASES: dict[str, GuidelineActionType] = {
    "encourage": "choose",
}
DEFAULT_GUIDELINE_ACTION_TYPE: GuidelineActionType = "choose"


def normalize_guideline_action_type(value: str) -> GuidelineActionType:
    """Normalize legacy action type aliases to the platform-supported enum."""
    if not isinstance(value, str):
        raise TypeError("Guideline action_type must be a string.")

    normalized = value.strip().lower()
    normalized = GUIDELINE_ACTION_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in GUIDELINE_ACTION_TYPES:
        allowed = ", ".join(f"'{item}'" for item in GUIDELINE_ACTION_TYPES)
        raise ValueError(
            f"Unsupported guideline action_type '{value}'. Use one of: {allowed}."
        )
    return cast(GuidelineActionType, normalized)


class GuidelineExtractionRunRequest(BaseModel):
    """Optional runtime overrides when scheduling guideline extraction."""

    model: str = Field(
        default_factory=get_default_model,
        description="OpenAI model used for page triage and guideline extraction",
    )
    dpi: int = Field(
        default_factory=get_default_dpi,
        ge=72,
        le=300,
        description="Render DPI used when converting PDF pages to images",
    )
    guide_id: Optional[str] = Field(
        default=None,
        description=(
            "Guide identifier or URN whose metadata (title, region, audience, year) "
            "is injected into the extraction prompts so rules extracted from a page "
            "carry the guide's population context. Strongly recommended."
        ),
    )
    profile_document: bool = Field(
        default=True,
        description=(
            "Read the guide's opening pages to establish its title, region, and "
            "above all the population it addresses, filling whatever the catalog "
            "record leaves blank. Disable only to save the extra model call when "
            "the catalog metadata is known to be complete."
        ),
    )
    profile_page_count: int = Field(
        default=DEFAULT_PROFILE_PAGE_COUNT,
        ge=1,
        le=20,
        description="How many leading pages the document profile pass reads",
    )
    force: bool = Field(
        default=False,
        description=(
            "Re-queue even when a job is already registered for this artifact. "
            "The per-artifact lock still prevents two workers extracting the "
            "same PDF, so this is safe; use it to recover a stalled job."
        ),
    )


class GuidelineProcessedPage(BaseModel):
    """Summary of a relevant processed PDF page."""

    page: int = Field(description="1-based page number")
    page_summary: str = Field(description="Short summary of the page content")
    guideline_count: int = Field(description="Number of guidelines extracted from the page")
    continues_from_previous: bool = Field(
        default=False,
        description="Whether this page continued a table, list, or sentence from the previous page",
    )


class GuidelineSkippedPage(BaseModel):
    """Summary of a skipped PDF page."""

    page: int = Field(description="1-based page number")
    decision: str = Field(description="Triage decision for the page")
    reason: str = Field(description="Why the page was skipped")
    continues_from_previous: bool = Field(
        default=False,
        description="Whether this page continued a structure from the previous page",
    )


class GuidelineGuideContext(BaseModel):
    """
    Guide context injected into the extraction prompts.

    Assembled from the catalog record plus, where that was silent, the guide
    document itself. ``derived_fields`` names the values that came from the
    document so a reviewer can see what was inferred rather than curated.
    """

    guide_urn: Optional[str] = None
    title: Optional[str] = None
    region: Optional[str] = None
    audience: Optional[str] = None
    target_audiences: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    publication_year: Optional[int] = None
    issuing_authority: Optional[str] = None
    population_note: Optional[str] = Field(
        default=None, description="Who the guidance is for, in the document's own terms"
    )
    age_min_months: Optional[int] = None
    age_max_months: Optional[int] = None
    scope_note: Optional[str] = None
    evidence: List[str] = Field(
        default_factory=list,
        description="Verbatim quotes the document-derived values were based on",
    )
    derived_fields: List[str] = Field(
        default_factory=list,
        description="Context fields read from the document rather than the catalog record",
    )


class GuidelineDocumentProfile(BaseModel):
    """Raw result of the profile pass over the guide's opening pages."""

    title: Optional[str] = None
    issuing_authority: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    publication_year: Optional[int] = None
    audience: Optional[str] = None
    population_note: Optional[str] = None
    age_min_months: Optional[int] = None
    age_max_months: Optional[int] = None
    scope_note: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    pages_read: List[int] = Field(default_factory=list)


class ExtractedGuidelineItem(BaseModel):
    """
    A single extracted guideline sentence plus the facet hints the page supported.

    Every field except ``page`` and ``text`` was added with extraction schema v2;
    results persisted under v1 validate with all of them absent.
    """

    page: int = Field(description="1-based source page number")
    text: str = Field(description="Extracted guideline text")
    section_label: Optional[str] = Field(
        default=None, description="Heading or table caption the rule sits under"
    )
    source_snippet: Optional[str] = Field(
        default=None, description="Verbatim span from the page the rule is based on"
    )
    target_population_hint: Optional[str] = Field(
        default=None, description="Free-text description of who the rule is for"
    )
    age_min_months: Optional[int] = Field(default=None)
    age_max_months: Optional[int] = Field(default=None)
    life_stage: List[str] = Field(default_factory=list)
    setting: List[str] = Field(default_factory=list)
    health_conditions: List[str] = Field(default_factory=list)
    nutrients: List[str] = Field(default_factory=list)
    guideline_type: Optional[str] = Field(default=None)
    topic: List[str] = Field(default_factory=list)
    action_type_hint: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)


class GuidelineArtifactStorageResponse(BaseModel):
    """Artifact-local temporary storage details for the downloaded PDF."""

    artifact_uuid: str = Field(description="Artifact UUID")
    workspace_root: str = Field(description="Root folder for temporary downloaded artifact PDFs")
    artifact_dir: str = Field(description="Temporary directory reserved for this artifact")
    pdf_filename: str = Field(description="Local temporary PDF filename")
    pdf_path: str = Field(description="Expected full path to the temporary local PDF")
    pdf_exists: bool = Field(description="Whether the temporary local PDF currently exists")


class GuidelineExtractionResponse(BaseModel):
    """Guideline extraction output for a staged artifact PDF."""

    artifact_uuid: str = Field(description="Artifact UUID")
    workspace_root: str = Field(description="Root folder for temporary downloaded artifact PDFs")
    artifact_dir: str = Field(description="Temporary directory used for this artifact")
    pdf_path: str = Field(description="Local temporary PDF path used during extraction")
    model: str = Field(description="OpenAI model used for extraction")
    dpi: int = Field(description="Render DPI used for page images")
    extracted_at: str = Field(description="ISO timestamp when the extraction completed")
    total_pages: int = Field(description="Total number of pages in the source PDF")
    total_processed_pages: int = Field(description="Number of relevant pages processed")
    total_skipped_pages: int = Field(description="Number of pages skipped during triage")
    total_guidelines: int = Field(description="Total number of extracted guideline entries")
    total_unique_guidelines: int = Field(description="Total number of deduplicated guidelines")
    processed_pages: List[GuidelineProcessedPage] = Field(
        description="Relevant pages and their extraction summaries"
    )
    skipped_pages: List[GuidelineSkippedPage] = Field(
        description="Skipped pages and the reason each page was excluded"
    )
    guidelines: List[ExtractedGuidelineItem] = Field(
        description="Per-page guideline strings before cross-page deduplication"
    )
    unique_guidelines: List[str] = Field(
        description="Deduplicated guideline strings across the full document"
    )
    schema_version: int = Field(
        default=1,
        description=(
            "Extraction result schema version. 1 = per-rule {page, text} only; "
            "2 = per-rule facet hints, provenance, and guide context. Results "
            "stored before v2 keep their version and are imported accordingly."
        ),
    )
    guide_context: Optional[GuidelineGuideContext] = Field(
        default=None,
        description="Guide context used during extraction (catalog record merged with the document profile)",
    )
    document_profile: Optional[GuidelineDocumentProfile] = Field(
        default=None,
        description="What the guide's opening pages said about the document, when profiling ran",
    )
    continuation_pages: List[int] = Field(
        default_factory=list,
        description="Pages that continued a structure from the preceding page",
    )


class GuidelineExtractionJobResponse(BaseModel):
    """Current queued/running/completed status for a guideline extraction job."""

    artifact_uuid: str = Field(description="Artifact UUID")
    status: Literal[
        "not_found", "queued", "running", "succeeded", "failed", "stalled"
    ] = Field(description="Current job state")
    stalled: bool = Field(
        default=False,
        description=(
            "The job claims to be running but no worker holds its lock — the "
            "process handling it died. Re-queueing is safe and will pick it up."
        ),
    )
    job_id: Optional[str] = Field(default=None, description="Latest job identifier for this artifact")
    model: Optional[str] = Field(default=None, description="Model requested for the latest job")
    dpi: Optional[int] = Field(default=None, description="DPI requested for the latest job")
    enqueued_at: Optional[str] = Field(default=None, description="ISO timestamp when the job was queued")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp when processing started")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp when processing finished")
    current_page: Optional[int] = Field(default=None, description="Current page being processed")
    total_pages: Optional[int] = Field(default=None, description="Total pages in the source PDF if known")
    error: Optional[str] = Field(default=None, description="Failure message if the latest job failed")
    storage: GuidelineArtifactStorageResponse = Field(
        description="Local temporary download directory and PDF location"
    )
    result: Optional[GuidelineExtractionResponse] = Field(
        default=None,
        description="Latest persisted extraction result for the artifact, if available",
    )


class GuidelineImportRequest(BaseModel):
    """Request body for importing extracted guidelines into a guide."""

    guide_id: str = Field(
        description="Guide identifier or URN understood by the WiseFood client"
    )
    dry_run: bool = Field(
        default=True,
        description="When true, return the planned imports without creating guidelines",
    )
    dedupe_against_guide: bool = Field(
        default=True,
        description="Skip extracted rules already present on the target guide",
    )
    action_type: GuidelineActionType = Field(
        default=DEFAULT_GUIDELINE_ACTION_TYPE,
        description=(
            "Fallback action type for guidelines whose extraction produced no "
            "per-rule action hint. Legacy 'encourage' inputs normalize to 'choose'."
        ),
    )
    existing_scan_limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=100000,
        description=(
            "Maximum number of existing guide guidelines to scan when deduping and "
            "calculating sequence numbers. Omit to scan every existing guideline, "
            "which is what correctness requires — a bounded scan silently misses "
            "duplicates and can reuse sequence numbers on large guides."
        ),
    )
    import_facets: bool = Field(
        default=True,
        description=(
            "Carry per-rule facet hints (life stage, setting, nutrients, ...) and "
            "source references from a v2 extraction result onto the created "
            "guidelines. No effect on v1 results, which carry no hints."
        ),
    )

    @field_validator("action_type", mode="before")
    @classmethod
    def validate_action_type(cls, value: str) -> GuidelineActionType:
        """Normalize backwards-compatible action type aliases before validation."""
        return normalize_guideline_action_type(value)


class GuidelineEnrichmentEnqueueRequest(BaseModel):
    """Request body for queueing guideline facet enrichment."""

    guide_urns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Guides to enrich. Omit to enrich every guide that has guidelines — "
            "the backfill form."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Re-enrich guidelines that already reached the current enrichment "
            "version. The catalog still refuses to overwrite human-edited values."
        ),
    )
    allow_pdf_profile: bool = Field(
        default=True,
        description=(
            "When a guide's catalog metadata does not establish who its rules are "
            "for, read the guide's PDF to recover it. This is what gives the "
            "already-extracted corpus its population context; disabling it makes "
            "enrichment markedly weaker on thinly-catalogued guides."
        ),
    )


class GuidelineEnrichmentPreviewRequest(BaseModel):
    """Request body for previewing enrichment proposals for one guide."""

    guide_urn: str = Field(description="Guide whose rules should be sampled")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="How many of the guide's rules to run the agent over",
    )
    allow_pdf_profile: bool = Field(
        default=True,
        description="Read the guide's PDF when its metadata leaves the population unestablished",
    )


class GuidelineImportItemResponse(BaseModel):
    """One guideline candidate or created import result."""

    rule_text: str = Field(description="Guideline rule text")
    page_no: Optional[int] = Field(default=None, description="Source PDF page number")
    action_type: GuidelineActionType = Field(
        description="Action type used for the guideline"
    )
    sequence_no: Optional[int] = Field(default=None, description="Assigned or proposed sequence number")
    status: Literal["would_create", "created", "skipped_existing"] = Field(
        description="Import outcome for this guideline"
    )
    reason: Optional[str] = Field(default=None, description="Why the guideline was skipped, if applicable")
    created_id: Optional[str] = Field(default=None, description="Created guideline identifier")
    facets: dict = Field(
        default_factory=dict,
        description="Facet fields and source references carried from the extraction result",
    )


class GuidelineImportResponse(BaseModel):
    """Import result for moving extracted guidelines into a guide."""

    artifact_uuid: str = Field(description="Artifact UUID")
    guide_id: str = Field(description="Target guide identifier or URN")
    dry_run: bool = Field(description="Whether the operation was a dry run")
    extracted_at: str = Field(description="Timestamp of the extraction result used for import")
    source_guideline_count: int = Field(
        description="Number of raw extracted guideline rows in the persisted extraction result"
    )
    total_candidates: int = Field(
        description="Number of unique extracted guidelines considered for import before guide-level dedupe"
    )
    existing_guidelines_scanned: int = Field(
        description="Number of existing guide guidelines scanned for dedupe and sequence numbers"
    )
    total_created: int = Field(description="Number of guidelines created on the target guide")
    total_skipped: int = Field(description="Number of guidelines skipped because they already existed")
    next_sequence_no_start: int = Field(
        description="First proposed or assigned sequence number for this import batch"
    )
    schema_version: int = Field(
        default=1,
        description="Schema version of the extraction result that was imported",
    )
    existing_scan_complete: bool = Field(
        default=True,
        description=(
            "False when a scan limit truncated the existing-guideline scan, meaning "
            "dedupe and sequence numbering saw only part of the guide"
        ),
    )
    items: List[GuidelineImportItemResponse] = Field(
        description="Per-guideline import outcomes in the order they were evaluated"
    )
