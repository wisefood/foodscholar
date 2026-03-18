"""Request and response models for guideline extraction endpoints."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from services.guideline_extractor import get_default_dpi, get_default_model


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


class GuidelineProcessedPage(BaseModel):
    """Summary of a relevant processed PDF page."""

    page: int = Field(description="1-based page number")
    page_summary: str = Field(description="Short summary of the page content")
    guideline_count: int = Field(description="Number of guidelines extracted from the page")


class GuidelineSkippedPage(BaseModel):
    """Summary of a skipped PDF page."""

    page: int = Field(description="1-based page number")
    decision: str = Field(description="Triage decision for the page")
    reason: str = Field(description="Why the page was skipped")


class ExtractedGuidelineItem(BaseModel):
    """A single extracted guideline sentence."""

    page: int = Field(description="1-based source page number")
    text: str = Field(description="Extracted guideline text")


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


class GuidelineExtractionJobResponse(BaseModel):
    """Current queued/running/completed status for a guideline extraction job."""

    artifact_uuid: str = Field(description="Artifact UUID")
    status: Literal["not_found", "queued", "running", "succeeded", "failed"] = Field(
        description="Current job state"
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
    action_type: str = Field(
        default="encourage",
        description="Action type to use for all newly created guidelines",
    )
    existing_scan_limit: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of existing guide guidelines to scan when deduping and calculating sequence numbers",
    )


class GuidelineImportItemResponse(BaseModel):
    """One guideline candidate or created import result."""

    rule_text: str = Field(description="Guideline rule text")
    page_no: Optional[int] = Field(default=None, description="Source PDF page number")
    action_type: str = Field(description="Action type used for the guideline")
    sequence_no: Optional[int] = Field(default=None, description="Assigned or proposed sequence number")
    status: Literal["would_create", "created", "skipped_existing"] = Field(
        description="Import outcome for this guideline"
    )
    reason: Optional[str] = Field(default=None, description="Why the guideline was skipped, if applicable")
    created_id: Optional[str] = Field(default=None, description="Created guideline identifier")


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
    items: List[GuidelineImportItemResponse] = Field(
        description="Per-guideline import outcomes in the order they were evaluated"
    )
