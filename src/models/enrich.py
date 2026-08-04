from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class ArticleInput(BaseModel):
    """Input model for article enrichment."""

    urn: str = Field(description="Article URN (unique identifier)")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract text")
    authors: Optional[str] = Field(
        default=None, description="Comma-separated list of authors"
    )


class EnrichmentResponse(BaseModel):
    """Response model for article enrichment."""

    urn: str = Field(description="Article URN")
    title: str = Field(description="Article title")
    keywords: list[str] = Field(description="Extracted and homogenized keywords")
    tags: list[str] = Field(description="Generic tags for aggregation across articles")
    reader_group: str = Field(description="Target reader group for this article")
    age_group: str = Field(description="Participant age group")
    population_group: str = Field(description="Study population group")
    geographic_context: Dict[str, Any] = Field(
        description="Geographic context (country/region and income setting)"
    )
    biological_model: str = Field(description="Human/Animal/In vitro/Mixed/Not stated")
    topics: list[str] = Field(description="1-3 normalized topic labels")
    study_type: str = Field(description="Type of study (RCT, meta-analysis, etc.)")
    hard_exclusion_flags: list[str] = Field(description="Hard exclusion flags for indexing")
    annotation_confidence: float = Field(description="0.0-1.0 confidence in classifications")
    evaluation: Dict[str, Any] = Field(
        description="User value score, actionability, verdict, and safety info"
    )
    annotations: Dict[str, Any] = Field(
        description="Simplified abstract, glossary, and Q&A sections"
    )


class EnrichmentJobRequest(BaseModel):
    """Options for queuing a selective enrichment job."""

    force: bool = Field(
        default=False,
        description=(
            "Re-enrich even if the article was already processed. Clears the "
            "sweeper's processed/failed bookkeeping first."
        ),
    )
    requested_by: Optional[str] = Field(
        default=None, description="Identifier of the operator who requested the run"
    )


class EnrichmentBatchRequest(EnrichmentJobRequest):
    """Options for queuing selective enrichment for several articles."""

    urns: List[str] = Field(
        min_length=1,
        max_length=200,
        description="Article URNs to enrich",
    )


class EnrichmentJobStatus(BaseModel):
    """Combined on-demand job state and sweeper bookkeeping for one article."""

    urn: str = Field(description="Article URN")
    status: str = Field(
        description="queued | running | succeeded | failed | not_found"
    )
    job_id: Optional[str] = Field(default=None, description="Latest job identifier")
    enqueued_at: Optional[str] = Field(default=None, description="When the job was queued")
    started_at: Optional[str] = Field(default=None, description="When processing began")
    completed_at: Optional[str] = Field(default=None, description="When processing ended")
    error: Optional[str] = Field(default=None, description="Failure reason, if any")
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Summary of what the last successful run wrote"
    )
    processed: bool = Field(
        default=False, description="Article is in the sweeper's processed set"
    )
    permanently_failed: bool = Field(
        default=False, description="Article exceeded sweeper retries"
    )


class EnrichmentBatchResponse(BaseModel):
    """Result of a batch enqueue."""

    total: int = Field(description="Number of distinct URNs queued or already in flight")
    jobs: List[EnrichmentJobStatus] = Field(description="Per-article job state")


class EnrichmentResetResponse(BaseModel):
    """Result of clearing sweeper bookkeeping for an article."""

    urn: str = Field(description="Article URN")
    cleared_processed: bool = Field(description="Removed from the processed set")
    cleared_failed: bool = Field(description="Removed from the permanently-failed set")


class EnrichmentWorkerStatus(BaseModel):
    """Combined status of both enrichment workers."""

    sweeper: Dict[str, Any] = Field(
        description="Catalog sweeper state (enabled, running, paused, stats, cursor)"
    )
    jobs: Dict[str, Any] = Field(
        description="On-demand job worker state (running, pending_jobs, stats)"
    )


class SweeperPauseRequest(BaseModel):
    """Pause or resume the catalog sweeper at runtime."""

    paused: bool = Field(description="True to pause the sweeper, False to resume it")


class EnrichmentWorkerRestartRequest(BaseModel):
    """Force the enrichment workers back into a running state."""

    sweeper: bool = Field(default=True, description="Restart the catalog sweeper")
    jobs: bool = Field(default=True, description="Restart the on-demand job worker")
    resume: bool = Field(
        default=True,
        description=(
            "Also clear the sweeper pause switch. This is what an operator "
            "usually means by restart: a pause set long ago has no expiry and "
            "survives every deploy, so restarting without clearing it would "
            "start a fresh thread that immediately parks itself again."
        ),
    )


class EnrichmentWorkerRestartResponse(BaseModel):
    """What the restart actually did, plus the resulting worker status."""

    sweeper: Optional[dict] = Field(
        default=None, description="Sweeper restart outcome, or null if not requested"
    )
    jobs: Optional[dict] = Field(
        default=None, description="Job worker restart outcome, or null if not requested"
    )
    status: "EnrichmentWorkerStatus" = Field(
        description="Worker status after the restart"
    )
