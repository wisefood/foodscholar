"""Article enrichment API endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from config import config
from models.enrich import (
    ArticleInput,
    EnrichmentBatchRequest,
    EnrichmentBatchResponse,
    EnrichmentBatchSummary,
    EnrichmentCriteriaBatchRequest,
    EnrichmentJobRequest,
    EnrichmentJobStatus,
    EnrichmentOverview,
    EnrichmentResetResponse,
    EnrichmentResponse,
    EnrichmentWorkerRestartRequest,
    EnrichmentWorkerRestartResponse,
    EnrichmentWorkerStatus,
    SweeperPauseRequest,
)
from services.article_enricher import ArticleEnricher
from services.enrichment_jobs import (
    ArticleNotFound,
    EnrichmentJobService,
    RedisUnavailable,
)
from workers.enrichment_job_worker import get_enrichment_job_worker
from workers.enrichment_worker import get_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enrich", tags=["Enrichment"])

article_enricher = ArticleEnricher()
job_service = EnrichmentJobService()


def _queue_unavailable(exc: Exception) -> HTTPException:
    logger.error("Redis unavailable for enrichment jobs: %s", exc)
    return HTTPException(
        status_code=503,
        detail="Enrichment job queue is unavailable",
    )


@router.post("/article", response_model=EnrichmentResponse)
async def enrich_article(request: ArticleInput):
    """
    Enrich a scientific article with annotations, keywords, and Q&A.

    Stateless: this accepts article metadata inline and returns the enrichment
    without touching the catalog. To enrich and persist a catalog article, use
    `POST /enrich/articles/{urn}`.

    This endpoint accepts article metadata and:
    1. Extracts and normalizes keywords from the abstract
    2. Classifies study type (RCT, meta-analysis, observational, etc.)
    3. Scores user value and actionability (0-5 scale)
    4. Generates a simplified abstract for general audiences
    5. Creates a glossary of technical terms
    6. Produces Q&A pairs for users, experts, and practitioners

    **Example Request:**
    ```json
    {
        "urn": "urn:article:12345",
        "title": "Effects of Omega-3 Supplementation on Cardiovascular Health",
        "abstract": "Background: Omega-3 fatty acids have been studied...",
        "authors": "Smith J, Doe A, Johnson B"
    }
    ```

    **Returns:**
    - Extracted keywords
    - Study type classification
    - User value and actionability scores
    - Simplified abstract
    - Glossary of key terms
    - Q&A for different expertise levels
    """
    try:
        logger.info(f"Enrichment request for article: {request.urn}")
        result = article_enricher.enrich_article(request)
        return result

    except Exception as e:
        logger.error(f"Error enriching article {request.urn}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error enriching article: {str(e)}",
        )


@router.get("/overview", response_model=EnrichmentOverview)
async def get_enrichment_overview():
    """
    Corpus-wide enrichment coverage with a per-journal breakdown.

    One aggregation over the live catalog: how many articles carry enrichment,
    how many are pending, and the same split for the largest journals — the
    administrator's map of where to point the next batch.
    """
    try:
        return job_service.overview()
    except Exception as exc:
        logger.error("Enrichment overview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/batches",
    response_model=EnrichmentBatchSummary,
    status_code=202,
)
async def enqueue_enrichment_batch_by_criteria(
    request: EnrichmentCriteriaBatchRequest,
):
    """
    Queue an enrichment batch by criteria (journal, missing-only, limit).

    Idempotent across runs: the default selection excludes already-enriched
    articles and never duplicates in-flight jobs, so re-running the same
    criteria continues where the previous batch stopped. Returns a batch_id
    to track progress via GET /enrich/batches/{batch_id}.
    """
    try:
        return job_service.enqueue_batch(
            venue=request.venue,
            only_missing=request.only_missing,
            force=request.force,
            limit=request.limit,
            requested_by=request.requested_by,
        )
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc
    except Exception as exc:
        logger.error("Criteria batch enqueue failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/batches", response_model=list[EnrichmentBatchSummary])
async def list_enrichment_batches():
    """Recent criteria batches, newest first."""
    try:
        return job_service.list_batches()
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.get("/batches/{batch_id}", response_model=EnrichmentBatchSummary)
async def get_enrichment_batch(batch_id: str):
    """One batch's live progress: per-status counts and recent failures."""
    try:
        batch = job_service.get_batch(batch_id)
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc
    if batch is None:
        raise HTTPException(status_code=404, detail="Unknown batch id.")
    return batch


@router.post(
    "/articles",
    response_model=EnrichmentBatchResponse,
    status_code=202,
)
async def enqueue_article_enrichment_batch(request: EnrichmentBatchRequest):
    """
    Queue selective enrichment for several catalog articles at once.

    Duplicates within the batch are collapsed, and articles already queued or
    running keep their existing job.
    """
    try:
        job_service.enqueue_many(
            request.urns,
            force=request.force,
            requested_by=request.requested_by,
        )
        statuses = job_service.get_statuses(request.urns)
        return EnrichmentBatchResponse(total=len(statuses), jobs=statuses)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.post(
    "/articles/{urn:path}",
    response_model=EnrichmentJobStatus,
    status_code=202,
)
async def enqueue_article_enrichment(
    urn: str,
    request: Optional[EnrichmentJobRequest] = None,
):
    """
    Queue selective enrichment for one catalog article.

    The job is drained by the on-demand enrichment worker, which runs
    independently of the catalog sweeper — selective enrichment keeps working
    while the sweeper is paused or disabled.

    If a job for this article is already queued or running, the existing job is
    returned and no duplicate is enqueued. Pass `force=true` to re-enrich an
    article that was already processed.
    """
    try:
        job_service.enqueue(
            urn,
            force=(request.force if request else False),
            requested_by=(request.requested_by if request else None),
        )
        return job_service.get_status(urn)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.get("/jobs", response_model=EnrichmentBatchResponse)
async def get_enrichment_statuses(
    urns: list[str] = Query(
        default=[],
        description="Article URNs to look up (repeat the parameter per URN)",
    ),
):
    """
    Enrichment status for a set of articles.

    Used by the console article list to badge each row without fetching a job
    per article.
    """
    if not urns:
        return EnrichmentBatchResponse(total=0, jobs=[])

    try:
        statuses = job_service.get_statuses(urns)
        return EnrichmentBatchResponse(total=len(statuses), jobs=statuses)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.get("/articles/{urn:path}", response_model=EnrichmentJobStatus)
async def get_article_enrichment_status(urn: str):
    """
    Enrichment status for one article.

    Merges the on-demand job record with the sweeper's processed/failed
    bookkeeping, so an article enriched by either path reports as `succeeded`.
    """
    try:
        return job_service.get_status(urn)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.delete("/articles/{urn:path}", response_model=EnrichmentResetResponse)
async def reset_article_enrichment(urn: str):
    """
    Clear sweeper bookkeeping for an article so it becomes eligible again.

    Removes it from the processed and permanently-failed sets and clears its
    retry counter. Does not delete enrichment already written to the catalog.
    """
    try:
        return job_service.reset_article(urn)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc


@router.get("/worker", response_model=EnrichmentWorkerStatus)
async def get_enrichment_worker_status():
    """
    Status of both enrichment workers.

    `sweeper` is the catalog-scanning worker (start/stop via
    `ENABLE_BACKGROUND_WORKER`, pause/resume at runtime). `jobs` is the
    on-demand worker that serves selective enrichment.
    """
    sweeper_enabled = bool(config.settings["ENABLE_BACKGROUND_WORKER"])
    sweeper: dict = {"enabled": sweeper_enabled}
    if sweeper_enabled:
        sweeper.update(get_worker().get_stats())
    else:
        # Still report the pause flag so the console shows a consistent switch.
        try:
            sweeper["paused"] = job_service.is_sweeper_paused()
        except RedisUnavailable:
            sweeper["paused"] = None
        sweeper["running"] = False

    jobs_enabled = bool(config.settings["ENABLE_ENRICHMENT_JOB_WORKER"])
    jobs: dict = {"enabled": jobs_enabled}
    if jobs_enabled:
        jobs.update(get_enrichment_job_worker().get_stats())
    else:
        jobs["running"] = False
        jobs["pending_jobs"] = job_service.pending_jobs()

    return EnrichmentWorkerStatus(sweeper=sweeper, jobs=jobs)


@router.post("/worker/pause", response_model=EnrichmentWorkerStatus)
async def set_sweeper_paused(request: SweeperPauseRequest):
    """
    Pause or resume the catalog sweeper at runtime.

    The switch lives in Redis, so it applies to every API replica and survives
    restarts. Selective enrichment jobs continue to be served while paused.
    """
    try:
        job_service.set_sweeper_paused(request.paused)
    except RedisUnavailable as exc:
        raise _queue_unavailable(exc) from exc

    logger.info(
        "Enrichment sweeper %s via API", "paused" if request.paused else "resumed"
    )
    return await get_enrichment_worker_status()


@router.post("/worker/restart", response_model=EnrichmentWorkerRestartResponse)
async def restart_enrichment_workers(request: EnrichmentWorkerRestartRequest):
    """
    Force the enrichment workers back into a running state.

    Resume is not always enough. Two states leave a worker stopped in a way the
    pause switch cannot describe:

    - The pause key in Redis has no expiry, so a pause set weeks ago survives
      every deploy. The console shows "paused" and resuming works, but nobody
      remembers who set it.
    - A worker thread that died leaves `running: true` behind, and `start()`
      reads that and decides there is nothing to do. The console shows
      "running" while nothing is being processed.

    This clears the pause switch and rebuilds the thread, so a worker that is
    stale-paused, dead, or both comes back in one action.
    """
    result: dict = {"sweeper": None, "jobs": None}

    if request.sweeper:
        if config.settings["ENABLE_BACKGROUND_WORKER"]:
            result["sweeper"] = get_worker().restart(resume=request.resume)
        else:
            # The thread is disabled by configuration, so there is nothing to
            # revive — but the pause switch is still worth clearing, otherwise
            # enabling the worker later starts it straight into a stale pause.
            cleared = None
            if request.resume:
                try:
                    cleared = job_service.is_sweeper_paused()
                    if cleared:
                        job_service.set_sweeper_paused(False)
                except RedisUnavailable as exc:
                    raise _queue_unavailable(exc) from exc
            result["sweeper"] = {
                "restarted": False,
                "reason": "ENABLE_BACKGROUND_WORKER is off in this replica",
                "pause_switch_was_set": cleared,
                "resumed": bool(cleared),
            }

    if request.jobs:
        if config.settings["ENABLE_ENRICHMENT_JOB_WORKER"]:
            result["jobs"] = get_enrichment_job_worker().restart()
        else:
            result["jobs"] = {
                "restarted": False,
                "reason": "ENABLE_ENRICHMENT_JOB_WORKER is off in this replica",
            }

    logger.info("Enrichment workers restarted via API: %s", result)
    status = await get_enrichment_worker_status()
    return EnrichmentWorkerRestartResponse(
        sweeper=result["sweeper"], jobs=result["jobs"], status=status
    )
