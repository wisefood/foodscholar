"""Guideline extraction and enrichment API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from models.guidelines import (
    GuideExportJobResponse,
    GuideExportRequest,
    GuidelineArtifactStorageResponse,
    GuidelineEnrichmentEnqueueRequest,
    GuidelineEnrichmentPreviewRequest,
    GuidelineExtractionJobResponse,
    GuidelineImportRequest,
    GuidelineImportResponse,
    GuidelineExtractionRunRequest,
)
from services.guide_export_jobs import (
    GuideExportError,
    GuideExportJobService,
    GuideExportNotFound,
    GuideExportNotReady,
    GuideExportQueueUnavailable,
)
from services.guide_export_workbook import XLSX_MEDIA_TYPE
from services.guideline_corpus import GuidelineCorpusService
from services.guideline_extractor import GuidelineExtractionError
from services.guideline_enrichment_jobs import (
    GuidelineEnrichmentJobService,
    GuidelineEnrichmentQueueUnavailable,
)
from services.guideline_jobs import (
    GuidelineImportError,
    GuidelineImportNotFoundError,
    GuidelineImportPreconditionError,
    GuidelineJobQueueUnavailable,
    GuidelineJobService,
)
from workers.guideline_extraction_worker import get_guideline_worker
from workers.guideline_enrichment_worker import get_guideline_enrichment_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guidelines", tags=["Guidelines"])

job_service = GuidelineJobService()
enrichment_job_service = GuidelineEnrichmentJobService()
corpus_service = GuidelineCorpusService()
export_service = GuideExportJobService()


@router.get("/storage/{artifact_uuid}", response_model=GuidelineArtifactStorageResponse)
async def get_guideline_storage(artifact_uuid: str):
    """
    Return the local workspace path reserved for a guideline artifact.

    The worker downloads the artifact PDF into this temporary location before
    extraction using the WiseFood platform client.
    """
    try:
        return job_service.get_storage(artifact_uuid)
    except GuidelineExtractionError as exc:
        logger.error("Error resolving guideline storage for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/extract/{artifact_uuid}",
    response_model=GuidelineExtractionJobResponse,
    status_code=202,
)
async def enqueue_guideline_extraction(
    artifact_uuid: str,
    request: Optional[GuidelineExtractionRunRequest] = None,
):
    """
    Queue a guideline extraction job for an artifact UUID.

    If a job is already queued or running for the artifact, the existing job status
    is returned and no duplicate job is enqueued.
    """
    try:
        job_service.enqueue_job(
            artifact_uuid=artifact_uuid,
            model=(request.model if request else None),
            dpi=(request.dpi if request else None),
            guide_id=(request.guide_id if request else None),
            profile_document=(request.profile_document if request else True),
            profile_page_count=(request.profile_page_count if request else None),
            force=(request.force if request else False),
        )
        return await job_service.get_job_response(artifact_uuid)
    except GuidelineExtractionError as exc:
        logger.error("Invalid guideline extraction request for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuidelineJobQueueUnavailable as exc:
        logger.error("Redis unavailable while queuing guideline job for %s: %s", artifact_uuid, exc)
        raise HTTPException(
            status_code=503,
            detail="Guideline job queue is unavailable",
        ) from exc


@router.get("/extract/{artifact_uuid}", response_model=GuidelineExtractionJobResponse)
async def get_guideline_extraction_status(artifact_uuid: str):
    """
    Return the latest job status and the latest persisted extraction result for an artifact.
    """
    try:
        return await job_service.get_job_response(artifact_uuid)
    except GuidelineExtractionError as exc:
        logger.error("Invalid guideline extraction status request for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuidelineJobQueueUnavailable as exc:
        logger.error("Redis unavailable while reading guideline job for %s: %s", artifact_uuid, exc)
        raise HTTPException(
            status_code=503,
            detail="Guideline job queue is unavailable",
        ) from exc


@router.post("/import/{artifact_uuid}", response_model=GuidelineImportResponse)
async def import_guidelines_to_guide(
    artifact_uuid: str,
    request: GuidelineImportRequest,
):
    """
    Import the latest completed extraction result into a WiseFood guide.

    This is on-demand by design. Use `dry_run=true` to preview sequence numbers and
    dedupe behavior before creating guide guidelines.
    """
    try:
        return await job_service.import_latest_result_to_guide(
            artifact_uuid=artifact_uuid,
            guide_id=request.guide_id,
            dry_run=request.dry_run,
            dedupe_against_guide=request.dedupe_against_guide,
            action_type=request.action_type,
            existing_scan_limit=request.existing_scan_limit,
            import_facets=request.import_facets,
        )
    except GuidelineExtractionError as exc:
        logger.error("Invalid guideline import request for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuidelineImportNotFoundError as exc:
        logger.error("No completed extraction result to import for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuidelineImportPreconditionError as exc:
        logger.error("Extraction not ready for import for %s: %s", artifact_uuid, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GuidelineImportError as exc:
        logger.error("Error importing guidelines for %s: %s", artifact_uuid, exc, exc_info=True)
        status_code = 404 if "not found" in str(exc).lower() else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post(
    "/export/{guide_urn}",
    response_model=GuideExportJobResponse,
    status_code=202,
)
async def enqueue_guide_export(guide_urn: str, request: Optional[GuideExportRequest] = None):
    """
    Queue an Excel export of every guideline of a guide.

    If an export of the guide is already queued or running, its status is
    returned and no duplicate is queued. The finished workbook is downloadable
    from `/export/{guide_urn}/download` until it expires.
    """
    try:
        export_service.enqueue_job(guide_urn, force=bool(request and request.force))
        return export_service.get_job_response(guide_urn)
    except GuideExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuideExportQueueUnavailable as exc:
        logger.error("Redis unavailable while queuing the export of %s: %s", guide_urn, exc)
        raise HTTPException(status_code=503, detail="Guide export queue is unavailable") from exc


@router.get("/export/{guide_urn}", response_model=GuideExportJobResponse)
async def get_guide_export_status(guide_urn: str):
    """Return the status of the latest Excel export of a guide."""
    try:
        return export_service.get_job_response(guide_urn)
    except GuideExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuideExportQueueUnavailable as exc:
        logger.error("Redis unavailable while reading the export of %s: %s", guide_urn, exc)
        raise HTTPException(status_code=503, detail="Guide export queue is unavailable") from exc


@router.get("/export/{guide_urn}/download")
async def download_guide_export(guide_urn: str):
    """Return the latest finished Excel export of a guide."""
    try:
        filename, content = export_service.get_file(guide_urn)
    except GuideExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuideExportNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GuideExportNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuideExportQueueUnavailable as exc:
        logger.error("Redis unavailable while reading the export of %s: %s", guide_urn, exc)
        raise HTTPException(status_code=503, detail="Guide export queue is unavailable") from exc

    return Response(
        content=content,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/worker/status")
async def get_guideline_worker_status():
    """Return guideline worker queue statistics."""
    worker = get_guideline_worker()
    return worker.get_stats()


@router.post("/enrichment/preview")
async def preview_guideline_enrichment(request: GuidelineEnrichmentPreviewRequest):
    """
    Run the enrichment agent over a sample of one guide's rules without writing.

    Returns both the proposed facets and the guide context they were inferred
    from, including which sources that context came from — the catalog record,
    a previous extraction run, or a profile pass over the guide's PDF. Check
    this before running a backfill: if the context is wrong, every facet under
    that guide will be wrong in the same way.
    """
    try:
        return enrichment_job_service.preview(
            guide_urn=request.guide_urn,
            limit=request.limit,
            allow_pdf_profile=request.allow_pdf_profile,
        )
    except Exception as exc:
        logger.error(
            "Failed to preview guideline enrichment for %s: %s",
            request.guide_urn,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/enrichment/enqueue", status_code=202)
async def enqueue_guideline_enrichment(
    request: Optional[GuidelineEnrichmentEnqueueRequest] = None,
):
    """
    Queue facet enrichment for specific guides, or for the whole corpus.

    One job per guide; each is independent, and a re-run skips guidelines that
    already reached the current enrichment version.
    """
    request = request or GuidelineEnrichmentEnqueueRequest()
    try:
        return enrichment_job_service.enqueue(
            guide_urns=request.guide_urns,
            force=request.force,
            allow_pdf_profile=request.allow_pdf_profile,
        )
    except GuidelineEnrichmentQueueUnavailable as exc:
        logger.error("Redis unavailable while queueing guideline enrichment: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Guideline enrichment queue is unavailable",
        ) from exc


@router.get("/enrichment/status")
async def get_guideline_enrichment_status():
    """Return per-guide enrichment progress and queue depth."""
    return enrichment_job_service.status()


@router.get("/enrichment/worker/status")
async def get_guideline_enrichment_worker_status():
    """Return guideline enrichment worker statistics."""
    return get_guideline_enrichment_worker().get_stats()


@router.post("/corpus/page-summaries/backfill/{guide_urn:path}")
async def backfill_guide_page_summaries(guide_urn: str, dry_run: bool = True):
    """
    Write extraction page summaries onto a guide's existing rules.

    Rules imported before page summaries were kept have none, and re-imports
    skip existing rules. Reads the stored extraction results behind the
    guide's rules and patches ``page_summary`` onto each rule missing one.
    Idempotent; ``dry_run`` (default) only counts what would change.
    """
    try:
        return await job_service.backfill_page_summaries(
            guide_urn, dry_run=dry_run
        )
    except Exception as exc:
        logger.error(
            "Page-summary backfill for %s failed: %s", guide_urn, exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/corpus/audit")
async def audit_guideline_corpus():
    """
    Break the stored guideline corpus down by status, review state and enrichment.

    Retrieval only surfaces active guidelines, so `retrievable` is what QA can
    actually cite. Check this before and after any activation pass.
    """
    try:
        return corpus_service.audit()
    except Exception as exc:
        logger.error("Guideline corpus audit failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/corpus/activation-plan")
async def get_guideline_activation_plan(require_verified: bool = True):
    """
    Per-guide preview of what activation would make retrievable. Writes nothing.
    """
    try:
        return corpus_service.activation_plan(require_verified=require_verified)
    except Exception as exc:
        logger.error("Guideline activation plan failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/corpus/activate/{guide_urn:path}")
async def activate_guide_guidelines(
    guide_urn: str,
    require_verified: bool = True,
    dry_run: bool = True,
):
    """
    Make one guide's rules retrievable.

    Defaults to a dry run and to verified rules only; both have to be turned off
    deliberately, because activation is what puts a rule in front of users.
    """
    try:
        return corpus_service.activate_guide(
            guide_urn,
            require_verified=require_verified,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.error(
            "Guideline activation failed for %s: %s", guide_urn, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
