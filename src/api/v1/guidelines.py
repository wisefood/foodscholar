"""Guideline extraction API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from models.guidelines import (
    GuidelineArtifactStorageResponse,
    GuidelineExtractionJobResponse,
    GuidelineImportRequest,
    GuidelineImportResponse,
    GuidelineExtractionRunRequest,
)
from services.guideline_extractor import GuidelineExtractionError
from services.guideline_jobs import (
    GuidelineImportError,
    GuidelineImportNotFoundError,
    GuidelineImportPreconditionError,
    GuidelineJobQueueUnavailable,
    GuidelineJobService,
)
from workers.guideline_extraction_worker import get_guideline_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guidelines", tags=["Guidelines"])

job_service = GuidelineJobService()


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


@router.get("/worker/status")
async def get_guideline_worker_status():
    """Return guideline worker queue statistics."""
    worker = get_guideline_worker()
    return worker.get_stats()
