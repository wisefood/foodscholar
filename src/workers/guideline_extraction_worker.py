"""Redis-backed background worker for guideline extraction jobs."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from config import config
from services.guideline_jobs import (
    GuidelineJobQueueUnavailable,
    GuidelineJobService,
    utcnow_iso,
)

logger = logging.getLogger(__name__)


class BackgroundGuidelineExtractionWorker:
    """Background thread that drains queued guideline extraction jobs from Redis."""

    def __init__(
        self,
        poll_interval: int = 5,
        *,
        job_service: GuidelineJobService | None = None,
    ):
        self.poll_interval = poll_interval
        self.job_service = job_service or GuidelineJobService()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._redis_down = False

        self.stats = {"processed": 0, "failed": 0, "started_at": None}

        logger.info("Background guideline extraction worker initialized")

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            logger.warning("Guideline extraction worker already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="GuidelineExtractionWorker",
        )
        self._thread.start()
        logger.info("Background guideline extraction worker started")

    def stop(self) -> None:
        """Stop the worker thread."""
        if not self._running:
            return

        logger.info("Stopping background guideline extraction worker...")
        self._running = False
        self._shutdown_event.set()

        if self._thread:
            self._thread.join(timeout=30)

        logger.info(
            "Background guideline extraction worker stopped. Stats: %s",
            self.stats,
        )

    def _on_progress(self, artifact_uuid: str, current_page: int, total_pages: int) -> None:
        """Refresh running job progress and keep the Redis lock alive."""
        try:
            self.job_service.update_progress(artifact_uuid, current_page, total_pages)
        except GuidelineJobQueueUnavailable:
            logger.warning(
                "Redis unavailable while updating guideline job progress for %s",
                artifact_uuid,
            )

    def _process_job(self, job: dict[str, Any]) -> None:
        """Process a single queued extraction job."""
        artifact_uuid = job["artifact_uuid"]

        current = self.job_service.get_job_state(artifact_uuid)
        if not current or current.get("job_id") != job.get("job_id"):
            logger.info(
                "Skipping stale guideline job for %s (job_id=%s)",
                artifact_uuid,
                job.get("job_id"),
            )
            return

        if not self.job_service.try_claim_job(artifact_uuid):
            logger.info(
                "Guideline job for %s is already locked by another worker",
                artifact_uuid,
            )
            return

        try:
            self.job_service.mark_running(job)
            storage = self.job_service.download_artifact_pdf(artifact_uuid)
            output = self.job_service.extractor.process_pdf(
                pdf_path=storage.pdf_path,
                model=job.get("model"),
                dpi=job.get("dpi"),
                progress_callback=lambda current_page, total_pages: self._on_progress(
                    artifact_uuid, current_page, total_pages
                ),
            )
            result = self.job_service.build_result(
                artifact_uuid=artifact_uuid,
                output=output,
                extracted_at=utcnow_iso(),
            )
            self.job_service.result_store.upsert_result(artifact_uuid, result)
            self.job_service.mark_succeeded(artifact_uuid, result)
            self.stats["processed"] += 1
            logger.info("Completed guideline extraction for %s", artifact_uuid)
        except Exception as exc:
            self.job_service.mark_failed(artifact_uuid, str(exc))
            self.stats["failed"] += 1
            logger.error(
                "Failed guideline extraction for %s: %s",
                artifact_uuid,
                exc,
                exc_info=True,
            )
        finally:
            self.job_service.release_lock_best_effort(artifact_uuid)

    def _run(self) -> None:
        """Worker loop."""
        logger.info("Guideline extraction worker thread started")

        while self._running:
            try:
                if not self.job_service.redis_available():
                    if not self._redis_down:
                        logger.error(
                            "Redis unavailable. Guideline extraction worker paused until Redis recovers."
                        )
                        self._redis_down = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._redis_down:
                    logger.info("Redis connection restored. Resuming guideline worker.")
                    self._redis_down = False

                job = self.job_service.pop_next_job(timeout=self.poll_interval)
                if job is None:
                    continue

                self._process_job(job)

            except GuidelineJobQueueUnavailable:
                self._redis_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)
            except Exception as exc:
                logger.error(
                    "Unexpected error in guideline extraction worker loop: %s",
                    exc,
                )
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

        logger.info("Guideline extraction worker thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Return current worker statistics."""
        pending_jobs = None
        try:
            pending_jobs = self.job_service.redis.client.llen(self.job_service.queue_key)
        except Exception:
            pending_jobs = None

        return {
            **self.stats,
            "running": self._running,
            "queue_key": self.job_service.queue_key,
            "pending_jobs": pending_jobs,
            "uptime_seconds": (
                (
                    datetime.now() - datetime.fromisoformat(self.stats["started_at"])
                ).total_seconds()
                if self.stats["started_at"]
                else 0
            ),
        }


_worker_instance: Optional[BackgroundGuidelineExtractionWorker] = None


def get_guideline_worker() -> BackgroundGuidelineExtractionWorker:
    """Get the global guideline extraction worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = BackgroundGuidelineExtractionWorker(
            poll_interval=int(config.settings.get("GUIDELINE_WORKER_POLL_INTERVAL", 5))
        )
    return _worker_instance


def start_guideline_worker() -> None:
    """Start the global guideline extraction worker."""
    get_guideline_worker().start()


def stop_guideline_worker() -> None:
    """Stop the global guideline extraction worker."""
    get_guideline_worker().stop()
