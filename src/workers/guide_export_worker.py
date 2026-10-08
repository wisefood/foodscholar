"""Redis-backed background worker for guide Excel exports."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from config import config
from services.guide_export_jobs import GuideExportJobService, GuideExportQueueUnavailable

logger = logging.getLogger(__name__)


class BackgroundGuideExportWorker:
    """Background thread that drains queued guide exports from Redis."""

    def __init__(self, poll_interval: int = 5, *, job_service: GuideExportJobService | None = None):
        self.poll_interval = poll_interval
        self.job_service = job_service or GuideExportJobService()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._redis_down = False

        self.stats = {"processed": 0, "failed": 0, "started_at": None}

    def start(self) -> None:
        if self._running:
            logger.warning("Guide export worker already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GuideExportWorker")
        self._thread.start()
        logger.info("Background guide export worker started")

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("Background guide export worker stopped. Stats: %s", self.stats)

    def _process_job(self, job: dict[str, Any]) -> None:
        """Process a single queued export."""
        guide_urn = job["guide_urn"]

        if not self.job_service.is_current_job(job):
            logger.info("Skipping superseded guide export for %s (job_id=%s)", guide_urn, job.get("job_id"))
            return

        if not self.job_service.try_claim_job(guide_urn):
            logger.info("Guide export for %s is already locked by another worker", guide_urn)
            return

        try:
            state = self.job_service.run_export(job)
            self.stats["processed"] += 1
            logger.info(
                "Exported %s guidelines of %s (%s bytes)",
                state.get("total_guidelines"),
                guide_urn,
                state.get("size_bytes"),
            )
        except Exception as exc:
            # An export is a few seconds of reads; a transient failure costs the
            # curator one more click, so it is reported rather than retried.
            self.stats["failed"] += 1
            logger.error("Guide export for %s failed: %s", guide_urn, exc, exc_info=True)
            try:
                self.job_service.mark_failed(guide_urn, str(exc) or exc.__class__.__name__)
            except GuideExportQueueUnavailable:
                logger.warning("Redis unavailable while recording the failed export of %s", guide_urn)
        finally:
            self.job_service.release_lock_best_effort(guide_urn)

    def _run(self) -> None:
        while self._running:
            try:
                if not self.job_service.redis_available():
                    if not self._redis_down:
                        logger.error("Redis unavailable. Guide export worker paused until Redis recovers.")
                        self._redis_down = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._redis_down:
                    logger.info("Redis connection restored. Resuming guide export worker.")
                    self._redis_down = False

                job = self.job_service.pop_next_job(timeout=self.poll_interval)
                if job is not None:
                    self._process_job(job)

            except GuideExportQueueUnavailable:
                self._redis_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)
            except Exception as exc:
                logger.error("Unexpected error in guide export worker loop: %s", exc)
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

    def get_stats(self) -> Dict[str, Any]:
        try:
            pending_jobs = self.job_service.redis.client.llen(self.job_service.queue_key)
        except Exception:
            pending_jobs = None

        return {
            **self.stats,
            "running": self._running,
            "queue_key": self.job_service.queue_key,
            "pending_jobs": pending_jobs,
        }


_worker_instance: Optional[BackgroundGuideExportWorker] = None


def get_guide_export_worker() -> BackgroundGuideExportWorker:
    """Get the global guide export worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = BackgroundGuideExportWorker(
            poll_interval=int(config.settings.get("GUIDE_EXPORT_WORKER_POLL_INTERVAL", 5))
        )
    return _worker_instance


def start_guide_export_worker() -> None:
    get_guide_export_worker().start()


def stop_guide_export_worker() -> None:
    get_guide_export_worker().stop()
