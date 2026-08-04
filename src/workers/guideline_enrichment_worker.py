"""Redis-backed background worker for guideline facet-enrichment jobs."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from config import config
from services.guideline_enrichment_jobs import (
    GuidelineEnrichmentJobService,
    GuidelineEnrichmentQueueUnavailable,
)

logger = logging.getLogger(__name__)


class BackgroundGuidelineEnrichmentWorker:
    """Background thread that drains queued guideline enrichment jobs."""

    def __init__(
        self,
        poll_interval: int = 5,
        *,
        job_service: GuidelineEnrichmentJobService | None = None,
    ):
        self.poll_interval = poll_interval
        self.job_service = job_service or GuidelineEnrichmentJobService()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._redis_down = False

        self.stats = {
            "processed": 0,
            "failed": 0,
            "guidelines_enriched": 0,
            "started_at": None,
        }

        logger.info("Background guideline enrichment worker initialized")

    def start(self) -> None:
        if self._running:
            logger.warning("Guideline enrichment worker already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="GuidelineEnrichmentWorker",
        )
        self._thread.start()
        logger.info("Background guideline enrichment worker started")

    def stop(self) -> None:
        if not self._running:
            return

        logger.info("Stopping background guideline enrichment worker...")
        self._running = False
        self._shutdown_event.set()

        if self._thread:
            self._thread.join(timeout=30)

        logger.info(
            "Background guideline enrichment worker stopped. Stats: %s", self.stats
        )

    def _process_job(self, job: Dict[str, Any]) -> None:
        guide_urn = job["guide_urn"]

        if not self.job_service.try_claim(guide_urn):
            logger.info(
                "Enrichment for guide %s is already locked by another worker",
                guide_urn,
            )
            return

        try:
            outcome = self.job_service.run_job(job)
            self.stats["processed"] += 1
            self.stats["guidelines_enriched"] += outcome.enriched
            logger.info(
                "Completed guideline enrichment for %s (%s enriched, %s skipped)",
                guide_urn,
                outcome.enriched,
                outcome.skipped_version,
            )
        except Exception as exc:
            self.stats["failed"] += 1
            logger.error(
                "Failed guideline enrichment for %s: %s", guide_urn, exc, exc_info=True
            )
        finally:
            self.job_service.release_lock_best_effort(guide_urn)

    def _run(self) -> None:
        logger.info("Guideline enrichment worker thread started")

        while self._running:
            try:
                if not self.job_service.redis_available():
                    if not self._redis_down:
                        logger.error(
                            "Redis unavailable. Guideline enrichment worker paused "
                            "until Redis recovers."
                        )
                        self._redis_down = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._redis_down:
                    logger.info(
                        "Redis connection restored. Resuming guideline enrichment worker."
                    )
                    self._redis_down = False

                job = self.job_service.pop_next_job(timeout=self.poll_interval)
                if job is None:
                    continue

                self._process_job(job)

            except GuidelineEnrichmentQueueUnavailable:
                self._redis_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)
            except Exception as exc:
                logger.error(
                    "Unexpected error in guideline enrichment worker loop: %s", exc
                )
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

        logger.info("Guideline enrichment worker thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        pending_jobs = None
        try:
            pending_jobs = self.job_service.redis.client.llen(
                self.job_service.queue_key
            )
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


_worker_instance: Optional[BackgroundGuidelineEnrichmentWorker] = None


def get_guideline_enrichment_worker() -> BackgroundGuidelineEnrichmentWorker:
    """Get the global guideline enrichment worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = BackgroundGuidelineEnrichmentWorker(
            poll_interval=int(
                config.settings.get("GUIDELINE_ENRICHMENT_WORKER_POLL_INTERVAL", 5)
            )
        )
    return _worker_instance


def start_guideline_enrichment_worker() -> None:
    """Start the global guideline enrichment worker."""
    get_guideline_enrichment_worker().start()


def stop_guideline_enrichment_worker() -> None:
    """Stop the global guideline enrichment worker."""
    get_guideline_enrichment_worker().stop()
