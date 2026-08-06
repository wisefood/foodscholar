"""
Redis-backed background worker for selective (on-demand) article enrichment.

Deliberately independent of the catalog sweeper in ``enrichment_worker``: the
sweeper can be disabled or paused while this worker keeps serving one-off
enrichment requests from the console.
"""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from config import config
from services.enrichment_jobs import (
    EnrichmentJobService,
    RedisUnavailable,
)

logger = logging.getLogger(__name__)


class EnrichmentJobWorker:
    """Background thread that drains queued per-article enrichment jobs."""

    JOIN_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        poll_interval: int = 5,
        *,
        job_service: EnrichmentJobService | None = None,
    ):
        self.poll_interval = poll_interval
        self.job_service = job_service or EnrichmentJobService()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._redis_down = False

        self.stats = {"processed": 0, "failed": 0, "skipped": 0, "started_at": None}

        logger.info("Enrichment job worker initialized")

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            logger.warning("Enrichment job worker already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EnrichmentJobWorker",
        )
        self._thread.start()
        logger.info("Enrichment job worker started")

    def is_alive(self) -> bool:
        """Whether the worker thread is actually running right now."""
        return bool(self._thread and self._thread.is_alive())

    def restart(self) -> dict:
        """
        Force the job worker back into a running state.

        ``start()`` short-circuits on ``_running``, which stays True if the
        thread died, so a crashed worker can never be revived by start alone.
        This tears the bookkeeping down first.
        """
        was_alive = self.is_alive()

        self._running = False
        self._shutdown_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)

        # Starting a second thread while the first refuses to stop would leave
        # two workers draining the same queue, and one more on every retry.
        if self._thread and self._thread.is_alive():
            logger.error(
                "Enrichment job worker thread did not stop within %ss; refusing "
                "to start a second one",
                self.JOIN_TIMEOUT_SECONDS,
            )
            return {
                "restarted": False,
                "reason": (
                    f"The worker thread did not stop within "
                    f"{self.JOIN_TIMEOUT_SECONDS}s. It is most likely blocked on "
                    f"a model call. Retry shortly."
                ),
                "thread_was_alive": was_alive,
                "running": True,
            }

        self._thread = None

        self.start()

        logger.info(
            "Enrichment job worker restarted (thread was %s)",
            "alive" if was_alive else "dead",
        )
        return {
            "restarted": True,
            "thread_was_alive": was_alive,
            "running": self.is_alive(),
        }

    def stop(self) -> None:
        """Stop the worker thread gracefully."""
        if not self._running:
            return

        logger.info("Stopping enrichment job worker...")
        self._running = False
        self._shutdown_event.set()

        if self._thread:
            self._thread.join(timeout=30)

        logger.info("Enrichment job worker stopped. Stats: %s", self.stats)

    def _process_job(self, job_ref: dict[str, Any]) -> None:
        """Process a single queued job reference."""
        urn = job_ref.get("urn")
        job_id = job_ref.get("job_id")
        if not isinstance(urn, str) or not urn.strip():
            logger.error("Discarding enrichment job with invalid URN: %r", urn)
            self.stats["skipped"] += 1
            return
        urn = urn.strip()

        # A newer request for the same article supersedes this one.
        if not self.job_service.is_current_job(urn, job_id):
            logger.info("Skipping stale enrichment job for %s (job_id=%s)", urn, job_id)
            self.stats["skipped"] += 1
            return

        if not self.job_service.try_claim_job(urn):
            logger.info("Enrichment job for %s is already locked by another worker", urn)
            self.stats["skipped"] += 1
            return

        try:
            self.job_service.mark_running(urn, job_id)
            summary = self.job_service.run_enrichment(urn)
            self.job_service.mark_succeeded(urn, summary)
            self.stats["processed"] += 1
            logger.info("Completed selective enrichment for %s", urn)
        except Exception as exc:
            self.stats["failed"] += 1
            logger.error("Failed selective enrichment for %s: %s", urn, exc, exc_info=True)
            try:
                self.job_service.mark_failed(urn, str(exc))
            except RedisUnavailable:
                logger.warning("Could not record enrichment failure for %s", urn)
        finally:
            self.job_service.release_job_lock_best_effort(urn)

    def _run(self) -> None:
        """Worker loop."""
        logger.info("Enrichment job worker thread started")

        while self._running:
            try:
                if not self.job_service.redis_available():
                    if not self._redis_down:
                        logger.error(
                            "Redis unavailable. Enrichment job worker paused until Redis recovers."
                        )
                        self._redis_down = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._redis_down:
                    logger.info("Redis connection restored. Resuming enrichment job worker.")
                    self._redis_down = False

                job_ref = self.job_service.pop_next_job(timeout=self.poll_interval)
                if job_ref is None:
                    continue

                self._process_job(job_ref)

            except RedisUnavailable:
                self._redis_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)
            except Exception as exc:
                logger.error("Unexpected error in enrichment job worker loop: %s", exc)
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

        logger.info("Enrichment job worker thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Return current worker statistics."""
        return {
            **self.stats,
            "running": self._running,
            "alive": self.is_alive(),
            "stalled": bool(self._running and not self.is_alive()),
            "queue_key": self.job_service.QUEUE_KEY,
            "pending_jobs": self.job_service.pending_jobs(),
            "uptime_seconds": (
                (
                    datetime.now() - datetime.fromisoformat(self.stats["started_at"])
                ).total_seconds()
                if self.stats["started_at"]
                else 0
            ),
        }


_worker_instance: Optional[EnrichmentJobWorker] = None


def get_enrichment_job_worker() -> EnrichmentJobWorker:
    """Get the global enrichment job worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = EnrichmentJobWorker(
            poll_interval=int(config.settings.get("ENRICHMENT_JOB_POLL_INTERVAL", 5))
        )
    return _worker_instance


def start_enrichment_job_worker() -> None:
    """Start the global enrichment job worker."""
    get_enrichment_job_worker().start()


def stop_enrichment_job_worker() -> None:
    """Stop the global enrichment job worker."""
    get_enrichment_job_worker().stop()
