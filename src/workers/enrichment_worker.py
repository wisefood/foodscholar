"""
Simplified background enrichment worker running as a thread in the FastAPI app.

This worker runs continuously in the background, scanning the data catalog
and enriching articles. Uses Redis to prevent duplicate processing across
multiple API replicas.

Selective, per-article enrichment lives in ``workers.enrichment_job_worker``.
Both share the persistence logic in ``services.enrichment_jobs`` so a manually
enriched article is written exactly like a swept one, and both read the same
Redis pause switch.
"""

import logging
import threading
import traceback
import time
from typing import Optional, Dict, Any
from datetime import datetime

from services.enrichment_jobs import (
    CatalogUnavailable,
    EnrichmentJobService,
    RedisUnavailable,
    extract_enrichment_fields,
    is_catalog_unavailable_error,
    persist_enrichment,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BackgroundEnrichmentWorker",
    "CatalogUnavailable",
    "RedisUnavailable",
    "get_worker",
    "start_background_worker",
    "stop_background_worker",
]


class BackgroundEnrichmentWorker:
    """
    Background worker that runs in a separate thread within the FastAPI app.

    Features:
    - Runs continuously in background thread
    - Uses Redis locks to prevent duplicate processing across replicas
    - Graceful shutdown on app termination
    - Automatic retry on failures
    """

    def __init__(
        self,
        batch_size: int = 50,
        poll_interval: int = 10,
        max_retries: int = 3,
        processing_timeout: int = 300,
        *,
        redis_client: Optional[Any] = None,
        enrichment_agent: Optional[Any] = None,
        job_service: Optional[EnrichmentJobService] = None,
    ):
        """
        Initialize the background worker.

        Args:
            batch_size: Number of articles to fetch per batch
            poll_interval: Seconds to wait between polling cycles
            max_retries: Maximum retry attempts per article
            processing_timeout: Seconds before a task lock expires
        """
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.processing_timeout = processing_timeout

        # Lazy imports + DI make this module testable without external deps.
        if redis_client is None:
            from backend.redis import RedisClientSingleton  # local import (optional in tests)

            self.redis = RedisClientSingleton()
        else:
            self.redis = redis_client

        if enrichment_agent is None:
            from agents.enrichment_agent import EnrichmentAgent  # local import (optional in tests)

            self.enrichment_agent = EnrichmentAgent()
        else:
            self.enrichment_agent = enrichment_agent

        # Shared with the on-demand worker: pause switch and Redis bookkeeping.
        self.job_service = job_service or EnrichmentJobService(
            redis_client=self.redis,
            enrichment_agent=self.enrichment_agent,
        )

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()

        # Statistics
        self.stats = {"processed": 0, "failed": 0, "skipped": 0, "started_at": None}

        # Pagination cursor - stored in Redis to persist across restarts
        # NOTE: This is a 0-based offset into `client.articles` (see wisefood EntityProxy slicing).
        self._cursor_key = "enrichment:cursor"

        # Redis outage tracking (avoid log spam)
        self._redis_down = False
        self._redis_last_error_log_at = 0.0  # monotonic seconds
        self._catalog_down = False
        self._catalog_last_error_log_at = 0.0  # monotonic seconds

        # Pause-state tracking (avoid log spam while parked)
        self._paused = False

        logger.info("Background enrichment worker initialized")

    @staticmethod
    def _is_catalog_unavailable_error(exc: Exception) -> bool:
        return is_catalog_unavailable_error(exc)

    def _redis_available(self) -> bool:
        """Best-effort check that Redis is reachable."""
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return False
            ping = getattr(client, "ping", None)
            if callable(ping):
                ping()
                return True
            # Fallback for test doubles without ping()
            client.get("__foodscholar_ping__")
            return True
        except Exception:
            return False

    def _redis_call(self, op: str, fn):
        try:
            return fn()
        except Exception as e:
            raise RedisUnavailable(op) from e

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("Worker already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="EnrichmentWorker"
        )
        self._thread.start()

        logger.info("Background enrichment worker started")

    def stop(self):
        """Stop the background worker thread gracefully."""
        if not self._running:
            return

        logger.info("Stopping background enrichment worker...")
        self._running = False
        self._shutdown_event.set()

        if self._thread:
            self._thread.join(timeout=30)

        logger.info(f"Background enrichment worker stopped. Stats: {self.stats}")

    def _try_claim_article(self, article_id: str) -> bool:
        """
        Try to claim an article for processing using Redis lock.

        Args:
            article_id: Article URN to claim

        Returns:
            True if successfully claimed, False otherwise
        """
        lock_key = f"enrichment:lock:{article_id}"

        client = self.redis.client

        # Use SET NX (set if not exists) with expiration
        acquired = self._redis_call(
            "redis.set(lock)",
            lambda: client.set(
                lock_key,
                threading.get_ident(),  # Use thread ID as lock value
                nx=True,
                ex=self.processing_timeout,
            ),
        )

        return bool(acquired)

    def _release_lock(self, article_id: str):
        """Release the processing lock for an article."""
        lock_key = f"enrichment:lock:{article_id}"
        client = self.redis.client
        self._redis_call("redis.delete(lock)", lambda: client.delete(lock_key))

    def _release_lock_best_effort(self, article_id: str):
        lock_key = f"enrichment:lock:{article_id}"
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return
            client.delete(lock_key)
        except Exception:
            return

    def _is_processed(self, article_id: str) -> bool:
        """Check if article has already been successfully processed."""
        client = self.redis.client
        return bool(
            self._redis_call(
                "redis.sismember(processed)",
                lambda: client.sismember("enrichment:processed", article_id),
            )
        )

    def _mark_processed(self, article_id: str):
        """Mark article as successfully processed."""
        client = self.redis.client
        self._redis_call(
            "redis.sadd(processed)",
            lambda: client.sadd("enrichment:processed", article_id),
        )

    def _is_permanently_failed(self, article_id: str) -> bool:
        """Check if article has permanently failed (exceeded max retries)."""
        client = self.redis.client
        return bool(
            self._redis_call(
                "redis.sismember(failed)",
                lambda: client.sismember("enrichment:failed", article_id),
            )
        )

    def _mark_permanently_failed(self, article_id: str):
        """Mark article as permanently failed (won't be retried)."""
        client = self.redis.client
        self._redis_call(
            "redis.sadd(failed)",
            lambda: client.sadd("enrichment:failed", article_id),
        )
        # Clean up the retry key since we won't retry anymore
        retry_key = f"enrichment:retry:{article_id}"
        self._redis_call("redis.delete(retry)", lambda: client.delete(retry_key))

    def _get_retry_count(self, article_id: str) -> int:
        """Get current retry count for an article."""
        retry_key = f"enrichment:retry:{article_id}"
        client = self.redis.client
        count = self._redis_call("redis.get(retry)", lambda: client.get(retry_key))
        return int(count) if count else 0

    def _increment_retry(self, article_id: str) -> int:
        """Increment and return retry count."""
        retry_key = f"enrichment:retry:{article_id}"
        client = self.redis.client
        new_count = self._redis_call("redis.incr(retry)", lambda: client.incr(retry_key))
        self._redis_call(
            "redis.expire(retry)",
            lambda: client.expire(retry_key, 86400),  # Expire after 24 hours
        )
        return new_count

    def _extract_enrichment_fields(self, enriched_data: Dict[str, Any]) -> tuple:
        """
        Extract fields from enriched data for storage.

        Returns:
            Tuple of (enhance_fields, article_fields, extras_fields)
        """
        return extract_enrichment_fields(enriched_data)

    def _process_article(self, article) -> bool:
        """
        Process a single article.

        Args:
            article: Article object from data catalog

        Returns:
            True if successful, False otherwise
        """
        article_id = getattr(article, "urn", None)
        if not isinstance(article_id, str) or not article_id.strip():
            logger.error(f"Failed to process article with missing/invalid URN: {article_id!r}")
            self.stats["failed"] += 1
            return False
        article_id = article_id.strip()

        try:
            # Check if already processed
            if self._is_processed(article_id):
                logger.debug(f"Article {article_id} already processed")
                self.stats["skipped"] += 1
                return True

            # Check if permanently failed (exceeded retries previously)
            if self._is_permanently_failed(article_id):
                logger.debug(f"Article {article_id} permanently failed, skipping")
                self.stats["skipped"] += 1
                return True

            # Try to claim the article
            if not self._try_claim_article(article_id):
                logger.debug(f"Article {article_id} locked by another worker")
                self.stats["skipped"] += 1
                return True

            # Check retry count
            retry_count = self._get_retry_count(article_id)
            if retry_count >= self.max_retries:
                logger.warning(f"Article {article_id} exceeded max retries, marking as permanently failed")
                self._mark_permanently_failed(article_id)
                self._release_lock(article_id)
                self.stats["failed"] += 1
                return False

            logger.info(f"Processing article {article_id}")

            # Enrich the article
            enriched_data = self.enrichment_agent.enrich_article(article)

            # Shared with the on-demand worker so both paths write identically.
            persist_enrichment(article, enriched_data)

            logger.info(f"Successfully enriched article {article_id}")

            # Mark as processed and release lock
            self._mark_processed(article_id)
            self._release_lock(article_id)
            self.stats["processed"] += 1

            return True

        except RedisUnavailable:
            self._release_lock_best_effort(article_id)
            raise

        except CatalogUnavailable:
            try:
                self._release_lock(article_id)
            except RedisUnavailable:
                self._release_lock_best_effort(article_id)
            raise

        except Exception as e:
            logger.error(f"Failed to process article {article_id}: {e}")
            logger.debug(traceback.format_exc())

            # Non-retriable failures (e.g. invalid URNs) should not spam retries.
            msg = str(e) if e is not None else ""
            try:
                if "invalid urn format" in msg.lower():
                    logger.warning(
                        f"Marking article {article_id} as permanently failed due to invalid URN"
                    )
                    self._mark_permanently_failed(article_id)
                else:
                    self._increment_retry(article_id)
            except RedisUnavailable:
                # Redis went away during failure handling; best-effort unlock below.
                pass

            try:
                self._release_lock(article_id)
            except RedisUnavailable:
                self._release_lock_best_effort(article_id)
            self.stats["failed"] += 1

            return False

    def _get_cursor(self) -> int:
        """Get current pagination offset from Redis (0-based)."""
        client = self.redis.client
        cursor = self._redis_call("redis.get(cursor)", lambda: client.get(self._cursor_key))
        if not cursor:
            return 0
        try:
            return max(0, int(cursor))
        except Exception:
            logger.warning("Invalid enrichment cursor value in Redis; resetting to 0")
            return 0

    def _set_cursor(self, cursor: int):
        """Save pagination offset to Redis."""
        client = self.redis.client
        self._redis_call(
            "redis.set(cursor)",
            lambda: client.set(self._cursor_key, str(max(0, int(cursor)))),
        )

    def _run(self):
        """Main worker loop running in background thread."""
        logger.info("Worker thread started")

        from backend.platform import WISEFOOD  # local import (optional in tests)

        while self._running:
            try:
                # If Redis is down, pause (locks/cursor/dedup depend on it).
                if not self._redis_available():
                    now = time.monotonic()
                    if (not self._redis_down) or (
                        now - self._redis_last_error_log_at > 30
                    ):
                        logger.error(
                            "Redis unavailable (connection refused). Worker paused until Redis recovers."
                        )
                        self._redis_last_error_log_at = now
                    self._redis_down = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._redis_down:
                    logger.info("Redis connection restored. Resuming enrichment worker.")
                    self._redis_down = False

                # Runtime pause switch (set from the console). Honored by every
                # replica; selective enrichment jobs keep running while parked.
                if self.job_service.is_sweeper_paused():
                    if not self._paused:
                        logger.info(
                            "Enrichment sweeper paused via runtime switch. "
                            "Selective enrichment jobs are unaffected."
                        )
                        self._paused = True
                    self._shutdown_event.wait(timeout=self.poll_interval)
                    continue

                if self._paused:
                    logger.info("Enrichment sweeper resumed via runtime switch.")
                    self._paused = False

                # Get current cursor position
                cursor = self._get_cursor()

                # Fetch and process using the same client instance so entities
                # created from it aren't used after the client is returned to the pool.
                client = WISEFOOD.get_client()
                try:
                    end_idx = cursor + self.batch_size
                    try:
                        articles = client.articles[cursor:end_idx]
                    except Exception as fetch_err:
                        if self._is_catalog_unavailable_error(fetch_err):
                            raise CatalogUnavailable(str(fetch_err)) from fetch_err
                        raise
                    logger.debug(f"Fetched {len(articles)} articles (cursor={cursor})")
                    if self._catalog_down:
                        logger.info("Data-catalog connection restored. Resuming enrichment worker.")
                        self._catalog_down = False

                    if not articles:
                        # No more articles - reset cursor to start over
                        logger.info("Reached end of catalog, resetting cursor to 0")
                        self._set_cursor(0)
                        self._shutdown_event.wait(timeout=self.poll_interval)
                        continue

                    # Process each article
                    processed_in_batch = 0
                    for article in articles:
                        if not self._running:
                            break

                        self._process_article(article)
                        processed_in_batch += 1

                    # Advance cursor for next batch
                    new_cursor = cursor + len(articles)
                    self._set_cursor(new_cursor)

                    # Wait before next batch
                    logger.debug(
                        f"Batch complete (processed {processed_in_batch}). Stats: {self.stats}"
                    )
                    self._shutdown_event.wait(timeout=self.poll_interval)
                finally:
                    WISEFOOD.return_client(client)

            except RedisUnavailable:
                # Redis went away mid-batch; pause and retry later.
                now = time.monotonic()
                if (not self._redis_down) or (now - self._redis_last_error_log_at > 30):
                    logger.error(
                        "Redis unavailable during processing. Worker paused until Redis recovers."
                    )
                    self._redis_last_error_log_at = now
                self._redis_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)

            except CatalogUnavailable:
                now = time.monotonic()
                if (not self._catalog_down) or (
                    now - self._catalog_last_error_log_at > 30
                ):
                    logger.error(
                        "Data-catalog unavailable. Worker paused until it recovers."
                    )
                    self._catalog_last_error_log_at = now
                self._catalog_down = True
                self._shutdown_event.wait(timeout=self.poll_interval)

            except Exception as e:
                self._catalog_down = False
                logger.error(f"Error in worker loop: {e}")
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

        logger.info("Worker thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current worker statistics."""
        try:
            cursor = self._get_cursor()
        except RedisUnavailable:
            cursor = None
        try:
            paused = self.job_service.is_sweeper_paused()
        except RedisUnavailable:
            paused = None
        return {
            **self.stats,
            "running": self._running,
            "paused": paused,
            "cursor": cursor,
            "uptime_seconds": (
                (
                    datetime.now() - datetime.fromisoformat(self.stats["started_at"])
                ).total_seconds()
                if self.stats["started_at"]
                else 0
            ),
        }


# Global worker instance
_worker_instance: Optional[BackgroundEnrichmentWorker] = None


def get_worker() -> BackgroundEnrichmentWorker:
    """Get the global worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = BackgroundEnrichmentWorker()
    return _worker_instance


def start_background_worker():
    """Start the background enrichment worker."""
    worker = get_worker()
    worker.start()


def stop_background_worker():
    """Stop the background enrichment worker."""
    worker = get_worker()
    worker.stop()
