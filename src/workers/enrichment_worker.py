"""
Simplified background enrichment worker running as a thread in the FastAPI app.

This worker runs continuously in the background, scanning the data catalog
and enriching articles. Uses Redis to prevent duplicate processing across
multiple API replicas.
"""

import logging
import threading
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

from backend.redis import RedisClientSingleton
from backend.platform import WISEFOOD
from agents.enrichment_agent import EnrichmentAgent

logger = logging.getLogger(__name__)


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

        self.redis = RedisClientSingleton()
        self.enrichment_agent = EnrichmentAgent()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()

        # Statistics
        self.stats = {"processed": 0, "failed": 0, "skipped": 0, "started_at": None}

        # Pagination cursor - stored in Redis to persist across restarts
        self._cursor_key = "enrichment:cursor"

        logger.info("Background enrichment worker initialized")

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

        # Use SET NX (set if not exists) with expiration
        acquired = self.redis.client.set(
            lock_key,
            threading.get_ident(),  # Use thread ID as lock value
            nx=True,
            ex=self.processing_timeout,
        )

        return bool(acquired)

    def _release_lock(self, article_id: str):
        """Release the processing lock for an article."""
        lock_key = f"enrichment:lock:{article_id}"
        self.redis.delete(lock_key)

    def _is_processed(self, article_id: str) -> bool:
        """Check if article has already been successfully processed."""
        return self.redis.client.sismember("enrichment:processed", article_id)

    def _mark_processed(self, article_id: str):
        """Mark article as successfully processed."""
        self.redis.client.sadd("enrichment:processed", article_id)

    def _is_permanently_failed(self, article_id: str) -> bool:
        """Check if article has permanently failed (exceeded max retries)."""
        return self.redis.client.sismember("enrichment:failed", article_id)

    def _mark_permanently_failed(self, article_id: str):
        """Mark article as permanently failed (won't be retried)."""
        self.redis.client.sadd("enrichment:failed", article_id)
        # Clean up the retry key since we won't retry anymore
        retry_key = f"enrichment:retry:{article_id}"
        self.redis.delete(retry_key)

    def _get_retry_count(self, article_id: str) -> int:
        """Get current retry count for an article."""
        retry_key = f"enrichment:retry:{article_id}"
        count = self.redis.get(retry_key)
        return int(count) if count else 0

    def _increment_retry(self, article_id: str) -> int:
        """Increment and return retry count."""
        retry_key = f"enrichment:retry:{article_id}"
        new_count = self.redis.client.incr(retry_key)
        self.redis.client.expire(retry_key, 86400)  # Expire after 24 hours
        return new_count

    def _extract_enrichment_fields(self, enriched_data: Dict[str, Any]) -> tuple:
        """
        Extract fields from enriched data for storage.

        Returns:
            Tuple of (direct_fields, extras_fields)
        """
        # Direct fields for enhance_self
        direct_fields = {}

        if "keywords" in enriched_data:
            direct_fields["ai_tags"] = enriched_data["keywords"]

        if "study_type" in enriched_data:
            direct_fields["ai_category"] = enriched_data["study_type"]

        if "evaluation" in enriched_data and "verdict" in enriched_data["evaluation"]:
            direct_fields["ai_key_takeaways"] = enriched_data["evaluation"]["verdict"]

        # Everything else goes to extras
        extras_fields = {
            "annotations": enriched_data.get("annotations", {}),
            "reader_group": enriched_data.get("reader_group"),
            "population_group": enriched_data.get("population_group"),
            "study_type": enriched_data.get("study_type"),
            "evaluation": enriched_data.get("evaluation", {}),
            "enriched_at": datetime.now().isoformat(),
        }

        return direct_fields, extras_fields

    def _process_article(self, article) -> bool:
        """
        Process a single article.

        Args:
            article: Article object from data catalog

        Returns:
            True if successful, False otherwise
        """
        article_id = article.urn

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

            # Extract fields
            direct_fields, extras_fields = self._extract_enrichment_fields(
                enriched_data
            )

            # Update article in catalog
            client = WISEFOOD.get_client()
            try:
                updated_article = client.articles[article_id]
                updated_article.enhance_self(
                    agent="foodscholar-v1", fields=direct_fields
                )
                updated_article.extras = extras_fields

                logger.info(f"Successfully enriched article {article_id}")
            finally:
                WISEFOOD.return_client(client)

            # Mark as processed and release lock
            self._mark_processed(article_id)
            self._release_lock(article_id)
            self.stats["processed"] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to process article {article_id}: {e}")
            logger.debug(traceback.format_exc())

            # Increment retry count and release lock
            self._increment_retry(article_id)
            self._release_lock(article_id)
            self.stats["failed"] += 1

            return False

    def _get_cursor(self) -> int:
        """Get current pagination cursor from Redis."""
        cursor = self.redis.get(self._cursor_key)
        return int(cursor) if cursor else 1

    def _set_cursor(self, cursor: int):
        """Save pagination cursor to Redis."""
        self.redis.set(self._cursor_key, str(cursor))

    def _run(self):
        """Main worker loop running in background thread."""
        logger.info("Worker thread started")

        while self._running:
            try:
                # Get current cursor position
                cursor = self._get_cursor()

                # Fetch articles from catalog with pagination
                client = WISEFOOD.get_client()
                try:
                    end_idx = cursor + self.batch_size - 1
                    articles = client.articles[cursor:end_idx]
                    logger.debug(f"Fetched {len(articles)} articles (cursor={cursor})")
                finally:
                    WISEFOOD.return_client(client)

                if not articles:
                    # No more articles - reset cursor to start over
                    logger.info("Reached end of catalog, resetting cursor to 1")
                    self._set_cursor(1)
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
                logger.debug(f"Batch complete (processed {processed_in_batch}). Stats: {self.stats}")
                self._shutdown_event.wait(timeout=self.poll_interval)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                logger.debug(traceback.format_exc())
                self._shutdown_event.wait(timeout=self.poll_interval)

        logger.info("Worker thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current worker statistics."""
        return {
            **self.stats,
            "running": self._running,
            "cursor": self._get_cursor(),
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
