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
        # NOTE: This is a 0-based offset into `client.articles` (see wisefood EntityProxy slicing).
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
            Tuple of (enhance_fields, article_fields, extras_fields)
        """
        # NOTE: The data-catalog `/enhance` endpoint currently validates `fields` keys
        # and only accepts: ai_tags, ai_category, ai_key_takeaways.
        enhance_fields: Dict[str, Any] = {}
        article_fields: Dict[str, Any] = {}

        def _clean_str_list(value: Any, *, default: Optional[list[str]] = None) -> list[str]:
            if isinstance(value, str) and value.strip():
                value = [value.strip()]
            if not isinstance(value, list):
                return default or []
            cleaned: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                s = item.strip()
                if not s or s in cleaned:
                    continue
                cleaned.append(s)
            return cleaned if cleaned or default is None else default

        def _clean_optional_str(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            s = value.strip()
            return s if s else None

        combined_tags: list[str] = []
        for src in (enriched_data.get("keywords"), enriched_data.get("tags")):
            if isinstance(src, str) and src.strip():
                src = [src.strip()]
            if not isinstance(src, list):
                continue
            for t in src:
                if not isinstance(t, str):
                    continue
                tt = t.strip()
                if not tt or tt in combined_tags:
                    continue
                combined_tags.append(tt)
        if combined_tags:
            enhance_fields["ai_tags"] = combined_tags

        keywords = _clean_str_list(enriched_data.get("keywords"), default=[])
        tags = _clean_str_list(enriched_data.get("tags"), default=["Other"])
        topics = _clean_str_list(enriched_data.get("topics"), default=["Other"])[:3]
        hard_exclusion_flags = _clean_str_list(
            enriched_data.get("hard_exclusion_flags"), default=["None"]
        )

        reader_group = _clean_optional_str(enriched_data.get("reader_group"))
        age_group = _clean_optional_str(enriched_data.get("age_group"))
        population_group = _clean_optional_str(enriched_data.get("population_group"))
        geographic_context = enriched_data.get("geographic_context")
        biological_model = _clean_optional_str(enriched_data.get("biological_model"))

        study_type = _clean_optional_str(enriched_data.get("study_type"))
        if study_type is not None:
            enhance_fields["ai_category"] = study_type

        try:
            conf_val = enriched_data.get("annotation_confidence")
            conf = float(conf_val) if conf_val is not None else None
        except Exception:
            conf = None
        annotation_confidence = max(0.0, min(1.0, conf)) if conf is not None else None

        evaluation = enriched_data.get("evaluation") if isinstance(enriched_data.get("evaluation"), dict) else {}
        verdict = _clean_str_list(evaluation.get("verdict"), default=[])
        if verdict:
            enhance_fields["ai_key_takeaways"] = verdict[:3]

        # Standard article fields should be updated via PATCH /articles/{urn} (save),
        # not via /enhance.
        article_fields["keywords"] = keywords
        article_fields["tags"] = tags
        article_fields["topics"] = topics
        article_fields["hard_exclusion_flags"] = hard_exclusion_flags
        if reader_group is not None:
            article_fields["reader_group"] = reader_group
        if age_group is not None:
            article_fields["age_group"] = age_group
        if population_group is not None:
            article_fields["population_group"] = population_group
        if isinstance(geographic_context, dict) and geographic_context:
            article_fields["geographic_context"] = geographic_context
        if biological_model is not None:
            article_fields["biological_model"] = biological_model
        if study_type is not None:
            article_fields["study_type"] = study_type
        if annotation_confidence is not None:
            article_fields["annotation_confidence"] = annotation_confidence

        # Everything else goes to extras
        extras_fields = {
            "annotations": enriched_data.get("annotations", {}),
            "evaluation": evaluation,
            "enriched_at": datetime.now().isoformat(),
        }

        return enhance_fields, article_fields, extras_fields

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
            enhance_fields, article_fields, extras_fields = self._extract_enrichment_fields(
                enriched_data
            )

            # Update article in catalog
            client = WISEFOOD.get_client()
            try:
                updated_article = client.articles[article_id]
                updated_article.enhance_self(
                    agent="foodscholar-v1", fields=enhance_fields
                )
                original_sync = getattr(updated_article, "sync", True)
                try:
                    updated_article.sync = False
                    for field_name, field_value in article_fields.items():
                        setattr(updated_article, field_name, field_value)
                    updated_article.extras = extras_fields
                    updated_article.save(only_dirty=True)
                finally:
                    updated_article.sync = original_sync

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
        """Get current pagination offset from Redis (0-based)."""
        cursor = self.redis.get(self._cursor_key)
        if not cursor:
            return 0
        try:
            return max(0, int(cursor))
        except Exception:
            logger.warning("Invalid enrichment cursor value in Redis; resetting to 0")
            return 0

    def _set_cursor(self, cursor: int):
        """Save pagination offset to Redis."""
        self.redis.set(self._cursor_key, str(max(0, int(cursor))))

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
                    end_idx = cursor + self.batch_size
                    articles = client.articles[cursor:end_idx]
                    logger.debug(f"Fetched {len(articles)} articles (cursor={cursor})")
                finally:
                    WISEFOOD.return_client(client)

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
