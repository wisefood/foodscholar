"""
Shared article-enrichment persistence plus Redis-backed on-demand job orchestration.

Two producers write enrichment into the catalog:

- ``workers.enrichment_worker`` sweeps the whole catalog in cursor order.
- ``workers.enrichment_job_worker`` drains selective, per-article jobs queued
  from the console.

Both go through :class:`EnrichmentJobService` so a manually enriched article is
byte-for-byte identical to a swept one. The sweeper can be paused at runtime
(Redis flag, honored by every replica) without stopping selective enrichment.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


# Terminal + in-flight job states. "not_found" is synthesized for URNs that have
# never been queued.
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_NOT_FOUND = "not_found"

ACTIVE_JOB_STATUSES = {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}


def utcnow_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


class RedisUnavailable(RuntimeError):
    """Raised when Redis is required but unreachable."""


class CatalogUnavailable(RuntimeError):
    """Raised when the data-catalog API is unreachable."""


class ArticleNotFound(RuntimeError):
    """Raised when a URN does not resolve to a catalog article."""


def is_catalog_unavailable_error(exc: Exception) -> bool:
    """Best-effort classification of transport failures against the catalog."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "connection refused",
            "failed to establish a new connection",
            "max retries exceeded",
            "httpconnectionpool",
            "newconnectionerror",
            "read timed out",
            "connect timeout",
            "temporarily unavailable",
        )
    )


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


def extract_enrichment_fields(enriched_data: Dict[str, Any]) -> tuple:
    """
    Split raw agent output into the three catalog write targets.

    Returns:
        Tuple of (enhance_fields, article_fields, extras_fields)
    """
    # NOTE: The data-catalog `/enhance` endpoint currently validates `fields` keys
    # and only accepts: ai_tags, ai_category, ai_key_takeaways.
    enhance_fields: Dict[str, Any] = {}
    article_fields: Dict[str, Any] = {}

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

    evaluation = (
        enriched_data.get("evaluation")
        if isinstance(enriched_data.get("evaluation"), dict)
        else {}
    )
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


def persist_enrichment(article, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write agent output onto a catalog article entity.

    Persists standard fields via PATCH first using the entity instance we already
    have (in-place), then attempts the optional /enhance call.

    Rationale: if /enhance fails it may leave the upstream client entity in a
    "dirty" state that causes a subsequent save() to include ai_* keys, which the
    base PATCH schema rejects.

    Returns:
        A summary dict describing what was written.
    """
    enhance_fields, article_fields, extras_fields = extract_enrichment_fields(
        enriched_data
    )

    if enhance_fields:
        extras_fields["enhance_agent"] = "foodscholar-v1"
        extras_fields["enhance_fields"] = enhance_fields

    original_sync = getattr(article, "sync", True)
    try:
        article.sync = False
        for field_name, field_value in article_fields.items():
            setattr(article, field_name, field_value)
        article.extras = extras_fields
        try:
            article.save(only_dirty=True)
        except Exception as save_err:
            if is_catalog_unavailable_error(save_err):
                raise CatalogUnavailable(str(save_err)) from save_err
            raise
    finally:
        article.sync = original_sync

    enhanced = False
    if enhance_fields:
        try:
            article.enhance_self(agent="foodscholar-v1", fields=enhance_fields)
            enhanced = True
        except Exception as enhance_err:
            logger.warning(
                "Enhance endpoint failed for article %s; skipping /enhance. Error: %s",
                getattr(article, "urn", "<unknown>"),
                enhance_err,
            )

    return {
        "enriched_at": extras_fields["enriched_at"],
        "enhanced": enhanced,
        "study_type": article_fields.get("study_type"),
        "keywords": article_fields.get("keywords", []),
        "tags": article_fields.get("tags", []),
        "topics": article_fields.get("topics", []),
        "ai_tags": enhance_fields.get("ai_tags", []),
        "ai_key_takeaways": enhance_fields.get("ai_key_takeaways", []),
        "annotation_confidence": article_fields.get("annotation_confidence"),
    }


class EnrichmentJobService:
    """Redis-backed orchestration for selective article enrichment."""

    # Sweeper bookkeeping (shared with BackgroundEnrichmentWorker).
    PROCESSED_SET = "enrichment:processed"
    FAILED_SET = "enrichment:failed"
    RETRY_PREFIX = "enrichment:retry"
    SWEEPER_LOCK_PREFIX = "enrichment:lock"

    # On-demand job bookkeeping.
    QUEUE_KEY = "enrichment:jobs:queue"
    JOB_PREFIX = "enrichment:job"
    JOB_LOCK_PREFIX = "enrichment:job:lock"
    PAUSE_KEY = "enrichment:sweeper:paused"

    # Jobs outlive the run so the console can show the last outcome.
    JOB_TTL_SECONDS = 7 * 24 * 3600

    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        enrichment_agent: Optional[Any] = None,
        catalog_pool: Optional[Any] = None,
        processing_timeout: int = 900,
    ):
        self._redis = redis_client
        self._enrichment_agent = enrichment_agent
        self._catalog_pool = catalog_pool
        self.processing_timeout = processing_timeout

    # ------------------------------------------------------------------ #
    # Lazy dependencies (keeps the module importable without Redis/catalog)
    # ------------------------------------------------------------------ #

    @property
    def redis(self):
        if self._redis is None:
            from backend.redis import RedisClientSingleton

            self._redis = RedisClientSingleton()
        return self._redis

    @property
    def enrichment_agent(self):
        if self._enrichment_agent is None:
            from agents.enrichment_agent import EnrichmentAgent

            self._enrichment_agent = EnrichmentAgent()
        return self._enrichment_agent

    @property
    def catalog_pool(self):
        if self._catalog_pool is None:
            from backend.platform import WISEFOOD

            self._catalog_pool = WISEFOOD
        return self._catalog_pool

    def redis_available(self) -> bool:
        """Best-effort check that Redis is reachable."""
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return False
            ping = getattr(client, "ping", None)
            if callable(ping):
                ping()
                return True
            client.get("__foodscholar_enrichment_ping__")
            return True
        except Exception:
            return False

    def _redis_call(self, op: str, fn):
        try:
            return fn()
        except Exception as exc:
            raise RedisUnavailable(op) from exc

    # ------------------------------------------------------------------ #
    # Sweeper pause switch
    # ------------------------------------------------------------------ #

    def is_sweeper_paused(self) -> bool:
        """Whether the catalog sweeper is paused at runtime (all replicas)."""
        value = self._redis_call(
            "redis.get(pause)", lambda: self.redis.client.get(self.PAUSE_KEY)
        )
        return str(value) == "1"

    def set_sweeper_paused(self, paused: bool) -> bool:
        """Pause or resume the catalog sweeper across every replica."""
        if paused:
            self._redis_call(
                "redis.set(pause)",
                lambda: self.redis.client.set(self.PAUSE_KEY, "1"),
            )
        else:
            self._redis_call(
                "redis.delete(pause)",
                lambda: self.redis.client.delete(self.PAUSE_KEY),
            )
        return paused

    # ------------------------------------------------------------------ #
    # Catalog access
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_urn(urn: str) -> str:
        """Normalize a caller-supplied article identifier."""
        if not isinstance(urn, str) or not urn.strip():
            raise ArticleNotFound("An article URN is required.")
        return urn.strip()

    def fetch_article(self, client, urn: str):
        """Fetch a single catalog article, mapping transport errors."""
        try:
            return client.articles.get(urn)
        except Exception as exc:
            if is_catalog_unavailable_error(exc):
                raise CatalogUnavailable(str(exc)) from exc
            raise ArticleNotFound(f"Article {urn} was not found in the catalog.") from exc

    # ------------------------------------------------------------------ #
    # Job state
    # ------------------------------------------------------------------ #

    def _job_key(self, urn: str) -> str:
        return f"{self.JOB_PREFIX}:{urn}"

    def _job_lock_key(self, urn: str) -> str:
        return f"{self.JOB_LOCK_PREFIX}:{urn}"

    def get_job_state(self, urn: str) -> Optional[dict[str, Any]]:
        """Read the latest job record for an article URN."""
        raw = self._redis_call(
            "redis.get(job)", lambda: self.redis.client.get(self._job_key(urn))
        )
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Discarding unreadable enrichment job state for %s", urn)
            return None

    def _set_job_state(self, urn: str, job_state: dict[str, Any]) -> None:
        payload = json.dumps(job_state)
        self._redis_call(
            "redis.set(job)",
            lambda: self.redis.client.set(
                self._job_key(urn), payload, ex=self.JOB_TTL_SECONDS
            ),
        )

    def enqueue(self, urn: str, *, force: bool = False, requested_by: Optional[str] = None) -> dict[str, Any]:
        """
        Queue a selective enrichment job unless one is already in flight.

        ``force`` also clears the sweeper's processed/failed bookkeeping so the
        article is eligible for a fresh pass.
        """
        urn = self.normalize_urn(urn)

        current = self.get_job_state(urn)
        if current and current.get("status") in ACTIVE_JOB_STATUSES:
            return current

        if force:
            self.reset_article(urn)

        job_state = {
            "urn": urn,
            "job_id": str(uuid.uuid4()),
            "status": JOB_STATUS_QUEUED,
            "force": bool(force),
            "requested_by": requested_by,
            "enqueued_at": utcnow_iso(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
        }
        self._set_job_state(urn, job_state)
        payload = json.dumps({"urn": urn, "job_id": job_state["job_id"]})
        self._redis_call(
            "redis.rpush(queue)",
            lambda: self.redis.client.rpush(self.QUEUE_KEY, payload),
        )
        logger.info("Queued selective enrichment for %s (force=%s)", urn, force)
        return job_state

    def enqueue_many(
        self,
        urns: Iterable[str],
        *,
        force: bool = False,
        requested_by: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Queue several articles, skipping duplicates within the batch."""
        seen: set[str] = set()
        jobs: list[dict[str, Any]] = []
        for raw_urn in urns:
            urn = self.normalize_urn(raw_urn)
            if urn in seen:
                continue
            seen.add(urn)
            jobs.append(self.enqueue(urn, force=force, requested_by=requested_by))
        return jobs

    def pop_next_job(self, timeout: int) -> Optional[dict[str, Any]]:
        """Block for the next queued job reference."""
        item = self._redis_call(
            "redis.blpop(queue)",
            lambda: self.redis.client.blpop(self.QUEUE_KEY, timeout=timeout),
        )
        if not item:
            return None
        _, payload = item
        try:
            return json.loads(payload)
        except Exception:
            logger.warning("Discarding unreadable enrichment queue entry")
            return None

    def pending_jobs(self) -> Optional[int]:
        """Queue depth, or None when Redis is unreachable."""
        try:
            return int(self.redis.client.llen(self.QUEUE_KEY))
        except Exception:
            return None

    def try_claim_job(self, urn: str) -> bool:
        """Acquire the per-article on-demand processing lock."""
        acquired = self._redis_call(
            "redis.set(job lock)",
            lambda: self.redis.client.set(
                self._job_lock_key(urn),
                utcnow_iso(),
                nx=True,
                ex=self.processing_timeout,
            ),
        )
        return bool(acquired)

    def release_job_lock_best_effort(self, urn: str) -> None:
        """Release the on-demand lock without surfacing failures."""
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return
            client.delete(self._job_lock_key(urn))
        except Exception:
            return

    def mark_running(self, urn: str, job_id: Optional[str] = None) -> dict[str, Any]:
        job = self.get_job_state(urn) or {"urn": urn, "job_id": job_id}
        job["status"] = JOB_STATUS_RUNNING
        job["started_at"] = job.get("started_at") or utcnow_iso()
        job["completed_at"] = None
        job["error"] = None
        self._set_job_state(urn, job)
        return job

    def mark_succeeded(self, urn: str, result: Dict[str, Any]) -> dict[str, Any]:
        job = self.get_job_state(urn) or {"urn": urn}
        job["status"] = JOB_STATUS_SUCCEEDED
        job["completed_at"] = utcnow_iso()
        job["error"] = None
        job["result"] = result
        self._set_job_state(urn, job)
        return job

    def mark_failed(self, urn: str, error: str) -> dict[str, Any]:
        job = self.get_job_state(urn) or {"urn": urn}
        job["status"] = JOB_STATUS_FAILED
        job["completed_at"] = utcnow_iso()
        job["error"] = error
        self._set_job_state(urn, job)
        return job

    def is_current_job(self, urn: str, job_id: Optional[str]) -> bool:
        """Whether the stored record still points at this job id."""
        current = self.get_job_state(urn)
        if not current:
            return False
        if job_id is None:
            return True
        return current.get("job_id") == job_id

    # ------------------------------------------------------------------ #
    # Sweeper bookkeeping
    # ------------------------------------------------------------------ #

    def is_processed(self, urn: str) -> bool:
        return bool(
            self._redis_call(
                "redis.sismember(processed)",
                lambda: self.redis.client.sismember(self.PROCESSED_SET, urn),
            )
        )

    def mark_processed(self, urn: str) -> None:
        self._redis_call(
            "redis.sadd(processed)",
            lambda: self.redis.client.sadd(self.PROCESSED_SET, urn),
        )

    def is_permanently_failed(self, urn: str) -> bool:
        return bool(
            self._redis_call(
                "redis.sismember(failed)",
                lambda: self.redis.client.sismember(self.FAILED_SET, urn),
            )
        )

    def reset_article(self, urn: str) -> dict[str, Any]:
        """
        Clear sweeper bookkeeping so an article becomes eligible again.

        Drops it from the processed and permanently-failed sets and clears its
        retry counter.
        """
        urn = self.normalize_urn(urn)
        client = self.redis.client
        removed_processed = bool(
            self._redis_call(
                "redis.srem(processed)",
                lambda: client.srem(self.PROCESSED_SET, urn),
            )
        )
        removed_failed = bool(
            self._redis_call(
                "redis.srem(failed)", lambda: client.srem(self.FAILED_SET, urn)
            )
        )
        self._redis_call(
            "redis.delete(retry)",
            lambda: client.delete(f"{self.RETRY_PREFIX}:{urn}"),
        )
        return {
            "urn": urn,
            "cleared_processed": removed_processed,
            "cleared_failed": removed_failed,
        }

    # ------------------------------------------------------------------ #
    # Status projection
    # ------------------------------------------------------------------ #

    def get_status(self, urn: str) -> dict[str, Any]:
        """
        Combined view of an article's enrichment state.

        Merges the on-demand job record with the sweeper's processed/failed
        bookkeeping so the console can render one badge per article.
        """
        urn = self.normalize_urn(urn)
        job = self.get_job_state(urn)

        processed = self.is_processed(urn)
        permanently_failed = self.is_permanently_failed(urn)

        if job is None:
            status = JOB_STATUS_NOT_FOUND
            if processed:
                status = JOB_STATUS_SUCCEEDED
            elif permanently_failed:
                status = JOB_STATUS_FAILED
            return {
                "urn": urn,
                "status": status,
                "job_id": None,
                "enqueued_at": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result": None,
                "processed": processed,
                "permanently_failed": permanently_failed,
            }

        return {
            "urn": urn,
            "status": job.get("status", JOB_STATUS_NOT_FOUND),
            "job_id": job.get("job_id"),
            "enqueued_at": job.get("enqueued_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "error": job.get("error"),
            "result": job.get("result"),
            "processed": processed,
            "permanently_failed": permanently_failed,
        }

    def get_statuses(self, urns: Iterable[str]) -> list[dict[str, Any]]:
        """Status for many URNs in one call (used by the article list page)."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for raw_urn in urns:
            urn = self.normalize_urn(raw_urn)
            if urn in seen:
                continue
            seen.add(urn)
            out.append(self.get_status(urn))
        return out

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run_enrichment(self, urn: str) -> Dict[str, Any]:
        """
        Enrich one catalog article end-to-end and persist the result.

        Also marks the article processed so the sweeper does not redo it.
        """
        urn = self.normalize_urn(urn)
        client = self.catalog_pool.get_client()
        try:
            article = self.fetch_article(client, urn)
            enriched_data = self.enrichment_agent.enrich_article(article)
            summary = persist_enrichment(article, enriched_data)
        finally:
            self.catalog_pool.return_client(client)

        try:
            self.mark_processed(urn)
        except RedisUnavailable:
            logger.warning(
                "Enriched %s but could not mark it processed (Redis unavailable)", urn
            )

        return summary
