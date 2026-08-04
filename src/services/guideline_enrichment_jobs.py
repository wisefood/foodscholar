"""Redis-backed orchestration for guideline facet-enrichment jobs."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config
from services.guideline_enricher import (
    GUIDELINE_ENRICHMENT_VERSION,
    EnrichmentOutcome,
    GuidelineEnricher,
)

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


class GuidelineEnrichmentQueueUnavailable(RuntimeError):
    """Raised when Redis-backed enrichment job operations are unavailable."""


class GuidelineEnrichmentProgressStore:
    """Persist per-guide enrichment progress in PostgreSQL."""

    def start(self, guide_urn: str, version: int) -> None:
        self._upsert(
            guide_urn,
            {
                "status": "running",
                "version": version,
                "started_at": utcnow(),
                "finished_at": None,
                "error": None,
            },
        )

    def finish(self, guide_urn: str, outcome: EnrichmentOutcome, version: int) -> None:
        self._upsert(
            guide_urn,
            {
                "status": "succeeded",
                "version": version,
                "total": outcome.total,
                "enriched": outcome.enriched,
                "skipped_version": outcome.skipped_version,
                "skipped_no_facets": outcome.skipped_no_facets,
                "failed": outcome.failed,
                "context_sources": outcome.context_sources,
                "finished_at": utcnow(),
                "error": None,
            },
        )

    def fail(self, guide_urn: str, error: str, version: int) -> None:
        self._upsert(
            guide_urn,
            {
                "status": "failed",
                "version": version,
                "finished_at": utcnow(),
                "error": error[:2000],
            },
        )

    def _upsert(self, guide_urn: str, values: Dict[str, Any]) -> None:
        try:
            from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
            from models.db import GuidelineEnrichmentRecord
            from sqlalchemy.dialects.postgresql import insert

            payload = {"guide_urn": guide_urn, **values}
            stmt = insert(GuidelineEnrichmentRecord).values(**payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[GuidelineEnrichmentRecord.guide_urn],
                set_={key: stmt.excluded[key] for key in values},
            )

            factory = POSTGRES_SYNC_SESSION_FACTORY()
            with factory() as session:
                session.execute(stmt)
                session.commit()
        except Exception as exc:
            # Progress bookkeeping must never abort an enrichment run.
            logger.warning(
                "Could not record enrichment progress for %s: %s", guide_urn, exc
            )

    def fetch_all(self) -> List[Dict[str, Any]]:
        try:
            from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
            from models.db import GuidelineEnrichmentRecord
            from sqlalchemy import select

            factory = POSTGRES_SYNC_SESSION_FACTORY()
            with factory() as session:
                rows = session.execute(select(GuidelineEnrichmentRecord)).scalars().all()
                return [
                    {
                        "guide_urn": row.guide_urn,
                        "status": row.status,
                        "version": row.version,
                        "total": row.total,
                        "enriched": row.enriched,
                        "skipped_version": row.skipped_version,
                        "skipped_no_facets": row.skipped_no_facets,
                        "failed": row.failed,
                        "context_sources": row.context_sources or [],
                        "error": row.error,
                        "started_at": row.started_at.isoformat() if row.started_at else None,
                        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.warning("Could not read enrichment progress: %s", exc)
            return []


class GuidelineEnrichmentJobService:
    """Queue and run guideline enrichment jobs, one per guide."""

    def __init__(
        self,
        redis_client: Any | None = None,
        enricher: GuidelineEnricher | None = None,
        progress_store: GuidelineEnrichmentProgressStore | None = None,
    ):
        self._redis = redis_client
        self._enricher = enricher
        self.progress = progress_store or GuidelineEnrichmentProgressStore()
        self.queue_key = str(
            config.settings.get(
                "GUIDELINE_ENRICHMENT_QUEUE_KEY", "guideline_enrichment:queue"
            )
        )
        self.lock_prefix = str(
            config.settings.get(
                "GUIDELINE_ENRICHMENT_LOCK_PREFIX", "guideline_enrichment:lock"
            )
        )
        self.processing_timeout = int(
            config.settings.get("GUIDELINE_ENRICHMENT_LOCK_TIMEOUT", 7200)
        )

    @property
    def redis(self):
        if self._redis is None:
            from backend.redis import RedisClientSingleton

            self._redis = RedisClientSingleton()
        return self._redis

    @property
    def enricher(self) -> GuidelineEnricher:
        if self._enricher is None:
            self._enricher = GuidelineEnricher()
        return self._enricher

    def redis_available(self) -> bool:
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return False
            ping = getattr(client, "ping", None)
            if callable(ping):
                ping()
                return True
            client.get("__foodscholar_guideline_enrichment_ping__")
            return True
        except Exception:
            return False

    def _redis_call(self, op: str, fn):
        try:
            return fn()
        except Exception as exc:
            raise GuidelineEnrichmentQueueUnavailable(op) from exc

    def _lock_key(self, guide_urn: str) -> str:
        return f"{self.lock_prefix}:{guide_urn}"

    def try_claim(self, guide_urn: str) -> bool:
        acquired = self._redis_call(
            "redis.set(lock)",
            lambda: self.redis.client.set(
                self._lock_key(guide_urn),
                utcnow_iso(),
                nx=True,
                ex=self.processing_timeout,
            ),
        )
        return bool(acquired)

    def release_lock_best_effort(self, guide_urn: str) -> None:
        try:
            client = getattr(self.redis, "client", None)
            if client is not None:
                client.delete(self._lock_key(guide_urn))
        except Exception:
            return

    def enqueue(
        self,
        guide_urns: Optional[List[str]] = None,
        *,
        force: bool = False,
        allow_pdf_profile: bool = True,
    ) -> Dict[str, Any]:
        """
        Queue enrichment for the named guides, or for every guide with rules.

        Enqueuing everything is the intended way to run the backfill: each guide
        is an independent job, so a failure affects one guide and a re-run skips
        whatever already reached the current version.
        """
        targets = guide_urns or self.enricher.list_guide_urns()
        version = self.enricher.version
        queued: List[str] = []

        for guide_urn in targets:
            job = {
                "job_id": str(uuid.uuid4()),
                "guide_urn": guide_urn,
                "version": version,
                "force": force,
                "allow_pdf_profile": allow_pdf_profile,
                "enqueued_at": utcnow_iso(),
            }
            payload = json.dumps(job)
            self._redis_call(
                "redis.rpush(queue)",
                lambda payload=payload: self.redis.client.rpush(
                    self.queue_key, payload
                ),
            )
            self.progress._upsert(
                guide_urn,
                {"status": "queued", "version": version, "error": None},
            )
            queued.append(guide_urn)

        return {
            "queued": len(queued),
            "guide_urns": queued,
            "version": version,
            "force": force,
        }

    def pop_next_job(self, timeout: int) -> Optional[Dict[str, Any]]:
        item = self._redis_call(
            "redis.blpop(queue)",
            lambda: self.redis.client.blpop(self.queue_key, timeout=timeout),
        )
        if not item:
            return None
        _, payload = item
        return json.loads(payload)

    def run_job(self, job: Dict[str, Any]) -> EnrichmentOutcome:
        """Execute one queued guide enrichment, recording progress either way."""
        guide_urn = job["guide_urn"]
        version = job.get("version", GUIDELINE_ENRICHMENT_VERSION)

        self.progress.start(guide_urn, version)
        try:
            outcome = self.enricher.enrich_guide(
                guide_urn,
                force=bool(job.get("force")),
                allow_pdf_profile=bool(job.get("allow_pdf_profile", True)),
            )
            self.progress.finish(guide_urn, outcome, version)
            return outcome
        except Exception as exc:
            self.progress.fail(guide_urn, str(exc), version)
            raise

    def preview(
        self,
        guide_urn: str,
        *,
        limit: int = 10,
        allow_pdf_profile: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the agent over a handful of a guide's rules and return the proposals.

        Writes nothing. This is the step to run before any real backfill: it
        shows both the facets that would be written and the guide context they
        were inferred from, which is what to sanity-check.
        """
        outcome = self.enricher.enrich_guide(
            guide_urn,
            dry_run=True,
            limit=limit,
            force=True,
            allow_pdf_profile=allow_pdf_profile,
        )
        return {
            "guide_urn": guide_urn,
            "version": self.enricher.version,
            "context_sources": outcome.context_sources,
            "guide_context": outcome.context_summary,
            "examined": outcome.total,
            "would_enrich": len(outcome.proposals),
            "no_facets": outcome.skipped_no_facets,
            "failed": outcome.failed,
            "proposals": outcome.proposals,
        }

    def status(self) -> Dict[str, Any]:
        pending = None
        try:
            pending = self.redis.client.llen(self.queue_key)
        except Exception:
            pending = None

        rows = self.progress.fetch_all()
        return {
            "version": self.enricher.version,
            "queue_key": self.queue_key,
            "pending_jobs": pending,
            "guides": rows,
            "totals": {
                "guides": len(rows),
                "enriched": sum(row.get("enriched") or 0 for row in rows),
                "skipped_version": sum(row.get("skipped_version") or 0 for row in rows),
                "failed": sum(row.get("failed") or 0 for row in rows),
            },
        }
