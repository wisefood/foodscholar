"""Redis-backed guideline extraction job orchestration and result persistence."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import config
from models.guidelines import (
    DEFAULT_GUIDELINE_ACTION_TYPE,
    GuidelineArtifactStorageResponse,
    GuidelineActionType,
    GuidelineExtractionJobResponse,
    GuidelineExtractionResponse,
    GuidelineImportItemResponse,
    GuidelineImportResponse,
    normalize_guideline_action_type,
)
from services.guideline_extractor import (
    DEFAULT_PROFILE_PAGE_COUNT,
    EXTRACTION_SCHEMA_VERSION,
    GUIDELINE_TYPE_VALUES,
    LIFE_STAGE_VALUES,
    SETTING_VALUES,
    GuideContext,
    GuidelineExtractionError,
    GuidelineExtractorService,
    get_default_dpi,
    get_default_model,
    normalize_guideline,
)

# Free-text population hints the extractor emits, mapped onto the catalog's
# closed `target_populations` enum. Anything unmatched is left off rather than
# forced into `other`, so the enrichment pass can still decide.
TARGET_POPULATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpregnan", "pregnant_people"),
    (r"\blactat|\bbreastfeed", "lactating_people"),
    (r"\binfant|\bbab(y|ies)|\bnewborn|\bunder\s*(1|one)\s*year", "infants"),
    (r"\btoddler|\bpreschool|\bunder\s*(5|five)|\b1\s*[-–to]+\s*4\b", "under_5_years"),
    (r"\bschool[- ]age|\badolescen|\bteen|\b5\s*[-–to]+\s*18\b", "ages_5_to_18"),
    (r"\belderly|\bolder adult|\bseniors?\b|\b65\+", "elderly"),
    (r"\badults?\b", "adults"),
    (r"\bgeneral population|\beveryone\b|\ball ages\b", "general_population"),
)

# Life stages imply a target population even when the hint text does not.
LIFE_STAGE_TO_TARGET_POPULATION: dict[str, str] = {
    "pregnancy": "pregnant_people",
    "lactation": "lactating_people",
    "infancy": "infants",
    "early_childhood": "under_5_years",
    "school_age": "ages_5_to_18",
    "adolescence": "ages_5_to_18",
    "adulthood": "adults",
    "older_adulthood": "elderly",
}

# The catalog's bulk import endpoint accepts up to 1000 guidelines per call.
IMPORT_BATCH_SIZE = 500

# Extraction touches a rate-limited service for every page, so a whole-job
# failure is usually transient. Bounded so an unparseable PDF stops eventually.
MAX_EXTRACTION_ATTEMPTS = int(
    config.settings.get("GUIDELINE_EXTRACTION_MAX_ATTEMPTS", 3)
)

logger = logging.getLogger(__name__)


def utcnow_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def guideline_compare_key(text: str) -> str:
    """Normalize guideline text for exact/near-exact comparisons."""
    normalized = normalize_guideline(text).lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class GuidelineJobQueueUnavailable(RuntimeError):
    """Raised when Redis-backed guideline job operations are unavailable."""


class GuidelineArtifactDownloadError(RuntimeError):
    """Raised when an artifact PDF cannot be downloaded locally."""


class GuidelineImportError(RuntimeError):
    """Raised when guideline import into a guide cannot be completed."""


class GuidelineImportNotFoundError(GuidelineImportError):
    """Raised when there is no completed extraction result to import."""


class GuidelineImportPreconditionError(GuidelineImportError):
    """Raised when import is attempted before extraction has completed."""


class GuidelineResultStore:
    """Persist and fetch the latest extraction result per artifact from PostgreSQL."""

    @staticmethod
    def _artifact_uuid(artifact_uuid: str) -> uuid.UUID:
        return uuid.UUID(artifact_uuid)

    async def fetch_result(self, artifact_uuid: str) -> Optional[GuidelineExtractionResponse]:
        """Fetch the latest persisted extraction result for an artifact."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import GuidelineExtractionRecord

        try:
            factory = POSTGRES_ASYNC_SESSION_FACTORY()
            async with factory() as session:
                record = await session.get(
                    GuidelineExtractionRecord,
                    self._artifact_uuid(artifact_uuid),
                )
                if record is None:
                    return None
                return GuidelineExtractionResponse.model_validate(record.result_json)
        except Exception as exc:
            logger.error(
                "Failed to fetch guideline extraction result for %s: %s",
                artifact_uuid,
                exc,
                exc_info=True,
            )
            return None

    def upsert_result(self, artifact_uuid: str, result: GuidelineExtractionResponse) -> None:
        """Insert or overwrite the latest extraction result for an artifact."""
        from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
        from models.db import GuidelineExtractionRecord
        from sqlalchemy.dialects.postgresql import insert

        payload = result.model_dump(mode="json")
        timestamp = datetime.now(timezone.utc)
        stmt = insert(GuidelineExtractionRecord).values(
            artifact_id=self._artifact_uuid(artifact_uuid),
            result_json=payload,
            created_at=timestamp,
            updated_at=timestamp,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[GuidelineExtractionRecord.artifact_id],
            set_={
                "result_json": stmt.excluded.result_json,
                "updated_at": timestamp,
            },
        )

        factory = POSTGRES_SYNC_SESSION_FACTORY()
        with factory() as session:
            session.execute(stmt)
            session.commit()


class GuidelineJobService:
    """Manage guideline extraction jobs in Redis and results in PostgreSQL."""

    def __init__(
        self,
        redis_client: Any | None = None,
        extractor: GuidelineExtractorService | None = None,
        result_store: GuidelineResultStore | None = None,
        platform_pool: Any | None = None,
    ):
        self._redis = redis_client
        self.extractor = extractor or GuidelineExtractorService()
        self.result_store = result_store or GuidelineResultStore()
        self._platform_pool = platform_pool
        self.queue_key = str(config.settings.get("GUIDELINE_JOB_QUEUE_KEY", "guidelines:queue"))
        self.lock_prefix = str(
            config.settings.get("GUIDELINE_JOB_LOCK_PREFIX", "guidelines:lock")
        )
        self.job_prefix = str(
            config.settings.get("GUIDELINE_JOB_STATUS_PREFIX", "guidelines:job")
        )
        self.processing_timeout = int(
            config.settings.get("GUIDELINE_JOB_LOCK_TIMEOUT", 7200)
        )

    @property
    def redis(self):
        """Get or lazily initialize the Redis singleton."""
        if self._redis is None:
            from backend.redis import RedisClientSingleton

            self._redis = RedisClientSingleton()
        return self._redis

    @property
    def platform_pool(self):
        """Get or lazily initialize the WiseFood client pool."""
        if self._platform_pool is None:
            from backend.platform import WISEFOOD

            self._platform_pool = WISEFOOD
        return self._platform_pool

    def redis_available(self) -> bool:
        """Best-effort check for Redis availability."""
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return False
            ping = getattr(client, "ping", None)
            if callable(ping):
                ping()
                return True
            client.get("__foodscholar_guideline_ping__")
            return True
        except Exception:
            return False

    def _job_key(self, artifact_uuid: str) -> str:
        return f"{self.job_prefix}:{artifact_uuid}"

    def _lock_key(self, artifact_uuid: str) -> str:
        return f"{self.lock_prefix}:{artifact_uuid}"

    def _redis_call(self, op: str, fn):
        try:
            return fn()
        except Exception as exc:
            raise GuidelineJobQueueUnavailable(op) from exc

    def get_storage(self, artifact_uuid: str) -> GuidelineArtifactStorageResponse:
        """Resolve the local temporary workspace for an artifact UUID."""
        storage = self.extractor.get_artifact_workspace(artifact_uuid)
        return GuidelineArtifactStorageResponse.model_validate(asdict(storage))

    def download_artifact_pdf(self, artifact_uuid: str) -> GuidelineArtifactStorageResponse:
        """Download an artifact PDF into the local temporary workspace."""
        storage = self.get_storage(artifact_uuid)
        pdf_path = Path(storage.pdf_path)
        if pdf_path.exists():
            pdf_path.unlink()

        client = self.platform_pool.get_client()
        try:
            client.artifacts.download_to(artifact_uuid, str(pdf_path))
        except Exception as exc:
            raise GuidelineArtifactDownloadError(
                f"Failed to download artifact PDF for {artifact_uuid}"
            ) from exc
        finally:
            self.platform_pool.return_client(client)

        if not pdf_path.exists():
            raise GuidelineArtifactDownloadError(
                f"Artifact download finished without a local PDF for {artifact_uuid}"
            )

        return self.get_storage(artifact_uuid)

    def get_job_state(self, artifact_uuid: str) -> Optional[dict[str, Any]]:
        """Fetch the latest Redis job state for an artifact UUID."""
        raw = self._redis_call(
            "redis.get(job)",
            lambda: self.redis.client.get(self._job_key(artifact_uuid)),
        )
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        return None

    def _register_job_state(
        self, artifact_uuid: str, job_state: dict[str, Any], *, overwrite: bool
    ) -> bool:
        """
        Write the initial job status, returning whether this caller won the slot.

        With ``overwrite=False`` this is a SET NX: only the first of several
        concurrent enqueues for the same artifact succeeds.
        """
        payload = json.dumps(job_state)
        key = self._job_key(artifact_uuid)
        return bool(
            self._redis_call(
                "redis.set(job, nx)",
                lambda: self.redis.client.set(key, payload, nx=not overwrite),
            )
        )

    def _set_job_state(self, artifact_uuid: str, job_state: dict[str, Any]) -> None:
        self._redis_call(
            "redis.set(job)",
            lambda: self.redis.client.set(
                self._job_key(artifact_uuid), json.dumps(job_state)
            ),
        )

    def is_current_job(self, job: dict[str, Any]) -> bool:
        """Return whether the Redis status still points at the same job id."""
        current = self.get_job_state(job["artifact_uuid"])
        return bool(current and current.get("job_id") == job.get("job_id"))

    def resolve_guide_context(self, guide_id: str | None) -> GuideContext:
        """
        Fetch the parent guide's self-description for the extraction prompts.

        Failure is not fatal: extraction still runs, just without the guide's
        population context, which is exactly the pre-existing behavior.
        """
        if not guide_id:
            return GuideContext()

        client = self.platform_pool.get_client()
        try:
            guide = client.guides.get(guide_id)
            data = getattr(guide, "data", None)
            if not isinstance(data, dict):
                data = {
                    name: getattr(guide, name, None)
                    for name in (
                        "urn",
                        "title",
                        "short_title",
                        "region",
                        "audience",
                        "target_audiences",
                        "language",
                        "publication_year",
                        "issuing_authority",
                    )
                }
            return GuideContext.from_guide(data)
        except Exception as exc:
            logger.warning(
                "Could not resolve guide context for %s; extracting without it: %s",
                guide_id,
                exc,
            )
            return GuideContext()
        finally:
            self.platform_pool.return_client(client)

    def has_lock(self, artifact_uuid: str) -> bool:
        """Whether some worker currently holds the claim on this artifact."""
        try:
            return bool(self.redis.client.exists(self._lock_key(artifact_uuid)))
        except Exception:
            # Unknown is treated as held: refusing to re-queue is the safe error,
            # since the alternative is two workers on the same PDF.
            return True

    def is_orphaned(self, job_state: dict[str, Any] | None) -> bool:
        """
        Whether a job claims to be in flight but nothing is working on it.

        Jobs are popped destructively, so a worker that dies mid-extraction
        leaves a status of ``running`` with no queue entry and no live lock.
        Without this check the artifact could never be extracted again: the
        status blocks re-queueing and the lock outlives the process by up to its
        full TTL.
        """
        if not job_state:
            return False
        if job_state.get("status") != "running":
            return False
        return not self.has_lock(job_state["artifact_uuid"])

    def enqueue_job(
        self,
        artifact_uuid: str,
        model: str | None = None,
        dpi: int | None = None,
        guide_id: str | None = None,
        profile_document: bool = True,
        profile_page_count: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Queue an extraction job unless one is genuinely already in flight.

        ``force`` re-queues regardless; the per-artifact lock still prevents two
        workers from extracting the same PDF at once, so the worst case is a
        superseded job that the stale-job check discards.
        """
        self.get_storage(artifact_uuid)
        current = self.get_job_state(artifact_uuid)

        if current and not force and current.get("status") in {"queued", "running"}:
            if self.is_orphaned(current):
                logger.warning(
                    "Guideline job %s for %s is orphaned (running with no live "
                    "worker); re-queueing.",
                    current.get("job_id"),
                    artifact_uuid,
                )
                self.release_lock_best_effort(artifact_uuid)
            else:
                return current

        job_state = {
            "artifact_uuid": artifact_uuid,
            "job_id": str(uuid.uuid4()),
            "status": "queued",
            "model": model or get_default_model(),
            "dpi": dpi or get_default_dpi(),
            "guide_id": guide_id,
            "profile_document": profile_document,
            "profile_page_count": profile_page_count or DEFAULT_PROFILE_PAGE_COUNT,
            "attempt": 1,
            "max_attempts": MAX_EXTRACTION_ATTEMPTS,
            "enqueued_at": utcnow_iso(),
            "started_at": None,
            "completed_at": None,
            "current_page": None,
            "total_pages": None,
            "error": None,
        }
        # Registering the job is a single conditional write, so two requests
        # arriving together cannot both decide they are the one enqueueing.
        # `force` and orphan recovery deliberately overwrite.
        registered = self._register_job_state(
            artifact_uuid, job_state, overwrite=force or bool(current)
        )
        if not registered:
            existing = self.get_job_state(artifact_uuid)
            if existing:
                return existing

        payload = json.dumps(job_state)
        self._redis_call(
            "redis.rpush(queue)",
            lambda: self.redis.client.rpush(self.queue_key, payload),
        )
        return job_state

    def pop_next_job(self, timeout: int) -> Optional[dict[str, Any]]:
        """Pop the next queued job from Redis."""
        item = self._redis_call(
            "redis.blpop(queue)",
            lambda: self.redis.client.blpop(self.queue_key, timeout=timeout),
        )
        if not item:
            return None
        _, payload = item
        return json.loads(payload)

    def try_claim_job(self, artifact_uuid: str) -> bool:
        """Acquire a Redis lock for the artifact being processed."""
        acquired = self._redis_call(
            "redis.set(lock)",
            lambda: self.redis.client.set(
                self._lock_key(artifact_uuid),
                utcnow_iso(),
                nx=True,
                ex=self.processing_timeout,
            ),
        )
        return bool(acquired)

    def refresh_lock(self, artifact_uuid: str) -> None:
        """Refresh the lock TTL for a running extraction job."""
        self._redis_call(
            "redis.expire(lock)",
            lambda: self.redis.client.expire(
                self._lock_key(artifact_uuid), self.processing_timeout
            ),
        )

    def release_lock(self, artifact_uuid: str) -> None:
        """Release the Redis lock for an artifact."""
        self._redis_call(
            "redis.delete(lock)",
            lambda: self.redis.client.delete(self._lock_key(artifact_uuid)),
        )

    def release_lock_best_effort(self, artifact_uuid: str) -> None:
        """Release the Redis lock without surfacing failures."""
        try:
            client = getattr(self.redis, "client", None)
            if client is None:
                return
            client.delete(self._lock_key(artifact_uuid))
        except Exception:
            return

    def mark_running(self, job: dict[str, Any]) -> dict[str, Any]:
        """Mark a queued job as running."""
        current = self.get_job_state(job["artifact_uuid"]) or job
        current["status"] = "running"
        current["started_at"] = current.get("started_at") or utcnow_iso()
        current["completed_at"] = None
        current["error"] = None
        self._set_job_state(job["artifact_uuid"], current)
        return current

    def update_progress(
        self,
        artifact_uuid: str,
        current_page: int,
        total_pages: int,
    ) -> None:
        """Update in-flight job progress and refresh the Redis lock."""
        job = self.get_job_state(artifact_uuid)
        if not job:
            return
        job["status"] = "running"
        job["current_page"] = current_page
        job["total_pages"] = total_pages
        job["started_at"] = job.get("started_at") or utcnow_iso()
        self._set_job_state(artifact_uuid, job)
        self.refresh_lock(artifact_uuid)

    def mark_succeeded(
        self,
        artifact_uuid: str,
        result: GuidelineExtractionResponse,
    ) -> dict[str, Any]:
        """Mark a job as succeeded after the result has been persisted."""
        job = self.get_job_state(artifact_uuid) or {}
        job["artifact_uuid"] = artifact_uuid
        job["status"] = "succeeded"
        job["model"] = result.model
        job["dpi"] = result.dpi
        job["current_page"] = result.total_pages
        job["total_pages"] = result.total_pages
        job["completed_at"] = result.extracted_at
        job["error"] = None
        self._set_job_state(artifact_uuid, job)
        return job

    def mark_failed(self, artifact_uuid: str, error: str) -> dict[str, Any]:
        """Mark a job as failed."""
        job = self.get_job_state(artifact_uuid) or {"artifact_uuid": artifact_uuid}
        job["status"] = "failed"
        job["completed_at"] = utcnow_iso()
        job["error"] = error
        self._set_job_state(artifact_uuid, job)
        return job

    def retry_or_fail(
        self, artifact_uuid: str, job: dict[str, Any], error: str
    ) -> dict[str, Any]:
        """
        Re-queue a failed extraction, or give up once attempts are exhausted.

        Extraction is long and touches a rate-limited service, so the common
        failure is transient — a 429 that outlasted the per-call backoff, or the
        artifact download flaking. Making an operator re-trigger those by hand
        is what left runs sitting failed for hours.

        The attempt counter is the bound: a PDF that genuinely cannot be parsed
        fails a fixed number of times and then stays failed, rather than
        cycling through the queue forever.
        """
        attempt = int(job.get("attempt", 1) or 1)
        max_attempts = int(job.get("max_attempts", MAX_EXTRACTION_ATTEMPTS) or 1)

        if attempt >= max_attempts:
            logger.error(
                "Guideline extraction for %s failed on attempt %s/%s; giving up: %s",
                artifact_uuid,
                attempt,
                max_attempts,
                error,
            )
            return self.mark_failed(artifact_uuid, error)

        retry_state = {
            **job,
            "status": "queued",
            "attempt": attempt + 1,
            "started_at": None,
            "completed_at": None,
            "current_page": None,
            "error": f"attempt {attempt} failed: {error}",
            "enqueued_at": utcnow_iso(),
        }
        self._set_job_state(artifact_uuid, retry_state)

        payload = json.dumps(retry_state)
        try:
            self._redis_call(
                "redis.rpush(queue)",
                lambda: self.redis.client.rpush(self.queue_key, payload),
            )
        except GuidelineJobQueueUnavailable:
            # If the queue is unreachable the retry cannot be scheduled; record
            # the failure rather than leaving a job claiming to be queued.
            return self.mark_failed(artifact_uuid, error)

        logger.warning(
            "Guideline extraction for %s failed on attempt %s/%s; re-queued: %s",
            artifact_uuid,
            attempt,
            max_attempts,
            error,
        )
        return retry_state

    def build_result(
        self,
        artifact_uuid: str,
        output,
        extracted_at: str,
    ) -> GuidelineExtractionResponse:
        """Build the persisted response payload from an extraction output bundle."""
        storage = self.get_storage(artifact_uuid)
        return GuidelineExtractionResponse(
            artifact_uuid=artifact_uuid,
            workspace_root=storage.workspace_root,
            artifact_dir=storage.artifact_dir,
            pdf_path=output.source_pdf,
            model=output.model,
            dpi=output.dpi,
            extracted_at=extracted_at,
            total_pages=output.total_pages,
            total_processed_pages=len(output.processed_pages),
            total_skipped_pages=len(output.skipped_pages),
            total_guidelines=len(output.guidelines),
            total_unique_guidelines=len(output.unique_guidelines),
            processed_pages=output.processed_pages,
            skipped_pages=output.skipped_pages,
            guidelines=output.guidelines,
            unique_guidelines=output.unique_guidelines,
            schema_version=getattr(output, "schema_version", EXTRACTION_SCHEMA_VERSION),
            guide_context=getattr(output, "guide_context", None),
            document_profile=getattr(output, "document_profile", None),
            continuation_pages=list(getattr(output, "continuation_pages", []) or []),
        )

    @staticmethod
    def _guideline_attr(item: Any, name: str, default: Any = None) -> Any:
        """Read an attribute from either a model/object or a plain dict."""
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _resolve_target_populations(
        hint: str | None, life_stage: list[str]
    ) -> list[str]:
        """Map a free-text population hint and life stages onto the catalog enum."""
        populations: list[str] = []

        if isinstance(hint, str) and hint.strip():
            lowered = hint.lower()
            for pattern, population in TARGET_POPULATION_PATTERNS:
                if re.search(pattern, lowered) and population not in populations:
                    populations.append(population)

        for stage in life_stage or []:
            population = LIFE_STAGE_TO_TARGET_POPULATION.get(stage)
            if population and population not in populations:
                populations.append(population)

        return populations

    @staticmethod
    def _page_summaries(result: GuidelineExtractionResponse) -> dict[int, str]:
        """
        Map page number to the summary the extractor wrote for it.

        The summary is the context a rule sentence loses the moment it is
        lifted off its page. It was previously computed, used to prime the next
        page, and then discarded at import; keeping it means a reviewer and the
        enrichment agent can both see what surrounded the rule.
        """
        summaries: dict[int, str] = {}
        for page in result.processed_pages or []:
            page_no = GuidelineJobService._guideline_attr(page, "page")
            summary = GuidelineJobService._guideline_attr(page, "page_summary")
            if page_no is None or not isinstance(summary, str) or not summary.strip():
                continue
            try:
                summaries[int(page_no)] = summary.strip()
            except (TypeError, ValueError):
                continue
        return summaries

    @staticmethod
    def _candidate_facets(
        row: Any,
        page_no: int | None,
        artifact_uuid: str,
        page_summaries: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """
        Build the catalog facet payload for one extracted rule.

        Only fields the extraction actually supported are included; a v1 row
        carries none of them and yields an empty payload, which keeps old
        results importable unchanged.
        """
        read = GuidelineJobService._guideline_attr
        facets: dict[str, Any] = {}

        life_stage = [
            value
            for value in (read(row, "life_stage") or [])
            if value in LIFE_STAGE_VALUES
        ]
        setting = [
            value for value in (read(row, "setting") or []) if value in SETTING_VALUES
        ]
        guideline_type = read(row, "guideline_type")
        health_conditions = list(read(row, "health_conditions") or [])
        nutrients = list(read(row, "nutrients") or [])
        topic = list(read(row, "topic") or [])
        age_min = read(row, "age_min_months")
        age_max = read(row, "age_max_months")

        if life_stage:
            facets["life_stage"] = life_stage
        if setting:
            facets["setting"] = setting
        if guideline_type in GUIDELINE_TYPE_VALUES:
            facets["guideline_type"] = guideline_type
        if health_conditions:
            facets["health_conditions"] = health_conditions
        if nutrients:
            facets["nutrients"] = nutrients
        if topic:
            facets["topic"] = topic
        if isinstance(age_min, int) and age_min >= 0:
            facets["age_min_months"] = age_min
        if isinstance(age_max, int) and age_max >= 0:
            facets["age_max_months"] = age_max

        target_populations = GuidelineJobService._resolve_target_populations(
            read(row, "target_population_hint"), life_stage
        )
        if target_populations:
            facets["target_populations"] = target_populations

        section_label = read(row, "section_label")
        if isinstance(section_label, str) and section_label.strip():
            facets["section_label"] = section_label.strip()

        if page_no is not None:
            source_ref: dict[str, Any] = {
                "artifact_id": artifact_uuid,
                "page_start": page_no,
                "page_end": page_no,
            }
            if isinstance(section_label, str) and section_label.strip():
                source_ref["section_label"] = section_label.strip()
            facets["source_refs"] = [source_ref]

            summary = (page_summaries or {}).get(page_no)
            if summary:
                facets["page_summary"] = summary

        # The verbatim span the rule came from is the strongest provenance we
        # have; it belongs on the record, not only in the extraction result.
        snippet = read(row, "source_snippet")
        if isinstance(snippet, str) and snippet.strip():
            facets["notes"] = f"Source excerpt: {snippet.strip()}"

        if facets:
            # Provenance and captured context are records of what the source
            # said, not machine-inferred values, so they are not claimed as
            # AI-generated and stay outside the enrichment no-clobber guard.
            provenance = (
                "source_refs",
                "notes",
                "page_summary",
                "section_label",
                "ai_generated_fields",
            )
            facets["ai_generated_fields"] = sorted(
                key for key in facets if key not in provenance
            )

        return facets

    @staticmethod
    def _collect_unique_import_candidates(
        result: GuidelineExtractionResponse,
        action_type: GuidelineActionType,
        *,
        artifact_uuid: str,
        import_facets: bool = True,
    ) -> list[dict[str, Any]]:
        """Collapse extracted guideline rows into unique import candidates."""
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        page_summaries = GuidelineJobService._page_summaries(result)

        for row in result.guidelines:
            rule_text = GuidelineJobService._guideline_attr(row, "text")
            page_no = GuidelineJobService._guideline_attr(row, "page")
            if not isinstance(rule_text, str):
                continue
            compare_key = guideline_compare_key(rule_text)
            if not compare_key or compare_key in seen:
                continue
            seen.add(compare_key)

            page_no = int(page_no) if page_no is not None else None

            # A per-rule hint beats the batch-wide constant: before this, all
            # ~2700 rules shared one action_type, which made the field useless.
            hinted_action = GuidelineJobService._guideline_attr(row, "action_type_hint")
            resolved_action = action_type
            if isinstance(hinted_action, str) and hinted_action.strip():
                try:
                    resolved_action = normalize_guideline_action_type(hinted_action)
                except (TypeError, ValueError):
                    resolved_action = action_type

            facets = (
                GuidelineJobService._candidate_facets(
                    row, page_no, artifact_uuid, page_summaries
                )
                if import_facets
                else {}
            )

            candidates.append(
                {
                    "rule_text": normalize_guideline(rule_text),
                    "page_no": page_no,
                    "action_type": resolved_action,
                    "compare_key": compare_key,
                    "facets": facets,
                }
            )

        return candidates

    def _load_existing_guide_guidelines(
        self,
        guide: Any,
        limit: int | None,
    ) -> tuple[list[Any], bool]:
        """
        Read the guide's existing guidelines for dedupe and sequence numbering.

        Uses the SDK's paged search rather than slicing the collection proxy:
        slicing returns lazy entities that fetch themselves individually on
        first attribute access, so reading `sequence_no` off 500 existing rules
        cost 500 HTTP round trips.

        Returns the items plus whether the scan was complete. A truncated scan
        silently misses duplicates and can reuse sequence numbers, so the caller
        surfaces the flag rather than swallowing it.
        """
        # Only what dedupe and numbering actually need.
        fields = ["id", "sequence_no", "rule_text"]

        try:
            items = guide.guidelines.fetch_all(fl=fields)
        except AttributeError:
            # Older SDK without fetch_all: fall back to the proxy slice. Slower,
            # but correctness does not depend on the fast path existing.
            items = list(guide.guidelines[0 : limit or 1000])

        if limit is not None and len(items) > limit:
            return items[:limit], False
        return items, True

    async def import_latest_result_to_guide(
        self,
        artifact_uuid: str,
        guide_id: str,
        *,
        dry_run: bool = True,
        dedupe_against_guide: bool = True,
        action_type: str = DEFAULT_GUIDELINE_ACTION_TYPE,
        existing_scan_limit: int | None = None,
        import_facets: bool = True,
    ) -> GuidelineImportResponse:
        """Import the latest persisted extraction result into a WiseFood guide."""
        try:
            normalized_action_type = normalize_guideline_action_type(action_type)
        except (TypeError, ValueError) as exc:
            raise GuidelineExtractionError(str(exc)) from exc

        self.get_storage(artifact_uuid)
        result = await self.result_store.fetch_result(artifact_uuid)
        if result is None:
            job = None
            try:
                job = self.get_job_state(artifact_uuid)
            except GuidelineJobQueueUnavailable:
                job = None

            if job and job.get("status") in {"queued", "running"}:
                raise GuidelineImportPreconditionError(
                    f"Guideline extraction for {artifact_uuid} has not completed yet."
                )
            raise GuidelineImportNotFoundError(
                f"No completed guideline extraction result found for artifact {artifact_uuid}."
            )

        schema_version = getattr(result, "schema_version", 1) or 1
        candidates = self._collect_unique_import_candidates(
            result,
            normalized_action_type,
            artifact_uuid=artifact_uuid,
            import_facets=import_facets and schema_version >= 2,
        )
        client = self.platform_pool.get_client()
        created_ids: list[str] = []

        try:
            guide = client.guides.get(guide_id)
            existing_guidelines, existing_scan_complete = (
                self._load_existing_guide_guidelines(guide, existing_scan_limit)
            )
            if not existing_scan_complete:
                logger.warning(
                    "Existing-guideline scan for guide %s was truncated at %s items; "
                    "dedupe and sequence numbering are based on a partial view.",
                    guide_id,
                    existing_scan_limit,
                )

            existing_sequence_nos: list[int] = []
            existing_compare_keys: set[str] = set()
            for guideline in existing_guidelines:
                sequence_no = self._guideline_attr(guideline, "sequence_no")
                if sequence_no is not None:
                    try:
                        existing_sequence_nos.append(int(sequence_no))
                    except Exception:
                        pass

                rule_text = self._guideline_attr(guideline, "rule_text")
                if isinstance(rule_text, str):
                    compare_key = guideline_compare_key(rule_text)
                    if compare_key:
                        existing_compare_keys.add(compare_key)

            next_sequence_no = max(existing_sequence_nos, default=0) + 1
            next_sequence_no_start = next_sequence_no
            items: list[GuidelineImportItemResponse] = []
            pending_creates: list[dict[str, Any]] = []

            for candidate in candidates:
                if (
                    dedupe_against_guide
                    and candidate["compare_key"] in existing_compare_keys
                ):
                    items.append(
                        GuidelineImportItemResponse(
                            rule_text=candidate["rule_text"],
                            page_no=candidate["page_no"],
                            action_type=candidate["action_type"],
                            sequence_no=None,
                            status="skipped_existing",
                            reason="A matching guideline already exists on the target guide.",
                            created_id=None,
                            facets=candidate["facets"],
                        )
                    )
                    continue

                sequence_no = next_sequence_no
                next_sequence_no += 1
                if dry_run:
                    items.append(
                        GuidelineImportItemResponse(
                            rule_text=candidate["rule_text"],
                            page_no=candidate["page_no"],
                            action_type=candidate["action_type"],
                            sequence_no=sequence_no,
                            status="would_create",
                            reason=None,
                            created_id=None,
                            facets=candidate["facets"],
                        )
                    )
                    continue

                item = {
                    "sequence_no": sequence_no,
                    "rule_text": candidate["rule_text"],
                    "action_type": candidate["action_type"],
                    "extractor_name": "guideline_extractor",
                    "extractor_run_id": artifact_uuid,
                    "extraction_model": result.model,
                }
                if candidate["page_no"] is not None:
                    item["page_no"] = candidate["page_no"]
                item.update(candidate["facets"])

                # Collected, not created: the whole batch goes in one request
                # below. One POST per rule cost a round trip each, and every one
                # re-resolved the guide and its artifacts server-side.
                pending_creates.append(item)
                items.append(
                    GuidelineImportItemResponse(
                        rule_text=candidate["rule_text"],
                        page_no=candidate["page_no"],
                        action_type=candidate["action_type"],
                        sequence_no=sequence_no,
                        status="created",
                        reason=None,
                        created_id=None,
                        facets=candidate["facets"],
                    )
                )
                existing_compare_keys.add(candidate["compare_key"])

            # The catalog caps a batch at 1000; chunk so an unusually large
            # extraction still lands in one import.
            for start in range(0, len(pending_creates), IMPORT_BATCH_SIZE):
                batch = pending_creates[start : start + IMPORT_BATCH_SIZE]
                response = guide.guidelines.bulk_import(batch)
                imported = (
                    response.get("imported_count", len(batch))
                    if isinstance(response, dict)
                    else len(batch)
                )
                created_ids.extend([""] * imported)

            return GuidelineImportResponse(
                artifact_uuid=artifact_uuid,
                guide_id=guide_id,
                dry_run=dry_run,
                extracted_at=result.extracted_at,
                source_guideline_count=len(result.guidelines),
                total_candidates=len(candidates),
                existing_guidelines_scanned=len(existing_guidelines),
                total_created=len(created_ids),
                total_skipped=sum(1 for item in items if item.status == "skipped_existing"),
                next_sequence_no_start=next_sequence_no_start,
                schema_version=schema_version,
                existing_scan_complete=existing_scan_complete,
                items=items,
            )
        except GuidelineImportError:
            raise
        except Exception as exc:
            message = str(exc)
            raise GuidelineImportError(
                f"Failed to import extracted guidelines into guide {guide_id}: {message}"
            ) from exc
        finally:
            self.platform_pool.return_client(client)

    async def get_job_response(
        self,
        artifact_uuid: str,
    ) -> GuidelineExtractionJobResponse:
        """Return combined Redis job state plus the latest persisted result."""
        storage = self.get_storage(artifact_uuid)
        result = await self.result_store.fetch_result(artifact_uuid)

        redis_error: GuidelineJobQueueUnavailable | None = None
        job = None
        try:
            job = self.get_job_state(artifact_uuid)
        except GuidelineJobQueueUnavailable as exc:
            redis_error = exc

        if job is None:
            if result is not None:
                return GuidelineExtractionJobResponse(
                    artifact_uuid=artifact_uuid,
                    status="succeeded",
                    job_id=None,
                    model=result.model,
                    dpi=result.dpi,
                    enqueued_at=None,
                    started_at=None,
                    completed_at=result.extracted_at,
                    current_page=result.total_pages,
                    total_pages=result.total_pages,
                    error=None,
                    storage=storage,
                    result=result,
                )

            if redis_error is not None:
                raise redis_error

            return GuidelineExtractionJobResponse(
                artifact_uuid=artifact_uuid,
                status="not_found",
                storage=storage,
                result=None,
            )

        # A job whose worker died still says "running"; surface that instead of
        # leaving the console spinning on a job nothing is working on.
        status = job["status"]
        stalled = self.is_orphaned(job)
        if stalled:
            status = "stalled"

        return GuidelineExtractionJobResponse(
            artifact_uuid=artifact_uuid,
            status=status,
            stalled=stalled,
            job_id=job.get("job_id"),
            model=job.get("model"),
            dpi=job.get("dpi"),
            enqueued_at=job.get("enqueued_at"),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            current_page=job.get("current_page"),
            total_pages=job.get("total_pages"),
            error=job.get("error"),
            storage=storage,
            result=result,
        )
