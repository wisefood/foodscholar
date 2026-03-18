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
    GuidelineArtifactStorageResponse,
    GuidelineExtractionJobResponse,
    GuidelineExtractionResponse,
    GuidelineImportItemResponse,
    GuidelineImportResponse,
)
from services.guideline_extractor import (
    GuidelineExtractorService,
    get_default_dpi,
    get_default_model,
    normalize_guideline,
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

    def enqueue_job(
        self,
        artifact_uuid: str,
        model: str | None = None,
        dpi: int | None = None,
    ) -> dict[str, Any]:
        """Queue a new extraction job unless one is already queued or running."""
        self.get_storage(artifact_uuid)
        current = self.get_job_state(artifact_uuid)
        if current and current.get("status") in {"queued", "running"}:
            return current

        job_state = {
            "artifact_uuid": artifact_uuid,
            "job_id": str(uuid.uuid4()),
            "status": "queued",
            "model": model or get_default_model(),
            "dpi": dpi or get_default_dpi(),
            "enqueued_at": utcnow_iso(),
            "started_at": None,
            "completed_at": None,
            "current_page": None,
            "total_pages": None,
            "error": None,
        }
        self._set_job_state(artifact_uuid, job_state)
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
        )

    @staticmethod
    def _guideline_attr(item: Any, name: str, default: Any = None) -> Any:
        """Read an attribute from either a model/object or a plain dict."""
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _collect_unique_import_candidates(
        result: GuidelineExtractionResponse,
        action_type: str,
    ) -> list[dict[str, Any]]:
        """Collapse extracted guideline rows into unique import candidates."""
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in result.guidelines:
            rule_text = GuidelineJobService._guideline_attr(row, "text")
            page_no = GuidelineJobService._guideline_attr(row, "page")
            if not isinstance(rule_text, str):
                continue
            compare_key = guideline_compare_key(rule_text)
            if not compare_key or compare_key in seen:
                continue
            seen.add(compare_key)
            candidates.append(
                {
                    "rule_text": normalize_guideline(rule_text),
                    "page_no": int(page_no) if page_no is not None else None,
                    "action_type": action_type,
                    "compare_key": compare_key,
                }
            )

        return candidates

    def _load_existing_guide_guidelines(
        self,
        guide: Any,
        limit: int,
    ) -> list[Any]:
        """Read existing guide guidelines up to the configured scan limit."""
        items: list[Any] = []
        start = 0
        page_size = min(100, max(1, limit))

        while start < limit:
            end = min(start + page_size, limit)
            batch = guide.guidelines[start:end]
            if not batch:
                break
            items.extend(batch)
            if len(batch) < (end - start):
                break
            start = end

        return items

    async def import_latest_result_to_guide(
        self,
        artifact_uuid: str,
        guide_id: str,
        *,
        dry_run: bool = True,
        dedupe_against_guide: bool = True,
        action_type: str = "encourage",
        existing_scan_limit: int = 500,
    ) -> GuidelineImportResponse:
        """Import the latest persisted extraction result into a WiseFood guide."""
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

        candidates = self._collect_unique_import_candidates(result, action_type)
        client = self.platform_pool.get_client()
        created_ids: list[str] = []

        try:
            guide = client.guides.get(guide_id)
            existing_guidelines = self._load_existing_guide_guidelines(
                guide, existing_scan_limit
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
                        )
                    )
                    continue

                create_kwargs = {
                    "sequence_no": sequence_no,
                    "rule_text": candidate["rule_text"],
                    "action_type": candidate["action_type"],
                }
                if candidate["page_no"] is not None:
                    create_kwargs["page_no"] = candidate["page_no"]

                created = guide.guidelines.create(**create_kwargs)
                created_id = self._guideline_attr(created, "id")
                if created_id is not None:
                    created_ids.append(str(created_id))
                items.append(
                    GuidelineImportItemResponse(
                        rule_text=candidate["rule_text"],
                        page_no=candidate["page_no"],
                        action_type=candidate["action_type"],
                        sequence_no=sequence_no,
                        status="created",
                        reason=None,
                        created_id=str(created_id) if created_id is not None else None,
                    )
                )
                existing_compare_keys.add(candidate["compare_key"])

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

        return GuidelineExtractionJobResponse(
            artifact_uuid=artifact_uuid,
            status=job["status"],
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
