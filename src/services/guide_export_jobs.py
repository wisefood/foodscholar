"""Redis-backed jobs that export one guide's guidelines to an Excel workbook."""

from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import config
from models.guidelines import GuideExportJobResponse
from services.guide_export_workbook import build_guide_workbook, export_filename

logger = logging.getLogger(__name__)

# Statuses that mean a job is still to finish. "stalled" is derived, never stored.
IN_FLIGHT = {"queued", "running"}


def utcnow_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


class GuideExportQueueUnavailable(RuntimeError):
    """Raised when Redis-backed export job operations are unavailable."""


class GuideExportError(ValueError):
    """Raised for a request that cannot name a guide."""


class GuideExportNotReady(RuntimeError):
    """Raised when the guide's latest export has not finished yet."""


class GuideExportNotFound(LookupError):
    """Raised when there is no finished export to download, or it has expired."""


def normalize_guide_urn(identifier: str) -> str:
    """A full ``urn:guide:<slug>`` for a URN or a bare slug."""
    from wisefood.entities.guides import Guide

    value = (identifier or "").strip()
    if not value or any(ch.isspace() for ch in value) or "/" in value:
        raise GuideExportError(f"Not a guide identifier: {identifier!r}")
    return Guide.build_identifier(value)


class GuideExportJobService:
    """
    One export job per guide, its status, and the finished file, all in Redis.

    The finished workbook lives next to its status under the same TTL rather
    than as a guide artifact: an export is a download, not a source document,
    and an artifact would sit in the guide's artifact list beside the PDFs that
    extraction reads from. The Redis client decodes responses to text, so the
    bytes are stored base64-encoded.
    """

    def __init__(self, redis_client: Any | None = None, platform_pool: Any | None = None):
        self._redis = redis_client
        self._platform_pool = platform_pool
        settings = config.settings
        self.queue_key = str(settings.get("GUIDE_EXPORT_QUEUE_KEY", "guide_export:queue"))
        self.job_prefix = str(settings.get("GUIDE_EXPORT_STATUS_PREFIX", "guide_export:job"))
        self.lock_prefix = str(settings.get("GUIDE_EXPORT_LOCK_PREFIX", "guide_export:lock"))
        self.file_prefix = str(settings.get("GUIDE_EXPORT_FILE_PREFIX", "guide_export:file"))
        self.lock_timeout = int(settings.get("GUIDE_EXPORT_LOCK_TIMEOUT", 900))
        self.ttl_seconds = int(settings.get("GUIDE_EXPORT_TTL_SECONDS", 86400))

    @property
    def redis(self):
        """Get or lazily initialize the Redis singleton."""
        if self._redis is None:
            from backend.redis import RedisClientSingleton

            self._redis = RedisClientSingleton()
        return self._redis

    @property
    def platform_pool(self):
        """Get or lazily initialize the WiseFood client pool (the service account)."""
        if self._platform_pool is None:
            from backend.platform import WISEFOOD

            self._platform_pool = WISEFOOD
        return self._platform_pool

    def redis_available(self) -> bool:
        """Best-effort check for Redis availability."""
        try:
            self.redis.client.ping()
            return True
        except Exception:
            return False

    def _job_key(self, guide_urn: str) -> str:
        return f"{self.job_prefix}:{guide_urn}"

    def _lock_key(self, guide_urn: str) -> str:
        return f"{self.lock_prefix}:{guide_urn}"

    def _file_key(self, job_id: str) -> str:
        return f"{self.file_prefix}:{job_id}"

    def _redis_call(self, op: str, fn):
        try:
            return fn()
        except Exception as exc:
            raise GuideExportQueueUnavailable(op) from exc

    # ------------------------------------------------------------- status --

    def get_job_state(self, guide_urn: str) -> Optional[dict[str, Any]]:
        raw = self._redis_call(
            "redis.get(job)", lambda: self.redis.client.get(self._job_key(guide_urn))
        )
        if not raw:
            return None
        return raw if isinstance(raw, dict) else json.loads(raw)

    def _set_job_state(self, guide_urn: str, state: dict[str, Any], *, nx: bool = False) -> bool:
        payload = json.dumps(state)
        return bool(
            self._redis_call(
                "redis.set(job)",
                lambda: self.redis.client.set(
                    self._job_key(guide_urn), payload, nx=nx, ex=self.ttl_seconds
                ),
            )
        )

    def has_lock(self, guide_urn: str) -> bool:
        try:
            return bool(self.redis.client.exists(self._lock_key(guide_urn)))
        except Exception:
            # Unknown is treated as held, so an unreachable Redis never makes a
            # live job look abandoned.
            return True

    def is_orphaned(self, state: dict[str, Any] | None) -> bool:
        """A job that says ``running`` while no worker holds its lock: the worker died."""
        return bool(state and state.get("status") == "running" and not self.has_lock(state["guide_urn"]))

    def file_exists(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        return bool(
            self._redis_call("redis.exists(file)", lambda: self.redis.client.exists(self._file_key(job_id)))
        )

    # ------------------------------------------------------------ enqueue --

    def enqueue_job(self, guide_urn: str, *, force: bool = False) -> dict[str, Any]:
        """
        Queue an export unless one for this guide is already in flight.

        Repeated clicks return the job already queued or running. A finished
        export is never reused implicitly: asking again exports the guide as it
        is now, since its rules may have been edited in between.
        """
        guide_urn = normalize_guide_urn(guide_urn)
        current = self.get_job_state(guide_urn)
        if current and not force and current.get("status") in IN_FLIGHT:
            if not self.is_orphaned(current):
                return current
            logger.warning(
                "Guide export %s for %s was orphaned; re-queueing.", current.get("job_id"), guide_urn
            )

        state = {
            "guide_urn": guide_urn,
            "job_id": str(uuid.uuid4()),
            "status": "queued",
            "stage": "queued",
            "enqueued_at": utcnow_iso(),
            "started_at": None,
            "completed_at": None,
            "total_guidelines": None,
            "filename": None,
            "size_bytes": None,
            "expires_at": None,
            "error": None,
        }
        # A single conditional write when nothing is there, so two clicks
        # arriving together cannot both enqueue; replacing a finished, failed
        # or orphaned job is deliberate and overwrites.
        if not self._set_job_state(guide_urn, state, nx=current is None):
            existing = self.get_job_state(guide_urn)
            if existing:
                return existing

        self._redis_call(
            "redis.rpush(queue)", lambda: self.redis.client.rpush(self.queue_key, json.dumps(state))
        )
        return state

    def pop_next_job(self, timeout: int) -> Optional[dict[str, Any]]:
        item = self._redis_call(
            "redis.blpop(queue)", lambda: self.redis.client.blpop(self.queue_key, timeout=timeout)
        )
        if not item:
            return None
        _, payload = item
        return json.loads(payload)

    def is_current_job(self, job: dict[str, Any]) -> bool:
        current = self.get_job_state(job["guide_urn"])
        return bool(current and current.get("job_id") == job.get("job_id"))

    def try_claim_job(self, guide_urn: str) -> bool:
        return bool(
            self._redis_call(
                "redis.set(lock)",
                lambda: self.redis.client.set(
                    self._lock_key(guide_urn), utcnow_iso(), nx=True, ex=self.lock_timeout
                ),
            )
        )

    def release_lock_best_effort(self, guide_urn: str) -> None:
        try:
            self.redis.client.delete(self._lock_key(guide_urn))
        except Exception:
            return

    # ------------------------------------------------------------ running --

    def _update(self, guide_urn: str, **changes: Any) -> dict[str, Any]:
        state = self.get_job_state(guide_urn) or {"guide_urn": guide_urn}
        state.update(changes)
        self._set_job_state(guide_urn, state)
        return state

    def fetch_guide(self, guide_urn: str) -> tuple[dict, list[dict]]:
        """The guide record and every one of its rules, as the service account sees them."""
        client = self.platform_pool.get_client()
        try:
            guide = client.guides.get(guide_urn)
            rules = guide.guidelines.fetch_all()
            return dict(guide.data), list(rules)
        finally:
            self.platform_pool.return_client(client)

    def run_export(self, job: dict[str, Any]) -> dict[str, Any]:
        """Fetch, build and store one export, recording each stage on the job."""
        guide_urn = job["guide_urn"]
        self._update(guide_urn, status="running", stage="fetching", started_at=utcnow_iso(), error=None)

        guide, rules = self.fetch_guide(guide_urn)
        self._update(guide_urn, stage="writing", total_guidelines=len(rules))

        content = build_guide_workbook(guide, rules, generated_at=utcnow_iso())
        filename = export_filename(guide)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)).isoformat()
        # The file first, then the status that points at it: a status saying
        # "succeeded" is never readable before its file is.
        file_record = json.dumps(
            {"filename": filename, "content": base64.b64encode(content).decode("ascii")}
        )
        self._redis_call(
            "redis.set(file)",
            lambda: self.redis.client.set(self._file_key(job["job_id"]), file_record, ex=self.ttl_seconds),
        )
        return self._update(
            guide_urn,
            status="succeeded",
            stage="done",
            completed_at=utcnow_iso(),
            filename=filename,
            size_bytes=len(content),
            expires_at=expires_at,
        )

    def mark_failed(self, guide_urn: str, error: str) -> dict[str, Any]:
        return self._update(guide_urn, status="failed", completed_at=utcnow_iso(), error=error)

    # ------------------------------------------------------------ reading --

    def get_job_response(self, guide_urn: str) -> GuideExportJobResponse:
        guide_urn = normalize_guide_urn(guide_urn)
        state = self.get_job_state(guide_urn)
        if state is None:
            return GuideExportJobResponse(guide_urn=guide_urn, status="not_found")

        stalled = self.is_orphaned(state)
        status = "stalled" if stalled else state.get("status", "not_found")
        return GuideExportJobResponse(
            guide_urn=guide_urn,
            status=status,
            stalled=stalled,
            job_id=state.get("job_id"),
            stage=state.get("stage"),
            enqueued_at=state.get("enqueued_at"),
            started_at=state.get("started_at"),
            completed_at=state.get("completed_at"),
            total_guidelines=state.get("total_guidelines"),
            filename=state.get("filename"),
            size_bytes=state.get("size_bytes"),
            expires_at=state.get("expires_at"),
            download_ready=status == "succeeded" and self.file_exists(state.get("job_id")),
            error=state.get("error"),
        )

    def get_file(self, guide_urn: str) -> tuple[str, bytes]:
        """The latest finished export of a guide, as ``(filename, content)``."""
        guide_urn = normalize_guide_urn(guide_urn)
        state = self.get_job_state(guide_urn)
        if state and state.get("status") in IN_FLIGHT and not self.is_orphaned(state):
            raise GuideExportNotReady(f"The export of {guide_urn} has not finished yet.")
        if not state or state.get("status") != "succeeded":
            raise GuideExportNotFound(f"There is no finished export of {guide_urn}.")

        raw = self._redis_call(
            "redis.get(file)", lambda: self.redis.client.get(self._file_key(state["job_id"]))
        )
        if not raw:
            raise GuideExportNotFound(f"The export of {guide_urn} has expired; run it again.")
        record = json.loads(raw)
        return record["filename"], base64.b64decode(record["content"])
