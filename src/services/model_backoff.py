"""
Retry with backoff for model calls.

The OpenAI SDK retries a couple of times on its own, which covers a blip but
not a sustained rate limit. Extraction is the case that matters: several
replicas each walking a PDF page by page can hold a 429 open for minutes, and
without backoff every page of every concurrent run fails in the same window and
the whole job dies.

Backoff is full-jitter exponential. Without jitter, workers that started
together retry together — they stay synchronised and keep colliding, which is
the failure this is meant to prevent.
"""

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 60.0

# Substrings that mark an error as worth retrying. Matched against the class
# name and message because the providers raise different exception types and we
# do not want a hard dependency on any one SDK's exception tree.
RETRYABLE_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
    "overloaded",
    "capacity",
)


def is_retryable(exc: BaseException) -> bool:
    """
    Whether an exception is worth another attempt.

    Deliberately conservative: an unrecognised error is treated as permanent.
    Retrying a malformed request or an auth failure just burns time and quota
    on something that cannot succeed.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False

    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in RETRYABLE_MARKERS)


def retry_delay(
    attempt: int,
    *,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_after: float | None = None,
) -> float:
    """
    Seconds to wait before attempt ``attempt`` (1-based).

    A server-supplied ``Retry-After`` wins: it is the only party that knows when
    the limit actually resets. Otherwise full-jitter exponential — the delay is
    drawn from ``[0, backoff]`` rather than being exactly ``backoff``, which is
    what breaks the synchronisation between workers that failed together.
    """
    if retry_after is not None and retry_after >= 0:
        return min(float(retry_after), max_delay)

    backoff = min(base_delay * (2 ** max(attempt - 1, 0)), max_delay)
    return random.uniform(0, backoff)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Read a Retry-After hint off the exception, if the SDK exposed one."""
    for attribute in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attribute, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for header in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
            try:
                raw = headers.get(header)
            except Exception:
                raw = None
            if raw is None:
                continue
            try:
                return float(str(raw).rstrip("s"))
            except (TypeError, ValueError):
                continue
    return None


def call_with_backoff(
    operation: Callable[[], T],
    *,
    description: str = "model call",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """
    Run ``operation``, retrying transient failures with jittered backoff.

    Re-raises the last exception once attempts are exhausted, or immediately for
    anything not classified as retryable.
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - classified below
            last_exc = exc
            if not is_retryable(exc) or attempt == max_attempts:
                raise

            delay = retry_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_after=_retry_after_seconds(exc),
            )
            logger.warning(
                "%s failed (attempt %s/%s): %s — retrying in %.1fs",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_exc  # type: ignore[misc]
