"""Optional Langfuse observability for LLM inference.

Tracing is opt-in: it activates only when both ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are present in the environment AND the ``langfuse``
package is importable. When either condition is unmet, every helper degrades
to a no-op so the application behaves exactly as it would without Langfuse.

Env vars (read directly by the Langfuse SDK):
    LANGFUSE_PUBLIC_KEY  - project public key ("pk-lf-...")
    LANGFUSE_SECRET_KEY  - project secret key ("sk-lf-...")
    LANGFUSE_BASE_URL    - host, e.g. https://cloud.langfuse.com (optional)
"""
import os
import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """Return True only if Langfuse is configured and importable."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning(
            "Langfuse keys are set but the 'langfuse' package is not "
            "importable; LLM tracing is disabled (%s)",
            exc,
        )
        return False
    return True


@lru_cache(maxsize=1)
def get_callback_handler() -> Optional[Any]:
    """Return a cached LangChain CallbackHandler, or None if disabled.

    The handler is stateless and reads credentials from the environment via
    the singleton Langfuse client, so a single shared instance can safely be
    attached to every ChatGroq client in the connection pool.
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        logger.info("Langfuse tracing enabled for LangChain LLM calls")
        return CallbackHandler()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize Langfuse CallbackHandler: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush buffered traces to Langfuse. Safe to call when disabled."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
        logger.info("Flushed pending Langfuse traces")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to flush Langfuse traces: %s", exc)
