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


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Any]:
    """Process-wide Langfuse client (shared connection + prompt cache).

    The Langfuse SDK is a singleton; per-request instantiation is discouraged
    by the docs to avoid memory leaks. This returns one configure-once client,
    reused for prompt fetching (and trace flushing) across all requests and
    threads. Returns None when observability is disabled.
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()  # reads keys + base_url from the environment
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush buffered traces to Langfuse. Safe to call when disabled."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
        logger.info("Flushed pending Langfuse traces")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to flush Langfuse traces: %s", exc)
