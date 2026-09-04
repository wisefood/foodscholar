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
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


def build_trace_config(
    *,
    run_name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a LangChain ``config`` dict carrying Langfuse trace attributes.

    Returns a ``config`` suitable for passing to ``Runnable.invoke(..., config=...)``
    so the Langfuse ``CallbackHandler`` (already attached to every pooled ChatGroq
    client in :mod:`backend.groq`) enriches the resulting trace. It maps to the
    Langfuse **v3** convention:

        - ``run_name``           -> a descriptive, filterable trace/observation name
        - ``langfuse_session_id``-> groups multi-turn conversations (Sessions view)
        - ``langfuse_user_id``   -> user/cost attribution and filtering
        - ``langfuse_tags``      -> per-feature analytics (e.g. ["qa", "rag"])

    PII policy: only opaque identifiers (``session_id``/``user_id``) and feature
    tags are attached here. Never pass personal data (allergies, dietary profile,
    member details) via ``extra_metadata`` — the LLM message payload remains the
    only place model-relevant context appears, which is the generation input by
    necessity.

    Works regardless of whether Langfuse is enabled: the returned ``run_name`` is
    a standard LangChain config key, and the ``langfuse_*`` metadata keys are
    simply ignored when no handler is present. ``None`` values are omitted, and
    identifiers are coerced to ``str`` (Langfuse requires string metadata values).
    """
    config: Dict[str, Any] = {"run_name": run_name}

    metadata: Dict[str, Any] = {}
    if session_id is not None:
        metadata["langfuse_session_id"] = str(session_id)
    if user_id is not None:
        metadata["langfuse_user_id"] = str(user_id)
    if tags:
        metadata["langfuse_tags"] = list(tags)
    if extra_metadata:
        metadata.update(extra_metadata)

    if metadata:
        config["metadata"] = metadata

    return config


def tracing_allowed() -> bool:
    """Whether the platform still wants traces produced.

    Separate from having keys configured. Before this, stopping tracing meant
    unsetting the Langfuse keys and rolling every pod — impossible mid-incident
    and impossible when a study participant withdraws. An admin can now switch
    it off from the console and every service stops within the flag refresh
    interval.

    Permissive when the switch cannot be read: a control plane that fails
    closed would take tracing down whenever the gateway hiccups.
    """
    try:
        import wf_telemetry

        return wf_telemetry.TELEMETRY.tracing_enabled("langfuse")
    except Exception:
        return True


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


class _SwitchedHandler(BaseCallbackHandler):
    """A LangChain callback handler that consults the tracing switch per call.

    The real Langfuse handler is attached ONCE, when a pooled model client is
    constructed, and the client then lives for the life of the process. Handing
    out `None` when the switch is off therefore only affected clients built
    after the flip — every already-warm pool kept tracing, and a service that
    booted with tracing off could never be switched back on. The console switch
    appeared to work and did nothing.

    This wrapper is what gets attached instead. It is always present; each
    callback checks the switch and forwards only while tracing is allowed, so
    the flip takes effect on the next model call whatever the pool's age.

    It **must** subclass `BaseCallbackHandler`. `ChatGroq` is a Pydantic model
    and validates `callbacks` with `is_instance_of`, so a duck-typed wrapper is
    rejected at construction and the service does not start.

    Subclassing brings its own trap, which is why the callbacks below are
    generated explicitly rather than left to `__getattr__`: the base class
    defines every `on_*` as a no-op, so ordinary attribute lookup finds those
    and `__getattr__` is never consulted. A wrapper written that way imports
    cleanly, starts cleanly, and silently traces nothing.
    """

    #: Copied rather than delegated: LangChain reads these off the instance,
    #: and both are plain class attributes on the base, so assignment works.
    raise_error: bool = False
    run_inline: bool = False

    def __init__(self, inner: Any):
        super().__init__()
        self._inner = inner
        for flag in ("raise_error", "run_inline"):
            value = getattr(inner, flag, None)
            if isinstance(value, bool):
                setattr(self, flag, value)

    # `ignore_*` are read-only properties on the base, so they cannot be copied
    # onto the instance the way the flags above are; they are delegated instead.
    @property
    def ignore_llm(self) -> bool:
        return bool(getattr(self._inner, "ignore_llm", False))

    @property
    def ignore_chain(self) -> bool:
        return bool(getattr(self._inner, "ignore_chain", False))

    @property
    def ignore_agent(self) -> bool:
        return bool(getattr(self._inner, "ignore_agent", False))

    @property
    def ignore_retriever(self) -> bool:
        return bool(getattr(self._inner, "ignore_retriever", False))

    @property
    def ignore_chat_model(self) -> bool:
        return bool(getattr(self._inner, "ignore_chat_model", False))

    @property
    def ignore_retry(self) -> bool:
        return bool(getattr(self._inner, "ignore_retry", False))

    @property
    def ignore_custom_event(self) -> bool:
        return bool(getattr(self._inner, "ignore_custom_event", False))


def _forward_callback(name: str):
    """One delegating callback: check the switch, then hand off to Langfuse."""

    def forward(self, *args, **kwargs):
        if not tracing_allowed():
            return None
        target = getattr(self._inner, name, None)
        if target is None:
            return None
        return target(*args, **kwargs)

    forward.__name__ = name
    forward.__qualname__ = f"_SwitchedHandler.{name}"
    return forward


# Every callback the base declares, overridden to delegate. Generated from the
# base class rather than listed, so a callback added by a future LangChain
# release is forwarded too instead of silently becoming a no-op.
for _callback in (name for name in dir(BaseCallbackHandler) if name.startswith("on_")):
    setattr(_SwitchedHandler, _callback, _forward_callback(_callback))
del _callback


def get_callback_handler() -> Optional[Any]:
    """The shared LangChain callback handler, switchable at runtime.

    Returns a handler that forwards to Langfuse only while the platform switch
    allows it. Checked per callback, not per construction — see
    `_SwitchedHandler` for why the obvious version did not work.
    """
    inner = _build_callback_handler()
    if inner is None:
        return None
    return _switched(inner)


@lru_cache(maxsize=1)
def _switched(inner: Any) -> Any:
    return _SwitchedHandler(inner)


@lru_cache(maxsize=1)
def _build_callback_handler() -> Optional[Any]:
    """Construct the handler once.

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


# The cache moved to the builder when the handler gained a runtime switch, but
# `get_callback_handler.cache_clear()` is the documented way to reset it and is
# used by the tests. Kept working rather than renamed at every call site.
get_callback_handler.cache_clear = _build_callback_handler.cache_clear


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Any]:
    """Process-wide Langfuse client (shared connection + prompt cache).

    The Langfuse SDK is a singleton; per-request instantiation is discouraged
    by the docs to avoid memory leaks. This returns one configure-once client,
    reused for prompt fetching (and trace flushing) across all requests and
    threads. Returns None when observability is disabled.

    Deliberately NOT gated on the tracing switch: this client is also how the
    prompt registry is read, and turning tracing off must not silently drop
    every prompt back to its in-code fallback. Only the callback handler — the
    thing that actually emits traces — is switched.
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
