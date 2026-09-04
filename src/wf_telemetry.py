"""Report activity back to the gateway.

VENDORED FILE — keep the copies in foodchat, foodscholar, RecipeWrangler and
wisefood-data-api identical. Dependency-free (stdlib only, ``urllib`` rather
than httpx/requests) so it can be copied rather than packaged, the same
delivery model as ``obs_context.py`` and the Langfuse integration.

**Why services report at all.** The gateway records what it can see: which route
was called, by whom, how long it took. It cannot see what only the service
knows — that a recipe search matched nothing until the constraints were relaxed,
that a turn cost 1,400 tokens, that an answer came from cache. Those facts are
reported here, tagged with the correlation id the gateway assigned, so they join
the request that caused them.

**Why over HTTP to the gateway rather than straight to the database.** One
writer owns the analytics schema. A service with database credentials for it
would be a second writer, a second migration surface, and a second place for
consent to be applied — and consent is the thing that must not have two
implementations.

Contract, matching ``wisefood-api/src/routers/analytics.py``:

    POST {ANALYTICS_INGEST_URL}
    X-WiseFood-Analytics-Signature: <issued_at>.<hmac-sha256(f"{issued_at}|" + body)>
    {"events": [{"kind": "search"|"llm_usage"|"event", "type": ..., "app": ...}]}

Enabled only when ``ANALYTICS_ENABLED`` is true *and* ``ANALYTICS_INGEST_SECRET``
is set. Unset means off, never open.

Wiring, two lines in the service's entry point::

    import wf_telemetry
    wf_telemetry.TELEMETRY.start(app="recipewrangler")   # atexit-flushed

Then, from anywhere::

    wf_telemetry.TELEMETRY.search(surface="recipes", raw_query=q, ...)
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAX = 5_000
_BATCH_MAX = 100
_FLUSH_INTERVAL = 2.0
_HTTP_TIMEOUT = 5.0
_SHUTDOWN_TIMEOUT = 5.0
#: How long a console change to the tracing switch takes to reach a service.
#: Short enough to be useful during an incident, long enough not to be a poll
#: storm from every pod.
_FLAGS_INTERVAL = 30.0
_SIGNATURE_HEADER = "X-WiseFood-Analytics-Signature"

_STOP = object()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Telemetry:
    """A background thread that batches events to the gateway.

    A thread rather than an asyncio task because these services are a mix of
    sync and async code — RecipeWrangler's search path is sync, FoodChat's
    orchestrator is async — and a thread is callable from both without the
    caller needing an event loop.

    Every method is safe to call from anywhere: nothing here blocks the caller,
    raises into it, or waits on the network.
    """

    def __init__(self):
        self._queue: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._app = "platform"
        self._enabled = False
        self._url = ""
        self._flags_url = ""
        self._secret = ""
        self.dropped = 0
        self.sent = 0
        self.failed = 0
        # Platform switches, refreshed on a timer from the gateway. Defaults
        # are permissive so a service that cannot reach the gateway keeps
        # behaving as it did before this existed — the switch is a control, not
        # a dependency.
        self._flags: Dict[str, Any] = {"tracing_enabled": True, "tracing_langfuse": True}
        self._flags_fetched_at = 0.0

    # ---------------------------------------------------------- lifecycle --
    def start(self, app: str) -> None:
        self._app = app
        self._url = (os.getenv("ANALYTICS_INGEST_URL") or "").strip()
        self._flags_url = (os.getenv("ANALYTICS_FLAGS_URL") or "").strip()
        if not self._flags_url and self._url.endswith("/internal/events"):
            # Derived rather than configured twice: two URLs that must agree is
            # two URLs that eventually will not.
            self._flags_url = self._url[: -len("/internal/events")] + "/runtime-flags"
        self._secret = (os.getenv("ANALYTICS_INGEST_SECRET") or "").strip()
        self._enabled = bool(
            _flag("ANALYTICS_ENABLED") and self._url and self._secret
        )
        # The flag poller runs even when event reporting is off: an operator
        # must be able to stop tracing platform-wide without analytics being
        # switched on first.
        self._polling = bool(self._flags_url and self._secret)
        if not self._enabled and not self._polling:
            logger.info(
                "telemetry.disabled (ANALYTICS_ENABLED=%s, url=%s, secret=%s)",
                _flag("ANALYTICS_ENABLED"),
                bool(self._url),
                bool(self._secret),
            )
            return
        if self._thread and self._thread.is_alive():
            return
        # Daemon: a stuck flush must never hold up interpreter exit. The atexit
        # hook below is the orderly path; the daemon flag is the guarantee.
        self._thread = threading.Thread(
            target=self._run, name="wf-telemetry", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)
        logger.info("telemetry.started app=%s url=%s", app, self._url)

    def stop(self) -> None:
        """Flush what is queued, briefly, then stop. Safe to call twice."""
        if not self._thread:
            return
        # Nothing new is accepted while stopping — otherwise events kept
        # accumulating in a queue nobody would ever drain again.
        self._enabled = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=_SHUTDOWN_TIMEOUT)
        self._thread = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def tracing_enabled(self, sink: Optional[str] = None) -> bool:
        """Whether this service may still produce traces.

        Consulted by each service's Langfuse module, so tracing can be stopped
        from the console for the whole platform rather than by unsetting keys
        and rolling every pod. Permissive when the gateway is unreachable: a
        control plane that fails closed would take tracing down with it.
        """
        if not self._flags.get("tracing_enabled", True):
            return False
        if sink is None:
            return True
        return bool(self._flags.get(f"tracing_{sink}", True))

    def flags(self) -> Dict[str, Any]:
        return dict(self._flags)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "polling": getattr(self, "_polling", False),
            "queued": self._queue.qsize(),
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
            "flags": dict(self._flags),
        }

    # ------------------------------------------------------------ recording --
    def _submit(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            # Every recorder passes `app` explicitly, possibly as None, so
            # setdefault never fired and the start(app=...) default was dead.
            if event.get("app") is None:
                event["app"] = self._app
            # The correlation id is what makes a service-reported fact joinable
            # to the request that caused it. Picked up from obs_context when it
            # is present, so callers never have to pass it.
            if "request_id" not in event:
                event["request_id"] = _current_request_id()
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.dropped += 1

    def event(
        self,
        event_type: str,
        *,
        props: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        member_id: Optional[str] = None,
        app: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:
        self._submit(
            {
                "kind": "event",
                "type": event_type,
                "props": props or {},
                "user_id": user_id,
                "member_id": member_id,
                "app": app,
                "route": route,
            }
        )

    def search(
        self,
        *,
        surface: str,
        raw_query: Optional[str],
        filters: Optional[Dict[str, Any]] = None,
        result_count_first_pass: Optional[int] = None,
        result_count_final: Optional[int] = None,
        relaxed: bool = False,
        lexical_fallback: bool = False,
        latency_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        member_id: Optional[str] = None,
        app: Optional[str] = None,
    ) -> None:
        """Report one search.

        ``result_count_first_pass`` and ``result_count_final`` are separate
        because a search that retries on an empty hit set hides its own miss in
        the returned total, and "found nothing" and "found nothing until we
        loosened the constraints" are different product problems.
        """
        self._submit(
            {
                "kind": "search",
                "type": "search.performed",
                "surface": surface,
                "raw_query": raw_query,
                "filters": filters or {},
                "result_count_first_pass": result_count_first_pass,
                "result_count_final": result_count_final,
                "relaxed": bool(relaxed),
                "lexical_fallback": bool(lexical_fallback),
                "latency_ms": _as_int(latency_ms),
                "user_id": user_id,
                "member_id": member_id,
                "app": app,
            }
        )

    def llm_usage(
        self,
        *,
        model: Optional[str],
        feature: Optional[str] = None,
        provider: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        member_id: Optional[str] = None,
        app: Optional[str] = None,
    ) -> None:
        """Report the cost of one model call.

        Langfuse holds the trace; this holds the number, because the Langfuse
        metrics API cannot group by user — high-cardinality dimensions are
        filter-only there — so "tokens per user" is not a report it can produce.
        """
        self._submit(
            {
                "kind": "llm_usage",
                "type": "llm.usage",
                "model": model,
                "feature": feature,
                "provider": provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "latency_ms": _as_int(latency_ms),
                "trace_id": trace_id,
                "user_id": user_id,
                "member_id": member_id,
                "app": app,
            }
        )

    def feedback(
        self,
        *,
        target_type: str,
        target_id: Optional[str],
        rating_kind: str,
        rating_value: Optional[str] = None,
        rating_value_num: Optional[float] = None,
        reason: Optional[str] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        member_id: Optional[str] = None,
        app: Optional[str] = None,
    ) -> None:
        """Mirror a feedback signal the service stores in its own tables.

        The service keeps its copy where it drives behaviour — FoodChat's
        feedback feeds personalisation — and mirrors here so an expert can see
        every surface's feedback in one place instead of four.
        """
        self._submit(
            {
                "kind": "feedback",
                "type": "feedback.given",
                "target_type": target_type,
                "target_id": target_id,
                "rating_kind": rating_kind,
                "rating_value": rating_value,
                "rating_value_num": rating_value_num,
                "reason": reason,
                "comment": comment,
                "user_id": user_id,
                "member_id": member_id,
                "app": app,
            }
        )

    # --------------------------------------------------------------- drain --
    def _run(self) -> None:
        # The first flag fetch happens here, on the worker, not in start(): a
        # blocking HTTP call on the main thread during service startup would
        # hold boot for the full timeout whenever the gateway is unreachable —
        # the one moment a control plane must not get in the way.
        self._refresh_flags(force=True)
        while True:
            batch = self._collect()
            if batch is None:
                return
            if batch:
                self._post(batch)
            self._refresh_flags()

    def _collect(self) -> Optional[List[Dict[str, Any]]]:
        """One batch, or None to stop."""
        batch: List[Dict[str, Any]] = []
        try:
            first = self._queue.get(timeout=_FLUSH_INTERVAL)
        except queue.Empty:
            return batch
        if first is _STOP:
            return self._final_batch()
        batch.append(first)

        deadline = time.monotonic() + _FLUSH_INTERVAL
        while len(batch) < _BATCH_MAX:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                self._post(batch)
                return self._final_batch()
            batch.append(item)
        return batch

    def _final_batch(self) -> None:
        """Send whatever is left, in batches, then signal the loop to end.

        Bounded by the shutdown budget rather than by one batch: an earlier
        version sent at most `_BATCH_MAX` and silently dropped the rest without
        counting it.
        """
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
        while not self._queue.empty() and time.monotonic() < deadline:
            remaining: List[Dict[str, Any]] = []
            while not self._queue.empty() and len(remaining) < _BATCH_MAX:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP:
                    remaining.append(item)
            if remaining:
                self._post(remaining)
        leftover = self._queue.qsize()
        if leftover:
            self.dropped += leftover
        return None

    def _refresh_flags(self, force: bool = False) -> None:
        """Pull the platform switches. Never raises, never blocks a caller."""
        if not getattr(self, "_polling", False):
            return
        if not force and (time.monotonic() - self._flags_fetched_at) < _FLAGS_INTERVAL:
            return
        self._flags_fetched_at = time.monotonic()
        body = b""
        request = urllib.request.Request(
            self._flags_url,
            data=None,
            method="GET",
            headers={_SIGNATURE_HEADER: self._sign(body)},
        )
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except Exception as exc:
            logger.debug("telemetry.flags_fetch_failed: %s", exc)
            return
        # The gateway wraps every response in {help, success, result}.
        values = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(values, dict):
            self._flags = values

    def _sign(self, body: bytes) -> str:
        issued_at = int(time.time())
        digest = hmac.new(
            self._secret.encode(), f"{issued_at}|".encode() + body, hashlib.sha256
        ).hexdigest()
        return f"{issued_at}.{digest}"

    def _post(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            body = json.dumps({"events": batch}, default=str).encode("utf-8")
        except Exception as exc:
            self.failed += len(batch)
            logger.warning("telemetry.encode_failed: %s", exc)
            return

        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                _SIGNATURE_HEADER: self._sign(body),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
                if 200 <= response.status < 300:
                    self.sent += len(batch)
                    return
                self.failed += len(batch)
                logger.warning("telemetry.rejected status=%s", response.status)
        except urllib.error.HTTPError as exc:
            # Dropped, not retried. A batch the gateway refuses will be refused
            # again, and a retry queue behind a persistent 4xx grows until the
            # process dies. Telemetry is never worth that.
            self.failed += len(batch)
            logger.warning("telemetry.http_error status=%s", exc.code)
        except Exception as exc:
            self.failed += len(batch)
            logger.warning("telemetry.post_failed: %s", exc)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


#: Where `obs_context` sits, per service. Flat `src/` in three of them, inside
#: the package in RecipeWrangler — tried in order so this file stays identical
#: everywhere.
_CONTEXT_MODULES = ("obs_context", "recipe_wrangler.api.obs_context")


_context_module: Any = None


def _current_request_id() -> Optional[str]:
    """The correlation id, if this service has obs_context wired in.

    The module is resolved once: re-attempting a failing import on every event
    is a measurable cost in the service whose copy lives under a package name.
    """
    global _context_module
    if _context_module is None:
        import importlib

        for module_name in _CONTEXT_MODULES:
            try:
                _context_module = importlib.import_module(module_name)
                break
            except Exception:
                continue
        else:
            _context_module = False
    if not _context_module:
        return None
    try:
        return _context_module.get_request_id()
    except Exception:
        return None


# ------------------------------------------------------- LangChain usage ----
#: Where the providers put token counts, and under what names. There is no one
#: shape: Groq and OpenAI-compatible endpoints report ``token_usage`` with
#: prompt/completion names, Anthropic reports ``usage`` with input/output ones,
#: Gemini counts under its own ``*_token_count`` names, Ollama writes
#: ``prompt_eval_count``/``eval_count`` straight into the response metadata, and
#: streaming drops ``llm_output`` altogether so the only surviving copy is on
#: the message. All of them are read, because it is the split that matters:
#: output tokens cost several times input, so a combined total cannot be turned
#: back into money afterwards.
_USAGE_ALIASES = {
    "input_tokens": (
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
        "prompt_eval_count",
    ),
    "output_tokens": (
        "output_tokens",
        "completion_tokens",
        "candidates_token_count",
        "eval_count",
    ),
    "total_tokens": ("total_tokens", "total_token_count"),
}

#: Keys under which a usage dict hides one level inside a metadata dict.
_USAGE_CONTAINERS = ("token_usage", "usage", "usage_metadata")


def usage_from_llm_result(result: Any) -> Dict[str, int]:
    """Token counts from a LangChain ``LLMResult``, or empty when absent.

    Deliberately duck-typed and total: a provider that reports nothing, or
    reports it somewhere new, yields an empty dict rather than an exception on
    the request path.
    """
    counts: Dict[str, int] = {}

    def absorb(source: Any) -> None:
        if not isinstance(source, dict):
            return
        for canonical, names in _USAGE_ALIASES.items():
            if counts.get(canonical) is not None:
                continue
            for name in names:
                value = source.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    counts[canonical] = int(value)
                    break

    def absorb_nested(source: Any) -> None:
        """A metadata dict, read both as itself and as a container.

        Ollama puts the counts at the top level of ``response_metadata``,
        OpenAI nests them under ``token_usage``, Anthropic under ``usage`` and
        Gemini under ``usage_metadata``. Reading every dict both ways is what
        keeps this from being one branch per provider.
        """
        if not isinstance(source, dict):
            return
        absorb(source)
        for key in _USAGE_CONTAINERS:
            absorb(source.get(key))

    try:
        # `llm_output` is the provider's aggregate for the whole call, so it is
        # read first and wins. The per-generation copies below are what is left
        # when streaming, which never populates it.
        absorb_nested(getattr(result, "llm_output", None))
        for generation_list in getattr(result, "generations", None) or []:
            for generation in generation_list or []:
                message = getattr(generation, "message", None)
                absorb(getattr(message, "usage_metadata", None))
                absorb_nested(getattr(message, "response_metadata", None))
                # Streaming runs carry the final chunk's counts here rather
                # than on `llm_output`, which is where the split used to be
                # lost and every streamed call reported nothing at all.
                absorb_nested(getattr(generation, "generation_info", None))
    except Exception:
        return {}

    if (
        counts.get("total_tokens") is None
        and counts.get("input_tokens") is not None
        and counts.get("output_tokens") is not None
    ):
        counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]
    return counts


def _current_langfuse_trace_id() -> Optional[str]:
    """The Langfuse trace a model call belongs to, when there is one.

    Read from the OpenTelemetry context that the Langfuse LangChain handler
    makes current for the duration of a run, rather than from that handler's
    own ``last_trace_id``: the handler is constructed once and attached to a
    pooled client, so ``last_trace_id`` is whichever call started most
    recently, not this one. Under concurrency that is a wrong answer, which is
    worse than none — a usage row linked to somebody else's trace.

    Only consulted once the service has actually initialised the SDK. Importing
    `langfuse` here would construct a client as a side effect, and telemetry
    must never be the thing that switches tracing on.
    """
    module = sys.modules.get("langfuse")
    if module is None:
        return None
    try:
        return module.get_client().get_current_trace_id()
    except Exception:
        logger.debug("telemetry.trace_id_unavailable", exc_info=True)
        return None


#: How many in-flight model calls the usage callback remembers. A run that
#: never ends — a cancelled request, a client torn down mid-call — leaves its
#: entry behind, and an unbounded dict on a long-lived pooled handler is a leak
#: that only shows up in production.
_MAX_INFLIGHT_LLM_RUNS = 512


def usage_callback(
    feature: str,
    *,
    provider: Optional[str] = None,
    app: Optional[str] = None,
    identity: Optional[Any] = None,
):
    """A LangChain callback reporting each model call's token cost and latency.

    ``identity`` is a *callable* returning ``{"user_id": ..., "member_id": ...}``,
    resolved when the call ends rather than when the handler is built — these
    handlers are attached once to a pooled client and then serve every user, so
    an identity captured at construction would label every call with whoever
    happened to warm the pool.

    Returns None when LangChain is unavailable, so a caller can splice the
    result into ``callbacks=[...]`` without guarding the import itself.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:  # pragma: no cover - langchain is present in practice
        return None

    class _UsageReporter(BaseCallbackHandler):
        def __init__(self) -> None:
            # Keyed by LangChain's `run_id`, not held as one attribute on the
            # handler: the same instance serves every concurrent call through
            # the pooled client, so a single start timestamp would belong to
            # whichever call started last and every latency would be someone
            # else's.
            self._runs: Dict[Any, Dict[str, Any]] = {}
            self._lock = threading.Lock()

        # Chat models fire `on_chat_model_start`, completion models
        # `on_llm_start`. Both are implemented rather than leaning on
        # LangChain's NotImplementedError fallback from the first to the
        # second, which costs a raised exception and a warning per call.
        def on_chat_model_start(self, serialized, messages, **kwargs) -> None:  # noqa: ANN001
            self._begin(kwargs.get("run_id"))

        def on_llm_start(self, serialized, prompts, **kwargs) -> None:  # noqa: ANN001
            self._begin(kwargs.get("run_id"))

        def _begin(self, run_id: Any) -> None:
            try:
                entry = {
                    "started": time.monotonic(),
                    # Captured at both ends of the call because the two
                    # handlers on the client are unordered: Langfuse makes its
                    # span current in `on_llm_start` and drops it again in
                    # `on_llm_end`, so whichever of the two hooks runs while
                    # the span is current is the one that can see the id.
                    "trace_id": _current_langfuse_trace_id(),
                }
                with self._lock:
                    if len(self._runs) >= _MAX_INFLIGHT_LLM_RUNS:
                        self._runs.pop(next(iter(self._runs)), None)
                    self._runs[run_id] = entry
            except Exception:  # pragma: no cover - never break a model call
                logger.debug("telemetry.usage_callback_failed", exc_info=True)

        def _finish(self, run_id: Any) -> Dict[str, Any]:
            """Take this run's start state, removing it from the map."""
            with self._lock:
                return self._runs.pop(run_id, None) or {}

        def on_llm_error(self, error, **kwargs) -> None:  # noqa: ANN001
            # A failed call has no usage to report, but its entry still has to
            # go: without this the map grows by one for every provider timeout.
            try:
                self._finish(kwargs.get("run_id"))
            except Exception:  # pragma: no cover - never break a model call
                logger.debug("telemetry.usage_callback_failed", exc_info=True)

        def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
            try:
                started = self._finish(kwargs.get("run_id"))
                counts = usage_from_llm_result(response)
                output = getattr(response, "llm_output", None) or {}
                model = None
                if isinstance(output, dict):
                    model = output.get("model_name") or output.get("model")
                who = {}
                if callable(identity):
                    try:
                        who = identity() or {}
                    except Exception:
                        who = {}
                latency_ms = None
                if started.get("started") is not None:
                    latency_ms = (time.monotonic() - started["started"]) * 1000.0
                TELEMETRY.llm_usage(
                    model=model,
                    feature=feature,
                    provider=provider,
                    app=app,
                    input_tokens=counts.get("input_tokens"),
                    output_tokens=counts.get("output_tokens"),
                    total_tokens=counts.get("total_tokens"),
                    latency_ms=latency_ms,
                    trace_id=started.get("trace_id") or _current_langfuse_trace_id(),
                    user_id=who.get("user_id"),
                    member_id=who.get("member_id"),
                )
            except Exception:  # pragma: no cover - never break a model call
                logger.debug("telemetry.usage_callback_failed", exc_info=True)

    return _UsageReporter()

TELEMETRY = Telemetry()
