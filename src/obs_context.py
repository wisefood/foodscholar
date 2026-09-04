"""Correlation context for a WiseFood service.

VENDORED FILE — keep the copies in wisefood-api, foodchat, foodscholar,
RecipeWrangler and wisefood-data-api identical. It is deliberately dependency-
free (stdlib + Starlette, both already present) so it can be copied rather than
packaged, the same delivery model the Langfuse integration uses.

What it buys: the gateway assigns every request an id and forwards it on every
proxied call, so one user action produces log lines carrying the *same* id in
every service it touched. Without that, correlating a slow answer with the
RecipeWrangler search and the LLM trace behind it means guessing from
timestamps.

Wiring, three lines in the service's entry point:

    import obs_context
    obs_context.install_log_filter()          # after logsys.configure()
    app.add_middleware(obs_context.RequestContextMiddleware)

This middleware never rejects a request and never raises. It is not
authentication.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

REQUEST_ID_HEADER = "X-Request-Id"
CLIENT_HEADER = "X-Client"

# A caller-supplied id reaches log lines and outbound headers, so anything
# outside this alphabet (CR/LF above all) is replaced with a generated id rather
# than sanitised in place: a malformed id must never become a half-honoured one.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9._:/+-]{1,64}$")

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("wf_request_id", default=None)
_CLIENT: ContextVar[Optional[str]] = ContextVar("wf_client", default=None)
#: Only set by a service that verifies the caller's token itself (the data
#: catalog does; foodchat and foodscholar are told who the caller is by the
#: gateway and must not invent one).
_USER_SUB: ContextVar[Optional[str]] = ContextVar("wf_user_sub", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def clean_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip()
    return value if _SAFE_ID.match(value) else None


def clean_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip()
    return value if _SAFE_LABEL.match(value) else None


def get_request_id() -> Optional[str]:
    return _REQUEST_ID.get()


def get_client() -> Optional[str]:
    return _CLIENT.get()


def get_user_sub() -> Optional[str]:
    return _USER_SUB.get()


def set_user_sub(sub: Optional[str]) -> None:
    """Record the caller, once their token has actually been verified."""
    _USER_SUB.set(sub or None)


def set_request_id(request_id: Optional[str]) -> None:
    """Adopt a correlation id outside an HTTP request (a worker, a job)."""
    _REQUEST_ID.set(request_id)


def outbound_headers() -> Dict[str, str]:
    """Headers to attach to a call this service makes onward."""
    request_id = _REQUEST_ID.get()
    return {REQUEST_ID_HEADER: request_id} if request_id else {}


def log_fields() -> Dict[str, Any]:
    return {"request_id": _REQUEST_ID.get() or "-", "client": _CLIENT.get() or "-"}


# ------------------------------------------------------------------- logging --
class RequestIdFilter(logging.Filter):
    """Stamp the in-flight request id onto every record passing a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in log_fields().items():
            if not hasattr(record, key) or getattr(record, key, None) in (None, ""):
                setattr(record, key, value)
        return True


_STANDARD_ATTRS = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


class ContextTextFormatter(logging.Formatter):
    """Text formatter that fills the correlation id in itself.

    The obvious alternative — a format string with ``%(request_id)s`` fed by a
    filter — turns any handler the filter missed into a formatting error, i.e.
    a logging change that can break a request. Reading the context here instead
    means the formatter is correct on its own and the filter is only an
    optimisation.
    """

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "request_id", None) in (None, ""):
            record.request_id = get_request_id() or "-"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including every ``extra={...}`` field.

    The text formatters across these services silently discard `extra`, so
    fields the code already passes never reach stdout.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in log_fields().items():
            payload.setdefault(key, value)
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:
            return json.dumps(
                {
                    "ts": payload["ts"],
                    "level": payload["level"],
                    "logger": payload["logger"],
                    "message": payload["message"],
                    "log_error": "payload not serialisable",
                }
            )


def install_log_filter() -> None:
    """Attach :class:`RequestIdFilter` to every configured handler.

    Handler-level rather than logger-level: every record reaching a handler
    passes that handler's filters, so no record can arrive at a formatter that
    references ``%(request_id)s`` without the attribute being set.

    Safe to call more than once.
    """
    seen = set()
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in list(logging.root.manager.loggerDict)
    ]
    for logger in loggers:
        for handler in getattr(logger, "handlers", []):
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
                handler.addFilter(RequestIdFilter())


# ---------------------------------------------------------------- middleware --
class RequestContextMiddleware:
    """Adopt the caller's correlation id, or mint one, for this request.

    Pure ASGI rather than ``BaseHTTPMiddleware``: the latter runs the app in a
    child task, which breaks ContextVars set downstream and interferes with
    streaming responses (FoodScholar proxies SSE).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = None
        client = None
        for raw_key, raw_value in scope.get("headers") or []:
            key = raw_key.decode("latin-1").lower()
            if key == REQUEST_ID_HEADER.lower():
                request_id = clean_id(raw_value.decode("latin-1"))
            elif key == CLIENT_HEADER.lower():
                client = clean_label(raw_value.decode("latin-1"))
        request_id = request_id or new_request_id()

        id_token = _REQUEST_ID.set(request_id)
        client_token = _CLIENT.set(client)
        user_token = _USER_SUB.set(None)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append(
                    (REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _REQUEST_ID.reset(id_token)
            _CLIENT.reset(client_token)
            _USER_SUB.reset(user_token)
