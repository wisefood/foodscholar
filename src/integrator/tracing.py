"""Putting an integrator run on the same trace board as everything else.

The console's LLM observability page already shows what every model call
cost. A research turn that can make a dozen of them — some with browser
search on the research model, some on the conversation model — would
otherwise be the one expensive thing on the platform nobody can see.

Deliberately best-effort and deliberately quiet: a tracing backend that is
down, misconfigured, or switched off must never cost a curator their answer.
Every function here returns something usable when Langfuse is unavailable,
and the caller does not branch on it.

Gated on the platform tracing switch through `tracing_allowed()`, the same
one the rest of FoodScholar honours — an operator who turned tracing off did
not mean "except the integrator".
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


@contextmanager
def trace_run(*, session_id: str, user_sub: Optional[str], question: str,
              model: str) -> Iterator["RunTrace"]:
    """One integrator turn, as one trace.

    Yields a handle the loop can hang tool calls and a final output on. When
    tracing is unavailable the handle is inert and every method is a no-op, so
    the loop is written once rather than once per configuration.
    """
    handle = RunTrace(session_id=session_id, user_sub=user_sub, model=model)
    handle._begin(question)
    try:
        yield handle
    finally:
        handle._end()


class RunTrace:
    """A trace, or a convincing impression of one when Langfuse is off."""

    def __init__(self, *, session_id: str, user_sub: Optional[str], model: str) -> None:
        self.session_id = session_id
        self.user_sub = user_sub
        self.model = model
        self._span: Any = None
        self._client: Any = None

    @property
    def active(self) -> bool:
        return self._span is not None

    def _begin(self, question: str) -> None:
        try:
            from backend.langfuse import get_langfuse_client, tracing_allowed

            if not tracing_allowed():
                return
            self._client = get_langfuse_client()
            if self._client is None:
                return
            self._span = self._client.start_span(
                name="integrator.turn",
                input={"question": question[:2000]},
                metadata={
                    "session_id": self.session_id,
                    "model": self.model,
                    # The subject, not a name or an email: this is a curator
                    # doing their job, and the trace board is shared.
                    "user_sub": self.user_sub,
                    "surface": "source-integrator",
                },
            )
        except Exception:  # noqa: BLE001 — see module docstring
            logger.debug("integrator: tracing unavailable", exc_info=True)
            self._span = None

    def tool(self, name: str, arguments: Any, ok: bool, result: Any,
             duration_ms: Optional[float]) -> None:
        """One tool call, as a child span.

        The result is truncated: a research call can return tens of kilobytes
        of page text, and a trace board is for seeing shape, not for storing a
        second copy of the web.
        """
        if not self.active:
            return
        try:
            child = self._span.start_span(
                name=f"tool.{name}",
                input=arguments if isinstance(arguments, dict) else {"raw": str(arguments)[:1000]},
                metadata={"ok": ok, "duration_ms": duration_ms},
            )
            child.update(output=_clip(result))
            child.end()
        except Exception:  # noqa: BLE001
            logger.debug("integrator: tool span failed", exc_info=True)

    def finish(self, *, reply: str, stop_reason: str, steps: int,
               tokens: int) -> None:
        if not self.active:
            return
        try:
            self._span.update(
                output={"reply": reply[:4000], "stop_reason": stop_reason},
                metadata={"steps": steps, "tokens": tokens,
                          "stop_reason": stop_reason},
            )
        except Exception:  # noqa: BLE001
            logger.debug("integrator: trace update failed", exc_info=True)

    def _end(self) -> None:
        if not self.active:
            return
        try:
            self._span.end()
            # Flushed rather than left to the background: a request-scoped run
            # in a worker that may go idle is exactly the trace that arrives
            # hours late or not at all.
            from backend.langfuse import flush_langfuse

            flush_langfuse()
        except Exception:  # noqa: BLE001
            logger.debug("integrator: trace end failed", exc_info=True)
        finally:
            self._span = None


def _clip(value: Any, limit: int = 4000) -> Any:
    """Enough of a result to recognise it by."""
    try:
        import json

        encoded = json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        encoded = str(value)
    if len(encoded) <= limit:
        return value
    return {"truncated": True, "bytes": len(encoded), "head": encoded[:limit]}
