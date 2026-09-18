"""Graph stream events and their SSE wire format.

Deliberately the same shape as services/qa_pipeline/events: one dataclass, one
formatter, one heartbeat constant, and no dependency beyond the standard
library. The QA stream proved the pattern works through this deployment's
ingress; there is no reason for the graph stream to invent a second one.

``name`` is the SSE event name and ``data`` must be JSON-serializable. Every
frame carries a monotonic ``seq`` so a client can detect a gap rather than
silently drawing an incomplete graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GraphEvent:
    """One frame of a graph stream."""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.name in ("done", "error")


def sse_format(event: GraphEvent) -> str:
    """Serialize an event as a single SSE frame."""
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.name}\ndata: {payload}\n\n"


SSE_HEARTBEAT_FRAME = ": keep-alive\n\n"

#: Headers every graph stream returns. X-Accel-Buffering is the one that
#: matters: without it an nginx-style proxy buffers the whole stream and the
#: progressive draw this exists for never happens.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
