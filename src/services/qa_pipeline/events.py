"""Pipeline events and their SSE wire format."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PipelineEvent:
    """One observable step of the pipeline.

    ``name`` is the SSE event name; ``data`` must be JSON-serializable. The
    orchestrator stamps ``request_id`` and a monotonic ``seq`` into every
    event's data so a client can reconcile out-of-order rendering.
    """

    name: str
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.name in ("done", "clarification", "error")


def sse_format(event: PipelineEvent) -> str:
    """Serialize an event as a single SSE frame."""
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.name}\ndata: {payload}\n\n"


SSE_HEARTBEAT_FRAME = ": keep-alive\n\n"
