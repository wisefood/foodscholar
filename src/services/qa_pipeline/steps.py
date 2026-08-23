"""User-visible step timeline for the agentic pipeline.

Turns pipeline stages into :class:`ReasoningStep` records the UI can render
as ChatGPT-style inline collapsible steps. Each step is streamed as a
``step`` SSE event when it starts and again (same id) when it completes with
its duration; the accumulated timeline also rides the final response so the
disclosure survives the end of the stream, page reloads, cache replays, and
the non-streaming endpoint.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from models.qa import ReasoningStep


class StepTracker:
    """Builds and finishes steps, keeping the ordered timeline."""

    def __init__(self) -> None:
        self._steps: Dict[str, ReasoningStep] = {}
        self._order: List[str] = []
        self._started_at: Dict[str, float] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"step-{self._counter}"

    def start(
        self,
        kind: str,
        title: str,
        *,
        detail: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        round: int = 0,
    ) -> ReasoningStep:
        step = ReasoningStep(
            id=self._next_id(),
            kind=kind,
            status="running",
            title=title,
            detail=detail,
            round=round,
            data=data or {},
        )
        self._steps[step.id] = step
        self._order.append(step.id)
        self._started_at[step.id] = time.monotonic()
        return step

    def finish(
        self,
        step_id: str,
        *,
        title: Optional[str] = None,
        detail: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStep:
        step = self._steps[step_id]
        step.status = "done"
        if title is not None:
            step.title = title
        if detail is not None:
            step.detail = detail
        if data:
            step.data = {**step.data, **data}
        started = self._started_at.get(step_id)
        if started is not None:
            step.elapsed_ms = int((time.monotonic() - started) * 1000)
        return step

    def add(
        self,
        kind: str,
        title: str,
        *,
        detail: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        round: int = 0,
    ) -> ReasoningStep:
        """A step that is born completed (rank summaries, notes, verdicts)."""
        step = self.start(kind, title, detail=detail, data=data, round=round)
        step.status = "done"
        step.elapsed_ms = 0
        return step

    def snapshot(self) -> List[ReasoningStep]:
        """The timeline in order; running steps are closed defensively."""
        steps = []
        for step_id in self._order:
            step = self._steps[step_id]
            if step.status == "running":
                self.finish(step_id)
            steps.append(step)
        return steps
