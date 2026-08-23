"""Agentic QA pipeline: plan → retrieve → rank → evaluate → repair → answer.

The orchestrator is an async generator of :class:`PipelineEvent` objects, so
the same pipeline serves both the SSE streaming endpoint (events forwarded as
they happen) and the classic ``POST /qa/ask`` endpoint (events drained, final
payload returned).
"""

from services.qa_pipeline.events import PipelineEvent, sse_format
from services.qa_pipeline.state import EvidenceItem, PipelineState

__all__ = [
    "PipelineEvent",
    "sse_format",
    "EvidenceItem",
    "PipelineState",
]
