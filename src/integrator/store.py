"""Postgres behind ``wisefood_mcp.stores.ProposalStore``.

The MCP package defines the protocol and the approval rule; this satisfies
the protocol with FoodScholar's own database so that both hosts — the agent
in-process, and anything reaching in over the API — see the same proposals
and the same approval state.

Sync SQLAlchemy on purpose. The tools are ordinary functions called from a
thread by the agent loop, and giving them an async store would mean either
an async tool protocol (which MCP hosts would then have to satisfy too) or
an event loop smuggled into a worker thread.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.postgres import PostgresConnectionSingleton
from models.db import IntegrationProposal, IntegratorToolCall
from wisefood_mcp.stores import Proposal

logger = logging.getLogger(__name__)

#: Model columns that carry straight onto the dataclass and back.
_FIELDS = (
    "id", "session_id", "backlog_id", "kind", "title", "source_url", "status",
    "country", "language", "population_group", "licence", "licence_confidence",
    "licence_evidence", "licence_override_reason", "proposed_rank",
    "expert_rank", "rationale", "plan", "created_by", "approved_by", "result",
)


def _to_dataclass(row: IntegrationProposal) -> Proposal:
    data: Dict[str, Any] = {f: getattr(row, f) for f in _FIELDS}
    data["metadata"] = row.proposal_metadata or {}
    data["licence_evidence"] = row.licence_evidence or []
    data["plan"] = row.plan or []
    data["result"] = row.result or {}
    for stamp in ("created_at", "updated_at", "approved_at"):
        value = getattr(row, stamp, None)
        data[stamp] = value.isoformat() if value else None
    # The dataclass defaults these; None from the database would override.
    data = {k: v for k, v in data.items() if v is not None}
    return Proposal(**data)


class PostgresProposalStore:
    """Proposals, in FoodScholar's schema."""

    def _session(self):
        return PostgresConnectionSingleton.get_sync_session_factory()()

    def create(self, proposal: Proposal) -> Proposal:
        with self._session() as db:
            row = IntegrationProposal(
                id=proposal.id,
                session_id=proposal.session_id,
                backlog_id=proposal.backlog_id,
                kind=proposal.kind,
                title=proposal.title[:500],
                source_url=proposal.source_url,
                status=proposal.status,
                country=proposal.country,
                language=proposal.language,
                population_group=proposal.population_group,
                licence=proposal.licence,
                licence_confidence=proposal.licence_confidence,
                licence_evidence=proposal.licence_evidence or [],
                licence_override_reason=proposal.licence_override_reason,
                proposed_rank=proposal.proposed_rank,
                expert_rank=proposal.expert_rank,
                rationale=proposal.rationale,
                plan=proposal.plan or [],
                proposal_metadata=proposal.metadata or {},
                created_by=proposal.created_by,
                result=proposal.result or {},
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return _to_dataclass(row)

    def get(self, proposal_id: str) -> Optional[Proposal]:
        with self._session() as db:
            row = db.get(IntegrationProposal, proposal_id)
            return _to_dataclass(row) if row else None

    def update(self, proposal_id: str, **changes: Any) -> Proposal:
        with self._session() as db:
            row = db.get(IntegrationProposal, proposal_id)
            if row is None:
                raise KeyError(proposal_id)
            for key, value in changes.items():
                # The dataclass calls it `metadata`; the class cannot.
                setattr(row, "proposal_metadata" if key == "metadata" else key, value)
            db.commit()
            db.refresh(row)
            return _to_dataclass(row)

    def list(self, *, session_id: Optional[str] = None, status: Optional[str] = None,
             limit: int = 100) -> List[Proposal]:
        with self._session() as db:
            q = db.query(IntegrationProposal)
            if session_id:
                q = q.filter(IntegrationProposal.session_id == session_id)
            if status:
                q = q.filter(IntegrationProposal.status == status)
            # Expert order first where one was set, then the agent's score.
            rows = (
                q.order_by(
                    IntegrationProposal.expert_rank.asc().nullslast(),
                    IntegrationProposal.proposed_rank.desc().nullslast(),
                    IntegrationProposal.created_at.asc(),
                )
                .limit(max(1, min(int(limit), 500)))
                .all()
            )
            return [_to_dataclass(r) for r in rows]


def record_tool_call(record: Dict[str, Any], *, session_id: Optional[str] = None) -> None:
    """Persist one tool call. Never raises — auditing must not break the run.

    The result is stored truncated: a research call can return tens of
    kilobytes of page text, and the audit needs to show what came back, not
    keep a second copy of the web.
    """
    try:
        result = record.get("result")
        if result is not None:
            import json

            encoded = json.dumps(result, default=str)
            if len(encoded) > 20_000:
                result = {"truncated": True, "bytes": len(encoded),
                          "head": encoded[:20_000]}
        with PostgresConnectionSingleton.get_sync_session_factory()() as db:
            db.add(IntegratorToolCall(
                session_id=session_id,
                proposal_id=(record.get("arguments") or {}).get("proposal_id"),
                tool=record.get("tool", "?")[:64],
                is_write=bool(record.get("write")),
                ok=bool(record.get("ok")),
                arguments=record.get("arguments"),
                result=result,
                error=record.get("error"),
                duration_ms=record.get("duration_ms"),
                actor=record.get("actor"),
            ))
            db.commit()
    except Exception:  # noqa: BLE001 — see docstring
        logger.warning("integrator: tool call not recorded", exc_info=True)
