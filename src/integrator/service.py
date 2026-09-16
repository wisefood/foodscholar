"""Wiring: sessions, turns, proposals, approval, backlog.

Everything the API layer needs, with the database and the tool context
assembled in one place. The approval function lives here rather than in a tool
module for the reason the whole design rests on: it is the one transition a
model must not be able to make, so it is not reachable as a tool at all.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.postgres import PostgresConnectionSingleton
from config import config
from models.db import (
    IntegratorBacklogItem, IntegratorMessage, IntegratorSession, IntegratorToolCall,
)
from wisefood_mcp import ToolContext, build_registry
from wisefood_mcp.stores import Proposal, approve as approve_proposal, new_proposal_id

from integrator.agent import IntegratorAgent, new_session_id, replay
from integrator.store import PostgresProposalStore, record_tool_call

logger = logging.getLogger(__name__)

_REGISTRY = build_registry()
_STORE = PostgresProposalStore()


def _session_factory():
    return PostgresConnectionSingleton.get_sync_session_factory()()


def _groq_client():
    """A raw Groq client.

    The raw SDK rather than the LangChain pool: the loop needs `tool_calls`
    round-tripped exactly as the provider returns them, and Compound's
    `executed_tools` are not something a chat wrapper surfaces.
    """
    import os

    from groq import Groq

    key = os.environ.get("GROQ_API_KEY") or config.settings.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return Groq(api_key=key)


def _data_client():
    """A catalog client, or None if this deployment has no credentials for one.

    None disables the catalog tools rather than failing the turn: research
    still works, and a curator gets a usable assistant instead of an error.
    """
    try:
        from wisefood.client import Credentials, DataClient
    except Exception:  # noqa: BLE001
        logger.info("integrator: wisefood client not installed")
        return None

    base = config.settings.get("WISEFOOD_API_URL")
    if not base:
        return None
    try:
        return DataClient(base_url=base, credentials=Credentials(
            client_id=config.settings.get("WISEFOOD_CLIENT_ID"),
            client_secret=config.settings.get("WISEFOOD_CLIENT_SECRET"),
        ))
    except Exception:  # noqa: BLE001
        logger.warning("integrator: catalog client unavailable", exc_info=True)
        return None


def tool_context(*, user_sub: str, session_id: Optional[str] = None) -> ToolContext:
    """The context every tool runs under, for this caller and this session."""
    return ToolContext(
        data_client=_data_client(),
        groq_client=_groq_client(),
        proposal_store=_STORE,
        writes_enabled=bool(config.settings.get("INTEGRATOR_WRITES_ENABLED", False)),
        research_model=config.settings.get("INTEGRATOR_RESEARCH_MODEL", "groq/compound"),
        actor=user_sub,
        contact_email=config.settings.get("INTEGRATOR_CONTACT_EMAIL"),
        recorder=lambda record: record_tool_call(record, session_id=session_id),
    )


# ------------------------------------------------------------------ sessions --

def create_session(*, user_sub: str, title: Optional[str] = None) -> Dict[str, Any]:
    row = IntegratorSession(id=new_session_id(), user_sub=user_sub, title=title)
    with _session_factory() as db:
        db.add(row)
        db.commit()
        db.refresh(row)
        return _session_dict(row)


def _session_dict(row: IntegratorSession) -> Dict[str, Any]:
    return {
        "id": row.id, "title": row.title, "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def list_sessions(*, user_sub: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _session_factory() as db:
        rows = (
            db.query(IntegratorSession)
            .filter(IntegratorSession.user_sub == user_sub)
            .order_by(IntegratorSession.updated_at.desc())
            .limit(max(1, min(int(limit), 200)))
            .all()
        )
        return [_session_dict(r) for r in rows]


def _owned_session(db, session_id: str, user_sub: str) -> IntegratorSession:
    row = db.get(IntegratorSession, session_id)
    if row is None or row.user_sub != user_sub:
        # Same answer for "not yours" and "not there": one curator should not
        # learn another's session ids by probing.
        raise LookupError("no such session")
    return row


def history(*, session_id: str, user_sub: str) -> List[Dict[str, Any]]:
    with _session_factory() as db:
        _owned_session(db, session_id, user_sub)
        rows = (
            db.query(IntegratorMessage)
            .filter(IntegratorMessage.session_id == session_id)
            .order_by(IntegratorMessage.seq.asc())
            .all()
        )
        return [
            {"seq": r.seq, "role": r.role, "content": r.content,
             "tool_name": r.tool_name, "steps": r.steps,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ]


def chat(*, session_id: str, user_sub: str, message: str) -> Dict[str, Any]:
    """One turn: persist the question, run the loop, persist what it produced."""
    with _session_factory() as db:
        _owned_session(db, session_id, user_sub)
        rows = (
            db.query(IntegratorMessage)
            .filter(IntegratorMessage.session_id == session_id)
            .order_by(IntegratorMessage.seq.asc())
            .all()
        )
        past = replay(rows)
        seq = (rows[-1].seq + 1) if rows else 0
        db.add(IntegratorMessage(session_id=session_id, seq=seq, role="user",
                                 content=message))
        db.commit()

    from integrator.tracing import trace_run

    model = config.settings.get("INTEGRATOR_MODEL", "openai/gpt-oss-120b")
    with trace_run(session_id=session_id, user_sub=user_sub,
                   question=message, model=model) as trace:
        agent = IntegratorAgent(
            registry=_REGISTRY,
            tool_context=tool_context(user_sub=user_sub, session_id=session_id),
            groq_client=_groq_client(),
            allow_writes=bool(config.settings.get("INTEGRATOR_WRITES_ENABLED", False)),
            trace=trace,
        )
        outcome = agent.run(past, message)

    with _session_factory() as db:
        session = db.get(IntegratorSession, session_id)
        for offset, turn in enumerate(outcome["messages"], start=1):
            db.add(IntegratorMessage(
                session_id=session_id, seq=seq + offset, role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
                tool_call_id=turn.get("tool_call_id"),
                tool_name=turn.get("tool_name"),
                # Only the final assistant turn carries the timeline, so
                # reopening the conversation shows one account of the work
                # rather than one per intermediate turn.
                steps=turn.get("steps"),
            ))
        # The first question names the session, so a curator's list reads as
        # what they asked rather than as a column of ids.
        if session is not None and not session.title:
            session.title = message.strip()[:120] or None
        db.commit()

    return {
        "session_id": session_id,
        "reply": outcome["reply"],
        "stop_reason": outcome["stop_reason"],
        "timeline": outcome["timeline"],
        "steps": outcome["steps"],
        "tokens": outcome["tokens"],
        "model": outcome["model"],
    }


# ----------------------------------------------------------------- proposals --

def list_proposals(*, session_id: Optional[str] = None, status: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
    return [p.to_dict() for p in _STORE.list(session_id=session_id, status=status,
                                             limit=limit)]


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    row = _STORE.get(proposal_id)
    return row.to_dict() if row else None


def create_proposal(*, user_sub: str, session_id: Optional[str], kind: str,
                    title: str, existing_similar: Optional[int] = None,
                    **fields: Any) -> Dict[str, Any]:
    """File a candidate, and score it by the rubric.

    The score is computed here rather than asked of the model. A number a
    model produced for reasons it did not record cannot be argued with; this
    one is arithmetic over facts the tools established, and every component
    of it is on the proposal for a curator to disagree with.
    """
    proposal = Proposal(
        id=new_proposal_id(), kind=kind, title=title, session_id=session_id,
        created_by=user_sub, status="proposed", **fields,
    )
    created = _STORE.create(proposal)
    return rescore(created.id, existing_similar=existing_similar)


def rescore(proposal_id: str, *, existing_similar: Optional[int] = None) -> Dict[str, Any]:
    """Recompute the rank. Called on create, and whenever the licence changes.

    The licence is the heaviest component, so a proposal scored before its
    licence was established is scored on a placeholder — leaving that stale
    would mean the queue is ordered by what we did not yet know.
    """
    from integrator import ranking

    row = _STORE.get(proposal_id)
    if row is None:
        raise LookupError("no such proposal")
    result = ranking.score(row.to_dict(), existing_similar=existing_similar,
                           settings=config.settings)
    metadata = dict(row.metadata or {})
    metadata["ranking"] = {"breakdown": result["breakdown"],
                           "weights": result["weights"]}
    return _STORE.update(
        proposal_id, proposed_rank=result["score"],
        rationale=row.rationale or result["rationale"], metadata=metadata,
    ).to_dict()


def approve(*, proposal_id: str, user_sub: str,
            override_reason: Optional[str] = None) -> Dict[str, Any]:
    """A person approves. Deliberately not a tool.

    There is no path from the model to this function: it is not in the
    registry, so it is not in the schemas the model is given, so it is not
    something the model can call. That is the wall.
    """
    return approve_proposal(_STORE, proposal_id, actor=user_sub,
                            override_reason=override_reason).to_dict()


def reject(*, proposal_id: str, user_sub: str, reason: str = "") -> Dict[str, Any]:
    row = _STORE.get(proposal_id)
    if row is None:
        raise LookupError("no such proposal")
    metadata = dict(row.metadata or {})
    metadata["rejected"] = {"by": user_sub, "reason": reason}
    return _STORE.update(proposal_id, status="rejected", metadata=metadata).to_dict()


def rerank(*, order: List[str], user_sub: str) -> List[Dict[str, Any]]:
    """The expert's order, stored beside the agent's rather than over it.

    Keeping both is what lets somebody ask later whether the rubric actually
    agrees with the people using it.
    """
    out = []
    for position, proposal_id in enumerate(order, start=1):
        out.append(_STORE.update(proposal_id, expert_rank=position).to_dict())
    return out


def tool_calls(*, session_id: Optional[str] = None, proposal_id: Optional[str] = None,
               limit: int = 100) -> List[Dict[str, Any]]:
    """The audit trail — what the agent actually did."""
    with _session_factory() as db:
        q = db.query(IntegratorToolCall)
        if session_id:
            q = q.filter(IntegratorToolCall.session_id == session_id)
        if proposal_id:
            q = q.filter(IntegratorToolCall.proposal_id == proposal_id)
        rows = (q.order_by(IntegratorToolCall.created_at.desc())
                 .limit(max(1, min(int(limit), 500))).all())
        return [
            {"id": r.id, "tool": r.tool, "write": r.is_write, "ok": r.ok,
             "arguments": r.arguments, "error": r.error,
             "duration_ms": r.duration_ms, "actor": r.actor,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ]


# ------------------------------------------------------------------- backlog --

def list_backlog(*, kind: Optional[str] = None, status: Optional[str] = None,
                 limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    with _session_factory() as db:
        q = db.query(IntegratorBacklogItem)
        if kind:
            q = q.filter(IntegratorBacklogItem.kind == kind)
        if status:
            q = q.filter(IntegratorBacklogItem.status == status)
        total = q.count()
        rows = (q.order_by(IntegratorBacklogItem.country.asc().nullslast(),
                           IntegratorBacklogItem.title.asc())
                 .offset(max(0, int(offset)))
                 .limit(max(1, min(int(limit), 500))).all())
        return {
            "total": total, "offset": int(offset),
            "items": [
                {"id": r.id, "kind": r.kind, "title": r.title, "url": r.url,
                 "country": r.country, "language": r.language,
                 "population_group": r.population_group, "status": r.status,
                 "attributes": r.attributes, "source_sheet": r.source_sheet}
                for r in rows
            ],
        }


def seed_backlog(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Idempotent import. Re-running must not multiply the queue."""
    added = skipped = 0
    with _session_factory() as db:
        for item in items:
            key = item["external_key"][:300]
            exists = (db.query(IntegratorBacklogItem)
                        .filter(IntegratorBacklogItem.external_key == key).first())
            if exists:
                skipped += 1
                continue
            db.add(IntegratorBacklogItem(
                id=uuid.uuid4().hex[:16], external_key=key,
                kind=item["kind"], title=item["title"][:500], url=item.get("url"),
                country=item.get("country"), language=item.get("language"),
                population_group=item.get("population_group"),
                attributes=item.get("attributes") or {},
                source_sheet=item.get("source_sheet"),
            ))
            added += 1
        db.commit()
    return {"added": added, "skipped": skipped}
