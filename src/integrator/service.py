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


def _data_client(access_token: Optional[str] = None):
    """A catalog client acting as the caller, or None if one cannot be built.

    `access_token` is the curator's own bearer, forwarded by the gateway. With
    it, every catalog read and write the agent performs carries that person's
    roles and is refused by the catalog wherever they would be refused
    directly — which is the whole point: an assistant must not be a way to do
    what the person driving it may not do.

    Without one there is no fallback to the service account. A missing token
    disables the catalog tools instead, because "act as the curator" quietly
    becoming "act as the platform" is the failure this is here to prevent, and
    a failure that makes the assistant *more* capable is the kind nobody
    reports. Research still works, so the turn is degraded rather than broken.
    """
    try:
        from wisefood.client import Credentials, DataClient
    except Exception:  # noqa: BLE001
        logger.info("integrator: wisefood client not installed")
        return None

    # DATA_API_URL, not WISEFOOD_API_URL. The first is the data catalog, which
    # serves /guides, /articles and the rest; the second is the gateway, which
    # does not — pointing at it returned {"detail": "Not Found"} for every
    # catalog tool, so the assistant could never see what the platform already
    # held. `backend/platform.py` has always used DATA_API_URL; this did not.
    base = config.settings.get("DATA_API_URL")
    if not base or not access_token:
        if not access_token:
            logger.info("integrator: no caller token, catalog tools disabled")
        return None
    try:
        return DataClient(base_url=base,
                          credentials=Credentials(access_token=access_token))
    except Exception:  # noqa: BLE001
        logger.warning("integrator: catalog client unavailable", exc_info=True)
        return None


def tool_context(*, user_sub: str, session_id: Optional[str] = None,
                 access_token: Optional[str] = None) -> ToolContext:
    """The context every tool runs under, for this caller and this session."""
    return ToolContext(
        data_client=_data_client(access_token),
        groq_client=_groq_client(),
        proposal_store=_STORE,
        writes_enabled=bool(config.settings.get("INTEGRATOR_WRITES_ENABLED", False)),
        research_model=config.settings.get("INTEGRATOR_RESEARCH_MODEL", "groq/compound"),
        inference_model=config.settings.get("INTEGRATOR_INFERENCE_MODEL"),
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


class RateLimited(RuntimeError):
    """Too much, too fast. Carries how long to wait, so a client can say so."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


def _turns_last_hour(db, user_sub: str) -> int:
    """How many questions this person has asked in the last hour.

    Counted from the messages already being written, so there is no separate
    counter to drift, and no cache whose outage silently lifts the limit. The
    join is by session owner rather than by message, because a message row
    does not carry a subject — the session it belongs to does.
    """
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    return (db.query(IntegratorMessage)
              .join(IntegratorSession, IntegratorSession.id == IntegratorMessage.session_id)
              .filter(IntegratorSession.user_sub == user_sub,
                      IntegratorMessage.role == "user",
                      IntegratorMessage.created_at >= since)
              .count())


def chat(*, session_id: str, user_sub: str, message: str,
         access_token: Optional[str] = None) -> Dict[str, Any]:
    """One turn: persist the question, run the loop, persist what it produced."""
    with _session_factory() as db:
        _owned_session(db, session_id, user_sub)
        ceiling = int(config.settings.get("INTEGRATOR_MAX_TURNS_PER_HOUR", 60))
        # Checked before the model is called, because the cost of a turn is
        # incurred there and refusing afterwards would be an apology rather
        # than a limit.
        if _turns_last_hour(db, user_sub) >= ceiling:
            raise RateLimited(
                f"that is {ceiling} questions in an hour, which is this "
                f"deployment's limit. It will clear as the hour rolls forward.",
                retry_after=300)
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
            tool_context=tool_context(user_sub=user_sub, session_id=session_id,
                                      access_token=access_token),
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


async def chat_stream(*, session_id: str, user_sub: str, message: str,
                      access_token: Optional[str] = None,
                      heartbeat: float = 15.0):
    """One turn, as it happens.

    The agent loop is synchronous — it blocks on Groq and on HTTP — so it runs
    in a thread and pushes events onto a queue this drains. The alternative,
    an async rewrite of the loop, would mean two implementations of the part
    that decides what the assistant may do, and that is the last thing to keep
    in two places.

    A heartbeat goes out whenever nothing has happened for a while. Without
    one an idle stream looks identical to a dead connection to every proxy
    between here and the browser, and a single web search can easily be
    quieter than their patience.
    """
    import asyncio
    import queue as queue_mod
    import threading

    from integrator.tracing import trace_run

    with _session_factory() as db:
        _owned_session(db, session_id, user_sub)
        ceiling = int(config.settings.get("INTEGRATOR_MAX_TURNS_PER_HOUR", 60))
        if _turns_last_hour(db, user_sub) >= ceiling:
            raise RateLimited(
                f"that is {ceiling} questions in an hour, which is this "
                f"deployment's limit. It will clear as the hour rolls forward.",
                retry_after=300)
        rows = (
            db.query(IntegratorMessage)
            .filter(IntegratorMessage.session_id == session_id)
            .order_by(IntegratorMessage.seq.asc()).all()
        )
        past = replay(rows)
        seq = (rows[-1].seq + 1) if rows else 0
        db.add(IntegratorMessage(session_id=session_id, seq=seq, role="user",
                                 content=message))
        db.commit()

    events: "queue_mod.Queue" = queue_mod.Queue()
    model = config.settings.get("INTEGRATOR_MODEL", "openai/gpt-oss-120b")

    def work():
        try:
            with trace_run(session_id=session_id, user_sub=user_sub,
                           question=message, model=model) as trace:
                agent = IntegratorAgent(
                    registry=_REGISTRY,
                    tool_context=tool_context(user_sub=user_sub,
                                              session_id=session_id,
                                              access_token=access_token),
                    groq_client=_groq_client(),
                    allow_writes=bool(config.settings.get(
                        "INTEGRATOR_WRITES_ENABLED", False)),
                    trace=trace,
                )
                outcome = agent.run_streamed(
                    past, message, lambda name, payload: events.put((name, payload)))
            _persist_turn(session_id, seq, outcome, message)
            events.put(("done", {
                "session_id": session_id, "reply": outcome["reply"],
                "stop_reason": outcome["stop_reason"],
                "timeline": outcome["timeline"], "steps": outcome["steps"],
                "tokens": outcome["tokens"], "model": outcome["model"],
            }))
        except Exception as exc:  # noqa: BLE001
            logger.exception("integrator: streamed turn failed")
            # The stream is already a 200 by the time this can happen, so a
            # failure has to arrive as an event or the client waits forever.
            events.put(("error", {"detail": f"{type(exc).__name__}: {exc}"[:400]}))
        finally:
            events.put((None, None))

    threading.Thread(target=work, name=f"integrator-turn-{session_id}",
                     daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        try:
            name, payload = await asyncio.wait_for(
                loop.run_in_executor(None, events.get), timeout=heartbeat)
        except asyncio.TimeoutError:
            yield ("heartbeat", None)
            continue
        if name is None:
            return
        yield (name, payload)


def _persist_turn(session_id: str, seq: int, outcome: Dict[str, Any],
                  message: str) -> None:
    """Write the turn down. Shared by the streamed and non-streamed paths."""
    with _session_factory() as db:
        session = db.get(IntegratorSession, session_id)
        for offset, turn in enumerate(outcome["messages"], start=1):
            db.add(IntegratorMessage(
                session_id=session_id, seq=seq + offset, role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
                tool_call_id=turn.get("tool_call_id"),
                tool_name=turn.get("tool_name"),
                steps=turn.get("steps"),
            ))
        if session is not None and not session.title:
            session.title = message.strip()[:120] or None
        db.commit()


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
               limit: int = 100, user_sub: Optional[str] = None,
               is_admin: bool = False) -> List[Dict[str, Any]]:
    """The audit trail — what the agent actually did.

    Scoped to the caller unless they are an admin. A tool call carries its
    arguments, which for this agent means the queries a curator typed and the
    URLs they were chasing; that is their work, not the room's. Admins see
    everything because an audit trail nobody can read in full is not one.

    The proposal queue itself stays shared — curators rank each other's
    candidates, which is the point of it — so this is the narrower rule for
    the narrower thing.
    """
    with _session_factory() as db:
        q = db.query(IntegratorToolCall)
        if not is_admin and user_sub:
            owned = [row.id for row in
                     db.query(IntegratorSession.id)
                       .filter(IntegratorSession.user_sub == user_sub).all()]
            q = q.filter(IntegratorToolCall.session_id.in_(owned or [""]))
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


# ------------------------------------------------------------------- runs --

#: A run whose heartbeat is older than this is not working, whatever its
#: status column says — the pod carrying it went away mid-extraction.
STALL_AFTER_SECONDS = 300


def _core_transport(loop):
    """POST and GET against FoodScholar's own guideline routes, in process.

    A loopback HTTP call would need a token minted for ourselves and would
    deadlock a single-worker deployment on its own request. Calling the
    service directly avoids both — but two of the three entry points are
    coroutines, and the async engine belongs to the event loop that built it.
    Touching it from the run's worker thread is how you get `Future attached
    to a different loop` an hour into an extraction.

    So the coroutines are handed back to the loop that owns them and the
    thread waits for the answer. `loop` is None only in tests, where there is
    no application loop and nothing pooled to confuse.
    """
    import asyncio

    from services.guideline_jobs import GuidelineJobService

    job_service = GuidelineJobService()

    def _enrichment_service():
        from services.enrichment_jobs import EnrichmentJobService

        return EnrichmentJobService()

    def _await(coro, timeout: float = 300.0):
        if loop is None:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    def _dump(model) -> Dict[str, Any]:
        return model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)

    def post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        parts = [p for p in path.split("/") if p]
        if "/enrich/articles/" in path:
            # The urn contains slashes, so it is whatever follows the segment
            # rather than the last component.
            urn = path.split("/enrich/articles/", 1)[1]
            service = _enrichment_service()
            service.enqueue(urn, force=bool(body.get("force")),
                            requested_by=body.get("requested_by"))
            return _dump(service.get_status(urn))
        # /api/v1/guidelines/<action>/<artifact_uuid>
        action, artifact_uuid = parts[-2], parts[-1]
        if action == "extract":
            job_service.enqueue_job(artifact_uuid=artifact_uuid,
                                    guide_id=body.get("guide_id"))
            return _dump(_await(job_service.get_job_response(artifact_uuid)))
        if action == "import":
            return _dump(_await(job_service.import_latest_result_to_guide(
                artifact_uuid=artifact_uuid,
                guide_id=body["guide_id"],
                dry_run=bool(body.get("dry_run", True)),
            ), timeout=900.0))
        raise ValueError(f"no in-process route for POST {path}")

    def get(path: str) -> Dict[str, Any]:
        parts = [p for p in path.split("/") if p]
        if "/enrich/articles/" in path:
            urn = path.split("/enrich/articles/", 1)[1]
            return _dump(_enrichment_service().get_status(urn))
        action, artifact_uuid = parts[-2], parts[-1]
        if action == "extract":
            return _dump(_await(job_service.get_job_response(artifact_uuid)))
        raise ValueError(f"no in-process route for GET {path}")

    return post, get


def _run_dict(row, *, now=None) -> Dict[str, Any]:
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    status = row.status
    if status == "running" and row.heartbeat_at is not None:
        beat = row.heartbeat_at
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=timezone.utc)
        if (now - beat).total_seconds() > STALL_AFTER_SECONDS:
            # Not written back: the pod may yet return and carry on. This is
            # what a reader is told, and it is the truth either way.
            status = "stalled"
    return {
        "id": row.id, "proposal_id": row.proposal_id, "session_id": row.session_id,
        "status": status, "stage": row.stage, "steps": row.steps or [],
        "error": row.error, "result": row.result or {},
        "wrote_anything": bool(row.wrote_anything), "dry_run": bool(row.dry_run),
        "started_by": row.started_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    from models.db import IntegrationRun

    with _session_factory() as db:
        row = db.get(IntegrationRun, run_id)
        return _run_dict(row) if row else None


def list_runs(*, proposal_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    from models.db import IntegrationRun

    with _session_factory() as db:
        q = db.query(IntegrationRun)
        if proposal_id:
            q = q.filter(IntegrationRun.proposal_id == proposal_id)
        rows = (q.order_by(IntegrationRun.created_at.desc())
                 .limit(max(1, min(int(limit), 100))).all())
        return [_run_dict(r) for r in rows]


def _active_run(db, proposal_id: str):
    """A run that is genuinely still going, ignoring ones whose pod died."""
    from datetime import datetime, timezone

    from models.db import IntegrationRun

    rows = (db.query(IntegrationRun)
              .filter(IntegrationRun.proposal_id == proposal_id,
                      IntegrationRun.status.in_(("queued", "running")))
              .all())
    now = datetime.now(timezone.utc)
    for row in rows:
        if _run_dict(row, now=now)["status"] != "stalled":
            return row
    return None


def _live_runs(db, user_sub: Optional[str] = None) -> int:
    """Runs genuinely in flight, ignoring ones whose pod died.

    Without discounting stalled runs the ceiling would be a one-way ratchet: a
    node failure would permanently consume capacity that nothing is using.
    """
    from datetime import datetime, timezone

    from models.db import IntegrationRun

    q = db.query(IntegrationRun).filter(
        IntegrationRun.status.in_(("queued", "running")))
    if user_sub:
        q = q.filter(IntegrationRun.started_by == user_sub)
    now = datetime.now(timezone.utc)
    return sum(1 for row in q.all() if _run_dict(row, now=now)["status"] != "stalled")


def _check_run_capacity(db, user_sub: str) -> None:
    """Two ceilings: one per person, one for the deployment.

    Each run holds a thread for as long as its extraction takes — minutes,
    sometimes an hour — so this bounds threads as much as it bounds spend.
    Without it, approving forty proposals and pressing integrate on each would
    put forty polling threads in one pod.
    """
    per_user = int(config.settings.get("INTEGRATOR_MAX_RUNS_PER_USER", 3))
    total = int(config.settings.get("INTEGRATOR_MAX_RUNS_TOTAL", 10))
    if _live_runs(db, user_sub) >= per_user:
        raise RateLimited(
            f"you already have {per_user} integrations running. Wait for one to "
            f"finish — they resume where they left off if anything goes wrong.")
    if _live_runs(db) >= total:
        raise RateLimited(
            f"the platform is already running {total} integrations, which is its "
            f"limit. Yours will start once one of them finishes.")


def start_integration(*, proposal_id: str, user_sub: str,
                      dry_run: bool = False,
                      access_token: Optional[str] = None) -> Dict[str, Any]:
    """Begin integrating an approved proposal. Returns the run immediately.

    The work happens on a thread because it waits on an extraction that takes
    minutes; the caller gets a run id and polls. Everything the thread does is
    written to the run row as it goes, so a reader who arrives late — or after
    a restart — sees the same account as one who watched.
    """
    import asyncio
    import threading
    from datetime import datetime, timezone

    from models.db import IntegrationRun

    from integrator.executor import Integration, new_run_id

    proposal = _STORE.get(proposal_id)
    if proposal is None:
        raise LookupError("no such proposal")
    if proposal.status != "approved":
        raise PermissionError("this proposal has not been approved")
    if not config.settings.get("INTEGRATOR_WRITES_ENABLED", False):
        raise PermissionError(
            "catalog writes are switched off in this deployment")

    with _session_factory() as db:
        if _active_run(db, proposal_id) is not None:
            raise RuntimeError("an integration is already running for this proposal")
        _check_run_capacity(db, user_sub)
        row = IntegrationRun(
            id=new_run_id(), proposal_id=proposal_id, session_id=proposal.session_id,
            status="queued", stage="queued", steps=[], result={},
            dry_run=bool(dry_run), started_by=user_sub,
            heartbeat_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
        run_id = row.id

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    ctx = tool_context(user_sub=user_sub, session_id=proposal.session_id,
                       access_token=access_token)
    ctx.core_post, ctx.core_get = _core_transport(loop)

    def persist(snapshot: Dict[str, Any]) -> None:
        with _session_factory() as db:
            live = db.get(IntegrationRun, run_id)
            if live is None:
                return
            live.status = snapshot["status"]
            live.stage = snapshot.get("stage")
            live.steps = snapshot.get("steps") or []
            live.result = snapshot.get("result") or {}
            live.error = snapshot.get("error")
            live.wrote_anything = bool(snapshot.get("wrote_anything"))
            live.heartbeat_at = datetime.now(timezone.utc)
            if snapshot.get("finished"):
                live.finished_at = datetime.now(timezone.utc)
            db.commit()

    def work() -> None:
        outcome = Integration(
            proposal=proposal, registry=_REGISTRY, ctx=ctx, persist=persist,
            poll_interval=float(config.settings.get("INTEGRATOR_POLL_INTERVAL", 20)),
            poll_timeout=float(config.settings.get("INTEGRATOR_EXTRACTION_TIMEOUT", 3600)),
            dry_run=bool(dry_run),
        ).run()
        # The proposal's own status follows its last run, so a curator reading
        # the queue sees what happened without opening each one.
        try:
            _STORE.update(proposal_id,
                          status="integrated" if outcome["status"] == "succeeded"
                          else "failed",
                          result={**(proposal.result or {}), **outcome["result"]})
        except Exception:  # noqa: BLE001
            logger.warning("integrator: proposal status not updated", exc_info=True)

    threading.Thread(target=work, name=f"integration-{run_id}", daemon=True).start()
    return get_run(run_id)
