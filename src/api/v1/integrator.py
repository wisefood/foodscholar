"""Source Integrator endpoints.

FoodScholar is internally unauthenticated and trusts the gateway to say who is
calling — the same contract every other surface here uses. The subject arrives
in the request body (`user_sub`), and the gateway is what proves it: the
console reaches these routes only through `/api/v1/foodscholar/integrator/*`,
which is gated on the admin and expert roles there.

There is no approve *tool*; there is an approve *endpoint*, and only a person
with a console session can reach it.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from routers.generic import render

router = APIRouter(prefix="/integrator", tags=["Source Integrator"])


class SessionCreate(BaseModel):
    user_sub: str
    title: Optional[str] = Field(default=None, max_length=300)


class ChatRequest(BaseModel):
    user_sub: str
    message: str = Field(min_length=1, max_length=8000)


class ProposalCreate(BaseModel):
    user_sub: str
    session_id: Optional[str] = None
    kind: str
    title: str = Field(max_length=500)
    source_url: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    population_group: Optional[str] = None
    licence: Optional[str] = None
    rationale: Optional[str] = None


class ApproveRequest(BaseModel):
    user_sub: str
    #: Required when the licence is undetermined or restrictive. Recorded with
    #: the approval, because that is what makes it answerable later.
    override_reason: Optional[str] = Field(default=None, max_length=1000)


class RejectRequest(BaseModel):
    user_sub: str
    reason: str = Field(default="", max_length=1000)


class RerankRequest(BaseModel):
    user_sub: str
    order: List[str]


@router.post("/sessions")
@render()
async def create_session(body: SessionCreate):
    """Start a conversation."""
    from integrator import service

    return service.create_session(user_sub=body.user_sub, title=body.title)


@router.get("/sessions")
@render()
async def list_sessions(user_sub: str, limit: int = 50):
    """This curator's conversations, most recent first."""
    from integrator import service

    return {"sessions": service.list_sessions(user_sub=user_sub, limit=limit)}


@router.get("/sessions/{session_id}/history")
@render()
async def session_history(session_id: str, user_sub: str):
    """Everything said in one conversation, including tool turns."""
    from integrator import service

    return {"messages": service.history(session_id=session_id, user_sub=user_sub)}


@router.post("/sessions/{session_id}/chat")
@render()
async def chat(session_id: str, body: ChatRequest):
    """One turn. The model may call tools; every call is recorded."""
    from integrator import service

    return service.chat(session_id=session_id, user_sub=body.user_sub,
                        message=body.message)


@router.get("/proposals")
@render()
async def list_proposals(session_id: Optional[str] = None,
                         status: Optional[str] = None, limit: int = 100):
    """Candidate sources, in the expert's order where one was set."""
    from integrator import service

    return {"proposals": service.list_proposals(session_id=session_id,
                                                status=status, limit=limit)}


@router.get("/proposals/{proposal_id}")
@render()
async def get_proposal(proposal_id: str):
    from integrator import service

    return service.get_proposal(proposal_id)


@router.post("/proposals")
@render()
async def create_proposal(body: ProposalCreate):
    """Add a candidate by hand, rather than through the conversation."""
    from integrator import service

    fields: Dict[str, Any] = body.model_dump(exclude_none=True)
    for key in ("user_sub", "session_id", "kind", "title"):
        fields.pop(key, None)
    return service.create_proposal(user_sub=body.user_sub, session_id=body.session_id,
                                   kind=body.kind, title=body.title, **fields)


@router.post("/proposals/{proposal_id}/approve")
@render()
async def approve(proposal_id: str, body: ApproveRequest):
    """A person approves. The only way a proposal becomes integratable.

    Refuses a proposal whose licence is undetermined unless a reason is given,
    and records the reason alongside the approval.
    """
    from integrator import service

    return service.approve(proposal_id=proposal_id, user_sub=body.user_sub,
                           override_reason=body.override_reason)


@router.post("/proposals/{proposal_id}/reject")
@render()
async def reject(proposal_id: str, body: RejectRequest):
    from integrator import service

    return service.reject(proposal_id=proposal_id, user_sub=body.user_sub,
                          reason=body.reason)


@router.post("/proposals/rerank")
@render()
async def rerank(body: RerankRequest):
    """The expert's ordering, kept beside the agent's rather than over it."""
    from integrator import service

    return {"proposals": service.rerank(order=body.order, user_sub=body.user_sub)}


@router.get("/backlog")
@render()
async def backlog(kind: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 100, offset: int = 0):
    """The queue of candidate sources, seeded from the project's catalogue."""
    from integrator import service

    return service.list_backlog(kind=kind, status=status, limit=limit, offset=offset)


@router.get("/audit")
@render()
async def audit(session_id: Optional[str] = None,
                proposal_id: Optional[str] = None,
                limit: int = Query(default=100, le=500)):
    """Every tool the agent ran, newest first."""
    from integrator import service

    return {"tool_calls": service.tool_calls(session_id=session_id,
                                             proposal_id=proposal_id, limit=limit)}
