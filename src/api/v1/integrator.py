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

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field

from routers.generic import render

router = APIRouter(prefix="/integrator", tags=["Source Integrator"])


#: The caller's own bearer, forwarded by the gateway.
#:
#: A header rather than a field in the body, for two reasons that both matter
#: here: bodies are logged and this one is persisted to an audit table, and a
#: credential that travels as data eventually gets treated as data. FoodScholar
#: does not verify it — the gateway already did — it only passes it to the
#: catalog, which verifies it properly and applies this person's roles.
DELEGATED_TOKEN_HEADER = "X-WiseFood-Delegated-Token"


def _token(value: Optional[str]) -> Optional[str]:
    return (value or "").strip() or None


def _service():
    """The integrator service, or a readable 503.

    Its tool layer lives in `wisefood_mcp`, which ships with wisefood-client
    0.0.31 and later. An image built against an older pin would otherwise
    raise ImportError from inside a route, which reads as a bug in the
    integrator rather than as a missing dependency.
    """
    try:
        from integrator import service
    except ImportError as exc:  # pragma: no cover - deployment shape
        from exceptions import APIException

        raise APIException(
            status_code=503,
            detail=(
                "The Source Integrator needs wisefood-client 0.0.31 or later "
                f"(wisefood_mcp is not importable: {exc})."
            ),
        ) from exc
    return service


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


class IntegrateRequest(BaseModel):
    user_sub: str
    #: Run every step up to the import and stop there, reporting what would be
    #: created. The entity and its artifact are still created — a preview of
    #: an extraction cannot happen without something to extract from.
    dry_run: bool = False


@router.post("/sessions")
@render()
async def create_session(request: Request, body: SessionCreate):
    """Start a conversation."""
    return _service().create_session(user_sub=body.user_sub, title=body.title)


@router.get("/sessions")
@render()
async def list_sessions(request: Request, user_sub: str, limit: int = 50):
    """This curator's conversations, most recent first."""
    service = _service()
    return {"sessions": service.list_sessions(user_sub=user_sub, limit=limit)}


@router.get("/sessions/{session_id}/history")
@render()
async def session_history(request: Request, session_id: str, user_sub: str):
    """Everything said in one conversation, including tool turns."""
    service = _service()
    return {"messages": service.history(session_id=session_id, user_sub=user_sub)}


@router.post("/sessions/{session_id}/chat")
@render()
async def chat(request: Request, session_id: str, body: ChatRequest,
               delegated: Optional[str] = Header(None, alias=DELEGATED_TOKEN_HEADER)):
    """One turn. The model may call tools; every call is recorded.

    The catalog tools run as the caller, using their forwarded token. Without
    one they are simply absent from the turn — the assistant can still search
    and read the web, and it never reads the catalog with rights the person
    asking does not have.
    """
    from exceptions import APIException

    service = _service()
    try:
        return service.chat(session_id=session_id, user_sub=body.user_sub,
                            message=body.message, access_token=_token(delegated))
    except service.RateLimited as exc:
        # 429 with Retry-After, so a looping client is told to stop rather than
        # left to read a 500 as something worth retrying immediately.
        raise APIException(status_code=429, detail=str(exc),
                           headers={"Retry-After": str(exc.retry_after)}) from exc


@router.get("/proposals")
@render()
async def list_proposals(request: Request, session_id: Optional[str] = None,
                         status: Optional[str] = None, limit: int = 100):
    """Candidate sources, in the expert's order where one was set."""
    service = _service()
    return {"proposals": service.list_proposals(session_id=session_id,
                                                status=status, limit=limit)}


@router.get("/proposals/{proposal_id}")
@render()
async def get_proposal(request: Request, proposal_id: str):
    return _service().get_proposal(proposal_id)


@router.post("/proposals")
@render()
async def create_proposal(request: Request, body: ProposalCreate):
    """Add a candidate by hand, rather than through the conversation."""
    service = _service()
    fields: Dict[str, Any] = body.model_dump(exclude_none=True)
    for key in ("user_sub", "session_id", "kind", "title"):
        fields.pop(key, None)
    return service.create_proposal(user_sub=body.user_sub, session_id=body.session_id,
                                   kind=body.kind, title=body.title, **fields)


@router.post("/proposals/{proposal_id}/approve")
@render()
async def approve(request: Request, proposal_id: str, body: ApproveRequest):
    """A person approves. The only way a proposal becomes integratable.

    Refuses a proposal whose licence is undetermined unless a reason is given,
    and records the reason alongside the approval.
    """
    return _service().approve(proposal_id=proposal_id, user_sub=body.user_sub,
                           override_reason=body.override_reason)


@router.post("/proposals/{proposal_id}/reject")
@render()
async def reject(request: Request, proposal_id: str, body: RejectRequest):
    return _service().reject(proposal_id=proposal_id, user_sub=body.user_sub,
                          reason=body.reason)


@router.post("/proposals/rerank")
@render()
async def rerank(request: Request, body: RerankRequest):
    """The expert's ordering, kept beside the agent's rather than over it."""
    service = _service()
    return {"proposals": service.rerank(order=body.order, user_sub=body.user_sub)}


@router.get("/backlog")
@render()
async def backlog(request: Request, kind: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 100, offset: int = 0):
    """The queue of candidate sources, seeded from the project's catalogue."""
    return _service().list_backlog(kind=kind, status=status, limit=limit, offset=offset)


@router.get("/audit")
@render()
async def audit(request: Request, user_sub: str,
                session_id: Optional[str] = None,
                proposal_id: Optional[str] = None,
                is_admin: bool = False,
                limit: int = Query(default=100, le=500)):
    """Every tool the agent ran, newest first — the caller's, unless admin.

    `user_sub` and `is_admin` come from the gateway, which took them from the
    token. They are required rather than optional so that a caller who is
    somehow not identified sees nothing instead of everything.
    """
    service = _service()
    return {"tool_calls": service.tool_calls(session_id=session_id,
                                             proposal_id=proposal_id, limit=limit,
                                             user_sub=user_sub, is_admin=is_admin)}


@router.post("/proposals/{proposal_id}/integrate")
@render()
async def integrate(request: Request, proposal_id: str, body: IntegrateRequest,
                    delegated: Optional[str] = Header(None, alias=DELEGATED_TOKEN_HEADER)):
    """Run an approved proposal into the catalog. Returns a run to poll.

    Approval and integration are separate on purpose. Approving says a person
    vouched for the source; integrating is when the catalog changes, and a
    curator may well want to approve a batch in one sitting and run them when
    somebody is around to watch.
    """
    from exceptions import APIException

    service = _service()
    try:
        return service.start_integration(
            proposal_id=proposal_id, user_sub=body.user_sub, dry_run=body.dry_run,
            access_token=_token(delegated))
    except LookupError as exc:
        raise APIException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise APIException(status_code=403, detail=str(exc)) from exc
    except service.RateLimited as exc:
        # Before RuntimeError, which it subclasses — the other order would
        # report every limit as a conflict.
        raise APIException(status_code=429, detail=str(exc),
                           headers={"Retry-After": str(exc.retry_after)}) from exc
    except RuntimeError as exc:
        # Already running. A second press should not start a second run.
        raise APIException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
@render()
async def run(request: Request, run_id: str):
    """One integration run, with its timeline. Poll this while it works."""
    from exceptions import APIException

    found = _service().get_run(run_id)
    if found is None:
        raise APIException(status_code=404, detail="no such run")
    return found


@router.get("/runs")
@render()
async def runs(request: Request, proposal_id: Optional[str] = None,
               limit: int = Query(default=20, le=100)):
    """Every attempt at a proposal, newest first — the failed ones included."""
    return {"runs": _service().list_runs(proposal_id=proposal_id, limit=limit)}
