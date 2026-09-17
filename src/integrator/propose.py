"""The tool that files a proposal.

Registered here rather than in the shared MCP package because filing a
proposal means *scoring* it, and the rubric lives in this service. The
standalone MCP process has no proposal queue for a curator to review, so it
has no use for the tool either.

This closes a gap that made the whole review surface unreachable: the agent
could research a source, establish its licence and describe in prose exactly
what it would propose — and then had no way to actually propose it. Every
conversation ended with a table of recommendations and an empty panel.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from wisefood_mcp.registry import ToolContext, ToolError

logger = logging.getLogger(__name__)

KINDS = ("guide", "article", "textbook", "fctable", "rcollection")


def propose_source(
    ctx: ToolContext,
    kind: str,
    title: str,
    source_url: Optional[str] = None,
    rationale: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    population_group: Optional[str] = None,
    licence: Optional[str] = None,
    licence_confidence: Optional[float] = None,
    licence_evidence: Optional[List[Dict[str, Any]]] = None,
    plan: Optional[List[str]] = None,
    existing_similar: Optional[int] = None,
) -> Dict[str, Any]:
    """File a candidate source for a curator to review in the console.

    This is the only way anything you find reaches a person. Call it once per
    source worth their attention — describing a proposal in your answer does
    not create one.

    Pass the licence exactly as `licence_evidence` established it, including
    its evidence, and leave it null when it could not be established. The
    rank is computed here from what the tools found, not taken from you.
    """
    from integrator.service import create_proposal

    if kind not in KINDS:
        raise ToolError(f"{kind!r} is not a kind of source",
                        allowed=list(KINDS))
    if not (title or "").strip():
        raise ToolError("a proposal needs a title")

    # The licence is the heaviest term in the rank and the thing a curator
    # decides on, so a claimed licence with nothing behind it is worse than
    # an honest gap: it scores well and reads as established.
    evidence = licence_evidence or []
    if licence and not evidence:
        raise ToolError(
            "a licence needs the evidence it came from — run licence_evidence "
            "on the source and pass what it returned, or leave the licence "
            "null and say it is undetermined",
            licence=licence)

    created = create_proposal(
        user_sub=ctx.actor or "",
        session_id=(ctx.extra or {}).get("session_id"),
        kind=kind,
        title=title.strip(),
        source_url=source_url,
        rationale=rationale,
        country=country,
        language=language,
        population_group=population_group,
        licence=licence,
        licence_confidence=licence_confidence,
        licence_evidence=evidence,
        plan=plan or [],
        existing_similar=existing_similar,
    )
    # Deliberately not the whole proposal: the model does not need its own
    # submission read back to it, and this result is re-sent on every
    # remaining step of the turn.
    return {
        "proposal_id": created["id"],
        "status": created["status"],
        "rank": created.get("proposed_rank"),
        "filed": f"{title.strip()} is now in the curator's review panel",
        "note": "a person approves it there; you cannot approve it yourself",
    }


def register(registry) -> None:
    """Add the tool to a registry.

    Not marked `write=True`. A write tool is one that changes the catalog,
    and those stay hidden until a deployment enables them. Filing a proposal
    changes nothing a reader can see — it is the request for permission, and
    gating it behind the same switch as the writes is what left the agent
    able to research and unable to say so.
    """
    registry.register(propose_source)
