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
    doi: Optional[str] = None,
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

    File it as soon as it looks credible. A proposal is a suggestion, not an
    entry: a curator reads it, and the integration run does the authoritative
    work when they approve. For an article, pass the `doi` and the run reads
    the publisher's record from Crossref itself — so you do not need to look
    a citation up before proposing it, and you should not type one out of a
    search result either. Give the title as you found it and let the record
    correct it.

    Pass the licence only if `licence_evidence` established one, with its
    evidence. Leave it null otherwise: undetermined is an ordinary outcome
    and the curator is asked for a reason at approval. The rank is computed
    here from what is known, not taken from you.

    :param doi: for an article — the run resolves the citation from it
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

    metadata = {"doi": doi.strip()} if doi and doi.strip() else {}
    created = create_proposal(
        user_sub=ctx.actor or "",
        session_id=(ctx.extra or {}).get("session_id"),
        kind=kind,
        title=title.strip(),
        source_url=source_url,
        metadata=metadata,
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
        "note": ("a person approves it there; you cannot approve it yourself. "
                 "Move on to the next candidate — this one is recorded."),
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
