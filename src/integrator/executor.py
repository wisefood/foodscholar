"""Integrating an approved proposal, one step at a time.

Phase 1 stopped at the proposal. This is what happens after a person presses
approve, and it is deliberately *not* the model's job. Three reasons, all of
which the chat loop fails on:

* **It is slow.** A guideline extraction over a 90-page PDF runs for minutes.
  A conversational turn has a step budget of twelve and a request that has to
  answer; neither survives a wait like that.
* **It has to resume.** Runs fail halfway — a download times out, a worker
  restarts. Resuming means knowing exactly which steps already landed, and
  repeating the ones that did not. That is a state machine, not a prompt.
* **It must be the same every time.** What gets written into a public-health
  catalog should not vary with a sampling temperature.

So the model proposes and a person approves; this executes. It executes
*through the same gated tools* rather than around them, which is what keeps
the guarantee honest: every step here goes through ``require_approved``, and
every step is written to the audit table by the registry, exactly as a
model-issued call would be.

The one thing the executor is allowed to decide is ordering, and it does not
decide that either — the order is fixed per kind, below.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

from wisefood_mcp.stores import content_permitted

from integrator.steps import StepTracker, finished_detail, running_title

logger = logging.getLogger(__name__)

#: Proposal kind -> the catalog tool that creates its entity.
CREATE_TOOL = {
    "guide": "create_guide",
    "article": "create_article",
    "textbook": "create_textbook",
}

#: Kinds whose artifact goes through the guideline extraction pipeline.
EXTRACTS = ("guide",)


class IntegrationError(RuntimeError):
    """A step failed in a way the run cannot continue past."""

    def __init__(self, message: str, *, detail: Any = None):
        super().__init__(message)
        self.detail = detail


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def guide_spec(proposal) -> Dict[str, Any]:
    """The catalog fields for a guide, from what the proposal established.

    A spec the assistant wrote wins, because it read the document and this
    has read a database row. What this fills in is the floor: a run should
    not fail because nobody typed a title that is sitting right there.
    """
    spec = dict((proposal.metadata or {}).get("spec") or {})
    spec.setdefault("title", proposal.title)
    if proposal.source_url:
        spec.setdefault("url", proposal.source_url)
    if proposal.country:
        spec.setdefault("region", proposal.country)
    if proposal.language:
        spec.setdefault("language", proposal.language)
    if proposal.rationale:
        spec.setdefault("description", proposal.rationale)
    return spec


def would_create_count(preview: Dict[str, Any]) -> int:
    """How many guidelines a dry run says it would create.

    Not ``total_created``: on a dry run the import creates nothing, so that
    field is 0 by definition and reading it would make every preview look
    like "nothing new to import". The plan is in the items, each marked
    ``would_create`` or ``skipped_existing``.
    """
    items = preview.get("items")
    if isinstance(items, list):
        return sum(1 for item in items
                   if (item or {}).get("status") == "would_create")
    # A caller that trimmed the items — fall back to the arithmetic.
    candidates = int(preview.get("total_candidates") or 0)
    return max(0, candidates - int(preview.get("total_skipped") or 0))


class Integration:
    """One run. Owns the order, the resume logic, and what it reports.

    Constructed with callbacks rather than a database handle so the whole
    thing is testable without one: `persist` is called after every step, and
    whatever it does with the snapshot is the host's business.
    """

    def __init__(self, *, proposal, registry, ctx, persist: Callable[[Dict[str, Any]], None],
                 poll_interval: float = 20.0, poll_timeout: float = 3600.0,
                 dry_run: bool = False, sleep: Callable[[float], None] = time.sleep):
        self.proposal = proposal
        self.registry = registry
        self.ctx = ctx
        self.persist = persist
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.dry_run = dry_run
        self.sleep = sleep
        self.steps = StepTracker()
        self.result: Dict[str, Any] = dict(proposal.result or {})
        self.wrote_anything = bool(self.result.get("urn"))
        self.stage: Optional[str] = None

    # --------------------------------------------------------------- plumbing --
    def _snapshot(self, status: str, *, error: Optional[str] = None,
                  finished: bool = False) -> Dict[str, Any]:
        return {
            "status": status, "stage": self.stage, "error": error,
            "steps": self.steps.snapshot(), "result": dict(self.result),
            "wrote_anything": self.wrote_anything, "finished": finished,
        }

    def _save(self, status: str = "running", **kw) -> None:
        try:
            self.persist(self._snapshot(status, **kw))
        except Exception:  # noqa: BLE001 — progress reporting must not kill a run
            logger.warning("integration run: progress not persisted", exc_info=True)

    def _call(self, tool: str, args: Dict[str, Any], *, stage: str,
              required: bool = True) -> Optional[Dict[str, Any]]:
        """Run one tool through the registry, narrating it as the chat does."""
        self.stage = stage
        kind, title, detail = running_title(tool, args)
        step = self.steps.start(kind, title, detail=detail, data={"tool": tool})
        self._save()

        outcome = self.registry.call(tool, json.dumps(args), self.ctx)
        ok = bool(outcome.get("ok"))
        self.steps.finish(step, ok=ok, outcome=finished_detail(
            tool, args, ok, outcome.get("result"), outcome.get("error")))
        self._save()

        if not ok:
            error = outcome.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            if required:
                raise IntegrationError(f"{tool}: {message}", detail=error)
            return None
        return outcome.get("result") or {}

    # ------------------------------------------------------------ the pipeline --
    def run(self) -> Dict[str, Any]:
        try:
            self._preflight()
            urn = self._create_entity()
            artifact_uuid = self._attach_file(urn)
            if artifact_uuid and self.proposal.kind in EXTRACTS:
                self._extract(urn, artifact_uuid)
                self._import(urn, artifact_uuid)
            self.stage = "done"
            self.steps.add("done", "Integration complete",
                           outcome=self._closing_line())
            self._save("succeeded", finished=True)
            return self._snapshot("succeeded", finished=True)
        except IntegrationError as exc:
            self.steps.add("stop", "Stopped", outcome=str(exc))
            self._save("failed", error=str(exc), finished=True)
            return self._snapshot("failed", error=str(exc), finished=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("integration run failed unexpectedly")
            message = f"{type(exc).__name__}: {exc}"[:500]
            self.steps.add("stop", "Stopped", outcome=message)
            self._save("failed", error=message, finished=True)
            return self._snapshot("failed", error=message, finished=True)

    def _closing_line(self) -> str:
        created = self.result.get("guidelines_created")
        if created is not None:
            return f"{created} guideline{'' if created == 1 else 's'} imported under {self.result.get('urn')}"
        if self.result.get("artifact_id"):
            return f"{self.result.get('urn')} created, with its file attached"
        return f"{self.result.get('urn')} registered"

    def _preflight(self) -> None:
        """Refuse before writing anything, not halfway through.

        A run that creates a guide and then discovers it has no PDF to attach
        leaves a hollow entry in the catalog that somebody has to notice and
        delete. Everything knowable up front is checked up front.
        """
        self.stage = "preflight"
        if self.proposal.status != "approved":
            raise IntegrationError("this proposal has not been approved")
        if self.ctx.data_client is None:
            raise IntegrationError(
                "this run has no catalog access as you — the session token was "
                "not forwarded or has expired. Sign in again and retry; nothing "
                "was written.")
        if self.proposal.kind not in CREATE_TOOL:
            raise IntegrationError(
                f"nothing is wired to integrate a {self.proposal.kind!r} yet; "
                f"guides, articles and textbooks are")

        handle = (self.proposal.metadata or {}).get("pending_artifact")
        wants_file = content_permitted(self.proposal)
        if self.proposal.kind in EXTRACTS and wants_file and not handle:
            raise IntegrationError(
                "no fetched file to extract from — open the source PDF with "
                "fetch_url first, so its handle is on the proposal")
        if not wants_file:
            self.steps.add(
                "licence", "Registering a pointer only",
                detail=self.proposal.licence,
                outcome=("this licence does not permit copying the content in, "
                         "so the catalog gets the reference and not the document"))

    def _create_entity(self) -> str:
        if self.result.get("urn"):
            self.steps.add("catalog", "Entity already created",
                           outcome=f"reusing {self.result['urn']} from an earlier attempt")
            return self.result["urn"]
        tool = CREATE_TOOL[self.proposal.kind]
        spec = guide_spec(self.proposal)
        created = self._call(tool, {"proposal_id": self.proposal.id, "spec": spec},
                             stage="create")
        urn = (created or {}).get("urn")
        if not urn:
            raise IntegrationError(f"{tool} returned no urn")
        self.result["urn"] = urn
        self.wrote_anything = True
        self._save()
        return urn

    def _attach_file(self, urn: str) -> Optional[str]:
        handle = (self.proposal.metadata or {}).get("pending_artifact")
        if not handle or not content_permitted(self.proposal):
            return None
        # The key is `artifact_id` because that is what `upload_artifact`
        # writes onto the proposal. Reading a different name here is how a
        # retry quietly attaches a second copy of the same PDF.
        if self.result.get("artifact_id"):
            self.steps.add("catalog", "File already attached",
                           outcome=f"reusing artifact {self.result['artifact_id']}")
            return self.result["artifact_id"]
        uploaded = self._call("upload_artifact", {
            "proposal_id": self.proposal.id, "parent_urn": urn,
            "pending_artifact": handle, "title": self.proposal.title,
        }, stage="upload")
        artifact_uuid = (uploaded or {}).get("artifact_id")
        if not artifact_uuid:
            raise IntegrationError("upload_artifact returned no artifact id")
        self.result["artifact_id"] = artifact_uuid
        self._save()
        return artifact_uuid

    def _extract(self, urn: str, artifact_uuid: str) -> None:
        """Queue the extraction, then wait on it, saying where it has got to."""
        self._call("enqueue_guideline_extraction", {
            "proposal_id": self.proposal.id, "artifact_uuid": artifact_uuid,
            "guide_urn": urn,
        }, stage="extract")

        self.stage = "extracting"
        step = self.steps.start("read", "Reading the document",
                                detail="extracting guidelines page by page")
        deadline = time.monotonic() + self.poll_timeout
        last_seen = None
        while True:
            self.sleep(self.poll_interval)
            state = self._call("guideline_extraction_status", {
                "proposal_id": self.proposal.id, "artifact_uuid": artifact_uuid,
            }, stage="extracting", required=False) or {}
            status = state.get("status")
            pages, total = state.get("current_page"), state.get("total_pages")
            if pages and total:
                # The detail, not the outcome: the run is still going, and a
                # reader watching it wants the page counter to move.
                step["detail"] = f"page {pages} of {total}"
                last_seen = f"{pages}/{total}"
            self.result["extraction"] = {k: state.get(k) for k in
                                         ("status", "current_page", "total_pages")}
            self._save()

            if status == "succeeded":
                found = state.get("guideline_count") or 0
                self.steps.finish(step, outcome=f"{found} guideline{'' if found == 1 else 's'} found")
                self.result["guidelines_extracted"] = found
                if not found:
                    raise IntegrationError(
                        "the extraction finished but found no guidelines; there is "
                        "nothing to import, and the guide is still there to inspect")
                return
            if status == "failed":
                self.steps.finish(step, ok=False,
                                  outcome=state.get("error") or "extraction failed")
                raise IntegrationError(
                    f"extraction failed: {state.get('error') or 'no reason given'}")
            if status == "not_found":
                self.steps.finish(step, ok=False, outcome="the job disappeared")
                raise IntegrationError(
                    "the extraction job is no longer known to the queue; it may have "
                    "been dropped by a restart. Retrying this run will re-queue it.")
            if time.monotonic() > deadline:
                self.steps.finish(step, ok=False, outcome=(
                    f"still running after {int(self.poll_timeout / 60)} minutes"
                    + (f" (last seen at page {last_seen})" if last_seen else "")))
                raise IntegrationError(
                    "the extraction is taking longer than this run waits for. It is "
                    "still going; retry this run to pick it up when it finishes.")

    def _import(self, urn: str, artifact_uuid: str) -> None:
        """Preview first, always, then import — unless this run is a preview.

        The dry run is not ceremony. It is the only place that reports how
        many extracted rules are actually new, and a preview showing zero is
        the signal that this document is already in the catalog.
        """
        preview = self._call("import_guidelines", {
            "proposal_id": self.proposal.id, "artifact_uuid": artifact_uuid,
            "guide_urn": urn, "dry_run": True,
        }, stage="preview") or {}
        would_create = would_create_count(preview)
        self.result["preview"] = {
            "candidates": preview.get("total_candidates"),
            "would_create": would_create,
            "would_skip": preview.get("total_skipped"),
        }
        self._save()
        if not would_create:
            skipped = int(preview.get("total_skipped") or 0)
            raise IntegrationError(
                f"nothing new to import — all {skipped} extracted rule"
                f"{'' if skipped == 1 else 's'} already exist on this guide"
                if skipped else "the extraction produced nothing importable")

        if self.dry_run:
            self.steps.add("done", "Preview only",
                           outcome=f"{would_create} guidelines would be created; "
                                   f"nothing was written")
            return

        imported = self._call("import_guidelines", {
            "proposal_id": self.proposal.id, "artifact_uuid": artifact_uuid,
            "guide_urn": urn, "dry_run": False,
        }, stage="import") or {}
        self.result["guidelines_created"] = imported.get("total_created")
        self.result["guidelines_skipped"] = imported.get("total_skipped")
        self._save()
