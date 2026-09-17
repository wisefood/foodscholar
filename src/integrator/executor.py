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
    "fctable": "create_fctable",
}

#: Kinds whose artifact goes through the guideline extraction pipeline.
EXTRACTS = ("guide",)

#: Kinds whose artifact is chunked into retrievable passages instead.
CHUNKS = ("textbook",)

#: Kinds whose file is *described* rather than ingested. A food composition
#: table is registered as a reference; the catalog has no row store to put one
#: in, so what the file yields is metadata about itself.
PROFILES = ("fctable",)


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


#: Crossref field -> catalog article field. Only the ones that mean the same
#: thing; anything needing judgement (topics, population group, study type) is
#: the enrichment's job, not a rename.
_CROSSREF_TO_ARTICLE = {
    "title": "title", "authors": "authors", "venue": "venue",
    "publication_year": "publication_year", "abstract": "abstract",
    "language": "language", "doi": "doi", "url": "url",
    "citation_count": "citation_count", "reference_count": "reference_count",
}


def article_spec(proposal) -> Dict[str, Any]:
    """The catalog fields for an article, Crossref first.

    Deliberately the opposite precedence from a free-text source: what the
    publisher deposited beats what an assistant wrote, because the one thing
    that must not happen to a scientific catalog is a citation with invented
    authors that reads perfectly well. An assistant's spec fills what Crossref
    has no field for, and only that.
    """
    metadata = proposal.metadata or {}
    spec = dict(metadata.get("spec") or {})
    record = metadata.get("doi_metadata") or {}

    if record.get("found"):
        for source, target in _CROSSREF_TO_ARTICLE.items():
            value = record.get(source)
            if value not in (None, "", [], {}):
                spec[target] = value
        if record.get("subjects") and not spec.get("keywords"):
            spec["keywords"] = list(record["subjects"])

    spec.setdefault("title", proposal.title)
    if proposal.source_url:
        spec.setdefault("url", proposal.source_url)
    if proposal.language:
        spec.setdefault("language", proposal.language)
    if proposal.population_group:
        spec.setdefault("population_group", proposal.population_group)
    if proposal.rationale and not spec.get("description"):
        spec["description"] = proposal.rationale
    doi = metadata.get("doi") or record.get("doi")
    if doi:
        spec.setdefault("doi", doi)
    return spec


#: What a table profile contributes to an `FCTable`. The rest of what the
#: profiler returns — the column names it judged from — is evidence for a
#: curator, not catalog metadata.
_PROFILE_TO_FCTABLE = (
    "number_of_entries", "nutrient_coverage", "completeness_percent",
    "completeness_description", "min_nutrients_per_item",
    "max_nutrients_per_item", "measurement_units", "reference_portions",
)


def fctable_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The measured fields, skipping anything the profiler could not tell."""
    return {key: profile[key] for key in _PROFILE_TO_FCTABLE
            if profile.get(key) not in (None, "", [], {})}


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
            # A recipe collection is harvested, not created and filled: there
            # is no document to attach and no entity to hang it from. It has
            # its own short path.
            if self.proposal.kind == "rcollection":
                self._harvest()
                self.stage = "done"
                self.steps.add("done", "Integration complete",
                               outcome=self._closing_line())
                self._save("succeeded", finished=True)
                return self._snapshot("succeeded", finished=True)
            if self.proposal.kind in PROFILES:
                self._profile()
            urn = self._create_entity()
            artifact_uuid = self._attach_file(urn)
            if artifact_uuid and self.proposal.kind in EXTRACTS:
                self._extract(urn, artifact_uuid)
                self._import(urn, artifact_uuid)
            elif artifact_uuid and self.proposal.kind in CHUNKS:
                self._chunk(urn, artifact_uuid)
            elif self.proposal.kind == "article":
                self._enrich(urn)
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
        harvest = self.result.get("harvest") or {}
        if harvest:
            return (f"{harvest.get('written') or 0} recipes imported, "
                    f"{harvest.get('skipped') or 0} skipped")
        dry = self.result.get("dry_run_harvest")
        if dry:
            return f"{dry.get('found') or 0} recipes found, none written"
        created = self.result.get("guidelines_created")
        if created is not None:
            return f"{created} guideline{'' if created == 1 else 's'} imported under {self.result.get('urn')}"
        if self.result.get("enrichment", {}).get("status") == "succeeded":
            return f"{self.result.get('urn')} created and enriched"
        entries = (self.result.get("profile") or {}).get("number_of_entries")
        if entries:
            return (f"{entries:,} entries profiled and registered as "
                    f"{self.result.get('urn')}")
        passages = self.result.get("passages")
        if passages:
            return (f"{passages} passage{'' if passages == 1 else 's'} from "
                    f"{self.result.get('page_count')} pages, under {self.result.get('urn')}")
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
        if self.proposal.kind not in CREATE_TOOL and self.proposal.kind != "rcollection":
            raise IntegrationError(
                f"nothing is wired to integrate a {self.proposal.kind!r} yet; "
                f"guides, articles, textbooks, composition tables and recipe "
                f"collections are")

        if self.proposal.kind == "article":
            self._resolve_the_citation()
            self._refuse_a_duplicate_doi()

        handle = (self.proposal.metadata or {}).get("pending_artifact")
        wants_file = content_permitted(self.proposal)
        if self.proposal.kind in (*EXTRACTS, *CHUNKS) and wants_file and not handle:
            raise IntegrationError(
                "no fetched file to read — open the source PDF with fetch_url "
                "first, so its handle is on the proposal")
        if not wants_file:
            self.steps.add(
                "licence", "Registering a pointer only",
                detail=self.proposal.licence,
                outcome=("this licence does not permit copying the content in, "
                         "so the catalog gets the reference and not the document"))

    def _resolve_the_citation(self) -> None:
        """Read the publisher's own record, here, before anything is created.

        This used to be the assistant's job at propose time, and the spec
        below preferred the record it was supposed to have stashed. Nothing
        ever stashed one — so the branch never fired and articles were
        created from whatever the assistant typed, which is precisely the
        failure the precedence was written to prevent.

        Doing it in the run fixes that and costs the conversation nothing. A
        citation is checked once, at the only moment it matters, instead of
        being checked for every candidate the assistant merely considered.
        The guarantee gets stronger for it: it no longer depends on the
        assistant having been diligent.
        """
        metadata = self.proposal.metadata or {}
        if (metadata.get("doi_metadata") or {}).get("found"):
            return
        doi = metadata.get("doi") or (metadata.get("spec") or {}).get("doi")
        if not doi:
            # An article with no DOI is proposed on its URL alone. Legitimate
            # for a report or a preprint; there is simply no record to read.
            self.steps.add("read", "No DOI to look up",
                           outcome="the entry is built from the proposal and its source")
            return

        record = self._call("doi_metadata", {"doi": doi},
                            stage="preflight", required=False) or {}
        if not record.get("found"):
            raise IntegrationError(
                f"Crossref has no record of {doi}, so there is no authoritative "
                f"citation to create this article from. Check the DOI on the "
                f"proposal — nothing was created.")

        # In memory for this run, which is all `article_spec` below needs.
        # A retry reads it again — one cheap call, and a fresher record than
        # one cached from a previous attempt.
        self.proposal.metadata = {**metadata, "doi_metadata": record}
        self.steps.add("read", "Citation confirmed with Crossref",
                       detail=doi,
                       outcome=f"{record.get('title') or 'the record'} — "
                               f"{record.get('publisher') or 'publisher unknown'}")

    def _refuse_a_duplicate_doi(self) -> None:
        """A DOI names one paper, so importing it twice is never right.

        Best effort, and it says so: the search is the catalog's own and is
        fuzzy, so this catches the duplicate it finds and does not promise
        there is no other. When it does fire it is certain — an exact DOI
        match is the same work, whatever the titles look like.
        """
        metadata = self.proposal.metadata or {}
        doi = (metadata.get("doi")
               or (metadata.get("doi_metadata") or {}).get("doi"))
        if not doi:
            return
        found = self._call("search_catalog",
                           {"kind": "article", "q": doi, "limit": 10},
                           stage="preflight", required=False) or {}
        for item in found.get("items") or []:
            if str(item.get("doi") or "").strip().lower() == str(doi).strip().lower():
                raise IntegrationError(
                    f"the catalog already holds {doi} as {item.get('urn') or 'an article'}; "
                    f"nothing was created. Reject this proposal, or open that entry "
                    f"if it needs updating.")

    def _harvest(self) -> None:
        """Read a recipe site into the corpus, dry first.

        The dry run is not a formality and is not skipped on the strength of
        the profile: `recipe_source` sampled a handful of pages, and this
        reads all of them. A source that looked fine in five pages and turns
        out to carry markup on a tenth of them is worth discovering without
        having written anything.

        On a real run the dry pass is skipped — the curator has already seen
        it — but `dry_run` at the top level still means "show me and write
        nothing", which is the preview button in the console.
        """
        metadata = self.proposal.metadata or {}
        location = (metadata.get("harvest_location")
                    or (metadata.get("spec") or {}).get("harvest_location")
                    or self.proposal.source_url)
        if not location:
            raise IntegrationError(
                "no sitemap or feed to harvest — run recipe_source on the site "
                "and put its harvest_location on the proposal")

        args = {
            "proposal_id": self.proposal.id,
            "location": location,
            "region": (self.proposal.country or "IE"),
            "limit": int(metadata.get("limit") or 200),
            "dry_run": True,
        }
        if metadata.get("include"):
            args["include"] = metadata["include"]

        started = self._call("import_recipe_source", args, stage="harvest") or {}
        dry = self._await_harvest(started.get("run_id"), "Reading the site")
        self.result["dry_run_harvest"] = dry

        found, written = dry.get("found") or 0, dry.get("written") or 0
        if not found:
            raise IntegrationError(
                f"the dry run read {location} and found no recipes to import; "
                f"nothing was written. Check the sitemap, or reject this "
                f"proposal.")

        if self.dry_run:
            self.steps.add("done", "Preview only",
                           outcome=f"{found} recipes found, none written")
            return

        args["dry_run"] = False
        real = self._call("import_recipe_source", args, stage="harvest") or {}
        outcome = self._await_harvest(real.get("run_id"), "Importing the recipes")
        self.result["harvest"] = outcome
        self.wrote_anything = bool(outcome.get("written"))
        if outcome.get("status") == "failed":
            raise IntegrationError(
                f"the import failed after writing {outcome.get('written') or 0} "
                f"recipes: {outcome.get('error') or 'no reason given'}")

    def _await_harvest(self, run_id, title: str) -> Dict[str, Any]:
        """Poll one import to its end, saying how far it has got."""
        if not run_id:
            raise IntegrationError("the recipe importer started no run")
        step = self.steps.start("read", title, detail=str(run_id))
        deadline = time.monotonic() + self.poll_timeout
        state: Dict[str, Any] = {}
        while True:
            self.sleep(self.poll_interval)
            state = self._call("recipe_import_status", {"run_id": run_id},
                               stage="harvesting", required=False) or {}
            found, written = state.get("found"), state.get("written")
            if found is not None:
                step["detail"] = (f"{written or 0} written of {found} found"
                                  if written is not None else f"{found} found")
            self._save("running")
            if state.get("finished"):
                break
            if time.monotonic() > deadline:
                raise IntegrationError(
                    f"the recipe import did not finish within "
                    f"{int(self.poll_timeout // 60)} minutes; run {run_id} may "
                    f"still be going — check it before starting another.")
        self.steps.finish(step, ok=state.get("status") != "failed", outcome=(
            f"{state.get('written') or 0} written, {state.get('skipped') or 0} "
            f"skipped, {state.get('failed') or 0} failed"))
        return state

    def _create_entity(self) -> str:
        if self.result.get("urn"):
            self.steps.add("catalog", "Entity already created",
                           outcome=f"reusing {self.result['urn']} from an earlier attempt")
            return self.result["urn"]
        tool = CREATE_TOOL[self.proposal.kind]
        spec = (article_spec if self.proposal.kind == "article" else guide_spec)(
            self.proposal)
        if self.proposal.kind in PROFILES and getattr(self, "_profiled", None):
            spec.update(fctable_fields(self._profiled))
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

    def _profile(self) -> None:
        """Describe the table before the entity is created, not after.

        The profile *is* most of an FCT's metadata — entries, nutrient
        coverage, completeness — so creating the entity first would mean
        writing a record everyone can see and then correcting it.
        """
        handle = (self.proposal.metadata or {}).get("pending_artifact")
        if not handle:
            # Not fatal. A table we may only point at still deserves its entry,
            # and a curator can fill the counts by hand as they do today.
            self.steps.add("catalog", "No table file to read",
                           outcome="registering what the proposal already knows")
            return
        profile = self._call("profile_fctable", {
            "proposal_id": self.proposal.id, "pending_artifact": handle,
        }, stage="profile") or {}
        self.result["profile"] = {
            k: profile.get(k) for k in
            ("number_of_entries", "nutrient_coverage", "completeness_percent")
        }
        self._profiled = profile
        self._save()

    def _chunk(self, urn: str, artifact_uuid: str) -> None:
        """Read the textbook into passages. Inline, because it is fast.

        No model, so no queue: a guide's rules must be understood to be
        extracted, whereas a passage is a span of the book with enough context
        to be retrieved. Seconds rather than minutes, and the run simply waits.
        """
        result = self._call("extract_textbook_passages", {
            "proposal_id": self.proposal.id, "textbook_urn": urn,
            "artifact_uuid": artifact_uuid,
        }, stage="chunk") or {}
        self.result["passages"] = result.get("passages")
        self.result["page_count"] = result.get("page_count")
        self.result["headings_found"] = result.get("headings_found")
        self._save()

    def _enrich(self, urn: str) -> None:
        """Queue the article's enrichment and wait on it.

        Unlike a guide's extraction, this is not what makes the entry worth
        having — the article is a usable catalog record the moment it exists.
        It is still waited on and still fails loudly, because an article that
        was never enriched is one the search will not surface well, and a
        silent half-integration is worse than a red one somebody can retry.
        """
        self._call("enqueue_article_enrichment", {
            "proposal_id": self.proposal.id, "article_urn": urn,
        }, stage="enrich")

        self.stage = "enriching"
        step = self.steps.start("read", "Enriching the article",
                                detail="keywords, study type, glossary and Q&A")
        deadline = time.monotonic() + self.poll_timeout
        while True:
            self.sleep(self.poll_interval)
            state = self._call("article_enrichment_status", {
                "proposal_id": self.proposal.id, "article_urn": urn,
            }, stage="enriching", required=False) or {}
            status = state.get("status")
            self.result["enrichment"] = {"status": status,
                                         "wrote": state.get("wrote")}
            self._save()

            if status == "succeeded":
                wrote = state.get("wrote") or []
                self.steps.finish(step, outcome=(
                    f"enriched — {', '.join(wrote)}" if wrote else "enriched"))
                return
            if status == "failed" or state.get("permanently_failed"):
                self.steps.finish(step, ok=False,
                                  outcome=state.get("error") or "enrichment failed")
                raise IntegrationError(
                    f"enrichment failed: {state.get('error') or 'no reason given'}. "
                    f"The article is in the catalog and can be enriched again.")
            if status == "not_found":
                self.steps.finish(step, ok=False, outcome="the job disappeared")
                raise IntegrationError(
                    "the enrichment job is no longer known to the queue; retrying "
                    "this run will re-queue it. The article is already there.")
            if time.monotonic() > deadline:
                self.steps.finish(step, ok=False, outcome=(
                    f"still running after {int(self.poll_timeout / 60)} minutes"))
                raise IntegrationError(
                    "the enrichment is taking longer than this run waits for. The "
                    "article is in the catalog; retry to pick the enrichment up.")

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
