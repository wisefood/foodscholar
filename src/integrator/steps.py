"""Saying what the assistant is doing, in words a curator can check.

The same shape FoodScholar's Q&A already streams — id, kind, status, title,
detail, elapsed, data — so the console renders both with one component. The
*vocabulary* is the integrator's own, because `ReasoningStep.kind` is a closed
list belonging to the Q&A pipeline and widening it would be changing somebody
else's contract for our convenience.

Why this exists at all: a tool badge reading `research` tells a curator that
something happened. "Searched the web for *Bulgaria dietary guidelines adults*
— 6 results" tells them what happened, and lets them notice that the assistant
searched for the wrong thing. That difference is the whole point of the
feature; an assistant that integrates sources into a public-health catalog has
to be checkable, not merely fast.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {many or one + 's'}"


#: Tool name -> (kind, what we say while it runs). The kinds are deliberately
#: few: a curator scanning a timeline wants to see shapes, not a taxonomy.
RUNNING: Dict[str, tuple] = {
    "research": ("search", "Searching the web"),
    "propose_source": ("propose", "Filing a proposal"),
    "fetch_url": ("read", "Opening the page"),
    "licence_evidence": ("licence", "Checking the licence"),
    "doi_metadata": ("read", "Looking up the DOI"),
    "journal_articles": ("search", "Listing what the journal has published"),
    "recipe_source": ("search", "Profiling the recipe site"),
    "import_recipe_source": ("write", "Harvesting the recipes"),
    "recipe_import_status": ("read", "Checking on the harvest"),
    "infer_guidelines": ("read", "Reading the rules out of the source"),
    "search_catalog": ("catalog", "Searching the catalog"),
    "catalog_coverage": ("catalog", "Checking what we already hold"),
    "get_entity": ("catalog", "Reading a catalog entry"),
    "list_organizations": ("catalog", "Looking up organisations"),
    "create_guide": ("write", "Creating a guide"),
    "create_article": ("write", "Creating an article"),
    "create_textbook": ("write", "Creating a textbook"),
    "create_fctable": ("write", "Registering the composition table"),
    "upload_artifact": ("write", "Attaching the file"),
    "enqueue_guideline_extraction": ("write", "Queuing the extraction"),
    "guideline_extraction_status": ("read", "Checking on the extraction"),
    "import_guidelines": ("write", "Importing the guidelines"),
    "extract_textbook_passages": ("write", "Reading the textbook into passages"),
    "profile_fctable": ("read", "Reading the composition table"),
    "enqueue_article_enrichment": ("write", "Queuing the enrichment"),
    "article_enrichment_status": ("read", "Checking on the enrichment"),
}


def running_title(tool: str, args: Dict[str, Any]) -> tuple:
    """What to show the moment a tool starts, before its result exists."""
    kind, title = RUNNING.get(tool, ("tool", f"Running {tool}"))
    detail = None
    if tool == "research":
        detail = str(args.get("query") or "")[:200] or None
    elif tool in ("fetch_url",):
        detail = str(args.get("url") or "")[:200] or None
    elif tool in ("licence_evidence", "doi_metadata"):
        detail = str(args.get("url") or args.get("doi") or "")[:200] or None
    elif tool in ("enqueue_article_enrichment", "article_enrichment_status"):
        detail = str(args.get("article_urn") or "")[:200] or None
    elif tool in ("search_catalog", "catalog_coverage"):
        bits = [str(args.get(k)) for k in ("q", "country", "population_group", "language")
                if args.get(k)]
        detail = " · ".join(bits)[:200] or None
    elif tool in ("create_guide", "create_article", "create_textbook",
                  "create_fctable"):
        detail = str((args.get("spec") or {}).get("title") or "")[:200] or None
    elif tool == "upload_artifact":
        detail = str(args.get("title") or args.get("parent_urn") or "")[:200] or None
    elif tool == "import_guidelines":
        detail = "preview" if args.get("dry_run") else "for real"
    return kind, title, detail


def finished_detail(tool: str, args: Dict[str, Any], ok: bool,
                    result: Any, error: Any) -> str:
    """What the step says once it is done — the *outcome*, not the call.

    This is the sentence a curator actually reads, so it names numbers and
    subjects rather than restating the tool. A failure says what failed in
    plain words, because a red badge with no reason is worse than no badge.
    """
    if not ok:
        message = (error or {}).get("message") if isinstance(error, dict) else None
        return f"Could not: {message or 'the step failed'}"[:300]

    data = result if isinstance(result, dict) else {}

    if tool == "research":
        findings = data.get("findings") or []
        used = ", ".join(data.get("tools_used") or []) or "search"
        return f"{_plural(len(findings), 'result')} via {used}"

    if tool == "fetch_url":
        if not data.get("fetched"):
            return f"Could not open it: {data.get('reason', 'no reason given')}"[:300]
        if data.get("kind") == "pdf":
            pages = data.get("pages")
            return f"PDF{f', {_plural(int(pages), 'page')}' if pages else ''} — ready to attach"
        chars = data.get("chars") or 0
        licences = len(data.get("licence_links") or [])
        tail = f", {_plural(licences, 'licence link')}" if licences else ""
        return f"Read {chars:,} characters{tail}"

    if tool == "licence_evidence":
        licence = data.get("proposed_licence")
        if not licence:
            return "No licence statement found — this cannot be approved without a reason"
        confidence = data.get("confidence")
        permitted = str(data.get("content_ingestion") or "")
        pct = f", {round(float(confidence) * 100)}% confident" if confidence else ""
        return f"Looks like {licence}{pct} — content {permitted.split(' until')[0]}"

    if tool in ("search_catalog", "catalog_coverage"):
        count = data.get("count")
        if count == 0:
            return "Nothing in the catalog matches — this would fill a gap"
        return f"{_plural(int(count or 0), 'match', 'matches')} already in the catalog"

    if tool == "propose_source":
        # Says where it went, because the panel is on the same screen and the
        # curator's next move is to look at it.
        return data.get("filed") or "Filed for review"

    if tool == "get_entity":
        return f"Read {data.get('title') or data.get('urn') or 'the entry'}"

    if tool == "list_organizations":
        return _plural(int(data.get("count") or 0), "organisation")

    if tool == "infer_guidelines":
        found = int(data.get("count") or 0)
        dropped = int(data.get("unverifiable_dropped") or 0)
        if not found:
            return "No rules stated in that text"
        shape = ("already listed as recommendations"
                 if data.get("already_aggregated") else "assembled from prose")
        tail = f", {dropped} dropped for unverifiable quotes" if dropped else ""
        return f"{_plural(found, 'rule')} — {shape}{tail}"

    if tool == "recipe_source":
        if not data.get("harvestable"):
            return (data.get("reason")
                    or f"no machine-readable recipes in {data.get('sampled') or 0} sampled pages")
        share = int(round(float(data.get("markup_share") or 0) * 100))
        estimate = data.get("estimated_recipes")
        return (f"{share}% of sampled pages carry recipe markup"
                + (f" — about {estimate:,} recipes" if estimate else ""))

    if tool == "import_recipe_source":
        where = data.get("location") or "the source"
        return (f"Dry run started on {where} — nothing will be written"
                if data.get("dry_run") else f"Import started on {where}")

    if tool == "recipe_import_status":
        if not data.get("finished"):
            found, written = data.get("found"), data.get("written")
            if found is None:
                return "Still going"
            return (f"{written or 0} written of {found} found"
                    if written is not None else f"{found} found so far")
        if data.get("status") == "failed":
            return data.get("error") or "The import failed"
        return (f"{data.get('written') or 0} written, "
                f"{data.get('skipped') or 0} skipped, "
                f"{data.get('failed') or 0} failed")

    if tool == "journal_articles":
        if not data.get("found"):
            return "More than one journal could be meant — needs an ISSN"
        n = int(data.get("count") or 0)
        fresh = data.get("new_to_the_catalog")
        tail = (f", {n - fresh} already held" if fresh is not None and n - fresh else "")
        return f"{_plural(n, 'article')} from {data.get('journal') or 'the journal'}{tail}"

    if tool == "doi_metadata":
        if not data.get("found"):
            return "Crossref has no record of that DOI"
        authors = data.get("authors") or []
        who = authors[0] if authors else "unknown author"
        if len(authors) > 1:
            who += f" and {_plural(len(authors) - 1, 'other')}"
        year = data.get("publication_year")
        venue = data.get("venue")
        bits = [who, str(year) if year else "", venue or ""]
        return " · ".join(b for b in bits if b)[:300]

    if tool == "profile_fctable":
        entries = int(data.get("number_of_entries") or 0)
        nutrients = data.get("nutrient_coverage") or []
        complete = data.get("completeness_percent")
        bits = [f"{entries:,} entries", _plural(len(nutrients), "nutrient")]
        if complete is not None:
            bits.append(f"{complete}% filled")
        return " · ".join(bits)

    if tool == "extract_textbook_passages":
        pages = data.get("page_count")
        headings = data.get("headings_found")
        bits = [_plural(int(data.get("passages") or 0), "passage")]
        if pages:
            bits.append(f"{_plural(int(pages), 'page')}")
        if headings:
            bits.append(f"{_plural(int(headings), 'heading')} found")
        return " · ".join(bits)

    if tool == "enqueue_article_enrichment":
        return f"Queued — {data.get('status') or 'waiting for a worker'}"

    if tool == "article_enrichment_status":
        wrote = data.get("wrote") or []
        tail = f" ({', '.join(wrote)})" if wrote else ""
        return f"{data.get('status') or 'unknown'}{tail}"[:300]

    if tool == "upload_artifact":
        return f"Attached as artifact {data.get('artifact_id')}"

    if tool == "enqueue_guideline_extraction":
        return f"Queued — {data.get('status') or 'waiting for a worker'}"

    if tool == "guideline_extraction_status":
        pages, total = data.get("current_page"), data.get("total_pages")
        where = f" at page {pages} of {total}" if pages and total else ""
        return f"{data.get('status') or 'unknown'}{where}"

    if tool == "import_guidelines":
        skipped = int(data.get("total_skipped") or 0)
        tail = f", {skipped} already there" if skipped else ""
        if data.get("dry_run"):
            from integrator.executor import would_create_count
            return f"would create {_plural(would_create_count(data), 'guideline')}{tail}"
        created = int(data.get("total_created") or 0)
        return f"Created {_plural(created, 'guideline')}{tail}"

    if data.get("urn"):
        return f"Created {data['urn']}"
    return "Done"


class StepTracker:
    """The timeline, built as the turn runs.

    Steps are started and finished rather than appended when complete, so a
    streaming client can show "Searching the web…" while it is happening. The
    non-streaming path gets the same list at the end; nothing is lost either
    way, which is why the shape is the same for both.
    """

    def __init__(self) -> None:
        self._steps: List[Dict[str, Any]] = []
        self._started: Dict[str, float] = {}
        self._counter = 0

    def start(self, kind: str, title: str, *, detail: Optional[str] = None,
              data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._counter += 1
        step = {
            "id": f"step-{self._counter}", "kind": kind, "status": "running",
            #: What was attempted — the query, the URL. Survives failure.
            "title": title, "detail": detail,
            #: What happened. Filled in by `finish`.
            "outcome": None, "elapsed_ms": None,
            "data": data or {},
        }
        self._steps.append(step)
        self._started[step["id"]] = time.monotonic()
        return step

    def finish(self, step: Dict[str, Any], *, outcome: Optional[str] = None,
               ok: bool = True, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Close a step with what happened.

        `detail` is left alone: it holds what was *attempted* — the query, the
        URL — and that is precisely what a curator needs when a step fails.
        Overwriting it with the error loses the one fact that makes the
        failure checkable, so the outcome gets its own field.
        """
        step["status"] = "done"
        step["ok"] = ok
        if outcome is not None:
            step["outcome"] = outcome
        if data:
            step["data"] = {**step["data"], **data}
        started = self._started.get(step["id"])
        if started is not None:
            step["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        return step

    def add(self, kind: str, title: str, *, detail: Optional[str] = None,
            outcome: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """A step that is born finished — a summary, a verdict, a stop."""
        step = self.start(kind, title, detail=detail, data=data)
        step["outcome"] = outcome or detail
        step["status"] = "done"
        step["ok"] = True
        step["elapsed_ms"] = 0
        return step

    def snapshot(self) -> List[Dict[str, Any]]:
        """The timeline, with anything still running closed off defensively."""
        for step in self._steps:
            if step["status"] == "running":
                self.finish(step)
        return list(self._steps)
