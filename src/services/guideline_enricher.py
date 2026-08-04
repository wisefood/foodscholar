"""
Post-extraction enrichment of dietary guidelines.

Existing guidelines were extracted page by page with no notion of the guide they
came from, so a rule like "Provide portions of red meat twice a week" sits in
the catalog with no population, no life stage, and no setting — while the QA
retriever boosts exactly those fields. This service walks the stored corpus and
fills them in, without re-running extraction.

The guide is the authority on who its rules apply to, so the crucial input is
guide context. It is assembled per guide, once, from three sources in
descending order of authority:

1. the catalog guide record;
2. the guide context captured at extraction time, when the guide was processed
   by a v2 extraction run;
3. a profile pass over the opening pages of the guide's own PDF.

The third source matters most for the existing corpus: those guides were
extracted before any context was captured, and their catalog metadata is often
too thin to establish a population. Reading the document recovers it.

Idempotency is by ``enrichment_version``: a guideline already enriched at the
current version is skipped, so a run can be repeated or resumed freely, and
bumping the version re-enriches everything. The catalog additionally refuses to
overwrite human-edited values, so re-running never undoes editorial work.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from agents.guideline_enrichment_agent import GuidelineEnrichmentAgent
from backend.elastic import ELASTIC_CLIENT
from config import config
from services.guideline_extractor import (
    DEFAULT_PROFILE_PAGE_COUNT,
    GuideContext,
    GuidelineExtractorService,
    extract_page_text,
    get_default_model,
    open_pdf,
    profile_guide_document,
    render_page_to_png,
    _load_openai_client,
    ensure_api_key,
)

logger = logging.getLogger(__name__)

GUIDELINE_INDEX = "guidelines"
GUIDE_INDEX = "guides"

# Bump to re-enrich the whole corpus with an improved prompt or vocabulary.
# Records at or above the current version are skipped.
GUIDELINE_ENRICHMENT_VERSION = 1

ENRICHMENT_AGENT_NAME = "guideline-enricher"

# The catalog caps a batch at 200 items.
ENRICH_BATCH_SIZE = 200
SCAN_PAGE_SIZE = 200

# Dry-run previews are read by a human; returning thousands of proposals
# would neither render nor help.
MAX_PREVIEW_PROPOSALS = 50

# Rules are independent and each is one mostly-waiting model call, so a few run
# at once. Bounded: the provider's rate limit is shared with every other worker
# and with extraction, and saturating it here would starve those.
DEFAULT_ENRICHMENT_CONCURRENCY = 8


@dataclass
class GuideContextResolution:
    """A guide's context plus where each part of it came from."""

    context: GuideContext
    sources: List[str] = field(default_factory=list)
    profiled_document: bool = False


@dataclass
class EnrichmentOutcome:
    """Result of enriching one guide's rules."""

    guide_urn: str
    total: int = 0
    enriched: int = 0
    skipped_version: int = 0
    skipped_no_facets: int = 0
    failed: int = 0
    context_sources: List[str] = field(default_factory=list)
    context_summary: str = ""
    proposals: List[Dict[str, Any]] = field(default_factory=list)


class GuidelineEnricher:
    """Walks stored guidelines and writes facets back to the catalog."""

    def __init__(
        self,
        agent: GuidelineEnrichmentAgent | None = None,
        platform_pool: Any | None = None,
        extractor: GuidelineExtractorService | None = None,
        version: int | None = None,
    ):
        self._agent = agent
        self._platform_pool = platform_pool
        self.extractor = extractor or GuidelineExtractorService()
        self.version = (
            version
            if version is not None
            else int(
                config.settings.get(
                    "GUIDELINE_ENRICHMENT_VERSION", GUIDELINE_ENRICHMENT_VERSION
                )
            )
        )
        self.concurrency = max(
            1,
            int(
                config.settings.get(
                    "GUIDELINE_ENRICHMENT_CONCURRENCY",
                    DEFAULT_ENRICHMENT_CONCURRENCY,
                )
            ),
        )
        self._context_cache: Dict[str, GuideContextResolution] = {}

    @property
    def agent(self) -> GuidelineEnrichmentAgent:
        if self._agent is None:
            self._agent = GuidelineEnrichmentAgent()
        return self._agent

    @property
    def platform_pool(self):
        if self._platform_pool is None:
            from backend.platform import WISEFOOD

            self._platform_pool = WISEFOOD
        return self._platform_pool

    # ------------------------------------------------------------------ #
    # Reading the corpus
    # ------------------------------------------------------------------ #

    def list_guide_urns(self) -> List[str]:
        """Every guide URN that has at least one non-deleted guideline."""
        response = ELASTIC_CLIENT.client.search(
            index=GUIDELINE_INDEX,
            body={
                "size": 0,
                "query": {"bool": {"must_not": [{"term": {"status": "deleted"}}]}},
                "aggs": {
                    "guides": {"terms": {"field": "guide_urn", "size": 1000}}
                },
            },
        )
        buckets = response["aggregations"]["guides"]["buckets"]
        return [bucket["key"] for bucket in buckets]

    def iter_guidelines(self, guide_urn: str) -> Iterator[Dict[str, Any]]:
        """
        Page through a guide's non-deleted guidelines.

        Uses ``search_after`` rather than ``from``/``size``: offset paging makes
        Elasticsearch re-sort and discard everything before the cursor on every
        page, and it stops dead at ``max_result_window``. ``search_after`` is
        flat-cost and unbounded, which matters because this walks a whole guide
        while an enrichment run mutates the documents underneath it.

        The tiebreaker on `id` is what makes the cursor total: `sequence_no`
        alone can repeat after a bad import, and a non-unique sort silently
        skips or repeats rows.
        """
        search_after: Optional[List[Any]] = None

        while True:
            body: Dict[str, Any] = {
                "size": SCAN_PAGE_SIZE,
                "sort": [{"sequence_no": "asc"}, {"id": "asc"}],
                "query": {
                    "bool": {
                        "filter": [{"term": {"guide_urn": guide_urn}}],
                        "must_not": [{"term": {"status": "deleted"}}],
                    }
                },
            }
            if search_after is not None:
                body["search_after"] = search_after

            response = ELASTIC_CLIENT.client.search(index=GUIDELINE_INDEX, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                return

            for hit in hits:
                yield hit.get("_source", {})

            if len(hits) < SCAN_PAGE_SIZE:
                return
            search_after = hits[-1].get("sort")
            if not search_after:
                # No sort values means we cannot advance; stopping is the only
                # safe option, since continuing would re-read the same page.
                logger.warning(
                    "Guideline scan for %s could not advance its cursor; "
                    "stopping after %s rows.",
                    guide_urn,
                    SCAN_PAGE_SIZE,
                )
                return

    def _get_guide_record(self, guide_urn: str) -> Optional[Dict[str, Any]]:
        try:
            response = ELASTIC_CLIENT.client.get(index=GUIDE_INDEX, id=guide_urn)
            return response.get("_source")
        except Exception as exc:
            logger.warning("Could not read guide %s: %s", guide_urn, exc)
            return None

    # ------------------------------------------------------------------ #
    # Guide context
    # ------------------------------------------------------------------ #

    def _stored_extraction_context(
        self, guide: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Recover the guide context a v2 extraction run captured, if there was one.

        Cheaper and more faithful than re-profiling, because it was produced
        while reading this exact document.
        """
        artifacts = guide.get("artifacts") or []
        artifact_ids = [
            str(artifact.get("id"))
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("id")
        ]
        if not artifact_ids:
            return None

        try:
            from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
            from models.db import GuidelineExtractionRecord
            import uuid as _uuid

            factory = POSTGRES_SYNC_SESSION_FACTORY()
            with factory() as session:
                for artifact_id in artifact_ids:
                    record = session.get(
                        GuidelineExtractionRecord, _uuid.UUID(artifact_id)
                    )
                    if record is None:
                        continue
                    result = record.result_json or {}
                    context = result.get("guide_context")
                    if context:
                        return context
        except Exception as exc:
            logger.debug(
                "No stored extraction context for %s: %s", guide.get("urn"), exc
            )
        return None

    def _profile_guide_pdf(self, guide: Dict[str, Any]) -> Optional[Any]:
        """
        Read the guide's PDF to recover what its metadata does not say.

        This is the path that rescues the already-extracted corpus: those guides
        were processed before any context was captured, and a title alone does
        not tell a rule who it is for.
        """
        artifacts = guide.get("artifacts") or []
        artifact_id = next(
            (
                str(artifact.get("id"))
                for artifact in artifacts
                if isinstance(artifact, dict) and artifact.get("id")
            ),
            None,
        )
        if not artifact_id:
            return None

        client = self.platform_pool.get_client()
        try:
            storage = self.extractor.get_artifact_workspace(artifact_id)
            pdf_path = storage.pdf_path
            if not storage.pdf_exists:
                client.artifacts.download_to(artifact_id, pdf_path)
        except Exception as exc:
            logger.warning(
                "Could not stage the PDF for guide %s: %s", guide.get("urn"), exc
            )
            return None
        finally:
            self.platform_pool.return_client(client)

        try:
            ensure_api_key()
            OpenAI = _load_openai_client()
            openai_client = OpenAI()
            doc = open_pdf(pdf_path)
            try:
                pages = []
                for index in range(min(DEFAULT_PROFILE_PAGE_COUNT, len(doc))):
                    page = doc[index]
                    pages.append(
                        (
                            index + 1,
                            extract_page_text(page),
                            render_page_to_png(page),
                        )
                    )
            finally:
                doc.close()

            return profile_guide_document(
                client=openai_client,
                model=get_default_model(),
                pages=pages,
            )
        except Exception as exc:
            logger.warning(
                "Could not profile the PDF for guide %s: %s", guide.get("urn"), exc
            )
            return None

    def resolve_guide_context(
        self, guide_urn: str, *, allow_pdf_profile: bool = True
    ) -> GuideContextResolution:
        """Assemble a guide's context once, then reuse it for all its rules."""
        cached = self._context_cache.get(guide_urn)
        if cached is not None:
            return cached

        guide = self._get_guide_record(guide_urn) or {}
        context = GuideContext.from_guide(guide)
        sources = ["catalog"] if not context.is_empty() else []
        profiled = False

        if context.needs_document_profile():
            stored = self._stored_extraction_context(guide)
            if stored:
                stored_context = GuideContext(**{
                    key: value
                    for key, value in stored.items()
                    if key in GuideContext.__dataclass_fields__
                })
                merged = GuideContext(
                    guide_urn=context.guide_urn or stored_context.guide_urn,
                    title=context.title or stored_context.title,
                    region=context.region or stored_context.region,
                    audience=context.audience or stored_context.audience,
                    target_audiences=(
                        context.target_audiences or stored_context.target_audiences
                    ),
                    language=context.language or stored_context.language,
                    publication_year=(
                        context.publication_year or stored_context.publication_year
                    ),
                    issuing_authority=(
                        context.issuing_authority or stored_context.issuing_authority
                    ),
                    population_note=stored_context.population_note,
                    age_min_months=stored_context.age_min_months,
                    age_max_months=stored_context.age_max_months,
                    scope_note=stored_context.scope_note,
                    evidence=list(stored_context.evidence),
                    derived_fields=list(stored_context.derived_fields),
                )
                context = merged
                sources.append("extraction_result")

        if allow_pdf_profile and context.needs_document_profile():
            profile = self._profile_guide_pdf(guide)
            if profile is not None:
                context = context.merge_document_profile(profile)
                sources.append("document_profile")
                profiled = True

        if context.is_empty():
            logger.warning(
                "No context could be established for guide %s; its rules will be "
                "enriched from their own text alone.",
                guide_urn,
            )

        resolution = GuideContextResolution(
            context=context, sources=sources, profiled_document=profiled
        )
        self._context_cache[guide_urn] = resolution
        return resolution

    # ------------------------------------------------------------------ #
    # Enrichment
    # ------------------------------------------------------------------ #

    def _needs_enrichment(self, guideline: Dict[str, Any]) -> bool:
        current = guideline.get("enrichment_version")
        if not isinstance(current, int):
            return True
        return current < self.version

    def _write_batch(self, items: List[Dict[str, Any]], dry_run: bool) -> int:
        """Send one enrichment batch to the catalog; returns items accepted."""
        if not items:
            return 0

        client = self.platform_pool.get_client()
        try:
            response = client.guidelines.enrich_batch(
                agent=ENRICHMENT_AGENT_NAME,
                items=items,
                dry_run=dry_run,
            )
        except Exception:
            # One bad batch must not discard a guide's completed work. The run
            # is resumable by enrichment_version, so the affected rules are
            # simply picked up next time.
            logger.warning(
                "Enrichment batch of %s item(s) failed; continuing with the rest "
                "of the guide.",
                len(items),
                exc_info=True,
            )
            return 0
        finally:
            self.platform_pool.return_client(client)

        succeeded = response.get("succeeded", 0) if isinstance(response, dict) else 0
        failures = [
            result
            for result in (response.get("results", []) if isinstance(response, dict) else [])
            if not result.get("ok")
        ]
        for failure in failures:
            logger.warning(
                "Enrichment rejected for guideline %s: %s",
                failure.get("id"),
                failure.get("error"),
            )
        return succeeded

    def enrich_guide(
        self,
        guide_urn: str,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        force: bool = False,
        allow_pdf_profile: bool = True,
        progress_callback=None,
    ) -> EnrichmentOutcome:
        """
        Enrich every guideline under one guide.

        ``dry_run`` runs the agent and reports the proposed facets without
        writing. ``force`` ignores the version check (but not the catalog's
        protection of human-edited values).
        """
        resolution = self.resolve_guide_context(
            guide_urn, allow_pdf_profile=allow_pdf_profile
        )
        context_block = resolution.context.as_prompt_block()

        outcome = EnrichmentOutcome(
            guide_urn=guide_urn,
            context_sources=resolution.sources,
            context_summary=context_block,
        )

        pending: List[Dict[str, Any]] = []

        def classify(guideline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """One rule through the agent. Runs on a worker thread."""
            guideline_id = guideline.get("id")
            if not guideline_id:
                return {"outcome": "failed"}
            try:
                facets = self.agent.enrich_guideline(guideline, context_block)
            except Exception as exc:
                logger.error(
                    "Enrichment agent failed for guideline %s: %s",
                    guideline_id,
                    exc,
                    exc_info=True,
                )
                return {"outcome": "failed"}

            if not facets:
                return {"outcome": "no_facets"}

            facets["enrichment_version"] = self.version
            return {
                "outcome": "enriched",
                "id": str(guideline_id),
                "rule_text": guideline.get("rule_text"),
                "facets": facets,
            }

        def collect(result: Optional[Dict[str, Any]]) -> None:
            """Fold one agent result into the outcome. Main thread only."""
            if result is None or result["outcome"] == "failed":
                outcome.failed += 1
                return
            if result["outcome"] == "no_facets":
                outcome.skipped_no_facets += 1
                return

            pending.append({"id": result["id"], "fields": result["facets"]})
            if dry_run and len(outcome.proposals) < MAX_PREVIEW_PROPOSALS:
                outcome.proposals.append(
                    {
                        "id": result["id"],
                        "rule_text": result["rule_text"],
                        "facets": result["facets"],
                    }
                )
            if progress_callback is not None:
                progress_callback(outcome.total)

        # Rules are independent, and each is one model call of mostly waiting.
        # Running a bounded number concurrently is what turns a 2700-rule
        # backfill from ~90 minutes into ~15; the cap keeps a single guide from
        # monopolising the provider's rate limit and starving other workers.
        # Batches are still written from this thread, in order.
        batch: list[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for guideline in self.iter_guidelines(guide_urn):
                if limit is not None and outcome.total >= limit:
                    break
                outcome.total += 1

                if not force and not self._needs_enrichment(guideline):
                    outcome.skipped_version += 1
                    continue

                batch.append(guideline)
                if len(batch) < self.concurrency:
                    continue

                for result in pool.map(classify, batch):
                    collect(result)
                batch = []

                if len(pending) >= ENRICH_BATCH_SIZE:
                    outcome.enriched += self._write_batch(pending, dry_run)
                    pending = []

            for result in pool.map(classify, batch):
                collect(result)

        outcome.enriched += self._write_batch(pending, dry_run)

        logger.info(
            "Enriched guide %s: total=%s enriched=%s skipped_version=%s "
            "skipped_no_facets=%s failed=%s context=%s",
            guide_urn,
            outcome.total,
            outcome.enriched,
            outcome.skipped_version,
            outcome.skipped_no_facets,
            outcome.failed,
            resolution.sources or "none",
        )
        return outcome
