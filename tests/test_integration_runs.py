"""Phase 2: what happens after a person approves.

The executor is the piece that actually changes the catalog, so these tests
are about the ways it must refuse and the ways it must not lose work:

* it writes nothing at all when a precondition fails, because a half-created
  guide is somebody's afternoon;
* a licence that forbids copying registers a pointer and stops;
* a retry after a failure reuses what already landed rather than creating a
  second copy of it;
* an import that would create nothing is a failure, not a success — the whole
  point of the run was to add something.

No Postgres and no network: an in-memory store, a fake catalog client and a
fake core transport, driven through the real registry so the approval gate
and the tool contracts are the ones that actually ship.
"""
from __future__ import annotations



import pytest


# --------------------------------------------------------------- fakes --

class FakeProxy:
    def __init__(self, urn):
        self.urn = urn
        self.created = []
        #: What a catalog search would return for this kind.
        self.hits = []

    def create(self, **fields):
        self.created.append(fields)
        return {"urn": self.urn, **fields}

    def search(self, q, limit=10):
        return list(self.hits)[:limit]


class FakeArtifacts:
    def __init__(self, artifact_id="art-1"):
        self.artifact_id = artifact_id
        self.uploads = []

    def upload(self, path, **kw):
        self.uploads.append({"path": path, **kw})
        return {"id": self.artifact_id}


class FakeDataClient:
    def __init__(self, urn="urn:wf:guide:1", artifact_id="art-1"):
        self.guides = FakeProxy(urn)
        self.articles = FakeProxy(urn)
        self.textbooks = FakeProxy(urn)
        self.artifacts = FakeArtifacts(artifact_id)


class FakeCore:
    """The guideline routes, scripted.

    `statuses` is consumed one per status check, so a test can make an
    extraction run for three polls and then succeed.
    """

    def __init__(self, statuses=None, imports=None):
        self.statuses = list(statuses or [{"status": "succeeded",
                                           "result": {"guidelines": [{}] * 5}}])
        self.imports = list(imports or [])
        self.posts = []

    def post(self, path, body):
        self.posts.append((path, dict(body)))
        if "/extract/" in path:
            return {"status": "queued"}
        if self.imports:
            return self.imports.pop(0)
        dry = bool(body.get("dry_run"))
        # Faithful to the core service: a dry run creates nothing, so
        # `total_created` is 0 and the plan is in the items.
        return {"total_created": 0 if dry else 5,
                "total_candidates": 5, "total_skipped": 0, "dry_run": dry,
                "items": [{"status": "would_create" if dry else "created"}] * 5}

    def get(self, path):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


@pytest.fixture
def registry():
    from wisefood_mcp import build_registry
    return build_registry()


@pytest.fixture
def store():
    from wisefood_mcp.stores import InMemoryProposalStore
    return InMemoryProposalStore()


HANDLE = "a" * 20


@pytest.fixture(autouse=True)
def fetched_file(tmp_path, monkeypatch):
    """A handle only resolves to a file that is actually on disk.

    `upload_artifact` checks, which is right — a handle from a pod that has
    since been replaced points at nothing — so the tests have to put a file
    where it looks rather than pass a plausible-looking string.
    """
    from wisefood_mcp.tools import research

    monkeypatch.setattr(research, "PENDING_DIR", tmp_path)
    (tmp_path / f"{HANDLE}.pdf").write_bytes(b"%PDF-1.4 fake")
    return tmp_path


def make_proposal(store, **kw):
    from wisefood_mcp.stores import Proposal, approve, new_proposal_id

    fields = dict(
        id=new_proposal_id(), kind="guide", title="Bulgarian FBDG for adults",
        source_url="https://ncpha.bg/fbdg.pdf", status="proposed",
        country="Bulgaria", language="Bulgarian", licence="CC-BY-4.0",
        metadata={"pending_artifact": HANDLE},
    )
    approved = kw.pop("approved", True)
    fields.update(kw)
    row = store.create(Proposal(**fields))
    if approved:
        approve(store, row.id, actor="curator-1")
    return store.get(row.id)


def make_context(store, core=None, data_client=None, **kw):
    from wisefood_mcp import ToolContext

    core = core or FakeCore()
    return ToolContext(
        data_client=data_client if data_client is not None else FakeDataClient(),
        proposal_store=store, writes_enabled=True,
        core_post=core.post, core_get=core.get, actor="curator-1", **kw,
    ), core


def run_integration(proposal, registry, ctx, **kw):
    from integrator.executor import Integration

    saved = []
    options = {"poll_interval": 0, "poll_timeout": 5, "sleep": lambda _s: None}
    options.update(kw)
    outcome = Integration(proposal=proposal, registry=registry, ctx=ctx,
                          persist=saved.append, **options).run()
    return outcome, saved


# --------------------------------------------------------- the happy path --

def test_a_guide_goes_all_the_way_through(registry, store):
    proposal = make_proposal(store)
    ctx, core = make_context(store)
    outcome, saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    assert outcome["result"]["urn"] == "urn:wf:guide:1"
    assert outcome["result"]["artifact_id"] == "art-1"
    assert outcome["result"]["guidelines_created"] == 5
    assert outcome["wrote_anything"] is True

    # Preview before import, and the import body uses the field name the core
    # route's request model actually declares.
    imports = [(p, b) for p, b in core.posts if "/import/" in p]
    assert [b["dry_run"] for _p, b in imports] == [True, False]
    assert all("guide_id" in b and "guide_urn" not in b for _p, b in imports)

    # Progress was persisted as it went, not only at the end.
    assert len(saved) > 4
    assert saved[-1]["finished"] is True


def test_the_timeline_says_what_it_did(registry, store):
    proposal = make_proposal(store)
    ctx, _core = make_context(store)
    outcome, _saved = run_integration(proposal, registry, ctx)

    outcomes = " | ".join(s.get("outcome") or "" for s in outcome["steps"])
    assert "Created urn:wf:guide:1" in outcomes
    assert "Attached as artifact art-1" in outcomes
    assert "5 guidelines found" in outcomes
    assert "would create" in outcomes
    assert "Created 5 guidelines" in outcomes


def test_provenance_travels_with_the_entity(registry, store):
    proposal = make_proposal(store)
    client = FakeDataClient()
    ctx, _core = make_context(store, data_client=client)
    run_integration(proposal, registry, ctx)

    integration = client.guides.created[0]["extras"]["integration"]
    assert integration["proposal_id"] == proposal.id
    assert integration["approved_by"] == "curator-1"
    assert integration["source_url"] == "https://ncpha.bg/fbdg.pdf"
    assert integration["licence"] == "CC-BY-4.0"


# ------------------------------------------------------------- refusals --

def test_an_unapproved_proposal_writes_nothing(registry, store):
    proposal = make_proposal(store, approved=False)
    client = FakeDataClient()
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "not been approved" in outcome["error"]
    assert client.guides.created == []
    assert outcome["wrote_anything"] is False


def test_a_guide_with_no_fetched_file_fails_before_creating_it(registry, store):
    """The check that exists so the catalog does not collect hollow guides."""
    proposal = make_proposal(store, metadata={})
    client = FakeDataClient()
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "fetch_url" in outcome["error"]
    assert client.guides.created == []


def test_a_restrictive_licence_registers_a_pointer_and_stops(registry, store):
    proposal = make_proposal(store, licence="proprietary")
    client = FakeDataClient()
    ctx, core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded"
    assert client.guides.created, "the reference itself is still worth having"
    assert client.artifacts.uploads == [], "the document must not be copied in"
    assert core.posts == [], "and nothing was extracted from it"
    assert "pointer" in " ".join(s["title"] for s in outcome["steps"]).lower()


def test_a_kind_with_no_pipeline_says_so(registry, store):
    proposal = make_proposal(store, kind="recipe_collection")
    ctx, _core = make_context(store)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "recipe_collection" in outcome["error"]


def test_writes_switched_off_stops_at_the_first_write(registry, store):
    proposal = make_proposal(store)
    client = FakeDataClient()
    ctx, _core = make_context(store, data_client=client)
    ctx.writes_enabled = False
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "switched off" in outcome["error"]
    assert client.guides.created == []


# -------------------------------------------------------------- failures --

def test_an_extraction_that_finds_nothing_is_a_failure(registry, store):
    proposal = make_proposal(store)
    ctx, core = make_context(store, core=FakeCore(
        statuses=[{"status": "succeeded", "result": {"guidelines": []}}]))
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "no guidelines" in outcome["error"]
    # The guide and its file are still there to look at, and the run says so.
    assert outcome["result"]["urn"] == "urn:wf:guide:1"
    assert outcome["wrote_anything"] is True


def test_a_failed_extraction_carries_its_reason(registry, store):
    proposal = make_proposal(store)
    ctx, _core = make_context(store, core=FakeCore(
        statuses=[{"status": "failed", "error": "the PDF has no text layer"}]))
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "no text layer" in outcome["error"]


def test_an_extraction_that_never_finishes_hands_the_wait_back(registry, store):
    proposal = make_proposal(store)
    ctx, _core = make_context(store, core=FakeCore(
        statuses=[{"status": "running", "current_page": 4, "total_pages": 90}]))
    outcome, _saved = run_integration(proposal, registry, ctx, poll_timeout=-1)

    assert outcome["status"] == "failed"
    assert "retry" in outcome["error"].lower()
    assert outcome["result"]["extraction"]["current_page"] == 4


def test_nothing_new_to_import_is_a_failure_not_a_success(registry, store):
    proposal = make_proposal(store)
    ctx, _core = make_context(store, core=FakeCore(imports=[
        {"total_created": 0, "total_skipped": 5, "total_candidates": 5,
         "dry_run": True, "items": [{"status": "skipped_existing"}] * 5},
    ]))
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "already exist" in outcome["error"]


# ---------------------------------------------------------------- resume --

def test_a_retry_reuses_what_already_landed(registry, store):
    """The reason a run is a row and not a column on the proposal."""
    proposal = make_proposal(store)
    ctx, _core = make_context(store, core=FakeCore(
        statuses=[{"status": "failed", "error": "worker restarted"}]))
    first, _saved = run_integration(proposal, registry, ctx)
    assert first["status"] == "failed"

    # The proposal now carries the urn and artifact the first attempt created.
    again = store.get(proposal.id)
    assert again.result["urn"] == "urn:wf:guide:1"

    client = FakeDataClient(urn="urn:wf:guide:SECOND")
    ctx2, _core2 = make_context(store, data_client=client)
    second, _saved2 = run_integration(again, registry, ctx2)

    assert second["status"] == "succeeded"
    assert second["result"]["urn"] == "urn:wf:guide:1", "not a second guide"
    assert client.guides.created == [], "and nothing was created again"
    assert client.artifacts.uploads == []


# ------------------------------------------------------------- dry runs --

def test_a_dry_run_previews_and_writes_no_guidelines(registry, store):
    proposal = make_proposal(store)
    ctx, core = make_context(store)
    outcome, _saved = run_integration(proposal, registry, ctx, dry_run=True)

    assert outcome["status"] == "succeeded"
    assert outcome["result"].get("guidelines_created") is None
    assert [b["dry_run"] for p, b in core.posts if "/import/" in p] == [True]
    assert outcome["result"]["preview"]["would_create"] == 5


# --------------------------------------------------------------- specs --

def test_the_spec_falls_back_to_what_the_proposal_knows(store):
    from integrator.executor import guide_spec

    proposal = make_proposal(store, rationale="Fills the Bulgaria gap")
    spec = guide_spec(proposal)
    assert spec["title"] == "Bulgarian FBDG for adults"
    assert spec["region"] == "Bulgaria"
    assert spec["language"] == "Bulgarian"
    assert spec["description"] == "Fills the Bulgaria gap"


def test_a_spec_the_assistant_wrote_wins(store):
    from integrator.executor import guide_spec

    proposal = make_proposal(store, metadata={
        "pending_artifact": HANDLE,
        "spec": {"title": "Хранене и здраве", "region": "BG", "publication_date": "2021"},
    })
    spec = guide_spec(proposal)
    assert spec["title"] == "Хранене и здраве"
    assert spec["region"] == "BG"
    assert spec["publication_date"] == "2021"
    assert spec["language"] == "Bulgarian", "and the floor still fills the rest"


# --------------------------------------------------------------- articles --

class FakeEnrichCore(FakeCore):
    """Adds the enrichment endpoints to the scripted core."""

    def __init__(self, statuses=None, **kw):
        super().__init__(**kw)
        self.enrichment = list(statuses or [{"status": "succeeded",
                                             "result": {"keywords": 1, "qa": 1}}])

    def post(self, path, body):
        self.posts.append((path, dict(body)))
        if "/enrich/articles/" in path:
            return {"status": "queued"}
        return super().post(path, body)

    def get(self, path):
        if "/enrich/articles/" in path:
            return (self.enrichment.pop(0) if len(self.enrichment) > 1
                    else self.enrichment[0])
        return super().get(path)


CROSSREF = {
    "found": True, "doi": "10.1136/bmj.n1234",
    "title": "Ultra-processed food and cardiovascular risk",
    "authors": ["Jane Smith", "Arto Virtanen"],
    "venue": "BMJ", "publisher": "BMJ Publishing Group",
    "publication_year": 2021, "abstract": "Background: we looked at things.",
    "language": "en", "url": "https://doi.org/10.1136/bmj.n1234",
    "subjects": ["Nutrition", "Cardiology"],
    "citation_count": 87, "reference_count": 45,
}


def make_article(store, **kw):
    metadata = {"doi": CROSSREF["doi"], "doi_metadata": CROSSREF}
    metadata.update(kw.pop("metadata", {}))
    return make_proposal(store, kind="article", title="whatever the model typed",
                         metadata=metadata, **kw)


def test_an_article_is_created_and_enriched(registry, store):
    proposal = make_article(store)
    client = FakeDataClient(urn="urn:wf:article:1")
    ctx, core = make_context(store, core=FakeEnrichCore(), data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    assert outcome["result"]["urn"] == "urn:wf:article:1"
    assert outcome["result"]["enrichment"]["status"] == "succeeded"
    assert any("/enrich/articles/" in p for p, _b in core.posts)
    # No extraction: an article has no PDF pipeline behind it.
    assert not any("/guidelines/" in p for p, _b in core.posts)


def test_crossref_beats_whatever_the_assistant_typed(registry, store):
    """The one thing that must not happen to a scientific catalog is a
    citation with invented authors that reads perfectly well."""
    proposal = make_article(store, metadata={"spec": {
        "title": "Ultraprocessed foods and heart disease",   # subtly wrong
        "authors": ["J. Smith", "A. Virtanen", "R. Invented"],
        "publication_year": 2020,
    }})
    client = FakeDataClient(urn="urn:wf:article:1")
    ctx, _core = make_context(store, core=FakeEnrichCore(), data_client=client)
    run_integration(proposal, registry, ctx)

    created = client.articles.created[0]
    assert created["title"] == CROSSREF["title"]
    assert created["authors"] == CROSSREF["authors"]
    assert created["publication_year"] == 2021
    assert created["venue"] == "BMJ"
    assert created["doi"] == CROSSREF["doi"]


def test_the_assistants_spec_still_fills_what_crossref_has_no_field_for(registry, store):
    proposal = make_article(store, population_group="adults", metadata={"spec": {
        "topics": ["cardiovascular health"], "reader_group": "practitioner",
    }})
    client = FakeDataClient(urn="urn:wf:article:1")
    ctx, _core = make_context(store, core=FakeEnrichCore(), data_client=client)
    run_integration(proposal, registry, ctx)

    created = client.articles.created[0]
    assert created["topics"] == ["cardiovascular health"]
    assert created["reader_group"] == "practitioner"
    assert created["population_group"] == "adults"
    # Crossref subjects become keywords when the assistant offered none.
    assert created["keywords"] == ["Nutrition", "Cardiology"]


def test_a_doi_we_already_hold_is_refused_before_anything_is_created(registry, store):
    """A DOI names one paper, so importing it twice is never right."""
    proposal = make_article(store)
    client = FakeDataClient(urn="urn:wf:article:NEW")
    client.articles.hits = [{"urn": "urn:wf:article:OLD", "doi": CROSSREF["doi"]}]
    ctx, _core = make_context(store, core=FakeEnrichCore(), data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "already holds" in outcome["error"]
    assert "urn:wf:article:OLD" in outcome["error"]
    assert client.articles.created == []


def test_a_different_doi_in_the_results_does_not_block(registry, store):
    """The search is fuzzy, so only an exact DOI match may refuse."""
    proposal = make_article(store)
    client = FakeDataClient(urn="urn:wf:article:1")
    client.articles.hits = [{"urn": "urn:wf:article:OTHER", "doi": "10.9999/other"}]
    ctx, _core = make_context(store, core=FakeEnrichCore(), data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    assert client.articles.created


def test_a_failed_enrichment_says_the_article_is_still_there(registry, store):
    proposal = make_article(store)
    ctx, _core = make_context(store, core=FakeEnrichCore(
        statuses=[{"status": "failed", "error": "the abstract was empty"}]))
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "empty" in outcome["error"]
    assert "in the catalog" in outcome["error"]
    assert outcome["result"]["urn"]
    assert outcome["wrote_anything"] is True


def test_the_article_timeline_reads_as_what_happened(registry, store):
    proposal = make_article(store)
    ctx, _core = make_context(store, core=FakeEnrichCore())
    outcome, _saved = run_integration(proposal, registry, ctx)

    outcomes = " | ".join(s.get("outcome") or "" for s in outcome["steps"])
    assert "Created urn:wf:guide:1" in outcomes or "Created urn:" in outcomes
    assert "enriched" in outcomes


def test_the_spec_needs_no_crossref_record_to_work(store):
    """A proposal filed before anyone looked the DOI up still integrates."""
    from integrator.executor import article_spec

    proposal = make_proposal(store, kind="article", title="A paper",
                             metadata={"doi": "10.1/x"})
    spec = article_spec(proposal)
    assert spec["title"] == "A paper"
    assert spec["doi"] == "10.1/x"
    assert spec["url"] == "https://ncpha.bg/fbdg.pdf"


# -------------------------------------------------------------- textbooks --

class FakeBoundPassages:
    def __init__(self, parent):
        self.parent = parent

    def bulk_replace(self, **kw):
        self.parent.replaced.append(kw)
        return {"total": len(kw.get("passages") or [])}


class FakePassagesProxy:
    def __init__(self):
        self.replaced = []

    def by_textbook(self, urn):
        self.replaced_urn = urn
        return FakeBoundPassages(self)


@pytest.fixture
def textbook_pdf(fetched_file):
    """A real PDF behind the pending handle, since the tool actually reads it."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for section in ("Chapter 1 Energy", "Chapter 2 Protein"):
        page = doc.new_page()
        page.insert_text((60, 60), section, fontsize=22)
        y = 100
        for _ in range(12):
            page.insert_text((60, y), "Dietary energy needs vary with age and "
                                      "activity across populations studied.",
                             fontsize=11)
            y += 15
    doc.save(str(fetched_file / f"{HANDLE}.pdf"))
    doc.close()
    return fetched_file


def make_textbook(store, **kw):
    return make_proposal(store, kind="textbook",
                         title="Human Nutrition, 6th edition", **kw)


def test_a_textbook_is_read_into_passages(registry, store, textbook_pdf):
    proposal = make_textbook(store)
    client = FakeDataClient(urn="urn:wf:textbook:1")
    client.textbook_passages = FakePassagesProxy()
    ctx, core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    assert outcome["result"]["urn"] == "urn:wf:textbook:1"
    assert outcome["result"]["passages"] > 0
    assert outcome["result"]["page_count"] == 2
    # No queue and no core API: chunking needs no model, so it runs inline.
    assert core.posts == []


def test_the_passages_carry_their_provenance(registry, store, textbook_pdf):
    proposal = make_textbook(store)
    client = FakeDataClient(urn="urn:wf:textbook:1")
    client.textbook_passages = FakePassagesProxy()
    ctx, _core = make_context(store, data_client=client)
    run_integration(proposal, registry, ctx)

    call = client.textbook_passages.replaced[0]
    assert call["artifact_id"] == "art-1"
    assert call["extractor_name"] == "wisefood-mcp/passages"
    assert call["extractor_run_id"] == proposal.id
    assert call["page_count"] == 2
    assert all(p["structure_path"] for p in call["passages"])


def test_a_textbook_with_no_fetched_file_fails_before_creating_it(registry, store):
    proposal = make_textbook(store, metadata={})
    client = FakeDataClient(urn="urn:wf:textbook:1")
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "fetch_url" in outcome["error"]
    assert client.textbooks.created == []


def test_a_scanned_textbook_fails_with_a_reason_a_person_can_act_on(
        registry, store, fetched_file):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    doc.save(str(fetched_file / f"{HANDLE}.pdf"))
    doc.close()

    proposal = make_textbook(store)
    client = FakeDataClient(urn="urn:wf:textbook:1")
    client.textbook_passages = FakePassagesProxy()
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert "OCR" in outcome["error"] or "scanned" in outcome["error"]
    # The textbook record itself is real and stays; only its content is missing.
    assert outcome["result"]["urn"] == "urn:wf:textbook:1"
    assert outcome["wrote_anything"] is True


def test_a_restrictive_licence_registers_the_textbook_without_its_text(
        registry, store, textbook_pdf):
    proposal = make_textbook(store, licence="proprietary")
    client = FakeDataClient(urn="urn:wf:textbook:1")
    client.textbook_passages = FakePassagesProxy()
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded"
    assert client.textbooks.created, "the reference is still worth having"
    assert client.artifacts.uploads == []
    assert client.textbook_passages.replaced == [], "no copying the book in"


# ------------------------------------------------- food composition tables --

@pytest.fixture
def table_file(fetched_file):
    """A real spreadsheet behind the handle — the profiler actually reads it."""
    pandas = pytest.importorskip("pandas")
    pandas.DataFrame({
        "Food code": [str(i) for i in range(50)],
        "Food name": [f"Food {i}" for i in range(50)],
        "Energy (kcal) per 100g": [100 + i for i in range(50)],
        "Protein (g)": [1.0 * i for i in range(50)],
        "Saturated fat (g)": [0.5 * i for i in range(50)],
        "Vitamin C (mg)": [None] * 25 + [1.0] * 25,
    }).to_csv(fetched_file / f"{HANDLE}.csv", index=False)
    (fetched_file / f"{HANDLE}.pdf").unlink(missing_ok=True)
    return fetched_file


def test_a_composition_table_is_profiled_into_its_metadata(registry, store, table_file):
    """`FCTable` is a metadata entity with no row store behind it, so what the
    file yields is a description of itself."""
    proposal = make_proposal(store, kind="fctable", title="National FCT 2023")
    client = FakeDataClient(urn="urn:wf:fctable:1")
    client.fctables = FakeProxy("urn:wf:fctable:1")
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    created = client.fctables.created[0]
    assert created["number_of_entries"] == 50
    assert "energy" in created["nutrient_coverage"]
    assert "saturated fat" in created["nutrient_coverage"]
    assert "vitamin C" in created["nutrient_coverage"]
    assert created["completeness_percent"] < 100, "half the vitamin C is missing"
    assert "kcal" in created["measurement_units"]


def test_the_profile_is_measured_before_the_entity_is_written(registry, store, table_file):
    """Otherwise the catalog gets a record everyone can see and then a
    correction."""
    proposal = make_proposal(store, kind="fctable", title="National FCT 2023")
    client = FakeDataClient(urn="urn:wf:fctable:1")
    client.fctables = FakeProxy("urn:wf:fctable:1")
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    # One create, carrying the numbers already.
    assert len(client.fctables.created) == 1
    assert client.fctables.created[0]["number_of_entries"] == 50
    assert outcome["result"]["profile"]["number_of_entries"] == 50


def test_a_table_with_no_file_is_still_registered(registry, store):
    """A table we may only point at still deserves its entry."""
    proposal = make_proposal(store, kind="fctable", title="National FCT 2023",
                             metadata={})
    client = FakeDataClient(urn="urn:wf:fctable:1")
    client.fctables = FakeProxy("urn:wf:fctable:1")
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "succeeded", outcome.get("error")
    assert client.fctables.created
    assert "number_of_entries" not in client.fctables.created[0]


def test_an_unreadable_table_fails_before_anything_is_created(registry, store, fetched_file):
    (fetched_file / f"{HANDLE}.pdf").unlink(missing_ok=True)
    (fetched_file / f"{HANDLE}.csv").write_text("\x00\x01 not a table at all")
    proposal = make_proposal(store, kind="fctable", title="Broken")
    client = FakeDataClient(urn="urn:wf:fctable:1")
    client.fctables = FakeProxy("urn:wf:fctable:1")
    ctx, _core = make_context(store, data_client=client)
    outcome, _saved = run_integration(proposal, registry, ctx)

    assert outcome["status"] == "failed"
    assert client.fctables.created == [], "the profile runs before the write"
    assert outcome["wrote_anything"] is False


def test_every_tool_says_what_it_is_doing_in_words(registry):
    """The narration is the feature, so a tool missing from it is a bug.

    `create_fctable` shipped without an entry and rendered as "Running
    create_fctable" — the raw function name, which is exactly what a curator
    is not supposed to have to read. Nothing catches that except this.
    """
    from integrator.steps import RUNNING, running_title

    tools = [t["function"]["name"]
             for t in registry.openai_schemas(include_writes=True)]
    missing = sorted(set(tools) - set(RUNNING))
    assert not missing, f"no words for: {missing}"

    for tool in tools:
        _kind, title, _detail = running_title(tool, {})
        assert not title.startswith("Running "), f"{tool} fell back to its own name"
        assert tool not in title, f"{tool} shows its function name to a curator"


class TestTheCitationIsResolvedInTheRun:
    """Where the "no invented citations" guarantee actually lives.

    `article_spec` preferred a Crossref record stashed on the proposal —
    but nothing ever stashed one, so the branch never fired and articles
    were created from whatever the assistant typed. Resolving it in the run
    fixes that and makes the guarantee unconditional: it no longer depends
    on the assistant having been diligent before proposing.
    """

    def _executor(self, proposal, calls, records):
        from wisefood_mcp import ToolContext

        from integrator.executor import Integration

        class Registry:
            def call(self, tool, args, ctx):
                import json as _json
                parsed = _json.loads(args) if isinstance(args, str) else args
                calls.append((tool, parsed))
                if tool == "doi_metadata":
                    return {"ok": True, "result": records.get(parsed["doi"],
                                                              {"found": False})}
                if tool == "search_catalog":
                    return {"ok": True, "result": {"items": []}}
                return {"ok": True, "result": {}}

        ctx = ToolContext(proposal_store=None, data_client=object())
        return Integration(proposal=proposal, registry=Registry(), ctx=ctx,
                           persist=lambda _s: None)

    def _proposal(self, **kw):
        from wisefood_mcp.stores import Proposal, new_proposal_id

        fields = dict(id=new_proposal_id(), kind="article", title="Typed title",
                      status="approved", licence="CC-BY-4.0")
        fields.update(kw)
        return Proposal(**fields)

    def test_the_publishers_record_replaces_what_was_typed(self):
        calls = []
        proposal = self._proposal(metadata={"doi": "10.1186/s12937-026-01386-8"})
        ex = self._executor(proposal, calls, {
            "10.1186/s12937-026-01386-8": {
                "found": True, "title": "Dietary patterns and cardiovascular risk",
                "authors": ["Real Author"], "publication_year": 2026,
                "publisher": "Springer", "doi": "10.1186/s12937-026-01386-8"}})
        ex._resolve_the_citation()

        assert ("doi_metadata", {"doi": "10.1186/s12937-026-01386-8"}) in calls
        from integrator.executor import article_spec
        spec = article_spec(ex.proposal)
        assert spec["title"] == "Dietary patterns and cardiovascular risk"
        assert spec["authors"] == ["Real Author"]

    def test_a_doi_crossref_does_not_know_stops_the_run(self):
        """Better to refuse than to create an article from a typed citation."""
        from integrator.executor import IntegrationError

        calls = []
        proposal = self._proposal(metadata={"doi": "10.9999/invented"})
        ex = self._executor(proposal, calls, {})
        with pytest.raises(IntegrationError) as caught:
            ex._resolve_the_citation()
        assert "no record" in str(caught.value)
        assert "nothing was created" in str(caught.value)

    def test_an_article_without_a_doi_is_still_allowed(self):
        """A report or a preprint may have none. There is simply no record to
        read, which is not the same as a citation that failed to check out."""
        calls = []
        proposal = self._proposal(metadata={})
        ex = self._executor(proposal, calls, {})
        ex._resolve_the_citation()
        assert not any(t == "doi_metadata" for t, _ in calls)

    def test_a_record_already_present_is_not_fetched_again(self):
        calls = []
        proposal = self._proposal(metadata={
            "doi": "10.1/x", "doi_metadata": {"found": True, "title": "Known"}})
        ex = self._executor(proposal, calls, {})
        ex._resolve_the_citation()
        assert not any(t == "doi_metadata" for t, _ in calls)
