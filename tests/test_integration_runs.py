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

    def create(self, **fields):
        self.created.append(fields)
        return {"urn": self.urn, **fields}


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
