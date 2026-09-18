"""What we send against what the catalog will actually accept.

Every failure in this area has looked the same from a curator's seat: they
approve a source, the run fetches the document, creates nothing, and stops on
a validation error naming a field nobody had heard of. `body.urn: Field
required`. `body.extras: Extra inputs are not permitted`. `body.region:
String should have at most 2 characters`.

Those were only ever found by running it in production, because nothing
compared the spec we build with the schema it is posted to. This does, using
the catalog's own Pydantic models — so the next divergence is a red test
rather than somebody's afternoon.

Skipped when `wisefood-data-api` is not beside this checkout, since the
schemas are its source of truth and copying them here would defeat the point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DATA_API = Path(__file__).resolve().parents[2] / "wisefood-data-api" / "src"


def _schemas():
    if not (DATA_API / "schemas" / "schemas.py").exists():
        pytest.skip("wisefood-data-api is not checked out beside this repo")
    if str(DATA_API) not in sys.path:
        sys.path.insert(0, str(DATA_API))
    return pytest.importorskip("schemas.schemas")


def _proposal(kind, **kw):
    from wisefood_mcp.stores import Proposal, new_proposal_id

    fields = dict(id=new_proposal_id(), kind=kind, title="Εθνικός Διατροφικός Οδηγός",
                  status="approved", licence="CC BY-NC-SA 4.0", country="Greece",
                  language="Greek", source_url="https://diatrofi.test/guide.pdf",
                  rationale="Fills the Greece gap")
    fields.update(kw)
    return Proposal(**fields)


#: proxy name per kind, as `_create` is called with.
PROXY = {"guide": "guides", "article": "articles",
         "textbook": "textbooks", "fctable": "fctables"}


def _posted(proposal, profile=None):
    """The body `_create` would actually send for this proposal."""
    from wisefood_mcp.licences import normalise_licence
    from wisefood_mcp.tools.writes import HAS_STATUS, _slug

    from integrator.executor import spec_for

    spec = spec_for(proposal, profile)
    spec.setdefault("url", proposal.source_url)
    spec["urn"] = _slug(spec.get("title") or proposal.title, proposal.id)
    spec["license"] = normalise_licence(proposal.licence)
    if PROXY[proposal.kind] in HAS_STATUS:
        spec.setdefault("status", "draft")
    return {k: v for k, v in spec.items() if v is not None}


def _model(schemas, kind):
    return {
        "guide": schemas.GuideCreationSchema,
        "article": schemas.ArticleCreationSchema,
        "textbook": schemas.TextbookCreationSchema,
        "fctable": schemas.FoodCompositionTableCreationSchema,
    }[kind]


@pytest.mark.parametrize("kind", ["guide", "article", "textbook", "fctable"])
def test_what_we_send_is_what_the_catalog_accepts(kind):
    schemas = _schemas()
    _model(schemas, kind)(**_posted(_proposal(kind)))


@pytest.mark.parametrize("kind", ["guide", "article", "textbook", "fctable"])
def test_a_proposal_that_knows_almost_nothing_still_validates(kind):
    """The floor: a title, a URL and a licence is all a curator has to have
    established. Everything else the spec fills in or leaves out."""
    schemas = _schemas()
    bare = _proposal(kind, country=None, language=None, rationale=None)
    _model(schemas, kind)(**_posted(bare))


def test_a_composition_table_carries_what_the_profiler_measured():
    schemas = _schemas()
    profile = {"number_of_entries": 1200, "nutrient_coverage": ["energy", "protein"]}
    body = _posted(_proposal("fctable"), profile)
    schemas.FoodCompositionTableCreationSchema(**body)
    assert body["number_of_entries"] == 1200


@pytest.mark.parametrize("kind", ["textbook", "fctable"])
def test_content_is_never_sent_to_a_kind_that_has_no_such_field(kind):
    """`guide_spec` was used for all three and added `content`. Neither of
    these declares it, and every schema is extra="forbid", so each of these
    integrations would have stopped at the create call."""
    schemas = _schemas()
    assert "content" not in _model(schemas, kind).model_fields
    assert "content" not in _posted(_proposal(kind))


def test_a_guide_does_send_content_because_its_schema_requires_it():
    schemas = _schemas()
    assert "content" in schemas.GuideCreationSchema.model_fields
    assert "content" in _posted(_proposal("guide"))


@pytest.mark.parametrize("kind", ["guide", "textbook", "fctable"])
def test_region_and_language_go_in_as_codes(kind):
    """"Greece" and "Greek" are refused with `String should have at most 2
    characters`."""
    body = _posted(_proposal(kind))
    assert body.get("language") == "el"
    if "region" in body:
        assert body["region"] == "GR"


def test_an_article_may_use_names_because_its_schema_allows_them():
    """Not every kind constrains these. An article takes free text in both,
    so discarding a name there would lose information for no reason."""
    schemas = _schemas()
    field = schemas.ArticleCreationSchema.model_fields["language"]
    assert field.annotation is not None
    schemas.ArticleCreationSchema(**_posted(_proposal("article")))


@pytest.mark.parametrize("kind", ["guide", "article", "textbook", "fctable"])
def test_no_extras_are_sent_to_a_schema_that_forbids_them(kind):
    """Only articles declare `extras`. Sending provenance to the others is a
    validation error, not a field quietly dropped."""
    schemas = _schemas()
    from wisefood_mcp.tools.writes import ACCEPTS_EXTRAS

    declares = "extras" in _model(schemas, kind).model_fields
    proxy = {"guide": "guides", "article": "articles",
             "textbook": "textbooks", "fctable": "fctables"}[kind]
    assert (proxy in ACCEPTS_EXTRAS) == declares, (
        f"{kind}: ACCEPTS_EXTRAS and the schema disagree")


@pytest.mark.parametrize("kind", ["guide", "article", "textbook", "fctable"])
def test_every_urn_we_build_matches_the_slug_pattern(kind):
    """A title in Greek reduces to nothing in ASCII, so the urn falls back to
    the proposal id rather than to an empty string the catalog rejects."""
    import re

    schemas = _schemas()
    for title in ("Εθνικός Διατροφικός Οδηγός", "Nutrition for Nurses (OpenStax)",
                  "—", "a" * 400):
        body = _posted(_proposal(kind, title=title))
        assert re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", body["urn"]), title
        assert len(body["urn"]) <= 100
        _model(schemas, kind)(**body)


@pytest.mark.parametrize("kind", ["guide", "textbook"])
def test_nothing_is_published_by_being_created(kind):
    """`GuideCreationSchema` defaults `status` to `active`, and the catalog
    refuses an active guide that nobody has verified — `entities/guides.py`
    raises "Guide must be verified before it can be published as active."
    So an integration asked for publication every time and was refused every
    time. Nothing the assistant brings in is published by being brought in.
    """
    _schemas()
    assert _posted(_proposal(kind))["status"] == "draft"


@pytest.mark.parametrize("kind", ["article", "fctable"])
def test_a_kind_with_no_status_field_is_never_sent_one(kind):
    schemas = _schemas()
    assert "status" not in _model(schemas, kind).model_fields
    assert "status" not in _posted(_proposal(kind))


@pytest.mark.parametrize("kind", ["guide", "textbook"])
def test_a_draft_does_not_need_a_verifier(kind):
    """The whole point of landing as a draft: the editorial gate applies to
    publishing, and this is not publishing."""
    schemas = _schemas()
    body = _posted(_proposal(kind))
    built = _model(schemas, kind)(**body)
    assert built.status == "draft"
    assert getattr(built, "review_status", "unreviewed") == "unreviewed"
