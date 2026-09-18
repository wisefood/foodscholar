"""Knowledge graph settings, where the deployment and the code have to agree.

These are cheap tests for a class of bug that is not cheap: a credential that
reads as the wrong string does not fail at startup — `get_graph()` is lazy —
it fails the first time somebody opens the graph tab, as a 503 that says the
stores could not be opened and nothing about why.
"""
import importlib
import sys

import pytest

sys.path.insert(0, "src")


@pytest.fixture()
def settings(monkeypatch):
    """A fresh Config.setup() under a controlled environment."""
    def build(**env):
        for key in list(env):
            monkeypatch.setenv(key, env[key])
        config_module = importlib.import_module("config")
        cfg = config_module.Config()
        cfg.setup()
        return cfg.settings
    return build


def _clear(monkeypatch):
    for key in ("KG_NEO4J_AUTH", "KG_NEO4J_USER", "KG_NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)


def test_the_graph_is_off_unless_a_deployment_says_otherwise(settings, monkeypatch):
    """A deployment that has never run the offline build should start clean and
    answer a readable 503, not fail one request at a time."""
    monkeypatch.delenv("KG_ENABLED", raising=False)
    assert settings()["KG_ENABLED"] is False


def test_neo4j_auth_is_split_into_user_and_password(settings, monkeypatch):
    """The platform stores ONE Neo4j credential: the `user/password` string
    Neo4j's own image is started with. Accepting that shape is what lets the
    deployment pass the secret through instead of keeping a second copy."""
    _clear(monkeypatch)
    s = settings(KG_NEO4J_AUTH="neo4j/s3cret")
    assert s["KG_NEO4J_USER"] == "neo4j"
    assert s["KG_NEO4J_PASSWORD"] == "s3cret"


def test_a_password_may_contain_the_separator(settings, monkeypatch):
    """Split on the FIRST slash only. A generated password containing one is
    otherwise silently truncated, and the failure is a 503 an hour later."""
    _clear(monkeypatch)
    s = settings(KG_NEO4J_AUTH="neo4j/a/b/c")
    assert s["KG_NEO4J_USER"] == "neo4j"
    assert s["KG_NEO4J_PASSWORD"] == "a/b/c"


def test_explicit_values_win_over_the_combined_one(settings, monkeypatch):
    """So a deployment that does hold them separately is not overridden."""
    _clear(monkeypatch)
    s = settings(
        KG_NEO4J_AUTH="neo4j/ignored",
        KG_NEO4J_USER="reader",
        KG_NEO4J_PASSWORD="other",
    )
    assert s["KG_NEO4J_USER"] == "reader"
    assert s["KG_NEO4J_PASSWORD"] == "other"


def test_no_auth_at_all_still_names_the_default_user(settings, monkeypatch):
    """An empty password is how the library is told there is none; an empty
    USER would be a connection attempt as nobody."""
    _clear(monkeypatch)
    s = settings()
    assert s["KG_NEO4J_USER"] == "neo4j"
    assert s["KG_NEO4J_PASSWORD"] == ""


def test_the_stream_depth_default_matches_what_the_interface_asks_for(settings):
    """The graph browser sends depth_max=2 explicitly so its depth control
    shows the truth on first paint. If this default moved, the two would
    disagree about what the map is showing."""
    assert settings()["KG_STREAM_DEFAULT_DEPTH"] == 2


def test_the_summary_model_separates_enabled_from_built():
    """An interface has to tell "switched off here" from "nobody has run the
    projector". Only the summary can answer the first: every other browse route
    reads the projected Elasticsearch index, which an index left behind by a
    deployment that later switched the graph off would still satisfy."""
    from models.kg import GraphSummary

    off = GraphSummary(enabled=False, built=False)
    assert off.enabled is False and off.built is False

    # Enabled but never projected — the case with a rebuild button behind it.
    not_built = GraphSummary(built=False)
    assert not_built.enabled is True and not_built.built is False
