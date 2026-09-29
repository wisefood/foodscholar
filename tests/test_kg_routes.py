"""Graph routes, addressed by the ids the graph actually mints.

Theme ids are slash-separated — `{facet}/{shelf_slug}/{label_slug}_{pass}{seq}`
— and card ids embed them. The server decodes `%2F` before it routes, so a
plain `{node_id}` parameter never matched one: every theme and every card was a
404 raised before any handler ran, while shelves (`foodon:…`, no slash) worked.
The browser's tree opened and nothing inside it did.

These go through the real router with the browse index stubbed, and assert
which handler ran and with which id. A status code alone cannot tell "no such
node" from "no such route", and the second is the one that shipped.
"""
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1 import graph

SHELF = "foodon:FOODON_03411222"
THEME = "foods/olive_oil/monounsaturated_fat_r1"
CARD = f"card:theme:{THEME}"


@pytest.fixture
def calls(monkeypatch):
    """Which browse lookup ran, and with what id, in call order."""
    seen = []
    empty_page = {"items": [], "total": 0, "next_cursor": None}

    def record(name, result=None):
        def stub(node_id, *args, **kwargs):
            seen.append((name, node_id))
            return result
        return stub

    monkeypatch.setattr(graph.browse, "get_node", record("get_node"))
    monkeypatch.setattr(graph.browse, "children", record("children", empty_page))
    monkeypatch.setattr(graph.browse, "themes_for", record("themes_for", empty_page))
    monkeypatch.setattr(graph.browse, "card_for", record("card_for"))
    return seen


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(graph.router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("node_id", [SHELF, THEME, CARD], ids=["shelf", "theme", "card"])
@pytest.mark.parametrize("template,lookup", [
    ("/nodes/{}", "get_node"),
    ("/nodes/{}/children", "children"),
    ("/nodes/{}/themes", "themes_for"),
    ("/nodes/{}/breadcrumb", "get_node"),
    ("/nodes/{}/chunks", "get_node"),
    ("/cards/{}", "card_for"),
])
def test_the_id_reaches_its_handler_whole(client, calls, template, lookup, node_id):
    """Encoded the way the browser encodes it, the id arrives intact — and a
    sub-route is not mistaken for the bare node route with `/children` glued
    onto the id, which is what `path` would do if the order were wrong."""
    client.get("/api/v1/graph" + template.format(quote(node_id, safe="")))
    assert calls, f"{template} matched no route for {node_id!r}"
    assert calls[0] == (lookup, node_id)
