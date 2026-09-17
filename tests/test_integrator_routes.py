"""The integrator's HTTP surface, checked against the decorator it uses.

Every route in this service is wrapped in `@render()`, which builds the
success envelope from the request — and raises at *call* time, not import
time, if the endpoint does not accept one. So a route missing
`request: Request` imports cleanly, registers cleanly, passes every test that
exercises the service layer, and 500s on the first real call.

That is exactly what shipped: all fifteen integrator routes were written
without it, and nothing caught them because the suite called
`integrator.service` directly. These tests look at the signatures the
decorator will inspect, so the next route written without a request fails
here rather than in production.
"""
from __future__ import annotations

import inspect

import pytest

from fastapi import Request


#: `render()`'s wrapper closes over exactly these. Identifying the decorator
#: this way rather than by name is what lets the check cover every router: a
#: route that is not render-wrapped has no such requirement, and asserting it
#: anyway would flag most of `enrich` and `qa`, which are correct as written.
_RENDER_FREEVARS = {"event", "func", "is_coro", "logger", "map_result"}


def _is_render_wrapped(endpoint) -> bool:
    code = getattr(endpoint, "__code__", None)
    return bool(code and _RENDER_FREEVARS.issubset(set(code.co_freevars)))


def _endpoints():
    """Every `@render()`-decorated route across the API, with its path."""
    import importlib

    out = []
    for name in ("integrator", "guidelines", "enrich", "qa", "search", "sessions"):
        try:
            module = importlib.import_module(f"api.v1.{name}")
        except Exception:  # noqa: BLE001 — a router needing config we lack
            continue
        for route in getattr(module, "router", None).routes if getattr(
                module, "router", None) else []:
            endpoint = getattr(route, "endpoint", None)
            # functools.wraps means the signature FastAPI and the decorator
            # both inspect is the wrapper's — which is the one that was wrong.
            if endpoint is not None and _is_render_wrapped(endpoint):
                out.append((f"{name}{getattr(route, 'path', '?')}", endpoint))
    return out


def test_render_decorated_routes_are_found():
    """If this finds nothing, the decorator changed shape and the check below
    is silently passing over everything."""
    found = _endpoints()
    assert found, "no @render() routes found — the detection is broken"
    assert any(path.startswith("integrator") for path, _ in found)


@pytest.mark.parametrize("path,endpoint", _endpoints(),
                         ids=[p for p, _ in _endpoints()])
def test_every_route_accepts_the_request_render_needs(path, endpoint):
    """`render()` raises RuntimeError without it, on every single call."""
    parameters = inspect.signature(endpoint).parameters
    named = [name for name, p in parameters.items()
             if p.annotation is Request or name == "request"]
    assert named, (
        f"{path} takes no `request: Request`; @render() will raise "
        f"'endpoint must accept a request parameter' on the first call")


def test_render_actually_rejects_an_endpoint_without_one():
    """The premise, asserted rather than assumed — if `render()` stops caring,
    the test above is guarding nothing and should be deleted."""
    import asyncio

    from routers.generic import render

    @render()
    async def no_request(thing: str):
        return {"thing": thing}

    with pytest.raises(RuntimeError, match="request"):
        asyncio.run(no_request(thing="x"))
