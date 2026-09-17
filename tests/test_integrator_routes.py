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


def test_the_streaming_route_is_not_render_wrapped():
    """It returns text/event-stream, and `@render()` builds a JSON envelope.

    Wrapping it would either corrupt the frames or fail on the first call, so
    this asserts the exception is deliberate rather than an oversight — the
    check above would otherwise make it look like one.
    """
    from api.v1 import integrator

    stream = [r for r in integrator.router.routes
              if getattr(r, "path", "").endswith("/chat/stream")]
    assert stream, "the streaming route disappeared"
    assert not _is_render_wrapped(stream[0].endpoint)


def test_the_streamed_turn_emits_each_step_as_it_happens():
    """The point of the stream: a step is visible when it starts, not when
    the whole turn ends a minute later."""
    import json

    from wisefood_mcp import ToolContext

    from integrator.agent import Budget, IntegratorAgent

    class Registry:
        def openai_schemas(self, include_writes=False):
            return [{"type": "function", "function": {"name": "research"}}]

        def call(self, name, args, ctx):
            return {"ok": True, "result": {"findings": [{"title": "x"}],
                                           "tools_used": ["search"]}}

    class Groq:
        def __init__(self):
            self.replies = [
                {"choices": [{"message": {"content": "", "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "research",
                                 "arguments": json.dumps({"query": "bulgaria"})}}]}}],
                 "usage": {"total_tokens": 5}},
                {"choices": [{"message": {"content": "Here you go."}}],
                 "usage": {"total_tokens": 5}},
            ]
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            data = self.replies.pop(0)
            return type("R", (), {"model_dump": lambda _self: data})()

    seen = []
    agent = IntegratorAgent(registry=Registry(),
                            tool_context=ToolContext(proposal_store=None),
                            groq_client=Groq(), budget=Budget(max_steps=10))
    outcome = agent.run_streamed([], "find a guide",
                                 lambda name, payload: seen.append((name, payload)))

    assert outcome["reply"] == "Here you go."
    names = [s[1]["title"] for s in seen if s[0] == "step"]
    assert "Searching the web" in names
    # Started and finished are both emitted, so a watcher sees it run.
    statuses = [s[1]["status"] for s in seen if s[0] == "step"]
    assert "running" in statuses and "done" in statuses


def test_a_listener_that_raises_does_not_kill_the_turn():
    """A browser that hung up is not a reason to abandon a turn already
    spending tokens — and the turn still has to be persisted."""
    from wisefood_mcp import ToolContext

    from integrator.agent import Budget, IntegratorAgent

    class Registry:
        def openai_schemas(self, include_writes=False):
            return []

        def call(self, name, args, ctx):
            return {"ok": True, "result": {}}

    class Groq:
        def __init__(self):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            return type("R", (), {"model_dump": lambda _s: {
                "choices": [{"message": {"content": "Answer."}}],
                "usage": {"total_tokens": 1}}})()

    agent = IntegratorAgent(registry=Registry(),
                            tool_context=ToolContext(proposal_store=None),
                            groq_client=Groq(), budget=Budget(max_steps=5))
    outcome = agent.run_streamed([], "hello", lambda *_a: 1 / 0)
    assert outcome["reply"] == "Answer."
