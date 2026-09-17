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
import json

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

    from wisefood_mcp import ToolContext

    from integrator.agent import Budget, IntegratorAgent

    class Registry:
        def openai_schemas(self, include_writes=False):
            return [{"type": "function", "function": {"name": "research"}}]

        def call(self, name, args, ctx):
            return {"ok": True, "result": {"findings": [{"title": "x"}],
                                           "tools_used": ["search"]}}

    class Groq:
        """A provider that streams, because a watched turn now asks it to."""

        def __init__(self):
            # Round one calls a tool, with the arguments split across frames
            # the way a real provider splits them. Round two writes an answer.
            self.replies = [
                [_delta(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "research"}}]),
                 _delta(tool_calls=[{"index": 0, "function": {
                     "arguments": '{"query": "bul'}}]),
                 _delta(tool_calls=[{"index": 0, "function": {
                     "arguments": 'garia"}'}}]),
                 _usage(5)],
                [_delta(content="Here "), _delta(content="you go."), _usage(5)],
            ]
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            assert kw.get("stream") is True, "a watched turn must be streamed"
            return iter(self.replies.pop(0))

    seen = []
    agent = IntegratorAgent(registry=Registry(),
                            tool_context=ToolContext(proposal_store=None),
                            groq_client=Groq(), budget=Budget(max_steps=10))
    outcome = agent.run_streamed([], "find a guide",
                                 lambda name, payload: seen.append((name, payload)))

    assert outcome["reply"] == "Here you go."
    # The answer arrives in pieces as it is written, not whole at the end.
    assert "".join(s[1]["delta"] for s in seen if s[0] == "text") == "Here you go."
    # And the fragmented tool arguments were put back together, or the tool
    # would have been called with a truncated query.
    assert outcome["messages"][1]["content"]
    names = [s[1]["title"] for s in seen if s[0] == "step"]
    assert "Searching the web" in names
    # Started and finished are both emitted, so a watcher sees it run.
    statuses = [s[1]["status"] for s in seen if s[0] == "step"]
    assert "running" in statuses and "done" in statuses


def _delta(content=None, tool_calls=None):
    """One streamed frame, in the shape the provider sends it."""
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    frame = {"choices": [{"delta": delta}]}
    return type("Chunk", (), {"model_dump": lambda _s, f=frame: f})()


def _usage(tokens):
    """The final frame, which carries the usage and no choice."""
    frame = {"choices": [], "usage": {"total_tokens": tokens}}
    return type("Chunk", (), {"model_dump": lambda _s, f=frame: f})()


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
            return iter([_delta(content="Answer."), _usage(1)])

    agent = IntegratorAgent(registry=Registry(),
                            tool_context=ToolContext(proposal_store=None),
                            groq_client=Groq(), budget=Budget(max_steps=5))
    outcome = agent.run_streamed([], "hello", lambda *_a: 1 / 0)
    assert outcome["reply"] == "Answer."


# ---------------------------------------------------------------- streaming --

class TestStreamedToolCalls:
    """Putting a streamed tool call back together.

    This is the one part of streaming that can change *what runs*: a call is
    sent as an id and a name once, then its arguments a few characters at a
    time, several calls interleaved and identified only by index. Getting it
    wrong does not fail loudly — it hands the loop a truncated argument
    string, which parses as `{}` and silently calls the tool with nothing.
    """

    def _merge(self, *frames):
        from integrator.agent import _merge_tool_call_deltas
        calls = {}
        for frame in frames:
            _merge_tool_call_deltas(calls, frame)
        return calls

    def test_arguments_split_across_frames_are_concatenated(self):
        calls = self._merge(
            [{"index": 0, "id": "c1", "type": "function",
              "function": {"name": "research", "arguments": ""}}],
            [{"index": 0, "function": {"arguments": '{"query": "Bul'}}],
            [{"index": 0, "function": {"arguments": 'garia FBDG"}'}}],
        )
        assert calls[0]["id"] == "c1"
        assert calls[0]["function"]["name"] == "research"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "query": "Bulgaria FBDG"}

    def test_two_calls_interleaved_stay_apart(self):
        # The provider does not finish one call before starting the next, and
        # the index is the only thing that says which is which.
        calls = self._merge(
            [{"index": 0, "id": "a", "function": {"name": "fetch_url"}},
             {"index": 1, "id": "b", "function": {"name": "research"}}],
            [{"index": 0, "function": {"arguments": '{"url": "'}},
             {"index": 1, "function": {"arguments": '{"query": "'}}],
            [{"index": 1, "function": {"arguments": 'greece"}'}},
             {"index": 0, "function": {"arguments": 'https://x.test"}'}}],
        )
        assert json.loads(calls[0]["function"]["arguments"])["url"] == "https://x.test"
        assert json.loads(calls[1]["function"]["arguments"])["query"] == "greece"

    def test_a_name_arriving_in_pieces_is_not_truncated(self):
        # Rarer, but a provider is allowed to split the name too, and the name
        # is what selects the tool.
        calls = self._merge(
            [{"index": 0, "id": "c", "function": {"name": "licence_"}}],
            [{"index": 0, "function": {"name": "evidence"}}],
        )
        assert calls[0]["function"]["name"] == "licence_evidence"

    def test_an_unstreamed_turn_stays_unstreamed(self):
        """Nobody watching means nothing to show deltas to, and one response
        is less to go wrong."""
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        seen = []

        class Registry:
            def openai_schemas(self, include_writes=False):
                return []

            def call(self, name, args, ctx):
                return {"ok": True, "result": {}}

        class Groq:
            def __init__(self):
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                seen.append(kw)
                return type("R", (), {"model_dump": lambda _s: {
                    "choices": [{"message": {"content": "Done."}}],
                    "usage": {"total_tokens": 1}}})()

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=5))
        assert agent.run([], "anything")["reply"] == "Done."
        assert seen[0].get("stream") is None


class TestHistoryIsCompacted:
    """Old tool results are shrunk before the conversation is re-sent.

    The turn that prompted this spent 120,000 tokens on one question without
    doing anything expensive: an 18,000-character PDF fetch was re-sent on
    every later step of the turn and again on every later turn of the
    session, so the same text was billed a dozen times over.
    """

    def _history(self, results):
        out = []
        for i, text in enumerate(results):
            out.append({"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{i}"}]})
            out.append({"role": "tool", "tool_call_id": f"c{i}",
                        "content": text})
        return out

    def test_the_recent_results_are_untouched(self):
        from integrator.agent import compact_history

        history = self._history(["a" * 5_000, "b" * 5_000])
        assert compact_history(history, keep_full=3) == history

    def test_an_older_result_is_cut_down(self):
        from integrator.agent import compact_history

        history = self._history(["a" * 5_000, "b" * 20, "c" * 20])
        out = compact_history(history, keep_full=2)
        assert len(out[1]["content"]) < 700
        assert out[1]["content"].startswith("a" * 400)
        # And it says the detail is gone, so the model does not spend a step
        # fetching it again.
        assert "will not restore" in out[1]["content"]
        # The recent two are left alone.
        assert out[3]["content"] == "b" * 20 and out[5]["content"] == "c" * 20

    def test_nothing_is_dropped_so_every_call_keeps_its_result(self):
        """A tool result whose matching call has gone is a 400 from the
        provider — a worse bug than a large bill."""
        from integrator.agent import compact_history

        history = self._history(["x" * 9_000 for _ in range(6)])
        out = compact_history(history, keep_full=1)
        assert len(out) == len(history)
        assert [m["role"] for m in out] == [m["role"] for m in history]
        assert [m.get("tool_call_id") for m in out] \
            == [m.get("tool_call_id") for m in history]

    def test_the_transcript_that_gets_persisted_is_not_compacted(self):
        """What is written down should be what happened, not what we could
        afford to re-send."""
        from integrator.agent import compact_history

        history = self._history(["a" * 9_000, "b" * 9_000, "c" * 9_000])
        compact_history(history, keep_full=1)
        assert history[1]["content"] == "a" * 9_000

    def test_a_short_result_is_left_as_it_is(self):
        from integrator.agent import compact_history

        history = self._history(["small", "b" * 9_000, "c" * 9_000])
        assert compact_history(history, keep_full=1)[1]["content"] == "small"


class TestProposingIsReachable:
    """Validation and visibility, neither of which needs a database."""

    def _ctx(self):
        from wisefood_mcp import ToolContext
        return ToolContext(proposal_store=None, actor="expert-1", extra={})

    def test_the_tool_is_visible_without_writes_being_enabled(self):
        """Proposing is asking permission, not writing to the catalog. Gating
        it behind the writes switch is what left the agent able to research a
        source and with no way to put it in front of anybody."""
        from integrator.service import _REGISTRY

        names = [s["function"]["name"]
                 for s in _REGISTRY.openai_schemas(include_writes=False)]
        assert "propose_source" in names

    def test_a_licence_without_evidence_is_refused(self):
        """A claimed licence with nothing behind it scores well and reads as
        established, which is worse than an honest gap."""
        from wisefood_mcp.registry import ToolError

        from integrator.propose import propose_source

        with pytest.raises(ToolError) as caught:
            propose_source(self._ctx(), kind="guide", title="Some guide",
                           licence="CC-BY-4.0")
        assert "evidence" in str(caught.value)

    def test_a_kind_that_does_not_exist_is_refused(self):
        from wisefood_mcp.registry import ToolError

        from integrator.propose import propose_source

        with pytest.raises(ToolError):
            propose_source(self._ctx(), kind="dataset", title="Something")

    def test_the_step_timeline_has_a_label_for_it(self):
        """Without one the timeline shows the raw tool name to a curator."""
        from integrator.steps import RUNNING, finished_detail

        assert "propose_source" in RUNNING
        assert finished_detail("propose_source", {}, True,
                               {"filed": "X is now in the panel"}, None) \
            == "X is now in the panel"
