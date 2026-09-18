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


def test_a_provider_that_rejects_stream_options_still_streams():
    """Not every provider — or proxy in front of one — takes the option. A
    turn that dies on an unknown parameter is worse than a turn whose tokens
    are estimated."""
    from wisefood_mcp import ToolContext

    from integrator.agent import Budget, IntegratorAgent

    class Registry:
        def openai_schemas(self, include_writes=False):
            return []

        def call(self, name, args, ctx):
            return {"ok": True, "result": {}}

    class Groq:
        def __init__(self):
            self.attempts = []
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            self.attempts.append(kw)
            if "stream_options" in kw:
                raise ValueError("unrecognized request argument: stream_options")
            return iter([_delta(content="Answer.")])

    groq = Groq()
    agent = IntegratorAgent(registry=Registry(),
                            tool_context=ToolContext(proposal_store=None),
                            groq_client=groq, budget=Budget(max_steps=5))
    outcome = agent.run_streamed([], "hello", lambda *_a: None)

    assert outcome["reply"] == "Answer."
    assert len(groq.attempts) == 2, "it retried without the option"
    # No usage frame came back, so the budget charges an estimate. A step that
    # costs nothing is a step the budget cannot stop.
    assert outcome["tokens"] > 0


class TestAnAuditRecordSurvivesAwkwardBytes:
    """Postgres refuses `\\u0000` in jsonb — it cannot be converted to text —
    and the whole insert fails with it. Fetching a PDF whose bytes do not
    decode cleanly puts a raw gzip header into a snippet, and the audit
    record for that call was being dropped with a warning. Losing the record
    of a tool call over one byte in its output defeats the audit trail."""

    def test_nul_is_stripped_from_every_corner(self):
        from integrator.store import _storable

        out = _storable({
            "query": "Greece guidelines",
            "results": [{"snippet": "\x1f�\b\x00binary\x00"}],
            "nested": {"deep": ["a\x00b"]},
            "count": 3,
            "ok": True,
            "nothing": None,
        })
        assert "\x00" not in str(out)
        assert out["results"][0]["snippet"] == "\x1f�\bbinary"
        assert out["nested"]["deep"] == ["ab"]
        # Everything else is left exactly as it was.
        assert out["count"] == 3 and out["ok"] is True and out["nothing"] is None

    def test_text_without_nul_is_untouched(self):
        from integrator.store import _storable

        value = {"quote": "Creative Commons — Attribution 4.0", "n": 1.5}
        assert _storable(value) == value


class TestWhatCountsAsTheSameCall:
    """Comparing arguments literally is too literal.

    One real run opened the same WHO PDF six times, each call asking for a
    different page, and searched the web five times for the same thing in
    five wordings. Every one was a fresh key and a fresh bill.
    """

    def test_the_same_document_is_the_same_call_whatever_page_is_asked_for(self):
        from integrator.agent import call_key

        keys = {
            call_key("fetch_url", {"url": "https://who.int/a.pdf", "page": 1,
                                   "max_chars": 2000}),
            call_key("fetch_url", {"url": "https://who.int/a.pdf", "page": 40,
                                   "max_chars": 5000}),
            call_key("fetch_url", {"url": "https://who.int/a.pdf/"}),
        }
        assert len(keys) == 1

    def test_a_search_reworded_is_the_same_search(self):
        from integrator.agent import call_key

        keys = {
            call_key("research", {"query": "Bulgaria national dietary guidelines PDF"}),
            call_key("research", {"query": "Bulgaria dietary guidelines 2020"}),
            call_key("research", {"query": "bulgaria  DIETARY guidelines!"}),
        }
        assert len(keys) == 1

    def test_a_genuinely_different_search_is_not_collapsed(self):
        """The guard must not silence a real second question — asking about
        children after asking about adults is new work."""
        from integrator.agent import call_key

        assert call_key("research", {"query": "Bulgaria dietary guidelines"}) \
            != call_key("research", {"query": "Bulgaria dietary guidelines children"})
        assert call_key("research", {"query": "Greece food composition table"}) \
            != call_key("research", {"query": "Bulgaria dietary guidelines"})

    def test_a_doi_is_matched_regardless_of_case(self):
        from integrator.agent import call_key

        assert call_key("doi_metadata", {"doi": "10.1186/S12937-026-01386-8"}) \
            == call_key("doi_metadata", {"doi": "10.1186/s12937-026-01386-8"})

    def test_anything_else_still_compares_its_arguments(self):
        from integrator.agent import call_key

        assert call_key("search_catalog", {"kind": "guide", "q": "a"}) \
            != call_key("search_catalog", {"kind": "guide", "q": "b"})
        # Argument order is not a difference.
        assert call_key("search_catalog", {"kind": "guide", "q": "a"}) \
            == call_key("search_catalog", {"q": "a", "kind": "guide"})


class TestTheRecipeTransportCarriesTheCaller:
    """An import the curator could not start by hand must not be one the
    assistant can start for them."""

    def test_the_callers_token_is_sent(self, monkeypatch):
        import integrator.service as service

        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL",
                            "http://wisefood-api:8000")
        post, get = service._recipes_transport("tok-abc")
        assert post is not None and get is not None

        seen = {}

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None, headers=None):
                seen.update(url=url, headers=headers, body=json)
                return type("R", (), {"raise_for_status": lambda _s: None,
                                      "json": lambda _s: {"run": {"id": "r1"}}})()

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeClient)
        assert post("/api/v1/recipewrangler/ingest/source", {"x": 1})["run"]["id"] == "r1"
        assert seen["headers"]["Authorization"] == "Bearer tok-abc"
        assert seen["url"].startswith("http://wisefood-api:8000/")

    def test_no_token_means_no_transport(self, monkeypatch):
        """Not a fallback to the service account: the tools then report that
        the importer is not configured, which is the honest answer."""
        import integrator.service as service

        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL",
                            "http://wisefood-api:8000")
        assert service._recipes_transport(None) == (None, None)
        assert service._recipes_transport("") == (None, None)

    def test_no_gateway_configured_means_no_transport(self, monkeypatch):
        import integrator.service as service

        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL", "")
        assert service._recipes_transport("tok") == (None, None)

    def test_the_gateway_is_used_not_the_catalog(self, monkeypatch):
        """The admin/expert check on these routes lives on the gateway, so
        going straight to the service would step around it."""
        import integrator.service as service

        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL",
                            "http://wisefood-api:8000")
        monkeypatch.setitem(service.config.settings, "DATA_API_URL",
                            "http://data-catalog:8000")
        seen = {}

        class FakeClient:
            def __init__(self, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url, headers=None):
                seen["url"] = url
                return type("R", (), {"raise_for_status": lambda _s: None,
                                      "json": lambda _s: {}})()

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeClient)
        _post, get = service._recipes_transport("tok")
        get("/api/v1/recipewrangler/ingest/source/runs/r1")
        assert "wisefood-api" in seen["url"] and "data-catalog" not in seen["url"]


class TestSuggestingRatherThanFiling:
    """The assistant printed a proposal as JSON and then said it had filed
    it. Neither half was any use: a code block is not something a curator can
    act on, and the proposal it claimed to have made did not exist, so
    looking for it was wasted time."""

    def _ctx(self):
        from wisefood_mcp import ToolContext
        return ToolContext(proposal_store=None, actor="expert-1", extra={})

    def test_a_suggestion_writes_nothing(self):
        from integrator.propose import suggest_source

        out = suggest_source(
            self._ctx(), kind="guide",
            title="Нутриционни препоръки за деца 3-7 г.",
            source_url="https://www.namama.bg/upload/Deca_3-7.pdf",
            country="Bulgaria", language="bg", population_group="children",
            rationale="Commercial brochure repeating official advice.")

        assert out["suggested"] is True
        assert out["suggestion"]["title"].startswith("Нутриционни")
        assert out["suggestion"]["population_group"] == "children"
        # No id, because nothing was filed. The whole point.
        assert "proposal_id" not in out

    def test_empty_fields_are_left_out(self):
        """The card renders what it is given; a licence of null should not
        arrive as a badge saying null."""
        from integrator.propose import suggest_source

        out = suggest_source(self._ctx(), kind="article", title="A paper")
        assert set(out["suggestion"]) == {"kind", "title"}

    def test_a_bad_kind_is_refused_the_same_as_filing(self):
        from wisefood_mcp.registry import ToolError

        from integrator.propose import suggest_source

        with pytest.raises(ToolError):
            suggest_source(self._ctx(), kind="dataset", title="Something")

    def test_the_suggestion_rides_on_its_step(self):
        """The console reads it off the step, which is already persisted and
        already streamed — no new column, no second request."""
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        suggestion = {"kind": "guide", "title": "Namama brochure"}

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "suggest_source"}}]

            def call(self, name, args, ctx):
                return {"ok": True, "result": {"suggested": True,
                                               "suggestion": suggestion}}

        class Groq:
            def __init__(self):
                self.replies = [
                    [_delta(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                         "function": {"name": "suggest_source",
                                                      "arguments": "{}"}}]),
                     _usage(1)],
                    [_delta(content="Yours to take or leave."), _usage(1)],
                ]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=6))
        out = agent.run_streamed([], "anything", lambda *_a: None)

        carried = [s for s in out["timeline"] if s.get("data", {}).get("suggestion")]
        assert carried, "the console has nothing to render the button from"
        assert carried[0]["data"]["suggestion"]["title"] == "Namama brochure"

    def test_the_timeline_says_it_is_the_curators_call(self):
        from integrator.steps import RUNNING, finished_detail

        assert "suggest_source" in RUNNING
        assert finished_detail(
            "suggest_source", {}, True,
            {"suggestion": {"title": "Namama brochure"}}, None
        ) == "Namama brochure — yours to take or leave"


class TestWhatATurnFiledIsRecordedNotNarrated:
    """The assistant reported eight filings for five calls, then produced ids
    for the difference when challenged, then invented a rule about titles and
    licences to explain it. The database knew all along; nothing showed it."""

    def _agent(self, filings, registry_result=None):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "propose_source"}}]

            def call(self, name, args, ctx):
                # The loop hands the registry the model's raw argument string.
                parsed = json.loads(args) if isinstance(args, str) else args
                return registry_result or {
                    "ok": True,
                    "result": {"proposal_id": f"id-{parsed.get('title', '?')}",
                               "status": "proposed"}}

        class Groq:
            def __init__(self):
                calls = [_delta(tool_calls=[{
                    "index": i, "id": f"c{i}", "type": "function",
                    "function": {"name": "propose_source",
                                 "arguments": json.dumps(
                                     {"kind": "article", "title": t})}}])
                    for i, t in enumerate(filings)]
                self.replies = [[*calls, _usage(1)],
                                [_delta(content="I filed eight."), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=20))
        return agent.run_streamed([], "search more", lambda *_a: None)

    def _filed(self, outcome):
        return [s["data"]["filed"] for s in outcome["timeline"]
                if s.get("data", {}).get("filed")]

    def test_the_record_counts_calls_not_claims(self):
        """Five calls, a reply claiming eight: the record says five."""
        out = self._agent(["A", "B", "C", "D", "E"])
        assert out["reply"] == "I filed eight."
        filed = self._filed(out)
        assert len(filed) == 5
        assert [f["title"] for f in filed] == ["A", "B", "C", "D", "E"]

    def test_each_entry_carries_the_id_the_tool_returned(self):
        """Not one the model wrote: an id it invented sends a curator looking
        for a proposal that does not exist."""
        out = self._agent(["Planetary Health Diet"])
        assert self._filed(out)[0]["proposal_id"] == "id-Planetary Health Diet"

    def test_a_failed_call_files_nothing(self):
        out = self._agent(["A"], registry_result={
            "ok": False, "error": {"message": "a licence needs evidence"}})
        assert self._filed(out) == []

    def test_a_call_that_returns_no_id_files_nothing(self):
        out = self._agent(["A"], registry_result={"ok": True, "result": {}})
        assert self._filed(out) == []

    def test_a_turn_that_filed_nothing_has_no_record(self):
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
                return iter([_delta(content="I filed five."), _usage(1)])

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=5))
        out = agent.run_streamed([], "anything", lambda *_a: None)
        assert self._filed(out) == []


class TestATransientFailureCanBeTriedAgain:
    """A 28 MB PDF died mid-transfer, and every retry after it came back
    "already run this turn" with the same error — the repeat guard was
    remembering failures, so one dropped connection became permanent for the
    rest of the turn."""

    def _agent(self, outcomes):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        calls = []
        queue = list(outcomes)

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "fetch_url"}}]

            def call(self, name, args, ctx):
                calls.append(args)
                return queue.pop(0) if queue else {"ok": True, "result": {}}

        class Groq:
            def __init__(self):
                one = [_delta(tool_calls=[{
                    "index": 0, "id": "c", "type": "function",
                    "function": {"name": "fetch_url",
                                 "arguments": json.dumps({"url": "https://g/KIDS.pdf"})}}]),
                    _usage(1)]
                self.replies = [list(one) for _ in range(4)] + [
                    [_delta(content="done"), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=8))
        return agent.run_streamed([], "read it", lambda *_a: None), calls

    def test_a_failure_does_not_become_the_permanent_answer(self):
        fail = {"ok": False, "error": {"message": "peer closed connection"}}
        good = {"ok": True, "result": {"pending_artifact": "p1"}}
        _out, calls = self._agent([fail, good])
        assert len(calls) >= 2, "the second attempt actually ran"

    def test_a_call_that_keeps_failing_still_stops(self):
        """Retrying is not looping: the third identical failure is the answer."""
        from integrator.agent import MAX_ATTEMPTS

        fail = {"ok": False, "error": {"message": "gone"}}
        _out, calls = self._agent([fail] * 6)
        assert len(calls) <= MAX_ATTEMPTS

    def test_a_success_is_still_only_run_once(self):
        good = {"ok": True, "result": {"pending_artifact": "p1"}}
        _out, calls = self._agent([good] * 6)
        assert len(calls) == 1


class TestACallThatCannotSucceedIsNotRepeated:
    """A refused argument is permanent: the same call makes the same refusal.
    Retrying spends a step to be told the same thing — and the model, seeing a
    fresh error rather than "you already asked", tends to try a third time."""

    def test_argument_refusals_are_permanent(self):
        from integrator.agent import is_permanent

        for code in ("unknown_kind", "unknown_licence", "not_a_doi",
                     "approval_required", "writes_disabled",
                     "licence_forbids_content", "journal_not_found"):
            assert is_permanent({"error": {"code": code}}), code

    def test_a_contract_violation_is_permanent(self):
        """Nothing about the world will make the same arguments validate."""
        from integrator.agent import is_permanent

        assert is_permanent({"error": {"code": "tool_error",
                                       "problems": [{"loc": ["q"]}]}})

    def test_a_dropped_connection_is_not_permanent(self):
        from integrator.agent import is_permanent

        assert not is_permanent({"error": {
            "code": "tool_error",
            "message": "RemoteProtocolError: peer closed connection"}})

    def test_a_shapeless_failure_is_treated_as_worth_one_retry(self):
        from integrator.agent import is_permanent

        assert not is_permanent({"error": {}})
        assert not is_permanent({})

    def test_a_permanent_failure_is_answered_from_the_record(self):
        """The second identical call gets "you already asked" rather than a
        fresh-looking error to react to."""
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        calls = []

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "propose_source"}}]

            def call(self, name, args, ctx):
                calls.append(args)
                return {"ok": False, "error": {"code": "unknown_kind",
                                               "message": "'artifact' is not a kind"}}

        class Groq:
            def __init__(self):
                one = [_delta(tool_calls=[{
                    "index": 0, "id": "c", "type": "function",
                    "function": {"name": "propose_source",
                                 "arguments": json.dumps(
                                     {"kind": "artifact", "title": "KIDS.pdf"})}}]),
                    _usage(1)]
                self.replies = [list(one) for _ in range(4)] + [
                    [_delta(content="done"), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=8))
        agent.run_streamed([], "file it", lambda *_a: None)
        assert len(calls) == 1, "one refusal is enough"


class TestTheLoopDoesNotGetStuck:
    """Ways a turn can be spent without going anywhere. Each of these has a
    bound, because the alternative is a curator watching a spinner until the
    token budget runs out."""

    def _agent(self, replies, registry_call=None, max_steps=8):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        calls = []

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "research"}}]

            def call(self, name, args, ctx):
                calls.append((name, args))
                if registry_call:
                    return registry_call(name, args)
                return {"ok": True, "result": {"findings": []}}

        class Groq:
            def __init__(self):
                self.replies = list(replies)
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                if not self.replies:
                    return iter([_delta(content="done"), _usage(1)])
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=max_steps))
        return agent, calls

    def _search(self, query):
        return [_delta(tool_calls=[{
            "index": 0, "id": f"c{abs(hash(query)) % 999}", "type": "function",
            "function": {"name": "research",
                         "arguments": json.dumps({"query": query})}}]), _usage(1)]

    def test_a_model_that_only_ever_calls_tools_hits_the_step_budget(self):
        agent, calls = self._agent([self._search(f"q{i}") for i in range(50)],
                                   max_steps=5)
        out = agent.run_streamed([], "search", lambda *_a: None)
        assert len(calls) <= 5
        # The reason is a sentence, because a curator reads it.
        assert "limit" in out["stop_reason"]
        assert out["reply"], "it still says something to the curator"

    def test_the_same_search_reworded_is_not_paid_for_twice(self):
        agent, calls = self._agent([
            self._search("Bulgaria national dietary guidelines PDF"),
            self._search("Bulgaria dietary guidelines 2020"),
            self._search("bulgaria DIETARY guidelines"),
        ])
        agent.run_streamed([], "search", lambda *_a: None)
        assert len(calls) == 1

    def test_a_tool_that_does_not_exist_does_not_end_the_turn(self):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        class Registry:
            def openai_schemas(self, include_writes=False):
                return []

            def call(self, name, args, ctx):
                return {"ok": False, "error": {"code": "unknown_tool",
                                               "message": f"no tool {name!r}"}}

        class Groq:
            def __init__(self):
                self.replies = [
                    [_delta(tool_calls=[{
                        "index": 0, "id": "c", "type": "function",
                        "function": {"name": "invented_tool", "arguments": "{}"}}]),
                     _usage(1)],
                    [_delta(content="I could not do that."), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=6))
        out = agent.run_streamed([], "do it", lambda *_a: None)
        assert out["reply"] == "I could not do that."

    def test_arguments_that_are_not_json_do_not_end_the_turn(self):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "research"}}]

            def call(self, name, args, ctx):
                return {"ok": True, "result": {}}

        class Groq:
            def __init__(self):
                self.replies = [
                    [_delta(tool_calls=[{
                        "index": 0, "id": "c", "type": "function",
                        "function": {"name": "research",
                                     "arguments": "{not json at all"}}]),
                     _usage(1)],
                    [_delta(content="recovered"), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=6))
        out = agent.run_streamed([], "go", lambda *_a: None)
        assert out["reply"] == "recovered"

    def test_a_turn_that_calls_nothing_still_answers(self):
        agent, calls = self._agent([[_delta(content="Here is what I know."), _usage(1)]])
        out = agent.run_streamed([], "tell me", lambda *_a: None)
        assert out["reply"] == "Here is what I know." and calls == []


class TestFilingAndSuggestingEdges:
    def _ctx(self):
        from wisefood_mcp import ToolContext
        return ToolContext(proposal_store=None, actor="expert-1", extra={})

    def test_a_title_that_is_only_whitespace(self):
        from wisefood_mcp.registry import ToolError

        from integrator.propose import suggest_source

        for empty in ("", "   ", "\n\t", "\u00a0"):
            with pytest.raises(ToolError):
                suggest_source(self._ctx(), kind="guide", title=empty)

    def test_a_title_in_another_alphabet_survives(self):
        from integrator.propose import suggest_source

        greek = "Εθνικός Διατροφικός Οδηγός για ενήλικες"
        out = suggest_source(self._ctx(), kind="guide", title=f"  {greek}  ")
        assert out["suggestion"]["title"] == greek

    def test_a_licence_a_page_wrote_is_normalised_in_a_suggestion_too(self):
        from integrator.propose import suggest_source

        out = suggest_source(self._ctx(), kind="guide", title="A guide",
                             licence="CC BY-NC-SA 4.0")
        assert out["suggestion"]["licence"] == "CCBYNCSA"

    def test_a_licence_nobody_can_place_is_refused_with_the_list(self):
        from wisefood_mcp.registry import ToolError

        from integrator.propose import propose_source

        with pytest.raises(ToolError) as caught:
            propose_source(self._ctx(), kind="guide", title="A guide",
                           licence="Free for everyone", licence_evidence=[{"x": 1}])
        assert caught.value.detail.get("code") == "unknown_licence"
        assert "CCBYNCSA" in caught.value.detail.get("allowed", [])

    def test_every_kind_is_named_when_one_is_wrong(self):
        from wisefood_mcp.registry import ToolError

        from integrator.propose import propose_source, KIND_MEANINGS

        with pytest.raises(ToolError) as caught:
            propose_source(self._ctx(), kind="pdf", title="x")
        assert set(caught.value.detail["allowed"]) == set(KIND_MEANINGS)


class TestCallKeyEdges:
    def test_a_url_with_and_without_its_trailing_slash(self):
        from integrator.agent import call_key

        assert (call_key("fetch_url", {"url": "https://x/a.pdf"})
                == call_key("fetch_url", {"url": "https://x/a.pdf/"}))

    def test_a_query_that_is_only_filler(self):
        """Two searches made of nothing but stopwords are the same search."""
        from integrator.agent import call_key

        assert (call_key("research", {"query": "the PDF of the"})
                == call_key("research", {"query": "a PDF for an"}))

    def test_a_missing_argument_does_not_raise(self):
        from integrator.agent import call_key

        for args in ({}, {"url": None}, {"query": ""}, {"doi": None}):
            assert isinstance(call_key("fetch_url", args), str)
            assert isinstance(call_key("research", args), str)

    def test_unicode_in_a_query_is_handled(self):
        from integrator.agent import call_key

        assert isinstance(call_key("research", {"query": "Οδηγός"}), str)

    def test_two_different_tools_never_share_a_key(self):
        from integrator.agent import call_key

        assert (call_key("fetch_url", {"url": "https://x/a"})
                != call_key("licence_evidence", {"url": "https://x/a"}))


class TestARunningCountTheModelCannotWriteAround:
    """Told once in the prompt not to overstate its filings, the assistant
    reported eight for five calls, then five for two. A number handed back
    after every call is harder to write around than an instruction it read at
    the start of the turn."""

    def _agent(self, filings):
        from wisefood_mcp import ToolContext

        from integrator.agent import Budget, IntegratorAgent

        results = []

        class Registry:
            def openai_schemas(self, include_writes=False):
                return [{"type": "function", "function": {"name": "propose_source"}}]

            def call(self, name, args, ctx):
                parsed = json.loads(args) if isinstance(args, str) else args
                return {"ok": True, "result": {
                    "proposal_id": f"id-{parsed['title']}", "status": "proposed"}}

        class Groq:
            def __init__(self):
                calls = [_delta(tool_calls=[{
                    "index": i, "id": f"c{i}", "type": "function",
                    "function": {"name": "propose_source",
                                 "arguments": json.dumps(
                                     {"kind": "guide", "title": t})}}])
                    for i, t in enumerate(filings)]
                self.replies = [[*calls, _usage(1)],
                                [_delta(content="done"), _usage(1)]]
                self.chat = type("C", (), {"completions": self})()

            def create(self, **kw):
                return iter(self.replies.pop(0))

        agent = IntegratorAgent(registry=Registry(),
                                tool_context=ToolContext(proposal_store=None),
                                groq_client=Groq(), budget=Budget(max_steps=20))
        out = agent.run_streamed([], "file them", lambda *_a: None)
        for message in out["messages"]:
            if message.get("role") == "tool":
                results.append(json.loads(message["content"]))
        return out, results

    def test_each_filing_is_told_how_many_there_have_been(self):
        _out, results = self._agent(["A", "B", "C"])
        counts = [r.get("filed_so_far_this_turn") for r in results]
        assert counts == [1, 2, 3]

    def test_the_count_is_said_in_words_it_will_read(self):
        _out, results = self._agent(["A", "B"])
        assert "2 filed in this turn" in results[-1]["note"]
        assert "do not count sources you only considered" in results[-1]["note"]

    def test_the_record_and_the_count_agree(self):
        """Two ways of saying the same thing, from the same source — if they
        ever disagree, the record is the one on screen."""
        out, results = self._agent(["A", "B", "C", "D"])
        filed = [s for s in out["timeline"] if s.get("data", {}).get("filed")]
        assert len(filed) == results[-1]["filed_so_far_this_turn"] == 4
