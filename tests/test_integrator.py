"""The Source Integrator: the wall, the loop, and the queue.

Three things are worth testing here and the rest is plumbing.

The **wall**: a proposal reaches ``approved`` only through a person, the write
tools refuse anything else, and an undetermined licence cannot be waved
through without a recorded reason. That runs against a real Postgres, because
the wall *is* a database column and a mock would prove nothing.

The **loop**: tool calls round-trip so the next turn can replay them, budgets
stop a runaway, and a failing tool becomes a value the model can read rather
than an exception that ends the conversation.

The **queue**: the spreadsheet imports completely and does not multiply.
"""
from __future__ import annotations

import copy
import os
import types
import uuid

import pathlib

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_HOST"),
    reason="needs a Postgres; set POSTGRES_HOST to run",
)


@pytest.fixture(scope="module", autouse=True)
def schema():
    from backend.db_init import init_db
    init_db()


@pytest.fixture
def store():
    from integrator.store import PostgresProposalStore
    return PostgresProposalStore()


@pytest.fixture
def proposal(store):
    from wisefood_mcp.stores import Proposal, new_proposal_id
    return store.create(Proposal(
        id=new_proposal_id(), kind="guide", title="Bulgarian FBDG for adults",
        source_url="https://ncpha.bg/fbdg.pdf", status="proposed",
        country="Bulgaria", language="Bulgarian",
    ))


class FakeGroq:
    """Replays a scripted list of provider responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        # Snapshot the messages. The loop mutates one list in place, and a
        # real client serialises at call time — keeping the reference would
        # mean every recorded call showed the conversation's final state.
        self.calls.append({**kw, "messages": copy.deepcopy(kw.get("messages", []))})
        return self.responses.pop(0) if self.responses else _say("done")


def _say(text, tool_calls=None, tokens=100):
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "openai/gpt-oss-120b", "choices": [{"message": message}],
            "usage": {"total_tokens": tokens}}


def _call(name, args, cid="call-1"):
    return [{"id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}]


# -------------------------------------------------------------------- store --

class TestTheStoreRoundTrips:
    def test_a_proposal_survives_the_database(self, store, proposal):
        again = store.get(proposal.id)
        assert again.title == "Bulgarian FBDG for adults"
        assert again.country == "Bulgaria" and again.status == "proposed"
        assert again.licence_evidence == [] and again.plan == []

    def test_metadata_is_stored_despite_the_reserved_name(self, store, proposal):
        # `metadata` is reserved on a declarative class, so the column and the
        # attribute differ; that mapping is exactly the kind of thing that
        # silently drops a field.
        store.update(proposal.id, metadata={"note": "ministry contacted"})
        assert store.get(proposal.id).metadata == {"note": "ministry contacted"}

    def test_the_expert_order_wins_over_the_agents(self, store):
        from wisefood_mcp.stores import Proposal, new_proposal_id
        tag = uuid.uuid4().hex[:8]
        low = store.create(Proposal(id=new_proposal_id(), kind="guide",
                                    title="low", session_id=tag, proposed_rank=0.1))
        high = store.create(Proposal(id=new_proposal_id(), kind="guide",
                                     title="high", session_id=tag, proposed_rank=0.9))
        assert [p.title for p in store.list(session_id=tag)] == ["high", "low"]
        store.update(low.id, expert_rank=1)
        assert [p.title for p in store.list(session_id=tag)][0] == "low"
        # The agent's score is kept, not overwritten — that is what makes a
        # rubric that keeps disagreeing with people visible later.
        assert store.get(low.id).proposed_rank == 0.1


# --------------------------------------------------------------------- wall --

class TestApprovalIsTheWall:
    def test_a_proposed_proposal_blocks_every_write(self, store, proposal):
        from wisefood_mcp.registry import ApprovalRequired
        from wisefood_mcp.stores import require_approved
        with pytest.raises(ApprovalRequired):
            require_approved(store, proposal.id)

    def test_an_undetermined_licence_needs_a_written_reason(self, store, proposal):
        from wisefood_mcp.registry import ToolError
        from wisefood_mcp.stores import approve
        with pytest.raises(ToolError) as exc:
            approve(store, proposal.id, actor="expert-1")
        assert exc.value.detail["code"] == "licence_unknown"

        approved = approve(store, proposal.id, actor="expert-1",
                           override_reason="Ministry confirmed reuse by email")
        assert approved.status == "approved"
        assert approved.approved_by == "expert-1" and approved.approved_at
        assert approved.licence_override_reason.startswith("Ministry")

    def test_a_clear_licence_approves_without_ceremony(self, store, proposal):
        from wisefood_mcp.stores import approve, require_approved
        store.update(proposal.id, licence="CC-BY-4.0", licence_confidence=0.9)
        approve(store, proposal.id, actor="expert-1")
        assert require_approved(store, proposal.id).licence == "CC-BY-4.0"

    def test_approval_is_not_reachable_as_a_tool(self):
        from wisefood_mcp import build_registry
        names = build_registry().names()
        assert not any("approve" in n for n in names)

    def test_the_service_approve_records_who_and_when(self, proposal):
        from integrator import service
        out = service.approve(proposal_id=proposal.id, user_sub="expert-9",
                              override_reason="public sector information")
        assert out["approved_by"] == "expert-9" and out["status"] == "approved"

    def test_a_rejected_proposal_keeps_why(self, proposal):
        from integrator import service
        out = service.reject(proposal_id=proposal.id, user_sub="expert-9",
                             reason="superseded by the 2025 edition")
        assert out["status"] == "rejected"
        assert out["metadata"]["rejected"]["reason"].startswith("superseded")


# --------------------------------------------------------------------- loop --

class TestTheLoop:
    def _agent(self, groq, **kw):
        from wisefood_mcp import ToolContext, build_registry
        from integrator.agent import IntegratorAgent
        return IntegratorAgent(
            registry=build_registry(),
            tool_context=ToolContext(proposal_store=None, **kw.pop("ctx", {})),
            groq_client=groq, **kw,
        )

    def test_a_plain_answer_needs_no_tools(self):
        agent = self._agent(FakeGroq(_say("Bulgaria has three guides.")))
        out = agent.run([], "what does the catalog hold for Bulgaria?")
        assert out["reply"] == "Bulgaria has three guides."
        assert out["stop_reason"] == "completed" and out["steps"] == 1

    def test_a_tool_call_round_trips_with_its_id(self):
        groq = FakeGroq(
            _say("", _call("licence_evidence", '{"text": "Licensed under CC BY 4.0"}')),
            _say("It is CC BY 4.0."),
        )
        agent = self._agent(groq)
        out = agent.run([], "what licence?")
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["assistant", "tool", "assistant"]
        tool_turn = out["messages"][1]
        assert tool_turn["tool_call_id"] == "call-1"
        assert tool_turn["tool_name"] == "licence_evidence"
        assert "CC-BY-4.0" in tool_turn["content"]
        # The second request must carry the whole exchange back.
        replayed = groq.calls[1]["messages"]
        assert replayed[-1]["role"] == "tool" and replayed[-1]["tool_call_id"] == "call-1"
        assert "tool_name" not in replayed[-1], "not a field the provider accepts"

    def test_a_failing_tool_is_a_value_the_model_can_read(self):
        groq = FakeGroq(
            _say("", _call("get_entity", '{"kind": "guide", "identifier": "urn:nope"}')),
            _say("I could not find it."),
        )
        out = self._agent(groq).run([], "fetch it")
        assert "error" in out["messages"][1]["content"]
        assert out["reply"] == "I could not find it.", "the conversation carries on"

    def test_a_malformed_argument_does_not_end_the_run(self):
        groq = FakeGroq(
            _say("", _call("search_catalog", '{"kind": "guide"}')),  # q missing
            _say("Let me try again."),
        )
        out = self._agent(groq).run([], "search")
        assert "did not match the tool" in out["messages"][1]["content"]

    def test_a_runaway_is_stopped_and_says_so(self):
        from integrator.agent import Budget
        # Always asks for another tool; only the budget can end this.
        groq = FakeGroq(*[_say("", _call("research", '{"query": "x"}', f"c{i}"))
                          for i in range(20)])
        agent = self._agent(groq, budget=Budget(max_steps=3, max_tokens=10**9))
        out = agent.run([], "find everything")
        assert out["stop_reason"].startswith("reached the 3-step")
        assert "stopped" in out["reply"] and out["steps"] == 3

    def test_a_token_ceiling_also_stops_it(self):
        from integrator.agent import Budget
        groq = FakeGroq(*[_say("", _call("research", '{"query": "x"}', f"c{i}"), tokens=400)
                          for i in range(20)])
        agent = self._agent(groq, budget=Budget(max_steps=99, max_tokens=1000))
        out = agent.run([], "find everything")
        assert "token limit" in out["stop_reason"]

    def test_phase_one_hides_the_write_tools_from_the_model(self):
        groq = FakeGroq(_say("ok"))
        agent = self._agent(groq, allow_writes=False)
        agent.run([], "hello")
        offered = {t["function"]["name"] for t in groq.calls[0]["tools"]}
        assert "search_catalog" in offered
        assert not {"create_guide", "upload_artifact"} & offered

    def test_the_research_model_is_not_the_conversation_model(self):
        from config import config
        assert config.settings["INTEGRATOR_RESEARCH_MODEL"].startswith("groq/compound")
        assert config.settings["INTEGRATOR_MODEL"] != config.settings["INTEGRATOR_RESEARCH_MODEL"]


class TestReplay:
    def test_stored_turns_become_provider_messages(self):
        from integrator.agent import replay
        rows = [
            types.SimpleNamespace(role="user", content="hi", tool_calls=None,
                                  tool_call_id=None),
            types.SimpleNamespace(role="assistant", content="",
                                  tool_calls=[{"id": "c1"}], tool_call_id=None),
            types.SimpleNamespace(role="tool", content="{}", tool_calls=None,
                                  tool_call_id="c1"),
        ]
        out = replay(rows)
        assert out[1]["tool_calls"] == [{"id": "c1"}]
        assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


# ------------------------------------------------------------------ backlog --

class TestTheBacklog:
    def test_the_whole_spreadsheet_imports(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, "scripts")
        from seed_integrator_backlog import read_ods, to_items

        ods = Path("../wisefood-client/Sources_Catalogue (1).ods")
        if not ods.exists():
            pytest.skip("source catalogue not checked out here")
        items, skipped = to_items(read_ods(ods))
        by_kind = {}
        for item in items:
            by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        # Every sheet lands, including the food-composition one that has no
        # Title column and needs a title built from its country.
        assert by_kind == {"guide": 123, "article": 61, "fctable": 23,
                           "textbook": 6, "rcollection": 4}
        assert len(skipped) == 2, "only the two country-only rows"
        assert all(i["external_key"] and i["title"] for i in items)

    def test_seeding_twice_does_not_multiply(self):
        from integrator import service
        tag = uuid.uuid4().hex[:8]
        items = [{"external_key": f"{tag}|one", "kind": "guide", "title": "One",
                  "url": "https://example.org/one"}]
        assert service.seed_backlog(items) == {"added": 1, "skipped": 0}
        assert service.seed_backlog(items) == {"added": 0, "skipped": 1}


class TestItSaysWhatItIsDoing:
    """Transparency is the feature, so it is tested like one.

    A curator approving a source into a public-health catalog has to be able
    to check the work: what was searched for, what was opened, what the
    licence evidence actually said. A badge reading `research` does not carry
    that; "Searched the web for *Bulgaria dietary guidelines* — 6 results"
    does, and lets them notice the assistant searched for the wrong thing.
    """

    def _agent(self, groq, **kw):
        from wisefood_mcp import ToolContext, build_registry
        from integrator.agent import IntegratorAgent
        return IntegratorAgent(registry=build_registry(),
                               tool_context=ToolContext(proposal_store=None),
                               groq_client=groq, **kw)

    def test_a_plain_answer_still_shows_that_it_thought(self):
        out = self._agent(FakeGroq(_say("Bulgaria has three guides."))).run([], "ask")
        timeline = out["timeline"]
        assert [s["kind"] for s in timeline] == ["plan"]
        assert timeline[0]["outcome"] == "Answered from what was already known"
        assert timeline[0]["status"] == "done"

    def test_the_plan_step_says_what_it_decided_to_do(self):
        groq = FakeGroq(
            _say("", _call("licence_evidence", '{"text": "CC BY 4.0"}')),
            _say("It is CC BY 4.0."),
        )
        out = self._agent(groq).run([], "what licence?")
        assert out["timeline"][0]["outcome"] == "Decided to check the licence"

    def test_each_tool_becomes_a_readable_step(self):
        groq = FakeGroq(
            _say("", _call("licence_evidence",
                           '{"text": "Licensed under CC BY 4.0"}', "c1")),
            _say("Done."),
        )
        out = self._agent(groq).run([], "check it")
        step = next(s for s in out["timeline"] if s["kind"] == "licence")
        assert step["title"] == "Checking the licence"
        assert "CC-BY-4.0" in step["outcome"] and "confident" in step["outcome"]
        assert step["status"] == "done" and step["ok"] is True
        assert step["elapsed_ms"] is not None

    def test_a_missing_licence_says_what_that_means_for_approval(self):
        groq = FakeGroq(
            _say("", _call("licence_evidence", '{"text": "A recipe for soup."}')),
            _say("Nothing found."),
        )
        out = self._agent(groq).run([], "check it")
        step = next(s for s in out["timeline"] if s["kind"] == "licence")
        assert "cannot be approved without a reason" in step["outcome"]

    def test_a_failed_tool_says_why_in_words(self):
        groq = FakeGroq(
            _say("", _call("research", '{"query": "x"}')),  # no groq client in ctx
            _say("I could not search."),
        )
        out = self._agent(groq).run([], "find sources")
        step = next(s for s in out["timeline"] if s["kind"] == "search")
        assert step["ok"] is False
        assert step["outcome"].startswith("Could not:")
        assert "Groq" in step["outcome"], "names what was missing, not just 'failed'"

    def test_the_search_step_shows_the_query_that_was_run(self):
        # The point of showing this: a curator can see it searched for the
        # wrong thing, which is invisible if only the result is shown.
        groq = FakeGroq(_say("", _call("research", '{"query": "Bulgaria FBDG adults"}')),
                        _say("Found some."))
        out = self._agent(groq).run([], "find them")
        step = next(s for s in out["timeline"] if s["kind"] == "search")
        assert step["detail"] == "Bulgaria FBDG adults"
        assert step["ok"] is False, "the query survives even when the search fails"

    def test_stopping_early_is_itself_a_step(self):
        from integrator.agent import Budget
        groq = FakeGroq(*[_say("", _call("research", '{"query": "x"}', f"c{i}"))
                          for i in range(20)])
        out = self._agent(groq, budget=Budget(max_steps=2, max_tokens=10**9)).run([], "go")
        stop = [s for s in out["timeline"] if s["kind"] == "stop"]
        assert stop and "step limit" in stop[0]["outcome"]

    def test_the_timeline_rides_the_final_assistant_turn(self):
        # So reopening the conversation tomorrow shows the same account of
        # the work, not an empty transcript.
        groq = FakeGroq(_say("", _call("licence_evidence", '{"text": "CC0"}')),
                        _say("Public domain."))
        out = self._agent(groq).run([], "check")
        finals = [m for m in out["messages"] if m["role"] == "assistant" and m.get("steps")]
        assert len(finals) == 1, "one account of the turn, not one per round"
        assert len(finals[0]["steps"]) == out["timeline"].__len__()

    def test_tool_turns_sent_to_the_provider_carry_no_ui_fields(self):
        # `steps` and `tool_name` are ours; a provider rejects unknown keys.
        groq = FakeGroq(_say("", _call("licence_evidence", '{"text": "CC0"}')),
                        _say("ok"))
        self._agent(groq).run([], "check")
        sent = groq.calls[1]["messages"]
        for message in sent:
            assert not set(message) - {"role", "content", "tool_calls", "tool_call_id"}


class TestTheRubric:
    """The score is arithmetic a curator can argue with, not a model's opinion.

    A number produced for reasons nobody recorded cannot be disagreed with.
    Every component here is named, weighted, and shown, so a curator who
    thinks it is wrong can point at the clause rather than at the number.
    """

    def _score(self, **fields):
        from integrator.ranking import score
        proposal = {"title": "x", "metadata": {}, **fields}
        return score(proposal, existing_similar=fields.pop("_existing", None))

    def test_a_permissive_licence_outranks_a_restrictive_one(self):
        from integrator.ranking import score
        base = {"title": "Guide", "source_url": "https://health.gov/g.pdf",
                "country": "X", "language": "Y", "metadata": {}}
        free = score({**base, "licence": "CC-BY-4.0"}, existing_similar=0)
        closed = score({**base, "licence": "Proprietary"}, existing_similar=0)
        assert free["score"] > closed["score"]

    def test_an_undetermined_licence_is_a_question_not_a_refusal(self):
        # Scoring it zero would bury everything nobody has checked yet, which
        # is most of a fresh backlog.
        from integrator.ranking import score
        base = {"title": "Guide", "source_url": "https://who.int/g.pdf", "metadata": {}}
        unknown = score({**base, "licence": None}, existing_similar=0)
        closed = score({**base, "licence": "Proprietary"}, existing_similar=0)
        assert unknown["score"] > closed["score"]
        assert "undetermined" in unknown["rationale"]

    def test_a_gap_outranks_something_already_covered(self):
        from integrator.ranking import score
        base = {"title": "Guide", "licence": "CC-BY-4.0",
                "source_url": "https://health.gov/g.pdf", "metadata": {}}
        gap = score(base, existing_similar=0)
        covered = score(base, existing_similar=5)
        assert gap["score"] > covered["score"]
        assert "fills a gap" in gap["rationale"]

    def test_authority_is_recognised_and_named(self):
        from integrator.ranking import score
        who = score({"title": "x", "source_url": "https://www.who.int/p/1",
                     "licence": "CC-BY-4.0", "metadata": {}}, existing_similar=0)
        blog = score({"title": "x", "source_url": "https://someblog.example/p",
                      "licence": "CC-BY-4.0", "metadata": {}}, existing_similar=0)
        assert who["score"] > blog["score"]
        assert any(row["why"] == "WHO" for row in who["breakdown"])

    def test_a_printed_only_source_is_marked_hard_to_ingest(self):
        from integrator.ranking import score
        printed = score({"title": "x", "licence": "CC-BY-4.0",
                         "metadata": {"attributes": {"Format": "Printed book"}}},
                        existing_similar=0)
        row = next(r for r in printed["breakdown"] if r["component"] == "tractability")
        assert "printed only" in row["why"]

    def test_every_component_is_shown_with_its_weight(self):
        result = self._score(licence="CC0", source_url="https://who.int/x")
        names = {row["component"] for row in result["breakdown"]}
        assert names == {"licence", "coverage_gap", "authority", "tractability",
                         "completeness"}
        for row in result["breakdown"]:
            assert row["why"] and 0 <= row["score"] <= 1 and row["weight"] > 0

    def test_the_rationale_leads_with_what_decided_it(self):
        # Ordered by weight x score, so the first clause is the one that moved
        # the number — not whichever component happened to be listed first.
        result = self._score(licence="CC0", source_url="https://who.int/x",
                             _existing=0)
        assert result["breakdown"][0]["why"] in result["rationale"].split("; ")[0]

    def test_weights_are_tunable_and_normalised(self):
        from integrator.ranking import weights
        default = weights({})
        assert abs(sum(default.values()) - 1.0) < 1e-9
        retuned = weights({"INTEGRATOR_WEIGHT_LICENCE": 0, "INTEGRATOR_WEIGHT_AUTHORITY": 8})
        assert retuned["licence"] == 0.0
        assert retuned["authority"] > default["authority"]
        assert abs(sum(retuned.values()) - 1.0) < 1e-9

    def test_a_nonsense_weight_falls_back_rather_than_crashing(self):
        from integrator.ranking import weights
        assert weights({"INTEGRATOR_WEIGHT_LICENCE": "not a number"})["licence"] > 0

    def test_creating_a_proposal_scores_it_and_keeps_the_breakdown(self):
        from integrator import service
        out = service.create_proposal(
            user_sub="expert-1", session_id=None, kind="guide",
            title="Bulgarian FBDG", source_url="https://ncpha.bg/f.pdf",
            country="Bulgaria", language="Bulgarian", licence="CC-BY-4.0",
            existing_similar=0,
        )
        assert out["proposed_rank"] and out["proposed_rank"] > 0.7
        assert out["rationale"]
        assert len(out["metadata"]["ranking"]["breakdown"]) == 5

    def test_rescoring_after_a_licence_is_established_moves_the_rank(self):
        # The licence is the heaviest component, so a proposal scored before
        # it was known is scored on a placeholder.
        from integrator import service
        out = service.create_proposal(
            user_sub="expert-1", session_id=None, kind="guide",
            title="Unknown licence guide", source_url="https://health.gov/g.pdf",
            existing_similar=0,
        )
        before = out["proposed_rank"]
        service._STORE.update(out["id"], licence="CC0")
        after = service.rescore(out["id"], existing_similar=0)["proposed_rank"]
        assert after > before


class TestTracingNeverCostsAnAnswer:
    """A trace backend that is down must not take the assistant with it."""

    class BrokenTrace:
        active = True
        def tool(self, *a, **kw): raise RuntimeError("langfuse is down")
        def finish(self, *a, **kw): raise RuntimeError("langfuse is down")

    def test_an_inert_handle_is_the_shape_the_loop_expects(self):
        from integrator.tracing import trace_run
        with trace_run(session_id="s", user_sub="u", question="q", model="m") as trace:
            # Langfuse is not installed in this environment, so this is the
            # real off-path rather than a stand-in for it.
            assert trace.active is False
            trace.tool("research", {"query": "x"}, True, {"findings": []}, 10.0)
            trace.finish(reply="hi", stop_reason="completed", steps=1, tokens=10)

    def test_a_turn_still_answers_when_tracing_raises(self):
        from wisefood_mcp import ToolContext, build_registry
        from integrator.agent import IntegratorAgent

        groq = FakeGroq(_say("", _call("licence_evidence", '{"text": "CC0"}')),
                        _say("Public domain."))
        agent = IntegratorAgent(
            registry=build_registry(),
            tool_context=ToolContext(proposal_store=None),
            groq_client=groq, trace=self.BrokenTrace(),
        )
        with pytest.raises(RuntimeError):
            # The handle itself raises — proving the test's premise. The
            # production handle swallows this; see `tracing.RunTrace`.
            agent.run([], "check it")

    def test_the_real_handle_swallows_backend_failures(self):
        from integrator.tracing import RunTrace
        trace = RunTrace(session_id="s", user_sub="u", model="m")

        class Exploding:
            def start_span(self, **kw): raise RuntimeError("down")
            def update(self, **kw): raise RuntimeError("down")
            def end(self): raise RuntimeError("down")

        trace._span = Exploding()
        # None of these may raise: an answer is worth more than its trace.
        trace.tool("research", {}, True, {}, 1.0)
        trace.finish(reply="x", stop_reason="completed", steps=1, tokens=1)
        trace._end()

    def test_a_large_result_is_clipped_before_it_reaches_the_board(self):
        from integrator.tracing import _clip
        small = {"findings": [1, 2, 3]}
        assert _clip(small) == small
        big = {"text": "x" * 50_000}
        clipped = _clip(big)
        assert clipped["truncated"] is True and len(clipped["head"]) <= 4000


# ------------------------------------------------- Phase 2: runs, in anger --

def test_a_run_row_survives_the_process_that_made_it(store, proposal, monkeypatch):
    """A run is readable by anyone, which is what makes the console possible."""
    from datetime import datetime, timezone

    from models.db import IntegrationRun
    from integrator import service

    with service._session_factory() as db:
        db.add(IntegrationRun(
            id=uuid.uuid4().hex[:16], proposal_id=proposal.id, status="running",
            stage="extracting", steps=[{"id": "step-1", "title": "Reading"}],
            result={"urn": "urn:wf:guide:1"}, wrote_anything=True,
            heartbeat_at=datetime.now(timezone.utc),
        ))
        db.commit()

    runs = service.list_runs(proposal_id=proposal.id)
    assert len(runs) == 1
    assert runs[0]["status"] == "running"
    assert runs[0]["result"]["urn"] == "urn:wf:guide:1"
    assert service.get_run(runs[0]["id"])["stage"] == "extracting"


def test_a_run_whose_pod_died_reads_stalled(store, proposal):
    """Not written back — the pod may yet return — but said out loud."""
    from datetime import datetime, timedelta, timezone

    from models.db import IntegrationRun
    from integrator import service

    run_id = uuid.uuid4().hex[:16]
    with service._session_factory() as db:
        db.add(IntegrationRun(
            id=run_id, proposal_id=proposal.id, status="running", stage="extracting",
            steps=[], result={},
            heartbeat_at=datetime.now(timezone.utc)
            - timedelta(seconds=service.STALL_AFTER_SECONDS + 60),
        ))
        db.commit()

    assert service.get_run(run_id)["status"] == "stalled"
    # And the column itself is untouched, so a worker that comes back finds
    # the run it was working on.
    with service._session_factory() as db:
        assert db.get(IntegrationRun, run_id).status == "running"


def test_an_unapproved_proposal_cannot_be_integrated(store, proposal, monkeypatch):
    from config import config
    from integrator import service

    monkeypatch.setitem(config.settings, "INTEGRATOR_WRITES_ENABLED", True)
    with pytest.raises(PermissionError, match="not been approved"):
        service.start_integration(proposal_id=proposal.id, user_sub="curator-1")


def test_integration_refuses_while_writes_are_off(store, proposal, monkeypatch):
    """The deployment switch, checked before a thread is ever started."""
    from config import config
    from wisefood_mcp.stores import approve
    from integrator import service

    # The fixture has no licence, so the wall requires a reason — which is
    # the behaviour under test elsewhere, and a precondition here.
    approve(store, proposal.id, actor="curator-1",
            override_reason="national agency, licence being confirmed by email")
    monkeypatch.setitem(config.settings, "INTEGRATOR_WRITES_ENABLED", False)
    with pytest.raises(PermissionError, match="switched off"):
        service.start_integration(proposal_id=proposal.id, user_sub="curator-1")


def test_a_second_press_does_not_start_a_second_run(store, proposal, monkeypatch):
    from datetime import datetime, timezone

    from config import config
    from models.db import IntegrationRun
    from wisefood_mcp.stores import approve
    from integrator import service

    # The fixture has no licence, so the wall requires a reason — which is
    # the behaviour under test elsewhere, and a precondition here.
    approve(store, proposal.id, actor="curator-1",
            override_reason="national agency, licence being confirmed by email")
    monkeypatch.setitem(config.settings, "INTEGRATOR_WRITES_ENABLED", True)
    with service._session_factory() as db:
        db.add(IntegrationRun(
            id=uuid.uuid4().hex[:16], proposal_id=proposal.id, status="running",
            steps=[], result={}, heartbeat_at=datetime.now(timezone.utc),
        ))
        db.commit()

    with pytest.raises(RuntimeError, match="already running"):
        service.start_integration(proposal_id=proposal.id, user_sub="curator-1")


def test_a_failed_proposal_can_be_approved_again(store, proposal, monkeypatch):
    """The dead end. A proposal's status follows its last run, so a failure
    left it neither `approved` (start_integration refused it) nor approvable
    (approve refused it), and the console showed the run panel without an
    Approve button — the only button on the card was the one that 403s.

    This stops at the gate rather than starting a run: what was broken is the
    pair of status checks, and the work behind them needs a live extraction.
    """
    from config import config
    from wisefood_mcp.stores import approve
    from integrator import service

    approve(store, proposal.id, actor="curator-1",
            override_reason="national agency, licence being confirmed by email")
    monkeypatch.setitem(config.settings, "INTEGRATOR_WRITES_ENABLED", True)

    # What the executor writes when a run does not finish.
    store.update(proposal.id, status="failed")
    with pytest.raises(PermissionError, match="not been approved"):
        service.start_integration(proposal_id=proposal.id, user_sub="curator-1")

    # Re-approving is the way back, and it records who asked for the retry.
    retried = approve(store, proposal.id, actor="curator-2",
                      override_reason="retrying after a timeout")
    assert retried.status == "approved"
    assert retried.approved_by == "curator-2"

    # Which is exactly what start_integration gates on, so the refusal above
    # no longer applies.
    assert store.get(proposal.id).status == "approved"


def test_a_stalled_run_does_not_block_a_retry(store, proposal, monkeypatch):
    """The point of computing `stalled`: otherwise a dead pod locks a proposal
    out of ever being integrated again."""
    from datetime import datetime, timedelta, timezone

    from config import config
    from models.db import IntegrationRun
    from wisefood_mcp.stores import approve
    from integrator import service

    # The fixture has no licence, so the wall requires a reason — which is
    # the behaviour under test elsewhere, and a precondition here.
    approve(store, proposal.id, actor="curator-1",
            override_reason="national agency, licence being confirmed by email")
    monkeypatch.setitem(config.settings, "INTEGRATOR_WRITES_ENABLED", True)
    with service._session_factory() as db:
        db.add(IntegrationRun(
            id=uuid.uuid4().hex[:16], proposal_id=proposal.id, status="running",
            steps=[], result={},
            heartbeat_at=datetime.now(timezone.utc)
            - timedelta(seconds=service.STALL_AFTER_SECONDS + 60),
        ))
        db.commit()
        assert service._active_run(db, proposal.id) is None


def test_the_core_transport_sends_the_field_the_route_declares(monkeypatch):
    """The 422 that Phase 1 would have hit.

    `GuidelineImportRequest` requires `guide_id`; the tool used to post
    `guide_urn`, which validates as a missing field and imports nothing.
    """


    from integrator import service

    seen = {}

    class FakeJobService:
        def enqueue_job(self, **kw):
            seen["enqueue"] = kw

        async def get_job_response(self, artifact_uuid):
            return {"status": "queued", "artifact_uuid": artifact_uuid}

        async def import_latest_result_to_guide(self, **kw):
            seen["import"] = kw
            return {"total_created": 3, "dry_run": kw["dry_run"]}

    monkeypatch.setattr("services.guideline_jobs.GuidelineJobService", FakeJobService)
    post, get = service._core_transport(None)

    post("/api/v1/guidelines/extract/abc", {"guide_id": "urn:wf:guide:1"})
    assert seen["enqueue"] == {"artifact_uuid": "abc", "guide_id": "urn:wf:guide:1"}

    out = post("/api/v1/guidelines/import/abc",
               {"guide_id": "urn:wf:guide:1", "dry_run": False})
    assert seen["import"]["guide_id"] == "urn:wf:guide:1"
    assert seen["import"]["dry_run"] is False
    assert out["total_created"] == 3

    assert get("/api/v1/guidelines/extract/abc")["status"] == "queued"


# --------------------------------------------- the agent acts as its caller --

class TestTheAgentHasNoMoreRightsThanTheCaller:
    """An assistant must not be a way to do what the person driving it may not.

    The catalog is the authority on that, so the only faithful way to honour it
    is to reach the catalog as them — and, crucially, to have no fallback when
    we cannot.
    """

    def test_no_caller_token_means_no_catalog_at_all(self, monkeypatch):
        """The failure that matters is the one that makes the agent *more*
        capable, because nobody reports it."""
        from integrator import service

        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL", "https://api.example")
        monkeypatch.setitem(service.config.settings, "GROQ_API_KEY", "gsk_test")
        assert service._data_client(None) is None
        assert service.tool_context(user_sub="curator-1").data_client is None

    def test_the_client_is_built_with_the_callers_token(self, monkeypatch):
        from integrator import service

        seen = {}

        class FakeClient:
            def __init__(self, base_url, credentials):
                seen["base"] = base_url
                seen["credentials"] = credentials

        import wisefood.client as wc
        monkeypatch.setattr(wc, "DataClient", FakeClient)
        monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL", "https://api.example")

        service._data_client("caller-token-abc")
        assert seen["credentials"].access_token == "caller-token-abc"
        assert seen["credentials"].is_delegated is True
        # Not the service account, under any circumstances.
        assert seen["credentials"].client_id is None
        assert seen["credentials"].client_secret is None

    def test_a_run_without_delegation_writes_nothing_and_says_why(
            self, store, proposal, monkeypatch):
        from wisefood_mcp import ToolContext
        from wisefood_mcp.stores import approve
        from integrator.executor import Integration
        from wisefood_mcp import build_registry

        approve(store, proposal.id, actor="curator-1",
                override_reason="national agency, licence being confirmed")
        row = store.get(proposal.id)
        ctx = ToolContext(proposal_store=store, writes_enabled=True, data_client=None)
        outcome = Integration(proposal=row, registry=build_registry(), ctx=ctx,
                              persist=lambda _s: None, poll_interval=0,
                              sleep=lambda _s: None).run()
        assert outcome["status"] == "failed"
        assert "token" in outcome["error"]
        assert outcome["wrote_anything"] is False


# ------------------------------------------------------------ flood control --

class TestNobodyCanFloodUs:
    """The routes are already admin-and-expert only, so this is not about
    strangers. It is about a client in a loop, or one person's credentials
    being used to spend the platform's budget."""

    def test_an_hour_of_questions_is_capped(self, monkeypatch):
        from config import config
        from integrator import service

        user = f"curator-{uuid.uuid4().hex[:8]}"
        session = service.create_session(user_sub=user, title="t")
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_TURNS_PER_HOUR", 2)

        with service._session_factory() as db:
            from models.db import IntegratorMessage
            for seq in range(2):
                db.add(IntegratorMessage(session_id=session["id"], seq=seq,
                                         role="user", content="q"))
            db.commit()

        with pytest.raises(service.RateLimited) as exc:
            service.chat(session_id=session["id"], user_sub=user, message="one more")
        assert exc.value.retry_after > 0

    def test_the_cap_is_per_person_not_shared(self, monkeypatch):
        """One busy curator must not lock everyone else out."""
        from config import config
        from integrator import service
        from models.db import IntegratorMessage

        busy = f"curator-{uuid.uuid4().hex[:8]}"
        other = f"curator-{uuid.uuid4().hex[:8]}"
        busy_session = service.create_session(user_sub=busy, title="t")
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_TURNS_PER_HOUR", 2)
        with service._session_factory() as db:
            for seq in range(5):
                db.add(IntegratorMessage(session_id=busy_session["id"], seq=seq,
                                         role="user", content="q"))
            db.commit()
            assert service._turns_last_hour(db, busy) >= 5
            assert service._turns_last_hour(db, other) == 0

    def test_concurrent_runs_are_bounded_per_person_and_overall(self, monkeypatch):
        from datetime import datetime, timezone

        from config import config
        from integrator import service
        from models.db import IntegrationRun

        user = f"curator-{uuid.uuid4().hex[:8]}"
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_RUNS_PER_USER", 1)
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_RUNS_TOTAL", 999)
        with service._session_factory() as db:
            db.add(IntegrationRun(
                id=uuid.uuid4().hex[:16], proposal_id=uuid.uuid4().hex[:12],
                status="running", steps=[], result={}, started_by=user,
                heartbeat_at=datetime.now(timezone.utc)))
            db.commit()
            with pytest.raises(service.RateLimited, match="already have"):
                service._check_run_capacity(db, user)

    def test_a_dead_pod_does_not_permanently_consume_capacity(self, monkeypatch):
        """Otherwise the ceiling is a one-way ratchet and a node failure
        eventually stops all integration."""
        from datetime import datetime, timedelta, timezone

        from config import config
        from integrator import service
        from models.db import IntegrationRun

        user = f"curator-{uuid.uuid4().hex[:8]}"
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_RUNS_PER_USER", 1)
        # Pinned, or this asserts about the per-user ceiling while the shared
        # test database quietly trips the global one on rows other tests left.
        monkeypatch.setitem(config.settings, "INTEGRATOR_MAX_RUNS_TOTAL", 999)
        with service._session_factory() as db:
            db.add(IntegrationRun(
                id=uuid.uuid4().hex[:16], proposal_id=uuid.uuid4().hex[:12],
                status="running", steps=[], result={}, started_by=user,
                heartbeat_at=datetime.now(timezone.utc)
                - timedelta(seconds=service.STALL_AFTER_SECONDS + 60)))
            db.commit()
            service._check_run_capacity(db, user)  # does not raise


# ------------------------------------------------------- one curator's work --

class TestTheAuditTrailIsNotTheRoomsToRead:
    def test_an_expert_sees_their_own_tool_calls_only(self):
        from integrator import service
        from integrator.store import record_tool_call

        mine = f"curator-{uuid.uuid4().hex[:8]}"
        theirs = f"curator-{uuid.uuid4().hex[:8]}"
        my_session = service.create_session(user_sub=mine, title="mine")
        their_session = service.create_session(user_sub=theirs, title="theirs")
        record_tool_call({"tool": "research", "ok": True, "actor": mine,
                          "arguments": {"query": "what I was chasing"}},
                         session_id=my_session["id"])
        record_tool_call({"tool": "research", "ok": True, "actor": theirs,
                          "arguments": {"query": "what they were chasing"}},
                         session_id=their_session["id"])

        seen = service.tool_calls(user_sub=mine, is_admin=False, limit=500)
        queries = [(c["arguments"] or {}).get("query") for c in seen]
        assert "what I was chasing" in queries
        assert "what they were chasing" not in queries

        everything = service.tool_calls(user_sub=mine, is_admin=True, limit=500)
        all_queries = [(c["arguments"] or {}).get("query") for c in everything]
        assert "what they were chasing" in all_queries

    def test_an_unidentified_caller_sees_nothing_rather_than_everything(self):
        from integrator import service
        from integrator.store import record_tool_call

        who = f"curator-{uuid.uuid4().hex[:8]}"
        session = service.create_session(user_sub=who, title="s")
        record_tool_call({"tool": "research", "ok": True, "actor": who,
                          "arguments": {"query": "private"}},
                         session_id=session["id"])
        assert service.tool_calls(user_sub="nobody", is_admin=False, limit=500) == []


def test_a_repeated_tool_call_is_answered_from_the_first_one(monkeypatch):
    """The first real run spent eight of fourteen steps fetching one identical
    PDF until the budget ran out. A model doing that is not making progress,
    and the network should not be asked again to prove it."""
    import json

    from wisefood_mcp import ToolContext

    from integrator.agent import Budget, IntegratorAgent

    calls = []

    class Registry:
        def openai_schemas(self, include_writes=False):
            return [{"type": "function", "function": {"name": "fetch_url"}}]

        def call(self, name, args, ctx):
            calls.append(json.loads(args) if isinstance(args, str) else args)
            return {"ok": True, "result": {"fetched": True, "kind": "pdf",
                                           "pending_artifact": "a" * 20}}

    def fetch(**_kw):
        return {"function": {"name": "fetch_url",
                             "arguments": json.dumps({"url": "https://x/f.pdf"})},
                "id": "c1", "type": "function"}

    responses = [
        {"choices": [{"message": {"content": "", "tool_calls": [fetch()]}}],
         "usage": {"total_tokens": 10}},
        {"choices": [{"message": {"content": "", "tool_calls": [fetch()]}}],
         "usage": {"total_tokens": 10}},
        {"choices": [{"message": {"content": "Done."}}], "usage": {"total_tokens": 10}},
    ]
    agent = IntegratorAgent(
        registry=Registry(), tool_context=ToolContext(proposal_store=None),
        groq_client=FakeGroq(*responses), budget=Budget(max_steps=10))
    outcome = agent.run([], "find the Bulgarian guide")

    assert len(calls) == 1, "the second identical fetch must not reach the tool"
    # And the model is told, so it stops rather than trying a third time.
    tool_turns = [m for m in outcome["messages"] if m["role"] == "tool"]
    assert "repeated_call" in tool_turns[1]["content"]
    assert "move on" in tool_turns[1]["content"]
    timeline = [s.get("outcome") for s in outcome["timeline"]]
    assert any("Already run this turn" in (o or "") for o in timeline)


def test_the_catalog_client_points_at_the_catalog_not_the_gateway(monkeypatch):
    """The gateway does not serve /guides or /articles, so aiming there made
    every catalog tool return {"detail": "Not Found"} — the assistant could
    never see what the platform already held."""
    from integrator import service

    seen = {}

    class FakeClient:
        def __init__(self, base_url, credentials):
            seen["base"] = base_url

    import wisefood.client as wc
    monkeypatch.setattr(wc, "DataClient", FakeClient)
    monkeypatch.setitem(service.config.settings, "DATA_API_URL", "http://data-catalog:8000")
    monkeypatch.setitem(service.config.settings, "WISEFOOD_API_URL", "http://wisefood-api:8000")

    service._data_client("tok")
    assert seen["base"] == "http://data-catalog:8000"



# ---------------------------------------------------------------- proposing --

class TestFilingAProposal:
    """The agent can actually file what it finds.

    It could not before: `propose_source` did not exist, so a conversation
    ended with a table of recommendations in prose and a review panel with
    nothing in it. `create_proposal` was in the service with no caller.
    """

    def _ctx(self, session_id=None):
        from wisefood_mcp import ToolContext
        return ToolContext(proposal_store=None, actor="expert-1",
                           extra={"session_id": session_id})

    def test_a_filed_source_turns_up_in_the_panel(self):
        from integrator import service
        from integrator.propose import propose_source

        session = service.create_session(user_sub="expert-1")
        out = propose_source(
            self._ctx(session["id"]), kind="guide",
            title="Bulgarian FBDG for adults",
            source_url="https://ncpha.bg/fbdg.pdf", country="Bulgaria",
            rationale="fills a gap: no Bulgarian guide is held",
        )
        assert out["status"] == "proposed"
        listed = service.list_proposals(session_id=session["id"])
        assert [p["title"] for p in listed] == ["Bulgarian FBDG for adults"]
        # Scored here, not by the model.
        assert listed[0]["proposed_rank"] is not None
        assert listed[0]["metadata"]["ranking"]["breakdown"]

    def test_what_comes_back_is_short(self):
        """The result is re-sent on every remaining step of the turn, so it
        carries what the model needs and not its own submission."""
        from integrator.propose import propose_source

        out = propose_source(self._ctx(), kind="article", title="A paper")
        assert set(out) == {"proposal_id", "status", "rank", "filed", "note"}
        assert len(str(out)) < 400
