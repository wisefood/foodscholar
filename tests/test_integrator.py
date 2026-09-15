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
