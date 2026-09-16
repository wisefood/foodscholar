"""The conversation loop.

One expert, one thread of conversation, a model that may call tools. The loop
is small on purpose — the interesting behaviour is in the tools and in the
approval wall, not here. What this owns is the four things a loop must not get
wrong:

* **Budgets.** A run stops after a fixed number of steps or tokens and says so.
  An agent with a search tool and no ceiling can spend an afternoon and a
  month's quota on one question.
* **Replay.** Every turn is persisted with its tool calls, and the next turn
  replays them, which is what "keeps its context" means. A tool result that
  cannot be matched to the call that produced it breaks the replay, so the
  ids are stored rather than reconstructed.
* **Audit.** Every tool call is written down before the model sees its result.
* **Separation.** The model chooses; we execute. The provider runs nothing but
  the web search inside the ``research`` tool.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from config import config

from integrator.steps import StepTracker, finished_detail, running_title

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You help a WiseFood curator find and integrate new sources into the platform's data catalog: national dietary guides, scientific articles, textbooks, food composition tables and recipe collections.

What you can do, and should say so when it is useful:

- search the web and open pages and PDFs
- check the catalog for what is already held, and where the gaps are
- work out a source's licence from its own statements, and for a DOI from Unpaywall and Crossref
- file a proposal for a curator to review

What you cannot do: approve anything, or put anything into the catalog yourself. A person reviews your proposals in the console and approves them there. Say that plainly rather than implying you will proceed.

How to work:

- Use `research` to search. Its URLs are leads, not facts — open the promising ones with `fetch_url` before you rely on what they say.
- For an article, always run `doi_metadata` first and propose from what it returns. Never type a title, author list or year out of a search result: a citation that reads perfectly and is wrong is the single worst thing you can put in this catalog, and Crossref has the record the publisher deposited. If a DOI is not registered there, say so rather than filling the gap yourself.
- Check `catalog_coverage` before proposing. A source filling a gap is worth more than a fourth guide for a country that already has three.
- Establish the licence with `licence_evidence` and quote what you found. Never assert a licence you have no evidence for; say it is undetermined instead, and say what that means — a curator can still approve it, but only by writing down why.
- Create a proposal for each candidate worth a curator's attention, with a rank, a short rationale, and the integration steps you intend.

Be transparent about your own work. Say what you searched for, which pages you actually read, and which of your conclusions rest on evidence you found versus on inference. When you are unsure, say what would settle it.

Be concise. Prefer official and primary sources: health ministries, public health agencies, WHO, EFSA, FAO, universities, peer-reviewed journals. When you cannot find something, say so — a confident wrong URL costs a curator more time than an honest gap."""


class Budget:
    """What one run may spend before it has to stop and explain itself."""

    def __init__(self, max_steps: int = 12, max_tokens: int = 120_000) -> None:
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.steps = 0
        self.tokens = 0

    def spend(self, tokens: int) -> None:
        self.steps += 1
        self.tokens += int(tokens or 0)

    @property
    def exhausted(self) -> Optional[str]:
        if self.steps >= self.max_steps:
            return f"reached the {self.max_steps}-step limit for one turn"
        if self.tokens >= self.max_tokens:
            return f"reached the {self.max_tokens:,}-token limit for one turn"
        return None


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


class IntegratorAgent:
    """Runs one turn of the conversation.

    Constructed per request rather than kept around: it holds the caller's
    identity, and an agent that outlived a request would be an agent acting
    with the last caller's authority.
    """

    def __init__(self, *, registry, tool_context, groq_client,
                 model: Optional[str] = None, budget: Optional[Budget] = None,
                 allow_writes: bool = False, trace=None):
        self.registry = registry
        self.ctx = tool_context
        self.groq = groq_client
        self.model = model or config.settings.get(
            "INTEGRATOR_MODEL", "openai/gpt-oss-120b")
        self.budget = budget or Budget(
            max_steps=int(config.settings.get("INTEGRATOR_MAX_STEPS", 12)),
            max_tokens=int(config.settings.get("INTEGRATOR_MAX_TOKENS", 120_000)),
        )
        # Phase 1 hides the write tools from the model entirely. A tool the
        # model cannot see is a tool it cannot try, which keeps the transcript
        # about research instead of about refusals.
        self.tools = self.registry.openai_schemas(include_writes=allow_writes)
        # Inert when tracing is off or unavailable, so the loop below never
        # has to ask whether it is being watched.
        self.trace = trace

    # ------------------------------------------------------------- one turn --
    def run(self, history: List[Dict[str, Any]], user_message: str) -> Dict[str, Any]:
        """Take one user message to the model's final answer.

        Returns the new messages to persist (assistant turns and tool results,
        in order), the final text, and what the run spent. Raises nothing the
        caller has to handle: a tool that fails hands the model an error and
        the conversation carries on, which is the whole reason tool failures
        are values rather than exceptions.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]
        produced: List[Dict[str, Any]] = []
        stop_reason = "completed"
        # What the curator sees while this runs and afterwards. Started here
        # rather than inside the tool loop so a turn that calls no tools still
        # says that it thought about the question.
        steps = StepTracker()
        thinking = steps.start("plan", "Working out what to look for")

        while True:
            limit = self.budget.exhausted
            if limit:
                stop_reason = limit
                note = (
                    f"I stopped because I {limit}. Ask me to continue and I will "
                    f"pick up from here."
                )
                steps.add("stop", "Stopped early", outcome=limit)
                produced.append({"role": "assistant", "content": note,
                                 "steps": steps.snapshot()})
                return self._result(produced, note, stop_reason, steps)

            completion = self.groq.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
                temperature=0.2,
                max_tokens=2000,
            )
            data = completion.model_dump() if hasattr(completion, "model_dump") else completion
            self.budget.spend((data.get("usage") or {}).get("total_tokens", 0))

            message = data["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if thinking is not None:
                steps.finish(thinking, outcome=(
                    f"Decided to {_describe_intent(calls)}" if calls
                    else "Answered from what was already known"
                ))
                thinking = None
            assistant_turn: Dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            if calls:
                assistant_turn["tool_calls"] = calls
            messages.append(assistant_turn)
            produced.append(assistant_turn)

            if not calls:
                assistant_turn["steps"] = steps.snapshot()
                return self._result(produced, assistant_turn["content"], stop_reason, steps)

            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or "?"
                raw_args = fn.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed_args = {}

                kind, title, detail = running_title(name, parsed_args or {})
                step = steps.start(kind, title, detail=detail,
                                   data={"tool": name})

                outcome = self.registry.call(name, raw_args, self.ctx)
                ok = bool(outcome.get("ok"))
                if self.trace is not None:
                    self.trace.tool(name, parsed_args, ok, outcome.get("result"),
                                    step.get("elapsed_ms"))
                payload = outcome.get("result") if ok else outcome.get("error")
                steps.finish(step, ok=ok, outcome=finished_detail(
                    name, parsed_args or {}, ok, outcome.get("result"),
                    outcome.get("error")))

                tool_turn = {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "tool_name": name,
                    "content": json.dumps(payload, default=str)[:24_000],
                }
                messages.append({k: v for k, v in tool_turn.items()
                                 if k not in ("tool_name", "steps")})
                produced.append(tool_turn)

    def _result(self, produced, text, stop_reason, steps) -> Dict[str, Any]:
        if self.trace is not None:
            self.trace.finish(reply=text or "", stop_reason=stop_reason,
                              steps=self.budget.steps, tokens=self.budget.tokens)
        return {
            "messages": produced,
            "reply": text,
            "stop_reason": stop_reason,
            # What it did, in order, for the curator to read.
            "timeline": steps.snapshot(),
            "steps": self.budget.steps,
            "tokens": self.budget.tokens,
            "model": self.model,
        }


def _describe_intent(calls) -> str:
    """"search the web and check the licence" — the plan, in its own words."""
    intents = {
        "research": "search the web",
        "fetch_url": "read the page",
        "licence_evidence": "check the licence",
        "doi_metadata": "look the DOI up",
        "search_catalog": "search the catalog",
        "catalog_coverage": "check what we already hold",
        "get_entity": "read a catalog entry",
        "list_organizations": "look up organisations",
    }
    names = []
    for call in calls:
        name = (call.get("function") or {}).get("name") or ""
        phrase = intents.get(name, name.replace("_", " "))
        if phrase and phrase not in names:
            names.append(phrase)
    if not names:
        return "use a tool"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def replay(rows) -> List[Dict[str, Any]]:
    """Turn stored messages back into what the model expects.

    The inverse of what :meth:`IntegratorAgent.run` returns. Assistant turns
    carry their ``tool_calls`` and tool turns their ``tool_call_id``, because
    a provider rejects a tool result whose call it cannot find — which is how
    a conversation that looked fine yesterday starts 400ing after a restart.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.role == "assistant":
            turn: Dict[str, Any] = {"role": "assistant", "content": row.content or ""}
            if row.tool_calls:
                turn["tool_calls"] = row.tool_calls
            out.append(turn)
        elif row.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": row.tool_call_id,
                "content": row.content or "",
            })
        else:
            out.append({"role": row.role, "content": row.content or ""})
    return out
