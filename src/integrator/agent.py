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
import re
import uuid
from typing import Any, Dict, List, Optional

from config import config

from integrator.steps import StepTracker, finished_detail, running_title

logger = logging.getLogger(__name__)

#: How much of one tool result the model is shown. A fetched PDF runs to tens
#: of thousands of characters and the useful part is near the front; what is
#: past this is almost always boilerplate that gets paid for on every
#: subsequent step of the turn.
TOOL_RESULT_CHARS = 8_000

#: How many recent tool results stay in full. Older ones are compacted — see
#: `compact_history`.
KEEP_FULL_RESULTS = 3

SYSTEM_PROMPT = """\
You help a WiseFood curator find and integrate new sources into the platform's data catalog: national dietary guides, scientific articles, textbooks, food composition tables and recipe collections.

What you can do, and should say so when it is useful:

- search the web and open pages and PDFs
- check the catalog for what is already held, and where the gaps are
- work out a source's licence from its own statements, and for a DOI from Unpaywall and Crossref
- file a proposal for a curator to review

What you cannot do: approve anything, or put anything into the catalog yourself. A person reviews your proposals in the console and approves them there. Say that plainly rather than implying you will proceed.

How to work:

- Use `research` to search. Its URLs are leads, not facts — open the promising ones with `fetch_url` before you rely on what they say. If a search does not turn something up, rewording it rarely will: say what you could not find and move on.
- Open a document once. `fetch_url` on a PDF tells you what it is, how many pages it has, and stashes it ready to attach — that is all you need to decide whether to propose it. Reading it page by page is the extraction pipeline's job and it does that after a curator approves, over the whole document, grounded in its pages.
- When a curator gives you a journal — a link or a name — use `journal_articles` to list what it has published. A publisher's own journal page is usually closed to us, so do not keep trying `fetch_url` on it. It marks what the catalog already holds and reads the licence the publisher registered, so propose the open ones that are new and say how many you skipped as already held.
- A recipe collection is a website, not a document, so there is nothing to open and read. Use `recipe_source` on the site: it finds where the site lists its pages, samples them, and reports how many carry machine-readable recipe markup. That share is the thing worth telling a curator — a site with thousands of pages and no markup has nothing we can import. Propose it with `harvest_location` as the source_url, and check the site's terms with `licence_evidence`, because a recipe corpus is content and copying it in needs a licence that permits that.
- When a source states dietary rules but does not already list them — advice in prose, a web page, a summary chapter — use `infer_guidelines` on the text you fetched. It returns rules with the verbatim quote each came from, and drops any it cannot quote. Say plainly that those rules were *inferred from the text* rather than extracted from a structured document, and never present them as the source's own list. For a guide that ships as a PDF of numbered recommendations, do not use this: propose it and let the extraction pipeline read it, which is grounded in pages rather than in your reading.
- For an article, pass the DOI to `propose_source` and move on. The integration run reads the publisher's record from Crossref when a curator approves, and refuses to create anything if that DOI has no record — so the citation is guaranteed at the point it matters and you do not have to establish it for every candidate you merely considered. Give the title as you found it; the record corrects it. Never present a citation you typed as if you had checked it.
- Check `catalog_coverage` before proposing. A source filling a gap is worth more than a fourth guide for a country that already has three. Its counts include drafts: an entry somebody has already brought in but not yet published is not a gap, so read `by_status` and say when what you found is already there as a draft. Give the country and language in whatever form you have — a name or an ISO code, both are resolved — and if it says it cannot resolve one, fix the name rather than reading the empty result as an absence.
- Establish the licence with `licence_evidence` when a source is the one you are recommending. Never assert a licence you have no evidence for; say it is undetermined instead, and say what that means — a curator can still approve it, but only by writing down why. Undetermined is a normal answer and is not a reason to withhold a proposal.
- File each candidate worth a curator's attention with `propose_source`, one call per source. When the call is genuinely the curator's — the source is off to the side of what they asked for, its licence is undetermined and copying matters, or it may duplicate something held — use `suggest_source` instead: it files nothing and shows them the candidate with a button.
- Never write a proposal out as JSON in your reply, and never say you have filed one unless `propose_source` returned an id for it. A code block is something nobody can act on, and a proposal that does not exist is worse than one you never offered, because a curator goes looking for it.
- Do not list what you filed, and never write a proposal id. The console shows the curator exactly what this turn filed, taken from the calls themselves — a list you write from memory competes with that and loses. Say what you found and what you think of it; the filings speak for themselves.
- If a call failed, say it failed and why. Never describe an intention as an accomplishment: "I looked for X and could not establish its licence" is useful, "I filed X" when you did not is a person searching a panel for something that is not there. This is the only way anything you find reaches a person: a source you describe in your answer but do not file does not exist as far as the console is concerned, and the curator's panel stays empty. Do it as you go, before you write your summary — not after, and never instead. Give it the licence and evidence exactly as `licence_evidence` returned them, a short rationale, and the integration steps you intend.
- Then say what you filed, with the titles. Do not tell the curator to create proposals; you have already created them and they are waiting in the panel.

Be transparent about your own work. Say what you searched for, which pages you actually read, and which of your conclusions rest on evidence you found versus on inference. When you are unsure, say what would settle it.

Work broadly, not exhaustively. A curator asking what exists for a country wants a handful of credible candidates filed and a clear account of them, not one source proved beyond doubt while the budget runs out. When something looks credible, file it and go to the next one; spend a second look only on what you are unsure of, or on what the curator asks about. It is better to return six proposals a person can judge than one they cannot act on. Say which ones you checked closely and which you are putting forward on their face — a curator can tell the difference and will thank you for it.

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
        # What has already been run this turn, keyed by call. A model that
        # fetches the same PDF seven times is not making progress, and the
        # first real run did exactly that: eight of fourteen steps went on one
        # identical URL until the budget ran out. Repeating a call returns what
        # it returned before, says so, and costs no network.
        seen: Dict[str, Any] = {}
        # What the curator sees while this runs and afterwards. Started here
        # rather than inside the tool loop so a turn that calls no tools still
        # says that it thought about the question.
        steps = StepTracker()
        self._emit = getattr(self, "_emit", None)
        thinking = steps.start("plan", "Working out what to look for")
        self._say("step", thinking)

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

            message, usage = self._complete(messages)
            self.budget.spend((usage or {}).get("total_tokens", 0))

            calls = message.get("tool_calls") or []
            if thinking is not None:
                steps.finish(thinking, outcome=(
                    f"Decided to {_describe_intent(calls)}" if calls
                    else "Answered from what was already known"
                ))
                self._say("step", thinking)
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
                self._say("step", step)

                key = call_key(name, parsed_args)
                repeat = key in seen
                if repeat:
                    outcome = seen[key]
                    if isinstance(outcome.get("result"), dict):
                        outcome = {**outcome, "result": {
                            **outcome["result"],
                            "repeated_call": (
                                "You already ran this in this turn and this is what "
                                "it returned. Nothing has changed. Use it and move "
                                "on — asking again, or asking the same thing in "
                                "different words, will not produce anything new."),
                        }}
                else:
                    outcome = self.registry.call(name, raw_args, self.ctx)
                    seen[key] = outcome
                ok = bool(outcome.get("ok"))
                if self.trace is not None:
                    self.trace.tool(name, parsed_args, ok, outcome.get("result"),
                                    step.get("elapsed_ms"))
                payload = outcome.get("result") if ok else outcome.get("error")
                # A suggestion rides on its step, which is already persisted
                # and already streamed — so the console can render it as a
                # button without a new column or a second request.
                if ok and name == "suggest_source" and isinstance(payload, dict):
                    step["data"] = {**step.get("data", {}),
                                    "suggestion": payload.get("suggestion") or {}}
                # What was actually filed, from the tool's own answer. The
                # console reports the turn from this and not from the reply:
                # asked to summarise its work, the model has written out
                # eight filings for five calls and then invented ids for the
                # difference. The database knew; nothing showed it.
                if ok and name == "propose_source" and isinstance(payload, dict):
                    if payload.get("proposal_id"):
                        step["data"] = {**step.get("data", {}), "filed": {
                            "proposal_id": payload["proposal_id"],
                            "title": (parsed_args or {}).get("title"),
                            "kind": (parsed_args or {}).get("kind"),
                        }}
                steps.finish(step, ok=ok, outcome=(
                    "Already run this turn — reusing the answer"
                    if repeat else finished_detail(
                        name, parsed_args or {}, ok, outcome.get("result"),
                        outcome.get("error"))))
                self._say("step", step)

                tool_turn = {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "tool_name": name,
                    "content": json.dumps(payload, default=str)[:TOOL_RESULT_CHARS],
                }
                messages.append({k: v for k, v in tool_turn.items()
                                 if k not in ("tool_name", "steps")})
                produced.append(tool_turn)

    # ---------------------------------------------------------- completion --
    def _complete(self, messages) -> tuple:
        """One model call, returning (message, usage).

        Streamed when a listener is attached, so the answer types out as the
        model writes it rather than landing whole after a silence — on a turn
        that has already spent a minute searching, the last thing a curator
        should get is another wait with nothing moving. Unstreamed otherwise:
        a background run has nobody to show deltas to, and the single response
        is less to go wrong.

        Either way the return shape is the same, so the loop above cannot
        tell which happened. That matters because the loop is what decides
        which tools run, and it should not grow a second path.
        """
        streaming = getattr(self, "_emit", None) is not None
        kwargs = dict(
            model=self.model,
            # Compacted here rather than in the loop: the loop's list is the
            # transcript that gets persisted, and what is persisted should be
            # what actually happened, not what we could afford to re-send.
            messages=compact_history(messages),
            tools=self.tools or None,
            tool_choice="auto" if self.tools else None,
            temperature=0.2,
            max_tokens=2000,
        )
        if not streaming:
            completion = self.groq.chat.completions.create(**kwargs)
            data = (completion.model_dump() if hasattr(completion, "model_dump")
                    else completion)
            return data["choices"][0]["message"], data.get("usage") or {}

        content: List[str] = []
        calls: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}
        # Ask for usage on the final frame: without it a streamed turn reports
        # no tokens at all, and the budget stops being a budget. Not every
        # provider or proxy in front of one accepts the option, and a turn
        # that dies on an unknown parameter is worse than a turn whose tokens
        # are estimated — so it is asked for, not required.
        try:
            stream = self.groq.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True})
        except TypeError:
            stream = self.groq.chat.completions.create(**kwargs, stream=True)
        except Exception as exc:  # noqa: BLE001
            if "stream_options" not in str(exc):
                raise
            logger.info("integrator: provider rejected stream_options; "
                        "streaming without usage")
            stream = self.groq.chat.completions.create(**kwargs, stream=True)
        for chunk in stream:
            frame = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            if frame.get("usage"):
                usage = frame["usage"]
            choices = frame.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                content.append(piece)
                self._say("text", {"delta": piece})
            _merge_tool_call_deltas(calls, delta.get("tool_calls") or [])

        if not usage:
            # No usage frame came back. Charge a rough estimate rather than
            # zero: a step that costs nothing is a step the budget cannot
            # stop, which is the runaway this whole class exists to prevent.
            spent = sum(len(str(m.get("content") or "")) for m in messages)
            usage = {"total_tokens": (spent + len("".join(content))) // 4}

        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return message, usage

    # ------------------------------------------------------ streamed turn --
    def run_streamed(self, history: List[Dict[str, Any]], user_message: str,
                     emit) -> Dict[str, Any]:
        """`run`, with a callback fired as each step starts and finishes.

        The loop itself is unchanged and deliberately so: a second
        implementation of the conversation would drift from this one, and the
        thing that would drift is the part that decides what the assistant is
        allowed to do. `emit` is called with (name, payload) and must not
        raise — a consumer that has hung up is not a reason to abandon a turn
        that is already spending tokens.
        """
        self._emit = emit
        try:
            return self.run(history, user_message)
        finally:
            self._emit = None

    def _say(self, name: str, payload: Dict[str, Any]) -> None:
        emit = getattr(self, "_emit", None)
        if emit is None:
            return
        try:
            # A copy, because a step is *mutated* when it finishes and the
            # consumer serialises later — off a queue, or across a thread.
            # Passing the live object means the "running" frame has already
            # become "done" by the time anyone reads it, and the stream shows
            # every step completing instantly, which is the one thing it
            # exists to avoid.
            emit(name, dict(payload))
        except Exception:  # noqa: BLE001 — see run_streamed
            logger.debug("integrator: listener dropped the %s event", name)

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


def compact_history(messages: List[Dict[str, Any]],
                    keep_full: int = KEEP_FULL_RESULTS) -> List[Dict[str, Any]]:
    """Shrink the tool results the model no longer needs in full.

    A conversation is re-sent in its entirety on every step, so a tool result
    is not paid for once — it is paid for on every step after it, and again on
    every later turn that replays it. One fetched PDF across an eight-step
    turn is the same eight thousand characters billed eight times, which is
    how a single question reaches a hundred thousand tokens without doing
    anything a curator would call expensive.

    What the model actually needs in full is the last few results — the ones
    it is reasoning about right now. Older ones it needs to *remember*: that
    it fetched this URL, and roughly what came back, so it does not fetch it
    again. That fits in a couple of hundred characters.

    The messages themselves are kept, never dropped. A tool result whose
    matching call has gone is a 400 from the provider, and a conversation
    that worked yesterday failing after a restart is a worse bug than a
    large bill.
    """
    indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(indices) <= keep_full:
        return messages

    stale = set(indices[:-keep_full]) if keep_full else set(indices)
    out: List[Dict[str, Any]] = []
    for i, message in enumerate(messages):
        if i in stale:
            out.append({**message, "content": _digest(message.get("content") or "")})
        else:
            out.append(message)
    return out


def _digest(content: str) -> str:
    """The head of a result, and an honest note that the rest is gone.

    Said in words the model can act on: it is told the content was trimmed
    and that re-running the call will not bring it back, because the obvious
    failure here is a model that notices something is missing and spends a
    step fetching it again.
    """
    if len(content) <= 400:
        return content
    return (content[:400]
            + f" …[{len(content) - 400:,} more characters trimmed to save "
              "context. This is the full result you already saw earlier in "
              "this turn; running the call again returns the same thing and "
              "will not restore the detail.]")


def call_key(name: str, args: Dict[str, Any]) -> str:
    """What counts as "the same call already made this turn".

    Comparing arguments literally is too literal, and a real run showed all
    three ways it fails. The same PDF was opened six times because each call
    asked for a different page. The web was searched five times for the same
    thing in five wordings — "Bulgaria national dietary guidelines PDF", then
    "Bulgaria dietary guidelines 2020", then "Bulgaria food based dietary
    guidelines children" — each a fresh key, each a fresh bill.

    So two calls are the same when they are after the same thing:

    * a document is identified by its URL. Which page, and how much text, are
      details of one reading of it. Paging through a PDF is the extraction
      pipeline's job and it does it after approval, not the assistant's.
    * a search is identified by its words, regardless of order, case,
      punctuation or the filler that gets shuffled between attempts.
    """
    if name in ("fetch_url", "licence_evidence") and args.get("url"):
        return f"{name}:{str(args['url']).strip().rstrip('/')}"
    if name == "doi_metadata" and args.get("doi"):
        return f"{name}:{str(args['doi']).strip().lower()}"
    if name == "research" and args.get("query"):
        words = re.findall(r"\w+", str(args["query"]).lower())
        # Words that carry no distinction between one attempt and the next.
        filler = {"pdf", "the", "a", "an", "of", "for", "in", "and", "or",
                  "national", "official", "latest", "new", "download"}
        stem = sorted(w for w in words if w not in filler and not w.isdigit())
        return f"research:{' '.join(stem)}"
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _merge_tool_call_deltas(calls: Dict[int, Dict[str, Any]], deltas) -> None:
    """Fold streamed tool-call fragments into `calls`, keyed by index.

    A streamed tool call arrives in pieces: the id and name once, then the
    arguments a few characters at a time across many frames, and several
    calls interleaved. The index is the only thing that identifies which
    call a fragment belongs to — ids are absent from every frame but the
    first — so it is what this keys on. Concatenating in arrival order is
    correct because a stream is ordered; the failure this avoids is
    *overwriting*, which silently truncates the arguments to their last
    fragment and hands the loop a call it cannot parse.
    """
    for delta in deltas:
        index = delta.get("index", 0)
        call = calls.setdefault(index, {
            "id": "", "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if delta.get("id"):
            call["id"] = delta["id"]
        if delta.get("type"):
            call["type"] = delta["type"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            call["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            call["function"]["arguments"] += fn["arguments"]


def _describe_intent(calls) -> str:
    """"search the web and check the licence" — the plan, in its own words."""
    intents = {
        "research": "search the web",
        "fetch_url": "read the page",
        "licence_evidence": "check the licence",
        "doi_metadata": "look the DOI up",
        "infer_guidelines": "read the rules out of it",
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
