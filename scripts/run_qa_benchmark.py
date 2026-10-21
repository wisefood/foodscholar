"""Run a persona question set through POST /api/v1/qa/ask and record every answer.

Each question is sent as its persona would ask it (expertise level, language,
country/region, age group) with ``bypass_cache`` set, so a re-run measures
the pipeline as deployed rather than replaying yesterday's answers. The
complete ``QAResponse`` is stored as returned; scoring (citation checks,
LLM-as-a-judge, red-team) reads the run files later and never needs to call
the API again.

A run directory holds two files:

* ``manifest.json`` — inputs (paths + sha256), options, and a
  ``GET /qa/runtime-info`` snapshot from before and after the run (git commit,
  models, prompt versions, ranking settings), plus any ``drift`` between the
  two. Written at the start and rewritten at the end, so a crashed run still
  says what it ran against.
* ``responses.jsonl`` — one record per question: what was sent, the HTTP
  status, latency, and the verbatim response (or the error).

Questions with ``follow_up_of`` run after that question, in its
``qa_thread_id``, so turn-2 questions see the conversation they refer to; a
filtered selection pulls their parent in automatically. The thread's summary
and verbatim turns are written by a background task after the parent's
answer is returned, so a follow-up waits ``--follow-up-delay`` seconds first;
one that still arrives without context (``turn_count`` < 2) is counted as
``follow_up_without_context``.

When FoodScholar asks a clarification, the persona answers it in the same
run (the pending clarification expires after 30 minutes, and the pipeline
asks at most one per thread). With ``--clarify persona`` (the default),
gpt-oss-20b on Groq picks the option this persona would, from their profile
(key: ``--groq-api-key`` or ``GROQ_API_KEY``; never written to the run files);
an entry in ``--clarification-answers`` (question_id -> option value(s))
overrides it.
The record keeps both turns: ``clarification_turns`` holds the clarification
response verbatim with the answer given and why, and ``response`` is the
final answer. If no answer can be given, the clarification stays the result
and that thread's follow-ups are skipped. ``--clarify none`` records the
clarification and stops, for every question. A search that errors
(an Elasticsearch timeout, say) does not fail the request, only weakens the
answer; such answers are counted as ``degraded_retrieval``. Re-running with
the same ``--run-id`` resumes: questions already recorded as ok are skipped,
except degraded ones, unanswered clarifications (when there is a clarifier),
and follow-ups that got no thread context, which are asked again, the last
together with their parent in a new thread (thread state lives a day, so
resume follow-ups within that).

Benchmark traffic is tagged ``user_id = "benchmark:<run_id>"``, which selects
a run in ``GET /qa/requests?user_id=...`` and in Langfuse traces.

Usage:
    python scripts/run_qa_benchmark.py --api http://localhost:8001 \\
        --personas personas.json --questions questions.json
    python scripts/run_qa_benchmark.py ... --persona P01 --type Factual --dry-run
    python scripts/run_qa_benchmark.py ... --run-id 20261021T101500Z   # resume
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

ASK_PATH = "/api/v1/qa/ask"
RUNTIME_INFO_PATH = "/api/v1/qa/runtime-info"

# Picking an option is not judging, so FoodScholar's own small model is fine.
CLARIFIER_MODEL = "openai/gpt-oss-20b"

CLARIFIER_SYSTEM = (
    "You are playing a specific person who uses FoodScholar, a nutrition "
    "question-answering assistant. Before answering their question, the "
    "assistant has asked them one clarification question. Answer it the way "
    "this person would, from their profile and their question: their "
    "household, age, location and concerns usually settle it. When the "
    "profile does not, pick what best fits their situation, never an option "
    "just because it is listed first. Give a one-sentence reason grounded in "
    "the profile."
)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def select_questions(
    questions: List[Dict[str, Any]],
    *,
    personas: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """The questions matching every given filter, plus the parents they follow up.

    Order is kept from the file. A follow-up without its parent would be
    asked out of context, so parents are always included even when a filter
    would have dropped them.
    """
    chosen = [
        q for q in questions
        if (not personas or q["persona_id"] in personas)
        and (not types or q["question_type"] in types)
        and (not ids or q["question_id"] in ids)
    ]
    if limit is not None:
        chosen = chosen[:limit]

    by_id = {q["question_id"]: q for q in questions}
    wanted = {q["question_id"] for q in chosen}
    pending = [q.get("follow_up_of") for q in chosen]
    while pending:
        parent = pending.pop()
        if parent and parent not in wanted:
            wanted.add(parent)
            pending.append(by_id[parent].get("follow_up_of"))
    return [q for q in questions if q["question_id"] in wanted]


def validate(questions: List[Dict[str, Any]], personas: Dict[str, Dict[str, Any]]) -> None:
    """Fail before any call if a question names a persona or parent that doesn't exist."""
    ids = {q["question_id"] for q in questions}
    problems = []
    for q in questions:
        if q["persona_id"] not in personas:
            problems.append(f"{q['question_id']}: unknown persona {q['persona_id']}")
        parent = q.get("follow_up_of")
        if parent and parent not in ids:
            problems.append(f"{q['question_id']}: follow_up_of unknown question {parent}")
    if len(ids) != len(questions):
        problems.append("duplicate question_id values")
    if problems:
        raise ValueError("Invalid question set:\n  " + "\n  ".join(problems))


def request_for(
    question: Dict[str, Any],
    persona: Dict[str, Any],
    *,
    run_id: str,
    language_from: str = "question",
    retriever: Optional[str] = None,
    qa_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The /qa/ask payload for one question asked as its persona.

    The single place that knows the question-set format: ``api_mapping``
    carries the request-shaped fields, and ``country_or_region`` is
    "City, Country", so the country is its last part.
    """
    mapping = persona.get("api_mapping") or {}
    region = mapping.get("country_or_region")
    language = (
        mapping.get("language") if language_from == "persona" else None
    ) or question.get("language") or mapping.get("language") or "en"
    payload: Dict[str, Any] = {
        "question": question["foodscholar_question"],
        "expertise_level": mapping.get("expertise_level", "intermediate"),
        "language": language,
        "region": region,
        "country": region.rsplit(",", 1)[-1].strip() if region else None,
        "member_age_group": mapping.get("member_age_group"),
        "user_id": f"benchmark:{run_id}",
        "bypass_cache": True,
        "qa_thread_id": qa_thread_id,
        "retriever": retriever,
    }
    return {k: v for k, v in payload.items() if v is not None}


def runtime_drift(start: Dict[str, Any], end: Dict[str, Any]) -> List[str]:
    """Settings that changed while the run was going, as dotted paths.

    A prompt that only shows up in the end snapshot was first served during
    the run; that is expected and not drift. A changed version is.
    """
    drift = []
    for key in sorted(set(start) | set(end)):
        if key == "generated_at":
            continue
        a, b = start.get(key), end.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if key == "prompt_versions" and sub not in a:
                    continue
                if a.get(sub) != b.get(sub):
                    drift.append(f"{key}.{sub}")
        elif a != b:
            drift.append(key)
    return drift


def search_failures(record: Dict[str, Any]) -> int:
    """Searches in this answer that errored instead of running.

    An Elasticsearch timeout does not fail the request: the pipeline answers
    from whatever the other searches found, so the answer is silently worse.
    The recorded step carries no ok flag (only the live stream does), so the
    title the orchestrator gives a failed search is the signal.
    """
    steps = (record.get("response") or {}).get("reasoning_steps") or []
    return sum(
        1 for s in steps
        if s.get("kind") == "search"
        and (
            (s.get("data") or {}).get("ok") is False
            or str(s.get("title") or "").startswith("Search unavailable")
        )
    )


def lacks_thread_context(record: Dict[str, Any]) -> bool:
    """A follow-up answered as if it were the first turn of its thread."""
    if not record.get("follow_up_of") or record.get("status") != "ok":
        return False
    context = (record.get("response") or {}).get("conversation_context") or {}
    return int(context.get("turn_count") or 0) < 2


def read_recorded(path: str) -> Dict[str, Dict[str, Any]]:
    """Records already in a run's responses.jsonl, last one per question wins."""
    recorded: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return recorded
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                recorded[record["question_id"]] = record
    return recorded


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def groq_client(api_key: Optional[str], purpose: str) -> Any:
    """An AsyncGroq client, or a RuntimeError that says what is missing."""
    try:  # only the steps that call a model need the SDK
        from groq import AsyncGroq, GroqError
    except ImportError as exc:
        raise RuntimeError(f"{purpose} needs the groq package (pip install groq)") from exc
    try:
        return AsyncGroq(api_key=api_key)  # None falls back to GROQ_API_KEY
    except GroqError as exc:
        raise RuntimeError(
            f"{purpose} needs a Groq API key: pass --groq-api-key or set GROQ_API_KEY"
        ) from exc


async def groq_structured(
    client: Any, *, model: str, system: str, prompt: str, schema: Dict[str, Any], name: str
) -> Tuple[Dict[str, Any], str]:
    """One call constrained to a JSON schema: (parsed object, model that served it).

    Raises ValueError when the output was cut off, so a caller never acts on
    half an answer.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        },
        # As backend.model_profiles sets them for gpt-oss: reasoning kept out
        # of the content and charged against a budget that fits it.
        reasoning_format="hidden",
        reasoning_effort="low",
        max_completion_tokens=2048,
        temperature=0,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("the model ran out of tokens")
    return json.loads(choice.message.content or ""), response.model


def _clarification_answer(
    clarification: Dict[str, Any],
    *,
    values: Optional[List[str]] = None,
    free_text: Optional[str] = None,
    **how: Any,
) -> Dict[str, Any]:
    return {
        "clarification_response": {
            "question_id": clarification.get("id"),
            "selected_values": values or [],
            "free_text": free_text,
        },
        **how,
    }


class PersonaClarifier:
    """Answers FoodScholar's clarification questions the way a persona would.

    ``answer()`` returns the ``clarification_response`` to send plus how it
    was chosen, or ``{"error": ...}``; it never raises, because a failed
    answer only means the clarification stays the recorded result.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = CLARIFIER_MODEL,
        scripted: Optional[Dict[str, Any]] = None,
        client: Any = None,
    ):
        if client is None:
            client = groq_client(api_key, "--clarify persona (or use --clarify none)")
        self.client = client
        self.model = model
        self.scripted = scripted or {}

    async def check(self) -> None:
        """Fail before the run, not at its first clarification, on a bad key or model id."""
        try:
            await self.client.models.retrieve(self.model)
        except Exception as exc:
            raise RuntimeError(
                f"--clarify persona cannot reach {self.model} on Groq "
                f"(check --groq-api-key, or use --clarify none): {exc}"
            ) from exc

    async def answer(
        self,
        question_id: str,
        persona: Dict[str, Any],
        question_text: str,
        clarification: Dict[str, Any],
    ) -> Dict[str, Any]:
        values = [o.get("value") for o in clarification.get("options") or []]
        scripted = self.scripted.get(question_id)
        if scripted is not None:
            chosen = scripted if isinstance(scripted, list) else [scripted]
            if not values:
                return _clarification_answer(
                    clarification, free_text=str(chosen[0])[:500], source="scripted"
                )
            if not set(chosen) <= set(values):
                return {"error": f"scripted answer {chosen} is not among the options {values}"}
            return _clarification_answer(clarification, values=chosen, source="scripted")
        try:
            return await self._ask_model(persona, question_text, clarification, values)
        except Exception as exc:
            return {"error": f"{exc.__class__.__name__}: {exc}"}

    async def _ask_model(
        self,
        persona: Dict[str, Any],
        question_text: str,
        clarification: Dict[str, Any],
        values: List[str],
    ) -> Dict[str, Any]:
        multiple = clarification.get("input_type") == "multiple_choice"
        if values and multiple:
            key, field = "choices", {"type": "array", "items": {"type": "string", "enum": values}}
        elif values:
            key, field = "choice", {"type": "string", "enum": values}
        else:
            key, field = "answer", {"type": "string"}
        schema = {
            "type": "object",
            "properties": {key: field, "reason": {"type": "string"}},
            "required": [key, "reason"],
            "additionalProperties": False,
        }
        # api_mapping is request plumbing, not something the person knows.
        profile = {k: v for k, v in persona.items() if k != "api_mapping"}
        options = "\n".join(
            f"- {o.get('value')}: {o.get('label')}" for o in clarification.get("options") or []
        ) or "(answer in at most one sentence)"
        prompt = (
            f"Your profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Your question to FoodScholar:\n{question_text}\n\n"
            f"FoodScholar asks: {clarification.get('question')}\n"
            f"Why it asks: {clarification.get('reason') or 'not given'}\n"
            f"Options:\n{options}"
        )
        data, served_by = await groq_structured(
            self.client, model=self.model, system=CLARIFIER_SYSTEM, prompt=prompt,
            schema=schema, name="clarification_answer",
        )
        how = {"source": "persona-llm", "model": served_by, "reason": data.get("reason")}
        if not values:
            return _clarification_answer(clarification, free_text=str(data[key])[:500], **how)
        chosen = data[key] if multiple else [data[key]]
        if not chosen or not set(chosen) <= set(values):
            return {"error": f"model chose {chosen}, not among the options {values}", **how}
        return _clarification_answer(clarification, values=chosen, **how)


class BenchmarkRun:
    """One run: schedules questions (parents before follow-ups) and records them."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        run_id: str,
        run_dir: str,
        questions: List[Dict[str, Any]],
        personas: Dict[str, Dict[str, Any]],
        concurrency: int = 2,
        retries: int = 1,
        retry_delay: float = 10.0,
        follow_up_delay: float = 20.0,
        language_from: str = "question",
        retriever: Optional[str] = None,
        clarifier: Optional[PersonaClarifier] = None,
    ):
        self.client = client
        self.clarifier = clarifier
        self.run_id = run_id
        self.questions = questions
        self.personas = personas
        self.semaphore = asyncio.Semaphore(max(concurrency, 1))
        self.retries = max(retries, 0)
        self.retry_delay = retry_delay
        self.follow_up_delay = follow_up_delay
        self.language_from = language_from
        self.retriever = retriever
        self.responses_path = os.path.join(run_dir, "responses.jsonl")
        self.recorded = read_recorded(self.responses_path)
        # A follow-up that got no thread context needs a new thread, so its
        # parent is asked again too (and the follow-up after it).
        self._fresh_threads = {
            r["follow_up_of"] for r in self.recorded.values() if lacks_thread_context(r)
        }
        self._tasks: Dict[str, asyncio.Task] = {}
        self._write_lock = asyncio.Lock()

    async def run(self) -> Dict[str, Dict[str, Any]]:
        for q in self.questions:
            self._tasks[q["question_id"]] = asyncio.create_task(self._question(q))
        results = await asyncio.gather(*self._tasks.values())
        return {r["question_id"]: r for r in results}

    async def _question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        qid = q["question_id"]
        parent_id = q.get("follow_up_of")
        parent = await self._tasks[parent_id] if parent_id else None
        # A follow-up is only as current as its parent: when the parent is
        # asked again, so is the follow-up, in the parent's new thread.
        parent_kept = parent is None or parent is self.recorded.get(parent_id)
        previous = self.recorded.get(qid)
        # Re-asked: degraded answers, and clarifications nobody answered when
        # there is now a clarifier to answer them.
        previous_response = (previous or {}).get("response") or {}
        unanswered = bool(self.clarifier and previous_response.get("needs_clarification"))
        if (
            previous
            and previous.get("status") == "ok"
            and not search_failures(previous)
            and not unanswered
            and qid not in self._fresh_threads
            and parent_kept
        ):
            return previous

        thread_id = None
        if parent is not None:
            parent_response = parent.get("response") or {}
            thread_id = parent_response.get("qa_thread_id")
            # A thread with a pending clarification re-serves it to any
            # follow-up, so asking would only record the clarification again.
            if (
                parent.get("status") != "ok"
                or not thread_id
                or parent_response.get("needs_clarification")
            ):
                return await self._record(
                    q, request=None, status="skipped", http_status=None,
                    latency_ms=None, started_at=now_iso(), response=None,
                    error=f"parent {parent_id} has no answered thread",
                )
            if not parent_kept:
                await asyncio.sleep(self.follow_up_delay)

        payload = request_for(
            q, self.personas[q["persona_id"]],
            run_id=self.run_id, language_from=self.language_from,
            retriever=self.retriever, qa_thread_id=thread_id,
        )
        async with self.semaphore:
            return await self._ask(q, payload)

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One /qa/ask call with retries: body (or None), status, error, latency."""
        error = None
        http_status = None
        body = None
        t0 = time.monotonic()
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                resp = await self.client.post(ASK_PATH, json=payload)
                http_status = resp.status_code
                if resp.status_code == 200:
                    body, error = resp.json(), None
                    break
                error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code < 500:
                    break  # a 4xx will fail the same way again
            except ValueError as exc:  # a 200 whose body is not JSON
                error = f"invalid JSON body: {exc}"
            except httpx.HTTPError as exc:
                error = f"{exc.__class__.__name__}: {exc}"
            if attempt < self.retries:
                await asyncio.sleep(self.retry_delay)
        return {
            "body": body,
            "http_status": http_status,
            "error": error,
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }

    async def _ask(self, q: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        started_at = now_iso()
        result = await self._post(payload)
        clarification_turns = []
        body = result["body"]
        if self.clarifier and body and body.get("needs_clarification") and body.get("clarification"):
            answer = await self.clarifier.answer(
                q["question_id"], self.personas[q["persona_id"]],
                payload["question"], body["clarification"],
            )
            clarification_turns.append({
                "request": payload,
                "response": body,
                "latency_ms": result["latency_ms"],
                "answer": answer,
            })
            if "error" not in answer:
                payload = {
                    **payload,
                    "qa_thread_id": body.get("qa_thread_id"),
                    "clarification_response": answer["clarification_response"],
                }
                result = await self._post(payload)
                # A failed second call leaves no final answer; the
                # clarification itself is still in clarification_turns.
        return await self._record(
            q, request=payload,
            status="ok" if result["body"] is not None else "error",
            http_status=result["http_status"], latency_ms=result["latency_ms"],
            started_at=started_at, response=result["body"], error=result["error"],
            clarification_turns=clarification_turns,
        )

    async def _record(self, q: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
        record = {
            "run_id": self.run_id,
            "question_id": q["question_id"],
            "persona_id": q["persona_id"],
            "question_type": q.get("question_type"),
            "language": q.get("language"),
            "turn": q.get("turn", 1),
            "follow_up_of": q.get("follow_up_of"),
            **fields,
        }
        async with self._write_lock:
            with open(self.responses_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        response = record.get("response") or {}
        flag = " [cache hit]" if response.get("cache_hit") else ""
        for turn in record.get("clarification_turns") or []:
            answer = turn.get("answer") or {}
            given = (answer.get("clarification_response") or {})
            flag += (
                f" [clarification unanswered: {answer['error']}]" if "error" in answer
                else f" [clarified: {', '.join(given.get('selected_values') or []) or given.get('free_text')}]"
            )
        flag += " [clarification]" if response.get("needs_clarification") else ""
        flag += " [no thread context]" if lacks_thread_context(record) else ""
        flag += " [search failed]" if search_failures(record) else ""
        print(f"{record['status']:>7}  {q['question_id']}{flag}", file=sys.stderr)
        return record


def summarize(records: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    responses = [r.get("response") or {} for r in records.values()]
    return {
        "questions": len(records),
        "ok": sum(r.get("status") == "ok" for r in records.values()),
        "error": sum(r.get("status") == "error" for r in records.values()),
        "skipped": sum(r.get("status") == "skipped" for r in records.values()),
        # FoodScholar asked, answered or not, and asked with nobody answering.
        "clarifications_asked": sum(
            bool(r.get("clarification_turns")) or bool((r.get("response") or {}).get("needs_clarification"))
            for r in records.values()
        ),
        "needs_clarification": sum(bool(r.get("needs_clarification")) for r in responses),
        # Should stay 0 with bypass_cache; anything else means a stale answer.
        "cache_hit": sum(bool(r.get("cache_hit")) for r in responses),
        # Should stay 0; raise --follow-up-delay if not.
        "follow_up_without_context": sum(lacks_thread_context(r) for r in records.values()),
        # Answered, but a search errored (e.g. an Elasticsearch timeout);
        # resuming the run re-asks these.
        "degraded_retrieval": sum(
            r.get("status") == "ok" and search_failures(r) > 0 for r in records.values()
        ),
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


async def fetch_runtime_info(client: httpx.AsyncClient) -> Dict[str, Any]:
    resp = await client.get(RUNTIME_INFO_PATH)
    if resp.status_code == 404:
        # runtime-info and bypass_cache shipped together: an API without the
        # one silently ignores the other and would serve cached answers.
        raise RuntimeError(
            f"{RUNTIME_INFO_PATH} not found: this deployment predates "
            "bypass_cache, so a run would replay cached answers."
        )
    resp.raise_for_status()
    return resp.json()


async def main_async(args: argparse.Namespace) -> int:
    personas = {p["persona_id"]: p for p in load_json(args.personas)}
    all_questions = load_json(args.questions)
    validate(all_questions, personas)
    questions = select_questions(
        all_questions, personas=args.persona, types=args.type,
        ids=args.only, limit=args.limit,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"run {run_id}: {len(questions)} question(s)", file=sys.stderr)

    if args.dry_run:
        for q in questions:
            payload = request_for(
                q, personas[q["persona_id"]], run_id=run_id,
                language_from=args.language_from, retriever=args.retriever,
                qa_thread_id="<thread of %s>" % q["follow_up_of"] if q.get("follow_up_of") else None,
            )
            print(json.dumps({"question_id": q["question_id"], **payload}, ensure_ascii=False))
        return 0

    clarifier = None
    scripted = load_json(args.clarification_answers) if args.clarification_answers else {}
    if args.clarify == "persona":
        clarifier = PersonaClarifier(
            api_key=args.groq_api_key, model=args.clarify_model, scripted=scripted
        )
        await clarifier.check()

    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.api, timeout=timeout) as client:
        # Before the run directory exists: an unreachable or outdated API
        # leaves nothing behind.
        runtime_start = await fetch_runtime_info(client)
        run_dir = os.path.join(args.out, run_id)
        os.makedirs(run_dir, exist_ok=True)
        manifest_path = os.path.join(run_dir, "manifest.json")
        previous = load_json(manifest_path) if os.path.exists(manifest_path) else {}
        manifest = {
            "run_id": run_id,
            "api": args.api,
            # A resumed run keeps its original start; each resume is listed.
            "started_at": previous.get("started_at") or now_iso(),
            "resumed_at": previous.get("resumed_at", []) + ([now_iso()] if previous else []),
            "finished_at": None,
            "inputs": {
                "personas": {"path": os.path.abspath(args.personas), "sha256": file_digest(args.personas), "count": len(personas)},
                "questions": {"path": os.path.abspath(args.questions), "sha256": file_digest(args.questions), "count": len(all_questions)},
            },
            "selection": {
                "persona": args.persona, "type": args.type, "only": args.only,
                "limit": args.limit, "selected": len(questions),
            },
            "options": {
                "concurrency": args.concurrency, "retries": args.retries,
                "timeout_s": args.timeout, "follow_up_delay_s": args.follow_up_delay,
                "language_from": args.language_from, "retriever": args.retriever,
                # The API key is deliberately not recorded.
                "clarify": args.clarify,
                "clarify_model": args.clarify_model if clarifier else None,
                "clarification_answers": (
                    {"path": os.path.abspath(args.clarification_answers),
                     "sha256": file_digest(args.clarification_answers)}
                    if args.clarification_answers else None
                ),
            },
            "runtime_info_start": previous.get("runtime_info_start") or runtime_start,
            "runtime_info_end": None,
            "drift": [],
            "counts": None,
        }
        write_manifest(manifest_path, manifest)

        run = BenchmarkRun(
            client, run_id=run_id, run_dir=run_dir, questions=questions,
            personas=personas, concurrency=args.concurrency,
            retries=args.retries, follow_up_delay=args.follow_up_delay,
            language_from=args.language_from, retriever=args.retriever,
            clarifier=clarifier,
        )
        records = await run.run()
        runtime_end = await fetch_runtime_info(client)

    manifest["finished_at"] = now_iso()
    manifest["runtime_info_end"] = runtime_end
    manifest["drift"] = runtime_drift(manifest["runtime_info_start"], runtime_end)
    manifest["counts"] = summarize(records)
    write_manifest(manifest_path, manifest)

    print(json.dumps(manifest["counts"]), file=sys.stderr)
    if manifest["drift"]:
        print("WARNING: settings changed during the run: " + ", ".join(manifest["drift"]), file=sys.stderr)
    print(run_dir)
    return 0 if manifest["counts"]["error"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api", required=True, help="FoodScholar base URL, e.g. http://localhost:8001")
    parser.add_argument("--personas", required=True, help="Personas JSON (list with persona_id, api_mapping)")
    parser.add_argument("--questions", required=True, help="Questions JSON (list with question_id, persona_id, foodscholar_question)")
    parser.add_argument("--out", default="benchmark-runs", help="Directory that holds one folder per run")
    parser.add_argument("--run-id", help="Defaults to the UTC start time; reuse one to resume that run")
    parser.add_argument("--persona", action="append", help="Only this persona (repeatable)")
    parser.add_argument("--type", action="append", help="Only this question_type (repeatable)")
    parser.add_argument("--only", action="append", help="Only this question_id (repeatable)")
    parser.add_argument("--limit", type=int, help="At most this many questions (before parents are added)")
    parser.add_argument("--concurrency", type=int, default=2, help="Questions in flight at once")
    parser.add_argument("--retries", type=int, default=1, help="Retries on 5xx/network errors")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout, seconds")
    parser.add_argument(
        "--follow-up-delay", type=float, default=20.0,
        help="Seconds a follow-up waits after its parent, for the thread summary to be written",
    )
    parser.add_argument(
        "--language-from", choices=["question", "persona"], default="question",
        help="Answer language: the question row's (default) or the persona's",
    )
    parser.add_argument("--retriever", choices=["rag", "kggen", "no_rag"], help="Override the API's default retriever")
    parser.add_argument(
        "--clarify", choices=["persona", "none"], default="persona",
        help="Answer FoodScholar's clarifications as the persona (default), or record them and stop",
    )
    parser.add_argument("--clarify-model", default=CLARIFIER_MODEL, help="Groq model that picks the persona's answer")
    parser.add_argument("--groq-api-key", help="Groq key for --clarify persona; defaults to GROQ_API_KEY")
    parser.add_argument(
        "--clarification-answers",
        help="JSON {question_id: option value or [values]} that overrides the persona model",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the payloads that would be sent; no calls")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
