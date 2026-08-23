# The Agentic QA Pipeline

*Status: implemented. `QA_PIPELINE_MODE=agentic` (default); `legacy` restores
the previous single-pass flow.*

## Why

The previous QA flow was a fixed sequence — one clarifier call producing
exactly two queries, one kNN article search plus one BM25 guideline search,
an editorial tier boost, one blocking answer call, one JSON response. It
could not reason about *what* to search or *why*, never noticed when its
evidence was poor, ignored the bibliometrics already stored on every article
(Semantic Scholar citation counts), and made the UI wait through several
sequential LLM calls with nothing to show.

The pipeline in `src/services/qa_pipeline/` replaces that with a bounded
reasoning loop whose every stage is observable over SSE:

```
plan (typed sub-questions, each with a user-visible "why" and
     attribute filters)
  → parallel hybrid retrieval per sub-question
      articles:   BM25 leg + informed kNN leg, fused client-side (RRF)
      guidelines: gated BM25 leg (+ gated kNN leg in hybrid mode)
  → deterministic rerank
      adjusted = rrf_norm × tier × recency × influence × study_design
  → sufficiency evaluator (LLM judge + research NOTES)
  → diagnosed repair round (bounded, targeted)  ─loop back once─┐
  → token-streamed answer with a citation trailer               │
                                                                └→ answer
```

## Module map

| File | Role |
|---|---|
| `qa_pipeline/orchestrator.py` | `run_pipeline(service, request)` — the loop, an async generator of `PipelineEvent` |
| `qa_pipeline/retrieval.py` | hybrid legs, attribute clauses, client-side RRF, evidence merging |
| `qa_pipeline/ranking.py` | recency / influence / study-design / tier adjustment, diversity cap, min-score threshold |
| `qa_pipeline/evaluator.py` | deterministic gates + LLM sufficiency judge + research notes |
| `qa_pipeline/repair.py` | diagnosis → targeted re-search mapping |
| `qa_pipeline/answering.py` | `astream` answer, sentinel protocol, citation validation |
| `qa_pipeline/state.py`, `events.py` | dataclasses and SSE framing |
| `src/agents/qa_planner.py` | `QAPlannerAgent` — safety + clarification + search decomposition in one call |

`QAService.answer_question` is now a dispatcher: agentic mode drains
`run_pipeline()` and returns the terminal payload (so `POST /qa/ask` keeps
its exact contract); `QA_PIPELINE_MODE=legacy` runs the old body, kept
verbatim as `_answer_question_legacy` (now with its blocking calls wrapped in
`asyncio.to_thread`).

## The planner

One LLM call (`QA_PLANNER_MODEL`, temp 0) produces everything the legacy
clarifier did — safety flags, guardrails, one material clarification,
canonical question — plus 1–`QA_MAX_SUBQUESTIONS` typed sub-questions. Each
carries:

- `why` — one user-visible sentence, streamed while the search runs;
- `qtype` (`quantity | mechanism | safety | recommendation | comparison |
  general`) and `branch` (`articles | guidelines | both`) — quantity and
  recommendation questions go to guidelines, mechanism and comparison to
  articles, safety to both;
- `lexical_query` (BM25 keywords) and `dense_query` (a full sentence for the
  vector leg);
- `filters` — structured attribute constraints extracted from the question
  (see below).

A deterministic fallback (one article search + one guideline search) covers
LLM failure; nothing user-facing depends on the planner succeeding.

## Metadata-aware ("informed") retrieval

`SubQuestionFilters` carries attribute constraints the planner extracted:
`year_min`/`year_max` ("recent evidence", "since 2020"), `open_access`,
`study_types`, `regions`, `target_populations`, `food_groups`, `nutrients`.

Application policy (`retrieval.article_attribute_clauses`):

- **Hard filters** only for deterministic, mapping-backed fields — the
  publication-year window and `open_access`. These are applied to the BM25
  leg **and inside the kNN leg's filter**, so vector search is a semantic
  search *within* the question's constraints, not a blind nearest-neighbour
  sweep.
- **Boosts** for everything vocabulary-shaped — study designs (matched
  against the LLM-assigned `ai_category`, whose coverage is partial),
  regions, populations, food groups, nutrients. Boosting instead of gating
  means an enrichment gap can never silently empty retrieval.
- Guidelines keep exactly one hard filter: the editorial gate
  (`status: active`, built only by `qa_retrievers.guideline_base_query` /
  `guideline_retrieval_filter`). Question attributes and user context both
  contribute `should` clauses. `tests/test_guideline_retrieval_gate.py`
  scans the pipeline's retrieval source to keep it that way.
- **Guideline hybrid is the default** (`QA_GUIDELINE_RETRIEVAL_MODE=hybrid`)
  now that guidelines carry embeddings: the pipeline runs a gated kNN leg and
  RRF-fuses it with BM25. `bm25` remains the opt-out for un-embedded
  deployments.
- **Age-group facets**: `member_age_group` (or an answered age clarification)
  maps to the enrichment vocabulary — `life_stage` terms plus the
  `age_min_months`/`age_max_months` window — as overlap boosts in
  `guideline_age_should_clauses` (shared by legacy, pipeline, and any future
  caller of `guideline_context_should_clauses`). Rules with the -1 "not
  stated" sentinel simply never earn the window boost.
- `health_conditions` joins the planner-extracted filters and boosts
  `health_conditions^4` on guidelines and abstract/tags on articles;
  `ai_tags` joins the article BM25 fields and facet boosts.

Legs are fused client-side with reciprocal rank fusion
(`score = Σ 1/(QA_RRF_K + rank)`, normalized to (0, 1]); a document found by
several sub-questions keeps its best score and accumulates the sub-question
ids that surfaced it.

## Ranking

`adjusted = rrf_norm × tier × recency × influence × study_design × affinity`,
all factors multiplicative around 1.0 and neutral on missing data:

- **tier** — the existing editorial boosts (`article_policy.tier_boost`).
- **recency** — exponential decay, half-life `QA_RECENCY_HALF_LIFE_YEARS`
  (6y), floored at `QA_RECENCY_FLOOR` (0.35): an old meta-analysis is
  discounted, not erased.
- **influence** — `1 + QA_INFLUENCE_WEIGHT · log1p(citations + 2·influential)
  / log1p(QA_INFLUENCE_CITATION_CAP)`, reading the Semantic Scholar fields
  already stored per article (`citationCount` / `citation_count`,
  `influentialCitationCount` / `influential_citation_count` — both spellings).
  Boost-only: a new paper with no citation history is never penalized.
- **study_design** — from `ai_category`: meta-analysis/systematic review 1.3,
  RCT 1.2, cohort 1.1, case report/in-vitro/animal 0.85.
- **affinity** (guidelines only) — how well a rule's enrichment facets fit
  the asker and the question: region match ×1.15, population/life-stage
  match (the age group mapped through the facet vocabulary) ×1.15, topical
  overlap on food groups/nutrients/health conditions ×1.1. Boost-only, so an
  unenriched rule ranks purely on relevance.

The answer prompt additionally shows each guideline's applicability facets
(life stage, human-readable age window, nutrients, health conditions,
action/frequency) and each article's citation counts, and the evaluator's
evidence digest carries study type + citations — both LLM judgments weigh
evidence quality and applicability, not just retrieval scores.

Then a per-document diversity cap (`QA_PER_DOC_CAP`, default 2 per parent
document), at most 5 guidelines, `top_k` articles, and the min-score
threshold `QA_MIN_SCORE` — **an empty evidence set is representable** and
flows to the evaluator instead of being papered over.

## Evaluator, repair, and research notes

Deterministic gates run first (all branches failed → `corpus_gap`; repair
budget spent → answer with what exists). Otherwise one small-model call
(`QA_EVALUATOR_MODEL`) judges coverage per sub-question and returns one
verdict:

| Verdict | Repair action |
|---|---|
| `sufficient` | answer |
| `vocabulary_mismatch` | swap in reformulated queries; re-search only the uncovered sub-questions |
| `wrong_granularity` | flip the sub-question's branch; re-search it |
| `decomposable_residue` | add 1–2 new sub-questions; search only those |
| `corpus_gap` | no retry — the answer discloses the gap honestly |
| `needs_user_clarification` | store the thread, emit `clarification`, end the stream |

One repair round by default (`QA_MAX_REPAIR_ROUNDS=1`). The
one-clarification-per-thread rule is enforced by the orchestrator, not
trusted to the model. The legacy "scout" region clarification survives as the
`needs_user_clarification` verdict with options derived from the retrieved
guideline regions.

**Research notes** are the evaluator's second product (same call, no extra
cost): 2–5 terse `ResearchNote`s of kind `finding` ("Two RCTs support
omega-3 lowering triglycerides"), `gap` (what the corpus lacked), or `lead`
(a direction worth searching next). Notes

- stream to the UI as `stage.notes` events,
- steer the repair round (they are in the evaluator's own context),
- persist per conversation thread in Redis (`qa_notes:{thread_id}`, 1 h TTL,
  capped at 20) so a **follow-up question's planner starts from what earlier
  searches established** — the prompt tells it not to re-search settled
  findings and to turn leads into sub-questions,
- land in `qa_requests.pipeline_meta` for analysis.

## Streaming answer

One `llm.astream` call, two-part output: the markdown answer (streamed as
`answer_delta` events), then the sentinel line `<<<END_ANSWER>>>`, then a
JSON trailer with verbatim citations, overall confidence, and follow-ups.
Citation building reuses the same module-level machinery as the legacy path
(`agents/qa_agent.py`: quote coercion to an exact source span, G-labels).

Defensive paths: a sentinel split across chunks is never emitted (a
sentinel-sized tail is held back); a model that ignores the protocol and
emits raw JSON is detected by its first character, buffered, and parsed as
the legacy shape; a missing trailer recovers citations from the inline
markdown links; a mid-stream provider failure still answers from what
streamed.

## SSE contract — `POST /api/v1/qa/ask/stream`

Body: the existing `QARequest`. Response: `text/event-stream`, frames
`event: <name>\ndata: <json>\n\n`, keep-alive comments every
`QA_STREAM_HEARTBEAT_SECONDS`. Every payload carries `request_id` and a
monotonic `seq`.

| Event | Payload highlights |
|---|---|
| `step` | **the collapsible-steps channel**: a `ReasoningStep` `{id, kind, status, title, detail, round, elapsed_ms, data}` — emitted once with `status: running` and again (same `id`) with `status: done` + duration. Kinds: `plan, search, rank, notes, evaluate, repair, answer, cache, clarification`. Titles/details are ready-to-render (dynamic parts localized via the prompts); `kind`+`data` carry the structured form. The full timeline also rides `done`/`clarification` payloads as `QAResponse.reasoning_steps`, so the disclosure survives stream end, reloads, cache replays, and the non-streaming endpoint. |
| `stage.start` | question, mode, model, retriever |
| `stage.plan` | canonical question, risk level, sub-questions with `why` + filters, prior notes |
| `stage.search_started` | sub_question_id, branch, why, lexical_query, round |
| `stage.search_results` | hit_count, ok, top-3 titles |
| `stage.rerank` | kept, dropped {below_threshold, over_doc_cap, over_budget}, top items with `score_parts` |
| `stage.notes` | new research notes (kind, text, source_urns) |
| `stage.evaluate` | round, verdict, reason, gaps |
| `stage.repair` | actions per sub-question |
| `stage.cache` | `{hit: true}` — followed by synthetic answer_delta replay |
| `answer_started` / `answer_delta` | model / text chunk |
| `citations` | validated `QACitation`s, confidence, follow-ups |
| `done` **(terminal)** | the full `QAResponse` (incl. memory_suggestions) for state reconciliation |
| `clarification` **(terminal)** | `QAResponse` with `needs_clarification`, `qa_thread_id` — client re-POSTs with `clarification_response`, same round-trip as `/qa/ask` |
| `error` **(terminal)** | title, detail, cause — emitted, not raised, once streaming began |

Cache hits replay through the same event vocabulary so the UI has one code
path. Invalid requests (bad model in advanced mode) are rejected with a 400
*before* the stream starts.

### Gateway and UI bridge (implemented in the sibling repos)

- **wisefood-api**: `POST /api/v1/foodscholar/qa/ask/stream`
  (`src/routers/foodscholar.py`) — auth + guest budget + member verification,
  then `FoodScholar.ask_question_stream` (`src/backend/foodscholar.py`)
  re-yields the upstream SSE bytes unbuffered through a `StreamingResponse`
  (no APIEnvelope). The stream is primed before the response starts so an
  upstream failure is a normal HTTP error, not a dead 200. The gateway's
  `QARequest` already mirrored every field. Read timeout is widened to 180 s;
  the upstream keep-alives feed it.
- **wisefood-ui**: `wisefoodRestApi.postStream()` (fetch + Authorization
  header — EventSource can't carry one), `foodscholarApi.askQuestionStream()`
  (SSE frame parser), `useFoodScholarQaStream` composable (live `steps`,
  `streamingAnswer`, sticky fallback to the classic call on a 404/405 from an
  older gateway), and `components/foodscholar/ReasoningSteps.vue` — the
  ChatGPT-style collapsible disclosure: narrates the running step in its
  header while live, auto-collapses to "How this was researched · N steps"
  when the answer lands, and re-renders from `reasoning_steps` on restored or
  cached answers. Wired into `pages/foodscholar/index.vue` (live steps +
  token-streamed markdown replace the skeleton; settled answers show the
  collapsed timeline).
- If a reverse proxy sits in front of either hop, verify it honors
  `X-Accel-Buffering: no` / has `proxy_buffering off` for this route.

## Configuration

All in `src/config.py`:

| Env var | Default | Meaning |
|---|---|---|
| `QA_PIPELINE_MODE` | `agentic` | `legacy` = rollback to the pre-pipeline flow |
| `QA_PLANNER_MODEL` | `QA_FAST_MODEL` | planner LLM |
| `QA_EVALUATOR_MODEL` | `QA_UTILITY_MODEL` | sufficiency judge LLM |
| `QA_MAX_SUBQUESTIONS` | 3 | plan decomposition cap (+2 headroom for repair additions) |
| `QA_MAX_REPAIR_ROUNDS` | 1 | bounded repair budget |
| `QA_RRF_K` | 60 | RRF constant |
| `QA_RRF_CANDIDATES` | 30 | per-leg candidate pool |
| `QA_RECENCY_HALF_LIFE_YEARS` | 6.0 | recency decay half-life |
| `QA_RECENCY_FLOOR` | 0.35 | old-paper discount floor |
| `QA_INFLUENCE_WEIGHT` | 0.3 | max citation boost (→ ×1.3) |
| `QA_INFLUENCE_CITATION_CAP` | 1000 | citations reaching full boost |
| `QA_MIN_SCORE` | 0.05 | evidence threshold (empty is representable) |
| `QA_PER_DOC_CAP` | 2 | diversity cap per parent document |
| `QA_STREAM_HEARTBEAT_SECONDS` | 15 | SSE keep-alive cadence |

Changing the ranking knobs or the pipeline mode invalidates the QA cache
naturally (they are part of the version-4 cache key).

## Prompts and tracing

New registry prompts (seeded to Langfuse on first deploy by the create-only
`sync_prompts`): `qa-planner-system`, `qa-evaluator-system`,
`qa-answer-stream-system`, `qa-answer-stream-user`. **Remember: once they
exist in Langfuse, the managed version wins — fallback edits in
`prompts.py` must be mirrored in the Langfuse UI.**

Langfuse run names per stage: `qa-planner`, `qa-evaluator`,
`qa-answer-stream` (plus the existing `qa-conversation-summary`), correlated
by an opaque `request_id` in metadata. PII policy unchanged: only opaque ids
and tags in trace metadata.

## Persistence

`qa_requests` gains a nullable `pipeline_meta JSONB` column (idempotent
`ALTER TABLE` in `db_init._apply_schema_updates`): sub-questions with
rationales, rounds, verdicts, repairs, notes, evidence counts, stage timings.

## What changed for existing features

- **Dual-answer A/B** — dropped from the agentic path (two answers cannot
  both token-stream; doubles cost). Contract fields remain, always null;
  the code lives on in the legacy path.
- **Scout clarification** — folded into the evaluator verdict.
- **Conversation summary** — kept; now a fire-and-forget async task after
  `done`.
- **Memory suggestions** — kept; computed by the pipeline and embedded in
  `done` (the `/qa/ask` handler only fills them in when absent, i.e. legacy).
- **`retriever=linearrag`** — single pass through the legacy adapter, then
  the same ranking and streamed answer; **`no_rag`** skips retrieval and
  streams a general-knowledge answer with no citations.
- **Tips / starter questions / feedback / models endpoints** — untouched.

## Tests

`PYTHONPATH=src python -m pytest tests/ -q`. New files:
`test_qa_pipeline_ranking.py`, `test_qa_pipeline_retrieval.py`,
`test_qa_planner.py`, `test_qa_evaluator.py`, `test_qa_answer_stream.py`,
`test_qa_pipeline_orchestrator.py`, `test_qa_sse_endpoint.py`. The guideline
gate suite now also scans the pipeline's retrieval functions.

## Deferred

Cross-encoder reranking (slots into `ranking.py` behind a future
`QA_RERANKER_MODEL` flag), guideline-embedding backfill (existing
`QA_GUIDELINE_RETRIEVAL_MODE=hybrid` flag), nested Langfuse spans, deleting
the legacy path + dual-answer code once agentic proves stable, moving
`QAService._qa_threads` fully to Redis for multi-replica correctness.
