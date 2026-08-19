# FoodScholar

<!-- Badges -->
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3128/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1c3c3c.svg?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-f55036.svg)](https://groq.com/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.14-005571.svg?logo=elasticsearch&logoColor=white)](https://www.elastic.co/)
[![Redis](https://img.shields.io/badge/Redis-6.4-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-async-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Langfuse](https://img.shields.io/badge/observability-Langfuse-0a0a0a.svg)](https://langfuse.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Catalog](https://img.shields.io/badge/catalog-wisefood--data--api-181717.svg?logo=github&logoColor=white)](https://github.com/wisefood/wisefood-data-api)
[![WiseFood EU](https://img.shields.io/badge/WiseFood-EU%20Project-2e7d32.svg)](https://wisefood-project.eu/)

**FoodScholar** is the AI reasoning layer of the WiseFood EU platform. It turns a curated corpus of
nutrition science — peer-reviewed articles and national dietary guidelines — into answers a
non-specialist can act on, without severing the link to the evidence behind them.

It provides a single backend for:

- grounded question answering over articles and dietary guidelines, with citations
- multi-document synthesis of search results into a summary with confidence levels
- conversational chat sessions with consented, per-user memory
- AI enrichment of catalog articles (plain-language abstracts, glossaries, tiered Q&A, scoring)
- extraction of structured dietary guidelines from source PDFs using a vision model
- editorial control over which articles reach which readers, and how strongly each is favoured

The service is built with FastAPI and LangChain. It reads its corpus from Elasticsearch, calls
Groq for chat/annotation inference and OpenAI for vision-based PDF extraction, keeps job state and
caches in Redis, persists Q&A records and extraction results in PostgreSQL, and traces every LLM
call to Langfuse when observability is enabled.

> **New here?** Jump to [Architecture](#architecture) for the component map, [API Reference](#api-reference)
> to start calling the service, or [Running the Service](#running-the-service) to get it up locally.

---

## Contents

- [Role in the WiseFood Platform](#role-in-the-wisefood-platform)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Retrieval & Editorial Policy](#retrieval--editorial-policy)
- [Background Workers](#background-workers)
- [Prompt Registry & Observability](#prompt-registry--observability)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Building the Image](#building-the-image)
- [Testing](#testing)
- [Development Notes](#development-notes)
- [Operational Caveats](#operational-caveats)
- [License](#license)

---

## Role in the WiseFood Platform

FoodScholar **does not own the corpus**. The authoritative store for articles, guides, guidelines
and their attached source files is the sibling
[**WiseFood Data API**](https://github.com/wisefood/wisefood-data-api), which also builds and owns
the Elasticsearch indices. FoodScholar is a consumer and an enricher of that catalog:

```
        ┌──────────────────┐         ┌──────────────────────┐
        │   wisefood-ui    │         │     wisefood-api     │
        │ reader + console │────────▶│    REST gateway      │
        └────────┬─────────┘         │  (auth, proxying)    │
                 │                   └──────────┬───────────┘
                 │ reads catalog                │ proxies /enrich/*
                 │ directly                     ▼
                 │                   ┌──────────────────────┐
                 │                   │    FoodScholar       │
                 │                   │  (this repository)   │
                 │                   └───┬──────────────┬───┘
                 │                       │              │
                 ▼                       │ reads        │ writes enrichment
        ┌──────────────────┐             │ indices      │ via Data API
        │ wisefood-data-api│◀────────────┴──────────────┘
        │  catalog + ES    │
        └──────────────────┘
```

- **Reads** the `articles` and `guidelines` Elasticsearch indices directly for retrieval.
- **Writes** enrichment back through the Data API (`PATCH /articles/{urn}` and
  `PATCH /articles/{urn}/enhance`) using the `wisefood` Python client, so the catalog stays the
  single source of truth.
- **Honours** editorial fields the catalog owns (`reader_visibility`, `indexing_tier`) rather than
  keeping any policy state of its own.
- **Is fronted** by the `wisefood-api` gateway, which supplies authentication. FoodScholar has no
  auth of its own — see [Operational Caveats](#operational-caveats).

---

## Capabilities

### 1. Grounded question answering

`POST /api/v1/qa/ask` answers a nutrition question against retrieved evidence. The pipeline is
deliberately more than "embed, retrieve, prompt":

- **Clarification & safety planning** — a structured pre-pass decides whether the question is
  answerable as asked, what to retrieve, and whether it touches a safety-sensitive area. When a
  detail genuinely changes the answer (country, age group), the service asks one targeted question
  and resumes from a short-lived thread rather than guessing.
- **Three retrieval strategies** — `rag` (Elasticsearch kNN over articles plus keyword search over
  guideline rules), `linearrag` (a graph-based passage retriever), or `no_rag` (model knowledge
  only, clearly labelled).
- **Expertise-adjusted answers** — `beginner` · `intermediate` · `expert` change both the prose and
  which articles are eligible as evidence.
- **Citations** — every answer carries the sources it used, with per-source scores.
- **Multilingual** — answers are produced in the requested ISO-639-1 language, including follow-ups.

### 2. Search summarisation

`POST /api/v1/search/summarize` accepts pre-fetched search results and synthesises them into one
markdown summary with key findings, citations mapped to specific articles, and follow-up questions.
Results are cached, and the cache can be invalidated per-pattern.

### 3. Chat sessions with consented memory

`/api/v1/sessions/*` manages conversational threads: session creation, chat turns, history,
per-user listing, and deletion. Sessions live in Redis and expire after `SESSION_TTL_SECONDS` of
inactivity, so guest state is reaped automatically.

Memory is **consent-first**. When a member phrases a durable preference, the service raises a nudge
("It seems you love lentils. Remember this?") and writes nothing until it is accepted. Acceptance
writes to the shared member profile with `source: "foodscholar"` provenance; a decline is recorded
as an opt-out, so neither FoodScholar nor FoodChat asks again. Safety-relevant data always requires
explicit consent regardless of confidence.

### 4. Article enrichment

Two producers write enrichment, both through the same persistence path so a manually enriched
article is byte-for-byte identical to a swept one:

- a **catalog sweeper** that walks the corpus in cursor order, and
- an **on-demand job queue** the console drives per article.

Enrichment produces homogenised keywords, a plain-language abstract rewrite, a glossary, three
tiers of Q&A (user / practitioner / expert), study-type classification, scoring, and a proposed
indexing tier.

### 5. Guideline extraction from PDFs

`/api/v1/guidelines/*` turns a national dietary-guideline PDF, attached to the catalog as an
artifact, into structured guideline records: pages are rendered and triaged, surviving pages are
extracted by a vision model under a strict JSON schema, results are persisted, then imported onto
the parent guide. A second **facet enrichment** pass adds the retrieval facets that make a rule
attributable (population, region, food groups), followed by explicit **activation**.

### 6. Editorial control over retrieval

Editors set two fields on catalog articles; FoodScholar enforces them at retrieval time. See
[Retrieval & Editorial Policy](#retrieval--editorial-policy).

---

## Architecture

### Three-layer architecture

FoodScholar separates HTTP concerns from business logic from LLM interaction. Dependencies flow in
one direction and layers are never skipped.

```
┌─────────────────────────────────────────────────────────┐
│                 Layer 1: API LAYER                      │
│              FastAPI HTTP Interface                     │
│  • Request/response handling                            │
│  • Input validation (Pydantic)                          │
│  • Consistent error envelope                            │
│              src/api/v1/ · src/routers/                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 2: SERVICE LAYER                     │
│           Business Logic Orchestration                  │
│  • Caching strategy                                     │
│  • Retrieval, policy enforcement, ranking               │
│  • Job orchestration & persistence                      │
│  • Data transformation, validation, error handling      │
│                     src/services/                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Layer 3: AGENT LAYER                      │
│          AI-Powered Intelligence (LangChain)            │
│  • LLM interactions (Groq, OpenAI)                      │
│  • Prompt composition from the registry                 │
│  • Multi-step reasoning & structured output recovery    │
│                      src/agents/                        │
└─────────────────────────────────────────────────────────┘
```

**[View the full architecture walkthrough →](docs/ARCHITECTURE_PRESENTATION.md)**

### Agents vs tools vs services

A distinction worth internalising before adding code, because putting logic in the wrong layer is
the most common source of drift in this repository.

| | Role | Owns | Example |
|---|---|---|---|
| 🤖 **Agent** | Intelligent decision-maker powered by an LLM | Prompt composition, multi-step reasoning, output parsing | `EnrichmentAgent` decides how to annotate an article |
| 🔧 **Tool** | A single capability an agent can use | One operation, no decisions | `synthesize_articles()`, `extract_keywords()` |
| ⚙️ **Service** | Business-logic orchestrator | Caching, validation, persistence, coordinating agents | `QAService` runs the whole answer pipeline |

**Analogy:** tools are kitchen appliances, agents are the chef deciding which to use, services are
the restaurant manager coordinating orders, inventory and staff.

### Component map

| Module | Responsibility |
|---|---|
| **`src/app.py`** | FastAPI app, lifespan (DB init, retriever warm-up, worker start/stop, prompt sync), CORS, router registration, `/health` |
| **`src/config.py`** | Every environment variable, read once into `config.settings` |
| **`src/routers/generic.py`** | Response envelope (`APIEnvelope`) and global exception handlers |
| **`src/api/v1/`** | HTTP surface: `qa`, `search`, `sessions`, `enrich`, `guidelines` |
| **`src/services/qa_service.py`** | The Q&A pipeline: planning, caching, retrieval, answer assembly, persistence |
| **`src/services/qa_retrievers.py`** | Retriever adapters (`rag`, `linearrag`, `no_rag`) normalising evidence into one shape |
| **`src/services/article_policy.py`** | Reader-visibility filtering and indexing-tier ranking |
| **`src/services/search_summarizer.py`** | Search-result synthesis with caching |
| **`src/services/memory_service.py`** | Consented per-user memory extraction and storage |
| **`src/services/session_store.py`** | Redis-backed chat sessions with TTL |
| **`src/services/enrichment_jobs.py`** | Article enrichment persistence + Redis job orchestration |
| **`src/services/article_enricher.py`** | Stateless enrichment for inline article payloads |
| **`src/services/guideline_extractor.py`** | PDF workspace, page triage/rendering, vision extraction |
| **`src/services/guideline_jobs.py`** | Extraction job orchestration and result persistence |
| **`src/services/guideline_enricher.py`** | Post-extraction facet enrichment of guideline rules |
| **`src/services/guideline_corpus.py`** | Corpus-level audit and the activation that makes rules retrievable |
| **`src/services/linearrag_service.py`** | Lazy singleton around the LinearRAG graph retriever |
| **`src/services/model_backoff.py`** | Retry-with-backoff wrapper for model calls |
| **`src/agents/`** | `qa_agent`, `qa_clarifier`, `synthesis_agent`, `enrichment_agent`, `guideline_enrichment_agent`, plus `json_output` for recovering JSON from imperfect completions |
| **`src/workers/`** | Four background threads — see [Background Workers](#background-workers) |
| **`src/backend/`** | Infrastructure adapters: `groq`, `elastic`, `redis`, `postgres`, `platform` (WiseFood clients), `prompts`, `langfuse`, `db_init` |
| **`src/models/`** | Pydantic request/response contracts and SQLAlchemy tables (`db.py`) |
| **`src/utilities/`** | `cache`, `chunking`, `citation_validator` |

### Runtime dependencies

| Dependency | Role | Required |
|---|---|---|
| **Elasticsearch 8.14** | Article + guideline retrieval (kNN and keyword). Indices are created and owned by the Data API | Yes |
| **Redis** | Job queues, worker coordination and locks, response caches, chat sessions | Yes |
| **PostgreSQL** | Q&A request/feedback records, starter questions, tips, guideline extraction and enrichment results | Yes |
| **WiseFood Data API** | Catalog reads and enrichment writes, via the `wisefood` client | Yes |
| **WiseFood Platform API** | Member and household context used to personalise answers | Yes |
| **Groq** | Chat and annotation inference (`GROQ_API_KEY`) | Yes |
| **OpenAI** | Vision-based guideline extraction from PDF pages (`OPENAI_API_KEY`) | For guideline extraction |
| **Sentence Transformers** | `all-MiniLM-L6-v2` query embeddings (384-dim, matching `ES_DIM`) | Yes (baked into the image) |
| **scispaCy** | `en_core_sci_sm` biomedical NER for LinearRAG | Yes (baked into the image) |
| **Langfuse** | LLM tracing and the prompt registry | Optional |

### Request lifecycle: `POST /api/v1/qa/ask`

1. **Validate** — FastAPI parses the request into `QARequest`.
2. **Resolve context** — member/household context is fetched from the Platform API when available.
3. **Plan** — the clarifier produces a structured plan: canonical question, retrieval queries, safety
   flags. If a clarification is required and would change the answer, the service returns it with a
   thread id instead of answering.
4. **Cache** — a key derived from question, retriever, model and user context is checked first.
5. **Retrieve** — the selected adapter fetches articles and guideline rules. Editorially restricted
   articles are excluded **in the Elasticsearch query**, so they never consume a `top_k` slot.
6. **Apply policy** — a second pass covers retrievers that cannot pre-filter and applies the
   indexing-tier boost, re-ranking articles among themselves.
7. **Answer** — the QA agent composes the prompt from the registry and calls Groq.
8. **Persist & trace** — the record is written to PostgreSQL, the response cached, and the LLM call
   traced to Langfuse.

### Enrichment lifecycle

```
console ──▶ POST /enrich/articles/{urn} ──▶ Redis queue ──▶ EnrichmentJobWorker
                                                                    │
sweeper ──▶ catalog cursor scan ─────────────────────────▶ EnrichmentAgent
                                                                    │
                                            keywords + annotation (Groq)
                                                                    │
                                              persist_enrichment(article)
                                                    │            │
                            PATCH /articles/{urn}   │            │  PATCH .../enhance
                            (standard fields)       ▼            ▼  (ai_* fields)
                                              WiseFood Data API
```

Both producers share `persist_enrichment`, and both record bookkeeping in Redis so the sweeper never
redoes an article the console just handled.

### Guideline extraction lifecycle

```
artifact PDF ──▶ download to workspace ──▶ render pages (PyMuPDF, GUIDELINE_RENDER_DPI)
                                                    │
                                      triage each page (skip TOC, covers, refs…)
                                                    │
                            extract surviving pages (OpenAI, strict JSON schema)
                                                    │
                            persist to PostgreSQL ──▶ import onto parent guide
                                                    │
                        facet enrichment (Groq) ──▶ activation ──▶ retrievable
```

---

## Repository Structure

```text
.
├── README.md
├── Dockerfile
├── Makefile                              # image build + push
├── docker-compose.yaml                   # dev convenience (see caveats)
├── requirements.txt
├── extract_guidelines.py                 # CLI wrapper around the extraction service
├── scripts/
│   └── seed_langfuse_prompts.py           # idempotent prompt seeding
├── docs/
│   ├── ARCHITECTURE_PRESENTATION.md       # architecture walkthrough
│   ├── BACKGROUND_WORKER.md               # worker design and operations
│   ├── guideline-facets-and-retrieval.md  # facets, context-aware extraction, retrieval gate
│   └── langfuse-integration-guide.md      # observability integration
├── src/
│   ├── app.py                            # FastAPI app + lifespan
│   ├── config.py                         # environment configuration
│   ├── entity.py                         # base entity abstraction
│   ├── exceptions.py
│   ├── logsys.py                         # logging setup
│   ├── schemas.py
│   ├── utils.py
│   ├── api/v1/
│   │   ├── qa.py                         # question answering
│   │   ├── search.py                     # search summarisation
│   │   ├── sessions.py                   # chat sessions
│   │   ├── enrich.py                     # article enrichment + workers
│   │   └── guidelines.py                 # guideline extraction/enrichment/corpus
│   ├── routers/
│   │   └── generic.py                    # response envelope + error handlers
│   ├── services/
│   │   ├── qa_service.py
│   │   ├── qa_retrievers.py
│   │   ├── article_policy.py
│   │   ├── search_summarizer.py
│   │   ├── memory_service.py
│   │   ├── session_store.py
│   │   ├── article_enricher.py
│   │   ├── enrichment_jobs.py
│   │   ├── guideline_extractor.py
│   │   ├── guideline_jobs.py
│   │   ├── guideline_enricher.py
│   │   ├── guideline_enrichment_jobs.py
│   │   ├── guideline_corpus.py
│   │   ├── model_backoff.py
│   │   ├── linearrag_service.py
│   │   └── linearrag/                    # graph retriever implementation
│   ├── agents/
│   │   ├── qa_agent.py
│   │   ├── qa_clarifier.py
│   │   ├── clarifier_fallback_i18n.py
│   │   ├── synthesis_agent.py
│   │   ├── enrichment_agent.py
│   │   ├── guideline_enrichment_agent.py
│   │   └── json_output.py
│   ├── workers/
│   │   ├── enrichment_worker.py          # catalog sweeper
│   │   ├── enrichment_job_worker.py      # on-demand enrichment
│   │   ├── guideline_extraction_worker.py
│   │   └── guideline_enrichment_worker.py
│   ├── backend/
│   │   ├── groq.py                       # ChatGroq connection pool
│   │   ├── elastic.py                    # Elasticsearch client + kNN helpers
│   │   ├── redis.py                       # Redis singleton
│   │   ├── postgres.py                   # async SQLAlchemy engine
│   │   ├── db_init.py                     # schema/table bootstrap
│   │   ├── platform.py                    # WiseFood Data + Platform client pools
│   │   ├── prompts.py                     # prompt registry (Langfuse + fallbacks)
│   │   └── langfuse.py                    # optional tracing
│   ├── models/                           # Pydantic contracts + SQLAlchemy tables
│   ├── utilities/                        # cache, chunking, citation validation
│   └── data/linearrag/                   # prebuilt graph + embeddings (shipped)
└── tests/                                # 179 tests
```

---

## Retrieval & Editorial Policy

Two catalog fields govern how an article may be used as evidence. Both are owned by
[wisefood-data-api](https://github.com/wisefood/wisefood-data-api) and set by editors from the
console; FoodScholar reads and enforces them.

### `reader_visibility`

| Value | Effect |
|---|---|
| `public` | Every reader. The default, and what an absent value means. |
| `expert_only` | Withheld from `beginner` and `intermediate` readers; experts still see it. |
| `hidden` | Withheld from all readers. Still visible to editors in the console. |

### `indexing_tier`

Scales the retrieval score, so ranking reflects editorial judgement and not just cosine similarity.
`prime` sits above every tier the enrichment agent can assign, making it unambiguously a human
decision. The agent's own proposal is written to `ai_indexing_tier` and never overwrites the
editor's field.

| Tier | Boost | Meaning |
|---|---|---|
| `prime` | ×1.6 | Influential work, surfaced ahead of better-matching ordinary articles |
| `core` | ×1.25 | Strong evidence, favoured |
| `supportive` | ×1.0 | Neutral — ranked purely on relevance |
| `specialized` | ×0.9 | Narrow relevance, slightly de-prioritised |
| `archive_only` | ×0.6 | Kept for completeness, rarely surfaced |
| `do_not_index` | — | Excluded from retrieval entirely |

### Enforcement rules

- **Excluded in-query.** The Elasticsearch filter drops restricted articles before `top_k` is
  applied, so a hidden article never displaces a usable one.
- **Post-pass for non-ES retrievers.** LinearRAG has no queryable policy fields, so a second pass
  filters and ranks uniformly across all retrievers.
- **Absent means permissive.** Every filter is phrased as an exclusion. Articles indexed before
  these fields existed carry neither, and a positive clause would have hidden the entire legacy
  corpus.
- **Guidelines are untouched.** Articles re-rank among themselves and guideline results keep their
  block position, because scores are not comparable across two indices.

> **Note:** reader expertise level is supplied by the client, so `expert_only` is a presentation
> control rather than a security boundary. See [Operational Caveats](#operational-caveats).

---

## Background Workers

Four workers run as daemon threads inside the API process, started and stopped by the FastAPI
lifespan. Each is independently switchable, because operational needs differ: the console must be
able to enrich one article while the corpus-wide sweeper is stopped.

| Worker | Flag | Default | Purpose |
|---|---|---|---|
| `enrichment_worker` | `ENABLE_BACKGROUND_WORKER` | `false` | Sweeps the whole catalog in cursor order. Pausable at runtime via Redis, honoured by every replica. |
| `enrichment_job_worker` | `ENABLE_ENRICHMENT_JOB_WORKER` | `true` | Drains per-article enrichment jobs queued from the console. |
| `guideline_extraction_worker` | `ENABLE_GUIDELINE_EXTRACTION_WORKER` | `true` | Runs PDF extraction jobs, with bounded retries for transient failures. |
| `guideline_enrichment_worker` | `ENABLE_GUIDELINE_ENRICHMENT_WORKER` | `true` | Adds retrieval facets to extracted rules. Version-gated and resumable. |

Worker state is inspectable at `GET /health`, `GET /api/v1/enrich/worker` and
`GET /api/v1/guidelines/worker/status`. The sweeper can be paused with
`POST /api/v1/enrich/worker/pause`.

**[Worker design and operations →](docs/BACKGROUND_WORKER.md)**

---

## Prompt Registry & Observability

### Prompt registry

All 17 production prompts live in a central registry (`src/backend/prompts.py`). Each has a Langfuse
name and an in-code fallback:

- **Langfuse is the source of truth.** A prompt edited in the Langfuse UI wins at runtime.
- **Fallbacks are a resilience net.** They are used when Langfuse is unreachable or disabled, and as
  a one-time seed for prompts that do not yet exist.
- **Startup sync is idempotent and never overwrites.** `sync_prompts()` runs in a daemon thread on
  every boot and only creates what is missing, so a deployment can never clobber a deliberate UI
  edit. Startup never blocks on or fails because of Langfuse.

> **Consequence worth knowing:** editing only the in-code fallback has no effect once the prompt
> exists in Langfuse. Mirror the change in the UI, or the managed version keeps winning.

Seed a fresh environment explicitly:

```bash
PYTHONPATH=src python scripts/seed_langfuse_prompts.py
```

### Observability

Tracing activates only when **both** `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set;
otherwise every hook is a no-op. Traces carry run names and tags per pipeline stage
(`enrichment-annotation`, `enrichment-keywords`, …), and buffered traces are flushed on shutdown.

**[Integration guide →](docs/langfuse-integration-guide.md)**

---

## API Reference

Base path `/api/v1`. Interactive OpenAPI documentation is served at **`/docs`** on any running
instance, with the raw spec at `/openapi.json`.

### Question answering — `/qa`

| Method | Path | Description |
|---|---|---|
| `POST` | `/qa/ask` | Answer a question with optional RAG, expertise level, language and retriever choice |
| `POST` | `/qa/memory` | Apply or decline a memory nudge raised by a previous turn |
| `POST` | `/qa/feedback` | Submit feedback on an answer (supports A/B dual-answer) |
| `GET` | `/qa/models` | Available Groq models and the default |
| `GET` | `/qa/questions` | Starter nutrition questions for a language |
| `GET` | `/qa/tips` | Tips of the day, grounded in guidelines or articles |
| `DELETE` | `/qa/cache/clear` | Invalidate cached answers by pattern |

### Search summarisation — `/search`

| Method | Path | Description |
|---|---|---|
| `POST` | `/search/summarize` | Synthesise pre-fetched search results into a cited summary |
| `DELETE` | `/search/cache/clear` | Invalidate cached summaries by pattern |
| `GET` | `/search/health` | Summariser health and cache configuration |

### Chat sessions — `/sessions`

| Method | Path | Description |
|---|---|---|
| `POST` | `/sessions/start` | Create a session |
| `POST` | `/sessions/chat` | Send a turn and receive a structured reply |
| `GET` | `/sessions/users/{user_id}` | List a user's sessions |
| `GET` | `/sessions/{session_id}/context` | Read accumulated context |
| `GET` | `/sessions/{session_id}/history` | Read message history |
| `DELETE` | `/sessions/{session_id}/history` | Clear history, keep the session |
| `DELETE` | `/sessions/{session_id}` | Delete the session |

### Article enrichment — `/enrich`

| Method | Path | Description |
|---|---|---|
| `POST` | `/enrich/article` | Stateless enrichment of an inline article payload (no catalog write) |
| `POST` | `/enrich/articles` | Queue enrichment for several catalog articles |
| `POST` | `/enrich/articles/{urn}` | Queue enrichment for one article (`force` re-enriches) |
| `GET` | `/enrich/jobs` | Batch job status for many URNs (one call per console page) |
| `GET` | `/enrich/articles/{urn}` | Job status for one article |
| `DELETE` | `/enrich/articles/{urn}` | Clear sweeper bookkeeping so the article is eligible again |
| `GET` | `/enrich/worker` | Status of both enrichment workers |
| `POST` | `/enrich/worker/pause` | Pause or resume the sweeper across all replicas |
| `POST` | `/enrich/worker/restart` | Restart the enrichment workers |

### Guidelines — `/guidelines`

| Method | Path | Description |
|---|---|---|
| `GET` | `/guidelines/storage/{artifact_uuid}` | Local workspace state for an artifact PDF |
| `POST` | `/guidelines/extract/{artifact_uuid}` | Queue extraction from an artifact PDF |
| `GET` | `/guidelines/extract/{artifact_uuid}` | Extraction job status and results |
| `POST` | `/guidelines/import/{artifact_uuid}` | Import extracted rules onto the parent guide |
| `GET` | `/guidelines/worker/status` | Extraction worker status |
| `POST` | `/guidelines/enrichment/preview` | Preview a facet-enrichment run |
| `POST` | `/guidelines/enrichment/enqueue` | Queue facet enrichment |
| `GET` | `/guidelines/enrichment/status` | Enrichment progress |
| `GET` | `/guidelines/enrichment/worker/status` | Enrichment worker status |
| `GET` | `/guidelines/corpus/audit` | Corpus-level audit of stored guidelines |
| `GET` | `/guidelines/corpus/activation-plan` | What activation would change |
| `POST` | `/guidelines/corpus/activate/{guide_urn}` | Activate a guide's rules for retrieval |

### Service endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service metadata and endpoint index |
| `GET` | `/health` | Health, cache config and per-worker statistics |
| `GET` | `/api/v1/worker/status` | Sweeper status and statistics |
| `GET` | `/docs` | Swagger UI |

### Worked example

```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{
        "question": "Is olive oil better than butter for heart health?",
        "expertise_level": "beginner",
        "language": "en",
        "retriever": "rag",
        "top_k": 5
      }'
```

---

## Configuration

All configuration is environment-driven and read once in `src/config.py` into `config.settings`.

> There is no `.env.example` in this repository yet. The tables below are the authoritative list.

### Core

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `DEBUG` | `true` | FastAPI debug mode and log verbosity |
| `CACHE_ENABLED` | `false` | Enable Redis-backed response caching |

### Data stores

| Variable | Default | Description |
|---|---|---|
| `ELASTIC_HOST` | `http://elasticsearch:9200` | Elasticsearch endpoint |
| `ES_DIM` | `384` | Embedding dimension — must match the embedding model |
| `REDIS_HOST` | `redis` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRES_DB` | `wisefood` | Database name |
| `POSTGRES_POOL_SIZE` | `10` | Connection pool size |
| `POSTGRES_MAX_OVERFLOW` | `20` | Pool overflow |
| `SESSION_TTL_SECONDS` | `604800` (7 days) | Chat session inactivity expiry |

### Platform integration

| Variable | Default | Description |
|---|---|---|
| `DATA_API_URL` | `http://data-catalog:8000` | WiseFood Data API base URL |
| `WISEFOOD_API_URL` | falls back to `DATA_API_URL` | WiseFood API base URL |
| `WISEFOOD_PLATFORM_API_URL` | falls back to `WISEFOOD_API_URL` | Platform API for members/households |
| `KEYCLOAK_CLIENT_ID` | `foodscholar` | Service account client id |
| `KEYCLOAK_CLIENT_SECRET` | *(unset)* | Service account secret |

### Model providers

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for chat and annotation inference |
| `OPENAI_API_KEY` | *(required for extraction)* | OpenAI key for vision-based PDF extraction |

### Models

Every model the application talks to is named by one of these variables and nowhere else, so a
provider retiring a model id is a config change and a restart. The roles are deliberately separate:
they are *not* interchangeable. The utility and enrichment roles run high-volume, low-stakes calls
where a small model is the right answer; the QA roles are user-facing; extraction is a vision call
through a different provider entirely.

| Variable | Default | Role |
|---|---|---|
| `QA_DEFAULT_MODEL` | `openai/gpt-oss-120b` | Answer model for simple mode and unspecified advanced requests |
| `QA_AVAILABLE_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b` | Comma-separated. Advertised by `GET /qa/models` and enforced on advanced-mode requests |
| `QA_FAST_MODEL` | `openai/gpt-oss-20b` | Cheap leg for A/B comparison strategies |
| `QA_UTILITY_MODEL` | `openai/gpt-oss-20b` | Starter questions, tips, conversation summaries |
| `SESSION_TITLE_MODEL` | `openai/gpt-oss-20b` | Chat session titles |
| `SESSION_CHAT_MODEL` | `openai/gpt-oss-120b` | Structured chat responses |
| `SYNTHESIS_MODEL` | `openai/gpt-oss-120b` | Search-result synthesis |
| `MEMORY_EXTRACTOR_MODEL` | `openai/gpt-oss-20b` | Consented-memory suggestion extraction |
| `ENRICHMENT_KEYWORD_MODEL` | `openai/gpt-oss-20b` | Article keyword extraction |
| `ENRICHMENT_ANNOTATION_MODEL` | `openai/gpt-oss-20b` | Article annotation |
| `GUIDELINE_ENRICHMENT_MODEL` | `openai/gpt-oss-20b` | Guideline facet enrichment |
| `GUIDELINE_EXTRACTION_MODEL` | `gpt-5.4` | Vision model for PDF page extraction (OpenAI, not Groq) |

`QA_DEFAULT_MODEL` must appear in `QA_AVAILABLE_MODELS` and the list must be non-empty — otherwise
`Config._validate_models` raises at startup rather than letting every advanced-mode request 400.

**Groq retirements (2026-08-16).** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were shut
down; announced 2026-06-17, effective 2026-08-16. Neither appears in any default here. The
replacements Groq named are `openai/gpt-oss-20b` and `openai/gpt-oss-120b` / `qwen/qwen3.6-27b`
respectively. Two consequences worth knowing:

- **There is no non-reasoning model left in the deployment.** Every Groq-backed role now runs on a
  reasoning family, so the roles that used a small chat model for a trivial job (session titles,
  classification, the A/B comparison leg) depend on the reasoning handling in
  `backend/model_profiles.py` rather than on the model being simple. Cost per call for those roles is
  higher than it was, bounded by `reasoning_effort=low` and the `max_tokens` floor.
- **`qwen/qwen3.6-27b` is in the picker deliberately** — one vendor's deprecation notice should not
  leave the service with no answer model, so a second family stays configured and reachable.

Retired ids are listed in `RETIRED` in `backend/model_profiles.py`, which warns at client
construction with the shutdown date and the replacement. A retired id still matches its family row,
so without that list nothing would flag it — the call would just fail at the provider.

Cluster deployment supplies these as a normal env map; the defaults above are what the application
runs with when it does not.

### Model portability

Swapping a model must not require an agent-by-agent audit, so the family-specific behaviour lives in
two places instead of at each call site:

- **`backend/model_profiles.py`** — a capability table keyed by model-id substring, applied by the
  Groq pool to every client it builds. It injects `reasoning_format=hidden`, a `reasoning_effort` and
  a `max_tokens` floor for reasoning families, raises a caller budget that is too low for one, and
  **drops parameters a family would reject** (`reasoning_*` on Llama, `temperature` on the OpenAI
  reasoning models). An unregistered id is used as-is and logs a warning.
- **`backend/model_output.py`** — `normalize_model_text` runs over every response before anything
  reads it: it flattens content blocks, strips `<think>`/`<reasoning>` blocks and OpenAI harmony
  channel residue (`analysis … assistantfinal`), and discards a fragment left by a completion cut off
  mid-reasoning. This is the second line of defence for families the profile table does not know.

All JSON recovery goes through `agents/json_output.py` (`parse_json_object` / `parse_json_array`).
When each call site had its own parser the weakest one decided which models the app could run, so
adding a new parser is a regression, not a local choice.

**Introducing a new model:** add a row to `_FAMILIES` in `backend/model_profiles.py` if it is a new
family, then point a role at it. An id from an unregistered family still runs, with caller defaults
only and a warning naming itself.

**Retiring one:** add it to `RETIRED` with the shutdown date and replacement, then move every role
off it. `tests/test_model_config.py` asserts no configured default references a retired id.

### Workers

| Variable | Default | Description |
|---|---|---|
| `ENABLE_BACKGROUND_WORKER` | `false` | Start the catalog sweeper |
| `WORKER_BATCH_SIZE` | `50` | Articles per sweep batch |
| `WORKER_POLL_INTERVAL` | `10` | Sweeper poll interval (s) |
| `ENABLE_ENRICHMENT_JOB_WORKER` | `true` | Start the on-demand enrichment worker |
| `ENRICHMENT_JOB_POLL_INTERVAL` | `5` | Job queue poll interval (s) |
| `ENABLE_GUIDELINE_EXTRACTION_WORKER` | `true` | Start the extraction worker |
| `GUIDELINE_WORKER_POLL_INTERVAL` | `5` | Extraction poll interval (s) |
| `ENABLE_GUIDELINE_ENRICHMENT_WORKER` | `true` | Start the facet-enrichment worker |
| `GUIDELINE_ENRICHMENT_WORKER_POLL_INTERVAL` | `5` | Enrichment poll interval (s) |

### Guideline extraction & enrichment

| Variable | Default | Description |
|---|---|---|
| `GUIDELINE_EXTRACTION_MODEL` | `gpt-5.4` | Vision model for page extraction |
| `GUIDELINE_RENDER_DPI` | `144` | PDF page render resolution |
| `GUIDELINE_PDF_WORKSPACE` | `/tmp/foodscholar/guideline_artifacts` | Local artifact workspace |
| `GUIDELINE_ARTIFACT_FILENAME` | `source.pdf` | Downloaded artifact filename |
| `GUIDELINE_EXTRACTION_MAX_ATTEMPTS` | `3` | Bounded retries for a failed job |
| `GUIDELINE_ENRICHMENT_VERSION` | `1` | Bump to re-enrich the corpus; lower versions are skipped |
| `GUIDELINE_ENRICHMENT_CONCURRENCY` | `8` | Rules enriched concurrently per guide |
| `GUIDELINE_JOB_QUEUE_KEY` | `guidelines:queue` | Redis queue key |
| `GUIDELINE_JOB_STATUS_PREFIX` | `guidelines:job` | Redis status key prefix |
| `GUIDELINE_JOB_LOCK_PREFIX` | `guidelines:lock` | Redis lock prefix |
| `GUIDELINE_JOB_LOCK_TIMEOUT` | `7200` | Lock TTL (s) |
| `GUIDELINE_ENRICHMENT_QUEUE_KEY` | `guideline_enrichment:queue` | Redis queue key |
| `GUIDELINE_ENRICHMENT_LOCK_PREFIX` | `guideline_enrichment:lock` | Redis lock prefix |
| `GUIDELINE_ENRICHMENT_LOCK_TIMEOUT` | `7200` | Lock TTL (s) |

### Retrieval tuning

| Variable | Default | Description |
|---|---|---|
| `QA_GUIDELINE_RETRIEVAL_MODE` | `bm25` | `bm25` or `hybrid` (BM25 + kNN). Hybrid only helps after the guideline embedding backfill has run |
| `QA_GUIDELINE_KNN_BOOST` | `1.0` | Weight of the vector leg in hybrid mode |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | *(unset)* | Enables tracing when set together with the secret key |
| `LANGFUSE_SECRET_KEY` | *(unset)* | Enables tracing when set together with the public key |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse endpoint |

---

## Running the Service

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Biomedical NER model used by LinearRAG
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

export GROQ_API_KEY=...            # required
export OPENAI_API_KEY=...          # required for guideline extraction
export ELASTIC_HOST=http://localhost:9200
export REDIS_HOST=localhost
export POSTGRES_HOST=localhost
export DATA_API_URL=http://localhost:8000

python src/app.py
```

The service listens on `PORT` (default `8000`). Reachable Elasticsearch, Redis, PostgreSQL and
WiseFood Data API endpoints are required — startup verifies the database connection and warms the
LinearRAG retriever before serving.

For autoreload during development:

```bash
cd src && uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker compose up --build
```

> **Read this before relying on Compose.** The bundled `docker-compose.yaml` is a thin dev
> convenience, not a full stack: it starts **only** the FoodScholar container and still expects
> external Elasticsearch, Redis, PostgreSQL and WiseFood API endpoints. It also carries three known
> rough edges — a stale `./app.py:/app/app.py` volume mount (the entrypoint is `src/app.py`), a
> healthcheck probing port `8000` while the service is configured on `8080`, and placeholder API
> keys. Fix or override these for your environment. See
> [Operational Caveats](#operational-caveats).

---

## Building the Image

```bash
make build          # docker build . -t wisefood/foodscholar:latest
make push           # docker push wisefood/foodscholar:latest
make all            # build + push
```

Override the tag as needed:

```bash
make build IMGTAG=ghcr.io/wisefood/foodscholar:1.0.0
```

### What the build does

The `Dockerfile` starts from `python:3.12.8-slim` and does all installation **in a single layer**,
deliberately:

1. installs `build-essential`, needed only to compile wheels;
2. installs the **CPU-only** PyTorch wheel (`torch==2.6.0`) from the PyTorch index, avoiding the
   multi-gigabyte CUDA build;
3. installs `requirements.txt`;
4. installs the `en_core_sci_sm` scispaCy model;
5. **pre-downloads** `sentence-transformers/all-MiniLM-L6-v2`, so the container never fetches the
   embedding model at runtime;
6. purges `build-essential` and clears apt/pip caches **in the same layer**.

Combining these matters: purging in a later layer would leave the toolchain in the image history and
the result would be too large to build or unpack on a constrained disk.

> **Port note:** the `EXPOSE 8001` line is documentation only. The actual listen port comes from
> `PORT` (default `8000` in `src/config.py`), so keep `EXPOSE`, `PORT` and your deployment's service
> port aligned.

The prebuilt LinearRAG artifacts in `src/data/linearrag/` (graph plus parquet embeddings) are copied
into the image and loaded at startup.

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

`PYTHONPATH=src` is **required** — the suite imports modules as `services.…`, `agents.…`, and a bare
`pytest tests/` fails collection with `ModuleNotFoundError`.

The suite is **179 tests** and runs in seconds, with no live Elasticsearch, Redis, PostgreSQL or LLM
provider required — infrastructure is faked at the seams. Coverage focuses on the logic where
mistakes are expensive and hard to spot in review:

| Area | File |
|---|---|
| Editorial policy enforcement and legacy-corpus defaults | `tests/test_article_policy.py` |
| Enrichment persistence, job orchestration, field extraction | `tests/test_enrichment_jobs.py` |
| Worker restart behaviour | `tests/test_enrichment_worker_restart.py` |
| Guideline extraction and job lifecycle | `tests/test_guideline_extractor.py`, `tests/test_guideline_jobs.py` |
| Guideline retrieval gate | `tests/test_guideline_retrieval_gate.py` |
| Consented memory | `tests/test_memory_service.py` |
| Model retry/backoff | `tests/test_model_backoff.py` |
| Model config, capability profiles, output normalization | `tests/test_model_config.py` |
| Prompt registry and Langfuse fallbacks | `tests/test_prompts_registry.py` |
| QA clarification flow and guideline RAG | `tests/test_qa_clarification.py`, `tests/test_qa_guideline_rag.py` |
| Tips generation and LLM token budgets | `tests/test_tips_generation.py` |

---

## Development Notes

### Reasoning models need an explicit token budget

`openai/gpt-oss-*` models are reasoning models. Called without `max_tokens` and a `reasoning_effort`
cap they can spend the entire completion on hidden reasoning and return **empty content** or a
payload truncated mid-object, which surfaces as an opaque JSON parse failure; left at the provider
default they can also return that reasoning *inside* `content`, where it is rendered as the answer.

This is no longer each call site's problem: `backend/model_profiles.py` applies the budget and
`reasoning_format=hidden` in the Groq pool, and `backend/model_output.py` strips leaked reasoning
before parsing. See [Model portability](#model-portability). A call site may still pass an explicit
`max_tokens` or `reasoning_effort` — an explicit value wins, except a budget below the family floor,
which is raised.

### Connection pooling

Groq clients (`backend/groq.py`) and WiseFood clients (`backend/platform.py`) are pooled and reused;
do not instantiate them per request. The Langfuse callback handler is attached automatically and
deliberately excluded from the pool key.

### Caching and idempotency

Redis caches are keyed on the full request shape, including retriever, model and user context, so
answers never leak across audiences. Job orchestration uses Redis locks with TTLs, and workers
check whether a newer request has superseded theirs before running.

### Database bootstrap

Tables are created idempotently at startup by `backend/db_init.py` from the SQLAlchemy models in
`models/db.py`. There is no migration tool wired up; additive schema changes are safe, destructive
ones need a manual plan.

### Adding an endpoint

1. Define request/response models in `src/models/`.
2. Add business logic to a service in `src/services/` (never in the router).
3. If it calls an LLM, put the prompt in the registry and the reasoning in an agent.
4. Register the route in `src/api/v1/` and include the router in `src/app.py`.
5. Add tests that fake the infrastructure seams.

---

## Operational Caveats

Things a new operator or reviewer should know before deploying this service.

- **No authentication.** FoodScholar exposes no auth of its own; it is an internal service designed
  to sit behind the `wisefood-api` gateway, which enforces Keycloak roles on the routes it proxies.
  **Do not expose this service directly to the internet.**
- **CORS is fully open.** `allow_origins=["*"]` in `src/app.py` is a development default and should
  be narrowed for production.
- **Reader expertise level is client-supplied.** `expert_only` visibility is therefore a
  presentation control, not a security boundary — anyone can declare themselves an expert. Material
  that genuinely must not be disclosed needs a role-backed, server-enforced gate.
- **`docker-compose.yaml` is not a full stack.** It starts only the API container and carries a
  stale volume mount, a mismatched healthcheck port and placeholder keys.
- **No `.env.example` yet.** Use the [Configuration](#configuration) tables as the reference.
- **Elasticsearch indices are not owned here.** Index creation and mappings live in
  wisefood-data-api. FoodScholar assumes `articles` and `guidelines` already exist.
- **Guideline extraction has a real cost.** It renders and sends PDF pages to a vision model.
  Triage exists to keep that bounded, but a large corpus run should be planned, not triggered
  casually.
- **`src/backend/cassandra.py` is currently unused** by the active runtime.

---

## Related Repositories

| Repository | Role |
|---|---|
| [wisefood-data-api](https://github.com/wisefood/wisefood-data-api) | Catalog and metadata service; owns articles, guides, guidelines and the Elasticsearch indices |
| [wisefood-client](https://github.com/wisefood/wisefood-client) | Official Python client for the catalog, used here for enrichment writes |

---

## License

This repository is distributed under the terms of the included
[Apache License 2.0](LICENSE).
