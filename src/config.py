"""Application configuration."""
import os
from typing import List


def _csv_list(raw: str) -> List[str]:
    """Parse a comma-separated env value, dropping blanks and duplicates."""
    seen = []
    for item in (raw or "").split(","):
        value = item.strip()
        if value and value not in seen:
            seen.append(value)
    return seen


class Config:
    """Application configuration."""

    def __init__(self):
        self.settings = {}

    def setup(self):
        """Read environment variables and store them in settings."""
        self.settings["HOST"] = os.getenv("HOST", "0.0.0.0")
        self.settings["PORT"] = int(os.getenv("PORT", 8000))
        self.settings["DEBUG"] = os.getenv("DEBUG", "true").lower() == "true"
        self.settings["ELASTIC_HOST"] = os.getenv(
            "ELASTIC_HOST", "http://elasticsearch:9200"
        )
        self.settings["ES_DIM"] = int(os.getenv("ES_DIM", 384))
        self.settings["KEYCLOAK_CLIENT_ID"] = os.getenv(
            "KEYCLOAK_CLIENT_ID", "foodscholar"
        )
        self.settings["KEYCLOAK_CLIENT_SECRET"] = os.getenv(
            "KEYCLOAK_CLIENT_SECRET", "***NOTSET***"
        )
        self.settings["DATA_API_URL"] = os.getenv(
            "DATA_API_URL", "http://data-catalog:8000"
        )
        self.settings["WISEFOOD_API_URL"] = os.getenv(
            "WISEFOOD_API_URL", self.settings["DATA_API_URL"]
        )
        self.settings["WISEFOOD_PLATFORM_API_URL"] = os.getenv(
            "WISEFOOD_PLATFORM_API_URL", self.settings["WISEFOOD_API_URL"]
        )
        self.settings["CACHE_ENABLED"] = (
            os.getenv("CACHE_ENABLED", "false").lower() == "true"
        )
        
        self.settings["REDIS_HOST"] = os.getenv("REDIS_HOST", "redis")
        self.settings["REDIS_PORT"] = int(os.getenv("REDIS_PORT", "6379"))

        # Chat sessions are kept in Redis and expire after this many seconds
        # of inactivity (default 7 days). Ephemeral/guest users rely on this
        # to have their conversational state reaped automatically.
        self.settings["SESSION_TTL_SECONDS"] = int(
            os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600))
        )

        # Background worker configuration
        self.settings["ENABLE_BACKGROUND_WORKER"] = (
            os.getenv("ENABLE_BACKGROUND_WORKER", "false").lower() == "true"
        )
        self.settings["WORKER_BATCH_SIZE"] = int(os.getenv("WORKER_BATCH_SIZE", "50"))
        self.settings["WORKER_POLL_INTERVAL"] = int(
            os.getenv("WORKER_POLL_INTERVAL", "10")
        )

        # Selective (on-demand) enrichment worker. Deliberately independent of
        # ENABLE_BACKGROUND_WORKER: the console must be able to enrich a single
        # article while the catalog sweeper is stopped or paused.
        self.settings["ENABLE_ENRICHMENT_JOB_WORKER"] = (
            os.getenv("ENABLE_ENRICHMENT_JOB_WORKER", "true").lower() == "true"
        )
        self.settings["ENRICHMENT_JOB_POLL_INTERVAL"] = int(
            os.getenv("ENRICHMENT_JOB_POLL_INTERVAL", "5")
        )
        self.settings["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "localhost")
        self.settings["POSTGRES_PORT"] = int(os.getenv("POSTGRES_PORT", 5432))
        self.settings["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "postgres")
        self.settings["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.settings["POSTGRES_DB"] = os.getenv("POSTGRES_DB", "wisefood")
        self.settings["POSTGRES_POOL_SIZE"] = int(os.getenv("POSTGRES_POOL_SIZE", 10))
        self.settings["POSTGRES_MAX_OVERFLOW"] = int(
            os.getenv("POSTGRES_MAX_OVERFLOW", 20)
        )
        self.settings["GUIDELINE_PDF_WORKSPACE"] = os.getenv(
            "GUIDELINE_PDF_WORKSPACE", "/tmp/foodscholar/guideline_artifacts"
        )
        self.settings["GUIDELINE_ARTIFACT_FILENAME"] = os.getenv(
            "GUIDELINE_ARTIFACT_FILENAME", "source.pdf"
        )
        self.settings["GUIDELINE_EXTRACTION_MODEL"] = os.getenv(
            "GUIDELINE_EXTRACTION_MODEL", "gpt-5.4"
        )
        self.settings["GUIDELINE_RENDER_DPI"] = int(
            os.getenv("GUIDELINE_RENDER_DPI", "144")
        )
        self.settings["ENABLE_GUIDELINE_EXTRACTION_WORKER"] = (
            os.getenv("ENABLE_GUIDELINE_EXTRACTION_WORKER", "true").lower() == "true"
        )
        self.settings["GUIDELINE_WORKER_POLL_INTERVAL"] = int(
            os.getenv("GUIDELINE_WORKER_POLL_INTERVAL", "5")
        )
        self.settings["GUIDELINE_JOB_QUEUE_KEY"] = os.getenv(
            "GUIDELINE_JOB_QUEUE_KEY", "guidelines:queue"
        )
        self.settings["GUIDELINE_JOB_STATUS_PREFIX"] = os.getenv(
            "GUIDELINE_JOB_STATUS_PREFIX", "guidelines:job"
        )
        self.settings["GUIDELINE_JOB_LOCK_PREFIX"] = os.getenv(
            "GUIDELINE_JOB_LOCK_PREFIX", "guidelines:lock"
        )
        self.settings["GUIDELINE_JOB_LOCK_TIMEOUT"] = int(
            os.getenv("GUIDELINE_JOB_LOCK_TIMEOUT", "7200")
        )

        # Guideline facet enrichment (post-extraction). Bumping the version
        # re-enriches the whole corpus on the next run; records at or above the
        # current version are skipped, so runs are resumable and repeatable.
        self.settings["GUIDELINE_ENRICHMENT_VERSION"] = int(
            os.getenv("GUIDELINE_ENRICHMENT_VERSION", "1")
        )
        self.settings["ENABLE_GUIDELINE_ENRICHMENT_WORKER"] = (
            os.getenv("ENABLE_GUIDELINE_ENRICHMENT_WORKER", "true").lower() == "true"
        )
        self.settings["GUIDELINE_ENRICHMENT_WORKER_POLL_INTERVAL"] = int(
            os.getenv("GUIDELINE_ENRICHMENT_WORKER_POLL_INTERVAL", "5")
        )
        self.settings["GUIDELINE_ENRICHMENT_QUEUE_KEY"] = os.getenv(
            "GUIDELINE_ENRICHMENT_QUEUE_KEY", "guideline_enrichment:queue"
        )
        self.settings["GUIDELINE_ENRICHMENT_LOCK_PREFIX"] = os.getenv(
            "GUIDELINE_ENRICHMENT_LOCK_PREFIX", "guideline_enrichment:lock"
        )
        self.settings["GUIDELINE_ENRICHMENT_LOCK_TIMEOUT"] = int(
            os.getenv("GUIDELINE_ENRICHMENT_LOCK_TIMEOUT", "7200")
        )

        # Guideline retrieval mode: "hybrid" (BM25 + kNN, default now that the
        # guideline embedding backfill has run) or "bm25" (keyword only — the
        # fallback for deployments whose guidelines are not yet embedded, where
        # the vector leg would favour the embedded minority).
        self.settings["QA_GUIDELINE_RETRIEVAL_MODE"] = os.getenv(
            "QA_GUIDELINE_RETRIEVAL_MODE", "hybrid"
        )
        self.settings["QA_GUIDELINE_KNN_BOOST"] = float(
            os.getenv("QA_GUIDELINE_KNN_BOOST", "1.0")
        )

        # How many rules one guide enriches at a time. The provider's rate
        # limit is shared with extraction and every other replica, so this is
        # deliberately modest rather than "as many as possible".
        self.settings["GUIDELINE_ENRICHMENT_CONCURRENCY"] = int(
            os.getenv("GUIDELINE_ENRICHMENT_CONCURRENCY", "8")
        )
        # A whole-job extraction failure is usually transient (a rate limit that
        # outlasted the per-call backoff, a flaky download), so it is retried a
        # bounded number of times before being recorded as failed.
        self.settings["GUIDELINE_EXTRACTION_MAX_ATTEMPTS"] = int(
            os.getenv("GUIDELINE_EXTRACTION_MAX_ATTEMPTS", "3")
        )

        # ------------------------------------------------------------------
        # Models
        #
        # Every model the app talks to is named here and nowhere else, so a
        # provider retiring an id (or a deployment wanting a cheaper one) is an
        # env change rather than a code change. The roles are separate on
        # purpose: they are not interchangeable. The utility/enrichment roles
        # run high-volume, low-stakes calls where a small model is the right
        # answer, the QA role is user-facing, and the extraction role needs
        # vision over rendered PDF pages through a different provider entirely.
        #
        # Family-specific quirks (reasoning budgets, which knobs a family
        # rejects) are handled once in backend.model_profiles, so any id from a
        # registered family can be dropped into any Groq-backed role here.
        # ------------------------------------------------------------------

        # User-facing Q&A. QA_AVAILABLE_MODELS is also the API contract: it is
        # what /qa/models advertises and what an advanced-mode request is
        # validated against, so the UI picker follows this list without a
        # redeploy.
        # --- Source Integrator -------------------------------------------
        # The conversation model. Tool-calling is required; Groq's Compound
        # systems cannot be used here because they refuse user-defined tools.
        self.settings["INTEGRATOR_MODEL"] = os.getenv(
            "INTEGRATOR_MODEL", "openai/gpt-oss-120b"
        )
        # The research model, which is the one place a provider executes
        # anything for us: Compound has built-in web search and decides on its
        # own when to use it.
        # Reading rules out of a source. A plain tool-calling model, not a
        # Compound one: Compound's value is its web search, and this reads text
        # it has already been handed.
        self.settings["INTEGRATOR_INFERENCE_MODEL"] = os.getenv(
            "INTEGRATOR_INFERENCE_MODEL", "openai/gpt-oss-120b"
        )
        self.settings["INTEGRATOR_RESEARCH_MODEL"] = os.getenv(
            "INTEGRATOR_RESEARCH_MODEL", "groq/compound"
        )
        # A run that can search the web needs a ceiling, or one question can
        # spend an afternoon and a month's quota.
        self.settings["INTEGRATOR_MAX_STEPS"] = int(
            os.getenv("INTEGRATOR_MAX_STEPS", 40)
        )
        self.settings["INTEGRATOR_MAX_TOKENS"] = int(
            os.getenv("INTEGRATOR_MAX_TOKENS", 400_000)
        )
        # The switch that lets an approved proposal actually reach the catalog.
        # Off by default, and deliberately not defaulted on now that Phase 2
        # works: a deployment should turn writes on when somebody is there to
        # watch the first one, not because it pulled a new image.
        # robots.txt addresses crawlers. The integrator is an expert pasting
        # one URL and waiting for an answer about that one document, under a
        # per-user rate limit — so it is off unless a deployment asks. It has
        # no bearing on the destination guard, which refuses a private
        # address whatever this says.
        self.settings["INTEGRATOR_RESPECT_ROBOTS"] = (
            os.getenv("INTEGRATOR_RESPECT_ROBOTS", "false").lower() == "true"
        )
        self.settings["INTEGRATOR_WRITES_ENABLED"] = (
            os.getenv("INTEGRATOR_WRITES_ENABLED", "false").lower() == "true"
        )
        # How an integration run waits on its extraction. The timeout is not a
        # kill — the job carries on in its own worker — it is how long this run
        # watches before handing the wait back to a person.
        # Flood control. The routes are already admin-and-expert only, so this
        # is not about strangers — it is about a client in a loop, or one
        # person's credentials being used to spend the platform's Groq budget.
        # Counted from the database rather than a cache, so an outage cannot
        # quietly turn the limit off.
        self.settings["INTEGRATOR_MAX_TURNS_PER_HOUR"] = int(
            os.getenv("INTEGRATOR_MAX_TURNS_PER_HOUR", 60)
        )
        # Concurrent integration runs. Each one holds a thread for as long as
        # its extraction takes, so the global figure is a ceiling on threads
        # and not only on spend.
        self.settings["INTEGRATOR_MAX_RUNS_PER_USER"] = int(
            os.getenv("INTEGRATOR_MAX_RUNS_PER_USER", 3)
        )
        self.settings["INTEGRATOR_MAX_RUNS_TOTAL"] = int(
            os.getenv("INTEGRATOR_MAX_RUNS_TOTAL", 10)
        )
        self.settings["INTEGRATOR_POLL_INTERVAL"] = float(
            os.getenv("INTEGRATOR_POLL_INTERVAL", 20)
        )
        self.settings["INTEGRATOR_EXTRACTION_TIMEOUT"] = float(
            os.getenv("INTEGRATOR_EXTRACTION_TIMEOUT", 3600)
        )
        # The ranking rubric's weights, tunable without a deploy. Licence
        # dominates because a source we may only point at is worth less than
        # one we may actually read. Normalised at use, so these are ratios
        # rather than percentages that have to add up.
        for component, default in (
            ("LICENCE", 0.40), ("COVERAGE_GAP", 0.25), ("AUTHORITY", 0.20),
            ("TRACTABILITY", 0.10), ("COMPLETENESS", 0.05),
        ):
            key = f"INTEGRATOR_WEIGHT_{component}"
            self.settings[key] = float(os.getenv(key, default))
        # Unpaywall asks callers to identify themselves. Ours, never a user's.
        self.settings["INTEGRATOR_CONTACT_EMAIL"] = os.getenv(
            "INTEGRATOR_CONTACT_EMAIL", ""
        )
        # Catalog credentials for the integrator's read tools. Unset leaves
        # those tools unavailable and research still working.
        self.settings["WISEFOOD_API_URL"] = os.getenv("WISEFOOD_API_URL", "")
        self.settings["WISEFOOD_CLIENT_ID"] = os.getenv("WISEFOOD_CLIENT_ID", "")
        self.settings["WISEFOOD_CLIENT_SECRET"] = os.getenv("WISEFOOD_CLIENT_SECRET", "")

        self.settings["QA_DEFAULT_MODEL"] = os.getenv(
            "QA_DEFAULT_MODEL", "openai/gpt-oss-120b"
        )
        self.settings["QA_AVAILABLE_MODELS"] = _csv_list(
            os.getenv(
                "QA_AVAILABLE_MODELS",
                "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b",
            )
        )
        # Cheap leg for classification and A/B comparison. Groq retired both
        # Llama ids on 2026-08-16, so there is no non-reasoning option left:
        # this is now the small reasoning model, and every call through it
        # depends on the reasoning handling in backend.model_profiles.
        self.settings["QA_FAST_MODEL"] = os.getenv(
            "QA_FAST_MODEL", "openai/gpt-oss-20b"
        )
        # Starter questions, tips, conversation summaries.
        self.settings["QA_UTILITY_MODEL"] = os.getenv(
            "QA_UTILITY_MODEL", "openai/gpt-oss-20b"
        )
        self.settings["SESSION_TITLE_MODEL"] = os.getenv(
            "SESSION_TITLE_MODEL", "openai/gpt-oss-20b"
        )
        self.settings["SESSION_CHAT_MODEL"] = os.getenv(
            "SESSION_CHAT_MODEL", "openai/gpt-oss-120b"
        )
        self.settings["SYNTHESIS_MODEL"] = os.getenv(
            "SYNTHESIS_MODEL", "openai/gpt-oss-120b"
        )
        self.settings["MEMORY_EXTRACTOR_MODEL"] = os.getenv(
            "MEMORY_EXTRACTOR_MODEL", "openai/gpt-oss-20b"
        )
        self.settings["ENRICHMENT_KEYWORD_MODEL"] = os.getenv(
            "ENRICHMENT_KEYWORD_MODEL", "openai/gpt-oss-20b"
        )
        self.settings["ENRICHMENT_ANNOTATION_MODEL"] = os.getenv(
            "ENRICHMENT_ANNOTATION_MODEL", "openai/gpt-oss-20b"
        )
        self.settings["GUIDELINE_ENRICHMENT_MODEL"] = os.getenv(
            "GUIDELINE_ENRICHMENT_MODEL", "openai/gpt-oss-20b"
        )

        # ------------------------------------------------------------------
        # Agentic QA pipeline (plan → retrieve → rank → evaluate → answer)
        # ------------------------------------------------------------------
        # "agentic" runs the reasoning pipeline; "legacy" is the rollback flag
        # for the pre-pipeline single-pass flow.
        self.settings["QA_PIPELINE_MODE"] = os.getenv("QA_PIPELINE_MODE", "agentic")
        # The planner decomposes the question; the evaluator judges evidence
        # sufficiency. Both are structured JSON calls where the small reasoning
        # model is the right cost point.
        self.settings["QA_PLANNER_MODEL"] = os.getenv(
            "QA_PLANNER_MODEL", self.settings["QA_FAST_MODEL"]
        )
        self.settings["QA_EVALUATOR_MODEL"] = os.getenv(
            "QA_EVALUATOR_MODEL", self.settings["QA_UTILITY_MODEL"]
        )
        self.settings["QA_MAX_SUBQUESTIONS"] = int(
            os.getenv("QA_MAX_SUBQUESTIONS", "3")
        )
        self.settings["QA_MAX_REPAIR_ROUNDS"] = int(
            os.getenv("QA_MAX_REPAIR_ROUNDS", "1")
        )
        # Client-side reciprocal rank fusion of the lexical and vector legs.
        self.settings["QA_RRF_K"] = int(os.getenv("QA_RRF_K", "60"))
        self.settings["QA_RRF_CANDIDATES"] = int(os.getenv("QA_RRF_CANDIDATES", "30"))
        # Ranking adjustment: exponential recency decay with a floor (an old
        # meta-analysis is discounted, not erased) and a log-scaled citation
        # boost that never punishes missing bibliometrics.
        self.settings["QA_RECENCY_HALF_LIFE_YEARS"] = float(
            os.getenv("QA_RECENCY_HALF_LIFE_YEARS", "6.0")
        )
        self.settings["QA_RECENCY_FLOOR"] = float(os.getenv("QA_RECENCY_FLOOR", "0.35"))
        self.settings["QA_INFLUENCE_WEIGHT"] = float(
            os.getenv("QA_INFLUENCE_WEIGHT", "0.3")
        )
        self.settings["QA_INFLUENCE_CITATION_CAP"] = int(
            os.getenv("QA_INFLUENCE_CITATION_CAP", "1000")
        )
        # Earned tier: an UNTIERED article whose citation record clears these
        # thresholds gets prime/core-like standing on its own — almost the
        # whole corpus carries no editorial tier, and field-shaping work should
        # not rank as a nobody while it waits for a curator. Earned boosts sit
        # below the curated ones (1.45 < 1.6, 1.15 < 1.25) and an explicit
        # tier, promotion or demotion, always wins.
        self.settings["QA_EARNED_TIER_ENABLED"] = os.getenv(
            "QA_EARNED_TIER_ENABLED", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.settings["QA_EARNED_PRIME_CITATIONS"] = int(
            os.getenv("QA_EARNED_PRIME_CITATIONS", "500")
        )
        self.settings["QA_EARNED_PRIME_INFLUENTIAL"] = int(
            os.getenv("QA_EARNED_PRIME_INFLUENTIAL", "25")
        )
        self.settings["QA_EARNED_CORE_CITATIONS"] = int(
            os.getenv("QA_EARNED_CORE_CITATIONS", "150")
        )
        self.settings["QA_EARNED_PRIME_BOOST"] = float(
            os.getenv("QA_EARNED_PRIME_BOOST", "1.45")
        )
        self.settings["QA_EARNED_CORE_BOOST"] = float(
            os.getenv("QA_EARNED_CORE_BOOST", "1.15")
        )
        self.settings["QA_MIN_SCORE"] = float(os.getenv("QA_MIN_SCORE", "0.05"))
        self.settings["QA_PER_DOC_CAP"] = int(os.getenv("QA_PER_DOC_CAP", "2"))
        self.settings["QA_STREAM_HEARTBEAT_SECONDS"] = int(
            os.getenv("QA_STREAM_HEARTBEAT_SECONDS", "15")
        )

        # ------------------------------------------------------------------
        # Knowledge graph browsing (foodscholar-lib)
        #
        # The graph itself is built offline: a job runs the library's phases
        # and writes shelves/themes/cards into Neo4j and chunks/cards into
        # Elasticsearch. This service never builds it. It projects that graph
        # into one denormalized browse index (services/kg_projector.py) and
        # serves every browse read from there, so replicas stay stateless and
        # a page of the tree is an inverted-index lookup rather than a graph
        # traversal.
        #
        # Off by default: a deployment that has never run the build should
        # start clean and answer a readable 503 on the browse routes, not
        # fail a request at a time.
        # ------------------------------------------------------------------
        self.settings["KG_ENABLED"] = (
            os.getenv("KG_ENABLED", "false").lower() == "true"
        )

        # Source stores, written by the offline build. KG_ES_URL follows
        # backend.elastic's ELASTIC_HOST rather than the ELASTIC_HOST setting
        # above: the two disagree on the service name, and backend.elastic is
        # the one that actually opens connections.
        self.settings["KG_ES_URL"] = os.getenv(
            "KG_ES_URL", os.getenv("ELASTIC_HOST", "http://elastic:9200")
        )
        self.settings["KG_CHUNK_INDEX"] = os.getenv(
            "KG_CHUNK_INDEX", "foodscholar_chunks"
        )
        self.settings["KG_CARD_INDEX"] = os.getenv(
            "KG_CARD_INDEX", "foodscholar_cards"
        )
        self.settings["KG_ES_API_KEY"] = os.getenv("KG_ES_API_KEY", "")
        self.settings["KG_ES_USERNAME"] = os.getenv("KG_ES_USERNAME", "")
        self.settings["KG_ES_PASSWORD"] = os.getenv("KG_ES_PASSWORD", "")
        self.settings["KG_NEO4J_URL"] = os.getenv(
            "KG_NEO4J_URL", "bolt://neo4j:7687"
        )
        # Credentials, from either shape.
        #
        # This platform already holds one Neo4j credential, as the secret the
        # database itself is started with: NEO4J_AUTH, a single "user/password"
        # string, which is the format Neo4j's own image defines. Accepting it
        # here means the deployment passes that same secret through rather than
        # storing the password a second time under a different key — and two
        # copies of one credential is a rotation that quietly half-applies.
        #
        # Split on the FIRST separator only: a password may contain "/", a
        # username may not.
        _neo4j_auth = os.getenv("KG_NEO4J_AUTH", "")
        _auth_user, _, _auth_password = _neo4j_auth.partition("/")
        self.settings["KG_NEO4J_USER"] = os.getenv(
            "KG_NEO4J_USER", _auth_user or "neo4j"
        )
        self.settings["KG_NEO4J_PASSWORD"] = os.getenv(
            "KG_NEO4J_PASSWORD", _auth_password
        )

        # The browse index. Reads go through the alias; the projector writes
        # <alias>_<graph_version> and repoints it, so a reader never sees a
        # half-built index and a rollback is one alias move.
        self.settings["KG_BROWSE_ALIAS"] = os.getenv(
            "KG_BROWSE_ALIAS", "foodscholar_browse"
        )
        self.settings["KG_BROWSE_BULK_SIZE"] = int(
            os.getenv("KG_BROWSE_BULK_SIZE", "500")
        )
        # Only one replica may project at a time: the job reads the whole
        # graph and rewrites an index, and two of them racing would burn the
        # cluster for no extra freshness.
        self.settings["KG_REINDEX_LOCK_KEY"] = os.getenv(
            "KG_REINDEX_LOCK_KEY", "kg:reindex:lock"
        )
        self.settings["KG_REINDEX_LOCK_TIMEOUT"] = int(
            os.getenv("KG_REINDEX_LOCK_TIMEOUT", "3600")
        )
        # Node positions are computed once, here, instead of by a force
        # simulation in every visitor's browser. The seed is fixed so the map
        # a user learns is the same map next week.
        self.settings["KG_LAYOUT_SEED"] = int(os.getenv("KG_LAYOUT_SEED", "42"))

        # --- Graph streaming (SSE) ----------------------------------------
        # Nodes per frame. Bigger frames flush less often and reveal the
        # graph in chunks; smaller frames draw more smoothly.
        self.settings["KG_STREAM_BATCH_SIZE"] = int(
            os.getenv("KG_STREAM_BATCH_SIZE", "250")
        )
        # Ceiling per stream, so one unfiltered request on a large graph
        # cannot hold a worker indefinitely. The terminal event reports
        # truncated=true when this bites, rather than lying by omission.
        self.settings["KG_STREAM_MAX_NODES"] = int(
            os.getenv("KG_STREAM_MAX_NODES", "20000")
        )
        # Default level of detail for a stream that asked for no depth.
        self.settings["KG_STREAM_DEFAULT_DEPTH"] = int(
            os.getenv("KG_STREAM_DEFAULT_DEPTH", "2")
        )
        self.settings["KG_STREAM_HEARTBEAT_SECONDS"] = int(
            os.getenv("KG_STREAM_HEARTBEAT_SECONDS", "15")
        )

        # --- Graph retrieval: the `kggen` QA retriever --------------------
        #
        # Extended KG-Gen hybrid scoring, served by the library over the same
        # stores as browsing. It replaced LinearRAG, whose 617MB index had to
        # be mounted beside the service; there is no artifact here, so the
        # only thing to configure is the scoring.
        #
        # The relation index is Layer 0, written by the offline
        # `build_relations()` pass. Without it the triplet and PageRank
        # branches stay silent and `kggen` degrades to plain kNN — which is a
        # usable ranking, so a deployment mid-build answers rather than fails.
        self.settings["KG_RELATION_INDEX"] = os.getenv(
            "KG_RELATION_INDEX", "foodscholar_relations"
        )
        # The query embedder, which MUST be the model the graph's chunks were
        # embedded with. Elasticsearch kNN compares a query vector against the
        # stored ones, so a mismatch is not a quality regression — it is a
        # dimension error (BGE-base is 768, MiniLM is 384) or, between two
        # same-size models, silently meaningless distances.
        #
        # This is deliberately NOT the ES_DIM/MiniLM embedder used for the
        # `rag` retriever's article index. That index and the graph's chunk
        # index are built by different pipelines with different models, and
        # each retriever has to use its own. The default matches the library's
        # `annotate.embedder` default, which is what a stock graph build uses.
        self.settings["KG_EMBED_MODEL"] = os.getenv(
            "KG_EMBED_MODEL", "BAAI/bge-base-en-v1.5"
        )
        # Chunks the text branch pulls for the other two branches to re-rank.
        # A passage outside this pool cannot be retrieved however well it
        # scores on the graph, so this is the recall knob.
        self.settings["KG_RETRIEVAL_CANDIDATE_K"] = int(
            os.getenv("KG_RETRIEVAL_CANDIDATE_K", "100")
        )
        # Branch weights. They must sum to 1.0 — the library refuses the
        # config otherwise, which surfaces here as a 503 on the first graph
        # question rather than a silently skewed ranking.
        self.settings["KG_RETRIEVAL_W_TEXT"] = float(
            os.getenv("KG_RETRIEVAL_W_TEXT", "0.3")
        )
        self.settings["KG_RETRIEVAL_W_TRIPLET"] = float(
            os.getenv("KG_RETRIEVAL_W_TRIPLET", "0.3")
        )
        self.settings["KG_RETRIEVAL_W_PPR"] = float(
            os.getenv("KG_RETRIEVAL_W_PPR", "0.4")
        )
        # Hops out from the query's entities. Each hop is one store call per
        # frontier entity, bounded by KG_RETRIEVAL_MAX_EXPANSION_CALLS.
        self.settings["KG_RETRIEVAL_SUBGRAPH_DEPTH"] = int(
            os.getenv("KG_RETRIEVAL_SUBGRAPH_DEPTH", "1")
        )
        self.settings["KG_RETRIEVAL_MAX_EXPANSION_CALLS"] = int(
            os.getenv("KG_RETRIEVAL_MAX_EXPANSION_CALLS", "50")
        )
        # Triples embedded per query, and the process-local cache that keeps
        # a warm replica from re-encoding the same corpus triples every time.
        self.settings["KG_RETRIEVAL_MAX_RELATIONS"] = int(
            os.getenv("KG_RETRIEVAL_MAX_RELATIONS", "500")
        )
        self.settings["KG_RETRIEVAL_EMBED_CACHE_SIZE"] = int(
            os.getenv("KG_RETRIEVAL_EMBED_CACHE_SIZE", "50000")
        )

        self._validate_models()

        # Langfuse observability (opt-in). Tracing activates only when both
        # the public and secret keys are provided. The Langfuse SDK reads
        # these from the environment directly; they are registered here for
        # centralization and documentation.
        self.settings["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.settings["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.settings["LANGFUSE_BASE_URL"] = os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )


    def _validate_models(self):
        """Fail fast on a model configuration that cannot serve a request.

        A misconfigured model list is not a degradation to absorb: an empty
        picker or a default the validator rejects turns every advanced-mode
        request into a 400 at runtime. Better to refuse to start.
        """
        available = self.settings["QA_AVAILABLE_MODELS"]
        default = self.settings["QA_DEFAULT_MODEL"]

        if not available:
            raise ValueError(
                "QA_AVAILABLE_MODELS is empty; it must list at least one model id"
            )
        if not default:
            raise ValueError("QA_DEFAULT_MODEL must be set to a model id")
        if default not in available:
            raise ValueError(
                f"QA_DEFAULT_MODEL '{default}' is not in QA_AVAILABLE_MODELS "
                f"{available}; a default the request validator rejects would "
                "fail every advanced-mode request"
            )


# Configure application settings
config = Config()
config.setup()
