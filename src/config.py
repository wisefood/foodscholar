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
