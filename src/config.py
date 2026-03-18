"""Application configuration."""
import os


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
        self.settings["CACHE_ENABLED"] = (
            os.getenv("CACHE_ENABLED", "false").lower() == "true"
        )
        
        self.settings["REDIS_HOST"] = os.getenv("REDIS_HOST", "redis")
        self.settings["REDIS_PORT"] = int(os.getenv("REDIS_PORT", "6379"))

        # Background worker configuration
        self.settings["ENABLE_BACKGROUND_WORKER"] = (
            os.getenv("ENABLE_BACKGROUND_WORKER", "false").lower() == "true"
        )
        self.settings["WORKER_BATCH_SIZE"] = int(os.getenv("WORKER_BATCH_SIZE", "50"))
        self.settings["WORKER_POLL_INTERVAL"] = int(
            os.getenv("WORKER_POLL_INTERVAL", "10")
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


# Configure application settings
config = Config()
config.setup()
