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


# Configure application settings
config = Config()
config.setup()
