"""FoodScholar API - Main Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import logsys
import uvicorn

from routers.generic import install_error_handler
from api.v1 import search, sessions

# Initialize logger
logger = logging.getLogger(__name__)
logsys.configure()


class Config:
    """Application configuration."""

    def __init__(self):
        self.settings = {}

    def setup(self):
        """Read environment variables and store them in settings."""
        self.settings["HOST"] = os.getenv("HOST", "127.0.0.1")
        self.settings["PORT"] = int(os.getenv("PORT", 8000))
        self.settings["DEBUG"] = os.getenv("DEBUG", "true").lower() == "true"
        self.settings["CONTEXT_PATH"] = os.getenv("CONTEXT_PATH", "")
        self.settings["APP_EXT_DOMAIN"] = os.getenv(
            "APP_EXT_DOMAIN", "http://wisefood.gr"
        )
        self.settings["ELASTIC_HOST"] = os.getenv(
            "ELASTIC_HOST", "http://elasticsearch:9200"
        )
        self.settings["ES_DIM"] = int(os.getenv("ES_DIM", 384))
        self.settings["FOODSCHOLAR_URL"] = os.getenv(
            "FOODSCHOLAR_URL", "http://foodscholar:8001"
        )
        self.settings["RECIPEWRANGLER_URL"] = os.getenv(
            "RECIPEWRANGLER_URL", "http://recipewrangler:8001"
        )
        self.settings["FOODCHAT_URL"] = os.getenv(
            "FOODCHAT_URL", "http://foodchat:8001"
        )
        self.settings["MINIO_ENDPOINT"] = os.getenv(
            "MINIO_ENDPOINT", "http://minio:9000"
        )
        self.settings["MINIO_ROOT"] = os.getenv("MINIO_ROOT", "root")
        self.settings["MINIO_ROOT_PASSWORD"] = os.getenv(
            "MINIO_ROOT_PASSWORD", "minioadmin"
        )
        self.settings["MINIO_EXT_URL_CONSOLE"] = os.getenv(
            "MINIO_EXT_URL_CONSOLE", "https://s3.wisefood.gr/console"
        )
        self.settings["MINIO_EXT_URL_API"] = os.getenv(
            "MINIO_EXT_URL_API", "https://s3.wisefood.gr"
        )
        self.settings["MINIO_BUCKET"] = os.getenv("MINIO_BUCKET", "system")
        self.settings["KEYCLOAK_URL"] = os.getenv(
            "KEYCLOAK_URL", "http://keycloak:8080"
        )
        self.settings["KEYCLOAK_EXT_URL"] = os.getenv(
            "KEYCLOAK_EXT_URL", "https://auth.wisefood.gr"
        )
        self.settings["KEYCLOAK_ISSUER_URL"] = os.getenv(
            "KEYCLOAK_ISSUER_URL", "https://auth.wisefood.gr/realms/master"
        )
        self.settings["KEYCLOAK_REALM"] = os.getenv("KEYCLOAK_REALM", "master")
        self.settings["KEYCLOAK_CLIENT_ID"] = os.getenv(
            "KEYCLOAK_CLIENT_ID", "wisefood-api"
        )
        self.settings["KEYCLOAK_CLIENT_SECRET"] = os.getenv(
            "KEYCLOAK_CLIENT_SECRET", "secret"
        )
        self.settings["CACHE_ENABLED"] = (
            os.getenv("CACHE_ENABLED", "false").lower() == "true"
        )
        self.settings["REDIS_HOST"] = os.getenv("REDIS_HOST", "redis")
        self.settings["REDIS_PORT"] = int(os.getenv("REDIS_PORT", 6379))
        self.settings["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "localhost")
        self.settings["POSTGRES_PORT"] = int(os.getenv("POSTGRES_PORT", 5432))
        self.settings["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "postgres")
        self.settings["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.settings["POSTGRES_DB"] = os.getenv("POSTGRES_DB", "wisefood")
        self.settings["POSTGRES_POOL_SIZE"] = int(os.getenv("POSTGRES_POOL_SIZE", 10))
        self.settings["POSTGRES_MAX_OVERFLOW"] = int(
            os.getenv("POSTGRES_MAX_OVERFLOW", 20)
        )
        self.settings["WISEFOOD_API_KEY"] = os.getenv("WISEFOOD_API_KEY", "")


# Configure application settings
config = Config()
config.setup()

# Initialize FastAPI app
app = FastAPI(
    title="FoodScholar API",
    description="""
FoodScholar: AI-powered scientific literature assistant for food science.

## Features

### Search Summaries
- Multi-document synthesis of scientific articles
- Citation-backed findings with confidence levels
- Expertise-level adjusted summaries (beginner/intermediate/expert)
- Intelligent caching for performance

### Chat Sessions
- Conversational interface for food and nutrition questions
- Context-aware responses with memory
- Structured fact extraction with references

## API Organization

- `/api/v1/search` - Search summarization endpoints
- `/api/v1/sessions` - Chat session management
    """,
    version="1.0.0",
    debug=config.settings["DEBUG"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Install error handlers
install_error_handler(app)

# Include routers
app.include_router(search.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "FoodScholar API",
        "version": "1.0.0",
        "description": "AI-powered scientific literature assistant for food science",
        "endpoints": {
            "search_summaries": "/api/v1/search/summarize",
            "trending": "/api/v1/search/trending",
            "chat": "/api/v1/sessions/chat",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FoodScholar",
        "version": "1.0.0",
        "cache_enabled": config.settings["CACHE_ENABLED"],
        "elasticsearch_host": config.settings["ELASTIC_HOST"],
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.settings["HOST"],
        port=config.settings["PORT"],
        log_level="debug" if config.settings["DEBUG"] else "info",
    )
