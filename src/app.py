"""FoodScholar API - Main Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import logsys
import uvicorn
from contextlib import asynccontextmanager

from routers.generic import install_error_handler
from api.v1 import search, sessions, enrich
from workers.enrichment_worker import (
    start_background_worker,
    stop_background_worker,
    get_worker,
)

# Initialize logger
logger = logging.getLogger(__name__)
logsys.configure()


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
        # Handle REDIS_HOST - parse from URL if needed
        redis_host_env = os.getenv("REDIS_HOST", "redis")
        if "://" in redis_host_env:
            # Extract host from URL like tcp://10.111.3.185:6379
            redis_host_env = redis_host_env.split("://")[1].split(":")[0]
        self.settings["REDIS_HOST"] = redis_host_env

        # Handle REDIS_PORT - parse from URL if needed
        redis_port_env = os.getenv("REDIS_PORT", "6379")
        if "://" in redis_port_env:
            # Extract port from URL like tcp://10.111.3.185:6379
            redis_port_env = redis_port_env.split(":")[-1]
        self.settings["REDIS_PORT"] = int(redis_port_env)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    if config.settings["ENABLE_BACKGROUND_WORKER"]:
        logger.info("Starting background enrichment worker...")
        start_background_worker()

    yield

    # Shutdown
    if config.settings["ENABLE_BACKGROUND_WORKER"]:
        logger.info("Stopping background enrichment worker...")
        stop_background_worker()


# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
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
app.include_router(enrich.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "FoodScholar API",
        "version": "1.0.0",
        "description": "AI-powered scientific literature assistant for food science",
        "endpoints": {
            "search_summaries": "/api/v1/search/summarize",
            "enrich_article": "/api/v1/enrich/article",
            "trending": "/api/v1/search/trending",
            "chat": "/api/v1/sessions/chat",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_data = {
        "status": "healthy",
        "service": "FoodScholar",
        "version": "1.0.0",
        "cache_enabled": config.settings["CACHE_ENABLED"],
        "elasticsearch_host": config.settings["ELASTIC_HOST"],
    }

    # Add worker status if enabled
    if config.settings["ENABLE_BACKGROUND_WORKER"]:
        worker = get_worker()
        health_data["background_worker"] = worker.get_stats()

    return health_data


@app.get("/api/v1/worker/status")
async def worker_status():
    """Get background worker status and statistics."""
    if not config.settings["ENABLE_BACKGROUND_WORKER"]:
        return {"enabled": False, "message": "Background worker is disabled"}

    worker = get_worker()
    return {"enabled": True, **worker.get_stats()}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.settings["HOST"],
        port=config.settings["PORT"],
        log_level="debug" if config.settings["DEBUG"] else "info",
    )
