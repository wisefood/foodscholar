"""FoodScholar API - Main Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import logsys
import uvicorn
from contextlib import asynccontextmanager

from routers.generic import install_error_handler
from api.v1 import search, sessions, enrich, qa
from workers.enrichment_worker import (
    start_background_worker,
    stop_background_worker,
    get_worker,
)
from config import config

# Initialize logger
logger = logging.getLogger(__name__)
logsys.configure()


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

### Question Answering
- Non-contextual Q&A with semantic article retrieval
- Simple mode (automatic RAG) and advanced mode (custom model/RAG selection)
- A/B testing with dual-answer feedback

### Chat Sessions
- Conversational interface for food and nutrition questions
- Context-aware responses with memory
- Structured fact extraction with references

## API Organization

- `/api/v1/search` - Search summarization endpoints
- `/api/v1/qa` - Question answering endpoints
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
app.include_router(qa.router, prefix="/api/v1")


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
            "qa_ask": "/api/v1/qa/ask",
            "qa_feedback": "/api/v1/qa/feedback",
            "qa_models": "/api/v1/qa/models",
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
