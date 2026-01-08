"""Search summary API endpoints."""
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from models.search import SearchSummaryRequest, SearchSummaryResponse
from services.search_summarizer import SearchSummarizer
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

# Initialize search summarizer
search_summarizer = SearchSummarizer(
    cache_enabled=os.getenv("CACHE_ENABLED", "false").lower() == "true",
    wisefood_api_key=os.getenv("WISEFOOD_API_KEY"),
)


@router.post("/summarize", response_model=SearchSummaryResponse)
async def summarize_search(request: SearchSummaryRequest):
    """
    Summarize search results across multiple scientific articles.

    This endpoint:
    1. Searches for articles matching the query
    2. Analyzes and synthesizes findings from multiple articles
    3. Returns a comprehensive summary with citations
    4. Caches results for faster subsequent requests

    **Example Request:**
    ```json
    {
        "query": "effects of omega-3 fatty acids on cardiovascular health",
        "filters": {
            "year_from": 2015,
            "categories": ["meta-analysis", "randomized controlled trial"]
        },
        "max_articles": 10,
        "expertise_level": "intermediate"
    }
    ```

    **Returns:**
    - Synthesized summary in markdown format
    - Key findings with citations to specific articles and sections
    - Follow-up question suggestions
    - Metadata about the search
    """
    try:
        logger.info(f"Search summary request: {request.query}")
        result = await search_summarizer.summarize_search(request)
        return result

    except Exception as e:
        logger.error(f"Error in summarize_search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating search summary: {str(e)}",
        )


@router.get("/trending", response_model=list[SearchSummaryResponse])
async def get_trending_summaries(
    limit: int = Query(default=5, ge=1, le=20, description="Number of trending summaries")
):
    """
    Get summaries for trending scientific articles.

    Returns pre-generated or cached summaries for articles that are:
    - Recently published
    - Highly cited
    - Frequently viewed

    **Parameters:**
    - limit: Number of trending summaries to return (1-20)
    """
    try:
        summaries = await search_summarizer.get_trending_summaries(limit=limit)
        return summaries

    except Exception as e:
        logger.error(f"Error getting trending summaries: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving trending summaries: {str(e)}",
        )


@router.delete("/cache/clear")
async def clear_search_cache(
    pattern: Optional[str] = Query(
        default="search_summary:*",
        description="Cache key pattern to clear (e.g., 'search_summary:*')"
    )
):
    """
    Clear cached search summaries.

    **Admin endpoint** - Use to invalidate cache when:
    - Article metadata is updated
    - Search algorithm changes
    - Manual cache refresh is needed

    **Parameters:**
    - pattern: Redis key pattern (default: clears all search summaries)

    **Returns:**
    Number of cache entries cleared
    """
    try:
        cleared = search_summarizer.clear_cache(pattern=pattern)
        return {
            "message": f"Cache cleared successfully",
            "pattern": pattern,
            "entries_cleared": cleared,
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}",
        )


@router.get("/health")
async def search_health_check():
    """
    Health check for search service.

    Returns service status and configuration.
    """
    return {
        "status": "healthy",
        "service": "search_summarizer",
        "cache_enabled": os.getenv("CACHE_ENABLED", "false").lower() == "true",
        "elasticsearch_configured": True,
    }
