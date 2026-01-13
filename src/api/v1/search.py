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
    cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true"
)


@router.post("/summarize", response_model=SearchSummaryResponse)
async def summarize_search(request: SearchSummaryRequest):
    """
    Synthesize pre-fetched search results into a comprehensive summary.

    This endpoint accepts article search results and:
    1. Analyzes and synthesizes findings from multiple articles
    2. Generates citations linking claims to specific articles
    3. Returns a comprehensive summary with evidence
    4. Caches results for faster subsequent requests

    **Example Request:**
    ```json
    {
        "query": "effects of omega-3 fatty acids on cardiovascular health",
        "results": [
            {
                "urn": "urn:article:12345",
                "title": "Omega-3 and Heart Health",
                "abstract": "Study findings...",
                "authors": ["Smith J", "Doe A"],
                "venue": "Journal of Nutrition",
                "publication_year": "2020-01-01",
                "category": "meta-analysis",
                "_score": 0.95
            }
        ],
        "expertise_level": "intermediate",
        "language": "en"
    }
    ```

    **Note:** The `_score` field can be passed as either `_score` or `score` - both are accepted.

    **Returns:**
    - Synthesized summary in markdown format
    - Key findings with citations to specific articles and sections
    - Follow-up question suggestions
    - Metadata about the synthesis
    """
    try:
        logger.info(f"Search summary request: {request.query} ({len(request.results)} articles)")
        result = await search_summarizer.summarize_search(request)
        return result

    except Exception as e:
        logger.error(f"Error in summarize_search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating search summary: {str(e)}",
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
    - Synthesis algorithm changes
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
    Health check for search summarization service.

    Returns service status and configuration.
    """
    return {
        "status": "healthy",
        "service": "search_summarizer",
        "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
    }
