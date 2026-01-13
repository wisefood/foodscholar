"""Search summarization service."""
import logging
from typing import List, Dict, Any
from models.search import SearchSummaryRequest, SearchSummaryResponse, ArticleResult
from agents.synthesis_agent import SynthesisAgent
from utilities.cache import CacheManager, TTL_SEARCH_SUMMARY
from utilities.citation_validator import CitationValidator

logger = logging.getLogger(__name__)


class SearchSummarizer:
    """Service for summarizing pre-fetched search results."""

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize search summarizer.

        Args:
            cache_enabled: Whether to enable caching
        """
        self.synthesis_agent = SynthesisAgent()
        self.cache_manager = CacheManager(enabled=cache_enabled)
        self.citation_validator = CitationValidator()

    async def summarize_search(
        self, request: SearchSummaryRequest
    ) -> SearchSummaryResponse:
        """
        Summarize pre-fetched search results.

        Args:
            request: SearchSummaryRequest with query and article results

        Returns:
            SearchSummaryResponse with synthesized findings
        """
        logger.info(
            f"Processing search summary request: {request.query} ({len(request.results)} articles)"
        )

        # Generate cache key (exclude user_id from key)
        cache_key = self.cache_manager.generate_cache_key(
            prefix="search_summary",
            data=request.model_dump(),
            exclude_keys=["user_id"],
        )

        # Check cache
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.info("Returning cached search summary")
            cached_result["cache_hit"] = True
            return SearchSummaryResponse(**cached_result)

        # Check if we have articles
        if not request.results:
            logger.warning(f"No articles provided for query: {request.query}")
            return self._create_empty_response(request)

        # Convert ArticleResult to dict format expected by synthesis agent
        articles = [self._convert_article_result(result) for result in request.results]

        logger.info(f"Synthesizing {len(articles)} articles")

        # Synthesize results
        summary_response = self.synthesis_agent.synthesize_search_results(
            query=request.query,
            articles=articles,
            expertise_level=request.expertise_level,
            language=request.language,
        )

        # Validate citations
        validation_report = self.citation_validator.validate_citations(
            summary_response.key_findings
        )
        logger.info(f"Citation validation: {validation_report}")

        if not validation_report["is_valid"]:
            logger.warning(
                f"Citation validation issues: {validation_report['findings_without_citations']}"
            )

        # Cache the result
        self.cache_manager.set(
            cache_key, summary_response.model_dump(), ttl=TTL_SEARCH_SUMMARY
        )

        return summary_response

    def _convert_article_result(self, result: ArticleResult) -> Dict[str, Any]:
        """
        Convert ArticleResult to dict format.

        Args:
            result: ArticleResult object

        Returns:
            Dictionary with article data
        """
        # Parse publication year
        year = None
        if result.publication_year:
            try:
                # Handle "YYYY-MM-DD" format
                year = int(result.publication_year.split("-")[0])
            except (ValueError, AttributeError):
                year = None

        return {
            "urn": result.urn,
            "title": result.title,
            "abstract": result.abstract or result.description or "",
            "authors": result.authors or [],
            "year": year,
            "journal": result.venue,
            "tags": result.tags or [],
            "category": result.category,
            "_score": result.score,  # Use the score field (aliased as _score)
        }

    def _create_empty_response(
        self, request: SearchSummaryRequest
    ) -> SearchSummaryResponse:
        """Create empty response when no articles provided."""
        from datetime import datetime

        return SearchSummaryResponse(
            query=request.query,
            summary=f"No articles provided for summarization. Please provide search results to analyze.",
            key_findings=[],
            total_articles_analyzed=0,
            all_citations=[],
            search_metadata={"expertise_level": request.expertise_level},
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            follow_up_suggestions=[],
        )

    def clear_cache(self, pattern: str = "search_summary:*") -> int:
        """
        Clear cached search summaries.

        Args:
            pattern: Cache key pattern to clear

        Returns:
            Number of cache entries cleared
        """
        return self.cache_manager.clear_pattern(pattern)
