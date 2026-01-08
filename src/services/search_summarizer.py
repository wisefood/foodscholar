"""Search summarization service."""
import logging
from typing import List, Dict, Any, Optional
from models.search import SearchSummaryRequest, SearchSummaryResponse
from agents.synthesis_agent import SynthesisAgent
from utils.cache import CacheManager, TTL_SEARCH_SUMMARY
from utils.citation_validator import CitationValidator
from backend.elastic import ELASTIC_CLIENT
from backend.platform import WisefoodClientSingleton

logger = logging.getLogger(__name__)


class SearchSummarizer:
    """Service for summarizing search results across multiple articles."""

    def __init__(
        self,
        cache_enabled: bool = True,
        wisefood_api_key: Optional[str] = None,
    ):
        """
        Initialize search summarizer.

        Args:
            cache_enabled: Whether to enable caching
            wisefood_api_key: API key for WiseFood platform
        """
        self.synthesis_agent = SynthesisAgent()
        self.cache_manager = CacheManager(enabled=cache_enabled)
        self.citation_validator = CitationValidator()
        self.elastic_client = ELASTIC_CLIENT.client
        self.wisefood_client = None

        if wisefood_api_key:
            self.wisefood_client = WisefoodClientSingleton(wisefood_api_key)

    async def summarize_search(
        self, request: SearchSummaryRequest
    ) -> SearchSummaryResponse:
        """
        Summarize search results.

        Args:
            request: SearchSummaryRequest with query and parameters

        Returns:
            SearchSummaryResponse with synthesized findings
        """
        logger.info(f"Processing search summary request: {request.query}")

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

        # Perform search
        articles = await self._search_articles(
            request.query, request.filters, request.max_articles
        )

        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return self._create_empty_response(request)

        logger.info(f"Found {len(articles)} articles for synthesis")

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
                f"Citation validation failed: {validation_report['findings_without_citations']}"
            )

        # Cache the result
        self.cache_manager.set(
            cache_key, summary_response.model_dump(), ttl=TTL_SEARCH_SUMMARY
        )

        return summary_response

    async def _search_articles(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        max_articles: int,
    ) -> List[Dict[str, Any]]:
        """
        Search for articles using Elasticsearch.

        Args:
            query: Search query
            filters: Optional filters
            max_articles: Maximum articles to return

        Returns:
            List of article dictionaries
        """
        try:
            # Build Elasticsearch query
            es_query = self._build_es_query(query, filters)

            # Execute search
            response = self.elastic_client.search(
                index="articles",  # Assuming 'articles' index
                body=es_query,
                size=max_articles,
            )

            # Extract and format results
            articles = []
            for hit in response["hits"]["hits"]:
                article_data = hit["_source"]
                article_data["urn"] = hit["_id"]
                article_data["_score"] = hit.get("_score")
                articles.append(article_data)

            return articles

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []

    def _build_es_query(
        self, query: str, filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build Elasticsearch query.

        Args:
            query: Search query string
            filters: Optional filters (year, journal, etc.)

        Returns:
            Elasticsearch query dict
        """
        # Base multi-match query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^3",
                                    "abstract^2",
                                    "keywords^2",
                                    "content",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": [],
                }
            },
            "sort": [{"_score": {"order": "desc"}}],
        }

        # Add filters if provided
        if filters:
            # Year range filter
            if "year_from" in filters or "year_to" in filters:
                year_filter = {"range": {"year": {}}}
                if "year_from" in filters:
                    year_filter["range"]["year"]["gte"] = filters["year_from"]
                if "year_to" in filters:
                    year_filter["range"]["year"]["lte"] = filters["year_to"]
                es_query["query"]["bool"]["filter"].append(year_filter)

            # Journal filter
            if "journals" in filters and filters["journals"]:
                es_query["query"]["bool"]["filter"].append(
                    {"terms": {"journal.keyword": filters["journals"]}}
                )

            # Category filter
            if "categories" in filters and filters["categories"]:
                es_query["query"]["bool"]["filter"].append(
                    {"terms": {"category.keyword": filters["categories"]}}
                )

            # Tags filter
            if "tags" in filters and filters["tags"]:
                es_query["query"]["bool"]["filter"].append(
                    {"terms": {"tags": filters["tags"]}}
                )

        return es_query

    def _create_empty_response(
        self, request: SearchSummaryRequest
    ) -> SearchSummaryResponse:
        """Create empty response when no articles found."""
        from datetime import datetime

        return SearchSummaryResponse(
            query=request.query,
            summary=f"No articles found for query: '{request.query}'. Try broadening your search terms or adjusting filters.",
            key_findings=[],
            total_articles_analyzed=0,
            all_citations=[],
            search_metadata=request.filters or {},
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            follow_up_suggestions=[
                "Try using broader search terms",
                "Remove some filters to expand results",
                "Check spelling of technical terms",
            ],
        )

    async def get_trending_summaries(
        self, limit: int = 5
    ) -> List[SearchSummaryResponse]:
        """
        Get summaries for trending articles.

        Args:
            limit: Number of trending summaries to generate

        Returns:
            List of SearchSummaryResponse objects
        """
        # TODO: Implement trending article detection
        # This could be based on:
        # - Recent views/downloads
        # - Citation count
        # - Recency
        # - Social media mentions
        logger.info("Getting trending article summaries")
        return []

    def clear_cache(self, pattern: str = "search_summary:*") -> int:
        """
        Clear cached search summaries.

        Args:
            pattern: Cache key pattern to clear

        Returns:
            Number of cache entries cleared
        """
        return self.cache_manager.clear_pattern(pattern)
