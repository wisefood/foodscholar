"""Article enrichment service for scientific articles."""
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from agents.enrichment_agent import EnrichmentAgent

logger = logging.getLogger(__name__)


class ArticleInput(BaseModel):
    """Input model for article enrichment."""

    urn: str = Field(description="Article URN (unique identifier)")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract text")
    authors: Optional[str] = Field(
        default=None, description="Comma-separated list of authors"
    )


class EnrichmentResponse(BaseModel):
    """Response model for article enrichment."""

    urn: str = Field(description="Article URN")
    title: str = Field(description="Article title")
    keywords: list[str] = Field(description="Extracted and homogenized keywords")
    reader_group: str = Field(description="Target reader group for this article")
    population_group: str = Field(description="Study population group")
    study_type: str = Field(description="Type of study (RCT, meta-analysis, etc.)")
    evaluation: Dict[str, Any] = Field(
        description="User value score, actionability, verdict, and safety info"
    )
    annotations: Dict[str, Any] = Field(
        description="Simplified abstract, glossary, and Q&A sections"
    )


class ArticleEnricher:
    """Service for enriching scientific articles with annotations and metadata."""

    def __init__(self):
        """Initialize article enricher."""
        self.enrichment_agent = EnrichmentAgent()

    def enrich_article(self, article: ArticleInput) -> EnrichmentResponse:
        """
        Enrich a single article with annotations, keywords, and Q&A.

        This method:
        1. Extracts keywords from the abstract
        2. Generates full annotations (study type, scores, Q&A, glossary)
        3. Returns the enriched article

        Args:
            article: ArticleInput with urn, title, abstract, and optional authors

        Returns:
            EnrichmentResponse with all enrichment data
        """
        logger.info(f"Processing enrichment request for article: {article.urn}")

        # Validate input
        if not article.abstract:
            logger.warning(f"No abstract provided for article: {article.urn}")
            return self._create_empty_response(article)

        # Convert to format expected by agent (object with attributes)
        article_obj = _ArticleObject(
            urn=article.urn,
            title=article.title,
            abstract=article.abstract,
            authors=article.authors or "",
        )

        # Run enrichment
        logger.info(f"Enriching article: {article.urn}")
        enriched = self.enrichment_agent.enrich_article(article_obj)

        # Build response
        return EnrichmentResponse(
            urn=enriched.get("urn", article.urn),
            title=enriched.get("title", article.title),
            keywords=enriched.get("keywords", []),
            reader_group=enriched.get("reader_group", "Not stated"),
            population_group=enriched.get("population_group", "Not stated"),
            study_type=enriched.get("study_type", "Not stated"),
            evaluation=enriched.get("evaluation", {}),
            annotations=enriched.get("annotations", {}),
        )

    def _create_empty_response(self, article: ArticleInput) -> EnrichmentResponse:
        """Create empty response when article cannot be enriched."""
        return EnrichmentResponse(
            urn=article.urn,
            title=article.title,
            keywords=[],
            reader_group="Not stated",
            population_group="Not stated",
            study_type="Not stated",
            evaluation={
                "user_value_score": 0,
                "actionability_score": 0,
                "verdict": ["Unable to analyze - no abstract provided"],
                "safety_sensitivity": "None",
                "recommended_user_framing": "This article could not be analyzed due to missing abstract.",
            },
            annotations={
                "abstract": "No abstract available for analysis.",
                "glosary": [],
                "user_qa": [],
                "expert_qa": [],
                "practitioner_qa": [],
            },
        )


class _ArticleObject:
    """Simple object wrapper for article data (agent expects object with attributes)."""

    def __init__(self, urn: str, title: str, abstract: str, authors: str):
        self.urn = urn
        self.title = title
        self.abstract = abstract
        self.authors = authors
