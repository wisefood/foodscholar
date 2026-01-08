"""Search summary models and schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class Citation(BaseModel):
    """Citation reference to a specific article and section."""

    article_urn: str = Field(description="Unique identifier for the article")
    article_title: str = Field(description="Title of the cited article")
    authors: Optional[List[str]] = Field(
        default=None, description="List of article authors"
    )
    year: Optional[int] = Field(default=None, description="Publication year")
    journal: Optional[str] = Field(default=None, description="Journal name")
    section: str = Field(
        description="Section of the article cited (e.g., 'abstract', 'methods', 'results', 'discussion')"
    )
    quote: Optional[str] = Field(
        default=None, description="Direct quote from the article if applicable"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this citation"
    )
    relevance_score: Optional[float] = Field(
        default=None, description="Relevance score from search (0-1)"
    )


class SynthesizedFinding(BaseModel):
    """A synthesized finding from multiple articles."""

    finding: str = Field(description="The synthesized finding or insight")
    supporting_citations: List[Citation] = Field(
        description="Citations supporting this finding"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence in this finding"
    )
    category: str = Field(
        description="Category of finding (e.g., 'nutrition', 'health outcomes', 'methodology', 'safety')"
    )


class SearchSummaryResponse(BaseModel):
    """Response model for search result summarization."""

    query: str = Field(description="Original search query")
    summary: str = Field(
        description="Natural language summary of search results in markdown format"
    )
    key_findings: List[SynthesizedFinding] = Field(
        description="List of key findings synthesized from articles"
    )
    total_articles_analyzed: int = Field(
        description="Number of articles analyzed for the summary"
    )
    all_citations: List[Citation] = Field(
        description="Complete list of all citations used in the summary"
    )
    search_metadata: Dict[str, Any] = Field(
        description="Metadata about the search (facets used, filters, etc.)"
    )
    generated_at: str = Field(
        description="ISO timestamp when summary was generated"
    )
    cache_hit: bool = Field(
        default=False, description="Whether this result came from cache"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested follow-up searches or questions based on findings",
    )


class SearchSummaryRequest(BaseModel):
    """Request model for search summarization."""

    query: str = Field(description="Search query string")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Facet filters to apply (e.g., year range, journal)"
    )
    max_articles: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of articles to analyze (1-50)",
    )
    language: str = Field(
        default="en", description="Language for the summary (ISO 639-1 code)"
    )
    user_id: Optional[str] = Field(
        default=None, description="User ID for tracking and personalization"
    )
    expertise_level: Literal["beginner", "intermediate", "expert"] = Field(
        default="intermediate",
        description="User expertise level to adjust summary complexity",
    )


class ArticleMetadata(BaseModel):
    """Metadata for a scientific article."""

    urn: str = Field(description="Article URN")
    title: str = Field(description="Article title")
    authors: Optional[List[str]] = Field(default=None)
    abstract: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None)
    journal: Optional[str] = Field(default=None)
    doi: Optional[str] = Field(default=None)
    keywords: Optional[List[str]] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    category: Optional[str] = Field(default=None)
    relevance_score: Optional[float] = Field(default=None)


class ArticleChunk(BaseModel):
    """A chunk of article content with metadata."""

    article_urn: str = Field(description="Article URN this chunk belongs to")
    section: str = Field(description="Section name (e.g., 'introduction', 'methods')")
    content: str = Field(description="Text content of the chunk")
    chunk_index: int = Field(description="Index of this chunk in the article")
    metadata: ArticleMetadata = Field(description="Article metadata")
