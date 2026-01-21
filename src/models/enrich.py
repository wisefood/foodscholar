from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

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