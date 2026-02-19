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
    tags: list[str] = Field(description="Generic tags for aggregation across articles")
    reader_group: str = Field(description="Target reader group for this article")
    age_group: str = Field(description="Participant age group")
    population_group: str = Field(description="Study population group")
    geographic_context: Dict[str, Any] = Field(
        description="Geographic context (country/region and income setting)"
    )
    biological_model: str = Field(description="Human/Animal/In vitro/Mixed/Not stated")
    topics: list[str] = Field(description="1-3 normalized topic labels")
    study_type: str = Field(description="Type of study (RCT, meta-analysis, etc.)")
    hard_exclusion_flags: list[str] = Field(description="Hard exclusion flags for indexing")
    annotation_confidence: float = Field(description="0.0-1.0 confidence in classifications")
    evaluation: Dict[str, Any] = Field(
        description="User value score, actionability, verdict, and safety info"
    )
    annotations: Dict[str, Any] = Field(
        description="Simplified abstract, glossary, and Q&A sections"
    )
