"""Article enrichment API endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from services.article_enricher import ArticleEnricher
from src.models.enrich import ArticleInput, EnrichmentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enrich", tags=["Enrichment"])

article_enricher = ArticleEnricher()


@router.post("/article", response_model=EnrichmentResponse)
async def enrich_article(request: ArticleInput):
    """
    Enrich a scientific article with annotations, keywords, and Q&A.

    This endpoint accepts article metadata and:
    1. Extracts and normalizes keywords from the abstract
    2. Classifies study type (RCT, meta-analysis, observational, etc.)
    3. Scores user value and actionability (0-5 scale)
    4. Generates a simplified abstract for general audiences
    5. Creates a glossary of technical terms
    6. Produces Q&A pairs for users, experts, and practitioners

    **Example Request:**
    ```json
    {
        "urn": "urn:article:12345",
        "title": "Effects of Omega-3 Supplementation on Cardiovascular Health",
        "abstract": "Background: Omega-3 fatty acids have been studied...",
        "authors": "Smith J, Doe A, Johnson B"
    }
    ```

    **Returns:**
    - Extracted keywords
    - Study type classification
    - User value and actionability scores
    - Simplified abstract
    - Glossary of key terms
    - Q&A for different expertise levels
    """
    try:
        logger.info(f"Enrichment request for article: {request.urn}")
        result = article_enricher.enrich_article(request)
        return result

    except Exception as e:
        logger.error(f"Error enriching article {request.urn}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error enriching article: {str(e)}",
        )
