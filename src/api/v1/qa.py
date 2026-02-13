"""Q&A API endpoints for non-contextual question answering."""
import logging
import os
from fastapi import APIRouter, Query
from typing import Optional

from models.qa import (
    QARequest,
    QAResponse,
    QAFeedbackRequest,
    QAFeedbackResponse,
    SimpleNutriQuestionsResponse,
    TipsOfTheDayResponse,
    AVAILABLE_GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
)
from services.qa_service import QAService
from exceptions import InvalidError, InternalError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["Question Answering"])

qa_service = QAService(
    cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true"
)


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a food science question and get a concise, cited answer.

    **Simple Mode (default):**
    Automatically retrieves relevant articles via semantic search
    and generates a citation-backed answer.

    **Advanced Mode:**
    Allows selecting a specific LLM model and toggling RAG on/off
    for comparison purposes.

    **Dual-Answer A/B Testing:**
    ~15-20% of simple-mode requests will return two answers generated
    with different approaches. Submit feedback via POST /qa/feedback.

    **Example Request (simple):**
    ```json
    {
        "question": "What are the health benefits of fermented foods?",
        "top_k": 5,
        "expertise_level": "intermediate"
    }
    ```

    **Example Request (advanced):**
    ```json
    {
        "question": "What are the health benefits of fermented foods?",
        "mode": "advanced",
        "model": "llama-3.1-8b-instant",
        "rag_enabled": false,
        "expertise_level": "expert"
    }
    ```
    """
    try:
        logger.info(
            "QA request: mode=%s, question='%s...'",
            request.mode, request.question[:80],
        )
        result = await qa_service.answer_question(request)
        return result
    except InvalidError:
        raise
    except Exception as e:
        logger.error("Error in ask_question: %s", e, exc_info=True)
        raise InternalError(
            detail="Error generating answer. Please try again.",
            extra={"cause": e.__class__.__name__},
        )


@router.post("/feedback", response_model=QAFeedbackResponse)
async def submit_feedback(request: QAFeedbackRequest):
    """
    Submit feedback on a QA answer.

    Use `request_id` from `/qa/ask` and provide either:
    - `preferred_answer` for dual-answer A/B preference, and/or
    - `helpfulness` for general answer quality feedback.

    **Example (dual-answer A/B):**
    ```json
    {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "preferred_answer": "a",
        "reason": "More detailed citations"
    }
    ```

    **Example (single-answer/general):**
    ```json
    {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "helpfulness": "helpful",
        "target_answer": "primary",
        "reason": "Clear and actionable"
    }
    ```
    """
    try:
        result = await qa_service.submit_feedback(request)
        return result
    except Exception as e:
        logger.error("Error submitting feedback: %s", e, exc_info=True)
        raise InternalError(
            detail="Error recording feedback.",
            extra={"cause": e.__class__.__name__},
        )


@router.get("/models")
async def list_available_models():
    """
    List available Groq models for advanced mode.

    Returns the list of models that can be passed in the `model` field
    when using `mode: "advanced"`.
    """
    return {
        "available_models": AVAILABLE_GROQ_MODELS,
        "default_model": DEFAULT_GROQ_MODEL,
    }


@router.get("/questions", response_model=SimpleNutriQuestionsResponse)
async def get_simple_nutri_questions():
    """Get 4 simple starter nutrition questions cached for 30 minutes."""
    try:
        return qa_service.get_simple_nutri_questions()
    except Exception as e:
        logger.error("Error getting starter nutrition questions: %s", e, exc_info=True)
        raise InternalError(
            detail="Error generating starter questions. Please try again.",
            extra={"cause": e.__class__.__name__},
        )


@router.get("/tips", response_model=TipsOfTheDayResponse)
async def get_tips_of_the_day():
    """Get 2 did_you_know facts and 2 tips, cached for 30 minutes."""
    try:
        return qa_service.get_tips_of_the_day()
    except Exception as e:
        logger.error("Error getting nutrition tips of the day: %s", e, exc_info=True)
        raise InternalError(
            detail="Error generating tips of the day. Please try again.",
            extra={"cause": e.__class__.__name__},
        )


@router.delete("/cache/clear")
async def clear_qa_cache(
    pattern: Optional[str] = Query(
        default="qa:*",
        description="Cache key pattern to clear",
    )
):
    """Clear cached Q&A responses."""
    try:
        cleared = qa_service.cache_manager.clear_pattern(pattern)
        return {
            "message": "Cache cleared successfully",
            "pattern": pattern,
            "entries_cleared": cleared,
        }
    except Exception as e:
        logger.error("Error clearing QA cache: %s", e, exc_info=True)
        raise InternalError(detail="Error clearing cache.")
