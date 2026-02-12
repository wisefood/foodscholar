"""Q&A models and schemas for non-contextual question answering."""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal


from models.search import Citation

AVAILABLE_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class QARequest(BaseModel):
    """Request model for the Q&A endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's question about food science or nutrition",
    )
    mode: Literal["simple", "advanced"] = Field(
        default="simple",
        description="Query mode: 'simple' for default RAG pipeline, 'advanced' for custom model/RAG selection",
    )
    model: Optional[str] = Field(
        default=None,
        description="Groq model to use (advanced mode only). Must be one of the available models.",
    )
    rag_enabled: bool = Field(
        default=True,
        description="Whether to use RAG retrieval (advanced mode only). When False, answers from LLM knowledge only.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of articles to retrieve via kNN search",
    )
    expertise_level: Literal["beginner", "intermediate", "expert"] = Field(
        default="intermediate",
        description="User expertise level to adjust answer complexity",
    )
    language: str = Field(
        default="en",
        description="Language for the answer (ISO 639-1 code)",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for tracking",
    )
    member_id: Optional[str] = Field(
        default=None,
        description="Optional member identifier for tracking",
    )


class QAAnswer(BaseModel):
    """A single Q&A answer with citations and metadata."""

    model_config = {"protected_namespaces": ()}

    answer: str = Field(
        description="Concise, explainable answer in markdown format"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations to articles supporting the answer",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence in the answer"
    )
    model_used: str = Field(
        description="The Groq model that generated this answer"
    )
    rag_used: bool = Field(
        description="Whether RAG retrieval was used for this answer"
    )
    articles_consulted: int = Field(
        default=0,
        description="Number of articles consulted for this answer",
    )


class RetrievedArticle(BaseModel):
    """An article retrieved by kNN vector search."""

    urn: str = Field(description="Article URN")
    title: str = Field(description="Article title")
    authors: Optional[List[str]] = Field(default=None, description="Article authors")
    venue: Optional[str] = Field(default=None, description="Publication venue")
    publication_year: Optional[str] = Field(
        default=None, description="Publication year"
    )
    category: Optional[str] = Field(default=None, description="Article category")
    tags: Optional[List[str]] = Field(default=None, description="Article tags")
    similarity_score: float = Field(
        description="Cosine similarity score from kNN search (0-1)"
    )


class DualAnswerFeedback(BaseModel):
    """Feedback structure for A/B testing dual-answer mode."""

    request_id: str = Field(description="Unique request identifier for tracking")
    answer_a_label: str = Field(
        description="Label describing approach A (e.g., 'model:llama-3.3-70b, temp:0.3')"
    )
    answer_b_label: str = Field(
        description="Label describing approach B (e.g., 'model:llama-3.1-8b, temp:0.3')"
    )


class QAResponse(BaseModel):
    """Response model for the Q&A endpoint."""

    question: str = Field(description="Original question")
    mode: Literal["simple", "advanced"] = Field(description="Mode used")
    primary_answer: QAAnswer = Field(description="The primary answer")
    secondary_answer: Optional[QAAnswer] = Field(
        default=None,
        description="Secondary answer for A/B comparison (present in ~15-20% of requests)",
    )
    dual_answer_feedback: Optional[DualAnswerFeedback] = Field(
        default=None,
        description="Feedback metadata when dual answers are provided",
    )
    retrieved_articles: List[RetrievedArticle] = Field(
        default_factory=list,
        description="Articles retrieved by vector search (shown for transparency)",
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested follow-up questions",
    )
    generated_at: str = Field(description="ISO timestamp of response generation")
    cache_hit: bool = Field(
        default=False, description="Whether this result came from cache"
    )
    request_id: str = Field(
        description="Unique request identifier for feedback tracking"
    )


class QAFeedbackRequest(BaseModel):
    """Request model for submitting feedback on QA answers."""

    request_id: str = Field(
        description="The request_id from the QAResponse being evaluated"
    )
    preferred_answer: Optional[Literal["a", "b", "neither", "both"]] = Field(
        default=None,
        description=(
            "Dual-answer preference (A/B feedback only). "
            "Use when both primary and secondary answers are shown."
        ),
    )
    helpfulness: Optional[Literal["helpful", "not_helpful"]] = Field(
        default=None,
        description=(
            "General helpfulness feedback. "
            "Use for single-answer or overall quality feedback."
        ),
    )
    target_answer: Literal["primary", "secondary", "overall"] = Field(
        default="overall",
        description="Which answer the feedback targets (default: overall).",
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional reason for preference",
    )

    @model_validator(mode="after")
    def validate_feedback_signal(self):
        """Require at least one concrete feedback signal."""
        if self.preferred_answer is None and self.helpfulness is None:
            raise ValueError(
                "Provide at least one of 'preferred_answer' or 'helpfulness'."
            )
        return self


class QAFeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    request_id: str = Field(description="Request identifier")
    status: str = Field(description="Feedback status")
    message: str = Field(description="Confirmation message")
