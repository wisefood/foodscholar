"""Q&A models and schemas for non-contextual question answering."""
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, Literal


AVAILABLE_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class ClarificationOption(BaseModel):
    """A selectable clarification option presented to the user."""

    label: str = Field(description="Human-readable option label")
    value: str = Field(description="Stable value to send back if selected")
    description: Optional[str] = Field(
        default=None,
        description="Optional short explanation of what this option means",
    )


class ClarificationRequest(BaseModel):
    """Structured short-horizon clarification requested before answering."""

    id: str = Field(description="Stable clarification question identifier")
    question: str = Field(description="The clarification question to present")
    input_type: Literal[
        "single_choice",
        "multiple_choice",
        "free_text",
        "number",
        "boolean",
    ] = Field(default="single_choice", description="Expected answer control")
    options: List[ClarificationOption] = Field(
        default_factory=list,
        description="Selectable options when the input type uses choices",
    )
    allow_free_text: bool = Field(
        default=True,
        description="Whether the user may provide a free-text clarification",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why this clarification materially changes the answer",
    )


class ClarificationAnswer(BaseModel):
    """Structured answer to a prior clarification request."""

    question_id: Optional[str] = Field(
        default=None,
        description="ClarificationRequest.id being answered",
    )
    selected_values: List[str] = Field(
        default_factory=list,
        description="Selected option values for choice-based clarifications",
    )
    free_text: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Free-text clarification when choices are insufficient",
    )


class QAUserContext(BaseModel):
    """Resolved member context used to personalize retrieval and answering."""

    country: Optional[str] = Field(
        default=None,
        description="Country inferred from the household region when available",
    )
    region: Optional[str] = Field(
        default=None,
        description="Household region from the WiseFood API when available",
    )
    experience_group: Optional[str] = Field(
        default=None,
        description="Audience/experience group used for answer formulation",
    )
    member_age_group: Optional[str] = Field(
        default=None,
        description="Age group from the selected household member when available",
    )
    profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Small safe subset of member profile data useful for QA",
    )


class QAClarifierSafetyPlan(BaseModel):
    """Structured output from the combined clarifier/safety planning step."""

    original_question: str = Field(description="Original user-facing question")
    canonical_question: str = Field(
        description="Canonical version of the question for internal planning"
    )
    article_query: str = Field(description="Query optimized for article retrieval")
    guideline_query: str = Field(
        description="Query optimized for dietary guideline retrieval"
    )
    output_language: Optional[str] = Field(
        default=None,
        description="Detected or requested answer language",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Nutrition/medical-nutrition safety risk level",
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description="Safety-sensitive factors detected in the question",
    )
    answer_guardrails: List[str] = Field(
        default_factory=list,
        description="Answer constraints to pass to the answer formulation step",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether a material clarification should be requested",
    )
    clarification: Optional[ClarificationRequest] = Field(
        default=None,
        description="Structured clarification prompt when needed",
    )
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Short operational reason for the plan",
    )


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
        description="Number of article sources to retrieve via kNN search",
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
    experience_group: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Optional audience/experience group used by retrieval and answer "
            "formulation. Defaults to expertise_level."
        ),
    )
    retriever: Literal["rag", "no_rag", "linearrag"] = Field(
        default="rag",
        description="Retrieval strategy: 'rag' for Elasticsearch kNN, 'linearrag' for graph-based retrieval, 'no_rag' for LLM-only",
    )
    qa_thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Short-horizon QA thread id returned when the service asks a "
            "clarification question."
        ),
    )
    clarification_response: Optional[ClarificationAnswer] = Field(
        default=None,
        description="Structured answer to a clarification request.",
    )


class QACitation(BaseModel):
    """Citation reference to a retrieved source."""

    source_type: Literal["article", "guideline"] = Field(
        description="Type of cited source",
    )
    source_id: str = Field(description="Source URN/id")
    source_title: str = Field(description="Source title")
    source_url: Optional[str] = Field(
        default=None,
        description="Application URL for the cited source when available",
    )
    authors: Optional[List[str]] = Field(
        default=None,
        description="Article authors; null for guideline sources",
    )
    year: Optional[int] = Field(default=None, description="Publication/source year")
    venue: Optional[str] = Field(
        default=None,
        description="Journal, publisher, country, or guideline region",
    )
    section: str = Field(description="Section cited, e.g. abstract or rule_text")
    quote: Optional[str] = Field(
        default=None,
        description="Direct quote from the source if applicable",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this citation",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Relevance score from retrieval",
    )


class QAAnswer(BaseModel):
    """A single Q&A answer with citations and metadata."""

    model_config = {"protected_namespaces": ()}

    answer: str = Field(
        description="Concise, explainable answer in markdown format"
    )
    citations: List[QACitation] = Field(
        default_factory=list,
        description="Citations to retrieved sources supporting the answer",
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
    sources_consulted: int = Field(
        default=0,
        description="Number of retrieved sources consulted for this answer",
    )
    articles_consulted: int = Field(
        default=0,
        exclude=True,
        description="Deprecated: use sources_consulted.",
    )


class RetrievedSource(BaseModel):
    """A source retrieved for RAG context."""

    source_type: Literal["article", "guideline"] = Field(
        default="article",
        description="Type of retrieved source",
    )
    urn: str = Field(description="Source URN/id")
    title: str = Field(description="Source title")
    authors: Optional[List[str]] = Field(
        default=None,
        description="Article authors; null for guidelines",
    )
    venue: Optional[str] = Field(
        default=None,
        description="Publication venue or guideline region",
    )
    publication_year: Optional[str] = Field(
        default=None, description="Publication year or source date"
    )
    category: Optional[str] = Field(default=None, description="Source category")
    tags: Optional[List[str]] = Field(default=None, description="Source tags")
    similarity_score: float = Field(
        description="Retriever relevance score"
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
    retrieved_sources: List[RetrievedSource] = Field(
        default_factory=list,
        description="Sources retrieved by RAG (shown for transparency)",
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
    qa_thread_id: Optional[str] = Field(
        default=None,
        description="Short-horizon QA thread id for clarification flow",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the client should collect clarification before answering",
    )
    clarification: Optional[ClarificationRequest] = Field(
        default=None,
        description="Structured clarification prompt to present to the user",
    )
    user_context: Optional[QAUserContext] = Field(
        default=None,
        description="Resolved country/experience context used by this response",
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


class SimpleNutriQuestionsResponse(BaseModel):
    """Response model for starter nutrition questions."""

    questions: List[str] = Field(
        description="A list of simple starter nutrition questions"
    )
    generated_at: str = Field(
        description="ISO timestamp when these questions were generated"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether this result came from cache",
    )

class TipEvidence(BaseModel):
    """Evidence payload for a generated tip/fact."""

    urn: str = Field(description="Source URN/id used as evidence")
    passage: str = Field(
        description="Short passage from the source text used for grounding"
    )
    title: Optional[str] = Field(
        default=None,
        description="Source title (optional)",
    )
    publication_year: Optional[str] = Field(
        default=None,
        description="Source date/year metadata (optional)",
    )


class TipWithEvidence(BaseModel):
    """A tip/fact with optional evidence."""

    text: str = Field(description="Tip/fact text")
    evidence: Optional[TipEvidence] = Field(
        default=None,
        description="Evidence used to create the item; omitted for fallbacks",
    )


class TipsOfTheDayResponse(BaseModel):
    """Response model for nutrition tips/facts of the day."""

    did_you_know: List[str] = Field(
        default_factory=list,
        description="Exactly 2 short human-focused 'Did you know?' nutrition facts"
    )
    tips: List[str] = Field(
        default_factory=list,
        description="Exactly 2 short human-focused nutrition tips"
    )
    did_you_know_detail: List[TipWithEvidence] = Field(
        default_factory=list,
        description="Did-you-know items including (when available) the source id + passage used",
    )
    tips_detail: List[TipWithEvidence] = Field(
        default_factory=list,
        description="Tip items including (when available) the source id + passage used",
    )
    generated_at: str = Field(
        description="ISO timestamp when these tips were generated"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether this result came from cache",
    )
