"""Q&A models and schemas for non-contextual question answering."""
from datetime import datetime

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, Literal

from config import config


# Both are deployment configuration (QA_AVAILABLE_MODELS / QA_DEFAULT_MODEL).
# The list is simultaneously the advanced-mode request validator and the
# contract /qa/models advertises to the UI, so retiring a model id is an env
# change: no rebuild, and the picker cannot drift from what the server accepts.
AVAILABLE_GROQ_MODELS = config.settings["QA_AVAILABLE_MODELS"]

DEFAULT_GROQ_MODEL = config.settings["QA_DEFAULT_MODEL"]


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


class SubQuestionFilters(BaseModel):
    """Structured attribute constraints the planner extracted for one search.

    Metadata-aware retrieval: deterministic, mapping-backed constraints
    (publication year window, open access) become hard Elasticsearch filters
    on BOTH legs — the kNN leg is informed by them, not just the BM25 leg.
    Vocabulary-shaped attributes (study design, regions, facets) become
    strong boosts instead, so an enrichment gap never silently empties
    retrieval.
    """

    year_min: Optional[int] = Field(
        default=None, description="Earliest publication year, inclusive"
    )
    year_max: Optional[int] = Field(
        default=None, description="Latest publication year, inclusive"
    )
    open_access: Optional[bool] = Field(
        default=None, description="Restrict to open-access articles when true"
    )
    study_types: List[str] = Field(
        default_factory=list,
        description="Preferred study designs, e.g. 'meta-analysis', 'rct'",
    )
    regions: List[str] = Field(
        default_factory=list,
        description="Guideline regions/countries the question targets",
    )
    target_populations: List[str] = Field(
        default_factory=list,
        description="Populations/life stages, e.g. 'pregnant_people', 'infants'",
    )
    food_groups: List[str] = Field(
        default_factory=list,
        description="Food groups the question concerns",
    )
    nutrients: List[str] = Field(
        default_factory=list,
        description="Nutrients the question concerns",
    )
    health_conditions: List[str] = Field(
        default_factory=list,
        description="Health conditions the question concerns, e.g. 'diabetes'",
    )

    def is_empty(self) -> bool:
        return not any(
            [
                self.year_min,
                self.year_max,
                self.open_access,
                self.study_types,
                self.regions,
                self.target_populations,
                self.food_groups,
                self.nutrients,
                self.health_conditions,
            ]
        )


class PlannedSubQuestion(BaseModel):
    """One typed sub-question the planner decided to search, and why."""

    id: str = Field(description="Stable id within the plan, e.g. 'sq1'")
    text: str = Field(description="The sub-question in natural language")
    why: str = Field(
        default="",
        description="One-line user-visible rationale for running this search",
    )
    qtype: Literal[
        "quantity",
        "mechanism",
        "safety",
        "recommendation",
        "comparison",
        "general",
    ] = Field(default="general", description="What kind of evidence it seeks")
    branch: Literal["articles", "guidelines", "both"] = Field(
        default="both",
        description="Which evidence branch(es) this sub-question targets",
    )
    lexical_query: str = Field(
        default="",
        description="Keyword-style query for the BM25 leg",
    )
    dense_query: str = Field(
        default="",
        description="Natural-sentence query for the vector leg",
    )
    filters: SubQuestionFilters = Field(
        default_factory=SubQuestionFilters,
        description="Attribute constraints applied to both retrieval legs",
    )
    round_added: int = Field(
        default=0,
        description="Pipeline round in which this sub-question was added",
    )


class QAPipelinePlan(QAClarifierSafetyPlan):
    """Clarifier/safety plan extended with the retrieval decomposition."""

    sub_questions: List[PlannedSubQuestion] = Field(
        default_factory=list,
        description="Typed sub-questions with reasoned queries",
    )


class ResearchNote(BaseModel):
    """A working note the pipeline keeps while researching a question.

    Notes accumulate across retrieval rounds ("found strong RCT evidence on X",
    "gap: no pediatric guidance retrieved", "lead: search for 'alpha-linolenic
    acid' instead") and persist on the QA thread so a follow-up question starts
    from what earlier searches already established.
    """

    text: str = Field(description="The note itself, one short sentence")
    kind: Literal["finding", "gap", "lead"] = Field(
        default="finding",
        description=(
            "finding = evidence located; gap = something the corpus lacked; "
            "lead = a promising direction for a subsequent search"
        ),
    )
    sub_question_id: Optional[str] = Field(
        default=None,
        description="Sub-question this note came from, when attributable",
    )
    source_urns: List[str] = Field(
        default_factory=list,
        description="Source URNs the note refers to",
    )


class ReasoningStep(BaseModel):
    """One user-visible step of the agentic pipeline, for collapsible UI.

    Streamed as ``step`` SSE events while running (same ``id`` re-emitted with
    an updated status) and included in full on the final response, so the UI
    can render a ChatGPT-style inline step disclosure both live and after the
    fact (page reload, cached answers, the non-streaming endpoint).

    ``title``/``detail`` are ready-to-render text (dynamic parts such as the
    planner's "why" come localized from the prompts); ``kind`` plus ``data``
    carry the structured form for UIs that prefer to compose their own labels.
    """

    id: str = Field(description="Stable step id within the request")
    kind: Literal[
        "plan",
        "search",
        "rank",
        "notes",
        "evaluate",
        "repair",
        "answer",
        "cache",
        "clarification",
    ] = Field(description="What kind of pipeline step this is")
    status: Literal["running", "done"] = Field(
        default="running",
        description="Steps stream twice: once running, once done",
    )
    title: str = Field(description="Short ready-to-render step label")
    detail: Optional[str] = Field(
        default=None,
        description="One optional supporting line under the title",
    )
    round: int = Field(
        default=0,
        description="Pipeline round the step belongs to (repairs bump it)",
    )
    elapsed_ms: Optional[int] = Field(
        default=None,
        description="Wall-clock duration, set when the step completes",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload (queries, counts, notes, verdicts...)",
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
    quote_context_before: Optional[str] = Field(
        default=None,
        description=(
            "Source text immediately preceding the quote, so a preview can "
            "show the cited line in its surroundings"
        ),
    )
    quote_context_after: Optional[str] = Field(
        default=None,
        description="Source text immediately following the quote",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this citation",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Relevance score from retrieval",
    )
    display_label: Optional[str] = Field(
        default=None,
        description="Short display label for inline citation, e.g. G1, G2 for guidelines",
    )
    guide_urn: Optional[str] = Field(
        default=None,
        description=(
            "Parent guide of a guideline citation. Lets a client land on the "
            "guide page even when the rule itself is no longer publicly "
            "readable (same contract as TipEvidence)."
        ),
    )
    region: Optional[str] = Field(
        default=None,
        description="Guide region of a guideline citation, for guide routing",
    )
    page_no: Optional[int] = Field(
        default=None,
        description="PDF page the cited rule came from, when known",
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
    guide_urn: Optional[str] = Field(
        default=None,
        description=(
            "Parent guide of a guideline source, so the UI can deep-link the "
            "guide page even when the rule is not publicly readable"
        ),
    )
    page_no: Optional[int] = Field(
        default=None,
        description="PDF page a guideline rule came from, when known",
    )
    citation_count: Optional[int] = Field(
        default=None,
        description=(
            "Semantic Scholar citation count, when stored — surfaces the "
            "prioritization signal to the reader"
        ),
    )
    influential_citation_count: Optional[int] = Field(
        default=None,
        description="Semantic Scholar influential citation count, when stored",
    )
    study_type: Optional[str] = Field(
        default=None,
        description="Enrichment-assigned study design (ai_category), when set",
    )


class DualAnswerFeedback(BaseModel):
    """Feedback structure for A/B testing dual-answer mode."""

    request_id: str = Field(description="Unique request identifier for tracking")
    answer_a_label: str = Field(
        description="Label describing approach A (e.g., 'model:openai/gpt-oss-120b, temp:0.3')"
    )
    answer_b_label: str = Field(
        description="Label describing approach B (e.g., 'model:openai/gpt-oss-20b, temp:0.3')"
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
    memory_suggestions: Optional[List["MemorySuggestion"]] = Field(
        default=None,
        description=(
            "Consent nudges for durable preferences the user expressed in the "
            "question ('It seems you love lentils — remember this?'). Written "
            "to the member profile only via POST /qa/memory on an explicit yes."
        ),
    )
    reasoning_steps: Optional[List[ReasoningStep]] = Field(
        default=None,
        description=(
            "The pipeline steps that produced this answer (searches with "
            "rationales, ranking, evidence checks, notes), for transparent "
            "collapsible rendering. Null on the legacy pipeline."
        ),
    )
    conversation_context: Optional["ConversationContext"] = Field(
        default=None,
        description=(
            "What this thread is carrying into the next turn — the compacted "
            "summary of earlier exchanges and the accumulated research notes — "
            "so the UI can show the user what FoodScholar remembers."
        ),
    )


class ConversationContext(BaseModel):
    """The thread memory made visible: summary + notes carried forward."""

    summary: Optional[str] = Field(
        default=None,
        description="Compacted summary of exchanges before the recent window",
    )
    notes: List[ResearchNote] = Field(
        default_factory=list,
        description="Research notes carried on this thread",
    )
    turn_count: int = Field(
        default=0,
        description="How many answered exchanges this thread has seen",
    )


class MemorySuggestion(BaseModel):
    """One durable-preference candidate awaiting the user's consent (same
    shape as FoodChat's nudges so the UI renders both identically)."""

    id: str = Field(description="Client-echoed suggestion id")
    kind: Literal[
        "like", "dislike", "cuisine", "allergy_hint", "goal", "dietary_pattern"
    ] = Field(description="What profile field an acceptance writes to")
    value: str = Field(description="Canonical lowercase item / goal slug / pattern")
    statement: str = Field(description="The nudge question shown to the user")
    source_text: Optional[str] = Field(
        default=None,
        description=(
            "The question this was inferred from, echoed back on accept and "
            "stored with the memory as its provenance"
        ),
    )


class MemoryDecisionRequest(BaseModel):
    """User's answer to a memory nudge (client echoes the suggestion back)."""

    member_id: str = Field(description="Member whose profile is affected")
    suggestion: MemorySuggestion
    decision: Literal["accept", "decline"]


class MemoryDecisionResponse(BaseModel):
    applied: bool = Field(
        description="True when a durable profile change was persisted"
    )
    decision: Literal["accept", "decline"]


# QAResponse forward-references MemorySuggestion (defined just above).
QAResponse.model_rebuild()


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
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Keycloak subject of the person giving feedback. Stamped by the "
            "gateway from the verified token, never supplied by a browser."
        ),
    )
    member_id: Optional[str] = Field(
        default=None, description="Household member the feedback was given as."
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


# ----------------------------- Review models -----------------------------
#
# The read side of `qa_requests` / `qa_feedback`. Both tables have been written
# on every question since they were added and never read: there was no GET for
# either, so the answers experts most need to see — the ones somebody marked
# unhelpful — were reachable only through a psql session.


class QARequestSummary(BaseModel):
    """One asked question, as it appears in a review list."""

    request_id: str
    question: str
    mode: str
    model: str
    language: str
    expertise_level: str
    created_at: datetime
    user_id: Optional[str] = None
    member_id: Optional[str] = None
    #: The gateway's X-Request-Id. For a question asked inside a FoodChat turn
    #: this is the only route back to the user, because FoodChat holds a signed
    #: member assertion rather than a Keycloak subject and cannot say who asked.
    correlation_id: Optional[str] = None
    confidence: Optional[str] = None
    articles_consulted: int = 0
    cache_hit: bool = False
    has_feedback: bool = False
    feedback_count: int = 0
    #: True when at least one piece of feedback on this answer was negative —
    #: the filter an expert reviewing quality actually wants.
    has_negative_feedback: bool = False
    answer_preview: Optional[str] = None


class QARequestDetail(QARequestSummary):
    """One asked question with everything needed to judge the answer."""

    primary_answer: Optional[Dict[str, Any]] = None
    secondary_answer: Optional[Dict[str, Any]] = None
    dual_strategy: Optional[str] = None
    retrieved_article_urns: List[str] = Field(default_factory=list)
    pipeline_meta: Optional[Dict[str, Any]] = None
    rag_enabled: bool = True
    top_k: int = 5
    feedback: List["QAFeedbackEntry"] = Field(default_factory=list)


class QAFeedbackEntry(BaseModel):
    """One piece of feedback, with enough context to act on it."""

    id: str
    request_id: str
    question: Optional[str] = None
    preferred_answer: Optional[str] = None
    helpfulness: Optional[str] = None
    target_answer: str = "overall"
    feedback_mode: str = "general"
    reason: Optional[str] = None
    user_id: Optional[str] = None
    member_id: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: datetime

    @property
    def is_negative(self) -> bool:
        return self.helpfulness == "not_helpful"


QARequestDetail.model_rebuild()


class QARequestListResponse(BaseModel):
    """A page of asked questions."""

    total: int = Field(description="Matching questions, before paging")
    limit: int
    offset: int
    items: List[QARequestSummary] = Field(default_factory=list)


class QAFeedbackListResponse(BaseModel):
    """A page of feedback."""

    total: int
    limit: int
    offset: int
    items: List[QAFeedbackEntry] = Field(default_factory=list)


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
    source_type: Optional[str] = Field(
        default=None, description="'guideline' or 'article'"
    )
    guide_urn: Optional[str] = Field(
        default=None,
        description=(
            "Parent guide of a guideline source. Lets a client deep-link even "
            "when the rule itself is no longer publicly readable."
        ),
    )
    page_no: Optional[int] = Field(
        default=None, description="Page the rule came from, when known"
    )
    region: Optional[str] = Field(
        default=None, description="Guide region, used to build the guide path"
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
