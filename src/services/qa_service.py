"""Q&A service for non-contextual question answering."""
import json
import logging
import random
import re
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

_TIP_PASSAGE_MAX_CHARS = 320
_TIP_PASSAGE_MIN_CHARS = 60

try:
    import json5  # type: ignore
except Exception:  # pragma: no cover
    json5 = None

from models.qa import (
    QARequest,
    QAResponse,
    QAAnswer,
    RetrievedSource,
    QAClarifierSafetyPlan,
    ClarificationOption,
    ClarificationRequest,
    QAUserContext,
    DualAnswerFeedback,
    QAFeedbackRequest,
    QAFeedbackResponse,
    SimpleNutriQuestionsResponse,
    TipsOfTheDayResponse,
    AVAILABLE_GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
)
from backend.elastic import ELASTIC_CLIENT
from backend.groq import GROQ_CHAT
from services.qa_retrievers import (
    QARetrieverAdapters,
    RetrievalResult,
    guideline_context_should_clauses,
)
from utilities.cache import CacheManager
from exceptions import InvalidError

logger = logging.getLogger(__name__)

TTL_QA_RESPONSE = 86400  # 1 day
TTL_QA_FEEDBACK = 2592000  # 30 days
TTL_QA_CLARIFICATION_THREAD = 1800  # 30 minutes
TTL_SIMPLE_NUTRI_QUESTIONS = 1800  # 30 minutes
TTL_TIPS_OF_THE_DAY = 1800  # 30 minutes
TIP_GROUNDING_TOP_K = 3
TIP_SOURCE_ARTICLE_POOL_SIZE = 10
TIP_SOURCE_ARTICLE_MIN_ABSTRACT_CHARS = 40
TIP_SOURCE_GUIDELINE_POOL_SIZE = 24
TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS = 12
TIP_SOURCE_TOPIC_QUERIES = [
    "fruit vegetables daily intake healthy eating",
    "whole grains bread rice pasta dietary guidance",
    "water drinks sugar beverages hydration guideline",
    "fish eggs legumes beans protein foods guideline",
    "milk dairy calcium dietary guide",
    "salt sodium limit food guide",
    "saturated fat oils spreads dietary guidance",
    "red meat processed meat weekly intake guideline",
    "breakfast cereals fiber wholegrain guideline",
    "snacks sweets alcohol moderation dietary guide",
]
QA_GUIDELINE_RAG_TOP_K_MAX = 5
MAX_TIP_REGEN_ATTEMPTS = 3
MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS = 3
REGION_LABELS = {
    "EU": "European Union",
    "GR": "Greece",
    "HU": "Hungary",
    "SI": "Slovenia",
    "UK": "United Kingdom",
    "US": "United States",
}
TIPS_OF_THE_DAY_TIPS_COUNT = 2
TIPS_OF_THE_DAY_DID_YOU_KNOW_COUNT = 2
SIMPLE_NUTRI_QUESTION_MODEL = "openai/gpt-oss-20b"
SIMPLE_QUESTION_BLOCKED_TERMS = [
    "do you",
    "what's your",
    "what is your",
    "your ",
    "go-to",
    "meal plan",
    "meal-planning",
    "plan meals",
    "lunch",
    "dinner",
    "breakfast",
    "snack",
    "recipe",
    "menu",
    "prep in",
    "prepare in",
    "what should i eat",
    "what to eat",
]
ANIMAL_EVIDENCE_TERMS = [
    "animal",
    "animals",
    "animal study",
    "animal studies",
    "animal model",
    "mouse",
    "mice",
    "dogs",
    "cats",
    "mouse model",
    "rat",
    "rats",
    "murine",
    "rodent",
    "zebrafish",
    "drosophila",
    "canine",
    "dog model",
    "pig model",
    "rabbit",
    "preclinical",
    "in vivo",
]
HUMAN_EVIDENCE_TERMS = [
    "human", "humans", "participant", "participants", "patient", "patients", "adult",
    "adults", "clinical trial", "randomized", "rct", "cohort", "meta-analysis",
    "systematic review", "cross-sectional", "observational study",
]


def _compile_term_patterns(terms: List[str]) -> List[re.Pattern]:
    return [
        re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        for term in terms
        if isinstance(term, str) and term.strip()
    ]


_ANIMAL_TERM_PATTERNS = _compile_term_patterns(ANIMAL_EVIDENCE_TERMS)
_HUMAN_TERM_PATTERNS = _compile_term_patterns(HUMAN_EVIDENCE_TERMS)


def _linearrag_retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    """Lazy-load LinearRAG so non-Linearrag paths do not require its dependencies."""
    from services.linearrag_service import retrieve

    return retrieve(question, top_k=top_k)

AMBIGUITY_KEYWORDS = [
    "better", "worse", "should", "recommend", "best", "opinion",
    "compared", "versus", "vs", "controversial", "debate",
    "safe", "dangerous", "risk", "healthy", "unhealthy",
]

DUAL_ANSWER_STRATEGIES = [
    ("model_comparison", {}, {"model": "llama-3.1-8b-instant"}),
    ("temperature_variation", {"temperature": 0.3}, {"temperature": 0.7}),
    ("top_k_variation", {"top_k": 5}, {"top_k": 10}),
]


class QAService:
    """Service for non-contextual Q&A with optional RAG."""

    INDEX_NAME = "articles"
    GUIDELINES_INDEX_NAME = "guidelines"

    def __init__(self, cache_enabled: bool = True):
        self.cache_manager = CacheManager(enabled=cache_enabled)
        self._embedder = None
        self._simple_question_llm = None
        self._simple_question_redis = None
        self._qa_threads: Dict[str, Dict[str, Any]] = {}
        self._retriever_adapters = QARetrieverAdapters(
            embed_query=self._embed_query,
            articles_index=self.INDEX_NAME,
            guidelines_index=self.GUIDELINES_INDEX_NAME,
        )

    @property
    def embedder(self):
        """Lazy-load the sentence-transformers model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Loaded sentence-transformers/all-MiniLM-L6-v2 embedding model")
        return self._embedder

    def _embed_query(self, question: str) -> List[float]:
        """Embed a question string into a 384-dim vector."""
        embedding = self.embedder.encode(question, normalize_embeddings=True)
        return embedding.tolist()

    @property
    def simple_question_llm(self):
        """Lazy-load Groq client for starter nutrition question generation."""
        if self._simple_question_llm is None:
            self._simple_question_llm = GROQ_CHAT.get_client(
                model=SIMPLE_NUTRI_QUESTION_MODEL,
                temperature=0.8,
            )
        return self._simple_question_llm

    @property
    def simple_question_redis(self):
        """Strict Redis client for starter question caching."""
        if self._simple_question_redis is None:
            from backend.redis import RedisClientSingleton

            client = RedisClientSingleton().client
            client.ping()
            self._simple_question_redis = client
        return self._simple_question_redis

    def get_simple_nutri_questions(self) -> SimpleNutriQuestionsResponse:
        """Return 4 starter nutrition questions with strict Redis caching."""
        cache_key = self.cache_manager.generate_cache_key(
            prefix="qa",
            data={
                "type": "simple_nutri_questions",
                "version": 2,
                "count": 4,
            },
        )

        try:
            cached = self.simple_question_redis.get(cache_key)
            if cached:
                payload = json.loads(cached) if isinstance(cached, str) else cached
                cached_questions = self._normalize_simple_questions(
                    payload.get("questions", []),
                    count=4,
                )
                if len(cached_questions) == 4:
                    payload["questions"] = cached_questions
                    payload["cache_hit"] = True
                    return SimpleNutriQuestionsResponse(**payload)
                logger.warning(
                    "Cached starter questions failed guardrails; regenerating."
                )
        except Exception as e:
            logger.error(
                "Redis read failed for starter questions cache key %s: %s",
                cache_key,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                "Redis is required for starter question caching but is unavailable."
            ) from e

        generated_questions = self._generate_simple_nutri_questions(count=4)
        response = SimpleNutriQuestionsResponse(
            questions=generated_questions,
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
        )

        self._persist_simple_questions_generation(
            cache_key=cache_key,
            questions=response.questions,
            generated_at=response.generated_at,
            model=SIMPLE_NUTRI_QUESTION_MODEL,
            count=4,
        )

        try:
            self.simple_question_redis.setex(
                cache_key,
                TTL_SIMPLE_NUTRI_QUESTIONS,
                json.dumps(response.model_dump()),
            )
        except Exception as e:
            logger.error(
                "Redis write failed for starter questions cache key %s: %s",
                cache_key,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                "Redis is required for starter question caching but is unavailable."
            ) from e

        return response

    def get_tips_of_the_day(self) -> TipsOfTheDayResponse:
        """Return 2 tips + 2 did-you-know facts with strict Redis caching."""
        tips_count = TIPS_OF_THE_DAY_TIPS_COUNT
        did_you_know_count = TIPS_OF_THE_DAY_DID_YOU_KNOW_COUNT
        cache_key = self.cache_manager.generate_cache_key(
            prefix="qa",
            data={
                "type": "tips_of_the_day",
                "version": 5,
                "tips_count": tips_count,
                "did_you_know_count": did_you_know_count,
            },
        )

        try:
            cached = self.simple_question_redis.get(cache_key)
            if cached:
                payload = json.loads(cached) if isinstance(cached, str) else cached
                payload = self._normalize_tips_payload(
                    payload,
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                )
                if self._is_tips_payload_appropriate(
                    payload,
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                ):
                    did_you_know_detail, tips_detail = self._normalize_tip_details(
                        payload,
                        tips_count=tips_count,
                        did_you_know_count=did_you_know_count,
                    )
                    payload["did_you_know_detail"] = did_you_know_detail
                    payload["tips_detail"] = tips_detail
                    payload["cache_hit"] = True
                    return TipsOfTheDayResponse(**payload)
                logger.warning(
                    "Cached tips payload failed guardrails; regenerating content."
                )
        except Exception as e:
            logger.error(
                "Redis read failed for tips cache key %s: %s",
                cache_key,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                "Redis is required for tips caching but is unavailable."
            ) from e

        generated_payload = self._generate_tips_of_the_day(
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )
        generated_payload = self._normalize_tips_payload(
            generated_payload,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )
        did_you_know_detail, tips_detail = self._normalize_tip_details(
            generated_payload,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )
        response = TipsOfTheDayResponse(
            did_you_know=generated_payload["did_you_know"],
            tips=generated_payload["tips"],
            did_you_know_detail=did_you_know_detail,
            tips_detail=tips_detail,
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
        )

        self._persist_tips_generation(
            cache_key=cache_key,
            tips=response.tips,
            did_you_know=response.did_you_know,
            tips_detail=[d.model_dump() for d in response.tips_detail],
            did_you_know_detail=[d.model_dump() for d in response.did_you_know_detail],
            generated_at=response.generated_at,
            model=SIMPLE_NUTRI_QUESTION_MODEL,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )

        try:
            self.simple_question_redis.setex(
                cache_key,
                TTL_TIPS_OF_THE_DAY,
                json.dumps(response.model_dump()),
            )
        except Exception as e:
            logger.error(
                "Redis write failed for tips cache key %s: %s",
                cache_key,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                "Redis is required for tips caching but is unavailable."
            ) from e

        return response

    def _parse_iso_datetime(self, value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _persist_simple_questions_generation(
        self,
        *,
        cache_key: str,
        questions: List[str],
        generated_at: str,
        model: Optional[str],
        count: int,
    ) -> None:
        """Best-effort persistence for starter questions (sync)."""
        with suppress(Exception):
            from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
            from models.db import SimpleNutriQuestionsRecord

            factory = POSTGRES_SYNC_SESSION_FACTORY()
            with factory() as session:
                session.add(
                    SimpleNutriQuestionsRecord(
                        cache_key=cache_key,
                        model=model,
                        count=count,
                        questions=questions,
                        generated_at=self._parse_iso_datetime(generated_at),
                    )
                )
                session.commit()

    def _persist_tips_generation(
        self,
        *,
        cache_key: str,
        tips: List[str],
        did_you_know: List[str],
        tips_detail: List[Dict[str, Any]],
        did_you_know_detail: List[Dict[str, Any]],
        generated_at: str,
        model: Optional[str],
        tips_count: int,
        did_you_know_count: int,
    ) -> None:
        """Best-effort persistence for tips (sync)."""
        with suppress(Exception):
            from backend.postgres import POSTGRES_SYNC_SESSION_FACTORY
            from models.db import TipsOfTheDayRecord

            factory = POSTGRES_SYNC_SESSION_FACTORY()
            with factory() as session:
                session.add(
                    TipsOfTheDayRecord(
                        cache_key=cache_key,
                        model=model,
                        tips_count=tips_count,
                        did_you_know_count=did_you_know_count,
                        tips=tips,
                        did_you_know=did_you_know,
                        tips_detail=tips_detail,
                        did_you_know_detail=did_you_know_detail,
                        generated_at=self._parse_iso_datetime(generated_at),
                    )
                )
                session.commit()

    async def answer_question(self, request: QARequest) -> QAResponse:
        """
        Main orchestration: validate, clarify if needed, retrieve, generate.

        Flow:
        1. Validate request (model selection in advanced mode)
        2. Resolve member/country/experience context
        3. Ask one structured clarification only when materially needed
        4. Check cache
        5. Retrieve sources, if enabled
        6. Generate primary answer via QAAgent
        7. Optionally generate secondary answer (dual-answer A/B)
        8. Cache and persist result
        """
        request_id = str(uuid.uuid4())

        self._validate_request(request)

        effective_model = self._resolve_model(request)
        effective_retriever = self._resolve_retriever(request)
        effective_rag = effective_retriever != "no_rag"
        user_context = self._resolve_user_context(request)
        thread_context = self._load_qa_thread(request.qa_thread_id)
        user_context = self._merge_thread_user_context(user_context, thread_context)

        if thread_context and request.clarification_response is None:
            clarification = ClarificationRequest(**thread_context["clarification"])
            return self._build_clarification_response(
                request=request,
                request_id=request_id,
                thread_id=thread_context["thread_id"],
                clarification=clarification,
                model_used=effective_model,
                user_context=user_context,
            )

        effective_question = self._compose_effective_question(
            request=request,
            thread_context=thread_context,
        )
        user_context = self._apply_clarification_to_user_context(
            user_context,
            request.clarification_response,
        )
        answered_ids = self._answered_clarification_ids(thread_context, request)
        plan = self._plan_clarification_safety(
            question=effective_question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
            model=effective_model,
        )

        clarification = plan.clarification if plan.needs_clarification else None
        ask_before_scout = (
            clarification is not None
            and self._should_ask_clarification_before_scout(
                clarification,
                effective_rag=effective_rag,
            )
        )
        if clarification and ask_before_scout:
            thread_id = request.qa_thread_id or str(uuid.uuid4())
            self._store_qa_thread(
                thread_id=thread_id,
                question=effective_question,
                request=request,
                clarification=clarification,
                user_context=user_context,
                answered_ids=answered_ids,
            )
            return self._build_clarification_response(
                request=request,
                request_id=request_id,
                thread_id=thread_id,
                clarification=clarification,
                model_used=effective_model,
                user_context=user_context,
            )

        cache_checked = False
        # Cache check
        cache_key = self._build_cache_key(
            request,
            question=effective_question,
            effective_retriever=effective_retriever,
            user_context=user_context,
        )
        if clarification is None:
            cache_checked = True
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.info("Cache hit for QA: %s...", effective_question[:50])
                cached["cache_hit"] = True
                cached["request_id"] = request_id
                if request.qa_thread_id and request.clarification_response:
                    self._clear_qa_thread(request.qa_thread_id)
                return QAResponse(**cached)

        # Retrieval
        source_payloads: List[Dict[str, Any]] = []
        retrieved_sources: List[RetrievedSource] = []
        retrieval_result = self._retrieve_sources(
            question=effective_question,
            plan=plan,
            top_k=request.top_k,
            retriever=effective_retriever,
            user_context=user_context,
        )
        if effective_rag:
            source_payloads = retrieval_result.source_payloads
            retrieved_sources = retrieval_result.retrieved_sources

        scout_clarification = self._build_scout_clarification(
            question=effective_question,
            plan=plan,
            pending_clarification=clarification,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
            source_payloads=source_payloads,
            retrieved_sources=retrieved_sources,
            retrieval_result=retrieval_result,
            effective_rag=effective_rag,
        )
        if scout_clarification:
            thread_id = request.qa_thread_id or str(uuid.uuid4())
            self._store_qa_thread(
                thread_id=thread_id,
                question=effective_question,
                request=request,
                clarification=scout_clarification,
                user_context=user_context,
                answered_ids=answered_ids,
            )
            return self._build_clarification_response(
                request=request,
                request_id=request_id,
                thread_id=thread_id,
                clarification=scout_clarification,
                model_used=effective_model,
                user_context=user_context,
            )

        if not cache_checked:
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.info("Cache hit for QA: %s...", effective_question[:50])
                cached["cache_hit"] = True
                cached["request_id"] = request_id
                if request.qa_thread_id and request.clarification_response:
                    self._clear_qa_thread(request.qa_thread_id)
                return QAResponse(**cached)

        # Primary answer
        from agents.qa_agent import QAAgent

        primary_agent = QAAgent(model=effective_model, temperature=0.3)
        context_payload = user_context.model_dump(exclude_none=True)
        context_payload["safety"] = {
            "risk_level": plan.risk_level,
            "flags": plan.safety_flags,
            "guardrails": plan.answer_guardrails,
        }
        context_payload["retrieval_plan"] = {
            "article_query": plan.article_query,
            "guideline_query": plan.guideline_query,
        }
        context_payload["retrieval_scout"] = self._summarize_retrieval_scout(
            retrieval_result,
            source_payloads,
        )
        if effective_rag and source_payloads:
            primary_answer, follow_ups = primary_agent.generate_answer_with_rag(
                question=effective_question,
                articles=source_payloads,
                expertise_level=request.expertise_level,
                language=request.language,
                retriever=effective_retriever,
                user_context=context_payload,
            )
        else:
            primary_answer, follow_ups = primary_agent.generate_answer_without_rag(
                question=effective_question,
                expertise_level=request.expertise_level,
                language=request.language,
                user_context=context_payload,
            )

        # Dual-answer logic (simple mode only)
        secondary_answer: Optional[QAAnswer] = None
        dual_feedback: Optional[DualAnswerFeedback] = None
        dual_strategy: Optional[str] = None
        if request.mode == "simple" and self._should_generate_dual_answer(
            effective_question
        ):
            secondary_answer, dual_feedback, dual_strategy = (
                self._generate_secondary_answer(
                    request,
                    source_payloads,
                    request_id,
                    question=effective_question,
                    retriever=effective_retriever,
                    user_context=context_payload,
                )
            )

        response = QAResponse(
            question=effective_question,
            mode=request.mode,
            primary_answer=primary_answer,
            secondary_answer=secondary_answer,
            dual_answer_feedback=dual_feedback,
            retrieved_sources=retrieved_sources,
            follow_up_suggestions=follow_ups or None,
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            request_id=request_id,
            qa_thread_id=request.qa_thread_id,
            needs_clarification=False,
            clarification=None,
            user_context=user_context,
        )

        # Cache only non-dual responses
        if secondary_answer is None:
            self.cache_manager.set(
                cache_key, response.model_dump(), ttl=TTL_QA_RESPONSE
            )

        # Persist to PostgreSQL
        persist_request = request.model_copy(
            update={
                "question": effective_question,
                "retriever": effective_retriever,
            }
        )
        await self._persist_request(
            persist_request, response, effective_model, effective_rag, dual_strategy
        )
        if request.qa_thread_id and request.clarification_response:
            self._clear_qa_thread(request.qa_thread_id)

        return response

    def _resolve_retriever(self, request: QARequest) -> str:
        """Resolve the effective retriever after advanced-mode toggles."""
        if request.mode == "advanced" and not request.rag_enabled:
            return "no_rag"
        return request.retriever

    def _plan_clarification_safety(
        self,
        *,
        question: str,
        request: QARequest,
        user_context: QAUserContext,
        answered_ids: Set[str],
        model: str,
    ) -> QAClarifierSafetyPlan:
        """Run the single structured clarifier/safety step with fallback."""
        from agents.qa_clarifier import (
            QAClarifierSafetyAgent,
            build_fallback_plan,
        )

        try:
            return QAClarifierSafetyAgent(model=model).plan(
                question=question,
                request=request,
                user_context=user_context,
                answered_ids=answered_ids,
            )
        except Exception as exc:
            logger.warning(
                "Using fallback QA clarifier/safety plan: %s",
                exc,
                exc_info=True,
            )
            return build_fallback_plan(
                question=question,
                request=request,
                user_context=user_context,
                answered_ids=answered_ids,
            )

    def _retrieve_sources(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        top_k: int,
        retriever: str,
        user_context: Optional[QAUserContext],
    ) -> RetrievalResult:
        """Retrieve evidence through the selected adapter."""
        self._retriever_adapters = QARetrieverAdapters(
            embed_query=self._embed_query,
            articles_index=self.INDEX_NAME,
            guidelines_index=self.GUIDELINES_INDEX_NAME,
        )
        adapter = self._retriever_adapters.get(retriever)
        result = adapter.retrieve(
            question=question,
            plan=plan,
            top_k=top_k,
            user_context=user_context,
        )
        logger.info("QA retrieval status: %s", result.status)
        return result

    @staticmethod
    def _should_ask_clarification_before_scout(
        clarification: ClarificationRequest,
        *,
        effective_rag: bool,
    ) -> bool:
        """Only defer clarification types that benefit from evidence scouting."""
        if clarification.id == "country_or_region" and effective_rag:
            return False
        return True

    def _build_scout_clarification(
        self,
        *,
        question: str,
        plan: QAClarifierSafetyPlan,
        pending_clarification: Optional[ClarificationRequest],
        request: QARequest,
        user_context: QAUserContext,
        answered_ids: Set[str],
        source_payloads: List[Dict[str, Any]],
        retrieved_sources: List[RetrievedSource],
        retrieval_result: RetrievalResult,
        effective_rag: bool,
    ) -> Optional[ClarificationRequest]:
        """Ask evidence-backed clarifications after the retrieval scout pass."""
        if not effective_rag or "country_or_region" in answered_ids:
            return None
        if user_context.country:
            return None

        wants_region = (
            pending_clarification is not None
            and pending_clarification.id == "country_or_region"
        ) or self._looks_region_sensitive_question(question, plan)
        if not wants_region:
            return None

        options = self._region_options_from_guideline_scout(
            source_payloads=source_payloads,
            retrieved_sources=retrieved_sources,
        )
        if not options and pending_clarification:
            options = pending_clarification.options
        if not options:
            return None

        if not any(option.value == "general" for option in options):
            options.append(
                ClarificationOption(
                    label="No preference",
                    value="general",
                    description="Use the best available evidence across regions.",
                )
            )

        status = retrieval_result.status or {}
        reason = (
            "Guideline candidates vary by region, and regional dietary "
            "guidance can change the practical recommendation."
        )
        if status.get("guideline_hits") == 0:
            reason = (
                "No strong regional guideline candidate was found, so the "
                "answer can use a general evidence base if preferred."
            )

        return ClarificationRequest(
            id="country_or_region",
            question="Which country or guideline region should the answer use?",
            input_type="single_choice",
            options=options[:5],
            allow_free_text=True,
            reason=reason,
        )

    @staticmethod
    def _looks_region_sensitive_question(
        question: str,
        plan: QAClarifierSafetyPlan,
    ) -> bool:
        text = " ".join(
            [
                question,
                plan.canonical_question,
                plan.guideline_query,
            ]
        ).lower()
        terms = [
            "how much",
            "how many",
            "how often",
            "weekly",
            "per week",
            "per day",
            "should",
            "recommend",
            "guideline",
            "intake",
            "serving",
            "portion",
        ]
        return any(term in text for term in terms)

    def _region_options_from_guideline_scout(
        self,
        *,
        source_payloads: List[Dict[str, Any]],
        retrieved_sources: List[RetrievedSource],
    ) -> List[ClarificationOption]:
        seen: Set[str] = set()
        options: List[ClarificationOption] = []

        for source in source_payloads:
            if source.get("source_type") != "guideline":
                continue
            value = self._source_region_value(source)
            if not value or value in seen:
                continue
            seen.add(value)
            options.append(
                ClarificationOption(
                    label=self._region_label(value),
                    value=value,
                    description="Found matching guideline candidates for this region.",
                )
            )
            if len(options) >= 4:
                return options

        for source in retrieved_sources:
            if source.source_type != "guideline":
                continue
            value = self._source_region_value(
                {
                    "country": source.venue,
                    "guide_region": source.venue,
                    "tags": source.tags,
                }
            )
            if not value or value in seen:
                continue
            seen.add(value)
            options.append(
                ClarificationOption(
                    label=self._region_label(value),
                    value=value,
                    description="Found matching guideline candidates for this region.",
                )
            )
            if len(options) >= 4:
                break

        return options

    @staticmethod
    def _source_region_value(source: Dict[str, Any]) -> Optional[str]:
        for key in ("country", "guide_region", "venue", "source"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        tags = source.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and 1 < len(tag.strip()) <= 8:
                    return tag.strip().upper()
        return None

    @staticmethod
    def _region_label(value: str) -> str:
        return REGION_LABELS.get(value.upper(), value)

    @staticmethod
    def _summarize_retrieval_scout(
        retrieval_result: RetrievalResult,
        source_payloads: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        guideline_regions = []
        for source in source_payloads:
            if source.get("source_type") != "guideline":
                continue
            region = QAService._source_region_value(source)
            if region and region not in guideline_regions:
                guideline_regions.append(region)
        return {
            "status": retrieval_result.status,
            "guideline_regions": guideline_regions[:6],
            "source_count": len(source_payloads),
        }

    def _resolve_user_context(self, request: QARequest) -> QAUserContext:
        """Resolve country/region and audience context for retrieval and answering."""
        context = QAUserContext(
            experience_group=request.experience_group or request.expertise_level,
        )

        if not request.member_id:
            return context

        try:
            member_context = self._lookup_member_context(request.member_id)
        except Exception as exc:
            logger.warning(
                "Unable to resolve WiseFood member context for %s: %s",
                request.member_id,
                exc,
                exc_info=True,
            )
            return context

        data = context.model_dump()
        data.update({k: v for k, v in member_context.items() if v is not None})
        if not data.get("country") and data.get("region"):
            data["country"] = self._country_from_region(data["region"])
        return QAUserContext(**data)

    def _lookup_member_context(self, member_id: str) -> Dict[str, Any]:
        """Fetch member household context through the WiseFood API client."""
        from backend.platform import WISEFOOD_PLATFORM

        client = WISEFOOD_PLATFORM.get_client()
        try:
            members_proxy = getattr(client, "members", None) or getattr(
                client, "member", None
            )
            households_proxy = getattr(client, "households", None) or getattr(
                client, "household", None
            )
            if not members_proxy or not households_proxy:
                raise RuntimeError("WiseFood Client does not expose members/households")

            member = members_proxy.get(member_id)
            household_id = self._object_value(member, "household_id")
            household = households_proxy.get(household_id) if household_id else None

            region = self._object_value(household, "region") if household else None
            profile = self._member_profile_dict(member)
            experience_group = (
                profile.get("experience_group")
                or profile.get("expertise_level")
                or profile.get("nutrition_experience")
            )

            return {
                "country": self._country_from_region(region),
                "region": region,
                "experience_group": experience_group,
                "member_age_group": self._object_value(member, "age_group"),
                "profile": self._safe_profile_subset(profile),
            }
        finally:
            WISEFOOD_PLATFORM.return_client(client)

    @staticmethod
    def _object_value(obj: Any, key: str) -> Optional[Any]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        if hasattr(obj, key):
            return getattr(obj, key)
        data = getattr(obj, "_data", None)
        if isinstance(data, dict):
            return data.get(key)
        return None

    def _member_profile_dict(self, member: Any) -> Dict[str, Any]:
        try:
            profile = getattr(member, "profile", None)
        except Exception:
            return {}
        if profile is None:
            return {}
        if isinstance(profile, dict):
            return profile
        if hasattr(profile, "to_dict"):
            data = profile.to_dict()
            return data if isinstance(data, dict) else {}
        data = getattr(profile, "_data", None)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _safe_profile_subset(profile: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "dietary_groups",
            "allergies",
            "nutritional_preferences",
            "properties",
        }
        return {key: profile[key] for key in allowed if key in profile}

    @staticmethod
    def _country_from_region(region: Optional[str]) -> Optional[str]:
        if not isinstance(region, str) or not region.strip():
            return None
        region = region.strip()
        if "-" in region:
            return region.split("-", 1)[0].upper()
        if len(region) <= 3:
            return region.upper()
        return region

    def _load_qa_thread(
        self, thread_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not thread_id:
            return None

        now = datetime.now(timezone.utc).timestamp()
        thread = self._qa_threads.get(thread_id)
        if thread and thread.get("expires_at", 0) > now:
            return thread
        if thread:
            self._qa_threads.pop(thread_id, None)

        cached = self.cache_manager.get(f"qa_thread:{thread_id}")
        if isinstance(cached, dict):
            cached["thread_id"] = thread_id
            cached.setdefault("expires_at", now + TTL_QA_CLARIFICATION_THREAD)
            self._qa_threads[thread_id] = cached
            return cached
        return None

    def _clear_qa_thread(self, thread_id: str) -> None:
        self._qa_threads.pop(thread_id, None)
        self.cache_manager.delete(f"qa_thread:{thread_id}")

    @staticmethod
    def _merge_thread_user_context(
        current: QAUserContext,
        thread_context: Optional[Dict[str, Any]],
    ) -> QAUserContext:
        if not thread_context or not isinstance(thread_context.get("user_context"), dict):
            return current

        stored = {
            key: value
            for key, value in thread_context["user_context"].items()
            if value not in (None, "", [], {})
        }
        current_data = current.model_dump()
        merged = {**stored, **{k: v for k, v in current_data.items() if v not in (None, "", [], {})}}
        return QAUserContext(**merged)

    @staticmethod
    def _answered_clarification_ids(
        thread_context: Optional[Dict[str, Any]],
        request: QARequest,
    ) -> Set[str]:
        answered = set()
        if thread_context:
            answered.update(
                str(item)
                for item in thread_context.get("answered_ids", [])
                if item
            )
        if request.clarification_response and request.clarification_response.question_id:
            answered.add(request.clarification_response.question_id)
        return answered

    def _apply_clarification_to_user_context(
        self,
        user_context: QAUserContext,
        clarification_response: Optional[Any],
    ) -> QAUserContext:
        if clarification_response is None or not clarification_response.question_id:
            return user_context

        data = user_context.model_dump()
        selected = [
            value
            for value in clarification_response.selected_values
            if isinstance(value, str) and value.strip()
        ]
        free_text = (
            clarification_response.free_text.strip()
            if isinstance(clarification_response.free_text, str)
            else None
        )

        if clarification_response.question_id == "country_or_region":
            value = selected[0] if selected else free_text
            if value and value != "general":
                data["region"] = value
                data["country"] = self._country_from_region(value)

        if clarification_response.question_id == "target_age_group":
            value = selected[0] if selected else free_text
            if value:
                data["member_age_group"] = value

        return QAUserContext(**data)

    def _store_qa_thread(
        self,
        *,
        thread_id: str,
        question: str,
        request: QARequest,
        clarification: ClarificationRequest,
        user_context: QAUserContext,
        answered_ids: Optional[Set[str]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).timestamp()
        payload = {
            "thread_id": thread_id,
            "question": question,
            "request": request.model_dump(),
            "clarification": clarification.model_dump(),
            "user_context": user_context.model_dump(),
            "answered_ids": sorted(answered_ids or set()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": now + TTL_QA_CLARIFICATION_THREAD,
        }
        self._qa_threads[thread_id] = payload
        self.cache_manager.set(
            f"qa_thread:{thread_id}",
            payload,
            ttl=TTL_QA_CLARIFICATION_THREAD,
        )

    def _compose_effective_question(
        self,
        *,
        request: QARequest,
        thread_context: Optional[Dict[str, Any]],
    ) -> str:
        if not thread_context or not request.clarification_response:
            return request.question

        base_question = thread_context.get("question") or request.question
        clarification = thread_context.get("clarification") or {}
        answer_text = self._format_clarification_answer(
            clarification=clarification,
            response=request.clarification_response.model_dump(),
        )
        if not answer_text:
            return base_question

        clarification_question = clarification.get("question") or "Clarification"
        return (
            f"{base_question}\n\n"
            f"User clarification - {clarification_question}: {answer_text}"
        )

    @staticmethod
    def _format_clarification_answer(
        *,
        clarification: Dict[str, Any],
        response: Dict[str, Any],
    ) -> str:
        options = clarification.get("options") or []
        labels_by_value = {
            str(option.get("value")): option.get("label", option.get("value"))
            for option in options
            if isinstance(option, dict)
        }
        selected = [
            labels_by_value.get(str(value), str(value))
            for value in response.get("selected_values", [])
            if value
        ]
        free_text = response.get("free_text")
        parts = selected
        if isinstance(free_text, str) and free_text.strip():
            parts.append(free_text.strip())
        return "; ".join(parts)

    def _build_material_clarification(
        self,
        *,
        question: str,
        request: QARequest,
        user_context: QAUserContext,
        answered_ids: Optional[Set[str]] = None,
    ) -> Optional[ClarificationRequest]:
        from agents.qa_clarifier import build_fallback_plan

        plan = build_fallback_plan(
            question=question,
            request=request,
            user_context=user_context,
            answered_ids=answered_ids,
        )
        return plan.clarification if plan.needs_clarification else None

    def _build_clarification_response(
        self,
        *,
        request: QARequest,
        request_id: str,
        thread_id: str,
        clarification: ClarificationRequest,
        model_used: str,
        user_context: QAUserContext,
    ) -> QAResponse:
        answer = QAAnswer(
            answer=f"Before answering, FoodScholar needs one detail: {clarification.question}",
            citations=[],
            confidence="low",
            model_used=model_used,
            rag_used=False,
            sources_consulted=0,
            articles_consulted=0,
        )
        return QAResponse(
            question=request.question,
            mode=request.mode,
            primary_answer=answer,
            retrieved_sources=[],
            follow_up_suggestions=None,
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            request_id=request_id,
            qa_thread_id=thread_id,
            needs_clarification=True,
            clarification=clarification,
            user_context=user_context,
        )

    async def _persist_request(
        self,
        request: QARequest,
        response: QAResponse,
        effective_model: str,
        effective_rag: bool,
        dual_strategy: Optional[str] = None,
    ) -> None:
        """Persist the QA request and response to PostgreSQL (best-effort)."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import QARequestRecord

        try:
            factory = POSTGRES_ASYNC_SESSION_FACTORY()
            async with factory() as session:
                record = QARequestRecord(
                    id=uuid.UUID(response.request_id),
                    question=request.question,
                    mode=request.mode,
                    model=effective_model,
                    rag_enabled=effective_rag,
                    top_k=request.top_k,
                    expertise_level=request.expertise_level,
                    language=request.language,
                    user_id=request.user_id,
                    member_id=request.member_id,
                    primary_answer=response.primary_answer.model_dump(),
                    secondary_answer=(
                        response.secondary_answer.model_dump()
                        if response.secondary_answer
                        else None
                    ),
                    dual_strategy=dual_strategy,
                    retrieved_article_urns=[
                        a.urn for a in response.retrieved_sources
                    ],
                    confidence=response.primary_answer.confidence,
                    articles_consulted=response.primary_answer.articles_consulted,
                    cache_hit=response.cache_hit,
                )
                session.add(record)
                await session.commit()
                logger.info(
                    "Persisted QA request %s to PostgreSQL", response.request_id
                )
        except Exception as e:
            logger.error(
                "Failed to persist QA request %s: %s",
                response.request_id, e, exc_info=True,
            )

    def _contextualize_retrieval_question(
        self,
        question: str,
        user_context: Optional[QAUserContext],
    ) -> str:
        """Append lightweight context that helps heterogeneous retrievers."""
        if not user_context:
            return question

        hints = []
        if user_context.country:
            hints.append(f"country {user_context.country}")
        if user_context.region and user_context.region != user_context.country:
            hints.append(f"region {user_context.region}")
        if user_context.member_age_group:
            hints.append(f"age group {user_context.member_age_group}")
        if user_context.experience_group:
            hints.append(f"audience {user_context.experience_group}")

        if not hints:
            return question
        return f"{question}\nContext: {', '.join(hints)}"

    @staticmethod
    def _infer_source_type(source: Dict[str, Any]) -> str:
        source_type = source.get("source_type")
        if source_type in {"article", "guideline"}:
            return source_type
        if source.get("rule_text") or source.get("guide_region") or source.get("guide_urn"):
            return "guideline"
        return "article"

    def _guideline_context_should_clauses(
        self,
        user_context: Optional[QAUserContext],
    ) -> List[Dict[str, Any]]:
        """Build non-restrictive boosts for region and audience-specific guidance."""
        return guideline_context_should_clauses(user_context)

    def _retrieve_articles(
        self,
        question: str,
        top_k: int,
        retriever: str = "rag",
        user_context: Optional[QAUserContext] = None,
    ) -> Tuple[List[Dict[str, Any]], List[RetrievedSource]]:
        plan = QAClarifierSafetyPlan(
            original_question=question,
            canonical_question=question,
            article_query=question,
            guideline_query=f"{question} dietary guideline recommendation",
        )
        result = self._retrieve_sources(
            question=question,
            plan=plan,
            top_k=top_k,
            retriever=retriever,
            user_context=user_context,
        )
        return result.source_payloads, result.retrieved_sources

    def _retrieve_article_rag_sources(
        self,
        question: str,
        top_k: int,
        user_context: Optional[QAUserContext] = None,
    ) -> Tuple[List[Dict[str, Any]], List[RetrievedSource]]:
        """Retrieve article sources for default Elasticsearch RAG."""
        try:
            query_vector = self._embed_query(question)
            raw_results = ELASTIC_CLIENT.knn_search(
                index_name=self.INDEX_NAME,
                query_vector=query_vector,
                k=top_k,
                num_candidates=max(top_k * 20, 100),
                field="embedding",
                filter_query={
                    "bool": {"must_not": {"term": {"status": "deleted"}}}
                },
                source_excludes=["embedding"],
            )
        except Exception as e:
            logger.error("Article RAG retrieval failed: %s", e, exc_info=True)
            return [], []

        articles: List[Dict[str, Any]] = []
        retrieved_models: List[RetrievedSource] = []
        for result in raw_results:
            result["source_type"] = "article"
            result["retriever"] = "rag"
            result["relevance_score"] = result.get("_score", 0.0)
            articles.append(result)
            retrieved_models.append(
                RetrievedSource(
                    source_type="article",
                    urn=result.get("urn", result.get("_id", "")),
                    title=result.get("title", ""),
                    authors=result.get("authors"),
                    venue=result.get("venue"),
                    publication_year=result.get("publication_year"),
                    category=result.get("category"),
                    tags=result.get("tags"),
                    similarity_score=result.get("_score", 0.0),
                )
            )
        return articles, retrieved_models

    def _retrieve_guideline_rag_sources(
        self,
        question: str,
        top_k: int,
        user_context: Optional[QAUserContext] = None,
    ) -> Tuple[List[Dict[str, Any]], List[RetrievedSource]]:
        """Retrieve dietary guideline rule_text sources for default RAG."""
        if top_k <= 0:
            return [], []

        context_should = self._guideline_context_should_clauses(user_context)
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": question,
                                "fields": [
                                    "rule_text^4",
                                    "title^2",
                                    "notes",
                                    "food_groups^2",
                                    "target_populations",
                                    "guide_region",
                                    "country",
                                    "population",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "must_not": [{"term": {"status": "deleted"}}],
                }
            },
        }
        if context_should:
            body["query"]["bool"]["should"] = context_should

        try:
            response = ELASTIC_CLIENT.client.search(
                index=self.GUIDELINES_INDEX_NAME,
                body=body,
            )
        except Exception as e:
            logger.error("Guideline RAG retrieval failed: %s", e, exc_info=True)
            return [], []

        guidelines: List[Dict[str, Any]] = []
        retrieved_models: List[RetrievedSource] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            if not isinstance(source, dict):
                continue

            guideline = {
                **source,
                "_id": hit.get("_id"),
                "_score": hit.get("_score", 0.0),
                "source_type": "guideline",
                "retriever": "rag",
            }
            rule_text = self._extract_guideline_rule_text(guideline)
            if len(rule_text) < TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS:
                continue

            urn = self._guideline_document_urn(guideline)
            guideline["urn"] = urn
            guideline["abstract"] = rule_text
            guideline["description"] = rule_text
            guideline["venue"] = guideline.get("guide_region")
            guideline["relevance_score"] = guideline.get("_score", 0.0)
            guideline["publication_year"] = self._guideline_publication_year(
                guideline
            )

            tags = self._guideline_tags(guideline)
            guidelines.append(guideline)
            retrieved_models.append(
                RetrievedSource(
                    source_type="guideline",
                    urn=urn,
                    title=guideline.get("title", "Dietary guideline"),
                    authors=None,
                    venue=guideline.get("guide_region"),
                    publication_year=guideline.get("publication_year"),
                    category="guideline",
                    tags=tags,
                    similarity_score=guideline.get("_score", 0.0),
                )
            )

        return guidelines, retrieved_models

    def _generate_simple_nutri_questions(self, count: int = 4) -> List[str]:
        """Generate assistant-directed nutrition questions with validation."""
        for attempt in range(1, MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS + 1):
            try:
                questions = self._generate_simple_questions_once(count=count)
                questions = self._normalize_simple_questions(questions, count=count)
                if len(questions) == count:
                    return questions
                logger.warning(
                    "Starter questions failed guardrails; regenerating (%d/%d).",
                    attempt,
                    MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS,
                )
            except Exception as e:
                logger.error(
                    "Error generating starter questions attempt %d/%d with %s: %s",
                    attempt,
                    MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS,
                    SIMPLE_NUTRI_QUESTION_MODEL,
                    e,
                    exc_info=True,
                )

        logger.warning(
            "Falling back to static-safe starter questions after %d failed attempts.",
            MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS,
        )
        return self._generate_fallback_questions(count=count)

    def _generate_simple_questions_once(self, count: int) -> List[str]:
        """Single pass generation for starter nutrition questions."""
        prompt = f"""You are creating starter questions that a user can ask an AI nutrition assistant.

Generate exactly {count} short nutrition-science questions that can be submitted by an average user, not expert.

Rules:
- Questions must be directed to the ΑΙ, so the user can submit them. Dont us first-person wording.
- Do NOT ask about the user's habits, preferences, or choices.
- Do NOT use wording like "do you", "your", "what's your", or "go-to".
- Do NOT generate meal-planning or food-suggestion content (no lunch/dinner/snack/recipe/menu/prep ideas).
- Focus on general nutrition science, food composition, and evidence-based concepts.
- Keep each question <= 16 words.
- Avoid diagnosis, treatments, and supplement dosage advice.
- Return ONLY valid JSON in this format:
{{"questions": ["q1", "q2", "q3", "q4"]}}
"""
        response = self.simple_question_llm.invoke(prompt)
        parsed = self._parse_questions_json(response.content)
        questions = parsed.get("questions", [])
        if not isinstance(questions, list):
            return []
        return [q for q in questions if isinstance(q, str)]

    def _normalize_simple_questions(self, questions: List[str], count: int) -> List[str]:
        """Normalize and validate starter questions."""
        unique = []
        seen = set()
        for question in questions:
            cleaned = " ".join(question.split()).strip()
            if not cleaned:
                continue
            if not cleaned.endswith("?"):
                cleaned += "?"
            if not self._is_simple_question_appropriate(cleaned):
                continue

            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(cleaned)

            if len(unique) >= count:
                break

        return unique[:count]

    def _is_simple_question_appropriate(self, question: str) -> bool:
        """Guardrails for starter questions."""
        normalized = question.lower()
        if any(term in normalized for term in SIMPLE_QUESTION_BLOCKED_TERMS):
            return False
        if self._violates_tip_safety_policy(normalized):
            return False
        if self._mentions_animal_testing(normalized):
            return False
        return True

    def _generate_tips_of_the_day(
        self,
        tips_count: int = TIPS_OF_THE_DAY_TIPS_COUNT,
        did_you_know_count: int = TIPS_OF_THE_DAY_DID_YOU_KNOW_COUNT,
    ) -> Dict[str, Any]:
        """Generate grounded tips/facts with validation and regeneration."""
        for attempt in range(1, MAX_TIP_REGEN_ATTEMPTS + 1):
            try:
                payload = self._generate_tips_payload_once(
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                    seed_offset=attempt,
                )
                # During generation attempts, don't auto-fill missing items with
                # fallbacks; allow regeneration to kick in.
                payload = self._normalize_tips_payload(
                    payload,
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                    fill_with_fallback=False,
                )

                if self._is_tips_payload_appropriate(
                    payload,
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                ):
                    # Final normalize can fill (should be no-op when complete).
                    return self._normalize_tips_payload(
                        payload,
                        tips_count=tips_count,
                        did_you_know_count=did_you_know_count,
                        fill_with_fallback=True,
                    )

                logger.warning(
                    "Tips payload failed guardrails; regenerating (%d/%d).",
                    attempt,
                    MAX_TIP_REGEN_ATTEMPTS,
                )
            except Exception as e:
                logger.error(
                    "Error generating tips attempt %d/%d with %s: %s",
                    attempt,
                    MAX_TIP_REGEN_ATTEMPTS,
                    SIMPLE_NUTRI_QUESTION_MODEL,
                    e,
                    exc_info=True,
                )

        logger.warning(
            "Falling back to static-safe tips after %d failed attempts.",
            MAX_TIP_REGEN_ATTEMPTS,
        )
        return self._format_tips_payload(
            self._generate_fallback_tips(
                tips_count=tips_count,
                did_you_know_count=did_you_know_count,
            ),
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )

    def _generate_tips_payload_once(
        self, tips_count: int, did_you_know_count: int, *, seed_offset: int = 0
    ) -> Dict[str, Any]:
        """Single generation pass for tips/facts."""
        total_count = tips_count + did_you_know_count
        candidate_count = max(total_count * 3, 12)

        # Use a daily seed so content varies day-to-day but stays stable within a day.
        daily_seed = int(datetime.now(timezone.utc).strftime("%Y%m%d")) + int(
            seed_offset or 0
        )
        guideline_query = self._tip_source_query(daily_seed, branch="guidelines")
        source_guidelines = self._get_random_tip_source_guidelines(
            size=max(TIP_SOURCE_GUIDELINE_POOL_SIZE, candidate_count),
            seed=daily_seed,
            query=guideline_query,
        )
        candidates = self._generate_tip_candidates_from_guidelines(
            guidelines=source_guidelines,
            candidate_count=candidate_count,
        )
        if not candidates:
            logger.warning(
                "No guideline-backed tips could be generated; falling back to article sources."
            )
            article_query = self._tip_source_query(daily_seed, branch="articles")
            source_articles = self._get_random_tip_source_articles(
                size=max(TIP_SOURCE_ARTICLE_POOL_SIZE, candidate_count),
                seed=daily_seed,
                query=article_query,
            )
            candidates = self._generate_tip_candidates_from_articles(
                articles=source_articles,
                candidate_count=candidate_count,
            )

        did_you_know_detail: List[Dict[str, Any]] = []
        tips_detail: List[Dict[str, Any]] = []
        seen = set()

        for item in candidates:
            if (
                len(did_you_know_detail) >= did_you_know_count
                and len(tips_detail) >= tips_count
            ):
                break
            if not isinstance(item, dict):
                continue

            kind = str(item.get("kind", "")).strip().lower()
            text = str(item.get("text", "")).strip()
            urn = str(item.get("urn", "")).strip()
            passage = str(item.get("passage", "")).strip()
            title = item.get("title", None)
            publication_year = item.get("publication_year", None)

            if kind not in ("tip", "did_you_know") or not text:
                continue
            if not self._is_tip_line_appropriate(text):
                continue
            if not urn or not passage:
                continue

            key = f"{kind}:{text.lower()}"
            if key in seen:
                continue
            seen.add(key)

            record = {
                "text": text,
                "evidence": {
                    "urn": urn,
                    "passage": passage,
                    "title": title if isinstance(title, str) and title.strip() else None,
                    "publication_year": (
                        publication_year
                        if isinstance(publication_year, str)
                        and publication_year.strip()
                        else None
                    ),
                },
            }

            if (
                kind == "did_you_know"
                and len(did_you_know_detail) < did_you_know_count
            ):
                did_you_know_detail.append(record)
            elif kind == "tip" and len(tips_detail) < tips_count:
                tips_detail.append(record)

        # No fallback here; allow regeneration to try a different random pool.
        return {
            "did_you_know": [d["text"] for d in did_you_know_detail],
            "tips": [d["text"] for d in tips_detail],
            "did_you_know_detail": did_you_know_detail,
            "tips_detail": tips_detail,
        }

    def _tip_source_query(self, seed: int, *, branch: str) -> str:
        """Select a varied topical query for daily tip source scouting."""
        offset = 0 if branch == "guidelines" else len(TIP_SOURCE_TOPIC_QUERIES) // 2
        idx = (int(seed) + offset) % len(TIP_SOURCE_TOPIC_QUERIES)
        return TIP_SOURCE_TOPIC_QUERIES[idx]

    def _get_random_tip_source_guidelines(
        self, *, size: int, seed: int, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch a randomized pool of dietary guideline rules for tips/facts."""
        try:
            guidelines = self._search_tip_source_guidelines(
                size=size,
                seed=seed,
                query=query,
            )
        except Exception:
            logger.exception(
                "Topic guideline fetch failed for tips; falling back to random search."
            )
            try:
                guidelines = ELASTIC_CLIENT.random_search(
                    index_name=self.GUIDELINES_INDEX_NAME,
                    size=size,
                    seed=seed,
                    filter_query={
                        "bool": {
                            "must": [{"exists": {"field": "rule_text"}}],
                            "must_not": [{"term": {"status": "deleted"}}],
                        }
                    },
                )
            except Exception:
                logger.exception("Random guideline fetch failed for tips.")
                return []

        cleaned: List[Dict[str, Any]] = []
        seen_rules = set()
        for guideline in guidelines:
            if not isinstance(guideline, dict):
                continue
            rule_text = self._extract_guideline_rule_text(guideline)
            if len(rule_text) < TIP_SOURCE_GUIDELINE_MIN_RULE_CHARS:
                continue
            if self._mentions_animal_testing(rule_text):
                continue
            key = rule_text.lower()
            if key in seen_rules:
                continue
            seen_rules.add(key)
            cleaned.append(guideline)

        return cleaned[:size]

    def _search_tip_source_guidelines(
        self,
        *,
        size: int,
        seed: int,
        query: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Search a topical-but-randomized guideline pool for daily tips."""
        if not query:
            return ELASTIC_CLIENT.random_search(
                index_name=self.GUIDELINES_INDEX_NAME,
                size=size,
                seed=seed,
                filter_query={
                    "bool": {
                        "must": [{"exists": {"field": "rule_text"}}],
                        "must_not": [{"term": {"status": "deleted"}}],
                    }
                },
            )

        body = {
            "bool": {
                "must": [
                    {"exists": {"field": "rule_text"}},
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "rule_text^4",
                                "title^2",
                                "notes",
                                "food_groups^2",
                                "target_populations",
                                "guide_region",
                            ],
                            "type": "best_fields",
                        }
                    },
                ],
                "must_not": [{"term": {"status": "deleted"}}],
            }
        }
        return ELASTIC_CLIENT.random_search(
            index_name=self.GUIDELINES_INDEX_NAME,
            size=size,
            seed=seed,
            filter_query=body,
        )

    def _generate_tip_candidates_from_guidelines(
        self, *, guidelines: List[Dict[str, Any]], candidate_count: int
    ) -> List[Dict[str, Any]]:
        """Generate tip/fact candidates grounded in dietary guideline rule_text."""
        if not guidelines:
            return []

        guideline_context = self._prepare_tip_guideline_context(guidelines)
        prompt = f"""You create safe daily nutrition content for a general audience.

Using ONLY the dietary guideline rules below, generate exactly {candidate_count} items with a mix of:
- practical nutrition tips
- "Did you know?" nutrition facts

Safety rules:
- General education only (no diagnosis, treatment, medication, or disease-management advice).
- No supplement dosage guidance.
- Use the guideline rule_text as the source of truth.

Style rules:
- Each item must be <= 18 words.
- One sentence per item.
- Avoid absolute guarantees (no "cures", "prevents", "always", "never").

Return ONLY valid JSON in this exact format:
{{
  "items": [
    {{"kind": "tip", "text": "item text", "guideline": 1}},
    {{"kind": "did_you_know", "text": "item text", "guideline": 2}}
  ]
}}

Dietary guideline rules:
{guideline_context}
"""
        items: List[Any] = []
        try:
            response = self.simple_question_llm.invoke(prompt)
            parsed = self._parse_tip_candidates_json(response.content)
            parsed_items = parsed.get("items", [])
            if isinstance(parsed_items, list):
                items = parsed_items
        except Exception as e:
            logger.warning(
                "Guideline tip JSON generation failed; using rule_text fallback: %s",
                e,
            )

        valid: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            if len(valid) >= candidate_count:
                break
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "tip")).strip().lower()
            text = str(item.get("text", "")).strip()
            guideline_idx = self._coerce_source_index(
                item.get("guideline", item.get("article")),
                len(guidelines),
            )
            if kind not in ("tip", "did_you_know"):
                continue
            if not text or guideline_idx is None:
                continue
            if self._mentions_animal_testing(
                text
            ) or self._violates_tip_safety_policy(text):
                continue

            guideline = guidelines[guideline_idx - 1]
            rule_text = self._extract_guideline_rule_text(guideline)
            if not rule_text:
                continue

            key = f"{kind}:{text.lower()}"
            if key in seen:
                continue
            seen.add(key)

            valid.append(
                {
                    "kind": kind,
                    "text": text,
                    "urn": self._guideline_source_urn(guideline),
                    "passage": self._extract_guideline_passage(guideline),
                    "title": guideline.get("title", None),
                    "publication_year": self._guideline_publication_year(guideline),
                }
            )

        if len(valid) < candidate_count:
            for fallback in self._generate_rule_based_tip_candidates_from_guidelines(
                guidelines,
                candidate_count=candidate_count - len(valid),
            ):
                key = (
                    f"{fallback.get('kind')}:"
                    f"{str(fallback.get('text', '')).lower()}"
                )
                if key in seen:
                    continue
                valid.append(fallback)
                seen.add(key)
                if len(valid) >= candidate_count:
                    break

        return valid

    def _prepare_tip_guideline_context(
        self, guidelines: List[Dict[str, Any]]
    ) -> str:
        """Prepare compact guideline rule context for tip generation."""
        snippets = []
        for idx, guideline in enumerate(guidelines, 1):
            rule_text = self._extract_guideline_rule_text(guideline)
            food_groups = guideline.get("food_groups") or []
            target_populations = guideline.get("target_populations") or []
            if isinstance(food_groups, list):
                food_groups_text = ", ".join(str(x) for x in food_groups if x)
            else:
                food_groups_text = str(food_groups)
            if isinstance(target_populations, list):
                populations_text = ", ".join(str(x) for x in target_populations if x)
            else:
                populations_text = str(target_populations)
            snippet = (
                f"Guideline {idx}:\n"
                f"- Title: {guideline.get('title', 'N/A')}\n"
                f"- Guide: {guideline.get('guide_urn', 'N/A')}\n"
                f"- Region: {guideline.get('guide_region', 'N/A')}\n"
                f"- Food groups: {food_groups_text or 'N/A'}\n"
                f"- Target populations: {populations_text or 'N/A'}\n"
                f"- Rule text: {rule_text[:500]}"
            )
            snippets.append(snippet)
        return "\n\n".join(snippets)

    def _generate_rule_based_tip_candidates_from_guidelines(
        self, guidelines: List[Dict[str, Any]], candidate_count: int
    ) -> List[Dict[str, Any]]:
        """Create safe guideline-backed items without relying on model JSON."""
        candidates: List[Dict[str, Any]] = []
        seen = set()

        for guideline in guidelines:
            if len(candidates) >= candidate_count:
                break
            rule_text = self._extract_guideline_rule_text(guideline)
            if not rule_text:
                continue

            source_payload = {
                "urn": self._guideline_source_urn(guideline),
                "passage": self._extract_guideline_passage(guideline),
                "title": guideline.get("title", None),
                "publication_year": self._guideline_publication_year(guideline),
            }
            for kind, text in (
                ("tip", self._guideline_rule_to_tip(rule_text)),
                ("did_you_know", self._guideline_rule_to_fact(rule_text)),
            ):
                if len(candidates) >= candidate_count:
                    break
                if not text or not self._is_tip_line_appropriate(text):
                    continue
                key = f"{kind}:{text.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"kind": kind, "text": text, **source_payload})

        return candidates[:candidate_count]

    def _extract_guideline_rule_text(self, guideline: Dict[str, Any]) -> str:
        """Extract normalized rule_text from a guideline document."""
        rule_text = guideline.get("rule_text") or ""
        if not isinstance(rule_text, str):
            rule_text = str(rule_text)
        return " ".join(rule_text.split()).strip()

    def _extract_guideline_passage(self, guideline: Dict[str, Any]) -> str:
        """Use guideline rule_text itself as the grounding passage."""
        rule_text = self._extract_guideline_rule_text(guideline)
        return rule_text[:_TIP_PASSAGE_MAX_CHARS].strip()

    def _guideline_source_urn(self, guideline: Dict[str, Any]) -> str:
        """Return the best stable identifier available for a guideline source."""
        for key in ("guide_urn", "urn", "id", "_id"):
            value = guideline.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        rule_text = self._extract_guideline_rule_text(guideline)
        if rule_text:
            return f"guideline:{uuid.uuid5(uuid.NAMESPACE_URL, rule_text)}"
        return "guidelines"

    def _guideline_document_urn(self, guideline: Dict[str, Any]) -> str:
        """Return a stable identifier for an individual guideline rule."""
        for key in ("id", "_id", "urn", "guide_urn"):
            value = guideline.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        rule_text = self._extract_guideline_rule_text(guideline)
        if rule_text:
            return f"guideline:{uuid.uuid5(uuid.NAMESPACE_URL, rule_text)}"
        return "guideline"

    def _guideline_tags(self, guideline: Dict[str, Any]) -> List[str]:
        """Collect guideline keyword metadata into displayable tags."""
        tags: List[str] = []
        for key in ("food_groups", "target_populations"):
            value = guideline.get(key)
            if isinstance(value, list):
                tags.extend(str(item) for item in value if item)
            elif isinstance(value, str) and value.strip():
                tags.append(value.strip())
        region = guideline.get("guide_region")
        if isinstance(region, str) and region.strip():
            tags.append(region.strip())
        return tags

    def _guideline_publication_year(self, guideline: Dict[str, Any]) -> Optional[str]:
        """Best-effort year-like metadata for guideline evidence."""
        for key in ("applicability_start_date", "created_at", "updated_at"):
            value = guideline.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _guideline_rule_to_tip(self, rule_text: str) -> str:
        """Turn a guideline rule into a concise actionable tip."""
        text = self._clean_guideline_generated_text(rule_text)
        lowered = text.lower()
        if lowered.startswith(
            ("people should ", "adults should ", "children should ")
        ):
            text = re.sub(
                r"^(people|adults|children)\s+should\s+",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = f"Aim to {text[0].lower()}{text[1:]}" if text else ""
        elif lowered.startswith("it is recommended that "):
            text = text[len("it is recommended that "):]
            text = f"Aim to {text[0].lower()}{text[1:]}" if text else ""
        elif not lowered.startswith(
            (
                "add ",
                "aim ",
                "avoid ",
                "choose ",
                "drink ",
                "eat ",
                "have ",
                "include ",
                "keep ",
                "limit ",
                "make ",
                "prefer ",
                "reduce ",
                "replace ",
                "select ",
                "use ",
            )
        ):
            text = f"Keep in mind that {text[0].lower()}{text[1:]}" if text else ""
        return self._shorten_tip_sentence(text, max_words=18)

    def _guideline_rule_to_fact(self, rule_text: str) -> str:
        """Turn a guideline rule into a concise did-you-know style fact."""
        text = self._clean_guideline_generated_text(rule_text)
        return self._shorten_tip_sentence(text, max_words=18)

    def _clean_guideline_generated_text(self, text: str) -> str:
        """Clean source text before deterministic tip/fact formatting."""
        text = " ".join(str(text).split()).strip(" -:;")
        text = re.sub(
            r"^(guideline|recommendation)\s*[:\-]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text

    def _shorten_tip_sentence(self, text: str, *, max_words: int) -> str:
        """Limit generated fallback text to a short single sentence."""
        text = " ".join(str(text).split()).strip()
        if not text:
            return ""

        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        words = sentence.split()
        if len(words) > max_words:
            sentence = " ".join(words[:max_words]).rstrip(" ,;:")
        sentence = sentence.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    def _coerce_source_index(self, value: Any, max_index: int) -> Optional[int]:
        """Coerce model source references like 1 or '1' into a valid index."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            idx = value
        elif isinstance(value, str) and value.strip().isdigit():
            idx = int(value.strip())
        else:
            return None
        if 1 <= idx <= max_index:
            return idx
        return None

    def _get_random_tip_source_articles(
        self, *, size: int, seed: int, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch a randomized pool of source articles for tips/facts."""
        try:
            articles = self._search_tip_source_articles(
                size=size,
                seed=seed,
                query=query,
            )
        except Exception:
            logger.exception(
                "Topic article fetch failed for tips; falling back to random search."
            )
            try:
                articles = ELASTIC_CLIENT.random_search(
                    index_name=self.INDEX_NAME,
                    size=size,
                    seed=seed,
                    filter_query={
                        "bool": {"must_not": [{"term": {"status": "deleted"}}]}
                    },
                    source_excludes=["embedding"],
                )
            except Exception:
                logger.exception("Random article fetch failed for tips.")
                return []

        cleaned: List[Dict[str, Any]] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            abstract = a.get("abstract") or a.get("description") or ""
            if not isinstance(abstract, str):
                abstract = str(abstract)
            abstract = abstract.strip()
            if len(abstract) < TIP_SOURCE_ARTICLE_MIN_ABSTRACT_CHARS:
                continue
            if self._mentions_animal_testing(abstract) or self._mentions_animal_testing(
                a.get("title", "") or ""
            ):
                continue
            cleaned.append(a)

        return cleaned[:size]

    def _search_tip_source_articles(
        self,
        *,
        size: int,
        seed: int,
        query: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Search a topical-but-randomized article pool for daily tips."""
        if not query:
            return ELASTIC_CLIENT.random_search(
                index_name=self.INDEX_NAME,
                size=size,
                seed=seed,
                filter_query={
                    "bool": {"must_not": [{"term": {"status": "deleted"}}]}
                },
                source_excludes=["embedding"],
            )

        body = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "abstract^2",
                                "description^2",
                                "tags",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
                "must_not": [{"term": {"status": "deleted"}}],
            }
        }
        return ELASTIC_CLIENT.random_search(
            index_name=self.INDEX_NAME,
            size=size,
            seed=seed,
            filter_query=body,
            source_excludes=["embedding"],
        )

    def _generate_tip_candidates_from_articles(
        self, *, articles: List[Dict[str, Any]], candidate_count: int
    ) -> List[Dict[str, Any]]:
        """Generate tip/fact candidates grounded in a randomized article pool."""
        if not articles:
            return []

        article_context = self._prepare_tip_article_context(articles)
        prompt = f"""You create safe daily nutrition content for a general audience.

Using ONLY the evidence in the article abstracts below, generate exactly {candidate_count} items with a mix of:
- practical nutrition tips
- "Did you know?" nutrition facts

Safety rules:
- General education only (no diagnosis, treatment, medication, or disease-management advice).
- No supplement dosage guidance.
- Do not mention animals, animal studies, mice, rats, or preclinical models.
- If an article is animal/preclinical-only or unclear, do NOT use it.

Style rules:
- Each item must be <= 18 words.
- One sentence per item.
- Avoid absolute guarantees (no "cures", "prevents", "always", "never").

Return ONLY valid JSON in this exact format:
{{
  "items": [
    {{"kind": "tip", "text": "item text", "article": 1}},
    {{"kind": "did_you_know", "text": "item text", "article": 2}}
  ]
}}

Evidence:
{article_context}
"""
        response = self.simple_question_llm.invoke(prompt)
        parsed = self._parse_tip_candidates_json(response.content)
        items = parsed.get("items", [])
        if not isinstance(items, list):
            return []

        valid: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "tip")).strip().lower()
            text = str(item.get("text", "")).strip()
            article_idx = self._coerce_source_index(
                item.get("article", None),
                len(articles),
            )
            if kind not in ("tip", "did_you_know"):
                continue
            if not text:
                continue
            if article_idx is None:
                continue
            if self._mentions_animal_testing(
                text
            ) or self._violates_tip_safety_policy(text):
                continue
            article = articles[article_idx - 1]
            urn = str(article.get("urn", article.get("_id", "")) or "").strip()
            passage = self._extract_tip_passage(article, text)
            if not urn or not passage:
                continue
            valid.append(
                {
                    "kind": kind,
                    "text": text,
                    "urn": urn,
                    "passage": passage,
                    "title": article.get("title", None),
                    "publication_year": article.get("publication_year", None),
                }
            )

        return valid

    def _extract_tip_passage(self, article: Dict[str, Any], tip_text: str) -> str:
        """Extract a short, best-matching passage from an article abstract/description."""
        abstract = article.get("abstract") or article.get("description") or ""
        if not isinstance(abstract, str):
            abstract = str(abstract)
        abstract = " ".join(abstract.split()).strip()
        if not abstract:
            return ""

        # Split into sentence-like chunks.
        chunks = re.split(r"(?<=[.!?])\s+", abstract)
        chunks = [c.strip() for c in chunks if c and c.strip()]
        if not chunks:
            chunks = [abstract]

        # Token overlap scoring (lightweight).
        stop = {
            "the", "and", "with", "from", "that", "this", "these", "those", "into", "onto",
            "are", "was", "were", "been", "being", "for", "to", "of", "in", "on", "at", "as",
            "a", "an", "or", "by", "it", "its", "their", "they", "them", "we", "you", "your",
        }

        def tokens(s: str) -> set[str]:
            return {
                t
                for t in re.findall(r"[a-zA-Z]{3,}", s.lower())
                if t not in stop
            }

        tip_tokens = tokens(tip_text)
        best = chunks[0]
        best_score = -1
        for c in chunks:
            c_tokens = tokens(c)
            if not c_tokens:
                continue
            score = len(tip_tokens & c_tokens) if tip_tokens else 0
            if score > best_score:
                best_score = score
                best = c

        passage = best
        if len(passage) < _TIP_PASSAGE_MIN_CHARS:
            passage = abstract[:_TIP_PASSAGE_MAX_CHARS]
        passage = passage[:_TIP_PASSAGE_MAX_CHARS].strip()
        return passage

    def _ground_tip_candidates(
        self, candidates: List[Dict[str, Any]], count: int
    ) -> List[str]:
        """Retrieve evidence per candidate and produce grounded safe tips."""
        grounded: List[str] = []
        seen = set()

        for candidate in candidates:
            if len(grounded) >= count:
                break

            if not isinstance(candidate, dict):
                continue

            kind = str(candidate.get("kind", "tip")).strip().lower()
            text = str(candidate.get("text", "")).strip()
            if not text:
                continue

            articles, _ = self._retrieve_articles(text, TIP_GROUNDING_TOP_K)
            if not articles:
                continue

            human_articles = self._filter_human_relevant_articles(articles)
            if not human_articles:
                continue

            grounded_text = self._ground_single_tip(
                kind=kind,
                text=text,
                articles=human_articles,
            )
            if not grounded_text:
                continue

            if self._violates_tip_safety_policy(grounded_text):
                continue
            if self._mentions_animal_testing(grounded_text):
                continue

            key = grounded_text.lower()
            if key in seen:
                continue

            grounded.append(grounded_text)
            seen.add(key)

        return grounded

    def _ground_single_tip(
        self, kind: str, text: str, articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Ground a single tip/fact using retrieved article evidence."""
        style = "Did you know?" if kind == "did_you_know" else "Tip:"
        article_context = self._prepare_tip_article_context(articles)
        prompt = f"""Rewrite the item below as one short, evidence-grounded nutrition line.

Candidate item: {text}
Style requirement: start with "{style}"

Only use the evidence provided in article abstracts.
If evidence is weak or unclear, return exactly: INSUFFICIENT_EVIDENCE

Safety rules:
- Use evidence from human studies only.
- Exclude animal-model or preclinical-only findings.
- Do not mention animals, animal studies, mice, rats, or rodent models.
- No diagnosis or treatment advice.
- No medication or supplement dosage guidance.
- No promises of curing or preventing disease.

Output rules:
- Single line only.
- Max 22 words.
- No citations or extra text.

Evidence:
{article_context}
"""
        try:
            response = self.simple_question_llm.invoke(prompt)
            output = response.content.strip()
            if "insufficient_evidence" in output.lower():
                return None

            if "```" in output:
                output = output.replace("```", "").strip()

            # Force single-line output.
            output = " ".join(output.split())

            if style == "Did you know?" and not output.lower().startswith("did you know?"):
                output = f"Did you know? {output.lstrip(':').strip()}"
            if style == "Tip:" and not output.lower().startswith("tip:"):
                output = f"Tip: {output.lstrip(':').strip()}"

            return output
        except Exception as e:
            logger.error("Error grounding tip candidate: %s", e, exc_info=True)
            return None

    def _prepare_tip_article_context(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare compact article abstract context for tip grounding."""
        snippets = []
        for idx, article in enumerate(articles, 1):
            abstract = (
                article.get("abstract")
                or article.get("description")
                or "No abstract available"
            )
            snippet = (
                f"Article {idx}:\n"
                f"- Title: {article.get('title', 'N/A')}\n"
                f"- Year: {article.get('publication_year', 'N/A')}\n"
                f"- Abstract: {abstract[:700]}"
            )
            snippets.append(snippet)
        return "\n\n".join(snippets)

    def _filter_human_relevant_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep articles that indicate human evidence and exclude animal-only evidence."""
        return [a for a in articles if self._is_human_relevant_article(a)]

    def _is_human_relevant_article(self, article: Dict[str, Any]) -> bool:
        """Heuristic check for human-focused evidence."""
        text_parts = [
            article.get("title", ""),
            article.get("abstract", ""),
            article.get("description", ""),
            article.get("ai_category", ""),
            " ".join(article.get("tags", []) or []),
        ]
        text = " ".join(str(p) for p in text_parts if p).lower()
        if not text:
            return False

        has_human = any(p.search(text) for p in _HUMAN_TERM_PATTERNS)
        has_animal = any(p.search(text) for p in _ANIMAL_TERM_PATTERNS)

        if has_animal:
            return False
        return has_human

    def _mentions_animal_testing(self, text: str) -> bool:
        """Guardrail for animal-testing phrasing in output."""
        normalized = str(text).lower()
        return any(p.search(normalized) for p in _ANIMAL_TERM_PATTERNS)

    def _violates_tip_safety_policy(self, text: str) -> bool:
        """Basic guardrail against medical advice in tips."""
        normalized = text.lower()
        blocked_terms = [
            "diagnose",
            "diagnosis",
            "treat",
            "treatment",
            "cure",
            "prevent disease",
            "medication",
            "prescription",
            "dose",
            "dosage",
            "mg ",
            "iu ",
            "hypertension",
            "diabetes management",
            "cancer",
        ]
        if any(term in normalized for term in blocked_terms):
            return True

        # Catch explicit supplement-dose patterns (e.g. "take 500mg") without
        # blocking common food/nutrition quantities in general tips.
        has_dose_unit = re.search(r"\b\d+\s?(mg|mcg|μg|iu)\b", normalized) is not None
        mentions_supplement = any(
            term in normalized
            for term in ("supplement", "capsule", "pill", "tablet", "take ")
        )
        if has_dose_unit and mentions_supplement:
            return True

        return False

    def _format_tips_payload(
        self,
        items: List[str],
        tips_count: int,
        did_you_know_count: int,
    ) -> Dict[str, List[str]]:
        """Split mixed lines into did_you_know and tips lists."""
        did_you_know, tips = self._split_tip_items(
            items,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )
        return {"did_you_know": did_you_know, "tips": tips}

    def _is_tips_payload_appropriate(
        self,
        payload: Dict[str, Any],
        tips_count: int,
        did_you_know_count: int,
    ) -> bool:
        """Validate final tip payload before returning/caching it."""
        did_you_know = payload.get("did_you_know", [])
        tips = payload.get("tips", [])

        if not isinstance(did_you_know, list) or not isinstance(tips, list):
            return False

        if len(did_you_know) < did_you_know_count or len(tips) < tips_count:
            return False

        for item in did_you_know + tips:
            if not self._is_tip_line_appropriate(item):
                return False

        return True

    def _is_tip_line_appropriate(self, text: str) -> bool:
        """Validate one line against animal and medical guardrails."""
        if not isinstance(text, str):
            return False
        normalized = text.strip()
        if not normalized:
            return False
        if self._mentions_animal_testing(normalized):
            return False
        if self._violates_tip_safety_policy(normalized):
            return False
        return True

    def _normalize_tips_payload(
        self,
        payload: Dict[str, Any],
        tips_count: int,
        did_you_know_count: int,
        *,
        fill_with_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Normalize cached/new payloads to did_you_know + tips shape."""
        # If detailed items exist, prefer them as the source of truth.
        details_dyk = payload.get("did_you_know_detail", [])
        details_tips = payload.get("tips_detail", [])
        if isinstance(details_dyk, list) and isinstance(details_tips, list) and (
            details_dyk or details_tips
        ):
            did_you_know = []
            tips = []
            for item in details_dyk:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    item["text"] = self._naturalize_tip_detail_text(
                        kind="did_you_know",
                        text=item["text"],
                        evidence=item.get("evidence"),
                    )
                    did_you_know.append(item["text"].strip())
            for item in details_tips:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    item["text"] = self._naturalize_tip_detail_text(
                        kind="tip",
                        text=item["text"],
                        evidence=item.get("evidence"),
                    )
                    tips.append(item["text"].strip())
            payload["did_you_know"] = did_you_know[:did_you_know_count]
            payload["tips"] = tips[:tips_count]
            return payload

        payload["did_you_know_detail"] = details_dyk
        payload["tips_detail"] = details_tips

        raw_did_you_know = payload.get("did_you_know", [])
        raw_tips = payload.get("tips", [])

        if not isinstance(raw_did_you_know, list):
            raw_did_you_know = []
        if not isinstance(raw_tips, list):
            raw_tips = []

        combined_items: List[str] = []
        for item in raw_did_you_know:
            if isinstance(item, str) and item.strip():
                combined_items.append(f"Did you know? {item.strip()}")
        for item in raw_tips:
            if isinstance(item, str) and item.strip():
                combined_items.append(item.strip())

        did_you_know, tips = self._split_tip_items(
            combined_items,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
            fill_with_fallback=fill_with_fallback,
        )
        payload["did_you_know"] = did_you_know
        payload["tips"] = tips
        return payload

    def _naturalize_tip_detail_text(
        self,
        *,
        kind: str,
        text: str,
        evidence: Any,
    ) -> str:
        """Clean old mechanical tip/fact wording while leaving evidence intact."""
        source_text = str(text or "").strip()
        passage = None
        if isinstance(evidence, dict):
            raw_passage = evidence.get("passage")
            if isinstance(raw_passage, str) and raw_passage.strip():
                passage = raw_passage.strip()

        lowered = source_text.lower()
        mechanical_prefixes = (
            "dietary guides advise:",
            "use dietary guide advice:",
            "did you know?",
            "tip:",
        )
        if passage and lowered.startswith(mechanical_prefixes):
            source_text = passage
        else:
            for prefix in mechanical_prefixes:
                if lowered.startswith(prefix):
                    source_text = source_text[len(prefix):].strip(" :")
                    break

        if kind == "did_you_know":
            return self._guideline_rule_to_fact(source_text)
        return self._guideline_rule_to_tip(source_text)

    def _normalize_tip_details(
        self,
        payload: Dict[str, Any],
        *,
        tips_count: int,
        did_you_know_count: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Normalize `*_detail` arrays, and synthesize them for fallback-only payloads."""
        did_you_know_detail = payload.get("did_you_know_detail", [])
        tips_detail = payload.get("tips_detail", [])

        def _coerce(items: Any, max_count: int) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            out: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = it.get("text")
                evidence = it.get("evidence", None)
                if not isinstance(text, str) or not text.strip():
                    continue
                if evidence is not None and not isinstance(evidence, dict):
                    evidence = None
                out.append({"text": text.strip(), "evidence": evidence})
                if len(out) >= max_count:
                    break
            return out

        did_you_know_detail = _coerce(did_you_know_detail, did_you_know_count)
        tips_detail = _coerce(tips_detail, tips_count)

        # If we don't have detailed evidence (e.g. fallback tips), synthesize entries
        # with `evidence=None` so clients have a consistent shape.
        if len(did_you_know_detail) < did_you_know_count:
            for text in payload.get("did_you_know", [])[:did_you_know_count]:
                if len(did_you_know_detail) >= did_you_know_count:
                    break
                if isinstance(text, str) and text.strip():
                    did_you_know_detail.append({"text": text.strip(), "evidence": None})
        if len(tips_detail) < tips_count:
            for text in payload.get("tips", [])[:tips_count]:
                if len(tips_detail) >= tips_count:
                    break
                if isinstance(text, str) and text.strip():
                    tips_detail.append({"text": text.strip(), "evidence": None})

        return did_you_know_detail[:did_you_know_count], tips_detail[:tips_count]

    def _split_tip_items(
        self,
        items: List[str],
        tips_count: int,
        did_you_know_count: int,
        *,
        fill_with_fallback: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Classify mixed tip strings into fact and tip arrays."""
        did_you_know: List[str] = []
        tips: List[str] = []
        seen = set()

        for item in items:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split()).strip()
            if not text:
                continue

            lower_text = text.lower()
            if lower_text.startswith("did you know?"):
                kind = "did_you_know"
                value = text[len("did you know?"):].strip(" :")
            elif lower_text.startswith("tip:"):
                kind = "tip"
                value = text[len("tip:"):].strip(" :")
            else:
                kind = "tip"
                value = text

            if not value:
                continue
            if not self._is_tip_line_appropriate(value):
                continue

            key = f"{kind}:{value.lower()}"
            if key in seen:
                continue
            seen.add(key)

            if kind == "did_you_know" and len(did_you_know) < did_you_know_count:
                did_you_know.append(value)
            elif kind == "tip" and len(tips) < tips_count:
                tips.append(value)

            if (
                len(did_you_know) >= did_you_know_count
                and len(tips) >= tips_count
            ):
                break

        if fill_with_fallback:
            missing_did_you_know = max(did_you_know_count - len(did_you_know), 0)
            missing_tips = max(tips_count - len(tips), 0)
            if missing_did_you_know > 0 or missing_tips > 0:
                fallback_items = self._generate_fallback_tips(
                    tips_count=missing_tips,
                    did_you_know_count=missing_did_you_know,
                )
                for fallback_item in fallback_items:
                    if not isinstance(fallback_item, str):
                        continue
                    text = " ".join(fallback_item.split()).strip()
                    if not text:
                        continue
                    lower_text = text.lower()
                    if lower_text.startswith("did you know?"):
                        value = text[len("did you know?"):].strip(" :")
                        key = f"did_you_know:{value.lower()}"
                        if (
                            value
                            and key not in seen
                            and len(did_you_know) < did_you_know_count
                        ):
                            did_you_know.append(value)
                            seen.add(key)
                    elif lower_text.startswith("tip:"):
                        value = text[len("tip:"):].strip(" :")
                        key = f"tip:{value.lower()}"
                        if value and key not in seen and len(tips) < tips_count:
                            tips.append(value)
                            seen.add(key)
                    if (
                        len(did_you_know) >= did_you_know_count
                        and len(tips) >= tips_count
                    ):
                        break

        return did_you_know, tips

    def _parse_questions_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON question payload from model output."""
        text = content.strip()

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        try:
            return json.loads(text)
        except Exception:
            # Common model failure mode: trailing commas.
            fixed = re.sub(r",\s*([}\]])", r"\1", text)
            if fixed != text:
                return json.loads(fixed)
            if json5 is None:
                raise
            return json5.loads(text)

    def _parse_tip_candidates_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON tip candidate payload from model output."""
        text = content.strip()
        if not text:
            return {"items": []}

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        candidates = [text]
        extracted = self._extract_first_json_object(text)
        if extracted and extracted != text:
            candidates.append(extracted)

        last_error: Optional[Exception] = None
        for candidate in candidates:
            if not candidate.strip():
                continue
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            for variant in (candidate, fixed):
                try:
                    return json.loads(variant)
                except Exception as e:
                    last_error = e
                if json5 is not None:
                    try:
                        return json5.loads(variant)
                    except Exception as e:
                        last_error = e

        logger.warning(
            "Unable to parse tip candidate JSON; using empty candidates: %s",
            last_error,
        )
        return {"items": []}

    def _extract_first_json_object(self, text: str) -> str:
        """Extract the first balanced JSON object from text when possible."""
        start = text.find("{")
        if start < 0:
            return text

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return text[start:]

    def _generate_fallback_questions(self, count: int) -> List[str]:
        """Emergency fallback without a fixed question pool."""
        pool = [
            "How does dietary fiber support gut health?",
            "What is the difference between soluble and insoluble fiber?",
            "How do whole grains differ nutritionally from refined grains?",
            "What factors affect how quickly blood sugar rises after meals?",
            "How does protein intake influence satiety?",
            "What nutrients are commonly low in highly processed diets?",
            "How does sodium intake relate to long-term cardiovascular health?",
            "What is the role of omega-3 fats in nutrition science?",
            "How does fermentation change the nutritional profile of foods?",
            "What does current evidence say about ultra-processed foods?",
        ]
        random.shuffle(pool)
        selected = self._normalize_simple_questions(pool, count=count)

        while len(selected) < count:
            idx = len(selected) + 1
            selected.append(
                f"What does current evidence show about nutrient bioavailability in foods? ({idx})"
            )

        return selected[:count]

    def _generate_fallback_tips(
        self, tips_count: int, did_you_know_count: int
    ) -> List[str]:
        """Emergency fallback tip generator."""
        tip_actions = [
            "Tip: Make vegetables or fruit a visible part of meals.",
            "Tip: Choose whole-grain breads, rice, or pasta more often.",
            "Tip: Make water your usual drink with meals.",
            "Tip: Include beans, lentils, fish, eggs, or lean meats for protein.",
        ]
        did_you_know_facts = [
            "Did you know? Dietary guides commonly emphasize vegetables, fruit, grains, and protein foods.",
            "Did you know? Many food guides recommend choosing mostly unsaturated oils and spreads.",
            "Did you know? Whole-grain foods are a recurring theme across dietary guides.",
            "Did you know? Food guides often recommend limiting sugary drinks.",
        ]

        random.shuffle(tip_actions)
        random.shuffle(did_you_know_facts)

        selected_tips = tip_actions[:tips_count]
        while len(selected_tips) < tips_count:
            idx = len(selected_tips) + 1
            selected_tips.append(
                f"Tip: Build one balanced plate with protein, fiber, and color. ({idx})"
            )

        selected_did_you_know = did_you_know_facts[:did_you_know_count]
        while len(selected_did_you_know) < did_you_know_count:
            idx = len(selected_did_you_know) + 1
            selected_did_you_know.append(
                f"Did you know? Whole foods often deliver multiple nutrients together. ({idx})"
            )

        mixed = selected_tips + selected_did_you_know
        random.shuffle(mixed)
        return mixed

    def _validate_request(self, request: QARequest) -> None:
        """Validate request parameters."""
        if request.mode == "advanced" and request.model:
            if request.model not in AVAILABLE_GROQ_MODELS:
                raise InvalidError(
                    detail=f"Model '{request.model}' is not available. "
                    f"Choose from: {AVAILABLE_GROQ_MODELS}",
                    extra={"available_models": AVAILABLE_GROQ_MODELS},
                )

    def _resolve_model(self, request: QARequest) -> str:
        """Determine the effective model to use."""
        if request.mode == "advanced" and request.model:
            return request.model
        return DEFAULT_GROQ_MODEL

    def _build_cache_key(
        self,
        request: QARequest,
        *,
        question: Optional[str] = None,
        effective_retriever: Optional[str] = None,
        user_context: Optional[QAUserContext] = None,
    ) -> str:
        """Generate cache key from question + effective settings."""
        effective_question = question or request.question
        retriever = effective_retriever or self._resolve_retriever(request)
        context = user_context.model_dump(exclude_none=True) if user_context else {}
        return self.cache_manager.generate_cache_key(
            prefix="qa",
            data={
                "version": 3,
                "question": effective_question.strip().lower(),
                "mode": request.mode,
                "model": self._resolve_model(request),
                "rag_enabled": retriever != "no_rag",
                "retriever": retriever,
                "top_k": request.top_k,
                "expertise_level": request.expertise_level,
                "experience_group": request.experience_group,
                "language": request.language,
                "user_context": context,
            },
        )

    def _should_generate_dual_answer(self, question: str) -> bool:
        """Decide whether to generate a dual answer for A/B testing."""
        question_lower = question.lower()

        ambiguity_matches = sum(
            1 for kw in AMBIGUITY_KEYWORDS if kw in question_lower
        )

        if ambiguity_matches >= 2:
            probability = 0.25
        elif ambiguity_matches >= 1:
            probability = 0.20
        else:
            probability = 0.15

        return random.random() < probability

    def _select_dual_strategy(self) -> Tuple[str, Dict, Dict]:
        """Select a random A/B testing strategy."""
        return random.choice(DUAL_ANSWER_STRATEGIES)

    def _generate_secondary_answer(
        self,
        request: QARequest,
        articles: List[Dict[str, Any]],
        request_id: str,
        *,
        question: Optional[str] = None,
        retriever: str = "rag",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[QAAnswer, DualAnswerFeedback, str]:
        """Generate a secondary answer using a different configuration.

        Returns:
            Tuple of (secondary_answer, dual_feedback, strategy_name)
        """
        strategy_name, primary_cfg, secondary_cfg = self._select_dual_strategy()

        secondary_model = secondary_cfg.get("model", DEFAULT_GROQ_MODEL)
        secondary_temp = secondary_cfg.get("temperature", 0.3)
        secondary_top_k = secondary_cfg.get("top_k", request.top_k)
        answer_question = question or request.question
        retrieval_user_context = (
            QAUserContext(**user_context) if isinstance(user_context, dict) else None
        )

        logger.info(
            "Generating dual answer (strategy=%s) for request %s",
            strategy_name, request_id,
        )

        from agents.qa_agent import QAAgent

        secondary_agent = QAAgent(
            model=secondary_model, temperature=secondary_temp
        )

        # If strategy varies top_k, re-retrieve articles
        secondary_articles = articles
        if (
            retriever != "no_rag"
            and "top_k" in secondary_cfg
            and secondary_cfg["top_k"] != request.top_k
        ):
            secondary_articles, _ = self._retrieve_articles(
                answer_question,
                secondary_cfg["top_k"],
                retriever=retriever,
                user_context=retrieval_user_context,
            )

        if secondary_articles:
            secondary_answer, _ = secondary_agent.generate_answer_with_rag(
                question=answer_question,
                articles=secondary_articles,
                expertise_level=request.expertise_level,
                language=request.language,
                retriever=retriever,
                user_context=user_context,
            )
        else:
            secondary_answer, _ = secondary_agent.generate_answer_without_rag(
                question=answer_question,
                expertise_level=request.expertise_level,
                language=request.language,
                user_context=user_context,
            )

        primary_label = (
            f"model:{DEFAULT_GROQ_MODEL}, temp:0.3, top_k:{request.top_k}"
        )
        secondary_label = (
            f"model:{secondary_model}, temp:{secondary_temp}, "
            f"top_k:{secondary_top_k}"
        )

        feedback = DualAnswerFeedback(
            request_id=request_id,
            answer_a_label=primary_label,
            answer_b_label=secondary_label,
        )

        return secondary_answer, feedback, strategy_name

    async def submit_feedback(
        self, feedback: QAFeedbackRequest
    ) -> QAFeedbackResponse:
        """Store user feedback to PostgreSQL."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import QAFeedbackRecord

        feedback_mode = (
            "ab_preference" if feedback.preferred_answer is not None else "general"
        )
        try:
            factory = POSTGRES_ASYNC_SESSION_FACTORY()
            async with factory() as session:
                record = QAFeedbackRecord(
                    request_id=uuid.UUID(feedback.request_id),
                    preferred_answer=feedback.preferred_answer,
                    helpfulness=feedback.helpfulness,
                    target_answer=feedback.target_answer,
                    feedback_mode=feedback_mode,
                    reason=feedback.reason,
                )
                session.add(record)
                await session.commit()
                logger.info(
                    "Recorded QA feedback for request %s: mode=%s, preference=%s, helpfulness=%s, target=%s",
                    feedback.request_id,
                    feedback_mode,
                    feedback.preferred_answer,
                    feedback.helpfulness,
                    feedback.target_answer,
                )
        except Exception as e:
            logger.error(
                "Failed to persist QA feedback for request %s: %s",
                feedback.request_id, e, exc_info=True,
            )

        return QAFeedbackResponse(
            request_id=feedback.request_id,
            status="recorded",
            message="Thank you for your feedback.",
        )
