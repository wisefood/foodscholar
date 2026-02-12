"""Q&A service for non-contextual question answering."""
import logging
import random
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from models.qa import (
    QARequest,
    QAResponse,
    QAAnswer,
    RetrievedArticle,
    DualAnswerFeedback,
    QAFeedbackRequest,
    QAFeedbackResponse,
    AVAILABLE_GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
)
from agents.qa_agent import QAAgent
from backend.elastic import ELASTIC_CLIENT
from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
from models.db import QARequestRecord, QAFeedbackRecord
from utilities.cache import CacheManager
from exceptions import InvalidError

logger = logging.getLogger(__name__)

TTL_QA_RESPONSE = 86400  # 1 day
TTL_QA_FEEDBACK = 2592000  # 30 days

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

    def __init__(self, cache_enabled: bool = True):
        self.cache_manager = CacheManager(enabled=cache_enabled)
        self._embedder = None

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

    async def answer_question(self, request: QARequest) -> QAResponse:
        """
        Main orchestration: validate, retrieve, generate, optionally dual-answer.

        Flow:
        1. Validate request (model selection in advanced mode)
        2. Check cache
        3. Embed query and kNN search (if RAG enabled)
        4. Generate primary answer via QAAgent
        5. Optionally generate secondary answer (dual-answer A/B)
        6. Cache result
        7. Return QAResponse
        """
        request_id = str(uuid.uuid4())

        self._validate_request(request)

        effective_model = self._resolve_model(request)
        effective_rag = request.rag_enabled if request.mode == "advanced" else True

        # Cache check
        cache_key = self._build_cache_key(request)
        cached = self.cache_manager.get(cache_key)
        if cached:
            logger.info("Cache hit for QA: %s...", request.question[:50])
            cached["cache_hit"] = True
            cached["request_id"] = request_id
            return QAResponse(**cached)

        # Retrieval
        articles: List[Dict[str, Any]] = []
        retrieved_articles: List[RetrievedArticle] = []
        if effective_rag:
            articles, retrieved_articles = self._retrieve_articles(
                request.question, request.top_k
            )

        # Primary answer
        primary_agent = QAAgent(model=effective_model, temperature=0.3)
        if effective_rag and articles:
            primary_answer, follow_ups = primary_agent.generate_answer_with_rag(
                question=request.question,
                articles=articles,
                expertise_level=request.expertise_level,
                language=request.language,
            )
        else:
            primary_answer, follow_ups = primary_agent.generate_answer_without_rag(
                question=request.question,
                expertise_level=request.expertise_level,
                language=request.language,
            )

        # Dual-answer logic (simple mode only)
        secondary_answer: Optional[QAAnswer] = None
        dual_feedback: Optional[DualAnswerFeedback] = None
        dual_strategy: Optional[str] = None
        if request.mode == "simple" and self._should_generate_dual_answer(
            request.question
        ):
            secondary_answer, dual_feedback, dual_strategy = (
                self._generate_secondary_answer(request, articles, request_id)
            )

        response = QAResponse(
            question=request.question,
            mode=request.mode,
            primary_answer=primary_answer,
            secondary_answer=secondary_answer,
            dual_answer_feedback=dual_feedback,
            retrieved_articles=retrieved_articles,
            follow_up_suggestions=follow_ups or None,
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            request_id=request_id,
        )

        # Cache only non-dual responses
        if secondary_answer is None:
            self.cache_manager.set(
                cache_key, response.model_dump(), ttl=TTL_QA_RESPONSE
            )

        # Persist to PostgreSQL
        await self._persist_request(
            request, response, effective_model, effective_rag, dual_strategy
        )

        return response

    async def _persist_request(
        self,
        request: QARequest,
        response: QAResponse,
        effective_model: str,
        effective_rag: bool,
        dual_strategy: Optional[str] = None,
    ) -> None:
        """Persist the QA request and response to PostgreSQL (best-effort)."""
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
                        a.urn for a in response.retrieved_articles
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

    def _retrieve_articles(
        self, question: str, top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[RetrievedArticle]]:
        """Embed the question and run kNN search against ES."""
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

            articles = []
            retrieved_models = []
            for r in raw_results:
                articles.append(r)
                retrieved_models.append(
                    RetrievedArticle(
                        urn=r.get("urn", r.get("_id", "")),
                        title=r.get("title", ""),
                        authors=r.get("authors"),
                        venue=r.get("venue"),
                        publication_year=r.get("publication_year"),
                        category=r.get("category"),
                        tags=r.get("tags"),
                        similarity_score=r.get("_score", 0.0),
                    )
                )

            logger.info(
                "Retrieved %d articles for question: %s...",
                len(articles), question[:50],
            )
            return articles, retrieved_models

        except Exception as e:
            logger.error("Error retrieving articles: %s", e, exc_info=True)
            return [], []

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

    def _build_cache_key(self, request: QARequest) -> str:
        """Generate cache key from question + effective settings."""
        return self.cache_manager.generate_cache_key(
            prefix="qa",
            data={
                "question": request.question.strip().lower(),
                "mode": request.mode,
                "model": self._resolve_model(request),
                "rag_enabled": (
                    request.rag_enabled if request.mode == "advanced" else True
                ),
                "top_k": request.top_k,
                "expertise_level": request.expertise_level,
                "language": request.language,
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
    ) -> Tuple[QAAnswer, DualAnswerFeedback, str]:
        """Generate a secondary answer using a different configuration.

        Returns:
            Tuple of (secondary_answer, dual_feedback, strategy_name)
        """
        strategy_name, primary_cfg, secondary_cfg = self._select_dual_strategy()

        secondary_model = secondary_cfg.get("model", DEFAULT_GROQ_MODEL)
        secondary_temp = secondary_cfg.get("temperature", 0.3)
        secondary_top_k = secondary_cfg.get("top_k", request.top_k)

        logger.info(
            "Generating dual answer (strategy=%s) for request %s",
            strategy_name, request_id,
        )

        secondary_agent = QAAgent(
            model=secondary_model, temperature=secondary_temp
        )

        # If strategy varies top_k, re-retrieve articles
        secondary_articles = articles
        if "top_k" in secondary_cfg and secondary_cfg["top_k"] != request.top_k:
            secondary_articles, _ = self._retrieve_articles(
                request.question, secondary_cfg["top_k"]
            )

        if secondary_articles:
            secondary_answer, _ = secondary_agent.generate_answer_with_rag(
                question=request.question,
                articles=secondary_articles,
                expertise_level=request.expertise_level,
                language=request.language,
            )
        else:
            secondary_answer, _ = secondary_agent.generate_answer_without_rag(
                question=request.question,
                expertise_level=request.expertise_level,
                language=request.language,
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
