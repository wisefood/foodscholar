"""Q&A service for non-contextual question answering."""
import json
import logging
import random
import re
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
    SimpleNutriQuestionsResponse,
    TipsOfTheDayResponse,
    AVAILABLE_GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
)
from agents.qa_agent import QAAgent
from backend.elastic import ELASTIC_CLIENT
from backend.groq import GROQ_CHAT
from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
from backend.redis import RedisClientSingleton
from models.db import QARequestRecord, QAFeedbackRecord
from utilities.cache import CacheManager
from exceptions import InvalidError

logger = logging.getLogger(__name__)

TTL_QA_RESPONSE = 86400  # 1 day
TTL_QA_FEEDBACK = 2592000  # 30 days
TTL_SIMPLE_NUTRI_QUESTIONS = 1800  # 30 minutes
TTL_TIPS_OF_THE_DAY = 1800  # 30 minutes
TIP_GROUNDING_TOP_K = 3
MAX_TIP_REGEN_ATTEMPTS = 3
MAX_SIMPLE_QUESTION_REGEN_ATTEMPTS = 3
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
        self._simple_question_llm = None
        self._simple_question_redis = None

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
                "version": 2,
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
        response = TipsOfTheDayResponse(
            did_you_know=generated_payload["did_you_know"],
            tips=generated_payload["tips"],
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
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
    ) -> Dict[str, List[str]]:
        """Generate grounded tips/facts with validation and regeneration."""
        for attempt in range(1, MAX_TIP_REGEN_ATTEMPTS + 1):
            try:
                payload = self._generate_tips_payload_once(
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                )
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
                    return payload

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
        self, tips_count: int, did_you_know_count: int
    ) -> Dict[str, List[str]]:
        """Single generation pass for tips/facts."""
        total_count = tips_count + did_you_know_count
        candidate_count = max(total_count * 2, 8)
        prompt = f"""You create daily nutrition content for a general audience.

Generate exactly {candidate_count} candidate items with a mix of:
- practical nutrition tips
- "Did you know?" nutrition facts

Rules:
- Keep each item short (<= 18 words).
- Keep content general and educational.
- Avoid diagnosis, treatment, medication, supplement dosage, or disease-management advice.
- Return ONLY valid JSON in this exact format:
{{
  "items": [
    {{"kind": "tip", "text": "item text"}},
    {{"kind": "did_you_know", "text": "item text"}}
  ]
}}
"""
        response = self.simple_question_llm.invoke(prompt)
        parsed = self._parse_tip_candidates_json(response.content)
        candidates = parsed.get("items", [])
        grounded_items = self._ground_tip_candidates(candidates, count=total_count)

        if len(grounded_items) < total_count:
            grounded_items.extend(
                self._generate_fallback_tips(
                    tips_count=tips_count,
                    did_you_know_count=did_you_know_count,
                )
            )

        return self._format_tips_payload(
            grounded_items,
            tips_count=tips_count,
            did_you_know_count=did_you_know_count,
        )

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

        has_human = any(term in text for term in HUMAN_EVIDENCE_TERMS)
        has_animal = any(term in text for term in ANIMAL_EVIDENCE_TERMS)

        if has_animal:
            return False
        return has_human

    def _mentions_animal_testing(self, text: str) -> bool:
        """Guardrail for animal-testing phrasing in output."""
        normalized = text.lower()
        return any(term in normalized for term in ANIMAL_EVIDENCE_TERMS)

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
        return any(term in normalized for term in blocked_terms)

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
    ) -> Dict[str, Any]:
        """Normalize cached/new payloads to did_you_know + tips shape."""
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
        )
        payload["did_you_know"] = did_you_know
        payload["tips"] = tips
        return payload

    def _split_tip_items(
        self,
        items: List[str],
        tips_count: int,
        did_you_know_count: int,
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
        return json.loads(text)

    def _parse_tip_candidates_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON tip candidate payload from model output."""
        text = content.strip()

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return json.loads(text)

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
            "Tip: Add one extra vegetable serving to your lunch.",
            "Tip: Pair fruit with nuts to balance energy and fullness.",
            "Tip: Choose whole grains more often than refined grains.",
            "Tip: Keep water nearby so hydration is easier all day.",
        ]
        did_you_know_facts = [
            "Did you know? Beans provide both fiber and protein in one food.",
            "Did you know? Frozen vegetables can retain nutrients very well.",
            "Did you know? Whole grains usually digest more slowly than refined grains.",
            "Did you know? Eating protein with meals can improve satiety.",
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
