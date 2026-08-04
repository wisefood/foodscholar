#!/usr/bin/env python3
"""
FoodScholar v2
==============
A robust multi-agent nutrition QA CLI over two Elasticsearch indices:
- articles: scientific article metadata + abstracts + dense embeddings
- guidelines: dietary guideline rules + metadata + dense embeddings

Design goals
------------
- Fast: parallel retrieval, short loops, bounded LLM calls
- Safe: explicit risk triage for pediatrics, pregnancy, chronic disease, medication/supplement interactions
- Grounded: hybrid retrieval, evidence ranking, inline citations, evidence snippets
- Practical: asks targeted clarification only when the missing detail materially changes the answer
- Multilingual: user can ask in many languages; retrieval pivots through English when useful
- Observable: emits live JSON-ish events to stderr for CLI/API integration

Architecture
------------
Specialist agents (6):
1) Safety Triage Agent
2) Question Clarifier Agent
3) Article Retrieval Agent
4) Guideline Retrieval Agent
5) Evidence Evaluator / Query Refiner Agent
6) Answer Formulation Agent

Frameworks
----------
- LangGraph for orchestration
- LangChain + Groq for LLM inference
- Elasticsearch for retrieval
- SentenceTransformers for query embeddings (CPU-friendly)

Run
---
python foodscholar_v2.py
python foodscholar_v2.py -q "How many times a week should my 8-year-old eat red meat?"
python foodscholar_v2.py -q "Πόσες φορές την εβδομάδα πρέπει να τρώει κόκκινο κρέας ένα παιδί;" --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from elasticsearch import Elasticsearch
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))
# Some ChatGroq installations support reasoning_effort; keep optional for compatibility.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Elasticsearch
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "")
VERIFY_CERTS = os.getenv("ELASTIC_VERIFY_CERTS", "false").lower() == "true"
ARTICLES_INDEX = os.getenv("ARTICLES_INDEX", "articles")
GUIDELINES_INDEX = os.getenv("GUIDELINES_INDEX", "guidelines")
USE_ELASTIC_RRF_RETRIEVER = os.getenv("USE_ELASTIC_RRF_RETRIEVER", "false").lower() == "true"
ELASTIC_REQUEST_TIMEOUT = int(os.getenv("ELASTIC_REQUEST_TIMEOUT", "20"))

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "torch")  # torch | onnx | openvino (if installed)
EMBEDDING_NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))

# Retrieval
TOP_K_ARTICLES = int(os.getenv("TOP_K_ARTICLES", "8"))
TOP_K_GUIDELINES = int(os.getenv("TOP_K_GUIDELINES", "8"))
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "80"))
MAX_REFINEMENT_ROUNDS = int(os.getenv("MAX_REFINEMENT_ROUNDS", "1"))
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", "8"))
HYBRID_LEXICAL_BOOST = float(os.getenv("HYBRID_LEXICAL_BOOST", "1.0"))
HYBRID_VECTOR_BOOST = float(os.getenv("HYBRID_VECTOR_BOOST", "2.0"))
MIN_ACCEPTABLE_HITS = int(os.getenv("MIN_ACCEPTABLE_HITS", "2"))
MIN_GUIDELINE_HITS_ACTIONABLE = int(os.getenv("MIN_GUIDELINE_HITS_ACTIONABLE", "1"))
GUIDELINE_RECENCY_SOFT_FLOOR = int(os.getenv("GUIDELINE_RECENCY_SOFT_FLOOR", "2015"))
ARTICLE_RECENCY_SOFT_FLOOR = int(os.getenv("ARTICLE_RECENCY_SOFT_FLOOR", "2016"))

# Ranking policy
GUIDELINE_PRIORITY_MULTIPLIER = float(os.getenv("GUIDELINE_PRIORITY_MULTIPLIER", "1.25"))
ARTICLE_PRIORITY_MULTIPLIER = float(os.getenv("ARTICLE_PRIORITY_MULTIPLIER", "1.0"))
ACTIONABLE_GUIDELINE_PRIORITY = os.getenv("ACTIONABLE_GUIDELINE_PRIORITY", "true").lower() == "true"

# UX / Observability
SHOW_DEBUG_EVENTS = os.getenv("SHOW_DEBUG_EVENTS", "true").lower() == "true"
OUTPUT_LANGUAGE_DEFAULT = os.getenv("OUTPUT_LANGUAGE_DEFAULT", "auto")
SHOW_EVIDENCE_SNIPPETS = os.getenv("SHOW_EVIDENCE_SNIPPETS", "true").lower() == "true"

# Field mappings: customize to your schema.
ARTICLE_TITLE_FIELD = os.getenv("ARTICLE_TITLE_FIELD", "title")
ARTICLE_ABSTRACT_FIELD = os.getenv("ARTICLE_ABSTRACT_FIELD", "abstract")
ARTICLE_EMBED_FIELD = os.getenv("ARTICLE_EMBED_FIELD", "embedding")
ARTICLE_YEAR_FIELD = os.getenv("ARTICLE_YEAR_FIELD", "year")
ARTICLE_JOURNAL_FIELD = os.getenv("ARTICLE_JOURNAL_FIELD", "journal")
ARTICLE_AUTHORS_FIELD = os.getenv("ARTICLE_AUTHORS_FIELD", "authors")
ARTICLE_DOI_FIELD = os.getenv("ARTICLE_DOI_FIELD", "doi")
ARTICLE_PMID_FIELD = os.getenv("ARTICLE_PMID_FIELD", "pmid")
ARTICLE_LANGUAGE_FIELD = os.getenv("ARTICLE_LANGUAGE_FIELD", "language")
ARTICLE_TYPE_FIELD = os.getenv("ARTICLE_TYPE_FIELD", "study_type")

GUIDELINE_RULE_FIELD = os.getenv("GUIDELINE_RULE_FIELD", "rule_text")
GUIDELINE_EMBED_FIELD = os.getenv("GUIDELINE_EMBED_FIELD", "embedding")
GUIDELINE_COUNTRY_FIELD = os.getenv("GUIDELINE_COUNTRY_FIELD", "country")
GUIDELINE_SOURCE_FIELD = os.getenv("GUIDELINE_SOURCE_FIELD", "source")
GUIDELINE_POPULATION_FIELD = os.getenv("GUIDELINE_POPULATION_FIELD", "population")
GUIDELINE_YEAR_FIELD = os.getenv("GUIDELINE_YEAR_FIELD", "year")
GUIDELINE_TOPIC_FIELD = os.getenv("GUIDELINE_TOPIC_FIELD", "topic")
GUIDELINE_LANGUAGE_FIELD = os.getenv("GUIDELINE_LANGUAGE_FIELD", "language")


# ============================================================
# UTILITIES
# ============================================================
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def log_event(stage: str, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    if not SHOW_DEBUG_EVENTS:
        return
    stamp = time.strftime("%H:%M:%S")
    eprint(f"[{stamp}] [{stage}] {message}")
    if payload:
        eprint(json.dumps(payload, ensure_ascii=False, indent=2))


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate(text: str, n: int = 280) -> str:
    text = compact_ws(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def normalize_language_code(code: Optional[str]) -> str:
    if not code:
        return OUTPUT_LANGUAGE_DEFAULT
    code = code.lower().strip()
    aliases = {
        "greek": "el",
        "gr": "el",
        "english": "en",
        "arabic": "ar",
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "italian": "it",
    }
    return aliases.get(code, code)


# ============================================================
# STRUCTURED MODELS
# ============================================================
class ClarificationOption(BaseModel):
    label: str
    value: str


class SafetyOutput(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    safety_flags: List[str] = Field(default_factory=list)
    medical_red_flags: List[str] = Field(default_factory=list)
    clarification_required_for_safety: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[ClarificationOption] = Field(default_factory=list)
    answer_guardrails: List[str] = Field(default_factory=list)


class ClarifierOutput(BaseModel):
    original_question: str
    canonical_english_question: str
    output_language: str
    user_intent: Literal[
        "actionable_guidance",
        "scientific_evidence",
        "comparison",
        "meal_planning",
        "safety",
        "myth_check",
        "other",
    ]
    answer_style: Literal["layperson", "clinical", "balanced"] = "balanced"
    target_population: Optional[str] = None
    age_group: Optional[str] = None
    health_conditions: List[str] = Field(default_factory=list)
    medications_or_supplements: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    material_missing_details: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[ClarificationOption] = Field(default_factory=list)
    article_query: str
    guideline_query: str
    metadata_hints: Dict[str, Any] = Field(default_factory=dict)


class RetrievalStatus(BaseModel):
    branch: Literal["articles", "guidelines"]
    ok: bool
    error: Optional[str] = None
    hit_count: int = 0
    used_query: str


class EvaluationOutput(BaseModel):
    satisfactory: bool
    reason: str
    evidence_gaps: List[str] = Field(default_factory=list)
    should_ask_user: bool = False
    user_question: Optional[str] = None
    user_options: List[ClarificationOption] = Field(default_factory=list)
    refine_articles: bool = False
    refine_guidelines: bool = False
    refined_article_query: Optional[str] = None
    refined_guideline_query: Optional[str] = None
    answer_even_if_partial: bool = True
    use_guideline_priority: bool = True


class FinalAnswerOutput(BaseModel):
    answer: str = Field(description="Must include inline citations like [G1], [A2] for substantive claims.")
    confidence: Literal["low", "medium", "high"]
    direct_answer_first: str
    evidence_summary: str
    uncertainty_note: Optional[str] = None
    needs_followup: bool = False
    followup_question: Optional[str] = None
    disclaimers: List[str] = Field(default_factory=list)


# ============================================================
# STATE
# ============================================================
class FoodScholarState(TypedDict, total=False):
    user_question: str
    refinement_round: int
    safety: Dict[str, Any]
    clarifier: Dict[str, Any]
    articles_results: List[Dict[str, Any]]
    guidelines_results: List[Dict[str, Any]]
    articles_status: Dict[str, Any]
    guidelines_status: Dict[str, Any]
    evaluation: Dict[str, Any]
    answer: Dict[str, Any]
    final_output: Dict[str, Any]
    last_error: str


@dataclass
class Runtime:
    llm: Any
    es: Elasticsearch
    embedder: SentenceTransformer

    @classmethod
    def build(cls) -> "Runtime":
        if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
            raise ValueError("Please set GROQ_API_KEY.")

        # Be defensive across langchain-groq versions.
        llm_kwargs = {
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "temperature": GROQ_TEMPERATURE,
            "max_retries": GROQ_MAX_RETRIES,
        }
        try:
            llm = ChatGroq(**llm_kwargs, reasoning_effort=GROQ_REASONING_EFFORT)
        except TypeError:
            llm = ChatGroq(**llm_kwargs)

        if ELASTIC_API_KEY:
            es = Elasticsearch(
                ELASTIC_URL,
                api_key=ELASTIC_API_KEY,
                verify_certs=VERIFY_CERTS,
                request_timeout=ELASTIC_REQUEST_TIMEOUT,
            )
        elif ELASTIC_USERNAME and ELASTIC_PASSWORD:
            es = Elasticsearch(
                ELASTIC_URL,
                basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
                verify_certs=VERIFY_CERTS,
                request_timeout=ELASTIC_REQUEST_TIMEOUT,
            )
        else:
            es = Elasticsearch(
                ELASTIC_URL,
                verify_certs=VERIFY_CERTS,
                request_timeout=ELASTIC_REQUEST_TIMEOUT,
            )

        embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
            backend=EMBEDDING_BACKEND,
        )
        return cls(llm=llm, es=es, embedder=embedder)

    def embed_query(self, text: str) -> List[float]:
        text = compact_ws(text)
        if not text:
            text = "nutrition"
        if hasattr(self.embedder, "encode_query"):
            vec = self.embedder.encode_query(
                text,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=EMBEDDING_NORMALIZE,
            )
        else:
            vec = self.embedder.encode(
                text,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=EMBEDDING_NORMALIZE,
            )
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)


RUNTIME: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = Runtime.build()
    return RUNTIME


# ============================================================
# EVIDENCE NORMALIZATION / RANKING
# ============================================================
def normalize_article_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source", {})
    return {
        "branch": "articles",
        "doc_id": hit.get("_id"),
        "score": float(hit.get("_score", 0.0) or 0.0),
        "title": src.get(ARTICLE_TITLE_FIELD, ""),
        "abstract": src.get(ARTICLE_ABSTRACT_FIELD, ""),
        "year": src.get(ARTICLE_YEAR_FIELD),
        "journal": src.get(ARTICLE_JOURNAL_FIELD),
        "authors": src.get(ARTICLE_AUTHORS_FIELD),
        "doi": src.get(ARTICLE_DOI_FIELD),
        "pmid": src.get(ARTICLE_PMID_FIELD),
        "language": src.get(ARTICLE_LANGUAGE_FIELD),
        "study_type": src.get(ARTICLE_TYPE_FIELD),
    }



def normalize_guideline_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source", {})
    return {
        "branch": "guidelines",
        "doc_id": hit.get("_id"),
        "score": float(hit.get("_score", 0.0) or 0.0),
        "rule_text": src.get(GUIDELINE_RULE_FIELD, ""),
        "country": src.get(GUIDELINE_COUNTRY_FIELD),
        "source": src.get(GUIDELINE_SOURCE_FIELD),
        "population": src.get(GUIDELINE_POPULATION_FIELD),
        "year": src.get(GUIDELINE_YEAR_FIELD),
        "topic": src.get(GUIDELINE_TOPIC_FIELD),
        "language": src.get(GUIDELINE_LANGUAGE_FIELD),
    }



def rank_article(result: Dict[str, Any]) -> float:
    score = float(result.get("score", 0.0) or 0.0) * ARTICLE_PRIORITY_MULTIPLIER
    year = safe_int(result.get("year"), 0)
    if year >= ARTICLE_RECENCY_SOFT_FLOOR:
        score += 0.2
    study_type = compact_ws(str(result.get("study_type") or "")).lower()
    if any(x in study_type for x in ["meta", "systematic", "guideline", "review"]):
        score += 0.3
    if result.get("doi") or result.get("pmid"):
        score += 0.1
    return round(score, 5)



def rank_guideline(result: Dict[str, Any]) -> float:
    score = float(result.get("score", 0.0) or 0.0) * GUIDELINE_PRIORITY_MULTIPLIER
    year = safe_int(result.get("year"), 0)
    if year >= GUIDELINE_RECENCY_SOFT_FLOOR:
        score += 0.2
    if result.get("population"):
        score += 0.1
    if result.get("source"):
        score += 0.1
    return round(score, 5)



def postprocess_results(results: List[Dict[str, Any]], branch: str) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for r in results:
        key = (r.get("doi") or r.get("pmid") or r.get("doc_id") or truncate(r.get("title") or r.get("rule_text") or "", 80)).strip()
        if key in seen:
            continue
        seen.add(key)
        r["rank_score"] = rank_guideline(r) if branch == "guidelines" else rank_article(r)
        deduped.append(r)
    deduped.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return deduped


# ============================================================
# ELASTICSEARCH QUERY BUILDERS
# ============================================================
def build_rrf_retriever(text_query: str, vector: List[float], branch: str, size: int) -> Dict[str, Any]:
    if branch == "articles":
        lexical_query = {
            "multi_match": {
                "query": text_query,
                "fields": [f"{ARTICLE_TITLE_FIELD}^3", f"{ARTICLE_ABSTRACT_FIELD}^2"],
                "type": "best_fields",
            }
        }
        knn_field = ARTICLE_EMBED_FIELD
    else:
        lexical_query = {
            "multi_match": {
                "query": text_query,
                "fields": [
                    f"{GUIDELINE_RULE_FIELD}^4",
                    f"{GUIDELINE_POPULATION_FIELD}^2",
                    f"{GUIDELINE_SOURCE_FIELD}^1.5",
                    f"{GUIDELINE_COUNTRY_FIELD}^1",
                    f"{GUIDELINE_TOPIC_FIELD}^1",
                ],
                "type": "best_fields",
            }
        }
        knn_field = GUIDELINE_EMBED_FIELD
    return {
        "size": size,
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": lexical_query}},
                    {"knn": {"field": knn_field, "query_vector": vector, "k": size, "num_candidates": NUM_CANDIDATES}},
                ]
            }
        },
        "_source": True,
    }



def build_hybrid_body(text_query: str, vector: List[float], branch: str, size: int, metadata_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata_hints = metadata_hints or {}
    if branch == "articles":
        fields = [f"{ARTICLE_TITLE_FIELD}^3", f"{ARTICLE_ABSTRACT_FIELD}^2"]
        embed_field = ARTICLE_EMBED_FIELD
        filter_clauses: List[Dict[str, Any]] = []
        lang = metadata_hints.get("language")
        if lang and ARTICLE_LANGUAGE_FIELD:
            filter_clauses.append({"term": {ARTICLE_LANGUAGE_FIELD: lang}})
    else:
        fields = [
            f"{GUIDELINE_RULE_FIELD}^4",
            f"{GUIDELINE_POPULATION_FIELD}^2",
            f"{GUIDELINE_SOURCE_FIELD}^1.5",
            f"{GUIDELINE_COUNTRY_FIELD}^1",
            f"{GUIDELINE_TOPIC_FIELD}^1",
        ]
        embed_field = GUIDELINE_EMBED_FIELD
        filter_clauses = []
        country = metadata_hints.get("country")
        if country:
            filter_clauses.append({"term": {GUIDELINE_COUNTRY_FIELD: country}})
        lang = metadata_hints.get("language")
        if lang and GUIDELINE_LANGUAGE_FIELD:
            filter_clauses.append({"term": {GUIDELINE_LANGUAGE_FIELD: lang}})

    query: Dict[str, Any] = {
        "size": size,
        "_source": True,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text_query,
                            "fields": fields,
                            "type": "best_fields",
                            "boost": HYBRID_LEXICAL_BOOST,
                        }
                    },
                    {
                        "knn": {
                            embed_field: {
                                "vector": vector,
                                "k": size,
                                "boost": HYBRID_VECTOR_BOOST,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
                "filter": filter_clauses,
            }
        },
    }
    return query



def elastic_search_branch(branch: str, query_text: str, size: int, metadata_hints: Optional[Dict[str, Any]] = None) -> Tuple[RetrievalStatus, List[Dict[str, Any]]]:
    rt = get_runtime()
    index = ARTICLES_INDEX if branch == "articles" else GUIDELINES_INDEX
    try:
        vector = rt.embed_query(query_text)
        if USE_ELASTIC_RRF_RETRIEVER:
            body = build_rrf_retriever(query_text, vector, branch, size)
        else:
            body = build_hybrid_body(query_text, vector, branch, size, metadata_hints=metadata_hints)
        log_event(f"retrieval.{branch}", "Searching Elasticsearch", {"index": index, "query": query_text, "rrf": USE_ELASTIC_RRF_RETRIEVER})
        resp = rt.es.search(index=index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if branch == "articles":
            results = postprocess_results([normalize_article_hit(h) for h in hits], branch)
        else:
            results = postprocess_results([normalize_guideline_hit(h) for h in hits], branch)
        return RetrievalStatus(branch=branch, ok=True, hit_count=len(results), used_query=query_text), results
    except Exception as exc:
        log_event(f"retrieval.{branch}", "Branch failed", {"error": repr(exc), "query": query_text})
        return RetrievalStatus(branch=branch, ok=False, error=repr(exc), used_query=query_text), []


# ============================================================
# PROMPTS
# ============================================================
SAFETY_SYSTEM = """
You are FoodScholar's Safety Triage Agent.

Classify the user query for nutrition/medical-nutrition risk.
Prioritize safety for: infants/children, pregnancy/breastfeeding, kidney disease, liver disease,
diabetes on medication, severe GI symptoms, eating disorders, food allergies, drug-supplement interactions,
and any red-flag symptom patterns.

Return only the schema.
Ask for clarification only when the missing detail materially changes the safe answer.
""".strip()


CLARIFIER_SYSTEM = """
You are FoodScholar's Question Clarifier Agent.

Responsibilities:
- Convert the user's question into a canonical English retrieval form.
- Preserve the user's desired output language.
- Create one query optimized for scientific articles and one for dietary guidelines.
- Extract age group, target population, conditions, meds/supplements, and material missing details.
- Only ask clarification when the missing information would substantially change the answer.
- Everyday practical questions should generally be answerable in a concise layperson style.
- Intermediary steps may be in English.

Return only the schema.
""".strip()


EVALUATOR_SYSTEM = """
You are FoodScholar's Evidence Evaluator and Query Refiner.

Decide whether the currently retrieved evidence is sufficient.
Policy:
- For actionable intake/frequency questions, prefer guideline evidence.
- Use article evidence for nuance, uncertainty, mechanisms, or if guidelines are sparse.
- Distinguish between:
  1) no evidence,
  2) partial evidence,
  3) branch/system failure,
  4) conflicting evidence,
  5) missing user detail.
- Keep loops short. Refine only if likely to materially improve retrieval.
- It is acceptable to answer partially when one branch succeeded and the other failed.

Return only the schema.
""".strip()


ANSWER_SYSTEM = """
You are FoodScholar's Answer Formulation Agent.

Write a safe, clear, helpful answer grounded in the provided evidence catalog.
Rules:
- Answer in the requested output language.
- Start with a direct answer.
- Every substantive claim must have inline citations like [G1], [A2], or [G1, A2].
- Prefer guideline citations for actionable recommendations.
- Use article citations for nuance, uncertainty, or mechanistic context.
- If evidence is partial, conflicting, or one retrieval branch failed, say so clearly.
- Do not overstate certainty. Do not diagnose.
- If a clarification would materially improve personalization, ask one short follow-up question.

Return only the schema.
""".strip()


# ============================================================
# AGENT NODES
# ============================================================
def safety_node(state: FoodScholarState) -> FoodScholarState:
    rt = get_runtime()
    question = state["user_question"]
    log_event("safety", "Triaging risk level", {"question": question})
    model = rt.llm.with_structured_output(SafetyOutput)
    out = model.invoke([("system", SAFETY_SYSTEM), ("human", question)])
    state["safety"] = out.model_dump()
    return state



def clarifier_node(state: FoodScholarState) -> FoodScholarState:
    rt = get_runtime()
    log_event("clarifier", "Planning retrieval queries")
    model = rt.llm.with_structured_output(ClarifierOutput)
    payload = {
        "question": state["user_question"],
        "safety": state.get("safety", {}),
        "default_output_language": OUTPUT_LANGUAGE_DEFAULT,
    }
    out = model.invoke([("system", CLARIFIER_SYSTEM), ("human", json.dumps(payload, ensure_ascii=False))])
    clar = out.model_dump()
    clar["output_language"] = normalize_language_code(clar.get("output_language"))
    state["clarifier"] = clar
    return state



def retrieve_parallel_node(state: FoodScholarState) -> FoodScholarState:
    clar = state["clarifier"]
    metadata_hints = clar.get("metadata_hints", {}) or {}
    tasks = [
        ("articles", clar["article_query"], TOP_K_ARTICLES),
        ("guidelines", clar["guideline_query"], TOP_K_GUIDELINES),
    ]
    results_map: Dict[str, List[Dict[str, Any]]] = {"articles": [], "guidelines": []}
    status_map: Dict[str, Dict[str, Any]] = {}

    log_event("retrieval", "Running retrieval branches in parallel", {"round": state.get("refinement_round", 0)})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(elastic_search_branch, branch, query, size, metadata_hints): branch
            for branch, query, size in tasks
        }
        for future in as_completed(futures):
            branch = futures[future]
            status, results = future.result()
            status_map[branch] = status.model_dump()
            results_map[branch] = results

    state["articles_results"] = results_map["articles"]
    state["guidelines_results"] = results_map["guidelines"]
    state["articles_status"] = status_map["articles"]
    state["guidelines_status"] = status_map["guidelines"]
    return state



def evaluator_node(state: FoodScholarState) -> FoodScholarState:
    rt = get_runtime()
    safety = state.get("safety", {})
    clar = state.get("clarifier", {})
    articles = state.get("articles_results", [])
    guidelines = state.get("guidelines_results", [])
    articles_status = state.get("articles_status", {})
    guidelines_status = state.get("guidelines_status", {})

    log_event(
        "evaluator",
        "Assessing evidence sufficiency",
        {
            "article_hits": len(articles),
            "guideline_hits": len(guidelines),
            "articles_ok": articles_status.get("ok"),
            "guidelines_ok": guidelines_status.get("ok"),
            "risk": safety.get("risk_level"),
        },
    )

    # Deterministic gates first.
    if safety.get("clarification_required_for_safety"):
        state["evaluation"] = EvaluationOutput(
            satisfactory=False,
            reason="Safety triage requires clarification before a safe personalized answer.",
            should_ask_user=True,
            user_question=safety.get("clarification_question"),
            user_options=[ClarificationOption(**x) for x in safety.get("clarification_options", [])],
            answer_even_if_partial=False,
        ).model_dump()
        return state

    if clar.get("needs_clarification"):
        state["evaluation"] = EvaluationOutput(
            satisfactory=False,
            reason="Question is materially underspecified.",
            should_ask_user=True,
            user_question=clar.get("clarification_question"),
            user_options=[ClarificationOption(**x) for x in clar.get("clarification_options", [])],
            answer_even_if_partial=False,
        ).model_dump()
        return state

    actionable = clar.get("user_intent") == "actionable_guidance"
    one_branch_ok = bool(articles_status.get("ok")) or bool(guidelines_status.get("ok"))
    enough_guidelines = len(guidelines) >= MIN_GUIDELINE_HITS_ACTIONABLE
    enough_any = len(articles) >= MIN_ACCEPTABLE_HITS or len(guidelines) >= MIN_ACCEPTABLE_HITS

    if actionable and ACTIONABLE_GUIDELINE_PRIORITY and enough_guidelines:
        state["evaluation"] = EvaluationOutput(
            satisfactory=True,
            reason="Guideline evidence is sufficient for an actionable answer.",
            use_guideline_priority=True,
        ).model_dump()
        return state

    if one_branch_ok and enough_any and state.get("refinement_round", 0) >= MAX_REFINEMENT_ROUNDS:
        state["evaluation"] = EvaluationOutput(
            satisfactory=True,
            reason="Reached refinement limit; answering with the best available evidence.",
            answer_even_if_partial=True,
            use_guideline_priority=actionable,
        ).model_dump()
        return state

    model = rt.llm.with_structured_output(EvaluationOutput)
    payload = {
        "user_question": state["user_question"],
        "safety": safety,
        "clarifier": clar,
        "current_round": state.get("refinement_round", 0),
        "articles_status": articles_status,
        "guidelines_status": guidelines_status,
        "articles_top": articles[:5],
        "guidelines_top": guidelines[:5],
    }
    out = model.invoke([("system", EVALUATOR_SYSTEM), ("human", json.dumps(payload, ensure_ascii=False))])
    state["evaluation"] = out.model_dump()
    return state



def refinement_router(state: FoodScholarState) -> str:
    evaluation = state.get("evaluation", {})
    round_no = state.get("refinement_round", 0)
    if evaluation.get("should_ask_user"):
        return "ask_user"
    if evaluation.get("satisfactory"):
        return "answer"
    if round_no >= MAX_REFINEMENT_ROUNDS:
        return "answer"
    if evaluation.get("refine_articles") or evaluation.get("refine_guidelines"):
        return "refine"
    return "answer"



def refine_queries_node(state: FoodScholarState) -> FoodScholarState:
    evaluation = state.get("evaluation", {})
    clar = dict(state.get("clarifier", {}))
    if evaluation.get("refine_articles") and evaluation.get("refined_article_query"):
        clar["article_query"] = evaluation["refined_article_query"]
    if evaluation.get("refine_guidelines") and evaluation.get("refined_guideline_query"):
        clar["guideline_query"] = evaluation["refined_guideline_query"]
    state["clarifier"] = clar
    state["refinement_round"] = state.get("refinement_round", 0) + 1
    log_event("refine", "Updated retrieval queries", {"round": state["refinement_round"], "article_query": clar.get("article_query"), "guideline_query": clar.get("guideline_query")})
    return state



def ask_user_node(state: FoodScholarState) -> FoodScholarState:
    evaluation = state.get("evaluation", {})
    prompt = evaluation.get("user_question") or "Could you clarify your question?"
    options = evaluation.get("user_options") or []
    log_event("clarification", "Awaiting user clarification", {"prompt": prompt, "options": options})

    eprint("\nFoodScholar needs one clarification before answering:")
    eprint(prompt)
    for idx, opt in enumerate(options, start=1):
        if isinstance(opt, dict):
            eprint(f"  {idx}. {opt.get('label')} ({opt.get('value')})")
        else:
            eprint(f"  {idx}. {opt.label} ({opt.value})")
    user_reply = input("\nYour answer: ").strip()
    state["user_question"] = f"{state['user_question']}\n\nUser clarification: {user_reply}"
    state["refinement_round"] = 0
    for key in ["evaluation", "clarifier", "articles_results", "guidelines_results", "articles_status", "guidelines_status", "answer", "final_output"]:
        state.pop(key, None)
    return state


# ============================================================
# EVIDENCE CATALOG / ANSWERING
# ============================================================
def build_evidence_catalog(state: FoodScholarState) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    clar = state.get("clarifier", {})
    actionable = clar.get("user_intent") == "actionable_guidance"

    guidelines = list(state.get("guidelines_results", []))[:MAX_CONTEXT_DOCS]
    articles = list(state.get("articles_results", []))[:MAX_CONTEXT_DOCS]

    if actionable and ACTIONABLE_GUIDELINE_PRIORITY:
        merged = [("guideline", g) for g in guidelines] + [("article", a) for a in articles]
    else:
        merged = sorted(
            [("guideline", g) for g in guidelines] + [("article", a) for a in articles],
            key=lambda t: t[1].get("rank_score", 0.0),
            reverse=True,
        )

    catalog: List[Dict[str, Any]] = []
    guideline_counter = 0
    article_counter = 0

    for branch, item in merged:
        if branch == "guideline":
            guideline_counter += 1
            cid = f"G{guideline_counter}"
            catalog.append(
                {
                    "citation_id": cid,
                    "branch": "guideline",
                    "doc_id": item.get("doc_id"),
                    "source": item.get("source"),
                    "country": item.get("country"),
                    "year": item.get("year"),
                    "population": item.get("population"),
                    "topic": item.get("topic"),
                    "evidence_snippet": truncate(item.get("rule_text", ""), 280),
                    "rank_score": item.get("rank_score"),
                }
            )
        else:
            article_counter += 1
            cid = f"A{article_counter}"
            catalog.append(
                {
                    "citation_id": cid,
                    "branch": "article",
                    "doc_id": item.get("doc_id"),
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "journal": item.get("journal"),
                    "doi": item.get("doi"),
                    "pmid": item.get("pmid"),
                    "study_type": item.get("study_type"),
                    "evidence_snippet": truncate(item.get("abstract", ""), 280),
                    "rank_score": item.get("rank_score"),
                }
            )

    meta = {
        "articles_status": state.get("articles_status", {}),
        "guidelines_status": state.get("guidelines_status", {}),
        "risk_level": state.get("safety", {}).get("risk_level"),
        "output_language": clar.get("output_language", OUTPUT_LANGUAGE_DEFAULT),
        "answer_style": clar.get("answer_style", "balanced"),
        "user_intent": clar.get("user_intent"),
        "refinement_rounds_used": state.get("refinement_round", 0),
    }
    return catalog, meta



def answer_node(state: FoodScholarState) -> FoodScholarState:
    rt = get_runtime()
    catalog, meta = build_evidence_catalog(state)
    payload = {
        "user_question": state.get("user_question"),
        "safety": state.get("safety", {}),
        "clarifier": state.get("clarifier", {}),
        "evaluation": state.get("evaluation", {}),
        "retrieval_meta": meta,
        "evidence_catalog": catalog,
    }
    log_event("answer", "Formulating final answer", {"catalog_items": len(catalog), "language": meta.get("output_language")})
    model = rt.llm.with_structured_output(FinalAnswerOutput)
    out = model.invoke([("system", ANSWER_SYSTEM), ("human", json.dumps(payload, ensure_ascii=False))])
    answer = out.model_dump()

    state["answer"] = answer
    state["final_output"] = {
        "answer": answer["answer"],
        "direct_answer_first": answer["direct_answer_first"],
        "evidence_summary": answer["evidence_summary"],
        "uncertainty_note": answer.get("uncertainty_note"),
        "confidence": answer["confidence"],
        "needs_followup": answer["needs_followup"],
        "followup_question": answer.get("followup_question"),
        "disclaimers": answer.get("disclaimers", []),
        "citations": catalog,
        "meta": {
            **meta,
            "model": GROQ_MODEL,
        },
    }
    return state


# ============================================================
# GRAPH
# ============================================================
def build_graph():
    graph = StateGraph(FoodScholarState)
    graph.add_node("safety", safety_node)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("retrieve_parallel", retrieve_parallel_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("refine", refine_queries_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("safety")
    graph.add_edge("safety", "clarifier")
    graph.add_edge("clarifier", "retrieve_parallel")
    graph.add_edge("retrieve_parallel", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        refinement_router,
        {"answer": "answer", "refine": "refine", "ask_user": "ask_user"},
    )
    graph.add_edge("refine", "retrieve_parallel")
    graph.add_edge("ask_user", "safety")
    graph.add_edge("answer", END)
    return graph.compile()


APP = build_graph()


# ============================================================
# CLI RENDERING
# ============================================================
def render_text_output(final_output: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(final_output.get("answer", ""))
    parts.append(f"\nConfidence: {final_output.get('confidence', 'unknown')}")

    uncertainty = final_output.get("uncertainty_note")
    if uncertainty:
        parts.append(f"\nUncertainty: {uncertainty}")

    disclaimers = final_output.get("disclaimers") or []
    if disclaimers:
        parts.append("\nDisclaimers:")
        for d in disclaimers:
            parts.append(f"- {d}")

    citations = final_output.get("citations") or []
    if citations:
        parts.append("\nSources:")
        for c in citations:
            if c.get("branch") == "guideline":
                line = f"- [{c.get('citation_id')}] guideline | {c.get('source')} | {c.get('country')} | {c.get('year')} | population={c.get('population')}"
            else:
                line = f"- [{c.get('citation_id')}] article | {c.get('title')} | {c.get('year')} | {c.get('journal')} | doi={c.get('doi')} | pmid={c.get('pmid')}"
            parts.append(line)
            if SHOW_EVIDENCE_SNIPPETS and c.get("evidence_snippet"):
                parts.append(f"  snippet: {c.get('evidence_snippet')}")

    if final_output.get("needs_followup") and final_output.get("followup_question"):
        parts.append(f"\nFollow-up: {final_output.get('followup_question')}")

    meta = final_output.get("meta", {})
    parts.append(
        "\nMeta: "
        + json.dumps(
            {
                "risk_level": meta.get("risk_level"),
                "user_intent": meta.get("user_intent"),
                "output_language": meta.get("output_language"),
                "refinement_rounds_used": meta.get("refinement_rounds_used"),
                "articles_ok": meta.get("articles_status", {}).get("ok"),
                "guidelines_ok": meta.get("guidelines_status", {}).get("ok"),
            },
            ensure_ascii=False,
        )
    )
    return "\n".join(parts)



def run_once(question: str, json_output: bool = False) -> int:
    initial_state: FoodScholarState = {"user_question": question, "refinement_round": 0}
    try:
        final_state = APP.invoke(initial_state)
        final_output = final_state.get("final_output", {})
        if json_output:
            print(json.dumps(final_output, ensure_ascii=False, indent=2))
        else:
            print(render_text_output(final_output))
        return 0
    except KeyboardInterrupt:
        eprint("Interrupted by user.")
        return 130
    except Exception as exc:
        log_event("error", "Unhandled exception", {"error": repr(exc)})
        payload = {"error": repr(exc)}
        if json_output:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Error: {exc}")
        return 1



def interactive_loop(json_output: bool = False) -> int:
    eprint("FoodScholar v2 CLI ready. Type your question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("foodscholar> ").strip()
        except EOFError:
            eprint()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        code = run_once(question, json_output=json_output)
        eprint("\n" + "-" * 90 + "\n")
        if code != 0:
            return code
    return 0



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FoodScholar multi-agent nutrition QA CLI")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.question:
        raise SystemExit(run_once(args.question, json_output=args.json))
    raise SystemExit(interactive_loop(json_output=args.json))
