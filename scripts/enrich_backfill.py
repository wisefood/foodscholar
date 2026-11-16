#!/usr/bin/env python3
"""
Backfill article enrichment from a self-hosted, OpenAI-compatible model.

Enriches every catalog article that has not been enriched yet (no
``ai_category``, the same marker the admin console uses) and writes the result
back to the data catalog.

What is reused from the service, unchanged:

* the prompts (``backend.prompts``: keyword system/user, annotation);
* the post-processing (``EnrichmentAgent._homogenize_keywords``,
  ``_normalize_enriched``, ``_to_ascii_obj``);
* the catalog field mapping (``services.enrichment_jobs.extract_enrichment_fields``)
  and the write order of ``persist_enrichment`` (PATCH, then /enhance).

What is different, and why:

* **Transport.** The service talks to Groq through LangChain. This talks to any
  OpenAI-compatible ``/v1/chat/completions`` (GPUStack / llama.cpp here).
* **Constrained decoding.** Every call sends a JSON schema as
  ``response_format``; the server turns it into a grammar, so the output is
  always well-formed JSON with the right keys, enums and array sizes.
* **Strict validation + retries.** Well-formed is not the same as good. Each
  output is checked again in Python (schema + content rules: no placeholders,
  exactly 3 Q&A per audience, questions end with "?", no duplicate questions,
  a real plain-language rewrite, ...). A rejected output is retried, alternating
  a "repair" turn that feeds the errors back to the model with a fresh sample at
  a higher temperature. Truncated outputs are retried with a larger budget.
* **Extras are merged, not replaced.** ``persist_enrichment`` overwrites the
  article's ``extras``; here keys the enrichment does not own are kept. The
  fields that are overwritten are saved to ``enriched.jsonl`` first (rollback).
* **Read-back verification.** After writing, the article is fetched again and
  the written fields are compared. Only then is it counted as done.

Operational behaviour:

* Resumable: every outcome is appended to ``<state>/results.jsonl``; a re-run
  skips what is done or already failed (``--retry-failed`` to retry those).
* Outages pause instead of failing articles: when the model server or the
  catalog stops answering, all workers wait while one probes it, and the
  article in flight is re-queued. An expired/invalid key stops the run.
* Ctrl-C / SIGTERM: stop taking new articles, finish the ones in flight.
  A second signal exits immediately.
* Progress: a line every ``--progress-every`` seconds, and
  ``<state>/progress.json`` (read by the ``status`` command).

Subcommands: ``run``, ``selftest`` (model only, no catalog), ``count``,
``status``. See ``scripts/enrich_backfill.sh`` for the usual entry point.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import logging
import os
import queue
import random
import re
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Production serves the in-code prompts; a stray Langfuse key in the operator's
# shell must not swap prompts in the middle of a backfill.
for _key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(_key, None)

# The service's Groq pool is constructed on import and warns about a missing
# GROQ_API_KEY. This script never uses it (the model is reached over the
# OpenAI-compatible API below), so the warning is only noise.
logging.getLogger("backend.groq").setLevel(logging.ERROR)

from config import config  # noqa: E402

config.setup()

from agents.enrichment_agent import EnrichmentAgent  # noqa: E402
from agents.json_output import parse_json_array  # noqa: E402
from backend.prompts import (  # noqa: E402
    ENRICHMENT_ANNOTATION,
    ENRICHMENT_KEYWORDS_SYSTEM,
    ENRICHMENT_KEYWORDS_USER,
)
from services.enrichment_jobs import extract_enrichment_fields  # noqa: E402

log = logging.getLogger("enrich_backfill")

VALIDATOR_VERSION = "2026-11-11.1"
ENHANCE_AGENT = "foodscholar-v1"  # same agent id the service writes
ENRICHED_MARKER_FIELD = "ai_category"  # same marker the admin console uses
MIN_ABSTRACT_WORDS = 25

# Token budgets. max_tokens caps hidden reasoning AND visible output together.
KEYWORD_MAX_TOKENS = 2048
ANNOTATION_MAX_TOKENS = 16384
ANNOTATION_MAX_TOKENS_ESCALATED = 24576

# Attempt n (1-based) uses TEMPERATURES[n-1] (last value repeats).
TEMPERATURES = (0.0, 0.2, 0.5, 0.7, 0.8)

# ---------------------------------------------------------------------------
# Output contract (mirrors the FIELD RULES of the annotation prompt)
# ---------------------------------------------------------------------------

READER_GROUPS = [
    "Academic Researchers", "Healthcare Professionals", "Industry/Policy", "General Public",
]
AGE_GROUPS = [
    "Infants (0-2)", "Children (3-12)", "Adolescents (13-18)", "Adults (18-64)",
    "Older adults (65+)", "Mixed", "Not stated",
]
INCOME_SETTINGS = ["High-income", "Middle-income", "Low-income", "Mixed", "Not stated"]
BIOLOGICAL_MODELS = ["Human", "Animal", "In vitro", "Mixed", "Not stated"]
TOPICS = [
    "Dietary patterns", "Macronutrients", "Micronutrients", "Fiber", "Ultra-processed foods",
    "Supplements", "Weight management", "Metabolism", "Cardiovascular health",
    "Diabetes & glycemic control", "Gut health & microbiome", "Cancer & oncology",
    "Inflammation & oxidative stress", "Bone health", "Physical Activity & Exercise",
    "Cognitive health", "Sports & performance", "Pregnancy & pediatrics", "Aging & longevity",
    "Food safety & allergens", "Public health nutrition", "Other",
]
STUDY_TYPES = [
    "Randomized Controlled Trial", "Non-randomized Intervention", "Observational (Cohort)",
    "Observational (Case-control)", "Observational (Cross-sectional)", "Systematic Review",
    "Meta-analysis", "Narrative Review", "Qualitative Study", "Animal Study",
    "In Vitro / Cell Study", "Mechanistic / Metabolic Study (Humans)", "Methods / Protocol",
    "Other", "Not stated",
]
INDEXING_TIERS = ["Core", "Supportive", "Specialized", "Archive-only", "Do not index"]
SAFETY_SENSITIVITY = [
    "None", "General nutrition advice", "Medical/disease-specific", "Pediatric/pregnancy",
    "Supplements/medication interactions", "Food safety/allergens", "Other",
]
HARD_EXCLUSION_FLAGS = [
    "Animal-only", "In vitro only", "No dietary exposure studied",
    "No nutrition-related outcomes", "Conference abstract only", "Retracted study", "None",
]


def _s() -> dict:
    # Lengths are checked in Python, not in the grammar: llama.cpp expands
    # {m,n} string bounds into nested rules, which gets expensive fast.
    return {"type": "string", "minLength": 1}


def _enum(values: List[str]) -> dict:
    return {"type": "string", "enum": list(values)}


def _arr(items: dict, lo: int, hi: int) -> dict:
    return {"type": "array", "items": items, "minItems": lo, "maxItems": hi}


def _obj(props: Dict[str, dict]) -> dict:
    # Property order matters: the grammar emits keys in this order, which
    # matches the skeleton in the prompt.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }


def _score() -> dict:
    return {"type": "integer", "minimum": 0, "maximum": 5}


_QA_ITEM = _obj({"question": _s(), "answer": _s(), "grounding": _s()})
_GLOSSARY_ITEM = _obj({"term": _s(), "definition": _s(), "rationale": _s()})

ANNOTATION_SCHEMA = _obj({
    "reader_group": _enum(READER_GROUPS),
    "age_group": _enum(AGE_GROUPS),
    "population_group": _enum(AGE_GROUPS),
    "geographic_context": _obj({
        "country_or_region": _s(),
        "income_setting": _enum(INCOME_SETTINGS),
    }),
    "biological_model": _enum(BIOLOGICAL_MODELS),
    "topics": _arr(_enum(TOPICS), 1, 3),
    "tags": _arr(_s(), 3, 8),
    "study_type": _enum(STUDY_TYPES),
    "evaluation": _obj({
        "user_value_score": _score(),
        "actionability_score": _score(),
        "relevance_score": _score(),
        "verdict": _arr(_s(), 1, 3),
        "indexing_tier": _enum(INDEXING_TIERS),
        "safety_sensitivity": _enum(SAFETY_SENSITIVITY),
        "recommended_user_framing": _s(),
    }),
    "hard_exclusion_flags": _arr(_enum(HARD_EXCLUSION_FLAGS), 1, 6),
    "annotation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "annotations": _obj({
        "abstract": _s(),
        "glosary": _arr(_GLOSSARY_ITEM, 3, 7),
        "user_qa": _arr(_QA_ITEM, 3, 3),
        "expert_qa": _arr(_QA_ITEM, 3, 3),
        "practitioner_qa": _arr(_QA_ITEM, 3, 3),
    }),
})

KEYWORD_SCHEMA = _arr(_s(), 1, 7)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def schema_errors(value: Any, schema: dict, path: str = "$") -> List[str]:
    """Check ``value`` against the subset of JSON Schema used above.

    The server is asked to enforce the same schema, so this normally finds
    nothing. It is here because "normally" is not "always": a server upgrade
    or a proxy that drops ``response_format`` would otherwise go unnoticed.
    """
    errors: List[str] = []
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected an object"]
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in schema["properties"]:
                    errors.append(f"{path}.{key}: unexpected key")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(schema_errors(value[key], sub, f"{path}.{key}"))
    elif kind == "array":
        if not isinstance(value, list):
            return [f"{path}: expected an array"]
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        if lo is not None and len(value) < lo:
            errors.append(f"{path}: needs at least {lo} items, got {len(value)}")
        if hi is not None and len(value) > hi:
            errors.append(f"{path}: allows at most {hi} items, got {len(value)}")
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{i}]"))
    elif kind == "string":
        if not isinstance(value, str):
            return [f"{path}: expected a string"]
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} is not one of the allowed values")
        if len(value.strip()) < schema.get("minLength", 0):
            errors.append(f"{path}: empty")
    elif kind in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path}: expected a number"]
        if kind == "integer" and not float(value).is_integer():
            errors.append(f"{path}: expected an integer")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    return errors


_PLACEHOLDERS = {
    "", "not stated", "n/a", "na", "none", "null", "...", "string", "tbd", "todo",
    "unknown", "-", "term", "question", "answer", "definition", "rationale",
    "grounding", "example", "placeholder", "lorem ipsum",
}
# Reasoning or channel markup that leaked into the payload.
_LEAK_RE = re.compile(r"<\|[^|>]*\|>|assistantfinal|```|</?think\b|</?analysis\b", re.I)


def _words(text: str) -> int:
    return len(text.split())


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_placeholder(text: str) -> bool:
    return _norm(text) in {_norm(p) for p in _PLACEHOLDERS}


def _walk_strings(value: Any, path: str = "$") -> Iterator[Tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")


def validate_keywords(keywords: Any, abstract: str) -> Tuple[List[str], List[str]]:
    errors = schema_errors(keywords, KEYWORD_SCHEMA)
    warnings: List[str] = []
    if errors:
        return errors, warnings
    seen = set()
    for i, kw in enumerate(keywords):
        if _is_placeholder(kw):
            errors.append(f"keyword[{i}] {kw!r} is a placeholder")
        if _LEAK_RE.search(kw):
            errors.append(f"keyword[{i}] contains markup")
        if _words(kw) > 6:
            errors.append(f"keyword[{i}] {kw!r} is longer than 6 words (max 3)")
        elif _words(kw) > 3:
            warnings.append(f"keyword {kw!r} is longer than 3 words")
        key = _norm(kw)
        if key in seen:
            errors.append(f"keyword[{i}] {kw!r} is a duplicate")
        seen.add(key)
    if not errors:
        text = _norm(abstract)
        found = sum(1 for kw in keywords if _norm(kw) and _norm(kw) in text)
        if found < max(1, len(keywords) // 2):
            warnings.append(
                f"only {found}/{len(keywords)} keywords appear verbatim in the abstract"
            )
    return errors, warnings


def validate_annotation(data: Any, abstract: str) -> Tuple[List[str], List[str]]:
    """Schema + content rules for the raw annotation (before normalisation)."""
    errors = schema_errors(data, ANNOTATION_SCHEMA)
    warnings: List[str] = []
    if errors:
        return errors, warnings

    for path, text in _walk_strings(data):
        if _LEAK_RE.search(text):
            errors.append(f"{path}: contains reasoning/markdown markup")

    def unique(values: List[str], path: str) -> None:
        seen = set()
        for v in values:
            key = _norm(v)
            if key in seen:
                errors.append(f"{path}: duplicate value {v!r}")
            seen.add(key)

    unique(data["topics"], "topics")
    unique(data["tags"], "tags")
    unique(data["hard_exclusion_flags"], "hard_exclusion_flags")
    if "None" in data["hard_exclusion_flags"] and len(data["hard_exclusion_flags"]) > 1:
        errors.append('hard_exclusion_flags: "None" cannot be combined with other flags')
    if "Other" in data["topics"] and len(data["topics"]) > 1:
        warnings.append('topics: "Other" listed alongside specific topics')
    for i, tag in enumerate(data["tags"]):
        if _is_placeholder(tag) and tag != "Other":
            errors.append(f"tags[{i}] {tag!r} is a placeholder")
        if _words(tag) > 4:
            errors.append(f"tags[{i}] {tag!r} is longer than 4 words (use 1-3)")

    ev = data["evaluation"]
    for i, bullet in enumerate(ev["verdict"]):
        if _is_placeholder(bullet) or _words(bullet) < 4:
            errors.append(f"evaluation.verdict[{i}] is too short or a placeholder: {bullet!r}")
    framing = ev["recommended_user_framing"]
    if _is_placeholder(framing) or _words(framing) < 8:
        errors.append("evaluation.recommended_user_framing must be 1-2 real sentences")
    elif _words(framing) > 90:
        errors.append("evaluation.recommended_user_framing is longer than 1-2 sentences")

    ann = data["annotations"]
    rewrite = ann["abstract"]
    orig_words = _words(abstract)
    min_words = max(25, min(60, int(orig_words * 0.3)))
    if _is_placeholder(rewrite) or _words(rewrite) < min_words:
        errors.append(
            f"annotations.abstract: the plain-language rewrite needs at least {min_words} "
            f"words, got {_words(rewrite)}"
        )
    elif _norm(rewrite) == _norm(abstract):
        errors.append("annotations.abstract: copied the original abstract instead of rewriting it")
    elif _words(rewrite) > max(400, int(orig_words * 1.6)):
        warnings.append("annotations.abstract: rewrite is much longer than the original")

    terms = []
    for i, entry in enumerate(ann["glosary"]):
        for key in ("term", "definition", "rationale"):
            if _is_placeholder(entry[key]):
                errors.append(f"annotations.glosary[{i}].{key} is a placeholder")
        if _words(entry["definition"]) < 4:
            errors.append(f"annotations.glosary[{i}].definition is too short")
        if _words(entry["rationale"]) < 4:
            errors.append(f"annotations.glosary[{i}].rationale is too short")
        if _words(entry["term"]) > 8:
            errors.append(f"annotations.glosary[{i}].term is a phrase, not a term")
        terms.append(entry["term"])
    unique(terms, "annotations.glosary terms")

    questions = []
    for group in ("user_qa", "expert_qa", "practitioner_qa"):
        for i, qa in enumerate(ann[group]):
            p = f"annotations.{group}[{i}]"
            q, a, g = qa["question"].strip(), qa["answer"].strip(), qa["grounding"].strip()
            if _is_placeholder(q) or _words(q) < 3:
                errors.append(f"{p}.question is too short or a placeholder")
            elif not q.endswith("?"):
                errors.append(f"{p}.question must end with a question mark")
            if _words(q) > 25:
                errors.append(f"{p}.question has {_words(q)} words (max 20)")
            elif _words(q) > 20:
                warnings.append(f"{p}.question has {_words(q)} words (max 20)")
            if _is_placeholder(a) or _words(a) < 5:
                errors.append(f"{p}.answer is too short or a placeholder")
            elif _words(a) > 120:
                errors.append(f"{p}.answer is longer than 1-2 sentences ({_words(a)} words)")
            if _is_placeholder(g) or _words(g) < 3:
                errors.append(f"{p}.grounding is too short or a placeholder")
            questions.append(q)
    unique(questions, "annotations Q&A questions")

    # Cross-field consistency: worth a look, not worth a retry.
    bio, st = data["biological_model"], data["study_type"]
    if bio == "Animal" and st not in ("Animal Study", "Other", "Not stated"):
        warnings.append(f"biological_model=Animal but study_type={st}")
    if st == "Animal Study" and bio not in ("Animal", "Mixed"):
        warnings.append(f"study_type=Animal Study but biological_model={bio}")
    if data["study_type"] == "Not stated":
        warnings.append("study_type is 'Not stated'")
    return errors, warnings


def validate_final(enriched: Dict[str, Any]) -> List[str]:
    """What will actually be written must still be complete after the service's
    normalisation (defaults filling, truncation, ASCII folding)."""
    errors: List[str] = []
    ann = enriched.get("annotations") or {}
    if not (ann.get("abstract") or "").strip():
        errors.append("annotations.abstract became empty after ASCII normalisation")
    for group in ("user_qa", "expert_qa", "practitioner_qa"):
        items = ann.get(group) or []
        if len(items) != 3:
            errors.append(f"annotations.{group}: {len(items)} items after normalisation")
        for i, qa in enumerate(items):
            if not all((qa.get(k) or "").strip() for k in ("question", "answer", "grounding")):
                errors.append(f"annotations.{group}[{i}] has an empty field after ASCII normalisation")
    for i, entry in enumerate(ann.get("glosary") or []):
        if not (entry.get("term") or "").strip():
            errors.append(
                f"annotations.glosary[{i}].term is empty after ASCII normalisation; "
                "write terms in plain ASCII (e.g. 'beta-carotene', not the Greek letter)"
            )
    if len(ann.get("glosary") or []) < 3:
        errors.append("annotations.glosary has fewer than 3 entries after normalisation")
    if not enriched.get("keywords"):
        errors.append("keywords became empty after normalisation")
    if not (enriched.get("evaluation") or {}).get("verdict"):
        errors.append("evaluation.verdict became empty after normalisation")
    return errors


# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------

class LLMFatal(RuntimeError):
    """Configuration/auth problem: retrying cannot help, stop the run."""


class LLMUnavailable(RuntimeError):
    """The model server did not answer after transport-level retries."""


class LLMBadRequest(RuntimeError):
    """The server rejected this particular request (e.g. context too long)."""


@dataclass
class Completion:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


_SIZE_KEYWORDS = ("minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength")


def basic_schema(schema: Any) -> Any:
    """The schema without size/range keywords, for providers whose structured
    output mode rejects them. Python validation still enforces all of them."""
    if isinstance(schema, dict):
        return {k: basic_schema(v) for k, v in schema.items() if k not in _SIZE_KEYWORDS}
    if isinstance(schema, list):
        return [basic_schema(v) for v in schema]
    return schema


class LLMClient:
    """OpenAI-compatible chat completions (GPUStack/llama.cpp, Groq, ...).

    ``schema_mode``: "full" sends the whole JSON schema (llama.cpp enforces
    it as a grammar); "basic" drops size/range keywords; "json_object" only
    asks for a JSON object. Validation in Python is the same in every mode.
    """

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: float,
                 transport_retries: int = 5, schema_mode: str = "full",
                 max_rate_limit_waits: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport_retries = transport_retries
        self.schema_mode = schema_mode
        self.max_rate_limit_waits = max_rate_limit_waits
        self.rate_limited = 0  # 429s seen (shown on the panel)
        self.http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15.0),
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        )

    def probe(self) -> None:
        """Raise unless the server is up and serves the configured model."""
        r = self.http.get("/models", timeout=20)
        if r.status_code in (401, 403):
            raise LLMFatal(f"model server rejected the API key ({r.status_code})")
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", [])]
        if self.model not in ids:
            raise LLMFatal(f"model {self.model!r} is not served here (available: {ids})")

    def chat(self, messages: List[dict], *, schema: dict, schema_name: str,
             max_tokens: int, temperature: float, reasoning_effort: str) -> Completion:
        response_format: Optional[dict]
        if self.schema_mode != "full" and schema.get("type") != "object":
            # OpenAI-style providers (Groq) only take an object at the top
            # level. The keyword call wants a bare array, so it goes without a
            # schema, exactly as the service sends it; validation still applies.
            response_format = None
        elif self.schema_mode == "json_object":
            response_format = {"type": "json_object"}
        else:
            sent = schema if self.schema_mode == "full" else basic_schema(schema)
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": sent},
            }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "seed": random.randint(1, 2**31 - 1),
        }
        if response_format is not None:
            body["response_format"] = response_format
        delay = 2.0
        last: Optional[str] = None
        attempt = 0
        rate_limit_waits = 0
        while attempt < self.transport_retries:
            attempt += 1
            t0 = time.monotonic()
            r = None
            try:
                r = self.http.post("/chat/completions", json=body)
            except httpx.TimeoutException as exc:
                last = f"timeout: {exc}"
            except httpx.TransportError as exc:
                last = f"transport: {exc}"
            else:
                if r.status_code == 429 and rate_limit_waits < self.max_rate_limit_waits:
                    # Rate limited: wait as long as the provider asks, and do
                    # not spend a transport retry on it.
                    rate_limit_waits += 1
                    attempt -= 1
                    try:
                        wait = float(r.headers.get("retry-after") or 0)
                    except ValueError:
                        wait = 0.0
                    wait = min(max(wait, 1.0), 120.0) * (1 + random.random() / 4)
                    self.rate_limited += 1
                    log.info("rate limited (429); waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                if r.status_code in (401, 403):
                    raise LLMFatal(f"model server rejected the API key ({r.status_code}): {r.text[:200]}")
                if r.status_code == 404:
                    raise LLMFatal(f"model endpoint not found (404): {r.text[:200]}")
                if r.status_code in (400, 413, 422):
                    raise LLMBadRequest(f"{r.status_code}: {r.text[:300]}")
                if r.status_code == 200:
                    try:
                        data = r.json()
                        choice = data["choices"][0]
                        msg = choice.get("message") or {}
                        usage = data.get("usage") or {}
                        return Completion(
                            content=msg.get("content") or "",
                            reasoning=msg.get("reasoning_content") or "",
                            finish_reason=choice.get("finish_reason") or "",
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            seconds=time.monotonic() - t0,
                        )
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        last = f"unreadable response: {exc}: {r.text[:200]}"
                else:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
            if attempt < self.transport_retries:
                sleep = delay * (0.75 + random.random() / 2)
                if r is not None and r.status_code == 498:  # Groq: capacity, back off longer
                    sleep = max(sleep, 15.0)
                log.warning("model call failed (%s); retry %d/%d in %.0fs",
                            last, attempt, self.transport_retries - 1, sleep)
                time.sleep(sleep)
                delay = min(delay * 2, 60)
        raise LLMUnavailable(last or "no response")


# ---------------------------------------------------------------------------
# Enrichment (LLM side)
# ---------------------------------------------------------------------------

class ContentFailure(RuntimeError):
    """Every attempt produced output that failed validation."""

    def __init__(self, stage: str, errors: List[str]):
        super().__init__(f"{stage}: " + "; ".join(errors[:6]))
        self.stage = stage
        self.errors = errors


@dataclass
class ArticleText:
    urn: str
    title: str
    abstract: str
    authors: str


@dataclass
class EnrichmentResult:
    enriched: Dict[str, Any]
    raw_keywords: List[str]
    warnings: List[str]
    attempts: List[dict]
    completion_tokens: int
    llm_seconds: float


def _parse_json(content: str, urn: str, array: bool = False) -> Any:
    try:
        return json.loads(content)
    except ValueError:
        # With a schema enforced this is unreachable; without one (the keyword
        # call on Groq) use the service's own recovery parsers.
        if array:
            return parse_json_array(content)
        return EnrichmentAgent._parse_annotation_content(content, urn)


def _repair_message(errors: List[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors[:15])
    return (
        "Your JSON was rejected by validation:\n"
        f"{listed}\n\n"
        "Return the complete corrected JSON object with every key and the same structure. "
        "Fix the listed problems; keep everything else grounded in the abstract. JSON only."
    )


class Enricher:
    def __init__(self, llm: LLMClient, max_attempts: int, stats: "Stats",
                 reasoning_effort: str = "low"):
        self.llm = llm
        self.max_attempts = max_attempts
        self.stats = stats
        self.reasoning_effort = reasoning_effort

    def _run_stage(self, stage: str, base_messages: List[dict], schema: dict,
                   max_tokens: int, check, attempts_log: List[dict]) -> Tuple[Any, List[str], Any]:
        """Call the model until ``check(parsed)`` returns no errors.

        ``check`` returns ``(errors, warnings, product)``. Odd attempts sample
        fresh; even attempts after a rejection send a repair turn with the
        errors. Truncation escalates the budget and samples fresh.
        """
        prev_content: Optional[str] = None
        prev_errors: List[str] = []
        budget = max_tokens
        for attempt in range(1, self.max_attempts + 1):
            temperature = TEMPERATURES[min(attempt, len(TEMPERATURES)) - 1]
            effort = self.reasoning_effort
            repair = prev_content is not None and attempt % 2 == 0
            messages = list(base_messages)
            if repair:
                messages += [
                    {"role": "assistant", "content": prev_content},
                    {"role": "user", "content": _repair_message(prev_errors)},
                ]
            entry = {"stage": stage, "attempt": attempt, "temperature": temperature,
                     "effort": effort, "repair": repair, "max_tokens": budget}
            try:
                comp = self.llm.chat(messages, schema=schema, schema_name=stage,
                                     max_tokens=budget, temperature=temperature,
                                     reasoning_effort=effort)
            except LLMBadRequest as exc:
                entry["error"] = str(exc)
                attempts_log.append(entry)
                # Most likely the context window: fall back to a smaller
                # budget and drop the repair turn.
                prev_content, prev_errors = None, [str(exc)]
                budget = max(4096, budget // 2)
                continue
            self.stats.add_tokens(comp.completion_tokens, comp.seconds)
            entry.update(finish=comp.finish_reason, completion_tokens=comp.completion_tokens,
                         seconds=round(comp.seconds, 2))
            if comp.finish_reason == "length":
                entry["errors"] = ["truncated by the token budget"]
                attempts_log.append(entry)
                prev_content, prev_errors = None, entry["errors"]
                budget = max(budget, ANNOTATION_MAX_TOKENS_ESCALATED if stage == "annotation"
                             else budget * 2)
                continue
            try:
                parsed = _parse_json(comp.content, stage, array=schema.get("type") == "array")
            except ValueError as exc:
                errors, warnings, product = [f"invalid JSON: {exc}"], [], None
            else:
                errors, warnings, product = check(parsed)
            entry["errors"] = errors
            attempts_log.append(entry)
            if not errors:
                return parsed, warnings, product
            log.info("%s attempt %d/%d rejected: %s", stage, attempt, self.max_attempts,
                     "; ".join(errors[:3]))
            prev_content, prev_errors = comp.content, errors
        raise ContentFailure(stage, prev_errors or ["no usable output"])

    def enrich(self, art: ArticleText) -> EnrichmentResult:
        attempts: List[dict] = []
        kw_messages = [
            {"role": "system", "content": ENRICHMENT_KEYWORDS_SYSTEM.compile()},
            {"role": "user", "content": ENRICHMENT_KEYWORDS_USER.compile(abstract=art.abstract)},
        ]

        def check_keywords(parsed):
            errors, warnings = validate_keywords(parsed, art.abstract)
            return errors, warnings, None

        raw_keywords, kw_warnings, _ = self._run_stage(
            "keywords", kw_messages, KEYWORD_SCHEMA, KEYWORD_MAX_TOKENS, check_keywords, attempts)
        # Same as EnrichmentAgent.extract_keywords (+ the [:10] in _generate_keywords).
        keywords = EnrichmentAgent._homogenize_keywords([k for k in raw_keywords if isinstance(k, str)][:10])

        ann_messages = [{
            "role": "user",
            "content": ENRICHMENT_ANNOTATION.compile(
                title=art.title, authors=art.authors, abstract=art.abstract),
        }]

        def check_annotation(parsed):
            errors, warnings = validate_annotation(parsed, art.abstract)
            if errors:
                return errors, warnings, None
            # Same post-processing as EnrichmentAgent.enrich_article.
            enriched = EnrichmentAgent._normalize_enriched(json.loads(json.dumps(parsed)), art.abstract)
            enriched["keywords"] = keywords
            enriched["urn"] = art.urn
            enriched["title"] = art.title
            enriched = EnrichmentAgent._to_ascii_obj(enriched)
            return validate_final(enriched), warnings, enriched

        _, ann_warnings, enriched = self._run_stage(
            "annotation", ann_messages, ANNOTATION_SCHEMA, ANNOTATION_MAX_TOKENS,
            check_annotation, attempts)
        return EnrichmentResult(
            enriched=enriched,
            raw_keywords=raw_keywords,
            warnings=kw_warnings + ann_warnings,
            attempts=attempts,
            completion_tokens=sum(a.get("completion_tokens", 0) for a in attempts),
            llm_seconds=round(sum(a.get("seconds", 0) for a in attempts), 2),
        )


# ---------------------------------------------------------------------------
# Catalog side
# ---------------------------------------------------------------------------

class CatalogFatal(RuntimeError):
    """Credentials rejected or no permission: stop the run."""


class CatalogUnavailable(RuntimeError):
    """The catalog did not answer after retries."""


class ArticleRejected(RuntimeError):
    """The catalog refused this article's request (not found, invalid payload)."""


def _classify_catalog_error(exc: Exception) -> str:
    """'fatal' | 'transient' | 'article'"""
    import requests
    from wisefood import exceptions as wf_exc

    if isinstance(exc, (wf_exc.AuthenticationError, wf_exc.AuthorizationError)):
        return "fatal"
    if isinstance(exc, (wf_exc.NotFoundError, wf_exc.InvalidError, wf_exc.DataError,
                        wf_exc.ConflictError, wf_exc.NotAllowedError)):
        return "article"
    if isinstance(exc, (wf_exc.RateLimitError, wf_exc.InternalError, wf_exc.BadGatewayError,
                        wf_exc.ServiceUnavailableError, wf_exc.GatewayTimeoutError,
                        requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                        requests.exceptions.RetryError)):
        return "transient"
    msg = str(exc).lower()
    if "authentication failed" in msg:
        return "fatal"
    if any(s in msg for s in ("connection", "timed out", "temporarily unavailable", "max retries")):
        return "transient"
    return "article"


class Catalog:
    def __init__(self, base_url: str, *, client_id: str = "", client_secret: str = "",
                 username: str = "", password: str = "", retries: int = 4):
        from wisefood import Credentials

        if client_id and client_secret:
            self.creds = Credentials(client_id=client_id, client_secret=client_secret)
        elif username and password:
            self.creds = Credentials(username=username, password=password)
        else:
            raise CatalogFatal(
                "catalog credentials missing: set CATALOG_CLIENT_ID/CATALOG_CLIENT_SECRET "
                "or CATALOG_USERNAME/CATALOG_PASSWORD")
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self._local = threading.local()

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            from wisefood import DataClient

            client = DataClient(self.base_url, self.creds, default_timeout=60.0)
            self._local.client = client
        return client

    def call(self, what: str, fn):
        """Run ``fn(client)`` with retries on transient failures."""
        delay = 3.0
        for attempt in range(1, self.retries + 1):
            try:
                return fn(self._client())
            except Exception as exc:  # classified below
                kind = _classify_catalog_error(exc)
                if kind == "fatal":
                    raise CatalogFatal(f"{what}: {exc}") from exc
                if kind == "article":
                    raise ArticleRejected(f"{what}: {exc}") from exc
                if attempt == self.retries:
                    raise CatalogUnavailable(f"{what}: {exc}") from exc
                # A broken session can stay broken; start a new one.
                self._local.client = None
                sleep = delay * (0.75 + random.random() / 2)
                log.warning("catalog %s failed (%s); retry %d/%d in %.0fs",
                            what, exc, attempt, self.retries - 1, sleep)
                time.sleep(sleep)
                delay = min(delay * 2, 60)

    def probe(self) -> None:
        self.call("login", lambda c: c.post("articles/search", json={"limit": 1, "fl": ["urn"]}))

    def search(self, fq: List[str], *, limit: int, sort: Optional[str] = None) -> Tuple[List[str], int]:
        payload: Dict[str, Any] = {
            "limit": limit, "offset": 0, "fl": ["urn"], "fq": fq,
            "fields": ["status"], "facet_limit": 1,
        }
        if sort:
            payload["sort"] = sort
        res = self.call("search", lambda c: c.post("articles/search", json=payload).json())
        result = res.get("result", {}) if isinstance(res, dict) else {}
        urns = [r.get("urn") for r in result.get("results", []) if isinstance(r, dict) and r.get("urn")]
        return urns, int(result.get("total") or 0)

    def fetch(self, urn: str):
        return self.call(f"fetch {urn}", lambda c: c.articles.get(urn))


def _qs_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


PENDING_FQ = [f"NOT _exists_:{ENRICHED_MARKER_FIELD}", "NOT status:deleted"]


def iter_pending(catalog: Catalog, extra_fq: List[str], page: int, stop: threading.Event) -> Iterator[str]:
    """Un-enriched URNs in URN order, by keyset pagination.

    Keyset rather than offset paging because the set shrinks while we write:
    offsets would skip articles, and they are capped at the ES result window.
    """
    last: Optional[str] = None
    while not stop.is_set():
        fq = PENDING_FQ + list(extra_fq)
        if last is not None:
            fq.append(f"urn:{{{_qs_quote(last)} TO *]")
        urns, _ = catalog.search(fq, limit=page, sort="urn asc")
        if not urns:
            return
        yield from urns
        last = urns[-1]


ARTICLE_FIELDS_BACKUP = (
    "keywords", "tags", "topics", "hard_exclusion_flags", "reader_group", "age_group",
    "population_group", "geographic_context", "biological_model", "study_type",
    "annotation_confidence", "extras", "ai_tags", "ai_category", "ai_key_takeaways",
    "ai_indexing_tier",
)


def write_enrichment(catalog: Catalog, article, enriched: Dict[str, Any], provenance: dict) -> dict:
    """Write like ``persist_enrichment`` (PATCH, then /enhance), merging extras."""
    enhance_fields, article_fields, extras_fields = extract_enrichment_fields(enriched)
    if enhance_fields:
        extras_fields["enhance_agent"] = ENHANCE_AGENT
        extras_fields["enhance_fields"] = enhance_fields
    extras_fields["enrichment_provenance"] = provenance
    existing = article.data.get("extras")
    merged_extras = {**(existing if isinstance(existing, dict) else {}), **extras_fields}

    def save(_client):
        original_sync = getattr(article, "sync", True)
        try:
            article.sync = False
            for name, value in article_fields.items():
                setattr(article, name, value)
            article.extras = merged_extras
            article.save(only_dirty=True)
        finally:
            article.sync = original_sync

    catalog.call(f"save {article.urn}", save)
    if enhance_fields:
        catalog.call(f"enhance {article.urn}",
                     lambda _c: article.enhance_self(agent=ENHANCE_AGENT, fields=enhance_fields))
    return {"enhance": enhance_fields, "article": article_fields, "extras": extras_fields}


def verify_written(catalog: Catalog, urn: str, written: dict) -> List[str]:
    fresh = catalog.fetch(urn).data
    problems: List[str] = []
    for key in ("study_type", "keywords", "topics", "tags"):
        if key in written["article"] and fresh.get(key) != written["article"][key]:
            problems.append(f"{key} does not match what was written")
    if "ai_category" in written["enhance"] and fresh.get("ai_category") != written["enhance"]["ai_category"]:
        problems.append("ai_category (the enriched marker) was not set")
    extras = fresh.get("extras") or {}
    if extras.get("enriched_at") != written["extras"]["enriched_at"]:
        problems.append("extras.enriched_at does not match this run")
    qa = ((extras.get("annotations") or {}).get("user_qa")) or []
    if len(qa) != 3:
        problems.append("extras.annotations.user_qa missing after write")
    return problems


# ---------------------------------------------------------------------------
# State, stats, progress
# ---------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    """Append-only JSONL records under one directory, safe across threads."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_path = root / "results.jsonl"
        self.enriched_path = root / "enriched.jsonl"
        self.progress_path = root / "progress.json"
        self._lock = threading.Lock()

    def lock_or_exit(self) -> None:
        self._lock_fh = open(self.root / ".lock", "w")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"another run is already using {self.root} (lock held); "
                     "stop it first or use a different --state-dir")
        self._lock_fh.write(str(os.getpid()))
        self._lock_fh.flush()

    def last_status(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if self.results_path.exists():
            with open(self.results_path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        out[rec["urn"]] = rec["status"]
                    except (ValueError, KeyError):
                        continue
        return out

    def append(self, path: Path, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def write_progress(self, snapshot: dict) -> None:
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2))
        tmp.replace(self.progress_path)


class Stats:
    def __init__(self, total_pending: Optional[int], limit: Optional[int]):
        self.lock = threading.Lock()
        self.started = time.time()
        self.total_pending = total_pending
        self.limit = limit
        self.counts = collections.Counter()
        self.recent = collections.deque()  # (finish time) of completed articles
        self.tokens = 0
        self.llm_seconds = 0.0
        self.in_flight = 0
        self.paused: Optional[str] = None
        self.stopping = False
        self.last_failure: Optional[str] = None
        self.attempt_hist = collections.Counter()

    def add_tokens(self, n: int, seconds: float) -> None:
        with self.lock:
            self.tokens += n
            self.llm_seconds += seconds

    def snapshot_tokens(self) -> Tuple[int, float]:
        with self.lock:
            return self.tokens, self.llm_seconds

    def finish(self, status: str, llm_calls: int = 0, failure: Optional[str] = None) -> None:
        with self.lock:
            self.counts[status] += 1
            if status in ("ok", "dry_run_ok", "failed", "dry_run_failed", "skipped"):
                self.recent.append(time.time())
            if llm_calls:
                self.attempt_hist[llm_calls] += 1
            if failure:
                self.last_failure = f"{datetime.now().strftime('%H:%M:%S')} {failure}"

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            while self.recent and now - self.recent[0] > 600:
                self.recent.popleft()
            elapsed = now - self.started
            window = min(600.0, elapsed) or 1.0
            rate = len(self.recent) / window * 60  # articles/min
            done = sum(self.counts.values())
            target = self.limit if self.limit is not None else self.total_pending
            if self.limit is not None and self.total_pending is not None:
                target = min(self.limit, self.total_pending)
            remaining = max(0, target - done) if target is not None else None
            eta = (remaining / rate * 60) if (remaining is not None and rate > 0) else None
            return {
                "updated_at": utcnow(),
                "elapsed_s": round(elapsed),
                "pending_at_start": self.total_pending,
                "target": target,
                "done": done,
                "counts": dict(self.counts),
                "in_flight": self.in_flight,
                "rate_per_min": round(rate, 2),
                "eta_s": round(eta) if eta is not None else None,
                "output_tokens": self.tokens,
                "output_tok_per_s": round(self.tokens / elapsed, 1) if elapsed else 0,
                "paused": self.paused,
                "stopping": self.stopping,
                "last_failure": self.last_failure,
                "llm_calls_per_article": dict(sorted(self.attempt_hist.items())),
            }


def _fmt_dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def progress_line(snap: dict) -> str:
    c = snap["counts"]
    target = snap["target"]
    done = snap["done"]
    pct = f" ({done / target * 100:.1f}%)" if target else ""
    paused = f" | PAUSED: {snap['paused']}" if snap.get("paused") else ""
    return (
        f"{done}/{target if target is not None else '?'}{pct} "
        f"ok={c.get('ok', 0) + c.get('dry_run_ok', 0)} "
        f"failed={c.get('failed', 0) + c.get('dry_run_failed', 0)} "
        f"skipped={c.get('skipped', 0)} deferred={c.get('deferred', 0)} "
        f"in-flight={snap['in_flight']} | {snap['rate_per_min']:.1f}/min "
        f"{snap['output_tok_per_s']:.0f} tok/s | ETA {_fmt_dur(snap['eta_s'])}{paused}"
    )


class Panel:
    """A few terminal lines redrawn in place (for tmux / an interactive shell)."""

    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.drawn = 0

    def render(self, snap: dict, header: str, footer: str) -> None:
        import shutil

        width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
        c = snap["counts"]
        target, done = snap["target"], snap["done"]
        frac = (done / target) if target else 0.0
        bar_w = max(10, min(50, width - 40))
        filled = int(bar_w * min(1.0, frac))
        ok = c.get("ok", 0) + c.get("dry_run_ok", 0)
        failed = c.get("failed", 0) + c.get("dry_run_failed", 0)
        calls = snap.get("llm_calls_per_article") or {}
        first_try = calls.get(2, 0)  # keywords + annotation, both accepted first time
        judged = sum(calls.values())
        if snap.get("stopping"):
            state = "STOPPING: finishing the articles in flight (Ctrl-C again to quit now)"
        elif snap.get("paused"):
            state = f"PAUSED: {snap['paused']} (probing every 30s)"
        else:
            state = "running"
        lines = [
            header,
            f"[{'#' * filled}{'.' * (bar_w - filled)}] {done:,} / {target:,}  {frac * 100:.1f}%"
            if isinstance(target, int) else f"{done:,} done",
            f"ok {ok:,}   failed {failed:,}   skipped {c.get('skipped', 0):,}   "
            f"deferred {c.get('deferred', 0):,}   in flight {snap['in_flight']}",
            f"rate {snap['rate_per_min']:.1f}/min   output {snap['output_tok_per_s']:.0f} tok/s   "
            f"elapsed {_fmt_dur(snap['elapsed_s'])}   ETA {_fmt_dur(snap['eta_s'])}   "
            f"rate-limit waits {snap.get('rate_limited', 0):,}",
            f"passed validation first time: "
            f"{(first_try / judged * 100) if judged else 0:.0f}% of {judged:,} enriched",
            f"status: {state}",
            f"last failure: {snap.get('last_failure') or '-'}",
            footer,
        ]
        lines = [ln if len(ln) <= width else ln[: width - 1] + "~" for ln in lines]
        # Redraw from the top-left of the pane every time. Relative cursor
        # moves break as soon as tmux resizes the pane (attach from another
        # terminal size) and leave duplicated lines behind.
        self.stream.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
        self.stream.flush()
        self.drawn = len(lines)


# ---------------------------------------------------------------------------
# Outage handling
# ---------------------------------------------------------------------------

class Breaker:
    """Shared pause for an upstream that stopped answering.

    The first worker to hit an outage probes the upstream until it answers;
    the others wait. Gives up (stops the run) after ``max_outage`` seconds.
    """

    def __init__(self, name: str, probe, stats: Stats, stop: threading.Event,
                 max_outage: float, interval: float = 30.0):
        self.name = name
        self.probe = probe
        self.stats = stats
        self.stop = stop
        self.max_outage = max_outage
        self.interval = interval
        self.ok = threading.Event()
        self.ok.set()
        self._prober = threading.Lock()
        self.gave_up = False

    def wait(self) -> bool:
        while not self.ok.wait(timeout=5):
            if self.stop.is_set():
                return False
        return not self.stop.is_set()

    def trip(self, reason: str) -> None:
        if not self._prober.acquire(blocking=False):
            self.wait()
            return
        try:
            self.ok.clear()
            started = time.monotonic()
            self.stats.paused = f"{self.name} unavailable since {utcnow()}"
            log.error("%s unavailable (%s); pausing all workers and probing every %.0fs",
                      self.name, reason, self.interval)
            while not self.stop.is_set():
                if self.stop.wait(self.interval):
                    break
                try:
                    self.probe()
                except (LLMFatal, CatalogFatal):
                    raise
                except Exception as exc:  # still down
                    down = time.monotonic() - started
                    log.warning("%s still unavailable after %s: %s", self.name, _fmt_dur(down), exc)
                    if down > self.max_outage:
                        log.error("%s down for more than %s; stopping the run",
                                  self.name, _fmt_dur(self.max_outage))
                        self.gave_up = True
                        self.stop.set()
                        break
                    continue
                log.info("%s is back after %s; resuming", self.name,
                         _fmt_dur(time.monotonic() - started))
                break
        finally:
            self.stats.paused = None
            self.ok.set()
            self._prober.release()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _authors_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(a) for a in value if a)
    return str(value or "")


class Runner:
    def __init__(self, args, llm: LLMClient, catalog: Catalog, state: State):
        self.args = args
        self.llm = llm
        self.catalog = catalog
        self.state = state
        self.stop = threading.Event()
        self.fatal: Optional[str] = None
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        self.stats: Stats  # set in run()
        self.requeue: "queue.Queue[Tuple[str, int]]" = queue.Queue()
        self.panel: Optional[Panel] = Panel() if use_panel(args) else None

    # -- per article -----------------------------------------------------------

    def _record(self, urn: str, status: str, t0: float, **extra) -> None:
        rec = {"urn": urn, "status": status, "run_id": self.run_id, "ts": utcnow(),
               "seconds": round(time.monotonic() - t0, 1), **extra}
        self.state.append(self.state.results_path, rec)
        failure = None
        if status in ("failed", "dry_run_failed", "deferred"):
            failure = f"{urn} [{extra.get('stage', status)}] {'; '.join((extra.get('errors') or [])[:2])}"
        self.stats.finish(status, llm_calls=len(extra.get("attempts") or []), failure=failure)

    def process(self, urn: str, deferrals: int) -> None:
        t0 = time.monotonic()
        dry = self.args.dry_run
        try:
            if not self.catalog_breaker.wait():
                return
            article = self.catalog.fetch(urn)
            data = article.data
            if data.get(ENRICHED_MARKER_FIELD) and not self.args.urn:
                self._record(urn, "skipped", t0, reason="already enriched")
                return
            abstract = (data.get("abstract") or "").strip()
            if _words(abstract) < MIN_ABSTRACT_WORDS:
                self._record(urn, "skipped", t0,
                             reason=f"abstract missing or under {MIN_ABSTRACT_WORDS} words "
                                    f"({_words(abstract)})")
                return
            art = ArticleText(urn=urn, title=(data.get("title") or "").strip(), abstract=abstract,
                              authors=_authors_text(data.get("authors")))

            if not self.llm_breaker.wait():
                return
            result = Enricher(self.llm, self.args.max_attempts, self.stats,
                              self.args.reasoning_effort).enrich(art)
            summary = {
                "attempts": result.attempts,
                "warnings": result.warnings,
                "completion_tokens": result.completion_tokens,
                "llm_seconds": result.llm_seconds,
                "study_type": result.enriched.get("study_type"),
            }
            provenance = {
                "model": self.llm.model,
                "source": "scripts/enrich_backfill.py",
                "run_id": self.run_id,
                "validator": VALIDATOR_VERSION,
                "llm_calls": len(result.attempts),
            }
            before = {k: data.get(k) for k in ARTICLE_FIELDS_BACKUP if k in data}
            self.state.append(self.state.enriched_path, {
                "urn": urn, "run_id": self.run_id, "ts": utcnow(), "dry_run": dry,
                "enriched": result.enriched, "raw_keywords": result.raw_keywords,
                "warnings": result.warnings, "before": before,
            })
            if dry:
                self._record(urn, "dry_run_ok", t0, **summary)
                log.info("DRY  %s  %s  calls=%d warnings=%d", urn, summary["study_type"],
                         len(result.attempts), len(result.warnings))
                return

            if not self.catalog_breaker.wait():
                self.requeue.put((urn, deferrals))
                return
            written = write_enrichment(self.catalog, article, result.enriched, provenance)
            problems = verify_written(self.catalog, urn, written)
            if problems and any("ai_category" in p for p in problems):
                # /enhance is the step the service lets fail silently; retry it once.
                self.catalog.call(f"enhance {urn}", lambda _c: article.enhance_self(
                    agent=ENHANCE_AGENT, fields=written["enhance"]))
                problems = verify_written(self.catalog, urn, written)
            if problems:
                self._record(urn, "failed", t0, stage="verify", errors=problems, **summary)
                log.error("FAIL %s  write not verified: %s", urn, "; ".join(problems))
                return
            self._record(urn, "ok", t0, **summary)
            log.info("OK   %s  %s  calls=%d warnings=%d %.0fs", urn, summary["study_type"],
                     len(result.attempts), len(result.warnings), time.monotonic() - t0)

        except ContentFailure as exc:
            status = "dry_run_failed" if dry else "failed"
            self._record(urn, status, t0, stage=exc.stage, errors=exc.errors[:20])
            log.error("FAIL %s  %s output never passed validation: %s", urn, exc.stage,
                      "; ".join(exc.errors[:3]))
        except ArticleRejected as exc:
            status = "dry_run_failed" if dry else "failed"
            self._record(urn, status, t0, stage="catalog", errors=[str(exc)[:500]])
            log.error("FAIL %s  catalog rejected the request: %s", urn, exc)
        except (LLMUnavailable, CatalogUnavailable) as exc:
            breaker = self.llm_breaker if isinstance(exc, LLMUnavailable) else self.catalog_breaker
            if deferrals < 3:
                self.requeue.put((urn, deferrals + 1))
            else:
                self._record(urn, "deferred", t0, errors=[str(exc)[:300]])
            breaker.trip(str(exc)[:200])
        except (LLMFatal, CatalogFatal) as exc:
            self.fatal = str(exc)
            log.critical("STOPPING: %s", exc)
            self.stop.set()
        except Exception as exc:  # a bug or something unforeseen: keep going, keep the evidence
            status = "dry_run_failed" if dry else "failed"
            self._record(urn, status, t0, stage="unexpected",
                         errors=[f"{type(exc).__name__}: {exc}"[:500]])
            log.exception("FAIL %s  unexpected error", urn)

    # -- orchestration ---------------------------------------------------------

    def _producer(self, work: "queue.Queue[Optional[Tuple[str, int]]]", skip: set) -> None:
        queued = 0
        try:
            if self.args.urn:
                source: Iterator[str] = iter(self.args.urn)
            else:
                source = iter_pending(self.catalog, self.args.fq, self.args.page_size, self.stop)
            for urn in source:
                if self.stop.is_set():
                    break
                if urn in skip:
                    continue
                if self.args.limit is not None and queued >= self.args.limit:
                    break
                while not self.stop.is_set():
                    try:
                        work.put((urn, 0), timeout=2)
                        queued += 1
                        break
                    except queue.Full:
                        continue
        except (CatalogFatal, LLMFatal) as exc:
            self.fatal = str(exc)
            self.stop.set()
        except CatalogUnavailable as exc:
            log.error("could not list pending articles: %s", exc)
            self.catalog_breaker.trip(str(exc))
        finally:
            self.producer_done.set()

    def _worker(self, work) -> None:
        while not self.stop.is_set():
            try:
                urn, deferrals = self.requeue.get_nowait()
            except queue.Empty:
                try:
                    item = work.get(timeout=1)
                except queue.Empty:
                    if self.producer_done.is_set() and self.requeue.empty() and work.empty():
                        return
                    continue
                urn, deferrals = item
            with self.stats.lock:
                self.stats.in_flight += 1
            try:
                self.process(urn, deferrals)
            finally:
                with self.stats.lock:
                    self.stats.in_flight -= 1

    def _snap(self) -> dict:
        snap = self.stats.snapshot()
        snap.update(run_id=self.run_id, dry_run=self.args.dry_run, pid=os.getpid(),
                    concurrency=self.args.concurrency, model=self.llm.model,
                    finished=self.finished.is_set(), fatal=self.fatal,
                    rate_limited=self.llm.rate_limited)
        return snap

    def _reporter(self) -> None:
        """Live panel on a terminal; periodic summary lines otherwise.

        Per-article lines go to run.log only, never to the terminal.
        """
        header = (f"FoodScholar enrichment | {self.llm.model} x{self.args.concurrency} | "
                  f"{'DRY RUN (no writes)' if self.args.dry_run else 'writing to catalog'} | "
                  f"run {self.run_id}")
        footer = f"log {self.state.root / 'run.log'} | Ctrl-C stops gracefully"
        last_file = last_line = 0.0
        while True:
            done = self.finished.wait(1 if self.panel else 5)
            snap = self._snap()
            now = time.monotonic()
            if done or now - last_file >= 5:
                self.state.write_progress(snap)
                last_file = now
            if self.panel:
                self.panel.render(snap, header, footer)
            elif done or now - last_line >= self.args.progress_every:
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {progress_line(snap)}", flush=True)
                last_line = now
            if now - last_line >= 300 and self.panel:
                log.info("PROGRESS %s", progress_line(snap))  # a trail in run.log too
                last_line = now
            if done:
                return

    def run(self) -> int:
        args = self.args
        log.info("run %s | model %s at %s | catalog %s | concurrency %d | %s",
                 self.run_id, self.llm.model, self.llm.base_url, self.catalog.base_url,
                 args.concurrency, "DRY RUN (no catalog writes)" if args.dry_run else "WRITING to the catalog")

        def say(msg: str) -> None:
            log.info("%s", msg)
            if self.panel:
                print(msg, flush=True)

        say("checking the model server and the catalog ...")
        self.llm.probe()
        self.catalog.probe()
        total = None
        if not args.urn:
            _, total = self.catalog.search(PENDING_FQ + list(args.fq), limit=1)
            say(f"un-enriched articles in the catalog: {total:,}")
        else:
            total = len(args.urn)

        previous = self.state.last_status()
        skip = {u for u, s in previous.items() if s in ("ok", "skipped")}
        if not args.retry_failed:
            skip |= {u for u, s in previous.items() if s == "failed"}
        if args.urn:
            skip = set()  # explicit URNs always run
        if previous:
            say(f"state {self.state.root}: {len(previous):,} earlier outcomes; skipping "
                f"{len(skip):,} (ok/skipped{'' if args.retry_failed else '/failed'})")
            if total is not None and not args.urn:
                total = max(0, total - sum(1 for u, s in previous.items()
                                           if s in ("failed", "skipped") and u in skip))

        self.stats = Stats(total, args.limit)
        self.llm_breaker = Breaker("model server", self.llm.probe, self.stats, self.stop, args.max_outage)
        self.catalog_breaker = Breaker("catalog", self.catalog.probe, self.stats, self.stop, args.max_outage)
        self.producer_done = threading.Event()
        self.finished = threading.Event()

        def on_signal(signum, _frame):
            if self.stop.is_set():
                log.warning("second signal: exiting now")
                print("\nexiting now; articles in flight are not written", flush=True)
                os._exit(130)
            log.warning("signal %d: finishing the articles in flight, then stopping "
                        "(signal again to exit immediately)", signum)
            self.stats.stopping = True
            self.stop.set()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        work: "queue.Queue[Optional[Tuple[str, int]]]" = queue.Queue(maxsize=args.concurrency * 2)
        producer = threading.Thread(target=self._producer, args=(work, skip), name="producer", daemon=True)
        workers = [threading.Thread(target=self._worker, args=(work,), name=f"w{i}", daemon=True)
                   for i in range(args.concurrency)]
        reporter = threading.Thread(target=self._reporter, name="reporter", daemon=True)
        reporter.start()
        producer.start()
        for w in workers:
            w.start()
        for w in workers:
            while w.is_alive():
                w.join(timeout=1)
        self.finished.set()
        reporter.join(timeout=10)

        snap = self.stats.snapshot()
        summary = [
            f"FINISHED {progress_line(snap)}",
            f"results: {self.state.results_path}",
            f"enriched payloads + rollback data: {self.state.enriched_path}",
        ]
        if self.fatal:
            summary.append(f"STOPPED because: {self.fatal}")
        elif self.llm_breaker.gave_up or self.catalog_breaker.gave_up:
            summary.append("STOPPED: an upstream stayed unavailable longer than --max-outage")
        for line in summary:
            log.info("%s", line)
            if self.panel:
                print(line, flush=True)
        if self.fatal:
            return 3
        if self.llm_breaker.gave_up or self.catalog_breaker.gave_up:
            return 4
        return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

SELFTEST_ARTICLE = ArticleText(
    urn="urn:article:selftest",
    title="Effect of a Mediterranean-style diet on blood pressure and LDL cholesterol in adults "
          "with metabolic syndrome: a 12-week randomized controlled trial",
    authors="A. Example, B. Sample, C. Placeholder",
    abstract=(
        "Background: Dietary patterns rich in vegetables, legumes, whole grains and olive oil "
        "have been associated with lower cardiovascular risk, but trial evidence in people with "
        "metabolic syndrome is limited. Methods: We randomly assigned 214 adults aged 40-70 years "
        "with metabolic syndrome in two outpatient clinics in southern Spain to a Mediterranean-style "
        "diet supplemented with 30 g/day of mixed nuts and extra-virgin olive oil (n=108) or to "
        "advice on a low-fat diet (n=106) for 12 weeks. The primary outcome was change in systolic "
        "blood pressure; secondary outcomes were LDL cholesterol, fasting glucose and body weight. "
        "Results: Adherence was high in both groups. Compared with the low-fat group, the "
        "Mediterranean-style group reduced systolic blood pressure by 4.1 mmHg (95% CI 1.8 to 6.4) "
        "and LDL cholesterol by 0.21 mmol/L (95% CI 0.05 to 0.37). Fasting glucose and body weight "
        "did not differ between groups. No serious adverse events were reported. Conclusions: In "
        "adults with metabolic syndrome, a Mediterranean-style diet with nuts and olive oil modestly "
        "lowered blood pressure and LDL cholesterol over 12 weeks. Longer trials are needed to "
        "establish effects on cardiovascular events."
    ),
)


def cmd_selftest(args, llm: LLMClient) -> int:
    """Enrich a built-in sample abstract (synthetic, no catalog access)."""
    llm.probe()
    stats = Stats(args.repeat, None)
    failures = 0
    results: List[EnrichmentResult] = []
    lock = threading.Lock()

    def one(i: int) -> None:
        nonlocal failures
        t0 = time.monotonic()
        try:
            res = Enricher(llm, args.max_attempts, stats, args.reasoning_effort).enrich(SELFTEST_ARTICLE)
        except Exception as exc:  # report and count
            with lock:
                failures += 1
            log.error("selftest #%d FAILED: %s", i, exc)
            return
        with lock:
            results.append(res)
        log.info("selftest #%d ok in %.0fs: %d model calls, %d output tokens, warnings=%s",
                 i, time.monotonic() - t0, len(res.attempts), res.completion_tokens, res.warnings)

    t0 = time.monotonic()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(1, args.repeat + 1)]
    sem = threading.Semaphore(args.concurrency)

    def gated(t: threading.Thread) -> None:
        with sem:
            t.run()

    runners = [threading.Thread(target=gated, args=(t,)) for t in threads]
    for r in runners:
        r.start()
    for r in runners:
        r.join()
    wall = time.monotonic() - t0
    tokens, _ = stats.snapshot_tokens()
    log.info("selftest: %d/%d ok in %.0fs | %.0f output tok/s aggregate | %.1f articles/min",
             len(results), args.repeat, wall, tokens / wall if wall else 0,
             len(results) / wall * 60 if wall else 0)
    if results and args.show:
        print(json.dumps(results[0].enriched, indent=2, ensure_ascii=False))
    return 0 if failures == 0 else 1


def cmd_count(args, catalog: Catalog) -> int:
    catalog.probe()
    _, pending = catalog.search(PENDING_FQ + list(args.fq), limit=1)
    _, total = catalog.search(["NOT status:deleted"] + list(args.fq), limit=1)
    print(json.dumps({"total": total, "enriched": total - pending, "pending": pending}, indent=2))
    return 0


def cmd_status(args) -> int:
    state = State(Path(args.state_dir))
    if state.progress_path.exists():
        snap = json.loads(state.progress_path.read_text())
        alive = False
        pid = snap.get("pid")
        if pid and not snap.get("finished"):
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                alive = False
        print(f"run {snap.get('run_id')} ({'RUNNING' if alive else 'not running'}"
              f"{', DRY RUN' if snap.get('dry_run') else ''}), last update {snap.get('updated_at')}")
        print("  " + progress_line(snap))
        if snap.get("fatal"):
            print(f"  stopped because: {snap['fatal']}")
    else:
        print(f"no progress file in {state.root}")
    last = state.last_status()
    if last:
        counts = collections.Counter(last.values())
        print(f"all runs in this state dir: {dict(counts)} ({len(last)} articles)")
        failed = [u for u, s in last.items() if s in ("failed", "deferred")]
        if failed:
            print(f"  failed/deferred (latest {min(10, len(failed))}): " + ", ".join(failed[-10:]))
            print("  details: grep the URN in " + str(state.results_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dir", default=os.getenv("ENRICH_STATE_DIR",
                                                    str(Path.home() / "foodscholar-enrich" / "state")))
    p.add_argument("--log-level", default=os.getenv("ENRICH_LOG_LEVEL", "INFO"))
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_opts(sp):
        sp.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
        sp.add_argument("--llm-model", default=os.getenv("LLM_MODEL"))
        sp.add_argument("--llm-timeout", type=float, default=float(os.getenv("LLM_TIMEOUT", "600")),
                        help="seconds per model call (default 600)")
        sp.add_argument("--max-attempts", type=int, default=int(os.getenv("ENRICH_MAX_ATTEMPTS", "5")),
                        help="model attempts per stage before the article fails (default 5)")
        sp.add_argument("--concurrency", type=int, default=int(os.getenv("ENRICH_CONCURRENCY", "8")))
        sp.add_argument("--reasoning-effort", default=os.getenv("LLM_REASONING_EFFORT", "low"),
                        help="gpt-oss reasoning effort; 'low' is its minimum (default low)")
        sp.add_argument("--schema-mode", choices=("full", "basic", "json_object"),
                        default=os.getenv("LLM_SCHEMA_MODE", "full"),
                        help="what structured-output request to send: full JSON schema (llama.cpp), "
                             "basic (no size/range keywords) or json_object; validation is identical")

    def fq_opt(sp):
        sp.add_argument("--fq", action="append", default=[],
                        help="extra catalog filter (Elasticsearch query_string), repeatable, "
                             'e.g. --fq \'venue:"Nutrients"\'')
        sp.add_argument("--fq-file", default=os.getenv("ENRICH_FQ_FILE") or None,
                        help="file with one extra filter clause per line (# comments allowed), "
                             "e.g. the list of journals to enrich")

    run = sub.add_parser("run", help="enrich un-enriched catalog articles")
    llm_opts(run)
    fq_opt(run)
    run.add_argument("--dry-run", action="store_true", help="enrich and validate, do not write")
    run.add_argument("--limit", type=int, default=None, help="stop after this many articles")
    run.add_argument("--urn", action="append", default=[], help="only these URNs (repeatable)")
    run.add_argument("--retry-failed", action="store_true",
                     help="also retry articles that failed in earlier runs")
    run.add_argument("--page-size", type=int, default=100)
    run.add_argument("--progress-every", type=float, default=30.0,
                     help="seconds between progress lines when not on a terminal")
    run.add_argument("--no-panel", action="store_true",
                     help="print progress lines instead of the live panel")
    run.add_argument("--max-outage", type=float, default=1800.0,
                     help="stop if the model server or catalog stays down this long (s)")

    st = sub.add_parser("selftest", help="enrich a built-in sample abstract; no catalog access")
    llm_opts(st)
    st.add_argument("--repeat", type=int, default=1)
    st.add_argument("--show", action="store_true", help="print the enriched JSON")

    cnt = sub.add_parser("count", help="count total / enriched / pending articles")
    fq_opt(cnt)

    sub.add_parser("status", help="show progress of the current/last run")
    return p


def use_panel(args) -> bool:
    return (getattr(args, "cmd", None) == "run" and not getattr(args, "no_panel", False)
            and sys.stdout.isatty())


def _setup_logging(level: str, logfile: Optional[Path], console: bool = True) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-4s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)] if console else []
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    log.setLevel(level.upper())
    # Library chatter (and the Groq pool's missing-key warning) is noise here.
    for noisy in ("httpx", "httpcore", "urllib3", "backend.groq", "backend.langfuse"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _llm_from_args(args) -> LLMClient:
    key = os.getenv("LLM_API_KEY", "")
    if not (args.llm_base_url and args.llm_model and key):
        sys.exit("LLM_BASE_URL, LLM_MODEL and LLM_API_KEY must be set")
    return LLMClient(args.llm_base_url, args.llm_model, key, timeout=args.llm_timeout,
                     schema_mode=args.schema_mode)


def _catalog_from_env() -> Catalog:
    return Catalog(
        os.getenv("CATALOG_URL", "https://demo.wisefood-project.eu/dc"),
        client_id=os.getenv("CATALOG_CLIENT_ID", ""),
        client_secret=os.getenv("CATALOG_CLIENT_SECRET", ""),
        username=os.getenv("CATALOG_USERNAME", ""),
        password=os.getenv("CATALOG_PASSWORD", ""),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "fq_file", None):
        clauses = [ln.strip() for ln in Path(args.fq_file).read_text().splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
        args.fq = list(args.fq) + clauses
    if args.cmd == "status":
        _setup_logging(args.log_level, None)
        return cmd_status(args)

    if args.cmd == "run":
        state = State(Path(args.state_dir))
        state.lock_or_exit()
        # With the live panel the terminal shows only the panel; details go to run.log.
        _setup_logging(args.log_level, state.root / "run.log", console=not use_panel(args))
    else:
        _setup_logging(args.log_level, None)

    def fail(code: int, msg: str) -> int:
        log.critical("%s", msg)
        if use_panel(args):
            print(msg, file=sys.stderr, flush=True)
        return code

    try:
        if args.cmd == "selftest":
            return cmd_selftest(args, _llm_from_args(args))
        if args.cmd == "count":
            return cmd_count(args, _catalog_from_env())
        runner = Runner(args, _llm_from_args(args), _catalog_from_env(), state)
        return runner.run()
    except (LLMFatal, CatalogFatal) as exc:
        return fail(3, f"STOPPED: {exc}")
    except (LLMUnavailable, CatalogUnavailable, httpx.HTTPError) as exc:
        return fail(4, f"upstream unavailable at start: {exc}")


if __name__ == "__main__":
    sys.exit(main())
