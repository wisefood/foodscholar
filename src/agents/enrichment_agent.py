"""
Enrichment agent for scientific articles.
Runs keyword extraction, homogenization, and full article enrichment.
"""

import json
import logging
import unicodedata
import copy
from typing import List, Dict, Any
from collections import Counter, defaultdict

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import (
    ENRICHMENT_ANNOTATION,
    ENRICHMENT_KEYWORDS_SYSTEM,
    ENRICHMENT_KEYWORDS_USER,
)

logger = logging.getLogger(__name__)


_DEFAULT_ANNOTATION_OUTPUT: Dict[str, Any] = {
    "reader_group": "Not stated",
    "age_group": "Not stated",
    "population_group": "Not stated",
    "geographic_context": {"country_or_region": "Not stated", "income_setting": "Not stated"},
    "biological_model": "Not stated",
    "topics": ["Other"],
    "tags": ["Other"],
    "study_type": "Not stated",
    "evaluation": {
        "user_value_score": 0,
        "actionability_score": 0,
        "relevance_score": 0,
        "verdict": ["Not stated"],
        "indexing_tier": "Archive-only",
        "safety_sensitivity": "None",
        "recommended_user_framing": "Not stated",
    },
    "hard_exclusion_flags": ["None"],
    "annotation_confidence": 0.0,
    "annotations": {
        "original_abstract": "",
        "abstract": "",
        "glosary": [],
        "user_qa": [],
        "expert_qa": [],
        "practitioner_qa": [],
    },
}


class EnrichmentAgent:
    """
    Agent for enriching scientific articles with annotations and metadata.

    This agent processes scientific articles to generate:
    - Keywords extracted from the abstract
    - Study type classification
    - User value and actionability scores
    - Simplified abstract rewrites for general audiences
    - Glossary of technical terms
    - Q&A pairs for different expertise levels

    Uses two LLM models:
    - A smaller model for keyword extraction (fast, cheap)
    - A larger model for full annotation (more capable)
    """

    def __init__(
        self,
        keyword_model: str = "openai/gpt-oss-20b",
        annotation_model: str = "openai/gpt-oss-20b",
        temperature: float = 0.0,
    ):
        """
        Initialize the enrichment agent with LLM clients.

        Args:
            keyword_model: Model ID for keyword extraction (default: gpt-oss-20b)
            annotation_model: Model ID for full annotation (default: gpt-oss-20b)
            temperature: LLM temperature for deterministic output (default: 0.0)
        """
        self.keyword_llm = GROQ_CHAT.get_client(
            model=keyword_model, temperature=temperature
        )
        self.annotation_llm = GROQ_CHAT.get_client(
            model=annotation_model, temperature=temperature
        )

    @staticmethod
    def _homogenize_keywords(keywords: List[str]) -> List[str]:
        """
        Normalize keyword variants to a single representative form.

        Groups keywords by uppercase form and selects the most common variant
        as the representative. This ensures consistent keyword representation
        across articles (e.g., "Omega-3" vs "omega-3" -> most common form).

        Args:
            keywords: List of raw keywords from extraction

        Returns:
            List of homogenized keywords with consistent casing
        """
        if not keywords:
            return []

        # Group keywords by uppercase form to find variants
        groups = defaultdict(list)
        for k in keywords:
            groups[k.upper()].append(k)

        # Select most common variant for each group
        representative = {
            key: Counter(variants).most_common(1)[0][0]
            for key, variants in groups.items()
        }

        return [representative[k.upper()] for k in keywords]

    def _generate_keywords(self, abstract: str) -> List[str]:
        """
        Extract keywords from article abstract using LLM.

        Uses a structured prompt to extract up to 10 representative keywords
        that describe the main topics and nutritional content of the article.

        Args:
            abstract: Article abstract text

        Returns:
            List of extracted keyword strings (max 10)
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ENRICHMENT_KEYWORDS_SYSTEM.langchain()),
                ("human", ENRICHMENT_KEYWORDS_USER.langchain()),
            ]
        )

        response = (prompt | self.keyword_llm).invoke(
            {"abstract": abstract},
            config=build_trace_config(
                run_name="enrichment-keywords",
                tags=["enrichment", "keywords"],
            ),
        )

        try:
            data = json.loads(response.content)
            return [k for k in data if isinstance(k, str)][:10]
        except Exception:
            logger.error(f"Keyword JSON invalid: {response.content}")
            return []

    def extract_keywords(self, abstract: str) -> List[str]:
        """
        Extract and homogenize keywords from an abstract.

        Combines keyword extraction and homogenization into a single operation.

        Args:
            abstract: Article abstract text

        Returns:
            List of homogenized keywords
        """
        raw_keywords = self._generate_keywords(abstract)
        return self._homogenize_keywords(raw_keywords)

    # ============================
    # ASCII normalization (UNCHANGED)
    # ============================

    DASH_MAP = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }

    PUNCT_MAP = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }

    @classmethod
    def _to_ascii_text(cls, s: str) -> str:
        if s is None:
            return s
        s = unicodedata.normalize("NFKC", s)
        for k, v in cls.DASH_MAP.items():
            s = s.replace(k, v)
        for k, v in cls.PUNCT_MAP.items():
            s = s.replace(k, v)
        s = s.encode("ascii", "ignore").decode("ascii")
        return " ".join(s.split())

    @classmethod
    def _to_ascii_obj(cls, obj):
        if isinstance(obj, str):
            return cls._to_ascii_text(obj)
        if isinstance(obj, list):
            return [cls._to_ascii_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {cls._to_ascii_text(k): cls._to_ascii_obj(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _deep_fill_defaults(defaults: Any, data: Any) -> Any:
        """
        Fill missing keys in a nested structure without overwriting provided values.

        Smaller models sometimes omit optional-looking keys; this normalizes the output.
        """
        if data is None:
            return copy.deepcopy(defaults)

        if isinstance(defaults, dict) and isinstance(data, dict):
            out = dict(data)
            for k, v in defaults.items():
                if k not in out:
                    out[k] = copy.deepcopy(v)
                else:
                    out[k] = EnrichmentAgent._deep_fill_defaults(v, out[k])
            return out

        return data

    @classmethod
    def _normalize_enriched(cls, enriched: Dict[str, Any], original_abstract: str) -> Dict[str, Any]:
        enriched = cls._deep_fill_defaults(_DEFAULT_ANNOTATION_OUTPUT, enriched)

        annotations = enriched.get("annotations") if isinstance(enriched.get("annotations"), dict) else {}
        if not annotations.get("original_abstract"):
            annotations["original_abstract"] = original_abstract or ""
        enriched["annotations"] = annotations

        age_group = enriched.get("age_group")
        population_group = enriched.get("population_group")
        if (not age_group or age_group == "Not stated") and population_group and population_group != "Not stated":
            enriched["age_group"] = population_group
        if (not population_group or population_group == "Not stated") and age_group and age_group != "Not stated":
            enriched["population_group"] = age_group

        glosary = annotations.get("glosary")
        if isinstance(glosary, str) and glosary.strip():
            glosary = [{"term": glosary.strip(), "definition": "Not stated", "rationale": "Not stated"}]
        if isinstance(glosary, list):
            normalized_glosary = []
            for entry in glosary:
                if isinstance(entry, str):
                    term = entry.strip()
                    if term:
                        normalized_glosary.append(
                            {"term": term, "definition": "Not stated", "rationale": "Not stated"}
                        )
                    continue
                if isinstance(entry, dict):
                    term = entry.get("term", "")
                    if isinstance(term, str):
                        term = term.strip()
                    else:
                        term = ""
                    if not term:
                        continue
                    definition = entry.get("definition", "Not stated")
                    if not isinstance(definition, str) or not definition.strip():
                        definition = "Not stated"
                    rationale = entry.get("rationale", "Not stated")
                    if not isinstance(rationale, str) or not rationale.strip():
                        rationale = "Not stated"
                    normalized_glosary.append(
                        {"term": term, "definition": definition.strip(), "rationale": rationale.strip()}
                    )
            annotations["glosary"] = normalized_glosary[:7]
        else:
            annotations["glosary"] = []
        enriched["annotations"] = annotations

        topics = enriched.get("topics")
        if isinstance(topics, str) and topics.strip():
            topics = [topics.strip()]
        if not isinstance(topics, list) or len(topics) == 0:
            topics = ["Other"]
        enriched["topics"] = topics[:3]

        tags = enriched.get("tags")
        if isinstance(tags, str) and tags.strip():
            tags = [tags.strip()]
        if not isinstance(tags, list):
            tags = []
        cleaned_tags = []
        for t in tags:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if not tt:
                continue
            if tt not in cleaned_tags:
                cleaned_tags.append(tt)
        if len(cleaned_tags) == 0:
            cleaned_tags = ["Other"]
        enriched["tags"] = cleaned_tags[:8]

        hard_flags = enriched.get("hard_exclusion_flags")
        if not isinstance(hard_flags, list) or len(hard_flags) == 0:
            enriched["hard_exclusion_flags"] = ["None"]

        evaluation = enriched.get("evaluation") if isinstance(enriched.get("evaluation"), dict) else {}
        verdict = evaluation.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            verdict = [verdict.strip()]
        if not isinstance(verdict, list) or len(verdict) == 0:
            evaluation["verdict"] = ["Not stated"]
        else:
            evaluation["verdict"] = verdict[:3]
        enriched["evaluation"] = evaluation

        try:
            conf = float(enriched.get("annotation_confidence", 0.0))
        except Exception:
            conf = 0.0
        enriched["annotation_confidence"] = max(0.0, min(1.0, conf))

        return enriched

    def enrich_article(self, article) -> Dict[str, Any]:
        """
        Enrich a scientific article with annotations, keywords, and Q&A.

        Performs full enrichment including:
        - Keyword extraction and homogenization
        - Study type classification
        - User value and actionability scoring
        - Simplified abstract rewrite
        - Glossary of key terms
        - Q&A for different audience levels (user, expert, practitioner)

        Args:
            article: Article object with title, abstract, authors, and urn attributes

        Returns:
            Dictionary containing all enrichment data, ASCII-normalized
        """
        # Extract and normalize keywords from abstract
        keywords = self.extract_keywords(article.abstract)

        # Build the annotation chain for full article enrichment
        enrichment_chain = (
            PromptTemplate(
                input_variables=["title", "abstract", "authors"],
                template=ENRICHMENT_ANNOTATION.langchain(),
            )
            | self.annotation_llm
            | JsonOutputParser()
        )

        # Generate enrichment annotations
        enriched = enrichment_chain.invoke(
            {
                "title": article.title,
                "abstract": article.abstract,
                "authors": article.authors,
            },
            config=build_trace_config(
                run_name="enrichment-annotation",
                tags=["enrichment", "annotation"],
            ),
        )

        enriched = self._normalize_enriched(enriched, article.abstract)

        # Add metadata fields
        enriched["keywords"] = keywords
        enriched["urn"] = article.urn
        enriched["title"] = article.title

        # Normalize to ASCII for consistent storage/display
        return self._to_ascii_obj(enriched)
