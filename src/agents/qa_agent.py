"""Question Answering agent for non-contextual Q&A with optional RAG.

The citation/context machinery lives as module-level functions so both the
legacy ``QAAgent`` (blocking, JSON-object answers) and the streaming pipeline
(``services.qa_pipeline.answering``) share one implementation of source
formatting, citation building, and quote coercion.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import ChatPromptTemplate

from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.model_output import normalize_model_text
from agents.json_output import parse_json_object
from backend.prompts import (
    QA_ANSWER_RAG_SYSTEM,
    QA_ANSWER_RAG_USER,
    QA_ANSWER_NORAG_SYSTEM,
    QA_ANSWER_NORAG_USER,
)
from models.qa import QAAnswer, QACitation, DEFAULT_GROQ_MODEL

logger = logging.getLogger(__name__)

# Register specifications per expertise level. These are deliberately DIVERGENT
# (not one "technicality dial") so beginner and expert answers read as written for
# genuinely different audiences — different vocabulary, framing, structure, and what
# is included vs omitted. See Claire/RCSI feedback: the two modes must be clearly
# differentiated, not the same answer with more/fewer words.
COMPLEXITY_INSTRUCTIONS = {
    "beginner": (
        "Write for a curious non-specialist with no science background.\n"
        "- Vocabulary: everyday words. If a technical term is unavoidable, define it in plain language the first time, in parentheses.\n"
        "- Framing: start from what the reader already experiences (food, meals, the body) and connect the science to that.\n"
        "- Use a concrete analogy or everyday example to make each key idea land.\n"
        "- Structure: short sentences, one idea at a time; lead with the practical takeaway, then the 'why'.\n"
        "- Omit: study designs, effect sizes, statistics, mechanism-level detail, and citations-as-argument. Reassure rather than caveat-heavily.\n"
        "- Tone: warm, encouraging, and confident without oversimplifying to the point of being wrong."
    ),
    "intermediate": (
        "Write for an informed general reader comfortable with basic science.\n"
        "- Use clear scientific language; define complex terms when first introduced.\n"
        "- Balance the practical takeaway with a brief, honest sense of the evidence behind it.\n"
        "- Structure: readable prose with the conclusion first, then supporting reasoning."
    ),
    "expert": (
        "Write for a nutrition/food-science professional or researcher.\n"
        "- Vocabulary: precise scientific and clinical terminology; do NOT define common domain terms or use analogies — assume fluency.\n"
        "- Framing: foreground mechanism, methodology, and the strength and shape of the evidence.\n"
        "- Where the sources support it, be specific about study designs (RCT, cohort, meta-analysis), effect sizes/direction, and statistical or clinical significance.\n"
        "- Structure: dense, information-first prose; no hand-holding, no motivational framing, no restating the question.\n"
        "- Surface limitations, heterogeneity, and open questions the way you would for a knowledgeable colleague."
    ),
}


def format_prior_conversation(prior_conversation: Optional[str]) -> str:
    """Render the running thread summary as a prompt block, or empty when absent.

    Mode-agnostic on purpose: this carries the FACTS of the conversation so a
    free-form follow-up ("what about for kids?") resolves, while expertise level
    and wording register stay decided per answer. Returns "" when there is no
    prior context so the ``{{prior_conversation}}`` slot collapses to nothing.
    """
    summary = (prior_conversation or "").strip()
    if not summary:
        return ""
    return (
        "\nPRIOR CONVERSATION (running summary of earlier turns in this thread; "
        "use it to interpret follow-up questions, and avoid needlessly repeating "
        "facts already given):\n"
        f"{summary}\n"
    )


# Kept for callers importing the previous private name.
_format_prior_conversation = format_prior_conversation


def is_guideline_source(source: Dict[str, Any]) -> bool:
    if source.get("source_type") == "guideline":
        return True
    return bool(source.get("rule_text")) and not source.get("abstract")


def join_field_values(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    if isinstance(value, str):
        return value.strip()
    return ""


def get_source_text(source: Dict[str, Any]) -> str:
    if is_guideline_source(source):
        text = source.get("rule_text") or source.get("abstract") or ""
    else:
        text = source.get("abstract") or source.get("description") or ""
    return text if isinstance(text, str) else ""


def default_source_section(source: Dict[str, Any]) -> str:
    return "rule_text" if is_guideline_source(source) else "abstract"


def coerce_quote_to_source_span(
    quote: Any, source_text: str, question: Optional[str] = None
) -> Optional[str]:
    """
    Ensure a quote is an exact substring of the provided source text.

    If the LLM returns a quote that differs only in whitespace/casing, attempt to
    recover the exact matching span from the source. If no quote is provided
    or no match can be found, fall back to a best-effort sentence from the source.
    """
    if not source_text:
        return None

    if isinstance(quote, str):
        candidate = quote.strip()
    else:
        candidate = ""

    if candidate and candidate in source_text:
        return candidate

    if candidate:
        # Try to match the candidate even if whitespace differs, then return
        # the exact span as it appears in the source for highlighting.
        tokens = candidate.split()
        if len(tokens) >= 3:
            pattern = r"\s+".join(re.escape(tok) for tok in tokens)
            match = re.search(pattern, source_text)
            if match:
                return match.group(0)
            match = re.search(pattern, source_text, flags=re.IGNORECASE)
            if match:
                return match.group(0)

    # Best-effort fallback: pick the most question-relevant sentence.
    return best_effort_quote_from_source(
        source_text=source_text, question=question
    )


def best_effort_quote_from_source(
    source_text: str, question: Optional[str]
) -> Optional[str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", source_text.strip())
        if s.strip()
    ]
    if not sentences:
        return None

    if not question:
        return sentences[0]

    q = question.lower()
    q_terms = {t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2}
    if not q_terms:
        return sentences[0]

    def score_sentence(sentence: str) -> int:
        s_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(q_terms & s_terms)

    best = max(sentences, key=score_sentence)
    # Keep the excerpt reasonably short while remaining an exact substring.
    words = best.split()
    if len(words) > 60:
        first_n = words[:60]
        pattern = r"\s+".join(re.escape(tok) for tok in first_n)
        match = re.search(pattern, best)
        if match:
            return match.group(0).strip()
        # If matching fails for any reason, prefer returning the full sentence
        # (still an exact substring) over returning a normalized variant.
    return best


def _format_age_range(min_months: Any, max_months: Any) -> str:
    """Human form of the enrichment age window; '' when not stated (-1/None)."""

    def _months(value: Any) -> Optional[int]:
        try:
            months = int(value)
        except (TypeError, ValueError):
            return None
        return months if months >= 0 else None

    low, high = _months(min_months), _months(max_months)
    if low is None and high is None:
        return ""

    def _label(months: int) -> str:
        if months < 24 and months % 12 != 0:
            return f"{months} months"
        years = months // 12
        return f"{years} year{'s' if years != 1 else ''}"

    if low is not None and high is not None:
        return f"{_label(low)} to {_label(high)}"
    if low is not None:
        return f"{_label(low)} and up"
    return f"up to {_label(high)}"


def _format_citation_counts(article: Dict[str, Any]) -> str:
    """'123 (influential: 4)' from either field spelling; '' when unstored."""

    def _count(*keys: str) -> Optional[int]:
        for key in keys:
            value = article.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    citations = _count("citationCount", "citation_count")
    if citations is None:
        return ""
    influential = _count("influentialCitationCount", "influential_citation_count")
    if influential:
        return f"{citations} (influential: {influential})"
    return str(citations)


def prepare_source_context(
    articles: List[Dict[str, Any]],
    retriever: str = "rag",
) -> str:
    """Format retrieved RAG sources for the LLM context window."""
    summaries = []
    g_counter = 1
    for idx, article in enumerate(articles, 1):
        if is_guideline_source(article):
            rule_text = get_source_text(article)
            food_groups = join_field_values(article.get("food_groups"))
            target_populations = join_field_values(
                article.get("target_populations")
            )
            g_label = f"G{g_counter}"
            g_counter += 1
            summary = f"""Guideline {idx} [{g_label}]:
- Source Type: guideline
- Retriever: {article.get('retriever', retriever)}
- URN: {article.get('urn', article.get('id', article.get('_id', 'N/A')))}
- Guide URN: {article.get('guide_urn', 'N/A')}
- Region: {article.get('guide_region', 'N/A')}
- Food Groups: {food_groups or 'N/A'}
- Target Populations: {target_populations or 'N/A'}
- Section: rule_text
- Rule Text: {rule_text}"""

            # Applicability facets, when enrichment filled them: the model
            # needs these to judge (and disclose) WHO a rule is for.
            life_stage = join_field_values(article.get("life_stage"))
            if life_stage:
                summary += f"\n- Life Stage: {life_stage}"
            age_range = _format_age_range(
                article.get("age_min_months"), article.get("age_max_months")
            )
            if age_range:
                summary += f"\n- Applies To Ages: {age_range}"
            nutrients = join_field_values(article.get("nutrients"))
            if nutrients:
                summary += f"\n- Nutrients: {nutrients}"
            health_conditions = join_field_values(article.get("health_conditions"))
            if health_conditions:
                summary += f"\n- Health Conditions: {health_conditions}"
            action = join_field_values(article.get("action_type"))
            frequency = join_field_values(article.get("frequency"))
            if action or frequency:
                summary += (
                    f"\n- Action/Frequency: {action or 'N/A'}"
                    f" / {frequency or 'N/A'}"
                )

            notes = article.get("notes")
            if isinstance(notes, str) and notes.strip():
                summary += f"\n- Notes: {notes.strip()[:500]}"

            summaries.append(summary)
            continue

        authors = article.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += "..."

        abstract = (
            article.get("abstract")
            or article.get("description")
            or "No abstract available"
        )

        summary = f"""Article {idx}:
- Source Type: article
- Retriever: {article.get('retriever', retriever)}
- URN: {article.get('urn', 'N/A')}
- Title: {article.get('title', 'N/A')}
- Authors: {author_str}
- Year: {article.get('publication_year', 'N/A')}
- Journal: {article.get('venue', 'N/A')}
- Study Type: {article.get('ai_category', 'N/A')}
- Abstract: {abstract}"""

        # Bibliometrics, when stored: lets the model weigh well-established
        # vs unproven findings the way rule 9 asks it to.
        citation_line = _format_citation_counts(article)
        if citation_line:
            summary += f"\n- Citations: {citation_line}"

        ai_key_takeaways = article.get("ai_key_takeaways", [])
        if ai_key_takeaways:
            takeaways = "; ".join(ai_key_takeaways)
            summary += f"\n- Key Takeaways: {takeaways}"

        summaries.append(summary)
    return "\n\n".join(summaries)


def format_answer_context(
    *,
    retriever: str,
    user_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Format retrieval and user context for the answer-formulation prompt."""
    context = user_context or {}
    parts = [f"- Retriever: {retriever}"]

    if retriever == "linearrag":
        parts.append(
            "- Evidence shape: graph/passage retrieval; each source text may be a passage rather than a full abstract."
        )
    elif retriever == "rag":
        parts.append(
            "- Evidence shape: Elastic RAG; sources may mix scientific article abstracts and dietary guideline rules."
        )
    else:
        parts.append(
            "- Evidence shape: no retrieved evidence; do not create citations."
        )

    region = context.get("region")
    country = context.get("country")
    experience_group = context.get("experience_group")
    member_age_group = context.get("member_age_group")
    if country or region:
        parts.append(
            f"- User geography: country={country or 'unknown'}, region={region or 'unknown'}."
        )
    if experience_group:
        parts.append(f"- Experience group: {experience_group}.")
    if member_age_group:
        parts.append(f"- Member age group: {member_age_group}.")

    profile = context.get("profile") if isinstance(context, dict) else None
    if isinstance(profile, dict):
        dietary_groups = profile.get("dietary_groups") or []
        allergies = profile.get("allergies") or []
        if dietary_groups:
            parts.append(f"- Dietary groups: {', '.join(map(str, dietary_groups[:5]))}.")
        if allergies:
            parts.append(f"- Allergies: {', '.join(map(str, allergies[:5]))}.")

    safety = context.get("safety") if isinstance(context, dict) else None
    if isinstance(safety, dict):
        risk_level = safety.get("risk_level")
        flags = safety.get("flags") or []
        guardrails = safety.get("guardrails") or []
        if risk_level:
            parts.append(f"- Safety risk level: {risk_level}.")
        if flags:
            parts.append(f"- Safety flags: {', '.join(map(str, flags[:6]))}.")
        for guardrail in guardrails[:4]:
            parts.append(f"- Guardrail: {guardrail}")

    scout = context.get("retrieval_scout") if isinstance(context, dict) else None
    if isinstance(scout, dict):
        status = scout.get("status") or {}
        regions = scout.get("guideline_regions") or []
        source_count = scout.get("source_count")
        if isinstance(status, dict):
            article_hits = status.get("article_hits")
            guideline_hits = status.get("guideline_hits")
            parts.append(
                f"- Retrieval scout: article_hits={article_hits}, guideline_hits={guideline_hits}."
            )
        if regions:
            parts.append(
                f"- Guideline regions found: {', '.join(map(str, regions[:6]))}."
            )
        if source_count is not None:
            parts.append(f"- Sources available for formulation: {source_count}.")

    # Agentic pipeline extras: the sub-question plan (with rationales) and the
    # final evaluator verdict, so the model can disclose gaps honestly.
    plan = context.get("retrieval_reasoning") if isinstance(context, dict) else None
    if isinstance(plan, dict):
        for sq in (plan.get("sub_questions") or [])[:6]:
            if isinstance(sq, dict) and sq.get("text"):
                why = f" (why: {sq['why']})" if sq.get("why") else ""
                parts.append(f"- Searched: {sq['text']}{why}")
        verdict = plan.get("verdict")
        if verdict and verdict != "sufficient":
            parts.append(f"- Evidence assessment: {verdict}.")
        for gap in (plan.get("gaps") or [])[:4]:
            parts.append(f"- Evidence gap to disclose: {gap}")

    return "\n".join(parts)


def create_source_citation(
    source: Dict[str, Any],
    section: str,
    quote: Optional[str] = None,
    confidence: str = "medium",
) -> QACitation:
    """Create a type-aware QA citation from retrieved source metadata."""
    raw_year = source.get("publication_year") or source.get("year")
    year = None
    if raw_year:
        try:
            year = int(str(raw_year)[:4])
        except (ValueError, TypeError):
            pass

    source_type = "guideline" if is_guideline_source(source) else "article"
    source_id = source.get("urn") or source.get("id") or source.get("_id") or ""
    source_title = source.get("title") or (
        "Dietary guideline" if source_type == "guideline" else "Unknown Title"
    )
    source_url = (
        f"/guidelines/{source_id}"
        if source_type == "guideline"
        else f"/articles/{source_id}"
    )

    authors = source.get("authors")
    if isinstance(authors, str):
        authors = [authors]

    # Guide-routing hints (guideline sources only): the UI needs these to land
    # on the guide page with the rule highlighted even when the rule itself is
    # outside the reader's visibility — same contract as TipEvidence.
    guide_urn = None
    region = None
    page_no = None
    if source_type == "guideline":
        authors = None
        raw_guide_urn = source.get("guide_urn")
        if isinstance(raw_guide_urn, str) and raw_guide_urn.strip():
            guide_urn = raw_guide_urn.strip()
        raw_region = source.get("guide_region")
        if isinstance(raw_region, str) and raw_region.strip():
            region = raw_region.strip()
        raw_page = source.get("page_no")
        if isinstance(raw_page, (int, float)) and not isinstance(raw_page, bool):
            page_no = int(raw_page)

    return QACitation(
        source_type=source_type,
        source_id=source_id,
        source_title=source_title,
        source_url=source_url,
        authors=authors,
        year=year,
        venue=source.get("venue")
        or source.get("journal")
        or source.get("guide_region")
        or source.get("country"),
        section=section,
        quote=quote,
        confidence=confidence,
        relevance_score=source.get("relevance_score") or source.get("_score"),
        guide_urn=guide_urn,
        region=region,
        page_no=page_no,
    )


def build_qa_answer(
    parsed: Dict[str, Any],
    *,
    question: str,
    articles: Optional[List[Dict[str, Any]]] = None,
    rag_used: bool = True,
    model_used: str,
) -> QAAnswer:
    """Convert parsed LLM JSON into a QAAnswer model."""
    citations = []
    if articles and rag_used:
        source_lookup: Dict[str, Dict[str, Any]] = {}
        for source in articles:
            for key in ("urn", "id", "_id", "guide_urn"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    source_lookup[value.strip()] = source

        # Pre-compute G-labels in source list order so they match the context
        g_label_map: Dict[str, str] = {}
        g_counter = 1
        for source in articles:
            if is_guideline_source(source):
                source_id = (
                    source.get("urn") or source.get("id") or source.get("_id") or ""
                )
                if source_id:
                    g_label_map[source_id] = f"G{g_counter}"
                g_counter += 1

        cited_sources = parsed.get("cited_sources")
        if not isinstance(cited_sources, list):
            cited_sources = parsed.get("cited_articles", [])

        for cited in cited_sources:
            if not isinstance(cited, dict):
                continue
            urn = cited.get("urn", "")
            if not isinstance(urn, str):
                continue
            source = source_lookup.get(urn.strip())
            if source:
                source_text = get_source_text(source)
                quote = coerce_quote_to_source_span(
                    cited.get("quote"), source_text, question=question
                )
                citation = create_source_citation(
                    source,
                    section=cited.get(
                        "section",
                        default_source_section(source),
                    ),
                    quote=quote,
                    confidence=cited.get("confidence", "medium"),
                )
                citation.display_label = g_label_map.get(citation.source_id)
                citations.append(citation)

    return QAAnswer(
        answer=parsed.get("answer", ""),
        citations=citations,
        confidence=parsed.get("overall_confidence", "medium"),
        model_used=model_used,
        rag_used=rag_used,
        sources_consulted=len(articles) if articles else 0,
        articles_consulted=len(articles) if articles else 0,
    )


class QAAgent:
    """Agent for answering food science questions with optional RAG context."""

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.3,
        trace_context: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = GROQ_CHAT.get_client(model=model, temperature=temperature)
        # Optional Langfuse trace context: {session_id, user_id}. Applied to every
        # LLM invocation so traces are grouped by conversation and attributed to a
        # user. Safe no-op when Langfuse is disabled.
        self.trace_context = trace_context or {}

    def generate_answer_with_rag(
        self,
        question: str,
        articles: List[Dict[str, Any]],
        expertise_level: str = "intermediate",
        language: str = "en",
        retriever: str = "rag",
        user_context: Optional[Dict[str, Any]] = None,
        prior_conversation: Optional[str] = None,
    ) -> Tuple[QAAnswer, List[str]]:
        """
        Generate an answer using retrieved articles as context (RAG mode).

        Args:
            question: User's question
            articles: Retrieved articles from kNN search
            expertise_level: beginner/intermediate/expert
            language: ISO 639-1 language code
            retriever: Retrieval strategy that produced the sources
            user_context: Optional country/member context for personalization

        Returns:
            Tuple of (QAAnswer with citations, follow-up suggestions)
        """
        logger.info(
            "Generating RAG answer for: '%s' (%d articles, model=%s)",
            question[:80], len(articles), self.model,
        )

        source_context = prepare_source_context(articles, retriever=retriever)
        answer_context = format_answer_context(
            retriever=retriever,
            user_context=user_context,
        )
        complexity = COMPLEXITY_INSTRUCTIONS.get(
            expertise_level, COMPLEXITY_INSTRUCTIONS["intermediate"]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_ANSWER_RAG_SYSTEM.langchain()),
            ("human", QA_ANSWER_RAG_USER.langchain()),
        ])

        parsed = self._invoke_and_parse(
            prompt,
            variables={
                "expertise_level": expertise_level,
                "complexity": complexity,
                "language": language,
                "answer_context": answer_context,
                "prior_conversation": format_prior_conversation(prior_conversation),
                "question": question,
                "source_context": source_context,
            },
            run_name="qa-answer-rag",
            tags=["qa", "answer", "rag"],
        )
        answer = self._build_qa_answer(
            parsed, question=question, articles=articles, rag_used=True
        )
        follow_ups = parsed.get("follow_ups", [])
        return answer, follow_ups

    def generate_answer_without_rag(
        self,
        question: str,
        expertise_level: str = "intermediate",
        language: str = "en",
        user_context: Optional[Dict[str, Any]] = None,
        prior_conversation: Optional[str] = None,
    ) -> Tuple[QAAnswer, List[str]]:
        """
        Generate an answer using only LLM parametric knowledge (no retrieval).

        Args:
            question: User's question
            expertise_level: beginner/intermediate/expert
            language: ISO 639-1 language code
            user_context: Optional country/member context for personalization

        Returns:
            Tuple of (QAAnswer with no article citations, follow-up suggestions)
        """
        logger.info(
            "Generating no-RAG answer for: '%s' (model=%s)",
            question[:80], self.model,
        )

        complexity = COMPLEXITY_INSTRUCTIONS.get(
            expertise_level, COMPLEXITY_INSTRUCTIONS["intermediate"]
        )
        answer_context = format_answer_context(
            retriever="no_rag",
            user_context=user_context,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_ANSWER_NORAG_SYSTEM.langchain()),
            ("human", QA_ANSWER_NORAG_USER.langchain()),
        ])

        parsed = self._invoke_and_parse(
            prompt,
            variables={
                "expertise_level": expertise_level,
                "complexity": complexity,
                "language": language,
                "answer_context": answer_context,
                "prior_conversation": format_prior_conversation(prior_conversation),
                "question": question,
            },
            run_name="qa-answer-norag",
            tags=["qa", "answer", "no_rag"],
        )
        answer = self._build_qa_answer(
            parsed, question=question, articles=None, rag_used=False
        )
        follow_ups = parsed.get("follow_ups", [])
        return answer, follow_ups

    def _format_answer_context(
        self,
        *,
        retriever: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return format_answer_context(
            retriever=retriever, user_context=user_context
        )

    def _prepare_article_context(
        self,
        articles: List[Dict[str, Any]],
        retriever: str = "rag",
    ) -> str:
        return prepare_source_context(articles, retriever=retriever)

    def _invoke_and_parse(
        self,
        prompt: ChatPromptTemplate,
        variables: Optional[Dict[str, Any]] = None,
        run_name: str = "qa-answer",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Invoke the LLM and parse JSON response.

        ``variables`` are substituted into the prompt template (registry-backed
        prompts carry their placeholders); omit for already-formatted prompts.
        ``run_name``/``tags`` name and tag the resulting Langfuse trace.
        """
        config = build_trace_config(
            run_name=run_name,
            session_id=self.trace_context.get("session_id"),
            user_id=self.trace_context.get("user_id"),
            tags=tags,
        )
        try:
            response = self.llm.invoke(
                prompt.format_messages(**(variables or {})), config=config
            )
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.error("Error invoking LLM: %s", e, exc_info=True)
            return {
                "answer": "Unable to generate an answer at this time. Please try again.",
                "overall_confidence": "low",
                "follow_ups": [],
            }

    def _parse_llm_response(self, content: Any) -> Dict[str, Any]:
        """Parse the answer JSON out of raw model output.

        Delegates to the shared recovery parser so this path tolerates exactly
        what the enrichment paths tolerate — fences, prose around the payload,
        trailing commas, content blocks, leaked reasoning. A model whose output
        needs recovery must not be the difference between an answer and the
        "unable to parse" placeholder.
        """
        try:
            return parse_json_object(content)
        except ValueError as e:
            logger.error("JSON parsing error: %s", e)
            logger.error(
                "Content (first 1000 chars): %s",
                normalize_model_text(content)[:1000],
            )
            return {
                "answer": "Unable to parse response. Please try again.",
                "overall_confidence": "low",
                "follow_ups": [],
            }

    def _build_qa_answer(
        self,
        parsed: Dict[str, Any],
        question: str,
        articles: Optional[List[Dict[str, Any]]] = None,
        rag_used: bool = True,
    ) -> QAAnswer:
        return build_qa_answer(
            parsed,
            question=question,
            articles=articles,
            rag_used=rag_used,
            model_used=self.model,
        )

    def _create_source_citation(
        self,
        source: Dict[str, Any],
        section: str,
        quote: Optional[str] = None,
        confidence: str = "medium",
    ) -> QACitation:
        return create_source_citation(
            source, section, quote=quote, confidence=confidence
        )

    @staticmethod
    def _is_guideline_source(source: Dict[str, Any]) -> bool:
        return is_guideline_source(source)

    @staticmethod
    def _join_field_values(value: Any) -> str:
        return join_field_values(value)

    @staticmethod
    def _get_source_text(source: Dict[str, Any]) -> str:
        return get_source_text(source)

    @staticmethod
    def _default_source_section(source: Dict[str, Any]) -> str:
        return default_source_section(source)

    @staticmethod
    def _coerce_quote_to_source_span(
        quote: Any, source_text: str, question: Optional[str] = None
    ) -> Optional[str]:
        return coerce_quote_to_source_span(quote, source_text, question=question)

    @staticmethod
    def _best_effort_quote_from_source(
        source_text: str, question: Optional[str]
    ) -> Optional[str]:
        return best_effort_quote_from_source(source_text, question)
