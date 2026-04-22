"""Question Answering agent for non-contextual Q&A with optional RAG."""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import ChatPromptTemplate

from backend.groq import GROQ_CHAT
from models.qa import QAAnswer, QACitation, DEFAULT_GROQ_MODEL

logger = logging.getLogger(__name__)

COMPLEXITY_INSTRUCTIONS = {
    "beginner": "Use simple, accessible language. Explain technical terms. Use analogies where helpful.",
    "intermediate": "Use clear scientific language. Define complex terms when first introduced.",
    "expert": "Use precise scientific terminology. Focus on methodology and statistical significance.",
}


class QAAgent:
    """Agent for answering food science questions with optional RAG context."""

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = GROQ_CHAT.get_client(model=model, temperature=temperature)

    def generate_answer_with_rag(
        self,
        question: str,
        articles: List[Dict[str, Any]],
        expertise_level: str = "intermediate",
        language: str = "en",
        retriever: str = "rag",
        user_context: Optional[Dict[str, Any]] = None,
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

        source_context = self._prepare_article_context(articles, retriever=retriever)
        answer_context = self._format_answer_context(
            retriever=retriever,
            user_context=user_context,
        )
        complexity = COMPLEXITY_INSTRUCTIONS.get(
            expertise_level, COMPLEXITY_INSTRUCTIONS["intermediate"]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Your task is to answer the user's question concisely and accurately using ONLY the provided retrieved sources as evidence. Sources may include scientific article abstracts and dietary guideline rules.

EXPERTISE LEVEL: {expertise_level}
{complexity}

LANGUAGE: Respond in {language}.

ANSWER FORMULATION CONTEXT:
{answer_context}

CRITICAL RULES:
1. Answer CONCISELY - aim for 2-4 paragraphs maximum.
2. Every factual claim MUST cite at least one retrieved source using a markdown link.
3. For article sources, cite as [First Author et al. (Year)](/articles/ARTICLE_URN). Use the first author's surname from the article metadata, followed by "et al." if there are multiple authors. Single-author articles: [Lee (2020)](/articles/URN).
4. For guideline sources, cite using the short label shown in brackets next to the source heading, e.g. [G1](/guidelines/GUIDELINE_URN), [G2](/guidelines/GUIDELINE_URN). Never use the full rule text as the link label.
5. If the retrieved sources do not contain sufficient information, say so explicitly.
6. Do NOT fabricate information beyond what the retrieved sources support.
7. Prefer dietary guideline rules for practical intake recommendations; use articles for study-specific mechanisms or evidence.
8. LinearRAG sources are passage-level snippets. Only cite them when the provided passage itself supports the claim.
9. If the user's country/region is known, prefer country- or region-specific guidance when the retrieved evidence supports it; otherwise state that the answer is general.
10. Clearly indicate when findings are preliminary vs well-established.
11. If sources disagree, present both perspectives.
12. For each cited source, include a "quote" field containing the EXACT verbatim passage from that source that best supports your answer to the user's question. For articles, quote from the abstract or passage text. For guidelines, quote from rule_text. The quote MUST be copied directly from the provided source text (no paraphrasing). Keep it short (ideally 1-2 sentences, <= 60 words).

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object.
Ensure all strings are properly escaped (use \\n for newlines, \\" for quotes).

JSON structure:
{{{{
  "answer": "Markdown-formatted concise answer with inline citations as markdown links",
  "cited_sources": [
    {{{{
      "urn": "the source URN",
      "section": "abstract or rule_text",
      "quote": "verbatim excerpt from the source supporting the answer",
      "confidence": "high"
    }}}}
  ],
  "overall_confidence": "high",
  "follow_ups": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}}}

IMPORTANT: Return ONLY the JSON object."""),
            ("human", f"""Question: {question}

Retrieved Sources:
{source_context}

Answer the question concisely using the sources above as evidence."""),
        ])

        parsed = self._invoke_and_parse(prompt)
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
        answer_context = self._format_answer_context(
            retriever="no_rag",
            user_context=user_context,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Answer the user's question using your training knowledge.

EXPERTISE LEVEL: {expertise_level}
{complexity}

LANGUAGE: Respond in {language}.

ANSWER FORMULATION CONTEXT:
{answer_context}

CRITICAL RULES:
1. Answer CONCISELY - aim for 2-4 paragraphs maximum.
2. Be honest about uncertainty. Use hedging language when appropriate.
3. Since no specific articles are provided, do NOT fabricate citations or article references.
4. Mention general knowledge sources where applicable (e.g., "according to WHO guidelines").
5. Clearly distinguish between well-established facts and emerging research.
6. If the user's country/region is known, localize the answer only when you can do so safely; otherwise say the guidance may vary by country.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object.

{{{{
  "answer": "Markdown-formatted concise answer",
  "overall_confidence": "high or medium or low",
  "follow_ups": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}}}"""),
            ("human", f"""Question: {question}

Answer the question concisely using your scientific knowledge."""),
        ])

        parsed = self._invoke_and_parse(prompt)
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

        return "\n".join(parts)

    def _prepare_article_context(
        self,
        articles: List[Dict[str, Any]],
        retriever: str = "rag",
    ) -> str:
        """Format retrieved RAG sources for the LLM context window."""
        summaries = []
        g_counter = 1
        for idx, article in enumerate(articles, 1):
            if self._is_guideline_source(article):
                rule_text = self._get_source_text(article)
                food_groups = self._join_field_values(article.get("food_groups"))
                target_populations = self._join_field_values(
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

            ai_key_takeaways = article.get("ai_key_takeaways", [])
            if ai_key_takeaways:
                takeaways = "; ".join(ai_key_takeaways)
                summary += f"\n- Key Takeaways: {takeaways}"

            summaries.append(summary)
        return "\n\n".join(summaries)

    def _invoke_and_parse(self, prompt: ChatPromptTemplate) -> Dict[str, Any]:
        """Invoke the LLM and parse JSON response."""
        try:
            response = self.llm.invoke(prompt.format_messages())
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.error("Error invoking LLM: %s", e, exc_info=True)
            return {
                "answer": "Unable to generate an answer at this time. Please try again.",
                "overall_confidence": "low",
                "follow_ups": [],
            }

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling code blocks and control chars."""
        content = content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Clean control characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error: %s", e)
            logger.error("Content (first 1000 chars): %s", content[:1000])
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
                if self._is_guideline_source(source):
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
                    source_text = self._get_source_text(source)
                    quote = self._coerce_quote_to_source_span(
                        cited.get("quote"), source_text, question=question
                    )
                    citation = self._create_source_citation(
                        source,
                        section=cited.get(
                            "section",
                            self._default_source_section(source),
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
            model_used=self.model,
            rag_used=rag_used,
            sources_consulted=len(articles) if articles else 0,
            articles_consulted=len(articles) if articles else 0,
        )

    def _create_source_citation(
        self,
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

        source_type = "guideline" if self._is_guideline_source(source) else "article"
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
        if source_type == "guideline":
            authors = None

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
        )

    @staticmethod
    def _is_guideline_source(source: Dict[str, Any]) -> bool:
        if source.get("source_type") == "guideline":
            return True
        return bool(source.get("rule_text")) and not source.get("abstract")

    @staticmethod
    def _join_field_values(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        if isinstance(value, str):
            return value.strip()
        return ""

    @staticmethod
    def _get_source_text(source: Dict[str, Any]) -> str:
        if QAAgent._is_guideline_source(source):
            text = source.get("rule_text") or source.get("abstract") or ""
        else:
            text = source.get("abstract") or source.get("description") or ""
        return text if isinstance(text, str) else ""

    @staticmethod
    def _default_source_section(source: Dict[str, Any]) -> str:
        return "rule_text" if QAAgent._is_guideline_source(source) else "abstract"

    @staticmethod
    def _coerce_quote_to_source_span(
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
                pattern = r"\\s+".join(re.escape(tok) for tok in tokens)
                match = re.search(pattern, source_text)
                if match:
                    return match.group(0)
                match = re.search(pattern, source_text, flags=re.IGNORECASE)
                if match:
                    return match.group(0)

        # Best-effort fallback: pick the most question-relevant sentence.
        return QAAgent._best_effort_quote_from_source(
            source_text=source_text, question=question
        )

    @staticmethod
    def _best_effort_quote_from_source(
        source_text: str, question: Optional[str]
    ) -> Optional[str]:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\\s+", source_text.strip())
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
            pattern = r"\\s+".join(re.escape(tok) for tok in first_n)
            match = re.search(pattern, best)
            if match:
                return match.group(0).strip()
            # If matching fails for any reason, prefer returning the full sentence
            # (still an exact substring) over returning a normalized variant.
        return best
