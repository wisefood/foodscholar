"""Question Answering agent for non-contextual Q&A with optional RAG."""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import ChatPromptTemplate

from backend.groq import GROQ_CHAT
from models.qa import QAAnswer, DEFAULT_GROQ_MODEL
from utilities.citation_validator import create_citation_from_article

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
    ) -> Tuple[QAAnswer, List[str]]:
        """
        Generate an answer using retrieved articles as context (RAG mode).

        Args:
            question: User's question
            articles: Retrieved articles from kNN search
            expertise_level: beginner/intermediate/expert
            language: ISO 639-1 language code

        Returns:
            Tuple of (QAAnswer with citations, follow-up suggestions)
        """
        logger.info(
            "Generating RAG answer for: '%s' (%d articles, model=%s)",
            question[:80], len(articles), self.model,
        )

        article_context = self._prepare_article_context(articles)
        complexity = COMPLEXITY_INSTRUCTIONS.get(
            expertise_level, COMPLEXITY_INSTRUCTIONS["intermediate"]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Your task is to answer the user's question concisely and accurately using ONLY the provided article abstracts as evidence.

EXPERTISE LEVEL: {expertise_level}
{complexity}

LANGUAGE: Respond in {language}.

CRITICAL RULES:
1. Answer CONCISELY - aim for 2-4 paragraphs maximum.
2. Every factual claim MUST cite at least one article by its URN.
3. If the articles do not contain sufficient information, say so explicitly.
4. Do NOT fabricate information beyond what the articles support.
5. Clearly indicate when findings are preliminary vs well-established.
6. If articles disagree, present both perspectives.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object.
Ensure all strings are properly escaped (use \\n for newlines, \\" for quotes).

JSON structure:
{{
  "answer": "Markdown-formatted concise answer with inline citations referencing article URNs",
  "cited_articles": [
    {{
      "urn": "the article URN",
      "section": "abstract",
      "confidence": "high"
    }}
  ],
  "overall_confidence": "high",
  "follow_ups": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}

IMPORTANT: Return ONLY the JSON object."""),
            ("human", f"""Question: {question}

Retrieved Articles:
{article_context}

Answer the question concisely using the articles above as evidence."""),
        ])

        parsed = self._invoke_and_parse(prompt)
        answer = self._build_qa_answer(parsed, articles=articles, rag_used=True)
        follow_ups = parsed.get("follow_ups", [])
        return answer, follow_ups

    def generate_answer_without_rag(
        self,
        question: str,
        expertise_level: str = "intermediate",
        language: str = "en",
    ) -> Tuple[QAAnswer, List[str]]:
        """
        Generate an answer using only LLM parametric knowledge (no retrieval).

        Args:
            question: User's question
            expertise_level: beginner/intermediate/expert
            language: ISO 639-1 language code

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

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Answer the user's question using your training knowledge.

EXPERTISE LEVEL: {expertise_level}
{complexity}

LANGUAGE: Respond in {language}.

CRITICAL RULES:
1. Answer CONCISELY - aim for 2-4 paragraphs maximum.
2. Be honest about uncertainty. Use hedging language when appropriate.
3. Since no specific articles are provided, do NOT fabricate citations or article references.
4. Mention general knowledge sources where applicable (e.g., "according to WHO guidelines").
5. Clearly distinguish between well-established facts and emerging research.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object.

{{
  "answer": "Markdown-formatted concise answer",
  "overall_confidence": "high or medium or low",
  "follow_ups": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}"""),
            ("human", f"""Question: {question}

Answer the question concisely using your scientific knowledge."""),
        ])

        parsed = self._invoke_and_parse(prompt)
        answer = self._build_qa_answer(parsed, articles=None, rag_used=False)
        follow_ups = parsed.get("follow_ups", [])
        return answer, follow_ups

    def _prepare_article_context(self, articles: List[Dict[str, Any]]) -> str:
        """Format retrieved articles for the LLM context window."""
        summaries = []
        for idx, article in enumerate(articles, 1):
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
- URN: {article.get('urn', 'N/A')}
- Title: {article.get('title', 'N/A')}
- Authors: {author_str}
- Year: {article.get('publication_year', 'N/A')}
- Journal: {article.get('venue', 'N/A')}
- Abstract: {abstract[:600]}"""
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
        articles: Optional[List[Dict[str, Any]]] = None,
        rag_used: bool = True,
    ) -> QAAnswer:
        """Convert parsed LLM JSON into a QAAnswer model."""
        citations = []
        if articles and rag_used:
            article_lookup = {a.get("urn", ""): a for a in articles}
            for cited in parsed.get("cited_articles", []):
                urn = cited.get("urn", "")
                if urn in article_lookup:
                    citation = create_citation_from_article(
                        article_lookup[urn],
                        section=cited.get("section", "abstract"),
                        confidence=cited.get("confidence", "medium"),
                    )
                    citations.append(citation)

        return QAAnswer(
            answer=parsed.get("answer", ""),
            citations=citations,
            confidence=parsed.get("overall_confidence", "medium"),
            model_used=self.model,
            rag_used=rag_used,
            articles_consulted=len(articles) if articles else 0,
        )
