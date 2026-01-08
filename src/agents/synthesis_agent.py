"""Synthesis agent for multi-document search summarization."""
import logging
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from backend.groq import GROQ_CHAT
from models.search import (
    Citation,
    SynthesizedFinding,
    SearchSummaryResponse,
    ArticleMetadata,
)
from utilities.citation_validator import create_citation_from_article
from datetime import datetime

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Agent for synthesizing information from multiple scientific articles."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
    ):
        """
        Initialize synthesis agent.

        Args:
            model: Groq model to use
            temperature: Model temperature (lower = more focused)
        """
        self.llm = GROQ_CHAT.get_client(model=model, temperature=temperature)

    def synthesize_search_results(
        self,
        query: str,
        articles: List[Dict[str, Any]],
        expertise_level: str = "intermediate",
        language: str = "en",
    ) -> SearchSummaryResponse:
        """
        Synthesize search results from multiple articles.

        Args:
            query: Original search query
            articles: List of article data dicts
            expertise_level: User expertise level
            language: Target language for summary

        Returns:
            SearchSummaryResponse with synthesized findings
        """
        logger.info(
            f"Synthesizing {len(articles)} articles for query: '{query}'"
        )

        # Prepare article summaries for the LLM
        article_summaries = self._prepare_article_summaries(articles)

        # Generate synthesis
        synthesis_result = self._generate_synthesis(
            query, article_summaries, expertise_level, language
        )

        # Extract citations and findings from the result
        findings = self._extract_findings(synthesis_result, articles)
        all_citations = self._collect_all_citations(findings)

        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(query, findings)

        return SearchSummaryResponse(
            query=query,
            summary=synthesis_result.get("summary", ""),
            key_findings=findings,
            total_articles_analyzed=len(articles),
            all_citations=all_citations,
            search_metadata={"expertise_level": expertise_level, "language": language},
            generated_at=datetime.now().isoformat(),
            cache_hit=False,
            follow_up_suggestions=follow_ups,
        )

    def _prepare_article_summaries(
        self, articles: List[Dict[str, Any]]
    ) -> str:
        """Prepare article summaries for the LLM."""
        summaries = []

        for idx, article in enumerate(articles, 1):
            summary = f"""
Article {idx}:
- URN: {article.get('urn', 'N/A')}
- Title: {article.get('title', 'N/A')}
- Authors: {', '.join(article.get('authors', [])[:3])}{'...' if len(article.get('authors', [])) > 3 else ''}
- Year: {article.get('year', 'N/A')}
- Journal: {article.get('journal', 'N/A')}
- Abstract: {article.get('abstract', 'No abstract available')[:500]}...
"""
            summaries.append(summary)

        return "\n\n".join(summaries)

    def _generate_synthesis(
        self,
        query: str,
        article_summaries: str,
        expertise_level: str,
        language: str,
    ) -> Dict[str, Any]:
        """Generate synthesis using LLM."""

        # Adjust complexity based on expertise level
        complexity_instructions = {
            "beginner": "Use simple, accessible language. Explain technical terms. Use analogies where helpful.",
            "intermediate": "Use clear scientific language. Define complex terms when first introduced.",
            "expert": "Use precise scientific terminology. Focus on methodology and statistical significance.",
        }

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""You are a scientific literature synthesis expert. Your task is to analyze multiple scientific articles and create a comprehensive, accurate summary that addresses the user's query.

EXPERTISE LEVEL: {expertise_level}
{complexity_instructions.get(expertise_level, complexity_instructions['intermediate'])}

CRITICAL RULES:
1. NEVER make claims without citing specific articles
2. Clearly distinguish between findings from individual studies vs consensus across studies
3. Note any contradictions or limitations in the research
4. Be precise about study methodologies (RCT, observational, meta-analysis, etc.)
5. Highlight strength of evidence (e.g., "preliminary findings suggest..." vs "strong evidence demonstrates...")

OUTPUT FORMAT:
Return a JSON object with:
- "summary": A comprehensive markdown-formatted summary (3-5 paragraphs)
- "findings": Array of objects, each with:
  - "finding": The specific finding or insight
  - "category": One of [nutrition, health outcomes, methodology, safety, mechanisms, epidemiology]
  - "confidence": One of [high, medium, low]
  - "supporting_article_urns": Array of URN strings that support this finding
  - "supporting_sections": Array of section names (abstract, methods, results, discussion)

SUMMARY STRUCTURE:
1. Opening: Direct answer to the query with key consensus
2. Body: Major findings organized thematically
3. Nuance: Conflicting evidence, limitations, or gaps
4. Conclusion: Practical implications or future research needs"""),
            ("human", f"""Query: {query}

Articles to synthesize:
{article_summaries}

Generate a synthesis that answers the query comprehensively.""")
        ])

        try:
            response = self.llm.invoke(prompt_template.format_messages())

            # Parse the JSON response
            import json
            content = response.content.strip()

            # Try to extract JSON from code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            # Fallback to simple summary
            return {
                "summary": f"Error generating synthesis: {e}",
                "findings": []
            }

    def _extract_findings(
        self, synthesis_result: Dict[str, Any], articles: List[Dict[str, Any]]
    ) -> List[SynthesizedFinding]:
        """Extract structured findings from synthesis result."""
        findings = []

        # Create article lookup by URN
        article_lookup = {article.get('urn', ''): article for article in articles}

        for finding_data in synthesis_result.get("findings", []):
            # Create citations for this finding
            citations = []

            urns = finding_data.get("supporting_article_urns", [])
            sections = finding_data.get("supporting_sections", ["abstract"] * len(urns))

            for urn, section in zip(urns, sections):
                if urn in article_lookup:
                    article = article_lookup[urn]
                    citation = create_citation_from_article(
                        article,
                        section,
                        confidence=finding_data.get("confidence", "medium")
                    )
                    citations.append(citation)

            findings.append(
                SynthesizedFinding(
                    finding=finding_data.get("finding", ""),
                    supporting_citations=citations,
                    confidence=finding_data.get("confidence", "medium"),
                    category=finding_data.get("category", "general"),
                )
            )

        return findings

    def _collect_all_citations(
        self, findings: List[SynthesizedFinding]
    ) -> List[Citation]:
        """Collect all unique citations from findings."""
        all_citations = []
        seen = set()

        for finding in findings:
            for citation in finding.supporting_citations:
                key = f"{citation.article_urn}:{citation.section}"
                if key not in seen:
                    all_citations.append(citation)
                    seen.add(key)

        return all_citations

    def _generate_follow_ups(
        self, query: str, findings: List[SynthesizedFinding]
    ) -> List[str]:
        """Generate follow-up question suggestions."""
        try:
            # Extract categories from findings
            categories = list(set(f.category for f in findings))

            prompt = f"""Based on this search query and findings, suggest 3 specific follow-up questions a user might ask.

Original query: {query}
Finding categories: {', '.join(categories)}

Return ONLY a JSON array of 3 strings, each a specific question. Example:
["What are the long-term effects?", "How does this compare to alternatives?", "What are the recommended dosages?"]
"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON array
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            import json
            suggestions = json.loads(content)

            return suggestions[:3]  # Ensure max 3

        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return []
