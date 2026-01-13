"""Synthesis agent for multi-document search summarization."""
import logging
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain import hub
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
        self._agent_executor = None
        self._tools = None

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
Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object.
Ensure all strings are properly escaped (use \\n for newlines, \\t for tabs, \\" for quotes).

JSON structure:
- "summary": A comprehensive markdown-formatted summary. Use rich markdown formatting:
  - **Bold** for key terms, important findings, and emphasis
  - *Italic* for study types, scientific names, and technical terms
  - Use bullet lists (- ) or numbered lists for clarity
  - Use headings (###) to organize sections
  - Use > blockquotes for notable quotes or key takeaways
  - Separate paragraphs with \\n\\n
  - Keep it focused and scannable with clear visual hierarchy
- "findings": Array of objects, each with:
  - "finding": The specific finding or insight (properly escaped string, can use markdown)
  - "category": One of [nutrition, health outcomes, methodology, safety, mechanisms, epidemiology]
  - "confidence": One of [high, medium, low]
  - "supporting_article_urns": Array of URN strings that support this finding
  - "supporting_sections": Array of section names (abstract, methods, results, discussion)

SUMMARY STRUCTURE:
1. Opening: Direct answer to the query with **key consensus in bold**
2. Body: Major findings organized with headings and bullet points
3. Nuance: Conflicting evidence, limitations, or gaps (use *italic* for caveats)
4. Conclusion: Practical implications with **bold takeaways**

FORMATTING EXAMPLES:
- "**Natto consumption** showed significant antithrombotic effects in *hypercholesterolemia rats*."
- "### Key Health Outcomes\\n\\n- Improved blood circulation\\n- Reduced cholesterol levels"
- "> Important: These findings are based on *animal studies* and may not directly translate to humans."

IMPORTANT: Return ONLY the JSON object. Do not wrap it in code blocks or add any other text."""),
            ("human", f"""Query: {query}

Articles to synthesize:
{article_summaries}

Generate a synthesis that answers the query comprehensively.""")
        ])

        try:
            response = self.llm.invoke(prompt_template.format_messages())

            # Parse the JSON response
            import json
            import re
            content = response.content.strip()

            # Try to extract JSON from code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Clean up control characters that break JSON parsing
            # Remove or escape control characters except for allowed ones
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

            # Log the cleaned content for debugging
            logger.debug(f"Cleaned LLM response (first 500 chars): {content[:500]}")

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Problematic content (first 1000 chars): {content[:1000] if 'content' in locals() else 'N/A'}")

            # Try to use JSON repair or extract what we can
            try:
                # Attempt to fix common JSON issues
                result = json.loads(content)
                logger.info("Successfully parsed with json5")
                return result
            except Exception as json5_error:
                logger.error(f"json5 parsing also failed: {json5_error}")

            # Final fallback
            return {
                "summary": f"Unable to parse LLM response. Please try again or check the logs.",
                "findings": []
            }
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

    def get_tools(self) -> List:
        """
        Get LangChain tools for this agent.

        Returns:
            List of LangChain tool functions
        """
        if self._tools is not None:
            return self._tools

        # Create a closure to capture self
        agent_instance = self

        @tool
        def synthesize_articles(
            query: str,
            articles_json: str,
            expertise_level: str = "intermediate",
            language: str = "en"
        ) -> str:
            """
            Synthesize information from multiple scientific articles into a comprehensive summary.

            Args:
                query: The search query or question to answer
                articles_json: JSON string containing array of article data
                expertise_level: Target expertise level (beginner, intermediate, expert)
                language: Target language code (default: en)

            Returns:
                JSON string containing synthesized findings and summary
            """
            import json

            try:
                articles = json.loads(articles_json)
                result = agent_instance.synthesize_search_results(
                    query=query,
                    articles=articles,
                    expertise_level=expertise_level,
                    language=language
                )
                return result.model_dump_json()
            except Exception as e:
                logger.error(f"Error in synthesize_articles tool: {e}")
                return json.dumps({"error": str(e)})

        @tool
        def generate_follow_up_questions(query: str, findings_json: str) -> str:
            """
            Generate follow-up questions based on a query and findings.

            Args:
                query: Original search query
                findings_json: JSON string containing array of findings

            Returns:
                JSON array of follow-up question strings
            """
            import json

            try:
                findings_data = json.loads(findings_json)
                findings = [
                    SynthesizedFinding(**f) for f in findings_data
                ]
                suggestions = agent_instance._generate_follow_ups(query, findings)
                return json.dumps(suggestions)
            except Exception as e:
                logger.error(f"Error in generate_follow_up_questions tool: {e}")
                return json.dumps([])

        self._tools = [synthesize_articles, generate_follow_up_questions]
        return self._tools

    @property
    def agent_executor(self) -> AgentExecutor:
        """
        Get a LangChain AgentExecutor that can be used in LangGraph graphs.

        This property lazily creates and caches an AgentExecutor with tools
        for synthesis operations.

        Returns:
            AgentExecutor configured with synthesis tools

        Example:
            >>> agent = SynthesisAgent()
            >>> executor = agent.agent_executor
            >>> result = executor.invoke({
            ...     "input": "Synthesize these articles about natto..."
            ... })
        """
        if self._agent_executor is not None:
            return self._agent_executor

        # Get tools
        tools = self.get_tools()

        # Pull a prompt from LangChain hub or use a custom one
        try:
            prompt = hub.pull("hwchase17/tool-calling-agent")
        except Exception as e:
            logger.warning(f"Could not pull prompt from hub: {e}. Using custom prompt.")
            # Fallback to custom prompt
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a scientific literature synthesis assistant.
You have access to tools for synthesizing research articles and generating follow-up questions.
Use these tools to help users understand scientific literature comprehensively."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

        # Create the agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # Create the executor
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

        return self._agent_executor

    def as_runnable(self):
        """
        Get the agent as a LangChain Runnable for use in LangGraph.

        This is an alias for agent_executor that's more explicit about
        LangGraph compatibility.

        Returns:
            AgentExecutor (which implements Runnable protocol)

        Example:
            >>> from langgraph.graph import StateGraph
            >>> agent = SynthesisAgent()
            >>> graph = StateGraph(...)
            >>> graph.add_node("synthesis", agent.as_runnable())
        """
        return self.agent_executor
