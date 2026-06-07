"""Tests for the centralized prompt registry and Langfuse client accessor."""
import os
import unittest


def _disable_langfuse():
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    from backend import langfuse as lf
    lf.get_langfuse_client.cache_clear()


class TestLangfuseClient(unittest.TestCase):
    def setUp(self):
        _disable_langfuse()

    def test_client_is_none_when_disabled(self):
        from backend.langfuse import get_langfuse_client
        self.assertIsNone(get_langfuse_client())


class TestPromptRegistry(unittest.TestCase):
    def setUp(self):
        _disable_langfuse()

    def test_disabled_returns_fallback_no_vars(self):
        from backend.prompts import ENRICHMENT_KEYWORDS_SYSTEM
        out = ENRICHMENT_KEYWORDS_SYSTEM.compile()
        self.assertIn("nutrition science expert", out)

    def test_compile_substitutes_variables(self):
        from backend.prompts import _Prompt
        p = _Prompt("test-x", fallback="Hello {{name}}!")
        self.assertEqual(p.compile(name="World"), "Hello World!")

    def test_keywords_user_has_abstract_var(self):
        from backend.prompts import ENRICHMENT_KEYWORDS_USER
        out = ENRICHMENT_KEYWORDS_USER.compile(abstract="ABC123")
        self.assertIn("ABC123", out)
        self.assertNotIn("{{abstract}}", out)

    def test_annotation_substitutes_and_keeps_json_braces(self):
        from backend.prompts import ENRICHMENT_ANNOTATION
        out = ENRICHMENT_ANNOTATION.compile(
            title="T1", authors="A1", abstract="AB1")
        self.assertIn("T1", out)
        self.assertIn("AB1", out)
        self.assertIn('"reader_group": "General Public"', out)
        self.assertNotIn("{{", out)

    def test_annotation_langchain_renders_via_prompttemplate(self):
        """langchain() must be a valid LangChain template: vars substitute,
        JSON skeleton braces survive as literal braces."""
        from langchain_core.prompts import PromptTemplate
        from backend.prompts import ENRICHMENT_ANNOTATION
        tmpl = PromptTemplate(
            input_variables=["title", "authors", "abstract"],
            template=ENRICHMENT_ANNOTATION.langchain(),
        )
        rendered = tmpl.format(title="TT", authors="AA", abstract="BB")
        self.assertIn("Title: TT", rendered)
        self.assertIn("Abstract: BB", rendered)
        self.assertIn('"reader_group": "General Public"', rendered)

    def test_keywords_langchain_renders_via_chatprompttemplate(self):
        from langchain.prompts import ChatPromptTemplate
        from backend.prompts import (
            ENRICHMENT_KEYWORDS_SYSTEM, ENRICHMENT_KEYWORDS_USER)
        tmpl = ChatPromptTemplate.from_messages([
            ("system", ENRICHMENT_KEYWORDS_SYSTEM.langchain()),
            ("human", ENRICHMENT_KEYWORDS_USER.langchain()),
        ])
        msgs = tmpl.format_messages(abstract="MYABSTRACT")
        self.assertIn("MYABSTRACT", msgs[1].content)
        self.assertIn("nutrition science expert", msgs[0].content)

    def test_qa_rag_prompt_renders_and_matches_legacy(self):
        from langchain.prompts import ChatPromptTemplate
        from backend.prompts import QA_ANSWER_RAG_SYSTEM, QA_ANSWER_RAG_USER
        v = dict(expertise_level="expert", complexity="CX", language="en",
                 answer_context="ACTX", question="Q?", source_context="SRC")
        tmpl = ChatPromptTemplate.from_messages([
            ("system", QA_ANSWER_RAG_SYSTEM.langchain()),
            ("human", QA_ANSWER_RAG_USER.langchain()),
        ])
        msgs = tmpl.format_messages(**v)
        # Rebuild the exact legacy f-strings to assert byte-equivalence.
        legacy_sys = f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Your task is to answer the user's question concisely and accurately using ONLY the provided retrieved sources as evidence. Sources may include scientific article abstracts and dietary guideline rules.

EXPERTISE LEVEL: {v['expertise_level']}
{v['complexity']}

LANGUAGE: Respond in {v['language']}.

ANSWER FORMULATION CONTEXT:
{v['answer_context']}

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
{{
  "answer": "Markdown-formatted concise answer with inline citations as markdown links",
  "cited_sources": [
    {{
      "urn": "the source URN",
      "section": "abstract or rule_text",
      "quote": "verbatim excerpt from the source supporting the answer",
      "confidence": "high"
    }}
  ],
  "overall_confidence": "high",
  "follow_ups": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}

IMPORTANT: Return ONLY the JSON object."""
        self.assertEqual(msgs[0].content, legacy_sys)
        self.assertIn("Q?", msgs[1].content)
        self.assertIn("SRC", msgs[1].content)

    def test_qa_norag_prompt_renders(self):
        from langchain.prompts import ChatPromptTemplate
        from backend.prompts import QA_ANSWER_NORAG_SYSTEM, QA_ANSWER_NORAG_USER
        tmpl = ChatPromptTemplate.from_messages([
            ("system", QA_ANSWER_NORAG_SYSTEM.langchain()),
            ("human", QA_ANSWER_NORAG_USER.langchain()),
        ])
        msgs = tmpl.format_messages(
            expertise_level="beginner", complexity="C", language="el",
            answer_context="AC", question="WHATQ")
        self.assertIn("beginner", msgs[0].content)
        self.assertIn('"overall_confidence": "high or medium or low"',
                      msgs[0].content)
        self.assertIn("WHATQ", msgs[1].content)


    def test_clarifier_system_fallback(self):
        from backend.prompts import QA_CLARIFIER_SYSTEM
        out = QA_CLARIFIER_SYSTEM.compile()
        self.assertIn("Clarifier and Safety planner", out)
        self.assertIn('"risk_level": "low | medium | high"', out)


if __name__ == "__main__":
    unittest.main()
