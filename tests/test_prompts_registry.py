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
                 answer_context="ACTX", prior_conversation="", question="Q?",
                 source_context="SRC")
        tmpl = ChatPromptTemplate.from_messages([
            ("system", QA_ANSWER_RAG_SYSTEM.langchain()),
            ("human", QA_ANSWER_RAG_USER.langchain()),
        ])
        msgs = tmpl.format_messages(**v)
        # Rebuild the exact legacy f-strings to assert byte-equivalence.
        legacy_sys = f"""You are FoodScholar, a scientific Q&A assistant specializing in food science, nutrition, and food safety. Your task is to answer the user's question concisely and accurately using ONLY the provided retrieved sources as evidence. Sources may include scientific article abstracts and dietary guideline rules.

EXPERTISE LEVEL: {v['expertise_level']}
{v['complexity']}

LANGUAGE: Write EVERY natural-language string you output in {v['language']} — this includes the "answer" prose AND every entry in "follow_ups". Do not switch to English for the follow-up questions or for any technical term that has a normal {v['language']} equivalent. Only the following may stay in their original form: proper nouns (author names, place names, organizations), source URNs, and established scientific Latin terms with no common {v['language']} word. Never leave stray English words in an otherwise {v['language']} answer.

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
            answer_context="AC", prior_conversation="", question="WHATQ")
        self.assertIn("beginner", msgs[0].content)
        self.assertIn('"overall_confidence": "high or medium or low"',
                      msgs[0].content)
        self.assertIn("WHATQ", msgs[1].content)


    def test_clarifier_system_fallback(self):
        from backend.prompts import QA_CLARIFIER_SYSTEM
        out = QA_CLARIFIER_SYSTEM.compile()
        self.assertIn("Clarifier and Safety planner", out)
        self.assertIn('"risk_level": "low | medium | high"', out)

    def test_qa_starter_renders_and_constrains_register(self):
        from backend.prompts import QA_STARTER_QUESTIONS
        out = QA_STARTER_QUESTIONS.compile(count=4, language="sl")
        # Variables substituted, no leftover mustache.
        self.assertNotIn("{{", out)
        self.assertIn("exactly 4", out)
        self.assertIn("Write every question in sl", out)
        # Register: aimed at everyday household users, not academic phrasing.
        self.assertIn("household", out.lower())
        self.assertIn("everyday", out.lower())
        # Guards the exact over-academic phrasings Claire (RCSI) flagged.
        self.assertIn("microbiota", out)
        self.assertIn("Explain how", out)
        # JSON output contract preserved.
        self.assertIn('{"questions": ["q1", "q2", "q3", "q4"]}', out)

    def test_qa_tips_from_guidelines_matches_legacy(self):
        from backend.prompts import QA_TIPS_FROM_GUIDELINES
        candidate_count = 3
        guideline_context = "RULES_HERE"
        legacy = f"""You create safe daily nutrition content for a general audience.

Using ONLY the dietary guideline rules below, generate exactly {candidate_count} items with a mix of:
- practical nutrition tips
- "Did you know?" nutrition facts

Safety rules:
- General education only (no diagnosis, treatment, medication, or disease-management advice).
- No supplement dosage guidance.
- Use the guideline rule_text as the source of truth.

Style rules:
- Each item must be <= 18 words.
- One sentence per item.
- Avoid absolute guarantees (no "cures", "prevents", "always", "never").

Return ONLY valid JSON in this exact format:
{{
  "items": [
    {{"kind": "tip", "text": "item text", "guideline": 1}},
    {{"kind": "did_you_know", "text": "item text", "guideline": 2}}
  ]
}}

Dietary guideline rules:
{guideline_context}
"""
        self.assertEqual(
            QA_TIPS_FROM_GUIDELINES.compile(
                candidate_count=candidate_count,
                guideline_context=guideline_context),
            legacy)

    def test_qa_tip_rewrite_matches_legacy(self):
        from backend.prompts import QA_TIP_REWRITE
        text, style, article_context = "ITEM", "Tip:", "EV"
        legacy = f"""Rewrite the item below as one short, evidence-grounded nutrition line.

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
        self.assertEqual(
            QA_TIP_REWRITE.compile(text=text, style=style,
                                   article_context=article_context),
            legacy)


class TestPromptSync(unittest.TestCase):
    """Idempotent startup sync: create when missing OR when fallback changed."""

    def _registry(self):
        from backend.prompts import _Prompt
        return [
            _Prompt("p-existing-same", fallback="SAME"),
            _Prompt("p-existing-diff", fallback="NEW TEXT"),
            _Prompt("p-missing", fallback="FRESH"),
        ]

    def test_sync_creates_only_missing_never_overwrites(self):
        """Langfuse is the source of truth: seed a prompt only when it does not
        exist. NEVER overwrite an existing prompt, even if its text differs from
        the in-code fallback (that text may be a deliberate UI edit)."""
        from backend import prompts as P

        class FakeManaged:
            def __init__(self, prompt):
                self.prompt = prompt

        ns = P._PROMPT_NAMESPACE  # prompts are stored under the namespace

        class FakeClient:
            def __init__(self):
                self.created = []
                self.store = {
                    ns + "p-existing-same": "SAME",     # exists -> skip
                    ns + "p-existing-diff": "UI EDIT",  # exists & differs -> STILL skip
                    # p-missing absent -> create
                }

            def get_prompt(self, name, **kwargs):
                if name not in self.store:
                    raise Exception(f"Prompt not found: '{name}'")
                return FakeManaged(self.store[name])

            def create_prompt(self, *, name, type, prompt, labels):
                self.created.append(name)

        fake = FakeClient()
        result = P.sync_prompts(client=fake, registry=self._registry())

        # Only the genuinely-missing prompt is created (namespaced).
        self.assertEqual(fake.created, [ns + "p-missing"])
        # The UI-edited existing prompt is left untouched.
        self.assertNotIn(ns + "p-existing-diff", fake.created)
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["skipped"], 2)

    def test_sync_noop_when_client_none(self):
        from backend import prompts as P
        result = P.sync_prompts(client=None, registry=self._registry())
        self.assertEqual(result, {"created": 0, "skipped": 0, "failed": 0})


if __name__ == "__main__":
    unittest.main()
