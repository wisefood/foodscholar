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

if __name__ == "__main__":
    unittest.main()
