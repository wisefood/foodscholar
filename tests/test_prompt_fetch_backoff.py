"""A dead Langfuse costs one failed lookup per prompt per backoff window.

The SDK hands back our fallback after a failed fetch but does not remember
the failure, so with Langfuse removed from the cluster every compile() paid a
failed lookup (about a second each, several per question).
"""

import unittest
from unittest.mock import patch

from backend import prompts


class _Fallback:
    is_fallback = True

    def __init__(self, text):
        self.text = text

    def compile(self, **v):
        return prompts._compile_fallback(self.text, v)


class _Managed:
    is_fallback = False
    version = 7

    def compile(self, **_v):
        return "managed text"


class _Client:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def get_prompt(self, name, fallback=None, **_kw):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result if self.result != "fallback" else _Fallback(fallback)


class PromptBackoffTests(unittest.TestCase):
    def _prompt(self):
        return prompts._Prompt("test-backoff", "Hello {{name}}")

    def test_failed_fetch_is_not_retried_within_the_window(self):
        client = _Client("fallback")
        p = self._prompt()
        with patch.object(prompts, "get_langfuse_client", return_value=client):
            self.assertEqual(p.compile(name="A"), "Hello A")
            self.assertEqual(p.compile(name="B"), "Hello B")
            self.assertEqual(p.compile(name="C"), "Hello C")
        self.assertEqual(client.calls, 1)

    def test_raising_fetch_also_backs_off(self):
        client = _Client(ConnectionError("Name or service not known"))
        p = self._prompt()
        with patch.object(prompts, "get_langfuse_client", return_value=client):
            p.compile(name="A")
            p.compile(name="B")
        self.assertEqual(client.calls, 1)

    def test_tries_again_after_the_window(self):
        client = _Client("fallback")
        p = self._prompt()
        with patch.object(prompts, "get_langfuse_client", return_value=client):
            p.compile(name="A")
            p._skip_until = 0.0  # the window has passed
            client.result = _Managed()
            self.assertEqual(p.compile(name="B"), "managed text")
            self.assertEqual(p.compile(name="C"), "managed text")
        self.assertEqual(client.calls, 3)  # successes are the SDK's to cache

    def test_no_client_means_fallback_without_lookup(self):
        p = self._prompt()
        with patch.object(prompts, "get_langfuse_client", return_value=None):
            self.assertEqual(p.compile(name="A"), "Hello A")


if __name__ == "__main__":
    unittest.main()
