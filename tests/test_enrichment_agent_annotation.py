"""Annotation invocation must survive budget-truncated completions.

Production logs showed gpt-oss-20b clipping the annotation JSON mid-string at
the max_tokens ceiling; at temperature 0 the plain retry reproduced the same
cut, so every affected article failed permanently. These tests pin the fix:
truncation is detected (finish_reason or unclosed JSON) and one final attempt
runs on the escalation chain with the larger budget.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.enrichment_agent import (
    _ANNOTATION_MAX_TOKENS,
    _ANNOTATION_MAX_TOKENS_ESCALATED,
    EnrichmentAgent,
)


def _agent() -> EnrichmentAgent:
    # __init__ builds real Groq clients; the annotation logic under test
    # never touches them.
    return object.__new__(EnrichmentAgent)


def _response(content: str, finish_reason: str | None = None):
    metadata = {"finish_reason": finish_reason} if finish_reason else {}
    return SimpleNamespace(content=content, response_metadata=metadata)


class _FakeChain:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, payload, config=None):
        self.calls += 1
        return self._responses.pop(0)


_ARTICLE = SimpleNamespace(
    urn="urn:article:x", title="T", abstract="A", authors=["B"]
)

_TRUNCATED = '{"reader_group":"General Public","tags":["Kefir","Probio'
_COMPLETE = '{"reader_group":"General Public","study_type":"RCT"}'


@patch("agents.enrichment_agent.build_trace_config", lambda **_: {})
class InvokeAnnotationTruncationTests(unittest.TestCase):
    def test_truncated_attempts_fall_through_to_escalation_chain(self):
        primary = _FakeChain([_response(_TRUNCATED), _response(_TRUNCATED)])
        escalated = _FakeChain([_response(_COMPLETE)])

        result = _agent()._invoke_annotation(
            primary, _ARTICLE, escalation_chain=escalated
        )

        self.assertEqual(result["study_type"], "RCT")
        self.assertEqual(primary.calls, 2)
        self.assertEqual(escalated.calls, 1)

    def test_finish_reason_length_counts_as_truncation(self):
        # Content that fails parsing for another surface reason still
        # escalates when the provider says the budget ran out.
        primary = _FakeChain(
            [_response("", "length"), _response("", "length")]
        )
        escalated = _FakeChain([_response(_COMPLETE)])

        result = _agent()._invoke_annotation(
            primary, _ARTICLE, escalation_chain=escalated
        )

        self.assertEqual(result["reader_group"], "General Public")
        self.assertEqual(escalated.calls, 1)

    def test_non_truncated_failure_does_not_escalate(self):
        primary = _FakeChain(
            [_response("no json here"), _response("still prose")]
        )
        escalated = _FakeChain([_response(_COMPLETE)])

        with self.assertRaises(ValueError):
            _agent()._invoke_annotation(
                primary, _ARTICLE, escalation_chain=escalated
            )
        self.assertEqual(escalated.calls, 0)

    def test_escalation_failure_raises_with_last_error(self):
        primary = _FakeChain([_response(_TRUNCATED), _response(_TRUNCATED)])
        escalated = _FakeChain([_response(_TRUNCATED)])

        with self.assertRaises(ValueError):
            _agent()._invoke_annotation(
                primary, _ARTICLE, escalation_chain=escalated
            )
        self.assertEqual(escalated.calls, 1)

    def test_success_on_first_attempt_never_escalates(self):
        primary = _FakeChain([_response(_COMPLETE)])
        escalated = _FakeChain([])

        result = _agent()._invoke_annotation(
            primary, _ARTICLE, escalation_chain=escalated
        )

        self.assertEqual(result["study_type"], "RCT")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(escalated.calls, 0)

    def test_escalated_budget_is_meaningfully_larger(self):
        self.assertGreater(
            _ANNOTATION_MAX_TOKENS_ESCALATED, _ANNOTATION_MAX_TOKENS
        )


if __name__ == "__main__":
    unittest.main()
