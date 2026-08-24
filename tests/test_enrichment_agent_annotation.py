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

# Cut mid-key: closing the open scopes yields a key with no value, so the
# unclosed-JSON salvage cannot rescue it and escalation must kick in.
_TRUNCATED = '{"reader_group":"General Public","tags'
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


@patch("agents.enrichment_agent.build_trace_config", lambda **_: {})
class UnclosedJsonSalvageTests(unittest.TestCase):
    """Production showed gpt-oss ending long annotations with finish_reason
    "stop" but one closing brace short — the payload is complete except for
    the closers, so it must be salvaged, not failed."""

    # The exact shape from the prod probe: ends '..."}]}' but the top-level
    # object needs one more '}'.
    _UNDER_CLOSED = (
        '{"reader_group":"General Public","study_type":"Meta-analysis",'
        '"annotations":{"user_qa":[{"question":"Q?","answer":"A."}]}'
    )

    def test_under_closed_payload_is_salvaged_on_first_attempt(self):
        primary = _FakeChain([_response(self._UNDER_CLOSED)])

        result = _agent()._invoke_annotation(primary, _ARTICLE)

        self.assertEqual(result["study_type"], "Meta-analysis")
        self.assertEqual(
            result["annotations"]["user_qa"][0]["answer"], "A."
        )
        self.assertEqual(primary.calls, 1)

    def test_open_string_and_dangling_comma_are_repaired(self):
        content = '{"tags":["Dairy","Prostate Cancer"],"verdict":["Some evi'
        primary = _FakeChain([_response(content)])

        result = _agent()._invoke_annotation(primary, _ARTICLE)

        self.assertEqual(result["tags"], ["Dairy", "Prostate Cancer"])
        self.assertEqual(result["verdict"], ["Some evi"])

    def test_mid_document_corruption_is_not_papered_over(self):
        # An unescaped quote breaks the JSON mid-document; the top-level
        # object still closes, so the closers repair must not fire and the
        # normal retry path handles it.
        broken = '{"verdict":["shows "significant" effects"],"score":3}'
        primary = _FakeChain([_response(broken), _response(broken)])

        with self.assertRaises(ValueError):
            _agent()._invoke_annotation(primary, _ARTICLE)
        self.assertEqual(primary.calls, 2)


class CloseUnclosedJsonTests(unittest.TestCase):
    def test_returns_none_for_balanced_json(self):
        from agents.json_output import close_unclosed_json

        self.assertIsNone(close_unclosed_json('{"a": 1}'))

    def test_returns_none_without_an_object(self):
        from agents.json_output import close_unclosed_json

        self.assertIsNone(close_unclosed_json("no json here"))

    def test_returns_none_for_mismatched_brackets(self):
        from agents.json_output import close_unclosed_json

        self.assertIsNone(close_unclosed_json('{"a": [1}'))

    def test_closes_nested_scopes_in_order(self):
        import json

        from agents.json_output import close_unclosed_json

        repaired = close_unclosed_json('{"a": {"b": [1, 2')
        self.assertEqual(json.loads(repaired), {"a": {"b": [1, 2]}})

    def test_escaped_quotes_do_not_confuse_string_tracking(self):
        import json

        from agents.json_output import close_unclosed_json

        repaired = close_unclosed_json('{"a": "say \\"hi\\"", "b": [1')
        self.assertEqual(json.loads(repaired), {"a": 'say "hi"', "b": [1]})


if __name__ == "__main__":
    unittest.main()
