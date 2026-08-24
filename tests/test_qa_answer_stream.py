"""Tests for the sentinel-protocol streaming answer generator."""

import unittest
from unittest.mock import patch

from services.qa_pipeline import answering


class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeStreamingLLM:
    """Yields the given chunk texts from astream."""

    def __init__(self, chunks, fail_after=None):
        self.chunks = chunks
        self.fail_after = fail_after

    async def astream(self, _messages, config=None):
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("provider dropped the stream")
            yield _FakeChunk(chunk)


PAYLOADS = [
    {
        "urn": "urn:article:a1",
        "source_type": "article",
        "title": "Whole grains and lipids",
        "authors": ["Doe"],
        "publication_year": "2021",
        "abstract": (
            "Whole-grain intake reduced LDL cholesterol in adults. "
            "Effects were consistent across cohorts."
        ),
        "relevance_score": 0.9,
    },
    {
        "urn": "guideline-1",
        "source_type": "guideline",
        "title": "Healthy eating guide",
        "rule_text": "Choose whole-grain cereals, bread, rice, or pasta more often.",
        "relevance_score": 5.0,
    },
]

TRAILER = (
    '{"cited_sources": [{"urn": "urn:article:a1", "section": "abstract", '
    '"quote": "Whole-grain intake reduced LDL cholesterol in adults.", '
    '"confidence": "high"}], "overall_confidence": "high", '
    '"follow_ups": ["What about oats?"]}'
)


async def _collect(chunks, payloads=PAYLOADS, fail_after=None):
    events = []
    llm = _FakeStreamingLLM(chunks, fail_after=fail_after)
    with patch.object(answering.GROQ_CHAT, "get_client", return_value=llm):
        async for event in answering.stream_answer(
            question="Do whole grains lower cholesterol?",
            payloads=payloads,
            expertise_level="intermediate",
            language="en",
            model="test-model",
        ):
            events.append(event)
    return events


def _deltas(events):
    return "".join(e["text"] for e in events if e["kind"] == "delta")


def _final(events):
    finals = [e for e in events if e["kind"] == "final"]
    assert len(finals) == 1
    return finals[0]


class SentinelStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_streams_answer_and_parses_trailer(self):
        answer = "Whole grains help [Doe et al. (2021)](/articles/urn:article:a1)."
        events = await _collect(
            [answer, "\n", answering.ANSWER_SENTINEL, "\n", TRAILER]
        )
        final = _final(events)
        self.assertEqual(_deltas(events).strip(), answer)
        self.assertNotIn(answering.ANSWER_SENTINEL, _deltas(events))
        self.assertTrue(final["parsed_trailer"])
        self.assertEqual(final["answer"].confidence, "high")
        self.assertEqual(len(final["answer"].citations), 1)
        citation = final["answer"].citations[0]
        self.assertEqual(citation.source_id, "urn:article:a1")
        # Quote is an exact substring of the source text.
        self.assertIn(citation.quote, PAYLOADS[0]["abstract"])
        self.assertEqual(final["follow_ups"], ["What about oats?"])

    async def test_sentinel_split_across_chunks_is_never_emitted(self):
        answer = "Answer text here."
        events = await _collect(
            [answer, " <<<END_", "ANSWER>>>", TRAILER[:20], TRAILER[20:]]
        )
        deltas = _deltas(events)
        self.assertNotIn("<<<", deltas)
        self.assertNotIn("END_ANSWER", deltas)
        self.assertEqual(deltas.strip(), answer)
        self.assertTrue(_final(events)["parsed_trailer"])

    async def test_missing_trailer_recovers_citations_from_links(self):
        answer = (
            "Whole grains help [Doe et al. (2021)](/articles/urn:article:a1) and "
            "guidelines agree [G1](/guidelines/guideline-1)."
        )
        events = await _collect([answer])
        final = _final(events)
        self.assertFalse(final["parsed_trailer"])
        cited_ids = {c.source_id for c in final["answer"].citations}
        self.assertEqual(cited_ids, {"urn:article:a1", "guideline-1"})
        # Best-effort quotes still land as exact source substrings.
        for citation in final["answer"].citations:
            self.assertTrue(citation.quote)

    async def test_json_mode_output_is_buffered_not_streamed_raw(self):
        # A model that ignored the protocol and returned the legacy JSON shape.
        legacy = (
            '{"answer": "Whole grains help.", "cited_sources": [], '
            '"overall_confidence": "medium", "follow_ups": []}'
        )
        events = await _collect([legacy[:30], legacy[30:]])
        final = _final(events)
        self.assertEqual(_deltas(events), "Whole grains help.")
        self.assertEqual(final["answer"].answer, "Whole grains help.")

    async def test_midstream_failure_still_produces_an_answer(self):
        events = await _collect(
            ["A partial answer that streamed fine before the crash. Then more words arrive here."],
            fail_after=None,
        )
        # Baseline sanity: full stream works.
        self.assertTrue(_final(events)["answer"].answer)

        events = await _collect(
            [
                "A partial answer that streamed fine before the crash. Then more words arrive here.",
                "never delivered",
            ],
            fail_after=1,
        )
        final = _final(events)
        self.assertEqual(final["answer"].confidence, "medium")
        self.assertIn("partial answer", final["answer"].answer)

    async def test_total_failure_yields_error_answer(self):
        events = await _collect(["ignored"], fail_after=0)
        final = _final(events)
        self.assertEqual(final["answer"].confidence, "low")
        self.assertIn("Unable to generate", final["answer"].answer)

    async def test_no_sources_never_fabricates_citations(self):
        answer = "General guidance without sources."
        events = await _collect(
            [answer, answering.ANSWER_SENTINEL,
             '{"cited_sources": [], "overall_confidence": "medium", "follow_ups": []}'],
            payloads=[],
        )
        final = _final(events)
        self.assertFalse(final["answer"].rag_used)
        self.assertEqual(final["answer"].citations, [])


class AnswerProseNormalizationTests(unittest.TestCase):
    """CJK citation brackets and banned dashes are repaired, not trusted away."""

    def test_cjk_brackets_become_a_parseable_link(self):
        from backend.model_output import normalize_answer_prose

        broken = "Fiber helps【Marcobal et al. (2024)](/articles/urn:article:a1)】."
        self.assertEqual(
            normalize_answer_prose(broken),
            "Fiber helps[Marcobal et al. (2024)](/articles/urn:article:a1).",
        )

    def test_spaced_dashes_become_commas_and_ranges_become_hyphens(self):
        from backend.model_output import normalize_answer_prose

        text = "Slows digestion — a traffic jam for sugar — over 20–35 % of intake."
        self.assertEqual(
            normalize_answer_prose(text),
            "Slows digestion, a traffic jam for sugar, over 20-35 % of intake.",
        )
        self.assertNotIn("—", normalize_answer_prose(text))
        self.assertNotIn("–", normalize_answer_prose(text))

    def test_empty_and_non_string_are_safe(self):
        from backend.model_output import normalize_answer_prose

        self.assertEqual(normalize_answer_prose(None), "")
        self.assertEqual(normalize_answer_prose(""), "")


class BrokenModelFormattingStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_cjk_bracket_citations_still_resolve_and_answer_is_clean(self):
        answer = (
            "Fiber slows sugar absorption — a lot"
            "【Doe et al. (2021)](/articles/urn:article:a1)】."
        )
        events = await _collect([answer])
        final = _final(events)
        # The settled answer parses as normal markdown with no banned glyphs
        # (the model's own label survives; only the brackets are repaired).
        self.assertIn(
            "[Doe et al. (2021)](/articles/urn:article:a1)",
            final["answer"].answer,
        )
        self.assertNotIn("【", final["answer"].answer)
        self.assertNotIn("—", final["answer"].answer)
        # And the citation was recovered from the repaired inline link.
        self.assertEqual(
            [c.source_id for c in final["answer"].citations],
            ["urn:article:a1"],
        )


class SentinelRobustnessTests(unittest.IsolatedAsyncioTestCase):
    """The trailer must never reach the settled answer, however the model
    mangles the sentinel (observed live: '<<>' rendering + full JSON leak)."""

    TRAILER_JSON = (
        '{"cited_sources": [{"urn": "urn:article:a1", "section": "abstract", '
        '"quote": "Whole-grain intake reduced LDL cholesterol in adults.", '
        '"confidence": "high"}], "overall_confidence": "high", "follow_ups": []}'
    )

    async def test_malformed_sentinel_variants_still_split(self):
        for sentinel in ("<<END_ANSWER>>", "<<< END_ANSWER >>>", "<<<END_ANSWER>>"):
            with self.subTest(sentinel=sentinel):
                events = await _collect(
                    [f"The answer body. {sentinel}", self.TRAILER_JSON]
                )
                final = _final(events)
                self.assertEqual(final["answer"].answer, "The answer body.")
                self.assertNotIn("cited_sources", _deltas(events))
                self.assertTrue(final["parsed_trailer"])
                self.assertEqual(len(final["answer"].citations), 1)

    async def test_sentinel_missing_entirely_trailer_is_salvaged(self):
        events = await _collect(
            ["The answer body.\n", self.TRAILER_JSON]
        )
        final = _final(events)
        self.assertEqual(final["answer"].answer, "The answer body.")
        self.assertNotIn("cited_sources", final["answer"].answer)
        self.assertTrue(final["parsed_trailer"])
        self.assertEqual(
            final["answer"].citations[0].source_id, "urn:article:a1"
        )

    async def test_bare_bracketed_urn_becomes_a_labeled_link(self):
        # Observed live: the model cited as 【urn:article:...】 with no label
        # and no URL at all. The evidence pool knows the source, so the URN
        # becomes a properly labeled markdown link and a recoverable citation.
        answer = (
            "Fiber supports the gut barrier【urn:article:a1】 in trials."
        )
        events = await _collect([answer])
        final = _final(events)
        self.assertIn(
            "[Doe (2021)](/articles/urn:article:a1)",
            final["answer"].answer,
        )
        self.assertNotIn("【", final["answer"].answer)
        self.assertEqual(
            [c.source_id for c in final["answer"].citations],
            ["urn:article:a1"],
        )

    def test_repair_labels_guidelines_with_matching_g_numbers(self):
        from agents.qa_agent import repair_citation_links

        sources = [
            PAYLOADS[0],  # article
            PAYLOADS[1],  # guideline -> G1
        ]
        text = "Guidance says so [guideline-1] and evidence agrees【urn:article:a1】."
        repaired = repair_citation_links(text, sources)
        self.assertIn("[G1](/guidelines/guideline-1)", repaired)
        self.assertIn("[Doe (2021)](/articles/urn:article:a1)", repaired)
        # A proper link is left alone.
        well_formed = "See [Doe (2021)](/articles/urn:article:a1)."
        self.assertEqual(repair_citation_links(well_formed, sources), well_formed)

    async def test_unclosed_citation_link_is_repaired(self):
        # "[Zhao et al. (2025)(/articles/urn)" — the model dropped the "](" .
        answer = (
            "Fiber helps [Doe et al. (2021)(/articles/urn:article:a1)."
            f" {answering.ANSWER_SENTINEL}"
        )
        events = await _collect([answer, TRAILER])
        final = _final(events)
        self.assertIn(
            "[Doe et al. (2021)](/articles/urn:article:a1)",
            final["answer"].answer,
        )


class InlineCitationRecoveryTests(unittest.TestCase):
    def test_unknown_urns_are_ignored(self):
        cited = answering.citations_from_inline_links(
            "See [X](/articles/urn:article:a1) and [Y](/articles/urn:article:unknown).",
            PAYLOADS,
        )
        self.assertEqual([c["urn"] for c in cited], ["urn:article:a1"])

    def test_duplicates_collapse(self):
        text = (
            "[A](/articles/urn:article:a1) then again [A](/articles/urn:article:a1)"
        )
        cited = answering.citations_from_inline_links(text, PAYLOADS)
        self.assertEqual(len(cited), 1)


if __name__ == "__main__":
    unittest.main()
