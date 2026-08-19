"""
Model configuration, capability profiles and output normalization.

These cover the two things that have to hold for a model swap to be an env
change rather than a code change: the roles resolve from configuration, and a
family's quirks are absorbed at the pool rather than at each call site.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.json_output import parse_json_array, parse_json_object  # noqa: E402
from backend.model_output import normalize_model_text  # noqa: E402
from backend.model_profiles import RETIRED, apply_profile, profile_for  # noqa: E402
from config import Config  # noqa: E402


class TestModelProfiles(unittest.TestCase):
    """Family quirks are declared once and applied at the pool."""

    def test_reasoning_family_gets_hidden_reasoning_and_a_budget(self):
        resolved = apply_profile("openai/gpt-oss-120b", {"temperature": 0.3})

        # 'hidden' is what keeps deliberation out of `content`, where it would
        # otherwise be parsed as the answer.
        self.assertEqual(resolved["reasoning_format"], "hidden")
        self.assertEqual(resolved["reasoning_effort"], "low")
        self.assertGreaterEqual(resolved["max_tokens"], 2048)
        self.assertEqual(resolved["temperature"], 0.3)

    def test_caller_budget_below_the_floor_is_raised(self):
        # 512 total tokens on a reasoning model is spent on reasoning, and the
        # JSON payload is truncated mid-object.
        resolved = apply_profile("openai/gpt-oss-20b", {"max_tokens": 512})
        self.assertEqual(resolved["max_tokens"], 2048)

    def test_caller_budget_above_the_floor_is_kept(self):
        resolved = apply_profile("openai/gpt-oss-20b", {"max_tokens": 8192})
        self.assertEqual(resolved["max_tokens"], 8192)

    def test_caller_reasoning_preferences_win(self):
        resolved = apply_profile(
            "openai/gpt-oss-20b", {"reasoning_effort": "high"}
        )
        self.assertEqual(resolved["reasoning_effort"], "high")

    def test_qwen_family_also_gets_hidden_reasoning_and_a_budget(self):
        # The cross-family alternative to gpt-oss-120b, so it has to be handled
        # as defensively as the family it stands in for.
        resolved = apply_profile("qwen/qwen3.6-27b", {"temperature": 0.3})
        self.assertEqual(resolved["reasoning_format"], "hidden")
        self.assertGreaterEqual(resolved["max_tokens"], 2048)

    def test_reasoning_params_are_dropped_for_families_that_reject_them(self):
        # Groq retired the Llama ids, but the family row stays: the same
        # weights are served by other providers, and a role pointed at one must
        # not 400 because an agent asked for reasoning_effort unconditionally.
        resolved = apply_profile(
            "llama-3.1-8b-instant",
            {"temperature": 0.0, "reasoning_effort": "low", "max_tokens": 512},
        )
        self.assertNotIn("reasoning_effort", resolved)
        self.assertNotIn("reasoning_format", resolved)
        self.assertEqual(resolved["max_tokens"], 512)

    def test_fixed_temperature_families_lose_the_temperature(self):
        resolved = apply_profile("gpt-5.4", {"temperature": 0.3})
        self.assertNotIn("temperature", resolved)

    def test_unknown_model_keeps_caller_intent_and_injects_nothing(self):
        profile = profile_for("some-vendor/brand-new-9000")
        self.assertFalse(profile.known)

        resolved = apply_profile(
            "some-vendor/brand-new-9000", {"temperature": 0.4, "reasoning_effort": "low"}
        )
        self.assertEqual(
            resolved, {"temperature": 0.4, "reasoning_effort": "low"}
        )

    def test_retired_ids_are_flagged_with_date_and_replacement(self):
        # A retired id still matches its family row, so the profile table alone
        # would not notice; the warning is the only signal before the provider
        # rejects the call.
        for retired in ("llama-3.1-8b-instant", "llama-3.3-70b-versatile"):
            self.assertIn(retired, RETIRED)
            shutdown, replacement = RETIRED[retired]
            self.assertEqual(shutdown, "2026-08-16")
            self.assertTrue(replacement)

        with self.assertLogs("backend.model_profiles", level="WARNING") as logs:
            # profile_for warns once per id, so use a fresh module-level set.
            from backend import model_profiles

            model_profiles._warned_retired.discard("llama-3.3-70b-versatile")
            profile_for("llama-3.3-70b-versatile")
        self.assertIn("2026-08-16", "".join(logs.output))
        self.assertIn("gpt-oss-120b", "".join(logs.output))

    def test_no_configured_default_role_uses_a_retired_id(self):
        from config import config

        roles = [
            v for k, v in config.settings.items()
            if k.endswith("_MODEL") or k == "QA_AVAILABLE_MODELS"
        ]
        configured = set()
        for value in roles:
            configured.update(value if isinstance(value, list) else [value])
        self.assertEqual(configured & set(RETIRED), set())

    def test_kwargs_are_not_mutated(self):
        original = {"temperature": 0.3}
        apply_profile("openai/gpt-oss-120b", original)
        self.assertEqual(original, {"temperature": 0.3})


class TestGroqPoolAppliesProfiles(unittest.TestCase):
    """The pool is the single choke point every Groq call goes through."""

    def _capture(self, model, **kwargs):
        from backend import groq as groq_module

        captured = {}

        class FakeChatGroq:
            def __init__(self, **init_kwargs):
                captured.update(init_kwargs)

        pool = groq_module.GroqConnectionPool()
        with patch.object(groq_module, "ChatGroq", FakeChatGroq):
            pool.get_client(model=model, **kwargs)
        return captured

    def test_reasoning_model_is_configured_defensively(self):
        captured = self._capture("openai/gpt-oss-120b", temperature=0.3)
        self.assertEqual(captured["model"], "openai/gpt-oss-120b")
        self.assertEqual(captured["reasoning_format"], "hidden")
        self.assertGreaterEqual(captured["max_tokens"], 2048)

    def test_unsupported_reasoning_knob_never_reaches_the_provider(self):
        captured = self._capture(
            "llama-3.1-8b-instant", temperature=0.0, reasoning_effort="low"
        )
        self.assertNotIn("reasoning_effort", captured)
        self.assertEqual(captured["temperature"], 0.0)

    def test_model_defaults_to_the_configured_qa_model(self):
        from config import config

        captured = self._capture(None)
        self.assertEqual(captured["model"], config.settings["QA_DEFAULT_MODEL"])


class TestModelOutputNormalization(unittest.TestCase):
    """Reasoning that leaks in-band is not an answer."""

    def test_think_block_is_stripped(self):
        self.assertEqual(
            normalize_model_text("<think>weigh the options</think>Eat more fibre."),
            "Eat more fibre.",
        )

    def test_harmony_channel_residue_is_stripped(self):
        self.assertEqual(
            normalize_model_text(
                "<|start|>assistant<|channel|>final<|message|>Eat more fibre."
            ),
            "Eat more fibre.",
        )
        self.assertEqual(
            normalize_model_text("analysis the user asks assistantfinal Eat fibre."),
            "Eat fibre.",
        )

    def test_content_blocks_are_flattened(self):
        blocks = [
            {"type": "text", "text": '{"answer": '},
            {"type": "text", "text": '"yes"}'},
        ]
        self.assertEqual(normalize_model_text(blocks), '{"answer": "yes"}')

    def test_truncated_reasoning_yields_nothing_rather_than_garbage(self):
        self.assertEqual(normalize_model_text("<think>cut off mid thought"), "")

    def test_ordinary_prose_is_untouched(self):
        text = "My assistant will finalize the meal plan."
        self.assertEqual(normalize_model_text(text), text)


class TestSharedJsonRecovery(unittest.TestCase):
    """Every agent path tolerates the same deviations."""

    def test_object_behind_reasoning_fences_and_a_trailing_comma(self):
        raw = '<think>plan</think>```json\n{"answer": "hi", "follow_ups": [],}\n```'
        self.assertEqual(
            parse_json_object(raw), {"answer": "hi", "follow_ups": []}
        )

    def test_object_after_harmony_residue(self):
        self.assertEqual(
            parse_json_object('analysis blah assistantfinal{"answer": "hi"}'),
            {"answer": "hi"},
        )

    def test_array_with_a_sentence_in_front(self):
        self.assertEqual(
            parse_json_array('Sure, here you go: ["fibre", "iron",]'),
            ["fibre", "iron"],
        )

    def test_unrecoverable_output_raises(self):
        with self.assertRaises(ValueError):
            parse_json_object("<think>truncated mid reasoning")


class TestModelConfigValidation(unittest.TestCase):
    """A model list that cannot serve a request stops the process."""

    def _setup_with(self, **env):
        with patch.dict(os.environ, env, clear=False):
            cfg = Config()
            cfg.setup()
            return cfg

    def test_roles_read_from_the_environment(self):
        cfg = self._setup_with(
            QA_DEFAULT_MODEL="openai/gpt-oss-20b",
            QA_AVAILABLE_MODELS="openai/gpt-oss-20b, qwen/qwen3.6-27b",
            SYNTHESIS_MODEL="qwen/qwen3.6-27b",
        )
        self.assertEqual(cfg.settings["QA_DEFAULT_MODEL"], "openai/gpt-oss-20b")
        self.assertEqual(
            cfg.settings["QA_AVAILABLE_MODELS"],
            ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
        )
        self.assertEqual(cfg.settings["SYNTHESIS_MODEL"], "qwen/qwen3.6-27b")

    def test_blank_and_duplicate_entries_are_dropped(self):
        cfg = self._setup_with(
            QA_DEFAULT_MODEL="openai/gpt-oss-20b",
            QA_AVAILABLE_MODELS=" openai/gpt-oss-20b , ,openai/gpt-oss-20b ",
        )
        self.assertEqual(cfg.settings["QA_AVAILABLE_MODELS"], ["openai/gpt-oss-20b"])

    def test_default_outside_the_available_list_is_refused(self):
        with self.assertRaises(ValueError):
            self._setup_with(
                QA_DEFAULT_MODEL="openai/gpt-oss-120b",
                QA_AVAILABLE_MODELS="qwen/qwen3.6-27b",
            )

    def test_empty_available_list_is_refused(self):
        with self.assertRaises(ValueError):
            self._setup_with(QA_AVAILABLE_MODELS=" , ")


if __name__ == "__main__":
    unittest.main()
