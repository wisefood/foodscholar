"""The platform tracing kill switch.

Tracing used to be a deploy-time fact: it ran if the Langfuse keys were set, and
stopping it meant unsetting them and rolling every pod. That is not something
anyone can do while an incident is in progress, or when a study participant
withdraws mid-session. An admin can now switch it off from the console and every
service stops within one flag-refresh interval.

Three properties have to hold, and each has a way of being quietly wrong:

* the switch must actually reach the handler — an ``lru_cache`` on the accessor
  would pin the first answer and the switch would appear to do nothing;
* it must not take *prompt management* down with it, since the same Langfuse
  client serves both;
* it must fail open. A control plane that fails closed would stop tracing every
  time the gateway hiccups.
"""
import sys
import unittest
from unittest import mock

sys.path.insert(0, "src")

import wf_telemetry  # noqa: E402
from backend import langfuse as lf  # noqa: E402


class TracingSwitchTests(unittest.TestCase):
    def setUp(self):
        lf.get_callback_handler.cache_clear()
        wf_telemetry.TELEMETRY._flags = {
            "tracing_enabled": True,
            "tracing_langfuse": True,
        }

    def tearDown(self):
        lf.get_callback_handler.cache_clear()
        wf_telemetry.TELEMETRY._flags = {
            "tracing_enabled": True,
            "tracing_langfuse": True,
        }

    def test_allowed_by_default(self):
        self.assertTrue(lf.tracing_allowed())

    def test_the_master_switch_stops_tracing(self):
        wf_telemetry.TELEMETRY._flags = {"tracing_enabled": False}
        self.assertFalse(lf.tracing_allowed())
        self.assertIsNone(lf.get_callback_handler())

    def test_the_langfuse_sink_can_be_stopped_alone(self):
        wf_telemetry.TELEMETRY._flags = {
            "tracing_enabled": True,
            "tracing_langfuse": False,
        }
        self.assertFalse(lf.tracing_allowed())
        self.assertIsNone(lf.get_callback_handler())

    def test_a_handler_attached_once_stops_forwarding_when_switched_off(self):
        """The bug this guards, and it is subtle.

        The handler is attached when a pooled model client is constructed, and
        that client then lives for the life of the process. Returning None from
        the accessor while the switch is off therefore only affected clients
        built *after* the flip — every already-warm pool kept tracing, and a
        service that booted with tracing off could never be switched on. The
        console switch appeared to work and did nothing.

        So the test does what the pool does: take the handler ONCE, then flip
        the switch, then use the handler it already has.
        """
        calls = []

        class FakeInner:
            raise_error = False

            def on_llm_end(self, *args, **kwargs):
                calls.append(args)
                return "forwarded"

        lf._switched.cache_clear()
        with mock.patch.object(lf, "_build_callback_handler", return_value=FakeInner()):
            handler = lf.get_callback_handler()          # what the pool keeps
            self.assertIsNotNone(handler)

            self.assertEqual(handler.on_llm_end("first"), "forwarded")
            self.assertEqual(len(calls), 1)

            wf_telemetry.TELEMETRY._flags = {"tracing_enabled": False}
            self.assertIsNone(handler.on_llm_end("while off"))
            self.assertEqual(len(calls), 1, "traced while the switch was off")

            wf_telemetry.TELEMETRY._flags = {"tracing_enabled": True}
            self.assertEqual(handler.on_llm_end("back on"), "forwarded")
            self.assertEqual(len(calls), 2)
        lf._switched.cache_clear()

    def test_the_wrapper_passes_through_langchain_flags(self):
        """LangChain reads these off the handler instance; a wrapper that hid
        them would change dispatch behaviour."""

        class FakeInner:
            raise_error = True
            ignore_chain = True

            def on_llm_end(self, *a, **k):
                return None

        lf._switched.cache_clear()
        with mock.patch.object(lf, "_build_callback_handler", return_value=FakeInner()):
            handler = lf.get_callback_handler()
            self.assertTrue(handler.raise_error)
            self.assertTrue(handler.ignore_chain)
        lf._switched.cache_clear()

    def test_prompt_management_survives_tracing_being_off(self):
        """The same client serves prompts. Switching tracing off must not drop
        every prompt back to its in-code fallback."""
        import inspect

        source = inspect.getsource(lf.get_langfuse_client)
        self.assertNotIn("tracing_allowed", source)

    def test_a_broken_flag_source_fails_open(self):
        with mock.patch.object(
            wf_telemetry.TELEMETRY,
            "tracing_enabled",
            side_effect=RuntimeError("gateway unreachable"),
        ):
            self.assertTrue(lf.tracing_allowed())


class FlagPropagationTests(unittest.TestCase):
    """What the telemetry client does with the switches it is handed."""

    def setUp(self):
        self.client = wf_telemetry.Telemetry()

    def test_defaults_are_permissive(self):
        """A service that has never reached the gateway behaves as it did
        before the switch existed."""
        self.assertTrue(self.client.tracing_enabled())
        self.assertTrue(self.client.tracing_enabled("langfuse"))

    def test_master_beats_a_sink_left_on(self):
        self.client._flags = {"tracing_enabled": False, "tracing_langfuse": True}
        self.assertFalse(self.client.tracing_enabled())
        self.assertFalse(self.client.tracing_enabled("langfuse"))

    def test_an_unknown_sink_is_allowed(self):
        self.client._flags = {"tracing_enabled": True}
        self.assertTrue(self.client.tracing_enabled("something_new"))

    def test_the_flags_url_is_derived_from_the_ingest_url(self):
        """Two URLs that must agree are two URLs that eventually will not."""
        with mock.patch.dict(
            "os.environ",
            {
                "ANALYTICS_INGEST_URL": "http://gw/api/v1/analytics/internal/events",
                "ANALYTICS_INGEST_SECRET": "s",
                "ANALYTICS_ENABLED": "false",
            },
            clear=False,
        ):
            client = wf_telemetry.Telemetry()
            with mock.patch.object(client, "_refresh_flags"):
                client.start(app="foodscholar")
            self.assertEqual(
                client._flags_url, "http://gw/api/v1/analytics/runtime-flags"
            )
            # Polls for the switch even with reporting off: an operator must be
            # able to stop tracing without turning analytics on first.
            self.assertTrue(client._polling)
            self.assertFalse(client.enabled)

    def test_no_secret_means_no_polling(self):
        with mock.patch.dict(
            "os.environ",
            {"ANALYTICS_INGEST_URL": "http://gw/api/v1/analytics/internal/events"},
            clear=True,
        ):
            client = wf_telemetry.Telemetry()
            client.start(app="foodscholar")
            self.assertFalse(client._polling)
            self.assertFalse(client.enabled)


if __name__ == "__main__":
    unittest.main()


class TestTheSwitchedHandlerIsAcceptedByPydantic:
    """`ChatGroq` is a Pydantic model that validates `callbacks` with
    `is_instance_of`. A duck-typed wrapper is rejected at construction, which
    is not a degraded trace — it is a service that will not start.

    This happened in production: FoodScholar crashed on import with
    `Input should be an instance of BaseCallbackHandler`.
    """

    def test_it_is_a_real_callback_handler(self):
        from langchain_core.callbacks.base import BaseCallbackHandler

        from backend.langfuse import _SwitchedHandler

        assert issubclass(_SwitchedHandler, BaseCallbackHandler)

    def test_a_pydantic_model_accepts_it(self):
        """The exact validation that crashed the service."""
        from typing import List, Optional

        from langchain_core.callbacks.base import BaseCallbackHandler
        from pydantic import BaseModel

        from backend.langfuse import _SwitchedHandler

        class ModelLike(BaseModel):
            callbacks: Optional[List[BaseCallbackHandler]] = None
            model_config = {"arbitrary_types_allowed": True}

        ModelLike(callbacks=[_SwitchedHandler(BaseCallbackHandler())])

    def test_every_callback_delegates_rather_than_inheriting_a_no_op(self):
        """The trap that comes with subclassing.

        The base defines every `on_*` as a no-op, so ordinary attribute lookup
        finds those and a `__getattr__`-based wrapper is never consulted. Such
        a wrapper imports cleanly, starts cleanly, and traces nothing at all —
        a worse failure than the crash, because nothing reports it.
        """
        from langchain_core.callbacks.base import BaseCallbackHandler

        from backend.langfuse import _SwitchedHandler

        callbacks = [n for n in dir(BaseCallbackHandler) if n.startswith("on_")]
        assert callbacks, "no callbacks found — has LangChain moved them?"
        for name in callbacks:
            assert getattr(_SwitchedHandler, name) is not getattr(
                BaseCallbackHandler, name
            ), f"{name} still resolves to the base class no-op"

    def test_it_forwards_only_while_tracing_is_allowed(self, monkeypatch):
        from langchain_core.callbacks.base import BaseCallbackHandler

        import backend.langfuse as module

        seen = []

        class Inner(BaseCallbackHandler):
            def on_llm_start(self, *args, **kwargs):
                seen.append("start")
                return "forwarded"

        handler = module._SwitchedHandler(Inner())

        monkeypatch.setattr(module, "tracing_allowed", lambda: True)
        assert handler.on_llm_start({}, []) == "forwarded"
        assert seen == ["start"]

        seen.clear()
        monkeypatch.setattr(module, "tracing_allowed", lambda: False)
        assert handler.on_llm_start({}, []) is None
        assert seen == [], "the inner handler was called with tracing off"

    def test_the_flags_langchain_reads_survive_wrapping(self):
        from langchain_core.callbacks.base import BaseCallbackHandler

        from backend.langfuse import _SwitchedHandler

        class Inner(BaseCallbackHandler):
            run_inline = True

        # run_inline decides whether Langfuse's handler runs in the caller's
        # context, which is what makes the current trace id available.
        assert _SwitchedHandler(Inner()).run_inline is True
