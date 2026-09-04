"""The vendored correlation module.

`src/obs_context.py` is copied byte-for-byte into wisefood-api, foodchat,
foodscholar, RecipeWrangler and wisefood-data-api. It is tested here because
FoodScholar is where the correlation id earns its keep: a nutrition question
asked inside a FoodChat turn arrives with no Keycloak subject, so the *only*
way to attribute it to a user is the id the gateway assigned and forwarded.

Two properties matter and neither is obvious:

* a caller-supplied id reaches log lines and outbound headers, so a malformed
  one must be replaced outright rather than patched up;
* the formatters must not depend on the log filter having run — a handler the
  filter missed would otherwise raise while formatting, turning a logging
  change into a failed request.
"""
import json
import logging
import sys
import unittest

sys.path.insert(0, "src")

import obs_context  # noqa: E402


class CleaningTests(unittest.TestCase):
    def test_safe_ids_pass_through(self):
        for value in ("abc123", "ui-1.2:3", "a" * 64):
            self.assertEqual(obs_context.clean_id(value), value)

    def test_hostile_ids_are_rejected(self):
        for value in ("with space", "a" * 65, "carriage\rreturn",
                      "new\nline", "semi;colon", "", None):
            self.assertIsNone(obs_context.clean_id(value))

    def test_generated_ids_are_unique_and_safe(self):
        ids = {obs_context.new_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)
        for value in ids:
            self.assertEqual(obs_context.clean_id(value), value)


class OutboundTests(unittest.TestCase):
    def tearDown(self):
        obs_context.set_request_id(None)

    def test_no_id_means_no_header(self):
        obs_context.set_request_id(None)
        self.assertEqual(obs_context.outbound_headers(), {})

    def test_id_is_forwarded(self):
        obs_context.set_request_id("rid-1")
        self.assertEqual(
            obs_context.outbound_headers(), {"X-Request-Id": "rid-1"}
        )


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


class FormatterTests(unittest.TestCase):
    """The formatters must work with no filter installed."""

    def setUp(self):
        self.handler = _Capture()
        self.logger = logging.getLogger("obs_context_test")
        self.logger.handlers = [self.handler]
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        obs_context.set_request_id(None)
        self.logger.handlers = []

    def test_text_formatter_fills_the_id_without_a_filter(self):
        self.handler.setFormatter(
            obs_context.ContextTextFormatter("[%(request_id)s] %(message)s")
        )
        obs_context.set_request_id("rid-text")
        self.logger.info("hello")
        self.assertEqual(self.handler.lines, ["[rid-text] hello"])

    def test_text_formatter_outside_a_request(self):
        self.handler.setFormatter(
            obs_context.ContextTextFormatter("[%(request_id)s] %(message)s")
        )
        obs_context.set_request_id(None)
        self.logger.info("hello")
        self.assertEqual(self.handler.lines, ["[-] hello"])

    def test_json_formatter_keeps_extra_fields(self):
        """The text formatters silently dropped these, which is why JSON exists."""
        self.handler.setFormatter(obs_context.JsonFormatter())
        obs_context.set_request_id("rid-json")
        self.logger.warning(
            "qa.persisted", extra={"question_id": "q1", "latency_ms": 42}
        )
        payload = json.loads(self.handler.lines[0])
        self.assertEqual(payload["message"], "qa.persisted")
        self.assertEqual(payload["level"], "WARNING")
        self.assertEqual(payload["request_id"], "rid-json")
        self.assertEqual(payload["question_id"], "q1")
        self.assertEqual(payload["latency_ms"], 42)

    def test_json_formatter_survives_an_unserialisable_extra(self):
        self.handler.setFormatter(obs_context.JsonFormatter())
        self.logger.info("odd", extra={"blob": object()})
        payload = json.loads(self.handler.lines[0])
        self.assertEqual(payload["message"], "odd")

    def test_install_log_filter_is_idempotent(self):
        obs_context.install_log_filter()
        obs_context.install_log_filter()
        installed = [
            f
            for f in self.handler.filters
            if isinstance(f, obs_context.RequestIdFilter)
        ]
        self.assertEqual(len(installed), 1)


class MiddlewareTests(unittest.TestCase):
    """Driven through the real ASGI protocol, no web framework required."""

    def _run(self, headers, app=None):
        import asyncio

        seen = {}

        async def default_app(scope, receive, send):
            seen["request_id"] = obs_context.get_request_id()
            seen["client"] = obs_context.get_client()
            seen["state"] = dict(scope.get("state") or {})
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = obs_context.RequestContextMiddleware(app or default_app)
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/qa/ask",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in headers.items()
            ],
        }
        asyncio.run(middleware(scope, receive, send))
        start = next(m for m in sent if m["type"] == "http.response.start")
        response_headers = {
            k.decode().lower(): v.decode() for k, v in start["headers"]
        }
        return seen, response_headers

    def test_id_is_minted_when_absent_and_echoed(self):
        seen, headers = self._run({})
        self.assertEqual(len(seen["request_id"]), 32)
        self.assertEqual(headers["x-request-id"], seen["request_id"])

    def test_gateway_id_is_adopted(self):
        seen, headers = self._run({"X-Request-Id": "from-gateway"})
        self.assertEqual(seen["request_id"], "from-gateway")
        self.assertEqual(headers["x-request-id"], "from-gateway")
        self.assertEqual(seen["state"]["request_id"], "from-gateway")

    def test_hostile_id_is_replaced(self):
        seen, headers = self._run({"X-Request-Id": "bad value"})
        self.assertNotEqual(seen["request_id"], "bad value")
        self.assertEqual(len(seen["request_id"]), 32)

    def test_client_label_is_captured(self):
        seen, _ = self._run({"X-Client": "wisefood-client/2.1.0"})
        self.assertEqual(seen["client"], "wisefood-client/2.1.0")

    def test_context_is_cleared_after_the_request(self):
        self._run({"X-Request-Id": "transient"})
        self.assertIsNone(obs_context.get_request_id())

    def test_context_is_cleared_even_when_the_app_raises(self):
        async def exploding_app(scope, receive, send):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            self._run({"X-Request-Id": "transient"}, app=exploding_app)
        self.assertIsNone(obs_context.get_request_id())

    def test_non_http_scopes_pass_straight_through(self):
        import asyncio

        called = {}

        async def app(scope, receive, send):
            called["type"] = scope["type"]

        async def noop(*_args):
            return {}

        middleware = obs_context.RequestContextMiddleware(app)
        asyncio.run(middleware({"type": "lifespan"}, noop, noop))
        self.assertEqual(called["type"], "lifespan")


if __name__ == "__main__":
    unittest.main()
