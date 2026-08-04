"""
Tests for model-call backoff and extraction retry.

Extraction makes two to three model calls per page, so several replicas walking
PDFs concurrently can hold a rate limit open for minutes. Without backoff every
page of every concurrent run fails inside that window and whole jobs die; with
naive fixed backoff the workers stay synchronised and keep colliding.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _RateLimited(Exception):
    def __init__(self, message="Rate limit reached", status_code=429, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        if retry_after is not None:
            self.retry_after = retry_after


class _BadRequest(Exception):
    def __init__(self, message="Invalid schema"):
        super().__init__(message)
        self.status_code = 400


class TestRetryClassification(unittest.TestCase):
    def test_rate_limits_and_server_errors_retry(self):
        from services.model_backoff import is_retryable

        self.assertTrue(is_retryable(_RateLimited()))
        self.assertTrue(is_retryable(Exception("503 Service Unavailable")))
        self.assertTrue(is_retryable(Exception("Connection reset by peer")))
        self.assertTrue(is_retryable(TimeoutError("timed out")))
        self.assertTrue(is_retryable(Exception("The engine is overloaded")))

    def test_client_errors_do_not_retry(self):
        """Retrying a malformed request burns quota on something that cannot work."""
        from services.model_backoff import is_retryable

        self.assertFalse(is_retryable(_BadRequest()))
        self.assertFalse(is_retryable(Exception("invalid api key")))

    def test_unrecognised_errors_are_treated_as_permanent(self):
        from services.model_backoff import is_retryable

        self.assertFalse(is_retryable(ValueError("something structural")))


class TestRetryDelay(unittest.TestCase):
    def test_delay_grows_and_is_capped(self):
        from services.model_backoff import retry_delay

        # Full jitter draws from [0, backoff], so assert the ceiling.
        for attempt, ceiling in ((1, 2.0), (2, 4.0), (3, 8.0)):
            samples = [
                retry_delay(attempt, base_delay=2.0, max_delay=60.0)
                for _ in range(50)
            ]
            self.assertLessEqual(max(samples), ceiling)
            self.assertGreaterEqual(min(samples), 0.0)

        capped = [retry_delay(20, base_delay=2.0, max_delay=10.0) for _ in range(50)]
        self.assertLessEqual(max(capped), 10.0)

    def test_jitter_desynchronises_workers(self):
        """
        Without jitter, workers that failed together retry together and keep
        colliding — the exact pileup backoff is meant to break up.
        """
        from services.model_backoff import retry_delay

        samples = {retry_delay(3, base_delay=2.0) for _ in range(50)}
        self.assertGreater(len(samples), 10)

    def test_server_retry_after_wins(self):
        from services.model_backoff import retry_delay

        self.assertEqual(retry_delay(1, retry_after=7.5), 7.5)
        # Still bounded: a hostile or broken header cannot stall a worker.
        self.assertEqual(retry_delay(1, retry_after=9999, max_delay=30.0), 30.0)


class TestCallWithBackoff(unittest.TestCase):
    def test_succeeds_without_retrying(self):
        from services.model_backoff import call_with_backoff

        calls = []
        result = call_with_backoff(lambda: calls.append(1) or "ok", sleep=lambda _: None)

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        from services.model_backoff import call_with_backoff

        attempts = {"n": 0}
        slept = []

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _RateLimited()
            return "recovered"

        result = call_with_backoff(flaky, sleep=slept.append)

        self.assertEqual(result, "recovered")
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(len(slept), 2)

    def test_gives_up_after_max_attempts(self):
        from services.model_backoff import call_with_backoff

        attempts = {"n": 0}

        def always_limited():
            attempts["n"] += 1
            raise _RateLimited()

        with self.assertRaises(_RateLimited):
            call_with_backoff(always_limited, max_attempts=4, sleep=lambda _: None)

        self.assertEqual(attempts["n"], 4)

    def test_permanent_errors_fail_immediately(self):
        from services.model_backoff import call_with_backoff

        attempts = {"n": 0}

        def bad_request():
            attempts["n"] += 1
            raise _BadRequest()

        with self.assertRaises(_BadRequest):
            call_with_backoff(bad_request, sleep=lambda _: None)

        self.assertEqual(attempts["n"], 1, "a 400 must not be retried")

    def test_honours_retry_after_from_the_exception(self):
        from services.model_backoff import call_with_backoff

        slept = []
        attempts = {"n": 0}

        def limited_once():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _RateLimited(retry_after=3.0)
            return "ok"

        call_with_backoff(limited_once, sleep=slept.append)
        self.assertEqual(slept, [3.0])


class TestExtractionCallsAreWrapped(unittest.TestCase):
    """
    Every model call in the extraction path must go through backoff. One
    unwrapped call is enough to kill a job during a rate-limit window.
    """

    def test_every_responses_create_is_wrapped(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "services"
            / "guideline_extractor.py"
        ).read_text(encoding="utf-8")

        # Each call site should appear inside a call_with_backoff lambda.
        self.assertEqual(source.count("client.responses.create"), 3)
        self.assertEqual(source.count("call_with_backoff("), 3)
        for fragment in source.split("client.responses.create")[:-1]:
            self.assertIn(
                "call_with_backoff(",
                fragment[-400:],
                "a responses.create call is not wrapped in backoff",
            )


if __name__ == "__main__":
    unittest.main()
