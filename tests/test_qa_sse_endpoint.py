"""Tests for the SSE endpoint framing of POST /qa/ask/stream."""

import json
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.qa_pipeline.events import PipelineEvent


def _app():
    import api.v1.qa as qa_api

    app = FastAPI()
    app.include_router(qa_api.router, prefix="/api/v1")
    return app, qa_api


def _parse_frames(raw: str):
    """Parse an SSE body into (event, data) pairs plus comment frames."""
    events = []
    comments = []
    for frame in raw.split("\n\n"):
        if not frame.strip():
            continue
        if frame.startswith(":"):
            comments.append(frame)
            continue
        lines = frame.split("\n")
        name = next(
            line[len("event: "):] for line in lines if line.startswith("event: ")
        )
        data = next(
            line[len("data: "):] for line in lines if line.startswith("data: ")
        )
        events.append((name, json.loads(data)))
    return events, comments


class SseEndpointTests(unittest.TestCase):
    def test_stream_frames_and_terminal_done(self):
        app, qa_api = _app()

        async def fake_pipeline(_request):
            yield PipelineEvent("stage.start", {"request_id": "r1", "seq": 0})
            yield PipelineEvent(
                "answer_delta", {"text": "Hello", "request_id": "r1", "seq": 1}
            )
            yield PipelineEvent("done", {"request_id": "r1", "seq": 2})

        with patch.object(
            qa_api.qa_service, "run_pipeline", side_effect=fake_pipeline
        ):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/v1/qa/ask/stream",
                json={"question": "Are whole grains healthy?"},
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith("text/event-stream")
                )
                self.assertEqual(response.headers["x-accel-buffering"], "no")
                body = "".join(response.iter_text())

        events, _comments = _parse_frames(body)
        self.assertEqual(
            [name for name, _ in events], ["stage.start", "answer_delta", "done"]
        )
        self.assertEqual(events[1][1]["text"], "Hello")

    def test_midstream_exception_becomes_error_event(self):
        app, qa_api = _app()

        async def broken_pipeline(_request):
            yield PipelineEvent("stage.start", {"request_id": "r1", "seq": 0})
            raise RuntimeError("boom")

        with patch.object(
            qa_api.qa_service, "run_pipeline", side_effect=broken_pipeline
        ):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/v1/qa/ask/stream",
                json={"question": "Are whole grains healthy?"},
            ) as response:
                self.assertEqual(response.status_code, 200)
                body = "".join(response.iter_text())

        events, _ = _parse_frames(body)
        self.assertEqual(events[0][0], "stage.start")
        self.assertEqual(events[-1][0], "error")
        self.assertIn("detail", events[-1][1])

    def test_stream_stops_after_terminal_clarification(self):
        app, qa_api = _app()

        async def clarifying_pipeline(_request):
            yield PipelineEvent("stage.start", {"request_id": "r1", "seq": 0})
            yield PipelineEvent(
                "clarification",
                {"request_id": "r1", "seq": 1, "needs_clarification": True},
            )
            yield PipelineEvent(  # pragma: no cover - must never be reached
                "done", {"request_id": "r1", "seq": 2}
            )

        with patch.object(
            qa_api.qa_service, "run_pipeline", side_effect=clarifying_pipeline
        ):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/v1/qa/ask/stream",
                json={"question": "How much fiber should children eat?"},
            ) as response:
                body = "".join(response.iter_text())

        events, _ = _parse_frames(body)
        self.assertEqual([name for name, _ in events], ["stage.start", "clarification"])

    def test_invalid_model_is_rejected_before_streaming(self):
        app, qa_api = _app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/qa/ask/stream",
            json={
                "question": "Are whole grains healthy?",
                "mode": "advanced",
                "model": "not-a-real-model",
            },
        )
        self.assertNotEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
