"""
Tests for forcing the enrichment workers back into a running state.

These cover the two ways a worker ends up stopped that pause/resume cannot
describe or repair:

- The pause switch is a Redis key with no expiry, so a pause set at any point
  survives every deploy and stays set until someone clears it.
- A worker thread that dies leaves `_running` True behind, and `start()` reads
  that flag and returns without doing anything — so the worker reports itself
  as running while nothing is processed, and cannot be revived by starting it.
"""

import threading
import time

import pytest

from workers.enrichment_job_worker import EnrichmentJobWorker
from workers.enrichment_worker import BackgroundEnrichmentWorker


class FakeJobService:
    """Just enough of EnrichmentJobService for the worker's restart path."""

    QUEUE_KEY = "enrichment:queue"

    def __init__(self, paused: bool = False):
        self._paused = paused
        self.pause_writes: list[bool] = []

    def is_sweeper_paused(self) -> bool:
        return self._paused

    def set_sweeper_paused(self, paused: bool) -> bool:
        self._paused = paused
        self.pause_writes.append(paused)
        return paused

    def pending_jobs(self) -> int:
        return 0


class FakeRedisClient:
    """`get_stats` reads the sweep cursor, so the fake needs a client shape."""

    class _Client:
        def get(self, _key):
            return None

    def __init__(self):
        self.client = self._Client()


def build_sweeper(paused: bool = False):
    """
    A sweeper wired to fakes.

    The constructor otherwise builds a real Redis client and a Groq-backed
    enrichment agent, neither of which the thread-lifecycle paths touch.
    """
    worker = BackgroundEnrichmentWorker(
        redis_client=FakeRedisClient(),
        enrichment_agent=object(),
        job_service=FakeJobService(paused=paused),
    )
    # The run loop talks to Redis and the catalog; restart only cares about
    # thread lifecycle, so park the thread on the shutdown event.
    worker._run = lambda: worker._shutdown_event.wait()
    return worker


@pytest.fixture
def sweeper():
    worker = build_sweeper()
    yield worker
    worker._running = False
    worker._shutdown_event.set()


@pytest.fixture
def job_worker():
    worker = EnrichmentJobWorker(job_service=FakeJobService())
    worker._run = lambda: worker._shutdown_event.wait()
    yield worker
    worker._running = False
    worker._shutdown_event.set()


def _wait_until_alive(worker, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if worker.is_alive():
            return True
        time.sleep(0.01)
    return False


class TestStaleRunningFlag:
    def test_start_refuses_to_revive_a_dead_thread(self, sweeper):
        """The bug restart exists to fix: start() trusts a flag, not the thread."""
        sweeper._running = True
        sweeper._thread = None

        sweeper.start()

        assert not sweeper.is_alive()

    def test_restart_revives_a_dead_thread(self, sweeper):
        sweeper._running = True
        sweeper._thread = None

        result = sweeper.restart()

        assert _wait_until_alive(sweeper)
        assert result["restarted"] is True
        assert result["thread_was_alive"] is False

    def test_stats_report_a_dead_thread_as_stalled(self, sweeper):
        sweeper._running = True
        sweeper._thread = None

        stats = sweeper.get_stats()

        assert stats["running"] is True
        assert stats["alive"] is False
        # `stalled` is what distinguishes "died" from "stopped on purpose".
        assert stats["stalled"] is True

    def test_a_healthy_worker_is_not_stalled(self, sweeper):
        sweeper.start()
        assert _wait_until_alive(sweeper)

        stats = sweeper.get_stats()

        assert stats["alive"] is True
        assert stats["stalled"] is False


class TestStalePause:
    def test_restart_clears_the_pause_switch(self):
        worker = build_sweeper(paused=True)
        service = worker.job_service

        try:
            result = worker.restart(resume=True)

            assert service.is_sweeper_paused() is False
            assert result["pause_switch_was_set"] is True
            assert result["resumed"] is True
        finally:
            worker._running = False
            worker._shutdown_event.set()

    def test_restart_can_leave_the_pause_in_place(self):
        """`resume=False` restarts the thread without overriding a deliberate pause."""
        worker = build_sweeper(paused=True)
        service = worker.job_service

        try:
            result = worker.restart(resume=False)

            assert service.is_sweeper_paused() is True
            assert service.pause_writes == []
            assert result["resumed"] is False
        finally:
            worker._running = False
            worker._shutdown_event.set()

    def test_restart_survives_redis_being_down(self, sweeper):
        """A dead thread must still be rebuilt when the pause switch is unreachable."""

        def explode():
            raise RuntimeError("redis is down")

        sweeper.job_service.is_sweeper_paused = explode

        result = sweeper.restart(resume=True)

        assert _wait_until_alive(sweeper)
        assert result["restarted"] is True


class TestRestartIsRepeatable:
    def test_restarting_a_running_worker_leaves_one_thread(self, sweeper):
        sweeper.start()
        assert _wait_until_alive(sweeper)
        before = threading.active_count()

        sweeper.restart()
        assert _wait_until_alive(sweeper)

        # The old thread is joined before the new one starts, so restarting
        # repeatedly must not accumulate parked threads.
        assert threading.active_count() == before
        assert sweeper.get_stats()["stalled"] is False

    def test_job_worker_restart_revives_a_dead_thread(self, job_worker):
        job_worker._running = True
        job_worker._thread = None

        result = job_worker.restart()

        assert _wait_until_alive(job_worker)
        assert result["restarted"] is True
        assert result["thread_was_alive"] is False

    def test_job_worker_stats_report_stalled(self, job_worker):
        job_worker._running = True
        job_worker._thread = None

        stats = job_worker.get_stats()

        assert stats["running"] is True
        assert stats["stalled"] is True
