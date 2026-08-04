import asyncio
import tempfile
import unittest
from pathlib import Path


class _FakeRedisClient:
    def __init__(self):
        self.values = {}
        self.queues = {}

    def ping(self):
        return True

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key):
        return 1 if self.values.pop(key, None) is not None else 0

    def exists(self, key):
        return 1 if key in self.values else 0

    def expire(self, key, seconds):
        return key in self.values

    def rpush(self, key, value):
        self.queues.setdefault(key, []).append(value)
        return len(self.queues[key])

    def blpop(self, key, timeout=0):
        queue = self.queues.get(key, [])
        if not queue:
            return None
        return key, queue.pop(0)

    def llen(self, key):
        return len(self.queues.get(key, []))


class _FakeRedisWrapper:
    def __init__(self):
        self.client = _FakeRedisClient()


class _FakeResultStore:
    def __init__(self):
        self.results = {}

    async def fetch_result(self, artifact_uuid):
        return self.results.get(artifact_uuid)

    def upsert_result(self, artifact_uuid, result):
        self.results[artifact_uuid] = result


class _FakeArtifactsAPI:
    def download_to(self, artifact_uuid, destination_path):
        Path(destination_path).write_bytes(f"pdf:{artifact_uuid}".encode("utf-8"))


class _FakeCreatedGuideline:
    def __init__(self, guideline_id, **kwargs):
        self.id = guideline_id
        self.sequence_no = kwargs.get("sequence_no")
        self.rule_text = kwargs.get("rule_text")
        self.page_no = kwargs.get("page_no")
        self.action_type = kwargs.get("action_type")
        self.payload = kwargs


class _FakeGuideGuidelinesCollection:
    """
    Stands in for the SDK proxy, and counts requests.

    The counters exist so a test can assert the import issues one bulk request
    rather than one per rule — the difference between a 300-rule import costing
    1 round trip and costing 300.
    """

    def __init__(self, existing=None):
        self._items = list(existing or [])
        self._created = 0
        self.create_calls = 0
        self.bulk_calls = 0
        self.fetch_all_calls = 0
        self.imported = []

    def __getitem__(self, item):
        return self._items[item]

    def fetch_all(self, *, page_size=500, fl=None):
        self.fetch_all_calls += 1
        return [
            {
                "id": getattr(item, "id", None),
                "sequence_no": getattr(item, "sequence_no", None),
                "rule_text": getattr(item, "rule_text", None),
            }
            for item in self._items
        ]

    def bulk_import(self, guidelines):
        self.bulk_calls += 1
        self.imported.extend(guidelines)
        for payload in guidelines:
            self._created += 1
            self._items.append(
                _FakeCreatedGuideline(
                    guideline_id=f"guideline-{self._created}", **payload
                )
            )
        return {"guide_urn": "urn:guide:test", "imported_count": len(guidelines)}

    def create(self, **kwargs):
        self.create_calls += 1
        self._created += 1
        guideline = _FakeCreatedGuideline(
            guideline_id=f"guideline-{self._created}",
            **kwargs,
        )
        self._items.append(guideline)
        return guideline


class _FakeGuide:
    def __init__(self, guide_id, existing=None):
        self.id = guide_id
        self.guidelines = _FakeGuideGuidelinesCollection(existing=existing)


class _FakeGuidesAPI:
    def __init__(self, guides=None):
        self._guides = guides or {}

    def get(self, guide_id):
        if guide_id not in self._guides:
            self._guides[guide_id] = _FakeGuide(guide_id)
        return self._guides[guide_id]


class _FakePlatformClient:
    def __init__(self, guides=None):
        self.artifacts = _FakeArtifactsAPI()
        self.guides = _FakeGuidesAPI(guides=guides)


class _FakePlatformPool:
    def __init__(self, guides=None):
        self.client = _FakePlatformClient(guides=guides)

    def get_client(self):
        return self.client

    def return_client(self, client):
        return None


class GuidelineJobServiceTests(unittest.TestCase):
    def _make_service(self, tmpdir, guides=None):
        from services.guideline_extractor import GuidelineExtractorService
        from services.guideline_jobs import GuidelineJobService

        redis_wrapper = _FakeRedisWrapper()
        result_store = _FakeResultStore()
        extractor = GuidelineExtractorService(workspace_root=tmpdir)
        platform_pool = _FakePlatformPool(guides=guides)
        service = GuidelineJobService(
            redis_client=redis_wrapper,
            extractor=extractor,
            result_store=result_store,
            platform_pool=platform_pool,
        )
        return service, result_store

    def test_enqueue_job_dedupes_while_queued(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service, _ = self._make_service(tmpdir)

            first = service.enqueue_job(artifact_uuid, model="gpt-test", dpi=144)
            second = service.enqueue_job(artifact_uuid, model="gpt-other", dpi=72)

            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(service.redis.client.llen(service.queue_key), 1)
            self.assertEqual(service.get_job_state(artifact_uuid)["status"], "queued")

    def test_running_job_progress_updates_current_page(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service, _ = self._make_service(tmpdir)

            service.enqueue_job(artifact_uuid)
            job = service.pop_next_job(timeout=0)

            self.assertTrue(service.try_claim_job(artifact_uuid))
            service.mark_running(job)
            service.update_progress(artifact_uuid, current_page=3, total_pages=10)

            state = service.get_job_state(artifact_uuid)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["current_page"], 3)
            self.assertEqual(state["total_pages"], 10)

    def test_running_job_is_not_requeued(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service, _ = self._make_service(tmpdir)

            first = service.enqueue_job(artifact_uuid)
            job = service.pop_next_job(timeout=0)
            service.try_claim_job(artifact_uuid)
            service.mark_running(job)

            second = service.enqueue_job(artifact_uuid)
            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(second["status"], "running")

    def test_status_falls_back_to_persisted_result(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service, result_store = self._make_service(tmpdir)

            from models.guidelines import GuidelineExtractionResponse

            result_store.upsert_result(
                artifact_uuid,
                GuidelineExtractionResponse(
                    artifact_uuid=artifact_uuid,
                    workspace_root=tmpdir,
                    artifact_dir=f"{tmpdir}/{artifact_uuid}",
                    pdf_path=f"{tmpdir}/{artifact_uuid}/source.pdf",
                    model="gpt-5.4",
                    dpi=144,
                    extracted_at="2026-03-17T00:00:00+00:00",
                    total_pages=12,
                    total_processed_pages=5,
                    total_skipped_pages=7,
                    total_guidelines=9,
                    total_unique_guidelines=6,
                    processed_pages=[],
                    skipped_pages=[],
                    guidelines=[],
                    unique_guidelines=[],
                ),
            )

            response = asyncio.run(service.get_job_response(artifact_uuid))
            self.assertEqual(response.status, "succeeded")
            self.assertIsNotNone(response.result)
            self.assertEqual(response.result.total_pages, 12)

    def test_download_artifact_pdf_uses_platform_client(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service, _ = self._make_service(tmpdir)

            storage = service.download_artifact_pdf(artifact_uuid)

            self.assertTrue(Path(storage.pdf_path).exists())
            self.assertEqual(
                Path(storage.pdf_path).read_bytes(),
                f"pdf:{artifact_uuid}".encode("utf-8"),
            )

    def test_guideline_import_request_normalizes_legacy_action_type(self):
        from models.guidelines import GuidelineImportRequest

        request = GuidelineImportRequest(
            guide_id="mediterranean_guide",
            action_type="encourage",
        )

        self.assertEqual(request.action_type, "choose")

    def test_import_latest_result_to_guide_dry_run_dedupes_existing(self):
        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
        guide_id = "mediterranean_guide"

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = [
                _FakeCreatedGuideline(
                    "existing-1",
                    sequence_no=4,
                    rule_text="Eat vegetables daily",
                    page_no=2,
                    action_type="choose",
                )
            ]
            service, result_store = self._make_service(
                tmpdir,
                guides={guide_id: _FakeGuide(guide_id, existing=existing)},
            )

            from models.guidelines import GuidelineExtractionResponse

            result_store.upsert_result(
                artifact_uuid,
                GuidelineExtractionResponse(
                    artifact_uuid=artifact_uuid,
                    workspace_root=tmpdir,
                    artifact_dir=f"{tmpdir}/{artifact_uuid}",
                    pdf_path=f"{tmpdir}/{artifact_uuid}/source.pdf",
                    model="gpt-5.4",
                    dpi=144,
                    extracted_at="2026-03-18T00:00:00+00:00",
                    total_pages=12,
                    total_processed_pages=5,
                    total_skipped_pages=7,
                    total_guidelines=3,
                    total_unique_guidelines=2,
                    processed_pages=[],
                    skipped_pages=[],
                    guidelines=[
                        {"page": 3, "text": "Eat vegetables daily"},
                        {"page": 4, "text": "Prefer whole grains over refined grains"},
                        {"page": 8, "text": "Prefer whole grains over refined grains"},
                    ],
                    unique_guidelines=[
                        "Eat vegetables daily",
                        "Prefer whole grains over refined grains",
                    ],
                ),
            )

            response = asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=True,
                    dedupe_against_guide=True,
                    existing_scan_limit=100,
                )
            )

            self.assertTrue(response.dry_run)
            self.assertEqual(response.total_candidates, 2)
            self.assertEqual(response.total_skipped, 1)
            self.assertEqual(response.total_created, 0)
            self.assertEqual(response.next_sequence_no_start, 5)
            self.assertEqual(response.items[0].status, "skipped_existing")
            self.assertEqual(response.items[0].action_type, "choose")
            self.assertEqual(response.items[1].status, "would_create")
            self.assertEqual(response.items[1].action_type, "choose")
            self.assertEqual(response.items[1].sequence_no, 5)

    def _seed_result(self, result_store, artifact_uuid, tmpdir, rules, *, schema_version=2):
        from models.guidelines import GuidelineExtractionResponse

        result_store.upsert_result(
            artifact_uuid,
            GuidelineExtractionResponse(
                artifact_uuid=artifact_uuid,
                workspace_root=tmpdir,
                artifact_dir=f"{tmpdir}/{artifact_uuid}",
                pdf_path=f"{tmpdir}/{artifact_uuid}/source.pdf",
                model="gpt-5.4",
                dpi=144,
                extracted_at="2026-03-18T00:00:00+00:00",
                total_pages=len(rules),
                total_processed_pages=len(rules),
                total_skipped_pages=0,
                total_guidelines=len(rules),
                total_unique_guidelines=len(rules),
                processed_pages=[
                    {
                        "page": 1,
                        "page_summary": "Portion guidance for 1-4 year olds.",
                        "guideline_count": len(rules),
                    }
                ],
                skipped_pages=[],
                guidelines=rules,
                unique_guidelines=[rule["text"] for rule in rules],
                schema_version=schema_version,
            ),
        )

    def test_real_import_issues_one_bulk_request_not_one_per_rule(self):
        """
        A 300-rule import must cost one round trip, not 300. Creating rules
        individually also made the server re-resolve the guide and its
        artifacts for every single one.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
            guide_id = "urn:guide:test"
            guide = _FakeGuide(guide_id)
            service, result_store = self._make_service(
                tmpdir, guides={guide_id: guide}
            )

            rules = [
                {"page": index + 1, "text": f"Rule number {index} about food."}
                for index in range(300)
            ]
            self._seed_result(result_store, artifact_uuid, tmpdir, rules)

            response = asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=False,
                )
            )

            self.assertEqual(response.total_created, 300)
            self.assertEqual(guide.guidelines.create_calls, 0)
            # 300 rules at a batch size of 500 is a single request.
            self.assertEqual(guide.guidelines.bulk_calls, 1)

    def test_large_import_is_chunked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
            guide_id = "urn:guide:test"
            guide = _FakeGuide(guide_id)
            service, result_store = self._make_service(
                tmpdir, guides={guide_id: guide}
            )

            rules = [
                {"page": index + 1, "text": f"Rule number {index} about food."}
                for index in range(1100)
            ]
            self._seed_result(result_store, artifact_uuid, tmpdir, rules)

            response = asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=False,
                )
            )

            self.assertEqual(response.total_created, 1100)
            self.assertEqual(guide.guidelines.bulk_calls, 3)

    def test_existing_scan_uses_one_paged_read_not_one_per_rule(self):
        """
        Reading existing rules by slicing the proxy returns lazy entities that
        fetch themselves on first attribute access — one HTTP GET per rule.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
            guide_id = "urn:guide:test"
            existing = [
                _FakeCreatedGuideline(
                    guideline_id=f"existing-{index}",
                    sequence_no=index + 1,
                    rule_text=f"Existing rule {index}",
                )
                for index in range(400)
            ]
            guide = _FakeGuide(guide_id, existing=existing)
            service, result_store = self._make_service(
                tmpdir, guides={guide_id: guide}
            )

            self._seed_result(
                result_store,
                artifact_uuid,
                tmpdir,
                [{"page": 1, "text": "A brand new rule about vegetables."}],
            )

            response = asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=False,
                )
            )

            self.assertEqual(guide.guidelines.fetch_all_calls, 1)
            self.assertTrue(response.existing_scan_complete)
            # Numbering continues past the 400 existing rules.
            self.assertEqual(response.next_sequence_no_start, 401)

    def test_page_summary_and_provenance_reach_the_created_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
            guide_id = "urn:guide:test"
            guide = _FakeGuide(guide_id)
            service, result_store = self._make_service(
                tmpdir, guides={guide_id: guide}
            )

            self._seed_result(
                result_store,
                artifact_uuid,
                tmpdir,
                [
                    {
                        "page": 1,
                        "text": "Provide portions of red meat twice a week.",
                        "section_label": "Protein foods",
                        "life_stage": ["early_childhood"],
                        "action_type_hint": "eat",
                    }
                ],
            )

            asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=False,
                )
            )

            imported = guide.guidelines.imported[0]
            self.assertEqual(
                imported["page_summary"], "Portion guidance for 1-4 year olds."
            )
            self.assertEqual(imported["section_label"], "Protein foods")
            self.assertEqual(imported["life_stage"], ["early_childhood"])
            # The per-rule hint must beat the batch-wide default.
            self.assertEqual(imported["action_type"], "eat")
            self.assertEqual(imported["extractor_name"], "guideline_extractor")
            self.assertEqual(imported["extraction_model"], "gpt-5.4")

    def test_v1_results_still_import(self):
        """Results stored before facet capture are never migrated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"
            guide_id = "urn:guide:test"
            guide = _FakeGuide(guide_id)
            service, result_store = self._make_service(
                tmpdir, guides={guide_id: guide}
            )

            self._seed_result(
                result_store,
                artifact_uuid,
                tmpdir,
                [{"page": 1, "text": "Eat vegetables daily."}],
                schema_version=1,
            )

            response = asyncio.run(
                service.import_latest_result_to_guide(
                    artifact_uuid=artifact_uuid,
                    guide_id=guide_id,
                    dry_run=False,
                )
            )

            self.assertEqual(response.total_created, 1)
            self.assertEqual(response.schema_version, 1)
            imported = guide.guidelines.imported[0]
            # No facet hints exist on a v1 row, so none are invented.
            self.assertNotIn("life_stage", imported)


class GuidelineJobConcurrencyTests(unittest.TestCase):
    """
    Concurrent and interrupted extractions.

    Jobs are popped destructively, so a worker that dies mid-extraction leaves a
    status of "running" with nothing working on it. Without orphan detection
    that artifact can never be extracted again — the status blocks re-queueing
    and the lock outlives the process.
    """

    def _service(self, tmpdir):
        service, _ = GuidelineJobServiceTests()._make_service(tmpdir)
        return service

    def test_second_enqueue_for_the_same_artifact_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            first = service.enqueue_job(artifact)
            second = service.enqueue_job(artifact)

            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(
                service.redis.client.llen(service.queue_key),
                1,
                "the same artifact was queued twice",
            )

    def test_a_claimed_job_is_not_treated_as_orphaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(artifact)
            service.try_claim_job(artifact)
            service.mark_running(job)

            state = service.get_job_state(artifact)
            self.assertFalse(service.is_orphaned(state))
            # A live job must not be displaced by another enqueue.
            self.assertEqual(service.enqueue_job(artifact)["job_id"], job["job_id"])

    def test_a_dead_worker_leaves_an_orphan_that_can_be_requeued(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(artifact)
            service.pop_next_job(timeout=0)          # worker takes it
            service.try_claim_job(artifact)
            service.mark_running(job)

            # The process dies: the lock expires, nothing released it, and the
            # queue no longer holds the job.
            service.redis.client.delete(service._lock_key(artifact))

            state = service.get_job_state(artifact)
            self.assertEqual(state["status"], "running")
            self.assertTrue(
                service.is_orphaned(state),
                "a running job with no live worker must be detectable",
            )

            requeued = service.enqueue_job(artifact)
            self.assertNotEqual(requeued["job_id"], job["job_id"])
            self.assertEqual(service.redis.client.llen(service.queue_key), 1)

    def test_stalled_jobs_are_reported_as_such(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(artifact)
            service.try_claim_job(artifact)
            service.mark_running(job)
            service.redis.client.delete(service._lock_key(artifact))

            response = asyncio.run(service.get_job_response(artifact))
            self.assertEqual(response.status, "stalled")
            self.assertTrue(response.stalled)

    def test_force_requeues_a_live_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            first = service.enqueue_job(artifact)
            service.try_claim_job(artifact)
            service.mark_running(first)

            second = service.enqueue_job(artifact, force=True)
            self.assertNotEqual(second["job_id"], first["job_id"])

    def test_a_superseded_job_is_skipped_by_the_worker(self):
        """
        Two jobs for one artifact must not both run. The stale check compares
        job ids, so only the most recently registered one proceeds.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            first = service.enqueue_job(artifact)
            second = service.enqueue_job(artifact, force=True)

            current = service.get_job_state(artifact)
            self.assertEqual(current["job_id"], second["job_id"])
            self.assertFalse(service.is_current_job(first))
            self.assertTrue(service.is_current_job(second))

    def test_a_transient_failure_is_requeued_not_abandoned(self):
        """
        Extraction fails for transient reasons far more often than structural
        ones — a rate limit outlasting the per-call backoff, a flaky download.
        Making an operator re-trigger those by hand left runs sitting failed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(artifact)
            service.pop_next_job(timeout=0)
            outcome = service.retry_or_fail(artifact, job, "429 rate limit")

            self.assertEqual(outcome["status"], "queued")
            self.assertEqual(outcome["attempt"], 2)
            self.assertEqual(service.redis.client.llen(service.queue_key), 1)

    def test_retries_stop_at_the_attempt_cap(self):
        """An unparseable PDF must stop, not cycle through the queue forever."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(artifact)
            max_attempts = job["max_attempts"]

            outcome = job
            retries = 0
            for _ in range(max_attempts + 2):
                outcome = service.retry_or_fail(artifact, outcome, "boom")
                if outcome["status"] == "failed":
                    break
                retries += 1

            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(
                retries,
                max_attempts - 1,
                "the first attempt is the enqueue, so there are cap-1 retries",
            )
            # One queue entry per attempt: the original plus each retry.
            self.assertEqual(
                service.redis.client.llen(service.queue_key), max_attempts
            )

    def test_a_retry_preserves_the_extraction_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            job = service.enqueue_job(
                artifact, guide_id="urn:guide:test", profile_document=False
            )
            retried = service.retry_or_fail(artifact, job, "boom")

            # Losing guide_id on a retry would silently produce context-free rules.
            self.assertEqual(retried["guide_id"], "urn:guide:test")
            self.assertFalse(retried["profile_document"])
            self.assertEqual(retried["job_id"], job["job_id"])

    def test_lock_prevents_two_workers_on_one_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            artifact = "123e4567-e89b-12d3-a456-426614174000"

            self.assertTrue(service.try_claim_job(artifact))
            self.assertFalse(
                service.try_claim_job(artifact),
                "a second worker acquired the same artifact",
            )
            service.release_lock(artifact)
            self.assertTrue(service.try_claim_job(artifact))


if __name__ == "__main__":
    unittest.main()
