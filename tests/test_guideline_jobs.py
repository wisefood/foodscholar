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


class _FakeGuideGuidelinesCollection:
    def __init__(self, existing=None):
        self._items = list(existing or [])
        self._created = 0

    def __getitem__(self, item):
        return self._items[item]

    def create(self, **kwargs):
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
                    action_type="encourage",
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
                    action_type="encourage",
                    existing_scan_limit=100,
                )
            )

            self.assertTrue(response.dry_run)
            self.assertEqual(response.total_candidates, 2)
            self.assertEqual(response.total_skipped, 1)
            self.assertEqual(response.total_created, 0)
            self.assertEqual(response.next_sequence_no_start, 5)
            self.assertEqual(response.items[0].status, "skipped_existing")
            self.assertEqual(response.items[1].status, "would_create")
            self.assertEqual(response.items[1].sequence_no, 5)


if __name__ == "__main__":
    unittest.main()
