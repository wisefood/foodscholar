import unittest


class _FakeRedisClient:
    def __init__(self):
        self.values = {}
        self.queues = {}
        self.sets = {}

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

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

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

    def sadd(self, key, value):
        members = self.sets.setdefault(key, set())
        if value in members:
            return 0
        members.add(value)
        return 1

    def srem(self, key, value):
        members = self.sets.setdefault(key, set())
        if value not in members:
            return 0
        members.discard(value)
        return 1

    def sismember(self, key, value):
        return value in self.sets.get(key, set())

    def lpush(self, key, value):
        self.queues.setdefault(key, []).insert(0, value)
        return len(self.queues[key])

    def ltrim(self, key, start, end):
        queue = self.queues.get(key, [])
        self.queues[key] = queue[start: end + 1]
        return True

    def lrange(self, key, start, end):
        queue = self.queues.get(key, [])
        return queue[start: end + 1]


class _FakeRedisWrapper:
    def __init__(self):
        self.client = _FakeRedisClient()


class _FakeArticle:
    """Stand-in for a wisefood client Article entity."""

    IMMUTABLE_FIELDS = ()

    def __init__(self, urn, *, enhance_fails=False):
        self.urn = urn
        self.sync = True
        self.saved = None
        self.enhanced = None
        self._enhance_fails = enhance_fails

    def save(self, only_dirty=False):
        self.saved = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"urn", "sync", "saved", "enhanced", "_enhance_fails"}
        }

    def enhance_self(self, *, agent, **fields):
        if self._enhance_fails:
            raise RuntimeError("enhance endpoint exploded")
        self.enhanced = {"agent": agent, **fields}


class _FakeArticlesProxy:
    def __init__(self, articles, *, missing=()):
        self._articles = articles
        self._missing = set(missing)

    def get(self, urn):
        if urn in self._missing:
            raise RuntimeError(f"404 not found: {urn}")
        return self._articles[urn]


class _FakeCatalogClient:
    def __init__(self, articles, *, missing=()):
        self.articles = _FakeArticlesProxy(articles, missing=missing)


class _FakeCatalogPool:
    def __init__(self, client):
        self._client = client
        self.returned = 0

    def get_client(self):
        return self._client

    def return_client(self, client):
        self.returned += 1


class _FakeEnrichmentAgent:
    def __init__(self, payload=None, fails=False):
        self.payload = payload or {
            "keywords": ["omega-3", "cardiovascular"],
            "tags": ["Nutrition"],
            "topics": ["Cardiometabolic"],
            "study_type": "RCT",
            "annotation_confidence": 0.8,
            "evaluation": {"verdict": ["Eat more fish"]},
            "annotations": {"abstract": "Simplified."},
        }
        self.fails = fails
        self.calls = []

    def enrich_article(self, article):
        self.calls.append(getattr(article, "urn", None))
        if self.fails:
            raise RuntimeError("agent exploded")
        return self.payload


def _make_service(articles=None, *, agent=None, missing=()):
    from services.enrichment_jobs import EnrichmentJobService

    articles = articles if articles is not None else {}
    pool = _FakeCatalogPool(_FakeCatalogClient(articles, missing=missing))
    service = EnrichmentJobService(
        redis_client=_FakeRedisWrapper(),
        enrichment_agent=agent or _FakeEnrichmentAgent(),
        catalog_pool=pool,
    )
    return service, pool


class TestEnrichmentFieldExtraction(unittest.TestCase):
    def test_enhance_fields_are_limited_to_catalog_accepted_keys(self):
        from services.enrichment_jobs import extract_enrichment_fields

        enhance, article, extras = extract_enrichment_fields(
            {
                "keywords": ["a", "b"],
                "tags": ["b", "c"],
                "study_type": "RCT",
                "evaluation": {"verdict": ["v1", "v2", "v3", "v4"]},
            }
        )

        # The /enhance endpoint validates keys and rejects anything else.
        self.assertEqual(set(enhance), {"ai_tags", "ai_category", "ai_key_takeaways"})
        # Keywords and tags merge into one deduped ai_tags list, order preserved.
        self.assertEqual(enhance["ai_tags"], ["a", "b", "c"])
        # Takeaways are capped at three.
        self.assertEqual(enhance["ai_key_takeaways"], ["v1", "v2", "v3"])
        self.assertEqual(article["study_type"], "RCT")
        self.assertIn("enriched_at", extras)

    def test_annotation_confidence_is_clamped(self):
        from services.enrichment_jobs import extract_enrichment_fields

        _, high, _ = extract_enrichment_fields({"annotation_confidence": 4.2})
        _, low, _ = extract_enrichment_fields({"annotation_confidence": -3})
        _, junk, _ = extract_enrichment_fields({"annotation_confidence": "nope"})

        self.assertEqual(high["annotation_confidence"], 1.0)
        self.assertEqual(low["annotation_confidence"], 0.0)
        self.assertNotIn("annotation_confidence", junk)

    def test_missing_fields_fall_back_to_catalog_safe_defaults(self):
        from services.enrichment_jobs import extract_enrichment_fields

        enhance, article, _ = extract_enrichment_fields({})

        self.assertEqual(enhance, {})
        self.assertEqual(article["tags"], ["Other"])
        self.assertEqual(article["topics"], ["Other"])
        self.assertEqual(article["hard_exclusion_flags"], ["None"])


class TestPersistEnrichment(unittest.TestCase):
    def test_persist_writes_fields_then_enhance(self):
        from services.enrichment_jobs import persist_enrichment

        article = _FakeArticle("urn:article:1")
        summary = persist_enrichment(
            article,
            {
                "keywords": ["omega-3"],
                "study_type": "RCT",
                "evaluation": {"verdict": ["Eat fish"]},
            },
        )

        self.assertEqual(article.study_type, "RCT")
        self.assertEqual(article.enhanced["agent"], "foodscholar-v1")
        self.assertEqual(article.enhanced["fields"]["ai_category"], "RCT")
        # sync is restored so the entity is not left in auto-write mode.
        self.assertTrue(article.sync)
        self.assertTrue(summary["enhanced"])
        self.assertEqual(summary["ai_key_takeaways"], ["Eat fish"])

    def test_enhance_failure_does_not_fail_the_run(self):
        from services.enrichment_jobs import persist_enrichment

        article = _FakeArticle("urn:article:1", enhance_fails=True)
        summary = persist_enrichment(article, {"study_type": "RCT"})

        # The PATCH already landed; /enhance is best-effort.
        self.assertEqual(article.study_type, "RCT")
        self.assertFalse(summary["enhanced"])

    def test_catalog_outage_during_save_is_classified(self):
        from services.enrichment_jobs import CatalogUnavailable, persist_enrichment

        article = _FakeArticle("urn:article:1")

        def _boom(only_dirty=False):
            raise RuntimeError("HTTPConnectionPool: Connection refused")

        article.save = _boom

        with self.assertRaises(CatalogUnavailable):
            persist_enrichment(article, {"study_type": "RCT"})
        # sync must still be restored on the failure path.
        self.assertTrue(article.sync)


class TestEnrichmentJobService(unittest.TestCase):
    def test_enqueue_creates_queued_job(self):
        service, _ = _make_service()

        job = service.enqueue("urn:article:1")

        self.assertEqual(job["status"], "queued")
        self.assertEqual(service.pending_jobs(), 1)
        self.assertEqual(service.get_status("urn:article:1")["status"], "queued")

    def test_enqueue_is_idempotent_while_in_flight(self):
        service, _ = _make_service()

        first = service.enqueue("urn:article:1")
        second = service.enqueue("urn:article:1")

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(service.pending_jobs(), 1)

    def test_force_enqueue_clears_sweeper_bookkeeping(self):
        service, _ = _make_service()
        service.mark_processed("urn:article:1")

        service.enqueue("urn:article:1", force=True)

        self.assertFalse(service.is_processed("urn:article:1"))

    def test_enqueue_many_collapses_duplicates(self):
        service, _ = _make_service()

        jobs = service.enqueue_many(["urn:article:1", "urn:article:1", "urn:article:2"])

        self.assertEqual(len(jobs), 2)
        self.assertEqual(service.pending_jobs(), 2)

    def test_blank_urn_is_rejected(self):
        from services.enrichment_jobs import ArticleNotFound

        service, _ = _make_service()

        with self.assertRaises(ArticleNotFound):
            service.enqueue("   ")

    def test_status_reports_swept_articles_as_succeeded(self):
        service, _ = _make_service()
        service.mark_processed("urn:article:1")

        status = service.get_status("urn:article:1")

        # No on-demand job exists, but the sweeper already handled it.
        self.assertEqual(status["status"], "succeeded")
        self.assertTrue(status["processed"])

    def test_status_is_not_found_for_untouched_articles(self):
        service, _ = _make_service()

        self.assertEqual(service.get_status("urn:article:9")["status"], "not_found")

    def test_reset_clears_processed_and_failed(self):
        service, _ = _make_service()
        service.mark_processed("urn:article:1")
        service.redis.client.sadd(service.FAILED_SET, "urn:article:1")

        result = service.reset_article("urn:article:1")

        self.assertTrue(result["cleared_processed"])
        self.assertTrue(result["cleared_failed"])
        self.assertEqual(service.get_status("urn:article:1")["status"], "not_found")

    def test_sweeper_pause_switch_round_trips(self):
        service, _ = _make_service()

        self.assertFalse(service.is_sweeper_paused())
        service.set_sweeper_paused(True)
        self.assertTrue(service.is_sweeper_paused())
        service.set_sweeper_paused(False)
        self.assertFalse(service.is_sweeper_paused())

    def test_run_enrichment_persists_and_marks_processed(self):
        article = _FakeArticle("urn:article:1")
        agent = _FakeEnrichmentAgent()
        service, pool = _make_service({"urn:article:1": article}, agent=agent)

        summary = service.run_enrichment("urn:article:1")

        self.assertEqual(agent.calls, ["urn:article:1"])
        self.assertEqual(article.study_type, "RCT")
        self.assertTrue(service.is_processed("urn:article:1"))
        self.assertEqual(summary["ai_key_takeaways"], ["Eat more fish"])
        # The pooled client must go back even on the happy path.
        self.assertEqual(pool.returned, 1)

    def test_run_enrichment_returns_client_on_failure(self):
        from services.enrichment_jobs import ArticleNotFound

        service, pool = _make_service({}, missing=["urn:article:missing"])

        with self.assertRaises(ArticleNotFound):
            service.run_enrichment("urn:article:missing")

        self.assertEqual(pool.returned, 1)


class TestEnrichmentJobWorker(unittest.TestCase):
    def _worker(self, service):
        from workers.enrichment_job_worker import EnrichmentJobWorker

        return EnrichmentJobWorker(poll_interval=0, job_service=service)

    def test_processing_a_job_marks_it_succeeded(self):
        article = _FakeArticle("urn:article:1")
        service, _ = _make_service({"urn:article:1": article})
        worker = self._worker(service)

        service.enqueue("urn:article:1")
        worker._process_job(service.pop_next_job(timeout=0))

        status = service.get_status("urn:article:1")
        self.assertEqual(status["status"], "succeeded")
        self.assertIsNotNone(status["completed_at"])
        self.assertEqual(worker.stats["processed"], 1)

    def test_agent_failure_marks_job_failed_with_reason(self):
        article = _FakeArticle("urn:article:1")
        service, _ = _make_service(
            {"urn:article:1": article}, agent=_FakeEnrichmentAgent(fails=True)
        )
        worker = self._worker(service)

        service.enqueue("urn:article:1")
        worker._process_job(service.pop_next_job(timeout=0))

        status = service.get_status("urn:article:1")
        self.assertEqual(status["status"], "failed")
        self.assertIn("agent exploded", status["error"])
        self.assertEqual(worker.stats["failed"], 1)
        # The lock must be released so a retry can claim the article.
        self.assertTrue(service.try_claim_job("urn:article:1"))

    def test_stale_job_is_skipped(self):
        article = _FakeArticle("urn:article:1")
        service, _ = _make_service({"urn:article:1": article})
        worker = self._worker(service)

        service.enqueue("urn:article:1")
        stale = service.pop_next_job(timeout=0)
        # A newer request supersedes the popped one.
        service.mark_succeeded("urn:article:1", {})
        service.enqueue("urn:article:1", force=True)

        worker._process_job(stale)

        self.assertEqual(worker.stats["skipped"], 1)
        self.assertEqual(worker.stats["processed"], 0)

    def test_locked_article_is_skipped(self):
        article = _FakeArticle("urn:article:1")
        service, _ = _make_service({"urn:article:1": article})
        worker = self._worker(service)

        service.enqueue("urn:article:1")
        job = service.pop_next_job(timeout=0)
        # Simulate another replica already holding the lock.
        service.try_claim_job("urn:article:1")

        worker._process_job(job)

        self.assertEqual(worker.stats["skipped"], 1)


class _FakeElasticForBatches:
    """Records queries; serves canned article hits and aggregations."""

    def __init__(self, urns=None, aggregations=None):
        self.bodies = []
        self.urns = urns or []
        self.aggregations = aggregations or {}

    @property
    def client(self):
        return self

    def search(self, index, body, **kwargs):
        self.bodies.append(body)
        if body.get("size") == 0:  # overview aggregation
            return {
                "hits": {"total": {"value": 100}},
                "aggregations": self.aggregations,
            }
        return {
            "hits": {
                "hits": [{"_id": urn, "_source": {"urn": urn}} for urn in self.urns]
            }
        }


class TestEnrichmentBatches(unittest.TestCase):
    def _with_fake_es(self, fake):
        from unittest.mock import patch
        import backend.elastic as elastic_module

        return patch.object(elastic_module, "ELASTIC_CLIENT", fake)

    def test_criteria_batch_selects_missing_only_and_tracks_progress(self):
        service, _pool = _make_service(articles={"urn:article:a1": _FakeArticle("urn:article:a1")})
        fake_es = _FakeElasticForBatches(urns=["urn:article:a1", "urn:article:a2"])

        with self._with_fake_es(fake_es):
            summary = service.enqueue_batch(
                venue="Nutrients", limit=50, requested_by="admin"
            )

        # Selection is idempotent by construction: enriched articles are
        # excluded and the venue filter is applied.
        body = fake_es.bodies[0]
        self.assertIn({"term": {"venue": "Nutrients"}}, body["query"]["bool"]["filter"])
        self.assertIn(
            {"exists": {"field": "ai_category"}}, body["query"]["bool"]["must_not"]
        )
        self.assertEqual(summary["selected"], 2)
        self.assertEqual(summary["queued"], 2)
        self.assertEqual(summary["criteria"]["venue"], "Nutrients")
        self.assertNotIn("urns", summary)

        # Progress reflects live job states for the batch's articles.
        batch = service.get_batch(summary["batch_id"])
        self.assertEqual(batch["progress"]["total"], 2)
        self.assertEqual(batch["progress"]["queued"], 2)
        self.assertEqual(batch["progress"]["done"], 0)

        service.mark_running("urn:article:a1")
        service.mark_succeeded("urn:article:a1", {"ok": True})
        service.mark_running("urn:article:a2")
        service.mark_failed("urn:article:a2", "boom")

        batch = service.get_batch(summary["batch_id"])
        self.assertEqual(batch["progress"]["succeeded"], 1)
        self.assertEqual(batch["progress"]["failed"], 1)
        self.assertEqual(batch["progress"]["percent"], 100.0)
        self.assertEqual(batch["failures"][0]["urn"], "urn:article:a2")

    def test_force_selects_already_enriched_articles_too(self):
        service, _pool = _make_service()
        fake_es = _FakeElasticForBatches(urns=["urn:article:a1"])
        with self._with_fake_es(fake_es):
            service.enqueue_batch(force=True, limit=10)
        body = fake_es.bodies[0]
        self.assertNotIn(
            {"exists": {"field": "ai_category"}}, body["query"]["bool"]["must_not"]
        )

    def test_batches_list_newest_first_without_urns(self):
        service, _pool = _make_service()
        fake_es = _FakeElasticForBatches(urns=["urn:article:a1"])
        with self._with_fake_es(fake_es):
            first = service.enqueue_batch(limit=10)
            second = service.enqueue_batch(limit=10)
        batches = service.list_batches()
        self.assertEqual(
            [b["batch_id"] for b in batches[:2]],
            [second["batch_id"], first["batch_id"]],
        )
        self.assertTrue(all("urns" not in b for b in batches))

    def test_unknown_batch_is_none(self):
        service, _pool = _make_service()
        self.assertIsNone(service.get_batch("nope"))

    def test_overview_reports_totals_and_venues(self):
        service, _pool = _make_service()
        fake_es = _FakeElasticForBatches(
            aggregations={
                "enriched": {"doc_count": 40},
                "venues": {
                    "buckets": [
                        {"key": "Nutrients", "doc_count": 30,
                         "enriched": {"doc_count": 10}},
                    ]
                },
            }
        )
        with self._with_fake_es(fake_es):
            overview = service.overview()
        self.assertEqual(overview["total"], 100)
        self.assertEqual(overview["enriched"], 40)
        self.assertEqual(overview["pending"], 60)
        self.assertEqual(
            overview["venues"],
            [{"venue": "Nutrients", "total": 30, "enriched": 10, "pending": 20}],
        )


if __name__ == "__main__":
    unittest.main()
