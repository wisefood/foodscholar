"""Tests for the QA source-coverage assessment (services.qa_coverage).

The gate checks run on production policy functions; the probe and the
concentration layer run the real legs, fusion, ranking and selection against
a fake Elasticsearch that understands just enough of the query bodies those
legs send: the multi_match text, the status / visibility / tier filters, kNN
and point-in-time paging.
"""
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from config import config
from services import qa_coverage as cov
from services.qa_pipeline import retrieval
from services.qa_pipeline.state import EvidenceItem
from models.qa import RetrievedSource

_WORD = re.compile(r"[a-z0-9]+")


def _words(text):
    return set(_WORD.findall(str(text or "").lower()))


def _find_multi_match(node):
    if isinstance(node, dict):
        if "multi_match" in node and isinstance(node["multi_match"], dict):
            return node["multi_match"].get("query")
        for value in node.values():
            found = _find_multi_match(value)
            if found:
                return found
    if isinstance(node, list):
        for value in node:
            found = _find_multi_match(value)
            if found:
                return found
    return None


class FakeES:
    """Two indexes held in memory; answers the bodies the legs build."""

    def __init__(self, articles=(), guidelines=(), knn_page_size=10):
        self.docs = {
            "articles": {doc_id: dict(src) for doc_id, src in articles},
            "guidelines": {doc_id: dict(src) for doc_id, src in guidelines},
        }
        self.knn_page_size = knn_page_size
        self.searches = 0
        self.pits = {}

    # -- point in time paging (census) --
    def open_point_in_time(self, index, keep_alive):
        pit = f"pit-{index}-{len(self.pits)}"
        self.pits[pit] = index
        return {"id": pit}

    def close_point_in_time(self, id=None, body=None):
        self.pits.pop(id, None)

    def search(self, index=None, body=None, knn=None, source=None, **_):
        self.searches += 1
        body = body or {}
        if "pit" in body:
            return self._page(self.pits[body["pit"]["id"]], body)
        if knn is not None:
            return self._knn(index, knn)
        return self._lexical(index, body)

    def _page(self, index, body):
        docs = sorted(self.docs[index].items())
        query = body.get("query") or {}
        if "bool" in query and "must_not" in query["bool"]:  # missing embedding
            docs = [(i, s) for i, s in docs if "embedding" not in s]
        start = (body.get("search_after") or [-1])[0] + 1
        page = docs[start : start + body["size"]]
        hits = []
        for offset, (doc_id, src) in enumerate(page):
            hit = {"_id": doc_id, "sort": [start + offset]}
            if body.get("_source") is not False:
                hit["_source"] = {k: v for k, v in src.items() if k != "embedding"}
            hits.append(hit)
        return {"hits": {"hits": hits}, "pit_id": body["pit"]["id"]}

    def _visible(self, index, src, body):
        if index == "guidelines":
            return src.get("status") == "active"
        if src.get("status") == "deleted" or src.get("indexing_tier") == "do_not_index":
            return False
        hidden = {"hidden"}
        if "expert_only" in json.dumps(body):
            hidden.add("expert_only")
        return src.get("reader_visibility") not in hidden

    def _lexical(self, index, body):
        query = _words(_find_multi_match(body.get("query")))
        scored = []
        for doc_id, src in sorted(self.docs[index].items()):
            if not self._visible(index, src, body):
                continue
            text = " ".join(str(src.get(k, "")) for k in ("title", "abstract", "rule_text", "keywords", "topic"))
            overlap = len(query & _words(text))
            if overlap:
                scored.append((overlap, doc_id, src))
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits = [
            {"_id": doc_id, "_score": float(score), "_source": {k: v for k, v in src.items() if k != "embedding"}}
            for score, doc_id, src in scored[: body.get("size", 10)]
        ]
        return {"hits": {"hits": hits}}

    def _knn(self, index, knn):
        flt = knn.get("filter")
        target = knn["query_vector"][0]
        pool = []
        for doc_id, src in sorted(self.docs[index].items()):
            if "embedding" not in src:
                continue
            if not self._visible(index, src, {"filter": flt}):
                continue
            pool.append((abs(src["embedding"][0] - target), doc_id, src))
        pool.sort(key=lambda t: (t[0], t[1]))
        # Like Elasticsearch without `size`: k neighbours, but one page of hits.
        limit = min(knn["k"], self.knn_page_size)
        hits = [
            {"_id": doc_id, "_score": 1.0 / (1.0 + dist), "_source": {k: v for k, v in src.items() if k != "embedding"}}
            for dist, doc_id, src in pool[:limit]
        ]
        return {"hits": {"hits": hits}}


def _article(doc_id, title, abstract="", **extra):
    src = {"urn": doc_id, "title": title, "abstract": abstract, "venue": extra.pop("venue", "Journal A")}
    src.update(extra)
    return doc_id, src


def _rule(doc_id, text, guide="urn:guide:g1", status="active", **extra):
    src = {"id": doc_id, "rule_text": text, "guide_urn": guide, "status": status, "title": "Guide title"}
    src.update(extra)
    return doc_id, src


def _record(doc_id, kind="article", status="eligible", stratum="s"):
    return cov.DocRecord(
        doc_id=doc_id, key=f"{kind}:{doc_id}", source_type=kind, stratum=stratum,
        year_bucket="2010s", title=doc_id, status=status, queries=[("title", doc_id)],
    )


def _item(key_id, score, *, guide=None, kind="article"):
    payload = {"urn": key_id, "source_type": kind}
    if guide:
        payload["guide_urn"] = guide
    item = EvidenceItem(payload=payload, source=RetrievedSource(source_type=kind, urn=key_id, title=key_id, similarity_score=score))
    item.adjusted_score = score
    return item


class GateTests(unittest.TestCase):
    def _classify(self, hit_src, has_embedding=True):
        return cov.article_record({"_id": "a1", "_source": hit_src}, has_embedding=has_embedding)

    def test_eligible_article(self):
        record = self._classify({"urn": "a1", "title": "Fibre and heart health", "abstract": "Text."})
        self.assertEqual(record.status, "eligible")
        self.assertEqual(record.key, "article:a1")
        self.assertEqual([k for k, _ in record.queries], ["title", "text"])

    def test_deleted_hidden_and_curated_do_not_index_are_unreachable(self):
        for src, gate in (
            ({"title": "x", "status": "deleted"}, "article.deleted"),
            ({"title": "x", "reader_visibility": "hidden"}, "article.hidden"),
            ({"title": "x", "indexing_tier": "do_not_index"}, "article.do_not_index"),
        ):
            with self.subTest(gate=gate):
                record = self._classify(src)
                self.assertIn(gate, record.gates)
                self.assertEqual(record.status, "unreachable")

    def test_agent_do_not_index_passes_es_but_scores_zero(self):
        record = self._classify({"title": "x", "ai_indexing_tier": "Do not index"})
        self.assertEqual(record.gates, ["article.do_not_index.scored_zero"])
        self.assertEqual(record.status, "unreachable")

    def test_editor_tier_overrides_agent_proposal(self):
        record = self._classify({"title": "x", "indexing_tier": "core", "ai_indexing_tier": "do_not_index"})
        self.assertEqual(record.status, "eligible")

    def test_expert_only_is_audience_limited(self):
        record = self._classify({"title": "x", "reader_visibility": "expert_only"})
        self.assertEqual(record.status, "audience_limited")

    def test_missing_vector_and_missing_text(self):
        self.assertEqual(self._classify({"title": "x"}, has_embedding=False).status, "lexical_only")
        self.assertEqual(self._classify({}, has_embedding=True).status, "vector_only")
        bare = self._classify({}, has_embedding=False)
        self.assertIn("article.no_text_no_vector", bare.gates)
        self.assertEqual(bare.status, "unreachable")

    def test_unknown_embedding_state_is_not_a_gate(self):
        self.assertEqual(self._classify({"title": "x"}, has_embedding=None).status, "eligible")

    def test_guideline_gates(self):
        doc_id, src = _rule("r1", "Eat five portions of fruit and vegetables a day.")
        ok = cov.guideline_record({"_id": doc_id, "_source": src}, has_embedding=True)
        self.assertEqual((ok.status, ok.key, ok.stratum), ("eligible", "guideline:r1", "urn:guide:g1"))
        draft = cov.guideline_record({"_id": "r2", "_source": {**src, "id": "r2", "status": "draft"}}, has_embedding=True)
        self.assertEqual(draft.status, "unreachable")
        self.assertEqual(draft.detail["rule_status"], "draft")
        short = cov.guideline_record({"_id": "r3", "_source": {**src, "id": "r3", "rule_text": "Eat fish"}}, has_embedding=True)
        self.assertIn("guideline.short_rule", short.gates)
        no_vec = cov.guideline_record({"_id": doc_id, "_source": src}, has_embedding=False)
        self.assertEqual(no_vec.status, "lexical_only")

    def test_every_gate_names_where_it_lives(self):
        for gate, meta in cov.GATES.items():
            with self.subTest(gate=gate):
                self.assertRegex(meta["code"], r"src/[\w/]+\.py:\d+")
                self.assertIn(meta["kind"], {"hard", "soft", "audience"})


class CensusTests(unittest.TestCase):
    def test_scan_pages_through_everything(self):
        es = FakeES(articles=[_article(f"a{i}", f"Title {i}") for i in range(7)])
        ids = [hit["_id"] for hit in cov.scan_index(es, "articles", page_size=3)]
        self.assertEqual(ids, [f"a{i}" for i in range(7)])
        self.assertEqual(es.pits, {})  # the point in time is closed

    def test_census_counts_and_first_gate(self):
        es = FakeES(
            articles=[
                _article("a1", "Fibre", embedding=[1.0]),
                _article("a2", "Gone", status="deleted", reader_visibility="hidden", embedding=[2.0]),
                _article("a3", "No vector"),
            ],
            guidelines=[
                _rule("r1", "Eat five portions of fruit a day.", embedding=[1.0]),
                _rule("r2", "Drafted rule about salt intake.", guide="urn:guide:g2", status="draft"),
            ],
        )
        records = cov.run_census(es)
        summary = cov.summarize_census(records)
        a, g = summary["article"], summary["guideline"]
        self.assertEqual((a["total"], a["reachable"]), (3, 2))
        self.assertEqual(a["unreachable_by_first_gate"], {"article.deleted": 1})
        self.assertEqual(a["gates"]["article.hidden"]["examples"], ["a2"])
        self.assertEqual(a["status"]["lexical_only"], 1)
        self.assertEqual((g["total"], g["reachable"]), (2, 1))
        self.assertEqual(g["guides_with_no_reachable_rule"], ["urn:guide:g2"])
        self.assertEqual(g["rule_status"], {"active": 1, "draft": 1})


class SamplingTests(unittest.TestCase):
    def test_all_when_unset(self):
        records = [_record(f"a{i}") for i in range(5)]
        self.assertEqual(len(cov.stratified_sample(records, None)), 5)

    def test_every_stratum_is_covered(self):
        records = [_record(f"big{i}", stratum="big") for i in range(50)]
        records += [_record("tiny", stratum="tiny"), _record("r1", kind="guideline", stratum="g")]
        chosen = cov.stratified_sample(records, 5, seed=3)
        self.assertEqual(len(chosen), 5)
        self.assertEqual({r.stratum for r in chosen}, {"big", "tiny", "g"})

    def test_guides_come_first_when_n_is_small(self):
        records = [_record(f"j{i}", stratum=f"journal{i}") for i in range(10)]
        records += [_record("r1", kind="guideline", stratum="g1"), _record("r2", kind="guideline", stratum="g2")]
        chosen = cov.stratified_sample(records, 2, seed=1)
        self.assertEqual({r.source_type for r in chosen}, {"guideline"})

    def test_deterministic(self):
        records = [_record(f"a{i}", stratum=f"s{i % 4}") for i in range(40)]
        first = [r.doc_id for r in cov.stratified_sample(records, 9, seed=7)]
        self.assertEqual(first, [r.doc_id for r in cov.stratified_sample(records, 9, seed=7)])


class DropReasonTests(unittest.TestCase):
    def test_selected_is_none(self):
        a = _item("a", 0.9)
        self.assertIsNone(cov.classify_drop("article:a", [a], [a], min_score=0.05, per_doc_cap=2))

    def test_below_threshold(self):
        a = _item("a", 0.01)
        self.assertEqual(cov.classify_drop("article:a", [a], [], min_score=0.05, per_doc_cap=2), "below_threshold")

    def test_over_doc_cap(self):
        items = [_item(f"r{i}", 0.9 - i / 10, guide="g", kind="guideline") for i in range(3)]
        self.assertEqual(
            cov.classify_drop("guideline:r2", items, items[:2], min_score=0.05, per_doc_cap=2), "over_doc_cap"
        )

    def test_over_budget_and_not_retrieved(self):
        items = [_item(f"a{i}", 0.9 - i / 100) for i in range(6)]
        self.assertEqual(cov.classify_drop("article:a5", items, items[:5], min_score=0.05, per_doc_cap=2), "over_budget")
        self.assertEqual(cov.classify_drop("article:zz", items, items[:5], min_score=0.05, per_doc_cap=2), "not_retrieved")

    def test_unreachable_reason_priority(self):
        q = {"fused_rank": None, "lexical_hits": 3, "knn_hits": 10, "leg_errors": {}, "fused_size": 12,
             "same_guide_share": 0.0, "drop_reason": "not_retrieved"}
        self.assertEqual(cov.unreachable_reason([q], candidates=30), "outranked")
        self.assertEqual(cov.unreachable_reason([q], candidates=30, vector_visible=False), "no_match")
        crowded = dict(q, same_guide_share=0.6)
        self.assertEqual(cov.unreachable_reason([crowded], candidates=30), "outranked_by_same_guide")
        capped = dict(q, fused_rank=3, drop_reason="over_doc_cap")
        self.assertEqual(cov.unreachable_reason([q, capped], candidates=30), "over_doc_cap")
        self.assertEqual(cov.unreachable_reason([], candidates=30), "no_query")


class MathAndInputTests(unittest.TestCase):
    def test_gini(self):
        self.assertEqual(cov.gini([3, 3, 3, 3]), 0.0)
        self.assertGreater(cov.gini([0, 0, 0, 0, 0, 0, 0, 0, 0, 100]), 0.85)
        self.assertEqual(cov.gini([]), 0.0)

    def test_load_questions_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = os.path.join(tmp, "q.txt")
            with open(lines, "w") as fh:
                fh.write("# comment\nIs fruit juice healthy?\n\n{\"question\": \"Are eggs ok?\"}\n")
            self.assertEqual(cov.load_questions(lines), ["Is fruit juice healthy?", "Are eggs ok?"])
            listed = os.path.join(tmp, "q.json")
            with open(listed, "w") as fh:
                json.dump([{"text": "Coffee?"}, "Salt?"], fh)
            self.assertEqual(cov.load_questions(listed), ["Coffee?", "Salt?"])


def _fake_embed(texts):
    return [[float(len(t) % 7)] for t in texts]


class EndToEndTests(unittest.TestCase):
    """Census, probe and concentration over the real legs and the fake cluster."""

    def setUp(self):
        self.es = FakeES(
            articles=[
                _article("a1", "Whole grains and heart disease", "Whole grains lower risk.", embedding=[1.0]),
                _article("a2", "Sugar drinks in teenagers", "Sugary drinks raise weight.", venue="Journal B", embedding=[2.0]),
                _article("a3", "Expert only vitamin D trial", "Vitamin D dosing.", reader_visibility="expert_only", embedding=[3.0]),
                _article("a4", "Deleted paper", status="deleted", embedding=[4.0]),
            ],
            guidelines=[
                _rule("r1", "Choose whole-grain bread, rice and pasta more often.", embedding=[1.0]),
                _rule("r2", "Limit sugary drinks for children aged 1 to 4.", guide="urn:guide:g2", embedding=[5.0]),
                _rule("r3", "Archived rule on salt intake for adults.", status="archived"),
            ],
        )
        self.patch = patch.object(retrieval.ELASTIC_CLIENT, "_client", self.es)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_probe_finds_eligible_documents_with_real_pipeline(self):
        records = cov.run_census(self.es)
        eligible = [r for r in records if r.status != "unreachable"]
        self.assertEqual(sorted(r.doc_id for r in eligible), ["a1", "a2", "a3", "r1", "r2"])
        ctx = cov.ProbeContext(embed=_fake_embed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.jsonl")
            rows = cov.run_probe(eligible, ctx, results_path=path, workers=2)
            self.assertEqual(len(rows), 5)
            by_id = {r["id"]: r for r in rows}
            self.assertTrue(by_id["a1"]["reachable"])
            self.assertEqual(by_id["a1"]["best_rank"], 1)
            self.assertTrue(by_id["r2"]["reachable"])
            # The expert-only study is probed as an expert reader would ask.
            self.assertTrue(by_id["a3"]["reachable"])

            # Resumable: a rerun searches nothing and returns the same rows.
            before = self.es.searches
            again = cov.run_probe(eligible, ctx, results_path=path, workers=2)
            self.assertEqual(self.es.searches, before)
            self.assertEqual(sorted(r["id"] for r in again), sorted(by_id))

            summary = cov.summarize_probe(rows, candidates=retrieval._candidates())
            self.assertEqual(summary["article"]["reachable"], 1.0)
            self.assertEqual(summary["guideline"]["recall"]["@1"], 1.0)

    def test_knn_page_cap_is_detected(self):
        self.es.docs["articles"].update(
            {f"x{i}": {"urn": f"x{i}", "title": f"Filler {i}", "embedding": [float(i)]} for i in range(20)}
        )
        census = cov.run_census(self.es)
        records = [r for r in census if r.doc_id == "a1"]
        with tempfile.TemporaryDirectory() as tmp:
            rows = cov.run_probe(records, cov.ProbeContext(embed=_fake_embed), results_path=os.path.join(tmp, "p.jsonl"))
        with patch.dict(config.settings, {"QA_RRF_CANDIDATES": 30}):
            summary = cov.summarize_probe(
                rows, candidates=retrieval._candidates(), vectors=cov.vector_population(census)
            )
        knn = summary["knn_response"]["article"]
        self.assertEqual((knn["max_hits_seen"], knn["documents_with_vectors"]), (10, 22))
        self.assertTrue(knn["capped_below_k"])

    def test_small_vector_corpus_is_not_reported_as_capped(self):
        census = cov.run_census(self.es)  # only three studies have vectors
        records = [r for r in census if r.doc_id == "a1"]
        with tempfile.TemporaryDirectory() as tmp:
            rows = cov.run_probe(records, cov.ProbeContext(embed=_fake_embed), results_path=os.path.join(tmp, "p.jsonl"))
        summary = cov.summarize_probe(rows, candidates=30, vectors=cov.vector_population(census))
        self.assertFalse(summary["knn_response"]["article"]["capped_below_k"])

    def test_crowded_rule_is_reported_with_its_reason(self):
        # Three rules of one guide share every word; one answer takes two per guide.
        text = "Offer milk water and fruit to toddlers every day."
        self.es.docs["guidelines"] = {
            f"t{i}": dict(_rule(f"t{i}", text, guide="urn:guide:toddlers")[1]) for i in range(3)
        }
        records = [r for r in cov.run_census(self.es) if r.source_type == "guideline"]
        with tempfile.TemporaryDirectory() as tmp:
            rows = cov.run_probe(records, cov.ProbeContext(embed=None), results_path=os.path.join(tmp, "p.jsonl"))
        reasons = {r["id"]: r["reason"] for r in rows}
        self.assertEqual(sorted(reasons.values(), key=str), [None, None, "over_doc_cap"])

    def test_concentration_and_markdown(self):
        records = cov.run_census(self.es)
        ctx = cov.ProbeContext(embed=_fake_embed)
        conc = cov.run_concentration(
            ["Are whole grains good for the heart?", "Whole grain bread every day?"], records, ctx, workers=1
        )
        self.assertEqual(conc["questions"], 2)
        self.assertGreaterEqual(conc["article"]["distinct_selected"], 1)
        self.assertIn("gini_selected", conc["guideline"])
        report = {
            "config": {"guideline_hybrid": True},
            "census": cov.summarize_census(records),
            "concentration": conc,
        }
        text = cov.render_markdown(report)
        self.assertIn("of 4 studies", text)
        self.assertIn("1 study can never be found", text)
        self.assertIn("can never be found, whatever the question", text)
        self.assertIn("deleted from the catalogue", text)
        self.assertIn("## Gates", text)


class _Indices:
    def __init__(self, es):
        self.es = es

    def exists(self, index):
        return index in self.es.docs


class ScriptSmokeTests(EndToEndTests):
    """The command line, end to end, on the fake cluster (keyword legs only)."""

    def test_script_writes_report_and_resumes(self):
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "assess_source_coverage", os.path.join(here, "..", "scripts", "assess_source_coverage.py")
        )
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        self.es.ping = lambda: True
        self.es.indices = _Indices(self.es)
        with tempfile.TemporaryDirectory() as tmp:
            questions = os.path.join(tmp, "q.txt")
            with open(questions, "w") as fh:
                fh.write("Are whole grains good for the heart?\n")
            args = ["--out-dir", tmp, "--no-embed", "--layers", "census,probe,concentration", "--questions", questions]
            with patch("sys.stdout"):
                self.assertEqual(script.main(args), 0)
            with open(os.path.join(tmp, "report.json")) as fh:
                report = json.load(fh)
            self.assertEqual(report["census"]["article"]["total"], 4)
            self.assertEqual(report["probe"]["article"]["probed"], 3)
            self.assertEqual(report["concentration"]["questions"], 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, "report.md")))
            before = self.es.searches
            with patch("sys.stdout"):
                script.main(args[:3] + ["--layers", "census,probe", "--reuse-census"])
            self.assertEqual(self.es.searches, before)  # nothing re-probed, census reused

    # The inherited end-to-end tests already ran in EndToEndTests.
    test_probe_finds_eligible_documents_with_real_pipeline = None
    test_knn_page_cap_is_detected = None
    test_small_vector_corpus_is_not_reported_as_capped = None
    test_crowded_rule_is_reported_with_its_reason = None
    test_concentration_and_markdown = None


if __name__ == "__main__":
    unittest.main()
