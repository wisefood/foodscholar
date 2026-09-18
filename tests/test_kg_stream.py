"""Tests for the SSE graph stream.

One invariant matters more than the rest: an edge must never reach a client
before both of its endpoints. A client that receives a dangling edge has to
buffer it or drop it, and both produce a drawing that is quietly wrong. The
tests use a deliberately small batch size so page and level boundaries fall in
awkward places.

The browse index is replaced with an in-memory stand-in built by the real
projector, so these exercise the walk and the ordering rather than
Elasticsearch.
"""
import asyncio
import unittest

from foodscholar.io.graph import Card, Shelf, Theme

from services import kg_browse_index as browse
from services import kg_projector, kg_stream
from services.kg_events import GraphEvent, sse_format


def _graph():
    shelves = [Shelf(shelf_id="s0", label="root", facet="foods", depth=0,
                     chunk_count=100)]
    for depth in range(1, 5):
        for n in range(3):
            shelves.append(Shelf(
                shelf_id=f"s{depth}_{n}", label=f"shelf {depth}.{n}",
                facet="foods", depth=depth,
                parent_shelf_id=("s0" if depth == 1 else f"s{depth - 1}_{n}"),
                chunk_count=100 - depth * 10 - n,
            ))
    # Each theme spans a shallow and a deep shelf, so its second attachment
    # cannot be sent until the walk reaches depth 4.
    themes = [
        Theme(theme_id=f"t{n}", label=f"theme {n}",
              shelf_ids=[f"s2_{n}", f"s4_{n}"], chunk_count=30 - n,
              discovered_by="leiden", discovery_version="v1", facet="foods",
              discovery_pass="merged")
        for n in range(3)
    ]
    cards = [
        Card(card_id="cA", target_id="s3_0", target_type="shelf", title="A",
             summary="a", evidence_quality="high", cited_chunk_ids=[],
             llm_model="m", prompt_version="v1"),
        Card(card_id="cB", target_id="t1", target_type="theme", title="B",
             summary="b", evidence_quality="low", cited_chunk_ids=[],
             llm_model="m", prompt_version="v1"),
    ]
    return list(kg_projector._documents(shelves, themes, cards, graph_version="v"))


def _matches(doc, f):
    if f.depth is not None and doc.get("depth") != f.depth:
        return False
    if f.depth_max is not None and (doc.get("depth") or 0) > f.depth_max:
        return False
    if f.kind and doc["kind"] not in f.kind:
        return False
    if f.facet and doc.get("facet") not in f.facet:
        return False
    return True


class _StreamCase(unittest.TestCase):
    """Swaps the browse index for an in-memory one for the duration of a test."""

    def setUp(self):
        self.docs = _graph()
        self._saved = (
            browse.level_page, browse.count, browse.max_depth,
            browse.indexed_graph_version,
        )
        docs = self.docs

        def level_page(filters, *, size, cursor=None):
            hits = sorted(
                (d for d in docs if _matches(d, filters)),
                key=lambda d: (-(d.get("chunk_count") or 0), d["node_id"]),
            )
            start = cursor[0] if cursor else 0
            window = hits[start : start + size]
            more = len(window) == size and start + size < len(hits)
            return window, ([start + size] if more else None)

        browse.level_page = level_page
        browse.count = lambda f: sum(1 for d in docs if _matches(d, f))
        browse.max_depth = lambda f: max(
            (d.get("depth") or 0) for d in docs if _matches(d, f)
        )
        browse.indexed_graph_version = lambda: "v"

    def tearDown(self):
        (browse.level_page, browse.count, browse.max_depth,
         browse.indexed_graph_version) = self._saved

    def stream(self, filters=None, **kw):
        filters = filters or browse.BrowseFilters(depth_max=10)

        async def collect():
            return [e async for e in kg_stream.stream_levels(filters, **kw)]

        return asyncio.run(collect())

    @staticmethod
    def replay(events):
        """Replay a stream the way a client would: what could it actually draw?"""
        seen, drawn, dangling = set(), [], []
        for event in events:
            if event.name == "nodes":
                seen.update(n["id"] for n in event.data["nodes"])
            elif event.name == "edges":
                for edge in event.data["edges"]:
                    if edge["source"] not in seen or edge["target"] not in seen:
                        dangling.append(edge)
                    drawn.append(edge)
        return seen, drawn, dangling


class TestOrdering(_StreamCase):
    def test_no_edge_arrives_before_its_endpoints(self):
        _, _, dangling = self.replay(self.stream(batch_size=2))
        self.assertEqual(dangling, [])

    def test_the_stream_opens_with_meta_and_closes_with_done(self):
        events = self.stream(batch_size=2)
        self.assertEqual(events[0].name, "meta")
        self.assertEqual(events[-1].name, "done")

    def test_seq_is_monotonic_so_a_client_can_spot_a_gap(self):
        events = self.stream(batch_size=2)
        self.assertEqual(
            [e.data["seq"] for e in events], list(range(1, len(events) + 1))
        )

    def test_nodes_arrive_shallowest_first(self):
        events = self.stream(batch_size=2)
        depths = [
            n["attrs"]["depth"]
            for e in events if e.name == "nodes" for n in e.data["nodes"]
        ]
        self.assertEqual(depths, sorted(depths))

    def test_every_matching_node_is_sent_exactly_once(self):
        events = self.stream(batch_size=2)
        ids = [n["id"] for e in events if e.name == "nodes" for n in e.data["nodes"]]
        self.assertEqual(len(ids), len(self.docs))
        self.assertEqual(len(set(ids)), len(self.docs))

    def test_frames_respect_the_batch_size(self):
        events = self.stream(batch_size=2)
        for event in events:
            if event.name in ("nodes", "edges"):
                self.assertLessEqual(len(event.data[event.name]), 2)

    def test_a_theme_spanning_two_depths_gets_both_attachments(self):
        _, drawn, _ = self.replay(self.stream(batch_size=2))
        attachments = [
            e for e in drawn if e["kind"] == "has_theme" and e["target"] == "t0"
        ]
        self.assertEqual(len(attachments), 2)

    def test_nodes_carry_their_precomputed_position(self):
        events = self.stream(batch_size=50)
        for event in events:
            if event.name == "nodes":
                for node in event.data["nodes"]:
                    self.assertIsNotNone(node["attrs"]["x"])
                    self.assertIsNotNone(node["attrs"]["y"])


class TestCeiling(_StreamCase):
    """One unfiltered request must not be able to hold a worker indefinitely."""

    def test_the_walk_stops_at_the_ceiling_and_says_so(self):
        done = self.stream(batch_size=2, max_nodes=5)[-1].data
        self.assertLessEqual(done["nodes"], 5)
        self.assertTrue(done["truncated"])

    def test_a_shallow_cut_strands_nothing(self):
        """Edges at depth 1 all point up at a root that has already been sent."""
        done = self.stream(batch_size=2, max_nodes=5)[-1].data
        self.assertEqual(done["dropped_edges"], 0)

    def test_a_deeper_cut_strands_edges_and_reports_them(self):
        """Themes are sent, the depth-4 shelves they span are not."""
        events = self.stream(batch_size=2, max_nodes=13)
        self.assertGreater(events[-1].data["dropped_edges"], 0)

    def test_the_dropped_count_is_truthful(self):
        """A sparse view and a broken one must be distinguishable."""
        events = self.stream(batch_size=2, max_nodes=13)
        seen, drawn, dangling = self.replay(events)
        expected = [
            edge for doc in self.docs if doc["node_id"] in seen
            for edge in kg_stream._candidate_edges(doc)
        ]
        unsendable = [
            e for e in expected
            if e["source"] not in seen or e["target"] not in seen
        ]
        self.assertEqual(dangling, [])
        self.assertEqual(events[-1].data["dropped_edges"], len(unsendable))
        self.assertEqual(len(drawn), len(expected) - len(unsendable))

    def test_the_invariant_survives_truncation(self):
        for ceiling in (1, 5, 13, 17):
            with self.subTest(max_nodes=ceiling):
                _, _, dangling = self.replay(
                    self.stream(batch_size=2, max_nodes=ceiling)
                )
                self.assertEqual(dangling, [])


class TestFilters(_StreamCase):
    def test_depth_max_bounds_the_walk(self):
        events = self.stream(browse.BrowseFilters(depth_max=1), batch_size=50)
        depths = {
            n["attrs"]["depth"]
            for e in events if e.name == "nodes" for n in e.data["nodes"]
        }
        self.assertTrue(depths <= {0, 1})

    def test_kind_filters_the_stream_not_just_the_view(self):
        """Filtering out a kind should stop it being sent, not hide it client-side."""
        events = self.stream(
            browse.BrowseFilters(kind=("shelf",), depth_max=10), batch_size=50
        )
        kinds = {
            n["kind"] for e in events if e.name == "nodes" for n in e.data["nodes"]
        }
        self.assertEqual(kinds, {"shelf"})

    def test_an_unknown_facet_is_refused_rather_than_silently_empty(self):
        from exceptions import APIException

        with self.assertRaises(APIException):
            self.stream(browse.BrowseFilters(facet=("not_a_facet",)))


class TestDisconnect(_StreamCase):
    """A client that leaves should not keep a generator paging the index."""

    def test_the_walk_stops_when_the_client_goes_away(self):
        calls = {"n": 0}

        async def gone():
            calls["n"] += 1
            return calls["n"] > 1

        events = self.stream(batch_size=2, is_disconnected=gone)
        self.assertFalse(any(e.name == "done" for e in events))
        self.assertLess(len(events), 10)


class TestWireFormat(unittest.TestCase):
    def test_a_frame_is_an_event_line_a_data_line_and_a_blank_line(self):
        frame = sse_format(GraphEvent(name="nodes", data={"seq": 1, "nodes": []}))
        self.assertTrue(frame.startswith("event: nodes\ndata: {"))
        self.assertTrue(frame.endswith("\n\n"))

    def test_only_done_and_error_end_a_stream(self):
        self.assertTrue(GraphEvent(name="done").is_terminal)
        self.assertTrue(GraphEvent(name="error").is_terminal)
        self.assertFalse(GraphEvent(name="nodes").is_terminal)
        self.assertFalse(GraphEvent(name="progress").is_terminal)


if __name__ == "__main__":
    unittest.main()
