"""Tests for projecting the knowledge graph into the browse index.

The projector derives three things the graph does not store, and a browse UI is
wrong in a different way for each of them: ancestor chains (breadcrumbs and
subtree filters), depths for themes and cards (the order the graph streams in),
and layout coordinates (whether the picture is stable between visits).

The ancestor-chain cases are the ones that caught a real bug: a memoized chain
is already root-first, and appending it to a child-first walk before reversing
produced a scrambled breadcrumb that still passed a set-membership check.
"""
import unittest

from foodscholar.io.graph import Card, Shelf, Theme

from services import kg_projector, kg_stream


def _shelves():
    return [
        Shelf(shelf_id="s-food", label="food product", facet="foods", depth=0,
              chunk_count=500, support_direct=50, support_lifted=450),
        Shelf(shelf_id="s-plant", label="plant food product", facet="foods",
              depth=1, parent_shelf_id="s-food", chunk_count=300),
        Shelf(shelf_id="s-oil", label="vegetable oil", facet="foods", depth=2,
              parent_shelf_id="s-plant", chunk_count=120,
              foodon_id="FOODON:00001234"),
        Shelf(shelf_id="s-olive", label="olive oil", display_label="Olive oil",
              facet="foods", depth=3, parent_shelf_id="s-oil", chunk_count=80),
        Shelf(shelf_id="s-folded", label="oil by source", facet="foods", depth=2,
              parent_shelf_id="s-plant", chunk_count=5, status="folded",
              see_also=["FOODON:00009999"]),
        Shelf(shelf_id="s-cvd", label="cardiovascular disease", facet="health",
              depth=0, chunk_count=200),
    ]


def _themes():
    return [
        Theme(theme_id="t-mufa", label="monounsaturated fat",
              shelf_ids=["s-olive"], chunk_count=40, discovered_by="leiden",
              discovery_version="v1", facet="foods", discovery_pass="merged",
              keyword_terms=["oleic", "mufa"]),
        Theme(theme_id="t-heart", label="heart health",
              shelf_ids=["s-oil", "s-olive"], chunk_count=60,
              discovered_by="leiden", discovery_version="v1", facet="foods",
              discovery_pass="relatedness", keyword_terms=["ldl"]),
    ]


def _cards():
    return [
        Card(card_id="c-olive", target_id="s-olive", target_type="shelf",
             title="Olive oil", summary="Associated with lower LDL.",
             evidence_quality="high", cited_chunk_ids=["ch1", "ch2"],
             llm_model="m", prompt_version="v1"),
        Card(card_id="c-heart", target_id="t-heart", target_type="theme",
             title="Heart health", summary="Mixed evidence.",
             evidence_quality="debated", cited_chunk_ids=["ch3"],
             llm_model="m", prompt_version="v1", safety_flagged=True),
    ]


def _project():
    docs = list(kg_projector._documents(
        _shelves(), _themes(), _cards(), graph_version="testhash"
    ))
    return docs, {d["node_id"]: d for d in docs}


class TestAncestorChains(unittest.TestCase):
    def setUp(self):
        self.docs, self.by_id = _project()

    def test_a_root_has_no_ancestors(self):
        self.assertEqual(self.by_id["s-food"]["ancestor_ids"], [])

    def test_the_chain_reads_root_first(self):
        """Breadcrumbs render in this order, so the order is the contract."""
        self.assertEqual(
            self.by_id["s-olive"]["ancestor_ids"],
            ["s-food", "s-plant", "s-oil"],
        )

    def test_memoized_chains_do_not_scramble_the_order(self):
        """Every chain is a prefix of its children's, whatever order they were built in."""
        for shelf in _shelves():
            chain = self.by_id[shelf.shelf_id]["ancestor_ids"]
            if shelf.parent_shelf_id:
                parent_chain = self.by_id[shelf.parent_shelf_id]["ancestor_ids"]
                self.assertEqual(chain, parent_chain + [shelf.parent_shelf_id])

    def test_a_theme_inherits_its_anchor_shelfs_chain(self):
        """So `under=<some ancestor>` finds the themes hanging beneath it too."""
        self.assertEqual(
            self.by_id["t-mufa"]["ancestor_ids"],
            ["s-food", "s-plant", "s-oil", "s-olive"],
        )

    def test_a_card_includes_whatever_it_describes(self):
        self.assertEqual(self.by_id["c-olive"]["ancestor_ids"][-1], "s-olive")
        self.assertEqual(self.by_id["c-heart"]["ancestor_ids"][-1], "t-heart")

    def test_a_malformed_parent_cycle_terminates(self):
        """A bad parent pointer should not hang the projection."""
        chains = kg_projector._ancestor_chains([
            Shelf(shelf_id="a", label="a", facet="foods", depth=0, parent_shelf_id="b"),
            Shelf(shelf_id="b", label="b", facet="foods", depth=1, parent_shelf_id="a"),
        ])
        self.assertEqual(set(chains), {"a", "b"})


class TestDerivedDepths(unittest.TestCase):
    """Depth drives the stream's breadth-first order across all three kinds."""

    def setUp(self):
        self.docs, self.by_id = _project()

    def test_a_shelf_keeps_its_own_depth(self):
        self.assertEqual(self.by_id["s-olive"]["depth"], 3)

    def test_a_theme_sits_one_level_below_its_shelf(self):
        self.assertEqual(self.by_id["t-mufa"]["depth"], 4)

    def test_a_theme_on_several_shelves_follows_the_shallowest(self):
        """Otherwise it would stream after a shelf it is supposed to introduce."""
        self.assertEqual(self.by_id["t-heart"]["depth"], 3)

    def test_a_card_sits_one_level_below_its_target(self):
        self.assertEqual(self.by_id["c-olive"]["depth"], 4)
        self.assertEqual(self.by_id["c-heart"]["depth"], 4)


class TestDocumentContents(unittest.TestCase):
    def setUp(self):
        self.docs, self.by_id = _project()

    def test_one_document_per_node(self):
        self.assertEqual(len(self.docs), 10)
        self.assertEqual(len({d["node_id"] for d in self.docs}), 10)

    def test_display_label_wins_when_a_shelf_has_one(self):
        """Resolved once here so no client has to know the rule."""
        self.assertEqual(self.by_id["s-olive"]["label"], "Olive oil")
        self.assertEqual(self.by_id["s-oil"]["label"], "vegetable oil")

    def test_counts_are_denormalized_onto_the_parent(self):
        self.assertEqual(self.by_id["s-plant"]["child_count"], 2)
        self.assertEqual(self.by_id["s-olive"]["theme_count"], 2)

    def test_has_card_marks_only_described_nodes(self):
        self.assertTrue(self.by_id["s-olive"]["has_card"])
        self.assertFalse(self.by_id["s-food"]["has_card"])

    def test_folded_shelves_keep_their_status_and_absorbed_ids(self):
        """Folded shelves stay in the data; listings filter them, not the projector."""
        self.assertEqual(self.by_id["s-folded"]["status"], "folded")
        self.assertEqual(self.by_id["s-folded"]["see_also"], ["FOODON:00009999"])

    def test_the_safety_flag_survives(self):
        self.assertTrue(self.by_id["c-heart"]["safety_flagged"])

    def test_no_embedding_reaches_the_index(self):
        """768 floats per document would make every response enormous."""
        for doc in self.docs:
            self.assertNotIn("embedding", doc)

    def test_every_document_is_stamped_with_the_build(self):
        for doc in self.docs:
            self.assertEqual(doc["graph_version"], "testhash")


class TestLayout(unittest.TestCase):
    """Positions are precomputed so the browser never runs a force simulation."""

    def setUp(self):
        self.docs, self.by_id = _project()

    def test_every_node_has_coordinates(self):
        for doc in self.docs:
            self.assertIsInstance(doc["x"], float)
            self.assertIsInstance(doc["y"], float)

    def test_the_same_graph_lays_out_the_same_way_twice(self):
        """A map you learn should still be that map next week."""
        again, _ = _project()
        self.assertEqual([d["x"] for d in again], [d["x"] for d in self.docs])
        self.assertEqual([d["y"] for d in again], [d["y"] for d in self.docs])

    def test_facets_occupy_different_regions(self):
        dx = abs(self.by_id["s-cvd"]["x"] - self.by_id["s-food"]["x"])
        dy = abs(self.by_id["s-cvd"]["y"] - self.by_id["s-food"]["y"])
        self.assertGreater(max(dx, dy), 0.5)

    def test_a_card_hangs_just_below_what_it_describes(self):
        self.assertEqual(self.by_id["c-olive"]["x"], self.by_id["s-olive"]["x"])
        self.assertLess(self.by_id["c-olive"]["y"], self.by_id["s-olive"]["y"])


class TestDerivedEdges(unittest.TestCase):
    """Edges come off the denormalized documents, so the stream never re-reads Neo4j."""

    def setUp(self):
        self.docs, self.by_id = _project()
        self.edges = [e for d in self.docs for e in kg_stream._candidate_edges(d)]

    def test_all_three_relations_are_derived(self):
        self.assertEqual(
            {e["kind"] for e in self.edges},
            {"parent_of", "has_theme", "describes"},
        )

    def test_parent_of_points_from_parent_to_child(self):
        self.assertIn(
            {"source": "s-oil", "target": "s-olive", "kind": "parent_of",
             "weight": 1.0, "attrs": {}},
            self.edges,
        )

    def test_a_theme_on_two_shelves_yields_two_attachments(self):
        attachments = [
            e for e in self.edges
            if e["kind"] == "has_theme" and e["target"] == "t-heart"
        ]
        self.assertEqual(len(attachments), 2)

    def test_no_edge_points_outside_the_projection(self):
        for edge in self.edges:
            self.assertIn(edge["source"], self.by_id)
            self.assertIn(edge["target"], self.by_id)

if __name__ == "__main__":
    unittest.main()
