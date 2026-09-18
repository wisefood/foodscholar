"""Tests for translating browse filters into Elasticsearch queries.

The filter panel is the whole point of "filter things out of the view", so the
translation from a filter value to a query is worth pinning: a clause that is
silently dropped looks like a graph with more in it than the user asked for,
which is indistinguishable from the filter not working.

Pure query construction — no cluster involved.
"""
import unittest

from exceptions import APIException

from services import kg_browse_index as browse

BROWSE = browse.BROWSE_MAPPING


def clauses(**kw):
    return browse._query(browse.BrowseFilters(**kw))


class TestFilterTranslation(unittest.TestCase):
    def test_no_filters_matches_everything(self):
        self.assertEqual(clauses(), {"match_all": {}})

    def test_a_single_facet_becomes_a_terms_clause(self):
        q = clauses(facet=("foods",))
        self.assertIn({"terms": {"facet": ["foods"]}}, q["bool"]["filter"])

    def test_under_matches_the_materialized_ancestor_chain(self):
        """This is what makes a subtree one lookup instead of a traversal."""
        q = clauses(under="s-plant")
        self.assertIn({"term": {"ancestor_ids": "s-plant"}}, q["bool"]["filter"])

    def test_ranges_are_translated_not_dropped(self):
        q = clauses(depth_max=3, min_chunks=50)
        self.assertIn({"range": {"depth": {"lte": 3}}}, q["bool"]["filter"])
        self.assertIn({"range": {"chunk_count": {"gte": 50}}}, q["bool"]["filter"])

    def test_every_filter_field_produces_a_clause(self):
        """A new field with no translation would narrow nothing and look fine."""
        q = clauses(
            kind=("shelf",), facet=("foods",), status=("active",), under="s1",
            depth=2, depth_max=4, min_chunks=10, discovered_by=("leiden",),
            evidence_quality=("high",), has_card=True, parent_id="s0",
            shelf_id="s2", target_id="t1",
        )
        self.assertEqual(len(q["bool"]["filter"]), 13)

    def test_a_query_string_searches_labels_above_descriptions(self):
        q = clauses(q="olive")
        match = q["bool"]["must"][0]["multi_match"]
        self.assertEqual(match["fields"], ["label^3", "description"])
        self.assertEqual(match["query"], "olive")

    def test_a_query_string_keeps_the_filters(self):
        """Searching inside a filtered view must not widen it back out."""
        q = clauses(q="olive", facet=("foods",))
        self.assertIn({"terms": {"facet": ["foods"]}}, q["bool"]["filter"])

    def test_has_card_false_is_a_filter_not_an_absence(self):
        """`has_card=False` means 'show the ones without a card', not 'no filter'."""
        q = clauses(has_card=False)
        self.assertIn({"term": {"has_card": False}}, q["bool"]["filter"])

    def test_min_chunks_zero_is_still_a_filter(self):
        """A falsy-but-present value must not be mistaken for unset."""
        q = clauses(min_chunks=0)
        self.assertIn({"range": {"chunk_count": {"gte": 0}}}, q["bool"]["filter"])

    def test_depth_zero_is_still_a_filter(self):
        q = clauses(depth=0)
        self.assertIn({"term": {"depth": 0}}, q["bool"]["filter"])


class TestRoots(unittest.TestCase):
    """A root is a shelf with no parent, which is a negative clause."""

    def test_roots_only_excludes_anything_with_a_parent(self):
        q = clauses(roots_only=True)
        self.assertEqual(q["bool"]["must_not"], [{"exists": {"field": "parent_id"}}])

    def test_roots_only_combines_with_positive_filters(self):
        q = clauses(roots_only=True, facet=("foods",))
        self.assertIn({"terms": {"facet": ["foods"]}}, q["bool"]["filter"])
        self.assertEqual(q["bool"]["must_not"], [{"exists": {"field": "parent_id"}}])

    def test_roots_only_survives_a_search_query(self):
        q = clauses(roots_only=True, q="oil")
        self.assertEqual(q["bool"]["must_not"], [{"exists": {"field": "parent_id"}}])


class TestValidation(unittest.TestCase):
    """A typo in a filter should be an error, not an empty result set."""

    def test_an_unknown_facet_is_refused(self):
        with self.assertRaises(APIException):
            browse.BrowseFilters(facet=("nutrients", "not_a_facet")).validate()

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(APIException):
            browse.BrowseFilters(kind=("shelf", "sandwich")).validate()

    def test_valid_values_pass(self):
        f = browse.BrowseFilters(facet=("foods", "health"), kind=("shelf", "card"))
        self.assertIs(f.validate(), f)


class TestCursors(unittest.TestCase):
    """Cursors are handed to clients, so they come back malformed eventually."""

    def test_a_cursor_round_trips(self):
        values = [42, "s-olive"]
        self.assertEqual(browse.decode_cursor(browse.encode_cursor(values)), values)

    def test_no_cursor_means_the_first_page(self):
        self.assertIsNone(browse.decode_cursor(None))
        self.assertIsNone(browse.decode_cursor(""))

    def test_a_malformed_cursor_is_a_client_error(self):
        for bad in ("not-base64!!", "eyJhIjogMX0=", "%%%"):
            with self.subTest(cursor=bad):
                with self.assertRaises(APIException):
                    browse.decode_cursor(bad)

    def test_a_cursor_is_url_safe(self):
        """It travels as a query parameter."""
        cursor = browse.encode_cursor([1, "a/b+c?d"])
        self.assertNotIn("/", cursor)
        self.assertNotIn("+", cursor)


class TestMapping(unittest.TestCase):
    def test_the_mapping_is_strict(self):
        """An unexpected field should fail the projection, not be inferred as text."""
        self.assertEqual(BROWSE["mappings"]["dynamic"], "strict")

    def test_every_filterable_field_is_in_the_mapping(self):
        """A filter on an unmapped field returns nothing, quietly."""
        properties = BROWSE["mappings"]["properties"]
        for field in (
            "kind", "facet", "status", "ancestor_ids", "parent_id", "shelf_ids",
            "target_id", "depth", "chunk_count", "discovered_by",
            "evidence_quality", "has_card",
        ):
            self.assertIn(field, properties)

    def test_sort_and_aggregation_fields_are_not_analyzed_text(self):
        """Sorting or aggregating on analyzed text raises at query time."""
        properties = BROWSE["mappings"]["properties"]
        for field in ("facet", "kind", "status", "evidence_quality"):
            self.assertEqual(properties[field]["type"], "keyword")
        self.assertEqual(properties["chunk_count"]["type"], "integer")
        self.assertIn("keyword", properties["label"]["fields"])

    def test_autocomplete_matches_inside_a_label(self):
        """Prefix-only would never find 'olive' in 'extra virgin olive oil'."""
        autocomplete = BROWSE["mappings"]["properties"]["label"]["fields"]["autocomplete"]
        self.assertEqual(autocomplete["type"], "search_as_you_type")

    def test_positions_are_numeric_so_a_viewport_can_range_query_them(self):
        properties = BROWSE["mappings"]["properties"]
        self.assertEqual(properties["x"]["type"], "float")
        self.assertEqual(properties["y"]["type"], "float")

    def test_no_vector_field(self):
        """Lexical search only; a vector field would drag in the 440 MB embedder."""
        for spec in BROWSE["mappings"]["properties"].values():
            self.assertNotEqual(spec.get("type"), "dense_vector")


if __name__ == "__main__":
    unittest.main()
