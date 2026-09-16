"""Tests for the population facet.

The Round 2 finding is pinned directly: "Plant-based diets and iron status"
came back full of pregnancy studies, and adding "in adults" fixed it.
"""
import unittest

from services.population_facet import (
    ADULTS,
    CHILDREN,
    OLDER_ADULTS,
    PREGNANCY,
    group_by_population,
    population_of_query,
    population_of_source,
    population_penalty,
    rerank_for_population,
)


class TestAnnotatedValues(unittest.TestCase):
    def test_every_spelling_of_adults_collapses(self):
        """The index holds these five spellings of one value."""
        for value in ("Adults (18-64)", "Adults", "Adults 18-64",
                      "Adults with obesity", "Adults (18"):
            with self.subTest(value=value):
                self.assertEqual(
                    population_of_source({"population_group": value}), {ADULTS}
                )

    def test_truncation_artefacts_still_resolve(self):
        """'Infants (0' is a truncated 'Infants (0-2)' from the annotator."""
        self.assertEqual(
            population_of_source({"population_group": "Infants (0"}), {"infants"}
        )

    def test_older_adults_is_not_read_as_adults(self):
        self.assertEqual(
            population_of_source({"population_group": "Older adults (65+)"}),
            {OLDER_ADULTS},
        )

    def test_not_stated_and_mixed_are_unknown(self):
        for value in ("Not stated", "Mixed", "Not", "", None):
            with self.subTest(value=value):
                self.assertEqual(population_of_source({"population_group": value}), set())


class TestPregnancyFromText(unittest.TestCase):
    def test_pregnancy_is_detected_from_text(self):
        """The vocabulary has no value for it; text is the only signal."""
        self.assertIn(
            PREGNANCY,
            population_of_source({"title": "Maternal iron supplementation trial"}),
        )

    def test_various_pregnancy_terms(self):
        for text in ("during pregnancy", "gestational diabetes", "antenatal care",
                     "lactating women", "postpartum recovery"):
            with self.subTest(text=text):
                self.assertIn(PREGNANCY, population_of_source({"abstract": text}))

    def test_unrelated_text_is_not_pregnancy(self):
        self.assertNotIn(
            PREGNANCY, population_of_source({"title": "Iron status in athletes"})
        )


class TestQueryParsing(unittest.TestCase):
    def test_the_broad_query_names_no_population(self):
        self.assertEqual(population_of_query("Plant-based diets and iron status"), set())

    def test_adding_in_adults_names_one(self):
        self.assertEqual(
            population_of_query("Plant-based diets and iron status in adults"), {ADULTS}
        )

    def test_older_adults_wins_over_adults(self):
        self.assertEqual(population_of_query("iron in older adults"), {OLDER_ADULTS})

    def test_colloquial_terms(self):
        self.assertEqual(population_of_query("is iron safe for kids"), {CHILDREN})


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"title": "Iron status in pregnancy: a cohort",
             "population_group": "Adults (18-64)"},
            {"title": "Iron and plant-based diets in adults",
             "population_group": "Adults (18-64)"},
            {"title": "Unlabelled iron review", "population_group": "Not stated"},
            {"title": "Iron in children", "population_group": "Children (3-12)"},
        ]

    def test_pregnancy_is_demoted_even_when_also_labelled_adults(self):
        """The reported failure.

        A pregnancy cohort IS an adults study, so an overlap test alone scores
        it a perfect match for "in adults" and it keeps the top slot.
        """
        question = "Plant-based diets and iron status in adults"
        ordered = rerank_for_population(question, self.pool)
        self.assertEqual(ordered[0]["title"], "Iron and plant-based diets in adults")
        self.assertEqual(ordered[-1]["title"], "Iron status in pregnancy: a cohort")

    def test_pregnancy_ranks_first_when_asked_for(self):
        ordered = rerank_for_population("Iron status during pregnancy", self.pool)
        self.assertIn("pregnancy", ordered[0]["title"].lower())

    def test_unannotated_sources_are_not_penalised(self):
        """~79% of the corpus is unannotated; penalising it would gut results."""
        wanted = population_of_query("iron in adults")
        self.assertEqual(population_penalty(wanted, {"title": "no annotation"}), 0)

    def test_a_query_naming_no_population_reorders_nothing(self):
        before = [s["title"] for s in self.pool]
        after = [s["title"] for s in rerank_for_population("iron status", self.pool)]
        self.assertEqual(before, after)

    def test_empty_pool(self):
        self.assertEqual(rerank_for_population("iron in adults", []), [])


class TestGrouping(unittest.TestCase):
    def test_sources_are_bucketed_for_segmenting_an_answer(self):
        grouped = group_by_population([
            {"title": "adults", "population_group": "Adults (18-64)"},
            {"title": "kids", "population_group": "Children (3-12)"},
            {"title": "unknown", "population_group": "Not stated"},
        ])
        self.assertEqual({"adults", "children", "unspecified"}, set(grouped))

    def test_a_multi_population_source_appears_in_each(self):
        """A maternal study annotated Adults is both, and is filed under both."""
        grouped = group_by_population([
            {"title": "iron in working adults", "population_group": "Adults (18-64)"},
        ])
        self.assertEqual(set(grouped), {"adults"})
        grouped = group_by_population([
            {"title": "maternal iron", "population_group": "Adults (18-64)"},
        ])
        self.assertEqual(set(grouped), {"adults", "pregnancy"})

    def test_empty_buckets_are_dropped(self):
        self.assertEqual(group_by_population([]), {})


if __name__ == "__main__":
    unittest.main()
