"""Tests for the evidence grading that orders sources by study design.

The three Round 2 findings this exists for are each pinned below: a mouse study
leading key findings, a conference item outranking stronger work, and
preclinical evidence reaching a reader as though it were about them.
"""
import unittest

from services.evidence_grade import (
    GRADE_EDITORIAL,
    GRADE_PRECLINICAL,
    GRADE_SYSTEMATIC,
    GRADE_UNKNOWN,
    grade_source,
    prefer_human,
    rank_key,
)


class TestGradeSource(unittest.TestCase):
    def test_systematic_review_in_humans_is_strongest(self):
        g = grade_source({"study_type": "Meta-analysis", "biological_model": "Human"})
        self.assertEqual(g.grade, GRADE_SYSTEMATIC)
        self.assertTrue(g.is_human)
        self.assertFalse(g.is_weak)

    def test_rct_in_animals_is_preclinical_despite_the_study_type(self):
        """An RCT in mice is still the wrong study for 'what should I eat'."""
        g = grade_source(
            {"study_type": "Randomized Controlled Trial", "biological_model": "Animal"}
        )
        self.assertEqual(g.grade, GRADE_PRECLINICAL)
        self.assertFalse(g.is_human)
        self.assertTrue(g.is_preclinical)

    def test_animal_study_is_labelled_for_the_reader(self):
        g = grade_source({"study_type": "Animal Study", "biological_model": "Animal"})
        self.assertEqual(g.label, "animal study")

    def test_in_vitro_is_labelled_as_a_laboratory_study(self):
        g = grade_source(
            {"study_type": "In Vitro / Cell Study", "biological_model": "In vitro"}
        )
        self.assertEqual(g.label, "laboratory study")

    def test_in_vitro_casing_variants_all_match(self):
        """The index holds three casings of this value; all must grade alike."""
        for model in ("In vitro", "In Vitro / Cell Study", "in vitro / cell study"):
            with self.subTest(model=model):
                self.assertFalse(grade_source({"biological_model": model}).is_human)

    def test_conference_item_cannot_outrank_stronger_evidence(self):
        g = grade_source({"study_type": "Systematic Review", "type": ["Conference"]})
        self.assertEqual(g.grade, GRADE_EDITORIAL)
        self.assertTrue(g.is_weak)

    def test_editorial_is_weak(self):
        self.assertTrue(grade_source({"type": ["Editorial", "JournalArticle"]}).is_weak)

    def test_unannotated_sources_outrank_preclinical(self):
        """~89% of the corpus has no study_type; it must not sink below mice."""
        unknown = grade_source({})
        animal = grade_source({"biological_model": "Animal"})
        self.assertEqual(unknown.grade, GRADE_UNKNOWN)
        self.assertLess(unknown.grade, animal.grade)
        self.assertFalse(unknown.is_weak)
        self.assertIsNone(unknown.is_human)

    def test_unknown_carries_no_label(self):
        self.assertEqual(grade_source({}).label, "")

    def test_never_raises_on_junk(self):
        for junk in (None, "string", 42, [], {"study_type": None}):
            with self.subTest(junk=junk):
                self.assertIsNotNone(grade_source(junk))


class TestRanking(unittest.TestCase):
    def test_relevance_still_orders_within_a_grade(self):
        weak_match = {"study_type": "Observational (Cohort)", "_score": 1.0}
        strong_match = {"study_type": "Observational (Cohort)", "_score": 9.0}
        self.assertLess(rank_key(strong_match), rank_key(weak_match))

    def test_a_highly_relevant_mouse_study_does_not_lead(self):
        """The Round 2 finding, as a test."""
        pool = [
            {"t": "mouse", "study_type": "Animal Study",
             "biological_model": "Animal", "_score": 9.9},
            {"t": "human", "study_type": "Observational (Cohort)",
             "biological_model": "Human", "_score": 2.0},
        ]
        self.assertEqual([s["t"] for s in prefer_human(pool)], ["human", "mouse"])

    def test_preclinical_is_kept_not_discarded(self):
        """A question with only animal evidence still gets an answer."""
        pool = [{"t": "mouse", "biological_model": "Animal", "_score": 5.0}]
        self.assertEqual(len(prefer_human(pool)), 1)

    def test_limit_is_applied_after_reordering(self):
        pool = [
            {"t": "mouse", "biological_model": "Animal", "_score": 9.9},
            {"t": "human", "biological_model": "Human", "_score": 1.0},
        ]
        self.assertEqual([s["t"] for s in prefer_human(pool, limit=1)], ["human"])

    def test_empty_pool(self):
        self.assertEqual(prefer_human([]), [])


if __name__ == "__main__":
    unittest.main()
