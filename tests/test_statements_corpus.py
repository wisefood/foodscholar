"""The experimental FSKG statements corpus (retriever="statements")."""

import unittest
from unittest.mock import patch

from agents.qa_agent import create_source_citation, prepare_source_context, repair_citation_links
from models.qa import PlannedSubQuestion, QAUserContext, SubQuestionFilters
from services.qa_pipeline import retrieval, statements
from services.qa_pipeline.answering import citations_from_inline_links
from services.qa_pipeline.state import EvidenceItem


def _hit(**doc):
    return {"_id": doc["statement_id"], "_score": 2.0, "_source": doc}


GUIDE = dict(statement_id="C0001", family="guide", text="Limit added sugar for children.",
             source="FSAI Healthy Eating (2019)", country="Ireland", section="Sugar",
             population="children", food_entities="added sugar, sweets", page_pdf=3,
             publication_year=2019, statement_type="Recommendation")
JOURNAL = dict(statement_id="NV0001", family="journal", text="SCFA lowered fasting insulin.",
               source="Nutrition Reviews", article_title="Short-chain fatty acids and insulin",
               citation="Pham et al. (2024), Nutr. Rev. 82:193-209", doi="10.1093/nutrit/nuad042",
               publication_year=2024, statement_type="Knowledge")
TEXTBOOK = dict(statement_id="HN0001", family="textbook", text="Nutrients are substances required by the body.",
                source="Human Nutrition: 2020 Edition", chapter="Chapter 1: Basic Concepts",
                section="What are Nutrients?", page_pdf=46, publication_year=2020,
                statement_type="Definition")


class DocumentTests(unittest.TestCase):
    def test_text_column_variants_and_native_language(self):
        hungarian = statements.statement_document({
            "Statement ID": "H0001", "Statement text (English)": "Add little salt.",
            "Statement text (Hungarian)": "Kevés sót!", "Section (Hungarian)": "Tányér",
            "Source type": "National Dietary Guide", "Country": "Hungary",
            "Publication date": "2021-11-10", "Quantity": "-", "Page (PDF)": 2,
        }, workbook="FSKG_Dietary_Guides_Hungary.xlsx")
        self.assertEqual(hungarian["family"], "guide")
        self.assertEqual(hungarian["text"], "Add little salt.")
        self.assertEqual((hungarian["text_native"], hungarian["native_language"]), ("Kevés sót!", "Hungarian"))
        self.assertEqual((hungarian["section"], hungarian["publication_year"], hungarian["page_pdf"]), ("Tányér", 2021, 2))
        self.assertIsNone(hungarian["quantity"])  # "-" means empty

        journal = statements.statement_document(
            {"Statement ID": "NV0001", "Text": "x", "Source type": "Journal article (abstract)"},
            workbook="FSKG_Journals_NutritionReviews.xlsx",
        )
        self.assertEqual(journal["family"], "journal")
        self.assertIsNone(statements.statement_document({"Statement ID": "X"}, workbook="w.xlsx"))


class NormalizeTests(unittest.TestCase):
    def test_each_family_becomes_the_type_the_pipeline_knows(self):
        _, guide, guide_source = statements.normalize_statement_hit(_hit(**GUIDE))
        self.assertEqual((guide["source_type"], guide["rule_text"], guide["guide_region"]), ("guideline", GUIDE["text"], "Ireland"))
        self.assertEqual(guide["urn"], "urn:fskg:statement:C0001")
        self.assertEqual((guide_source.page_no, guide_source.source_type), (3, "guideline"))

        _, journal, journal_source = statements.normalize_statement_hit(_hit(**JOURNAL))
        self.assertEqual((journal["source_type"], journal["authors"], journal["title"]), ("article", ["Pham", "et al."], JOURNAL["article_title"]))
        self.assertEqual(journal_source.doi, JOURNAL["doi"])

        _, textbook, textbook_source = statements.normalize_statement_hit(_hit(**TEXTBOOK))
        self.assertEqual((textbook["source_type"], textbook_source.source_type), ("textbook", "textbook"))
        self.assertEqual(textbook["venue"], TEXTBOOK["chapter"])
        self.assertTrue(textbook["parent_doc"].startswith("textbook:human-nutrition"))

    def test_textbook_cap_counts_chapters_not_statements(self):
        _, a, _ = statements.normalize_statement_hit(_hit(**TEXTBOOK))
        _, b, _ = statements.normalize_statement_hit(_hit(**{**TEXTBOOK, "statement_id": "HN0002"}))
        self.assertEqual(EvidenceItem(payload=a, source=None).parent_doc_key,
                         EvidenceItem(payload=b, source=None).parent_doc_key)


class MetadataTests(unittest.TestCase):
    def test_question_attributes_boost_statement_fields(self):
        filters = SubQuestionFilters(target_populations=["children"], nutrients=["sugar"], regions=["Ireland"])
        clauses = statements.metadata_should_clauses(filters, user_context=None, families=["journal"])
        fields = [c["multi_match"]["fields"][0] for c in clauses]
        self.assertEqual(fields, ["country^5", "population^3", "food_entities^3"])

    def test_asker_context_only_steers_guides(self):
        ctx = QAUserContext(country="Ireland", member_age_group="older adult")
        self.assertEqual(statements.metadata_should_clauses(None, user_context=ctx, families=["journal", "textbook"]), [])
        guide = statements.metadata_should_clauses(None, user_context=ctx, families=["guide"])
        self.assertEqual({c["multi_match"]["query"] for c in guide}, {"Ireland", "older adult"})


class BranchTests(unittest.TestCase):
    def test_planner_routing_maps_to_statement_families(self):
        self.assertEqual(statements.STATEMENT_BRANCH_FAMILIES[statements.STATEMENT_BRANCH_FOR["guidelines"]], ["guide"])
        self.assertEqual(statements.STATEMENT_BRANCH_FAMILIES[statements.STATEMENT_BRANCH_FOR["articles"]], ["journal", "textbook"])

    def test_run_branch_fuses_lexical_and_vector_legs(self):
        sq = PlannedSubQuestion(id="sq1", text="q", why="w", qtype="general", branch="articles",
                                lexical_query="q", dense_query="q")
        hit = statements.normalize_statement_hit(_hit(**JOURNAL))
        with patch.object(statements, "search_statements_lexical", return_value=[hit]) as lexical, \
                patch.object(statements, "search_statements_knn", return_value=[hit]) as knn:
            outcome = retrieval.run_branch(sq, branch="statement_literature", vector=[0.1], user_context=None, expertise_level="beginner")
        self.assertTrue(outcome.ok)
        self.assertEqual(lexical.call_args.kwargs["families"], ["journal", "textbook"])
        self.assertEqual(knn.call_args.kwargs["families"], ["journal", "textbook"])
        self.assertEqual(len(outcome.items), 1)  # same statement from both legs, fused
        self.assertAlmostEqual(outcome.items[0].rrf_norm, 1.0)


class TextbookCitationTests(unittest.TestCase):
    def setUp(self):
        _, self.textbook, _ = statements.normalize_statement_hit(_hit(**TEXTBOOK))

    def test_prompt_block_says_how_to_cite(self):
        block = prepare_source_context([self.textbook])
        self.assertIn("Textbook 1:", block)
        self.assertIn(f"(/textbooks/{self.textbook['urn']})", block)
        self.assertIn("Statement: Nutrients are substances", block)

    def test_citation_and_links_use_the_textbook_type(self):
        citation = create_source_citation(self.textbook, section="statement", quote="Nutrients are substances")
        self.assertEqual((citation.source_type, citation.source_url, citation.page_no), ("textbook", f"/textbooks/{self.textbook['urn']}", 46))
        text = repair_citation_links(f"See [{self.textbook['urn']}].", [self.textbook])
        self.assertIn(f"](/textbooks/{self.textbook['urn']})", text)
        cited = citations_from_inline_links(f"[Human (2020)](/textbooks/{self.textbook['urn']})", [self.textbook])
        self.assertEqual([c["urn"] for c in cited], [self.textbook["urn"]])


if __name__ == "__main__":
    unittest.main()
