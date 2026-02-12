"""Citation validation and tracking utilities."""
import logging
from typing import List, Dict, Any, Set
from models.search import Citation, SynthesizedFinding

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates and tracks citations to ensure accuracy."""

    def __init__(self):
        self.citation_map: Dict[str, Set[str]] = {}  # claim -> set of article URNs

    def validate_citations(
        self, findings: List[SynthesizedFinding]
    ) -> Dict[str, Any]:
        """
        Validate that all findings have proper citations.

        Returns a validation report with:
        - is_valid: bool
        - total_findings: int
        - total_citations: int
        - findings_without_citations: List[str]
        - low_confidence_findings: List[str]
        """
        report = {
            "is_valid": True,
            "total_findings": len(findings),
            "total_citations": 0,
            "findings_without_citations": [],
            "low_confidence_findings": [],
            "duplicate_citations": [],
        }

        seen_citations = set()

        for finding in findings:
            # Check if finding has citations
            if not finding.supporting_citations:
                report["is_valid"] = False
                report["findings_without_citations"].append(finding.finding)
                continue

            # Count citations
            report["total_citations"] += len(finding.supporting_citations)

            # Track low confidence
            if finding.confidence == "low":
                report["low_confidence_findings"].append(finding.finding)

            # Check for duplicate citations
            for citation in finding.supporting_citations:
                citation_key = f"{citation.article_urn}:{citation.section}"
                if citation_key in seen_citations:
                    report["duplicate_citations"].append(citation_key)
                seen_citations.add(citation_key)

        return report

    def extract_unique_articles(self, citations: List[Citation]) -> List[str]:
        """Extract unique article URNs from citations."""
        return list(set(citation.article_urn for citation in citations))

    def group_citations_by_article(
        self, citations: List[Citation]
    ) -> Dict[str, List[Citation]]:
        """Group citations by article URN."""
        grouped: Dict[str, List[Citation]] = {}
        for citation in citations:
            if citation.article_urn not in grouped:
                grouped[citation.article_urn] = []
            grouped[citation.article_urn].append(citation)
        return grouped

    def calculate_citation_diversity(self, citations: List[Citation]) -> float:
        """
        Calculate citation diversity (0-1).
        Higher score means citations are spread across more articles.
        """
        if not citations:
            return 0.0

        unique_articles = len(self.extract_unique_articles(citations))
        total_citations = len(citations)

        return unique_articles / total_citations

    def validate_citation_metadata(self, citation: Citation) -> List[str]:
        """
        Validate that a citation has required metadata.
        Returns list of missing fields.
        """
        missing = []

        if not citation.article_urn:
            missing.append("article_urn")
        if not citation.article_title:
            missing.append("article_title")
        if not citation.section:
            missing.append("section")
        if not citation.confidence:
            missing.append("confidence")

        return missing

    def ensure_citation_quality(
        self, citations: List[Citation], min_confidence: str = "medium"
    ) -> List[Citation]:
        """
        Filter citations to ensure minimum quality.
        Returns only citations meeting the confidence threshold.
        """
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_order[min_confidence]

        return [
            c
            for c in citations
            if confidence_order.get(c.confidence, 0) >= min_level
        ]

    def track_citation(self, claim: str, article_urn: str):
        """Track which articles support which claims."""
        if claim not in self.citation_map:
            self.citation_map[claim] = set()
        self.citation_map[claim].add(article_urn)

    def get_supporting_articles(self, claim: str) -> Set[str]:
        """Get all articles that support a claim."""
        return self.citation_map.get(claim, set())


def create_citation_from_article(
    article_metadata: Dict[str, Any],
    section: str,
    quote: str = None,
    confidence: str = "medium",
) -> Citation:
    """
    Helper function to create a Citation from article metadata.

    Args:
        article_metadata: Dict containing article metadata
        section: Section name being cited
        quote: Optional direct quote
        confidence: Citation confidence level

    Returns:
        Citation object
    """
    # Extract year as int from publication_year (e.g. "2017-01-01" -> 2017)
    raw_year = article_metadata.get("publication_year") or article_metadata.get("year")
    year = None
    if raw_year:
        try:
            year = int(str(raw_year)[:4])
        except (ValueError, TypeError):
            pass

    return Citation(
        article_urn=article_metadata.get("urn", ""),
        article_title=article_metadata.get("title", "Unknown Title"),
        authors=article_metadata.get("authors", []),
        year=year,
        journal=article_metadata.get("venue") or article_metadata.get("journal"),
        section=section,
        quote=quote,
        confidence=confidence,
        relevance_score=article_metadata.get("relevance_score"),
    )
