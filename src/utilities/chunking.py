"""Text chunking utilities for article processing."""
import re
from typing import List, Dict, Any, Tuple
from models.search import ArticleChunk, ArticleMetadata


class ArticleChunker:
    """Chunks scientific articles into manageable sections."""

    # Common scientific article sections
    STANDARD_SECTIONS = [
        "abstract",
        "introduction",
        "background",
        "methods",
        "methodology",
        "materials and methods",
        "results",
        "findings",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
    ]

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_by_sections(
        self, article_text: str, metadata: ArticleMetadata
    ) -> List[ArticleChunk]:
        """
        Chunk article by detected sections.

        Args:
            article_text: Full article text
            metadata: Article metadata

        Returns:
            List of ArticleChunk objects
        """
        sections = self._detect_sections(article_text)
        chunks = []

        for idx, (section_name, section_text) in enumerate(sections):
            # If section is too large, split it further
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_large_section(section_text)
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    chunks.append(
                        ArticleChunk(
                            article_urn=metadata.urn,
                            section=f"{section_name}_part{sub_idx + 1}",
                            content=sub_chunk,
                            chunk_index=len(chunks),
                            metadata=metadata,
                        )
                    )
            else:
                chunks.append(
                    ArticleChunk(
                        article_urn=metadata.urn,
                        section=section_name,
                        content=section_text,
                        chunk_index=idx,
                        metadata=metadata,
                    )
                )

        return chunks

    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect sections in article text.

        Returns:
            List of (section_name, section_text) tuples
        """
        sections = []
        lines = text.split("\n")

        current_section = "abstract"  # Default first section
        current_text = []

        for line in lines:
            # Check if line is a section header
            section_name = self._is_section_header(line)

            if section_name:
                # Save previous section
                if current_text:
                    sections.append(
                        (current_section, "\n".join(current_text).strip())
                    )

                # Start new section
                current_section = section_name
                current_text = []
            else:
                current_text.append(line)

        # Add last section
        if current_text:
            sections.append((current_section, "\n".join(current_text).strip()))

        return sections

    def _is_section_header(self, line: str) -> str:
        """
        Check if a line is a section header.

        Returns:
            Section name if header, empty string otherwise
        """
        line_lower = line.strip().lower()

        # Check against known sections
        for section in self.STANDARD_SECTIONS:
            # Match exact or with number prefix (e.g., "1. Introduction")
            if line_lower == section or re.match(
                rf"^\d+\.?\s*{re.escape(section)}$", line_lower
            ):
                return section

        # Check for numbered sections (1., 2., etc.)
        if re.match(r"^\d+\.?\s+[A-Z][a-z]+", line.strip()):
            # Extract section name after number
            match = re.match(r"^\d+\.?\s+([A-Za-z\s]+)", line.strip())
            if match:
                return match.group(1).strip().lower()

        return ""

    def _split_large_section(self, text: str) -> List[str]:
        """
        Split a large section into smaller chunks with overlap.

        Args:
            text: Section text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within overlap range
                sentence_end = text.rfind(". ", start, end)
                if sentence_end > start:
                    end = sentence_end + 1

            chunks.append(text[start:end].strip())

            # Move start forward with overlap
            start = end - self.overlap if end - self.overlap > start else end

        return chunks

    def chunk_fixed_size(
        self, text: str, metadata: ArticleMetadata
    ) -> List[ArticleChunk]:
        """
        Chunk text into fixed-size chunks (fallback method).

        Args:
            text: Text to chunk
            metadata: Article metadata

        Returns:
            List of ArticleChunk objects
        """
        chunks = self._split_large_section(text)
        return [
            ArticleChunk(
                article_urn=metadata.urn,
                section=f"chunk_{idx}",
                content=chunk,
                chunk_index=idx,
                metadata=metadata,
            )
            for idx, chunk in enumerate(chunks)
        ]

    def extract_key_sentences(
        self, text: str, max_sentences: int = 5
    ) -> List[str]:
        """
        Extract key sentences from text (simple heuristic).

        Args:
            text: Text to extract from
            max_sentences: Maximum sentences to return

        Returns:
            List of key sentences
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Simple heuristic: prefer longer sentences with keywords
        keywords = [
            "significant",
            "important",
            "found",
            "demonstrated",
            "results",
            "conclusion",
            "however",
            "therefore",
        ]

        scored = []
        for sent in sentences:
            score = len(sent)  # Base score on length
            for kw in keywords:
                if kw in sent.lower():
                    score += 50

            scored.append((score, sent))

        # Sort by score and return top N
        scored.sort(reverse=True, key=lambda x: x[0])
        return [sent for _, sent in scored[:max_sentences]]


def create_article_metadata(article_data: Dict[str, Any]) -> ArticleMetadata:
    """
    Create ArticleMetadata from article data dictionary.

    Args:
        article_data: Dict containing article data from WiseFood API or Elasticsearch

    Returns:
        ArticleMetadata object
    """
    return ArticleMetadata(
        urn=article_data.get("urn", article_data.get("id", "")),
        title=article_data.get("title", "Unknown Title"),
        authors=article_data.get("authors", []),
        abstract=article_data.get("abstract"),
        year=article_data.get("year"),
        journal=article_data.get("journal"),
        doi=article_data.get("doi"),
        keywords=article_data.get("keywords", []),
        tags=article_data.get("tags", []),
        category=article_data.get("category"),
        relevance_score=article_data.get("_score"),  # Elasticsearch score
    )
