"""The FSKG statements corpus as an alternative QA evidence source (experimental).

``retriever="statements"`` answers from short, self-contained statements
extracted from national dietary guides, review-article abstracts and nutrition
textbooks (the FSKG workbooks), instead of the article and guideline indices.
Everything else in the pipeline stays the same, so the two corpora can be
compared question for question.

Retrieval mirrors the article/guideline branches signal for signal: a BM25 leg
with metadata boosts, an embedding leg with metadata filters, fused by RRF in
``retrieval.run_branch``. The planner's routing is kept: a ``guidelines``
sub-question searches guide statements, an ``articles`` one searches journal
and textbook statements.

Each statement becomes a payload of the type the rest of the pipeline already
handles: guide statements are guidelines (country as region, PDF page), journal
statements are articles (DOI, article title, first author from the citation),
and textbook statements are the one new type, ``textbook``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.elastic import ELASTIC_CLIENT
from config import config
from models.qa import QAUserContext, RetrievedSource, SubQuestionFilters

FAMILY_GUIDE = "guide"
FAMILY_JOURNAL = "journal"
FAMILY_TEXTBOOK = "textbook"

# Planner branch -> statement branch, and statement branch -> families searched.
STATEMENT_BRANCH_FOR = {"guidelines": "statement_guides", "articles": "statement_literature"}
STATEMENT_BRANCH_FAMILIES = {
    "statement_guides": [FAMILY_GUIDE],
    "statement_literature": [FAMILY_JOURNAL, FAMILY_TEXTBOOK],
}

LEXICAL_FIELDS = [
    "text^3",
    "food_entities^2",
    "population^2",
    "article_title^2",
    "section",
    "chapter",
    "text_native",
]

INDEX_MAPPING: Dict[str, Any] = {
    "properties": {
        "statement_id": {"type": "keyword"},
        "family": {"type": "keyword"},
        "workbook": {"type": "keyword"},
        "statement_type": {"type": "keyword"},
        "dimension": {"type": "keyword"},
        "action": {"type": "keyword"},
        "text": {"type": "text"},
        "text_native": {"type": "text"},
        "native_language": {"type": "keyword"},
        "quantity": {"type": "text"},
        "population": {"type": "text"},
        "food_entities": {"type": "text"},
        "section": {"type": "text"},
        "chapter": {"type": "text"},
        "article_title": {"type": "text"},
        "source": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "source_kind": {"type": "keyword"},
        "country": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "publication_date": {"type": "keyword"},
        "publication_year": {"type": "integer"},
        "page_pdf": {"type": "integer"},
        "page_printed": {"type": "keyword"},
        "doi": {"type": "keyword"},
        "citation": {"type": "text"},
        "publisher": {"type": "text"},
        "license": {"type": "keyword"},
        "embedding": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "cosine",
        },
    }
}


def statements_index() -> str:
    return str(config.settings.get("STATEMENTS_INDEX") or "fskg_statements")


# --------------------------------------------------------------------------- #
# Workbook rows -> index documents (used by scripts/index_statements.py)
# --------------------------------------------------------------------------- #


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN from pandas
        return None
    text = str(value).strip()
    return text if text and text != "-" else None


def _int(value: Any) -> Optional[int]:
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def family_of(source_kind: Optional[str], workbook: str) -> str:
    kind = (source_kind or "").lower()
    if "guide" in kind or "_Dietary_Guides_" in workbook:
        return FAMILY_GUIDE
    if "journal" in kind or "_Journals_" in workbook:
        return FAMILY_JOURNAL
    return FAMILY_TEXTBOOK


def statement_document(row: Dict[str, Any], *, workbook: str) -> Optional[Dict[str, Any]]:
    """One Statements-sheet row as an index document (no embedding yet).

    The workbooks name the text column differently (``Text``, ``Statement
    text``, ``Statement text (English)`` with the original language beside
    it); the English text is the one searched and embedded.
    """
    text = _clean(row.get("Text") or row.get("Statement text") or row.get("Statement text (English)"))
    statement_id = _clean(row.get("Statement ID"))
    if not text or not statement_id:
        return None
    native_column = next(
        (c for c in row if c.startswith("Statement text (") and not c.endswith("(English)")), None
    )
    section = _clean(row.get("Section")) or next(
        (_clean(row[c]) for c in row if c.startswith("Section (")), None
    )
    published = _clean(row.get("Publication date"))
    year = _int(published[:4]) if published else None
    source_kind = _clean(row.get("Source type"))
    return {
        "statement_id": statement_id,
        "family": family_of(source_kind, workbook),
        "workbook": workbook,
        "statement_type": _clean(row.get("Statement type")),
        "dimension": _clean(row.get("Dimension")),
        "action": _clean(row.get("Action")),
        "text": text,
        "text_native": _clean(row.get(native_column)) if native_column else None,
        "native_language": native_column[len("Statement text ("):-1] if native_column else None,
        "quantity": _clean(row.get("Quantity")),
        "population": _clean(row.get("Population")),
        "food_entities": _clean(row.get("Food entities")),
        "section": section,
        "chapter": _clean(row.get("Chapter")),
        "article_title": _clean(row.get("Article")),
        "source": _clean(row.get("Source")),
        "source_kind": source_kind,
        "country": _clean(row.get("Country")),
        "publication_date": published,
        "publication_year": year,
        "page_pdf": _int(row.get("Page (PDF)")),
        "page_printed": _clean(row.get("Page (printed)")),
        "doi": _clean(row.get("DOI")),
        "citation": _clean(row.get("Citation")),
        "publisher": _clean(row.get("Publisher")),
        "license": _clean(row.get("License")),
    }


# --------------------------------------------------------------------------- #
# Search legs
# --------------------------------------------------------------------------- #


def _family_filter(families: List[str]) -> Dict[str, Any]:
    return {"terms": {"family": families}}


def _year_filters(filters: Optional[SubQuestionFilters]) -> List[Dict[str, Any]]:
    if filters is None or not (filters.year_min or filters.year_max):
        return []
    years: Dict[str, Any] = {}
    if filters.year_min:
        years["gte"] = filters.year_min
    if filters.year_max:
        years["lte"] = filters.year_max
    return [{"range": {"publication_year": years}}]


def _boost(term: str, fields: List[str]) -> Dict[str, Any]:
    return {"multi_match": {"query": term, "fields": fields, "type": "best_fields"}}


def metadata_should_clauses(
    filters: Optional[SubQuestionFilters],
    *,
    user_context: Optional[QAUserContext],
    families: List[str],
) -> List[Dict[str, Any]]:
    """Boosts from the question's attributes and, for guides, the asker's context.

    The statement fields stand in for the ones the article and guideline
    clauses use: Population for population_group / target_populations, Food
    entities for food_groups / nutrients, Country for guide_region. Boosts
    only, as for guidelines: a hard filter would hide statements whose
    metadata cell was left empty.
    """
    clauses: List[Dict[str, Any]] = []
    if filters is not None and not filters.is_empty():
        for region in filters.regions[:3]:
            clauses.append(_boost(region, ["country^5", "source"]))
        for population in filters.target_populations[:4]:
            clauses.append(_boost(population, ["population^3", "text"]))
        for term in [*filters.food_groups[:4], *filters.nutrients[:4]]:
            clauses.append(_boost(term, ["food_entities^3", "text^2"]))
        for condition in filters.health_conditions[:3]:
            clauses.append(_boost(condition, ["population^3", "text^2"]))
    # The asker's geography and age only steer guideline-like evidence, as
    # guideline_context_should_clauses does for the guideline index.
    if user_context and FAMILY_GUIDE in families:
        for place in {user_context.country, user_context.region} - {None, ""}:
            clauses.append(_boost(place, ["country^4", "source"]))
        if user_context.member_age_group:
            clauses.append(_boost(user_context.member_age_group, ["population^2"]))
    return clauses


def search_statements_lexical(
    query: str,
    *,
    families: List[str],
    size: int,
    user_context: Optional[QAUserContext],
    filters: Optional[SubQuestionFilters] = None,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    bool_query: Dict[str, Any] = {
        "must": [_boost(query, LEXICAL_FIELDS)],
        "filter": [_family_filter(families), *_year_filters(filters)],
    }
    should = metadata_should_clauses(filters, user_context=user_context, families=families)
    if should:
        bool_query["should"] = should
    response = ELASTIC_CLIENT.client.search(
        index=statements_index(),
        body={"size": size, "query": {"bool": bool_query}, "_source": {"excludes": ["embedding"]}},
    )
    return [normalize_statement_hit(hit) for hit in response.get("hits", {}).get("hits", [])]


def search_statements_knn(
    vector: List[float],
    *,
    families: List[str],
    size: int,
    filters: Optional[SubQuestionFilters] = None,
) -> List[Tuple[str, Dict[str, Any], RetrievedSource]]:
    raw_results = ELASTIC_CLIENT.knn_search(
        index_name=statements_index(),
        query_vector=vector,
        k=size,
        num_candidates=max(size * 4, 100),
        field="embedding",
        filter_query=[_family_filter(families), *_year_filters(filters)],
        source_excludes=["embedding"],
    )
    results = []
    for raw in raw_results:
        hit = {"_id": raw.pop("_id", None), "_score": raw.pop("_score", 0.0), "_source": raw}
        results.append(normalize_statement_hit(hit))
    return results


# --------------------------------------------------------------------------- #
# Hits -> payloads the pipeline already understands
# --------------------------------------------------------------------------- #

_FIRST_AUTHOR = re.compile(r"^\s*([^,(]+?)(\s+et al\.?)?\s*\(")


def _split_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def statement_urn(statement_id: str) -> str:
    return f"urn:fskg:statement:{statement_id}"


def normalize_statement_hit(hit: Dict[str, Any]) -> Tuple[str, Dict[str, Any], RetrievedSource]:
    """(identity key, payload, display source) for one statement hit."""
    doc = dict(hit.get("_source") or {})
    statement_id = doc.get("statement_id") or str(hit.get("_id"))
    family = doc.get("family") or FAMILY_TEXTBOOK
    text = doc.get("text") or ""
    year = doc.get("publication_year")
    score = float(hit.get("_score") or 0.0)
    payload: Dict[str, Any] = {
        **doc,
        "_id": hit.get("_id"),
        "_score": score,
        "urn": statement_urn(statement_id),
        "retriever": "statements",
        "relevance_score": score,
        "abstract": text,
        "description": text,
        "publication_year": str(year) if year else None,
        "page_no": doc.get("page_pdf"),
    }
    source_name = doc.get("source") or "FSKG statement"

    if family == FAMILY_GUIDE:
        payload.update(
            source_type="guideline",
            rule_text=text,
            title=doc.get("section") or source_name,
            guide_urn=f"urn:fskg:guide:{_slug(source_name)}",
            guide_region=doc.get("country"),
            target_populations=_split_list(doc.get("population")),
            food_groups=_split_list(doc.get("food_entities")),
        )
        source = RetrievedSource(
            source_type="guideline", urn=payload["urn"], title=payload["title"],
            venue=doc.get("country"), publication_year=payload["publication_year"],
            category=doc.get("statement_type"), similarity_score=score,
            guide_urn=payload["guide_urn"], page_no=doc.get("page_pdf"),
        )
    elif family == FAMILY_JOURNAL:
        match = _FIRST_AUTHOR.match(doc.get("citation") or "")
        authors = None
        if match:
            authors = [match.group(1).strip()] + (["et al."] if match.group(2) else [])
        payload.update(
            source_type="article",
            title=doc.get("article_title") or source_name,
            authors=authors,
            venue=source_name,
            population_group=doc.get("population"),
        )
        source = RetrievedSource(
            source_type="article", urn=payload["urn"], title=payload["title"],
            authors=authors, venue=source_name, publication_year=payload["publication_year"],
            category=doc.get("statement_type"), similarity_score=score, doi=doc.get("doi"),
        )
    else:
        payload.update(
            source_type="textbook",
            title=source_name,
            venue=doc.get("chapter") or doc.get("publisher"),
            # The per-document diversity cap counts a chapter, not a statement:
            # one textbook must not be able to fill every evidence slot.
            parent_doc=f"textbook:{_slug(source_name)}:{_slug(doc.get('chapter') or '')}",
        )
        source = RetrievedSource(
            source_type="textbook", urn=payload["urn"], title=source_name,
            venue=payload["venue"], publication_year=payload["publication_year"],
            category=doc.get("statement_type"), similarity_score=score, page_no=doc.get("page_pdf"),
        )
    return f"{payload['source_type']}:{payload['urn']}", payload, source
