"""
Enforcement of the catalog's editorial policy during retrieval.

Two fields on a catalog article govern how it may be used as evidence:

``reader_visibility``
    ``public`` | ``expert_only`` | ``hidden`` — which readers an article reaches.
    Enforced against the QA request's ``expertise_level``.

``indexing_tier`` / ``ai_indexing_tier``
    Editorial tier, falling back to the one the enrichment agent proposed.
    Scales the retrieval score, so an editorially promoted ("prime") article
    outranks its neighbours and an archived one sinks.

Both live in wisefood-data-api (``ArticleSchema``) and are written from the
console. This module only *reads* them off retrieval payloads — FoodScholar owns
no policy state of its own.

Articles indexed before those fields existed carry neither. Every function here
treats an absent value as "public, untiered", so the legacy corpus behaves
exactly as it did before the fields were introduced.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vocabulary (mirrors wisefood-data-api schemas.IndexingTier / ReaderVisibility)
# --------------------------------------------------------------------------- #

TIER_PRIME = "prime"
TIER_CORE = "core"
TIER_SUPPORTIVE = "supportive"
TIER_SPECIALIZED = "specialized"
TIER_ARCHIVE_ONLY = "archive_only"
TIER_DO_NOT_INDEX = "do_not_index"

# Score multipliers. Neutral is `supportive` (1.0), so an article with no tier
# ranks exactly where retrieval put it.
TIER_BOOSTS: Dict[str, float] = {
    TIER_PRIME: 1.6,
    TIER_CORE: 1.25,
    TIER_SUPPORTIVE: 1.0,
    TIER_SPECIALIZED: 0.9,
    TIER_ARCHIVE_ONLY: 0.6,
    TIER_DO_NOT_INDEX: 0.0,
}

INDEXING_TIERS: Tuple[str, ...] = tuple(TIER_BOOSTS)

VISIBILITY_PUBLIC = "public"
VISIBILITY_EXPERT_ONLY = "expert_only"
VISIBILITY_HIDDEN = "hidden"

# Reader expertise levels that may see `expert_only` articles. Mirrors
# catalog_access.EXPERT_READER_LEVELS on the catalog side.
EXPERT_AUDIENCES = frozenset({"expert"})

# The enrichment prompt speaks in title case ("Archive-only", "Do not index");
# the catalog stores slugs. Normalize both, plus the obvious variants.
_TIER_LOOKUP: Dict[str, str] = {tier: tier for tier in INDEXING_TIERS}
_TIER_LOOKUP.update(
    {
        "archive-only": TIER_ARCHIVE_ONLY,
        "archive only": TIER_ARCHIVE_ONLY,
        "archive": TIER_ARCHIVE_ONLY,
        "do not index": TIER_DO_NOT_INDEX,
        "do-not-index": TIER_DO_NOT_INDEX,
        "excluded": TIER_DO_NOT_INDEX,
    }
)


def normalize_tier(value: Any) -> Optional[str]:
    """Map any spelling of a tier onto the catalog's slug, or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    return _TIER_LOOKUP.get(value.strip().lower())


def tier_boost(tier: Optional[str]) -> float:
    """Score multiplier for a tier; unknown or absent tiers are neutral."""
    if not tier:
        return 1.0
    return TIER_BOOSTS.get(tier, 1.0)


def is_expert_audience(expertise_level: Optional[str]) -> bool:
    """Whether an audience may see ``expert_only`` articles."""
    if not isinstance(expertise_level, str):
        return False
    return expertise_level.strip().lower() in EXPERT_AUDIENCES


def reader_visibility_of(payload: Dict[str, Any]) -> str:
    """Reader visibility of an article payload; absent means ``public``."""
    value = payload.get("reader_visibility")
    if not isinstance(value, str) or not value.strip():
        return VISIBILITY_PUBLIC
    normalized = value.strip().lower()
    if normalized in {VISIBILITY_EXPERT_ONLY, VISIBILITY_HIDDEN, VISIBILITY_PUBLIC}:
        return normalized
    logger.debug("Unknown reader_visibility %r; treating as public", value)
    return VISIBILITY_PUBLIC


def effective_tier(payload: Dict[str, Any]) -> Optional[str]:
    """
    The tier that applies: the editor's if set, otherwise the agent's proposal.

    ``extras.evaluation.indexing_tier`` is checked last, for articles enriched
    before ``ai_indexing_tier`` became a catalog field.
    """
    candidates: List[Any] = [
        payload.get("indexing_tier"),
        payload.get("ai_indexing_tier"),
    ]

    extras = payload.get("extras")
    if isinstance(extras, dict):
        evaluation = extras.get("evaluation")
        if isinstance(evaluation, dict):
            candidates.append(evaluation.get("indexing_tier"))

    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        candidates.append(evaluation.get("indexing_tier"))

    for candidate in candidates:
        tier = normalize_tier(candidate)
        if tier:
            return tier
    return None


def is_visible_to(payload: Dict[str, Any], *, expert: bool) -> bool:
    """Whether an article payload may be shown to this audience."""
    visibility = reader_visibility_of(payload)
    if visibility == VISIBILITY_HIDDEN:
        return False
    if visibility == VISIBILITY_EXPERT_ONLY:
        return expert
    return True


# --------------------------------------------------------------------------- #
# Elasticsearch pre-filtering
# --------------------------------------------------------------------------- #


def reader_visibility_filter(expertise_level: Optional[str]) -> Dict[str, Any]:
    """
    The ES clause that keeps restricted articles out of the result set.

    Phrased as ``must_not``: a positive ``term`` on ``reader_visibility`` would
    drop every article indexed before the field existed.
    """
    excluded = [VISIBILITY_HIDDEN]
    if not is_expert_audience(expertise_level):
        excluded.append(VISIBILITY_EXPERT_ONLY)

    return {"terms": {"reader_visibility": excluded}}


def excluded_tiers_filter() -> Dict[str, Any]:
    """The ES clause that drops articles an editor removed from retrieval."""
    return {"terms": {"indexing_tier": [TIER_DO_NOT_INDEX]}}


def article_filter_query(expertise_level: Optional[str]) -> Dict[str, Any]:
    """
    The full article retrieval filter: not deleted, and readable by this audience.

    Replaces the bare `status != deleted` filter the retrievers used before.
    """
    return {
        "bool": {
            "must_not": [
                {"term": {"status": "deleted"}},
                reader_visibility_filter(expertise_level),
                excluded_tiers_filter(),
            ]
        }
    }


# --------------------------------------------------------------------------- #
# Post-retrieval enforcement
# --------------------------------------------------------------------------- #


def filter_and_rank(
    payloads: List[Dict[str, Any]],
    sources: List[Any],
    *,
    expertise_level: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Drop articles this audience may not see and re-rank the rest by tier.

    Belt-and-braces for the ES pre-filter, and the only enforcement available for
    retrievers that do not go through Elasticsearch (LinearRAG). Guideline
    results pass through untouched, and articles keep their block position in the
    payload list — only their order *within* that block changes, because
    retrieval scores are not comparable across the two indices.

    ``sources`` are ``RetrievedSource`` models, which do not carry the policy
    fields; they are matched to their payload by URN.
    """
    expert = is_expert_audience(expertise_level)

    policy_by_urn: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        urn = payload.get("urn") or payload.get("_id")
        if isinstance(urn, str) and urn:
            policy_by_urn.setdefault(urn, payload)

    def payload_is_article(payload: Dict[str, Any]) -> bool:
        return payload.get("source_type") != "guideline"

    def source_is_article(source: Any) -> bool:
        return getattr(source, "source_type", None) != "guideline"

    kept_payloads = [
        p for p in payloads if not payload_is_article(p) or is_visible_to(p, expert=expert)
    ]

    def source_visible(source: Any) -> bool:
        if not source_is_article(source):
            return True
        payload = policy_by_urn.get(getattr(source, "urn", None) or "")
        # A source with no matching payload has no policy to apply; keep it.
        return payload is None or is_visible_to(payload, expert=expert)

    kept_sources = [s for s in sources if source_visible(s)]

    ranked_payloads = _rank_articles(
        kept_payloads,
        is_article=payload_is_article,
        score_of=lambda p: _score_value(p.get("_score", p.get("relevance_score", 0.0))),
        tier_of=effective_tier,
        annotate=_annotate_payload,
    )
    ranked_sources = _rank_articles(
        kept_sources,
        is_article=source_is_article,
        score_of=lambda s: _score_value(getattr(s, "similarity_score", 0.0)),
        tier_of=lambda s: effective_tier(
            policy_by_urn.get(getattr(s, "urn", None) or "", {})
        ),
        annotate=None,
    )

    if limit is not None:
        ranked_payloads = _truncate_articles(
            ranked_payloads, limit, is_article=payload_is_article
        )
        ranked_sources = _truncate_articles(
            ranked_sources, limit, is_article=source_is_article
        )

    return ranked_payloads, ranked_sources


def _rank_articles(items, *, is_article, score_of, tier_of, annotate):
    """Reorder the article subsequence by boosted score, slot for slot."""
    slots = [i for i, item in enumerate(items) if is_article(item)]
    if not slots:
        return items

    scored = []
    for position, index in enumerate(slots):
        item = items[index]
        tier = tier_of(item)
        boost = tier_boost(tier)
        boosted = score_of(item) * boost
        if annotate is not None:
            annotate(item, tier, boost, boosted)
        # `position` keeps the sort stable at equal boosted scores.
        scored.append((-boosted, position, item))

    scored.sort(key=lambda entry: (entry[0], entry[1]))

    ordered = list(items)
    for index, (_, _, item) in zip(slots, scored):
        ordered[index] = item
    return ordered


def _annotate_payload(
    payload: Dict[str, Any],
    tier: Optional[str],
    boost: float,
    boosted: float,
) -> None:
    """Record why a payload ranked where it did (surfaces in QA debug output)."""
    payload["effective_indexing_tier"] = tier
    payload["editorial_boost"] = boost
    payload["editorial_score"] = boosted


def _truncate_articles(items: List[Any], limit: int, *, is_article) -> List[Any]:
    """Keep at most ``limit`` articles, leaving non-articles untouched."""
    out: List[Any] = []
    kept = 0
    for item in items:
        if is_article(item):
            if kept >= limit:
                continue
            kept += 1
        out.append(item)
    return out


def _score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
