"""The corpus's own tags, and how a search uses them.

Every guideline rule and every enriched article is tagged: who it is for,
which food group and nutrient, which condition, the setting, the study type,
whether it is a human study, and more. The agent used to write such
constraints as free text ("toddlers") against keyword fields that hold slugs
("under_5_years"), which matches nothing, and only ever as soft boosts.

Here the agent works with the tags as they are:

- ``catalog()`` reads each tag's vocabulary from the live indices (cached), and
  the planner gets it with the question, so it writes exact values for any
  aspect the question has, and says which are requirements (``must``);
- ``canonical`` maps what it wrote onto the vocabulary and drops the rest;
- ``clauses`` turns a search's wants into Elasticsearch boosts, and its
  requirements into filters that still let an untagged source through (most
  articles are untagged; a requirement must not hide them);
- ``fit`` scores one source against one search's wants for ranking;
- ``tags`` is what the relevance grader and the evidence check are shown.

Nothing here knows what a tag means. Age is the one aspect that is a number,
not a tag; ``audience.py`` handles it and feeds the same wants.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tag fields per index, with what the planner is told about each.
FIELDS: Dict[str, Dict[str, str]] = {
    "guidelines": {
        "target_populations": "who the rule is for",
        "life_stage": "life stage the rule is for",
        "food_groups": "food group the rule is about",
        "nutrients": "nutrient the rule is about",
        "health_conditions": "condition the rule addresses",
        "topic": "what the rule is about",
        "setting": "where it applies",
        "audience": "who acts on it: the person or their caregiver",
        "frequency": "how often",
        "action_type": "what the rule tells you to do",
    },
    "articles": {
        "study_type": "study design",
        "biological_model": "studied in humans, animals or in vitro",
        "population_group": "who was studied",
        "age_group": "age of those studied",
        "topics": "research area",
    },
}
# Values that say nothing: a source carrying only these is untagged.
NEUTRAL = {"not stated", "not", "mixed", "none", "unknown", "general", "general_population", "n/a"}
# Vocabulary shown to the planner per field (the rest still canonicalises).
SHOWN = 40
CACHE_SECONDS = 3600

PREFER_MATCH, PREFER_MISS = 1.2, 0.85
MUST_MATCH, MUST_MISS = 1.3, 0.25
# Nearly every rule is tagged, so a match there means something. Only about
# one article in eight is, so rewarding a match would reward being tagged,
# not being right: for articles a match earns nothing and a mismatch costs.
MATCH_EARNS = {"guidelines": True, "articles": False}
# A want naming more than this share of a field's values says nothing.
TOO_BROAD = 0.5
MAX_VALUES = 4
# Matches stack only this far: five agreeing tags are not five times the
# evidence that a source answers the question (the grader decides that).
FIT_FLOOR, FIT_CEILING = 0.15, 1.6
# A requirement that leaves fewer hits than this is relaxed to a preference.
RELAX_BELOW = 3

# The vocabularies on 2026-11-11, for when the index cannot be asked.
_FALLBACK: Dict[str, Dict[str, List[str]]] = {
    "guidelines": {
        "target_populations": ["under_5_years", "general_population", "adults", "ages_5_to_18", "elderly", "lactating_people", "pregnant_people"],
        "life_stage": ["early_childhood", "adulthood", "older_adulthood", "adolescence", "school_age", "lactation", "pregnancy"],
        "food_groups": ["grains", "protein_foods", "fats_and_oils", "vegetables", "fruits", "dairy", "beverages", "sugars_and_sweets"],
        "nutrients": ["fat", "sugar", "protein", "salt", "iron", "calcium", "vitamin_d"],
        "setting": ["general", "home", "clinical", "retail", "workplace"],
        "audience": ["individual", "caregiver"],
        "frequency": ["daily", "weekly", "per_meal", "occasional"],
        "action_type": ["eat", "do", "drink", "limit", "use"],
    },
    "articles": {
        "study_type": ["Observational (Cross-sectional)", "Observational (Cohort)", "Randomized Controlled Trial", "Animal Study", "Narrative Review", "Systematic Review", "Meta-analysis"],
        "biological_model": ["Human", "Animal", "In vitro"],
        "age_group": ["Adults (18-64)", "Children (3-12)", "Older adults (65+)", "Adolescents (13-18)", "Infants (0-2)"],
        "population_group": ["Adults (18-64)", "Children (3-12)", "Older adults (65+)", "Adolescents (13-18)", "Infants (0-2)"],
    },
}

_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _neutral(value: Any) -> bool:
    return str(value).strip().lower() in NEUTRAL or _norm(value) in NEUTRAL or not str(value).strip()


def _load() -> Dict[str, Dict[str, List[str]]]:
    from backend.elastic import ELASTIC_CLIENT
    from services.qa_retrievers import guideline_retrieval_filter

    out: Dict[str, Dict[str, List[str]]] = {}
    for index, fields in FIELDS.items():
        base = guideline_retrieval_filter() if index == "guidelines" else {"match_all": {}}
        query = {"bool": {"filter": base if isinstance(base, list) else [base]}}
        aggs = {field: {"terms": {"field": field, "size": 400}} for field in fields}
        response = ELASTIC_CLIENT.client.search(
            index=index, body={"size": 0, "query": query, "aggs": aggs}
        )
        out[index] = {
            field: [
                str(bucket["key"])
                for bucket in response["aggregations"][field]["buckets"]
                # One-off spellings and truncated values are noise, not vocabulary.
                if not _neutral(bucket["key"]) and bucket["doc_count"] >= (2 if index == "articles" else 1)
            ]
            for field in fields
        }
    return out


def catalog() -> Dict[str, Dict[str, List[str]]]:
    """Each tag's vocabulary, most used first; cached, never raises."""
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["value"]
    try:
        value = _load()
    except Exception as exc:  # the index is a convenience here
        logger.warning("Facet catalog unavailable, using the built-in one: %s", exc)
        # Ask again in a minute rather than keeping the stand-in for an hour.
        _cache.update(at=now - CACHE_SECONDS + 60, value=_cache["value"] or _FALLBACK)
        return _cache["value"]
    _cache.update(at=now, value=value)
    return value


def for_planner() -> Dict[str, Any]:
    """What the planner (and the evidence check) is given with the question."""
    vocab = catalog()
    return {
        "how_to_use": (
            "Tags the sources carry. In a sub-question's filters.facets, name a "
            "field and the values the question calls for, using ONLY these "
            "fields and values (guideline fields for guideline searches, article "
            "fields for article searches). Use every aspect the question has: "
            "who it is about, the food group, the nutrient, a condition, the "
            "setting, the kind of study. List a field in filters.must only "
            "when a source for anything else would be wrong for this question "
            "(e.g. the life stage of the person asked about); everything else "
            "is a preference. Leave out aspects the question does not have."
        ),
        **{
            index: {
                field: {"what": FIELDS[index][field], "values": values[:SHOWN]}
                for field, values in fields.items()
                if values
            }
            for index, fields in vocab.items()
        },
    }


def index_of(field: str) -> Optional[str]:
    for index, fields in FIELDS.items():
        if field in fields:
            return index
    return None


def canonical(field: str, values: Iterable[Any]) -> List[str]:
    """What the agent wrote for a field, as the index spells it.

    Exact after normalising first; then a value whose words are contained in
    a vocabulary value's (or the reverse), so "toddlers under 5" still finds
    "under_5_years" and "Adults" finds every spelling of the adult group.
    Anything else is dropped: a tag that is not in the index matches nothing.
    """
    index = index_of(field)
    if index is None:
        return []
    vocabulary = catalog().get(index, {}).get(field) or []
    # Most used spelling first: "Adults (18-64)" and "Adults 18-64" normalise
    # alike, and the common one must not be lost to the rare one.
    by_norm: Dict[str, str] = {}
    for v in vocabulary:
        by_norm.setdefault(_norm(v), v)
    out: List[str] = []
    for value in list(values or [])[:MAX_VALUES]:
        key = _norm(value)
        if not key:
            continue
        if key in by_norm:
            hits = [by_norm[key]]
        else:
            words = {w.rstrip("s") for w in key.split("_") if len(w) > 2}
            hits = [
                v for n, v in by_norm.items()
                if words and (
                    words <= {w.rstrip("s") for w in n.split("_")}
                    or {w.rstrip("s") for w in n.split("_") if len(w) > 2} <= words
                )
            ]
        # Every spelling of the same group ("Adults", "Adults (18-64)").
        if index == "articles" and hits:
            head = _norm(hits[0]).split("_")[0]
            hits += [v for v in vocabulary if _norm(v).split("_")[0] == head]
        for hit in hits:
            if hit not in out:
                out.append(hit)
    # "All of them" is not a want (seen: every study type listed for a
    # question about what a food contains).
    families = {_norm(v).split("_")[0] for v in out} if index == "articles" else set(out)
    all_families = {_norm(v).split("_")[0] for v in vocabulary} if index == "articles" else set(vocabulary)
    if len(all_families) >= 4 and len(families) > TOO_BROAD * len(all_families):
        return []
    return out[:12]


def wants(filters: Any, index: str) -> Dict[str, Tuple[List[str], bool]]:
    """One search's wants for an index: field -> (values, is a requirement)."""
    if filters is None:
        return {}
    must = set(getattr(filters, "must", None) or [])
    out: Dict[str, Tuple[List[str], bool]] = {}
    for field, values in (getattr(filters, "facets", None) or {}).items():
        if index_of(field) == index and values:
            out[field] = (list(values), field in must)
    return out


def clauses(filters: Any, index: str, *, relax: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(filters, boosts) for one search. A requirement admits a source tagged
    with a wanted value or not tagged for that field at all."""
    hard: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []
    for field, (values, required) in wants(filters, index).items():
        soft.append({"terms": {field: values, "boost": 4.0 if required else 2.5}})
        if required and not relax:
            vocabulary = catalog().get(index, {}).get(field) or []
            others = [v for v in vocabulary if v not in values]
            # "Not one of the other values" keeps the untagged and the neutral.
            if others:
                hard.append({"bool": {"should": [
                    {"terms": {field: values}},
                    {"bool": {"must_not": {"terms": {field: others}}}},
                ], "minimum_should_match": 1}})
    return hard, soft


def _values(payload: Dict[str, Any], field: str) -> List[str]:
    raw = payload.get(field)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    return [str(v) for v in raw if not _neutral(v)]


def fit(payload: Dict[str, Any], filters: Any) -> Tuple[float, Dict[str, str]]:
    """How one source fits one search's wants: (multiplier, per-field outcome).

    A source that carries a wanted tag rises; one tagged with something else
    for that field sinks, hard when the field is a requirement; an untagged
    source is left alone, because silence is not a mismatch.
    """
    index = "guidelines" if payload.get("source_type") == "guideline" else "articles"
    factor = 1.0
    detail: Dict[str, str] = {}
    for field, (wanted, required) in wants(filters, index).items():
        have = _values(payload, field)
        if not have:
            continue
        if set(have) & set(wanted):
            if MATCH_EARNS[index]:
                factor *= MUST_MATCH if required else PREFER_MATCH
            detail[field] = "match"
        else:
            factor *= MUST_MISS if required else PREFER_MISS
            detail[field] = "miss"
    return min(max(factor, FIT_FLOOR), FIT_CEILING), detail


def tags(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A source's own tags, for the grader and the evidence check."""
    index = "guidelines" if payload.get("source_type") == "guideline" else "articles"
    out: Dict[str, Any] = {}
    for field in FIELDS[index]:
        have = _values(payload, field)
        if have:
            out[field] = have if len(have) > 1 else have[0]
    return out


def describe(filters: Any) -> Dict[str, Any]:
    """A search's wants, for the grader and the evidence check."""
    if filters is None:
        return {}
    out: Dict[str, Any] = {}
    if getattr(filters, "facets", None):
        out["facets"] = dict(filters.facets)
    if getattr(filters, "must", None):
        out["must"] = list(filters.must)
    low, high = getattr(filters, "age_min_months", None), getattr(filters, "age_max_months", None)
    if low is not None or high is not None:
        out["age_months"] = [low, high]
    return out
