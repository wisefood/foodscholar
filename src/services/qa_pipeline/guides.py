"""A rule is read in its guide's context.

Rules were tagged one at a time, from their own text. "Count 30g cooked beef
as 1 serving" says nothing about age, so it was tagged ``general_population``
with no life stage and no age window; its guide is "Healthy Ireland Food
Pyramid: a guide for children aged 1-4". In that guide 21 of 26 rules carry no
life stage, and three other guides are the same (59 of 67, 19 of 20, 17 of
17). Tags alone therefore cannot tell that the rule is a toddler's portion.

``context()`` learns, per guide, who it is for: from what the guide says of
itself (title and description), else from what its tagged rules agree on.
``attach`` gives every retrieved rule its guide's title and description, and a
rule that states no audience of its own inherits the guide's, marked as
inherited. Ranking, the grader, the evidence check and the answer then all
see the rule as its guide meant it.

The lasting fix is in enrichment (tag rules with their guide in view); until
the corpus is re-tagged this makes the agent right at question time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from services.qa_pipeline import audience as audience_mod

logger = logging.getLogger(__name__)

CACHE_SECONDS = 3600
# Tagged rules of a guide "agree" when this share of them names one stage.
AGREE_SHARE = 0.8
AGREE_MIN = 3
SPECIFIC_STAGES = ("life_stage",)

_cache: Dict[str, Any] = {"at": 0.0, "value": None}


@dataclass
class GuideContext:
    title: str = ""
    description: str = ""
    life_stages: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    min_months: Optional[int] = None
    max_months: Optional[int] = None
    source: str = ""  # "guide" (it says so) or "rules" (its rules agree)

    @property
    def names_somebody(self) -> bool:
        return bool(self.life_stages or self.min_months is not None)


def _from_guide_text(title: str, description: str) -> Optional[audience_mod.Audience]:
    return audience_mod._from_text(f"{title}. {description}", "guide")


def _load() -> Dict[str, GuideContext]:
    from backend.elastic import ELASTIC_CLIENT
    from services.qa_retrievers import guideline_retrieval_filter

    es = ELASTIC_CLIENT.client
    out: Dict[str, GuideContext] = {}
    guides = es.search(
        index="guides",
        body={"size": 500, "query": {"match_all": {}}, "_source": ["urn", "title", "description"]},
    )
    for hit in guides.get("hits", {}).get("hits", []):
        src = hit.get("_source") or {}
        urn = src.get("urn")
        if not urn:
            continue
        ctx = GuideContext(title=str(src.get("title") or ""), description=str(src.get("description") or ""))
        said = _from_guide_text(ctx.title, ctx.description)
        if said is not None:
            ctx.life_stages, ctx.populations = list(said.life_stages), list(said.populations)
            ctx.min_months, ctx.max_months, ctx.source = said.min_months, said.max_months, "guide"
        out[urn] = ctx

    base = guideline_retrieval_filter()
    agg = es.search(
        index="guidelines",
        body={
            "size": 0,
            "query": {"bool": {"filter": base if isinstance(base, list) else [base]}},
            "aggs": {"g": {"terms": {"field": "guide_urn", "size": 500}, "aggs": {
                "stages": {"terms": {"field": "life_stage", "size": 3}},
                "tagged": {"filter": {"exists": {"field": "life_stage"}}},
                "low": {"filter": {"range": {"age_min_months": {"gte": 0}}}, "aggs": {"v": {"min": {"field": "age_min_months"}}}},
                "high": {"filter": {"range": {"age_max_months": {"gte": 0}}}, "aggs": {"v": {"max": {"field": "age_max_months"}}}},
            }}},
        },
    )
    for bucket in agg["aggregations"]["g"]["buckets"]:
        ctx = out.setdefault(bucket["key"], GuideContext())
        if ctx.names_somebody:
            continue  # what the guide says of itself wins
        tagged = bucket["tagged"]["doc_count"]
        stages = bucket["stages"]["buckets"]
        if tagged >= AGREE_MIN and stages and stages[0]["doc_count"] / tagged >= AGREE_SHARE:
            ctx.life_stages = [stages[0]["key"]]
            ctx.source = "rules"
            low, high = bucket["low"]["v"].get("value"), bucket["high"]["v"].get("value")
            # A window only when the stage is one that has an age to it.
            if ctx.life_stages[0] != "adulthood" and low is not None and high is not None:
                ctx.min_months, ctx.max_months = int(low), int(high)
    return out


def context() -> Dict[str, GuideContext]:
    """Every guide's context; cached, never raises ({} when unavailable)."""
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["value"]
    try:
        value = _load()
    except Exception as exc:
        logger.warning("Guide context unavailable; rules keep their own tags: %s", exc)
        _cache.update(at=now - CACHE_SECONDS + 60, value=_cache["value"] or {})
        return _cache["value"]
    _cache.update(at=now, value=value)
    return value


def _states_audience(payload: Dict[str, Any]) -> bool:
    """Whether the rule itself says who it is for (beyond 'everyone')."""
    low, high = audience_mod.rule_window(payload)
    return bool(audience_mod._slugs(payload.get("life_stage"))) or low is not None or high is not None


def attach(payloads: Iterable[Dict[str, Any]]) -> None:
    """Give each rule its guide's title, description and, where the rule is
    silent about who it is for, the guide's audience."""
    known = None
    for payload in payloads:
        if payload.get("source_type") != "guideline":
            continue
        if known is None:
            known = context()
        ctx = known.get(payload.get("guide_urn") or "")
        if ctx is None:
            continue
        if ctx.title:
            payload.setdefault("guide_title", ctx.title)
        if ctx.description:
            payload.setdefault("guide_description", ctx.description[:300])
        if not ctx.names_somebody or _states_audience(payload):
            continue
        payload["life_stage"] = list(ctx.life_stages)
        if ctx.populations:
            payload["target_populations"] = list(ctx.populations)
        if ctx.min_months is not None:
            payload["age_min_months"] = ctx.min_months
            payload["age_max_months"] = ctx.max_months if ctx.max_months is not None else -1
        payload["audience_from_guide"] = ctx.source
