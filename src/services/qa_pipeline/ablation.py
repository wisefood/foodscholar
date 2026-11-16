"""Per-request pipeline switches for ablation studies.

A request may carry ``ablation``: a few settings from ``SWITCHES`` with the
values to answer it with ("answer this one without the planner"). They are
applied as a context-local overlay on ``config.settings`` for that request
only, so one deployment answers every configuration of a study, interleaved,
from the same code, index and models.

Off by default: unless the deployment sets ``QA_ABLATION_ENABLED=true``, a
request that carries ``ablation`` is refused, never silently answered with the
defaults (a study would then compare the full system with itself).

An ablated answer never shares a cache entry with a normal one: the overlay is
part of the answer cache key (``QAService._build_cache_key``).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from config import SETTINGS_OVERLAY, config


class AblationError(ValueError):
    """An ablation the deployment does not allow, or a malformed one."""


@dataclass(frozen=True)
class Switch:
    kind: type
    what: str
    choices: Optional[Tuple[str, ...]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    # Choices known only at run time (the deployment's models).
    choices_from: Optional[Callable[[], Iterable[str]]] = None
    # A comma list of choices ("recency,influence"); "" is allowed.
    many: bool = False


RANKING_PRIORS = ("tier", "recency", "influence", "design", "affinity")


def _models() -> Iterable[str]:
    """Every model this deployment is configured with, for any role."""
    names = set(dict.get(config.settings, "QA_AVAILABLE_MODELS") or [])
    for key in ("QA_DEFAULT_MODEL", "QA_FAST_MODEL", "QA_UTILITY_MODEL",
                "QA_PLANNER_MODEL", "QA_EVALUATOR_MODEL"):
        value = dict.get(config.settings, key)
        if value:
            names.add(value)
    return sorted(names)


# Every setting a request may override, with what turning it does. Only
# settings read at call time belong here; one read at import would ignore it.
SWITCHES: Dict[str, Switch] = {
    # Baselines
    "QA_PIPELINE_MODE": Switch(
        str, "'legacy': the single-pass flow (one retrieval, one answer)",
        choices=("agentic", "legacy"),
    ),
    # Planning
    "QA_PLANNER_ENABLED": Switch(
        bool, "off: the question as asked, one article and one guideline search"
    ),
    "QA_MAX_SUBQUESTIONS": Switch(int, "sub-questions the planner may write", low=1, high=6),
    "QA_CLARIFY_ENABLED": Switch(bool, "off: never ask the asker for a detail"),
    "QA_SEARCH_FILTERS_ENABLED": Switch(
        bool, "off: searches ignore the planner's filters (years, study design, population)"
    ),
    "QA_THREAD_MEMORY_ENABLED": Switch(
        bool, "off: a follow-up is answered without the thread's earlier turns and notes"
    ),
    "QA_USER_CONTEXT_ENABLED": Switch(
        bool, "off: nothing known about the asker (country, region, age group)"
    ),
    # Evidence check and repair
    "QA_MAX_REPAIR_ROUNDS": Switch(
        int, "0: no evidence check and no follow-up searches", low=0, high=3
    ),
    "QA_GAP_RECHECK_ENABLED": Switch(
        bool, "off: a first-round 'corpus gap' ends the search (as before)"
    ),
    # Retrieval
    "QA_RETRIEVAL_LEGS": Switch(
        str, "which search legs run", choices=("hybrid", "lexical", "vector")
    ),
    "QA_GUIDELINE_RETRIEVAL_MODE": Switch(
        str, "'bm25': guidelines by keywords only", choices=("hybrid", "bm25")
    ),
    "QA_KNN_CANDIDATES": Switch(
        int, "hits from the vector leg (0 = QA_RRF_CANDIDATES; 10 = before the kNN fix)",
        low=0, high=200,
    ),
    "QA_RRF_CANDIDATES": Switch(int, "hits taken from each leg", low=5, high=200),
    "QA_RRF_K": Switch(int, "rank fusion constant", low=1, high=1000),
    # Ranking
    "QA_RERANK_ENABLED": Switch(bool, "off: no relevance grading of candidates before the cut"),
    "QA_RANKING_PRIORS_ENABLED": Switch(
        bool, "off: rank by fused relevance alone (no tier, recency, influence, design, affinity)"
    ),
    "QA_RANKING_PRIORS_OFF": Switch(
        str, "rank without these multipliers only (comma list)", choices=RANKING_PRIORS, many=True
    ),
    "QA_RECENCY_HALF_LIFE_YEARS": Switch(float, "recency decay half-life", low=0.5, high=100),
    "QA_RECENCY_FLOOR": Switch(float, "1.0: no recency decay", low=0, high=1),
    "QA_INFLUENCE_WEIGHT": Switch(float, "0: no citation boost", low=0, high=5),
    "QA_EARNED_TIER_ENABLED": Switch(bool, "off: no earned tier for highly cited articles"),
    "QA_MIN_SCORE": Switch(float, "score floor after ranking", low=0, high=1),
    "QA_PER_DOC_CAP": Switch(int, "rules per guide / chunks per article", low=1, high=50),
    # Evidence handed to the answer
    "QA_ARTICLE_TOP_K": Switch(int, "articles the answer reads (0 = the request's top_k)", low=0, high=20),
    "QA_GUIDELINE_TOP_K": Switch(int, "guideline rules the answer reads (0 = default)", low=0, high=20),
    # Models per role
    "QA_DEFAULT_MODEL": Switch(str, "the model that writes the answer", choices_from=_models),
    "QA_PLANNER_MODEL": Switch(str, "the model that plans the searches", choices_from=_models),
    "QA_EVALUATOR_MODEL": Switch(str, "the model that checks the evidence", choices_from=_models),
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def enabled() -> bool:
    """Whether this deployment accepts ablation requests (env, not overlay)."""
    return bool(dict.get(config.settings, "QA_ABLATION_ENABLED", False))


def _coerce(name: str, switch: Switch, value: Any) -> Any:
    if switch.kind is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise AblationError(f"{name} must be true or false, not {value!r}")
    if switch.kind is str:
        choices = tuple(switch.choices_from()) if switch.choices_from else switch.choices
        if switch.choices_from:  # model ids keep their case
            text = str(value).strip()
        else:
            text = str(value).strip().lower()
        if switch.many:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            bad = [p for p in parts if choices and p not in choices]
            if bad:
                raise AblationError(f"{name} takes a comma list of {list(choices)}, not {bad}")
            return ",".join(p for p in (choices or parts) if p in parts)
        if choices and text not in choices:
            raise AblationError(f"{name} must be one of {list(choices)}, not {value!r}")
        return text
    if isinstance(value, bool):
        raise AblationError(f"{name} must be a number, not {value!r}")
    try:
        number = switch.kind(value)
    except (TypeError, ValueError):
        raise AblationError(f"{name} must be a number, not {value!r}") from None
    if switch.kind is int and float(value) != number:
        raise AblationError(f"{name} must be a whole number, not {value!r}")
    if (switch.low is not None and number < switch.low) or (
        switch.high is not None and number > switch.high
    ):
        raise AblationError(f"{name} must be between {switch.low} and {switch.high}, not {value!r}")
    return number


def validate(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The overlay a request asks for, typed; raises on anything not allowed."""
    if not raw:
        return {}
    if not enabled():
        raise AblationError(
            "This deployment does not accept ablation requests "
            "(QA_ABLATION_ENABLED is off)."
        )
    unknown = sorted(set(raw) - set(SWITCHES))
    if unknown:
        raise AblationError(f"Unknown ablation setting(s): {unknown}; allowed: {sorted(SWITCHES)}")
    return {name: _coerce(name, SWITCHES[name], value) for name, value in sorted(raw.items())}


def current() -> Dict[str, Any]:
    """The overlay the request being answered runs with ({} for a normal one)."""
    return dict(SETTINGS_OVERLAY.get() or {})


@contextmanager
def active(overlay: Optional[Mapping[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Answer the enclosed request with ``overlay`` on top of the settings."""
    if not overlay:
        yield {}
        return
    merged = {**current(), **overlay}
    token = SETTINGS_OVERLAY.set(merged)
    try:
        yield merged
    finally:
        try:
            SETTINGS_OVERLAY.reset(token)
        except ValueError:
            # An async generator closed from another context (garbage
            # collection): that context never saw the overlay.
            pass


def flag(name: str, default: bool = True) -> bool:
    """A boolean switch's effective value for the request being answered."""
    value = config.settings.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def catalog() -> Dict[str, Any]:
    """What runtime-info reports: whether requests may ablate, and what."""
    return {
        "enabled": enabled(),
        "switches": {
            name: {
                "type": switch.kind.__name__,
                "default": dict.get(config.settings, name),
                "what": switch.what,
                **(
                    {"choices": list(switch.choices_from() if switch.choices_from else switch.choices)}
                    if (switch.choices or switch.choices_from) else {}
                ),
            }
            for name, switch in SWITCHES.items()
        },
    }
