"""Age: the one aspect of a source that is a number, not a tag.

Tags (who, food group, nutrient, study type, …) are handled generically in
``facets.py``; the planner names them. Age needs arithmetic: guideline rules
carry a window in months (``age_min_months`` / ``age_max_months``, -1 when not
stated), articles a group like "Children (3-12)", and a question says "my
3-year-old". In production 208 of 531 searchable rules are written for
children, and nothing pushed them down when an adult asked, so the answer to
"how much meat is a serving" was the 30 g meant for 1-4 year olds.

This module is the safety net under the planner, not a second brain:

- ``resolve`` reads who the question is about from its own words, then the
  plan, then the asker's profile, and ``apply`` fills a search's wants with it
  only where the planner named nobody;
- ``age_fit`` scores a source's age against a search's age window;
- ``unstated_fit`` makes rules written only for a special group (children,
  pregnancy, lactation) step back when the search names nobody.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

ADULT_MONTHS = 18 * 12

MATCH_FACTOR = 1.3      # the rule's age window overlaps the audience's
MISMATCH_FACTOR = 0.25  # written for another age entirely
UNSTATED_FACTOR = 0.5   # nobody named: a group-only rule steps back


@dataclass(frozen=True)
class Group:
    tokens: Tuple[str, ...]
    life_stages: Tuple[str, ...]
    populations: Tuple[str, ...]
    months: Tuple[Optional[int], Optional[int]]


# First match wins: "older adult" before "adult", "school" before "child".
GROUPS: List[Group] = [
    Group(("pregnan",), ("pregnancy",), ("pregnant_people",), (None, None)),
    Group(("lactat", "breastfeed", "nursing mother"), ("lactation",), ("lactating_people",), (None, None)),
    Group(("infant", "baby", "babies", "newborn"), ("infancy",), ("under_5_years",), (0, 12)),
    Group(("toddler", "preschool", "pre-school", "nursery"), ("early_childhood",), ("under_5_years",), (12, 60)),
    Group(("teen", "adolescen"), ("adolescence",), ("ages_5_to_18",), (144, 216)),
    Group(("school", "primary"), ("school_age",), ("ages_5_to_18",), (60, 144)),
    Group(("child", "kid", "son", "daughter"), ("early_childhood", "school_age"), ("under_5_years", "ages_5_to_18"), (12, 216)),
    Group(("elder", "older", "senior", "aged 65", "over 65", "retire"), ("older_adulthood",), ("elderly",), (780, 1560)),
    Group(("adult", "grown-up"), ("adulthood",), ("adults", "general_population"), (216, 780)),
]

_AGE = re.compile(
    r"\b(\d{1,2})\s*(?:-|\s)?\s*(year|yr|month|mo)s?(?:\s*-?\s*old|\s+of\s+age)?\b", re.I
)
_RANGE = re.compile(r"\baged?\s+(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})\b", re.I)


@dataclass
class Audience:
    """The group a question is about; empty when nobody is named."""

    min_months: Optional[int] = None
    max_months: Optional[int] = None
    life_stages: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def known(self) -> bool:
        return bool(self.life_stages or self.populations or self.min_months is not None)

    def describe(self) -> Dict[str, Any]:
        return {
            "min_months": self.min_months,
            "max_months": self.max_months,
            "life_stages": self.life_stages,
            "populations": self.populations,
            "source": self.source,
        }


def _from_group(group: Group, source: str) -> Audience:
    low, high = group.months
    return Audience(low, high, list(group.life_stages), list(group.populations), source)


def _from_months(months: int, source: str) -> Audience:
    """An exact age ("3-year-old") as a window, with the matching slugs."""
    for group in GROUPS:
        low, high = group.months
        if low is not None and high is not None and low <= months < high and group.tokens[0] != "child":
            return Audience(months, months, list(group.life_stages), list(group.populations), source)
    return Audience(months, months, [], [], source)


def _from_text(text: str, source: str) -> Optional[Audience]:
    if not text:
        return None
    lowered = text.lower()
    found = _RANGE.search(lowered)
    if found:
        low, high = sorted(int(v) for v in found.groups())
        audience = _from_months(low * 12, source)
        audience.min_months, audience.max_months = low * 12, high * 12 + 11
        return audience
    found = _AGE.search(lowered)
    if found:
        number, unit = int(found.group(1)), found.group(2).lower()
        return _from_months(number if unit.startswith("mo") else number * 12, source)
    for group in GROUPS:
        if any(token in lowered for token in group.tokens):
            return _from_group(group, source)
    return None


def resolve(
    question: str,
    *,
    populations: Iterable[str] = (),
    member_age_group: Optional[str] = None,
) -> Audience:
    """The question's own words first, then the plan, then the profile."""
    return (
        _from_text(question, "question")
        or _from_text(" ".join(p for p in populations if p), "plan")
        or _from_text(member_age_group or "", "profile")
        or Audience()
    )


def _months(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        months = int(value)
    except (TypeError, ValueError):
        return None
    return months if months >= 0 else None


def _slugs(value: Any) -> set:
    if isinstance(value, str):
        value = [value]
    return {str(v).strip().lower() for v in (value or []) if str(v).strip()}


def rule_window(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    return _months(payload.get("age_min_months")), _months(payload.get("age_max_months"))


def _group_only(payload: Dict[str, Any]) -> bool:
    """A rule for children, pregnancy or lactation only, not for everyone."""
    low, high = rule_window(payload)
    if high is not None and high < ADULT_MONTHS:
        return True
    stages = _slugs(payload.get("life_stage"))
    return bool(stages) and stages <= {"pregnancy", "lactation", "infancy", "early_childhood", "school_age", "adolescence"}


POPULATION_FIELDS = ("target_populations", "life_stage", "population_group", "age_group")


def names_somebody(filters: Any) -> bool:
    """Whether a search's wants say who it is about, by tag or by age."""
    if filters is None:
        return False
    facets = getattr(filters, "facets", None) or {}
    return (
        any(facets.get(field) for field in POPULATION_FIELDS)
        or getattr(filters, "age_min_months", None) is not None
        or getattr(filters, "age_max_months", None) is not None
    )


def apply(filters: Any, audience: Optional[Audience]) -> Any:
    """Fill a search's wants with who the question is about, where the
    planner named nobody. Stated in the question it is a requirement; taken
    from the asker's profile it is a preference (they may be asking for
    someone else)."""
    if filters is None or audience is None or not audience.known or names_somebody(filters):
        return filters
    from services.qa_pipeline import facets as corpus

    wanted = dict(filters.facets)
    for field, values in (("life_stage", audience.life_stages), ("target_populations", audience.populations)):
        found = corpus.canonical(field, values)
        if found:
            wanted[field] = found
    stated = audience.source in ("question", "plan")
    must = list(filters.must)
    if stated:
        must += [f for f in ("life_stage",) if f in wanted and f not in must]
        if audience.min_months is not None and "age" not in must:
            must.append("age")
    return filters.model_copy(update={
        "facets": wanted,
        "must": must,
        "age_min_months": audience.min_months,
        "age_max_months": audience.max_months,
    })


def _article_window(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    studied = _from_text(str(payload.get("age_group") or payload.get("population_group") or ""), "article")
    if studied is None:
        return None, None
    return studied.min_months, studied.max_months


def age_fit(payload: Dict[str, Any], filters: Any) -> float:
    """A source's age against the search's age window (1.0 when either is silent)."""
    low = getattr(filters, "age_min_months", None) if filters is not None else None
    high = getattr(filters, "age_max_months", None) if filters is not None else None
    if low is None and high is None:
        return 1.0
    guideline = payload.get("source_type") == "guideline"
    src_low, src_high = rule_window(payload) if guideline else _article_window(payload)
    if src_low is None and src_high is None:
        return 1.0
    overlaps = (src_low or 0) <= (high if high is not None else 10_000) and (
        src_high if src_high is not None else 10_000
    ) >= (low or 0)
    if overlaps:
        return MATCH_FACTOR if guideline else 1.2
    required = "age" in (getattr(filters, "must", None) or [])
    # A rule for another age gives the wrong amounts; a study in another age
    # group can still inform.
    if guideline:
        return MISMATCH_FACTOR if required else 0.6
    return 0.5 if required else 0.8


def unstated_fit(payload: Dict[str, Any], filters: Any) -> float:
    """Nobody named: a rule only for a special group steps back."""
    if payload.get("source_type") != "guideline" or names_somebody(filters):
        return 1.0
    return UNSTATED_FACTOR if _group_only(payload) else 1.0


def age_clauses(filters: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(filters, boosts) on the guideline age window for one search."""
    low = getattr(filters, "age_min_months", None) if filters is not None else None
    high = getattr(filters, "age_max_months", None) if filters is not None else None
    if low is None and high is None:
        return [], []
    overlap = {"bool": {"must": [
        {"range": {"age_min_months": {"gte": 0, "lte": high if high is not None else 10_000}}},
        {"range": {"age_max_months": {"gte": low or 0}}},
    ]}}
    soft = [{**overlap, "bool": {**overlap["bool"], "boost": 3.0}}]
    hard: List[Dict[str, Any]] = []
    if "age" in (getattr(filters, "must", None) or []):
        # In the window, or with no window stated (-1 / missing).
        hard.append({"bool": {"should": [
            overlap,
            {"bool": {"must_not": {"range": {"age_max_months": {"gte": 0}}}}},
        ], "minimum_should_match": 1}})
    return hard, soft


def describe_in_words(audience: Optional[Audience]) -> str:
    """For the grader and the evidence check."""
    if audience is None or not audience.known:
        return "not stated: a general adult reader"
    parts = [s.replace("_", " ") for s in audience.life_stages or audience.populations]
    if audience.min_months is not None:
        high = audience.max_months
        if high is not None and high == audience.min_months:
            parts.append(f"age {audience.min_months // 12} years" if audience.min_months >= 24 else f"age {audience.min_months} months")
        else:
            parts.append(f"ages {audience.min_months}-{high if high is not None else '…'} months")
    return ", ".join(dict.fromkeys(parts))


def age_of(payload: Dict[str, Any]) -> Optional[str]:
    """A source's age window in words, when it states one."""
    if payload.get("source_type") == "guideline":
        low, high = rule_window(payload)
        if low is None and high is None:
            return None
        return f"{low if low is not None else 0}-{high if high is not None else '…'} months"
    group = str(payload.get("age_group") or "").strip()
    return None if not group or group.lower().startswith(("not", "mixed")) else group
