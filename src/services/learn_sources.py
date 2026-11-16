"""Source sampling for the Learn content: the daily quiz and the tips of the day.

Three things keep the daily content from repeating itself:

- balance: guideline rules are taken guide by guide in turn, and studies
  journal by journal, so one large guide or one journal cannot fill the head
  of the pool the model reads first; studies are blended in every day, not
  only used to top up a short pool of rules;
- breadth: a profile's topics steer half of the pool and the rest is drawn
  from the whole corpus, so a narrow profile does not shrink the pool to the
  same handful of rules every day;
- memory: sources shown on recent days go to the back of the queue.

And one thing keeps it true: a rule written for a group (toddlers, pregnancy,
older adults) carries that group into the prompt, with its guide's title, and
its age window into the evidence, so a 1-4 year old's portion is never asked
about as if it were everyone's.

Everything here is pure, except the two Redis helpers, which never raise.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Days a shown source stays at the back of the queue.
LEARN_RECENT_DAYS = 14
# Age windows reaching adulthood apply to everyone; only narrower ones are a
# group worth naming.
ADULT_MONTHS = 18 * 12

# Put in front of the sources the model writes from. It sits in the data, not
# in the prompt template, so it holds whichever version of a prompt is live.
APPLIES_TO_NOTE = (
    "Each source applies only to the group named in its 'Guide' and 'Applies to' "
    "lines. Anything written from a source must name that group (for example "
    "'For children aged 1 to 4, ...') and must never present an amount or rule "
    "meant for one group as advice for everyone."
)


def dedupe(items: Iterable[T], key: Callable[[T], Any]) -> List[T]:
    """Keep the first of each key; items without a key are dropped."""
    seen: Set[Any] = set()
    out: List[T] = []
    for item in items:
        k = key(item)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def interleave_by_group(
    items: Sequence[T],
    group: Callable[[T], Optional[str]],
    per_group: Optional[int] = None,
) -> List[T]:
    """Round-robin across groups: one from each group in turn.

    Nothing is dropped unless ``per_group`` caps it, so a corpus with only a
    few guides keeps its pool; what changes is the order, and the prompt only
    sees the head of the list, so the head is spread across groups. The order
    within a group is kept (a fresh-first pool stays fresh-first). Items with
    no group each count as their own group.
    """
    buckets: Dict[str, List[T]] = {}
    order: List[str] = []
    for index, item in enumerate(items):
        name = group(item) or f"\0solo{index}"
        if name not in buckets:
            buckets[name] = []
            order.append(name)
        if per_group is None or len(buckets[name]) < per_group:
            buckets[name].append(item)
    out: List[T] = []
    depth = max((len(b) for b in buckets.values()), default=0)
    for round_no in range(depth):
        for name in order:
            if round_no < len(buckets[name]):
                out.append(buckets[name][round_no])
    return out


def prefer_unseen(items: Sequence[T], key: Callable[[T], Any], recent: Set[str]) -> List[T]:
    """Sources not shown recently first, then the rest (each in its order)."""
    if not recent:
        return list(items)
    fresh = [item for item in items if key(item) not in recent]
    shown = [item for item in items if key(item) in recent]
    return fresh + shown


def balanced_order(
    items: Sequence[T],
    key: Callable[[T], Any],
    group: Callable[[T], Optional[str]],
    recent: Set[str],
    limit: Optional[int] = None,
) -> List[T]:
    """Unseen sources first, then recently shown ones; each part group by group.

    ``limit`` keeps the pool (and so the prompt) the size it was before it was
    widened with a corpus-wide draw.
    """
    fresh = [item for item in items if key(item) not in recent]
    shown = [item for item in items if key(item) in recent]
    ordered = interleave_by_group(fresh, group) + interleave_by_group(shown, group)
    return ordered[:limit] if limit is not None else ordered


def blend(primary: Sequence[T], secondary: Sequence[T], run: int = 2) -> List[T]:
    """``run`` of primary, then one of secondary, until both run out."""
    out: List[T] = []
    i = j = 0
    run = max(int(run), 1)
    while i < len(primary) or j < len(secondary):
        for _ in range(run):
            if i < len(primary):
                out.append(primary[i])
                i += 1
        if j < len(secondary):
            out.append(secondary[j])
            j += 1
    return out


def _months(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        months = int(value)
    except (TypeError, ValueError):
        return None
    return months if months >= 0 else None  # -1 means "not stated"


def guideline_age_window(guideline: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """A rule's age window in months, only when it names a group.

    A window that reaches adulthood from early on ("from 1 year") applies to
    everyone; children's windows end before 18, older adults' start after.
    """
    low = _months(guideline.get("age_min_months"))
    high = _months(guideline.get("age_max_months"))
    if low is None and high is None:
        return None, None
    if (high is not None and high < ADULT_MONTHS) or (low is not None and low > ADULT_MONTHS):
        return low, high
    return None, None


def format_age_window(low: Optional[int], high: Optional[int]) -> str:
    """English form for the prompt ("ages 1 to 4 years"); '' without a window."""
    if low is None and high is None:
        return ""
    in_months = (high is not None and high <= 24) or any(
        v is not None and v % 12 for v in (low, high)
    )

    def unit(v: int) -> str:
        return str(v) if in_months else str(v // 12)

    label = "months" if in_months else "years"
    if low is not None and low == high:
        return f"age {unit(low)} {label}"
    if low is not None and high is not None:
        return f"ages {unit(low)} to {unit(high)} {label}"
    if low is not None:
        return f"ages {unit(low)} {label} and over"
    return f"ages up to {unit(high)} {label}"  # type: ignore[arg-type]


def _join(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v).replace("_", " ").strip() for v in value if v)
    return str(value or "").replace("_", " ").strip()


def guideline_applies_to(guideline: Dict[str, Any]) -> str:
    """Who a rule is for, in English for the prompt; '' when nothing is stated."""
    parts: List[str] = []
    populations = _join(guideline.get("target_populations"))
    life_stage = _join(guideline.get("life_stage"))
    if populations:
        parts.append(populations)
    if life_stage and life_stage.lower() not in populations.lower():
        parts.append(life_stage)
    window = format_age_window(*guideline_age_window(guideline))
    if window:
        parts.append(window)
    return "; ".join(parts)


# --- Recently shown sources (Redis sorted set: member = urn, score = day) ---


def _recent_key(kind: str) -> str:
    return f"qa:learn:recent:{kind}"


def read_recent_sources(redis: Any, kind: str, today: int, days: int = LEARN_RECENT_DAYS) -> Set[str]:
    """Sources shown in the last ``days`` days; empty when Redis is unavailable."""
    if redis is None:
        return set()
    try:
        members = redis.zrangebyscore(_recent_key(kind), today - days, "+inf")
    except Exception as exc:  # Redis is a convenience here
        logger.warning("Learn recent-sources read failed (%s): %s", kind, exc)
        return set()
    return {m.decode() if isinstance(m, bytes) else str(m) for m in members or []}


def remember_sources(
    redis: Any, kind: str, urns: Iterable[Optional[str]], today: int, days: int = LEARN_RECENT_DAYS
) -> None:
    """Record today's sources and forget those older than the window."""
    keys = {u for u in urns if u}
    if redis is None or not keys:
        return
    key = _recent_key(kind)
    try:
        redis.zadd(key, {u: today for u in keys})
        redis.zremrangebyscore(key, "-inf", today - days - 1)
        redis.expire(key, (days + 2) * 86400)
    except Exception as exc:
        logger.warning("Learn recent-sources write failed (%s): %s", kind, exc)
