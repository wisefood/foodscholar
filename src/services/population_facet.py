"""Which population a question is about, and which one a source studied.

Round 2: a broad search for "Plant-based diets and iron status" came back full
of pregnancy studies, and adding "in adults" fixed it. A second finding is the
same gap from the other side — one answer mixing adults, pregnant women and
children, which is hard to follow and easy to misapply.

The annotation does not cover this on its own. `population_group` and
`age_group` are near-duplicate fields with the same values, and **neither has
any value for pregnancy** — while 8,481 articles mention it in their title or
abstract. So a reader had no way to express "not pregnancy" and the retriever
had no way to honour it.

Hence two sources of signal, in order of trust:

  1. the annotated field, normalised — it carries several spellings of the same
     value ("Adults (18-64)", "Adults", "Adults 18-64") plus truncation
     artefacts from the annotation pipeline ("Adults (18", "Infants (0", "Not"),
     all of which have to collapse before anything can match;
  2. a text signal, used ONLY for pregnancy, because that is the one population
     the vocabulary cannot express at all.

The text signal is deliberately not extended to the other populations. Where
the field exists it is the better answer, and inferring "children" from the
word "children" in an abstract would mislabel every paper that merely compares
itself to one.

Nothing here filters. A population mismatch demotes a source; it never removes
it, because the annotation is absent on ~79% of the corpus and a filter would
throw away most of the library to enforce a facet it cannot see.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Set

ADULTS = "adults"
OLDER_ADULTS = "older_adults"
ADOLESCENTS = "adolescents"
CHILDREN = "children"
INFANTS = "infants"
PREGNANCY = "pregnancy"

#: Every population this facet knows, in life-stage order.
POPULATIONS = (INFANTS, CHILDREN, ADOLESCENTS, ADULTS, OLDER_ADULTS, PREGNANCY)

#: Human-readable, for segmenting an answer or offering a refinement.
POPULATION_LABELS = {
    INFANTS: "infants",
    CHILDREN: "children",
    ADOLESCENTS: "adolescents",
    ADULTS: "adults",
    OLDER_ADULTS: "older adults",
    PREGNANCY: "pregnancy",
}

#: Leading text of an annotated value -> population. Matched as a prefix after
#: lowercasing, which is what collapses "Adults (18-64)", "Adults 18-64",
#: "Adults", "Adults with obesity" and the truncated "Adults (18" onto one key.
#: Order matters: "older adults" must be tested before "adults".
_VALUE_PREFIXES = (
    ("older adult", OLDER_ADULTS),
    ("adolescent", ADOLESCENTS),
    ("infant", INFANTS),
    ("child", CHILDREN),
    ("adult", ADULTS),
)

#: Pregnancy in a query or a source's text. Word-bounded so "pregnancy" is not
#: found inside an unrelated token.
_PREGNANCY_RE = re.compile(
    r"\b(pregnan\w*|gestation\w*|maternal|antenatal|prenatal|lactat\w*|"
    r"breastfeed\w*|postpartum)\b",
    re.IGNORECASE,
)

#: Population named in a QUESTION. Broader than the annotation prefixes because
#: people write "in adults", "for the elderly", "in kids".
_QUERY_PATTERNS = (
    (OLDER_ADULTS, re.compile(r"\b(older adults?|elderly|seniors?|65\+|geriatric)\b", re.I)),
    (ADOLESCENTS, re.compile(r"\b(adolescents?|teenagers?|teens?|youths?)\b", re.I)),
    (INFANTS, re.compile(r"\b(infants?|babies|baby|neonat\w*|newborns?)\b", re.I)),
    (CHILDREN, re.compile(r"\b(children|child|kids?|paediatric|pediatric|schoolchildren)\b", re.I)),
    (ADULTS, re.compile(r"\b(adults?|grown[- ]ups?|men|women)\b", re.I)),
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _from_annotated_value(value: Any) -> Optional[str]:
    text = _norm(value)
    if not text or text.startswith("not") or text == "mixed":
        return None
    for prefix, population in _VALUE_PREFIXES:
        if text.startswith(prefix):
            return population
    return None


def population_of_source(source: Dict[str, Any]) -> Set[str]:
    """Populations a source studied. Empty when nothing says.

    Empty is a real and common answer — most of the corpus is unannotated —
    and callers must treat it as "unknown", never as "not a match".
    """
    if not isinstance(source, dict):
        return set()

    found: Set[str] = set()
    for field in ("population_group", "age_group"):
        population = _from_annotated_value(source.get(field))
        if population:
            found.add(population)

    # Pregnancy only, and only from text: the vocabulary has no value for it.
    haystack = " ".join(
        str(source.get(field) or "") for field in ("title", "abstract", "description")
    )
    if _PREGNANCY_RE.search(haystack):
        found.add(PREGNANCY)
    return found


def population_of_query(question: str) -> Set[str]:
    """Populations a question names. Empty when it does not say."""
    if not question:
        return set()
    found: Set[str] = set()
    if _PREGNANCY_RE.search(question):
        found.add(PREGNANCY)
    for population, pattern in _QUERY_PATTERNS:
        if pattern.search(question):
            found.add(population)
    # "older adults" matches the adults pattern too; the more specific wins.
    if OLDER_ADULTS in found:
        found.discard(ADULTS)
    return found


def population_penalty(query_populations: Set[str], source: Dict[str, Any]) -> int:
    """0 when the source fits the question, higher when it does not.

    An unannotated source scores 0, not a penalty. Ranking it below a labelled
    mismatch would punish sources for the annotation gap rather than for being
    the wrong study.
    """
    if not query_populations:
        return 0
    source_populations = population_of_source(source)
    if not source_populations:
        return 0

    # Pregnancy specialises, so it is checked before any overlap.
    #
    # "Iron status in pregnancy: a cohort" is annotated Adults (18-64) — which
    # it is — so an overlap test alone scores it a perfect match for "iron
    # status in adults" and it keeps the top slot. That is precisely the
    # reported failure. A reader asking about adults in general does not want
    # the pregnancy-specific literature, even though pregnant adults are
    # adults, so a pregnancy source is demoted whenever the question did not
    # ask for it.
    if PREGNANCY in source_populations and PREGNANCY not in query_populations:
        return 2

    if source_populations & query_populations:
        return 0
    return 1


def rerank_for_population(question: str, sources: list) -> list:
    """Stable reorder putting population-appropriate sources first."""
    if not sources:
        return []
    wanted = population_of_query(question)
    if not wanted:
        return list(sources)
    return sorted(sources, key=lambda s: population_penalty(wanted, s))


def group_by_population(sources: Iterable[Dict[str, Any]]) -> Dict[str, list]:
    """Sources bucketed by population, for segmenting an answer.

    Keyed by the values in POPULATIONS plus ``unspecified``. A source studying
    more than one population appears under each, because that is what it is.
    """
    grouped: Dict[str, list] = {p: [] for p in POPULATIONS}
    grouped["unspecified"] = []
    for source in sources or []:
        populations = population_of_source(source)
        if not populations:
            grouped["unspecified"].append(source)
            continue
        for population in populations:
            grouped.setdefault(population, []).append(source)
    return {k: v for k, v in grouped.items() if v}
