"""One evidence signal, derived from what the article index already records.

Round 2 produced three separate findings that are the same missing signal:

  * mouse-model studies surfaced as key findings, where a reader was asking a
    question about themselves;
  * a conference-presentation summary ranked above stronger evidence;
  * menopause supplement answers built on preclinical work and claims that a
    reviewer did not consider evidence-based for humans.

Nothing had to be inferred to fix these. The `articles` index carries
`study_type`, `biological_model` and `population_group` on ~15k documents, and
the retriever already reads that index directly. What was missing was a single
ordering over those values, and a label the reader can see.

Two deliberate choices:

`UNKNOWN` sits above preclinical, not below it. Roughly 89% of the corpus has
no `study_type`, so treating "unannotated" as "weak" would bury most of the
library beneath every mouse study that happens to be labelled. Absence of
evidence about the evidence is not evidence of poor evidence.

`is_human` is tracked separately from the grade rather than folded into it.
A well-conducted animal study is not a bad study; it is the wrong study for
"how much fibre should I eat", and the right one for a mechanism question. The
caller decides what to do with that — `human_only` for key findings, a label
everywhere else — instead of the ranking silently deciding for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Best first. The number is an ordering, not a score to do arithmetic on.
GRADE_SYSTEMATIC = 0
GRADE_RCT = 1
GRADE_INTERVENTION = 2
GRADE_COHORT = 3
GRADE_CASE_CONTROL = 4
GRADE_CROSS_SECTIONAL = 5
GRADE_QUALITATIVE = 6
GRADE_REVIEW = 7
GRADE_UNKNOWN = 8
GRADE_PRECLINICAL = 9
GRADE_EDITORIAL = 10

#: `study_type` as the index spells it, lowercased, to a grade.
_STUDY_TYPE_GRADE: Dict[str, int] = {
    "systematic review": GRADE_SYSTEMATIC,
    "meta-analysis": GRADE_SYSTEMATIC,
    "meta analysis": GRADE_SYSTEMATIC,
    "randomized controlled trial": GRADE_RCT,
    "randomised controlled trial": GRADE_RCT,
    "non-randomized intervention": GRADE_INTERVENTION,
    "non-randomised intervention": GRADE_INTERVENTION,
    "observational (cohort)": GRADE_COHORT,
    "observational (case-control)": GRADE_CASE_CONTROL,
    "observational (cross-sectional)": GRADE_CROSS_SECTIONAL,
    "qualitative study": GRADE_QUALITATIVE,
    "narrative review": GRADE_REVIEW,
    "animal study": GRADE_PRECLINICAL,
    "in vitro / cell study": GRADE_PRECLINICAL,
    "in vitro": GRADE_PRECLINICAL,
    "methods / protocol": GRADE_EDITORIAL,
    "not stated": GRADE_UNKNOWN,
    "other": GRADE_UNKNOWN,
}

#: Semantic Scholar publication types that place a document below the study
#: types above regardless of what `study_type` claims. A conference summary is
#: the specific case Round 2 reported.
_WEAK_PUBLICATION_TYPES = {
    "editorial",
    "lettersandcomments",
    "conference",
    "news",
}

#: `biological_model` values meaning the work was not done in humans. The index
#: holds three casings of the in-vitro value, so matching is lowercased.
_NON_HUMAN_MODELS = {"animal", "in vitro", "in vitro / cell study", "plant"}
_HUMAN_MODELS = {"human", "mixed"}

_GRADE_LABELS = {
    GRADE_SYSTEMATIC: "systematic review",
    GRADE_RCT: "randomised controlled trial",
    GRADE_INTERVENTION: "intervention study",
    GRADE_COHORT: "cohort study",
    GRADE_CASE_CONTROL: "case-control study",
    GRADE_CROSS_SECTIONAL: "cross-sectional study",
    GRADE_QUALITATIVE: "qualitative study",
    GRADE_REVIEW: "narrative review",
    GRADE_UNKNOWN: "",
    GRADE_PRECLINICAL: "preclinical",
    GRADE_EDITORIAL: "editorial or conference item",
}


@dataclass(frozen=True)
class EvidenceGrade:
    """How much weight a source's design supports, and what to call it."""

    grade: int
    #: True for human work, False for preclinical, None when not recorded.
    is_human: Optional[bool]
    #: Shown to the reader beside the citation; empty when there is nothing
    #: worth saying.
    label: str

    @property
    def is_preclinical(self) -> bool:
        return self.is_human is False or self.grade == GRADE_PRECLINICAL

    @property
    def is_weak(self) -> bool:
        return self.grade >= GRADE_PRECLINICAL


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _publication_types(source: Dict[str, Any]) -> set:
    raw = source.get("type")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return set()
    return {_norm(v) for v in raw if v}


def grade_source(source: Dict[str, Any]) -> EvidenceGrade:
    """Grade one retrieved article. Never raises; unknown is a valid answer."""
    if not isinstance(source, dict):
        return EvidenceGrade(GRADE_UNKNOWN, None, "")

    model = _norm(source.get("biological_model"))
    if model in _NON_HUMAN_MODELS:
        is_human: Optional[bool] = False
    elif model in _HUMAN_MODELS:
        is_human = True
    else:
        is_human = None

    grade = _STUDY_TYPE_GRADE.get(_norm(source.get("study_type")), GRADE_UNKNOWN)

    # A non-human model overrides an optimistic study_type: an RCT in mice is
    # still preclinical for a reader asking about themselves.
    if is_human is False and grade < GRADE_PRECLINICAL:
        grade = GRADE_PRECLINICAL

    # A conference summary or editorial outranks nothing, whatever it claims.
    if _publication_types(source) & _WEAK_PUBLICATION_TYPES:
        grade = max(grade, GRADE_EDITORIAL)

    label = _GRADE_LABELS.get(grade, "")
    if is_human is False and grade == GRADE_PRECLINICAL:
        # Say which kind, because "preclinical" is not a word most readers use.
        label = "animal study" if model == "animal" else "laboratory study"
    return EvidenceGrade(grade=grade, is_human=is_human, label=label)


def rank_key(source: Dict[str, Any]) -> tuple:
    """Sort key placing better-designed, human evidence first.

    Relevance stays in the key — this reorders within comparable evidence
    rather than replacing the retriever's judgement with a hierarchy. A
    perfectly relevant cohort study should still beat a barely-relevant
    systematic review about something else.
    """
    graded = grade_source(source)
    score = source.get("_score") or source.get("similarity_score") or 0.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return (graded.grade, -score)


def prefer_human(sources: list, limit: Optional[int] = None) -> list:
    """Human evidence first, preclinical only to fill a shortfall.

    For key findings, where Round 2 found mouse studies answering questions
    people asked about themselves. Preclinical work is not discarded — it falls
    to the back, so a question that only has animal evidence still gets an
    answer, and the label says what it is.
    """
    if not sources:
        return []
    human, other = [], []
    for source in sources:
        (other if grade_source(source).is_preclinical else human).append(source)
    ordered = sorted(human, key=rank_key) + sorted(other, key=rank_key)
    return ordered[:limit] if limit else ordered
