"""Required coverage for topics people keep asking about.

Round 2, on cholesterol, twice: the answer should distinguish dietary
cholesterol from blood cholesterol and explain HDL/LDL in plain terms, and it
should cover the things that actually move blood lipids — saturated and
unsaturated fat, refined carbohydrate, fibre, plant sterols and stanols,
physical activity — rather than stopping at dietary cholesterol.

Both are the same shape of problem. The answer is not wrong; it is partial, and
it is partial in a way that misleads, because a reader who asks about
cholesterol and is told only about eggs will go on believing eggs are the
lever. Retrieval cannot fix that: the sources returned for "cholesterol" are
about cholesterol, and the missing material is missing because nobody asked
for it.

So this is a checklist, not a retrieval change. For a handful of topics where
the shape of a complete answer is well established and the cost of a partial
one is a misinformed reader, the prompt is told what a complete answer covers.
It is a prompt input rather than a hard requirement because the sources may not
support every point, and an answer that invents plant sterol evidence to
satisfy a checklist is worse than one that omits it.

Deliberately small. Every entry here is a claim that we know what a complete
answer looks like, which is a claim worth making carefully and only where a
domain reviewer would agree. Adding a topic is a content decision, not a code
one — it wants the same sign-off as any other nutrition guidance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TopicCoverage:
    """What a complete answer on a topic addresses."""

    topic: str
    #: Matched against the question, case-insensitively.
    pattern: "re.Pattern[str]"
    #: Distinctions a reader is likely to conflate, stated plainly.
    distinctions: Tuple[str, ...] = ()
    #: Factors a complete answer should mention where the sources allow, each
    #: paired with the terms that count as having mentioned it. The terms are
    #: explicit rather than derived from the label: "dietary fibre, especially
    #: soluble fibre" is covered by an answer that only says "soluble fibre",
    #: and deriving a keyword from the first words of the label would score
    #: that as missing.
    factors: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


TOPICS: Tuple[TopicCoverage, ...] = (
    TopicCoverage(
        topic="cholesterol",
        pattern=re.compile(
            r"\b(cholesterol|ldl|hdl|blood lipids?|lipid profile|statins?)\b", re.I
        ),
        distinctions=(
            "dietary cholesterol (what is eaten) is not the same as blood "
            "cholesterol (what is measured), and for most people the first has "
            "a modest effect on the second",
            "LDL and HDL, in plain words, and why they are reported separately",
        ),
        factors=(
            (
                "saturated fat, and replacing it with unsaturated fat",
                ("saturated fat", "unsaturated"),
            ),
            (
                "refined carbohydrate and free sugars",
                ("refined carb", "free sugar", "added sugar"),
            ),
            (
                "dietary fibre, especially soluble fibre",
                ("fibre", "fiber"),
            ),
            (
                "plant sterols and stanols",
                ("sterol", "stanol"),
            ),
            (
                "physical activity",
                ("physical activity", "exercise"),
            ),
        ),
    ),
)


def coverage_for(question: str) -> Optional[TopicCoverage]:
    """The coverage checklist for a question, or None for most questions.

    First match wins. The registry is small enough that overlap is a review
    problem rather than something to resolve at runtime.
    """
    if not question:
        return None
    for entry in TOPICS:
        if entry.pattern.search(question):
            return entry
    return None


def coverage_prompt_lines(question: str) -> List[str]:
    """Prompt lines naming what a complete answer covers. Empty for most.

    Phrased as "where the sources support it" throughout: the checklist steers
    what the answer reaches for, and must never become a reason to assert
    something the evidence does not carry.
    """
    entry = coverage_for(question)
    if entry is None:
        return []

    lines = [
        f"- TOPIC COVERAGE ({entry.topic}): readers reported answers on this "
        f"topic as incomplete in ways that mislead. Cover the following where "
        f"the retrieved sources support it, and say plainly when they do not."
    ]
    for distinction in entry.distinctions:
        lines.append(f"  - Distinguish: {distinction}.")
    if entry.factors:
        joined = "; ".join(label for label, _ in entry.factors)
        lines.append(
            f"  - Do not stop at the single most obvious factor. A complete "
            f"answer considers: {joined}."
        )
    return lines


def missing_coverage(question: str, answer: str) -> List[str]:
    """Checklist items a produced answer never mentions.

    For review and measurement, not for gating: an answer may legitimately omit
    a point the sources did not support, and this cannot tell the difference. It
    exists so "are cholesterol answers still partial?" has an answer that is not
    somebody's impression.
    """
    entry = coverage_for(question)
    if entry is None or not answer:
        return []

    text = answer.lower()
    return [
        label
        for label, terms in entry.factors
        if not any(term in text for term in terms)
    ]
