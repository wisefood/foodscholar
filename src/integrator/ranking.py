"""Scoring a candidate source, out loud.

The plan asks for an explicit rubric stored with the proposal and editable by
a curator — not a hidden prompt. That distinction is the whole design: a
number a model produced for reasons it did not record cannot be argued with,
and a curator who disagrees with it has nowhere to push.

So the score is arithmetic over facts the tools established, every component
is named in the returned breakdown, and the weights live in settings where
the console can change them without a deploy. When a curator drags a proposal
somewhere else, that lands in `expert_rank` beside this rather than over it —
which is how anyone later can ask whether the rubric actually agrees with the
people using it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Licences under which content itself may be ingested, best first. A source
#: we may only point at is worth less than one we may actually read, which is
#: why this dominates the rubric.
LICENCE_SCORES: Dict[str, float] = {
    "CC0": 1.0, "public-domain": 1.0,
    "CC-BY-4.0": 0.95, "CCBY": 0.95, "MIT": 0.95, "Apache-2.0": 0.95,
    "CC-BY-SA-4.0": 0.9, "CCBYSA": 0.9, "GPL-3.0": 0.9,
    "CCBYNC": 0.75, "CCBYNCSA": 0.7,
    # Readable but not redistributable without care; still worth having.
    "publisher-specific-oa": 0.6, "other-oa": 0.55, "unspecified-oa": 0.5,
    # No content, only a pointer.
    "CCBYNCND": 0.35, "Proprietary": 0.15,
}

#: Words in a publisher or URL that mark a source as authoritative. Crude on
#: purpose: it is a prior, not a judgement, and the curator sees the reason.
AUTHORITY_MARKERS: List[Tuple[str, float, str]] = [
    ("who.int", 1.0, "WHO"),
    ("efsa.europa.eu", 1.0, "EFSA"),
    ("fao.org", 1.0, "FAO"),
    ("europa.eu", 0.9, "EU institution"),
    ("ministry", 0.9, "government ministry"),
    ("gov.", 0.9, "government"),
    (".gov", 0.9, "government"),
    ("health", 0.7, "health body"),
    ("nhs.", 0.9, "NHS"),
    (".edu", 0.75, "university"),
    ("ac.", 0.75, "university"),
    ("univ", 0.7, "university"),
    ("institute", 0.6, "institute"),
    ("society", 0.55, "learned society"),
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "licence": 0.40,
    "coverage_gap": 0.25,
    "authority": 0.20,
    "tractability": 0.10,
    "completeness": 0.05,
}


def weights(settings: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The rubric's weights, normalised so the score stays in 0–1.

    Read from settings so the console can retune them without a deploy. A
    weight somebody sets to zero removes that component honestly rather than
    leaving it in the breakdown contributing nothing visible.
    """
    values = dict(DEFAULT_WEIGHTS)
    for key in values:
        raw = (settings or {}).get(f"INTEGRATOR_WEIGHT_{key.upper()}")
        if raw is not None:
            try:
                values[key] = max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    total = sum(values.values()) or 1.0
    return {k: v / total for k, v in values.items()}


def _licence_component(licence: Optional[str]) -> Tuple[float, str]:
    if not licence:
        # Not zero: an undetermined licence is a question, not a refusal, and
        # scoring it zero would bury sources nobody has checked yet.
        return 0.3, "licence undetermined — needs checking before approval"
    score = LICENCE_SCORES.get(licence, 0.4)
    if score >= 0.9:
        return score, f"{licence} — content may be reused freely"
    if score >= 0.5:
        return score, f"{licence} — reusable with conditions"
    return score, f"{licence} — pointer only, content may not be copied"


def _authority_component(*args: Optional[str]) -> Tuple[float, str]:
    haystack = " ".join(a.lower() for a in args if a)
    best, why = 0.35, "no recognised issuing body"
    for marker, score, label in AUTHORITY_MARKERS:
        if marker in haystack and score > best:
            best, why = score, label
    return best, why


def _coverage_component(existing: Optional[int]) -> Tuple[float, str]:
    """A gap is worth more than a fourth copy of something we have."""
    if existing is None:
        return 0.5, "coverage not checked"
    if existing == 0:
        return 1.0, "nothing like it in the catalog — fills a gap"
    if existing <= 2:
        return 0.6, f"{existing} similar already held"
    return 0.25, f"{existing} similar already held — well covered"


def _tractability_component(attributes: Dict[str, Any], url: Optional[str]) -> Tuple[float, str]:
    """How much work it would be to actually ingest."""
    kind = str(attributes.get("Type") or attributes.get("Format") or "").lower()
    pages = attributes.get("Pages") or attributes.get("Number of Pages")
    if "web" in kind or "html" in kind or (url or "").endswith((".htm", ".html")):
        return 0.9, "web page — straightforward to read"
    if "spreadsheet" in kind or "csv" in kind or "xls" in kind:
        return 0.85, "spreadsheet — structured already"
    if "printed" in kind or "book" in kind:
        return 0.2, "printed only — no machine-readable copy"
    if "pdf" in kind or (url or "").lower().endswith(".pdf"):
        try:
            count = int(str(pages).strip())
        except (TypeError, ValueError):
            return 0.7, "PDF — extractable"
        if count > 300:
            return 0.45, f"PDF, {count} pages — a long extraction"
        return 0.75, f"PDF, {count} pages"
    return 0.55, "format unknown"


def _completeness_component(proposal: Dict[str, Any]) -> Tuple[float, str]:
    """Whether we know enough about it to act."""
    known = [k for k in ("source_url", "country", "language", "title")
             if proposal.get(k)]
    ratio = len(known) / 4
    if ratio == 1:
        return 1.0, "fully described"
    return ratio, f"missing {4 - len(known)} of country, language, title, URL"


def score(proposal: Dict[str, Any], *, existing_similar: Optional[int] = None,
          settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one candidate, and say why.

    Returns the number, the weights used, and a line per component. The
    rationale is prose a curator reads on the card — the point is that they
    can disagree with a specific clause rather than with a number.
    """
    w = weights(settings)
    attributes = proposal.get("metadata", {}).get("attributes") or {}

    components: Dict[str, Tuple[float, str]] = {
        "licence": _licence_component(proposal.get("licence")),
        "coverage_gap": _coverage_component(existing_similar),
        "authority": _authority_component(
            proposal.get("source_url"), attributes.get("Publisher"),
            proposal.get("title")),
        "tractability": _tractability_component(attributes, proposal.get("source_url")),
        "completeness": _completeness_component(proposal),
    }

    total = sum(w[name] * value for name, (value, _) in components.items())
    breakdown = [
        {"component": name, "score": round(value, 3),
         "weight": round(w[name], 3), "why": why}
        for name, (value, why) in components.items()
    ]
    # Ordered by what actually moved the number, so the first clause a curator
    # reads is the one that decided it.
    breakdown.sort(key=lambda row: row["weight"] * row["score"], reverse=True)
    rationale = "; ".join(row["why"] for row in breakdown[:3])

    return {
        "score": round(total, 3),
        "rationale": rationale,
        "breakdown": breakdown,
        "weights": w,
    }
