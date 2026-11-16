"""How a research run is going, round by round: the research log.

After each retrieval round the pipeline asks four cheap questions of what it
just did, without a model call:

- Did it search anything new, or repeat a query it had already run?
- Did the round add anything to the evidence the answer will use, or only
  re-find what it already had?
- Does what it newly kept still talk about the question (shared key terms),
  or has the search wandered off?
- Which sub-questions does the evaluator still see as uncovered?

From those it gives the round a status:

- ``blocked``: every search in the round failed;
- ``stalled``: a repair round that added nothing new to the kept evidence,
  or only re-ran queries already tried;
- ``drifting``: what the round newly kept shares far fewer of the question's
  key terms than the first round's evidence did;
- ``sufficient``: the evaluator judged the evidence enough (a real judgement,
  not the end-of-budget default);
- ``on_track``: otherwise.

Observe-only for now: the status and the log are recorded with the answer
(``pipeline_meta``) and in the logs, and nothing acts on them yet. The
thresholds are first guesses, to be calibrated on the benchmark before any
of this steers the search.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# A round's newly kept sources drift when they share less than this fraction
# of the overlap the first round's sources had with the question.
DRIFT_RATIO = 0.5
# In the first round, kept sources sharing less than this with the question
# have drifted from the start (the plan's searches missed it).
DRIFT_FLOOR = 0.1

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
# Short, common words carry no topic; anything of four letters or more that
# is not in this list counts as a key term.
_STOP = {
    "about", "after", "also", "been", "being", "best", "both", "does", "doing",
    "each", "eating", "every", "from", "good", "have", "healthy", "into", "just",
    "like", "many", "more", "most", "much", "need", "only", "other", "over",
    "same", "should", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "very", "want", "what", "when",
    "where", "which", "while", "will", "with", "would", "your",
}


def query_key(branch: str, text: Optional[str]) -> Tuple[str, str]:
    """A search's identity: its branch and its words, case and spacing aside."""
    return (branch or "", " ".join(str(text or "").lower().split()))


def key_terms(text: Optional[str]) -> Set[str]:
    """Topic words of a text: four letters or more, not a common word.

    A crude stem (drop a plural "s") lets "eggs" meet "egg".
    """
    terms = set()
    for word in _WORD_RE.findall(str(text or "").lower()):
        if word in _STOP:
            continue
        terms.add(word[:-1] if word.endswith("s") and len(word) > 4 else word)
    return terms


def _source_text(payload: Dict[str, Any]) -> str:
    parts = [payload.get("title"), payload.get("rule_text")]
    abstract = payload.get("abstract") or payload.get("description")
    if abstract:
        parts.append(str(abstract)[:400])
    return " ".join(str(p) for p in parts if p)


def overlap_with_question(question_terms: Set[str], payloads: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Mean share of the question's key terms each source mentions; None if no sources."""
    if not question_terms:
        return None
    shares = [
        len(question_terms & key_terms(_source_text(p))) / len(question_terms)
        for p in payloads
    ]
    return round(sum(shares) / len(shares), 3) if shares else None


def round_signals(
    *,
    round_index: int,
    searched: Sequence[Tuple[str, str, str]],
    queries_tried_before: Set[Tuple[str, str]],
    branch_statuses: Sequence[Dict[str, Any]],
    pool_keys_before: Set[str],
    pool_keys: Set[str],
    selected_keys_before: Set[str],
    selected: Sequence[Any],
    question_terms: Set[str],
    per_sub_question: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """The deterministic signals of one round.

    ``searched`` is (sub_question_id, branch, query) for each search run this
    round; ``selected`` the EvidenceItems kept after ranking (with ``key`` and
    ``payload``).
    """
    keys = [query_key(branch, query) for _, branch, query in searched]
    repeated = sorted({q for (_, q) in (k for k in keys if k in queries_tried_before)})
    failed = sum(1 for s in branch_statuses if not s.get("ok", False))
    new_selected = [item for item in selected if item.key not in selected_keys_before]
    uncovered = [
        str(entry.get("id") or entry.get("sub_question_id") or "")
        for entry in per_sub_question
        if entry.get("covered") is False or entry.get("gap")
    ]
    return {
        "round": round_index,
        "searches": len(searched),
        "failed": failed,
        "repeated_queries": repeated,
        "new_in_pool": len(pool_keys - pool_keys_before),
        "new_in_selected": len(new_selected),
        "selected": len(selected),
        "question_overlap_selected": overlap_with_question(
            question_terms, (item.payload for item in selected)
        ),
        "question_overlap_new": overlap_with_question(
            question_terms, (item.payload for item in new_selected)
        ),
        "uncovered_sub_questions": [u for u in uncovered if u],
    }


def assess_round(
    signals: Dict[str, Any],
    *,
    verdict: Optional[str],
    used_llm: bool,
    baseline_overlap: Optional[float],
) -> Tuple[str, str]:
    """A round's status and a one-line reason (see the module docstring)."""
    searches = signals.get("searches", 0)
    if searches and signals.get("failed", 0) >= searches:
        return "blocked", "Every search in this round failed."
    if signals["round"] > 0:
        if searches and len(signals.get("repeated_queries", [])) >= searches:
            return "stalled", "This round only repeated searches already run."
        if signals.get("new_in_selected", 0) == 0:
            return "stalled", "This round added nothing new to the evidence used."
    new_overlap = signals.get("question_overlap_new")
    if new_overlap is not None:
        if signals["round"] == 0 and new_overlap < DRIFT_FLOOR:
            return "drifting", "The evidence found shares almost none of the question's key terms."
        if (
            signals["round"] > 0
            and baseline_overlap
            and new_overlap < baseline_overlap * DRIFT_RATIO
        ):
            return "drifting", "What this round added is further from the question than the first round's evidence."
    if verdict == "sufficient" and used_llm:
        return "sufficient", "The evaluator judged the evidence enough to answer."
    if verdict == "sufficient" and not used_llm:
        return "on_track", "Search budget reached; answering with the best evidence found."
    return "on_track", "New, on-topic evidence; still looking."


def log_entry(
    signals: Dict[str, Any],
    status: str,
    reason: str,
    *,
    searched: Sequence[Tuple[str, str, str]],
    verdict: Optional[str],
    gaps: Sequence[str] = (),
) -> Dict[str, Any]:
    """One research-log line: what was tried, what came of it, what is missing."""
    return {
        "round": signals["round"],
        "status": status,
        "reason": reason,
        "verdict": verdict,
        "tried": [f"{branch}: {query}" for _, branch, query in searched][:12],
        "found_new": signals.get("new_in_pool", 0),
        "kept_new": signals.get("new_in_selected", 0),
        "missing": [g for g in gaps if g][:6],
        "next": None,  # filled in when the round's repair plan is known
    }
