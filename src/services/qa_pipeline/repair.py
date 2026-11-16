"""Mapping from an evaluator diagnosis to a targeted repair action.

Each diagnosis repairs differently — a reformulation, a branch flip, or new
sub-questions — and only the affected sub-questions are re-searched.

A ``corpus_gap`` is the honest answer once the corpus has really been
searched. Called after the FIRST round, it is only an impression from one
search and a few snippets (live: "no evidence on oat milk calcium" while the
catalog held three surveys of exactly that), so with search budget left it is
rechecked once, with the searches the evaluator itself suggested in its
notes (``QA_GAP_RECHECK_ENABLED``; an ablation switch).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from config import config
from models.qa import PlannedSubQuestion
from services.qa_pipeline.evaluator import EvaluationResult
from services.qa_pipeline.state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class RepairPlan:
    """Sub-questions to (re-)search in the next round, with an action log."""

    to_search: List[PlannedSubQuestion] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.to_search)


def _uncovered_ids(evaluation: EvaluationResult) -> List[str]:
    ids = []
    for entry in evaluation.per_sub_question:
        if entry.get("covered") is False and entry.get("id"):
            ids.append(str(entry["id"]))
    return ids


# A query the evaluator spelled out in a lead note: 'search "oat milk calcium"'.
_QUOTED = re.compile(r"[\"“”]([^\"“”]{3,80})[\"“”]")
MAX_GAP_RECHECKS = 2


def gap_recheck_enabled() -> bool:
    return str(config.settings.get("QA_GAP_RECHECK_ENABLED", True)).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _recheck_queries(evaluation: EvaluationResult) -> List[str]:
    """Searches the evaluator suggested: reformulations first, then leads."""
    queries: List[str] = []
    for entry in evaluation.reformulated_queries:
        if entry.get("lexical_query"):
            queries.append(str(entry["lexical_query"]))
    for note in evaluation.notes:
        if note.kind != "lead" or not note.text:
            continue
        quoted = _QUOTED.findall(note.text)
        queries.extend(quoted if quoted else [note.text])
    seen: Set[str] = set()
    out = []
    for query in queries:
        key = " ".join(query.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(query.strip())
    return out


def _untried(
    query_options: List[str], branch: str, tried: Set[Tuple[str, str]], current: Optional[str]
) -> Optional[str]:
    current_key = " ".join(str(current or "").lower().split())
    for query in query_options:
        key = " ".join(query.lower().split())
        if key != current_key and (branch or "", key) not in tried:
            return query
    return None


def build_repair_plan(
    state: PipelineState,
    evaluation: EvaluationResult,
    *,
    max_total_sub_questions: int,
) -> RepairPlan:
    """Translate the diagnosis into the next round's search list."""
    plan = RepairPlan()
    verdict = evaluation.verdict
    sub_questions = {sq.id: sq for sq in state.sub_questions()}

    if verdict == "vocabulary_mismatch":
        reformulated_by_id = {
            entry["id"]: entry for entry in evaluation.reformulated_queries
        }
        target_ids = _uncovered_ids(evaluation) or list(reformulated_by_id)
        for sq_id in target_ids:
            sq = sub_questions.get(sq_id)
            entry = reformulated_by_id.get(sq_id)
            if sq is None or entry is None:
                continue
            if entry.get("lexical_query"):
                sq.lexical_query = entry["lexical_query"]
            if entry.get("dense_query"):
                sq.dense_query = entry["dense_query"]
            plan.to_search.append(sq)
            plan.actions.append(
                {
                    "sub_question_id": sq.id,
                    "action": "reformulate",
                    "new_query": sq.lexical_query,
                }
            )

    elif verdict == "wrong_granularity":
        for sq_id in _uncovered_ids(evaluation):
            sq = sub_questions.get(sq_id)
            if sq is None:
                continue
            sq.branch = (
                "guidelines"
                if sq.branch == "articles"
                else "articles"
                if sq.branch == "guidelines"
                else "both"
            )
            plan.to_search.append(sq)
            plan.actions.append(
                {
                    "sub_question_id": sq.id,
                    "action": "switch_branch",
                    "new_query": sq.branch,
                }
            )

    elif verdict == "decomposable_residue":
        room = max(max_total_sub_questions - len(sub_questions), 0)
        for sq in evaluation.new_sub_questions[:room]:
            if sq.id in sub_questions:
                sq.id = f"sq{len(sub_questions) + len(plan.to_search) + 1}"
            state.plan.sub_questions.append(sq)
            plan.to_search.append(sq)
            plan.actions.append(
                {
                    "sub_question_id": sq.id,
                    "action": "add_sub_question",
                    "new_query": sq.text,
                }
            )

    elif verdict == "corpus_gap" and state.round == 0 and gap_recheck_enabled():
        options = _recheck_queries(evaluation)
        tried = set(getattr(state, "queries_tried", set()) or set())
        targets = _uncovered_ids(evaluation) or list(sub_questions)
        for sq_id in targets:
            if len(plan.to_search) >= MAX_GAP_RECHECKS:
                break
            sq = sub_questions.get(sq_id)
            if sq is None:
                continue
            # The evaluator's own suggestion, else the question in plain words
            # (the planner's keyword query is what just came up empty).
            query = _untried(options, sq.branch, tried, sq.lexical_query) or _untried(
                [sq.text], sq.branch, tried, sq.lexical_query
            )
            if not query:
                continue
            sq.lexical_query = query
            sq.dense_query = query
            plan.to_search.append(sq)
            plan.actions.append(
                {"sub_question_id": sq.id, "action": "recheck_gap", "new_query": query}
            )

    # "sufficient" and "needs_user_clarification" repair nothing; nor does a
    # "corpus_gap" after a recheck.
    return plan
