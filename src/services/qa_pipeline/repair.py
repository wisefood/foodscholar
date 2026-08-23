"""Mapping from an evaluator diagnosis to a targeted repair action.

Each diagnosis repairs differently — a reformulation, a branch flip, or new
sub-questions — and only the affected sub-questions are re-searched. A
``corpus_gap`` never retries: the honest move is to answer and disclose.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

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

    # "corpus_gap", "sufficient", and "needs_user_clarification" repair nothing.
    return plan
