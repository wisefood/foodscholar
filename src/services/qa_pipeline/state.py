"""Dataclasses carrying pipeline state between stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.qa import (
    PlannedSubQuestion,
    QAPipelinePlan,
    QARequest,
    QAUserContext,
    ResearchNote,
    RetrievedSource,
)


@dataclass
class EvidenceItem:
    """One retrieved source with its fused and adjusted ranking scores."""

    payload: Dict[str, Any]
    source: RetrievedSource
    sub_question_ids: List[str] = field(default_factory=list)
    rrf_score: float = 0.0
    rrf_norm: float = 0.0
    adjusted_score: float = 0.0
    score_parts: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identity used for deduplication across legs and sub-questions."""
        payload = self.payload
        for k in ("urn", "doi", "id", "_id"):
            value = payload.get(k)
            if isinstance(value, str) and value.strip():
                return f"{payload.get('source_type', 'article')}:{value.strip()}"
        return f"{payload.get('source_type', 'article')}:{id(payload)}"

    @property
    def parent_doc_key(self) -> str:
        """Identity of the parent document, for the per-document diversity cap."""
        payload = self.payload
        if payload.get("source_type") == "guideline":
            for k in ("guide_urn", "urn", "_id"):
                value = payload.get(k)
                if isinstance(value, str) and value.strip():
                    return f"guide:{value.strip()}"
        for k in ("doi", "urn", "_id"):
            value = payload.get(k)
            if isinstance(value, str) and value.strip():
                return f"doc:{value.strip()}"
        return self.key


@dataclass
class PipelineState:
    """Mutable state threaded through one pipeline run."""

    request: QARequest
    request_id: str
    effective_question: str
    user_context: QAUserContext
    effective_model: str
    effective_retriever: str
    plan: Optional[QAPipelinePlan] = None
    round: int = 0
    evidence: List[EvidenceItem] = field(default_factory=list)
    branch_statuses: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    repairs: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[ResearchNote] = field(default_factory=list)
    prior_notes: List[ResearchNote] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(default_factory=dict)

    def sub_questions(self) -> List[PlannedSubQuestion]:
        return list(self.plan.sub_questions) if self.plan else []

    def add_notes(self, notes: List[ResearchNote]) -> List[ResearchNote]:
        """Accumulate notes, skipping duplicates by normalized text."""
        seen = {note.text.strip().lower() for note in self.notes}
        added: List[ResearchNote] = []
        for note in notes:
            text = (note.text or "").strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            self.notes.append(note)
            added.append(note)
        return added

    def evidence_payloads(self) -> List[Dict[str, Any]]:
        return [item.payload for item in self.evidence]

    def retrieved_sources(self) -> List[RetrievedSource]:
        return [item.source for item in self.evidence]

    def pipeline_meta(self) -> Dict[str, Any]:
        """The persisted/observable summary of how this answer was produced."""
        sub_questions = [
            {
                "id": sq.id,
                "text": sq.text,
                "why": sq.why,
                "qtype": sq.qtype,
                "branch": sq.branch,
                "round_added": sq.round_added,
            }
            for sq in self.sub_questions()
        ]
        articles = sum(
            1
            for item in self.evidence
            if item.payload.get("source_type") != "guideline"
        )
        return {
            "mode": "agentic",
            "sub_questions": sub_questions,
            "rounds": self.round + 1,
            "verdicts": self.verdicts,
            "repairs": self.repairs,
            "notes": [note.model_dump() for note in self.notes],
            "evidence_counts": {
                "articles": articles,
                "guidelines": len(self.evidence) - articles,
            },
            "timings_ms": self.timings_ms,
        }
