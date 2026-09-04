"""Reading back what was asked, and what people thought of the answers.

`qa_requests` and `qa_feedback` have been written on every question since they
were added, and never read: there was no GET for either table. The data an
expert most needs — the answers somebody marked unhelpful, and what they said
about them — was reachable only from a psql session.

This is the read side. It is deliberately a separate service from `QAService`:
that one is on the answer path and must stay fast, this one runs list queries
for a console and can afford joins.

Every query is bounded, ordered and paged. An expert console that fetches
"all questions" against a table that grows forever is a slow request today and
an outage later.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import func, select

logger = logging.getLogger(__name__)

MAX_PAGE = 200
DEFAULT_PAGE = 50
#: Enough of the answer to recognise it in a list without shipping the whole
#: thing for every row.
PREVIEW_CHARS = 240


def _answer_preview(answer: Any) -> Optional[str]:
    if not isinstance(answer, dict):
        return None
    for key in ("answer", "text", "content", "summary"):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            trimmed = " ".join(value.split())
            return trimmed[:PREVIEW_CHARS]
    return None


class QAReviewService:
    """List and inspect asked questions and the feedback on them."""

    @staticmethod
    def _clamp(limit: Optional[int], offset: Optional[int]) -> Tuple[int, int]:
        try:
            size = int(limit) if limit is not None else DEFAULT_PAGE
        except (TypeError, ValueError):
            size = DEFAULT_PAGE
        try:
            start = int(offset) if offset is not None else 0
        except (TypeError, ValueError):
            start = 0
        return max(1, min(size, MAX_PAGE)), max(0, start)

    async def list_requests(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        user_id: Optional[str] = None,
        member_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        language: Optional[str] = None,
        mode: Optional[str] = None,
        search: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        has_feedback: Optional[bool] = None,
        negative_only: bool = False,
    ) -> Dict[str, Any]:
        """A page of asked questions, newest first."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import QAFeedbackRecord, QARequestRecord

        size, start = self._clamp(limit, offset)

        # Feedback counts are aggregated in a subquery rather than joined in,
        # so a question with three pieces of feedback stays one row and the
        # page size means what it says.
        feedback_counts = (
            select(
                QAFeedbackRecord.request_id.label("request_id"),
                func.count().label("feedback_count"),
                func.count()
                .filter(QAFeedbackRecord.helpfulness == "not_helpful")
                .label("negative_count"),
            )
            .group_by(QAFeedbackRecord.request_id)
            .subquery()
        )

        conditions = []
        if user_id:
            conditions.append(QARequestRecord.user_id == user_id)
        if member_id:
            conditions.append(QARequestRecord.member_id == member_id)
        if correlation_id:
            conditions.append(QARequestRecord.correlation_id == correlation_id)
        if language:
            conditions.append(QARequestRecord.language == language)
        if mode:
            conditions.append(QARequestRecord.mode == mode)
        if since:
            conditions.append(QARequestRecord.created_at >= since)
        if until:
            conditions.append(QARequestRecord.created_at <= until)
        if search:
            conditions.append(QARequestRecord.question.ilike(f"%{search}%"))
        if negative_only:
            conditions.append(feedback_counts.c.negative_count > 0)
        elif has_feedback is True:
            conditions.append(feedback_counts.c.feedback_count > 0)
        elif has_feedback is False:
            conditions.append(feedback_counts.c.feedback_count.is_(None))

        base = select(
            QARequestRecord,
            func.coalesce(feedback_counts.c.feedback_count, 0),
            func.coalesce(feedback_counts.c.negative_count, 0),
        ).outerjoin(
            feedback_counts, feedback_counts.c.request_id == QARequestRecord.id
        )
        for condition in conditions:
            base = base.where(condition)

        async with POSTGRES_ASYNC_SESSION_FACTORY()() as db:
            total = await db.scalar(
                select(func.count()).select_from(base.subquery())
            )
            rows = (
                await db.execute(
                    base.order_by(QARequestRecord.created_at.desc())
                    .limit(size)
                    .offset(start)
                )
            ).all()

        return {
            "total": int(total or 0),
            "limit": size,
            "offset": start,
            "items": [
                self._summary(record, count, negative)
                for record, count, negative in rows
            ],
        }

    async def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """One question, its answers, its sources, and its feedback."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import QAFeedbackRecord, QARequestRecord

        try:
            key = uuid.UUID(str(request_id))
        except (ValueError, AttributeError):
            return None

        async with POSTGRES_ASYNC_SESSION_FACTORY()() as db:
            record = (
                await db.execute(
                    select(QARequestRecord).where(QARequestRecord.id == key)
                )
            ).scalar_one_or_none()
            if record is None:
                return None
            feedback = (
                (
                    await db.execute(
                        select(QAFeedbackRecord)
                        .where(QAFeedbackRecord.request_id == key)
                        .order_by(QAFeedbackRecord.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )

        entries = [self._feedback(row, record.question) for row in feedback]
        detail = self._summary(
            record,
            len(entries),
            sum(1 for e in entries if e["helpfulness"] == "not_helpful"),
        )
        detail.update(
            {
                "primary_answer": record.primary_answer,
                "secondary_answer": record.secondary_answer,
                "dual_strategy": record.dual_strategy,
                "retrieved_article_urns": list(record.retrieved_article_urns or []),
                "pipeline_meta": record.pipeline_meta,
                "rag_enabled": bool(record.rag_enabled),
                "top_k": int(record.top_k or 0),
                "feedback": entries,
            }
        )
        return detail

    async def list_feedback(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        negative_only: bool = False,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """A page of feedback, newest first, with the question it is about."""
        from backend.postgres import POSTGRES_ASYNC_SESSION_FACTORY
        from models.db import QAFeedbackRecord, QARequestRecord

        size, start = self._clamp(limit, offset)

        base = select(QAFeedbackRecord, QARequestRecord.question).outerjoin(
            QARequestRecord, QARequestRecord.id == QAFeedbackRecord.request_id
        )
        if negative_only:
            base = base.where(QAFeedbackRecord.helpfulness == "not_helpful")
        if user_id:
            # Feedback rows only recently gained an identity of their own, so
            # older ones are attributable through the question alone.
            base = base.where(
                (QAFeedbackRecord.user_id == user_id)
                | (QARequestRecord.user_id == user_id)
            )
        if since:
            base = base.where(QAFeedbackRecord.created_at >= since)

        async with POSTGRES_ASYNC_SESSION_FACTORY()() as db:
            total = await db.scalar(
                select(func.count()).select_from(base.subquery())
            )
            rows = (
                await db.execute(
                    base.order_by(QAFeedbackRecord.created_at.desc())
                    .limit(size)
                    .offset(start)
                )
            ).all()

        return {
            "total": int(total or 0),
            "limit": size,
            "offset": start,
            "items": [self._feedback(row, question) for row, question in rows],
        }

    # ----------------------------------------------------------- mapping --
    @staticmethod
    def _summary(record: Any, feedback_count: int, negative_count: int) -> Dict[str, Any]:
        return {
            "request_id": str(record.id),
            "question": record.question,
            "mode": record.mode,
            "model": record.model,
            "language": record.language,
            "expertise_level": record.expertise_level,
            "created_at": record.created_at,
            "user_id": record.user_id,
            "member_id": record.member_id,
            "correlation_id": getattr(record, "correlation_id", None),
            "confidence": record.confidence,
            "articles_consulted": int(record.articles_consulted or 0),
            "cache_hit": bool(record.cache_hit),
            "has_feedback": feedback_count > 0,
            "feedback_count": int(feedback_count or 0),
            "has_negative_feedback": int(negative_count or 0) > 0,
            "answer_preview": _answer_preview(record.primary_answer),
        }

    @staticmethod
    def _feedback(row: Any, question: Optional[str]) -> Dict[str, Any]:
        return {
            "id": str(row.id),
            "request_id": str(row.request_id),
            "question": question,
            "preferred_answer": row.preferred_answer,
            "helpfulness": row.helpfulness,
            "target_answer": row.target_answer or "overall",
            "feedback_mode": row.feedback_mode or "general",
            "reason": row.reason,
            "user_id": getattr(row, "user_id", None),
            "member_id": getattr(row, "member_id", None),
            "correlation_id": getattr(row, "correlation_id", None),
            "created_at": row.created_at,
        }


QA_REVIEW_SERVICE = QAReviewService()
