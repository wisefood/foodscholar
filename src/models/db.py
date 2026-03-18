"""SQLAlchemy table models for persistent QA storage."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from backend.postgres import Base


SCHEMA = "foodscholar"


class QARequestRecord(Base):
    """Persisted QA request and its response."""

    __tablename__ = "qa_requests"
    __table_args__ = {"schema": SCHEMA}

    id = Column(UUID(as_uuid=True), primary_key=True)
    question = Column(Text, nullable=False)
    mode = Column(String(16), nullable=False, default="simple")
    model = Column(String(64), nullable=False)
    rag_enabled = Column(Boolean, nullable=False, default=True)
    top_k = Column(Integer, nullable=False, default=5)
    expertise_level = Column(String(16), nullable=False, default="intermediate")
    language = Column(String(8), nullable=False, default="en")

    user_id = Column(String(255), nullable=True)
    member_id = Column(String(255), nullable=True)

    primary_answer = Column(JSONB, nullable=False)
    secondary_answer = Column(JSONB, nullable=True)
    dual_strategy = Column(String(64), nullable=True)

    retrieved_article_urns = Column(JSONB, nullable=True)
    confidence = Column(String(16), nullable=True)
    articles_consulted = Column(Integer, nullable=False, default=0)
    cache_hit = Column(Boolean, nullable=False, default=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class QAFeedbackRecord(Base):
    """User feedback on a QA response."""

    __tablename__ = "qa_feedback"
    __table_args__ = {"schema": SCHEMA}

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id = Column(
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.qa_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    preferred_answer = Column(String(16), nullable=True)
    helpfulness = Column(String(24), nullable=True)
    target_answer = Column(String(16), nullable=False, default="overall")
    feedback_mode = Column(String(24), nullable=False, default="general")
    reason = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class SimpleNutriQuestionsRecord(Base):
    """Persisted 'starter questions' generations."""

    __tablename__ = "simple_nutri_questions"
    __table_args__ = {"schema": SCHEMA}

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    cache_key = Column(String(128), nullable=False, index=True)
    model = Column(String(64), nullable=True)
    count = Column(Integer, nullable=False, default=4)
    questions = Column(JSONB, nullable=False)
    generated_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class TipsOfTheDayRecord(Base):
    """Persisted 'tips of the day' generations (including evidence details)."""

    __tablename__ = "tips_of_the_day"
    __table_args__ = {"schema": SCHEMA}

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    cache_key = Column(String(128), nullable=False, index=True)
    model = Column(String(64), nullable=True)
    tips_count = Column(Integer, nullable=False, default=2)
    did_you_know_count = Column(Integer, nullable=False, default=2)

    tips = Column(JSONB, nullable=False)
    did_you_know = Column(JSONB, nullable=False)
    tips_detail = Column(JSONB, nullable=True)
    did_you_know_detail = Column(JSONB, nullable=True)

    generated_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class GuidelineExtractionRecord(Base):
    """Latest persisted guideline extraction result for an artifact."""

    __tablename__ = "guideline_extractions"
    __table_args__ = {"schema": SCHEMA}

    artifact_id = Column(UUID(as_uuid=True), primary_key=True)
    result_json = Column(JSONB, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
