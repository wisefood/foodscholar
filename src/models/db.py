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
    Float,
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
    # The gateway's X-Request-Id for the request that produced this question.
    # `id` above is FoodScholar's own request identifier (the one the UI quotes
    # back when submitting feedback); this is the platform-wide correlation id,
    # and it is how a question asked *inside a FoodChat turn* — which reaches
    # this service without a Keycloak subject — is attributed to a user, by
    # joining the gateway's activity record for the same id.
    correlation_id = Column(String(64), nullable=True, index=True)

    primary_answer = Column(JSONB, nullable=False)
    secondary_answer = Column(JSONB, nullable=True)
    dual_strategy = Column(String(64), nullable=True)

    retrieved_article_urns = Column(JSONB, nullable=True)
    # Agentic pipeline metadata: sub-questions (with rationales), rounds,
    # evaluator verdicts, repairs, research notes, timings. Null on legacy.
    pipeline_meta = Column(JSONB, nullable=True)
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

    # Who is complaining. Previously recoverable only by joining back to
    # qa_requests, and then only when that row had an identity of its own —
    # which left feedback on a chat-originated question attributable to nobody.
    user_id = Column(String(255), nullable=True, index=True)
    member_id = Column(String(255), nullable=True)
    correlation_id = Column(String(64), nullable=True, index=True)

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


class GuidelineEnrichmentRecord(Base):
    """
    Progress of the facet-enrichment pass over one guide's guidelines.

    Keyed by guide rather than by rule: enrichment is driven per guide because
    the guide's context is what every rule under it inherits, and that context
    is resolved once per guide.
    """

    __tablename__ = "guideline_enrichments"
    __table_args__ = {"schema": SCHEMA}

    guide_urn = Column(String(512), primary_key=True)
    status = Column(String(16), nullable=False, default="queued")
    version = Column(Integer, nullable=False, default=1)
    total = Column(Integer, nullable=False, default=0)
    enriched = Column(Integer, nullable=False, default=0)
    skipped_version = Column(Integer, nullable=False, default=0)
    skipped_no_facets = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    context_sources = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
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


# ---------------------------------------------------------------------------
# Source Integrator
#
# The conversational agent that researches candidate sources and integrates
# approved ones into the catalog. Four tables, and the shape of them is the
# design: a session holds the conversation, a proposal is the unit a *person*
# approves, a run is one attempt at integrating an approved proposal, and a
# tool call is the audit — every single thing the agent did, with what it was
# given and what came back.
#
# Nothing here trusts the model. The approval state lives in a column, not in
# a prompt, and `wisefood_mcp.stores.require_approved` reads that column
# before any write tool will run.
# ---------------------------------------------------------------------------


class IntegratorSession(Base):
    """One expert's conversation with the integrator.

    Separate from the FoodChat session tables even though the shape rhymes:
    those are keyed by household member and carry a participant's meal talk;
    this is keyed by Keycloak subject and carries an expert's research. Sharing
    them would put a curator's source hunt in a participant's history.
    """

    __tablename__ = "integrator_sessions"
    __table_args__ = {"schema": SCHEMA}

    id = Column(String(64), primary_key=True)
    user_sub = Column(String(100), nullable=False, index=True)
    title = Column(String(300), nullable=True)
    status = Column(String(16), nullable=False, default="open")
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class IntegratorMessage(Base):
    """One turn. Assistant turns keep the tool calls they made.

    The whole message list is replayed to the model on the next turn, which is
    what "keeps its context" means in practice — so `tool_calls` and
    `tool_call_id` are stored rather than derived: a tool result that cannot
    be matched back to the call that produced it breaks the next replay.
    """

    __tablename__ = "integrator_messages"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    seq = Column(Integer, nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=True)
    tool_calls = Column(JSONB, nullable=True)
    tool_call_id = Column(String(128), nullable=True)
    tool_name = Column(String(64), nullable=True)
    #: What the assistant did to produce this turn, as ReasoningSteps — the
    #: same shape FoodScholar's Q&A already streams, so the console renders
    #: both the same way. Stored rather than streamed-and-forgotten: a curator
    #: reopening a conversation tomorrow needs to see how an answer was
    #: reached, which is most of what makes it checkable.
    steps = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class IntegrationProposal(Base):
    """One candidate source, and everything worked out about it.

    `status` is the wall. It reaches ``approved`` only through the console,
    by a person, and every write tool refuses a proposal that is anything
    else. `licence` and `licence_evidence` travel together on purpose: a
    licence without the quotes behind it is a guess that has been promoted to
    a fact, and this platform has to be able to answer a challenge years later.
    """

    __tablename__ = "integration_proposals"
    __table_args__ = {"schema": SCHEMA}

    id = Column(String(32), primary_key=True)
    session_id = Column(String(64), nullable=True, index=True)
    backlog_id = Column(String(64), nullable=True, index=True)

    kind = Column(String(24), nullable=False)
    title = Column(String(500), nullable=False)
    source_url = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="researching", index=True)

    country = Column(String(120), nullable=True)
    language = Column(String(80), nullable=True)
    population_group = Column(String(160), nullable=True)

    licence = Column(String(64), nullable=True)
    licence_confidence = Column(Float, nullable=True)
    licence_evidence = Column(JSONB, nullable=False, default=list)
    #: Set only when a person approved despite an undetermined or restrictive
    #: licence. Its presence is what makes that approval defensible.
    licence_override_reason = Column(Text, nullable=True)

    #: What the assistant scored it, and what the expert dragged it to. Both,
    #: so a rubric that keeps disagreeing with people is visible as a pattern.
    proposed_rank = Column(Float, nullable=True)
    expert_rank = Column(Integer, nullable=True)
    rationale = Column(Text, nullable=True)
    plan = Column(JSONB, nullable=False, default=list)
    proposal_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    created_by = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(JSONB, nullable=False, default=dict)


class IntegratorToolCall(Base):
    """Every tool the agent ran, whether it worked, and how long it took.

    The audit the provenance chain points at. Kept for read tools too, not
    only writes: "why did it propose this licence" is answered by the research
    and evidence calls that came before, and those are exactly the ones a
    write-only log would have thrown away.
    """

    __tablename__ = "integrator_tool_calls"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=True, index=True)
    proposal_id = Column(String(32), nullable=True, index=True)
    tool = Column(String(64), nullable=False)
    is_write = Column(Boolean, nullable=False, default=False)
    ok = Column(Boolean, nullable=False, default=True)
    arguments = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(JSONB, nullable=True)
    duration_ms = Column(Float, nullable=True)
    actor = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class IntegratorBacklogItem(Base):
    """A candidate source waiting to be looked at.

    Seeded from the project's own source catalogue spreadsheet (219 rows) and
    added to by experts or by the assistant. `external_key` is what makes
    re-seeding idempotent — the spreadsheet will be re-imported, and it must
    not multiply.
    """

    __tablename__ = "integrator_backlog"
    __table_args__ = {"schema": SCHEMA}

    id = Column(String(64), primary_key=True)
    external_key = Column(String(300), nullable=False, unique=True, index=True)
    kind = Column(String(24), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    url = Column(Text, nullable=True)
    country = Column(String(120), nullable=True)
    language = Column(String(80), nullable=True)
    population_group = Column(String(160), nullable=True)
    #: Whatever else the source sheet carried — page counts, CiteScore,
    #: publisher, declared licence. Kept verbatim rather than normalised,
    #: because the columns differ per sheet and the assistant reads them.
    attributes = Column(JSONB, nullable=False, default=dict)
    status = Column(String(16), nullable=False, default="pending", index=True)
    source_sheet = Column(String(80), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class IntegrationRun(Base):
    """One attempt at integrating an approved proposal.

    Separate from the proposal because a run is an *attempt*, and attempts
    fail: a PDF that will not download, an extraction the model gives up on,
    an import that finds nothing to import. Folding the state onto the
    proposal would mean a retry erases the evidence of why the first try
    failed, which is exactly the evidence somebody needs.

    `steps` holds the same shape the chat timeline uses, so the console
    renders a run with the component it already has. `heartbeat_at` is what
    tells a reader the difference between a run that is working and a run
    whose pod died mid-extraction — the two look identical from `status`
    alone, and one of them needs a person.
    """

    __tablename__ = "integration_runs"
    __table_args__ = {"schema": SCHEMA}

    id = Column(String(32), primary_key=True)
    proposal_id = Column(String(32), nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)

    #: queued | running | succeeded | failed | stalled
    status = Column(String(16), nullable=False, default="queued", index=True)
    #: Which pipeline stage it is on, for a progress line that means something.
    stage = Column(String(40), nullable=True)
    steps = Column(JSONB, nullable=False, default=list)
    error = Column(Text, nullable=True)
    #: urn, artifact_uuid, extraction job, import counts — accumulated as the
    #: run goes, so a failure still shows everything that did land.
    result = Column(JSONB, nullable=False, default=dict)
    #: True once anything has been created in the catalog. A failed run that
    #: got this far needs cleaning up by hand, and should say so.
    wrote_anything = Column(Boolean, nullable=False, default=False)

    dry_run = Column(Boolean, nullable=False, default=False)
    started_by = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
