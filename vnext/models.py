"""Engram Memory vNext 规范数据与派生检索数据 ORM 模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    func,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from .enums import (
    ActorType,
    CandidateActionTargetRole,
    CandidateActionType,
    CandidateStatus,
    ClaimBasis,
    ConfidenceLevel,
    EventTimeOrigin,
    EventTimePrecision,
    EvidenceRole,
    EvidenceSourceType,
    MemoryEventType,
    MemoryKind,
    MemoryStatus,
    OutboxObjectType,
    OutboxOperation,
    OutboxStatus,
    ParticipantKind,
    ParticipantRole,
    ProvenanceQuality,
    RelationType,
    RetrievalEntryType,
    RevisionChangeReason,
    SalienceLevel,
    SleepCandidateOutcome,
    SleepSessionStatus,
    SleepTriggerType,
    StabilityLevel,
    SubjectKind,
    VectorIndexStatus,
)


class Base(DeclarativeBase):
    """vNext 独立 SQLAlchemy 声明基类。"""


class UTCDateTime(TypeDecorator[datetime]):
    """以 ISO-8601 UTC 文本持久化带时区时间。"""

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> str | None:
        """拒绝无时区时间并统一转换为 UTC。"""
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Engram vNext 时间必须包含时区")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def process_result_value(self, value: str | None, dialect: Any) -> datetime | None:
        """将数据库 UTC 文本恢复为带时区时间。"""
        del dialect
        if value is None:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _enum(enum_type: type[Any], name: str) -> SqlEnum[Any]:
    """构造带数据库 CHECK 约束的字符串枚举列类型。"""
    return SqlEnum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda values: [item.value for item in values],
    )


class SchemaVersionModel(Base):
    """记录 vNext Schema 版本，不承载认知数据。"""

    __tablename__ = "engram_vnext_schema_version"

    schema_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class MemoryModel(Base):
    """正式记忆身份，不保存正文或认知评估副本。"""

    __tablename__ = "engram_vnext_memory"

    memory_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[MemoryStatus] = mapped_column(
        _enum(MemoryStatus, "engram_vnext_memory_status"), nullable=False
    )
    anchor_title: Mapped[str] = mapped_column(Text, nullable=False)
    current_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "engram_vnext_memory_revision.revision_id",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=False,
    )
    current_assessment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "engram_vnext_memory_assessment.assessment_id",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_by_type: Mapped[ActorType] = mapped_column(
        _enum(ActorType, "engram_vnext_memory_creator_type"), nullable=False
    )
    last_experienced_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        Index("idx_engram_vnext_memory_status", "status"),
        Index("idx_engram_vnext_memory_experienced", "last_experienced_at"),
    )


class MemoryRevisionModel(Base):
    """正式记忆不可变语义版本。"""

    __tablename__ = "engram_vnext_memory_revision"

    revision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_kind: Mapped[MemoryKind] = mapped_column(
        _enum(MemoryKind, "engram_vnext_memory_kind"), nullable=False
    )
    confidence: Mapped[ConfidenceLevel] = mapped_column(
        _enum(ConfidenceLevel, "engram_vnext_confidence"), nullable=False
    )
    confidence_reason: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    event_start_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    event_end_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    event_time_precision: Mapped[EventTimePrecision] = mapped_column(
        _enum(EventTimePrecision, "engram_vnext_event_time_precision"), nullable=False
    )
    event_time_origin: Mapped[EventTimeOrigin] = mapped_column(
        _enum(EventTimeOrigin, "engram_vnext_event_time_origin"), nullable=False
    )
    change_reason: Mapped[RevisionChangeReason] = mapped_column(
        _enum(RevisionChangeReason, "engram_vnext_revision_change_reason"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_by_type: Mapped[ActorType] = mapped_column(
        _enum(ActorType, "engram_vnext_revision_creator_type"), nullable=False
    )
    created_by_ref: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("memory_id", "revision_no", name="uq_engram_vnext_revision_no"),
        CheckConstraint("revision_no >= 1", name="ck_engram_vnext_revision_positive"),
        CheckConstraint(
            "event_end_at IS NULL OR event_start_at IS NOT NULL",
            name="ck_engram_vnext_revision_event_range_start",
        ),
        Index("idx_engram_vnext_revision_memory", "memory_id"),
        Index("idx_engram_vnext_revision_kind", "memory_kind"),
    )


class MemoryRevisionSubjectModel(Base):
    """每个记忆版本唯一的主体。"""

    __tablename__ = "engram_vnext_memory_revision_subject"

    revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id"), primary_key=True
    )
    subject_kind: Mapped[SubjectKind] = mapped_column(
        _enum(SubjectKind, "engram_vnext_subject_kind"), nullable=False
    )
    person_id: Mapped[str | None] = mapped_column(Text)
    subject_key: Mapped[str | None] = mapped_column(Text)
    subject_label: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "subject_kind != 'PERSON' OR person_id IS NOT NULL",
            name="ck_engram_vnext_person_subject_id",
        ),
        Index("idx_engram_vnext_subject_person", "person_id"),
    )


class MemoryRevisionParticipantModel(Base):
    """记忆版本参与者。"""

    __tablename__ = "engram_vnext_memory_revision_participant"

    participant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id"), nullable=False
    )
    participant_kind: Mapped[ParticipantKind] = mapped_column(
        _enum(ParticipantKind, "engram_vnext_participant_kind"), nullable=False
    )
    person_id: Mapped[str | None] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    role: Mapped[ParticipantRole] = mapped_column(
        _enum(ParticipantRole, "engram_vnext_participant_role"), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "participant_kind != 'PERSON' OR person_id IS NOT NULL",
            name="ck_engram_vnext_person_participant_id",
        ),
        Index("idx_engram_vnext_participant_revision", "revision_id"),
        Index("idx_engram_vnext_participant_person", "person_id"),
    )


class EvidenceModel(Base):
    """正式认知或候选素材的来源证据。"""

    __tablename__ = "engram_vnext_evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[EvidenceSourceType] = mapped_column(
        _enum(EvidenceSourceType, "engram_vnext_evidence_source_type"), nullable=False
    )
    claim_basis: Mapped[ClaimBasis] = mapped_column(
        _enum(ClaimBasis, "engram_vnext_claim_basis"), nullable=False
    )
    provenance_quality: Mapped[ProvenanceQuality] = mapped_column(
        _enum(ProvenanceQuality, "engram_vnext_provenance_quality"), nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class EvidenceMessageLinkModel(Base):
    """证据到核心消息的有序软引用。"""

    __tablename__ = "engram_vnext_evidence_message_link"

    evidence_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_evidence.evidence_id"), primary_key=True
    )
    message_id: Mapped[str] = mapped_column(Text, primary_key=True)
    stream_id: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("ordinal >= 0", name="ck_engram_vnext_evidence_ordinal"),
        Index("idx_engram_vnext_evidence_message_stream", "stream_id"),
    )


class RevisionEvidenceModel(Base):
    """记忆版本与证据的多对多关系。"""

    __tablename__ = "engram_vnext_revision_evidence"

    revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_evidence.evidence_id"), primary_key=True
    )
    evidence_role: Mapped[EvidenceRole] = mapped_column(
        _enum(EvidenceRole, "engram_vnext_evidence_role"), primary_key=True
    )
    linked_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class MemoryAssessmentModel(Base):
    """正式记忆不可变认知评估。"""

    __tablename__ = "engram_vnext_memory_assessment"

    assessment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    based_on_revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id"), nullable=False
    )
    stability: Mapped[StabilityLevel] = mapped_column(
        _enum(StabilityLevel, "engram_vnext_stability"), nullable=False
    )
    salience: Mapped[SalienceLevel] = mapped_column(
        _enum(SalienceLevel, "engram_vnext_salience"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_by_type: Mapped[ActorType] = mapped_column(
        _enum(ActorType, "engram_vnext_assessment_creator_type"), nullable=False
    )

    __table_args__ = (Index("idx_engram_vnext_assessment_memory", "memory_id"),)


class MemoryRelationModel(Base):
    """具有固定方向语义且可撤销的记忆关系。"""

    __tablename__ = "engram_vnext_memory_relation"

    relation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    target_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    relation_type: Mapped[RelationType] = mapped_column(
        _enum(RelationType, "engram_vnext_relation_type"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_by_type: Mapped[ActorType] = mapped_column(
        _enum(ActorType, "engram_vnext_relation_creator_type"), nullable=False
    )
    retracted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    retract_reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "source_memory_id != target_memory_id", name="ck_engram_vnext_relation_not_self"
        ),
        Index("idx_engram_vnext_relation_source", "source_memory_id"),
        Index("idx_engram_vnext_relation_target", "target_memory_id"),
        Index(
            "uq_engram_vnext_relation_active_directed",
            "source_memory_id",
            "target_memory_id",
            "relation_type",
            unique=True,
            sqlite_where=text("retracted_at IS NULL"),
            postgresql_where=text("retracted_at IS NULL"),
        ),
        Index(
            "uq_engram_vnext_relation_active_symmetric",
            func.min(source_memory_id, target_memory_id),
            func.max(source_memory_id, target_memory_id),
            "relation_type",
            unique=True,
            sqlite_where=text(
                "retracted_at IS NULL AND relation_type IN ('CONTRADICTS', 'RELATED_TO')"
            ),
            postgresql_where=text(
                "retracted_at IS NULL AND relation_type IN ('CONTRADICTS', 'RELATED_TO')"
            ),
        ),
    )


class CandidateModel(Base):
    """尚未成为正式记忆的全局候选素材。"""

    __tablename__ = "engram_vnext_candidate"

    candidate_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[CandidateStatus] = mapped_column(
        _enum(CandidateStatus, "engram_vnext_candidate_status"), nullable=False
    )
    rough_title: Mapped[str] = mapped_column(Text, nullable=False)
    rough_content: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_kind: Mapped[MemoryKind | None] = mapped_column(
        _enum(MemoryKind, "engram_vnext_candidate_kind")
    )
    confidence_hint: Mapped[ConfidenceLevel | None] = mapped_column(
        _enum(ConfidenceLevel, "engram_vnext_candidate_confidence")
    )
    salience_hint: Mapped[SalienceLevel | None] = mapped_column(
        _enum(SalienceLevel, "engram_vnext_candidate_salience")
    )
    retention_reason: Mapped[str] = mapped_column(Text, nullable=False)
    uncertainty_note: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    processing_session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id")
    )
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("idx_engram_vnext_candidate_status", "status", "created_at"),)


class CandidateEvidenceModel(Base):
    """候选素材与证据关系。"""

    __tablename__ = "engram_vnext_candidate_evidence"

    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_evidence.evidence_id"), primary_key=True
    )


class CandidateSubjectModel(Base):
    """候选素材的初步主体判断。"""

    __tablename__ = "engram_vnext_candidate_subject"

    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), primary_key=True
    )
    subject_kind: Mapped[SubjectKind] = mapped_column(
        _enum(SubjectKind, "engram_vnext_candidate_subject_kind"), nullable=False
    )
    person_id: Mapped[str | None] = mapped_column(Text)
    subject_key: Mapped[str | None] = mapped_column(Text)
    subject_label: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "subject_kind != 'PERSON' OR person_id IS NOT NULL",
            name="ck_engram_vnext_candidate_person_subject_id",
        ),
    )


class CandidateParticipantModel(Base):
    """候选素材的初步参与者判断。"""

    __tablename__ = "engram_vnext_candidate_participant"

    participant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), nullable=False
    )
    participant_kind: Mapped[ParticipantKind] = mapped_column(
        _enum(ParticipantKind, "engram_vnext_candidate_participant_kind"), nullable=False
    )
    person_id: Mapped[str | None] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    role: Mapped[ParticipantRole] = mapped_column(
        _enum(ParticipantRole, "engram_vnext_candidate_participant_role"), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "participant_kind != 'PERSON' OR person_id IS NOT NULL",
            name="ck_engram_vnext_candidate_person_participant_id",
        ),
        Index("idx_engram_vnext_candidate_participant", "candidate_id"),
    )


class CandidateEncoderCursorModel(Base):
    """按聊天流保存经历编码器复合游标。"""

    __tablename__ = "engram_vnext_candidate_encoder_cursor"

    stream_id: Mapped[str] = mapped_column(Text, primary_key=True)
    last_processed_message_time: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_processed_message_id: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class SleepSessionModel(Base):
    """每次睡眠整理运行的审计会话。"""

    __tablename__ = "engram_vnext_sleep_session"

    sleep_session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trigger_type: Mapped[SleepTriggerType] = mapped_column(
        _enum(SleepTriggerType, "engram_vnext_sleep_trigger_type"), nullable=False
    )
    status: Mapped[SleepSessionStatus] = mapped_column(
        _enum(SleepSessionStatus, "engram_vnext_sleep_session_status"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("candidate_count >= 0", name="ck_engram_vnext_sleep_candidate_count"),
        Index("idx_engram_vnext_sleep_status", "status", "started_at"),
    )


class CandidateActionModel(Base):
    """睡眠整理对候选素材产生的追加式动作。"""

    __tablename__ = "engram_vnext_candidate_action"

    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), nullable=False
    )
    sleep_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id"), nullable=False
    )
    action_type: Mapped[CandidateActionType] = mapped_column(
        _enum(CandidateActionType, "engram_vnext_candidate_action_type"), nullable=False
    )
    result_revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id")
    )
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (Index("idx_engram_vnext_action_candidate", "candidate_id"),)


class SleepActionOperationModel(Base):
    """睡眠动作的不可变意图与可恢复执行状态。"""

    __tablename__ = "engram_vnext_sleep_action_operation"

    operation_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), nullable=False
    )
    sleep_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id"), nullable=False
    )
    action_type: Mapped[CandidateActionType] = mapped_column(
        _enum(CandidateActionType, "engram_vnext_sleep_operation_action_type"), nullable=False
    )
    intent_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PREPARED', 'EXECUTING', 'COMPLETED', 'FAILED')",
            name="ck_engram_vnext_sleep_operation_status",
        ),
        Index(
            "idx_engram_vnext_sleep_operation_resume",
            "sleep_session_id",
            "status",
            "updated_at",
        ),
        Index("idx_engram_vnext_sleep_operation_candidate", "candidate_id", "created_at"),
    )


class SleepActionPlanModel(Base):
    """候选动作计划及其恢复游标。"""

    __tablename__ = "engram_vnext_sleep_action_plan"

    plan_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), nullable=False
    )
    sleep_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id"), nullable=False
    )
    intents_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    operation_keys_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    action_count: Mapped[int] = mapped_column(Integer, nullable=False)
    next_action_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    target_memory_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PREPARED', 'EXECUTING', 'COMPLETED', 'FAILED')",
            name="ck_engram_vnext_sleep_plan_status",
        ),
        CheckConstraint(
            "action_count >= 0 AND next_action_index >= 0 AND next_action_index <= action_count",
            name="ck_engram_vnext_sleep_plan_cursor",
        ),
        Index(
            "idx_engram_vnext_sleep_plan_resume",
            "sleep_session_id",
            "status",
            "updated_at",
        ),
        Index("idx_engram_vnext_sleep_plan_candidate", "candidate_id", "created_at"),
    )


class CandidateActionTargetModel(Base):
    """候选整理动作涉及的正式记忆。"""

    __tablename__ = "engram_vnext_candidate_action_target"

    action_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate_action.action_id"), primary_key=True
    )
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), primary_key=True
    )
    target_role: Mapped[CandidateActionTargetRole] = mapped_column(
        _enum(CandidateActionTargetRole, "engram_vnext_candidate_target_role"), primary_key=True
    )


class SleepSessionCandidateModel(Base):
    """睡眠会话对候选素材的认领与释放记录。"""

    __tablename__ = "engram_vnext_sleep_session_candidate"

    sleep_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id"), primary_key=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_candidate.candidate_id"), primary_key=True
    )
    claimed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    outcome: Mapped[SleepCandidateOutcome | None] = mapped_column(
        _enum(SleepCandidateOutcome, "engram_vnext_sleep_candidate_outcome")
    )


class MemoryEventModel(Base):
    """记忆不可删除的追加式审计事件。"""

    __tablename__ = "engram_vnext_memory_event"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id")
    )
    event_type: Mapped[MemoryEventType] = mapped_column(
        _enum(MemoryEventType, "engram_vnext_memory_event_type"), nullable=False
    )
    actor_type: Mapped[ActorType] = mapped_column(
        _enum(ActorType, "engram_vnext_memory_event_actor_type"), nullable=False
    )
    actor_ref: Mapped[str | None] = mapped_column(Text)
    stream_id: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    __table_args__ = (
        Index("idx_engram_vnext_event_memory", "memory_id", "occurred_at"),
        Index("idx_engram_vnext_event_type", "event_type", "occurred_at"),
    )


class DomainOperationModel(Base):
    """领域写操作的原子幂等结果。"""

    __tablename__ = "engram_vnext_domain_operation"

    operation_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    __table_args__ = (
        Index("idx_engram_vnext_domain_operation_type", "operation_type", "created_at"),
    )


class PersonPersonaModel(Base):
    """由正式记忆派生的当前人物印象。"""

    __tablename__ = "engram_vnext_person_persona"

    person_id: Mapped[str] = mapped_column(Text, primary_key=True)
    impression_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_sleep_session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id")
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)


class PersonaUpdateLogModel(Base):
    """人物印象更新审计日志。"""

    __tablename__ = "engram_vnext_persona_update_log"

    update_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    person_id: Mapped[str] = mapped_column(Text, nullable=False)
    sleep_session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_sleep_session.sleep_session_id")
    )
    old_content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    new_content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (Index("idx_engram_vnext_persona_log_person", "person_id", "created_at"),)


class PersonaUpdateMemoryModel(Base):
    """人物印象更新所参考记忆的审计关联。"""

    __tablename__ = "engram_vnext_persona_update_memory"

    update_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_persona_update_log.update_id"), primary_key=True
    )
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), primary_key=True
    )


class MemoryRetrievalEntryModel(Base):
    """可完全重建的文本检索入口。"""

    __tablename__ = "engram_vnext_memory_retrieval_entry"

    entry_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory.memory_id"), nullable=False
    )
    revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("engram_vnext_memory_revision.revision_id")
    )
    entry_type: Mapped[RetrievalEntryType] = mapped_column(
        _enum(RetrievalEntryType, "engram_vnext_retrieval_entry_type"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    generator_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        Index("idx_engram_vnext_retrieval_memory", "memory_id"),
        Index("idx_engram_vnext_retrieval_revision", "revision_id"),
        Index("idx_engram_vnext_retrieval_type", "entry_type"),
    )


class VectorOutboxModel(Base):
    """Canonical Commit 后更新向量索引的事务 Outbox。"""

    __tablename__ = "engram_vnext_vector_outbox"

    outbox_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    object_type: Mapped[OutboxObjectType] = mapped_column(
        _enum(OutboxObjectType, "engram_vnext_outbox_object_type"), nullable=False
    )
    object_id: Mapped[str] = mapped_column(String(36), nullable=False)
    operation: Mapped[OutboxOperation] = mapped_column(
        _enum(OutboxOperation, "engram_vnext_outbox_operation"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(Text, nullable=False)
    # The target index is nullable for rows created before the first ACTIVE
    # manifest is bootstrapped.  Workers resolve those rows to the sole
    # current ACTIVE manifest and fence the claim before touching the sink.
    index_id: Mapped[str | None] = mapped_column(String(36))
    status: Mapped[OutboxStatus] = mapped_column(
        _enum(OutboxStatus, "engram_vnext_outbox_status"), nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_token: Mapped[str | None] = mapped_column(String(36))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        CheckConstraint("attempt_count >= 0", name="ck_engram_vnext_outbox_attempt_count"),
        Index("idx_engram_vnext_outbox_status", "status", "updated_at"),
        Index("idx_engram_vnext_outbox_index_status", "index_id", "status", "updated_at"),
    )


class VectorIndexManifestModel(Base):
    """可替换向量索引版本的工程清单。"""

    __tablename__ = "engram_vnext_vector_index_manifest"

    index_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    index_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    flashback_threshold: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[VectorIndexStatus] = mapped_column(
        _enum(VectorIndexStatus, "engram_vnext_vector_index_status"), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "embedding_dimension > 0", name="ck_engram_vnext_manifest_dimension"
        ),
        Index("idx_engram_vnext_manifest_status", "status"),
        Index(
            "uq_engram_vnext_manifest_active",
            "status",
            unique=True,
            sqlite_where=text("status = 'ACTIVE'"),
            postgresql_where=text("status = 'ACTIVE'"),
        ),
    )


IMMUTABLE_MODELS: ClassVar[tuple[type[Base], ...]] = (
    MemoryRevisionModel,
    MemoryRevisionSubjectModel,
    MemoryRevisionParticipantModel,
    EvidenceModel,
    EvidenceMessageLinkModel,
    RevisionEvidenceModel,
    MemoryAssessmentModel,
    CandidateActionModel,
    CandidateActionTargetModel,
    MemoryEventModel,
    PersonaUpdateLogModel,
    PersonaUpdateMemoryModel,
)


def _reject_immutable_change(mapper: Any, connection: Any, target: Base) -> None:
    """阻止不可变历史模型被 ORM 更新或删除。"""
    del mapper, connection
    raise ValueError(f"{type(target).__name__} 是追加式历史记录，不允许修改或删除")


for _model in IMMUTABLE_MODELS:
    event.listen(_model, "before_update", _reject_immutable_change)
    event.listen(_model, "before_delete", _reject_immutable_change)


ALL_MODELS: tuple[type[Base], ...] = tuple(Base.metadata.tables) and (
    SchemaVersionModel,
    MemoryModel,
    MemoryRevisionModel,
    MemoryRevisionSubjectModel,
    MemoryRevisionParticipantModel,
    EvidenceModel,
    EvidenceMessageLinkModel,
    RevisionEvidenceModel,
    MemoryAssessmentModel,
    MemoryRelationModel,
    CandidateModel,
    CandidateEvidenceModel,
    CandidateSubjectModel,
    CandidateParticipantModel,
    CandidateEncoderCursorModel,
    SleepSessionModel,
    CandidateActionModel,
    SleepActionOperationModel,
    SleepActionPlanModel,
    CandidateActionTargetModel,
    SleepSessionCandidateModel,
    MemoryEventModel,
    DomainOperationModel,
    PersonPersonaModel,
    PersonaUpdateLogModel,
    PersonaUpdateMemoryModel,
    MemoryRetrievalEntryModel,
    VectorOutboxModel,
    VectorIndexManifestModel,
)
