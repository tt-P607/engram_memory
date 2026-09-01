"""Engram Memory vNext 固定领域枚举。"""

from __future__ import annotations

from enum import StrEnum


class MemoryStatus(StrEnum):
    """正式记忆状态。"""

    ACTIVE = "ACTIVE"
    MERGED = "MERGED"
    TOMBSTONED = "TOMBSTONED"


class MemoryKind(StrEnum):
    """记忆语义类型。"""

    EVENT = "EVENT"
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    RELATIONSHIP = "RELATIONSHIP"
    COMMITMENT = "COMMITMENT"
    PERSONAL_STATE = "PERSONAL_STATE"
    SOCIAL_PATTERN = "SOCIAL_PATTERN"


class ConfidenceLevel(StrEnum):
    """当前版本的认知置信等级。"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class StabilityLevel(StrEnum):
    """当前记忆认知稳定等级。"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class SalienceLevel(StrEnum):
    """记忆深刻度等级。"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class EventTimePrecision(StrEnum):
    """事件时间精度。"""

    EXACT = "EXACT"
    DAY = "DAY"
    APPROXIMATE = "APPROXIMATE"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class EventTimeOrigin(StrEnum):
    """事件时间来源。"""

    SOURCE_EXPLICIT = "SOURCE_EXPLICIT"
    SOURCE_RELATIVE_RESOLVED = "SOURCE_RELATIVE_RESOLVED"
    RECONSTRUCTED_FROM_EVIDENCE = "RECONSTRUCTED_FROM_EVIDENCE"
    LLM_INFERRED = "LLM_INFERRED"
    UNKNOWN = "UNKNOWN"


class RevisionChangeReason(StrEnum):
    """记忆版本变化原因。"""

    INITIAL = "INITIAL"
    DEVELOPMENT = "DEVELOPMENT"
    CORRECTION = "CORRECTION"
    CLARIFICATION = "CLARIFICATION"
    NEW_EVIDENCE = "NEW_EVIDENCE"
    EXPLICIT_CORRECTION = "EXPLICIT_CORRECTION"
    ADMIN_CORRECTION = "ADMIN_CORRECTION"
    MIGRATION_REWRITE = "MIGRATION_REWRITE"


class SubjectKind(StrEnum):
    """记忆主体类型。"""

    PERSON = "PERSON"
    GROUP = "GROUP"
    EVENT = "EVENT"
    TOPIC = "TOPIC"
    SELF = "SELF"
    WORLD = "WORLD"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ParticipantKind(StrEnum):
    """记忆参与者类型。"""

    PERSON = "PERSON"
    SELF = "SELF"
    OTHER = "OTHER"


class ParticipantRole(StrEnum):
    """记忆参与者角色。"""

    INITIATOR = "INITIATOR"
    PARTICIPANT = "PARTICIPANT"
    RECIPIENT = "RECIPIENT"
    OBSERVER = "OBSERVER"
    MENTIONED = "MENTIONED"


class EvidenceSourceType(StrEnum):
    """证据来源类型。"""

    MESSAGE_SET = "MESSAGE_SET"
    ACTOR_WRITE = "ACTOR_WRITE"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    LEGACY_RECORD = "LEGACY_RECORD"
    ADMIN = "ADMIN"
    EXTERNAL = "EXTERNAL"


class ClaimBasis(StrEnum):
    """证据所支持主张的形成基础。"""

    DIRECT_STATEMENT = "DIRECT_STATEMENT"
    OBSERVED_BEHAVIOR = "OBSERVED_BEHAVIOR"
    THIRD_PARTY_STATEMENT = "THIRD_PARTY_STATEMENT"
    BOT_INFERENCE = "BOT_INFERENCE"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    EXPLICIT_MEMORY_WRITE = "EXPLICIT_MEMORY_WRITE"
    LEGACY_IMPORT = "LEGACY_IMPORT"
    ADMIN_ASSERTION = "ADMIN_ASSERTION"


class ProvenanceQuality(StrEnum):
    """证据来源追踪质量。"""

    EXACT = "EXACT"
    HIGH = "HIGH"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class EvidenceRole(StrEnum):
    """证据与记忆版本之间的作用。"""

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    CONTEXT = "CONTEXT"
    CORRECTION_SOURCE = "CORRECTION_SOURCE"


class RelationType(StrEnum):
    """正式记忆关系类型。"""

    CONTINUES = "CONTINUES"
    CAUSES = "CAUSES"
    REINFORCES = "REINFORCES"
    CONTRADICTS = "CONTRADICTS"
    RELATED_TO = "RELATED_TO"
    SUPERSEDES = "SUPERSEDES"
    MERGED_INTO = "MERGED_INTO"


class CandidateStatus(StrEnum):
    """候选素材处理状态。"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DEFERRED = "DEFERRED"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"


class CandidateActionType(StrEnum):
    """候选素材整理动作。"""

    CREATE_NEW = "CREATE_NEW"
    REINFORCE = "REINFORCE"
    REVISE = "REVISE"
    MERGE = "MERGE"
    RELATE = "RELATE"
    IGNORE = "IGNORE"
    DEFER = "DEFER"


class CandidateActionTargetRole(StrEnum):
    """候选动作中的记忆角色。"""

    SOURCE = "SOURCE"
    TARGET = "TARGET"
    CANONICAL = "CANONICAL"
    RESULT = "RESULT"


class SleepTriggerType(StrEnum):
    """睡眠整理会话触发类型。"""

    DAILY = "DAILY"
    PRESSURE = "PRESSURE"
    MANUAL = "MANUAL"
    RECOVERY = "RECOVERY"


class SleepSessionStatus(StrEnum):
    """睡眠整理会话状态。"""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class SleepCandidateOutcome(StrEnum):
    """睡眠会话处理候选素材的结果。"""

    RESOLVED = "RESOLVED"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"


class MemoryEventType(StrEnum):
    """记忆追加式事件类型。"""

    CREATED = "CREATED"
    REINFORCED = "REINFORCED"
    REVISED = "REVISED"
    ANCHOR_CHANGED = "ANCHOR_CHANGED"
    RELATED = "RELATED"
    RELATION_RETRACTED = "RELATION_RETRACTED"
    MERGED = "MERGED"
    RECALLED = "RECALLED"
    READ = "READ"
    FLASHBACK_EXPOSED = "FLASHBACK_EXPOSED"
    TOMBSTONED = "TOMBSTONED"
    RESTORED = "RESTORED"


class ActorType(StrEnum):
    """领域动作执行主体。"""

    ACTOR = "ACTOR"
    SLEEP_AGENT = "SLEEP_AGENT"
    SYSTEM = "SYSTEM"
    ADMIN = "ADMIN"
    MIGRATION = "MIGRATION"


class RetrievalEntryType(StrEnum):
    """可重建检索入口类型。"""

    ANCHOR = "ANCHOR"
    CURRENT_REVISION = "CURRENT_REVISION"
    HISTORICAL_REVISION = "HISTORICAL_REVISION"
    TAG = "TAG"
    GENERATED_CUE = "GENERATED_CUE"


class OutboxObjectType(StrEnum):
    """向量 Outbox 可处理对象类型。"""

    RETRIEVAL_ENTRY = "RETRIEVAL_ENTRY"


class OutboxOperation(StrEnum):
    """向量 Outbox 操作。"""

    UPSERT = "UPSERT"
    DELETE = "DELETE"


class OutboxStatus(StrEnum):
    """向量 Outbox 处理状态。"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class VectorIndexStatus(StrEnum):
    """向量索引版本状态。"""

    BUILDING = "BUILDING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    FAILED = "FAILED"