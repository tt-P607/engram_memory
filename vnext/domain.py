"""Engram Memory vNext 领域输入与结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .enums import (
    ActorType,
    EvidenceSourceType,
    MemoryKind,
    ParticipantKind,
    SubjectKind,
    RevisionChangeReason,
    RelationType,
    CandidateActionType,
    CandidateStatus,
    CandidateActionTargetRole,
    SleepSessionStatus,
    SleepTriggerType,
    MemoryEventType,
)


@dataclass(frozen=True, slots=True)
class SubjectInput:
    """记忆版本主体输入。"""

    subject_kind: SubjectKind
    person_id: str | None = None
    subject_key: str | None = None
    subject_label: str | None = None

    def validate(self) -> None:
        """验证人物主体具有核心人物标识。"""
        if self.subject_kind is SubjectKind.PERSON and not self.person_id:
            raise ValueError("PERSON 主体必须提供 person_id")


@dataclass(frozen=True, slots=True)
class ParticipantInput:
    """记忆版本参与者输入。"""

    participant_kind: ParticipantKind
    person_id: str | None = None
    label: str | None = None

    def validate(self) -> None:
        """验证人物参与者具有核心人物标识。"""
        if self.participant_kind is ParticipantKind.PERSON and not self.person_id:
            raise ValueError("PERSON 参与者必须提供 person_id")


@dataclass(frozen=True, slots=True)
class EvidenceMessageInput:
    """证据关联的原始消息软引用。"""

    message_id: str
    stream_id: str
    snapshot: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    """正式记忆写入所需的证据输入。"""

    source_type: EvidenceSourceType
    observed_at: datetime
    messages: tuple[EvidenceMessageInput, ...] = ()
    source_ref: str | None = None
    note: str | None = None

    def validate(self) -> None:
        """验证消息证据拥有消息引用且引用不重复。"""
        if self.source_type is EvidenceSourceType.MESSAGE_SET and not self.messages:
            raise ValueError("MESSAGE_SET Evidence 必须关联至少一条消息")
        keys = {(item.stream_id, item.message_id) for item in self.messages}
        if len(keys) != len(self.messages):
            raise ValueError("同一 Evidence 不能重复关联消息")


@dataclass(frozen=True, slots=True)
class CreateMemoryInput:
    """创建正式记忆及初始认知状态的输入。"""

    title: str
    content: str
    memory_kind: MemoryKind
    subject: SubjectInput
    observed_at: datetime
    evidence: tuple[EvidenceInput, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    participants: tuple[ParticipantInput, ...] = ()

    def validate(self) -> None:
        """验证正式记忆创建所需的领域完整性。"""
        required_text = {
            "title": self.title,
            "content": self.content,
        }
        for field_name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if not self.evidence and not self.evidence_ids:
            raise ValueError("正式记忆必须至少关联一份 Evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids 不能重复")
        self.subject.validate()
        for participant in self.participants:
            participant.validate()
        for evidence in self.evidence:
            evidence.validate()


@dataclass(frozen=True, slots=True)
class ReinforceMemoryInput:
    """正式记忆强化输入。"""

    memory_id: str
    based_on_revision_id: str
    reason: str
    evidence: tuple[EvidenceInput, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        """验证强化不缺少目标、原因或新增证据。"""
        if not self.memory_id or not self.based_on_revision_id:
            raise ValueError("强化必须指定 memory_id 与 based_on_revision_id")
        if not self.reason.strip():
            raise ValueError("reinforce reason 不能为空")
        if not self.evidence and not self.evidence_ids:
            raise ValueError("强化必须提供新 Evidence 或已有 evidence_ids")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids 不能重复")
        for evidence in self.evidence:
            evidence.validate()


@dataclass(frozen=True, slots=True)
class ReviseMemoryInput:
    """正式记忆线性修订输入。"""

    memory_id: str
    based_on_revision_id: str
    title: str
    content: str
    memory_kind: MemoryKind
    subject: SubjectInput
    observed_at: datetime
    change_reason: RevisionChangeReason
    evidence: tuple[EvidenceInput, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    participants: tuple[ParticipantInput, ...] = ()

    def validate(self) -> None:
        """验证线性修订所需的语义内容与证据。"""
        if not self.memory_id or not self.based_on_revision_id:
            raise ValueError("修订必须指定 memory_id 与 based_on_revision_id")
        for field_name, value in {
            "title": self.title,
            "content": self.content,
        }.items():
            if not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if self.change_reason is RevisionChangeReason.INITIAL:
            raise ValueError("修订不能使用 INITIAL change_reason")
        if not self.evidence and not self.evidence_ids:
            raise ValueError("修订必须提供 Evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids 不能重复")
        self.subject.validate()
        for participant in self.participants:
            participant.validate()
        for evidence in self.evidence:
            evidence.validate()


@dataclass(frozen=True, slots=True)
class RelateMemoryInput:
    """建立正式记忆关系的输入。"""

    source_memory_id: str
    target_memory_id: str
    relation_type: RelationType
    reason: str

    def validate(self) -> None:
        """验证关系目标、方向和原因。"""
        if not self.source_memory_id or not self.target_memory_id:
            raise ValueError("关系必须指定 source_memory_id 与 target_memory_id")
        if self.source_memory_id == self.target_memory_id:
            raise ValueError("不能建立 Memory 自环关系")
        if not self.reason.strip():
            raise ValueError("relation reason 不能为空")


@dataclass(frozen=True, slots=True)
class MergeMemoryInput:
    """将多个正式记忆合并到已有 Canonical Memory 的输入。"""

    source_memory_ids: tuple[str, ...]
    canonical_memory_id: str | None = None
    reason: str = ""
    mode: str = "EXISTING_CANONICAL"
    new_memory: CreateMemoryInput | None = None
    evidence_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        """验证合并至少包含一个不同于 Canonical 的来源。"""
        if not self.source_memory_ids:
            raise ValueError("合并必须指定 source_memory_ids")
        if self.mode not in {"EXISTING_CANONICAL", "NEW_CANONICAL"}:
            raise ValueError("Merge mode 无效")
        if self.mode == "EXISTING_CANONICAL" and not self.canonical_memory_id:
            raise ValueError("EXISTING_CANONICAL 必须指定 canonical_memory_id")
        if self.mode == "NEW_CANONICAL" and self.new_memory is None:
            raise ValueError("NEW_CANONICAL 必须指定 new_memory")
        if self.canonical_memory_id is not None and self.canonical_memory_id in self.source_memory_ids:
            raise ValueError("Canonical Memory 不能同时作为 Merge Source")
        if len(set(self.source_memory_ids)) != len(self.source_memory_ids):
            raise ValueError("source_memory_ids 不能重复")
        if any(not isinstance(item, str) or not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids 必须包含非空 ID")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids 不能重复")
        if self.mode == "NEW_CANONICAL" and self.evidence_ids:
            raise ValueError("NEW_CANONICAL 的 Evidence 必须通过 new_memory 指定")
        if not self.reason.strip():
            raise ValueError("merge reason 不能为空")
        if self.new_memory is not None:
            self.new_memory.validate()


@dataclass(frozen=True, slots=True)
class CandidateInput:
    """经历编码器产生的候选素材输入。"""

    rough_title: str
    rough_content: str
    observed_at: datetime
    evidence: tuple[EvidenceInput, ...]
    proposed_kind: MemoryKind | None = None
    subject: SubjectInput | None = None
    participants: tuple[ParticipantInput, ...] = ()

    def validate(self) -> None:
        """验证候选素材文本、来源和初步结构。"""
        for field_name, value in {
            "rough_title": self.rough_title,
            "rough_content": self.rough_content,
        }.items():
            if not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if not self.evidence:
            raise ValueError("Candidate 必须至少关联一份 Evidence")
        if self.subject is not None:
            self.subject.validate()
        for participant in self.participants:
            participant.validate()
        for evidence in self.evidence:
            evidence.validate()


@dataclass(frozen=True, slots=True)
class CandidateActionInput:
    """候选素材处理动作输入。"""

    candidate_id: str
    sleep_session_id: str
    action_type: CandidateActionType
    note: str | None = None
    result_revision_id: str | None = None
    targets: tuple[tuple[str, CandidateActionTargetRole], ...] = ()
    operation_key: str | None = None

    def validate(self) -> None:
        """验证候选动作的目标和说明。"""
        if not self.candidate_id or not self.sleep_session_id:
            raise ValueError("Candidate Action 必须指定候选和 Sleep Session")
        if self.action_type is CandidateActionType.DEFER and not (self.note or "").strip():
            raise ValueError("DEFER Action 必须说明暂缓原因")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("Candidate Action targets 不能重复")


@dataclass(frozen=True, slots=True)
class SleepSessionInput:
    """创建睡眠整理审计会话的输入。"""

    trigger_type: SleepTriggerType
    model_id: str
    prompt_version: str

    def validate(self) -> None:
        """验证模型与 Prompt 标识非空。"""
        if not self.model_id.strip() or not self.prompt_version.strip():
            raise ValueError("Sleep Session 必须指定 model_id 与 prompt_version")


@dataclass(frozen=True, slots=True)
class SleepSessionResult:
    """睡眠会话创建或完成后的状态。"""

    sleep_session_id: str
    status: SleepSessionStatus
    candidate_count: int


@dataclass(frozen=True, slots=True)
class CandidateStateTransition:
    """候选状态转换结果。"""

    previous_status: CandidateStatus
    current_status: CandidateStatus


@dataclass(frozen=True, slots=True)
class MemoryObservationEventInput:
    """不改变认知内容的记忆观察事件输入。"""

    memory_id: str
    event_type: MemoryEventType
    revision_id: str | None = None
    payload: dict[str, object] | None = None

    def validate(self) -> None:
        """只允许读取、召回和闪回暴露事件。"""
        if not self.memory_id:
            raise ValueError("Memory Event 必须指定 memory_id")
        allowed = {
            MemoryEventType.READ,
            MemoryEventType.RECALLED,
            MemoryEventType.FLASHBACK_EXPOSED,
        }
        if self.event_type not in allowed:
            raise ValueError("MemoryEventService 只允许记录观察事件")


@dataclass(frozen=True, slots=True)
class MemoryLifecycleInput:
    """正式记忆工程生命周期操作输入。"""

    memory_id: str
    reason: str

    def validate(self) -> None:
        """验证生命周期操作目标和原因。"""
        if not self.memory_id:
            raise ValueError("Memory Lifecycle 必须指定 memory_id")
        if not self.reason.strip():
            raise ValueError("lifecycle reason 不能为空")


@dataclass(frozen=True, slots=True)
class PersonaUpdateInput:
    """由正式记忆派生人物印象的更新输入。"""

    person_id: str
    impression_text: str
    reason: str
    memory_ids: tuple[str, ...]
    sleep_session_id: str | None = None

    def validate(self) -> None:
        """验证人物、正文、原因和审计记忆引用。"""
        if not self.person_id:
            raise ValueError("Persona Update 必须指定 person_id")
        if not self.impression_text.strip():
            raise ValueError("impression_text 不能为空")
        if not self.reason.strip():
            raise ValueError("persona update reason 不能为空")
        if not self.memory_ids:
            raise ValueError("Persona Update 必须引用 Formal Memory")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("Persona memory_ids 不能重复")


@dataclass(frozen=True, slots=True)
class PersonaUpdateResult:
    """人物印象更新结果。"""

    person_id: str
    changed: bool
    update_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """混合检索查询输入。"""

    text: str
    top_k: int = 10
    person_ids: tuple[str, ...] = ()
    memory_kinds: tuple[MemoryKind, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None

    def validate(self) -> None:
        """验证查询文本与过滤条件。"""
        if not self.text.strip():
            raise ValueError("检索 text 不能为空")
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if len(set(self.person_ids)) != len(self.person_ids):
            raise ValueError("person_ids 不能重复")
        if self.start_time is not None and self.end_time is not None:
            if self.start_time > self.end_time:
                raise ValueError("start_time 不能晚于 end_time")


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """单条正式记忆按 memory_id 聚合后的融合检索结果。"""

    memory_id: str
    title: str
    matched_by: tuple[str, ...]
    rrf_score: float
    lexical_rank: int | None
    vector_rank: int | None
    vector_similarity: float | None = None


@dataclass(frozen=True, slots=True)
class PersonImpression:
    """人物印象与近期相关记忆。"""

    person_id: str
    impression_text: str | None
    recent_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VectorUpsert:
    """向量工作项处理的输入文本与元数据。"""

    entry_id: str
    memory_id: str
    revision_id: str | None
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class WriteContext:
    """领域写操作的执行者与来源上下文。"""

    actor_type: ActorType
    actor_ref: str | None = None
    stream_id: str | None = None
    operation_key: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """正式记忆写事务产生的稳定标识。"""

    memory_id: str
    revision_id: str
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
