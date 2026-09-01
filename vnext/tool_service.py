"""Engram Memory vNext Tool 门面。

按 Technical Spec 第 75-105 节实现 Tool Contract，并落实第 82/83 节
Actor / Sleep Agent 权限矩阵。BaseTool 层只能薄封装本模块；
权限判定依据 Runtime 注入的身份，不信任 LLM 自报身份。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4

from src.app.plugin_system.api import person_api
from sqlalchemy import desc, func, select

from .framework_bridge import read_message_snapshots
from .domain import (
    CreateMemoryInput,
    EvidenceInput,
    EvidenceMessageInput,
    MergeMemoryInput,
    PersonaUpdateInput,
    ReinforceMemoryInput,
    RelateMemoryInput,
    RetrievalQuery,
    ReviseMemoryInput,
    WriteContext,
)
from .enums import (
    ActorType,
    ClaimBasis,
    EvidenceSourceType,
    MemoryEventType,
    MemoryKind,
    MemoryStatus,
    ProvenanceQuality,
    RelationType,
)
from .memory_service import MemoryService
from .models import (
    EvidenceMessageLinkModel,
    MemoryEventModel,
    MemoryAssessmentModel,
    MemoryModel,
    MemoryRelationModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
)
from .persona_service import PersonaService
from .relation_service import MergeService, RelationService
from .repository import MemoryRepository
from .retrieval_service import RetrievalService, VectorSearchBackend
from .schema import VNextSchema

# 普通 Actor 可用工具（§82）；memory_revise 仅限 EXPLICIT_CORRECTION（§72）
ACTOR_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"memory_search", "memory_read", "memory_write", "person_lookup", "memory_revise"}
)
# Sleep Agent 可用工具（§83）
SLEEP_AGENT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "memory_search",
        "memory_read",
        "evidence_read",
        "memory_write",
        "memory_reinforce",
        "memory_revise",
        "memory_merge",
        "memory_relate",
        "person_lookup",
        "person_impression_update",
    }
)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Tool 调用身份与来源，由 Runtime 注入。"""

    actor_type: ActorType
    actor_ref: str | None = None
    stream_id: str | None = None
    sleep_session_id: str | None = None
    evidence_message_ids: tuple[str, ...] = field(default_factory=tuple)
    operation_key: str | None = None

    def to_write_context(self) -> WriteContext:
        """转换为领域写上下文。"""
        return WriteContext(
            actor_type=self.actor_type,
            actor_ref=self.actor_ref,
            stream_id=self.stream_id,
            operation_key=self.operation_key,
        )


class VNextToolService:
    """vNext 全部 Tool 的领域门面，内含权限矩阵硬约束。"""

    def __init__(
        self,
        schema: VNextSchema,
        vector_backend: VectorSearchBackend,
        *,
        embedding_model_id: str = "tool-embedding",
        persona_max_length: int = 500,
        recent_memory_limit: int = 10,
        default_search_limit: int = 5,
        max_search_limit: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """装配领域服务与 Tool 级参数。

        参数:
            schema: vNext Schema。
            vector_backend: 向量检索后端。
            embedding_model_id: 检索入口记录的向量模型标识。
            persona_max_length: 人物印象正文长度上限。
            recent_memory_limit: person_lookup 返回的近期记忆条数。
            default_search_limit: memory_search 默认返回条数。
            max_search_limit: memory_search 最大返回条数。
            rrf_k: Reciprocal Rank Fusion 的常数项。
        """
        if persona_max_length <= 0:
            raise ValueError("persona_max_length 必须大于 0")
        if recent_memory_limit <= 0:
            raise ValueError("recent_memory_limit 必须大于 0")
        if default_search_limit <= 0 or max_search_limit < default_search_limit:
            raise ValueError("default_search_limit/max_search_limit 参数非法")
        self._schema = schema
        self._memory = MemoryService(schema, embedding_model_id)
        self._relation = RelationService(schema)
        self._merge = MergeService(schema, self._memory)
        self._repository = MemoryRepository(schema)
        self._retrieval = RetrievalService(schema, vector_backend, rrf_k=rrf_k)
        self._persona = PersonaService(schema, max_length=persona_max_length)
        self._recent_memory_limit = recent_memory_limit
        self._default_search_limit = default_search_limit
        self._max_search_limit = max_search_limit

    def require_permission(self, tool: str, context: ToolContext) -> None:
        """按 §82/§83 权限矩阵校验调用资格。"""
        if context.actor_type is ActorType.SLEEP_AGENT:
            allowed = SLEEP_AGENT_ALLOWED_TOOLS
        elif context.actor_type is ActorType.ACTOR:
            allowed = ACTOR_ALLOWED_TOOLS
        elif context.actor_type is ActorType.ADMIN:
            allowed = SLEEP_AGENT_ALLOWED_TOOLS | ACTOR_ALLOWED_TOOLS
        else:
            allowed = frozenset()
        if tool not in allowed:
            raise PermissionError(f"{context.actor_type.value} 无权调用 {tool}")

    async def memory_search(
        self,
        query: str,
        context: ToolContext,
        *,
        person_ids: tuple[str, ...] = (),
        memory_kinds: tuple[str, ...] = (),
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[dict[str, object], ...]:
        """memory_search：仅暴露语义参数（§75），返回候选目录（§76）。"""
        self.require_permission("memory_search", context)
        if limit is not None and limit <= 0:
            raise ValueError("limit 必须大于 0")
        kinds = tuple(MemoryKind(kind) for kind in memory_kinds)
        top_k = min(limit or self._default_search_limit, self._max_search_limit)
        results = await self._retrieval.search(
            RetrievalQuery(
                text=query,
                top_k=top_k,
                person_ids=person_ids,
                memory_kinds=kinds,
                start_time=start_time,
                end_time=end_time,
            ),
            context.to_write_context(),
        )
        views = await self._search_views(tuple(item.memory_id for item in results))
        return tuple(
            {
                "memory_id": item.memory_id,
                "status": views[item.memory_id]["status"],
                "anchor_title": item.anchor_title,
                "current_revision_title": views[item.memory_id][
                    "current_revision_title"
                ],
                "current_content_preview": views[item.memory_id][
                    "current_content_preview"
                ],
                "memory_kind": views[item.memory_id]["memory_kind"],
                "subject": views[item.memory_id]["subject"],
                "last_experienced_at": views[item.memory_id][
                    "last_experienced_at"
                ],
                "salience": views[item.memory_id]["salience"],
                "matched_by": list(item.matched_by),
                "rrf_score": item.rrf_score,
            }
            for item in results
        )

    async def _search_views(
        self,
        memory_ids: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        """读取 memory_search 候选目录所需的当前派生字段。"""
        if not memory_ids:
            return {}
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.execute(
                        select(
                            MemoryModel.memory_id,
                            MemoryModel.status,
                            MemoryRevisionModel.title,
                            MemoryRevisionModel.content,
                            MemoryRevisionModel.memory_kind,
                            MemoryAssessmentModel.salience,
                            MemoryModel.last_experienced_at,
                            MemoryRevisionSubjectModel.subject_kind,
                            MemoryRevisionSubjectModel.person_id,
                            MemoryRevisionSubjectModel.subject_key,
                            MemoryRevisionSubjectModel.subject_label,
                        )
                        .join(
                            MemoryRevisionModel,
                            MemoryRevisionModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .join(
                            MemoryAssessmentModel,
                            MemoryAssessmentModel.assessment_id
                            == MemoryModel.current_assessment_id,
                        )
                        .join(
                            MemoryRevisionSubjectModel,
                            MemoryRevisionSubjectModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(MemoryModel.memory_id.in_(memory_ids))
                    )
                ).all()
            )
        return {
            row.memory_id: {
                "status": row.status.value,
                "current_revision_title": row.title,
                "current_content_preview": row.content[:160],
                "memory_kind": row.memory_kind.value,
                "subject": {
                    "subject_kind": row.subject_kind.value,
                    "person_id": row.person_id,
                    "subject_key": row.subject_key,
                    "subject_label": row.subject_label,
                },
                "last_experienced_at": row.last_experienced_at,
                "salience": row.salience.value,
            }
            for row in rows
        }

    async def memory_read(
        self,
        memory_id: str,
        view: str,
        context: ToolContext,
    ) -> dict[str, object]:
        """memory_read：current/history/full 三视图（§77-§80）。"""
        self.require_permission("memory_read", context)
        if view not in {"current", "history", "full"}:
            raise ValueError("view 只能是 current/history/full")
        memory = await self._repository.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} 不存在")
        await self._record_read_event(memory_id, context)
        revision = await self._repository.get_current_revision(memory_id)
        assessment = await self._repository.get_current_assessment(memory_id)
        merged_into = await self._resolve_merged_into(memory_id, memory.status)
        current: dict[str, object] = {
            "memory_id": memory.memory_id,
            "status": memory.status.value,
            "merged_into": merged_into,
            "anchor_title": memory.anchor_title,
            "last_experienced_at": memory.last_experienced_at,
            "current_revision": self._revision_view(revision),
            "current_subject": await self._subject_view(revision),
            "current_participants": await self._participants_view(revision),
            "current_assessment": self._assessment_view(assessment),
            "evidence_summary": await self._evidence_summary(memory_id, full=False),
        }
        if view == "current":
            return current
        history = [
            self._revision_history_item(item)
            for item in await self._repository.list_revisions(memory_id)
        ]
        if view == "history":
            return {**current, "history": history}
        return {
            **current,
            "history": history,
            "evidence_metadata": await self._evidence_summary(memory_id, full=True),
            "assessment_history": [
                self._assessment_view(item)
                for item in await self._repository.list_assessments(memory_id)
            ],
            "events": [
                {
                    "event_id": item.event_id,
                    "event_type": item.event_type.value,
                    "actor_type": item.actor_type.value,
                    "occurred_at": item.occurred_at,
                }
                for item in await self._repository.list_events(memory_id)
            ],
        }

    async def memory_write(
        self,
        data: CreateMemoryInput,
        context: ToolContext,
        *,
        evidence_message_ids: tuple[str, ...] = (),
    ) -> dict[str, str]:
        """memory_write：主动写入直接创建 Formal Memory（§67）。

        Runtime 自动附加 ACTOR_WRITE / EXPLICIT_MEMORY_WRITE Evidence。
        """
        self.require_permission("memory_write", context)
        message_ids = evidence_message_ids or context.evidence_message_ids
        auto_evidence = EvidenceInput(
            source_type=EvidenceSourceType.ACTOR_WRITE,
            claim_basis=ClaimBasis.EXPLICIT_MEMORY_WRITE,
            provenance_quality=ProvenanceQuality.EXACT,
            observed_at=datetime.now(UTC),
            messages=self._message_links(message_ids, context.stream_id),
        )
        merged = replace(data, evidence=data.evidence + (auto_evidence,))
        result = await self._memory.create_memory(merged, context.to_write_context())
        return {
            "memory_id": result.memory_id,
            "revision_id": result.revision_id,
            "assessment_id": result.assessment_id,
        }

    async def memory_reinforce(
        self,
        data: ReinforceMemoryInput,
        context: ToolContext,
    ) -> dict[str, str]:
        """memory_reinforce：同认知新增证据（§69）。"""
        self.require_permission("memory_reinforce", context)
        data = self._with_reinforce_evidence(data, context.evidence_message_ids, context)
        result = await self._memory.reinforce_memory(data, context.to_write_context())
        return {
            "memory_id": result.memory_id,
            "revision_id": result.revision_id,
            "assessment_id": result.assessment_id,
        }

    async def memory_revise(
        self,
        data: ReviseMemoryInput,
        context: ToolContext,
    ) -> dict[str, str]:
        """memory_revise：认知变化创建新 Revision（§70-§72）。

        普通 Actor 只允许 EXPLICIT_CORRECTION。
        """
        self.require_permission("memory_revise", context)
        if (
            context.actor_type is ActorType.ACTOR
            and data.change_reason.value != "EXPLICIT_CORRECTION"
        ):
            raise PermissionError("普通 Actor 只能以 EXPLICIT_CORRECTION 执行 memory_revise")
        if context.actor_type is ActorType.ACTOR and not context.evidence_message_ids:
            raise PermissionError("普通 Actor 的 EXPLICIT_CORRECTION 必须绑定当前纠错消息证据")
        data = self._with_revise_evidence(data, context.evidence_message_ids, context)
        result = await self._memory.revise_memory(data, context.to_write_context())
        return {
            "memory_id": result.memory_id,
            "revision_id": result.revision_id,
            "assessment_id": result.assessment_id,
        }

    async def memory_merge(
        self,
        data: MergeMemoryInput,
        context: ToolContext,
    ) -> dict[str, str]:
        """memory_merge：多条重复记忆合并到既有 Canonical（§73）。"""
        self.require_permission("memory_merge", context)
        canonical_id, relation_ids = await self._merge.merge(
            data, context.to_write_context()
        )
        return {
            "canonical_memory_id": canonical_id,
            "relation_ids": ",".join(relation_ids),
        }

    async def memory_relate(
        self,
        data: RelateMemoryInput,
        context: ToolContext,
    ) -> dict[str, str]:
        """memory_relate：建立有方向且防重复的记忆关系（§74）。"""
        self.require_permission("memory_relate", context)
        relation_id = await self._relation.relate_memory(data, context.to_write_context())
        return {"relation_id": relation_id}

    async def evidence_read(
        self,
        evidence_ids: tuple[str, ...],
        context: ToolContext,
    ) -> tuple[dict[str, object], ...]:
        """evidence_read：从公共消息源读取证据引用及原始消息（§81）。"""
        self.require_permission("evidence_read", context)
        if not evidence_ids:
            raise ValueError("evidence_ids 不能为空")
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(EvidenceMessageLinkModel)
                        .where(EvidenceMessageLinkModel.evidence_id.in_(evidence_ids))
                        .order_by(
                            EvidenceMessageLinkModel.evidence_id,
                            EvidenceMessageLinkModel.ordinal,
                        )
                    )
                ).all()
            )
        links_by_evidence: dict[str, list[tuple[str, str, int]]] = {
            evidence_id: [] for evidence_id in evidence_ids
        }
        for row in rows:
            links_by_evidence.setdefault(row.evidence_id, []).append(
                (row.message_id, row.stream_id, row.ordinal)
            )
        source_messages = await self._load_evidence_messages(
            tuple(
                link
                for links in links_by_evidence.values()
                for link in links
            )
        )
        result: list[dict[str, object]] = []
        for evidence_id in evidence_ids:
            messages: list[dict[str, object]] = []
            for message_id, stream_id, ordinal in sorted(
                links_by_evidence.get(evidence_id, ()),
                key=lambda item: item[2],
            ):
                message = source_messages.get((stream_id, message_id))
                if message is None:
                    message = source_messages.get(("", message_id))
                if message is None and not stream_id:
                    message = next(
                        (
                            row
                            for (_, candidate_id), row in source_messages.items()
                            if candidate_id == message_id
                        ),
                        None,
                    )
                if message is None:
                    continue
                messages.append(
                    {
                        "message_id": message_id,
                        "stream_id": stream_id or message.get("stream_id", ""),
                        "ordinal": ordinal,
                        "time": message.get("time"),
                        "sender_id": message.get("sender_id"),
                        "sender_name": message.get("sender_name"),
                        "sender_cardname": message.get("sender_cardname"),
                        "person_id": message.get("person_id"),
                        "platform": message.get("platform"),
                        "message_type": message.get("message_type"),
                        "content": message.get("content", ""),
                        "processed_plain_text": message.get(
                            "processed_plain_text", ""
                        ),
                        "text": message.get("processed_plain_text")
                        or message.get("content", ""),
                        "source": "message_store",
                    }
                )
            result.append({"evidence_id": evidence_id, "messages": messages})
        return tuple(result)

    async def _load_evidence_messages(
        self,
        links: tuple[tuple[str, str, int], ...],
    ) -> dict[tuple[str, str], dict[str, object]]:
        """通过集中 Bridge 按消息 ID 批量读取，不扫描完整消息历史。"""
        refs = tuple((stream_id, message_id) for message_id, stream_id, _ in links)
        snapshots = await read_message_snapshots(refs)
        unscoped_ids = {
            message_id for message_id, stream_id, _ in links if not stream_id
        }
        result: dict[tuple[str, str], dict[str, object]] = {}
        for snapshot in snapshots:
            row = snapshot.to_dict()
            result[(snapshot.stream_id, snapshot.message_id)] = row
            if snapshot.message_id in unscoped_ids:
                result[("", snapshot.message_id)] = row
        return result

    async def person_lookup(
        self,
        person_id: str,
        context: ToolContext,
    ) -> dict[str, object]:
        """person_lookup：三段式返回基本信息 + Persona + Recent Memory（§98-§100）。"""
        self.require_permission("person_lookup", context)
        if not person_id.strip():
            raise ValueError("person_id 不能为空")
        basic_person_info = await self._basic_person_info(person_id)
        persona = await self._persona.get_persona(person_id)
        recent = await self._recent_memories(person_id)
        return {
            "person_id": person_id,
            "basic_person_info": basic_person_info,
            "persona_impression": persona.impression_text if persona else None,
            "persona_updated_at": persona.updated_at if persona else None,
            "recent_memories": [
                {
                    "memory_id": row.memory_id,
                    "anchor_title": row.anchor_title,
                    "current_revision_title": row.title,
                    "current_content_preview": row.content[:80],
                    "last_experienced_at": row.last_experienced_at or row.created_at,
                }
                for row in recent
            ],
        }

    @staticmethod
    async def _basic_person_info(person_id: str) -> dict[str, object] | None:
        """按 ``platform:user_id`` 读取公开 Person API 的基本资料。"""
        if ":" not in person_id:
            return None
        platform, user_id = person_id.split(":", 1)
        if not platform.strip() or not user_id.strip():
            return None
        person = await person_api.get_person(platform, user_id)
        if person is None:
            return None
        return {
            "platform": platform,
            "user_id": user_id,
            "nickname": person.nickname,
            "cardname": person.cardname,
        }

    async def person_impression_update(
        self,
        data: PersonaUpdateInput,
        context: ToolContext,
    ) -> dict[str, object]:
        """person_impression_update：仅 Sleep Agent / ADMIN（§90）。"""
        self.require_permission("person_impression_update", context)
        if context.sleep_session_id is not None:
            data = replace(data, sleep_session_id=context.sleep_session_id)
        result = await self._persona.update_persona(data, context.to_write_context())
        return {
            "person_id": result.person_id,
            "changed": result.changed,
            "update_id": result.update_id,
        }

    @staticmethod
    def _with_reinforce_evidence(
        data: ReinforceMemoryInput,
        message_ids: tuple[str, ...],
        context: ToolContext,
    ) -> ReinforceMemoryInput:
        """为强化输入自动绑定当前对话消息证据。"""
        if not message_ids:
            return data
        auto = EvidenceInput(
            source_type=EvidenceSourceType.ACTOR_WRITE,
            claim_basis=ClaimBasis.EXPLICIT_MEMORY_WRITE,
            provenance_quality=ProvenanceQuality.EXACT,
            observed_at=datetime.now(UTC),
            messages=VNextToolService._message_links(message_ids, context.stream_id),
        )
        return replace(data, evidence=data.evidence + (auto,))

    @staticmethod
    def _with_revise_evidence(
        data: ReviseMemoryInput,
        message_ids: tuple[str, ...],
        context: ToolContext,
    ) -> ReviseMemoryInput:
        """为修订输入自动绑定纠错消息证据（§72 要求纠错 Message Evidence）。"""
        if not message_ids:
            return data
        auto = EvidenceInput(
            source_type=EvidenceSourceType.ACTOR_WRITE,
            claim_basis=ClaimBasis.EXPLICIT_MEMORY_WRITE,
            provenance_quality=ProvenanceQuality.EXACT,
            observed_at=datetime.now(UTC),
            messages=VNextToolService._message_links(message_ids, context.stream_id),
        )
        return replace(data, evidence=data.evidence + (auto,))

    @staticmethod
    def _message_links(
        message_ids: tuple[str, ...],
        stream_id: str | None,
    ) -> tuple[EvidenceMessageInput, ...]:
        """构造消息证据链接。"""
        return tuple(
            EvidenceMessageInput(message_id=message_id, stream_id=stream_id or "")
            for message_id in message_ids
        )

    async def _recent_memories(self, person_id: str) -> tuple[object, ...]:
        """按 §99 读取人物相关近期正式记忆。

        当前 Revision 的 Subject 或 Participants 命中人物、status 为 ACTIVE，
        按 last_experienced_at（缺省 created_at）降序取前 recent_memory_limit 条。
        """
        async with self._schema.database.session() as session:
            subject_ids = set(
                (
                    await session.scalars(
                        select(MemoryRevisionSubjectModel.revision_id).where(
                            MemoryRevisionSubjectModel.person_id == person_id
                        )
                    )
                ).all()
            )
            participant_ids = set(
                (
                    await session.scalars(
                        select(MemoryRevisionParticipantModel.revision_id).where(
                            MemoryRevisionParticipantModel.person_id == person_id
                        )
                    )
                ).all()
            )
            hit_revision_ids = subject_ids | participant_ids
            if not hit_revision_ids:
                return ()
            rows = tuple(
                (
                    await session.execute(
                        select(
                            MemoryModel.memory_id,
                            MemoryModel.anchor_title,
                            MemoryModel.last_experienced_at,
                            MemoryModel.created_at,
                            MemoryRevisionModel.title,
                            MemoryRevisionModel.content,
                        )
                        .join(
                            MemoryRevisionModel,
                            MemoryRevisionModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(
                            MemoryModel.status == MemoryStatus.ACTIVE,
                            MemoryModel.current_revision_id.in_(hit_revision_ids),
                        )
                        .order_by(
                            desc(
                                func.coalesce(
                                    MemoryModel.last_experienced_at,
                                    MemoryModel.created_at,
                                )
                            )
                        )
                        .limit(self._recent_memory_limit)
                    )
                ).all()
            )
        return rows

    async def _record_read_event(self, memory_id: str, context: ToolContext) -> None:
        """memory_read 命中即记录 READ 观察事件（不改认知属性）。"""
        async with self._schema.database.session() as session:
            session.add(
                MemoryEventModel(
                    event_id=str(uuid4()),
                    memory_id=memory_id,
                    revision_id=None,
                    event_type=MemoryEventType.READ,
                    actor_type=context.actor_type,
                    actor_ref=context.actor_ref,
                    stream_id=context.stream_id,
                    occurred_at=datetime.now(UTC),
                    payload_json=None,
                )
            )

    async def _resolve_merged_into(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> str | None:
        """读取 MERGED 记忆的 Canonical 指向（§39）。"""
        if status is not MemoryStatus.MERGED:
            return None
        async with self._schema.database.session() as session:
            row = (
                await session.scalars(
                    select(MemoryRelationModel.target_memory_id).where(
                        MemoryRelationModel.source_memory_id == memory_id,
                        MemoryRelationModel.relation_type == RelationType.MERGED_INTO,
                        MemoryRelationModel.retracted_at.is_(None),
                    )
                )
            ).first()
        return row

    async def _evidence_summary(
        self,
        memory_id: str,
        *,
        full: bool,
    ) -> list[dict[str, object]]:
        """读取记忆关联证据摘要或完整元数据。"""
        evidences = await self._repository.list_evidence(memory_id)
        if not full:
            return [
                {
                    "evidence_id": item.evidence_id,
                    "source_type": item.source_type.value,
                    "claim_basis": item.claim_basis.value,
                }
                for item in evidences
            ]
        return [
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type.value,
                "claim_basis": item.claim_basis.value,
                "provenance_quality": item.provenance_quality.value,
                "observed_at": item.observed_at,
                "source_ref": item.source_ref,
                "note": item.note,
                "created_at": item.created_at,
            }
            for item in evidences
        ]

    async def _subject_view(self, revision: object | None) -> dict[str, object] | None:
        """读取当前版本主体。"""
        if revision is None:
            return None
        revision_id = revision.revision_id  # type: ignore[attr-defined]
        async with self._schema.database.session() as session:
            row = await session.get(MemoryRevisionSubjectModel, revision_id)
        if row is None:
            # 外键约束下每条 Revision 必有 Subject，防御悬空指针场景
            return None  # pragma: no cover
        return {
            "subject_kind": row.subject_kind.value,
            "person_id": row.person_id,
            "subject_key": row.subject_key,
            "subject_label": row.subject_label,
        }

    async def _participants_view(self, revision: object | None) -> list[dict[str, object]]:
        """读取当前版本参与者。"""
        if revision is None:
            return []
        revision_id = revision.revision_id  # type: ignore[attr-defined]
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(MemoryRevisionParticipantModel).where(
                            MemoryRevisionParticipantModel.revision_id == revision_id
                        )
                    )
                ).all()
            )
        return [
            {
                "participant_kind": row.participant_kind.value,
                "person_id": row.person_id,
                "label": row.label,
                "role": row.role.value,
            }
            for row in rows
        ]

    @staticmethod
    def _revision_view(revision: object | None) -> dict[str, object] | None:
        """构造当前版本视图。"""
        if revision is None:
            return None
        return {
            "revision_id": revision.revision_id,  # type: ignore[attr-defined]
            "revision_no": revision.revision_no,  # type: ignore[attr-defined]
            "title": revision.title,  # type: ignore[attr-defined]
            "content": revision.content,  # type: ignore[attr-defined]
            "memory_kind": revision.memory_kind.value,  # type: ignore[attr-defined]
            "confidence": revision.confidence.value,  # type: ignore[attr-defined]
        }

    @staticmethod
    def _revision_history_item(revision: object) -> dict[str, object]:
        """构造历史版本条目（§79）。"""
        return {
            "revision_id": revision.revision_id,  # type: ignore[attr-defined]
            "revision_no": revision.revision_no,  # type: ignore[attr-defined]
            "title": revision.title,  # type: ignore[attr-defined]
            "content": revision.content,  # type: ignore[attr-defined]
            "memory_kind": revision.memory_kind.value,  # type: ignore[attr-defined]
            "confidence": revision.confidence.value,  # type: ignore[attr-defined]
            "confidence_reason": revision.confidence_reason,  # type: ignore[attr-defined]
            "observed_at": revision.observed_at,  # type: ignore[attr-defined]
            "event_start_at": revision.event_start_at,  # type: ignore[attr-defined]
            "event_end_at": revision.event_end_at,  # type: ignore[attr-defined]
            "change_reason": revision.change_reason.value,  # type: ignore[attr-defined]
        }

    @staticmethod
    def _assessment_view(assessment: object | None) -> dict[str, object] | None:
        """构造认知评估视图。"""
        if assessment is None:
            return None
        return {
            "assessment_id": assessment.assessment_id,  # type: ignore[attr-defined]
            "stability": assessment.stability.value,  # type: ignore[attr-defined]
            "salience": assessment.salience.value,  # type: ignore[attr-defined]
            "reason": assessment.reason,  # type: ignore[attr-defined]
        }
