"""Engram Memory vNext Tool 门面。

按 Technical Spec 第 75-105 节实现 Tool Contract，并落实第 82/83 节
Actor / Sleep Agent 权限矩阵。BaseTool 层只能薄封装本模块；
权限判定依据 Runtime 注入的身份，不信任 LLM 自报身份。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import desc, func, select

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
    EvidenceSourceType,
    MemoryEventType,
    MemoryKind,
    MemoryStatus,
    RelationType,
)
from .evidence_service import EvidenceService
from .framework_bridge import read_message_context_snapshots, read_message_snapshots
from .memory_service import MemoryService
from .models import (
    EvidenceMessageLinkModel,
    EvidenceMessageSnapshotModel,
    EvidenceModel,
    MemoryEventModel,
    MemoryModel,
    MemoryRelationModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    CandidateEvidenceModel,
    RevisionEvidenceModel,
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
        "message_context_read",
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


def _context_message_time(message: dict[str, object]) -> float:
    """返回快照时间戳；无法解析的时间排在有效时间之后。"""
    value = message.get("time")
    if isinstance(value, datetime):
        return value.timestamp() if value.tzinfo is not None else float("inf")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return float("inf")
            return parsed.timestamp() if parsed.tzinfo is not None else float("inf")
    return float("inf")


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
        on_actor_memory_changed: Callable[[tuple[str, ...]], Awaitable[None]] | None = None,
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
        self._evidence = EvidenceService(schema)
        self._retrieval = RetrievalService(schema, vector_backend, rrf_k=rrf_k)
        self._persona = PersonaService(schema, max_length=persona_max_length)
        self._recent_memory_limit = recent_memory_limit
        self._default_search_limit = default_search_limit
        self._max_search_limit = max_search_limit
        self._on_actor_memory_changed = on_actor_memory_changed

    def require_permission(self, tool: str, context: ToolContext) -> None:
        """按 §82/§83 权限矩阵校验调用资格。"""
        if context.actor_type is ActorType.SLEEP_AGENT:
            allowed = SLEEP_AGENT_ALLOWED_TOOLS
        elif context.actor_type is ActorType.MIGRATION:
            allowed = frozenset({"evidence_read"})
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
                "title": item.title,
                "current_content_preview": views[item.memory_id][
                    "current_content_preview"
                ],
                "memory_kind": views[item.memory_id]["memory_kind"],
                "subject": views[item.memory_id]["subject"],
                "last_experienced_at": views[item.memory_id][
                    "last_experienced_at"
                ],
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
        merged_into = await self._resolve_merged_into(memory_id, memory.status)
        revision_id = revision.revision_id if revision is not None else None
        current: dict[str, object] = {
            "memory_id": memory.memory_id,
            "status": memory.status.value,
            "merged_into": merged_into,
            "created_at": memory.created_at,
            "last_experienced_at": memory.last_experienced_at,
            "current_revision": self._revision_view(revision),
            "current_subject": await self._subject_view(revision),
            "current_participants": await self._participants_view(revision),
            "evidence_summary": await self._evidence_summary(
                memory_id,
                revision_id=revision_id,
                full=False,
            ),
        }
        if view == "current":
            return current
        history = [
            self._revision_history_item(item)
            for item in await self._repository.list_revisions(memory_id)
        ]
        if view == "history":
            return {**current, "history": history}
        evidence_metadata: list[dict[str, object]] = []
        for item in await self._repository.list_revisions(memory_id):
            evidence_metadata.extend(
                await self._evidence_summary(
                    memory_id,
                    revision_id=item.revision_id,
                    full=True,
                )
            )
        return {
            **current,
            "history": history,
            "evidence_metadata": evidence_metadata,
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

        Runtime 自动附加与当前聊天原始消息关联的 ACTOR_WRITE Evidence。
        """
        self.require_permission("memory_write", context)
        message_ids = evidence_message_ids or context.evidence_message_ids
        if context.actor_type is ActorType.ACTOR and not message_ids:
            raise ValueError("主动写入正式记忆必须引用当前聊天的原始消息")
        evidence = data.evidence
        if context.actor_type is ActorType.ACTOR and any(
            message.stream_id != context.stream_id
            or message.message_id not in message_ids
            for item in evidence
            for message in item.messages
        ):
            raise ValueError("主动写入证据必须属于当前聊天已验证的来源消息")
        linked_messages = {
            (message.stream_id, message.message_id)
            for item in evidence
            for message in item.messages
        }
        missing_ids = tuple(
            message_id
            for message_id in message_ids
            if (context.stream_id or "", message_id) not in linked_messages
        )
        if missing_ids:
            evidence += (
                EvidenceInput(
                    source_type=EvidenceSourceType.ACTOR_WRITE,
                    observed_at=datetime.now(UTC),
                    messages=self._message_links(missing_ids, context.stream_id),
                ),
            )
        merged = replace(data, evidence=evidence)
        result = await self._memory.create_memory(merged, context.to_write_context())
        if context.actor_type is ActorType.ACTOR and self._on_actor_memory_changed is not None:
            await self._on_actor_memory_changed((result.memory_id,))
        return {
            "memory_id": result.memory_id,
            "revision_id": result.revision_id,
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
        if context.actor_type is ActorType.ACTOR and self._on_actor_memory_changed is not None:
            await self._on_actor_memory_changed((result.memory_id,))
        return {
            "memory_id": result.memory_id,
            "revision_id": result.revision_id,
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
        """读取已授权候选或正式记忆关联的长期消息快照。"""
        self.require_permission("evidence_read", context)
        if not evidence_ids:
            raise ValueError("evidence_ids 不能为空")
        if context.actor_type is ActorType.MIGRATION:
            async with self._schema.database.session() as session:
                allowed_ids = set(
                    (
                        await session.scalars(
                            select(CandidateEvidenceModel.evidence_id).distinct()
                        )
                    ).all()
                )
            if set(evidence_ids) - allowed_ids:
                raise PermissionError("Migration 只能读取 Candidate 关联的 Evidence")
        elif context.actor_type is ActorType.SLEEP_AGENT:
            async with self._schema.database.session() as session:
                candidate_ids = set(
                    (
                        await session.scalars(
                            select(CandidateEvidenceModel.evidence_id).distinct()
                        )
                    ).all()
                )
                revision_ids = set(
                    (
                        await session.scalars(
                            select(RevisionEvidenceModel.evidence_id).distinct()
                        )
                    ).all()
                )
            if set(evidence_ids) - (candidate_ids | revision_ids):
                raise PermissionError("Sleep Agent 只能读取已关联的 Candidate 或 Memory Evidence")
        return await self._read_evidence_records(evidence_ids, include_messages=True)

    async def message_context_read(
        self,
        evidence_id: str,
        message_id: str,
        context: ToolContext,
        *,
        before: int = 8,
        after: int = 20,
    ) -> dict[str, object]:
        """读取已关联 Evidence anchor 的有限原始聊天上下文。"""
        if context.actor_type is not ActorType.SLEEP_AGENT:
            raise PermissionError("只有 Sleep Agent 可以读取消息上下文")
        if not evidence_id.strip() or not message_id.strip():
            raise ValueError("evidence_id 和 message_id 不能为空")
        if (
            isinstance(before, bool)
            or not isinstance(before, int)
            or not 0 <= before <= 30
        ):
            raise ValueError("before 必须是 0 到 30 的整数")
        if (
            isinstance(after, bool)
            or not isinstance(after, int)
            or not 0 <= after <= 30
        ):
            raise ValueError("after 必须是 0 到 30 的整数")

        async with self._schema.database.session() as session:
            candidate_link = (
                await session.scalars(
                    select(CandidateEvidenceModel.evidence_id)
                    .where(CandidateEvidenceModel.evidence_id == evidence_id)
                    .limit(1)
                )
            ).first()
            revision_link = (
                await session.scalars(
                    select(RevisionEvidenceModel.evidence_id)
                    .where(RevisionEvidenceModel.evidence_id == evidence_id)
                    .limit(1)
                )
            ).first()
            if candidate_link is None and revision_link is None:
                raise PermissionError(
                    "Sleep Agent 只能读取已关联 Candidate 或 Memory 的 Evidence"
                )
            anchor_links = tuple(
                (
                    await session.scalars(
                        select(EvidenceMessageLinkModel).where(
                            EvidenceMessageLinkModel.evidence_id == evidence_id,
                            EvidenceMessageLinkModel.message_id == message_id,
                        )
                    )
                ).all()
            )
            stream_ids = tuple(dict.fromkeys(item.stream_id for item in anchor_links))
            if not stream_ids:
                raise PermissionError("anchor 消息不属于指定 Evidence")
            if len(stream_ids) != 1:
                raise ValueError("anchor 消息在指定 Evidence 下对应多个 stream")
            stream_id = stream_ids[0]
            saved_anchor_rows = tuple(
                (
                    await session.scalars(
                        select(EvidenceMessageSnapshotModel).where(
                            EvidenceMessageSnapshotModel.stream_id == stream_id,
                            EvidenceMessageSnapshotModel.message_id == message_id,
                        )
                    )
                ).all()
            )

        saved_anchor: dict[str, object] | None = None
        if saved_anchor_rows:
            row = saved_anchor_rows[0]
            saved_anchor = dict(row.payload)
            saved_anchor.update(
                message_id=row.message_id,
                stream_id=row.stream_id,
                source="evidence_snapshot",
                captured_at=row.captured_at.isoformat(),
                redacted=row.redacted_at is not None,
            )
            if row.redacted_at is not None:
                raise PermissionError("anchor 消息已隐私删除，不能读取上下文")

        saved_reply_metadata = saved_anchor is not None and "reply_to" in saved_anchor
        reply_to_message_id = (
            str(saved_anchor.get("reply_to") or "").strip()
            if saved_reply_metadata and saved_anchor is not None
            else None
        )
        core_snapshots = await read_message_context_snapshots(
            stream_id,
            message_id,
            before,
            after,
            reply_to_message_id=reply_to_message_id,
            use_core_reply_to=not saved_reply_metadata,
        )
        core_by_id = {item.message_id: item for item in core_snapshots}
        core_anchor = core_by_id.get(message_id)
        if not saved_reply_metadata and core_anchor is not None:
            reply_to_message_id = core_anchor.reply_to
            if saved_anchor is not None:
                saved_anchor = {**saved_anchor, "reply_to": core_anchor.reply_to}

        references = tuple(
            dict.fromkeys(
                [(stream_id, item.message_id) for item in core_snapshots]
                + ([(stream_id, reply_to_message_id)] if reply_to_message_id else [])
            )
        )
        saved_by_key = {
            (str(item["stream_id"]), str(item["message_id"])): item
            for item in await self._evidence.read_messages(references)
        }
        saved_anchor = saved_by_key.get((stream_id, message_id), saved_anchor)
        if (
            saved_anchor is not None
            and "reply_to" not in saved_anchor
            and core_anchor is not None
        ):
            saved_anchor = {**saved_anchor, "reply_to": core_anchor.reply_to}
        if saved_anchor is not None and saved_anchor.get("redacted"):
            raise PermissionError("anchor 消息已隐私删除，不能读取上下文")

        messages: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for snapshot in core_snapshots:
            key = (snapshot.stream_id, snapshot.message_id)
            saved = saved_by_key.get(key)
            if saved is not None and saved.get("redacted"):
                view: dict[str, object] = {
                    "message_id": snapshot.message_id,
                    "stream_id": snapshot.stream_id,
                    "time": snapshot.time,
                    "source": "evidence_snapshot",
                    "redacted": True,
                }
            elif snapshot.message_id == message_id and saved_anchor is not None:
                view = dict(saved_anchor)
            else:
                view = dict(saved) if saved is not None else snapshot.to_dict()
            messages.append(view)
            seen_ids.add(snapshot.message_id)

        reply_target_status = "none"
        has_snapshot_only_reply = False
        if core_anchor is None:
            if saved_anchor is not None:
                messages = [dict(saved_anchor)]
                seen_ids = {message_id}
                if reply_to_message_id:
                    key = (stream_id, reply_to_message_id)
                    saved_reply = saved_by_key.get(key)
                    if saved_reply is not None and saved_reply.get("redacted"):
                        reply_view = {
                            "message_id": reply_to_message_id,
                            "stream_id": stream_id,
                            "source": "evidence_snapshot",
                            "redacted": True,
                        }
                        reply_target_status = "redacted"
                    else:
                        reply_snapshot = await read_message_snapshots((key,))
                        reply_view = (
                            dict(saved_reply)
                            if saved_reply is not None
                            else reply_snapshot[0].to_dict()
                            if reply_snapshot
                            else {}
                        )
                        reply_target_status = "included" if reply_view else "missing"
                    if reply_view and reply_to_message_id != message_id:
                        messages.append(reply_view)
                        seen_ids.add(reply_to_message_id)
                messages.sort(key=_context_message_time)
        elif reply_to_message_id:
            if reply_to_message_id in seen_ids:
                target = saved_by_key.get((stream_id, reply_to_message_id))
                reply_target_status = (
                    "redacted"
                    if target is not None and target.get("redacted")
                    else "included"
                )
            else:
                key = (stream_id, reply_to_message_id)
                saved_reply = saved_by_key.get(key)
                if saved_reply is not None and saved_reply.get("redacted"):
                    reply_view = {
                        "message_id": reply_to_message_id,
                        "stream_id": stream_id,
                        "source": "evidence_snapshot",
                        "redacted": True,
                    }
                    reply_target_status = "redacted"
                else:
                    reply_snapshot = await read_message_snapshots((key,))
                    reply_view = (
                        dict(saved_reply)
                        if saved_reply is not None
                        else reply_snapshot[0].to_dict()
                        if reply_snapshot
                        else {}
                    )
                    reply_target_status = "included" if reply_view else "missing"
                    has_snapshot_only_reply = bool(saved_reply and reply_view)
                if reply_view:
                    anchor_index = next(
                        (
                            index
                            for index, item in enumerate(messages)
                            if item.get("message_id") == message_id
                        ),
                        len(messages),
                    )
                    if _context_message_time(reply_view) < _context_message_time(
                        messages[anchor_index]
                    ):
                        messages.insert(anchor_index, reply_view)
                    else:
                        messages.insert(anchor_index + 1, reply_view)
                    seen_ids.add(reply_to_message_id)

        anchor_source = (
            "evidence_snapshot"
            if saved_anchor is not None
            else "core_message"
            if core_anchor is not None
            else "missing"
        )
        context_status: dict[str, object] = {
            "anchor_source": anchor_source,
            "core_window": "available" if core_anchor is not None else "anchor_missing",
            "ordering": (
                "core_time_id_with_snapshot_reply"
                if has_snapshot_only_reply
                else "core_time_id"
                if core_anchor is not None
                else "snapshot_time_only"
                if saved_anchor is not None
                else "unavailable"
            ),
            "reply_target": reply_target_status,
        }
        return {
            "evidence_id": evidence_id,
            "anchor_message_id": message_id,
            "stream_id": stream_id,
            "messages": messages,
            "context_status": context_status,
        }

    async def memory_evidence_read(
        self,
        memory_id: str,
        context: ToolContext,
        *,
        revision_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """按已授权 Memory 与 Revision 范围读取关联 Evidence 快照。"""
        self.require_permission("memory_read", context)
        memory = await self._repository.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} 不存在")
        selected_revision_id = revision_id or memory.current_revision_id
        revisions = await self._repository.list_revisions(memory_id)
        if not any(item.revision_id == selected_revision_id for item in revisions):
            raise ValueError("revision_id 不属于指定 Memory")
        evidence = await self._repository.list_evidence(
            memory_id,
            revision_id=selected_revision_id,
        )
        await self._record_read_event(memory_id, context)
        return await self._read_evidence_records(
            tuple(item.evidence_id for item in evidence),
            include_messages=True,
            revision_id=selected_revision_id,
        )

    async def _read_evidence_records(
        self,
        evidence_ids: tuple[str, ...],
        *,
        include_messages: bool,
        revision_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """读取 Evidence 元数据和其精确关联的长期消息快照。"""
        if not evidence_ids:
            return ()
        unique_ids = tuple(dict.fromkeys(evidence_ids))
        async with self._schema.database.session() as session:
            evidence_rows = tuple(
                (
                    await session.scalars(
                        select(EvidenceModel).where(
                            EvidenceModel.evidence_id.in_(unique_ids)
                        )
                    )
                ).all()
            )
            links = tuple(
                (
                    await session.scalars(
                        select(EvidenceMessageLinkModel)
                        .where(EvidenceMessageLinkModel.evidence_id.in_(unique_ids))
                        .order_by(
                            EvidenceMessageLinkModel.evidence_id,
                            EvidenceMessageLinkModel.ordinal,
                        )
                    )
                ).all()
            )
        evidence_by_id = {row.evidence_id: row for row in evidence_rows}
        links_by_id: dict[str, list[EvidenceMessageLinkModel]] = {
            evidence_id: [] for evidence_id in unique_ids
        }
        for link in links:
            links_by_id.setdefault(link.evidence_id, []).append(link)

        snapshots_by_ref: dict[tuple[str, str], dict[str, object]] = {}
        if include_messages:
            references = tuple(
                dict.fromkeys((link.stream_id, link.message_id) for link in links)
            )
            snapshots = await self._evidence.read_messages(references)
            snapshots_by_ref = {
                (str(item.get("stream_id") or ""), str(item.get("message_id") or "")): item
                for item in snapshots
            }

        result: list[dict[str, object]] = []
        for evidence_id in unique_ids:
            row = evidence_by_id.get(evidence_id)
            if row is None:
                result.append(
                    {
                        "evidence_id": evidence_id,
                        "source_status": "MISSING_EVIDENCE",
                        "revision_id": revision_id,
                    }
                )
                continue
            record: dict[str, object] = {
                "evidence_id": row.evidence_id,
                "source_type": row.source_type.value,
                "observed_at": row.observed_at,
                "source_ref": row.source_ref,
                "note": row.note,
                "created_at": row.created_at,
                "revision_id": revision_id,
            }
            if include_messages:
                messages: list[dict[str, object]] = []
                missing_messages: list[dict[str, object]] = []
                for link in links_by_id.get(evidence_id, ()):
                    reference = (link.stream_id, link.message_id)
                    snapshot = snapshots_by_ref.get(reference)
                    status = (
                        "MISSING"
                        if snapshot is None
                        else "REDACTED"
                        if snapshot.get("redacted")
                        else "AVAILABLE"
                    )
                    message: dict[str, object] = {
                        "stream_id": link.stream_id,
                        "message_id": link.message_id,
                        "ordinal": link.ordinal,
                        "source_status": status,
                    }
                    if snapshot is not None and status == "AVAILABLE":
                        message["snapshot"] = snapshot
                    messages.append(message)
                    if status != "AVAILABLE":
                        missing_messages.append(message)
                record["messages"] = messages
                record["missing_messages"] = missing_messages
            result.append(record)
        return tuple(result)

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
        person_aliases = await self._repository.resolve_person_aliases(person_id)
        persona = None
        for alias in person_aliases:
            persona = await self._persona.get_persona(alias)
            if persona is not None:
                break
        recent = await self._recent_memories(person_id)
        return {
            "person_id": person_id,
            "core_person_id": persona.person_id if persona else None,
            "basic_person_info": basic_person_info,
            "persona_impression": (persona.impression_text or None) if persona else None,
            "persona_updated_at": persona.updated_at if persona else None,
            "recent_memories": [
                {
                    "memory_id": row.memory_id,
                    "title": row.title,
                    "current_content_preview": row.content[:80],
                    "last_experienced_at": row.last_experienced_at or row.created_at,
                }
                for row in recent
            ],
        }

    async def _basic_person_info(self, person_id: str) -> dict[str, object] | None:
        """优先读取核心身份中心，再使用已保存的消息人物元数据。"""
        person = await self._persona.get_core_person(person_id)
        if person is not None:
            return {
                "platform": person.platform,
                "user_id": person.user_id,
                "nickname": person.nickname,
                "cardname": person.cardname,
            }
        snapshot_info = await self._repository.get_person_metadata(person_id)
        if snapshot_info is not None:
            return snapshot_info
        return None

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

    async def person_impression_review_complete(
        self, data: PersonaUpdateInput, context: ToolContext,
    ) -> None:
        """登记 Sleep 对核心印象的保持决定及其正式记忆依据。"""
        self.require_permission("person_impression_update", context)
        if context.sleep_session_id is not None:
            data = replace(data, sleep_session_id=context.sleep_session_id)
        await self._persona.record_unchanged_review(data, context.to_write_context())

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
        person_ids = await self._repository.resolve_person_aliases(person_id)
        async with self._schema.database.session() as session:
            subject_ids = set(
                (
                    await session.scalars(
                        select(MemoryRevisionSubjectModel.revision_id).where(
                            MemoryRevisionSubjectModel.person_id.in_(person_ids)
                        )
                    )
                ).all()
            )
            participant_ids = set(
                (
                    await session.scalars(
                        select(MemoryRevisionParticipantModel.revision_id).where(
                            MemoryRevisionParticipantModel.person_id.in_(person_ids)
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
        revision_id: str | None,
        full: bool,
    ) -> list[dict[str, object]]:
        """读取记忆指定版本关联的证据摘要或长期快照。"""
        evidences = await self._repository.list_evidence(
            memory_id,
            revision_id=revision_id,
        )
        if not full:
            return [
                {
                    "evidence_id": item.evidence_id,
                    "source_type": item.source_type.value,
                    "observed_at": item.observed_at,
                    "note": item.note,
                }
                for item in evidences
            ]
        return list(
            await self._read_evidence_records(
                tuple(item.evidence_id for item in evidences),
                include_messages=True,
                revision_id=revision_id,
            )
        )

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
            "observed_at": revision.observed_at,  # type: ignore[attr-defined]
            "created_at": revision.created_at,  # type: ignore[attr-defined]
            "change_reason": revision.change_reason.value,  # type: ignore[attr-defined]
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
            "observed_at": revision.observed_at,  # type: ignore[attr-defined]
            "created_at": revision.created_at,  # type: ignore[attr-defined]
            "change_reason": revision.change_reason.value,  # type: ignore[attr-defined]
        }
