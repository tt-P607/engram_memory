"""Engram Memory vNext 记忆关系与合并领域服务。"""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import or_, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .domain import MergeMemoryInput, RelateMemoryInput, WriteContext
from .enums import MemoryEventType, MemoryStatus, RelationType
from .models import DomainOperationModel, MemoryEventModel, MemoryModel, MemoryRelationModel
from .schema import VNextSchema

if TYPE_CHECKING:
    from .memory_service import MemoryService

SYMMETRIC_RELATIONS = frozenset({RelationType.CONTRADICTS, RelationType.RELATED_TO})


def _new_id() -> str:
    """生成统一 UUID 字符串。"""
    return str(uuid4())


async def _claim_domain_operation(
    session: AsyncSession,
    operation_key: str | None,
    operation_type: str,
) -> dict[str, object] | None:
    """在当前写事务中原子声明关系领域操作。"""
    if operation_key is None:
        return None
    await session.execute(
        sqlite_insert(DomainOperationModel)
        .values(
            operation_key=operation_key,
            operation_type=operation_type,
            result_json=None,
            created_at=datetime.now(UTC),
            completed_at=None,
        )
        .prefix_with("OR IGNORE")
    )
    operation = await session.get(DomainOperationModel, operation_key)
    if operation is None or operation.operation_type != operation_type:
        raise ValueError("operation_key 已绑定不同领域操作")
    return dict(operation.result_json) if operation.result_json is not None else None


async def _complete_domain_operation(
    session: AsyncSession,
    operation_key: str | None,
    result: dict[str, object],
) -> None:
    """在关系领域事务内固化操作结果。"""
    if operation_key is None:
        return
    operation = await session.get(DomainOperationModel, operation_key)
    if operation is None:
        raise ValueError("领域操作声明不存在")
    if operation.result_json is None:
        operation.result_json = result
        operation.completed_at = datetime.now(UTC)


class RelationService:
    """管理正式记忆之间可审计、可撤回的关系。"""

    def __init__(self, schema: VNextSchema) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema

    async def _find_operation_relation(
        self,
        session: AsyncSession,
        operation_key: str | None,
    ) -> str | None:
        """按稳定动作 key 找到已提交的关系事件。"""
        if operation_key is None:
            return None
        events = tuple(
            (
                await session.scalars(
                    select(MemoryEventModel).where(
                        MemoryEventModel.event_type == MemoryEventType.RELATED
                    )
                )
            ).all()
        )
        for event in events:
            if (
                isinstance(event.payload_json, dict)
                and event.payload_json.get("operation_key") == operation_key
                and event.payload_json.get("relation_id")
            ):
                return str(event.payload_json["relation_id"])
        return None

    async def relate_memory(
        self,
        data: RelateMemoryInput,
        context: WriteContext,
    ) -> str:
        """建立非重复关系并写入双方审计事件。"""
        data.validate()
        if data.relation_type is RelationType.MERGED_INTO:
            raise ValueError("MERGED_INTO 只能通过 MergeService 创建")
        now = datetime.now(UTC)
        relation_id = _new_id()
        async with self._schema.database.session() as session:
            existing_result = await _claim_domain_operation(
                session, context.operation_key, "RELATE"
            )
            if existing_result is not None:
                return str(existing_result["relation_id"])
            existing_relation = await self._find_operation_relation(
                session, context.operation_key
            )
            if existing_relation is not None:
                await _complete_domain_operation(
                    session,
                    context.operation_key,
                    {"relation_id": existing_relation},
                )
                return existing_relation
            await self._require_memories(session, data.source_memory_id, data.target_memory_id)
            if await self._find_active_relation(
                session,
                data.source_memory_id,
                data.target_memory_id,
                data.relation_type,
            ):
                raise ValueError("Memory Relation 已存在")
            session.add(
                MemoryRelationModel(
                    relation_id=relation_id,
                    source_memory_id=data.source_memory_id,
                    target_memory_id=data.target_memory_id,
                    relation_type=data.relation_type,
                    reason=data.reason,
                    created_at=now,
                    created_by_type=context.actor_type,
                    retracted_at=None,
                    retract_reason=None,
                )
            )
            self._add_relation_events(session, data, context, relation_id, now)
            await _complete_domain_operation(
                session,
                context.operation_key,
                {"relation_id": relation_id},
            )
        return relation_id

    async def retract_relation(
        self,
        relation_id: str,
        reason: str,
        context: WriteContext,
    ) -> None:
        """撤回现有关系并保留原记录。"""
        if not reason.strip():
            raise ValueError("retract reason 不能为空")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            relation = await session.get(MemoryRelationModel, relation_id)
            if relation is None:
                raise ValueError("Memory Relation 不存在")
            if relation.retracted_at is not None:
                raise ValueError("Memory Relation 已撤回")
            result = await session.execute(
                update(MemoryRelationModel)
                .where(
                    MemoryRelationModel.relation_id == relation_id,
                    MemoryRelationModel.retracted_at.is_(None),
                )
                .values(retracted_at=now, retract_reason=reason)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise ValueError("Memory Relation 已被并发撤回")
            relation.retracted_at = now
            relation.retract_reason = reason
            for memory_id in {relation.source_memory_id, relation.target_memory_id}:
                session.add(
                    MemoryEventModel(
                        event_id=_new_id(),
                        memory_id=memory_id,
                        revision_id=None,
                        event_type=MemoryEventType.RELATION_RETRACTED,
                        actor_type=context.actor_type,
                        actor_ref=context.actor_ref,
                        stream_id=context.stream_id,
                        occurred_at=now,
                        payload_json={"relation_id": relation_id, "reason": reason},
                    )
                )

    async def _require_memories(
        self,
        session: AsyncSession,
        source_memory_id: str,
        target_memory_id: str,
    ) -> None:
        """确认关系两端的正式记忆存在。"""
        rows = list(
            (
                await session.scalars(
                    select(MemoryModel).where(
                        MemoryModel.memory_id.in_((source_memory_id, target_memory_id))
                    )
                )
            ).all()
        )
        if len(rows) != 2:
            raise ValueError("关系目标 Memory 不存在")
        if any(row.status is MemoryStatus.TOMBSTONED for row in rows):
            raise ValueError("TOMBSTONED Memory 不能建立新关系")

    async def _find_active_relation(
        self,
        session: AsyncSession,
        source_memory_id: str,
        target_memory_id: str,
        relation_type: RelationType,
    ) -> MemoryRelationModel | None:
        """按方向语义查找未撤回的重复关系。"""
        if relation_type in SYMMETRIC_RELATIONS:
            endpoints = or_(
                (MemoryRelationModel.source_memory_id == source_memory_id)
                & (MemoryRelationModel.target_memory_id == target_memory_id),
                (MemoryRelationModel.source_memory_id == target_memory_id)
                & (MemoryRelationModel.target_memory_id == source_memory_id),
            )
        else:
            endpoints = (
                (MemoryRelationModel.source_memory_id == source_memory_id)
                & (MemoryRelationModel.target_memory_id == target_memory_id)
            )
        statement = select(MemoryRelationModel).where(
            endpoints,
            MemoryRelationModel.relation_type == relation_type,
            MemoryRelationModel.retracted_at.is_(None),
        )
        return (await session.scalars(statement)).one_or_none()

    @staticmethod
    def _add_relation_events(
        session: AsyncSession,
        data: RelateMemoryInput,
        context: WriteContext,
        relation_id: str,
        occurred_at: datetime,
    ) -> None:
        """为关系两端追加 RELATED 事件。"""
        payload = {
            "relation_id": relation_id,
            "relation_type": data.relation_type.value,
            "operation_key": context.operation_key,
        }
        for memory_id in {data.source_memory_id, data.target_memory_id}:
            session.add(
                MemoryEventModel(
                    event_id=_new_id(),
                    memory_id=memory_id,
                    revision_id=None,
                    event_type=MemoryEventType.RELATED,
                    actor_type=context.actor_type,
                    actor_ref=context.actor_ref,
                    stream_id=context.stream_id,
                    occurred_at=occurred_at,
                    payload_json=payload,
                )
            )


class MergeService:
    """将重复正式记忆合并到已有 Canonical Memory。"""

    def __init__(
        self,
        schema: VNextSchema,
        memory_service: MemoryService | None = None,
    ) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema
        self._memory_service = memory_service

    async def _find_operation_merge(
        self,
        session: AsyncSession,
        operation_key: str | None,
    ) -> tuple[str, tuple[str, ...]] | None:
        """按稳定动作 key 恢复已提交 Merge 的 Canonical 与关系。"""
        if operation_key is None:
            return None
        events = tuple(
            (
                await session.scalars(
                    select(MemoryEventModel).where(
                        MemoryEventModel.event_type == MemoryEventType.MERGED
                    )
                )
            ).all()
        )
        canonical_id: str | None = None
        relation_ids: list[str] = []
        for event in events:
            payload = event.payload_json
            if not isinstance(payload, dict) or payload.get("operation_key") != operation_key:
                continue
            canonical_id = canonical_id or (
                str(payload["canonical_memory_id"])
                if payload.get("canonical_memory_id")
                else None
            )
            if payload.get("relation_id"):
                relation_ids.append(str(payload["relation_id"]))
        if canonical_id is None or not relation_ids:
            return None
        return canonical_id, tuple(dict.fromkeys(relation_ids))

    async def merge(
        self,
        data: MergeMemoryInput,
        context: WriteContext,
    ) -> tuple[str, tuple[str, ...]]:
        """按输入模式合并到既有或新建的 Canonical Memory。"""
        data.validate()
        if data.mode == "EXISTING_CANONICAL":
            if data.canonical_memory_id is None:
                raise ValueError("EXISTING_CANONICAL 必须指定 canonical_memory_id")
            return data.canonical_memory_id, await self.merge_into_existing(data, context)
        if self._memory_service is None or data.new_memory is None:
            raise ValueError("NEW_CANONICAL 未装配 MemoryService")
        prepared_memory = replace(
            data.new_memory,
            evidence=await self._memory_service.prepare_evidence(data.new_memory.evidence),
        )
        data = replace(data, new_memory=prepared_memory)
        async with self._schema.database.session() as session:
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            completed = await _claim_domain_operation(
                session, context.operation_key, "MERGE"
            )
            if completed is not None:
                return (
                    str(completed["canonical_memory_id"]),
                    tuple(str(item) for item in completed.get("relation_ids", ())),
                )
            canonical_context = replace(
                context,
                operation_key=(
                    f"{context.operation_key}:canonical-create"
                    if context.operation_key is not None
                    else None
                ),
            )
            created = await self._memory_service.create_memory_in_session(
                session, data.new_memory, canonical_context
            )
            existing_mode = MergeMemoryInput(
                source_memory_ids=data.source_memory_ids,
                canonical_memory_id=created.memory_id,
                reason=data.reason,
            )
            relation_ids = await self._merge_into_existing_session(
                session, existing_mode, context, datetime.now(UTC)
            )
            await _complete_domain_operation(
                session,
                context.operation_key,
                {
                    "canonical_memory_id": created.memory_id,
                    "relation_ids": list(relation_ids),
                },
            )
        return created.memory_id, relation_ids

    async def _merge_into_existing_session(
        self,
        session: AsyncSession,
        data: MergeMemoryInput,
        context: WriteContext,
        now: datetime,
    ) -> tuple[str, ...]:
        """在调用方事务中完成 Canonical Merge 的全部关系、状态和事件写入。"""
        canonical = await session.get(MemoryModel, data.canonical_memory_id)
        if canonical is None or canonical.status is not MemoryStatus.ACTIVE:
            raise ValueError("Canonical Memory 必须是 ACTIVE")
        sources = list(
            (
                await session.scalars(
                    select(MemoryModel).where(
                        MemoryModel.memory_id.in_(data.source_memory_ids)
                    )
                )
            ).all()
        )
        if len(sources) != len(data.source_memory_ids):
            raise ValueError("存在无效 Merge Source")
        if any(source.status is not MemoryStatus.ACTIVE for source in sources):
            raise ValueError("Merge Source 必须是 ACTIVE")
        relation_ids: list[str] = []
        for source in sources:
            if await self._would_create_cycle(session, source.memory_id, canonical.memory_id):
                raise ValueError("MERGED_INTO 不允许形成 Cycle")
            relation_id = _new_id()
            claim_result = await session.execute(
                update(MemoryModel)
                .where(
                    MemoryModel.memory_id == source.memory_id,
                    MemoryModel.status == MemoryStatus.ACTIVE,
                )
                .values(status=MemoryStatus.MERGED, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            if claim_result.rowcount != 1:
                raise ValueError("Merge Source 在合并前已被并发修改")
            source.status = MemoryStatus.MERGED
            source.updated_at = now
            relation_ids.append(relation_id)
            session.add(
                MemoryRelationModel(
                    relation_id=relation_id,
                    source_memory_id=source.memory_id,
                    target_memory_id=canonical.memory_id,
                    relation_type=RelationType.MERGED_INTO,
                    reason=data.reason,
                    created_at=now,
                    created_by_type=context.actor_type,
                    retracted_at=None,
                    retract_reason=None,
                )
            )
            session.add(
                MemoryEventModel(
                    event_id=_new_id(),
                    memory_id=source.memory_id,
                    revision_id=source.current_revision_id,
                    event_type=MemoryEventType.MERGED,
                    actor_type=context.actor_type,
                    actor_ref=context.actor_ref,
                    stream_id=context.stream_id,
                    occurred_at=now,
                    payload_json={
                        "canonical_memory_id": canonical.memory_id,
                        "relation_id": relation_id,
                        "reason": data.reason,
                        "operation_key": context.operation_key,
                    },
                )
            )
        session.add(
            MemoryEventModel(
                event_id=_new_id(),
                memory_id=canonical.memory_id,
                revision_id=canonical.current_revision_id,
                event_type=MemoryEventType.MERGED,
                actor_type=context.actor_type,
                actor_ref=context.actor_ref,
                stream_id=context.stream_id,
                occurred_at=now,
                payload_json={
                    "source_memory_ids": list(data.source_memory_ids),
                    "reason": data.reason,
                    "operation_key": context.operation_key,
                },
            )
        )
        return tuple(relation_ids)

    async def merge_into_existing(
        self,
        data: MergeMemoryInput,
        context: WriteContext,
    ) -> tuple[str, ...]:
        """保留来源历史并建立到既有 Canonical 的合并血缘。"""
        data.validate()
        if data.mode != "EXISTING_CANONICAL":
            raise ValueError("merge_into_existing 只接受 EXISTING_CANONICAL")
        now = datetime.now(UTC)
        relation_ids: list[str] = []
        async with self._schema.database.session() as session:
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            completed = await _claim_domain_operation(
                session, context.operation_key, "MERGE"
            )
            if completed is not None:
                return tuple(str(item) for item in completed.get("relation_ids", ()))
            canonical = await session.get(MemoryModel, data.canonical_memory_id)
            if canonical is None or canonical.status is not MemoryStatus.ACTIVE:
                raise ValueError("Canonical Memory 必须是 ACTIVE")
            sources = list(
                (
                    await session.scalars(
                        select(MemoryModel).where(
                            MemoryModel.memory_id.in_(data.source_memory_ids)
                        )
                    )
                ).all()
            )
            if len(sources) != len(data.source_memory_ids):
                raise ValueError("存在无效 Merge Source")
            if data.evidence_ids:
                if self._memory_service is None:
                    raise ValueError("关联 Merge Evidence 需要装配 MemoryService")
                if not canonical.current_revision_id:
                    raise ValueError("Canonical Memory 缺少当前 Revision")
                await self._memory_service._attach_evidence(
                    session,
                    canonical.current_revision_id,
                    (),
                    data.evidence_ids,
                    now,
                )
            if any(source.status is not MemoryStatus.ACTIVE for source in sources):
                existing = tuple(
                    (
                        await session.scalars(
                            select(MemoryRelationModel).where(
                                MemoryRelationModel.source_memory_id.in_(data.source_memory_ids),
                                MemoryRelationModel.target_memory_id
                                == data.canonical_memory_id,
                                MemoryRelationModel.relation_type == RelationType.MERGED_INTO,
                                MemoryRelationModel.retracted_at.is_(None),
                            )
                        )
                    ).all()
                )
                if len(existing) == len(data.source_memory_ids) and all(
                    source.status is MemoryStatus.MERGED for source in sources
                ):
                    return tuple(item.relation_id for item in existing)
                raise ValueError("Merge Source 必须是 ACTIVE")
            for source in sources:
                if await self._would_create_cycle(
                    session,
                    source.memory_id,
                    canonical.memory_id,
                ):
                    raise ValueError("MERGED_INTO 不允许形成 Cycle")
                relation_id = _new_id()
                relation_ids.append(relation_id)
                claim_result = await session.execute(
                    update(MemoryModel)
                    .where(
                        MemoryModel.memory_id == source.memory_id,
                        MemoryModel.status == MemoryStatus.ACTIVE,
                    )
                    .values(status=MemoryStatus.MERGED, updated_at=now)
                    .execution_options(synchronize_session=False)
                )
                if claim_result.rowcount != 1:
                    raise ValueError("Merge Source 在合并前已被并发修改")
                source.status = MemoryStatus.MERGED
                source.updated_at = now
                session.add(
                    MemoryRelationModel(
                        relation_id=relation_id,
                        source_memory_id=source.memory_id,
                        target_memory_id=canonical.memory_id,
                        relation_type=RelationType.MERGED_INTO,
                        reason=data.reason,
                        created_at=now,
                        created_by_type=context.actor_type,
                        retracted_at=None,
                        retract_reason=None,
                    )
                )
                session.add(
                    MemoryEventModel(
                        event_id=_new_id(),
                        memory_id=source.memory_id,
                        revision_id=source.current_revision_id,
                        event_type=MemoryEventType.MERGED,
                        actor_type=context.actor_type,
                        actor_ref=context.actor_ref,
                        stream_id=context.stream_id,
                        occurred_at=now,
                        payload_json={
                            "canonical_memory_id": canonical.memory_id,
                            "relation_id": relation_id,
                            "reason": data.reason,
                            "operation_key": context.operation_key,
                        },
                    )
                )
            session.add(
                MemoryEventModel(
                    event_id=_new_id(),
                    memory_id=canonical.memory_id,
                    revision_id=canonical.current_revision_id,
                    event_type=MemoryEventType.MERGED,
                    actor_type=context.actor_type,
                    actor_ref=context.actor_ref,
                    stream_id=context.stream_id,
                    occurred_at=now,
                    payload_json={
                        "source_memory_ids": list(data.source_memory_ids),
                        "reason": data.reason,
                        "operation_key": context.operation_key,
                    },
                )
            )
            await _complete_domain_operation(
                session,
                context.operation_key,
                {
                    "canonical_memory_id": data.canonical_memory_id,
                    "relation_ids": list(relation_ids),
                },
            )
        return tuple(relation_ids)

    async def _would_create_cycle(
        self,
        session: AsyncSession,
        source_memory_id: str,
        canonical_memory_id: str,
    ) -> bool:
        """检查 Canonical 的既有合并链是否最终指回 Source。"""
        current_id = canonical_memory_id
        visited: set[str] = set()
        while current_id not in visited:
            if current_id == source_memory_id:
                return True
            visited.add(current_id)
            statement = select(MemoryRelationModel.target_memory_id).where(
                MemoryRelationModel.source_memory_id == current_id,
                MemoryRelationModel.relation_type == RelationType.MERGED_INTO,
                MemoryRelationModel.retracted_at.is_(None),
            )
            next_id = (await session.scalars(statement)).one_or_none()
            if next_id is None:
                return False
            current_id = next_id
        return True
