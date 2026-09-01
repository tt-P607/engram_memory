"""Engram Memory vNext 人物印象派生领域服务。"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .domain import PersonaUpdateInput, PersonaUpdateResult, WriteContext
from .enums import ActorType, MemoryStatus, SleepSessionStatus
from .models import (
    MemoryModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    PersonaUpdateLogModel,
    PersonaUpdateMemoryModel,
    PersonPersonaModel,
    SleepSessionModel,
)
from .schema import VNextSchema

EMPTY_CONTENT_HASH = sha256(b"").hexdigest()


def _content_hash(text_value: str) -> str:
    """计算规范化人物印象正文的稳定摘要。"""
    return sha256(text_value.encode("utf-8")).hexdigest()


class PersonaService:
    """管理由正式 Memory 单向派生的人物印象及其审计记录。"""

    def __init__(self, schema: VNextSchema, max_length: int) -> None:
        """绑定 vNext Schema 并设置人物印象最大长度。"""
        if max_length <= 0:
            raise ValueError("persona max_length 必须大于 0")
        self._schema = schema
        self._max_length = max_length

    async def get_persona(self, person_id: str) -> PersonPersonaModel | None:
        """按人物标识读取当前人物印象。"""
        if not person_id:
            raise ValueError("Persona 查询必须指定 person_id")
        async with self._schema.database.session() as session:
            return await session.get(PersonPersonaModel, person_id)

    async def update_persona(
        self,
        data: PersonaUpdateInput,
        context: WriteContext,
    ) -> PersonaUpdateResult:
        """在完成的整理阶段后更新人物印象并写入审计关联。"""
        data.validate()
        self._require_writer(context)
        final_text = data.impression_text.strip()[: self._max_length]
        content_hash = _content_hash(final_text)
        async with self._schema.database.session() as session:
            sleep_session = await self._validate_sleep_context(session, data, context)
            memories = await self._load_relevant_memories(session, data)
            persona = await session.get(PersonPersonaModel, data.person_id)
            if persona is not None and persona.content_hash == content_hash:
                return PersonaUpdateResult(
                    person_id=data.person_id,
                    changed=False,
                    update_id=None,
                    content_hash=persona.content_hash,
                )
            now = datetime.now(UTC)
            old_hash = persona.content_hash if persona is not None else EMPTY_CONTENT_HASH
            inserted = False
            if persona is None:
                insert_result = await session.execute(
                    sqlite_insert(PersonPersonaModel)
                    .values(
                        person_id=data.person_id,
                        impression_text=final_text,
                        created_at=now,
                        updated_at=now,
                        last_sleep_session_id=(
                            data.sleep_session_id if sleep_session is not None else None
                        ),
                        content_hash=content_hash,
                    )
                    .prefix_with("OR IGNORE")
                )
                if insert_result.rowcount == 1:
                    inserted = True
                    persona = await session.get(PersonPersonaModel, data.person_id)
                    if persona is None:  # pragma: no cover - same transaction
                        raise ValueError("Persona 创建失败")
                else:
                    persona = await session.get(PersonPersonaModel, data.person_id)
                    if persona is None:  # pragma: no cover - same transaction
                        raise ValueError("Persona 并发创建失败")
                    old_hash = persona.content_hash
                    if persona.content_hash == content_hash:
                        return PersonaUpdateResult(
                            person_id=data.person_id,
                            changed=False,
                            update_id=None,
                            content_hash=content_hash,
                        )
            if not inserted:
                if persona is None:  # pragma: no cover - guarded above
                    raise ValueError("Persona 不存在")
                result = await session.execute(
                    update(PersonPersonaModel)
                    .where(
                        PersonPersonaModel.person_id == data.person_id,
                        PersonPersonaModel.content_hash == persona.content_hash,
                        PersonPersonaModel.updated_at == persona.updated_at,
                    )
                    .values(
                        impression_text=final_text,
                        updated_at=now,
                        last_sleep_session_id=(
                            data.sleep_session_id
                            if sleep_session is not None
                            else persona.last_sleep_session_id
                        ),
                        content_hash=content_hash,
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    current = await session.get(PersonPersonaModel, data.person_id)
                    if current is not None and current.content_hash == content_hash:
                        return PersonaUpdateResult(
                            person_id=data.person_id,
                            changed=False,
                            update_id=None,
                            content_hash=current.content_hash,
                        )
                    raise ValueError("Persona 在更新前已被并发修改")
                persona.impression_text = final_text
                persona.updated_at = now
                persona.last_sleep_session_id = (
                    data.sleep_session_id if sleep_session is not None else persona.last_sleep_session_id
                )
                persona.content_hash = content_hash
            update_id = str(uuid4())
            session.add(
                PersonaUpdateLogModel(
                    update_id=update_id,
                    person_id=data.person_id,
                    sleep_session_id=data.sleep_session_id,
                    old_content_hash=old_hash,
                    new_content_hash=content_hash,
                    reason=data.reason,
                    created_at=now,
                )
            )
            await session.flush()
            for memory in memories:
                session.add(
                    PersonaUpdateMemoryModel(
                        update_id=update_id,
                        memory_id=memory.memory_id,
                    )
                )
        return PersonaUpdateResult(
            person_id=data.person_id,
            changed=True,
            update_id=update_id,
            content_hash=content_hash,
        )

    @staticmethod
    def _require_writer(context: WriteContext) -> None:
        """确认更新由 Sleep Agent 或 Admin 执行。"""
        if context.actor_type not in {ActorType.SLEEP_AGENT, ActorType.ADMIN}:
            raise PermissionError("只有 SLEEP_AGENT 或 ADMIN 可以更新 Persona")

    async def _validate_sleep_context(
        self,
        session: AsyncSession,
        data: PersonaUpdateInput,
        context: WriteContext,
    ) -> SleepSessionModel | None:
        """校验 Sleep Agent 的完成会话约束。"""
        if context.actor_type is ActorType.SLEEP_AGENT and not data.sleep_session_id:
            raise ValueError("SLEEP_AGENT 更新 Persona 必须指定 sleep_session_id")
        if data.sleep_session_id is None:
            return None
        sleep_session = await session.get(SleepSessionModel, data.sleep_session_id)
        if sleep_session is None:
            raise ValueError("Sleep Session 不存在")
        if context.actor_type is ActorType.SLEEP_AGENT:
            if sleep_session.status not in {
                SleepSessionStatus.COMPLETED,
                SleepSessionStatus.PARTIAL,
            }:
                raise ValueError(
                    "Persona Review 必须绑定已完成的 Sleep Session（允许 PARTIAL）"
                )
        elif sleep_session.status is SleepSessionStatus.RUNNING:
            raise ValueError("Admin Persona 更新不能绑定 RUNNING Session")
        return sleep_session

    async def _load_relevant_memories(
        self,
        session: AsyncSession,
        data: PersonaUpdateInput,
    ) -> tuple[MemoryModel, ...]:
        """读取并校验人物相关正式记忆。"""
        memories = tuple(
            (
                await session.scalars(
                    select(MemoryModel).where(MemoryModel.memory_id.in_(data.memory_ids))
                )
            ).all()
        )
        if len(memories) != len(data.memory_ids):
            raise ValueError("Persona 引用了不存在的 Formal Memory")
        if any(memory.status is MemoryStatus.TOMBSTONED for memory in memories):
            raise ValueError("TOMBSTONED Memory 不能作为 Persona 更新依据")
        for memory in memories:
            revision = await session.get(MemoryRevisionModel, memory.current_revision_id)
            if revision is None:
                # 外键约束下正常不可达，防御 Doctor 场景的悬空指针
                raise ValueError("Persona 参考 Memory 的当前 Revision 不存在")  # pragma: no cover
            subject = await session.get(MemoryRevisionSubjectModel, revision.revision_id)
            participant_person_ids = set(
                (
                    await session.scalars(
                        select(MemoryRevisionParticipantModel.person_id).where(
                            MemoryRevisionParticipantModel.revision_id == revision.revision_id,
                            MemoryRevisionParticipantModel.person_id == data.person_id,
                        )
                    )
                ).all()
            )
            subject_matches = subject is not None and subject.person_id == data.person_id
            if not subject_matches and not participant_person_ids:
                raise ValueError("Persona 参考 Memory 与目标人物无关")
        return memories
