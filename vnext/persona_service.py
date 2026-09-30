"""Engram Memory vNext 人物印象派生领域服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.plugin_system.api import database_api, person_api
from src.app.plugin_system.api.message_api import PersonInfo

from .domain import PersonaUpdateInput, PersonaUpdateResult, WriteContext
from .enums import ActorType, MemoryStatus, SleepSessionStatus
from .models import (
    MemoryModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    PersonaUpdateLogModel,
    PersonaUpdateMemoryModel,
    SleepSessionModel,
)
from .repository import MemoryRepository
from .schema import VNextSchema

EMPTY_CONTENT_HASH = sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class PersonaSnapshot:
    """核心人物记录中当前印象的只读视图。"""

    person_id: str
    impression_text: str
    updated_at: datetime | None


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
        self._repository = MemoryRepository(schema)

    async def get_core_person(self, person_id: str) -> PersonInfo | None:
        """经公开数据库与人物 API 读取核心人物记录。"""
        if not person_id:
            raise ValueError("Persona 查询必须指定 person_id")
        if ":" in person_id:
            platform, user_id = person_id.split(":", 1)
            if not platform or not user_id:
                raise ValueError("人物平台身份不完整")
            return await person_api.get_person(platform, user_id)
        return await database_api.get_by(PersonInfo, person_id=person_id)

    async def get_persona(self, person_id: str) -> PersonaSnapshot | None:
        """只从核心 PersonInfo.impression 读取当前人物印象。"""
        person = await self.get_core_person(person_id)
        if person is None:
            return None
        return PersonaSnapshot(
            person_id=person.person_id,
            impression_text=person.impression or "",
            updated_at=(
                datetime.fromtimestamp(person.updated_at, UTC)
                if person.updated_at is not None else None
            ),
        )

    async def update_persona(
        self,
        data: PersonaUpdateInput,
        context: WriteContext,
    ) -> PersonaUpdateResult:
        """在完成的整理阶段后更新人物印象并写入审计关联。"""
        data.validate()
        self._require_writer(context)
        final_text = data.impression_text.strip()
        if len(final_text) > self._max_length:
            raise ValueError(f"人物印象超过 {self._max_length} 字，请重新凝练完整正文")
        content_hash = _content_hash(final_text)
        person = await self.get_core_person(data.person_id)
        if person is None:
            raise ValueError("核心人物记录不存在，不能另建人物印象")
        old_hash = _content_hash(person.impression or "")
        person_ids = await self._repository.resolve_person_aliases(data.person_id)
        async with self._schema.database.session() as session:
            await self._validate_sleep_context(session, data, context)
            memories = await self._load_relevant_memories(session, data, person_ids)
            if old_hash != content_hash:
                if not await person_api.update_user_impression(
                    person.platform, person.user_id, final_text,
                ):
                    raise ValueError("核心人物印象更新失败")
                reread = await self.get_persona(person.person_id)
                if reread is None or reread.impression_text != final_text:
                    raise ValueError("核心人物印象回读与写入不一致")
            update_id = self._append_review_log(
                session, data, person.person_id, memories, old_hash, content_hash,
            )
        return PersonaUpdateResult(
            person_id=person.person_id,
            changed=old_hash != content_hash,
            update_id=update_id,
            content_hash=content_hash,
        )

    async def record_unchanged_review(
        self, data: PersonaUpdateInput, context: WriteContext,
    ) -> None:
        """记录保持现有核心印象的审查，避免重复调查同一批正式记忆。"""
        self._require_writer(context)
        if not data.memory_ids or not data.reason.strip():
            raise ValueError("保持印象也须记录正式记忆依据和理由")
        person = await self.get_core_person(data.person_id)
        if person is None:
            raise ValueError("核心人物记录不存在")
        person_ids = await self._repository.resolve_person_aliases(data.person_id)
        current_hash = _content_hash(person.impression or "")
        async with self._schema.database.session() as session:
            await self._validate_sleep_context(session, data, context)
            memories = await self._load_relevant_memories(session, data, person_ids)
            self._append_review_log(
                session, data, person.person_id, memories, current_hash, current_hash,
            )

    @staticmethod
    def _append_review_log(
        session: AsyncSession, data: PersonaUpdateInput, person_id: str,
        memories: tuple[MemoryModel, ...], old_hash: str, new_hash: str,
    ) -> str:
        """追加审查摘要及其正式记忆关联，不保存另一份人物印象正文。"""
        update_id = str(uuid4())
        session.add(PersonaUpdateLogModel(
            update_id=update_id, person_id=person_id,
            sleep_session_id=data.sleep_session_id,
            old_content_hash=old_hash, new_content_hash=new_hash,
            reason=data.reason, created_at=datetime.now(UTC),
        ))
        for memory in memories:
            session.add(PersonaUpdateMemoryModel(
                update_id=update_id, memory_id=memory.memory_id,
            ))
        return update_id

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
        person_ids: tuple[str, ...],
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
                            MemoryRevisionParticipantModel.person_id.in_(person_ids),
                        )
                    )
                ).all()
            )
            subject_matches = subject is not None and subject.person_id in person_ids
            if not subject_matches and not participant_person_ids:
                raise ValueError("Persona 参考 Memory 与目标人物无关")
        return memories
