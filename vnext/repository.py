"""Engram Memory vNext 规范数据只读仓储。"""

from __future__ import annotations

from sqlalchemy import select

from .models import (
    EvidenceModel,
    MemoryAssessmentModel,
    MemoryEventModel,
    MemoryModel,
    MemoryRevisionModel,
    RevisionEvidenceModel,
)
from .schema import VNextSchema


class MemoryRepository:
    """提供不绕过领域约束的正式记忆读取能力。"""

    def __init__(self, schema: VNextSchema) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema

    async def get_memory(self, memory_id: str) -> MemoryModel | None:
        """按稳定标识读取记忆身份。"""
        async with self._schema.database.session() as session:
            return await session.get(MemoryModel, memory_id)

    async def get_current_revision(self, memory_id: str) -> MemoryRevisionModel | None:
        """读取正式记忆当前版本。"""
        statement = (
            select(MemoryRevisionModel)
            .join(MemoryModel, MemoryModel.current_revision_id == MemoryRevisionModel.revision_id)
            .where(MemoryModel.memory_id == memory_id)
        )
        async with self._schema.database.session() as session:
            return (await session.scalars(statement)).one_or_none()

    async def get_current_assessment(self, memory_id: str) -> MemoryAssessmentModel | None:
        """读取正式记忆当前认知评估。"""
        statement = (
            select(MemoryAssessmentModel)
            .join(
                MemoryModel,
                MemoryModel.current_assessment_id == MemoryAssessmentModel.assessment_id,
            )
            .where(MemoryModel.memory_id == memory_id)
        )
        async with self._schema.database.session() as session:
            return (await session.scalars(statement)).one_or_none()

    async def list_revisions(self, memory_id: str) -> tuple[MemoryRevisionModel, ...]:
        """按版本号升序读取完整 Revision 历史。"""
        statement = (
            select(MemoryRevisionModel)
            .where(MemoryRevisionModel.memory_id == memory_id)
            .order_by(MemoryRevisionModel.revision_no)
        )
        async with self._schema.database.session() as session:
            return tuple((await session.scalars(statement)).all())

    async def list_assessments(self, memory_id: str) -> tuple[MemoryAssessmentModel, ...]:
        """按创建时间升序读取认知评估历史。"""
        statement = (
            select(MemoryAssessmentModel)
            .where(MemoryAssessmentModel.memory_id == memory_id)
            .order_by(MemoryAssessmentModel.created_at, MemoryAssessmentModel.assessment_id)
        )
        async with self._schema.database.session() as session:
            return tuple((await session.scalars(statement)).all())

    async def list_evidence(self, memory_id: str) -> tuple[EvidenceModel, ...]:
        """读取正式记忆全部 Revision 关联的去重 Evidence。"""
        statement = (
            select(EvidenceModel)
            .join(RevisionEvidenceModel, RevisionEvidenceModel.evidence_id == EvidenceModel.evidence_id)
            .join(
                MemoryRevisionModel,
                MemoryRevisionModel.revision_id == RevisionEvidenceModel.revision_id,
            )
            .where(MemoryRevisionModel.memory_id == memory_id)
            .distinct()
            .order_by(EvidenceModel.created_at, EvidenceModel.evidence_id)
        )
        async with self._schema.database.session() as session:
            return tuple((await session.scalars(statement)).all())

    async def list_events(self, memory_id: str) -> tuple[MemoryEventModel, ...]:
        """按发生时间稳定排序读取追加式事件历史。"""
        statement = (
            select(MemoryEventModel)
            .where(MemoryEventModel.memory_id == memory_id)
            .order_by(MemoryEventModel.occurred_at, MemoryEventModel.event_id)
        )
        async with self._schema.database.session() as session:
            return tuple((await session.scalars(statement)).all())
