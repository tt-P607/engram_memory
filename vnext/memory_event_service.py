"""Engram Memory vNext 非认知观察事件领域服务。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from .domain import MemoryObservationEventInput, WriteContext
from .models import MemoryEventModel, MemoryModel, MemoryRevisionModel
from .schema import VNextSchema


class MemoryEventService:
    """记录不会改变 Memory 认知状态的观察事件。"""

    def __init__(self, schema: VNextSchema) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema

    async def record_observation(
        self,
        data: MemoryObservationEventInput,
        context: WriteContext,
    ) -> str:
        """追加观察事件并验证可选 Revision 属于目标 Memory。"""
        data.validate()
        event_id = str(uuid4())
        async with self._schema.database.session() as session:
            memory = await session.get(MemoryModel, data.memory_id)
            if memory is None:
                raise ValueError("Memory 不存在")
            if data.revision_id is not None:
                revision = await session.get(MemoryRevisionModel, data.revision_id)
                if revision is None or revision.memory_id != data.memory_id:
                    raise ValueError("Revision 不属于目标 Memory")
            session.add(
                MemoryEventModel(
                    event_id=event_id,
                    memory_id=data.memory_id,
                    revision_id=data.revision_id,
                    event_type=data.event_type,
                    actor_type=context.actor_type,
                    actor_ref=context.actor_ref,
                    stream_id=context.stream_id,
                    occurred_at=datetime.now(UTC),
                    payload_json=data.payload,
                )
            )
        return event_id