"""将已提交的 Memory 变化串行合并到人物印象刷新。"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from src.app.plugin_system.api import log_api

from .domain import MemoryChanged
from .framework_bridge import ManagedTaskHandle, cancel_managed_task, create_managed_task
from .persona_service import PersonaService
from .repository import MemoryRepository


logger = log_api.get_logger(
    "engram_memory.vnext.persona_updater",
    display="Engram 记忆",
    color=log_api.COLOR.CYAN,
)


class PersonaUpdater:
    """维护人物维度的待处理变化并运行唯一后台刷新任务。"""

    def __init__(
        self,
        service: PersonaService,
        repository: MemoryRepository,
    ) -> None:
        """绑定共享 Persona 服务和人物别名仓储。"""
        self._service = service
        self._repository = repository
        self._pending: dict[str, list[MemoryChanged]] = defaultdict(list)
        self._task: ManagedTaskHandle | None = None
        self._closed = False
        self.last_errors: dict[str, Exception] = {}

    async def enqueue(self, change: MemoryChanged) -> None:
        """按变化前后关联人物合并一次记忆变化。"""
        if self._closed:
            raise RuntimeError("PersonaUpdater 已关闭")
        for person_id in change.affected_person_ids:
            self._pending[person_id].append(change)
        if self._pending and (self._task is None or self._task.task is None or self._task.task.done()):
            self._task = create_managed_task(
                self._run(),
                name="engram_memory_persona_updater",
                daemon=True,
            )

    async def close(self) -> None:
        """取消并等待唯一后台刷新任务。"""
        self._closed = True
        task = self._task
        self._task = None
        if task is not None and task.task is not None and not task.task.done():
            cancel_managed_task(task.task_id)
            await asyncio.gather(task.task, return_exceptions=True)

    async def _run(self) -> None:
        """逐批刷新人物印象并保留失败状态。"""
        try:
            while self._pending and not self._closed:
                source_person_id, changes = self._pending.popitem()
                person_id = source_person_id
                try:
                    aliases = await self._repository.resolve_person_aliases(source_person_id)
                    person_id = next(
                        (item for item in aliases if ":" not in item),
                        source_person_id,
                    )
                    for alias in aliases:
                        changes.extend(self._pending.pop(alias, ()))
                    batch = tuple(dict.fromkeys(changes))
                    result = await self._service.refresh(person_id, batch)
                    self.last_errors.pop(person_id, None)
                    if result is None:
                        logger.warning(f"Persona 刷新未处理任何变化: {person_id}")
                except Exception as error:  # noqa: BLE001
                    self.last_errors[person_id] = error
                    logger.error(f"Persona 刷新失败 {person_id}: {error}")
        finally:
            self._task = None