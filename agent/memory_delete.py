"""engram_memory memory_delete 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..service.memory_service import MemoryService


class MemoryDeleteTool(BaseTool):
    """软删除记忆（单条）。"""

    name = "memory_delete"
    description = "软删除一条记忆（is_deleted 标记，可恢复）。危险操作，仅支持单条删除。"

    async def execute(
        self,
        memory_id: Annotated[str, "要删除的记忆 ID"],
    ) -> tuple[bool, dict[str, Any]]:
        """执行软删除。"""
        result = await MemoryService(self.plugin).delete_memory(memory_id)
        return True, {"action": "memory_delete", **result}
