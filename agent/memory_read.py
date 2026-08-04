"""engram_memory memory_read 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..service.memory_service import MemoryService


class MemoryReadTool(BaseTool):
    """批量读取记忆全文。"""

    name = "memory_read"
    description = "按记忆 ID 批量读取全文，可传入单个或多个 ID。当 memory_search 命中的记忆需要查看完整内容时使用。"

    async def execute(
        self,
        memory_ids: Annotated[list[str], "记忆 ID 列表，可传入单个或多个"],
    ) -> tuple[bool, dict[str, Any]]:
        """执行批量读取。"""
        result = await MemoryService(self.plugin).read_memories(memory_ids)
        return True, {"action": "memory_read", "ok": True, **result}
