"""engram_memory journal_read 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..config import EngramMemoryConfig
from ..store import shared_store


class JournalReadTool(BaseTool):
    """按日期范围/流名翻看日记。"""

    name = "journal_read"
    description = "按日期范围（可选按聊天流名称过滤）读取过去的日记，模拟翻日记本回溯历史。"

    async def execute(
        self,
        date_from: Annotated[str, "开始日期，格式 YYYY-MM-DD"],
        date_to: Annotated[str | None, "结束日期，默认为 date_from"] = None,
        stream_name: Annotated[str | None, "按聊天流名称过滤"] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """执行日记读取。"""
        plugin = self.plugin

        def _config_factory() -> EngramMemoryConfig:
            if isinstance(plugin.config, EngramMemoryConfig):
                return plugin.config
            return EngramMemoryConfig()

        store = shared_store(plugin, _config_factory)
        end = date_to or date_from
        items = await store.read_journal(date_from, end, stream_name)
        return True, {
            "action": "journal_read",
            "ok": True,
            "count": len(items),
            "items": items,
        }
