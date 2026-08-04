"""engram_memory memory_search 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..service.memory_service import MemoryService


class MemorySearchTool(BaseTool):
    """语义检索记忆。"""

    name = "memory_search"
    description = "语义检索记忆，支持指定层级/人物/标签。当你需要回忆相关记忆时使用。"

    async def execute(
        self,
        query: Annotated[str, "搜索关键词或自然语言描述"],
        layer: Annotated[str, "检索层级：short_term/active/archived/all，默认 all"] = "all",
        person_id: Annotated[str | None, "按人物过滤，格式 platform:user_id"] = None,
        core_tags: Annotated[list[str] | None, "按核心标签过滤（可选）"] = None,
        top_n: Annotated[int, "返回结果数量，默认 10"] = 10,
    ) -> tuple[bool, dict[str, Any]]:
        """执行语义检索。"""
        service = MemoryService(self.plugin)
        results = await service.search_memories(
            query=query,
            layer=layer,
            person_id=person_id,
            core_tags=core_tags,
            top_n=top_n,
        )
        if not results:
            return True, {
                "action": "memory_search",
                "ok": True,
                "count": 0,
                "results": [],
                "hint": "未找到匹配记忆，可尝试更换关键词或降低查询具体度",
            }
        return True, {
            "action": "memory_search",
            "ok": True,
            "count": len(results),
            "results": results,
        }
