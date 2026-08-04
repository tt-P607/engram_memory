"""engram_memory person_lookup 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..service.person_service import PersonService


class PersonLookupTool(BaseTool):
    """查询人物认知 + 记忆索引目录。"""

    name = "person_lookup"
    description = (
        "查询一个人物的认知（昵称、历史昵称、印象、交互时间线）以及相关记忆索引目录。"
        "参数可以是 person_id（platform:user_id）或昵称。"
    )

    async def execute(
        self,
        query: Annotated[str, "人物标识：person_id 或 nickname"],
    ) -> tuple[bool, dict[str, Any]]:
        """执行人物查询。"""
        result = await PersonService(self.plugin).lookup_person(query)
        ok = bool(result.get("ok", True))
        return ok, result
