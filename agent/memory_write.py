"""engram_memory memory_write 工具。"""

from __future__ import annotations

from typing import Annotated, Any

from src.app.plugin_system.base import BaseTool

from ..service.memory_service import MemoryService


class MemoryWriteTool(BaseTool):
    """创建或更新记忆。"""

    name = "memory_write"
    description = "创建一条新记忆或更新已有记忆（传入 memory_id 时为更新）。重要信息请主动写入。"

    async def execute(
        self,
        title: Annotated[str, "记忆标题"],
        content: Annotated[str, "记忆完整内容"],
        core_tags: Annotated[list[str], "核心标签列表"],
        diffusion_tags: Annotated[list[str], "扩散标签列表"],
        opposing_tags: Annotated[list[str], "对立标签列表"],
        event_time: Annotated[float | None, "事件发生时间（Unix 时间戳），不填则用当前时间"] = None,
        layer: Annotated[str, "写入层级：short_term/active/archived，默认 active"] = "active",
        person_id: Annotated[str | None, "关联核心人物，格式 platform:user_id"] = None,
        related_people: Annotated[list[str] | None, "涉及的其他人物列表"] = None,
        relation_memory_ids: Annotated[list[str] | None, "关联的其他记忆 ID 列表"] = None,
        memory_id: Annotated[str | None, "已有记忆 ID，传入则为更新"] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """执行创建或更新。"""
        result = await MemoryService(self.plugin).write_memory(
            title=title,
            content=content,
            core_tags=core_tags,
            diffusion_tags=diffusion_tags,
            opposing_tags=opposing_tags,
            event_time=event_time,
            layer=layer,
            person_id=person_id,
            related_people=related_people,
            relation_memory_ids=relation_memory_ids,
            memory_id=memory_id,
            stream_id=self.get_current_stream_id(),
        )
        return True, {"action": "memory_write", "ok": True, **result}
