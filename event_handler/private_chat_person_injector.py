"""engram_memory 私聊人物认知注入器。

在 ON_PROMPT_BUILD 事件中，当 chat_type 为 private 时，从当前消息
提取对话对象的平台与 sender_id，反查 PersonInfo 并格式化人物认知
注入流私有 actor bucket（FIXED，缓存友好）。群聊不注入。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.app.plugin_system.api import log_api, person_api, prompt_api
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import EventType
from src.core.models.message import Message
from src.core.prompt import SystemReminderConsumeType, SystemReminderInsertType
from src.kernel.event import EventDecision

logger = log_api.get_logger("engram_memory.private_chat_person_injector")


def _format_date(timestamp: float) -> str:
    """将时间戳格式化为 YYYY-MM-DD；非法时返回空字符串。"""
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except (OSError, ValueError, OverflowError):
        return ""


class PrivateChatPersonInjector(BaseEventHandler):
    """私聊人物认知注入。"""

    name = "private_chat_person_injector"
    description = "私聊时注入当前对话对象的人物认知（FIXED）"
    weight = 30
    init_subscribe: list[EventType | str] = [EventType.ON_PROMPT_BUILD]
    _REMINDER_NAME = "engram_memory_person"

    def _journal_enabled(self) -> bool:
        """判断日记回顾是否启用（人物认知印象由日记回顾更新）。"""
        from ..config import EngramMemoryConfig

        config = self.plugin.config
        if isinstance(config, EngramMemoryConfig):
            return bool(config.journal.enabled)
        return True

    def _get_msg(self, params: dict[str, Any]) -> Any | None:
        """从事件参数提取触发消息。"""
        values = params.get("values") or {}
        return values.get("message")

    def _get_stream_id(self, params: dict[str, Any]) -> str:
        """从事件参数提取流 ID。"""
        values = params.get("values") or {}
        return str(values.get("stream_id") or "").strip()

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 ON_PROMPT_BUILD 事件。"""
        stream_id = self._get_stream_id(params)
        if not stream_id:
            return EventDecision.SUCCESS, params

        message = self._get_msg(params)
        if message is None:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        if not isinstance(message, Message):
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 仅私聊注入；日记回顾关闭时人物认知（含印象）不注入
        if not self._journal_enabled():
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        chat_type = str(message.chat_type or "").strip()
        if chat_type != "private":
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        platform = str(message.platform or "").strip()
        sender_id = str(message.sender_id or "").strip()
        if not platform or not sender_id:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        try:
            person = await person_api.get_person(platform, sender_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"读取人物信息失败 {platform}:{sender_id}: {exc}")
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        if person is None:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 复用 person_lookup 逻辑：取人物认知 + 相关记忆（含关系线索）
        related_memories: list[dict[str, Any]] = []
        try:
            from ..service.person_service import PersonService

            lookup = await PersonService(self.plugin).lookup_person(
                f"{platform}:{sender_id}"
            )
            if lookup.get("ok"):
                related_memories = list(lookup.get("memories") or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"私聊注入获取相关记忆失败 {platform}:{sender_id}: {exc}")

        content = self._format_content(person, related_memories)
        try:
            prompt_api.add_stream_reminder(
                stream_id=stream_id,
                bucket="actor",
                name=self._REMINDER_NAME,
                content=content,
                insert_type=SystemReminderInsertType.FIXED,
                consume=SystemReminderConsumeType.FOREVER,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"写入人物认知 reminder 失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
        return EventDecision.SUCCESS, params

    def _format_content(
        self, person: Any, related_memories: list[dict[str, Any]] | None = None
    ) -> str:
        """格式化人物认知文本（含印象 + 相关记忆/关系线索）。"""
        nickname = str(person.nickname or "") or "未知用户"

        lines: list[str] = ["## 当前对话对象", f"昵称：{nickname}"]

        # 历史昵称
        history = self._nickname_history_text(person)
        if history:
            lines.append(history)

        # 交互时间线
        first_date = _format_date(float(person.first_interaction or 0.0))
        last_date = _format_date(float(person.last_interaction or 0.0))
        if first_date:
            lines.append(f"认识于：{first_date}")
        if last_date:
            lines.append(f"最近活跃：{last_date}")

        # 印象
        impression = str(person.impression or "").strip()
        lines.append(f"印象：{impression if impression else '暂无印象'}")

        # 相关记忆（含与这个人有关的过往与关系线索）
        memories = related_memories or []
        if memories:
            titles = [str(m.get("title") or "").strip() for m in memories]
            titles = [t for t in titles if t]
            if titles:
                lines.append("相关记忆：" + "；".join(titles[:8]))

        return "\n".join(lines)

    @staticmethod
    def _nickname_history_text(person: Any) -> str:
        """生成历史昵称行文本；无历史时返回空字符串。"""
        try:
            history = person.nickname_history or ""
        except AttributeError:
            return ""
        if not history:
            return ""
        import json

        try:
            entries = json.loads(history)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(entries, list):
            return ""
        names = [str(e.get("name") or "") for e in entries if isinstance(e, dict) and e.get("name")]
        names = list(dict.fromkeys(names))
        if not names:
            return ""
        return (
            "历史曾用名："
            + "、".join(names)
            + "（以上为该用户曾使用的 QQ 昵称，框架自动追踪，仅供参考，不代表当前身份）"
        )

    def _clear(self, stream_id: str) -> None:
        """非私聊或无人物时删除该 reminder。"""
        try:
            prompt_api.delete_stream_reminder(stream_id, "actor", self._REMINDER_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"清理人物认知 reminder 失败 stream={stream_id}: {exc}")
