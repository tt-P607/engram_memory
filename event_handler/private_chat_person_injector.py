"""engram_memory 私聊人物认知注入器。

在 ON_PROMPT_BUILD 事件中，当 chat_type 为 private 时，从当前消息
提取对话对象的平台与 sender_id，反查 PersonInfo 并格式化人物认知
注入流私有 actor bucket（FIXED，缓存友好）。群聊不注入。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import log_api, prompt_api
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import EventType
from src.core.prompt import SystemReminderConsumeType, SystemReminderInsertType
from src.kernel.event import EventDecision

if TYPE_CHECKING:
    from src.core.models.sql_alchemy import ChatStreams, PersonInfo

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

    def _get_stream_id(self, params: dict[str, Any]) -> str:
        """从事件参数提取流 ID。"""
        values = params.get("values") or {}
        return str(values.get("stream_id") or "").strip()

    async def _get_stream_identity(
        self, stream_id: str
    ) -> tuple[str, str, str] | None:
        """查 ChatStreams 拿 (chat_type, platform, person_id)。

        事件参数不含 message，从流记录取对话对象。
        person_id 为哈希（sha256），需反查 PersonInfo 拿 platform/user_id。
        """
        try:
            from src.app.plugin_system.api import database_api
            from src.core.models.sql_alchemy import ChatStreams

            row: ChatStreams | None = await database_api.get_by(
                ChatStreams, stream_id=stream_id
            )
            if not row:
                return None
            chat_type = str(row.chat_type or "").strip()
            person_id = str(row.person_id or "").strip()
            platform = str(row.platform or "").strip()
            if not chat_type or not person_id:
                return None
            return chat_type, platform, person_id
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"查询流身份失败 stream={stream_id}: {exc}")
            return None

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 ON_PROMPT_BUILD 事件。"""
        stream_id = self._get_stream_id(params)
        if not stream_id:
            return EventDecision.SUCCESS, params

        identity = await self._get_stream_identity(stream_id)
        if identity is None:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        chat_type, platform, hashed_person_id = identity

        # 仅私聊注入
        if chat_type != "private":
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 哈希 person_id → 反查 PersonInfo（主键即哈希）
        try:
            from src.app.plugin_system.api import database_api
            from src.core.models.sql_alchemy import PersonInfo

            person: PersonInfo | None = await database_api.get_by(
                PersonInfo, person_id=hashed_person_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"反查人物失败 {hashed_person_id[:8]}: {exc}")
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        if person is None:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        sender_id = str(person.user_id or "").strip()
        platform = str(person.platform or platform or "").strip()
        if not sender_id:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 取相关记忆（轻量直查，不走 lookup_person 的完整链路，避免触发
        # 懒蒸馏等重操作阻塞 prompt 构建）
        related_memories: list[dict[str, Any]] = []
        try:
            from ..service.person_service import PersonService

            person_service = PersonService(self.plugin)
            related_memories = await person_service._search_person_memories(
                person_service._memory_service(None),
                f"{platform}:{sender_id}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"私聊注入获取相关记忆失败 {platform}:{sender_id}: {exc}")

        # 缺印象时调度后台蒸馏（不阻塞本次注入；完成后下次对话生效）
        if not str(person.impression or "").strip():
            try:
                from ..service.person_service import schedule_background_distill

                schedule_background_distill(self.plugin, platform, sender_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"调度后台蒸馏失败 {platform}:{sender_id}: {exc}")

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
        self,
        person: PersonInfo,
        related_memories: list[dict[str, Any]] | None = None,
    ) -> str:
        """格式化人物认知文本（含印象 + 相关记忆/关系线索）。"""
        nickname = str(person.nickname or "") or "未知用户"
        user_id = str(person.user_id or "").strip()
        platform = str(person.platform or "").strip()

        lines: list[str] = ["## 当前对话对象", f"昵称：{nickname}"]
        if user_id:
            if platform:
                lines.append(f"账号/ID：{platform}:{user_id}")
            else:
                lines.append(f"账号/ID：{user_id}")

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

        # 相关记忆（含与这个人有关的过往与关系线索；带 id 供 memory_read 查询）
        memories = related_memories or []
        if memories:
            entries = []
            for m in memories[:8]:
                title = str(m.get("title") or "").strip()
                mid = str(m.get("memory_id") or "").strip()
                if not title:
                    continue
                entries.append(f"{title}（{mid}）" if mid else title)
            if entries:
                lines.append("相关记忆：" + "；".join(entries))
                lines.append("（需要某条记忆详情时，可用 memory_read 按上方 id 查询）")

        return "\n".join(lines)

    @staticmethod
    def _nickname_history_text(person: PersonInfo) -> str:
        """生成历史昵称行文本；无历史时返回空字符串。"""
        history = str(person.nickname_history or "").strip()
        if not history:
            return ""
        import json

        try:
            entries = json.loads(history)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(entries, list):
            return ""
        names = [
            str(e.get("name") or "")
            for e in entries
            if isinstance(e, dict) and e.get("name")
        ]
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
