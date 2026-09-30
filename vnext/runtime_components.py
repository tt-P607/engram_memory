"""Engram Memory vNext 的框架组件薄门面。

组件只负责把框架调用转换为 vNext Runtime Owner 调用，不在这里保存
Canonical 状态，也不复制领域规则。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import prompt_api, stream_api
from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.base import BaseEventHandler, BaseRouter, BaseService, BaseTool
from src.app.plugin_system.types import EventType

from .domain import (
    CreateMemoryInput,
    EvidenceInput,
    EvidenceMessageInput,
    ParticipantInput,
    ReviseMemoryInput,
    SubjectInput,
)
from .enums import (
    EvidenceSourceType,
    MemoryKind,
    ParticipantKind,
    SubjectKind,
)
from .tool_service import ToolContext

if TYPE_CHECKING:
    from .runtime_owner import VNextRuntimeOwner


def _owner(plugin: Any) -> VNextRuntimeOwner:
    """取得插件实例持有的唯一 vNext Runtime Owner。"""
    from .runtime_owner import VNextRuntimeOwner

    runtime = getattr(plugin, "runtime_owner", None)
    if not isinstance(runtime, VNextRuntimeOwner):
        raise RuntimeError("Engram vNext Runtime 尚未初始化")
    return runtime


def _actor_context(tool: BaseTool) -> ToolContext:
    """从框架绑定的当前消息构造不可伪造的 Actor 上下文。"""
    # 长会话可能持有重载前的工具，权限枚举和输入类型须从当前模块读取。
    from .enums import ActorType
    from .tool_service import ToolContext

    message = tool.trigger_message
    stream_id = tool.get_current_stream_id() or None
    message_id = str(getattr(message, "message_id", "") or "").strip()
    platform = str(getattr(message, "platform", "") or "").strip()
    sender_id = str(getattr(message, "sender_id", "") or "").strip()
    actor_ref = f"{platform}:{sender_id}" if platform and sender_id else sender_id or None
    evidence_ids = (message_id,) if message_id else ()
    return ToolContext(
        actor_type=ActorType.ACTOR,
        actor_ref=actor_ref,
        stream_id=stream_id,
        evidence_message_ids=evidence_ids,
    )


def _actor_message_value(message: object, field: str, default: object = None) -> object:
    """读取当前 Actor 消息字段，不把结构化内容转换成伪原文。"""
    if isinstance(message, Mapping):
        return message.get(field, default)
    return getattr(message, field, default)


def _actor_source_directory(
    available: dict[str, dict[str, object]], *, include_person_id: bool = False,
) -> list[dict[str, object]]:
    """生成可供 Actor 选择来源的消息目录。"""
    fields = ["message_id", "time", "sender", "content", "quoteable", "reason"]
    if include_person_id:
        fields.append("person_id")
    return [
        {field: message[field] for field in fields if field in message}
        for message in available.values()
    ]


async def _actor_source_context(
    tool: BaseTool, payload: dict[str, object], *, correction: bool = False,
) -> tuple[ToolContext | None, dict[str, object], dict[str, object] | None]:
    """验证 Actor 原话来源并生成中性正式记忆输入。"""
    from .runtime import message_to_encoder_message

    context = _actor_context(tool)
    if context.stream_id is None:
        return None, {"error": "当前聊天未绑定，无法保存记忆来源"}, None
    stream = await stream_api.get_stream(context.stream_id)
    messages = [*stream.context.history_messages, *stream.context.unread_messages] if stream else []
    if tool.trigger_message is not None:
        messages.append(tool.trigger_message)
    available: dict[str, dict[str, object]] = {}
    for message in messages:
        message_id_value = _actor_message_value(message, "message_id") or _actor_message_value(
            message, "id"
        )
        message_stream_value = _actor_message_value(message, "stream_id")
        if not isinstance(message_id_value, str) or not isinstance(message_stream_value, str):
            continue
        message_id = message_id_value.strip()
        message_stream_id = message_stream_value.strip()
        if not message_id or message_stream_id != context.stream_id:
            continue
        raw_content = _actor_message_value(message, "content")
        raw_time = _actor_message_value(message, "time")
        if isinstance(raw_time, datetime):
            display_time = raw_time.isoformat()
        elif isinstance(raw_time, (str, int, float)):
            display_time = str(raw_time)
        else:
            display_time = ""
        sender_id_value = _actor_message_value(message, "sender_id")
        sender_name_value = _actor_message_value(message, "sender_name")
        sender_cardname_value = _actor_message_value(message, "sender_cardname")
        sender_role_value = _actor_message_value(message, "sender_role")
        sender_id = sender_id_value.strip() if isinstance(sender_id_value, str) else ""
        sender_name = sender_name_value.strip() if isinstance(sender_name_value, str) else ""
        sender_cardname = (
            sender_cardname_value.strip()
            if isinstance(sender_cardname_value, str)
            else ""
        )
        sender_role = sender_role_value.strip() if isinstance(sender_role_value, str) else ""
        person_id_value = _actor_message_value(message, "person_id")
        if not person_id_value:
            extra = _actor_message_value(message, "extra")
            if isinstance(extra, Mapping):
                person_id_value = extra.get("person_id")
        person_id = person_id_value.strip() if isinstance(person_id_value, str) else ""
        is_bot = (
            sender_role.casefold() == "bot"
            or sender_id.casefold() == "bot"
            or person_id.casefold() == "bot"
        )
        sender = (
            "Bot"
            if is_bot
            else sender_name or sender_cardname or sender_id or "未知发言者"
        )
        base_source: dict[str, object] = {
            "message_id": message_id,
            "time": display_time,
            "sender": sender,
            "content": raw_content if isinstance(raw_content, str) else None,
            "quoteable": False,
            "reason": "Message.content 不是可逐字引用的纯文本" if not isinstance(raw_content, str) else "",
            "person_id": person_id or None,
            "speaker_is_bot": is_bot,
            "snapshot": None,
            "valid_message": False,
        }
        if not isinstance(raw_content, str) or not raw_content.strip():
            if isinstance(raw_content, str):
                base_source["reason"] = "Message.content 为空文本，不能逐字引用"
            available[message_id] = base_source
            continue
        try:
            adapted = message_to_encoder_message(message)
        except ValueError:
            base_source["reason"] = "消息缺少可持久化来源快照字段"
            available[message_id] = base_source
            continue
        if adapted.message_id != message_id or adapted.stream_id != context.stream_id:
            continue
        snapshot = dict(adapted.snapshot or {})
        processed_content = _actor_message_value(message, "processed_plain_text")
        snapshot.update(
            {
                "message_id": adapted.message_id,
                "stream_id": adapted.stream_id,
                "time": adapted.time.isoformat(),
                "content": raw_content,
                "processed_plain_text": (
                    processed_content if isinstance(processed_content, str) else raw_content
                ),
            }
        )
        reply_to = _actor_message_value(message, "reply_to")
        snapshot["reply_to"] = (
            str(reply_to).strip()
            if isinstance(reply_to, (str, int)) and str(reply_to).strip()
            else None
        )
        message_type = _actor_message_value(message, "message_type")
        message_type = getattr(message_type, "value", message_type)
        if message_type is None or isinstance(message_type, (str, int, float, bool)):
            snapshot["message_type"] = message_type
        available[message_id] = {
            **base_source,
            "time": adapted.time.isoformat(),
            "sender": "Bot" if (adapted.snapshot or {}).get("speaker_is_bot") else sender,
            "person_id": (adapted.snapshot or {}).get("person_id"),
            "speaker_is_bot": bool((adapted.snapshot or {}).get("speaker_is_bot")),
            "quoteable": True,
            "reason": "",
            "snapshot": snapshot,
            "valid_message": True,
            "observed_at": adapted.time,
        }

    if correction:
        requested = payload.get("source_message_ids")
        if not isinstance(requested, list) or not requested or any(
            not isinstance(value, str) or not value.strip() for value in requested
        ):
            return None, {
                "error": "请选择直接支持修订内容的原始消息 ID",
                "source_messages": _actor_source_directory(available, include_person_id=True),
            }, None
        source_ids = tuple(dict.fromkeys(value.strip() for value in requested))
        if any(value not in available or not available[value]["valid_message"] for value in source_ids):
            return None, {
                "error": "来源 ID 不属于当前聊天上下文，请使用返回目录中的准确 ID",
                "source_messages": _actor_source_directory(available, include_person_id=True),
            }, None
        subject = payload.get("subject")
        associations = [subject] if isinstance(subject, dict) else []
        participants = payload.get("participants")
        if isinstance(participants, list):
            associations.extend(item for item in participants if isinstance(item, dict))
        source_people = {available[value]["person_id"] for value in source_ids}
        if any(item.get("person_id") and item["person_id"] not in source_people for item in associations):
            return None, {
                "error": "人物关联必须使用已选来源目录中的准确 person_id；昵称不能充当人物 ID",
                "source_messages": _actor_source_directory(
                    available, include_person_id=True
                ),
            }, None
        source_ids = tuple(dict.fromkeys((*context.evidence_message_ids, *source_ids)))
        validated_payload = payload
        operation_payload = {
            key: value for key, value in payload.items() if key != "source_message_ids"
        }
    else:
        source_quotes = payload.get("source_quotes")
        source_directory = _actor_source_directory(available)
        if not isinstance(source_quotes, list) or not source_quotes:
            return None, {
                "error": "请从当前聊天来源目录选择逐字原文；source_quotes 可先传空数组取得目录",
                "source_messages": source_directory,
            }, None
        try:
            memory_kind = _enum(MemoryKind, payload.get("memory_kind"), "memory_kind")
        except ValueError:
            return None, {
                "error": "memory_kind 无效，请从工具 schema 的枚举中选择",
                "source_messages": source_directory,
            }, None
        checked_quotes: list[tuple[str, str, dict[str, object]]] = []
        seen_quotes: set[tuple[str, str]] = set()
        for quote in source_quotes:
            if not isinstance(quote, dict):
                return None, {
                    "error": "source_quotes 每项都须包含 message_id 与 exact_text",
                    "source_messages": source_directory,
                }, None
            message_id = quote.get("message_id")
            exact_text = quote.get("exact_text")
            normalized_id = message_id.strip() if isinstance(message_id, str) else ""
            source = available.get(normalized_id)
            if source is None:
                return None, {
                    "error": "message_id 不属于当前聊天，请从返回目录选择准确 ID",
                    "source_messages": source_directory,
                }, None
            if not source["quoteable"]:
                return None, {
                    "error": source["reason"] or "该消息不能用于逐字来源引用",
                    "source_messages": source_directory,
                }, None
            original_content = source["content"]
            if (
                not isinstance(exact_text, str)
                or not exact_text.strip()
                or not isinstance(original_content, str)
                or exact_text not in original_content
            ):
                return None, {
                    "error": "exact_text 必须是该消息 Message.content 中的非空连续原文",
                    "source_messages": source_directory,
                }, None
            quote_key = (normalized_id, exact_text)
            if quote_key not in seen_quotes:
                seen_quotes.add(quote_key)
                checked_quotes.append((normalized_id, exact_text, source))
        if not checked_quotes:
            return None, {
                "error": "至少选择一条非空的逐字原文",
                "source_messages": source_directory,
            }, None
        quoted_ids = tuple(dict.fromkeys(item[0] for item in checked_quotes))
        reply_context_ids: list[str] = []
        for message_id in quoted_ids:
            snapshot = available[message_id].get("snapshot")
            reply_to = snapshot.get("reply_to") if isinstance(snapshot, dict) else None
            reply_source = available.get(reply_to) if isinstance(reply_to, str) else None
            if reply_source is not None and reply_source["valid_message"]:
                reply_context_ids.append(reply_to)
        source_ids = tuple(dict.fromkeys((*quoted_ids, *reply_context_ids)))
        source_snapshots = {
            message_id: available[message_id]["snapshot"] for message_id in source_ids
        }
        observed_at = max(
            source["observed_at"]
            for _, _, source in checked_quotes
            if isinstance(source.get("observed_at"), datetime)
        )
        validated_quotes = [
            {"message_id": message_id, "exact_text": exact_text}
            for message_id, exact_text, _ in checked_quotes
        ]
        title = checked_quotes[0][1][:60]
        content_lines = ["聊天原话；消息内的回复引用仍按原说话人理解："]
        for _, exact_text, source in checked_quotes:
            sender = str(source.get("sender") or "未知发言者")
            timestamp = str(source.get("time") or "")
            content_lines.append(f"- {timestamp}，{sender}的消息摘录：「{exact_text}」")
        content = "\n".join(content_lines)
        participants_by_id: dict[str, dict[str, object]] = {}
        for _, _, source in checked_quotes:
            person_id = source.get("person_id")
            if (
                isinstance(person_id, str)
                and person_id.strip()
                and person_id.casefold() != "bot"
                and not source.get("speaker_is_bot")
            ):
                participants_by_id.setdefault(
                    person_id,
                    {
                        "participant_kind": ParticipantKind.OTHER.value,
                        "person_id": person_id,
                        "label": "来源发言账号",
                    },
                )
        validated_payload = {
            "title": title,
            "content": content,
            "memory_kind": memory_kind.value,
            "subject": {"subject_kind": SubjectKind.UNKNOWN.value},
            "participants": list(participants_by_id.values()),
            "source_quotes": validated_quotes,
            "source_message_ids": source_ids,
            "source_snapshots": source_snapshots,
            "observed_at": observed_at,
            "_source_directory": source_directory,
        }
        operation_payload = {
            "memory_kind": memory_kind.value,
            "source_quotes": validated_quotes,
        }
    operation = json.dumps(
        {"stream_id": context.stream_id, "sources": sorted(source_ids),
         "payload": operation_payload},
        ensure_ascii=False, sort_keys=True,
    )
    return replace(
        context, evidence_message_ids=source_ids,
        operation_key="actor-memory:" + sha256(operation.encode()).hexdigest(),
    ), {}, validated_payload


def _text(value: object, field_name: str) -> str:
    """读取非空文本字段。"""
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} 不能为空")
    return result


def _enum(enum_type: type[Any], value: object, field_name: str, default: Any = None) -> Any:
    """把 JSON 字段转换为固定领域枚举。"""
    if value is None and default is not None:
        return default
    try:
        return enum_type(str(value))
    except ValueError as error:
        raise ValueError(f"{field_name} 值无效") from error


def _datetime(value: object, field_name: str) -> datetime:
    """解析 ISO 时间并统一为带时区 UTC。"""
    if value is None:
        raise ValueError(f"{field_name} 不能为空")
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} 必须包含时区")
    return parsed.astimezone(UTC)


def _optional_datetime(value: object, field_name: str) -> datetime | None:
    """解析可选的带时区 ISO 时间或 datetime。"""
    if value is None:
        return None
    return _datetime(value, field_name)


def _subject(payload: dict[str, object]) -> SubjectInput:
    """从工具 JSON 构造 Subject。"""
    from .domain import SubjectInput
    from .enums import SubjectKind

    raw = payload.get("subject")
    data = raw if isinstance(raw, dict) else payload
    return SubjectInput(
        subject_kind=_enum(
            SubjectKind,
            data.get("subject_kind"),
            "subject_kind",
            SubjectKind.UNKNOWN,
        ),
        person_id=str(data["person_id"]).strip() if data.get("person_id") else None,
        subject_key=str(data["subject_key"]).strip() if data.get("subject_key") else None,
        subject_label=str(data["subject_label"]).strip() if data.get("subject_label") else None,
    )


def _participants(payload: dict[str, object]) -> tuple[ParticipantInput, ...]:
    """从工具 JSON 构造可选参与者。"""
    from .domain import ParticipantInput
    from .enums import ParticipantKind

    raw = payload.get("participants")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("participants 必须是对象数组")
    result: list[ParticipantInput] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("participants 的每项必须是对象")
        result.append(
            ParticipantInput(
                participant_kind=_enum(
                    ParticipantKind,
                    item.get("participant_kind"),
                    "participant_kind",
                    ParticipantKind.OTHER,
                ),
                person_id=str(item["person_id"]).strip() if item.get("person_id") else None,
                label=str(item["label"]).strip() if item.get("label") else None,
            )
        )
    return tuple(result)


_MEMORY_SUBJECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "subject_kind": {
            "type": "string",
            "enum": [member.value for member in SubjectKind],
        },
        "person_id": {"type": "string"},
        "subject_key": {"type": "string"},
        "subject_label": {"type": "string"},
    },
    "required": ["subject_kind"],
    "additionalProperties": False,
}
_MEMORY_PARTICIPANT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "participant_kind": {
            "type": "string",
            "enum": [member.value for member in ParticipantKind],
        },
        "person_id": {"type": "string"},
        "label": {"type": "string"},
    },
    "required": ["participant_kind"],
    "additionalProperties": False,
}
_MEMORY_WRITE_PAYLOAD_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "memory_kind": {
            "type": "string",
            "description": "选择最符合所选原话类型的记忆类型。",
            "enum": [member.value for member in MemoryKind],
        },
        "source_quotes": {
            "type": "array",
            "description": "当前聊天原始消息中的逐字引用；传空数组可取得可引用来源目录。",
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "exact_text": {
                        "type": "string",
                        "description": "该消息 Message.content 中非空且连续的原文子串。",
                    },
                },
                "required": ["message_id", "exact_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memory_kind", "source_quotes"],
    "additionalProperties": False,
}
_MEMORY_REVISE_PAYLOAD_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "memory_id": {"type": "string"},
        "based_on_revision_id": {"type": "string"},
        "source_message_ids": {
            "type": "array", "items": {"type": "string"},
            "description": "支持更正的当前聊天消息 ID；当前明确纠错消息会一并保存。",
        },
        "title": {"type": "string"},
        "content": {"type": "string"},
        "memory_kind": {
            "type": "string",
            "enum": [member.value for member in MemoryKind],
        },
        "subject": _MEMORY_SUBJECT_SCHEMA,
        "participants": {
            "type": "array",
            "items": _MEMORY_PARTICIPANT_SCHEMA,
        },
    },
    "required": [
        "memory_id",
        "based_on_revision_id",
        "title",
        "content",
        "memory_kind",
        "subject",
    ],
    "additionalProperties": False,
}


def _create_input(
    payload: dict[str, object], *, observed_at: datetime,
    evidence: tuple[EvidenceInput, ...] = (),
) -> CreateMemoryInput:
    """把经来源校验的 Actor 输入转换为领域记忆。"""
    from .domain import CreateMemoryInput
    from .enums import MemoryKind

    return CreateMemoryInput(
        title=_text(payload.get("title"), "title"),
        content=_text(payload.get("content"), "content"),
        memory_kind=_enum(MemoryKind, payload.get("memory_kind"), "memory_kind"),
        subject=_subject(payload),
        participants=_participants(payload),
        observed_at=observed_at,
        evidence=evidence,
    )


def _revise_input(payload: dict[str, object]) -> ReviseMemoryInput:
    """把显式纠错参数转换为线性 Revision 输入。"""
    from .domain import ReviseMemoryInput
    from .enums import MemoryKind, RevisionChangeReason

    now = datetime.now(UTC)
    return ReviseMemoryInput(
        memory_id=_text(payload.get("memory_id"), "memory_id"),
        based_on_revision_id=_text(
            payload.get("based_on_revision_id"), "based_on_revision_id"
        ),
        title=_text(payload.get("title"), "title"),
        content=_text(payload.get("content"), "content"),
        memory_kind=_enum(MemoryKind, payload.get("memory_kind"), "memory_kind"),
        subject=_subject(payload),
        participants=_participants(payload),
        observed_at=now,
        change_reason=RevisionChangeReason.EXPLICIT_CORRECTION,
    )


class VNextMemorySearchTool(BaseTool):
    """向主模型提供 vNext 全局混合记忆搜索。"""

    name = "memory_search"
    description = "搜索全局 vNext 正式记忆；可显式按人物、类型和时间筛选。"

    async def execute(
        self,
        query: str,
        person_ids: list[str] | None = None,
        memory_kinds: list[str] | None = None,
        limit: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[bool, str | dict[str, object]]:
        """执行混合检索并返回候选目录。"""
        owner = _owner(self.plugin)
        context = _actor_context(self)
        result = await owner.tools.memory_search(
            query,
            context,
            person_ids=tuple(person_ids or ()),
            memory_kinds=tuple(memory_kinds or ()),
            limit=limit,
            start_time=_optional_datetime(start_time, "start_time"),
            end_time=_optional_datetime(end_time, "end_time"),
        )
        return True, {"memories": list(result)}


class VNextMemoryReadTool(BaseTool):
    """向主模型提供正式记忆的 current/history/full 读取。"""

    name = "memory_read"
    description = "读取 vNext 正式记忆的当前状态、修订历史或完整审计视图。"

    async def execute(
        self,
        memory_id: str,
        view: str = "current",
    ) -> tuple[bool, str | dict[str, object]]:
        """读取一条正式记忆。"""
        result = await _owner(self.plugin).tools.memory_read(
            memory_id,
            view,
            _actor_context(self),
        )
        return True, result


class VNextMemoryWriteTool(BaseTool):
    """向主模型提供带证据的主动正式记忆写入。"""

    name = "memory_write"
    description = (
        "确认是新的长期记忆时，先用 memory_search 查同一人物或经历，再提交 memory_kind 和逐字可核对的 source_quotes。"
        "每条来源须使用当前聊天的准确 message_id 和 Message.content 连续原文子串；"
        "短答或指代依赖前文时，同时引用必要的提问或回复目标。正式记忆由系统按来源元数据和原文构造。"
        "source_quotes 传空数组可先查看来源目录。"
    )

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        """Expose the current formal-memory write fields to the LLM."""
        return {
            "type": "function",
            "function": {
                "name": f"tool-{cls.name}",
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {"payload": _MEMORY_WRITE_PAYLOAD_SCHEMA},
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(
        self,
        payload: dict[str, object],
    ) -> tuple[bool, str | dict[str, object]]:
        """校验结构化输入并直接创建 Formal Memory。"""
        context, error, validated_payload = await _actor_source_context(self, payload)
        if context is None or validated_payload is None:
            return False, error
        source_ids = validated_payload.get("source_message_ids")
        source_snapshots = validated_payload.get("source_snapshots")
        observed_at = validated_payload.get("observed_at")
        if (
            context.stream_id is None
            or not isinstance(source_ids, tuple)
            or not isinstance(source_snapshots, dict)
            or not isinstance(observed_at, datetime)
            or any(not isinstance(source_snapshots.get(message_id), dict) for message_id in source_ids)
        ):
            source_directory = validated_payload.get("_source_directory")
            return False, {
                "error": "所选原话缺少可保存的当前消息快照，请重新选择来源",
                "source_messages": source_directory if isinstance(source_directory, list) else [],
            }
        evidence = (
            EvidenceInput(
                source_type=EvidenceSourceType.ACTOR_WRITE,
                observed_at=observed_at,
                messages=tuple(
                    EvidenceMessageInput(
                        message_id=message_id,
                        stream_id=context.stream_id,
                        snapshot=source_snapshots[message_id],
                    )
                    for message_id in source_ids
                ),
                note=json.dumps(
                    validated_payload.get("source_quotes", []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        result = await _owner(self.plugin).tools.memory_write(
            _create_input(validated_payload, observed_at=observed_at, evidence=evidence),
            context,
        )
        return True, result


class VNextMemoryReviseTool(BaseTool):
    """向主模型提供当前聊天明确纠错的受限 Revision。"""

    name = "memory_revise"
    description = "仅在当前聊天明确纠错时，基于当前 revision 创建新版本。"

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        """Expose the complete linear-revision input without retired fields."""
        return {
            "type": "function",
            "function": {
                "name": f"tool-{cls.name}",
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {"payload": _MEMORY_REVISE_PAYLOAD_SCHEMA},
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(
        self,
        payload: dict[str, object],
    ) -> tuple[bool, str | dict[str, object]]:
        """执行 EXPLICIT_CORRECTION Revision。"""
        context, error, _ = await _actor_source_context(self, payload, correction=True)
        if context is None:
            return False, error
        result = await _owner(self.plugin).tools.memory_revise(
            _revise_input(payload),
            context,
        )
        return True, result


class VNextPersonLookupTool(BaseTool):
    """向主模型提供人物印象与近期正式记忆索引。"""

    name = "person_lookup"
    description = "查询人物的派生印象与按时间新鲜度排序的近期正式记忆。"

    async def execute(
        self,
        person_id: str,
    ) -> tuple[bool, str | dict[str, object]]:
        """查询人物信息。"""
        result = await _owner(self.plugin).tools.person_lookup(
            person_id,
            _actor_context(self),
        )
        return True, result


class VNextMemoryService(BaseService):
    """向其他插件暴露带上下文的 vNext Memory Service。"""

    name = "memory_service"
    description = "Engram Memory vNext 规范记忆服务。"

    async def search(
        self,
        query: str,
        context: ToolContext,
        limit: int | None = None,
    ) -> tuple[dict[str, object], ...]:
        """执行带权限上下文的全局混合检索。"""
        return await _owner(self.plugin).tools.memory_search(
            query,
            context,
            limit=limit,
        )

    async def read(
        self,
        memory_id: str,
        view: str,
        context: ToolContext,
    ) -> dict[str, object]:
        """读取正式记忆。"""
        return await _owner(self.plugin).tools.memory_read(memory_id, view, context)


class VNextMessageEventHandler(BaseEventHandler):
    """观察新消息并安排经历编码，不在消息事件中执行 LLM。"""

    name = "vnext_message_observer"
    description = "将新消息交给 vNext Experience Encoder 调度器。"
    init_subscribe = [EventType.ON_MESSAGE_RECEIVED]
    timeout = 1.0

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """登记消息并保留事件参数结构。"""
        message = params.get("message")
        if message is not None:
            _owner(self.plugin).observe_message(message)
        return EventDecision.SUCCESS, params


class VNextFlashbackEventHandler(BaseEventHandler):
    """将已想起的记忆注入聊天流的固定 SystemReminder。"""

    name = "vnext_flashback_injector"
    description = "在当前回复生成前执行有预算的 vNext 自然闪回。"
    init_subscribe = [EventType.ON_PROMPT_BUILD]
    timeout = 2.0

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """使用原生 fixed + forever 保留历史闪回并刷新失效内容。"""
        if params.get("name") not in {
            "default_chatter_user_prompt",
            "neo_default_chatter_user_prompt",
            "kfc_user_prompt",
        }:
            return EventDecision.SUCCESS, params
        values = params.get("values")
        if not isinstance(values, dict):
            return EventDecision.SUCCESS, params
        stream_id = str(values.get("stream_id") or "").strip()
        if not stream_id:
            return EventDecision.SUCCESS, params
        cue = str(values.get("content") or values.get("unreads") or "").strip()
        try:
            owner = _owner(self.plugin)
        except Exception:
            return EventDecision.SUCCESS, params
        candidates: tuple[object, ...] = ()
        if cue:
            try:
                candidates = await owner.consume_flashback_prefetch(stream_id)
            except Exception:
                pass
        await self._reconcile_stream_reminders(owner, stream_id, candidates)
        return EventDecision.SUCCESS, params

    async def _reconcile_stream_reminders(
        self,
        owner: VNextRuntimeOwner,
        stream_id: str,
        candidates: tuple[object, ...],
    ) -> None:
        """Refresh current revisions and remove invalid stream-local reminders."""
        prefix = "engram_memory_flashback_"
        tracked_names = self.plugin._flashback_reminder_streams.setdefault(
            stream_id, set()
        )
        valid_names: list[str] = []
        memory_ids: list[str] = []
        for name in tuple(tracked_names):
            memory_id = name.removeprefix(prefix)
            if memory_id == name or not memory_id.strip():
                try:
                    prompt_api.delete_stream_reminder(stream_id, "actor", name)
                except Exception:
                    continue
                tracked_names.discard(name)
                continue
            valid_names.append(name)
            memory_ids.append(memory_id)
        for candidate in candidates:
            memory_id = str(getattr(candidate, "memory_id", "") or "").strip()
            if memory_id:
                memory_ids.append(memory_id)
        normalized_ids = tuple(dict.fromkeys(memory_ids))
        if not normalized_ids:
            if not tracked_names:
                self.plugin._flashback_reminder_streams.pop(stream_id, None)
            return

        try:
            current = await owner.flashback.current_reminder_candidates(normalized_ids)
        except Exception:
            for name in valid_names:
                try:
                    prompt_api.delete_stream_reminder(stream_id, "actor", name)
                except Exception:
                    continue
            return

        for name in valid_names:
            memory_id = name.removeprefix(prefix)
            candidate = current.get(memory_id)
            if candidate is None:
                try:
                    prompt_api.delete_stream_reminder(stream_id, "actor", name)
                except Exception:
                    continue
                tracked_names.discard(name)
                continue
            self._upsert_stream_reminder(stream_id, name, candidate)

        for candidate in candidates:
            memory_id = str(getattr(candidate, "memory_id", "") or "").strip()
            refreshed = current.get(memory_id)
            if refreshed is None:
                continue
            name = f"{prefix}{memory_id}"
            self._upsert_stream_reminder(stream_id, name, refreshed)
            tracked_names.add(name)

        if not tracked_names:
            self.plugin._flashback_reminder_streams.pop(stream_id, None)

    def _upsert_stream_reminder(
        self,
        stream_id: str,
        name: str,
        candidate: Any,
    ) -> None:
        """Keep the original exposure time while replacing stale revision text."""
        rendered = prompt_api.get_stream_reminder(stream_id, "actor", names=[name])
        marker = f"[{name}]\n"
        previous = rendered[len(marker) :] if rendered.startswith(marker) else ""
        recalled_at, separator, previous_block = previous.partition("\n\n")
        current_block = candidate.to_prompt_block().replace(
            "<system_reminder", "&lt;system_reminder"
        ).replace("</system_reminder>", "&lt;/system_reminder&gt;")
        if separator and previous_block == current_block:
            return
        if not separator:
            recalled_at = f"想起这段往事的时间：{datetime.now(UTC).isoformat()}"
        prompt_api.add_stream_reminder(
            stream_id=stream_id,
            bucket="actor",
            name=name,
            content=f"{recalled_at}\n\n{current_block}",
            insert_type=prompt_api.SystemReminderInsertType.FIXED,
            consume=prompt_api.SystemReminderConsumeType.FOREVER,
        )


class VNextDoctorRouter(BaseRouter):
    """提供只读 vNext Doctor 健康检查路由。"""

    name = "vnext_doctor"
    description = "Engram Memory vNext Canonical/Derived 一致性检查。"
    custom_route_path = "/api/engram-vnext"

    def register_endpoints(self) -> None:
        """注册只读一致性检查端点。"""

        @self.app.get("/check")
        async def check() -> dict[str, object]:
            """返回当前 vNext Doctor 报告。"""
            doctor = _owner(self.plugin).doctor
            if doctor is None:
                raise RuntimeError("Engram vNext Doctor 尚未初始化")
            report = await doctor.check()
            return {
                "healthy": report.healthy,
                "issues": [
                    {
                        "code": issue.code,
                        "object_id": issue.object_id,
                        "repairable": issue.repairable,
                        "details": issue.details,
                    }
                    for issue in report.issues
                ],
            }


__all__ = [
    "VNextDoctorRouter",
    "VNextFlashbackEventHandler",
    "VNextMemoryReadTool",
    "VNextMemoryReviseTool",
    "VNextMemorySearchTool",
    "VNextMemoryService",
    "VNextMemoryWriteTool",
    "VNextMessageEventHandler",
    "VNextPersonLookupTool",
]
