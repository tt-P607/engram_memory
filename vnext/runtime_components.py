"""Engram Memory vNext 的框架组件薄门面。

组件只负责把框架调用转换为 vNext Runtime Owner 调用，不在这里保存
Canonical 状态，也不复制领域规则。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.base import BaseEventHandler, BaseRouter, BaseService, BaseTool
from src.app.plugin_system.types import EventType

from .domain import (
    AssessmentInput,
    CreateMemoryInput,
    ParticipantInput,
    ReviseMemoryInput,
    SubjectInput,
)
from .enums import (
    ActorType,
    ConfidenceLevel,
    EventTimeOrigin,
    EventTimePrecision,
    MemoryKind,
    ParticipantKind,
    ParticipantRole,
    RevisionChangeReason,
    SalienceLevel,
    StabilityLevel,
    SubjectKind,
)
from .runtime_owner import VNextRuntimeOwner
from .tool_service import ToolContext


def _owner(plugin: Any) -> VNextRuntimeOwner:
    """取得插件实例持有的唯一 vNext Runtime Owner。"""
    runtime = getattr(plugin, "runtime_owner", None)
    if not isinstance(runtime, VNextRuntimeOwner):
        raise RuntimeError("Engram vNext Runtime 尚未初始化")
    return runtime


def _actor_context(tool: BaseTool) -> ToolContext:
    """从框架绑定的当前消息构造不可伪造的 Actor 上下文。"""
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


def _text(value: object, field_name: str, *, default: str | None = None) -> str:
    """读取非空文本字段。"""
    if value is None and default is not None:
        return default
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


def _datetime(value: object, field_name: str, default: datetime | None = None) -> datetime:
    """解析 ISO 时间并统一为带时区 UTC。"""
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} 不能为空")
        return default
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
                role=_enum(
                    ParticipantRole,
                    item.get("role"),
                    "participant role",
                    ParticipantRole.PARTICIPANT,
                ),
            )
        )
    return tuple(result)


def _create_input(payload: dict[str, object]) -> CreateMemoryInput:
    """把 Actor 的结构化写入参数转换为领域输入。"""
    now = datetime.now(UTC)
    return CreateMemoryInput(
        anchor_title=_text(payload.get("anchor_title"), "anchor_title"),
        title=_text(payload.get("title"), "title"),
        content=_text(payload.get("content"), "content"),
        memory_kind=_enum(MemoryKind, payload.get("memory_kind"), "memory_kind"),
        subject=_subject(payload),
        participants=_participants(payload),
        confidence=_enum(
            ConfidenceLevel,
            payload.get("confidence"),
            "confidence",
            ConfidenceLevel.MEDIUM,
        ),
        confidence_reason=_text(
            payload.get("confidence_reason"),
            "confidence_reason",
            default="Actor 明确要求保存该经历",
        ),
        observed_at=_datetime(payload.get("observed_at"), "observed_at", now),
        event_start_at=(
            _datetime(payload.get("event_start_at"), "event_start_at")
            if payload.get("event_start_at") is not None
            else None
        ),
        event_end_at=(
            _datetime(payload.get("event_end_at"), "event_end_at")
            if payload.get("event_end_at") is not None
            else None
        ),
        event_time_precision=_enum(
            EventTimePrecision,
            payload.get("event_time_precision"),
            "event_time_precision",
            EventTimePrecision.UNKNOWN,
        ),
        event_time_origin=_enum(
            EventTimeOrigin,
            payload.get("event_time_origin"),
            "event_time_origin",
            EventTimeOrigin.UNKNOWN,
        ),
        stability=_enum(
            StabilityLevel,
            payload.get("stability"),
            "stability",
            StabilityLevel.MEDIUM,
        ),
        salience=_enum(
            SalienceLevel,
            payload.get("salience"),
            "salience",
            SalienceLevel.MEDIUM,
        ),
        assessment_reason=_text(
            payload.get("assessment_reason"),
            "assessment_reason",
            default="Actor 主动保存的正式记忆",
        ),
    )


def _revise_input(payload: dict[str, object]) -> ReviseMemoryInput:
    """把显式纠错参数转换为线性 Revision 输入。"""
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
        confidence=_enum(ConfidenceLevel, payload.get("confidence"), "confidence"),
        confidence_reason=_text(payload.get("confidence_reason"), "confidence_reason"),
        observed_at=_datetime(payload.get("observed_at"), "observed_at", now),
        event_time_precision=_enum(
            EventTimePrecision,
            payload.get("event_time_precision"),
            "event_time_precision",
            EventTimePrecision.UNKNOWN,
        ),
        event_time_origin=_enum(
            EventTimeOrigin,
            payload.get("event_time_origin"),
            "event_time_origin",
            EventTimeOrigin.UNKNOWN,
        ),
        change_reason=RevisionChangeReason.EXPLICIT_CORRECTION,
        assessment=AssessmentInput(
            stability=_enum(
                StabilityLevel,
                payload.get("stability"),
                "stability",
                StabilityLevel.MEDIUM,
            ),
            salience=_enum(
                SalienceLevel,
                payload.get("salience"),
                "salience",
                SalienceLevel.MEDIUM,
            ),
            reason=_text(payload.get("assessment_reason"), "assessment_reason"),
        ),
        event_start_at=(
            _datetime(payload.get("event_start_at"), "event_start_at")
            if payload.get("event_start_at") is not None
            else None
        ),
        event_end_at=(
            _datetime(payload.get("event_end_at"), "event_end_at")
            if payload.get("event_end_at") is not None
            else None
        ),
        new_anchor_title=(
            str(payload["new_anchor_title"]).strip()
            if payload.get("new_anchor_title")
            else None
        ),
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
    description = "主动保存一条有证据的 vNext 正式记忆；不创建 Archived 层。"

    async def execute(
        self,
        payload: dict[str, object],
    ) -> tuple[bool, str | dict[str, object]]:
        """校验结构化输入并直接创建 Formal Memory。"""
        context = _actor_context(self)
        result = await _owner(self.plugin).tools.memory_write(
            _create_input(payload),
            context,
        )
        return True, result


class VNextMemoryReviseTool(BaseTool):
    """向主模型提供当前聊天明确纠错的受限 Revision。"""

    name = "memory_revise"
    description = "仅在当前聊天明确纠错时，基于当前 revision 创建新版本。"

    async def execute(
        self,
        payload: dict[str, object],
    ) -> tuple[bool, str | dict[str, object]]:
        """执行 EXPLICIT_CORRECTION Revision。"""
        result = await _owner(self.plugin).tools.memory_revise(
            _revise_input(payload),
            _actor_context(self),
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
    """在默认用户 Prompt 构建前注入统一 vNext Flashback。"""

    name = "vnext_flashback_injector"
    description = "在当前回复生成前执行有预算的 vNext 自然闪回。"
    init_subscribe = [EventType.ON_PROMPT_BUILD]
    timeout = 2.0

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """读取最近多轮语境并把闪回候选加入当前 Prompt extra。"""
        if params.get("name") not in {
            "default_chatter_user_prompt",
            "neo_default_chatter_user_prompt",
        }:
            return EventDecision.SUCCESS, params
        values = params.get("values")
        if not isinstance(values, dict):
            return EventDecision.SUCCESS, params
        stream_id = str(values.get("stream_id") or "").strip()
        if not stream_id:
            return EventDecision.SUCCESS, params
        try:
            owner = _owner(self.plugin)
            candidates = await owner.consume_flashback_prefetch(stream_id)
        except Exception:
            return EventDecision.SUCCESS, params
        blocks = [candidate.to_prompt_block() for candidate in candidates]
        if blocks:
            existing = str(values.get("extra") or "").strip()
            values["extra"] = "\n\n".join(item for item in (existing, *blocks) if item)
            params["values"] = values
        return EventDecision.SUCCESS, params


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
