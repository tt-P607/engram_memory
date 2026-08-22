"""engram_memory 短期记忆后台总结逻辑。

周期任务从各聊天流读取自上次锚点以来的新消息，达到阈值时调用
LLM 总结为 1-3 条短期记忆并写入短期层（48h TTL），随后更新增量锚点。
单流失败只记录日志，不中断其他流。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import log_api, stream_api
from src.core.models.message import Message

from ..metrics import get_metrics
from ..prompts import SUMMARY_PROMPT, SUMMARY_PROMPT_NAME
from ..sub_agent import call_json_sub_agent, resolve_prompt
from .memory_service import MemoryService, normalize_person_id

if TYPE_CHECKING:
    from ..store import MemoryStore

logger = log_api.get_logger("engram_memory.short_term_summarizer")


def _message_time(message: Message) -> float:
    """获取消息时间戳（秒）。"""
    value = message.time
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _person_id_of(message: Message) -> str:
    """从消息提取人物 ID（platform:user_id），信息不足返回空。"""
    platform = str(message.platform or "").strip()
    sender_id = str(message.sender_id or "").strip()
    if not platform or not sender_id:
        return ""
    prefix = f"{platform}:"
    if sender_id.startswith(prefix):
        return sender_id
    return f"{prefix}{sender_id}"


def _person_name_of(message: Message) -> str:
    """从消息提取人物展示名称。"""
    name = str(message.sender_name or "").strip()
    if name:
        return name
    sender_id = str(message.sender_id or "").strip()
    return sender_id or "未知"


def _format_messages(messages: list[Message]) -> str:
    """将消息列表格式化为聊天记录文本（按时间先后）。

    说话人带 person_id（platform:user_id），使 LLM 总结时能直接从消息
    上下文提取真实人物 ID，而不是依赖昵称猜测。
    """
    lines: list[str] = []
    for message in messages:
        text = str(message.processed_plain_text or "").strip()
        if not text:
            continue
        sender = _person_name_of(message)
        person_id = _person_id_of(message)
        if person_id:
            lines.append(f"{sender}（{person_id}）: {text}")
        else:
            lines.append(f"{sender}: {text}")
    return "\n".join(lines)


async def _summarize_stream(
    plugin: Any,
    store: "MemoryStore",
    stream_id: str,
    config: Any,
    memory_service: MemoryService,
) -> int:
    """对单个聊天流执行一次短期总结。

    Args:
        plugin: 插件实例。
        store: 共享存储层。
        stream_id: 聊天流 ID。
        config: 插件配置。
        memory_service: 记忆服务实例。

    Returns:
        新写入的短期记忆条数。
    """
    anchors = await store.read_anchors()
    stream_anchor = anchors.get(stream_id, {})
    last_time = float(stream_anchor.get("last_message_time", 0.0) or 0.0)

    messages = await stream_api.get_stream_messages(stream_id, limit=300)
    # 过滤：时间晚于锚点，且非 bot 消息
    new_messages = [
        message
        for message in messages
        if _message_time(message) > last_time
        and str(message.sender_role or "").lower() != "bot"
    ]
    if not new_messages:
        return 0

    threshold = int(config.short_term.summarizer_message_threshold)
    if len(new_messages) < threshold:
        return 0

    # 从最旧的消息开始取 threshold*2 条作为本批（避免超长）。锚点推进到
    # 本批末尾：若积压超过批次上限，较新的消息留待下轮继续总结，
    # 不会被永久跳过
    if len(new_messages) > threshold * 2:
        batch = new_messages[: threshold * 2]
    else:
        batch = new_messages
    chat_flow = _format_messages(batch)
    if not chat_flow.strip():
        return 0

    # 流名称（群聊 group_name；私聊回退 stream_id 前 8 位）
    stream_name = f"[{stream_id[:8]}]"
    try:
        info = await stream_api.get_stream_info(stream_id)
        if info:
            stream_name = str(info.get("group_name") or "") or stream_name
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"获取流信息失败 {stream_id}: {exc}")

    system_prompt = resolve_prompt(SUMMARY_PROMPT_NAME, SUMMARY_PROMPT).format(
        bot_name="Engram Memory",
        stream_name=stream_name,
    )
    items, status = await call_json_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_short_term_summary",
        system=system_prompt,
        user=chat_flow,
        stream_id=stream_id,
    )
    if status == "error":
        # 调用/解析失败：不更新锚点，下轮自动重试本批消息
        get_metrics(plugin).incr("summarizer_parse_errors")
        return 0
    if not items:
        return 0

    # 本批真实人物集合（roster）：LLM 输出的 person_id 必须在此集合内，
    # 杜绝格式合法但事实错误（编造）的人物 ID 落库
    roster = {_person_id_of(m) for m in batch}
    roster.discard("")

    # 批次中间时间戳作为 event_time
    times = [_message_time(m) for m in batch]
    mid_time = sum(times) / len(times) if times else time.time()

    written = 0
    for item in items:
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title or not content:
            continue

        # person_id/related_people 写入前校验：格式合法 + 必须在本批 roster 内
        person_id = normalize_person_id(item.get("person_id"))
        if person_id and roster and person_id not in roster:
            logger.warning(f"短期总结剔除编造 person_id: {person_id}")
            person_id = None
        related_people: list[str] = []
        for raw in (item.get("related_people") or []):
            cleaned = normalize_person_id(raw)
            if not cleaned:
                continue
            if roster and cleaned not in roster:
                logger.warning(f"短期总结剔除编造 related_people: {cleaned}")
                continue
            if cleaned not in related_people:
                related_people.append(cleaned)
        try:
            await memory_service.write_memory(
                title=title,
                content=content,
                core_tags=[str(t) for t in (item.get("core_tags") or [])],
                diffusion_tags=[str(t) for t in (item.get("diffusion_tags") or [])],
                opposing_tags=[str(t) for t in (item.get("opposing_tags") or [])],
                event_time=mid_time,
                layer="short_term",
                person_id=person_id,
                related_people=related_people or None,
                stream_id=stream_id,
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"短期记忆写入失败 stream={stream_id}: {exc}")

    # 更新锚点：推进到本批最后一条（未入批的积压消息保持待消费状态）
    if batch:
        last_message = batch[-1]
        anchors[stream_id] = {
            "last_message_time": _message_time(last_message),
            "last_message_id": str(last_message.message_id or ""),
        }
        await store.write_anchors(anchors)
    return written


async def summarize_short_term(
    plugin: Any,
    store: "MemoryStore",
    memory_service: MemoryService,
) -> dict[str, Any]:
    """对所有活跃流执行短期总结。

    Args:
        plugin: 插件实例。
        store: 共享存储层。
        memory_service: 记忆服务实例。

    Returns:
        含 scanned / written / errors 的统计字典。
    """
    config = plugin.config
    try:
        stream_ids = await stream_api.get_stream_ids_from_db()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"获取流列表失败: {exc}")
        return {"scanned": 0, "written": 0, "errors": 0}

    scanned = 0
    written = 0
    errors = 0
    for stream_id in stream_ids:
        try:
            scanned += 1
            written += await _summarize_stream(
                plugin, store, stream_id, config, memory_service
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.error(f"短期总结流失败 {stream_id}: {exc}")

    logger.info(f"短期总结完成: scanned={scanned} written={written} errors={errors}")
    metrics = get_metrics(plugin)
    metrics.incr("summarizer_scanned", scanned)
    metrics.incr("summarizer_written", written)
    metrics.incr("summarizer_errors", errors)
    return {"scanned": scanned, "written": written, "errors": errors}
