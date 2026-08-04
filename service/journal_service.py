"""engram_memory 日记回顾逻辑。

两步回顾：
- 第一步：分聊天流并行回顾（生成流日记 + 提取长期记忆写入 archived）。
- 第二步：所有流完成后串行执行三个子任务（跨流记忆关联建立 →
  人物印象更新 → 中期层整理）。

失败保护：单流失败只记录日志、不推进消费标记；整个流程（含第二步）
全部成功后才更新 ``.last_journal_review`` 时间戳，任一步失败则本次不更新。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import database_api, log_api, person_api, stream_api
from src.core.models.message import Message
from src.core.models.sql_alchemy import PersonInfo
from src.kernel.concurrency import get_task_manager

from ..prompts import (
    ACTIVE_REVIEW_PROMPT,
    ACTIVE_REVIEW_PROMPT_NAME,
    ASSOCIATE_PROMPT,
    ASSOCIATE_PROMPT_NAME,
    EXTRACT_PROMPT,
    EXTRACT_PROMPT_NAME,
    IMPRESSION_PROMPT,
    IMPRESSION_PROMPT_NAME,
    JOURNAL_PROMPT,
    JOURNAL_PROMPT_NAME,
    SHORT_TERM_REVIEW_PROMPT,
    SHORT_TERM_REVIEW_PROMPT_NAME,
)
from ..sub_agent import call_sub_agent, extract_json_array, resolve_prompt
from .memory_service import MemoryService
from .person_service import PersonService

if TYPE_CHECKING:
    from ..store import MemoryStore

logger = log_api.get_logger("engram_memory.journal_service")

# 第一步并行回顾的并发数上限
_STEP1_CONCURRENCY = 4


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
    """将消息列表格式化为聊天记录文本。"""
    lines: list[str] = []
    for message in messages:
        text = str(message.processed_plain_text or "").strip()
        if not text:
            continue
        sender = _person_name_of(message)
        lines.append(f"{sender}: {text}")
    return "\n".join(lines)


def _collect_roster(messages: list[Message]) -> dict[str, str]:
    """从消息收集本流人物清单（person_id -> 名字），排除 bot。"""
    roster: dict[str, str] = {}
    for message in messages:
        if str(message.sender_role or "").lower() == "bot":
            continue
        person_id = _person_id_of(message)
        if not person_id or person_id in roster:
            continue
        roster[person_id] = _person_name_of(message)
    return roster


def _format_roster(roster: dict[str, str]) -> str:
    """将人物清单格式化为 prompt 文本。"""
    if not roster:
        return "（本流人物清单为空）"
    return "\n".join(f"- {person_id}（{name}）" for person_id, name in roster.items())


def _validate_person_field(
    raw_value: Any,
    *,
    roster: dict[str, str],
    is_private: bool,
    private_person_id: str | None,
) -> str | None:
    """校验并清洗单个 person_id 字段。

    群聊流必须在本流 roster 内；私聊流必须等于该流唯一对应的 person_id。
    不满足时返回 None（该人物字段置空，杜绝编造 ID 落库）。

    Args:
        raw_value: LLM 返回的原始 person_id。
        roster: 本流人物清单（person_id -> 名字）。
        is_private: 是否为私聊流。
        private_person_id: 私聊流对应的唯一 person_id。

    Returns:
        通过校验的 person_id；否则 None。
    """
    value = str(raw_value or "").strip()
    if not value:
        return None
    if is_private:
        if private_person_id and value == private_person_id:
            return value
        return None
    if value in roster:
        return value
    return None


def _clean_tags(raw_value: Any) -> list[str]:
    """清洗标签列表（去空白、转小写、去重）。"""
    seen: set[str] = set()
    result: list[str] = []
    for tag in (raw_value or []):
        cleaned = str(tag).strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


async def journal_review(
    plugin: Any,
    store: "MemoryStore",
    memory_service: MemoryService,
    person_service: PersonService,
) -> dict[str, Any]:
    """执行一次完整日记回顾。

    Args:
        plugin: 插件实例。
        store: 共享存储层。
        memory_service: 记忆服务实例。
        person_service: 人物连接服务实例。

    Returns:
        含 streams_reviewed / new_memories / step 结果的统计字典。
    """
    config = plugin.config
    now = time.time()
    start_ts = await store.read_last_review() or (now - 24 * 3600.0)
    end_ts = now

    # 收集活跃流（last_active_time 距今 <= active_stream_hours）
    try:
        all_stream_ids = await stream_api.get_stream_ids_from_db()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"获取流列表失败: {exc}")
        return {"streams_reviewed": 0, "new_memories": 0, "error": str(exc)}

    active_streams: list[str] = []
    active_cutoff = now - int(config.journal.active_stream_hours) * 3600.0
    for stream_id in all_stream_ids:
        try:
            info = await stream_api.get_stream_info(stream_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"获取流信息失败 {stream_id}: {exc}")
            info = None
        if not info:
            continue
        last_active = float(info.get("last_active_time") or 0.0)
        if last_active >= active_cutoff:
            active_streams.append(stream_id)

    if not active_streams:
        # 无活跃流：仍视为一次成功回顾（避免下次又补偿）
        await store.write_last_review(end_ts)
        return {"streams_reviewed": 0, "new_memories": 0}

    # 第一步：分聊天流并行回顾
    tm = get_task_manager()
    batches = [
        active_streams[i : i + _STEP1_CONCURRENCY]
        for i in range(0, len(active_streams), _STEP1_CONCURRENCY)
    ]
    stream_results: list[dict[str, Any]] = []
    for batch in batches:
        results = await tm.gather(
            *(
                _review_stream(
                    plugin, store, memory_service, stream_id, start_ts, end_ts, config
                )
                for stream_id in batch
            ),
            return_exceptions=True,
            group_name="engram_memory_journal_step1",
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"第一步回顾流失败: {result}")
                continue
            if result:
                stream_results.append(result)

    all_new_memories: list[dict[str, Any]] = []
    for stream_result in stream_results:
        all_new_memories.extend(stream_result.get("new_memories", []))

    # 第二步：全局整理（串行四子任务）
    try:
        # 1. 短期记忆审查（所有未过期），有价值复制晋升 archived
        await _review_short_term(plugin, memory_service, config)
        # 2. 中期层增量审查（当天新增 active），promote→archived / discard→软删
        await _review_active_layer(plugin, memory_service, start_ts, end_ts, config)
        # 3. 每流内部记忆关联（不跨流）
        await _stream_association(plugin, memory_service, all_new_memories, config)
        # 4. 人物印象更新
        await _update_impressions(
            plugin, person_service, stream_results, all_new_memories, config
        )
    except Exception as exc:  # noqa: BLE001
        # 第二步失败：本次不更新 .last_journal_review，等待下轮重试
        logger.error(f"第二步综合回顾失败: {exc}", exc_info=True)
        return {
            "streams_reviewed": len(stream_results),
            "new_memories": len(all_new_memories),
            "error": str(exc),
        }

    # 全部成功 → 更新时间戳
    await store.write_last_review(end_ts)
    return {
        "streams_reviewed": len(stream_results),
        "new_memories": len(all_new_memories),
    }


# ------------------------------------------------------------------
# 第一步：单流回顾
# ------------------------------------------------------------------


async def _review_stream(
    plugin: Any,
    store: "MemoryStore",
    memory_service: MemoryService,
    stream_id: str,
    start_ts: float,
    end_ts: float,
    config: Any,
) -> dict[str, Any]:
    """单流回顾：拉历史 → 收集人物清单 → 生成流日记 → 存盘 → 提取长期记忆。

    返回 {"stream_id", "stream_name", "journal_summary", "new_memories"}。
    任一步失败抛异常，由调用方隔离。
    """
    messages = await stream_api.get_stream_messages(stream_id, limit=1000)
    window_messages = [
        message
        for message in messages
        if start_ts <= _message_time(message) <= end_ts
        and str(message.sender_role or "").lower() != "bot"
    ]
    if not window_messages:
        return {
            "stream_id": stream_id,
            "stream_name": "",
            "journal_summary": "",
            "new_memories": [],
        }

    roster = _collect_roster(window_messages)

    # 流信息与流名称
    info = await stream_api.get_stream_info(stream_id)
    chat_type = str((info or {}).get("chat_type") or "")
    is_private = chat_type == "private"
    stream_name = ""
    if info:
        if chat_type == "group":
            stream_name = str(info.get("group_name") or "") or stream_id[:8]
        elif chat_type == "private":
            hashed_person_id = str(info.get("person_id") or "")
            if hashed_person_id:
                try:
                    person = await database_api.get_by(PersonInfo, person_id=hashed_person_id)
                    if person is not None:
                        stream_name = str(person.nickname or "") or "未知用户"
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"私聊流反查人物失败 {hashed_person_id}: {exc}")
            if not stream_name:
                stream_name = "私聊_" + stream_id[:8]
    if not stream_name:
        stream_name = stream_id[:8]

    chat_flow = _format_messages(window_messages)
    date_str = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")

    # 读取 bot 人设（昵称/核心人格/人格侧面/身份）
    from src.app.plugin_system.api import config_api

    persona_text = ""
    bot_name = "Engram Memory"
    try:
        core_config = config_api.get_core_config()
        personality = core_config.personality
        bot_name = str(personality.nickname or "") or bot_name
        parts: list[str] = []
        if personality.personality_core:
            parts.append(f"性格：{personality.personality_core}")
        if personality.personality_side:
            parts.append(f"人格侧面：{personality.personality_side}")
        if personality.identity:
            parts.append(f"身份：{personality.identity}")
        persona_text = "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"读取 bot 人设失败: {exc}")

    # 1. 生成流日记
    journal_prompt = resolve_prompt(JOURNAL_PROMPT_NAME, JOURNAL_PROMPT).format(
        bot_name=bot_name,
        stream_name=stream_name,
        date=date_str,
        persona=persona_text or "（无）",
        messages=chat_flow,
    )
    journal_raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_journal_generate",
        system=journal_prompt,
        user=chat_flow,
        stream_id=stream_id,
        persona=persona_text or None,
    )
    if not journal_raw.strip():
        raise ValueError("流日记生成为空")
    journal_content = journal_raw.strip()

    # 2. 存盘
    await store.write_journal(date_str, stream_name, journal_content)

    # 3. 提取长期记忆
    extract_prompt = resolve_prompt(EXTRACT_PROMPT_NAME, EXTRACT_PROMPT).format(
        bot_name=bot_name,
        date=date_str,
        stream_name=stream_name,
        person_roster=_format_roster(roster),
        journal_content=journal_content,
    )
    extract_raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_journal_extract",
        system=extract_prompt,
        user=journal_content,
        stream_id=stream_id,
    )
    items = extract_json_array(extract_raw)

    # 私聊流唯一人物
    private_person_id: str | None = None
    if is_private:
        private_person_id = _person_id_of(window_messages[0])

    # 4. 字段清洗 + 人物 ID 校验 + 写入长期层
    new_memories: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    day_zero = datetime.fromtimestamp(end_ts).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()

    for item in items:
        memory_id = str(item.get("memory_id") or "").strip()
        if memory_id and memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)

        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title or not content:
            continue

        person_id = _validate_person_field(
            item.get("person_id"),
            roster=roster,
            is_private=is_private,
            private_person_id=private_person_id,
        )
        related_people = [
            p
            for p in (
                _validate_person_field(
                    raw,
                    roster=roster,
                    is_private=is_private,
                    private_person_id=private_person_id,
                )
                for raw in (item.get("related_people") or [])
            )
            if p
        ]

        raw_event_time = item.get("event_time")
        try:
            event_time = float(raw_event_time) if raw_event_time else day_zero
        except (TypeError, ValueError):
            event_time = day_zero

        result = await memory_service.write_memory(
            title=title,
            content=content,
            core_tags=_clean_tags(item.get("core_tags")),
            diffusion_tags=_clean_tags(item.get("diffusion_tags")),
            opposing_tags=_clean_tags(item.get("opposing_tags")),
            event_time=event_time,
            layer="archived",
            person_id=person_id,
            related_people=related_people or None,
            relation_memory_ids=[
                str(mid) for mid in (item.get("relation_memory_ids") or []) if str(mid).strip()
            ]
            or None,
            stream_id=stream_id,
        )
        new_memories.append(
            {
                "memory_id": result["memory_id"],
                "title": title,
                "content_summary": content[:100],
                "stream_id": stream_id,
                "person_id": person_id,
            }
        )

    return {
        "stream_id": stream_id,
        "stream_name": stream_name,
        "journal_summary": _summarize_journal(journal_content),
        "new_memories": new_memories,
    }


def _summarize_journal(journal_content: str) -> str:
    """提取日记摘要（前 300 字）。"""
    return journal_content[:300]


# ------------------------------------------------------------------
# 第二步：子任务 3 — 每流内部记忆关联建立
# ------------------------------------------------------------------


async def _stream_association(
    plugin: Any,
    memory_service: MemoryService,
    all_new_memories: list[dict[str, Any]],
    config: Any,
) -> None:
    """每流内部记忆关联建立：按流分组，组内判定关联对，写入 relation_memory_ids 双向。"""
    # 按 stream_id 分组
    by_stream: dict[str, list[dict[str, Any]]] = {}
    for mem in all_new_memories:
        sid = str(mem.get("stream_id") or "")
        by_stream.setdefault(sid, []).append(mem)

    repo = memory_service._get_repo()
    for sid, mems in by_stream.items():
        if len(mems) < 2:
            continue
        new_ids = {m["memory_id"] for m in mems}

        # 该流已有记忆候选（每条新记忆 top 3，去重合并，排除本次新记忆）
        existing_candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for mem in mems:
            try:
                hits = await memory_service.search_memories(
                    mem.get("content_summary") or mem["title"],
                    layer="all",
                    top_n=3,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"预检索已有记忆失败 {mem['memory_id']}: {exc}")
                continue
            for hit in hits:
                hit_id = str(hit.get("memory_id") or "")
                if hit_id in new_ids or hit_id in seen_ids:
                    continue
                seen_ids.add(hit_id)
                existing_candidates.append(
                    {
                        "memory_id": hit_id,
                        "title": hit.get("title") or "",
                        "content_summary": hit.get("summary") or "",
                        "source": hit.get("source") or "",
                        "is_existing": True,
                    }
                )

        new_payload = [
            {
                "memory_id": m["memory_id"],
                "title": m["title"],
                "content_summary": m.get("content_summary") or "",
            }
            for m in mems
        ]

        prompt = resolve_prompt(ASSOCIATE_PROMPT_NAME, ASSOCIATE_PROMPT).format(
            bot_name="Engram Memory",
            stream_name=sid[:8],
            new_memories=json.dumps(new_payload, ensure_ascii=False),
            existing_candidates=json.dumps(existing_candidates, ensure_ascii=False),
        )
        raw = await call_sub_agent(
            task=str(config.internal_llm.task_name),
            request_name="engram_memory_journal_associate",
            system=prompt,
            user=json.dumps(
                {"new_memories": new_payload, "existing_candidates": existing_candidates},
                ensure_ascii=False,
            ),
        )
        pairs = extract_json_array(raw)

        for pair in pairs:
            a = str(pair.get("a") or "").strip()
            b = str(pair.get("b") or "").strip()
            if not a or not b or a == b:
                continue
            records = await repo.get_records_map([a, b])
            if a not in records or b not in records:
                continue
            try:
                await memory_service.write_memory(
                    title=records[a].title,
                    content=records[a].content,
                    core_tags=records[a].core_tags,
                    diffusion_tags=records[a].diffusion_tags,
                    opposing_tags=records[a].opposing_tags,
                    event_time=records[a].event_time,
                    layer=records[a].layer,
                    person_id=records[a].person_id,
                    related_people=records[a].related_people,
                    relation_memory_ids=list(set(records[a].relation_memory_ids) | {b}),
                    memory_id=a,
                    stream_id=records[a].stream_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"写入关联失败 {a}->{b}: {exc}")


# ------------------------------------------------------------------
# 第二步：子任务 2 — 人物印象更新
# ------------------------------------------------------------------


async def _update_impressions(
    plugin: Any,
    person_service: PersonService,
    stream_results: list[dict[str, Any]],
    all_new_memories: list[dict[str, Any]],
    config: Any,
) -> None:
    """人物印象更新：基于流日记摘要 + 新记忆，更新 PersonInfo.impression。"""
    # 收集出现人物（person_id 非空）去重
    person_ids: set[str] = set()
    for mem in all_new_memories:
        pid = mem.get("person_id")
        if pid:
            person_ids.add(pid)

    # 流日记摘要按流分组
    stream_summaries: dict[str, str] = {
        result.get("stream_id", ""): result.get("journal_summary", "")
        for result in stream_results
    }

    for raw_person_id in person_ids:
        platform, _, user_id = raw_person_id.partition(":")
        if not platform or not user_id:
            continue

        person = await person_api.get_person(platform, user_id)
        if person is None:
            continue
        current_impression = str(person.impression or "")

        today_memories = [
            mem for mem in all_new_memories if mem.get("person_id") == raw_person_id
        ]
        today_summaries_parts: list[str] = []
        for mem in today_memories:
            summary = stream_summaries.get(mem.get("stream_id", ""))
            if summary:
                today_summaries_parts.append(summary)
        today_summaries = "\n".join(dict.fromkeys(today_summaries_parts))
        if not today_summaries and not today_memories:
            continue

        prompt = resolve_prompt(IMPRESSION_PROMPT_NAME, IMPRESSION_PROMPT).format(
            bot_name="Engram Memory",
            nickname=str(person.nickname or ""),
            current_impression=current_impression or "（暂无印象）",
            today_summaries=today_summaries or "（无）",
            today_memories=json.dumps(
                [
                    {"title": m.get("title"), "content": m.get("content_summary")}
                    for m in today_memories
                ],
                ensure_ascii=False,
            ),
        )
        raw = await call_sub_agent(
            task=str(config.internal_llm.task_name),
            request_name="engram_memory_journal_impression",
            system=prompt,
            user=prompt,
        )
        new_impression = str(raw or "").strip()
        if not new_impression:
            continue
        max_chars = int(config.journal.impression_max_chars)
        new_impression = new_impression[:max_chars]
        try:
            await person_service.update_impression(platform, user_id, new_impression)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"更新人物印象失败 {raw_person_id}: {exc}")


# ------------------------------------------------------------------
# 第二步：子任务 2 — 短期记忆审查（复制晋升）
# ------------------------------------------------------------------


async def _review_short_term(
    plugin: Any,
    memory_service: MemoryService,
    config: Any,
) -> None:
    """审查所有未过期短期记忆，有价值的复制晋升到 archived。

    使用 ``copy_promote_memory`` 保留短期原记录（layer/expires_at 不变），
    在 archived 层新建一条独立记忆。
    """
    repo = memory_service._get_repo()
    now = time.time()
    short_term_memories = await repo.list_short_term_all_unexpired(now=now)
    if not short_term_memories:
        return

    payload = [
        {
            "memory_id": mem.memory_id,
            "title": mem.title,
            "content_summary": mem.content[:100],
        }
        for mem in short_term_memories
    ]
    prompt = resolve_prompt(
        SHORT_TERM_REVIEW_PROMPT_NAME, SHORT_TERM_REVIEW_PROMPT
    ).format(
        bot_name="Engram Memory",
        short_term_memories=json.dumps(payload, ensure_ascii=False),
    )
    raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_journal_short_term_review",
        system=prompt,
        user=json.dumps(payload, ensure_ascii=False),
    )
    decisions = extract_json_array(raw)

    for decision in decisions:
        memory_id = str(decision.get("memory_id") or "").strip()
        action = str(decision.get("action") or "").strip().lower()
        if not memory_id or action != "promote":
            continue
        try:
            new_id = await memory_service.copy_promote_memory(memory_id, "archived")
            if new_id:
                logger.info(f"短期记忆复制晋升: {memory_id} -> {new_id}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"短期记忆晋升失败 {memory_id}: {exc}")


# ------------------------------------------------------------------
# 第二步：子任务 1 — 中期层增量审查（当天新增 active）
# ------------------------------------------------------------------


async def _review_active_layer(
    plugin: Any,
    memory_service: MemoryService,
    start_ts: float,
    end_ts: float,
    config: Any,
) -> None:
    """审查当天新增的 active 记忆，判定 promote/discard/keep。"""
    repo = memory_service._get_repo()
    active_memories = await repo.list_active_by_created_range(
        start_ts=start_ts, end_ts=end_ts, limit=500
    )
    if not active_memories:
        return

    payload = [
        {"memory_id": mem.memory_id, "title": mem.title, "content_summary": mem.content[:100]}
        for mem in active_memories
    ]
    prompt = resolve_prompt(ACTIVE_REVIEW_PROMPT_NAME, ACTIVE_REVIEW_PROMPT).format(
        bot_name="Engram Memory",
        active_memories=json.dumps(payload, ensure_ascii=False),
    )
    raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_journal_active_review",
        system=prompt,
        user=json.dumps(payload, ensure_ascii=False),
    )
    decisions = extract_json_array(raw)

    valid_actions = {"promote", "discard", "keep"}
    for decision in decisions:
        memory_id = str(decision.get("memory_id") or "").strip()
        action = str(decision.get("action") or "").strip().lower()
        if not memory_id or action not in valid_actions:
            continue
        try:
            if action == "promote":
                await memory_service.promote_memory(memory_id, "archived")
            elif action == "discard":
                await memory_service.delete_memory(memory_id)
            # keep: 不处理
        except Exception as exc:  # noqa: BLE001
            logger.error(f"中期层处置失败 {memory_id}: {exc}")
