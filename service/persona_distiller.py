"""engram_memory 人物印象懒加载蒸馏服务。

基于人物本人跨流文本消息，分块提炼特征并融合为整体印象，写入
``PersonInfo.impression``。采用「最近窗口」模式：每次取最近 N 条文本
消息蒸馏，不维护增量游标，印象反映当前状态。
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import log_api, message_api, person_api

from ..prompts import DISTILL_PROMPT, DISTILL_PROMPT_NAME, MERGE_PROMPT, MERGE_PROMPT_NAME
from ..sub_agent import call_sub_agent, extract_json_array, resolve_prompt

if TYPE_CHECKING:
    from ..config import EngramMemoryConfig
    from ..store import MemoryStore

logger = log_api.get_logger("engram_memory.persona_distiller")

# 参与蒸馏的消息类型白名单（只留文本）
_TEXT_TYPES = {"text"}


def _has_messages_budget(messages: list[dict[str, Any]], min_messages: int) -> bool:
    """判断文本消息条数是否达到蒸馏门槛。

    Args:
        messages: 拉取的消息字典列表（已过滤）。
        min_messages: 最少文本条数。

    Returns:
        达到门槛返回 True。
    """
    return len(messages) >= min_messages


def _filter_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留文本消息，并按时间正序。

    Args:
        messages: 消息字典列表。

    Returns:
        文本消息列表（时间正序）。
    """
    texts = [
        msg
        for msg in messages
        if str(msg.get("message_type") or "").lower() in _TEXT_TYPES
    ]
    texts.sort(key=lambda m: float(m.get("time") or 0.0))
    return texts


def _format_chunk(messages: list[dict[str, Any]]) -> str:
    """将一批消息格式化为聊天记录文本。

    Args:
        messages: 文本消息列表。

    Returns:
        格式化后的聊天记录。
    """
    lines: list[str] = []
    for msg in messages:
        text = str(msg.get("processed_plain_text") or msg.get("content") or "").strip()
        if not text:
            continue
        sender = str(msg.get("sender_name") or msg.get("sender_id") or "未知")
        lines.append(f"{sender}: {text}")
    return "\n".join(lines)


def _chunk_messages(
    messages: list[dict[str, Any]], chunk_size: int
) -> list[list[dict[str, Any]]]:
    """将消息列表按 chunk_size 切成块。

    Args:
        messages: 文本消息列表。
        chunk_size: 每块条数。

    Returns:
        消息块列表。
    """
    return [
        messages[i : i + chunk_size] for i in range(0, len(messages), chunk_size)
    ]


async def _distill_chunk(
    plugin: Any,
    config: "EngramMemoryConfig",
    chunk: list[dict[str, Any]],
) -> dict[str, Any]:
    """对单个消息块执行一次特征提炼。

    Args:
        plugin: 插件实例。
        config: 插件配置。
        chunk: 单个消息块。

    Returns:
        结构化观察字典；失败返回空字典。
    """
    chat_flow = _format_chunk(chunk)
    if not chat_flow.strip():
        return {}
    system_prompt = resolve_prompt(DISTILL_PROMPT_NAME, DISTILL_PROMPT).format(
        bot_name="Engram Memory",
        messages=chat_flow,
    )
    raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_persona_distill_chunk",
        system=system_prompt,
        user=chat_flow,
    )
    items = extract_json_array(raw)
    if not items:
        return {}
    return items[0]


def _merge_observations(observations: list[dict[str, Any]]) -> str:
    """将多块观察序列化为 prompt 文本。"""
    parts: list[str] = []
    for index, obs in enumerate(observations, start=1):
        parts.append(f"【片段 {index}】\n{json.dumps(obs, ensure_ascii=False)}")
    return "\n\n".join(parts)


def _load_bot_persona() -> tuple[str, str]:
    """读取真实 bot 人设（昵称/核心人格/人格侧面/身份）。

    Returns:
        (bot 昵称, 人设文本)；读取失败时返回 ("Engram Memory", "")。
    """
    try:
        from src.app.plugin_system.api import config_api

        personality = config_api.get_core_config().personality
        bot_name = str(personality.nickname or "") or "Engram Memory"
        parts: list[str] = []
        if personality.personality_core:
            parts.append(f"性格：{personality.personality_core}")
        if personality.personality_side:
            parts.append(f"人格侧面：{personality.personality_side}")
        if personality.identity:
            parts.append(f"身份：{personality.identity}")
        if personality.reply_style:
            parts.append(f"表达风格：{personality.reply_style}")
        return bot_name, "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"读取 bot 人设失败: {exc}")
        return "Engram Memory", ""


async def distill_person(
    plugin: Any,
    store: "MemoryStore",
    config: "EngramMemoryConfig",
    raw_person_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """对单个人物执行一次印象蒸馏（懒加载/巡检/手动共用）。

    Args:
        plugin: 插件实例。
        store: 共享存储层。
        config: 插件配置。
        raw_person_id: 人物原始 ID（platform:user_id）。
        force: 是否强制蒸馏（无视消息数门槛）。

    Returns:
        蒸馏结果字典，含 ok / distilled / reason / impression。
    """
    platform, _, user_id = raw_person_id.partition(":")
    if not platform or not user_id:
        return {"ok": False, "distilled": False, "reason": "person_id 格式非法"}

    person = await person_api.get_person(platform, user_id)
    if person is None:
        return {"ok": False, "distilled": False, "reason": "人物不存在"}

    # 哈希 person_id（Messages 表索引）
    hashed_person_id = person_api.generate_person_id(platform, user_id)

    max_messages = int(config.persona.max_messages)
    min_messages = int(config.persona.min_messages)
    chunk_size = int(config.persona.chunk_size)

    try:
        raw_messages = await message_api.get_messages_by_time_for_users(
            start_time=0.0,
            end_time=time.time(),
            person_ids=[hashed_person_id],
            limit=max_messages,
            limit_mode="latest",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"拉取人物消息失败 {raw_person_id}: {exc}")
        return {"ok": False, "distilled": False, "reason": f"拉取消息失败: {exc}"}

    text_messages = _filter_text_messages(raw_messages)
    if not force and not _has_messages_budget(text_messages, min_messages):
        return {
            "ok": True,
            "distilled": False,
            "reason": f"样本不足（文本消息 {len(text_messages)} < {min_messages}）",
        }

    if not text_messages:
        return {"ok": True, "distilled": False, "reason": "无文本消息可蒸馏"}

    # 分块提炼
    observations: list[dict[str, Any]] = []
    for chunk in _chunk_messages(text_messages, chunk_size):
        obs = await _distill_chunk(plugin, config, chunk)
        if obs:
            observations.append(obs)
    if not observations:
        return {"ok": True, "distilled": False, "reason": "特征提炼为空"}

    # 融合为最终印象（注入真实 bot 人设，以 bot 口吻写）
    max_chars = int(config.journal.impression_max_chars)
    bot_name, persona_text = _load_bot_persona()
    merge_prompt = resolve_prompt(MERGE_PROMPT_NAME, MERGE_PROMPT).format(
        bot_name=bot_name,
        persona=persona_text or "（无）",
        observations=_merge_observations(observations),
        max_chars=max_chars,
    )
    impression_text = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_persona_merge",
        system=merge_prompt,
        user=merge_prompt,
        persona=persona_text or None,
    )
    impression_text = str(impression_text or "").strip()
    if not impression_text:
        return {"ok": True, "distilled": False, "reason": "融合结果为空"}

    # 写印象
    try:
        await person_api.update_user_impression(
            platform, user_id, impression=impression_text[:max_chars]
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"写入印象失败 {raw_person_id}: {exc}")
        return {"ok": False, "distilled": False, "reason": f"写入印象失败: {exc}"}

    # 记录蒸馏元数据
    try:
        meta = await store.read_distill_meta()
        meta[raw_person_id] = {
            "last_distilled_at": time.time(),
            "message_count": len(text_messages),
        }
        await store.write_distill_meta(meta)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"记录蒸馏元数据失败 {raw_person_id}: {exc}")

    logger.info(f"人物蒸馏完成 {raw_person_id}: 文本 {len(text_messages)} 条")
    return {
        "ok": True,
        "distilled": True,
        "reason": f"蒸馏完成（文本 {len(text_messages)} 条）",
        "impression": impression_text[:max_chars],
    }


async def ensure_person_impression(
    plugin: Any,
    store: "MemoryStore",
    config: "EngramMemoryConfig",
    raw_person_id: str,
) -> tuple[bool, str]:
    """懒加载入口：保证人物有可用印象。

    若从未蒸馏且样本足够则蒸馏一次；已蒸馏但已有新文本消息则增量蒸馏。
    返回 (是否有可用印象, 说明)。

    Args:
        plugin: 插件实例。
        store: 共享存储层。
        config: 插件配置。
        raw_person_id: 人物原始 ID（platform:user_id）。

    Returns:
        (是否有可用印象, 说明文本)。
    """
    platform, _, user_id = raw_person_id.partition(":")
    if not platform or not user_id:
        return False, "person_id 格式非法"

    person = await person_api.get_person(platform, user_id)
    if person is None:
        return False, "人物不存在"

    meta = await store.read_distill_meta()
    entry = meta.get(raw_person_id) or {}

    has_impression = bool(str(person.impression or "").strip())
    last_distilled_at = float(entry.get("last_distilled_at") or 0.0)

    # 已有印象且距上次蒸馏无新文本消息（最新文本时间不晚于蒸馏时间）→ 不重复蒸馏
    if has_impression and last_distilled_at > 0.0:
        hashed_person_id = person_api.generate_person_id(platform, user_id)
        try:
            recent = await message_api.get_messages_by_time_for_users(
                start_time=last_distilled_at,
                end_time=time.time(),
                person_ids=[hashed_person_id],
                limit=1,
                limit_mode="latest",
            )
            if not recent:
                return True, "印象已蒸馏且无新增文本消息"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"检查新消息失败 {raw_person_id}: {exc}")

    # 无印象或已有新文本消息 → 尝试蒸馏
    result = await distill_person(plugin, store, config, raw_person_id)
    if result.get("distilled"):
        return True, str(result.get("reason") or "")
    return has_impression, str(result.get("reason") or "")
