"""engram_memory 人物印象懒加载蒸馏服务。

基于「对话窗口」蒸馏人物印象：定位目标人物的真实发言点（锚点），
围绕锚点挖取带上下文的对话窗口（含 bot 回复与旁人反应），分块提炼
特征并融合为整体印象，写入 ``PersonInfo.impression``。支持按人物配置
蒸馏范围（all/group/private）。
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import log_api, message_api, person_api

from ..prompts import DISTILL_PROMPT, DISTILL_PROMPT_NAME, MERGE_PROMPT, MERGE_PROMPT_NAME
from ..sub_agent import call_json_sub_agent, call_sub_agent, resolve_prompt

if TYPE_CHECKING:
    from ..config import EngramMemoryConfig
    from ..store import MemoryStore

logger = log_api.get_logger("engram_memory.persona_distiller")

# 参与蒸馏的消息类型白名单（只留文本）
_TEXT_TYPES = {"text"}

# 合法 scope 值
_SCOPES = {"all", "group", "private"}

# 单个锚点挖窗的时间半径上限（秒），防止冷清流一次拉全量历史
_HALF_WINDOW_SECONDS = 900.0


def _normalize_scope(scope: Any) -> str:
    """归一化 scope 值，非法值回退 all。"""
    value = str(scope or "").strip().lower()
    return value if value in _SCOPES else "all"


def _resolve_scope(
    raw_person_id: str,
    meta: dict[str, dict[str, Any]],
    default_scope: str,
) -> str:
    """解析蒸馏范围：人物级覆盖优先，其次全局配置。

    Args:
        raw_person_id: 人物原始 ID（platform:user_id）。
        meta: 已读取的蒸馏元数据（含人物级 scope 覆盖）。
        default_scope: 全局默认范围。

    Returns:
        该人物生效的 scope（all/group/private）。
    """
    entry = meta.get(raw_person_id) or {}
    override = str(entry.get("scope") or "").strip().lower()
    if override in _SCOPES:
        return override
    return _normalize_scope(default_scope)


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


def _format_window(
    messages: list[dict[str, Any]],
    *,
    target_hashed_person_id: str,
    bot_sender_ids: set[str],
    stream_label: str,
) -> str:
    """将一个对话窗口格式化为带角色标注的聊天文本。

    Args:
        messages: 窗口消息列表（时间正序）。
        target_hashed_person_id: 目标人物的哈希 person_id。
        bot_sender_ids: bot 的 sender_id 集合（用于标注「我」）。
        stream_label: 流显示名（群名/私聊标记）。

    Returns:
        格式化文本。
    """
    lines: list[str] = [f"【{stream_label}】"]
    for msg in messages:
        if str(msg.get("message_type") or "").lower() not in _TEXT_TYPES:
            continue
        text = str(msg.get("processed_plain_text") or msg.get("content") or "").strip()
        if not text:
            continue
        sender_id = str(msg.get("sender_id") or "")
        sender = str(msg.get("sender_name") or sender_id or "未知")
        if sender_id in bot_sender_ids:
            speaker = "我"
        elif str(msg.get("person_id") or "") == target_hashed_person_id:
            speaker = f"{sender}（主角）"
        else:
            speaker = sender
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _pick_window_anchors(
    anchors: list[dict[str, Any]],
    window_count: int,
) -> list[dict[str, Any]]:
    """跨流均匀分配窗口锚点。

    每个流按其锚点占比分摊窗口配额，流内采样并优先较新锚点；
    所有流共享 window_count 总预算。

    Args:
        anchors: 锚点列表（时间正序，元素含 stream_id/time）。
        window_count: 目标窗口数。

    Returns:
        选中的锚点列表（时间正序）。
    """
    if len(anchors) <= window_count:
        return list(anchors)
    grouped: dict[str, list[int]] = {}
    for index, anchor in enumerate(anchors):
        stream_id = str(anchor.get("stream_id") or "")
        if stream_id:
            grouped.setdefault(stream_id, []).append(index)
    total = len(anchors)
    picked_indexes: set[int] = set()
    # 第一轮：按占比分配配额（每流至少 1）
    quotas: dict[str, int] = {}
    for stream_id, indexes in grouped.items():
        quotas[stream_id] = max(1, window_count * len(indexes) // total)
    # 超额削减（占比分配总和可能超预算）
    while sum(quotas.values()) > window_count:
        richest = max(quotas, key=lambda s: quotas[s])
        if quotas[richest] > 1:
            quotas[richest] -= 1
        else:
            break
    for stream_id, quota in quotas.items():
        indexes = grouped[stream_id]
        take = min(quota, len(indexes))
        if take == len(indexes):
            picked_indexes.update(indexes)
        else:
            # 流内优先较新（尾部）锚点，等距采样
            step = len(indexes) / take
            picked_indexes.update(
                indexes[len(indexes) - 1 - round(k * step)] for k in range(take)
            )
    # 第二轮：从最新开始补齐剩余预算
    remaining = window_count - len(picked_indexes)
    if remaining > 0:
        for index in range(len(anchors) - 1, -1, -1):
            if remaining <= 0:
                break
            if index not in picked_indexes:
                picked_indexes.add(index)
                remaining -= 1
    return [anchors[i] for i in sorted(picked_indexes)]


def _chunk_texts(texts: list[str], max_chars: int) -> list[list[str]]:
    """将窗口文本列表按累计字符数切块（每块近似 max_chars）。

    Args:
        texts: 窗口文本列表。
        max_chars: 每块目标字符数。

    Returns:
        窗口文本块列表。
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        length = len(text) + 1
        if current and size + length > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(text)
        size += length
    if current:
        chunks.append(current)
    return chunks


async def _distill_chunk(
    plugin: Any,
    config: "EngramMemoryConfig",
    chunk_text: str,
) -> dict[str, Any]:
    """对单个窗口块文本执行一次特征提炼。

    Args:
        plugin: 插件实例。
        config: 插件配置。
        chunk_text: 已格式化的对话块文本。

    Returns:
        结构化观察字典；失败返回空字典。
    """
    if not chunk_text.strip():
        return {}
    system_prompt = resolve_prompt(DISTILL_PROMPT_NAME, DISTILL_PROMPT).format(
        bot_name="Engram Memory",
    )
    items, status = await call_json_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_persona_distill_chunk",
        system=system_prompt,
        user=chunk_text,
    )
    if status == "error":
        logger.warning("人物蒸馏分块调用/解析失败，跳过该块")
        return {}
    return items[0] if items else {}


def _merge_observations(observations: list[dict[str, Any]]) -> str:
    """将多块观察序列化为 prompt 文本。"""
    parts: list[str] = []
    for index, obs in enumerate(observations, start=1):
        parts.append(f"【片段 {index}】\n{json.dumps(obs, ensure_ascii=False)}")
    return "\n\n".join(parts)


async def _collect_memory_material(
    plugin: Any,
    raw_person_id: str,
    *,
    observations: list[dict[str, Any]] | None = None,
) -> list[str]:
    """收集该人物的既有记忆素材（选记忆环节 + 直接检索兜底）。

    流程：查询记忆目录（含 related_people 匹配）→ 交给子代理按观察
    摘要挑选补全型条目 → 取选中条目全文。子代理挑选失败时回退为最近
    N 条（不因选记忆环节故障中断蒸馏）。

    Args:
        plugin: 插件实例（提供共享 repo 与配置）。
        raw_person_id: 人物原始 ID（platform:user_id）。
        observations: 已提炼的观察列表（供挑选参考），None 时跳过选记忆。

    Returns:
        「标题：内容摘要」文本行列表；无记忆或查询失败返回空列表。
    """
    try:
        from ..store import shared_repo

        def _config_factory() -> Any:
            from ..config import EngramMemoryConfig

            config = plugin.config
            if isinstance(config, EngramMemoryConfig):
                return config
            return EngramMemoryConfig()

        config = _config_factory()
        repo = shared_repo(plugin, _config_factory)
        # 幂等初始化（插件加载时已建表则直接返回；独立脚本环境补齐）
        await repo.initialize()
        index = await repo.search_person_memory_index(
            person_id=raw_person_id,
            layers=["active", "archived"],
            limit=int(config.persona.memory_index_limit),
        )
        if not index:
            return []

        chosen_ids: list[str] = []
        if observations is not None:
            chosen_ids = await _pick_memories(plugin, config, observations, index)

        if not chosen_ids:
            # 选记忆未产出（明确空选或挑选失败）：回退取最近 N 条全文
            fallback = await repo.search_by_person(
                person_id=raw_person_id,
                layers=["active", "archived"],
                limit=int(config.persona.memory_material_limit),
                include_related=True,
            )
            records = fallback
        else:
            record_map = await repo.get_records_map(chosen_ids)
            # 保持目录时间序（新→旧）
            records = [
                record_map[mid]
                for mid in chosen_ids
                if (record_map.get(mid)) is not None
            ]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"检索人物记忆素材失败 {raw_person_id}: {exc}")
        return []

    lines: list[str] = []
    for record in records:
        summary = (record.content or "").strip()[:100]
        lines.append(f"- {record.title}：{summary}")
    return lines[: int(config.persona.memory_material_limit)]


async def _pick_memories(
    plugin: Any,
    config: "EngramMemoryConfig",
    observations: list[dict[str, Any]],
    index: list[dict[str, Any]],
) -> list[str]:
    """让子代理从记忆目录中挑选补全印象的条目。

    Args:
        plugin: 插件实例。
        config: 插件配置。
        observations: 已提炼的观察列表。
        index: 记忆目录（含 memory_id/title/updated_at）。

    Returns:
        选中的 memory_id 列表（保持目录顺序）；未选出或失败返回空列表。
    """
    from datetime import datetime

    from ..prompts import MEMORY_PICK_PROMPT, MEMORY_PICK_PROMPT_NAME
    from ..sub_agent import call_sub_agent, extract_int_array, resolve_prompt

    def _date(ts: float) -> str:
        """时间戳转简短日期；非法回退空串。"""
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except (OSError, ValueError, OverflowError):
            return ""

    # 观察摘要：各块观察的关键词串联（控制 token）
    obs_keys: list[str] = []
    for obs in observations:
        keys = [
            str(v)
            for v in obs.values()
            if isinstance(v, str) and v and v != "不详"
        ]
        if keys:
            obs_keys.append("、".join(keys)[:120])
    obs_summary = "；".join(obs_keys)[:800] or "（暂无）"

    max_pick = int(config.persona.memory_material_limit)
    catalog_lines = [
        f"{i}. {entry['title']}（{_date(entry['updated_at'])}）"
        for i, entry in enumerate(index, start=1)
    ]
    user_text = (
        "【已有观察摘要】\n"
        + obs_summary
        + "\n\n【记忆目录】\n"
        + "\n".join(catalog_lines)
    )
    system_prompt = resolve_prompt(
        MEMORY_PICK_PROMPT_NAME, MEMORY_PICK_PROMPT
    ).format(max_pick=max_pick)

    raw = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_persona_memory_pick",
        system=system_prompt,
        user=user_text,
    )
    picked = extract_int_array(raw)
    if not picked:
        return []
    # 编号（1 起）→ memory_id；越界忽略；去重保序
    chosen: list[str] = []
    seen: set[int] = set()
    for number in picked:
        if 1 <= number <= len(index) and number not in seen:
            seen.add(number)
            chosen.append(index[number - 1]["memory_id"])
    if picked and not chosen:
        logger.debug("选记忆输出编号全部越界，忽略")
    return chosen


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


async def _load_bot_sender_ids(platforms: set[str]) -> set[str]:
    """查询各平台 bot 的 sender_id 集合（用于窗口内标注「我」）。

    Args:
        platforms: 涉及的平台名集合。

    Returns:
        bot sender_id 集合；查询失败的平台跳过。
    """
    from src.app.plugin_system.api import adapter_api

    bot_ids: set[str] = set()
    for platform in platforms:
        try:
            info = await adapter_api.get_bot_info_by_platform(platform)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"查询 bot 信息失败 platform={platform}: {exc}")
            continue
        if info and info.get("bot_id"):
            bot_ids.add(str(info["bot_id"]))
    return bot_ids


async def _stream_display_labels(stream_ids: list[str]) -> dict[str, str]:
    """查询流显示名（群名或私聊标记）。

    Args:
        stream_ids: 流 ID 列表。

    Returns:
        ``{stream_id: 显示名}``；查询失败的流回退流 ID 前 8 位。
    """
    from src.app.plugin_system.api import stream_api

    labels: dict[str, str] = {}
    for stream_id in stream_ids:
        fallback = f"[{stream_id[:8]}]"
        try:
            info = await stream_api.get_stream_info(stream_id)
        except Exception:  # noqa: BLE001
            info = None
        if not info:
            labels[stream_id] = fallback
            continue
        chat_type = str(info.get("chat_type") or "")
        if chat_type == "group":
            labels[stream_id] = str(info.get("group_name") or "") or fallback
        elif chat_type == "private":
            labels[stream_id] = "私聊"
        else:
            labels[stream_id] = fallback
    return labels


async def _count_scoped_own_messages(
    config: "EngramMemoryConfig",
    hashed_person_id: str,
    scope: str,
) -> int:
    """统计目标人物在生效 scope 内的本人消息数（门槛判定用）。

    基于锚点池（本人最近消息）做 scope 内计数，只查一次库，
    不逐流循环（流多时逐流查询会退化为上百次 DB 往返）。

    Args:
        config: 插件配置。
        hashed_person_id: 哈希 person_id。
        scope: 生效范围。

    Returns:
        消息条数（截断到查询上限）；查询失败返回 0。
    """
    from src.app.plugin_system.api import stream_api

    cap = int(config.persona.max_messages)
    try:
        rows = await message_api.get_messages_by_time_for_users(
            start_time=0.0,
            end_time=time.time(),
            person_ids=[hashed_person_id],
            limit=cap,
            limit_mode="latest",
        )
        if scope == "all":
            return len(rows)
        # group/private：用流列表过滤后计数
        allowed = set(await stream_api.get_stream_ids_from_db(scope))
        return sum(
            1 for msg in rows if str(msg.get("stream_id") or "") in allowed
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"统计 scope 内消息数失败 scope={scope}: {exc}")
        return 0


async def _collect_conversation_windows(
    config: "EngramMemoryConfig",
    hashed_person_id: str,
    scope: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """采集带上下文的对话窗口（scope 过滤 + 锚点采样 + 挖窗）。

    Args:
        config: 插件配置。
        hashed_person_id: 目标人物哈希 person_id。
        scope: 生效的蒸馏范围（all/group/private）。

    Returns:
        ``(stream_id, 窗口消息列表)`` 元组列表（时间正序）。
    """
    from src.app.plugin_system.api import stream_api

    # 1) 锚点定位：目标人物最近的本人文本消息
    anchor_messages = await message_api.get_messages_by_time_for_users(
        start_time=0.0,
        end_time=time.time(),
        person_ids=[hashed_person_id],
        limit=int(config.persona.max_messages),
        limit_mode="latest",
    )
    anchor_messages = _filter_text_messages(anchor_messages)
    if not anchor_messages:
        return []

    # 2) scope 过滤：按流的聊天类型剔除锚点
    if scope != "all":
        try:
            allowed = set(await stream_api.get_stream_ids_from_db(scope))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"获取流列表失败 scope={scope}: {exc}")
            return []
        anchor_messages = [
            msg
            for msg in anchor_messages
            if str(msg.get("stream_id") or "") in allowed
        ]
        if not anchor_messages:
            return []

    # 3) 跨流均匀采样窗口锚点
    anchors = [
        {
            "stream_id": str(msg.get("stream_id") or ""),
            "time": float(msg.get("time") or 0.0),
            "message_id": str(msg.get("message_id") or ""),
        }
        for msg in anchor_messages
    ]
    chosen = _pick_window_anchors(anchors, int(config.persona.window_count))
    if not chosen:
        return []

    # 4) 逐锚点挖窗：锚点前 radius 条 + 锚点起 earliest span 条
    radius = int(config.persona.window_radius)
    span = radius * 2 + 1
    windows: list[tuple[str, list[dict[str, Any]]]] = []
    for anchor in chosen:
        stream_id = anchor["stream_id"]
        anchor_time = anchor["time"]
        try:
            before = await message_api.get_messages_by_time_in_chat(
                stream_id=stream_id,
                start_time=anchor_time - _HALF_WINDOW_SECONDS,
                end_time=anchor_time,
                limit=radius,
                limit_mode="latest",
            )
            after = await message_api.get_messages_by_time_in_chat_inclusive(
                stream_id=stream_id,
                start_time=anchor_time,
                end_time=anchor_time + _HALF_WINDOW_SECONDS,
                limit=span,
                limit_mode="earliest",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"挖窗失败 stream={stream_id} t={anchor_time}: {exc}")
            continue
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for msg in [*before, *after]:
            key = str(msg.get("message_id") or msg.get("id") or "")
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)
            merged.append(msg)
        merged.sort(key=lambda m: float(m.get("time") or 0.0))
        if merged:
            windows.append((stream_id, merged))
    return windows


async def distill_person(
    plugin: Any,
    store: "MemoryStore",
    config: "EngramMemoryConfig",
    raw_person_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """对单个人物执行一次印象蒸馏（懒加载/巡检/手动共用）。

    管线：scope 解析 → 门槛判定 → 对话窗口采集 → 分块提炼 → 融合 → 写印象。

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

    # scope 解析：人物级覆盖优先，其次全局配置
    logger.info(f"蒸馏开始 {raw_person_id}")
    meta = await store.read_distill_meta()
    scope = _resolve_scope(raw_person_id, meta, str(config.persona.scope))
    logger.info(f"生效范围 {raw_person_id} scope={scope}")

    # 门槛判定：以生效 scope 内的本人消息量为准（force 跳过统计）
    min_messages = int(config.persona.min_messages)
    scoped_count = await _count_scoped_own_messages(config, hashed_person_id, scope)
    logger.info(f"门槛统计 {raw_person_id}: scope 内 {scoped_count} 条 / 需 {min_messages}")
    if scoped_count <= 0:
        return {"ok": True, "distilled": False, "reason": "无消息可蒸馏"}
    if not force and scoped_count < min_messages:
        return {
            "ok": True,
            "distilled": False,
            "reason": (
                f"scope={scope} 范围内本人消息不足"
                f"（{scoped_count} < {min_messages}）"
            ),
        }

    # 采集对话窗口（锚点+挖窗，含 bot 回复与旁人反应）
    windows = await _collect_conversation_windows(config, hashed_person_id, scope)
    logger.info(f"窗口采集完成 {raw_person_id}: {len(windows)} 个")
    if not windows:
        return {"ok": True, "distilled": False, "reason": f"scope={scope} 范围内无可蒸对话样本"}

    # bot 发言人集合（窗口内标注「我」）
    platforms = {
        str(msg.get("platform") or "")
        for _, msgs in windows
        for msg in msgs
        if msg.get("platform")
    }
    bot_sender_ids = await _load_bot_sender_ids(platforms)

    # 格式化窗口（群名/私聊标记 + 角色标注）；无主角发言的窗口丢弃，
    # 避免浪费提炼调用（采样到主角锚点附近但主角消息全为非文本类型）
    unique_stream_ids = list(dict.fromkeys(sid for sid, _ in windows))
    labels = await _stream_display_labels(unique_stream_ids)
    window_texts: list[str] = []
    for sid, msgs in windows:
        text = _format_window(
            msgs,
            target_hashed_person_id=hashed_person_id,
            bot_sender_ids=bot_sender_ids,
            stream_label=labels.get(sid, f"[{sid[:8]}]"),
        )
        if "（主角）" in text:
            window_texts.append(text)
        else:
            logger.debug(f"丢弃无主角窗口 {sid}")
    if not window_texts:
        return {"ok": True, "distilled": False, "reason": "窗口格式化为空"}

    # 分块提炼（按字符切块；chunk_size*12 近似单块消息规模）
    chunks = _chunk_texts(window_texts, max_chars=int(config.persona.chunk_size) * 12)
    observations: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        logger.info(f"分块提炼 {raw_person_id}: 块 {index}/{len(chunks)}")
        obs = await _distill_chunk(plugin, config, "\n\n".join(chunk))
        if obs:
            observations.append(obs)
    if not observations:
        return {"ok": True, "distilled": False, "reason": "特征提炼为空"}

    logger.info(f"观察融合 {raw_person_id}: {len(observations)} 份观察")

    # 融合为最终印象：指令+人设走 system（persona 由 call_sub_agent 拼接到
    # system 开头），观察数据走 user，避免整份 prompt 双发
    max_chars = int(config.journal.impression_max_chars)
    bot_name, persona_text = _load_bot_persona()
    merge_prompt = resolve_prompt(MERGE_PROMPT_NAME, MERGE_PROMPT).format(
        bot_name=bot_name,
        max_chars=max_chars,
    )
    # 素材 = 聊天观察 + 既有记忆条目（选记忆环节按观察摘要挑选补全型
    # 条目，印象与记忆库保持一致）
    user_material = _merge_observations(observations)
    memory_lines = await _collect_memory_material(
        plugin, raw_person_id, observations=observations
    )
    if memory_lines:
        user_material = (
            "== 聊天观察 ==\n"
            + user_material
            + "\n\n== 你记下的关于他的记忆条目（真实事实） ==\n"
            + "\n".join(memory_lines)
        )
    impression_text = await call_sub_agent(
        task=str(config.internal_llm.task_name),
        request_name="engram_memory_persona_merge",
        system=merge_prompt,
        user=user_material,
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

    # 记录蒸馏元数据（保留人物级 scope 覆盖）
    try:
        entry = meta.get(raw_person_id) or {}
        entry["last_distilled_at"] = time.time()
        entry["message_count"] = scoped_count
        meta[raw_person_id] = entry
        await store.write_distill_meta(meta)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"记录蒸馏元数据失败 {raw_person_id}: {exc}")

    logger.info(
        f"人物蒸馏完成 {raw_person_id}: 窗口 {len(windows)} 个 scope={scope}"
    )
    return {
        "ok": True,
        "distilled": True,
        "reason": f"蒸馏完成（窗口 {len(windows)} 个，scope={scope}）",
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
