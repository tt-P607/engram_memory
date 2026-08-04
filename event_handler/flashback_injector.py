"""engram_memory 记忆闪回注入器。

在 ON_PROMPT_BUILD 事件中，按配置概率触发，检索中长期层灰色地带
（相似度在 [gray_zone_min, gray_zone_max]）的记忆，按激活次数反向
加权随机抽取一条注入流私有 actor bucket（DYNAMIC）。
此命中不增加 activation_count（被动注入不计激活）。
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from src.app.plugin_system.api import log_api, prompt_api
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import EventType
from src.core.models.message import Message
from src.core.prompt import SystemReminderConsumeType, SystemReminderInsertType
from src.kernel.event import EventDecision
from src.kernel.vector_db import get_vector_db_service

from ..config import EngramMemoryConfig
from ..service.rag.vector_ops import cosine_similarity, embed_texts, to_float_vector

logger = log_api.get_logger("engram_memory.flashback_injector")

T = TypeVar("T")


def clamp_probability(value: float) -> float:
    """将概率值裁剪到 [0, 1]。"""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def weighted_choice(items: Sequence[T], weights: Sequence[float], *, u: float) -> T | None:
    """按权重从 items 中抽取一个元素。"""
    if not items:
        return None
    if len(items) != len(weights):
        raise ValueError("items 与 weights 长度必须一致")
    safe_weights = [max(0.0, float(w)) for w in weights]
    total = sum(safe_weights)
    if total <= 0.0:
        return items[-1]
    threshold = float(u) * total
    acc = 0.0
    for item, w in zip(items, safe_weights, strict=False):
        acc += w
        if acc >= threshold:
            return item
    return items[-1]


def activation_weight(*, activation_count: int, exponent: float) -> float:
    """根据激活次数计算抽取权重（激活越低权重越高）。"""
    count = max(0, int(activation_count))
    exp = float(exponent)
    if exp <= 0.0:
        exp = 1.0
    return 1.0 / ((count + 1) ** exp)


class FlashbackInjector(BaseEventHandler):
    """记忆闪回注入。"""

    name = "flashback_injector"
    description = "基于语义关联的概率性闪回（DYNAMIC）"
    weight = 10
    init_subscribe: list[EventType | str] = [EventType.ON_PROMPT_BUILD]
    _REMINDER_NAME = "engram_memory_flashback"
    _COLLECTIONS = ("engram_memory_active", "engram_memory_archived")

    def __init__(self, plugin: Any) -> None:
        """初始化闪回注入器。"""
        super().__init__(plugin)
        # memory_id -> 注入时间戳（内存冷却表，重启清空可接受）
        self._recent_flashbacks: dict[str, float] = {}

    def _get_config(self) -> EngramMemoryConfig:
        config = self.plugin.config
        if isinstance(config, EngramMemoryConfig):
            return config
        return EngramMemoryConfig()

    def _get_msg_text(self, params: dict[str, Any]) -> str:
        """从事件参数提取消息文本；无消息或无文本时返回空字符串。"""
        values = params.get("values") or {}
        message = values.get("message")
        if message is None:
            return ""
        if not isinstance(message, Message):
            return ""
        text = str(message.processed_plain_text or "").strip()
        if not text:
            text = str(message.content or "").strip()
        return text

    async def _collect_candidates(
        self,
        query_vector: list[float],
        config: EngramMemoryConfig,
        now: float,
    ) -> list[dict[str, Any]]:
        """检索中长期层灰色地带记忆，排除冷却期项。"""
        vector_db = get_vector_db_service(str(config.storage.vector_db_path))
        gray_min = float(config.flashback.gray_zone_min)
        gray_max = float(config.flashback.gray_zone_max)
        candidate_limit = int(config.flashback.candidate_limit)
        cooldown_seconds = int(config.flashback.cooldown_seconds)

        # 清理过期冷却项
        expired = [
            mid
            for mid, ts in self._recent_flashbacks.items()
            if now - ts >= cooldown_seconds
        ]
        for mid in expired:
            self._recent_flashbacks.pop(mid, None)

        candidates: list[dict[str, Any]] = []
        for collection in self._COLLECTIONS:
            try:
                count = await vector_db.count(collection)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"统计集合失败 {collection}: {exc}")
                continue
            if count <= 0:
                continue
            n_results = min(candidate_limit, count)
            try:
                result = await vector_db.query(
                    collection_name=collection,
                    query_embeddings=[query_vector],
                    n_results=n_results,
                    include=["ids", "metadatas", "embeddings", "documents"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"检索集合失败 {collection}: {exc}")
                continue

            ids_row = result.get("ids", [[]])[0] if result.get("ids") else []
            metadatas_row = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
            embeddings_row = result.get("embeddings", [[]])[0] if result.get("embeddings") else []
            documents_row = result.get("documents", [[]])[0] if result.get("documents") else []

            for index, memory_id in enumerate(ids_row):
                if memory_id in self._recent_flashbacks:
                    continue
                embedding = to_float_vector(
                    embeddings_row[index] if index < len(embeddings_row) else [],
                    expected_dim=len(query_vector),
                    source="inject.flashback",
                    collection_name=collection,
                )
                if not embedding:
                    continue
                similarity = cosine_similarity(query_vector, embedding)
                if similarity < gray_min or similarity > gray_max:
                    continue
                metadata = metadatas_row[index] if index < len(metadatas_row) else {}
                metadata = metadata if isinstance(metadata, dict) else {}
                candidates.append(
                    {
                        "memory_id": memory_id,
                        "title": str(metadata.get("title") or ""),
                        "document": documents_row[index] if index < len(documents_row) else "",
                        "activation_count": int(metadata.get("activation_count") or 0),
                        "similarity": similarity,
                    }
                )
        return candidates

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 ON_PROMPT_BUILD 事件。"""
        values = params.get("values") or {}
        stream_id = str(values.get("stream_id") or "").strip()
        if not stream_id:
            return EventDecision.SUCCESS, params

        config = self._get_config()
        if not config.plugin.enabled or not config.flashback.enabled:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 概率判定
        if not (random.random() < clamp_probability(float(config.flashback.trigger_probability))):
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        text = self._get_msg_text(params)
        if not text:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        now = time.time()
        try:
            query_vector = (await embed_texts(
                [text],
                task_name=str(config.internal_llm.embedding_task_name),
                request_name="engram_memory_inject_flashback",
            ))[0]
            candidates = await self._collect_candidates(query_vector, config, now)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"记忆闪回失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        if not candidates:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 权重选择 1 条
        weights = [
            activation_weight(
                activation_count=int(candidate.get("activation_count") or 0),
                exponent=float(config.flashback.activation_weight_exponent),
            )
            for candidate in candidates
        ]
        chosen = weighted_choice(candidates, weights, u=random.random())
        if chosen is None:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        title = str(chosen.get("title") or "")
        document = str(chosen.get("document") or "")
        summary = document[100:] if len(document) > 100 else document

        content = (
            "## 记忆闪回\n"
            "就在刚才，你突然回忆起了一些事情：\n"
            f"{title}：{summary}\n"
            "- 这是你无征兆的回忆起的东西，你可以按实际情况处理，可以选择忽视，也可以选择其他做法。\n"
            "- 注：这是你记忆中已经存在的内容，不需要重新写入。"
        )

        # 记录冷却
        self._recent_flashbacks[str(chosen["memory_id"])] = now

        try:
            prompt_api.add_stream_reminder(
                stream_id=stream_id,
                bucket="actor",
                name=self._REMINDER_NAME,
                content=content,
                insert_type=SystemReminderInsertType.DYNAMIC,
                consume=SystemReminderConsumeType.FOREVER,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"写入闪回 reminder 失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
        return EventDecision.SUCCESS, params

    def _clear(self, stream_id: str) -> None:
        """不触发时删除该 reminder（add_stream_reminder 空 content 会抛异常）。"""
        try:
            prompt_api.delete_stream_reminder(stream_id, "actor", self._REMINDER_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"清理闪回 reminder 失败 stream={stream_id}: {exc}")
