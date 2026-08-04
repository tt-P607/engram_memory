"""engram_memory 短期记忆被动注入器。

在 ON_PROMPT_BUILD 事件中按当前消息向量检索短期层记忆，
将相关近期记忆注入流私有 actor bucket（DYNAMIC），实现跨群信息互通。
此检索命中不增加 activation_count（被动注入不计激活）。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api import log_api, prompt_api
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import EventType
from src.core.models.message import Message
from src.core.prompt import SystemReminderConsumeType, SystemReminderInsertType
from src.kernel.event import EventDecision
from src.kernel.vector_db import get_vector_db_service

from ..config import EngramMemoryConfig
from ..service.memory_service import MemoryService
from ..service.rag.vector_ops import cosine_similarity, embed_texts, to_float_vector

logger = log_api.get_logger("engram_memory.short_term_injector")


class ShortTermInjector(BaseEventHandler):
    """短期记忆被动注入。"""

    name = "short_term_injector"
    description = "按当前消息检索短期层并注入相关近期记忆（DYNAMIC）"
    weight = 20
    init_subscribe: list[EventType | str] = [EventType.ON_PROMPT_BUILD]
    _REMINDER_NAME = "engram_memory_short_term"
    _COLLECTION = "engram_memory_short_term"

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
    ) -> list[dict[str, Any]]:
        """检索短期层记忆并过滤相似度阈值。"""
        vector_db = get_vector_db_service(str(config.storage.vector_db_path))
        try:
            count = await vector_db.count(self._COLLECTION)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"统计短期集合失败: {exc}")
            return []
        if count <= 0:
            return []
        n_results = min(int(config.short_term.inject_max) * 5, count)
        try:
            result = await vector_db.query(
                collection_name=self._COLLECTION,
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["ids", "metadatas", "embeddings", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"检索短期集合失败: {exc}")
            return []

        ids_row = result.get("ids", [[]])[0] if result.get("ids") else []
        metadatas_row = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
        embeddings_row = result.get("embeddings", [[]])[0] if result.get("embeddings") else []
        documents_row = result.get("documents", [[]])[0] if result.get("documents") else []

        threshold = float(config.short_term.inject_threshold)
        candidates: list[dict[str, Any]] = []
        for index, memory_id in enumerate(ids_row):
            embedding = to_float_vector(
                embeddings_row[index] if index < len(embeddings_row) else [],
                expected_dim=len(query_vector),
                source="inject.short_term",
                collection_name=self._COLLECTION,
            )
            if not embedding:
                continue
            similarity = cosine_similarity(query_vector, embedding)
            if similarity < threshold:
                continue
            metadata = metadatas_row[index] if index < len(metadatas_row) else {}
            metadata = metadata if isinstance(metadata, dict) else {}
            candidates.append(
                {
                    "memory_id": memory_id,
                    "title": str(metadata.get("title") or ""),
                    "stream_id": str(metadata.get("stream_id") or ""),
                    "similarity": similarity,
                    "document": documents_row[index] if index < len(documents_row) else "",
                }
            )
        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        return candidates[: int(config.short_term.inject_max)]

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
        if not config.plugin.enabled:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params
        if not config.short_term.enabled:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        text = self._get_msg_text(params)
        if not text:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        try:
            query_vector = (await embed_texts(
                [text],
                task_name=str(config.internal_llm.embedding_task_name),
                request_name="engram_memory_inject_short_term",
            ))[0]
            candidates = await self._collect_candidates(query_vector, config)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"短期记忆注入失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        if not candidates:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        # 组装注入文本
        memory_service = MemoryService(self.plugin)
        lines: list[str] = ["## 近期群聊记忆", "以下是你近期在其他聊天流中记住的事情，供你参考："]
        for candidate in candidates:
            source = await memory_service.map_source(candidate.get("stream_id") or "")
            lines.append(f"- {candidate.get('title') or '未命名'}（来源：{source}）")
        lines.append("注：这些是自动总结的近期记忆，你可能需要主动回忆更多细节。")
        content = "\n".join(lines)

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
            logger.error(f"写入短期记忆 reminder 失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
        return EventDecision.SUCCESS, params

    def _clear(self, stream_id: str) -> None:
        """无匹配时删除该 reminder（add_stream_reminder 空 content 会抛异常）。"""
        try:
            prompt_api.delete_stream_reminder(stream_id, "actor", self._REMINDER_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"清理短期记忆 reminder 失败 stream={stream_id}: {exc}")
