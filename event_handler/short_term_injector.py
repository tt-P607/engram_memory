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
from ..metrics import get_metrics
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

    async def _build_query_text(
        self,
        stream_id: str,
        current_text: str,
    ) -> str:
        """构建检索文本：最近几轮消息 + 当前消息，增强短消息/代词场景召回。

        拉取历史失败时退化为仅当前消息，不阻塞注入。
        """
        try:
            from src.app.plugin_system.api import stream_api

            messages = await stream_api.get_stream_messages(stream_id, limit=8)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"拉取近期消息失败 stream={stream_id}: {exc}")
            return current_text
        recent: list[str] = []
        for message in messages:
            role = str(message.sender_role or "").lower()
            if role == "bot":
                text = str(message.processed_plain_text or message.content or "").strip()
                if text:
                    recent.append(f"我: {text[:60]}")
            else:
                name = str(message.sender_name or message.sender_id or "").strip()
                text = str(message.processed_plain_text or message.content or "").strip()
                if text:
                    recent.append(f"{name}: {text[:60]}")
        # 最近在末尾；保留最近 6 条 + 当前消息，当前消息权重最高放最后
        window = recent[-6:]
        window.append(current_text)
        return "\n".join(window)

    async def _collect_candidates(
        self,
        query_vector: list[float],
        config: EngramMemoryConfig,
        *,
        exclude_stream_id: str = "",
    ) -> list[dict[str, Any]]:
        """检索短期层记忆并过滤相似度阈值。

        Args:
            query_vector: 查询向量。
            config: 插件配置。
            exclude_stream_id: 排除该流来源的记忆（跨群注入不混入本流）。

        Returns:
            候选记忆列表（按相似度降序，截断到 inject_max）。
        """
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
            memory_stream_id = str(metadata.get("stream_id") or "")
            if exclude_stream_id and memory_stream_id == exclude_stream_id:
                continue
            candidates.append(
                {
                    "memory_id": memory_id,
                    "title": str(metadata.get("title") or ""),
                    "stream_id": memory_stream_id,
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
            # 多轮窗口检索文本（短消息/代词场景召回更稳）
            query_text = await self._build_query_text(stream_id, text)
            query_vector = (await embed_texts(
                [query_text],
                task_name=str(config.internal_llm.embedding_task_name),
                request_name="engram_memory_inject_short_term",
            ))[0]
            candidates = await self._collect_candidates(
                query_vector, config, exclude_stream_id=stream_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"短期记忆注入失败 stream={stream_id}: {exc}")
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        if not candidates:
            self._clear(stream_id)
            return EventDecision.SUCCESS, params

        get_metrics(self.plugin).incr("short_term_injected")

        # 组装注入文本：标题 + 事实摘要（document 为 "# 标题\n正文"，
        # 剥离标题行后取正文，让 LLM 不必二次调工具即可消费）
        memory_service = MemoryService(self.plugin)
        lines: list[str] = ["## 近期记忆", "以下是你近期记住的事情（来自各聊天流的自动总结），可直接使用："]
        for candidate in candidates:
            title = str(candidate.get("title") or "").strip() or "未命名"
            document = str(candidate.get("document") or "")
            body = document
            if title != "未命名" and document.startswith(f"# {title}"):
                body = document[len(f"# {title}") :].lstrip("\n")
            summary = body[:80].strip()
            source = await memory_service.map_source(candidate.get("stream_id") or "")
            if summary:
                lines.append(f"- {title}：{summary}（来源：{source}）")
            else:
                lines.append(f"- {title}（来源：{source}）")
        lines.append("注：以上为自动总结的近期记忆；需要完整细节时可用 memory_read 读取。")
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
