"""engram_memory 记忆核心服务。

提供三层记忆（short_term / active / archived）的写入、检索、读取、
删除、晋升与来源映射。共享状态（repo / 向量库 / 锁）均通过
``self.plugin`` 上的 store 获取（Service 非单例）。
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

import numpy as np

from src.app.plugin_system.api import database_api, log_api, stream_api
from src.app.plugin_system.base import BaseService
from src.core.models.sql_alchemy import PersonInfo
from src.kernel.vector_db import get_vector_db_service

from ..config import EngramMemoryConfig
from ..store import shared_repo
from .metadata_repository import EngramMemoryMetadataRepository, EngramMemoryRecord
from .rag import epa
from .rag.deduplicator import EngramMemoryDeduplicator
from .rag.vector_ops import (
    cosine_similarity,
    embed_texts,
    sanitize_vector_metadata,
    to_float_vector,
    vector_norm_sq,
)

logger = log_api.get_logger("engram_memory.memory_service")

# 三层记忆对应的向量库 collection 名
COLLECTION_BY_LAYER: dict[str, str] = {
    "short_term": "engram_memory_short_term",
    "active": "engram_memory_active",
    "archived": "engram_memory_archived",
}

_LAYERS: tuple[str, ...] = ("short_term", "active", "archived")

# person_id 合法格式：platform:user_id（platform 与 user_id 均非空，不含空格）
_PERSON_ID_RE = re.compile(r"^[^\s:]+:[^\s:]+$")


def normalize_person_id(value: Any) -> str | None:
    """校验并规范化单个 person_id。

    合法格式为 ``platform:user_id``（如 ``qq:123456``、``minecraft:test``、
    ``live:xxx``）。名字、纯数字、含空格、空值均视为非法，返回 None。

    Args:
        value: 原始 person_id。

    Returns:
        合法的 person_id 字符串；非法返回 None。
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    if not _PERSON_ID_RE.match(raw):
        logger.warning(f"person_id 格式非法，已置空: {raw!r}")
        return None
    return raw


class MemoryService(BaseService):
    """三层记忆的写入、检索、读取、删除、晋升与来源映射。"""

    name: str = "memory_service"
    description: str = "engram_memory 记忆核心服务：三层记忆 CRUD + 检索 + EPA + 晋升"
    version: str = "1.0.0"
    dependencies: list[str] = []

    def _get_config(self) -> EngramMemoryConfig:
        """返回插件配置。"""
        config = self.plugin.config
        if isinstance(config, EngramMemoryConfig):
            return config
        return EngramMemoryConfig()

    @staticmethod
    def _config_factory(plugin: Any) -> EngramMemoryConfig:
        """构造插件配置（供 store 惰性创建）。"""
        if isinstance(plugin.config, EngramMemoryConfig):
            return plugin.config
        return EngramMemoryConfig()

    def _get_repo(self) -> EngramMemoryMetadataRepository:
        """返回共享元数据仓储。"""
        plugin = self.plugin

        def _config_factory() -> EngramMemoryConfig:
            if isinstance(plugin.config, EngramMemoryConfig):
                return plugin.config
            return EngramMemoryConfig()

        return shared_repo(plugin, _config_factory)

    def _get_vector_db(self) -> Any:
        """返回共享向量库服务。"""
        config = self._get_config()
        return get_vector_db_service(str(config.storage.vector_db_path))

    def _get_deduplicator(self) -> EngramMemoryDeduplicator:
        """返回检索结果去重器。"""
        return EngramMemoryDeduplicator()

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str]:
        """归一化标签：去空白、转小写、去重。"""
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags or []:
            cleaned = str(tag).strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    @staticmethod
    def _extract_title(content: str) -> str:
        """从内容中提取标题（取首行，最多 30 字）。"""
        first_line = next(
            (line.strip() for line in content.splitlines() if line.strip()),
            "",
        )
        return first_line[:30] or "未命名记忆"

    def _layer_collection(self, layer: str) -> str:
        """返回层对应的 collection 名。"""
        if layer not in COLLECTION_BY_LAYER:
            raise ValueError(f"非法 layer: {layer!r}")
        return COLLECTION_BY_LAYER[layer]

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    async def write_memory(
        self,
        *,
        title: str,
        content: str,
        core_tags: list[str],
        diffusion_tags: list[str],
        opposing_tags: list[str],
        event_time: float | None = None,
        layer: str = "active",
        person_id: str | None = None,
        related_people: list[str] | None = None,
        relation_memory_ids: list[str] | None = None,
        memory_id: str | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        """创建或更新记忆。

        新建：生成 UUID4 + embedding + 计算新颖度能量比；
        更新：合并字段 + 重新 embedding（如 content 变化）。
        ``relation_memory_ids`` 双向自动维护（写 A→B 自动追加 B→A，
        B 不存在时忽略该关联并在返回值中提示）。

        Args:
            title: 记忆标题。
            content: 记忆全文。
            core_tags: 核心标签列表。
            diffusion_tags: 扩散标签列表。
            opposing_tags: 对立标签列表。
            event_time: 事件发生时间戳。
            layer: 所在层（short_term/active/archived），默认 active。
            person_id: 关联核心人物（platform:user_id）。
            related_people: 涉及人物列表。
            relation_memory_ids: 关联记忆 ID 列表。
            memory_id: 已有记忆 ID，传入则为更新。
            stream_id: 来源聊天流 ID。

        Returns:
            含 memory_id / is_new / novelty_energy / ignored_relations 的字典。
        """
        config = self._get_config()
        # event_time 缺省时用当前时间（LLM 无需填时间戳）
        if event_time is None:
            event_time = time.time()
        else:
            try:
                event_time = float(event_time)
            except (TypeError, ValueError):
                event_time = time.time()
        title_clean = str(title or "").strip() or self._extract_title(str(content or ""))
        content_clean = str(content or "").strip()
        if not content_clean:
            raise ValueError("记忆内容不能为空")

        layer = str(layer or "active").strip().lower()
        if layer not in _LAYERS:
            raise ValueError(f"非法 layer: {layer!r}")

        core_tags = self._normalize_tags(core_tags)
        diffusion_tags = self._normalize_tags(diffusion_tags)
        opposing_tags = self._normalize_tags(opposing_tags)
        # person_id / related_people 强制 platform:user_id 格式，非法置空/剔除
        person_id = normalize_person_id(person_id)
        related_people = [
            p for p in (normalize_person_id(p) for p in (related_people or [])) if p
        ]

        # 存储文本统一为 "标题\n内容"
        text = f"# {title_clean}\n{content_clean}" if title_clean else content_clean
        vector = (await embed_texts(
            [text],
            task_name=config.internal_llm.embedding_task_name,
            request_name="engram_memory_write_embed",
        ))[0]

        is_new = memory_id is None
        if is_new:
            memory_id = f"mem-{uuid.uuid4().hex}"
            novelty_energy = await self._compute_novelty(
                vector, layer, config.write_conflict.top_n
            )
        else:
            novelty_energy = 0.0
            # 更新：先删除旧向量，再重新写入
            try:
                await self._get_vector_db().delete(
                    collection_name=self._layer_collection(layer),
                    ids=[memory_id],
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"删除旧向量失败 {memory_id}: {exc}")

        now = time.time()
        expires_at = None
        if layer == "short_term":
            expires_at = now + config.short_term.ttl_hours * 3600.0

        metadata = {
            "title": title_clean,
            "layer": layer,
            "event_time": float(event_time),
            "stream_id": str(stream_id or ""),
            "person_id": person_id or "",
            "core_tags": __import__("json").dumps(core_tags, ensure_ascii=False),
            "diffusion_tags": __import__("json").dumps(diffusion_tags, ensure_ascii=False),
            "opposing_tags": __import__("json").dumps(opposing_tags, ensure_ascii=False),
            "novelty_energy": float(novelty_energy),
            "activation_count": 0,
            "last_activated_at": 0.0,
            "is_deleted": 0,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self._get_vector_db().add(
                collection_name=self._layer_collection(layer),
                embeddings=[vector],
                documents=[text],
                metadatas=[sanitize_vector_metadata(metadata)],
                ids=[memory_id],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"向量入库失败 {memory_id}: {exc}")
            raise

        repo = self._get_repo()
        await repo.upsert_record(
            memory_id=memory_id,
            title=title_clean,
            content=content_clean,
            layer=layer,
            event_time=float(event_time),
            stream_id=str(stream_id or ""),
            person_id=person_id,
            related_people=related_people,
            core_tags=core_tags,
            diffusion_tags=diffusion_tags,
            opposing_tags=opposing_tags,
            relation_memory_ids=relation_memory_ids,
            novelty_energy=float(novelty_energy),
            expires_at=expires_at,
        )

        ignored_relations: list[str] = []
        if relation_memory_ids:
            await self._maintain_relations(memory_id, relation_memory_ids, ignored_relations)

        return {
            "memory_id": memory_id,
            "is_new": is_new,
            "novelty_energy": float(novelty_energy),
            "ignored_relations": ignored_relations,
        }

    async def _compute_novelty(
        self,
        vector: list[float],
        layer: str,
        top_n: int,
    ) -> float:
        """计算新向量的新颖度能量比（相对同层已有向量）。"""
        vector_db = self._get_vector_db()
        collection = self._layer_collection(layer)
        try:
            count = await vector_db.count(collection)
            if count <= 0:
                return 1.0
            result = await vector_db.query(
                collection_name=collection,
                query_embeddings=[vector],
                n_results=min(int(top_n), count),
                include=["embeddings"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"新颖度检索失败 {collection}: {exc}")
            return 1.0

        embeddings_row = self._safe_first_row(result.get("embeddings", [[]]))
        basis_vectors = [
            to_float_vector(
                embeddings_row[i] if i < len(embeddings_row) else [],
                expected_dim=len(vector),
                source="write.novelty",
                collection_name=collection,
            )
            for i in range(len(embeddings_row))
        ]
        basis_vectors = [v for v in basis_vectors if v and vector_norm_sq(v) > 1e-12]
        return epa.novelty_energy_ratio(vector, basis_vectors)

    async def _maintain_relations(
        self,
        source_id: str,
        target_ids: list[str],
        ignored: list[str],
    ) -> None:
        """双向维护记忆关联（写 A→B 自动追加 B→A）。"""
        repo = self._get_repo()
        targets = await repo.get_records_map(target_ids)
        for target_id in target_ids:
            target = targets.get(target_id)
            if target is None:
                ignored.append(target_id)
                continue
            new_relations = list(target.relation_memory_ids)
            if source_id not in new_relations:
                new_relations.append(source_id)
            await repo.update_record(target_id, relation_memory_ids=new_relations)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        layer: str = "all",
        person_id: str | None = None,
        core_tags: list[str] | None = None,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """语义检索记忆。

        流程：embedding → 并行检索指定层 collection → EPA 重塑 →
        残差去重 → person_id/core_tags 过滤 → 命中记忆激活计数 +1 →
        返回结果列表。

        Args:
            query: 搜索关键词或自然语言描述。
            layer: 检索层级（short_term/active/archived/all），默认 all。
            person_id: 按人物过滤（platform:user_id）。
            core_tags: 按核心标签过滤。
            top_n: 返回结果数量。

        Returns:
            结果列表，每项含 memory_id/title/summary/source/event_time/layer/
            person_id/core_tags/score。
        """
        config = self._get_config()
        query_text = str(query or "").strip()
        if not query_text:
            return []

        query_vector = (await embed_texts(
            [query_text],
            task_name=config.internal_llm.embedding_task_name,
            request_name="engram_memory_search_embed",
        ))[0]

        target_layers = _LAYERS if layer == "all" else (str(layer).strip().lower(),)
        for target in target_layers:
            if target not in _LAYERS:
                raise ValueError(f"非法 layer: {target!r}")

        vector_db = self._get_vector_db()
        n_results = max(1, int(top_n))

        # 候选收集（并行各层）
        candidates = await self._collect_candidates(
            vector_db, query_vector, target_layers, max(n_results, n_results * 3)
        )
        if not candidates:
            return []

        records = await self._get_repo().get_records_map(
            [str(item["memory_id"]) for item in candidates]
        )

        # 填充记录元数据
        for item in candidates:
            record = records.get(str(item["memory_id"]))
            if record is not None:
                item["record"] = record

        # EPA 重塑（短期层可跳过）
        skip_epa = config.retrieval.epa_skip_short_term and set(target_layers) == {"short_term"}
        reshaped_vector = query_vector
        beta = config.retrieval.base_beta
        if not skip_epa:
            evidence_vectors = [
                to_float_vector(
                    item.get("embedding", []),
                    expected_dim=len(query_vector),
                    source="retrieve.evidence",
                    collection_name=str(item.get("collection", "unknown")),
                )
                for item in candidates
                if item.get("embedding") is not None
            ]
            logic_depth = epa.projection_entropy_logic_depth(query_vector, evidence_vectors)
            query_core_tags = {str(t).strip().lower() for t in (core_tags or []) if str(t).strip()}
            resonance = epa.estimate_resonance(query_text, query_core_tags, set(), set())
            beta = max(
                0.0,
                min(
                    1.0,
                    config.retrieval.base_beta
                    + logic_depth * config.retrieval.logic_depth_scale
                    + (0.1 if resonance else 0.0),
                ),
            )

            core_vectors: list[tuple[list[float], float]] = []
            diffusion_vectors: list[tuple[list[float], float]] = []
            opposing_vectors: list[tuple[list[float], float]] = []
            core_boost_center = (
                config.retrieval.core_boost_min + config.retrieval.core_boost_max
            ) / 2
            query_tokens = {token for token in query_text.lower().split() if token}

            for item in candidates:
                record = item.get("record")
                if record is None:
                    continue
                embedding = to_float_vector(
                    item.get("embedding", []),
                    expected_dim=len(query_vector),
                    source="retrieve.reshape",
                    collection_name=str(item.get("collection", "unknown")),
                )
                if len(embedding) != len(query_vector):
                    continue
                similarity = max(0.0, cosine_similarity(query_vector, embedding))
                if similarity <= 1e-12:
                    continue
                item_core = set(record.core_tags)
                item_diffusion = set(record.diffusion_tags)
                item_opposing = set(record.opposing_tags)
                core_match = (query_core_tags or query_tokens) & item_core
                if core_match:
                    core_vectors.append(
                        (embedding, similarity * core_boost_center * len(core_match))
                    )
                diffusion_match = query_tokens & item_diffusion
                if diffusion_match:
                    diffusion_vectors.append(
                        (embedding, similarity * config.retrieval.diffusion_boost * len(diffusion_match))
                    )
                opposing_match = query_tokens & item_opposing
                if opposing_match:
                    opposing_vectors.append(
                        (embedding, similarity * config.retrieval.opposing_penalty * len(opposing_match))
                    )

            reshaped_vector = epa.reshape_query_vector(
                query_vector,
                beta=beta,
                core_vectors=core_vectors,
                diffusion_vectors=diffusion_vectors,
                opposing_vectors=opposing_vectors,
                energy_cutoff=config.write_conflict.energy_cutoff,
            )
            if vector_norm_sq(reshaped_vector) <= 1e-12:
                reshaped_vector = query_vector

        # 二次收集 + 打分
        candidates = await self._collect_candidates(
            vector_db, reshaped_vector, target_layers, n_results
        )
        records = await self._get_repo().get_records_map(
            [str(item["memory_id"]) for item in candidates]
        )
        for item in candidates:
            record = records.get(str(item["memory_id"]))
            if record is None:
                continue
            item["record"] = record
            embedding = to_float_vector(
                item.get("embedding", []),
                expected_dim=len(reshaped_vector),
                source="retrieve.score",
                collection_name=str(item.get("collection", "unknown")),
            )
            similarity = cosine_similarity(reshaped_vector, embedding)
            item["score"] = self._match_score_with_tags(
                query_text=query_text,
                similarity=similarity,
                record=record,
                beta=beta,
                core_tags=core_tags,
            )

        # 过滤：person_id / core_tags / 软删除
        filtered: list[dict[str, Any]] = []
        for item in candidates:
            record = item.get("record")
            if record is None:
                continue
            if record.is_deleted:
                continue
            if person_id and record.person_id != person_id:
                continue
            if core_tags:
                query_core = {str(t).strip().lower() for t in core_tags}
                if not query_core.intersection(set(record.core_tags)):
                    continue
            filtered.append(item)

        if not filtered:
            return []

        selected = self._get_deduplicator().select(
            filtered,
            limit=n_results,
            similarity_threshold=config.retrieval.deduplication_threshold,
        )

        repo = self._get_repo()
        results: list[dict[str, Any]] = []
        for item in selected:
            record = item.get("record")
            if record is None:
                continue
            # 主动检索命中 → 激活计数 +1
            await repo.update_activated(record.memory_id)
            results.append(
                {
                    "memory_id": record.memory_id,
                    "title": record.title,
                    "summary": record.content[:100],
                    "source": await self.map_source(record.stream_id),
                    "event_time": record.event_time,
                    "layer": record.layer,
                    "person_id": record.person_id,
                    "core_tags": record.core_tags,
                    "score": float(item.get("score", 0.0)),
                }
            )
        return results

    async def _collect_candidates(
        self,
        vector_db: Any,
        embedding: list[float],
        layers: tuple[str, ...],
        per_collection_limit: int,
    ) -> list[dict[str, Any]]:
        """对所有目标层集合并行查询，收集候选记忆。"""
        collected: list[dict[str, Any]] = []
        for layer in layers:
            collection = self._layer_collection(layer)
            try:
                count = await vector_db.count(collection)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"统计集合失败 {collection}: {exc}")
                continue
            if count <= 0:
                continue
            try:
                result = await vector_db.query(
                    collection_name=collection,
                    query_embeddings=[embedding],
                    n_results=min(per_collection_limit, count),
                    include=["embeddings", "metadatas", "documents", "distances"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"查询集合失败 {collection}: {exc}")
                continue

            ids_row = self._safe_first_row(result.get("ids", [[]]))
            documents_row = self._safe_first_row(result.get("documents", [[]]))
            metadatas_row = self._safe_first_row(result.get("metadatas", [[]]))
            embeddings_row = self._safe_first_row(result.get("embeddings", [[]]))

            for index, memory_id in enumerate(ids_row):
                item_embedding = to_float_vector(
                    embeddings_row[index] if index < len(embeddings_row) else [],
                    expected_dim=len(embedding),
                    source=f"retrieve.collect[{collection}][{index}]",
                    collection_name=collection,
                )
                collected.append(
                    {
                        "memory_id": memory_id,
                        "document": documents_row[index] if index < len(documents_row) else "",
                        "metadata": metadatas_row[index] if index < len(metadatas_row) else {},
                        "embedding": item_embedding,
                        "score": 0.0,
                        "collection": collection,
                        "record": None,
                    }
                )
        return collected

    def _match_score_with_tags(
        self,
        query_text: str,
        similarity: float,
        record: EngramMemoryRecord,
        beta: float,
        core_tags: list[str] | None = None,
    ) -> float:
        """基于 TAG 三角标签重叠对底层向量相似度进行修正。"""
        config = self._get_config()
        query_core_tags = {str(t).strip().lower() for t in (core_tags or []) if str(t).strip()}
        query_tokens = {token for token in query_text.lower().split() if token}

        core_overlap = len((query_core_tags or query_tokens) & set(record.core_tags))
        diffusion_overlap = len(query_tokens & set(record.diffusion_tags))
        opposing_overlap = len(query_tokens & set(record.opposing_tags))

        core_boost = (config.retrieval.core_boost_min + config.retrieval.core_boost_max) / 2
        score_delta = (
            core_boost * core_overlap
            + config.retrieval.diffusion_boost * diffusion_overlap
            - config.retrieval.opposing_penalty * opposing_overlap
        )
        return similarity + beta * score_delta

    # ------------------------------------------------------------------
    # 读取 / 删除
    # ------------------------------------------------------------------

    async def read_memories(self, memory_ids: list[str]) -> dict[str, Any]:
        """批量读取记忆全文。

        过滤 is_deleted=1；命中记忆激活计数 +1。
        返回 {"items": [...], "not_found": [...]}。

        Args:
            memory_ids: 记忆 ID 列表。

        Returns:
            含 items（完整内容 + 元数据 + source + relation_memory_ids）
            与 not_found 的字典。
        """
        ids = [str(mid).strip() for mid in (memory_ids or []) if str(mid).strip()]
        if not ids:
            return {"items": [], "not_found": []}
        records = await self._get_repo().get_records_map(ids)
        found_ids: list[str] = []
        items: list[dict[str, Any]] = []
        for mid in ids:
            record = records.get(mid)
            if record is None or record.is_deleted:
                continue
            found_ids.append(mid)
            items.append(
                {
                    "memory_id": record.memory_id,
                    "title": record.title,
                    "content": record.content,
                    "layer": record.layer,
                    "event_time": record.event_time,
                    "stream_id": record.stream_id,
                    "source": await self.map_source(record.stream_id),
                    "person_id": record.person_id,
                    "related_people": record.related_people,
                    "core_tags": record.core_tags,
                    "diffusion_tags": record.diffusion_tags,
                    "opposing_tags": record.opposing_tags,
                    "relation_memory_ids": record.relation_memory_ids,
                    "novelty_energy": record.novelty_energy,
                    "activation_count": record.activation_count,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
            )
        for mid in found_ids:
            await self._get_repo().update_activated(mid)
        not_found = [mid for mid in ids if mid not in records or records[mid].is_deleted]
        return {"items": items, "not_found": not_found}

    async def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """软删除记忆（is_deleted=1，不删向量库数据）。"""
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            return {"ok": False, "error": "memory_id 不能为空"}
        ok = await self._get_repo().soft_delete_record(memory_id)
        return {"ok": ok}

    # ------------------------------------------------------------------
    # 来源映射
    # ------------------------------------------------------------------

    async def map_source(self, stream_id: str) -> str:
        """将 stream_id 映射为可读来源文本。

        群聊: ``[平台] 群名 (群号: xxx)``；
        私聊: ``[平台] 私聊 昵称 (user_id)``；失败回退 ``[stream_id 前 8 位]``。

        Args:
            stream_id: 聊天流 ID。

        Returns:
            可读来源字符串。
        """
        stream_id = str(stream_id or "").strip()
        if not stream_id:
            return ""
        try:
            info = await stream_api.get_stream_info(stream_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"获取流信息失败 {stream_id}: {exc}")
            info = None
        if not info:
            return f"[{stream_id[:8]}]"

        platform = str(info.get("platform") or "")
        chat_type = str(info.get("chat_type") or "")
        if chat_type == "group":
            group_name = str(info.get("group_name") or "")
            group_id = str(info.get("group_id") or "")
            return f"[{platform}] {group_name} (群号: {group_id})"
        if chat_type == "private":
            # 私聊：流信息中 person_id 为哈希，反查 PersonInfo 获取昵称 + user_id
            hashed_person_id = str(info.get("person_id") or "")
            nickname = "未知用户"
            user_id = ""
            if hashed_person_id:
                try:
                    person = await database_api.get_by(PersonInfo, person_id=hashed_person_id)
                    if person is not None:
                        nickname = str(person.nickname or "") or "未知用户"
                        user_id = str(person.user_id or "")
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"反查私聊人物失败 {hashed_person_id}: {exc}")
            return f"[{platform}] 私聊 {nickname} ({user_id})"
        return f"[{stream_id[:8]}]"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_first_row(value: Any) -> list[Any]:
        """提取二维结构的首行（返回首行的记录列表）。

        ChromaDB query 返回形如 ``[numpy(rows, dim)]``（外层单元素、内层二维
        numpy 数组）或 ``[rows]``。本方法将其展开为 ``rows`` 条记录的一维列表。

        Args:
            value: 可能为 ``[numpy(rows, dim)]`` / ``[rows]`` / ``numpy(rows, dim)``。

        Returns:
            首行的记录列表；无法识别时返回空列表。
        """
        # 直接是 numpy 二维数组 → 返回行向量列表
        if isinstance(value, np.ndarray):
            if value.ndim >= 2:
                return [value[i] for i in range(value.shape[0])]
            return list(value)
        # 外层包裹：取首个元素（可能是 numpy 二维数组或 list 列表）
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, np.ndarray):
                if first.ndim >= 2:
                    return [first[i] for i in range(first.shape[0])]
                return [first]
            if isinstance(first, (list, tuple)):
                return list(first)
            return list(value)
        return []

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        """将 list-like 值安全转换为 list。"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return []
