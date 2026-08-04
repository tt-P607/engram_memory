"""engram_memory 记忆元数据仓储（基于 PluginDatabase）。

提供三层记忆（short_term / active / archived）的元数据 CRUD、检索、
激活计数、软删除、晋升与清理查询。底层使用 :class:`PluginDatabase`
与 SQLAlchemy ORM，独立 SQLite 存储，与主程序数据库隔离。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.app.plugin_system.api.storage_api import PluginDatabase

from ..models import EngramMemoryRecordModel

# 三层记忆合法 layer 值
_LAYERS: frozenset[str] = frozenset({"short_term", "active", "archived"})

# update_record 中用于区分「不更新 expires_at」与「显式清空 expires_at」的哨兵
_UNSET: object = object()


@dataclass(slots=True)
class EngramMemoryRecord:
    """记忆元数据记录。"""

    memory_id: str
    title: str
    content: str
    layer: str
    event_time: float
    stream_id: str
    person_id: str | None
    related_people: list[str]
    core_tags: list[str]
    diffusion_tags: list[str]
    opposing_tags: list[str]
    relation_memory_ids: list[str]
    novelty_energy: float
    activation_count: int
    last_activated_at: float
    is_deleted: bool
    created_at: float
    updated_at: float
    expires_at: float | None


class EngramMemoryMetadataRepository:
    """engram_memory 的 SQLAlchemy + PluginDatabase 元数据仓储。"""

    def __init__(self, db_path: str) -> None:
        """初始化仓储。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self._db = PluginDatabase(db_path, [EngramMemoryRecordModel])

    async def initialize(self) -> None:
        """初始化数据库（建表）。"""
        await self._db.initialize()

    async def close(self) -> None:
        """关闭底层 PluginDatabase 连接。"""
        await self._db.close()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_list(raw_value: Any) -> list[str]:
        """将 JSON 列文本解析为字符串列表，失败时返回空列表。"""
        if isinstance(raw_value, list):
            return [str(item) for item in raw_value if str(item).strip()]
        if not isinstance(raw_value, str) or not raw_value.strip():
            return []
        try:
            parsed = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()]

    @staticmethod
    def _dump_json_list(values: list[str] | None) -> str:
        """将字符串列表序列化为 JSON 文本。"""
        return json.dumps(list(values or []), ensure_ascii=False)

    @staticmethod
    def _normalize_layer(layer: str) -> str:
        """校验并归一化 layer 值，非法时抛 ValueError。"""
        normalized = str(layer or "").strip().lower()
        if normalized not in _LAYERS:
            raise ValueError(f"非法 layer: {layer!r}，允许值: {sorted(_LAYERS)}")
        return normalized

    def _to_record(self, row: EngramMemoryRecordModel) -> EngramMemoryRecord:
        """将 ORM 实例转换为公共 dataclass（字段一一对应，直接属性访问）。"""
        return EngramMemoryRecord(
            memory_id=row.memory_id,
            title=row.title,
            content=row.content,
            layer=row.layer,
            event_time=float(row.event_time),
            stream_id=row.stream_id,
            person_id=row.person_id,
            related_people=self._parse_json_list(row.related_people),
            core_tags=self._parse_json_list(row.core_tags),
            diffusion_tags=self._parse_json_list(row.diffusion_tags),
            opposing_tags=self._parse_json_list(row.opposing_tags),
            relation_memory_ids=self._parse_json_list(row.relation_memory_ids),
            novelty_energy=float(row.novelty_energy),
            activation_count=int(row.activation_count),
            last_activated_at=float(row.last_activated_at),
            is_deleted=bool(row.is_deleted),
            created_at=float(row.created_at),
            updated_at=float(row.updated_at),
            expires_at=float(row.expires_at) if row.expires_at is not None else None,
        )

    # ------------------------------------------------------------------
    # 写入 / 更新
    # ------------------------------------------------------------------

    async def upsert_record(
        self,
        *,
        memory_id: str,
        title: str,
        content: str,
        layer: str,
        event_time: float,
        stream_id: str,
        person_id: str | None = None,
        related_people: list[str] | None = None,
        core_tags: list[str] | None = None,
        diffusion_tags: list[str] | None = None,
        opposing_tags: list[str] | None = None,
        relation_memory_ids: list[str] | None = None,
        novelty_energy: float = 0.0,
        activation_count: int | None = None,
        last_activated_at: float | None = None,
        expires_at: float | None = None,
    ) -> None:
        """写入或更新记忆元数据。

        已存在时保留 ``created_at``，更新其余字段；更新时若
        ``activation_count`` / ``last_activated_at`` 传 None 则保留原值。

        Args:
            memory_id: 记忆唯一 ID。
            title: 标题。
            content: 全文内容。
            layer: 所在层（short_term/active/archived）。
            event_time: 事件发生时间戳。
            stream_id: 来源聊天流 ID。
            person_id: 关联核心人物（platform:user_id）。
            related_people: 涉及人物列表。
            core_tags: 核心标签列表。
            diffusion_tags: 扩散标签列表。
            opposing_tags: 对立标签列表。
            relation_memory_ids: 关联记忆 ID 列表。
            novelty_energy: 新颖度能量比。
            activation_count: 激活次数，None 表示更新时保留原值。
            last_activated_at: 最近激活时间戳，None 表示更新时保留原值。
            expires_at: TTL 到期时间戳（短期层），非短期层为 None。
        """
        now = time.time()
        normalized_layer = self._normalize_layer(layer)
        R = EngramMemoryRecordModel

        async with self._db.session() as s:
            # 保留已有记录的 created_at
            existing = await s.execute(
                select(R.created_at).where(R.memory_id == memory_id)
            )
            row = existing.first()
            created_at = float(row[0]) if row else now

            set_values: dict[str, Any] = dict(
                memory_id=memory_id,
                title=title,
                content=content,
                layer=normalized_layer,
                event_time=float(event_time),
                stream_id=stream_id,
                person_id=person_id,
                related_people=self._dump_json_list(related_people),
                core_tags=self._dump_json_list(core_tags),
                diffusion_tags=self._dump_json_list(diffusion_tags),
                opposing_tags=self._dump_json_list(opposing_tags),
                relation_memory_ids=self._dump_json_list(relation_memory_ids),
                novelty_energy=float(novelty_energy),
                is_deleted=0,
                created_at=created_at,
                updated_at=now,
                expires_at=expires_at,
            )
            if activation_count is not None:
                set_values["activation_count"] = int(activation_count)
            if last_activated_at is not None:
                set_values["last_activated_at"] = float(last_activated_at)

            # 冲突更新时不再写主键列（主键用于定位，不参与 SET）
            conflict_set = {
                key: value for key, value in set_values.items() if key != "memory_id"
            }
            stmt = sqlite_insert(R).values(**set_values).on_conflict_do_update(
                index_elements=["memory_id"],
                set_=conflict_set,
            )
            await s.execute(stmt)

    async def update_record(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        layer: str | None = None,
        event_time: float | None = None,
        stream_id: str | None = None,
        person_id: str | None = None,
        related_people: list[str] | None = None,
        core_tags: list[str] | None = None,
        diffusion_tags: list[str] | None = None,
        opposing_tags: list[str] | None = None,
        relation_memory_ids: list[str] | None = None,
        novelty_energy: float | None = None,
        expires_at: float | object = _UNSET,
    ) -> bool:
        """按 memory_id 部分更新记忆字段。

        仅更新传入的非 None 字段（``expires_at`` 例外：用 ``_UNSET``
        哨兵区分「不更新」与「显式清空（传 None）」）。

        Args:
            memory_id: 目标记忆 ID。
            title: 新标题（可选）。
            content: 新内容（可选）。
            layer: 新层级（可选）。
            event_time: 新事件时间戳（可选）。
            stream_id: 新来源流 ID（可选）。
            person_id: 新核心人物（可选）。
            related_people: 新涉及人物列表（可选）。
            core_tags: 新核心标签（可选）。
            diffusion_tags: 新扩散标签（可选）。
            opposing_tags: 新对立标签（可选）。
            relation_memory_ids: 新关联记忆列表（可选）。
            novelty_energy: 新新颖度能量比（可选）。
            expires_at: 新 TTL 到期时间（可选）；传 None 表示清空。

        Returns:
            True 表示更新成功；False 表示记录不存在或已删除。
        """
        now = time.time()
        R = EngramMemoryRecordModel

        async with self._db.session() as s:
            existing = (
                await s.execute(
                    select(R).where(R.memory_id == memory_id, R.is_deleted == 0)
                )
            ).scalar_one_or_none()
            if existing is None:
                return False

            update_vals: dict[str, Any] = {"updated_at": now}
            if title is not None:
                update_vals["title"] = title
            if content is not None:
                update_vals["content"] = content
            if layer is not None:
                update_vals["layer"] = self._normalize_layer(layer)
            if event_time is not None:
                update_vals["event_time"] = float(event_time)
            if stream_id is not None:
                update_vals["stream_id"] = stream_id
            if person_id is not None:
                update_vals["person_id"] = person_id
            if related_people is not None:
                update_vals["related_people"] = self._dump_json_list(related_people)
            if core_tags is not None:
                update_vals["core_tags"] = self._dump_json_list(core_tags)
            if diffusion_tags is not None:
                update_vals["diffusion_tags"] = self._dump_json_list(diffusion_tags)
            if opposing_tags is not None:
                update_vals["opposing_tags"] = self._dump_json_list(opposing_tags)
            if relation_memory_ids is not None:
                update_vals["relation_memory_ids"] = self._dump_json_list(relation_memory_ids)
            if novelty_energy is not None:
                update_vals["novelty_energy"] = float(novelty_energy)
            if expires_at is not _UNSET:
                update_vals["expires_at"] = expires_at

            await s.execute(
                update(R).where(R.memory_id == memory_id).values(**update_vals)
            )
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    async def get_record(
        self, memory_id: str, *, include_deleted: bool = False
    ) -> EngramMemoryRecord | None:
        """按 memory_id 查询单条元数据。"""
        records = await self.get_records_map([memory_id], include_deleted=include_deleted)
        return records.get(memory_id)

    async def get_records_map(
        self, memory_ids: list[str], *, include_deleted: bool = False
    ) -> dict[str, EngramMemoryRecord]:
        """按 memory_id 列表批量查询元数据。

        Args:
            memory_ids: 待查询的 memory_id 列表，为空时返回空字典。
            include_deleted: 是否包含已软删记录，默认 False。

        Returns:
            ``{memory_id: EngramMemoryRecord}`` 字典，找不到的 id 不出现。
        """
        if not memory_ids:
            return {}
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = select(R).where(R.memory_id.in_(memory_ids))
            if not include_deleted:
                stmt = stmt.where(R.is_deleted == 0)
            rows = (await s.execute(stmt)).scalars().all()
        return {row.memory_id: self._to_record(row) for row in rows}

    async def soft_delete_record(self, memory_id: str) -> bool:
        """将指定记忆标记为软删除（is_deleted=1）。"""
        now = time.time()
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            result = await s.execute(
                update(R)
                .where(R.memory_id == memory_id, R.is_deleted == 0)
                .values(is_deleted=1, updated_at=now)
            )
        return bool(getattr(result, "rowcount", 0) or 0)

    async def hard_delete_records(self, memory_ids: list[str]) -> int:
        """物理删除指定记忆记录。"""
        if not memory_ids:
            return 0
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            result = await s.execute(delete(R).where(R.memory_id.in_(memory_ids)))
        return int(getattr(result, "rowcount", 0) or 0)

    async def update_activated(self, memory_id: str) -> None:
        """原子将指定记忆激活计数 +1 并更新最近激活时间。"""
        now = time.time()
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            await s.execute(
                update(R)
                .where(R.memory_id == memory_id)
                .values(
                    activation_count=R.activation_count + 1,
                    last_activated_at=now,
                    updated_at=now,
                )
            )

    async def list_by_layer(
        self,
        *,
        layer: str,
        limit: int = 200,
        include_deleted: bool = False,
    ) -> list[EngramMemoryRecord]:
        """按层列出记忆（updated_at 倒序）。"""
        normalized_layer = self._normalize_layer(layer)
        qb = self._db.query(EngramMemoryRecordModel).filter(layer=normalized_layer)
        if not include_deleted:
            qb = qb.filter(is_deleted=0)
        rows = await qb.order_by("-updated_at").limit(max(1, int(limit))).all()
        return [self._to_record(r) for r in rows]  # type: ignore[arg-type]

    async def list_active(
        self, *, limit: int = 200
    ) -> list[EngramMemoryRecord]:
        """列出中期层（active）未删除记忆，updated_at 倒序。"""
        return await self.list_by_layer(layer="active", limit=limit)

    async def list_all_active_for_review(
        self, *, limit: int = 500
    ) -> list[EngramMemoryRecord]:
        """列出中期层全量（供日记回顾子任务 3 审查）。"""
        return await self.list_by_layer(layer="active", limit=limit)

    async def list_expired_short_term(
        self, *, now: float
    ) -> list[EngramMemoryRecord]:
        """列出已过期的短期记忆（expires_at <= now 且未删除）。"""
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(R.layer == "short_term", R.is_deleted == 0, R.expires_at <= now)
                .order_by(R.expires_at)
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def list_short_term_promotable(
        self, *, now: float
    ) -> list[EngramMemoryRecord]:
        """列出短期层中可晋升的记忆（activation_count>0 且未过期、未删除）。"""
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(
                    R.layer == "short_term",
                    R.is_deleted == 0,
                    R.activation_count > 0,
                    R.expires_at > now,
                )
                .order_by(R.last_activated_at.desc())
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def list_short_term_all_unexpired(
        self, *, now: float
    ) -> list[EngramMemoryRecord]:
        """列出所有未过期且未删除的短期记忆。

        Args:
            now: 当前时间戳。

        Returns:
            短期记忆列表。
        """
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(
                    R.layer == "short_term",
                    R.is_deleted == 0,
                    R.expires_at > now,
                )
                .order_by(R.created_at.desc())
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def list_short_term_oldest(
        self, *, limit: int = 1
    ) -> list[EngramMemoryRecord]:
        """列出最旧的未删除短期记忆（created_at 升序）。

        用于短期记忆数量超限时清理最旧的条目。

        Args:
            limit: 返回条数。

        Returns:
            按创建时间从旧到新的短期记忆列表。
        """
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(R.layer == "short_term", R.is_deleted == 0)
                .order_by(R.created_at.asc())
                .limit(max(1, int(limit)))
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def list_active_by_created_range(
        self, *, start_ts: float, end_ts: float, limit: int = 500
    ) -> list[EngramMemoryRecord]:
        """列出创建时间在 [start_ts, end_ts] 区间内、未删除的 active 记忆。

        Args:
            start_ts: 区间起点。
            end_ts: 区间终点。
            limit: 最大条数。

        Returns:
            active 记忆列表。
        """
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(
                    R.layer == "active",
                    R.is_deleted == 0,
                    R.created_at >= start_ts,
                    R.created_at <= end_ts,
                )
                .order_by(R.created_at.desc())
                .limit(max(1, int(limit)))
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def search_by_person(
        self,
        *,
        person_id: str,
        layers: list[str],
        limit: int = 10,
    ) -> list[EngramMemoryRecord]:
        """按人物查询记忆（限定层，updated_at 倒序）。

        Args:
            person_id: 人物原始 ID（platform:user_id）。
            layers: 允许的层列表（如 ["active", "archived"]）。
            limit: 最大返回条数。

        Returns:
            匹配的人物相关记忆列表。
        """
        normalized_layers = [self._normalize_layer(layer) for layer in layers]
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = (
                select(R)
                .where(
                    R.person_id == person_id,
                    R.layer.in_(normalized_layers),
                    R.is_deleted == 0,
                )
                .order_by(R.updated_at.desc())
                .limit(max(1, int(limit)))
            )
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def search_records(
        self,
        *,
        keyword: str | None = None,
        layer: str | None = None,
        person_id: str | None = None,
        stream_id: str | None = None,
        include_deleted: bool = False,
        limit: int = 20,
    ) -> list[EngramMemoryRecord]:
        """按结构化约束查询记忆记录（keyword 匹配 title/content）。"""
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = select(R)
            if layer and layer.strip().lower() != "all":
                stmt = stmt.where(R.layer == self._normalize_layer(layer))
            if person_id:
                stmt = stmt.where(R.person_id == person_id)
            if stream_id:
                stmt = stmt.where(R.stream_id == stream_id)
            if keyword:
                pattern = f"%{keyword}%"
                stmt = stmt.where((R.title.like(pattern)) | (R.content.like(pattern)))
            if not include_deleted:
                stmt = stmt.where(R.is_deleted == 0)
            stmt = stmt.order_by(R.updated_at.desc()).limit(max(1, int(limit)))
            rows = (await s.execute(stmt)).scalars().all()
        return [self._to_record(r) for r in rows]

    async def count_by_layer(self, *, include_deleted: bool = False) -> dict[str, int]:
        """统计各层记忆数量。"""
        R = EngramMemoryRecordModel
        async with self._db.session() as s:
            stmt = select(R.layer, R.memory_id)
            if not include_deleted:
                stmt = stmt.where(R.is_deleted == 0)
            rows = (await s.execute(stmt)).all()
        counts: dict[str, int] = {layer: 0 for layer in _LAYERS}
        for layer, _ in rows:
            counts[str(layer)] = counts.get(str(layer), 0) + 1
        return counts
