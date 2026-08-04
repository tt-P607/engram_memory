"""engram_memory 记忆数据模型。

使用独立的 ``declarative_base``，与主程序数据库的 Base 完全隔离。
由 ``PluginDatabase`` 负责在指定 SQLite 文件中按需建表。
"""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

# 独立 Base，与核心数据库隔离
Base = declarative_base()


class EngramMemoryRecordModel(Base):
    """记忆主表，对应 ``engram_memory_records``。

    三层记忆（short_term / active / archived）共用一张表，通过 ``layer`` 字段区分。
    短期层有 ``expires_at``（TTL 到期时间），中期/长期层为 NULL。
    ``person_id`` 存储原始格式 ``platform:user_id``，与 ``PersonInfo`` 的
    ``platform`` + ``user_id`` 字段拼接一致（用于人物关联检索）。
    """

    __tablename__ = "engram_memory_records"

    memory_id: Mapped[str] = mapped_column(Text, primary_key=True, comment="唯一 ID（UUID4）")
    title: Mapped[str] = mapped_column(Text, nullable=False, comment="标题")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="全文内容")
    layer: Mapped[str] = mapped_column(Text, nullable=False, comment="层级 short_term/active/archived")
    event_time: Mapped[float] = mapped_column(Float, nullable=False, comment="事件发生时间戳")
    stream_id: Mapped[str] = mapped_column(Text, nullable=False, comment="来源聊天流 ID")
    person_id: Mapped[str | None] = mapped_column(Text, nullable=True, comment="核心人物 platform:user_id")
    related_people: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="涉及人物列表（JSON）")
    core_tags: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="核心标签（JSON）")
    diffusion_tags: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="扩散标签（JSON）")
    opposing_tags: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="对立标签（JSON）")
    relation_memory_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="关联记忆 ID 列表（JSON）")
    novelty_energy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, comment="新颖度能量比")
    activation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="激活次数")
    last_activated_at: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, comment="最近激活时间戳")
    is_deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="软删除标记 0/1")
    created_at: Mapped[float] = mapped_column(Float, nullable=False, comment="创建时间戳")
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, comment="更新时间戳")
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True, comment="短期层 TTL 到期时间，非短期层为 NULL")

    __table_args__ = (
        Index("idx_engram_memory_layer_deleted", "layer", "is_deleted"),
        Index("idx_engram_memory_person_id", "person_id"),
        Index("idx_engram_memory_stream_id", "stream_id"),
        Index("idx_engram_memory_event_time", "event_time"),
    )
