"""Engram Memory vNext Schema 初始化与版本校验。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from src.app.plugin_system.api.storage_api import PluginDatabase

from .models import ALL_MODELS, SchemaVersionModel

SCHEMA_KEY = "engram_memory_vnext"
SCHEMA_VERSION = 1


class VNextSchema:
    """管理隔离的 vNext PluginDatabase。"""

    def __init__(self, db_path: str) -> None:
        """创建 Schema 管理器。"""
        self.database = PluginDatabase(db_path, list(ALL_MODELS))

    async def initialize(self) -> None:
        """建表并校验当前 Schema 版本。"""
        await self.database.initialize()
        async with self.database.session() as session:
            result = await session.execute(
                select(SchemaVersionModel).where(SchemaVersionModel.schema_key == SCHEMA_KEY)
            )
            current = result.scalar_one_or_none()
            if current is None:
                session.add(
                    SchemaVersionModel(
                        schema_key=SCHEMA_KEY,
                        version=SCHEMA_VERSION,
                        applied_at=datetime.now(UTC),
                    )
                )
            elif current.version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"不支持的 Engram vNext Schema 版本: {current.version}，"
                    f"当前代码要求 {SCHEMA_VERSION}"
                )

    async def close(self) -> None:
        """关闭数据库连接。"""
        await self.database.close()
