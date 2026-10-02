"""Engram Memory vNext Schema 初始化与版本校验。"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sqlite3
import tempfile

from sqlalchemy import select

from src.app.plugin_system.api.storage_api import PluginDatabase

from .models import ALL_MODELS, SchemaVersionModel

SCHEMA_KEY = "engram_memory_vnext"
SCHEMA_VERSION = 4


class VNextSchema:
    """管理隔离的 vNext PluginDatabase。"""

    def __init__(self, db_path: str) -> None:
        """创建 Schema 管理器。"""
        self._db_path = Path(db_path)
        self.database = PluginDatabase(db_path, list(ALL_MODELS))

    @property
    def db_path(self) -> Path:
        """返回规范数据库路径，供关联的来源日志维护使用。"""
        return self._db_path

    def _check_existing_version(self) -> None:
        """只在字节副本上检查旧库，避免版本拒绝前触碰来源 WAL。"""
        if not self._db_path.is_file():
            return
        with tempfile.TemporaryDirectory(prefix="engram-schema-") as directory:
            probe = Path(directory) / "probe.db"
            shutil.copyfile(self._db_path, probe)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self._db_path}{suffix}")
                if sidecar.exists():
                    shutil.copyfile(sidecar, Path(f"{probe}{suffix}"))
            with closing(sqlite3.connect(probe)) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='engram_vnext_schema_version'"
                ).fetchone()
                if exists is None:
                    if connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
                    ).fetchone() is not None:
                        raise RuntimeError("已有数据库缺少 Engram Schema 版本，拒绝隐式适配")
                    return
                row = connection.execute(
                    "SELECT version FROM engram_vnext_schema_version WHERE schema_key=?",
                    (SCHEMA_KEY,),
                ).fetchone()
                if row is None or row[0] != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"已有 Engram 库要求显式副本迁移至 Schema {SCHEMA_VERSION}；"
                        "启动不会修改旧数据。请使用 plugins/engram_memory/scripts/migrate_schema.py"
                    )

    async def initialize(self) -> None:
        """建表并校验当前 Schema 版本。"""
        self._check_existing_version()
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
