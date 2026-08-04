"""engram_memory 共享存储层。

统一管理记忆元数据库仓储实例、后台任务互斥锁与短期总结锚点文件的读写。
所有并发写盘在 ``asyncio.Lock`` 内完成，写盘采用临时文件 + ``os.replace``
原子替换，避免后台任务重叠运行时互相覆盖或读到半写文件。

所有共享对象（repo / store / 任务锁）都挂载在插件实例上，因为
``BaseService`` 非单例（``service_api.get_service`` 每次新建实例），
实例级字段无法跨调用共享。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.app.plugin_system.api.log_api import get_logger

from .service.metadata_repository import EngramMemoryMetadataRepository

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin

logger = get_logger("engram_memory.store")

# 插件实例上挂载共享对象用的属性名
_PLUGIN_ATTR_STORE = "_engram_memory_store"
_PLUGIN_ATTR_REPO = "_engram_memory_repo"
_PLUGIN_ATTR_LOCKS = "_engram_memory_job_locks"


def shared_repo(
    plugin: "BasePlugin",
    config_factory: Callable[[], Any],
) -> EngramMemoryMetadataRepository:
    """获取挂载在插件实例上的记忆元数据库仓储单例（懒创建 + 初始化）。

    Args:
        plugin: 插件实例。
        config_factory: 构造配置的回调，仅在首次创建时调用。

    Returns:
        记忆元数据库仓储实例。
    """
    repo = getattr(plugin, _PLUGIN_ATTR_REPO, None)
    if repo is None:
        config = config_factory()
        db_path = str(config.storage.metadata_db_path)
        repo = EngramMemoryMetadataRepository(db_path)
        setattr(plugin, _PLUGIN_ATTR_REPO, repo)
    return repo


def shared_locks(plugin: "BasePlugin") -> dict[str, asyncio.Lock]:
    """获取挂载在插件实例上的任务互斥锁表（key=任务名）。"""
    locks = getattr(plugin, _PLUGIN_ATTR_LOCKS, None)
    if locks is None:
        locks = {}
        setattr(plugin, _PLUGIN_ATTR_LOCKS, locks)
    return locks


def shared_store(
    plugin: "BasePlugin",
    config_factory: Callable[[], Any],
) -> "MemoryStore":
    """获取插件级共享 MemoryStore 单例（懒创建并挂载到插件实例）。

    Args:
        plugin: 插件实例。
        config_factory: 构造配置的回调，仅在首次创建时调用。

    Returns:
        MemoryStore 共享存储实例。
    """
    store = getattr(plugin, _PLUGIN_ATTR_STORE, None)
    if store is None:
        store = MemoryStore(plugin, config_factory)
        setattr(plugin, _PLUGIN_ATTR_STORE, store)
    return store


def _atomic_write_text(path: Path, text: str) -> None:
    """以原子方式写文本文件：先写临时文件再 os.replace 替换。"""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class MemoryStore:
    """状态文件读写入口（挂插件实例单例）。

    所有文件读写均在单个 ``asyncio.Lock`` 内完成（读内存 + 改内存 + 写盘
    为同一临界区），写盘用原子替换，保证后台任务重叠时数据一致。
    """

    def __init__(
        self,
        plugin: "BasePlugin",
        config_factory: Callable[[], Any],
    ) -> None:
        """初始化存储层。

        Args:
            plugin: 插件实例。
            config_factory: 构造配置的回调，惰性获取最新配置。
        """
        self._plugin = plugin
        self._config_factory = config_factory
        self._lock = asyncio.Lock()

    def _get_anchors_path(self) -> Path:
        """返回短期总结锚点文件路径。"""
        config = self._config_factory()
        data_dir = Path(str(config.storage.metadata_db_path)).parent
        return data_dir / ".summarizer_anchors.json"

    async def read_anchors(self) -> dict[str, dict[str, Any]]:
        """读取短期总结增量锚点，格式 ``{stream_id: {last_message_time, last_message_id}}``。"""
        async with self._lock:
            path = self._get_anchors_path()
            try:
                text = path.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                return {}
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {}
            if not isinstance(parsed, dict):
                return {}
            return {
                str(key): value
                for key, value in parsed.items()
                if isinstance(value, dict)
            }

    async def write_anchors(self, anchors: dict[str, dict[str, Any]]) -> None:
        """原子写入短期总结增量锚点。"""
        async with self._lock:
            _atomic_write_text(
                self._get_anchors_path(),
                json.dumps(anchors, ensure_ascii=False, indent=2),
            )
