"""engram_memory 共享存储层。

统一管理记忆元数据库仓储实例、后台任务互斥锁、回顾状态文件与日记
文件的读写。所有并发写盘在 ``asyncio.Lock`` 内完成，写盘采用临时文件 +
``os.replace`` 原子替换，避免后台任务重叠运行时互相覆盖或读到半写文件。

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

# 日记文件名中的非法字符（Windows 路径限制）
_INVALID_FILENAME_CHARS = '/\\:*?"<>|'


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
    """状态文件与日记读写入口（挂插件实例单例）。

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

    def _get_config(self) -> Any:
        """获取当前插件配置。"""
        return self._config_factory()

    def _get_journal_dir(self) -> Path:
        """返回日记目录路径（相对项目根目录解析）。"""
        config = self._get_config()
        return Path(str(config.storage.journal_dir))

    def _get_last_review_path(self) -> Path:
        """返回上次日记回顾时间戳文件路径。"""
        config = self._get_config()
        data_dir = Path(str(config.storage.metadata_db_path)).parent
        return data_dir / ".last_journal_review"

    def _get_anchors_path(self) -> Path:
        """返回短期总结锚点文件路径。"""
        config = self._get_config()
        data_dir = Path(str(config.storage.metadata_db_path)).parent
        return data_dir / ".summarizer_anchors.json"

    # ------------------------------------------------------------------
    # 回顾状态
    # ------------------------------------------------------------------

    async def read_last_review(self) -> float | None:
        """读取上次日记回顾时间戳；不存在或损坏返回 None。"""
        async with self._lock:
            path = self._get_last_review_path()
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (FileNotFoundError, OSError):
                return None
            try:
                value = float(text)
            except ValueError:
                return None
            return value

    async def write_last_review(self, ts: float) -> None:
        """原子写入上次日记回顾时间戳。"""
        async with self._lock:
            _atomic_write_text(self._get_last_review_path(), str(float(ts)))

    # ------------------------------------------------------------------
    # 短期总结锚点
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 日记文件
    # ------------------------------------------------------------------

    def journal_relative_path(self, date_str: str, stream_name: str) -> Path:
        """生成日记相对路径：``{流目录}/{日期}.md``。

        Args:
            date_str: 日期字符串（YYYY-MM-DD）。
            stream_name: 聊天流名称（群聊为 group_name，私聊为 nickname）。

        Returns:
            相对路径（流目录名 + 日期文件名）。
        """
        safe_name = "".join(
            "_" if char in _INVALID_FILENAME_CHARS else char for char in stream_name
        ).strip()
        safe_name = safe_name or "unknown"
        return Path(safe_name) / f"{date_str}.md"

    async def write_journal(self, date_str: str, stream_name: str, content: str) -> Path:
        """按聊天流分目录原子写入一篇日记 Markdown 文件。

        Args:
            date_str: 日期字符串（YYYY-MM-DD）。
            stream_name: 聊天流名称。

        Returns:
            写入的文件路径。
        """
        relative = self.journal_relative_path(date_str, stream_name)
        path = self._get_journal_dir() / relative
        async with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, content)
        return path

    async def read_journal(
        self,
        date_from: str,
        date_to: str,
        stream_name: str | None = None,
        limit: int = 7,
    ) -> list[dict[str, str]]:
        """按日期范围（含端点）读取日记。

        遍历 ``journal_dir`` 下各聊天流子目录（``{流目录}/{日期}.md``）中
        日期在区间内、stream_name 可选匹配的 ``.md`` 文件，按日期倒序取前
        ``limit`` 篇。

        Args:
            date_from: 开始日期（YYYY-MM-DD）。
            date_to: 结束日期（YYYY-MM-DD）。
            stream_name: 按流名称过滤（可选，模糊包含匹配）。
            limit: 最大返回篇数。

        Returns:
            日记列表，每项 ``{"date": str, "stream_name": str, "content": str}``。
        """
        if date_to < date_from:
            return []
        async with self._lock:
            journal_dir = self._get_journal_dir()
            if not journal_dir.is_dir():
                return []
            items: list[dict[str, str]] = []
            for path in sorted(journal_dir.glob("**/*.md"), reverse=True):
                if path.parent == journal_dir:
                    continue
                filename = path.name
                if not filename.endswith(".md") or len(filename) != 14:
                    continue
                date_str = filename[:10]
                if not (date_from <= date_str <= date_to):
                    continue
                name_part = path.parent.name
                if stream_name and stream_name not in name_part:
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning(f"读取日记失败: {path} ({exc})")
                    continue
                items.append(
                    {
                        "date": date_str,
                        "stream_name": name_part,
                        "content": content,
                    }
                )
                if len(items) >= max(1, int(limit)):
                    break
            return items
