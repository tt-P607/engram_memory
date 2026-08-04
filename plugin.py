"""engram_memory 插件入口。

三层记忆（短期/中期/长期）+ 人物连接。负责组件注册、生命周期、提示词
模板注册与短期记忆后台任务调度。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.app.plugin_system.api import log_api, prompt_api
from src.app.plugin_system.base import BasePlugin, register_plugin
from src.app.plugin_system.types import PromptTemplate
from src.kernel.concurrency import get_task_manager

from .agent.memory_delete import MemoryDeleteTool
from .agent.memory_read import MemoryReadTool
from .agent.memory_search import MemorySearchTool
from .agent.memory_write import MemoryWriteTool
from .agent.person_lookup import PersonLookupTool
from .config import EngramMemoryConfig
from .event_handler.flashback_injector import FlashbackInjector
from .event_handler.private_chat_person_injector import PrivateChatPersonInjector
from .event_handler.short_term_injector import ShortTermInjector
from .prompts import PROMPT_TEMPLATES
from .router.memory_admin_router import MemoryAdminRouter
from .service.memory_service import MemoryService
from .service.person_service import PersonService
from .service.short_term_summarizer import summarize_short_term
from .store import shared_repo, shared_store

logger = log_api.get_logger("engram_memory.plugin")

# 短期总结默认间隔（分钟）
_SUMMARIZER_INTERVAL_DEFAULT = 30
# 短期清理间隔（秒，每小时）
_CLEANUP_INTERVAL_SECONDS = 3600
# 调度注册重试次数
_SCHEDULE_RETRY_MAX = 600


def make_config_factory(plugin: Any) -> Any:
    """构造无参配置回调（供 store 惰性创建）。

    Args:
        plugin: 插件实例。

    Returns:
        无参配置回调，返回插件配置实例。
    """

    def _config_factory() -> EngramMemoryConfig:
        if isinstance(plugin.config, EngramMemoryConfig):
            return plugin.config
        return EngramMemoryConfig()

    return _config_factory


@register_plugin
class EngramMemoryPlugin(BasePlugin):
    """三层记忆（短期/中期/长期）+ 人物连接插件。"""

    plugin_name: str = "engram_memory"
    plugin_description: str = "三层记忆（短期/中期/长期）+ 人物连接"
    plugin_version: str = "1.0.0"

    configs: list[type] = [EngramMemoryConfig]
    dependent_components: list[str] = []

    def __init__(self, config: EngramMemoryConfig | None = None) -> None:
        """初始化插件。"""
        super().__init__(config)
        self._schedule_ids: list[str] = []
        self._register_task_id: str | None = None
        self._job_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # 组件
    # ------------------------------------------------------------------

    def get_components(self) -> list[type]:
        """返回插件组件类（配置禁用时返回空列表）。"""
        if isinstance(self.config, EngramMemoryConfig) and not self.config.plugin.enabled:
            return []
        return [
            MemorySearchTool,
            MemoryReadTool,
            MemoryWriteTool,
            MemoryDeleteTool,
            PersonLookupTool,
            MemoryService,
            PersonService,
            ShortTermInjector,
            FlashbackInjector,
            PrivateChatPersonInjector,
            MemoryAdminRouter,
        ]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def on_plugin_loaded(self) -> None:
        """插件加载完成后：注册提示词模板、预创建共享 store 并启动后台任务注册。"""
        for name, template in PROMPT_TEMPLATES.items():
            try:
                prompt_api.register_template(PromptTemplate(name=name, template=template))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"注册提示词模板失败 {name}: {exc}")

        # 预创建共享 store（含 repo 初始化）
        try:
            _ = shared_store(self, make_config_factory(self))
            await shared_repo(self, make_config_factory(self)).initialize()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"初始化共享存储失败: {exc}")

        tm = get_task_manager()
        task = tm.create_task(
            self._register_schedules_when_ready(),
            name="engram_memory_register_schedule",
            daemon=True,
        )
        self._register_task_id = task.task_id

    async def on_plugin_unloaded(self) -> None:
        """插件卸载前：关闭 repo、移除调度并取消后台注册任务。"""
        try:
            repo = shared_repo(self, make_config_factory(self))
            await repo.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"关闭仓储失败: {exc}")

        from src.kernel.scheduler import get_unified_scheduler

        scheduler = get_unified_scheduler()
        for schedule_id in list(self._schedule_ids):
            try:
                await scheduler.remove_schedule(schedule_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"移除调度失败 {schedule_id}: {exc}")
        self._schedule_ids.clear()

        if self._register_task_id:
            try:
                get_task_manager().cancel_task(self._register_task_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"取消注册任务失败: {exc}")
            self._register_task_id = None

    # ------------------------------------------------------------------
    # 后台任务注册
    # ------------------------------------------------------------------

    async def _register_schedules_when_ready(self) -> None:
        """等待 scheduler 运行后注册短期记忆后台任务。"""
        from src.kernel.scheduler import TriggerType, get_unified_scheduler

        if not isinstance(self.config, EngramMemoryConfig):
            logger.warning("engram_memory config 未加载，无法注册 schedule")
            return

        config = self.config
        summarizer_interval = (
            int(config.short_term.summarizer_interval_minutes) * 60
            if config.short_term.summarizer_interval_minutes
            else _SUMMARIZER_INTERVAL_DEFAULT * 60
        )

        # 短期记忆相关任务仅在 short_term.enabled 时注册
        plans: list[tuple[str, int, Any]] = []
        if config.short_term.enabled:
            plans.append(
                ("engram_memory_short_term_summarizer", summarizer_interval, self._run_summarizer_job)
            )
            plans.append(
                ("engram_memory_short_term_cleanup", _CLEANUP_INTERVAL_SECONDS, self._run_cleanup_job)
            )

        scheduler = get_unified_scheduler()
        for attempt in range(_SCHEDULE_RETRY_MAX):
            try:
                registered: list[str] = []
                for task_name, interval_seconds, callback in plans:
                    schedule_id = await scheduler.create_schedule(
                        callback=callback,
                        trigger_type=TriggerType.TIME,
                        trigger_config={"interval_seconds": interval_seconds},
                        is_recurring=True,
                        task_name=task_name,
                        force_overwrite=True,
                    )
                    registered.append(schedule_id)
                self._schedule_ids = registered
                logger.info(f"engram_memory 后台任务已注册: {len(registered)} 个")
                break
            except RuntimeError:
                await asyncio.sleep(0.5)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"注册后台任务失败: {exc}")
                await asyncio.sleep(2.0)
        else:
            logger.warning("等待 scheduler 就绪超时，engram_memory 后台任务未注册")

    # ------------------------------------------------------------------
    # 任务回调
    # ------------------------------------------------------------------

    async def _run_summarizer_job(self) -> None:
        """短期总结任务回调。"""
        await self._run_job_locked("summarizer", self._do_summarizer)

    async def _run_cleanup_job(self) -> None:
        """短期清理任务回调。"""
        await self._run_job_locked("cleanup", self._do_cleanup)

    async def _run_job_locked(self, name: str, job) -> None:
        """带互斥锁执行后台任务，防止重叠运行。"""
        lock = self._job_locks.setdefault(name, asyncio.Lock())
        if lock.locked():
            logger.info(f"engram_memory {name} 任务已在运行，跳过本次")
            return
        try:
            async with lock:
                await job()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"engram_memory {name} 任务执行失败: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # 具体任务
    # ------------------------------------------------------------------

    async def _do_summarizer(self) -> None:
        """执行短期总结。"""
        if not isinstance(self.config, EngramMemoryConfig):
            return
        store = shared_store(self, make_config_factory(self))
        memory_service = MemoryService(self)
        await summarize_short_term(self, store, memory_service)

    async def _do_cleanup(self) -> None:
        """执行短期层清理：TTL 过期 + 数量上限（超限删最旧）。"""
        if not isinstance(self.config, EngramMemoryConfig):
            return
        from src.kernel.vector_db import get_vector_db_service

        repo = shared_repo(self, make_config_factory(self))
        now = time.time()

        vector_db = get_vector_db_service(str(self.config.storage.vector_db_path))

        async def _soft_delete(record: Any) -> None:
            """软删一条短期记忆并同步删向量。"""
            await repo.soft_delete_record(record.memory_id)
            await vector_db.delete(
                collection_name="engram_memory_short_term",
                ids=[record.memory_id],
            )

        cleaned = 0

        # 1) TTL 过期清理
        expired = await repo.list_expired_short_term(now=now)
        for record in expired:
            try:
                await _soft_delete(record)
                cleaned += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(f"清理过期短期记忆失败 {record.memory_id}: {exc}")

        # 2) 数量上限清理：未过期短期总数超过 max_items 时，删最旧的直到达标
        max_items = int(self.config.short_term.max_items)
        unexpired = await repo.list_short_term_all_unexpired(now=now)
        overflow = len(unexpired) - max_items
        if overflow > 0:
            oldest = await repo.list_short_term_oldest(limit=overflow)
            for record in oldest:
                try:
                    await _soft_delete(record)
                    cleaned += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"清理超限短期记忆失败 {record.memory_id}: {exc}")

        logger.info(f"短期清理完成: cleaned={cleaned}")
