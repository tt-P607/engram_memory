"""Engram Memory vNext 插件入口。

本入口只装配 vNext Canonical/Derived Runtime，不注册旧的短期、归档、
EPA、随机闪回或即时人物蒸馏组件。旧模块保留在目录中供历史兼容代码
读取，但不再属于本插件的生产组件图。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from typing import Any

from src.app.plugin_system.api import log_api, prompt_api, router_api
from src.app.plugin_system.base import BasePlugin, register_plugin

from .config import EngramMemoryConfig
from .router.memory_admin_router import VNextMemoryAdminRouter
from .prompts import MEMORY_GUIDE_REMINDER
from .vnext.framework_bridge import (
    cancel_managed_task,
    create_managed_task,
    create_time_schedule,
    delete_owned_reminder,
    get_managed_task,
    remove_owned_schedule,
)
from .vnext.runtime_components import (
    VNextDoctorRouter,
    VNextFlashbackEventHandler,
    VNextMemoryReadTool,
    VNextMemoryReviseTool,
    VNextMemorySearchTool,
    VNextMemoryService,
    VNextMemoryWriteTool,
    VNextMessageEventHandler,
    VNextPersonLookupTool,
)
from .vnext.runtime_owner import VNextRuntimeOwner


logger = log_api.get_logger("engram_memory.plugin")

_SCHEDULE_RETRY_MAX = 600
_ENCODER_SCHEDULE_NAME = "engram_memory_vnext_encoder_flush"
_SLEEP_SCHEDULE_NAME = "engram_memory_vnext_daily_sleep"


def make_config_factory(plugin: Any) -> Any:
    """构造无参配置回调，供仍需读取插件配置的兼容模块使用。"""

    def _config_factory() -> EngramMemoryConfig:
        if isinstance(plugin.config, EngramMemoryConfig):
            return plugin.config
        return EngramMemoryConfig()

    return _config_factory


@register_plugin
class EngramMemoryPlugin(BasePlugin):
    """装配 Engram Memory vNext 的唯一插件 Owner。"""

    plugin_name: str = "engram_memory"
    configs: list[type] = [EngramMemoryConfig]
    dependent_components: list[str] = []

    def __init__(self, config: EngramMemoryConfig | None = None) -> None:
        """初始化插件及其延迟创建的 Runtime Owner。"""
        if config is not None and not isinstance(config, EngramMemoryConfig):
            # 热重载后的配置缓存可能仍属于旧模块中的同名配置类。
            config = EngramMemoryConfig.model_validate(config.model_dump())
        super().__init__(config)
        self.runtime_owner: VNextRuntimeOwner | None = None
        self._schedule_ids: list[str] = []
        self._register_task_id: str | None = None
        self._daily_schedule_id: str | None = None
        self._unloading = False
        self._flashback_reminder_streams: dict[str, set[str]] = {}

    def get_components(self) -> list[type]:
        """返回 vNext Actor 工具、服务、事件处理器和 Doctor 路由。"""
        if isinstance(self.config, EngramMemoryConfig) and not self.config.plugin.enabled:
            return []
        return [
            VNextMemorySearchTool,
            VNextMemoryReadTool,
            VNextMemoryWriteTool,
            VNextMemoryReviseTool,
            VNextPersonLookupTool,
            VNextMemoryService,
            VNextMessageEventHandler,
            VNextFlashbackEventHandler,
            VNextDoctorRouter,
            VNextMemoryAdminRouter,
        ]

    async def on_plugin_loaded(self) -> None:
        """初始化 vNext Runtime、注册引导语并安排后台生命周期。"""
        if isinstance(self.config, EngramMemoryConfig) and not self.config.plugin.enabled:
            return
        self._unloading = False
        self.runtime_owner = VNextRuntimeOwner(self)
        try:
            await self.runtime_owner.initialize()
        except BaseException:
            try:
                await self.runtime_owner.close()
            except Exception as error:  # noqa: BLE001
                logger.warning(f"清理初始化失败的 vNext Runtime 失败: {error}")
            self.runtime_owner = None
            raise
        for component in (VNextDoctorRouter, VNextMemoryAdminRouter):
            signature = f"{self.plugin_name}:router:{component.name}"
            if router_api.get_mounted_router(signature) is not None:
                await router_api.reload_router(signature, self)
        reminder_registered = False
        try:
            prompt_api.add_system_reminder(
                bucket="actor",
                name="engram_memory_guide",
                content=MEMORY_GUIDE_REMINDER,
                insert_type=(
                    prompt_api.SystemReminderInsertType.DYNAMIC
                    if self.runtime_owner.config.vnext.prompt_injection.reminder_at_end
                    else prompt_api.SystemReminderInsertType.FIXED
                ),
            )
            reminder_registered = True
        except Exception as error:  # noqa: BLE001
            logger.warning(f"注册 vNext 记忆引导语失败: {error}")
        register_coro = self._register_schedules_when_ready()
        try:
            task = create_managed_task(
                register_coro,
                name="engram_memory_vnext_register_schedule",
                daemon=True,
            )
            self._register_task_id = task.task_id
        except BaseException:
            register_coro.close()
            if reminder_registered:
                try:
                    delete_owned_reminder("actor", "engram_memory_guide")
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"清理 vNext 记忆引导语失败: {error}")
            try:
                await self.runtime_owner.close()
            except Exception as error:  # noqa: BLE001
                logger.warning(f"清理 vNext Runtime 失败: {error}")
            self.runtime_owner = None
            raise

    async def on_plugin_unloaded(self) -> None:
        """移除 vNext 调度、停止 Owner 并清理全局引导语。"""
        self._unloading = True
        self._daily_schedule_id = None
        if self._register_task_id:
            task_id = self._register_task_id
            self._register_task_id = None
            try:
                task = get_managed_task(task_id).task
                cancel_managed_task(task_id)
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
            except Exception as error:  # noqa: BLE001
                logger.debug(f"停止 vNext 调度注册任务失败: {error}")
        for schedule_id in tuple(self._schedule_ids):
            try:
                await remove_owned_schedule(schedule_id)
            except Exception as error:  # noqa: BLE001
                logger.debug(f"移除 vNext 调度 {schedule_id} 失败: {error}")
        self._schedule_ids.clear()
        for stream_id, names in self._flashback_reminder_streams.items():
            for name in names:
                try:
                    prompt_api.delete_stream_reminder(stream_id, "actor", name)
                except Exception:  # noqa: BLE001
                    logger.debug("移除流闪回 reminder 失败")
        self._flashback_reminder_streams.clear()
        if self.runtime_owner is not None:
            await self.runtime_owner.close()
            self.runtime_owner = None
        try:
            delete_owned_reminder("actor", "engram_memory_guide")
        except Exception as error:  # noqa: BLE001
            logger.debug(f"移除 vNext 记忆引导语失败: {error}")

    async def _register_schedules_when_ready(self) -> None:
        """等待统一 Scheduler 就绪后注册编码与每日 Sleep 任务。"""
        if self.runtime_owner is None:
            return
        config = self.runtime_owner.config
        flush_interval = config.vnext.candidate_encoder.max_wait_minutes * 60
        registered: list[str] = []
        for _ in range(_SCHEDULE_RETRY_MAX):
            try:
                registered.append(
                    await create_time_schedule(
                        callback=self.runtime_owner.flush_all_streams,
                        trigger_config={"interval_seconds": flush_interval},
                        is_recurring=True,
                        task_name=_ENCODER_SCHEDULE_NAME,
                        force_overwrite=True,
                    )
                )
                registered.append(await self._schedule_daily_sleep())
                self._schedule_ids = registered
                logger.info("engram_memory vNext 后台任务已注册")
                return
            except asyncio.CancelledError:
                await self._remove_schedules(registered)
                raise
            except RuntimeError:
                await self._remove_schedules(registered)
                await asyncio.sleep(0.5)
                registered.clear()
            except Exception as error:  # noqa: BLE001
                logger.warning(f"注册 vNext 后台任务失败: {error}")
                await self._remove_schedules(registered)
                await asyncio.sleep(2.0)
                registered.clear()
        logger.warning("等待 Scheduler 就绪超时，vNext 后台任务未注册")

    async def _schedule_daily_sleep(self) -> str:
        """按本地日历安排下一次每日整理，避免周期触发忽略指定时刻。"""
        if self.runtime_owner is None:
            raise RuntimeError("vNext Runtime 尚未初始化")
        daily_time = time.fromisoformat(self.runtime_owner.config.vnext.sleep.daily_time)
        now = datetime.now()
        target = now.replace(
            hour=daily_time.hour, minute=daily_time.minute,
            second=daily_time.second, microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        schedule_id = await create_time_schedule(
            callback=self._run_daily_sleep_and_reschedule,
            trigger_config={"trigger_at": target},
            is_recurring=False,
            task_name=f"{_SLEEP_SCHEDULE_NAME}_{target.strftime('%Y%m%d_%H%M%S')}",
            force_overwrite=True,
        )
        self._daily_schedule_id = schedule_id
        return schedule_id

    async def _run_daily_sleep_and_reschedule(self) -> None:
        """执行每日整理；成功或失败后均安排下一个本地日期。"""
        owner = self.runtime_owner
        if owner is None or self._unloading:
            return
        previous_id = self._daily_schedule_id
        try:
            await owner.run_daily_sleep()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.warning(f"vNext 每日整理失败: {error}")
        finally:
            if self.runtime_owner is owner and not self._unloading:
                if previous_id in self._schedule_ids:
                    self._schedule_ids.remove(previous_id)
                try:
                    self._schedule_ids.append(await self._schedule_daily_sleep())
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"安排 vNext 下一次每日整理失败: {error}")

    @staticmethod
    async def _remove_schedules(schedule_ids: list[str]) -> None:
        """清理一次未完成的调度注册，避免重试留下半套任务。"""
        for schedule_id in tuple(schedule_ids):
            try:
                await remove_owned_schedule(schedule_id)
            except Exception:  # noqa: BLE001
                continue


__all__ = ["EngramMemoryPlugin", "make_config_factory"]
