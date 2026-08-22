"""engram_memory 人物连接服务：封装 person_api。

提供人物认知读写、昵称历史、印象更新与 person_lookup 逻辑。
记忆表 ``person_id`` 存储原始格式 ``platform:user_id``，与 PersonInfo
的 ``platform`` + ``user_id`` 字段拼接一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api import log_api, person_api
from src.app.plugin_system.base import BaseService

if TYPE_CHECKING:
    from .memory_service import MemoryService

logger = log_api.get_logger("engram_memory.person_service")

# 后台蒸馏防重复集合挂在插件实例上的属性名
_PENDING_DISTILL_ATTR = "_engram_memory_pending_distills"


def _pending_distills(plugin: Any) -> set[str]:
    """获取挂载在插件实例上的蒸馏挂起人物集合（防重复调度）。"""
    pending = getattr(plugin, _PENDING_DISTILL_ATTR, None)
    if not isinstance(pending, set):
        pending = set()
        setattr(plugin, _PENDING_DISTILL_ATTR, pending)
    return pending


def schedule_background_distill(plugin: Any, platform: str, user_id: str) -> bool:
    """将人物印象蒸馏调度为后台任务（不阻塞调用方）。

    同一人物同时只允许一个蒸馏任务在跑；已在跑时跳过。

    Args:
        plugin: 插件实例。
        platform: 平台标识。
        user_id: 平台用户 ID。

    Returns:
        是否成功创建了后台任务。
    """
    raw_person_id = f"{platform}:{user_id}"
    pending = _pending_distills(plugin)
    if raw_person_id in pending:
        return False

    async def _run() -> None:
        """后台执行一次懒蒸馏，结束后从挂起集合移除。"""
        try:
            await ensure_person_impression_lazy(plugin, raw_person_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"后台蒸馏失败 {raw_person_id}: {exc}")
        finally:
            pending.discard(raw_person_id)

    try:
        from src.kernel.concurrency import get_task_manager

        task_manager = get_task_manager()
        task_manager.create_task(
            _run(),
            name=f"engram_memory_distill_{raw_person_id}",
            daemon=True,
        )
    except Exception as exc:  # noqa: BLE001
        pending.discard(raw_person_id)
        logger.error(f"调度后台蒸馏失败 {raw_person_id}: {exc}")
        return False
    return True


def make_config_factory(plugin: Any) -> Any:
    """构造无参配置回调（供 store 惰性创建）。"""

    def _config_factory() -> Any:
        from ..config import EngramMemoryConfig

        if isinstance(plugin.config, EngramMemoryConfig):
            return plugin.config
        return EngramMemoryConfig()

    return _config_factory


async def ensure_person_impression_lazy(plugin: Any, raw_person_id: str) -> None:
    """在当前协程内执行懒蒸馏（供后台任务调用，失败不抛出）。"""
    from .persona_distiller import ensure_person_impression
    from ..store import shared_store
    from ..config import EngramMemoryConfig

    config = plugin.config
    if not isinstance(config, EngramMemoryConfig):
        config = EngramMemoryConfig()
    store = shared_store(plugin, make_config_factory(plugin))
    await ensure_person_impression(plugin, store, config, raw_person_id)


class PersonService(BaseService):
    """人物认知读写与 person_lookup 逻辑。"""

    name: str = "person_service"
    description: str = "engram_memory 人物连接服务"
    dependencies: list[str] = []

    def _memory_service(self, memory_service: "MemoryService | None") -> "MemoryService":
        """返回记忆服务实例（传入则复用，否则新建）。"""
        if memory_service is not None:
            return memory_service
        from .memory_service import MemoryService

        return MemoryService(self.plugin)

    def _get_config(self) -> Any:
        """返回插件配置。"""
        from ..config import EngramMemoryConfig

        config = self.plugin.config
        if isinstance(config, EngramMemoryConfig):
            return config
        return EngramMemoryConfig()

    def _get_store(self) -> Any:
        """返回共享 store。"""
        from ..store import shared_store

        return shared_store(self.plugin, self._get_config)

    async def _ensure_impression_lazy(self, platform: str, user_id: str) -> None:
        """懒加载蒸馏（同步执行，仅供巡检/手动路径使用）。"""
        from .persona_distiller import ensure_person_impression

        await ensure_person_impression(
            self.plugin,
            self._get_store(),
            self._get_config(),
            f"{platform}:{user_id}",
        )

    # ------------------------------------------------------------------
    # 人物查询
    # ------------------------------------------------------------------

    async def lookup_person(
        self,
        query: str,
        *,
        memory_service: "MemoryService | None" = None,
    ) -> dict[str, Any]:
        """查询人物认知 + 记忆索引目录。

        query 含 ``:`` 视为 person_id（platform:user_id）；否则视为昵称，
        通过 ``person_api.resolve_user_id`` 反查（需 platform，默认 "qq"）。

        Args:
            query: 人物标识（person_id 或 nickname）。
            memory_service: 可选记忆服务实例。

        Returns:
            成功返回 ``{"ok": True, "person": {...}, "memories": [...]}``；
            失败返回 ``{"ok": False, "error": ...}``。
        """
        query_text = str(query or "").strip()
        if not query_text:
            return {"ok": False, "error": "查询内容不能为空"}

        identity = await self._resolve_identity(query_text)
        if identity is None:
            return {"ok": False, "error": "未找到该人物，请确认昵称或提供 person_id"}
        platform, user_id = identity

        person = await person_api.get_person(platform, user_id)
        if person is None:
            return {"ok": False, "error": "person_id 不存在"}

        # 懒加载蒸馏：缺印象或样本足够时调度到后台执行，阻塞路径只读不写。
        # 蒸馏完成后写入 PersonInfo.impression，下次对话/查询即生效
        config = self._get_config()
        if config.persona.enabled:
            try:
                schedule_background_distill(self.plugin, platform, user_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"调度后台蒸馏失败 {platform}:{user_id}: {exc}")

        memory_svc = self._memory_service(memory_service)
        memories = await self._search_person_memories(
            memory_svc, f"{platform}:{user_id}"
        )
        person_data = await self._format_person(person, platform, user_id)

        return {
            "ok": True,
            "person": person_data,
            "memories": memories,
        }

    async def _resolve_identity(self, query: str) -> tuple[str, str] | None:
        """解析 query 为 (platform, user_id)。

        含 ``:`` 直接拆分为 platform:user_id；否则用默认平台 "qq" 做昵称反查。
        反查失败或命中不唯一时返回 None。
        """
        if ":" in query:
            platform, _, user_id = query.partition(":")
            platform_clean = str(platform).strip()
            user_id_clean = str(user_id).strip()
            if not platform_clean or not user_id_clean:
                return None
            return platform_clean, user_id_clean
        # 昵称反查（默认平台 qq）
        try:
            resolved = await person_api.resolve_user_id("qq", query)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"昵称反查失败 {query}: {exc}")
            resolved = None
        if not resolved:
            return None
        return "qq", resolved

    async def _search_person_memories(
        self,
        memory_service: "MemoryService",
        raw_person_id: str,
    ) -> list[dict[str, Any]]:
        """检索与该人物相关的中长期记忆（active + archived，按 updated_at 倒序）。"""
        try:
            records = await memory_service._get_repo().search_by_person(
                person_id=raw_person_id,
                layers=["active", "archived"],
                limit=10,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"检索人物记忆失败 {raw_person_id}: {exc}")
            return []
        return [
            {
                "memory_id": record.memory_id,
                "title": record.title,
                "updated_at": record.updated_at,
            }
            for record in records
        ]

    async def _format_person(
        self,
        person: Any,
        platform: str,
        user_id: str,
    ) -> dict[str, Any]:
        """将 PersonInfo 格式化为返回字典（异步读取昵称历史）。"""
        nickname_history = await self.get_nickname_history(platform, user_id)
        return {
            "nickname": str(person.nickname or ""),
            "nickname_history": nickname_history,
            "platform": platform,
            "user_id": user_id,
            "person_id": f"{platform}:{user_id}",
            "first_interaction": float(person.first_interaction or 0.0),
            "last_interaction": float(person.last_interaction or 0.0),
            "impression": str(person.impression or "").strip() or "暂无印象",
        }

    # ------------------------------------------------------------------
    # 印象写入
    # ------------------------------------------------------------------

    async def update_impression(
        self,
        platform: str,
        user_id: str,
        impression: str,
    ) -> bool:
        """写入 PersonInfo.impression。

        Args:
            platform: 平台标识。
            user_id: 平台用户 ID。
            impression: 新的印象文本（非空）。

        Returns:
            是否更新成功。
        """
        impression_text = str(impression or "").strip()
        if not impression_text:
            return False
        return await person_api.update_user_impression(
            platform,
            user_id,
            impression=impression_text,
        )

    async def get_person_info(
        self,
        platform: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """读取人物信息为字典；不存在返回 None。"""
        person = await person_api.get_person(platform, user_id)
        if person is None:
            return None
        return await self._format_person(person, platform, user_id)

    async def get_nickname_history(
        self,
        platform: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """读取昵称历史。"""
        try:
            return await person_api.get_nickname_history(platform, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"读取昵称历史失败 {platform}:{user_id}: {exc}")
            return []
