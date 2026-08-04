"""engram_memory 管理后台 Router。

提供记忆管理 Web 页面与 REST API：状态概览、记忆列表/详情/创建/更新/
删除、人物认知查询。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.app.plugin_system.api import log_api
from src.app.plugin_system.base import BaseRouter

from ..config import EngramMemoryConfig
from ..service.memory_service import MemoryService
from ..service.person_service import PersonService
from ..store import shared_repo

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin

logger = log_api.get_logger("engram_memory.admin_router")


class MemoryCreatePayload(BaseModel):
    """创建记忆请求体。"""

    title: str = Field(..., description="记忆标题")
    content: str = Field(..., description="记忆全文")
    layer: str = Field(default="active", description="层级：short_term/active/archived")
    event_time: float = Field(default=0.0, description="事件发生时间戳")
    stream_id: str = Field(default="", description="来源聊天流 ID")
    person_id: str | None = Field(default=None, description="核心人物 platform:user_id")
    related_people: list[str] = Field(default_factory=list, description="涉及人物列表")
    core_tags: list[str] = Field(default_factory=list, description="核心标签")
    diffusion_tags: list[str] = Field(default_factory=list, description="扩散标签")
    opposing_tags: list[str] = Field(default_factory=list, description="对立标签")
    relation_memory_ids: list[str] = Field(default_factory=list, description="关联记忆 ID")


class MemoryUpdatePayload(BaseModel):
    """更新记忆请求体。"""

    title: str | None = Field(default=None, description="记忆标题")
    content: str | None = Field(default=None, description="记忆全文")
    layer: str | None = Field(default=None, description="层级")
    event_time: float | None = Field(default=None, description="事件发生时间戳")
    person_id: str | None = Field(default=None, description="核心人物")
    related_people: list[str] | None = Field(default=None, description="涉及人物")
    core_tags: list[str] | None = Field(default=None, description="核心标签")
    diffusion_tags: list[str] | None = Field(default=None, description="扩散标签")
    opposing_tags: list[str] | None = Field(default=None, description="对立标签")
    relation_memory_ids: list[str] | None = Field(default=None, description="关联记忆 ID")


class MemoryAdminRouter(BaseRouter):
    """记忆管理后台：Web 页面 + REST API。"""

    name: str = "memory_admin"
    description: str = "engram_memory 记忆管理后台"
    custom_route_path: str = "/engram-memory"
    cors_origins: list[str] = ["*"]

    def __init__(self, plugin: "BasePlugin") -> None:
        """初始化 Router。"""
        super().__init__(plugin)
        self._memory_service: MemoryService | None = None

    def _get_memory_service(self) -> MemoryService:
        """懒加载记忆服务实例。"""
        if self._memory_service is None:
            self._memory_service = MemoryService(plugin=self.plugin)
        return self._memory_service

    def _get_config(self) -> EngramMemoryConfig:
        config = self.plugin.config
        if isinstance(config, EngramMemoryConfig):
            return config
        return EngramMemoryConfig()

    @staticmethod
    def _html_path() -> Path:
        """返回后台页面 HTML 文件路径。"""
        return Path(__file__).with_name("memory_admin.html")

    def _load_html(self) -> str:
        """读取后台页面 HTML 文件。"""
        try:
            return self._html_path().read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise RuntimeError("memory_admin.html 不存在，无法加载管理后台") from exc

    def register_endpoints(self) -> None:
        """注册管理后台端点。"""
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            """管理后台首页。"""
            return self._load_html()

        @app.get("/api/status")
        async def status() -> dict[str, Any]:
            """各层计数 + 最近记录概览。"""
            repo = shared_repo(self.plugin, lambda: self._get_config())
            counts = await repo.count_by_layer()
            recent = await repo.list_by_layer(layer="active", limit=5)
            return {
                "ok": True,
                "counts": counts,
                "recent": [
                    {
                        "memory_id": r.memory_id,
                        "title": r.title,
                        "layer": r.layer,
                        "person_id": r.person_id,
                        "updated_at": r.updated_at,
                    }
                    for r in recent
                ],
            }

        @app.get("/api/stats")
        async def stats() -> dict[str, Any]:
            """全面统计：各层计数、软删数、人物数、流数、过期短期数、向量计数。"""
            from src.kernel.vector_db import get_vector_db_service

            repo = shared_repo(self.plugin, lambda: self._get_config())
            config = self._get_config()

            counts = await repo.count_by_layer()
            counts_deleted = await repo.count_by_layer(include_deleted=True)
            deleted = {
                layer: counts_deleted.get(layer, 0) - counts.get(layer, 0)
                for layer in counts
            }

            # 人物数：distinct person_id（非空）
            from ..service.metadata_repository import EngramMemoryRecordModel

            R = EngramMemoryRecordModel
            async with repo._db.session() as s:  # type: ignore[attr-defined]
                person_rows = (
                    await s.execute(
                        select(R.person_id)
                        .where(R.person_id.is_not(None), R.person_id != "")
                        .distinct()
                    )
                ).scalars().all()
            person_count = len([p for p in person_rows if p])

            # 流数：distinct stream_id（非空）
            async with repo._db.session() as s:  # type: ignore[attr-defined]
                stream_rows = (
                    await s.execute(
                        select(R.stream_id)
                        .where(R.stream_id.is_not(None), R.stream_id != "")
                        .distinct()
                    )
                ).scalars().all()
            stream_count = len([p for p in stream_rows if p])

            # 过期短期记忆数
            import time as _time

            now = _time.time()
            expired = await repo.list_expired_short_term(now=now)

            # 向量库各 collection 计数
            vector_counts: dict[str, int] = {}
            try:
                vdb = get_vector_db_service(str(config.storage.vector_db_path))
                for col in ("engram_memory_short_term", "engram_memory_active", "engram_memory_archived"):
                    try:
                        vector_counts[col.replace("engram_memory_", "")] = await vdb.count(col)
                    except Exception:  # noqa: BLE001
                        vector_counts[col.replace("engram_memory_", "")] = -1
            except Exception:  # noqa: BLE001
                pass

            return {
                "ok": True,
                "counts": counts,
                "deleted": deleted,
                "person_count": person_count,
                "stream_count": stream_count,
                "expired_short_term": len(expired),
                "vector_counts": vector_counts,
            }

        @app.get("/api/memories")
        async def list_memories(
            layer: str | None = Query(default=None, description="层级过滤"),
            person_id: str | None = Query(default=None, description="人物过滤"),
            stream_id: str | None = Query(default=None, description="流过滤"),
            q: str | None = Query(default=None, description="关键词过滤"),
            include_deleted: bool = Query(default=False, description="是否包含软删"),
            limit: int = Query(default=50, ge=1, le=500),
        ) -> dict[str, Any]:
            """记忆列表（支持 layer/person_id/stream_id/q/include_deleted 过滤）。"""
            repo = shared_repo(self.plugin, lambda: self._get_config())
            if q or person_id or stream_id:
                records = await repo.search_records(
                    keyword=q,
                    layer=layer,
                    person_id=person_id,
                    stream_id=stream_id,
                    include_deleted=include_deleted,
                    limit=limit,
                )
            elif layer:
                records = await repo.list_by_layer(
                    layer=layer, limit=limit, include_deleted=include_deleted
                )
            else:
                records = await repo.list_by_layer(
                    layer="active", limit=limit, include_deleted=include_deleted
                )
            return {
                "ok": True,
                "items": [
                    {
                        "memory_id": r.memory_id,
                        "title": r.title,
                        "layer": r.layer,
                        "person_id": r.person_id,
                        "core_tags": r.core_tags,
                        "event_time": r.event_time,
                        "updated_at": r.updated_at,
                        "is_deleted": r.is_deleted,
                    }
                    for r in records
                ],
            }

        @app.get("/api/memories/{memory_id}")
        async def get_memory(memory_id: str) -> dict[str, Any]:
            """记忆详情。"""
            repo = shared_repo(self.plugin, lambda: self._get_config())
            record = await repo.get_record(memory_id, include_deleted=True)
            if record is None:
                raise HTTPException(status_code=404, detail="记忆不存在")
            return {
                "ok": True,
                "item": {
                    "memory_id": record.memory_id,
                    "title": record.title,
                    "content": record.content,
                    "layer": record.layer,
                    "event_time": record.event_time,
                    "stream_id": record.stream_id,
                    "person_id": record.person_id,
                    "related_people": record.related_people,
                    "core_tags": record.core_tags,
                    "diffusion_tags": record.diffusion_tags,
                    "opposing_tags": record.opposing_tags,
                    "relation_memory_ids": record.relation_memory_ids,
                    "novelty_energy": record.novelty_energy,
                    "activation_count": record.activation_count,
                    "is_deleted": record.is_deleted,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                },
            }

        @app.post("/api/memories", status_code=201)
        async def create_memory(payload: MemoryCreatePayload) -> dict[str, Any]:
            """创建记忆。"""
            service = self._get_memory_service()
            result = await service.write_memory(
                title=payload.title,
                content=payload.content,
                core_tags=payload.core_tags,
                diffusion_tags=payload.diffusion_tags,
                opposing_tags=payload.opposing_tags,
                event_time=payload.event_time or time.time(),
                layer=payload.layer,
                person_id=payload.person_id,
                related_people=payload.related_people or None,
                relation_memory_ids=payload.relation_memory_ids or None,
                stream_id=payload.stream_id,
            )
            return {"ok": True, **result}

        @app.put("/api/memories/{memory_id}")
        async def update_memory(
            memory_id: str, payload: MemoryUpdatePayload
        ) -> dict[str, Any]:
            """更新记忆。"""
            service = self._get_memory_service()
            record = await service._get_repo().get_record(memory_id)
            if record is None:
                raise HTTPException(status_code=404, detail="记忆不存在")
            result = await service.write_memory(
                title=payload.title or record.title,
                content=payload.content or record.content,
                core_tags=payload.core_tags if payload.core_tags is not None else record.core_tags,
                diffusion_tags=(
                    payload.diffusion_tags
                    if payload.diffusion_tags is not None
                    else record.diffusion_tags
                ),
                opposing_tags=(
                    payload.opposing_tags
                    if payload.opposing_tags is not None
                    else record.opposing_tags
                ),
                event_time=payload.event_time or record.event_time,
                layer=payload.layer or record.layer,
                person_id=(
                    payload.person_id
                    if payload.person_id is not None
                    else record.person_id
                ),
                related_people=(
                    payload.related_people
                    if payload.related_people is not None
                    else record.related_people
                ),
                relation_memory_ids=(
                    payload.relation_memory_ids
                    if payload.relation_memory_ids is not None
                    else record.relation_memory_ids
                ),
                memory_id=memory_id,
                stream_id=record.stream_id,
            )
            return {"ok": True, **result}

        @app.delete("/api/memories/{memory_id}")
        async def delete_memory(memory_id: str) -> dict[str, Any]:
            """软删除记忆。"""
            service = self._get_memory_service()
            result = await service.delete_memory(memory_id)
            if not result.get("ok"):
                raise HTTPException(status_code=404, detail="记忆不存在或已删除")
            return {"ok": True}

        @app.get("/api/persons")
        async def lookup_person(
            q: str = Query(..., description="昵称或 person_id"),
        ) -> dict[str, Any]:
            """人物认知查询 + 相关记忆。"""
            person_service = PersonService(self.plugin)
            return await person_service.lookup_person(q)

        @app.get("/api/streams")
        async def list_streams(
            layer: str | None = Query(default=None, description="按层过滤"),
        ) -> dict[str, Any]:
            """按流聚合：每个流 ID + 该流记忆数（可含层过滤）。"""
            from ..service.metadata_repository import EngramMemoryRecordModel

            R = EngramMemoryRecordModel
            repo = shared_repo(self.plugin, lambda: self._get_config())
            async with repo._db.session() as s:  # type: ignore[attr-defined]
                stmt = (
                    select(R.stream_id, R.layer)
                    .where(R.stream_id.is_not(None), R.stream_id != "")
                )
                if layer and layer.strip().lower() != "all":
                    stmt = stmt.where(R.layer == layer)
                rows = (await s.execute(stmt)).all()
            agg: dict[str, dict[str, int]] = {}
            for sid, lyr in rows:
                sid = str(sid or "")
                if not sid:
                    continue
                item = agg.setdefault(sid, {"short_term": 0, "active": 0, "archived": 0})
                if lyr in item:
                    item[lyr] += 1
            streams = [
                {
                    "stream_id": sid,
                    "counts": item,
                    "total": sum(item.values()),
                }
                for sid, item in sorted(agg.items(), key=lambda kv: -sum(kv[1].values()))
            ]
            return {"ok": True, "streams": streams}

        @app.get("/api/persons-list")
        async def list_persons(
            limit: int = Query(default=200, ge=1, le=500),
        ) -> dict[str, Any]:
            """所有关联人物列表：person_id + 记忆数（非空，跨层）。"""
            from ..service.metadata_repository import EngramMemoryRecordModel

            R = EngramMemoryRecordModel
            repo = shared_repo(self.plugin, lambda: self._get_config())
            async with repo._db.session() as s:  # type: ignore[attr-defined]
                rows = (
                    await s.execute(
                        select(R.person_id, R.layer)
                        .where(R.person_id.is_not(None), R.person_id != "")
                    )
                ).all()
            agg: dict[str, dict[str, int]] = {}
            for pid, lyr in rows:
                pid = str(pid or "")
                if not pid:
                    continue
                item = agg.setdefault(pid, {"short_term": 0, "active": 0, "archived": 0})
                if lyr in item:
                    item[lyr] += 1
            persons = [
                {
                    "person_id": pid,
                    "counts": item,
                    "total": sum(item.values()),
                }
                for pid, item in sorted(agg.items(), key=lambda kv: -sum(kv[1].values()))
            ][: int(limit)]
            return {"ok": True, "persons": persons}
