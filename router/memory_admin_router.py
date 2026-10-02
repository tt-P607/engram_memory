"""正式记忆、版本来源与核心人物印象的本地只读管理页。"""

from __future__ import annotations

from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy import func, select, tuple_

from src.app.plugin_system.api import database_api
from src.app.plugin_system.api.message_api import PersonInfo
from src.app.plugin_system.base import BaseRouter

from ..vnext.enums import ActorType, MemoryStatus
from ..vnext.models import (
    EvidenceMessageLinkModel,
    EvidenceMessageSnapshotModel,
    EvidenceModel,
    MemoryModel,
    MemoryRelationModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
)
from ..vnext.runtime_owner import VNextRuntimeOwner
from ..vnext.tool_service import ToolContext

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin


def _iso(value: datetime | None) -> str | None:
    """把可选时间编码为 ISO-8601 文本。"""
    return value.isoformat() if value is not None else None


def _enum(value: Any) -> str | None:
    """把 ORM 枚举转换为其持久化值。"""
    return value.value if value is not None else None


def _snapshot_text(payload: object) -> str | None:
    """只从已保存的原始消息快照中取出正文。"""
    if not isinstance(payload, dict):
        return None
    for field in ("processed_plain_text", "content"):
        text = payload.get(field)
        if isinstance(text, str) and text.strip():
            return text
    return None


class VNextMemoryAdminRouter(BaseRouter):
    """提供 loopback 限定的只读记忆档案页。"""

    name = "memory_admin"
    description = "Engram Memory vNext 本地记忆档案页"
    custom_route_path = "/engram-memory"
    cors_origins = None

    def __init__(self, plugin: BasePlugin) -> None:
        """绑定插件，并注册只读管理端点。"""
        super().__init__(plugin)

    def _owner(self) -> VNextRuntimeOwner:
        """读取插件生命周期中已创建的唯一 vNext Owner。"""
        # runtime_owner 是插件启动时动态附加的运行时字段，BasePlugin 不定义此属性。
        owner = getattr(self.plugin, "runtime_owner", None)
        if not isinstance(owner, VNextRuntimeOwner):
            raise HTTPException(status_code=503, detail="Engram vNext Runtime 尚未就绪")
        return owner

    @staticmethod
    def _html_path() -> Path:
        """返回静态管理页面路径。"""
        return Path(__file__).with_name("memory_admin.html")

    def _load_html(self) -> str:
        """读取管理页面 HTML。"""
        try:
            return self._html_path().read_text(encoding="utf-8")
        except OSError as error:
            raise HTTPException(status_code=500, detail="管理页面暂不可用") from error

    async def _source_views(
        self,
        owner: VNextRuntimeOwner,
        evidences: tuple[EvidenceModel, ...],
    ) -> list[dict[str, Any]]:
        """返回证据元数据及其数据库中已保存的来源快照。"""
        if not evidences:
            return []
        evidence_ids = tuple(item.evidence_id for item in evidences)
        async with owner.schema.database.session() as session:
            links = tuple(
                (
                    await session.scalars(
                        select(EvidenceMessageLinkModel)
                        .where(EvidenceMessageLinkModel.evidence_id.in_(evidence_ids))
                        .order_by(
                            EvidenceMessageLinkModel.evidence_id,
                            EvidenceMessageLinkModel.ordinal,
                        )
                    )
                ).all()
            )
            snapshots: dict[tuple[str, str], EvidenceMessageSnapshotModel] = {}
            if links:
                keys = tuple((item.stream_id, item.message_id) for item in links)
                snapshot_rows = (
                    await session.scalars(
                        select(EvidenceMessageSnapshotModel).where(
                            tuple_(
                                EvidenceMessageSnapshotModel.stream_id,
                                EvidenceMessageSnapshotModel.message_id,
                            ).in_(keys)
                        )
                    )
                ).all()
                snapshots = {
                    (item.stream_id, item.message_id): item for item in snapshot_rows
                }

        links_by_evidence: dict[str, list[EvidenceMessageLinkModel]] = {}
        for link in links:
            links_by_evidence.setdefault(link.evidence_id, []).append(link)

        result: list[dict[str, Any]] = []
        for evidence in evidences:
            messages: list[dict[str, Any]] = []
            for link in links_by_evidence.get(evidence.evidence_id, []):
                snapshot = snapshots.get((link.stream_id, link.message_id))
                if snapshot is None:
                    snapshot_status = "NOT_CAPTURED"
                    payload: dict[str, Any] | None = None
                elif snapshot.redacted_at is not None:
                    snapshot_status = "REDACTED"
                    payload = None
                else:
                    snapshot_status = "AVAILABLE"
                    payload = snapshot.payload
                messages.append(
                    {
                        "stream_id": link.stream_id,
                        "message_id": link.message_id,
                        "ordinal": link.ordinal,
                        "snapshot_status": snapshot_status,
                        "captured_at": _iso(snapshot.captured_at) if snapshot else None,
                        "speaker": (
                            payload.get("sender_cardname")
                            or payload.get("sender_name")
                            or payload.get("speaker")
                            if payload
                            else None
                        ),
                        "observed_at": payload.get("time") if payload else None,
                        "content": _snapshot_text(payload) if payload else None,
                    }
                )
            result.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "source_type": _enum(evidence.source_type),
                    "source_ref": evidence.source_ref,
                    "note": evidence.note,
                    "observed_at": _iso(evidence.observed_at),
                    "created_at": _iso(evidence.created_at),
                    "messages": messages,
                }
            )
        return result

    async def _subject_view(
        self,
        owner: VNextRuntimeOwner,
        revision_id: str,
        participants: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """读取记忆版本的主要人物或次要人物及其核心 ID。"""
        async with owner.schema.database.session() as session:
            if participants:
                rows = (
                    await session.scalars(
                        select(MemoryRevisionParticipantModel)
                        .where(MemoryRevisionParticipantModel.revision_id == revision_id)
                        .order_by(MemoryRevisionParticipantModel.participant_id)
                    )
                ).all()
                return [
                    {
                        "participant_kind": _enum(row.participant_kind),
                        "person_id": row.person_id,
                        "label": row.label,
                    }
                    for row in rows
                ]
            row = await session.get(MemoryRevisionSubjectModel, revision_id)
        if row is None:
            return None
        return {
            "subject_kind": _enum(row.subject_kind),
            "person_id": row.person_id,
            "subject_key": row.subject_key,
            "subject_label": row.subject_label,
        }

    def register_endpoints(self) -> None:
        """注册页面、正式记忆与核心人物印象的只读 API。"""
        app = self.app

        @app.middleware("http")
        async def require_loopback(request: Request, call_next: Any) -> Response:
            """限制页面以外的请求只能来自本机回环地址。"""
            if request.scope.get("path") == "/":
                return await call_next(request)
            client_host = request.client.host if request.client is not None else ""
            try:
                address = ip_address(client_host)
            except ValueError:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "管理 API 仅允许本机访问"},
                )
            mapped_address = address.ipv4_mapped if address.version == 6 else None
            if not address.is_loopback and not (
                mapped_address is not None and mapped_address.is_loopback
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "管理 API 仅允许本机访问"},
                )
            hostname = request.url.hostname or ""
            try:
                local_hostname = ip_address(hostname).is_loopback
            except ValueError:
                local_hostname = hostname == "localhost"
            if not local_hostname:
                return JSONResponse(status_code=403, content={"detail": "管理 API 仅接受本机地址"})
            return await call_next(request)

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            """返回不含数据的静态档案页。"""
            return self._load_html()

        @app.get("/api/status")
        async def status() -> dict[str, Any]:
            """返回正式记忆计数及自然闪回可用状态。"""
            owner = self._owner()
            async with owner.schema.database.session() as session:
                revision_count = int(await session.scalar(select(func.count()).select_from(MemoryRevisionModel)) or 0)
                memory_rows = (
                    await session.execute(
                        select(MemoryModel.status, func.count())
                        .group_by(MemoryModel.status)
                    )
                ).all()
            flashback_enabled = owner.config.vnext.flashback.enabled
            flashback_threshold = await owner.flashback.active_threshold()
            flashback_limit = owner.config.vnext.flashback.max_memories
            flashback_effective = (
                flashback_enabled and flashback_limit > 0 and flashback_threshold is not None
            )
            if not flashback_enabled:
                flashback_reason = "配置中的自动闪回开关已关闭。"
            elif flashback_limit == 0:
                flashback_reason = "配置已开启，但每轮可注入的记忆数量为0。"
            elif flashback_threshold is None:
                flashback_reason = "配置已开启，当前索引尚未设置经过实测的相关性门槛，自动注入暂不执行；主动检索仍可使用。"
            else:
                flashback_reason = "自动闪回已具备运行条件；每次仍需通过相关性、冷却及耗时检查。"
            return {
                "ok": True,
                "revision_count": revision_count,
                "memory_counts": {
                    _enum(item): int(count) for item, count in memory_rows
                },
                "flashback": {
                    "enabled": flashback_enabled,
                    "effective": flashback_effective,
                    "threshold": flashback_threshold,
                    "reason": flashback_reason,
                },
            }

        @app.get("/api/memories")
        async def list_memories(
            q: str = Query(default="", max_length=200),
            status_filter: str = Query(default="ACTIVE", alias="status", max_length=24),
            limit: int = Query(default=30, ge=1, le=100),
        ) -> dict[str, Any]:
            """按正文检索或按状态浏览正式记忆。"""
            owner = self._owner()
            normalized_query = q.strip()
            if status_filter != "ALL" and status_filter not in {
                item.value for item in MemoryStatus
            }:
                raise HTTPException(status_code=422, detail="记忆状态无效")
            if normalized_query:
                hits = await owner.tools.memory_search(
                    normalized_query,
                    ToolContext(actor_type=ActorType.ADMIN),
                    limit=limit,
                )
                items = [
                    {
                        "memory_id": hit["memory_id"],
                        "status": hit["status"],
                        "title": hit["title"],
                        "preview": hit["current_content_preview"],
                        "memory_kind": hit["memory_kind"],
                        "subject": hit["subject"],
                        "last_experienced_at": _iso(
                            cast(datetime | None, hit["last_experienced_at"])
                        ),
                        "matched_by": hit["matched_by"],
                    }
                    for hit in hits
                    if status_filter == "ALL" or hit["status"] == status_filter
                ]
                return {"ok": True, "items": items}

            statement = (
                select(MemoryModel, MemoryRevisionModel, MemoryRevisionSubjectModel)
                .join(
                    MemoryRevisionModel,
                    MemoryRevisionModel.revision_id == MemoryModel.current_revision_id,
                )
                .outerjoin(
                    MemoryRevisionSubjectModel,
                    MemoryRevisionSubjectModel.revision_id == MemoryModel.current_revision_id,
                )
                .order_by(MemoryModel.updated_at.desc(), MemoryModel.memory_id)
                .limit(limit)
            )
            if status_filter != "ALL":
                statement = statement.where(MemoryModel.status == MemoryStatus(status_filter))
            async with owner.schema.database.session() as session:
                rows = (await session.execute(statement)).all()
            return {
                "ok": True,
                "items": [
                    {
                        "memory_id": memory.memory_id,
                        "status": _enum(memory.status),
                        "title": revision.title,
                        "preview": revision.content[:180],
                        "memory_kind": _enum(revision.memory_kind),
                        "subject": (
                            {
                                "subject_kind": _enum(subject.subject_kind),
                                "person_id": subject.person_id,
                                "subject_key": subject.subject_key,
                                "subject_label": subject.subject_label,
                            }
                            if subject
                            else None
                        ),
                        "created_at": _iso(memory.created_at),
                        "updated_at": _iso(memory.updated_at),
                        "last_experienced_at": _iso(memory.last_experienced_at),
                        "revision_no": revision.revision_no,
                    }
                    for memory, revision, subject in rows
                ],
            }

        @app.get("/api/memories/{memory_id}")
        async def get_memory(memory_id: str) -> dict[str, Any]:
            """读取正式记忆版本、人物、关联证据和消息快照。"""
            owner = self._owner()
            memory = await owner.repository.get_memory(memory_id)
            if memory is None:
                raise HTTPException(status_code=404, detail="记忆不存在")
            revisions = await owner.repository.list_revisions(memory_id)
            revision_views: list[dict[str, Any]] = []
            sources: list[dict[str, Any]] = []
            for revision in revisions:
                revision_views.append(
                    {
                        "revision_id": revision.revision_id,
                        "revision_no": revision.revision_no,
                        "title": revision.title,
                        "content": revision.content,
                        "memory_kind": _enum(revision.memory_kind),
                        "observed_at": _iso(revision.observed_at),
                        "created_at": _iso(revision.created_at),
                        "change_reason": _enum(revision.change_reason),
                        "created_by_type": _enum(revision.created_by_type),
                        "subject": await self._subject_view(
                            owner, revision.revision_id
                        ),
                        "participants": await self._subject_view(
                            owner, revision.revision_id, participants=True
                        ),
                    }
                )
                revision_evidence = await owner.repository.list_evidence(
                    memory_id, revision_id=revision.revision_id
                )
                sources.extend(
                    {
                        **source,
                        "revision_no": revision.revision_no,
                    }
                    for source in await self._source_views(owner, revision_evidence)
                )
            current = await owner.repository.get_current_revision(memory_id)
            async with owner.schema.database.session() as session:
                relations = (
                    await session.scalars(
                        select(MemoryRelationModel)
                        .where(
                            (
                                (MemoryRelationModel.source_memory_id == memory_id)
                                | (MemoryRelationModel.target_memory_id == memory_id)
                            )
                        )
                        .order_by(MemoryRelationModel.created_at, MemoryRelationModel.relation_id)
                    )
                ).all()
            return {
                "ok": True,
                "item": {
                    "memory_id": memory.memory_id,
                    "status": _enum(memory.status),
                    "created_at": _iso(memory.created_at),
                    "updated_at": _iso(memory.updated_at),
                    "created_by_type": _enum(memory.created_by_type),
                    "last_experienced_at": _iso(memory.last_experienced_at),
                    "current_revision_id": memory.current_revision_id,
                    "current_revision_no": current.revision_no if current else None,
                    "revisions": revision_views,
                    "sources": sources,
                    "relations": [
                        {
                            "relation_id": relation.relation_id,
                            "related_memory_id": (
                                relation.target_memory_id
                                if relation.source_memory_id == memory_id
                                else relation.source_memory_id
                            ),
                            "relation_type": _enum(relation.relation_type),
                            "reason": relation.reason,
                            "created_at": _iso(relation.created_at),
                            "retracted_at": _iso(relation.retracted_at),
                        }
                        for relation in relations
                    ],
                },
            }

        @app.get("/api/personas")
        async def list_personas(
            q: str = Query(default="", max_length=200),
            limit: int = Query(default=60, ge=1, le=100),
        ) -> dict[str, Any]:
            """浏览核心数据库人物印象及人物 ID。"""
            people: dict[str, PersonInfo] = {}
            search_fields = ("person_id", "nickname", "cardname", "impression") if q.strip() else (None,)
            for field in search_fields:
                query = database_api.query(PersonInfo).filter(
                    impression__isnull=False, impression__ne="",
                )
                if field is not None:
                    query = query.filter(**{f"{field}__like": f"%{q.strip()}%"})
                rows = cast(
                    list[PersonInfo],
                    await query.order_by("-updated_at", "person_id").limit(limit).all(),
                )
                people.update((row.person_id, row) for row in rows)
            personas = sorted(
                people.values(), key=lambda row: (-(row.updated_at or 0), row.person_id),
            )[:limit]
            return {
                "ok": True,
                "storage": "core.person_info.impression",
                "items": [
                    {
                        "person_id": item.person_id,
                        "display_name": item.cardname or item.nickname,
                        "impression": item.impression,
                        "created_at": (
                            datetime.fromtimestamp(item.first_interaction).astimezone().isoformat()
                            if item.first_interaction is not None else None
                        ),
                        "updated_at": (
                            datetime.fromtimestamp(item.updated_at).astimezone().isoformat()
                            if item.updated_at is not None else None
                        ),
                    }
                    for item in personas
                ],
            }
