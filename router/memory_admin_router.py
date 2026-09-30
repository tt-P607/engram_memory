"""Engram Memory vNext 本地管理页、来源回读与新候选整理入口。"""

from __future__ import annotations

import json
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy import func, select, tuple_

from src.app.plugin_system.api import database_api
from src.app.plugin_system.api.message_api import PersonInfo
from src.app.plugin_system.base import BaseRouter

from ..vnext.enums import ActorType, CandidateStatus, MemoryStatus
from ..vnext.models import (
    CandidateActionModel,
    CandidateEvidenceModel,
    CandidateModel,
    CandidateParticipantModel,
    CandidateSubjectModel,
    EvidenceMessageLinkModel,
    EvidenceMessageSnapshotModel,
    EvidenceModel,
    MemoryModel,
    MemoryRelationModel,
    MemoryRevisionModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    PersonaUpdateLogModel,
    SchemaVersionModel,
    SleepActionOperationModel,
    SleepActionPlanModel,
    SleepSessionCandidateModel,
    SleepSessionModel,
)
from ..vnext.runtime_owner import VNextRuntimeOwner
from ..vnext.schema import SCHEMA_KEY, SCHEMA_VERSION
from ..vnext.tool_service import ToolContext

if TYPE_CHECKING:
    from src.app.plugin_system.base import BasePlugin


_BACKLOG_STATUSES = (
    CandidateStatus.PENDING, CandidateStatus.PROCESSING,
    CandidateStatus.DEFERRED, CandidateStatus.FAILED,
)


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


def _sleep_trace(path: Path, sleep_session_id: str) -> tuple[list[dict[str, Any]], str | None]:
    """读取指定会话的实际事件，明确区分历史缺失与日志读取失败。"""
    events: list[dict[str, Any]] = []
    incomplete = False
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    incomplete = True
                    continue
                if isinstance(row, dict) and row.get("sleep_session_id") == sleep_session_id:
                    events.append(row)
    except FileNotFoundError:
        pass
    except OSError:
        return [], "本次会话的对话日志暂不可读，以下仍可查看数据库中的操作记录。"
    if not events:
        return [], "本次历史会话未保存逐轮模型对话，以下展示真实数据库操作记录，不补造对话。"
    if incomplete:
        return events, "日志中有未完整写入的行，当前展示可读取的实际记录。"
    if not any(row.get("event") == "model_input" for row in events):
        return events, "本次会话保存了模型输出和操作，未保存当时输入及全部工具返回；缺失部分不会补造。"
    return events, None


class VNextMemoryAdminRouter(BaseRouter):
    """提供 loopback 限定的记忆档案页和受限整理入口。"""

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

    async def _candidate_sources(
        self, owner: VNextRuntimeOwner, candidate_id: str
    ) -> list[dict[str, Any]]:
        """读取指定候选所关联的证据和来源快照。"""
        async with owner.schema.database.session() as session:
            evidence_ids = tuple(
                (
                    await session.scalars(
                        select(CandidateEvidenceModel.evidence_id).where(
                            CandidateEvidenceModel.candidate_id == candidate_id
                        )
                    )
                ).all()
            )
            evidences = (
                tuple(
                    (
                        await session.scalars(
                            select(EvidenceModel)
                            .where(EvidenceModel.evidence_id.in_(evidence_ids))
                            .order_by(EvidenceModel.observed_at, EvidenceModel.evidence_id)
                        )
                    ).all()
                )
                if evidence_ids
                else ()
            )
        return await self._source_views(owner, evidences)

    async def _subject_view(
        self,
        owner: VNextRuntimeOwner,
        revision_id: str,
        participants: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """读取记忆版本的主体或参与者。"""
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

    async def _candidate_people(
        self, owner: VNextRuntimeOwner, candidate_id: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """读取候选的主体与参与者判断。"""
        async with owner.schema.database.session() as session:
            subject = await session.get(CandidateSubjectModel, candidate_id)
            participants = (
                await session.scalars(
                    select(CandidateParticipantModel)
                    .where(CandidateParticipantModel.candidate_id == candidate_id)
                    .order_by(CandidateParticipantModel.participant_id)
                )
            ).all()
        subject_view = (
            {
                "subject_kind": _enum(subject.subject_kind),
                "person_id": subject.person_id,
                "subject_key": subject.subject_key,
                "subject_label": subject.subject_label,
            }
            if subject
            else None
        )
        participant_views = [
            {
                "participant_kind": _enum(row.participant_kind),
                "person_id": row.person_id,
                "label": row.label,
            }
            for row in participants
        ]
        return subject_view, participant_views

    def register_endpoints(self) -> None:
        """注册页面、查询 API 和当前新候选的整理入口。"""
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
            """返回仅含计数、布尔和整理起点的运行概览。"""
            owner = self._owner()
            async with owner.schema.database.session() as session:
                revision_count = int(await session.scalar(select(func.count()).select_from(MemoryRevisionModel)) or 0)
                memory_rows = (
                    await session.execute(
                        select(MemoryModel.status, func.count())
                        .group_by(MemoryModel.status)
                    )
                ).all()
                candidate_rows = (
                    await session.execute(
                        select(CandidateModel.status, func.count())
                        .group_by(CandidateModel.status)
                    )
                ).all()
                automatic_since = owner.config.vnext.sleep.automatic_since
                try:
                    cutoff = datetime.fromisoformat(
                        automatic_since.replace("Z", "+00:00")
                    )
                except ValueError:
                    cutoff = None
                historical_backlog = 0
                if cutoff is not None:
                    historical_backlog = int(
                        await session.scalar(
                            select(func.count())
                            .select_from(CandidateModel)
                            .where(
                                CandidateModel.created_at < cutoff,
                                CandidateModel.status.in_(_BACKLOG_STATUSES),
                            )
                        )
                        or 0
                    )
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
                "candidate_counts": {
                    _enum(item): int(count) for item, count in candidate_rows
                },
                "automatic_backlog_count": await owner._backlog_count(),
                "historical_backlog_count": historical_backlog,
                "automatic_since": automatic_since,
                "encoder_paused": bool(owner._encoder_backlog_blocked),
                "received_message_buffer_count": sum(
                    len(messages) for messages in owner._pending_messages.values()
                ),
                "encoder_message_threshold": owner.config.vnext.candidate_encoder.message_threshold,
                "encoder_max_wait_minutes": owner.config.vnext.candidate_encoder.max_wait_minutes,
                "flashback": {
                    "enabled": flashback_enabled,
                    "effective": flashback_effective,
                    "threshold": flashback_threshold,
                    "reason": flashback_reason,
                },
            }

        @app.post("/api/sleep/run")
        async def run_sleep(request: Request) -> dict[str, Any]:
            """从本机管理页运行正常 Sleep，仅处理自动整理范围的新候选。"""
            expected_origin = f"{request.url.scheme}://{request.url.netloc}"
            if request.headers.get("origin") != expected_origin:
                raise HTTPException(status_code=403, detail="请从本机管理页发起整理")
            owner = self._owner()
            if owner._automatic_since is None:
                raise HTTPException(status_code=409, detail="尚未设置新聊天整理起点，保留旧积压")
            ran = await owner.run_daily_sleep()
            return {"ran": ran, "sleep_session_id": owner.sleep_agent.last_session_id if ran else None}

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
                        "last_experienced_at": _iso(hit["last_experienced_at"]),
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

        @app.get("/api/candidates")
        async def list_candidates(
            q: str = Query(default="", max_length=200),
            status_filter: str | None = Query(default=None, alias="status", max_length=24),
            limit: int = Query(default=40, ge=1, le=100),
        ) -> dict[str, Any]:
            """按状态或候选原文浏览积压素材。"""
            owner = self._owner()
            statement = select(CandidateModel).order_by(
                CandidateModel.created_at.desc(), CandidateModel.candidate_id
            )
            if status_filter:
                try:
                    candidate_status = CandidateStatus(status_filter)
                except ValueError as error:
                    raise HTTPException(status_code=422, detail="候选状态无效") from error
                statement = statement.where(CandidateModel.status == candidate_status)
            if q.strip():
                statement = statement.where(
                    CandidateModel.rough_title.ilike(f"%{q.strip()}%")
                    | CandidateModel.rough_content.ilike(f"%{q.strip()}%")
                )
            statement = statement.limit(limit)
            async with owner.schema.database.session() as session:
                candidates = tuple((await session.scalars(statement)).all())
                evidence_counts: dict[str, int] = {}
                if candidates:
                    rows = (
                        await session.execute(
                            select(
                                CandidateEvidenceModel.candidate_id,
                                func.count(CandidateEvidenceModel.evidence_id),
                            )
                            .where(
                                CandidateEvidenceModel.candidate_id.in_(
                                    tuple(item.candidate_id for item in candidates)
                                )
                            )
                            .group_by(CandidateEvidenceModel.candidate_id)
                        )
                    ).all()
                    evidence_counts = {candidate_id: int(count) for candidate_id, count in rows}
            return {
                "ok": True,
                "items": [
                    {
                        "candidate_id": item.candidate_id,
                        "status": _enum(item.status),
                        "title": item.rough_title,
                        "preview": item.rough_content[:180],
                        "proposed_kind": _enum(item.proposed_kind),
                        "created_at": _iso(item.created_at),
                        "observed_at": _iso(item.observed_at),
                        "processing_session_id": item.processing_session_id,
                        "evidence_count": evidence_counts.get(item.candidate_id, 0),
                    }
                    for item in candidates
                ],
            }

        @app.get("/api/candidates/{candidate_id}")
        async def get_candidate(candidate_id: str) -> dict[str, Any]:
            """读取候选原文、状态、判断和受限关联来源。"""
            owner = self._owner()
            async with owner.schema.database.session() as session:
                candidate = await session.get(CandidateModel, candidate_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail="候选不存在")
            subject, participants = await self._candidate_people(owner, candidate_id)
            return {
                "ok": True,
                "item": {
                    "candidate_id": candidate.candidate_id,
                    "status": _enum(candidate.status),
                    "title": candidate.rough_title,
                    "content": candidate.rough_content,
                    "proposed_kind": _enum(candidate.proposed_kind),
                    "observed_at": _iso(candidate.observed_at),
                    "created_at": _iso(candidate.created_at),
                    "processing_session_id": candidate.processing_session_id,
                    "last_error": candidate.last_error,
                    "subject": subject,
                    "participants": participants,
                    "sources": await self._candidate_sources(owner, candidate_id),
                },
            }

        @app.get("/api/sleep")
        async def list_sleep_sessions(
            limit: int = Query(default=30, ge=1, le=100),
        ) -> dict[str, Any]:
            """按开始时间倒序列出真实 Sleep 会话。"""
            owner = self._owner()
            async with owner.schema.database.session() as session:
                sessions = (
                    await session.scalars(
                        select(SleepSessionModel)
                        .order_by(SleepSessionModel.started_at.desc())
                        .limit(limit)
                    )
                ).all()
            return {
                "ok": True,
                "items": [
                    {
                        "sleep_session_id": item.sleep_session_id,
                        "trigger_type": _enum(item.trigger_type),
                        "status": _enum(item.status),
                        "started_at": _iso(item.started_at),
                        "finished_at": _iso(item.finished_at),
                        "candidate_count": item.candidate_count,
                        "model_id": item.model_id,
                        "prompt_version": item.prompt_version,
                        "error_summary": item.error_summary,
                    }
                    for item in sessions
                ],
            }

        @app.get("/api/sleep/{sleep_session_id}")
        async def get_sleep_session(sleep_session_id: str) -> dict[str, Any]:
            """读取真实 Sleep 会话、认领记录与步骤和计划动作审计。"""
            owner = self._owner()
            async with owner.schema.database.session() as session:
                sleep_session = await session.get(SleepSessionModel, sleep_session_id)
                if sleep_session is None:
                    raise HTTPException(status_code=404, detail="Sleep 会话不存在")
                claims = (
                    await session.scalars(
                        select(SleepSessionCandidateModel)
                        .where(
                            SleepSessionCandidateModel.sleep_session_id
                            == sleep_session_id
                        )
                        .order_by(SleepSessionCandidateModel.claimed_at)
                    )
                ).all()
                candidate_ids = tuple(item.candidate_id for item in claims)
                candidate_titles: dict[str, str] = {}
                if candidate_ids:
                    candidate_rows = (
                        await session.execute(
                            select(CandidateModel.candidate_id, CandidateModel.rough_title)
                            .where(CandidateModel.candidate_id.in_(candidate_ids))
                        )
                    ).all()
                    candidate_titles = dict(candidate_rows)
                operations = (
                    await session.scalars(
                        select(SleepActionOperationModel)
                        .where(
                            SleepActionOperationModel.sleep_session_id
                            == sleep_session_id
                        )
                        .order_by(
                            SleepActionOperationModel.created_at,
                            SleepActionOperationModel.operation_key,
                        )
                    )
                ).all()
                plans = (
                    await session.scalars(
                        select(SleepActionPlanModel)
                        .where(SleepActionPlanModel.sleep_session_id == sleep_session_id)
                        .order_by(SleepActionPlanModel.created_at, SleepActionPlanModel.plan_key)
                    )
                ).all()
                actions = (
                    await session.scalars(
                        select(CandidateActionModel)
                        .where(CandidateActionModel.sleep_session_id == sleep_session_id)
                        .order_by(CandidateActionModel.created_at, CandidateActionModel.action_id)
                    )
                ).all()
                revision_ids = {
                    operation.result_json["revision_id"]
                    for operation in operations
                    if isinstance(operation.result_json, dict)
                    and isinstance(operation.result_json.get("revision_id"), str)
                }
                result_memories: dict[str, dict[str, Any]] = {}
                if revision_ids:
                    revision_rows = (
                        await session.execute(
                            select(
                                MemoryRevisionModel,
                                MemoryModel.status,
                                MemoryModel.current_revision_id,
                            )
                            .join(MemoryModel, MemoryModel.memory_id == MemoryRevisionModel.memory_id)
                            .where(MemoryRevisionModel.revision_id.in_(tuple(revision_ids)))
                        )
                    ).all()
                    result_memories = {
                        revision.revision_id: {
                            "memory_id": revision.memory_id,
                            "title": revision.title,
                            "content": revision.content,
                            "revision_id": revision.revision_id,
                            "revision_no": revision.revision_no,
                            "status": _enum(memory_status),
                            "is_current": revision.revision_id == current_revision_id,
                        }
                        for revision, memory_status, current_revision_id in revision_rows
                    }
            trace_events, trace_notice = _sleep_trace(
                Path(owner.config.storage.vnext_db_path).resolve().parent / "sleep-events.jsonl",
                sleep_session_id,
            )
            return {
                "ok": True,
                "item": {
                    "sleep_session_id": sleep_session.sleep_session_id,
                    "trigger_type": _enum(sleep_session.trigger_type),
                    "status": _enum(sleep_session.status),
                    "started_at": _iso(sleep_session.started_at),
                    "finished_at": _iso(sleep_session.finished_at),
                    "candidate_count": sleep_session.candidate_count,
                    "model_id": sleep_session.model_id,
                    "prompt_version": sleep_session.prompt_version,
                    "error_summary": sleep_session.error_summary,
                    "trace_events": trace_events,
                    "trace_available": bool(trace_events),
                    "trace_notice": trace_notice,
                    "claims": [
                        {
                            "candidate_id": claim.candidate_id,
                            "candidate_title": candidate_titles.get(claim.candidate_id),
                            "claimed_at": _iso(claim.claimed_at),
                            "released_at": _iso(claim.released_at),
                            "outcome": _enum(claim.outcome),
                        }
                        for claim in claims
                    ],
                    "operations": [
                        {
                            "operation_key": operation.operation_key,
                            "candidate_id": operation.candidate_id,
                            "action_type": _enum(operation.action_type),
                            "intent": operation.intent_json,
                            "status": operation.status,
                            "result": operation.result_json,
                            "error": operation.error,
                            "created_at": _iso(operation.created_at),
                            "updated_at": _iso(operation.updated_at),
                            "result_memories": [
                                result_memories[operation.result_json["revision_id"]]
                            ] if (
                                isinstance(operation.result_json, dict)
                                and operation.result_json.get("revision_id") in result_memories
                            ) else [],
                            "memory_readback_available": bool(
                                isinstance(operation.result_json, dict)
                                and operation.result_json.get("revision_id") in result_memories
                            ),
                        }
                        for operation in operations
                    ],
                    "plans": [
                        {
                            "plan_key": plan.plan_key,
                            "candidate_id": plan.candidate_id,
                            "intents": plan.intents_json,
                            "operation_keys": plan.operation_keys_json,
                            "action_count": plan.action_count,
                            "next_action_index": plan.next_action_index,
                            "status": plan.status,
                            "target_memory_ids": plan.target_memory_ids_json,
                            "created_at": _iso(plan.created_at),
                            "updated_at": _iso(plan.updated_at),
                        }
                        for plan in plans
                    ],
                    "actions": [
                        {
                            "action_id": action.action_id,
                            "candidate_id": action.candidate_id,
                            "action_type": _enum(action.action_type),
                            "result_revision_id": action.result_revision_id,
                            "note": action.note,
                            "created_at": _iso(action.created_at),
                        }
                        for action in actions
                    ],
                },
            }

        @app.get("/api/personas")
        async def list_personas(
            q: str = Query(default="", max_length=200),
            limit: int = Query(default=60, ge=1, le=100),
        ) -> dict[str, Any]:
            """浏览核心数据库人物印象及最近一次记忆审查时间。"""
            owner = self._owner()
            people: dict[str, PersonInfo] = {}
            search_fields = ("person_id", "nickname", "cardname", "impression") if q.strip() else (None,)
            for field in search_fields:
                query = database_api.query(PersonInfo).filter(
                    impression__isnull=False, impression__ne="",
                )
                if field is not None:
                    query = query.filter(**{f"{field}__like": f"%{q.strip()}%"})
                rows = await query.order_by("-updated_at", "person_id").limit(limit).all()
                people.update((row.person_id, row) for row in rows)
            personas = sorted(
                people.values(), key=lambda row: (-(row.updated_at or 0), row.person_id),
            )[:limit]
            async with owner.schema.database.session() as session:
                applied_at = await session.scalar(
                    select(SchemaVersionModel.applied_at).where(
                        SchemaVersionModel.schema_key == SCHEMA_KEY,
                        SchemaVersionModel.version == SCHEMA_VERSION,
                    )
                )
                reviews = (await session.scalars(
                    select(PersonaUpdateLogModel)
                    .where(
                        PersonaUpdateLogModel.person_id.in_(tuple(people)),
                        PersonaUpdateLogModel.created_at >= applied_at,
                    )
                    .order_by(PersonaUpdateLogModel.created_at.desc())
                )).all() if people and applied_at is not None else ()
            last_sessions: dict[str, str | None] = {}
            for review in reviews:
                last_sessions.setdefault(review.person_id, review.sleep_session_id)
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
                        "last_sleep_session_id": last_sessions.get(item.person_id),
                    }
                    for item in personas
                ],
            }
