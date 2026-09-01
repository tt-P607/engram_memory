"""Engram Memory vNext 候选素材与睡眠会话领域服务。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .domain import (
    CandidateActionInput,
    CandidateInput,
    CandidateStateTransition,
    SleepSessionInput,
    SleepSessionResult,
)
from .enums import (
    CandidateActionType,
    CandidateStatus,
    SleepCandidateOutcome,
    SleepSessionStatus,
)
from .models import (
    CandidateActionModel,
    CandidateActionTargetModel,
    CandidateEncoderCursorModel,
    CandidateEvidenceModel,
    CandidateModel,
    CandidateParticipantModel,
    CandidateSubjectModel,
    EvidenceMessageLinkModel,
    EvidenceModel,
    MemoryModel,
    MemoryRevisionModel,
    SleepSessionCandidateModel,
    SleepActionOperationModel,
    SleepActionPlanModel,
    SleepSessionModel,
)
from .schema import VNextSchema

TERMINAL_ACTIONS = frozenset({CandidateActionType.IGNORE, CandidateActionType.DEFER})
OPERATION_PREPARED = "PREPARED"
OPERATION_EXECUTING = "EXECUTING"
OPERATION_COMPLETED = "COMPLETED"
OPERATION_FAILED = "FAILED"


def _new_id() -> str:
    """生成统一 UUID 字符串。"""
    return str(uuid4())


def _operation_action_id(operation_key: str) -> str:
    """将稳定动作 key 映射为 CandidateAction 主键。"""
    return str(uuid5(NAMESPACE_URL, f"engram-vnext-sleep-action:{operation_key}"))


class CandidateService:
    """管理经历候选、编码游标及其来源证据。"""

    def __init__(self, schema: VNextSchema) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema

    async def create_candidates(
        self,
        candidates: tuple[CandidateInput, ...],
        *,
        cursor: tuple[str, datetime, str] | None = None,
    ) -> tuple[str, ...]:
        """批量创建候选，并可在同一事务内推进编码游标。"""
        for candidate in candidates:
            candidate.validate()
        if not candidates and cursor is None:
            return ()
        now = datetime.now(UTC)
        candidate_ids = tuple(_new_id() for _ in candidates)
        async with self._schema.database.session() as session:
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            for candidate_id, candidate in zip(candidate_ids, candidates, strict=True):
                session.add(
                    CandidateModel(
                        candidate_id=candidate_id,
                        status=CandidateStatus.PENDING,
                        rough_title=candidate.rough_title,
                        rough_content=candidate.rough_content,
                        proposed_kind=candidate.proposed_kind,
                        confidence_hint=candidate.confidence_hint,
                        salience_hint=candidate.salience_hint,
                        retention_reason=candidate.retention_reason,
                        uncertainty_note=candidate.uncertainty_note,
                        observed_at=candidate.observed_at,
                        created_at=now,
                        processing_session_id=None,
                        last_error=None,
                    )
                )
                if candidate.subject is not None:
                    session.add(
                        CandidateSubjectModel(
                            candidate_id=candidate_id,
                            subject_kind=candidate.subject.subject_kind,
                            person_id=candidate.subject.person_id,
                            subject_key=candidate.subject.subject_key,
                            subject_label=candidate.subject.subject_label,
                        )
                    )
                for participant in candidate.participants:
                    session.add(
                        CandidateParticipantModel(
                            participant_id=_new_id(),
                            candidate_id=candidate_id,
                            participant_kind=participant.participant_kind,
                            person_id=participant.person_id,
                            label=participant.label,
                            role=participant.role,
                        )
                    )
                for evidence in candidate.evidence:
                    evidence_id = _new_id()
                    session.add(
                        EvidenceModel(
                            evidence_id=evidence_id,
                            source_type=evidence.source_type,
                            claim_basis=evidence.claim_basis,
                            provenance_quality=evidence.provenance_quality,
                            observed_at=evidence.observed_at,
                            source_ref=evidence.source_ref,
                            note=evidence.note,
                            created_at=now,
                        )
                    )
                    session.add(
                        CandidateEvidenceModel(
                            candidate_id=candidate_id,
                            evidence_id=evidence_id,
                        )
                    )
                    for ordinal, message in enumerate(evidence.messages):
                        session.add(
                            EvidenceMessageLinkModel(
                                evidence_id=evidence_id,
                                message_id=message.message_id,
                                stream_id=message.stream_id,
                                ordinal=ordinal,
                            )
                        )
            if cursor is not None:
                await self._update_cursor_in_session(session, *cursor)
        return candidate_ids

    async def update_cursor(
        self,
        stream_id: str,
        message_time: datetime,
        message_id: str,
    ) -> None:
        """原子更新指定聊天流的经历编码复合游标。"""
        if not stream_id or not message_id:
            raise ValueError("Encoder Cursor 必须指定 stream_id 与 message_id")
        async with self._schema.database.session() as session:
            await self._update_cursor_in_session(
                session,
                stream_id,
                message_time,
                message_id,
            )

    async def _update_cursor_in_session(
        self,
        session: AsyncSession,
        stream_id: str,
        message_time: datetime,
        message_id: str,
    ) -> None:
        """在调用方事务中推进指定聊天流的复合游标。"""
        if not stream_id or not message_id:
            raise ValueError("Encoder Cursor 必须指定 stream_id 与 message_id")
        now = datetime.now(UTC)
        await session.execute(
            sqlite_insert(CandidateEncoderCursorModel)
            .values(
                stream_id=stream_id,
                last_processed_message_time=message_time,
                last_processed_message_id=message_id,
                updated_at=now,
            )
            .prefix_with("OR IGNORE")
        )
        await session.flush()
        result = await session.execute(
            update(CandidateEncoderCursorModel)
            .where(
                CandidateEncoderCursorModel.stream_id == stream_id,
                (
                    (CandidateEncoderCursorModel.last_processed_message_time < message_time)
                    | (
                        (CandidateEncoderCursorModel.last_processed_message_time == message_time)
                        & (
                            CandidateEncoderCursorModel.last_processed_message_id
                            <= message_id
                        )
                    )
                ),
            )
            .values(
                last_processed_message_time=message_time,
                last_processed_message_id=message_id,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 1:
            return
        cursor = await session.get(CandidateEncoderCursorModel, stream_id)
        if cursor is not None and (
            message_time,
            message_id,
        ) < (
            cursor.last_processed_message_time,
            cursor.last_processed_message_id,
        ):
            raise ValueError("Encoder Cursor 不允许回退")


class SleepSessionService:
    """管理睡眠会话、候选认领、动作审计和异常恢复。"""

    def __init__(self, schema: VNextSchema) -> None:
        """绑定 vNext Schema。"""
        self._schema = schema

    async def start_session(
        self,
        data: SleepSessionInput,
        candidate_ids: tuple[str, ...],
    ) -> SleepSessionResult:
        """创建 RUNNING 会话并原子认领 PENDING 或 DEFERRED Candidate。"""
        data.validate()
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Sleep Session candidate_ids 不能重复")
        now = datetime.now(UTC)
        sleep_session_id = _new_id()
        async with self._schema.database.session() as session:
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            sleep_session = SleepSessionModel(
                sleep_session_id=sleep_session_id,
                trigger_type=data.trigger_type,
                status=SleepSessionStatus.RUNNING,
                started_at=now,
                finished_at=None,
                candidate_count=len(candidate_ids),
                model_id=data.model_id,
                prompt_version=data.prompt_version,
                error_summary=None,
            )
            session.add(sleep_session)
            await session.flush([sleep_session])
            if candidate_ids:
                claim_result = await session.execute(
                    update(CandidateModel)
                    .where(
                        CandidateModel.candidate_id.in_(candidate_ids),
                        CandidateModel.status.in_(
                            (CandidateStatus.PENDING, CandidateStatus.DEFERRED)
                        ),
                    )
                    .values(
                        status=CandidateStatus.PROCESSING,
                        processing_session_id=sleep_session_id,
                        last_error=None,
                    )
                    .execution_options(synchronize_session=False)
                )
                if claim_result.rowcount != len(candidate_ids):
                    existing_ids = set(
                        (
                            await session.scalars(
                                select(CandidateModel.candidate_id).where(
                                    CandidateModel.candidate_id.in_(candidate_ids)
                                )
                            )
                        ).all()
                    )
                    if existing_ids != set(candidate_ids):
                        raise ValueError("存在无效 Candidate")
                    raise ValueError("Candidate 当前状态不可认领或发生认领冲突")
            for candidate_id in candidate_ids:
                session.add(
                    SleepSessionCandidateModel(
                        sleep_session_id=sleep_session_id,
                        candidate_id=candidate_id,
                        claimed_at=now,
                        released_at=None,
                        outcome=None,
                    )
                )
        return SleepSessionResult(
            sleep_session_id=sleep_session_id,
            status=SleepSessionStatus.RUNNING,
            candidate_count=len(candidate_ids),
        )

    async def abort_session(
        self,
        sleep_session_id: str,
        error_summary: str,
    ) -> SleepSessionResult:
        """失败收口 RUNNING 会话并释放仍由其持有的全部候选。"""
        if not error_summary.strip():
            raise ValueError("abort_session error_summary 不能为空")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            sleep_session = await session.get(SleepSessionModel, sleep_session_id)
            if sleep_session is None:
                raise ValueError("Sleep Session 不存在")
            if sleep_session.status is not SleepSessionStatus.RUNNING:
                return SleepSessionResult(
                    sleep_session_id,
                    sleep_session.status,
                    sleep_session.candidate_count,
                )
            candidates = tuple(
                (
                    await session.scalars(
                        select(CandidateModel).where(
                            CandidateModel.status == CandidateStatus.PROCESSING,
                            CandidateModel.processing_session_id == sleep_session_id,
                        )
                    )
                ).all()
            )
            for candidate in candidates:
                claim = await session.get(
                    SleepSessionCandidateModel,
                    (sleep_session_id, candidate.candidate_id),
                )
                result = await session.execute(
                    update(CandidateModel)
                    .where(
                        CandidateModel.candidate_id == candidate.candidate_id,
                        CandidateModel.status == CandidateStatus.PROCESSING,
                        CandidateModel.processing_session_id == sleep_session_id,
                    )
                    .values(
                        status=CandidateStatus.FAILED,
                        processing_session_id=None,
                        last_error=error_summary,
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount == 1 and claim is not None and claim.released_at is None:
                    claim.released_at = now
                    claim.outcome = SleepCandidateOutcome.FAILED
            result = await session.execute(
                update(SleepSessionModel)
                .where(
                    SleepSessionModel.sleep_session_id == sleep_session_id,
                    SleepSessionModel.status == SleepSessionStatus.RUNNING,
                )
                .values(
                    status=SleepSessionStatus.FAILED,
                    finished_at=now,
                    error_summary=error_summary,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise ValueError("Sleep Session 在 abort 前已被并发结束")
            sleep_session.status = SleepSessionStatus.FAILED
            sleep_session.finished_at = now
            sleep_session.error_summary = error_summary
            candidate_count = sleep_session.candidate_count
        return SleepSessionResult(
            sleep_session_id,
            SleepSessionStatus.FAILED,
            candidate_count,
        )

    async def prepare_operation(
        self,
        candidate_id: str,
        sleep_session_id: str,
        action_type: CandidateActionType,
        operation_key: str,
        intent: dict[str, object],
    ) -> None:
        """原子保存不可变动作意图，重复 key 必须指向同一动作。"""
        if not operation_key.strip():
            raise ValueError("operation_key 不能为空")
        try:
            normalized_intent = json.loads(json.dumps(intent, sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ValueError("动作意图必须是 JSON 对象") from error
        if not isinstance(normalized_intent, dict):
            raise ValueError("动作意图必须是 JSON 对象")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            existing = await session.get(SleepActionOperationModel, operation_key)
            if existing is not None:
                if (
                    existing.candidate_id != candidate_id
                    or existing.sleep_session_id != sleep_session_id
                    or existing.action_type is not action_type
                    or existing.intent_json != normalized_intent
                ):
                    raise ValueError("operation_key 已绑定不同的动作意图")
                return
            await self._require_claimed_candidate(session, candidate_id, sleep_session_id)
            await session.execute(
                sqlite_insert(SleepActionOperationModel)
                .values(
                    operation_key=operation_key,
                    candidate_id=candidate_id,
                    sleep_session_id=sleep_session_id,
                    action_type=action_type,
                    intent_json=normalized_intent,
                    status=OPERATION_PREPARED,
                    result_json=None,
                    error=None,
                    created_at=now,
                    updated_at=now,
                )
                .prefix_with("OR IGNORE")
            )
            operation = await session.get(SleepActionOperationModel, operation_key)
            if operation is None:  # pragma: no cover - INSERT/SELECT same transaction
                raise ValueError("无法保存 Sleep Action Operation")
            if (
                operation.candidate_id != candidate_id
                or operation.sleep_session_id != sleep_session_id
                or operation.action_type is not action_type
                or operation.intent_json != normalized_intent
            ):
                raise ValueError("operation_key 已绑定不同的动作意图")

    async def prepare_action_plan(
        self,
        candidate_id: str,
        sleep_session_id: str,
        plan_key: str,
        actions: tuple[tuple[str, CandidateActionType, dict[str, object]], ...],
    ) -> None:
        """在领域动作执行前一次性持久化完整动作计划。"""
        if not plan_key.strip() or not actions:
            raise ValueError("动作计划必须包含稳定 plan_key 与至少一个动作")
        operation_keys = tuple(item[0] for item in actions)
        if len(set(operation_keys)) != len(operation_keys):
            raise ValueError("动作计划 operation_key 不能重复")
        try:
            intents = [json.loads(json.dumps(item[2], sort_keys=True)) for item in actions]
        except (TypeError, ValueError) as error:
            raise ValueError("动作计划意图必须是 JSON 对象") from error
        if not all(isinstance(item, dict) for item in intents):
            raise ValueError("动作计划意图必须是 JSON 对象")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            existing = await session.get(SleepActionPlanModel, plan_key)
            if existing is not None:
                if (
                    existing.candidate_id != candidate_id
                    or existing.sleep_session_id != sleep_session_id
                    or existing.intents_json != intents
                    or existing.operation_keys_json != list(operation_keys)
                ):
                    raise ValueError("plan_key 已绑定不同的动作计划")
                return
            await self._require_claimed_candidate(session, candidate_id, sleep_session_id)
            session.add(
                SleepActionPlanModel(
                    plan_key=plan_key,
                    candidate_id=candidate_id,
                    sleep_session_id=sleep_session_id,
                    intents_json=intents,
                    operation_keys_json=list(operation_keys),
                    action_count=len(actions),
                    next_action_index=0,
                    status=OPERATION_PREPARED,
                    target_memory_ids_json=[],
                    created_at=now,
                    updated_at=now,
                )
            )
            for operation_key, action_type, intent in actions:
                await session.execute(
                    sqlite_insert(SleepActionOperationModel)
                    .values(
                        operation_key=operation_key,
                        candidate_id=candidate_id,
                        sleep_session_id=sleep_session_id,
                        action_type=action_type,
                        intent_json=intent,
                        status=OPERATION_PREPARED,
                        result_json=None,
                        error=None,
                        created_at=now,
                        updated_at=now,
                    )
                    .prefix_with("OR IGNORE")
                )
            await session.flush()
            for operation_key, action_type, intent in actions:
                operation = await session.get(SleepActionOperationModel, operation_key)
                if (
                    operation is None
                    or operation.candidate_id != candidate_id
                    or operation.sleep_session_id != sleep_session_id
                    or operation.action_type is not action_type
                    or operation.intent_json != intent
                ):
                    raise ValueError("动作计划 operation_key 已绑定不同动作")

    async def get_action_plan(self, plan_key: str) -> SleepActionPlanModel | None:
        """读取动作计划及其恢复游标。"""
        async with self._schema.database.session() as session:
            return await session.get(SleepActionPlanModel, plan_key)

    async def list_incomplete_plans(self) -> tuple[SleepActionPlanModel, ...]:
        """列出尚未完成的动作计划。"""
        async with self._schema.database.session() as session:
            return tuple(
                (
                    await session.scalars(
                        select(SleepActionPlanModel)
                        .where(
                            SleepActionPlanModel.status.in_(
                                (OPERATION_PREPARED, OPERATION_EXECUTING)
                            )
                        )
                        .order_by(SleepActionPlanModel.created_at)
                    )
                ).all()
            )

    async def advance_action_plan(
        self,
        plan_key: str,
        expected_index: int,
        target_memory_ids: tuple[str, ...],
    ) -> None:
        """以条件 CAS 推进动作计划游标。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            plan = await session.get(SleepActionPlanModel, plan_key)
            if plan is None:
                raise ValueError("Sleep Action Plan 不存在")
            if plan.next_action_index > expected_index:
                return
            if plan.next_action_index != expected_index:
                raise ValueError("Sleep Action Plan 游标发生并发冲突")
            result = await session.execute(
                update(SleepActionPlanModel)
                .where(
                    SleepActionPlanModel.plan_key == plan_key,
                    SleepActionPlanModel.next_action_index == expected_index,
                    SleepActionPlanModel.status.in_((OPERATION_PREPARED, OPERATION_EXECUTING)),
                )
                .values(
                    next_action_index=expected_index + 1,
                    status=(
                        OPERATION_COMPLETED
                        if expected_index + 1 == plan.action_count
                        else OPERATION_EXECUTING
                    ),
                    target_memory_ids_json=list(
                        dict.fromkeys(plan.target_memory_ids_json + list(target_memory_ids))
                    ),
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise ValueError("Sleep Action Plan 已被并发推进")

    async def finalize_action_plan(self, plan_key: str) -> tuple[str, ...]:
        """仅在完整计划完成且原会话仍持有 Candidate 时收口。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            plan = await session.get(SleepActionPlanModel, plan_key)
            if plan is None or plan.status != OPERATION_COMPLETED:
                raise ValueError("Sleep Action Plan 尚未完成")
            if plan.next_action_index != plan.action_count:
                raise ValueError("Sleep Action Plan 游标未到末尾")
            candidate_result = await session.execute(
                update(CandidateModel)
                .where(
                    CandidateModel.candidate_id == plan.candidate_id,
                    CandidateModel.status == CandidateStatus.PROCESSING,
                    CandidateModel.processing_session_id == plan.sleep_session_id,
                )
                .values(status=CandidateStatus.RESOLVED, processing_session_id=None)
                .execution_options(synchronize_session=False)
            )
            if candidate_result.rowcount != 1:
                candidate = await session.get(CandidateModel, plan.candidate_id)
                if candidate is None:
                    raise ValueError("Candidate 不存在")
                if candidate.status not in {
                    CandidateStatus.RESOLVED,
                    CandidateStatus.DEFERRED,
                }:
                    raise ValueError("Candidate 已被其他 Session 认领")
            claim_result = await session.execute(
                update(SleepSessionCandidateModel)
                .where(
                    SleepSessionCandidateModel.sleep_session_id == plan.sleep_session_id,
                    SleepSessionCandidateModel.candidate_id == plan.candidate_id,
                    SleepSessionCandidateModel.released_at.is_(None),
                )
                .values(released_at=now, outcome=SleepCandidateOutcome.RESOLVED)
                .execution_options(synchronize_session=False)
            )
            if claim_result.rowcount not in {0, 1}:
                raise ValueError("Candidate Claim 收口发生并发冲突")
            return tuple(str(item) for item in plan.target_memory_ids_json)

    async def get_operation(self, operation_key: str) -> SleepActionOperationModel | None:
        """读取动作 journal，供崩溃恢复使用。"""
        async with self._schema.database.session() as session:
            return await session.get(SleepActionOperationModel, operation_key)

    async def list_incomplete_operations(self) -> tuple[SleepActionOperationModel, ...]:
        """列出尚未失败的动作计划，供新 Sleep Session 续跑。"""
        async with self._schema.database.session() as session:
            return tuple(
                (
                    await session.scalars(
                        select(SleepActionOperationModel)
                        .where(
                            SleepActionOperationModel.status.in_(
                                (OPERATION_PREPARED, OPERATION_EXECUTING, OPERATION_COMPLETED)
                            )
                        )
                        .order_by(SleepActionOperationModel.created_at)
                    )
                ).all()
            )

    async def complete_operation(
        self,
        operation_key: str,
        result: dict[str, object],
    ) -> dict[str, object]:
        """持久化领域动作结果；已完成动作只返回首次结果。"""
        try:
            normalized_result = json.loads(json.dumps(result, sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ValueError("动作结果必须是 JSON 对象") from error
        if not isinstance(normalized_result, dict):
            raise ValueError("动作结果必须是 JSON 对象")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            operation = await session.get(SleepActionOperationModel, operation_key)
            if operation is None:
                raise ValueError("Sleep Action Operation 不存在")
            if operation.status == OPERATION_COMPLETED:
                return dict(operation.result_json or {})
            if operation.status not in {OPERATION_PREPARED, OPERATION_EXECUTING}:
                raise ValueError("失败的 Sleep Action Operation 不能完成")
            result = await session.execute(
                update(SleepActionOperationModel)
                .where(
                    SleepActionOperationModel.operation_key == operation_key,
                    SleepActionOperationModel.status.in_((OPERATION_PREPARED, OPERATION_EXECUTING)),
                )
                .values(
                    status=OPERATION_COMPLETED,
                    result_json=normalized_result,
                    updated_at=now,
                    error=None,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                current = await session.get(SleepActionOperationModel, operation_key)
                if current is not None and current.status == OPERATION_COMPLETED:
                    return dict(current.result_json or {})
                raise ValueError("Sleep Action Operation 已被并发修改")
        return normalized_result

    async def mark_operation_executing(self, operation_key: str) -> None:
        """将动作 journal 标记为执行中，允许重试保持同一 key。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            result = await session.execute(
                update(SleepActionOperationModel)
                .where(
                    SleepActionOperationModel.operation_key == operation_key,
                    SleepActionOperationModel.status == OPERATION_PREPARED,
                )
                .values(status=OPERATION_EXECUTING, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount == 0:
                operation = await session.get(SleepActionOperationModel, operation_key)
                if operation is None:
                    raise ValueError("Sleep Action Operation 不存在")
                if operation.status not in {OPERATION_EXECUTING, OPERATION_COMPLETED}:
                    raise ValueError("Sleep Action Operation 不可执行")

    async def record_operation_action(self, operation_key: str) -> str:
        """从已完成 journal 补写 CandidateAction，重复恢复保持幂等。"""
        async with self._schema.database.session() as session:
            operation = await session.get(SleepActionOperationModel, operation_key)
            if operation is None:
                raise ValueError("Sleep Action Operation 不存在")
            if operation.status != OPERATION_COMPLETED or operation.result_json is None:
                raise ValueError("Sleep Action Operation 尚未完成")
            intent = operation.intent_json
            result = operation.result_json
            raw_targets = result.get("targets", ())
            targets = tuple(
                (str(item[0]), item[1])
                for item in raw_targets
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            action = CandidateActionInput(
                candidate_id=operation.candidate_id,
                sleep_session_id=operation.sleep_session_id,
                action_type=operation.action_type,
                note=intent.get("note") if isinstance(intent.get("note"), str) else None,
                result_revision_id=(
                    str(result["revision_id"])
                    if result.get("revision_id") is not None
                    else None
                ),
                targets=targets,
                operation_key=operation_key,
            )
            return await self._record_action_in_session(session, action, allow_recovery=True)

    async def finalize_recovered_operation(self, operation_key: str) -> None:
        """在动作审计补齐后收口其候选，避免新会话再次询问模型。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            operation = await session.get(SleepActionOperationModel, operation_key)
            if operation is None or operation.status != OPERATION_COMPLETED:
                raise ValueError("Sleep Action Operation 尚未完成")
            candidate = await session.get(CandidateModel, operation.candidate_id)
            if candidate is None:
                raise ValueError("Candidate 不存在")
            target_status = (
                CandidateStatus.DEFERRED
                if operation.action_type is CandidateActionType.DEFER
                else CandidateStatus.RESOLVED
            )
            candidate_result = await session.execute(
                update(CandidateModel)
                .where(
                    CandidateModel.candidate_id == operation.candidate_id,
                    CandidateModel.status == CandidateStatus.PROCESSING,
                    CandidateModel.processing_session_id == operation.sleep_session_id,
                )
                .values(status=target_status, processing_session_id=None)
                .execution_options(synchronize_session=False)
            )
            if candidate_result.rowcount == 0 and candidate.status not in {
                CandidateStatus.RESOLVED,
                CandidateStatus.DEFERRED,
            }:
                raise ValueError("Candidate 已被其他 Session 认领")
            await session.execute(
                update(SleepSessionCandidateModel)
                .where(
                    SleepSessionCandidateModel.sleep_session_id == operation.sleep_session_id,
                    SleepSessionCandidateModel.candidate_id == operation.candidate_id,
                    SleepSessionCandidateModel.released_at.is_(None),
                )
                .values(
                    released_at=now,
                    outcome=(
                        SleepCandidateOutcome.DEFERRED
                        if target_status is CandidateStatus.DEFERRED
                        else SleepCandidateOutcome.RESOLVED
                    ),
                )
                .execution_options(synchronize_session=False)
            )

    async def _record_action_in_session(
        self,
        session: AsyncSession,
        data: CandidateActionInput,
        *,
        allow_recovery: bool,
    ) -> str:
        """在调用方事务中追加或读取幂等 CandidateAction。"""
        operation = None
        if data.operation_key is not None:
            operation = await session.get(SleepActionOperationModel, data.operation_key)
            if operation is None:
                raise ValueError("Sleep Action Operation 不存在")
            if (
                operation.candidate_id != data.candidate_id
                or operation.sleep_session_id != data.sleep_session_id
                or operation.action_type is not data.action_type
            ):
                raise ValueError("operation_key 与 CandidateAction 不匹配")
            if operation.result_json is not None:
                result = operation.result_json
                data = CandidateActionInput(
                    candidate_id=data.candidate_id,
                    sleep_session_id=data.sleep_session_id,
                    action_type=data.action_type,
                    note=data.note,
                    result_revision_id=(
                        str(result["revision_id"])
                        if result.get("revision_id") is not None
                        else data.result_revision_id
                    ),
                    targets=tuple(
                        (str(item[0]), item[1])
                        for item in result.get("targets", data.targets)
                        if isinstance(item, (list, tuple)) and len(item) == 2
                    ),
                    operation_key=data.operation_key,
                )
            elif operation.status != OPERATION_COMPLETED:
                raise ValueError("Sleep Action Operation 尚未完成")
        if allow_recovery:
            candidate = await session.get(CandidateModel, data.candidate_id)
            claim = await session.get(
                SleepSessionCandidateModel,
                (data.sleep_session_id, data.candidate_id),
            )
            if candidate is None or claim is None:
                raise ValueError("Candidate 或 Sleep Session Claim 不存在")
            claim_is_current = (
                candidate.status is CandidateStatus.PROCESSING
                and candidate.processing_session_id == data.sleep_session_id
                and claim.released_at is None
            )
        else:
            candidate, claim = await self._require_claimed_candidate(
                session, data.candidate_id, data.sleep_session_id
            )
            claim_is_current = True
        action_id = (
            _operation_action_id(data.operation_key)
            if data.operation_key is not None
            else _new_id()
        )
        existing_action = await session.get(CandidateActionModel, action_id)
        if existing_action is not None:
            return action_id
        if data.result_revision_id is not None:
            revision = await session.get(MemoryRevisionModel, data.result_revision_id)
            if revision is None:
                raise ValueError("result_revision_id 不存在")
        target_ids = {memory_id for memory_id, _ in data.targets}
        if target_ids:
            existing_target_ids = set(
                (
                    await session.scalars(
                        select(MemoryModel.memory_id).where(MemoryModel.memory_id.in_(target_ids))
                    )
                ).all()
            )
            if existing_target_ids != target_ids:
                raise ValueError("Candidate Action target Memory 不存在")
        now = datetime.now(UTC)
        action_insert = await session.execute(
            sqlite_insert(CandidateActionModel)
            .values(
                action_id=action_id,
                candidate_id=data.candidate_id,
                sleep_session_id=data.sleep_session_id,
                action_type=data.action_type,
                result_revision_id=data.result_revision_id,
                note=data.note,
                created_at=now,
            )
            .prefix_with("OR IGNORE")
        )
        if action_insert.rowcount == 0:
            return action_id
        for memory_id, target_role in data.targets:
            await session.execute(
                sqlite_insert(CandidateActionTargetModel)
                .values(
                    action_id=action_id,
                    memory_id=memory_id,
                    target_role=target_role,
                )
                .prefix_with("OR IGNORE")
            )
        if data.action_type is CandidateActionType.IGNORE and claim_is_current:
            self._release_candidate(candidate, claim, CandidateStatus.RESOLVED,
                                    SleepCandidateOutcome.RESOLVED, now)
        elif data.action_type is CandidateActionType.DEFER and claim_is_current:
            self._release_candidate(candidate, claim, CandidateStatus.DEFERRED,
                                    SleepCandidateOutcome.DEFERRED, now)
        return action_id

    async def record_action(self, data: CandidateActionInput) -> str:
        """为当前会话认领的候选追加动作与正式记忆目标审计。"""
        data.validate()
        async with self._schema.database.session() as session:
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            return await self._record_action_in_session(session, data, allow_recovery=False)

    async def resolve_candidate(
        self,
        candidate_id: str,
        sleep_session_id: str,
    ) -> CandidateStateTransition:
        """在一个或多个非终态动作完成后显式结束候选处理。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            candidate, _claim = await self._require_claimed_candidate(
                session,
                candidate_id,
                sleep_session_id,
            )
            action_count = len(
                (
                    await session.scalars(
                        select(CandidateActionModel.action_id).where(
                            CandidateActionModel.candidate_id == candidate_id,
                            CandidateActionModel.sleep_session_id == sleep_session_id,
                        )
                    )
                ).all()
            )
            if action_count == 0:
                raise ValueError("Candidate 尚无可完成的 Action")
            previous = candidate.status
            result = await session.execute(
                update(CandidateModel)
                .where(
                    CandidateModel.candidate_id == candidate_id,
                    CandidateModel.status == CandidateStatus.PROCESSING,
                    CandidateModel.processing_session_id == sleep_session_id,
                )
                .values(status=CandidateStatus.RESOLVED, processing_session_id=None)
                .execution_options(synchronize_session=False)
            )
            claim_result = await session.execute(
                update(SleepSessionCandidateModel)
                .where(
                    SleepSessionCandidateModel.sleep_session_id == sleep_session_id,
                    SleepSessionCandidateModel.candidate_id == candidate_id,
                    SleepSessionCandidateModel.released_at.is_(None),
                )
                .values(released_at=now, outcome=SleepCandidateOutcome.RESOLVED)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1 or claim_result.rowcount != 1:
                raise ValueError("Candidate 在收口前已被并发修改")
        return CandidateStateTransition(previous, CandidateStatus.RESOLVED)

    async def fail_candidate(
        self,
        candidate_id: str,
        sleep_session_id: str,
        error: str,
    ) -> CandidateStateTransition:
        """记录候选处理失败并释放当前认领。"""
        if not error.strip():
            raise ValueError("Candidate failure error 不能为空")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            candidate, claim = await self._require_claimed_candidate(
                session,
                candidate_id,
                sleep_session_id,
            )
            previous = candidate.status
            result = await session.execute(
                update(CandidateModel)
                .where(
                    CandidateModel.candidate_id == candidate_id,
                    CandidateModel.status == CandidateStatus.PROCESSING,
                    CandidateModel.processing_session_id == sleep_session_id,
                )
                .values(
                    status=CandidateStatus.FAILED,
                    processing_session_id=None,
                    last_error=error,
                )
                .execution_options(synchronize_session=False)
            )
            claim_result = await session.execute(
                update(SleepSessionCandidateModel)
                .where(
                    SleepSessionCandidateModel.sleep_session_id == sleep_session_id,
                    SleepSessionCandidateModel.candidate_id == candidate_id,
                    SleepSessionCandidateModel.released_at.is_(None),
                )
                .values(released_at=now, outcome=SleepCandidateOutcome.FAILED)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1 or claim_result.rowcount != 1:
                raise ValueError("Candidate 在失败收口前已被并发修改")
        return CandidateStateTransition(previous, CandidateStatus.FAILED)

    async def finish_session(
        self,
        sleep_session_id: str,
        status: SleepSessionStatus,
        error_summary: str | None = None,
    ) -> SleepSessionResult:
        """结束 RUNNING 睡眠会话并保存最终状态。"""
        if status is SleepSessionStatus.RUNNING:
            raise ValueError("finish_session 不能使用 RUNNING 状态")
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            sleep_session = await session.get(SleepSessionModel, sleep_session_id)
            if sleep_session is None:
                raise ValueError("Sleep Session 不存在")
            if sleep_session.status is not SleepSessionStatus.RUNNING:
                raise ValueError("Sleep Session 已结束")
            result = await session.execute(
                update(SleepSessionModel)
                .where(
                    SleepSessionModel.sleep_session_id == sleep_session_id,
                    SleepSessionModel.status == SleepSessionStatus.RUNNING,
                )
                .values(status=status, finished_at=now, error_summary=error_summary)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise ValueError("Sleep Session 已被并发结束")
            sleep_session.status = status
            sleep_session.finished_at = now
            sleep_session.error_summary = error_summary
            candidate_count = sleep_session.candidate_count
        return SleepSessionResult(sleep_session_id, status, candidate_count)

    async def recover_abandoned_sessions(self) -> tuple[str, ...]:
        """终结上一次 Runtime 遗留的运行会话并恢复可安全重试的候选。"""
        now = datetime.now(UTC)
        async with self._schema.database.session() as session:
            running_sessions = tuple(
                (
                    await session.scalars(
                        select(SleepSessionModel).where(
                            SleepSessionModel.status == SleepSessionStatus.RUNNING
                        )
                    )
                ).all()
            )
            for running_session in running_sessions:
                result = await session.execute(
                    update(SleepSessionModel)
                    .where(
                        SleepSessionModel.sleep_session_id == running_session.sleep_session_id,
                        SleepSessionModel.status == SleepSessionStatus.RUNNING,
                    )
                    .values(
                        status=SleepSessionStatus.FAILED,
                        finished_at=now,
                        error_summary="Runtime 中断后恢复遗留会话",
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount == 1:
                    running_session.status = SleepSessionStatus.FAILED
                    running_session.finished_at = now
                    running_session.error_summary = "Runtime 中断后恢复遗留会话"
        return await self.recover_interrupted_candidates()

    async def recover_interrupted_candidates(self) -> tuple[str, ...]:
        """释放非 RUNNING Session 中无终态动作的 PROCESSING Candidate。"""
        now = datetime.now(UTC)
        recovered: list[str] = []
        async with self._schema.database.session() as session:
            candidates = list(
                (
                    await session.scalars(
                        select(CandidateModel).where(
                            CandidateModel.status == CandidateStatus.PROCESSING,
                            CandidateModel.processing_session_id.is_not(None),
                        )
                    )
                ).all()
            )
            for candidate in candidates:
                session_id = candidate.processing_session_id
                sleep_session = await session.get(SleepSessionModel, session_id)
                if sleep_session is not None and sleep_session.status is SleepSessionStatus.RUNNING:
                    continue
                terminal_action = (
                    await session.scalars(
                        select(CandidateActionModel.action_id).where(
                            CandidateActionModel.candidate_id == candidate.candidate_id,
                            CandidateActionModel.sleep_session_id == session_id,
                            CandidateActionModel.action_type.in_(TERMINAL_ACTIONS),
                        )
                    )
                ).first()
                if terminal_action is not None:
                    continue
                claim = await session.get(
                    SleepSessionCandidateModel,
                    (session_id, candidate.candidate_id),
                )
                result = await session.execute(
                    update(CandidateModel)
                    .where(
                        CandidateModel.candidate_id == candidate.candidate_id,
                        CandidateModel.status == CandidateStatus.PROCESSING,
                        CandidateModel.processing_session_id == session_id,
                    )
                    .values(
                        status=CandidateStatus.PENDING,
                        processing_session_id=None,
                        last_error=None,
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    continue
                candidate.status = CandidateStatus.PENDING
                candidate.processing_session_id = None
                candidate.last_error = None
                if claim is not None and claim.released_at is None:
                    claim.released_at = now
                    claim.outcome = SleepCandidateOutcome.FAILED
                recovered.append(candidate.candidate_id)
        return tuple(recovered)

    async def _require_claimed_candidate(
        self,
        session: object,
        candidate_id: str,
        sleep_session_id: str,
    ) -> tuple[CandidateModel, SleepSessionCandidateModel]:
        """确认候选由指定 RUNNING Session 持有。"""
        candidate = await session.get(CandidateModel, candidate_id)  # type: ignore[attr-defined]
        sleep_session = await session.get(SleepSessionModel, sleep_session_id)  # type: ignore[attr-defined]
        if candidate is None or sleep_session is None:
            raise ValueError("Candidate 或 Sleep Session 不存在")
        if sleep_session.status is not SleepSessionStatus.RUNNING:
            raise ValueError("Sleep Session 不是 RUNNING")
        if (
            candidate.status is not CandidateStatus.PROCESSING
            or candidate.processing_session_id != sleep_session_id
        ):
            raise ValueError("Candidate 未被当前 Sleep Session 认领")
        claim = await session.get(  # type: ignore[attr-defined]
            SleepSessionCandidateModel,
            (sleep_session_id, candidate_id),
        )
        if claim is None or claim.released_at is not None:
            raise ValueError("Candidate Claim 不存在或已释放")
        return candidate, claim

    @staticmethod
    def _release_candidate(
        candidate: CandidateModel,
        claim: SleepSessionCandidateModel,
        status: CandidateStatus,
        outcome: SleepCandidateOutcome,
        released_at: datetime,
    ) -> None:
        """更新候选状态并关闭本次认领记录。"""
        candidate.status = status
        candidate.processing_session_id = None
        claim.released_at = released_at
        claim.outcome = outcome
