"""Engram Memory vNext 睡眠整理编排器。

按 Technical Spec §55-§64 与 Core Design §26-§30 实现睡眠会话编排：
混合触发判定（每日/压力/安静期）、候选认领、多步 Agent 循环、
执行后 self-check 重读、恢复与审计。Sleep Agent 使用 VNextToolService
的 SLEEP_AGENT 权限执行整理工具；每个候选的动作通过
CandidateService 审计留痕。
"""

from __future__ import annotations

import inspect
import json
from hashlib import sha256
from dataclasses import dataclass, replace
from datetime import UTC, datetime, time
from typing import Mapping
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select

from .candidate_service import CandidateService, SleepSessionService
from .domain import (
    AssessmentInput,
    CandidateActionInput,
    CreateMemoryInput,
    MergeMemoryInput,
    ParticipantInput,
    RelateMemoryInput,
    ReinforceMemoryInput,
    ReviseMemoryInput,
    SleepSessionInput,
    SubjectInput,
)
from .enums import (
    ActorType,
    CandidateActionType,
    CandidateActionTargetRole,
    CandidateStatus,
    ConfidenceLevel,
    EventTimeOrigin,
    EventTimePrecision,
    MemoryKind,
    RelationType,
    RevisionChangeReason,
    SalienceLevel,
    SleepSessionStatus,
    SleepTriggerType,
    StabilityLevel,
    SubjectKind,
    ParticipantKind,
    ParticipantRole,
)
from .models import (
    CandidateEvidenceModel,
    CandidateModel,
    CandidateParticipantModel,
    CandidateSubjectModel,
)
from .schema import VNextSchema
from .tool_service import ToolContext, VNextToolService


@dataclass(frozen=True, slots=True)
class SleepTriggerState:
    """一次触发判定的输入状态。"""

    pending_count: int
    now: datetime
    daily_time: time
    pressure_threshold: int
    quiet_period_minutes: int
    last_finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """触发判定结果。"""

    should_trigger: bool
    trigger_type: SleepTriggerType | None
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """一次会话认领的候选批次。"""

    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidateContext:
    """执行 Sleep 动作所需的候选 Canonical 输入快照。"""

    candidate_id: str
    rough_title: str
    rough_content: str
    retention_reason: str
    observed_at: datetime
    evidence_ids: tuple[str, ...]
    subject: SubjectInput
    participants: tuple[ParticipantInput, ...]
    proposed_kind: MemoryKind | None
    confidence_hint: ConfidenceLevel | None
    salience_hint: SalienceLevel | None


class AgentStepProducer:
    """单候选整理步骤 LLM 调用协议。

    输入候选素材与工具视图，输出该候选的动作意图列表；
    空列表合法（对应模型未形成安全动作，候选按 DEFER 释放）。
    """

    def __call__(self, candidate_payload: str) -> tuple[dict[str, object], ...]:
        """按候选素材文本产出动作意图。"""
        raise NotImplementedError


class AgentProtocolError(ValueError):
    """Sleep Agent Producer 返回违反结构化契约时的错误。"""


class SleepAgentOrchestrator:
    """睡眠整理会话编排器。"""

    def __init__(
        self,
        schema: VNextSchema,
        tools: VNextToolService,
        *,
        sleep_service: SleepSessionService | None = None,
        batch_size: int,
        model_id: str,
        prompt_version: str,
    ) -> None:
        """绑定 Schema、工具门面与会话参数。

        参数:
            schema: vNext Schema。
            tools: SLEEP_AGENT 上下文的工具门面。
            batch_size: 每次会话认领的候选上限。
            model_id: 本次会话使用的模型标识（§55）。
            prompt_version: Sleep Agent Prompt 显式版本号（§148）。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if not model_id.strip() or not prompt_version.strip():
            raise ValueError("model_id 与 prompt_version 不能为空")
        self._schema = schema
        self._tools = tools
        self._sessions = sleep_service or SleepSessionService(schema)
        self._candidates = CandidateService(schema)
        self._batch_size = batch_size
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._context = ToolContext(actor_type=ActorType.SLEEP_AGENT)
        self.last_session_id: str | None = None
        self.last_changed_memory_ids: tuple[str, ...] = ()

    @property
    def sleep_service(self) -> SleepSessionService:
        """返回 Owner 注入的 Sleep Session 服务。"""
        return self._sessions

    @staticmethod
    def decide_trigger(state: SleepTriggerState) -> TriggerDecision:
        """按每日固定时间 + 压力阈值 + 安静期判定触发。

        规则（§60）：
        - 每日 Daily：当前时间越过当日 daily_time；
        - 压力 Pressure：pending 数达到阈值即触发（安静期由调用方
          在触发前等待对话停顿，此处只输出决策）；
        - 都不满足则不触发。
        """
        if state.pressure_threshold > 0 and state.pending_count >= state.pressure_threshold:
            return TriggerDecision(True, SleepTriggerType.PRESSURE, "候选压力达到阈值")
        local = state.now.astimezone()
        today_target = local.replace(
            hour=state.daily_time.hour,
            minute=state.daily_time.minute,
            second=state.daily_time.second,
            microsecond=0,
        )
        already_ran_today = state.last_finished_at is not None and (
            state.last_finished_at.astimezone().date() == local.date()
            and state.last_finished_at.astimezone() >= today_target
        )
        if local >= today_target and not already_ran_today:
            return TriggerDecision(True, SleepTriggerType.DAILY, "到达每日整理时间")
        return TriggerDecision(False, None, "未满足触发条件")

    async def pending_candidates(self, limit: int | None = None) -> tuple[str, ...]:
        """读取待处理候选（PENDING 优先，DEFERRED 兜底）。"""
        size = limit if limit is not None else self._batch_size
        async with self._schema.database.session() as session:
            pending_rows = tuple(
                (
                    await session.scalars(
                        select(CandidateModel.candidate_id)
                        .where(CandidateModel.status == CandidateStatus.PENDING)
                        .order_by(CandidateModel.created_at, CandidateModel.candidate_id)
                        .limit(size)
                    )
                ).all()
            )
            if pending_rows:
                return pending_rows
            deferred_rows = tuple(
                (
                    await session.scalars(
                        select(CandidateModel.candidate_id)
                        .where(CandidateModel.status == CandidateStatus.DEFERRED)
                        .order_by(CandidateModel.created_at, CandidateModel.candidate_id)
                        .limit(size)
                    )
                ).all()
            )
            return deferred_rows

    async def pending_count(self) -> int:
        """统计待处理候选总数。"""
        async with self._schema.database.session() as session:
            return int(
                await session.scalar(
                    select(func.count())
                    .select_from(CandidateModel)
                    .where(CandidateModel.status == CandidateStatus.PENDING)
                )
            )

    async def actionable_count(self) -> int:
        """统计可由下一次 Sleep 认领的 PENDING 与 DEFERRED 候选。"""
        async with self._schema.database.session() as session:
            return int(
                await session.scalar(
                    select(func.count())
                    .select_from(CandidateModel)
                    .where(
                        CandidateModel.status.in_(
                            (CandidateStatus.PENDING, CandidateStatus.DEFERRED)
                        )
                    )
                )
            )

    async def candidate_payload(self, candidate_id: str) -> str:
        """把候选素材格式化为 Agent 步骤输入文本。"""
        async with self._schema.database.session() as session:
            candidate = await session.get(CandidateModel, candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate {candidate_id} 不存在")
        return (
            f"candidate_id: {candidate.candidate_id}\n"
            f"rough_title: {candidate.rough_title}\n"
            f"rough_content: {candidate.rough_content}\n"
            f"retention_reason: {candidate.retention_reason}\n"
            f"proposed_kind: {candidate.proposed_kind.value if candidate.proposed_kind else None}\n"
            f"confidence_hint: {candidate.confidence_hint.value if candidate.confidence_hint else None}\n"
            f"salience_hint: {candidate.salience_hint.value if candidate.salience_hint else None}\n"
            f"observed_at: {candidate.observed_at.isoformat()}\n"
            f"uncertainty_note: {candidate.uncertainty_note}"
        )

    async def _candidate_context(self, candidate_id: str) -> _CandidateContext:
        """读取候选及其初步主体、参与者和证据 ID。"""
        async with self._schema.database.session() as session:
            candidate = await session.get(CandidateModel, candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} 不存在")
            evidence_ids = tuple(
                (
                    await session.scalars(
                        select(CandidateEvidenceModel.evidence_id).where(
                            CandidateEvidenceModel.candidate_id == candidate_id
                        )
                    )
                ).all()
            )
            subject_row = await session.get(CandidateSubjectModel, candidate_id)
            participant_rows = tuple(
                (
                    await session.scalars(
                        select(CandidateParticipantModel)
                        .where(CandidateParticipantModel.candidate_id == candidate_id)
                        .order_by(CandidateParticipantModel.participant_id)
                    )
                ).all()
            )
        subject = SubjectInput(
            subject_kind=subject_row.subject_kind if subject_row else SubjectKind.UNKNOWN,
            person_id=subject_row.person_id if subject_row else None,
            subject_key=subject_row.subject_key if subject_row else None,
            subject_label=subject_row.subject_label if subject_row else None,
        )
        participants = tuple(
            ParticipantInput(
                participant_kind=row.participant_kind,
                person_id=row.person_id,
                label=row.label,
                role=row.role,
            )
            for row in participant_rows
        )
        return _CandidateContext(
            candidate_id=candidate.candidate_id,
            rough_title=candidate.rough_title,
            rough_content=candidate.rough_content,
            retention_reason=candidate.retention_reason,
            observed_at=candidate.observed_at,
            evidence_ids=evidence_ids,
            subject=subject,
            participants=participants,
            proposed_kind=candidate.proposed_kind,
            confidence_hint=candidate.confidence_hint,
            salience_hint=candidate.salience_hint,
        )

    async def run_session(
        self,
        trigger_type: SleepTriggerType,
        step_producer: AgentStepProducer,
    ) -> SleepSessionStatus:
        """执行一次完整睡眠会话并返回最终状态。

        流程（§63）：认领批次 → 逐候选多步整理（动作执行 +
        self-check 重读）→ 终态收口 → 会话结束。全部成功
        COMPLETED；存在失败候选 PARTIAL；启动失败 FAILED。
        """
        recovered_memory_ids = await self._resume_action_operations()
        await self._sessions.recover_interrupted_candidates()
        candidate_ids = await self.pending_candidates()
        self.last_changed_memory_ids = tuple(dict.fromkeys(recovered_memory_ids))
        session_result = await self._sessions.start_session(
            SleepSessionInput(
                trigger_type=trigger_type,
                model_id=self._model_id,
                prompt_version=self._prompt_version,
            ),
            candidate_ids,
        )
        self.last_session_id = session_result.sleep_session_id
        changed_memory_ids: list[str] = list(recovered_memory_ids)
        failed = 0
        try:
            for candidate_id in candidate_ids:
                try:
                    ok = await self._process_candidate(
                        session_result.sleep_session_id,
                        candidate_id,
                        step_producer,
                    )
                    changed_memory_ids.extend(ok[1])
                    if not ok[0]:
                        await self._sessions.fail_candidate(
                            candidate_id,
                            session_result.sleep_session_id,
                            error="Agent 未产出可执行动作",
                        )
                        failed += 1
                except AgentProtocolError:
                    await self._sessions.abort_session(
                        session_result.sleep_session_id,
                        "Agent Producer 返回了无效结构化结果",
                    )
                    raise
                except Exception as error:
                    failed += 1
                    try:
                        await self._sessions.fail_candidate(
                            candidate_id,
                            session_result.sleep_session_id,
                            error=str(error) or error.__class__.__name__,
                        )
                    except ValueError:
                        # 动作可能已经释放候选；会话级收口仍会处理剩余认领。
                        pass
            status = (
                SleepSessionStatus.COMPLETED
                if failed == 0
                else SleepSessionStatus.PARTIAL
            )
            await self._sessions.finish_session(
                session_result.sleep_session_id,
                status,
                error_summary=None if failed == 0 else f"{failed} 个候选处理失败",
            )
            self.last_changed_memory_ids = tuple(dict.fromkeys(changed_memory_ids))
            return status
        except Exception as error:
            await self._sessions.abort_session(
                session_result.sleep_session_id,
                str(error) or error.__class__.__name__,
            )
            raise

    async def _resume_action_operations(self) -> tuple[str, ...]:
        """在新会话询问模型前续跑完整动作计划。"""
        changed_memory_ids: list[str] = []
        plans = await self._sessions.list_incomplete_plans()
        planned_operation_keys: set[str] = set()
        for plan in plans:
            intents = tuple(plan.intents_json)
            operation_keys = tuple(plan.operation_keys_json)
            planned_operation_keys.update(operation_keys)
            if len(intents) != plan.action_count or len(operation_keys) != plan.action_count:
                raise ValueError("Sleep Action Plan 内容与游标不一致")
            for index in range(plan.next_action_index, plan.action_count):
                intent = intents[index]
                if not isinstance(intent, dict):
                    raise ValueError("Sleep Action Plan intent_json 无效")
                action_type = self._parse_action_type(intent.get("action_type"))
                target_ids = await self._execute_intent(
                    plan.sleep_session_id,
                    plan.candidate_id,
                    intent,
                    replace(
                        self._context,
                        sleep_session_id=plan.sleep_session_id,
                    ),
                    action_index=index,
                    operation_key=operation_keys[index],
                    plan_key=plan.plan_key,
                    expected_action_type=action_type,
                )
                await self._sessions.advance_action_plan(plan.plan_key, index, target_ids)
                changed_memory_ids.extend(target_ids)
            changed_memory_ids.extend(
                await self._sessions.finalize_action_plan(plan.plan_key)
            )
        for operation in await self._sessions.list_incomplete_operations():
            if operation.operation_key in planned_operation_keys:
                continue
            if operation.status != "COMPLETED":
                continue
            await self._sessions.record_operation_action(operation.operation_key)
            await self._sessions.finalize_recovered_operation(operation.operation_key)
            result = operation.result_json or {}
            changed_memory_ids.extend(
                str(item) for item in result.get("target_memory_ids", ())
            )
        return tuple(dict.fromkeys(changed_memory_ids))

    async def _process_candidate(
        self,
        sleep_session_id: str,
        candidate_id: str,
        step_producer: AgentStepProducer,
    ) -> tuple[bool, tuple[str, ...]]:
        """处理单个候选：执行动作意图并审计，self-check 重读结果。

        返回值的第一个元素表示是否完成候选处理，第二个元素是本轮
        触及的 Formal Memory ID；模型未产出动作时第一个元素为 False。
        """
        candidate_context = await self._candidate_context(candidate_id)
        context = replace(
            self._context,
            sleep_session_id=sleep_session_id,
        )
        payload = await self.candidate_payload(candidate_id)
        intents = await self._agent_decide(
            candidate_context,
            context,
            payload,
            step_producer,
        )
        if not intents:
            # An empty decision is a valid safety outcome.  It must not turn
            # into a terminal failure, otherwise uncertain evidence becomes a
            # permanently failed Candidate instead of a retryable DEFER.
            await self._sessions.record_action(
                CandidateActionInput(
                    candidate_id=candidate_id,
                    sleep_session_id=sleep_session_id,
                    action_type=CandidateActionType.DEFER,
                    note="Sleep Agent 未能基于当前证据形成安全动作",
                )
            )
            return True, ()
        has_terminal = False
        changed_memory_ids: list[str] = []
        action_specs: list[tuple[str, CandidateActionType, dict[str, object]]] = []
        for index, intent in enumerate(intents):
            if not isinstance(intent, Mapping):
                raise AgentProtocolError("每个动作意图必须是对象")
            intent = dict(intent)
            action_type = self._parse_action_type(intent.get("action_type"))
            if action_type in (CandidateActionType.IGNORE, CandidateActionType.DEFER):
                if index != len(intents) - 1:
                    raise ValueError("终态动作必须是候选的最后一个动作")
                has_terminal = True
            action_specs.append(
                (
                    self._operation_key(sleep_session_id, candidate_id, index, intent),
                    action_type,
                    intent,
                )
            )
        plan_key = self._plan_key(sleep_session_id, candidate_id, tuple(item[2] for item in action_specs))
        await self._sessions.prepare_action_plan(
            candidate_id,
            sleep_session_id,
            plan_key,
            tuple(action_specs),
        )
        for index, (operation_key, action_type, intent) in enumerate(action_specs):
            target_ids = await self._execute_intent(
                sleep_session_id,
                candidate_id,
                intent,
                context,
                action_index=index,
                operation_key=operation_key,
                plan_key=plan_key,
                expected_action_type=action_type,
            )
            await self._sessions.advance_action_plan(plan_key, index, target_ids)
            changed_memory_ids.extend(target_ids)
            for memory_id in target_ids:
                await self._tools.memory_read(memory_id, "full", context)
        if has_terminal:
            # 终态动作已释放候选，self-check 确认可读取即可
            await self.candidate_payload(candidate_id)
            return True, tuple(dict.fromkeys(changed_memory_ids))
        # 非终态动作完成完整计划后才收口 Candidate。
        changed_memory_ids.extend(await self._sessions.finalize_action_plan(plan_key))
        return True, tuple(dict.fromkeys(changed_memory_ids))

    async def _agent_decide(
        self,
        candidate_context: _CandidateContext,
        tool_context: ToolContext,
        payload: str,
        step_producer: AgentStepProducer,
    ) -> tuple[dict[str, object], ...]:
        """运行单一 Sleep Agent 的有界观察循环并返回最终动作。"""
        produce = getattr(step_producer, "produce", None)
        if produce is None:
            research = await self._research_candidate(candidate_context, tool_context)
            if research:
                payload = f"{payload}\n\nSleep research:\n{research}"
            return await self._produce_step(step_producer, payload)

        observations: list[dict[str, object]] = []
        for step_number in range(1, 9):
            transcript = repr(observations[-8:])
            step_payload = (
                f"{payload}\n\nAgent step: {step_number}/8\n"
                "Choose exactly one observation tool step or one final DECIDE.\n"
                f"Tool observations:\n{transcript}"
            )
            steps = await self._produce_step(step_producer, step_payload)
            if not steps:
                return ()
            if len(steps) != 1:
                raise AgentProtocolError("单 Agent 每一步必须只返回一个 step")
            step = steps[0]
            if not isinstance(step, Mapping):
                raise AgentProtocolError("Agent step 必须是对象")
            step_type = self._optional_text(
                step.get("step_type") or step.get("action_type")
            )
            if step_type is None:
                raise AgentProtocolError("Agent step 缺少 step_type")
            normalized = step_type.upper()
            if normalized == "DECIDE":
                raw_actions = step.get("actions")
                if not isinstance(raw_actions, (list, tuple)):
                    raise AgentProtocolError("DECIDE.actions 必须是数组")
                actions = tuple(raw_actions)
                if not actions:
                    return ()
                if not all(self._is_final_action(item) for item in actions):
                    raise AgentProtocolError("DECIDE.actions 包含非法动作")
                return actions
            if normalized in {item.value for item in CandidateActionType}:
                raise AgentProtocolError("produce 路径必须通过 DECIDE.actions 返回动作")
            observation = await self._observe_step(
                normalized,
                step,
                tool_context,
            )
            observations.append(
                {
                    "step": step_number,
                    "tool": normalized,
                    "result": self._compact_observation(observation),
                }
            )
        return (
            {
                "action_type": CandidateActionType.DEFER.value,
                "note": "Agent 达到最大观察步数，无法安全形成决定",
            },
        )

    async def _observe_step(
        self,
        step_type: str,
        step: Mapping[str, object],
        context: ToolContext,
    ) -> object:
        """执行一个只读观察步骤并把结果回填 Agent。"""
        if step_type == "SEARCH":
            query = self._required_text(step.get("query"), "SEARCH query")
            return await self._tools.memory_search(query, context, limit=8)
        if step_type == "MEMORY_READ":
            memory_id = self._required_text(step.get("memory_id"), "MEMORY_READ memory_id")
            view = self._optional_text(step.get("view")) or "full"
            return await self._tools.memory_read(memory_id, view, context)
        if step_type == "EVIDENCE_READ":
            raw_ids = step.get("evidence_ids")
            if not isinstance(raw_ids, (list, tuple)):
                raise AgentProtocolError("EVIDENCE_READ.evidence_ids 必须是数组")
            evidence_ids = tuple(
                self._required_text(value, "EVIDENCE_READ evidence_id")
                for value in raw_ids
            )
            if not evidence_ids:
                raise AgentProtocolError("EVIDENCE_READ.evidence_ids 不能为空")
            return await self._tools.evidence_read(evidence_ids, context)
        if step_type == "PERSON_LOOKUP":
            person_id = self._required_text(step.get("person_id"), "PERSON_LOOKUP person_id")
            return await self._tools.person_lookup(person_id, context)
        raise AgentProtocolError(f"不支持的 Agent step: {step_type}")

    @staticmethod
    def _is_final_action(value: object) -> bool:
        """判断对象是否为最终候选动作。"""
        return isinstance(value, Mapping) and isinstance(
            value.get("action_type"), str
        ) and value.get("action_type", "").strip().upper() in {
            item.value for item in CandidateActionType
        }

    @staticmethod
    def _compact_observation(value: object) -> object:
        """限制工具观察写回 LLM 的大小，不保存私有思维链。"""
        text = repr(value)
        return text if len(text) <= 6000 else f"{text[:6000]}..."

    async def _research_candidate(
        self,
        context: _CandidateContext,
        tool_context: ToolContext,
    ) -> str:
        """执行 Sleep Agent 的受控全局检索、读取和证据复核。"""
        queries = tuple(
            dict.fromkeys(
                item.strip()
                for item in (
                    f"{context.rough_title}\n{context.rough_content}",
                    context.rough_title,
                    context.retention_reason,
                )
                if item.strip()
            )
        )
        result_by_memory_id: dict[str, dict[str, object]] = {}
        for query in queries:
            search_results = await self._tools.memory_search(
                query,
                tool_context,
                limit=8,
            )
            for item in search_results:
                memory_id = item.get("memory_id")
                if isinstance(memory_id, str) and memory_id.strip():
                    result_by_memory_id.setdefault(memory_id, item)
            if len(result_by_memory_id) >= 8:
                break
        if not result_by_memory_id:
            return "无相关 Formal Memory 命中。"
        readable: list[dict[str, object]] = []
        for item in tuple(result_by_memory_id.values())[:8]:
            memory_id = item.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id.strip():
                continue
            full = await self._tools.memory_read(memory_id, "full", tool_context)
            evidence_summary = full.get("evidence_summary", ())
            evidence_ids = tuple(
                evidence.get("evidence_id")
                for evidence in evidence_summary
                if isinstance(evidence, dict)
                and isinstance(evidence.get("evidence_id"), str)
            )
            evidence = (
                await self._tools.evidence_read(evidence_ids, tool_context)
                if evidence_ids
                else ()
            )
            readable.append(
                {
                    "search": item,
                    "memory": {
                        "memory_id": memory_id,
                        "current": full.get("current_revision"),
                        "subject": full.get("current_subject"),
                        "assessment": full.get("current_assessment"),
                        "evidence": evidence,
                    },
                }
            )
        return repr(readable)

    async def _produce_step(
        self,
        step_producer: AgentStepProducer,
        payload: str,
    ) -> tuple[dict[str, object], ...]:
        """执行一个 Agent 决策步骤，兼容同步测试生产者与异步运行时生产者。"""
        produce = getattr(step_producer, "produce", None)
        if produce is None:
            result = step_producer(payload)
        else:
            result = produce(payload)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, tuple):
            raise AgentProtocolError("step producer 必须返回 tuple")
        return result

    async def _execute_intent(
        self,
        sleep_session_id: str,
        candidate_id: str,
        intent: dict[str, object],
        tool_context: ToolContext | None = None,
        *,
        action_index: int = 0,
        operation_key: str | None = None,
        plan_key: str | None = None,
        expected_action_type: CandidateActionType | None = None,
    ) -> tuple[str, ...]:
        """执行单个动作意图并写候选动作审计。"""
        action_type = self._parse_action_type(intent.get("action_type"))
        if expected_action_type is not None and action_type is not expected_action_type:
            raise AgentProtocolError("Sleep Action Plan action_type 已被篡改")
        operation_key = operation_key or self._operation_key(
            sleep_session_id, candidate_id, action_index, intent
        )
        if plan_key is None:
            await self._sessions.prepare_operation(
                candidate_id,
                sleep_session_id,
                action_type,
                operation_key,
                intent,
            )
        operation = await self._sessions.get_operation(operation_key)
        if operation is not None and operation.status == "COMPLETED":
            await self._sessions.record_operation_action(operation_key)
            result = operation.result_json or {}
            return tuple(str(item) for item in result.get("target_memory_ids", ()))
        await self._sessions.mark_operation_executing(operation_key)
        context = await self._candidate_context(candidate_id)
        actor_context = replace(
            tool_context or self._context,
            sleep_session_id=sleep_session_id,
            operation_key=operation_key,
        )
        result_revision_id: str | None = None
        targets: tuple[tuple[str, CandidateActionTargetRole], ...] = ()
        target_memory_ids: list[str] = []
        domain_result: dict[str, object] = {}
        if action_type is CandidateActionType.CREATE_NEW:
            result = await self._tools.memory_write(
                self._create_memory_input(context, intent),
                actor_context,
            )
            result_revision_id = result["revision_id"]
            targets = ((result["memory_id"], CandidateActionTargetRole.RESULT),)
            target_memory_ids.append(result["memory_id"])
            domain_result = dict(result)
        elif action_type is CandidateActionType.REINFORCE:
            result = await self._tools.memory_reinforce(
                self._reinforce_memory_input(context, intent),
                actor_context,
            )
            result_revision_id = result["revision_id"]
            targets = ((result["memory_id"], CandidateActionTargetRole.TARGET),)
            target_memory_ids.append(result["memory_id"])
            domain_result = dict(result)
        elif action_type is CandidateActionType.REVISE:
            result = await self._tools.memory_revise(
                self._revise_memory_input(context, intent),
                actor_context,
            )
            result_revision_id = result["revision_id"]
            targets = ((result["memory_id"], CandidateActionTargetRole.TARGET),)
            target_memory_ids.append(result["memory_id"])
            domain_result = dict(result)
        elif action_type is CandidateActionType.MERGE:
            merge = self._merge_memory_input(intent, context)
            result = await self._tools.memory_merge(merge, actor_context)
            canonical_memory_id = result["canonical_memory_id"]
            targets = tuple(
                (memory_id, CandidateActionTargetRole.SOURCE)
                for memory_id in merge.source_memory_ids
            ) + ((canonical_memory_id, CandidateActionTargetRole.CANONICAL),)
            target_memory_ids.extend(merge.source_memory_ids)
            target_memory_ids.append(canonical_memory_id)
            domain_result = dict(result)
        elif action_type is CandidateActionType.RELATE:
            relate = self._relate_memory_input(intent)
            result = await self._tools.memory_relate(relate, actor_context)
            targets = (
                (relate.source_memory_id, CandidateActionTargetRole.SOURCE),
                (relate.target_memory_id, CandidateActionTargetRole.TARGET),
            )
            target_memory_ids.extend((relate.source_memory_id, relate.target_memory_id))
            domain_result = dict(result)
        domain_result["revision_id"] = result_revision_id
        domain_result["targets"] = [list(item) for item in targets]
        domain_result["target_memory_ids"] = list(dict.fromkeys(target_memory_ids))
        await self._sessions.complete_operation(operation_key, domain_result)
        await self._sessions.record_operation_action(operation_key)
        return tuple(dict.fromkeys(target_memory_ids))

    @staticmethod
    def _plan_key(
        sleep_session_id: str,
        candidate_id: str,
        intents: tuple[dict[str, object], ...],
    ) -> str:
        """生成由候选、会话和完整动作列表决定的稳定计划 key。"""
        canonical = json.dumps(intents, sort_keys=True, separators=(",", ":"), default=str)
        return str(
            uuid5(
                NAMESPACE_URL,
                f"engram-vnext-sleep-plan:{sleep_session_id}:{candidate_id}:{canonical}",
            )
        )

    @staticmethod
    def _operation_key(
        sleep_session_id: str,
        candidate_id: str,
        action_index: int,
        intent: Mapping[str, object],
    ) -> str:
        """生成由会话、候选、动作序号和完整意图决定的稳定 key。"""
        canonical = json.dumps(intent, sort_keys=True, separators=(",", ":"), default=str)
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        return str(
            uuid5(
                NAMESPACE_URL,
                f"engram-vnext-sleep:{sleep_session_id}:{candidate_id}:{action_index}:{digest}",
            )
        )

    @staticmethod
    def _create_memory_input(
        context: _CandidateContext,
        intent: Mapping[str, object],
    ) -> CreateMemoryInput:
        """把 CREATE_NEW 意图转换为正式记忆输入。"""
        kind = SleepAgentOrchestrator._enum_value(
            MemoryKind,
            intent.get("memory_kind"),
            context.proposed_kind or MemoryKind.EVENT,
        )
        confidence = SleepAgentOrchestrator._enum_value(
            ConfidenceLevel,
            intent.get("confidence"),
            context.confidence_hint or ConfidenceLevel.MEDIUM,
        )
        salience = SleepAgentOrchestrator._enum_value(
            SalienceLevel,
            intent.get("salience"),
            context.salience_hint or SalienceLevel.MEDIUM,
        )
        return CreateMemoryInput(
            anchor_title=SleepAgentOrchestrator._text_or_default(
                intent.get("anchor_title"), context.rough_title
            ),
            title=SleepAgentOrchestrator._text_or_default(
                intent.get("title"), context.rough_title
            ),
            content=SleepAgentOrchestrator._text_or_default(
                intent.get("content"), context.rough_content
            ),
            memory_kind=kind,
            subject=SleepAgentOrchestrator._subject_value(intent.get("subject"), context.subject),
            participants=SleepAgentOrchestrator._participants_value(
                intent.get("participants"), context.participants
            ),
            confidence=confidence,
            confidence_reason=SleepAgentOrchestrator._text_or_default(
                intent.get("confidence_reason"), context.retention_reason
            ),
            observed_at=SleepAgentOrchestrator._datetime_value(
                intent.get("observed_at"), context.observed_at
            ),
            event_time_precision=SleepAgentOrchestrator._enum_value(
                EventTimePrecision,
                intent.get("event_time_precision"),
                EventTimePrecision.UNKNOWN,
            ),
            event_time_origin=SleepAgentOrchestrator._enum_value(
                EventTimeOrigin,
                intent.get("event_time_origin"),
                EventTimeOrigin.UNKNOWN,
            ),
            stability=SleepAgentOrchestrator._enum_value(
                StabilityLevel,
                intent.get("stability"),
                StabilityLevel.LOW,
            ),
            salience=salience,
            assessment_reason=SleepAgentOrchestrator._text_or_default(
                intent.get("assessment_reason"), context.retention_reason
            ),
            evidence_ids=context.evidence_ids,
            event_start_at=SleepAgentOrchestrator._optional_datetime(
                intent.get("event_start_at")
            ),
            event_end_at=SleepAgentOrchestrator._optional_datetime(
                intent.get("event_end_at")
            ),
        )

    @staticmethod
    def _reinforce_memory_input(
        context: _CandidateContext,
        intent: Mapping[str, object],
    ) -> ReinforceMemoryInput:
        """把 REINFORCE 意图转换为正式记忆输入。"""
        memory_id = SleepAgentOrchestrator._required_text(
            intent.get("memory_id"), "REINFORCE memory_id"
        )
        revision_id = SleepAgentOrchestrator._required_text(
            intent.get("based_on_revision_id"), "REINFORCE based_on_revision_id"
        )
        assessment = SleepAgentOrchestrator._assessment_value(intent)
        return ReinforceMemoryInput(
            memory_id=memory_id,
            based_on_revision_id=revision_id,
            reason=SleepAgentOrchestrator._text_or_default(
                intent.get("reason"), context.retention_reason
            ),
            evidence_ids=context.evidence_ids,
            assessment=assessment,
        )

    @staticmethod
    def _revise_memory_input(
        context: _CandidateContext,
        intent: Mapping[str, object],
    ) -> ReviseMemoryInput:
        """把 REVISE 意图转换为正式记忆输入。"""
        memory_id = SleepAgentOrchestrator._required_text(
            intent.get("memory_id"), "REVISE memory_id"
        )
        based_on = SleepAgentOrchestrator._required_text(
            intent.get("based_on_revision_id"), "REVISE based_on_revision_id"
        )
        return ReviseMemoryInput(
            memory_id=memory_id,
            based_on_revision_id=based_on,
            title=SleepAgentOrchestrator._text_or_default(
                intent.get("title"), context.rough_title
            ),
            content=SleepAgentOrchestrator._text_or_default(
                intent.get("content"), context.rough_content
            ),
            memory_kind=SleepAgentOrchestrator._enum_value(
                MemoryKind,
                intent.get("memory_kind"),
                context.proposed_kind or MemoryKind.EVENT,
            ),
            subject=SleepAgentOrchestrator._subject_value(intent.get("subject"), context.subject),
            participants=SleepAgentOrchestrator._participants_value(
                intent.get("participants"), context.participants
            ),
            confidence=SleepAgentOrchestrator._enum_value(
                ConfidenceLevel,
                intent.get("confidence"),
                context.confidence_hint or ConfidenceLevel.MEDIUM,
            ),
            confidence_reason=SleepAgentOrchestrator._text_or_default(
                intent.get("confidence_reason"), context.retention_reason
            ),
            observed_at=SleepAgentOrchestrator._datetime_value(
                intent.get("observed_at"), context.observed_at
            ),
            event_time_precision=SleepAgentOrchestrator._enum_value(
                EventTimePrecision,
                intent.get("event_time_precision"),
                EventTimePrecision.UNKNOWN,
            ),
            event_time_origin=SleepAgentOrchestrator._enum_value(
                EventTimeOrigin,
                intent.get("event_time_origin"),
                EventTimeOrigin.UNKNOWN,
            ),
            change_reason=SleepAgentOrchestrator._enum_value(
                RevisionChangeReason,
                intent.get("change_reason"),
                RevisionChangeReason.DEVELOPMENT,
            ),
            assessment=SleepAgentOrchestrator._assessment_value(intent)
            or AssessmentInput(
                stability=StabilityLevel.LOW,
                salience=context.salience_hint or SalienceLevel.MEDIUM,
                reason=context.retention_reason,
            ),
            evidence_ids=context.evidence_ids,
            event_start_at=SleepAgentOrchestrator._optional_datetime(
                intent.get("event_start_at")
            ),
            event_end_at=SleepAgentOrchestrator._optional_datetime(
                intent.get("event_end_at")
            ),
            new_anchor_title=SleepAgentOrchestrator._optional_text(
                intent.get("new_anchor_title")
            ),
        )

    @staticmethod
    def _merge_memory_input(
        intent: Mapping[str, object],
        context: _CandidateContext | None = None,
    ) -> MergeMemoryInput:
        """把 MERGE 意图转换为合并输入。"""
        raw_sources = intent.get("source_memory_ids")
        if not isinstance(raw_sources, (list, tuple)):
            raise ValueError("MERGE source_memory_ids 必须是数组")
        sources = tuple(
            SleepAgentOrchestrator._required_text(item, "MERGE source_memory_id")
            for item in raw_sources
        )
        mode = (SleepAgentOrchestrator._optional_text(intent.get("mode")) or "EXISTING_CANONICAL").upper()
        new_memory_value = intent.get("new_memory")
        new_memory = None
        if mode == "NEW_CANONICAL":
            if not isinstance(new_memory_value, Mapping):
                raise ValueError("NEW_CANONICAL new_memory 必须是对象")
            if context is None:
                raise ValueError("NEW_CANONICAL 需要 Candidate 上下文")
            new_memory = SleepAgentOrchestrator._create_memory_input(context, new_memory_value)
        canonical_memory_id = (
            SleepAgentOrchestrator._required_text(
                intent.get("canonical_memory_id"), "MERGE canonical_memory_id"
            )
            if mode == "EXISTING_CANONICAL"
            else None
        )
        return MergeMemoryInput(
            source_memory_ids=sources,
            canonical_memory_id=canonical_memory_id,
            reason=SleepAgentOrchestrator._required_text(
                intent.get("reason"), "MERGE reason"
            ),
            mode=mode,
            new_memory=new_memory,
        )

    @staticmethod
    def _relate_memory_input(intent: Mapping[str, object]) -> RelateMemoryInput:
        """把 RELATE 意图转换为关系输入。"""
        return RelateMemoryInput(
            source_memory_id=SleepAgentOrchestrator._required_text(
                intent.get("source_memory_id"), "RELATE source_memory_id"
            ),
            target_memory_id=SleepAgentOrchestrator._required_text(
                intent.get("target_memory_id"), "RELATE target_memory_id"
            ),
            relation_type=SleepAgentOrchestrator._enum_value(
                RelationType,
                intent.get("relation_type"),
                RelationType.RELATED_TO,
            ),
            reason=SleepAgentOrchestrator._required_text(
                intent.get("reason"), "RELATE reason"
            ),
        )

    @staticmethod
    def _assessment_value(intent: Mapping[str, object]) -> AssessmentInput | None:
        """解析可选的认知评估对象。"""
        raw = intent.get("assessment")
        source: Mapping[str, object] = raw if isinstance(raw, Mapping) else intent
        if raw is None and not any(
            key in intent for key in ("stability", "salience", "assessment_reason")
        ):
            return None
        return AssessmentInput(
            stability=SleepAgentOrchestrator._enum_value(
                StabilityLevel, source.get("stability"), StabilityLevel.LOW
            ),
            salience=SleepAgentOrchestrator._enum_value(
                SalienceLevel, source.get("salience"), SalienceLevel.MEDIUM
            ),
            reason=SleepAgentOrchestrator._text_or_default(
                source.get("reason", source.get("assessment_reason")),
                "Sleep Agent 依据候选证据更新评估",
            ),
        )

    @staticmethod
    def _subject_value(
        value: object,
        fallback: SubjectInput,
    ) -> SubjectInput:
        """解析主体对象并在缺失时使用候选初步主体。"""
        if not isinstance(value, Mapping):
            return fallback
        return SubjectInput(
            subject_kind=SleepAgentOrchestrator._enum_value(
                SubjectKind, value.get("subject_kind"), fallback.subject_kind
            ),
            person_id=SleepAgentOrchestrator._optional_text(value.get("person_id")),
            subject_key=SleepAgentOrchestrator._optional_text(value.get("subject_key")),
            subject_label=SleepAgentOrchestrator._optional_text(value.get("subject_label")),
        )

    @staticmethod
    def _participants_value(
        value: object,
        fallback: tuple[ParticipantInput, ...],
    ) -> tuple[ParticipantInput, ...]:
        """解析参与者数组并在缺失时使用候选初步参与者。"""
        if not isinstance(value, (list, tuple)):
            return fallback
        participants: list[ParticipantInput] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError("participants 的每一项必须是对象")
            participants.append(
                ParticipantInput(
                    participant_kind=SleepAgentOrchestrator._enum_value(
                        ParticipantKind, item.get("participant_kind"), ParticipantKind.OTHER
                    ),
                    role=SleepAgentOrchestrator._enum_value(
                        ParticipantRole, item.get("role"), ParticipantRole.MENTIONED
                    ),
                    person_id=SleepAgentOrchestrator._optional_text(item.get("person_id")),
                    label=SleepAgentOrchestrator._optional_text(item.get("label")),
                )
            )

        return tuple(participants)

    @staticmethod
    def _enum_value(enum_type: type, value: object, fallback: object) -> object:
        """解析字符串枚举值，缺失时返回指定默认值。"""
        if value is None:
            return fallback
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return enum_type(value.strip().upper())
            except ValueError as error:
                raise ValueError(f"无效枚举值: {value}") from error
        raise ValueError("枚举字段必须是字符串")

    @staticmethod
    def _datetime_value(value: object, fallback: datetime) -> datetime:
        """解析带时区的时间值。"""
        parsed = SleepAgentOrchestrator._optional_datetime(value)
        return parsed or fallback

    @staticmethod
    def _optional_datetime(value: object) -> datetime | None:
        """解析可空 ISO 时间。"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"无效时间值: {value}") from error
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        raise ValueError("时间字段必须是 ISO 文本或 datetime")

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        """读取非空文本字段。"""
        text_value = SleepAgentOrchestrator._optional_text(value)
        if not text_value:
            raise ValueError(f"{field_name} 不能为空")
        return text_value

    @staticmethod
    def _optional_text(value: object) -> str | None:
        """读取可空文本字段。"""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("文本字段必须是字符串")
        text_value = value.strip()
        return text_value or None

    @staticmethod
    def _text_or_default(value: object, fallback: str) -> str:
        """读取文本字段，缺失时使用非空默认值。"""
        return SleepAgentOrchestrator._optional_text(value) or fallback

    @staticmethod
    def _parse_action_type(value: object) -> CandidateActionType:
        """把意图字段解析为候选动作枚举。"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("动作意图缺少 action_type")
        return CandidateActionType(value.strip())

    async def attempt_final_resolution(
        self,
        sleep_session_id: str,
        candidate_id: str,
    ) -> bool:
        """对只有非终态动作的候选显式收口为 RESOLVED。

        返回是否成功收口（无动作时 False，由调用方按失败处理）。
        """
        try:
            await self._sessions.resolve_candidate(candidate_id, sleep_session_id)
        except ValueError:
            return False
        return True
