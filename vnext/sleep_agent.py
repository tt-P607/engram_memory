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
from datetime import UTC, datetime, time, timedelta
import re
from time import perf_counter
from typing import Callable, Mapping
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select

from src.app.plugin_system.api.log_api import COLOR, get_logger

from .candidate_service import CandidateService, SleepSessionService
from .domain import (
    CandidateActionInput,
    CreateMemoryInput,
    EvidenceMessageInput,
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
    MemoryKind,
    MemoryStatus,
    RelationType,
    RevisionChangeReason,
    SleepSessionStatus,
    SleepTriggerType,
    SubjectKind,
    ParticipantKind,
)
from .models import (
    CandidateActionModel,
    CandidateEvidenceModel,
    CandidateModel,
    CandidateParticipantModel,
    CandidateSubjectModel,
    MemoryModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionModel,
    MemoryRevisionSubjectModel,
    SleepActionOperationModel,
    SleepSessionCandidateModel,
)
from .message_display import has_unseparated_reply_preview, split_reply_preview
from .schema import VNextSchema
from .tool_service import ToolContext, VNextToolService

logger = get_logger(
    "engram_memory.vnext.sleep_agent",
    display="Engram Sleep",
    color=COLOR.CYAN,
)

MAX_CANDIDATES_PER_GROUP = 100
MAX_SLEEP_AGENT_STEPS = 8


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
    observed_at: datetime
    evidence_ids: tuple[str, ...]
    source_snapshots: tuple[dict[str, object], ...]
    identity_aliases: tuple[str, ...]
    subject: SubjectInput
    participants: tuple[ParticipantInput, ...]
    proposed_kind: MemoryKind | None


class AgentStepProducer:
    """候选批次整理步骤 LLM 调用协议。

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
        automatic_since: datetime | None = None,
    ) -> None:
        """绑定 Schema、工具门面与会话参数。

        参数:
            schema: vNext Schema。
            tools: SLEEP_AGENT 上下文的工具门面。
            batch_size: 每次会话认领的候选上限。
            model_id: 本次会话使用的模型标识（§55）。
            prompt_version: Sleep Agent Prompt 显式版本号（§148）。
            automatic_since: 自动处理允许选取和恢复的候选起始时间。
        """
        if batch_size <= 0 or batch_size > MAX_CANDIDATES_PER_GROUP:
            raise ValueError(
                f"batch_size 必须在 1 到 {MAX_CANDIDATES_PER_GROUP} 之间"
            )
        if not model_id.strip() or not prompt_version.strip():
            raise ValueError("model_id 与 prompt_version 不能为空")
        self._schema = schema
        self._tools = tools
        self._sessions = sleep_service or SleepSessionService(schema)
        self._candidates = CandidateService(schema)
        self._batch_size = batch_size
        self._automatic_since = (
            automatic_since.astimezone(UTC)
            if automatic_since is not None and automatic_since.tzinfo is not None
            else automatic_since.replace(tzinfo=UTC)
            if automatic_since is not None
            else None
        )
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._context = ToolContext(actor_type=ActorType.SLEEP_AGENT)
        self.last_session_id: str | None = None
        self.last_changed_memory_ids: tuple[str, ...] = ()
        self.last_llm_calls = 0

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

    async def pending_candidates(
        self, limit: int | None = None, *, offset: int = 0
    ) -> tuple[str, ...]:
        """按最近尝试时间读取 PENDING 与 DEFERRED 候选。"""
        if offset < 0:
            raise ValueError("offset 不能小于 0")
        size = limit if limit is not None else self._batch_size
        if size < 0:
            raise ValueError("limit 不能小于 0")
        if size == 0:
            return ()
        cutoff = (
            (CandidateModel.created_at >= self._automatic_since,)
            if self._automatic_since is not None
            else ()
        )
        last_attempt_at = (
            select(func.max(CandidateActionModel.created_at))
            .where(CandidateActionModel.candidate_id == CandidateModel.candidate_id)
            .scalar_subquery()
        )
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(CandidateModel.candidate_id)
                        .where(
                            CandidateModel.status.in_(
                                (CandidateStatus.PENDING, CandidateStatus.DEFERRED)
                            )
                        )
                        .where(*cutoff)
                        .order_by(
                            func.coalesce(last_attempt_at, CandidateModel.created_at),
                            CandidateModel.candidate_id,
                        )
                        .offset(offset)
                        .limit(size)
                    )
                ).all()
            )
            return rows

    async def pending_count(self) -> int:
        """统计待处理候选总数。"""
        statement = (
            select(func.count())
            .select_from(CandidateModel)
            .where(CandidateModel.status == CandidateStatus.PENDING)
        )
        if self._automatic_since is not None:
            statement = statement.where(
                CandidateModel.created_at >= self._automatic_since
            )
        async with self._schema.database.session() as session:
            return int(await session.scalar(statement))

    async def run_grouped_session(
        self,
        trigger_type: SleepTriggerType,
        step_producer: AgentStepProducer,
        candidate_ids: tuple[str, ...],
        trace: Callable[[str, dict[str, object]], None] | None = None,
    ) -> SleepSessionStatus:
        """按配置批量整理指定候选，并记录可选的决策轨迹。"""
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("候选 ID 不能重复")
        recovered_memory_ids = await self._resume_action_operations()
        await self._sessions.recover_interrupted_candidates(
            automatic_since=self._automatic_since
        )
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(CandidateModel).where(
                            CandidateModel.candidate_id.in_(candidate_ids)
                        )
                    )
                ).all()
            )
        if len(rows) != len(candidate_ids) or any(
            candidate.status
            not in {CandidateStatus.PENDING, CandidateStatus.DEFERRED}
            for candidate in rows
        ):
            raise ValueError("本轮候选必须全部是 PENDING 或 DEFERRED")
        return await self._run_candidate_batch(
            trigger_type,
            step_producer,
            candidate_ids,
            trace,
            recovered_memory_ids,
        )

    async def _candidate_groups(
        self, candidate_ids: tuple[str, ...]
    ) -> tuple[tuple[str, ...], ...]:
        """按人物、主题与时间关联排列候选并限制每组大小。"""
        contexts = tuple([
            await self._candidate_context(candidate_id)
            for candidate_id in candidate_ids
        ])
        remaining = list(contexts)
        ordered_groups: list[tuple[str, ...]] = []
        while remaining:
            seed = min(
                remaining,
                key=lambda item: (item.observed_at, item.candidate_id),
            )
            remaining.remove(seed)
            group = [seed]
            while remaining and len(group) < self._batch_size:
                next_context = max(
                    remaining,
                    key=lambda item: self._candidate_group_affinity(item, tuple(group)),
                )
                people, subjects, topics, *_ = self._candidate_group_affinity(
                    next_context, tuple(group)
                )
                if not people and not subjects and topics < 2:
                    break
                remaining.remove(next_context)
                group.append(next_context)
            ordered_groups.append(tuple(item.candidate_id for item in group))
        return tuple(ordered_groups)

    @staticmethod
    def _candidate_group_affinity(
        candidate: _CandidateContext,
        group: tuple[_CandidateContext, ...],
    ) -> tuple[int, int, int, float, float, str]:
        """Return lightweight person, subject, topic and time affinity."""
        candidate_people = {
            person_id
            for person_id in (
                candidate.subject.person_id,
                *(item.person_id for item in candidate.participants),
            )
            if person_id and person_id.casefold() != "bot"
        }
        candidate_subjects = {
            value.casefold()
            for value in (candidate.subject.subject_key, candidate.subject.subject_label)
            if value
        }
        candidate_topics = SleepAgentOrchestrator._topic_tokens(
            candidate.rough_title
        )
        best = (0, 0, 0, float("-inf"), float("-inf"), "")
        for related in group:
            related_people = {
                person_id
                for person_id in (
                    related.subject.person_id,
                    *(item.person_id for item in related.participants),
                )
                if person_id and person_id.casefold() != "bot"
            }
            related_subjects = {
                value.casefold()
                for value in (related.subject.subject_key, related.subject.subject_label)
                if value
            }
            related_topics = SleepAgentOrchestrator._topic_tokens(
                related.rough_title
            )
            shared_people = len(candidate_people & related_people)
            shared_subjects = len(candidate_subjects & related_subjects)
            shared_topics = len(candidate_topics & related_topics)
            distance = abs((candidate.observed_at - related.observed_at).total_seconds())
            score = (
                shared_people,
                shared_subjects,
                shared_topics,
                -distance,
                -candidate.observed_at.timestamp(),
                candidate.candidate_id,
            )
            if score > best:
                best = score
        return best

    @staticmethod
    def _topic_tokens(value: str) -> frozenset[str]:
        """Extract small English terms and Chinese character bigrams."""
        normalized = value.casefold()
        tokens = set(re.findall(r"[a-z0-9]{2,}", normalized))
        for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
            if len(segment) == 2:
                tokens.add(segment)
            else:
                tokens.update(
                    segment[index : index + 2]
                    for index in range(len(segment) - 1)
                )
        return frozenset(tokens)

    async def _run_candidate_batch(
        self,
        trigger_type: SleepTriggerType,
        step_producer: AgentStepProducer,
        candidate_ids: tuple[str, ...],
        trace: Callable[[str, dict[str, object]], None] | None,
        recovered_memory_ids: tuple[str, ...],
    ) -> SleepSessionStatus:
        """认领候选组、隔离组级失败并结束 Sleep Session。"""
        started_at = perf_counter()
        log_enabled = bool(candidate_ids)

        def emit(event: str, payload: dict[str, object]) -> None:
            if log_enabled:
                self._log_trace_event(event, payload)
            if trace:
                trace(event, payload)

        groups = await self._candidate_groups(candidate_ids)
        session_result = await self._sessions.start_session(
            SleepSessionInput(trigger_type, self._model_id, self._prompt_version), candidate_ids
        )
        self.last_session_id = session_result.sleep_session_id
        self.last_llm_calls = 0
        changed: list[str] = list(recovered_memory_ids)
        failed = 0
        session_payload = {
            "sleep_session_id": session_result.sleep_session_id,
            "candidate_count": len(candidate_ids),
            "group_count": len(groups),
            "groups": groups,
        }
        if log_enabled:
            logger.info(
                f"[bold cyan]Sleep[/bold cyan] [bold green]批次开始[/bold green] "
                f"触发={trigger_type.value} 候选={len(candidate_ids)} 组={len(groups)} "
                f"会话={self._short_log_id(session_result.sleep_session_id)}",
                event="sleep_batch_start",
                trigger=trigger_type.value,
                candidate_count=len(candidate_ids),
                group_count=len(groups),
            )
        if trace:
            trace("session_start", session_payload)
        try:
            for group_index, group in enumerate(groups, start=1):
                group_started_at = perf_counter()
                if log_enabled:
                    logger.info(
                        f"[bold cyan]Sleep[/bold cyan] [bold blue]开始整理[/bold blue] "
                        f"第 {group_index}/{len(groups)} 组 候选={len(group)} "
                        f"ID={self._short_log_ids(group)}",
                        event="sleep_group_start",
                        group_index=group_index,
                        candidate_count=len(group),
                    )
                emit("group_start", {"candidate_ids": group})
                try:
                    changed.extend(await self._process_group(
                        session_result.sleep_session_id, group, step_producer, emit
                    ))
                except Exception as error:
                    if log_enabled:
                        logger.error(
                            f"[bold red]Sleep[/bold red] [red]分组失败[/red] "
                            f"候选={len(group)} ID={self._short_log_ids(group)} "
                            f"类型={type(error).__name__}",
                            event="sleep_group_failed",
                            candidate_count=len(group),
                            error_type=type(error).__name__,
                        )
                    if trace:
                        trace("group_error", {"candidate_ids": group, "error": str(error)})
                    async with self._schema.database.session() as session:
                        remaining = tuple((await session.scalars(
                            select(CandidateModel.candidate_id).where(
                                CandidateModel.candidate_id.in_(group),
                                CandidateModel.status == CandidateStatus.PROCESSING,
                                CandidateModel.processing_session_id == session_result.sleep_session_id,
                            )
                        )).all())
                    failed += len(remaining)
                    for candidate_id in group:
                        if candidate_id in remaining:
                            await self._sessions.fail_candidate(
                                candidate_id, session_result.sleep_session_id,
                                str(error) or type(error).__name__,
                            )
                else:
                    if log_enabled:
                        logger.info(
                            f"[bold cyan]Sleep[/bold cyan] [green]分组完成[/green] "
                            f"耗时={perf_counter() - group_started_at:.1f}s "
                            f"累计变更记忆={len(set(changed))}",
                            event="sleep_group_end",
                            elapsed_seconds=round(perf_counter() - group_started_at, 1),
                            changed_memory_count=len(set(changed)),
                        )
            status = SleepSessionStatus.PARTIAL if failed else SleepSessionStatus.COMPLETED
            await self._sessions.finish_session(
                session_result.sleep_session_id, status,
                error_summary=f"{failed} 个候选处理失败" if failed else None,
            )
            self.last_changed_memory_ids = tuple(dict.fromkeys(changed))
            if log_enabled:
                logger.info(
                    f"[bold cyan]Sleep[/bold cyan] [bold green]批次结束[/bold green] "
                    f"状态={status.value} 候选={len(candidate_ids)} 失败={failed} "
                    f"变更记忆={len(self.last_changed_memory_ids)} "
                    f"模型调用={self.last_llm_calls} 耗时={perf_counter() - started_at:.1f}s "
                    f"会话={self._short_log_id(session_result.sleep_session_id)}",
                    event="sleep_batch_end",
                    status=status.value,
                    candidate_count=len(candidate_ids),
                    failed_count=failed,
                    changed_memory_count=len(self.last_changed_memory_ids),
                    llm_calls=self.last_llm_calls,
                    elapsed_seconds=round(perf_counter() - started_at, 1),
                )
            emit("session_end", {"status": status.value, "llm_calls": self.last_llm_calls})
            return status
        except Exception as error:
            if log_enabled:
                logger.error(
                    f"[bold red]Sleep[/bold red] [red]会话中止[/red] "
                    f"候选={len(candidate_ids)} 类型={type(error).__name__} "
                    f"耗时={perf_counter() - started_at:.1f}s",
                    event="sleep_session_aborted",
                    candidate_count=len(candidate_ids),
                    error_type=type(error).__name__,
                    elapsed_seconds=round(perf_counter() - started_at, 1),
                )
            await self._sessions.abort_session(session_result.sleep_session_id, str(error))
            raise

    async def _process_group(
        self,
        session_id: str,
        group: tuple[str, ...],
        producer: AgentStepProducer,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> tuple[str, ...]:
        """多步调查后处理一组候选，并逐条记录来源与动作目标。"""
        contexts = {
            candidate_id: await self._candidate_context(candidate_id)
            for candidate_id in group
        }
        tool_context = replace(self._context, sleep_session_id=session_id)
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for context in contexts.values()
                for evidence_id in context.evidence_ids
            )
        )
        source_records = (
            await self._tools.evidence_read(evidence_ids, tool_context)
            if evidence_ids
            else ()
        )
        if trace and evidence_ids:
            trace(
                "EVIDENCE_READ",
                {"evidence_ids": evidence_ids, "result": source_records},
            )
        snapshots_by_evidence = self._snapshots_by_evidence(source_records)
        contexts = {
            candidate_id: replace(
                context,
                source_snapshots=tuple(
                    {**snapshot, "evidence_id": evidence_id}
                    for evidence_id in context.evidence_ids
                    for snapshot in snapshots_by_evidence.get(evidence_id, ())
                ),
            )
            for candidate_id, context in contexts.items()
        }
        group_context = self._combine_candidate_contexts(
            tuple(contexts[candidate_id] for candidate_id in group)
        )
        payloads = tuple(
            self._candidate_context_payload(contexts[candidate_id])
            for candidate_id in group
        )
        group_subjects = {
            (context.subject.subject_kind.value, context.subject.person_id or context.subject.subject_key)
            for context in contexts.values()
            if context.subject is not None
        }
        group_person_ids = {
            context.subject.person_id
            for context in contexts.values()
            if context.subject.subject_kind is SubjectKind.PERSON
            and context.subject.person_id
        }
        group_person_ids.update(
            participant.person_id
            for context in contexts.values()
            for participant in context.participants
            if participant.participant_kind is ParticipantKind.PERSON
            and participant.person_id
        )
        recent_memories = tuple(
            memory for memory in await self._recent_memories(
                tuple(sorted(group_person_ids))
            )
            if (
                (str(memory["subject_kind"]), memory["person_id"] or memory["subject_key"])
                in group_subjects
                or (
                    str(memory["subject_kind"]) == SubjectKind.PERSON.value
                    and memory["person_id"] in group_person_ids
                )
                or group_person_ids.intersection(memory["participant_person_ids"])
            )
        )
        if trace:
            trace("recent_formal_memories", {"memories": recent_memories})
        recent_directory = tuple(
            {
                key: memory[key]
                for key in (
                    "memory_id", "revision_id", "created_at_local", "updated_at_local",
                    "created_by_type", "title", "memory_kind", "person_id",
                )
                if memory.get(key) is not None
            }
            for memory in recent_memories
        )
        prompt = (
            "本组是待核对的候选素材，请按系统指引判断后续意义、查证来源并整理。"
            "在reason或note简要说明保存、忽略或暂缓的实际理由；不以写入数量为目标。"
            "先判断本组实际值得保留的记忆单元，再把共享同一计划或事件主线的候选归到同一动作；"
            "同一人物、同一天或同一会话不等于同一经历。"
            "可核对但没有后续意义的短时状态用IGNORE，不混进其它记忆；"
            "有意义的单次经历仍可保留。CREATE_NEW的reason请说明未来在什么情境下有用，"
            "不要只说有记录价值。"
            "每个准备写入的候选先SEARCH自身主题，可用queries批量查询。"
            "相关旧Memory先MEMORY_READ view=full，核对版本和证据；同一事实优先强化或修订。\n"
            "因旧记忆已覆盖而IGNORE时，也要核对旧正文与候选原始消息及必要的回复上下文；"
            "旧文的人物、行动或时间无来源支持时先修订或暂缓，不用错误旧文作为去重依据。\n"
            "候选标题和建议类型只作检索线索，正文依据原始消息。原话含短答、指代、引用、问句、转折时，"
            "用MESSAGE_CONTEXT_READ查来源消息前后及回复目标；同时提供evidence_id、message_id，"
            "before默认8、after默认20，各最多30。candidate_ids指定这段上下文用于哪些候选，"
            "不填时关联拥有该Evidence的候选。从旧记忆证据补查时请明确candidate_ids。"
            "新读到的真实消息可以引用，执行器会为实际引用的新来源存证；未读取的消息不能引用。\n"
            "准备写入的每个候选至少用MESSAGE_CONTEXT_READ核对一个来源锚点的前后对话，再DECIDE。"
            "一段上下文适用于多个候选时，用candidate_ids一并关联。"
            "source_message_ids必须包括支撑正文关键对象和行动的原话，旧记忆正文不能替代来源。\n"
            "对照完整语境核对人物、时间、对象及限定语，正文与标题一致。"
            "缺上下文先补查；有价值且明确的核心可以单独保留，关键含义仍不明再DEFER。\n"
            "本组最多8次模型调用，调查及写前复核最多7次，为写后回读复核预留1次。"
            "每次只返回一个步骤对象的JSON数组。DECIDE.actions恰好覆盖本组全部候选，"
            "每项用candidate_ids数组，相关候选可以共同支持一项动作。"
            "写入前会再次核对正文和完整来源；目标旧记忆必须来自本轮检索且已读取。\n"
            "CREATE_NEW提供title/content/memory_kind/source_message_ids；计划的memory_kind用COMMITMENT，不用PLAN。"
            "REINFORCE提供memory_id/reason/source_message_ids；"
            "REVISE提供memory_id/title/content/memory_kind/change_reason/reason/source_message_ids；"
            "MERGE提供canonical_memory_id/source_memory_ids/reason/source_message_ids；"
            "RELATE提供source_memory_id/target_memory_id/relation_type/reason；"
            "IGNORE或DEFER提供note。不得填based_on_revision_id。"
            "source_message_ids列出直接支持正文的原始账号消息，以及理解问答、指代所必需的上下文消息；"
            "所有引用必须已保存或为本组实际读取。Bot提问和建议只作上下文，不能当作本人确认。\n"
            "最近正式记忆含过去24小时创建或更新的当前版本及Bot主动写入的内容，先核对避免重复。\n"
            f"最近正式记忆目录（正文和来源须读取核对）：{json.dumps(recent_directory, ensure_ascii=False, default=str)}\n"
            f"候选数：{len(group)}（上限 {self._batch_size}）\n候选：\n"
            + "\n\n".join(payloads)
        )
        decision_started_at = self.last_llm_calls
        actions = await self._agent_decide(
            group_context,
            tool_context,
            prompt,
            producer,
            trace=trace,
            candidate_contexts=contexts,
            known_memory_ids=tuple(str(item["memory_id"]) for item in recent_memories),
            max_steps=MAX_SLEEP_AGENT_STEPS - 1,
        )
        decision_steps = self.last_llm_calls - decision_started_at
        group_context = self._combine_candidate_contexts(
            tuple(contexts[candidate_id] for candidate_id in group)
        )
        actions = self._require_human_source_messages(
            actions, group_context, trace=trace
        )
        if trace:
            trace("group_decision", {"candidate_ids": group, "actions": actions})
        assigned: set[str] = set()
        normalized_actions: list[dict[str, object]] = []
        for action in actions:
            if not isinstance(action, Mapping):
                raise AgentProtocolError("每个分组动作必须是对象")
            action = dict(action)
            action_type = self._parse_action_type(action.get("action_type"))
            raw_ids = action.get("candidate_ids")
            if raw_ids is None and action_type in {
                CandidateActionType.DEFER,
                CandidateActionType.IGNORE,
            }:
                ids = [candidate_id for candidate_id in group if candidate_id not in assigned]
                action["candidate_ids"] = ids
            elif raw_ids is None and len(group) == 1:
                ids = [group[0]]
                action["candidate_ids"] = ids
            elif isinstance(raw_ids, list):
                ids = raw_ids
            else:
                raise AgentProtocolError("每个分组动作必须包含 candidate_ids 数组")
            if not ids or any(not isinstance(item, str) or item not in contexts or item in assigned for item in ids):
                raise AgentProtocolError("分组动作引用了未知或重复候选")
            assigned.update(ids)
            normalized_actions.append(action)
        if assigned != set(group):
            raise AgentProtocolError("分组决定没有覆盖全部候选")
        actions = tuple(normalized_actions)
        memory_action_types = {
            CandidateActionType.CREATE_NEW,
            CandidateActionType.REINFORCE,
            CandidateActionType.REVISE,
            CandidateActionType.MERGE,
            CandidateActionType.RELATE,
        }
        review_budget = MAX_SLEEP_AGENT_STEPS - decision_steps
        if review_budget < 1 and any(
            self._parse_action_type(action.get("action_type")) in memory_action_types
            for action in actions
        ):
            actions = tuple(
                {
                    **action,
                    "action_type": CandidateActionType.DEFER.value,
                    "note": "本组没有剩余模型调用预算完成写入后复核，暂缓写入",
                }
                if self._parse_action_type(action.get("action_type"))
                in memory_action_types
                else action
                for action in actions
            )
        changed: list[str] = []
        post_action_results: list[dict[str, object]] = []
        post_action_contexts: dict[str, _CandidateContext] = {}
        post_action_ids: dict[str, set[str]] = {}
        candidates_to_resolve: set[str] = set()
        for index, action in enumerate(actions):
            ids = tuple(action["candidate_ids"])
            action_type = self._parse_action_type(action.get("action_type"))
            if action_type in (CandidateActionType.REVISE, CandidateActionType.REINFORCE) and not self._optional_text(action.get("memory_id")):
                action = {**action, "action_type": "DEFER", "note": "模型未指定可核实的目标记忆 ID"}
                action_type = CandidateActionType.DEFER
            if action_type is CandidateActionType.DEFER and not self._optional_text(action.get("note")):
                action = {**action, "note": "当前证据不足以安全整理"}
            combined = self._combine_candidate_contexts(
                tuple(contexts[candidate_id] for candidate_id in ids)
            )
            if action_type in memory_action_types:
                combined = await self._save_action_sources(action, combined, trace)
                contexts[ids[0]] = replace(contexts[ids[0]], evidence_ids=combined.evidence_ids)
            operation_key = self._operation_key(session_id, ids[0], index, action)
            if trace:
                trace("action_start", {"candidate_ids": ids, "action": action})
            target_ids = await self._execute_intent(
                session_id, ids[0], action, tool_context, action_index=index,
                operation_key=operation_key, candidate_context=combined,
            )
            async with self._schema.database.session() as session:
                operation = await session.get(SleepActionOperationModel, operation_key)
                if operation is None or operation.result_json is None:
                    raise ValueError("分组动作缺少已提交的操作结果")
                result = operation.result_json
            targets = tuple(
                (str(item[0]), CandidateActionTargetRole(item[1]))
                for item in result["targets"]
            )
            for candidate_id in ids[1:]:
                await self._sessions.record_action(CandidateActionInput(
                    candidate_id=candidate_id, sleep_session_id=session_id,
                    action_type=action_type, note=self._optional_text(action.get("note")),
                    result_revision_id=result.get("revision_id"), targets=targets,
                ))
            reread = await self._reread_action_result(
                action_type, combined, target_ids, tool_context, trace
            )
            if action_type in memory_action_types:
                post_action_results.append(
                    {
                        "candidate_ids": ids,
                        "action": action,
                        "readback": reread,
                    }
                )
                for candidate_id in ids:
                    post_action_contexts[candidate_id] = replace(
                        contexts[candidate_id],
                        evidence_ids=combined.evidence_ids,
                        source_snapshots=combined.source_snapshots,
                    )
                if action_type in {
                    CandidateActionType.CREATE_NEW,
                    CandidateActionType.REINFORCE,
                    CandidateActionType.REVISE,
                    CandidateActionType.MERGE,
                }:
                    active_target_ids = {
                        str(item["memory_id"])
                        for item in reread
                        if isinstance(item.get("memory_id"), str)
                        and isinstance(item.get("memory"), Mapping)
                        and item["memory"].get("status") == MemoryStatus.ACTIVE.value
                    }
                    for candidate_id in ids:
                        post_action_ids.setdefault(candidate_id, set()).update(
                            active_target_ids
                        )
            if action_type not in (CandidateActionType.IGNORE, CandidateActionType.DEFER):
                candidates_to_resolve.update(ids)
            changed.extend(target_ids)
            if trace:
                trace("action_end", {"candidate_ids": ids, "action_type": action_type.value, "target_memory_ids": target_ids})
        corrected_candidates: set[str] = set()
        review_deferred_candidates: set[str] = set()
        if post_action_results:
            review_actions = await self._review_action_results(
                post_action_contexts,
                post_action_ids,
                post_action_results,
                producer,
                tool_context,
                decision_steps + 1,
                trace,
            )
            if review_actions is None:
                review_deferred_candidates.update(post_action_contexts)
                for candidate_id in sorted(review_deferred_candidates):
                    await self._sessions.record_action(
                        CandidateActionInput(
                            candidate_id=candidate_id,
                            sleep_session_id=session_id,
                            action_type=CandidateActionType.DEFER,
                            note="正式记忆已写入，但写后复核失败，需后续复核",
                        )
                    )
                if trace:
                    trace(
                        "post_action_review_deferred",
                        {
                            "candidate_ids": tuple(sorted(review_deferred_candidates)),
                            "committed_memory_ids": tuple(dict.fromkeys(changed)),
                        },
                    )
                review_actions = ()
            for correction_index, correction in enumerate(review_actions):
                candidate_id = correction["candidate_ids"][0]
                memory_id = self._required_text(
                    correction.get("memory_id"), "REVISE memory_id"
                )
                read_memory = next(
                    item["memory"]
                    for result in post_action_results
                    for item in result["readback"]
                    if item.get("memory_id") == memory_id
                    and isinstance(item.get("memory"), Mapping)
                )
                current_revision = read_memory.get("current_revision")
                revision_id = (
                    current_revision.get("revision_id")
                    if isinstance(current_revision, Mapping)
                    else None
                )
                if not isinstance(revision_id, str) or not revision_id:
                    if trace:
                        trace(
                            "post_action_correction_rejected",
                            {
                                "candidate_id": candidate_id,
                                "memory_id": memory_id,
                                "reason": "回读结果没有当前 Revision ID",
                                "committed_memory_ids": tuple(sorted(post_action_ids.get(candidate_id, ()))),
                            },
                        )
                    continue
                correction = {
                    **correction,
                    "based_on_revision_id": revision_id,
                }
                plan_key = self._plan_key(
                    session_id, candidate_id, (correction,)
                )
                operation_key = self._operation_key(
                    session_id,
                    candidate_id,
                    MAX_CANDIDATES_PER_GROUP + correction_index + 1,
                    correction,
                )
                post_action_contexts[candidate_id] = await self._save_action_sources(
                    correction, post_action_contexts[candidate_id], trace,
                )
                await self._sessions.prepare_action_plan(
                    candidate_id,
                    session_id,
                    plan_key,
                    ((operation_key, CandidateActionType.REVISE, correction),),
                )
                correction_targets = await self._execute_intent(
                    session_id,
                    candidate_id,
                    correction,
                    tool_context,
                    action_index=0,
                    operation_key=operation_key,
                    plan_key=plan_key,
                    expected_action_type=CandidateActionType.REVISE,
                    candidate_context=post_action_contexts[candidate_id],
                )
                await self._sessions.advance_action_plan(
                    plan_key, 0, correction_targets
                )
                changed.extend(
                    await self._sessions.finalize_action_plan(plan_key)
                )
                corrected_candidates.add(candidate_id)
                await self._reread_action_result(
                    CandidateActionType.REVISE,
                    post_action_contexts[candidate_id],
                    correction_targets,
                    tool_context,
                    trace,
                )
                if trace:
                    trace(
                        "post_action_correction_applied",
                        {
                            "candidate_id": candidate_id,
                            "memory_id": memory_id,
                            "plan_key": plan_key,
                            "operation_key": operation_key,
                            "target_memory_ids": correction_targets,
                        },
                    )
        for candidate_id in sorted(
            candidates_to_resolve - corrected_candidates - review_deferred_candidates
        ):
            await self._sessions.resolve_candidate(candidate_id, session_id)
        return tuple(dict.fromkeys(changed))

    async def _save_action_sources(
        self,
        action: Mapping[str, object],
        context: _CandidateContext,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> _CandidateContext:
        """在首次动作及写后纠正前保存实际引用的来源和精确回复目标。"""
        source_ids = set(action.get("source_message_ids") or ())
        if not source_ids:
            return context
        selected = {
            (str(source["stream_id"]), str(source["message_id"])): source
            for source in context.source_snapshots if source.get("message_id") in source_ids
        }
        reply_ids = {
            (str(source["stream_id"]), str(source["reply_to"]))
            for source in selected.values() if source.get("reply_to")
        }
        for source in context.source_snapshots:
            key = (str(source.get("stream_id")), str(source.get("message_id")))
            if key in reply_ids:
                selected.setdefault(key, source)
        extra_ids = await CandidateService(self._schema).attach_context_evidence(
            context.candidate_id,
            tuple(EvidenceMessageInput(stream_id=stream, message_id=message, snapshot=source)
                  for (stream, message), source in selected.items()),
        )
        if extra_ids and trace:
            trace("context_sources_saved", {
                "candidate_id": context.candidate_id, "evidence_ids": extra_ids,
                "message_ids": tuple(message for _, message in selected),
            })
        return replace(context, evidence_ids=tuple(dict.fromkeys((*context.evidence_ids, *extra_ids))))

    async def _review_action_results(
        self,
        candidate_contexts: Mapping[str, _CandidateContext],
        correctable_memory_ids: Mapping[str, set[str]],
        action_results: list[dict[str, object]],
        producer: AgentStepProducer,
        tool_context: ToolContext,
        step_number: int,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> tuple[dict[str, object], ...] | None:
        """Ask the Agent to inspect all group write results once."""
        candidate_ids = tuple(candidate_contexts)
        committed_memory_ids = tuple(
            dict.fromkeys(
                str(item["memory_id"])
                for result in action_results
                for item in result.get("readback", ())
                if isinstance(item, Mapping)
                and isinstance(item.get("memory_id"), str)
            )
        )
        current_results = [
            {
                "candidate_ids": result["candidate_ids"],
                "readback": [
                    {
                        "memory_id": item["memory_id"],
                        "evidence": item["evidence"],
                        "memory": {
                            key: value for key, value in item["memory"].items()
                            if key in {
                                "memory_id", "status", "merged_into", "current_revision",
                                "current_subject", "current_participants",
                            }
                        },
                    }
                    for item in result.get("readback", ())
                ],
            }
            for result in action_results
        ]
        review_payload = (
            f"Agent step: {step_number}/{MAX_SLEEP_AGENT_STEPS} POST_ACTION_REVIEW\n"
            "检查本组刚执行的正式记忆，下面提供当前版本和实际保存的原始 Evidence；"
            "先独立阅读原始消息，分清被引用内容与本人新增回答，再逐项核对当前标题和正文。"
            "核对的是原话是否实际表达了这层意思，不是有没有附上消息ID。不能因为已有写入或先前决定就认定正确。"
            "检查表述是否增加了原话的确定性或具体程度；有歧义或需要修订时，"
            "在reason或note中引用足以说明判断的简短原话。"
            "特别核对因果词连接的各项事实：相邻陈述不能自动组成原因与结果；"
            "若本人没有明确说出该联系，用并列陈述或省去无关细节。"
            "发现实际内容错误、遗漏、不受来源支持的断言，"
            "或无关细节明显遮蔽记忆核心时，用REVISE纠正。"
            "核对回读 Memory 的 evidence 中实际保存的消息能否支撑正文。"
            "若支撑正文所必需的原话尚未存证，用 REVISE 补齐准确 source_message_ids；"
            "现有存证足够且内容一致时无需修订。"
            "输出一个标准 DECIDE，actions 恰好覆盖下方 review_candidate_ids。"
            "每个动作都必须使用 candidate_ids 数组字段，即使只关联一个候选也必须写数组；"
            "禁止使用单数 candidate_id 字段或省略候选字段。"
            '格式示例：[{"step_type":"DECIDE","actions":['
            '{"action_type":"IGNORE","candidate_ids":["<本轮真实候选ID>"],"note":"无需纠正"}]}]。'
            "不需要纠正的候选用 IGNORE（这里只表示无需后续纠正，不会新增 CandidateAction）；"
            "需要纠正时仅用 REVISE，且每个 REVISE 的 candidate_ids 数组长度必须为 1；"
            "memory_id 必须是该候选下方回读的 ACTIVE 目标，"
            "source_message_ids 必须精确引用下方该候选直接支持正文的原始账号消息。"
            "不创建新记忆，不强化，不合并，不关联，不延伸新事实；不要填写 based_on_revision_id。"
            "REVISE 必须提供 reason、title、content、memory_kind、change_reason 和 source_message_ids。"
            "如果回读与来源一致，全部使用 IGNORE。\n"
            f"review_candidate_ids: {candidate_ids!r}\n"
            "候选关联的原始来源及补查上下文（不以候选摘要作为证据）：\n"
            + "\n\n".join(
                json.dumps({
                    "candidate_id": candidate_id,
                    "original_sources": self._source_payload(
                        candidate_contexts[candidate_id].source_snapshots
                    ),
                }, ensure_ascii=False, default=str)
                for candidate_id in candidate_ids
            )
            + "\n\n本组当前记忆与已存来源：\n"
            + json.dumps(current_results, ensure_ascii=False, default=str)
        )
        if trace:
            trace(
                "post_action_review_input",
                {"step": step_number, "candidate_ids": candidate_ids, "prompt": review_payload},
            )
        self.last_llm_calls += 1
        try:
            steps = await self._produce_step(producer, review_payload)
            if trace:
                trace(
                    "post_action_review_output",
                    {
                        "step": step_number,
                        "candidate_ids": candidate_ids,
                        "steps": steps,
                    },
                )
            if len(steps) == 1 and isinstance(steps[0], Mapping):
                step = steps[0]
                step_type = self._optional_text(
                    step.get("step_type") or step.get("action_type")
                )
                if step_type is not None and step_type.upper() == "DECIDE":
                    raw_actions = step.get("actions")
                    if not isinstance(raw_actions, (list, tuple)):
                        raise AgentProtocolError("执行后 DECIDE.actions 必须是数组")
                    actions = tuple(raw_actions)
                elif self._is_final_action(step):
                    actions = steps
                else:
                    raise AgentProtocolError(
                        "执行后复核必须直接返回 DECIDE，不得请求未预算的额外工具步骤"
                    )
            elif steps and all(self._is_final_action(item) for item in steps):
                actions = steps
            else:
                raise AgentProtocolError("执行后复核未返回合法 DECIDE")
            normalized_actions, validation_error = self._validate_decision_actions(
                actions, candidate_contexts
            )
            if validation_error:
                raise AgentProtocolError(validation_error)
            corrections: list[dict[str, object]] = []
            selected_targets: set[str] = set()
            for action in normalized_actions:
                action_type = self._parse_action_type(action.get("action_type"))
                if action_type is CandidateActionType.IGNORE:
                    continue
                if action_type is not CandidateActionType.REVISE:
                    raise AgentProtocolError(
                        "执行后复核只允许 REVISE 或 IGNORE"
                    )
                action_candidate_ids = tuple(action["candidate_ids"])
                if len(action_candidate_ids) != 1:
                    raise AgentProtocolError(
                        "执行后 REVISE 必须只关联一个 Candidate"
                    )
                candidate_id = action_candidate_ids[0]
                memory_id = self._required_text(
                    action.get("memory_id"), "REVISE memory_id"
                )
                if memory_id not in correctable_memory_ids.get(candidate_id, set()):
                    raise AgentProtocolError(
                        "执行后 REVISE 目标不是该候选刚回读的 ACTIVE Memory"
                    )
                if memory_id in selected_targets:
                    raise AgentProtocolError(
                        "执行后复核不能对同一 Memory 提交多条并行 REVISE"
                    )
                selected_targets.add(memory_id)
                corrections.append(action)
            source_checked: list[dict[str, object]] = []
            for action in corrections:
                candidate_id = str(action["candidate_ids"][0])
                checked = self._require_human_source_messages(
                    (action,), candidate_contexts[candidate_id], trace=trace
                )[0]
                source_checked.append(checked)
            corrections = source_checked
            rejected = tuple(
                action
                for action in corrections
                if self._parse_action_type(action.get("action_type"))
                is not CandidateActionType.REVISE
            )
            if rejected:
                raise AgentProtocolError(
                    "执行后 REVISE 缺少可核对的直接账号来源"
                )
            if trace:
                trace(
                    "post_action_review_complete",
                    {
                        "candidate_ids": candidate_ids,
                        "committed_memory_ids": committed_memory_ids,
                        "correction_count": len(corrections),
                    },
                )
            return tuple(corrections)
        except Exception as error:
            if trace:
                trace(
                    "post_action_review_failed",
                    {
                        "candidate_ids": candidate_ids,
                        "committed_memory_ids": committed_memory_ids,
                        "writes_committed": True,
                        "needs_follow_up_review": True,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            return None

    async def _recent_memories(
        self, person_ids: tuple[str, ...]
    ) -> tuple[dict[str, object], ...]:
        """读取最近一天创建或更新的正式记忆当前正文。"""
        async with self._schema.database.session() as session:
            rows = (await session.execute(
                select(MemoryModel, MemoryRevisionModel, MemoryRevisionSubjectModel)
                .join(MemoryRevisionModel, MemoryRevisionModel.revision_id == MemoryModel.current_revision_id)
                .join(MemoryRevisionSubjectModel, MemoryRevisionSubjectModel.revision_id == MemoryModel.current_revision_id)
                .where(
                    MemoryModel.status == MemoryStatus.ACTIVE,
                    MemoryModel.updated_at >= datetime.now(UTC) - timedelta(days=1),
                )
                .order_by(MemoryModel.updated_at.desc(), MemoryModel.memory_id)
            )).all()
            revision_ids = tuple(revision.revision_id for _, revision, _ in rows)
            participants_by_revision: dict[str, set[str]] = {}
            if revision_ids and person_ids:
                participant_rows = await session.execute(
                    select(
                        MemoryRevisionParticipantModel.revision_id,
                        MemoryRevisionParticipantModel.person_id,
                    ).where(
                        MemoryRevisionParticipantModel.revision_id.in_(revision_ids),
                        MemoryRevisionParticipantModel.participant_kind == ParticipantKind.PERSON,
                        MemoryRevisionParticipantModel.person_id.in_(person_ids),
                        MemoryRevisionParticipantModel.person_id.is_not(None),
                    )
                )
                for revision_id, person_id in participant_rows:
                    if isinstance(person_id, str) and person_id:
                        participants_by_revision.setdefault(revision_id, set()).add(person_id)
        return tuple({
            "memory_id": memory.memory_id,
            "revision_id": revision.revision_id,
            "created_at_local": self._local_datetime_text(memory.created_at),
            "updated_at_local": self._local_datetime_text(memory.updated_at),
            "created_by_type": memory.created_by_type.value,
            "title": revision.title,
            "content": revision.content,
            "memory_kind": revision.memory_kind.value,
            "subject_kind": subject.subject_kind.value,
            "subject_key": subject.subject_key,
            "person_id": subject.person_id,
            "participant_person_ids": tuple(
                sorted(participants_by_revision.get(revision.revision_id, ()))
            ),
        } for memory, revision, subject in rows)

    @staticmethod
    def _combine_candidate_contexts(
        contexts: tuple[_CandidateContext, ...],
    ) -> _CandidateContext:
        """合并同一动作候选的来源、人物和正文。"""
        if not contexts:
            raise ValueError("Candidate 上下文不能为空")
        first = contexts[0]
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for context in contexts
                for evidence_id in context.evidence_ids
            )
        )
        participants = tuple(
            dict.fromkeys(
                participant
                for context in contexts
                for participant in context.participants
            )
        )
        source_snapshots: dict[tuple[str, str], dict[str, object]] = {}
        identity_aliases: list[str] = []
        for context in contexts:
            for snapshot in context.source_snapshots:
                message_key = (
                    str(snapshot.get("stream_id") or ""),
                    str(snapshot.get("message_id") or ""),
                )
                source_snapshots.setdefault(message_key, snapshot)
            for alias in context.identity_aliases:
                if alias and alias not in identity_aliases:
                    identity_aliases.append(alias)
        return replace(
            first,
            rough_title=" / ".join(dict.fromkeys(context.rough_title for context in contexts)),
            rough_content="\n".join(
                f"[{SleepAgentOrchestrator._local_datetime_text(context.observed_at)}] "
                f"{context.rough_content}"
                for context in contexts
            ),
            evidence_ids=evidence_ids,
            source_snapshots=tuple(source_snapshots.values()),
            identity_aliases=tuple(identity_aliases),
            participants=participants,
            proposed_kind=first.proposed_kind,
        )

    async def _reread_action_result(
        self,
        action_type: CandidateActionType,
        context: _CandidateContext,
        target_memory_ids: tuple[str, ...],
        tool_context: ToolContext,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> tuple[dict[str, object], ...]:
        """动作提交后重读正式记忆及其长期原始来源。"""
        results: list[dict[str, object]] = []
        if target_memory_ids:
            for memory_id in target_memory_ids:
                memory = await self._tools.memory_read(memory_id, "full", tool_context)
                evidence_summary = memory.get("evidence_summary", ())
                evidence_ids = tuple(
                    item["evidence_id"]
                    for item in evidence_summary
                    if isinstance(item, Mapping)
                    and isinstance(item.get("evidence_id"), str)
                )
                sources = (
                    await self._tools.evidence_read(evidence_ids, tool_context)
                    if evidence_ids
                    else ()
                )
                results.append(
                    {
                        "memory_id": memory_id,
                        "memory": memory,
                        "evidence": sources,
                    }
                )
                if trace:
                    trace(
                        "self_check",
                        {
                            "memory_id": memory_id,
                            "memory": memory,
                            "evidence": sources,
                        },
                    )
        elif action_type in {CandidateActionType.DEFER, CandidateActionType.IGNORE}:
            sources = (
                await self._tools.evidence_read(context.evidence_ids, tool_context)
                if context.evidence_ids
                else ()
            )
            results.append(
                {
                    "candidate_evidence_ids": context.evidence_ids,
                    "evidence": sources,
                }
            )
            if trace:
                trace(
                    "self_check",
                    {
                        "candidate_evidence_ids": context.evidence_ids,
                        "evidence": sources,
                    },
                )
        return tuple(results)

    async def actionable_count(self) -> int:
        """统计可由下一次 Sleep 认领的 PENDING 与 DEFERRED 候选。"""
        statement = (
            select(func.count())
            .select_from(CandidateModel)
            .where(
                CandidateModel.status.in_(
                    (CandidateStatus.PENDING, CandidateStatus.DEFERRED)
                )
            )
        )
        if self._automatic_since is not None:
            statement = statement.where(
                CandidateModel.created_at >= self._automatic_since
            )
        async with self._schema.database.session() as session:
            return int(await session.scalar(statement))

    @staticmethod
    def _short_log_id(value: object) -> str:
        """将内部候选、记忆或会话 ID 缩短用于运行日志。"""
        if not isinstance(value, str) or not value.strip():
            return "-"
        return value.strip()[:8]

    @classmethod
    def _short_log_ids(cls, values: object, *, limit: int = 4) -> str:
        """格式化有限数量的内部 ID，避免日志过长。"""
        if not isinstance(values, (list, tuple)):
            return "-"
        items = tuple(
            cls._short_log_id(value)
            for value in values
            if isinstance(value, str) and value.strip()
        )
        if not items:
            return "-"
        suffix = ",…" if len(items) > limit else ""
        return ",".join(items[:limit]) + suffix

    @staticmethod
    def _log_action_label(value: object) -> str:
        """把动作类型转换为面向运行日志的中文标签。"""
        labels = {
            "CREATE_NEW": "新建",
            "REINFORCE": "补强",
            "REVISE": "修订",
            "MERGE": "合并",
            "RELATE": "关联",
            "DEFER": "暂缓",
            "IGNORE": "忽略",
        }
        action_type = value.value if isinstance(value, CandidateActionType) else value
        return labels.get(str(action_type).upper(), "未知动作")

    @classmethod
    def _log_trace_event(cls, event: str, payload: dict[str, object]) -> None:
        """只将脱敏的 Sleep 事件摘要写入 INFO 日志。"""
        prefix = "[bold cyan]Sleep[/bold cyan]"
        if event == "model_input":
            step = payload.get("step")
            logger.info(
                f"{prefix} [blue]第 {step}/8 步[/blue] 请求模型",
                event="sleep_step_start",
                step=step if isinstance(step, int) else None,
            )
            return
        if event == "model_output":
            step = payload.get("step")
            logger.info(
                f"{prefix} [blue]第 {step}/8 步[/blue] 收到模型响应",
                event="sleep_step_response",
                step=step if isinstance(step, int) else None,
            )
            return
        if event == "SEARCH":
            results = payload.get("results", payload.get("result"))
            memory_ids = cls._search_memory_ids(results)
            candidate_ids = payload.get("candidate_ids")
            logger.info(
                f"{prefix} [blue]检索记忆[/blue] 候选={cls._short_log_ids(candidate_ids)} "
                f"命中={len(memory_ids)} ID={cls._short_log_ids(memory_ids)}",
                event="sleep_search",
                candidate_count=len(candidate_ids) if isinstance(candidate_ids, (list, tuple)) else 0,
                result_count=len(memory_ids),
            )
            return
        if event == "MEMORY_READ":
            memory_id = payload.get("memory_id")
            request = payload.get("request")
            if memory_id is None and isinstance(request, Mapping):
                memory_id = request.get("memory_id")
            logger.info(
                f"{prefix} [blue]读取记忆全文与历史[/blue] ID={cls._short_log_id(memory_id)}",
                event="sleep_memory_read",
                memory_id_short=cls._short_log_id(memory_id),
            )
            return
        if event == "EVIDENCE_READ":
            evidence_ids = payload.get("evidence_ids")
            records = payload.get("results", payload.get("result"))
            record_count = len(records) if isinstance(records, (list, tuple)) else 0
            message_count = sum(
                len(record.get("messages", ()))
                for record in records
                if isinstance(record, Mapping)
                and isinstance(record.get("messages", ()), (list, tuple))
            ) if isinstance(records, (list, tuple)) else 0
            evidence_count = (
                len(evidence_ids) if isinstance(evidence_ids, (list, tuple)) else record_count
            )
            if evidence_count or record_count:
                logger.info(
                    f"{prefix} [blue]读取原始证据[/blue] 证据={evidence_count} 消息快照={message_count}",
                    event="sleep_evidence_read",
                    evidence_count=evidence_count,
                    record_count=record_count,
                    message_count=message_count,
                )
            return
        if event == "MESSAGE_CONTEXT_READ":
            result = payload.get("result")
            messages = result.get("messages", ()) if isinstance(result, Mapping) else ()
            count = len(messages) if isinstance(messages, (list, tuple)) else 0
            logger.info(
                f"{prefix} [blue]读取来源前后对话与回复目标[/blue] 消息={count}",
                event="sleep_message_context_read", message_count=count,
            )
            return
        if event == "context_sources_saved":
            message_ids = payload.get("message_ids", ())
            count = len(message_ids) if isinstance(message_ids, (list, tuple)) else 0
            logger.info(
                f"{prefix} [green]保存本次引用的上下文证据[/green] 关联消息={count}",
                event="sleep_context_sources_saved", message_count=count,
            )
            return
        if event == "PERSON_LOOKUP":
            logger.info(f"{prefix} [blue]读取人物资料[/blue]（标识已隐藏）", event="sleep_person_lookup")
            return
        if event == "recent_formal_memories":
            memories = payload.get("memories")
            count = len(memories) if isinstance(memories, (list, tuple)) else 0
            if count:
                logger.info(
                    f"{prefix} [blue]预读近期正式记忆[/blue] 数量={count}",
                    event="sleep_recent_memories",
                    memory_count=count,
                )
            return
        if event == "group_decision":
            raw_actions = payload.get("actions")
            actions = raw_actions if isinstance(raw_actions, (list, tuple)) else ()
            summary = "；".join(cls._format_log_action(action) for action in actions)
            logger.info(
                f"{prefix} [bold magenta]最终决定[/bold magenta] {summary or '无动作'}",
                event="sleep_group_decision",
                action_count=len(actions),
            )
            return
        if event == "action_start":
            logger.info(
                f"{prefix} [yellow]执行操作[/yellow] {cls._format_log_action(payload.get('action'))}",
                event="sleep_action_start",
            )
            return
        if event == "action_end":
            targets = payload.get("target_memory_ids")
            logger.info(
                f"{prefix} [green]操作完成[/green] {cls._log_action_label(payload.get('action_type'))} "
                f"记忆={cls._short_log_ids(targets)}",
                event="sleep_action_end",
                action_type=str(payload.get("action_type", "")),
                target_count=len(targets) if isinstance(targets, (list, tuple)) else 0,
            )
            return
        if event == "self_check":
            memory_id = payload.get("memory_id")
            evidence = payload.get("evidence")
            evidence_count = len(evidence) if isinstance(evidence, (list, tuple)) else 0
            if memory_id is not None:
                logger.info(
                    f"{prefix} [green]回读核对完成[/green] 记忆={cls._short_log_id(memory_id)} "
                    f"证据={evidence_count}",
                    event="sleep_self_check",
                    memory_id_short=cls._short_log_id(memory_id),
                    evidence_count=evidence_count,
                )
            elif payload.get("candidate_evidence_ids"):
                candidate_evidence_ids = payload.get("candidate_evidence_ids")
                candidate_evidence_count = (
                    len(candidate_evidence_ids)
                    if isinstance(candidate_evidence_ids, (list, tuple))
                    else 0
                )
                logger.info(
                    f"{prefix} [green]终态核对完成[/green] "
                    f"候选证据={candidate_evidence_count}",
                    event="sleep_candidate_self_check",
                    evidence_count=candidate_evidence_count,
                )
            return
        if event == "post_action_review_input":
            candidate_ids = payload.get("candidate_ids")
            candidate_count = (
                len(candidate_ids) if isinstance(candidate_ids, (list, tuple)) else 0
            )
            logger.info(
                f"{prefix} [blue]开始整组写后复核[/blue] 候选={candidate_count}",
                event="sleep_post_action_review_start",
                candidate_count=candidate_count,
            )
            return
        if event == "post_action_review_output":
            steps = payload.get("steps")
            logger.info(
                f"{prefix} [blue]收到整组写后复核结果[/blue] 步骤="
                f"{len(steps) if isinstance(steps, (list, tuple)) else 0}",
                event="sleep_post_action_review_output",
                step_count=len(steps) if isinstance(steps, (list, tuple)) else 0,
            )
            return
        if event == "post_action_review_complete":
            candidate_ids = payload.get("candidate_ids")
            committed_ids = payload.get("committed_memory_ids")
            candidate_count = (
                len(candidate_ids) if isinstance(candidate_ids, (list, tuple)) else 0
            )
            committed_count = (
                len(committed_ids) if isinstance(committed_ids, (list, tuple)) else 0
            )
            logger.info(
                f"{prefix} [green]整组写后复核完成[/green] 候选={candidate_count} "
                f"记忆={committed_count} 纠正={payload.get('correction_count', 0)}",
                event="sleep_post_action_review_complete",
                candidate_count=candidate_count,
                committed_memory_count=committed_count,
                correction_count=payload.get("correction_count", 0),
            )
            return
        if event == "post_action_review_failed":
            committed_ids = payload.get("committed_memory_ids")
            committed_count = (
                len(committed_ids) if isinstance(committed_ids, (list, tuple)) else 0
            )
            logger.warning(
                f"{prefix} [yellow]动作已提交，但整组写后复核失败[/yellow] "
                f"需后续复核记忆={committed_count} 错误={payload.get('error_type', '未知')}",
                event="sleep_post_action_review_failed",
                committed_memory_count=committed_count,
                error_type=str(payload.get("error_type", "")),
            )
            return
        if event == "post_action_review_deferred":
            candidate_ids = payload.get("candidate_ids")
            candidate_count = (
                len(candidate_ids) if isinstance(candidate_ids, (list, tuple)) else 0
            )
            committed_ids = payload.get("committed_memory_ids")
            committed_count = (
                len(committed_ids) if isinstance(committed_ids, (list, tuple)) else 0
            )
            logger.warning(
                f"{prefix} [yellow]写后复核未完成，候选已暂缓且记忆仍保留[/yellow] "
                f"候选={candidate_count} 已写入记忆={committed_count}",
                event="sleep_post_action_review_deferred",
                candidate_count=candidate_count,
                committed_memory_count=committed_count,
            )
            return
        if event == "post_action_correction_rejected":
            logger.warning(
                f"{prefix} [yellow]复核提出的纠正未通过来源或目标校验[/yellow]",
                event="sleep_post_action_correction_rejected",
            )
            return
        if event == "post_action_correction_applied":
            candidate_id = payload.get("candidate_id")
            memory_id = payload.get("memory_id")
            logger.info(
                f"{prefix} [green]已应用复核纠正[/green] 候选={cls._short_log_id(candidate_id)} "
                f"目标={cls._short_log_id(memory_id)}",
                event="sleep_post_action_correction_applied",
                candidate_id_short=cls._short_log_id(candidate_id),
                memory_id_short=cls._short_log_id(memory_id),
            )
            return
        if event == "decision_review_requested":
            logger.info(f"{prefix} [yellow]进入来源复核[/yellow]", event="sleep_source_review")
            return
        if event in {"SOURCE_ALREADY_COVERED", "source_already_covered"}:
            logger.info(
                f"{prefix} [yellow]来源已被现有记忆覆盖[/yellow]，要求模型复用、修订或暂缓",
                event="sleep_source_covered",
            )
            return
        if event in {
            "decision_deferred",
            "unsafe_action_deferred",
            "unverified_target_deferred",
        }:
            logger.info(f"{prefix} [yellow]候选暂缓[/yellow]，等待后续重新评估", event="sleep_deferred")
            return
        if event == "protocol_error":
            logger.info(
                f"{prefix} [yellow]步骤格式需修正[/yellow]，在本轮继续尝试：{payload.get('message', '')}",
                event="sleep_protocol_feedback",
            )
            return
        if event == "model_error":
            logger.warning(f"{prefix} [yellow]模型步骤失败[/yellow]，原始响应未写入普通日志", event="sleep_model_error")

    @classmethod
    def _format_log_action(cls, action: object) -> str:
        """仅格式化动作类型及内部候选、记忆短 ID。"""
        if not isinstance(action, Mapping):
            return "未知动作"
        action_type = action.get("action_type")
        candidate_ids = action.get("candidate_ids")
        target_ids: list[str] = []
        for key in (
            "memory_id",
            "canonical_memory_id",
            "source_memory_id",
            "target_memory_id",
        ):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                target_ids.append(value)
        source_ids = action.get("source_memory_ids")
        if isinstance(source_ids, (list, tuple)):
            target_ids.extend(
                value for value in source_ids if isinstance(value, str) and value.strip()
            )
        return (
            f"{cls._log_action_label(action_type)} 候选={cls._short_log_ids(candidate_ids)} "
            f"目标={cls._short_log_ids(target_ids)}"
        )

    async def candidate_payload(self, candidate_id: str) -> str:
        """把候选素材格式化为 Agent 步骤输入文本。"""
        context = await self._candidate_context(candidate_id)
        return self._candidate_context_payload(context)

    @staticmethod
    def _candidate_context_payload(context: _CandidateContext) -> str:
        """格式化候选线索、原始来源及可信的人物标识。"""
        participants = [
            {
                "participant_kind": item.participant_kind.value,
                "person_id": item.person_id,
                "label": item.label,
            }
            for item in context.participants
        ]
        return (
            f"candidate_id: {context.candidate_id}\n"
            f"rough_title: {context.rough_title}\n"
            f"proposed_kind: {context.proposed_kind.value if context.proposed_kind else None}\n"
            f"observed_at: {SleepAgentOrchestrator._local_datetime_text(context.observed_at)}\n"
            f"subject: {context.subject}\nparticipants: {participants}\n"
            f"evidence_ids: {context.evidence_ids}\n"
            f"original_sources: {SleepAgentOrchestrator._source_payload(context.source_snapshots)}\n"
            "保留原文中的陈述归属和语气，例如本人表示、他人转述、可能或尚未确认。"
        )

    @staticmethod
    def _snapshots_by_evidence(
        evidence: tuple[dict[str, object], ...],
    ) -> dict[str, tuple[dict[str, object], ...]]:
        """提取每份候选 Evidence 下可用的长期消息快照。"""
        result: dict[str, tuple[dict[str, object], ...]] = {}
        for record in evidence:
            evidence_id = record.get("evidence_id")
            messages = record.get("messages")
            if not isinstance(evidence_id, str) or not isinstance(messages, (list, tuple)):
                continue
            result[evidence_id] = tuple(
                dict(snapshot)
                for message in messages
                if isinstance(message, Mapping)
                and message.get("source_status") == "AVAILABLE"
                and isinstance((snapshot := message.get("snapshot")), Mapping)
            )
        return result

    @staticmethod
    def _source_payload(
        snapshots: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        """格式化快照，明确标识 Bot 与可核对的原始消息正文。"""
        payload: list[dict[str, object]] = []
        for snapshot in snapshots:
            raw_text = str(snapshot.get("processed_plain_text") or snapshot.get("content") or "")
            own_text, reply_preview = split_reply_preview(
                raw_text,
                snapshot.get("reply_to"),
            )
            person_id = SleepAgentOrchestrator._optional_text(snapshot.get("person_id"))
            speaker_kind = (
                "BOT"
                if snapshot.get("speaker_is_bot") is True
                or str(snapshot.get("sender_role") or "").casefold() == "bot"
                or str(snapshot.get("speaker_kind") or "").upper() == "BOT"
                or (person_id and person_id.casefold() == "bot")
                else "ACCOUNT"
                if person_id
                else "UNKNOWN"
            )
            payload.append(
                {
                    "message_id": snapshot.get("message_id"),
                    "evidence_id": snapshot.get("evidence_id"),
                    "stream_id": snapshot.get("stream_id"),
                    "time": snapshot.get("time"),
                    "message_time_local": SleepAgentOrchestrator._local_message_time(
                        snapshot.get("time")
                    ),
                    "speaker_kind": speaker_kind,
                    "sender_role": snapshot.get("sender_role"),
                    "person_id": person_id,
                    "sender_name": snapshot.get("sender_name"),
                    "sender_cardname": snapshot.get("sender_cardname"),
                    "message_type": snapshot.get("message_type"),
                    "reply_to": snapshot.get("reply_to"),
                    "text": own_text,
                    **({"reply_preview_not_speaker_text": reply_preview} if reply_preview is not None else {}),
                    **({"reply_boundary_unresolved": True} if has_unseparated_reply_preview(raw_text, snapshot.get("reply_to")) else {}),
                }
            )
        return tuple(payload)

    @staticmethod
    def _local_datetime_text(value: datetime) -> str:
        """将带时区的观察时间格式化为本地时区并保留偏移。"""
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone().isoformat()

    @staticmethod
    def _local_message_time(value: object) -> str | None:
        """把快照中的 Unix UTC 或 ISO 时间转换为本地偏移显示。"""
        if isinstance(value, datetime):
            moment = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                moment = datetime.fromtimestamp(value, tz=UTC)
            except (OverflowError, OSError, ValueError):
                return None
        elif isinstance(value, str):
            try:
                moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone().isoformat()

    @staticmethod
    def _defer_group_actions(
        actions: tuple[dict[str, object], ...] | tuple[object, ...],
        note: str,
    ) -> tuple[dict[str, object], ...]:
        """把整组动作安全降级为 DEFER。"""
        return tuple(
            {
                **dict(action),
                "action_type": CandidateActionType.DEFER.value,
                "note": note,
            }
            for action in actions
            if isinstance(action, Mapping)
        )

    @staticmethod
    def _require_human_source_messages(
        actions: tuple[dict[str, object], ...],
        context: _CandidateContext,
        *,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> tuple[dict[str, object], ...]:
        """要求写入引用可定位账号的原话，并保留必要的上下文来源。"""
        snapshots_by_id: dict[str, list[dict[str, object]]] = {}
        for snapshot in context.source_snapshots:
            message_id = snapshot.get("message_id")
            if isinstance(message_id, str) and message_id:
                snapshots_by_id.setdefault(message_id, []).append(snapshot)
        result: list[dict[str, object]] = []
        for action in actions:
            action_type = SleepAgentOrchestrator._parse_action_type(
                action.get("action_type")
            )
            requires_source = action_type in {
                CandidateActionType.CREATE_NEW,
                CandidateActionType.REVISE,
                CandidateActionType.REINFORCE,
            } or (
                action_type is CandidateActionType.MERGE
                and isinstance(action.get("new_memory"), Mapping)
            )
            if not requires_source:
                result.append(action)
                continue
            raw_ids = action.get("source_message_ids")
            ids_are_well_formed = (
                isinstance(raw_ids, (list, tuple))
                and bool(raw_ids)
                and all(isinstance(item, str) and item for item in raw_ids)
            )
            source_ids = tuple(
                item for item in raw_ids if isinstance(item, str) and item
            ) if isinstance(raw_ids, (list, tuple)) else ()
            valid = ids_are_well_formed and len(set(source_ids)) == len(source_ids)
            selected: list[dict[str, object]] = []
            if valid:
                for message_id in source_ids:
                    matches = snapshots_by_id.get(message_id, ())
                    if len(matches) != 1:
                        valid = False
                        break
                    snapshot = matches[0]
                    selected.append(snapshot)
            if valid:
                valid = any(
                    isinstance(snapshot.get("person_id"), str)
                    and snapshot["person_id"].strip()
                    and snapshot["person_id"].casefold() != "bot"
                    and snapshot.get("speaker_is_bot") is not True
                    and str(snapshot.get("sender_role") or "").casefold() != "bot"
                    and str(snapshot.get("speaker_kind") or "").upper() != "BOT"
                    for snapshot in selected
                )
            if valid:
                result.append(action)
                continue
            deferred = {
                **action,
                "action_type": CandidateActionType.DEFER.value,
                "note": "缺少可核对的直接账号原始消息，暂缓写入",
            }
            result.append(deferred)
            if trace:
                trace(
                    "unsafe_action_deferred",
                    {
                        "action_type": action_type.value,
                        "source_message_ids": raw_ids,
                        "reason": "来源缺失、不唯一、未实际读取或没有可定位账号原话",
                    },
                )
        return tuple(result)

    @staticmethod
    def _trusted_subject(
        context: _CandidateContext,
        subject: SubjectInput,
    ) -> SubjectInput:
        """只保留原始来源可证明或能精确唯一映射的人物 ID。"""
        if subject.subject_kind is not SubjectKind.PERSON:
            return subject
        person_ids = {
            person_id
            for snapshot in context.source_snapshots
            if (person_id := SleepAgentOrchestrator._optional_text(snapshot.get("person_id")))
            and person_id.casefold() != "bot"
        }
        if subject.person_id in person_ids:
            return subject
        matching_person_ids = {
            person_id
            for snapshot in context.source_snapshots
            if (person_id := SleepAgentOrchestrator._optional_text(snapshot.get("person_id")))
            and person_id.casefold() != "bot"
            and any(
                alias in (snapshot.get("sender_name"), snapshot.get("sender_cardname"))
                for alias in context.identity_aliases
            )
        }
        if len(matching_person_ids) == 1:
            return SubjectInput(
                subject_kind=SubjectKind.PERSON,
                person_id=next(iter(matching_person_ids)),
                subject_label=subject.subject_label,
            )
        return SubjectInput(
            subject_kind=SubjectKind.UNKNOWN,
            subject_label=subject.subject_label,
        )

    @staticmethod
    def _trusted_participants(
        context: _CandidateContext,
        participants: tuple[ParticipantInput, ...],
    ) -> tuple[ParticipantInput, ...]:
        """移除不受原始消息身份元数据支持的人物参与者 ID。"""
        person_ids = {
            person_id
            for snapshot in context.source_snapshots
            if (person_id := SleepAgentOrchestrator._optional_text(snapshot.get("person_id")))
            and person_id.casefold() != "bot"
        }
        trusted: list[ParticipantInput] = []
        for participant in participants:
            if participant.participant_kind is not ParticipantKind.PERSON:
                trusted.append(participant)
                continue
            if participant.person_id in person_ids:
                trusted.append(participant)
                continue
            matching_person_ids = {
                person_id
                for snapshot in context.source_snapshots
                if (person_id := SleepAgentOrchestrator._optional_text(snapshot.get("person_id")))
                and person_id.casefold() != "bot"
                and any(
                    alias in (snapshot.get("sender_name"), snapshot.get("sender_cardname"))
                    for alias in (participant.person_id, participant.label)
                    if alias
                )
            }
            if len(matching_person_ids) == 1:
                trusted.append(
                    ParticipantInput(
                        participant_kind=ParticipantKind.PERSON,
                        person_id=next(iter(matching_person_ids)),
                        label=participant.label,
                    )
                )
            else:
                trusted.append(
                    ParticipantInput(
                        participant_kind=ParticipantKind.OTHER,
                        label=participant.label,
                    )
                )
        return tuple(trusted)

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
            )
            for row in participant_rows
        )
        identity_aliases = tuple(
            dict.fromkeys(
                value
                for value in (
                    subject.person_id,
                    subject.subject_key,
                    subject.subject_label,
                    *(
                        alias
                        for participant in participants
                        for alias in (participant.person_id, participant.label)
                    ),
                )
                if value
            )
        )
        return _CandidateContext(
            candidate_id=candidate.candidate_id,
            rough_title=candidate.rough_title,
            rough_content=candidate.rough_content,
            observed_at=candidate.observed_at,
            evidence_ids=evidence_ids,
            source_snapshots=(),
            identity_aliases=identity_aliases,
            subject=subject,
            participants=participants,
            proposed_kind=candidate.proposed_kind,
        )

    async def run_session(
        self,
        trigger_type: SleepTriggerType,
        step_producer: AgentStepProducer,
        trace: Callable[[str, dict[str, object]], None] | None = None,
        *,
        candidate_ids: tuple[str, ...] | None = None,
    ) -> SleepSessionStatus:
        """执行一次完整睡眠会话并返回最终状态。

        流程（§63）：认领批次 → 全批候选多步整理、
        动作执行及 self-check 重读 → 会话结束。全部成功 COMPLETED；
        存在失败候选 PARTIAL。
        """
        recovered_memory_ids = await self._resume_action_operations()
        await self._sessions.recover_interrupted_candidates(
            automatic_since=self._automatic_since
        )
        if candidate_ids is None:
            candidate_ids = await self.pending_candidates()
        return await self._run_candidate_batch(
            trigger_type,
            step_producer,
            candidate_ids,
            trace,
            recovered_memory_ids,
        )

    async def _resume_action_operations(self) -> tuple[str, ...]:
        """在新会话询问模型前续跑完整动作计划。"""
        changed_memory_ids: list[str] = []
        plans = await self._sessions.list_incomplete_plans()
        operations = await self._sessions.list_incomplete_operations()
        if self._automatic_since is not None:
            candidate_ids = {
                plan.candidate_id for plan in plans
            } | {
                operation.candidate_id for operation in operations
            }
            if candidate_ids:
                async with self._schema.database.session() as session:
                    eligible_candidate_ids = set(
                        (
                            await session.scalars(
                                select(CandidateModel.candidate_id).where(
                                    CandidateModel.candidate_id.in_(candidate_ids),
                                    CandidateModel.created_at >= self._automatic_since,
                                )
                            )
                        ).all()
                    )
                plans = tuple(
                    plan for plan in plans
                    if plan.candidate_id in eligible_candidate_ids
                )
                operations = tuple(
                    operation for operation in operations
                    if operation.candidate_id in eligible_candidate_ids
                )
        planned_operation_keys: set[str] = set()
        for plan in plans:
            intents = tuple(plan.intents_json)
            operation_keys = tuple(plan.operation_keys_json)
            planned_operation_keys.update(operation_keys)
            if len(intents) != plan.action_count or len(operation_keys) != plan.action_count:
                raise ValueError("Sleep Action Plan 内容与游标不一致")
            if any(
                isinstance(intent, dict)
                and intent.get("action_type") in ("REVISE", "REINFORCE")
                and not intent.get("based_on_revision_id")
                for intent in intents[plan.next_action_index:]
            ):
                await self._sessions.discard_unexecutable_plan(
                    plan.plan_key, "旧计划缺少 revision 基线，作废后重开"
                )
                continue
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
        for operation in operations:
            if operation.operation_key in planned_operation_keys:
                continue
            if operation.status != "COMPLETED":
                await self._sessions.discard_stale_operation(
                    operation.operation_key,
                    "旧候选已收口且动作未提交，作废孤立操作",
                )
                continue
            action_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"engram-vnext-sleep-action:{operation.operation_key}",
                )
            )
            async with self._schema.database.session() as session:
                candidate = await session.get(CandidateModel, operation.candidate_id)
                claim = await session.get(
                    SleepSessionCandidateModel,
                    (operation.sleep_session_id, operation.candidate_id),
                )
                action = await session.get(CandidateActionModel, action_id)
            claim_is_current = (
                candidate is not None
                and candidate.status is CandidateStatus.PROCESSING
                and candidate.processing_session_id == operation.sleep_session_id
                and claim is not None
                and claim.released_at is None
            )
            if action is not None and not claim_is_current:
                continue
            await self._sessions.record_operation_action(operation.operation_key)
            if claim_is_current:
                await self._sessions.finalize_recovered_operation(
                    operation.operation_key
                )
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
            candidate_contexts={candidate_id: candidate_context},
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
        *,
        trace: Callable[[str, dict[str, object]], None] | None = None,
        candidate_contexts: dict[str, _CandidateContext] | None = None,
        known_memory_ids: tuple[str, ...] = (),
        max_steps: int = MAX_SLEEP_AGENT_STEPS,
    ) -> tuple[dict[str, object], ...]:
        """在限定步数内调查候选并返回最终动作。"""
        if not 1 <= max_steps <= MAX_SLEEP_AGENT_STEPS:
            raise ValueError(
                f"max_steps 必须在 1 到 {MAX_SLEEP_AGENT_STEPS} 之间"
            )
        produce = getattr(step_producer, "produce", None)
        if not callable(produce):
            research = await self._research_candidate(
                candidate_context, tool_context, trace=trace
            )
            if research:
                payload = f"{payload}\n\nSleep research:\n{research}"
            if trace:
                trace("model_input", {"step": 1, "prompt": payload})
            self.last_llm_calls += 1
            try:
                steps = await self._produce_step(step_producer, payload)
            except Exception as error:
                if trace:
                    trace(
                        "model_error",
                        {
                            "step": 1,
                            "error": str(error),
                            "raw_response": getattr(step_producer, "last_raw_response", None),
                        },
                    )
                raise
            if trace:
                trace("model_output", {"step": 1, "steps": steps})
            if len(steps) == 1 and steps[0].get("step_type") == "DECIDE":
                actions = steps[0].get("actions")
                if not isinstance(actions, (list, tuple)) or not all(
                    self._is_final_action(item) for item in actions
                ):
                    raise AgentProtocolError("DECIDE.actions 包含非法动作")
                return tuple(actions)
            if not steps:
                return (
                    {
                        "action_type": CandidateActionType.DEFER.value,
                        "note": "Sleep Agent 未形成可核实的决定",
                    },
                )
            return steps

        from .runtime import LLMResponseFormatError

        observations: list[dict[str, object]] = []
        candidate_contexts = candidate_contexts or {
            candidate_context.candidate_id: candidate_context
        }
        read_memory_ids: set[str] = set()
        read_full_memory_ids: set[str] = set()
        read_memories: dict[str, dict[str, object]] = {}
        read_evidence_ids: set[str] = set()
        searched_memory_ids: set[str] = set(known_memory_ids)
        searched_candidate_ids: set[str] = set()
        context_read_candidate_ids: set[str] = set()
        review_requested = False
        review_actions: tuple[dict[str, object], ...] = ()

        def read_target_snapshots(memory_id: str) -> tuple[dict[str, object], ...]:
            """Return only original snapshots from a fully read target Memory."""
            if memory_id not in read_full_memory_ids:
                return ()
            memory = read_memories.get(memory_id)
            if not isinstance(memory, Mapping):
                return ()
            evidence_records = memory.get("evidence_metadata", ())
            if not isinstance(evidence_records, (list, tuple)):
                return ()
            snapshots_by_evidence = self._snapshots_by_evidence(
                tuple(
                    record
                    for record in evidence_records
                    if isinstance(record, Mapping)
                )
            )
            snapshots_by_key: dict[tuple[str, str], dict[str, object]] = {}
            for evidence_id, snapshots in snapshots_by_evidence.items():
                if evidence_id not in read_evidence_ids:
                    continue
                for snapshot in snapshots:
                    stream_id = snapshot.get("stream_id")
                    message_id = snapshot.get("message_id")
                    if (
                        not isinstance(stream_id, str)
                        or not stream_id
                        or not isinstance(message_id, str)
                        or not message_id
                        or snapshot.get("redacted") is True
                    ):
                        continue
                    snapshots_by_key.setdefault(
                        (stream_id, message_id),
                        {**snapshot, "evidence_id": evidence_id},
                    )
            return tuple(snapshots_by_key.values())

        def include_read_target_sources(actions: tuple[object, ...]) -> None:
            """Attach already-read target evidence only to final update actions."""
            for action in actions:
                if not isinstance(action, Mapping):
                    continue
                try:
                    action_type = self._parse_action_type(action.get("action_type"))
                except ValueError:
                    continue
                if action_type not in {
                    CandidateActionType.REVISE,
                    CandidateActionType.REINFORCE,
                    CandidateActionType.MERGE,
                }:
                    continue
                raw_candidate_ids = action.get("candidate_ids")
                if not isinstance(raw_candidate_ids, (list, tuple)):
                    continue
                target_sources_by_id: dict[str, dict[str, object]] = {}
                ambiguous_ids: set[str] = set()
                for memory_id in self._action_target_memory_ids((action,)):
                    memory = read_memories.get(memory_id)
                    if (
                        memory_id not in read_full_memory_ids
                        or not isinstance(memory, Mapping)
                        or memory.get("status") != MemoryStatus.ACTIVE.value
                    ):
                        continue
                    for snapshot in read_target_snapshots(memory_id):
                        message_id = snapshot["message_id"]
                        previous = target_sources_by_id.get(message_id)
                        if previous is not None and (
                            previous.get("stream_id") != snapshot.get("stream_id")
                        ):
                            target_sources_by_id.pop(message_id, None)
                            ambiguous_ids.add(message_id)
                        elif message_id not in ambiguous_ids:
                            target_sources_by_id.setdefault(message_id, snapshot)
                for candidate_id in raw_candidate_ids:
                    if not isinstance(candidate_id, str):
                        continue
                    context = candidate_contexts.get(candidate_id)
                    if context is None:
                        continue
                    existing_ids = {
                        str(snapshot.get("message_id"))
                        for snapshot in context.source_snapshots
                        if snapshot.get("message_id")
                    }
                    additions = tuple(
                        snapshot
                        for message_id, snapshot in target_sources_by_id.items()
                        if message_id not in existing_ids
                        and message_id not in ambiguous_ids
                    )
                    if additions:
                        candidate_contexts[candidate_id] = replace(
                            context,
                            source_snapshots=(*context.source_snapshots, *additions),
                        )

        async def read_search_hits(
            results: object, step_number: int, candidate_ids: tuple[str, ...] = ()
        ) -> None:
            """读取本轮最相关的旧记忆及来源，供写入前比较。"""
            remaining = max(0, 6 - len(read_full_memory_ids))
            related = tuple(candidate_contexts[item] for item in candidate_ids)
            people = {
                context.subject.person_id
                for context in related
                if context.subject.subject_kind is SubjectKind.PERSON
                and context.subject.person_id
                and context.subject.person_id.casefold() != "bot"
            }
            people.update(
                participant.person_id
                for context in related
                for participant in context.participants
                if participant.participant_kind is ParticipantKind.PERSON
                and participant.person_id
                and participant.person_id.casefold() != "bot"
            )
            subjects = {
                (context.subject.subject_kind.value, context.subject.subject_key)
                for context in related
                if context.subject.subject_kind is not SubjectKind.PERSON
                and context.subject.subject_key
            }
            hits = tuple(item for item in results if isinstance(item, Mapping)) if isinstance(
                results, (list, tuple)
            ) else ()
            preferred = tuple(
                str(item["memory_id"])
                for item in hits
                if isinstance(item.get("memory_id"), str)
                and isinstance(item.get("subject"), Mapping)
                and (
                    item["subject"].get("person_id") in people
                    or (
                        item["subject"].get("subject_kind"),
                        item["subject"].get("subject_key"),
                    ) in subjects
                )
            )
            memory_ids = tuple(
                memory_id
                for memory_id in dict.fromkeys(
                    preferred[:2] or self._search_memory_ids(results)[:1]
                )
                if memory_id not in read_full_memory_ids
            )[:remaining]
            if not memory_ids:
                return
            reads = await self._read_memories_with_evidence(
                memory_ids,
                tool_context,
                read_memory_ids=read_memory_ids,
                read_full_memory_ids=read_full_memory_ids,
                read_evidence_ids=read_evidence_ids,
                trace=trace,
            )
            read_memories.update(
                (str(item["memory_id"]), dict(item["memory"])) for item in reads
            )
            observations.append(
                {
                    "step": step_number,
                    "tool": "MEMORY_READ",
                    "candidate_ids": candidate_ids,
                    "result": tuple(self._compact_observation(item) for item in reads),
                }
            )

        if candidate_context.evidence_ids:
            candidate_sources = self._source_payload(candidate_context.source_snapshots)
            read_evidence_ids.update(candidate_context.evidence_ids)
            observations.append(
                {
                    "step": 0,
                    "tool": "EVIDENCE_READ",
                    "purpose": "核对候选引用的长期原始消息快照",
                    "result": self._compact_observation(candidate_sources),
                }
            )
        searched = False
        for step_number in range(1, max_steps + 1):
            transcript = repr(observations[-8:])
            context_requests = tuple(
                {
                    "step_type": "MESSAGE_CONTEXT_READ",
                    "candidate_ids": [key],
                    "evidence_id": item.source_snapshots[0].get("evidence_id") or item.evidence_ids[0],
                    "message_id": item.source_snapshots[0]["message_id"],
                }
                for key, item in candidate_contexts.items()
                if key not in context_read_candidate_ids and item.evidence_ids and item.source_snapshots
            )
            research_status = (
                "无价值或仍待核实可直接DECIDE。准备写入时，先完成未做的查证。"
                f"尚未SEARCH的候选：{tuple(key for key in candidate_contexts if key not in searched_candidate_ids)}。"
                "SEARCH使用queries:[{candidate_ids:[ID],query:该候选主题}]。"
                f"尚未核对上下文的准确请求：{json.dumps(context_requests, ensure_ascii=False)}。"
                "每次只选择其中一项步骤，不同时返回读取和DECIDE。\n"
            )
            step_payload = (
                f"{payload}\n\nAgent step: {step_number}/{max_steps}\n"
                f"{research_status}"
                "Choose one observation tool step or one final DECIDE. Before writing, read source context and search relevant history. "
                "Before using an existing Memory, read its full current view and source evidence.\n"
                f"Tool observations:\n{transcript}"
            )
            if review_requested:
                review_items: list[dict[str, object]] = []
                for action_index, action in enumerate(review_actions):
                    action_contexts = tuple(
                        candidate_contexts[item] for item in action["candidate_ids"]
                    )
                    sources = {
                        (str(item.get("stream_id")), str(item.get("message_id"))): item
                        for context in action_contexts
                        for item in self._source_payload(context.source_snapshots)
                    }
                    targets = []
                    target_ids = list(self._action_target_memory_ids((action,)))
                    if observations[-1].get("tool") in {
                        "SOURCE_ALREADY_COVERED", "SOURCE_COVERAGE_REVIEW_REQUIRED",
                    }:
                        covered_id = observations[-1]["existing_targets"].get(action_index)
                        if covered_id and covered_id not in target_ids:
                            target_ids.append(covered_id)
                    for memory_id in target_ids:
                        memory = read_memories.get(memory_id, {})
                        revision = memory.get("current_revision") or {}
                        if not isinstance(revision, Mapping):
                            revision = {}
                        targets.append({
                            "memory_id": memory_id,
                            "status": memory.get("status"),
                            "current_revision": revision,
                            "statement_time_local": self._local_message_time(
                                revision.get("observed_at")
                            ),
                            "original_messages": self._source_payload(
                                read_target_snapshots(memory_id)
                            ),
                        })
                    review_items.append({
                        "action_type": self._parse_action_type(
                            action.get("action_type")
                        ).value,
                        "candidate_ids": tuple(action["candidate_ids"]),
                        "target_memory_ids": tuple(target_ids),
                        "action_targets": {
                            key: action[key]
                            for key in (
                                "memory_id",
                                "canonical_memory_id",
                                "source_memory_id",
                                "target_memory_id",
                                "source_memory_ids",
                            )
                            if key in action
                        },
                        "original_messages": tuple(sources.values()),
                        "candidate_subjects": [context.subject for context in action_contexts],
                        "read_targets": targets,
                    })
                feedback = {
                    key: value for key, value in observations[-1].items()
                    if key not in {"source_messages", "draft_actions", "actions", "result"}
                }
                step_payload = (
                    f"Agent step: {step_number}/{max_steps}\nFINAL SOURCE REVIEW — SOURCE-FIRST FINALIZATION\n"
                    f"全部候选ID：{tuple(candidate_contexts)}\n"
                    f"逐动作原始消息、拟执行动作和目标：{review_items!r}\n"
                    f"最近反馈：{feedback!r}\n"
                    '严格返回[{"step_type":"DECIDE","actions":[...]}]；不要直接返回动作数组。'
                    "actions恰好覆盖全部候选。只依据逐动作提供的原始消息，从头生成最终正文；"
                    "本步骤不提供先前拟写的title、content或reason，不要从候选摘要或旧正文推导新事实。"
                    "保留candidate_ids和已核对的目标ID。若原判断是旧记忆已覆盖，"
                    "逐项核对旧正文与旧来源；有实质错误时用REVISE纠正，准确覆盖时才IGNORE。"
                    "其他拟写动作保留动作类型；若来源显示不应写入，可改为IGNORE或DEFER。"
                    "按每位说话者自己的原话分别归属信息。短答只确认其明确回应的范围；"
                    "另一位说话者提出的目的、理由、建议或计划不能自动成为被回应者本人的想法或计划，除非对方明确确认。"
                    "不补来源未支持的事实、归属、因果或时间关系；相邻陈述不能自动组成因果。"
                    "CREATE_NEW的source_message_ids只从该动作的original_messages中选择；"
                    "REVISE、REINFORCE或MERGE可引用本动作original_messages及其匹配目标的read_targets.original_messages。"
                    "read_targets中的正文只用于确定更新目标，事实须由其original_messages支持；未读目标或缺少原始来源时不可引用。"
                    "若更新需要保留的旧事实没有可核对的原始来源，请DEFER，不要静默丢弃或用旧正文代替来源。"
                    "横向比较全部动作：同一计划或事件主线由多条候选支持时可合成一条记忆，"
                    "仅同人同日不能合并；无关或没有后续意义的候选单独IGNORE，"
                    "IGNORE也算覆盖候选，不能为凑齐候选将其混入其它记忆。"
                    "关键上下文不足时先MESSAGE_CONTEXT_READ；旧文也按同样标准核对，错误用REVISE纠正。"
                    "保留有后续意义的核心及必要语境，包括有意义的情感和重要单次经历；"
                    "仅服务当前回合的内容用IGNORE，有价值但关键来源或含义仍待核实才DEFER。"
                    "已有记忆覆盖当前原始消息时，复用或修订，不另建重复记忆。"
                    "日期有助于理解状态或进度时，注明来源中的发言日期；"
                    "发言日期不能当作事件发生日期。"
                    "旧记忆缺少日期会造成实质时间歧义时，用REVISE澄清。"
                    "CREATE_NEW提供title/content/memory_kind；"
                    "REINFORCE提供memory_id/reason；REVISE提供memory_id/title/content/memory_kind/"
                    "change_reason（CLARIFICATION或CORRECTION）/reason；"
                    "MERGE提供canonical_memory_id/source_memory_ids/reason；"
                    "RELATE提供source_memory_id/target_memory_id/relation_type/reason。"
                    "所有写入动作提供精确source_message_ids；DEFER提供note。不要填写based_on_revision_id。"
                )
            if trace:
                trace("model_input", {"step": step_number, "prompt": step_payload})
            self.last_llm_calls += 1
            try:
                steps = await self._produce_step(step_producer, step_payload)
            except Exception as error:
                if trace:
                    trace(
                        "model_error",
                        {
                            "step": step_number,
                            "error": str(error),
                            "raw_response": getattr(step_producer, "last_raw_response", None),
                        },
                    )
                if isinstance(error, (AgentProtocolError, LLMResponseFormatError)):
                    self._append_protocol_error(
                        observations,
                        step_number,
                        f"模型输出格式无效：{error}。请返回完整且语法合法的JSON数组，"
                        '每次只有一个step；最终为[{"step_type":"DECIDE","actions":[所有动作]}]。',
                        trace=trace,
                    )
                    continue
                raise
            if trace:
                trace("model_output", {"step": step_number, "steps": steps})
            if not steps:
                return ({
                    "action_type": CandidateActionType.DEFER.value,
                    "note": "Sleep Agent 未形成可核实的决定",
                },)
            if len(steps) != 1:
                self._append_protocol_error(
                    observations,
                    step_number,
                    '单 Agent 每步只返回一个step。最终动作必须放在'
                    '[{"step_type":"DECIDE","actions":[所有动作]}]中，不能直接返回动作数组。',
                    trace=trace,
                )
                continue
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
                    return ({
                        "action_type": CandidateActionType.DEFER.value,
                        "note": "Sleep Agent 未形成可核实的决定",
                    },)
                if not all(self._is_final_action(item) for item in actions):
                    raise AgentProtocolError("DECIDE.actions 包含非法动作")
                if review_requested:
                    include_read_target_sources(actions)
                    candidate_context = self._combine_candidate_contexts(
                        tuple(candidate_contexts.values())
                    )
                normalized_actions, validation_error = self._validate_decision_actions(
                    actions, candidate_contexts
                )
                if validation_error:
                    self._append_protocol_error(
                        observations,
                        step_number,
                        validation_error,
                        trace=trace,
                    )
                    review_requested = False
                    continue
                actions = normalized_actions
                review_actions = actions
                write_types = {
                    CandidateActionType.CREATE_NEW,
                    CandidateActionType.REINFORCE,
                    CandidateActionType.REVISE,
                    CandidateActionType.MERGE,
                }
                if all(self._parse_action_type(action.get("action_type")) in {
                    CandidateActionType.IGNORE, CandidateActionType.DEFER,
                } for action in actions):
                    if review_requested:
                        return actions
                    covered_actions: dict[int, str] = {}
                    for index, action in enumerate(actions):
                        if self._parse_action_type(action.get("action_type")) is not CandidateActionType.IGNORE:
                            continue
                        candidate_keys = {
                            (str(source.get("stream_id")), str(source.get("message_id")))
                            for candidate_id in action["candidate_ids"]
                            for source in candidate_contexts[candidate_id].source_snapshots
                        }
                        for memory_id in read_memories:
                            target_keys = {
                                (str(source.get("stream_id")), str(source.get("message_id")))
                                for source in read_target_snapshots(memory_id)
                            }
                            if candidate_keys & target_keys:
                                covered_actions[index] = memory_id
                                break
                    if not covered_actions:
                        return actions
                    if step_number >= max_steps:
                        return tuple(
                            {**action, "action_type": CandidateActionType.DEFER.value,
                             "note": "旧记忆覆盖判断尚未完成来源复核"}
                            if index in covered_actions else action
                            for index, action in enumerate(actions)
                        )
                    observations.append({
                        "step": step_number,
                        "tool": "SOURCE_COVERAGE_REVIEW_REQUIRED",
                        "existing_targets": covered_actions,
                    })
                    review_actions = actions
                    review_requested = True
                    continue
                unsearched_candidates = tuple(
                    candidate_id
                    for action in actions
                    if self._parse_action_type(action.get("action_type")) in write_types
                    for candidate_id in action["candidate_ids"]
                    if candidate_id not in searched_candidate_ids
                )
                if unsearched_candidates:
                    searched = True
                    for candidate_id in dict.fromkeys(unsearched_candidates):
                        context = candidate_contexts[candidate_id]
                        query = f"{context.rough_title}\n{context.rough_content}"
                        search_result = await self._tools.memory_search(
                            query, tool_context, limit=12
                        )
                        searched_memory_ids.update(
                            self._search_memory_ids(search_result)
                        )
                        searched_candidate_ids.add(candidate_id)
                        observation = {
                            "step": step_number,
                            "tool": "SEARCH",
                            "candidate_ids": (candidate_id,),
                            "query": query,
                            "result": self._compact_observation(search_result),
                        }
                        observations.append(observation)
                        if trace:
                            trace(
                                "SEARCH",
                                {
                                    "candidate_ids": (candidate_id,),
                                    "query": query,
                                    "results": search_result,
                                },
                            )
                        await read_search_hits(search_result, step_number, (candidate_id,))
                unread_contexts = tuple(dict.fromkeys(
                    candidate_id
                    for action in actions
                    if self._parse_action_type(action.get("action_type")) in write_types
                    for candidate_id in action["candidate_ids"]
                    if candidate_id not in context_read_candidate_ids
                ))
                if unread_contexts:
                    context_requests: dict[tuple[str, str], list[str]] = {}
                    for candidate_id in unread_contexts:
                        context = candidate_contexts[candidate_id]
                        if not context.evidence_ids or not context.source_snapshots:
                            continue
                        source = context.source_snapshots[0]
                        evidence_id = source.get("evidence_id") or context.evidence_ids[0]
                        message_id = source.get("message_id")
                        if not isinstance(evidence_id, str) or not isinstance(message_id, str):
                            continue
                        context_requests.setdefault(
                            (evidence_id, message_id), []
                        ).append(candidate_id)
                    unread_without_source = tuple(
                        candidate_id
                        for candidate_id in unread_contexts
                        if not any(
                            candidate_id in candidate_ids
                            for candidate_ids in context_requests.values()
                        )
                    )
                    if unread_without_source:
                        self._append_protocol_error(
                            observations, step_number,
                            "写入前需用 MESSAGE_CONTEXT_READ 核对候选来源前后对话，"
                            f"但这些候选缺少可读取的来源锚点：{unread_without_source}。",
                            trace=trace,
                        )
                        review_requested = False
                        continue
                    for (evidence_id, message_id), candidate_ids in context_requests.items():
                        request = {
                            "step_type": "MESSAGE_CONTEXT_READ",
                            "evidence_id": evidence_id,
                            "message_id": message_id,
                            "candidate_ids": tuple(candidate_ids),
                        }
                        context_result = await self._observe_step(
                            "MESSAGE_CONTEXT_READ", request, tool_context
                        )
                        messages = (
                            context_result.get("messages", ())
                            if isinstance(context_result, Mapping)
                            else ()
                        )
                        if isinstance(messages, (list, tuple)):
                            for candidate_id in candidate_ids:
                                current = candidate_contexts[candidate_id]
                                sources = {
                                    (str(source.get("stream_id")), str(source.get("message_id"))): dict(source)
                                    for source in current.source_snapshots
                                }
                                for message in messages:
                                    if not isinstance(message, Mapping) or message.get("redacted"):
                                        continue
                                    if not message.get("message_id") or not message.get("stream_id"):
                                        continue
                                    key = (str(message["stream_id"]), str(message["message_id"]))
                                    if key not in sources:
                                        sources[key] = dict(message)
                                    elif message.get("reply_to") and not sources[key].get("reply_to"):
                                        sources[key]["reply_to"] = message["reply_to"]
                                candidate_contexts[candidate_id] = replace(
                                    current, source_snapshots=tuple(sources.values()),
                                )
                                context_read_candidate_ids.add(candidate_id)
                        observation = {
                            "step": step_number,
                            "tool": "MESSAGE_CONTEXT_READ",
                            "evidence_id": evidence_id,
                            "message_id": message_id,
                            "candidate_ids": tuple(candidate_ids),
                            "result": self._compact_observation(context_result),
                        }
                        observations.append(observation)
                        if trace:
                            trace(
                                "MESSAGE_CONTEXT_READ",
                                {"request": request, "result": context_result},
                            )
                    candidate_context = self._combine_candidate_contexts(
                        tuple(candidate_contexts.values())
                    )
                if unsearched_candidates or unread_contexts:
                    review_requested = False
                    continue
                if not searched:
                    query = f"{candidate_context.rough_title}\n{candidate_context.rough_content}"
                    search_result = await self._tools.memory_search(
                        query, tool_context, limit=12
                    )
                    searched_memory_ids.update(self._search_memory_ids(search_result))
                    observations.append(
                        {
                            "step": step_number,
                            "tool": "SEARCH",
                            "query": query,
                            "result": self._compact_observation(search_result),
                        }
                    )
                    searched = True
                    if trace:
                        trace("SEARCH", {"query": query, "results": search_result})
                    await read_search_hits(
                        search_result, step_number, tuple(candidate_contexts)
                    )
                    continue
                target_memory_ids = self._action_target_memory_ids(actions)
                unsearched_targets = tuple(
                    memory_id
                    for memory_id in target_memory_ids
                    if memory_id not in searched_memory_ids
                )
                if unsearched_targets:
                    search_result = await self._tools.memory_search(
                        " ".join(unsearched_targets), tool_context, limit=12
                    )
                    searched_memory_ids.update(self._search_memory_ids(search_result))
                    observations.append(
                        {
                            "step": step_number,
                            "tool": "SEARCH",
                            "query": " ".join(unsearched_targets),
                            "result": self._compact_observation(search_result),
                        }
                    )
                    if trace:
                        trace(
                            "SEARCH",
                            {
                                "query": " ".join(unsearched_targets),
                                "results": search_result,
                            },
                        )
                    unavailable_targets = {
                        memory_id
                        for memory_id in unsearched_targets
                        if memory_id not in searched_memory_ids
                    }
                    if unavailable_targets:
                        actions = tuple(
                            {
                                **dict(action),
                                "action_type": CandidateActionType.DEFER.value,
                                "note": "目标记忆未出现在本轮检索结果中",
                            }
                            if isinstance(action, Mapping)
                            and unavailable_targets.intersection(
                                self._action_target_memory_ids((action,))
                            )
                            else action
                            for action in actions
                        )
                        if trace:
                            trace(
                                "unverified_target_deferred",
                                {
                                    "memory_ids": tuple(sorted(unavailable_targets)),
                                },
                            )
                        if review_requested:
                            return actions
                        observations.append(
                            {
                                "step": step_number,
                                "tool": "UNVERIFIED_TARGETS_DEFERRED",
                                "draft_actions": actions,
                            }
                        )
                        continue
                    continue
                unread_targets = tuple(
                    memory_id
                    for memory_id in target_memory_ids
                    if memory_id not in read_full_memory_ids
                )
                if unread_targets:
                    target_observations = await self._read_memories_with_evidence(
                        unread_targets,
                        tool_context,
                        read_memory_ids=read_memory_ids,
                        read_full_memory_ids=read_full_memory_ids,
                        read_evidence_ids=read_evidence_ids,
                        trace=trace,
                    )
                    observations.extend(
                        {
                            "step": step_number,
                            "tool": "MEMORY_READ",
                            "result": self._compact_observation(item),
                        }
                        for item in target_observations
                    )
                    read_memories.update({
                        str(item["memory_id"]): dict(item["memory"])
                        for item in target_observations
                    })
                    continue
                covered_actions: dict[int, str] = {}
                for index, action in enumerate(actions):
                    if self._parse_action_type(action.get("action_type")) is not CandidateActionType.CREATE_NEW:
                        continue
                    cited_ids = set(action.get("source_message_ids", ()))
                    cited_keys = {
                        (str(source.get("stream_id")), str(source.get("message_id")))
                        for candidate_id in action["candidate_ids"]
                        for source in candidate_contexts[candidate_id].source_snapshots
                        if source.get("message_id") in cited_ids
                    }
                    for memory_id, memory in read_memories.items():
                        if memory.get("status") != "ACTIVE":
                            continue
                        records = memory.get("evidence_metadata") or ()
                        existing_keys = {
                            (str(message.get("stream_id")), str(message.get("message_id")))
                            for record in records
                            for message in record.get("messages", ())
                            if message.get("source_status") == "AVAILABLE"
                        }
                        if cited_keys and cited_keys <= existing_keys:
                            covered_actions[index] = memory_id
                            break
                if covered_actions:
                    feedback = {
                        "tool": "SOURCE_ALREADY_COVERED",
                        "actions": actions,
                        "existing_targets": covered_actions,
                        "message": "这些CREATE引用的全部消息已在已读正式记忆覆盖，请复用、修订或暂缓。",
                    }
                    observations.append(feedback)
                    if trace:
                        trace("source_already_covered", feedback)
                    if step_number < max_steps:
                        review_actions = actions
                        review_requested = True
                        continue
                    actions = tuple(
                        {"action_type": "DEFER", "candidate_ids": action["candidate_ids"],
                         "note": "全部引用消息已被正式记忆覆盖，剩余步数不足以核对新的重复项"}
                        if index in covered_actions else action
                        for index, action in enumerate(actions)
                    )
                if not review_requested:
                    if step_number >= max_steps:
                        if trace:
                            trace(
                                "decision_deferred",
                                {
                                    "reason": "没有剩余步数执行最终原文复核",
                                    "actions": actions,
                                },
                            )
                        return self._defer_group_actions(
                            actions, "缺少最终原文复核，安全起见暂缓"
                        )
                    observations.append(
                        {
                            "step": step_number,
                            "tool": "FINAL_SOURCE_REVIEW_REQUIRED",
                            "draft_actions": actions,
                            "source_messages": self._source_payload(
                                candidate_context.source_snapshots
                            ),
                        }
                    )
                    review_requested = True
                    review_actions = actions
                    if trace:
                        trace(
                            "decision_review_requested",
                            {"step": step_number, "actions": actions},
                        )
                    continue
                actions = self._require_human_source_messages(
                    actions, candidate_context, trace=trace
                )
                return actions
            if normalized in {item.value for item in CandidateActionType}:
                protocol_error = "produce 路径必须通过 DECIDE.actions 返回动作"
                observations.append(
                    {
                        "step": step_number,
                        "tool": "PROTOCOL_ERROR",
                        "message": protocol_error,
                        "received": dict(step),
                    }
                )
                if trace:
                    trace(
                        "protocol_error",
                        {
                            "step": step_number,
                            "message": protocol_error,
                            "received": dict(step),
                            "raw_response": getattr(
                                step_producer, "last_raw_response", None
                            ),
                        },
                    )
                continue
            if normalized == "SEARCH":
                search_requests, search_error = self._parse_search_requests(
                    step, candidate_contexts
                )
                if search_error:
                    self._append_protocol_error(
                        observations,
                        step_number,
                        search_error,
                        trace=trace,
                    )
                    continue
                searched = True
                for candidate_ids, query in search_requests:
                    search_result = await self._tools.memory_search(
                        query, tool_context, limit=12
                    )
                    searched_memory_ids.update(
                        self._search_memory_ids(search_result)
                    )
                    searched_candidate_ids.update(candidate_ids)
                    observations.append(
                        {
                            "step": step_number,
                            "tool": "SEARCH",
                            "candidate_ids": candidate_ids,
                            "query": query,
                            "result": self._compact_observation(search_result),
                        }
                    )
                    if trace:
                        trace(
                            "SEARCH",
                            {
                                "candidate_ids": candidate_ids,
                                "query": query,
                                "results": search_result,
                            },
                        )
                    await read_search_hits(search_result, step_number, candidate_ids)
                continue
            try:
                if normalized == "MESSAGE_CONTEXT_READ":
                    evidence_id = self._required_text(step.get("evidence_id"), "MESSAGE_CONTEXT_READ evidence_id")
                    if evidence_id not in read_evidence_ids:
                        raise AgentProtocolError("先读取本组候选或旧记忆的 Evidence，再查看其消息上下文")
                    raw_ids = step.get("candidate_ids")
                    if raw_ids is not None and (
                        not isinstance(raw_ids, (list, tuple)) or not raw_ids
                        or any(not isinstance(key, str) or key not in candidate_contexts for key in raw_ids)
                    ):
                        raise AgentProtocolError("MESSAGE_CONTEXT_READ.candidate_ids 必须引用本组候选")
                observation = await self._observe_step(normalized, step, tool_context)
            except (AgentProtocolError, ValueError, PermissionError) as error:
                self._append_protocol_error(
                    observations, step_number, str(error), trace=trace,
                )
                continue
            if normalized == "MESSAGE_CONTEXT_READ" and isinstance(observation, Mapping):
                evidence_id = self._required_text(step.get("evidence_id"), "MESSAGE_CONTEXT_READ evidence_id")
                raw_candidate_ids = step.get("candidate_ids")
                if raw_candidate_ids is None:
                    context_ids = tuple(
                        key for key, item in candidate_contexts.items()
                        if evidence_id in item.evidence_ids
                    )
                elif isinstance(raw_candidate_ids, (list, tuple)) and raw_candidate_ids and all(
                    isinstance(key, str) and key in candidate_contexts for key in raw_candidate_ids
                ):
                    context_ids = tuple(dict.fromkeys(raw_candidate_ids))
                else:
                    self._append_protocol_error(
                        observations, step_number,
                        "MESSAGE_CONTEXT_READ.candidate_ids 必须引用本组候选", trace=trace,
                    )
                    continue
                messages = observation.get("messages", ())
                if isinstance(messages, (list, tuple)):
                    for candidate_id in context_ids:
                        current = candidate_contexts[candidate_id]
                        sources = {
                            (str(source.get("stream_id")), str(source.get("message_id"))): dict(source)
                            for source in current.source_snapshots
                        }
                        for message in messages:
                            if not isinstance(message, Mapping) or message.get("redacted"):
                                continue
                            if not message.get("message_id") or not message.get("stream_id"):
                                continue
                            key = (str(message["stream_id"]), str(message["message_id"]))
                            if key not in sources:
                                sources[key] = dict(message)
                            elif message.get("reply_to") and not sources[key].get("reply_to"):
                                sources[key]["reply_to"] = message["reply_to"]
                        candidate_contexts[candidate_id] = replace(
                            current, source_snapshots=tuple(sources.values()),
                        )
                        context_read_candidate_ids.add(candidate_id)
                    candidate_context = self._combine_candidate_contexts(tuple(candidate_contexts.values()))
            if normalized == "MEMORY_READ":
                memory_id = self._required_text(step.get("memory_id"), "MEMORY_READ memory_id")
                read_memory_ids.add(memory_id)
                view = self._optional_text(step.get("view")) or "full"
                if view == "full":
                    read_full_memory_ids.add(memory_id)
                if isinstance(observation, Mapping):
                    if view == "full":
                        read_memories[memory_id] = dict(observation)
                    evidence_ids = self._memory_evidence_ids(observation)
                    unread_evidence_ids = tuple(
                        evidence_id
                        for evidence_id in evidence_ids
                        if evidence_id not in read_evidence_ids
                    )
                    if unread_evidence_ids:
                        evidence = await self._tools.evidence_read(
                            unread_evidence_ids, tool_context
                        )
                        read_evidence_ids.update(unread_evidence_ids)
                        observation = {
                            "memory": observation,
                            "evidence": evidence,
                        }
                        if trace:
                            trace(
                                "EVIDENCE_READ",
                                {
                                    "evidence_ids": unread_evidence_ids,
                                    "results": evidence,
                                },
                            )
            elif normalized == "EVIDENCE_READ":
                raw_ids = step.get("evidence_ids")
                if isinstance(raw_ids, (list, tuple)):
                    read_evidence_ids.update(
                        item for item in raw_ids if isinstance(item, str)
                    )
            observations.append(
                {
                    "step": step_number,
                    "tool": normalized,
                    "result": self._compact_observation(observation),
                }
            )
            if trace:
                trace(normalized, {"request": dict(step), "result": observation})
        return (
            {
                "action_type": CandidateActionType.DEFER.value,
                "note": f"Agent 达到 {max_steps} 步上限，无法安全形成决定",
            },
        )

    async def _read_memories_with_evidence(
        self,
        memory_ids: tuple[str, ...],
        context: ToolContext,
        *,
        read_memory_ids: set[str],
        read_full_memory_ids: set[str],
        read_evidence_ids: set[str],
        trace: Callable[[str, dict[str, object]], None] | None = None,
    ) -> tuple[dict[str, object], ...]:
        """读取目标 Memory 当前 Revision、历史及长期原始证据。"""
        results: list[dict[str, object]] = []
        for memory_id in memory_ids:
            memory = await self._tools.memory_read(memory_id, "full", context)
            read_memory_ids.add(memory_id)
            read_full_memory_ids.add(memory_id)
            evidence_ids = self._memory_evidence_ids(memory)
            unread_evidence_ids = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in read_evidence_ids
            )
            evidence = (
                await self._tools.evidence_read(unread_evidence_ids, context)
                if unread_evidence_ids
                else ()
            )
            read_evidence_ids.update(unread_evidence_ids)
            result = {
                "memory_id": memory_id,
                "memory": memory,
                "evidence": evidence,
            }
            results.append(result)
            if trace:
                trace("MEMORY_READ", {"memory_id": memory_id, "result": memory})
                if unread_evidence_ids:
                    trace(
                        "EVIDENCE_READ",
                        {"evidence_ids": unread_evidence_ids, "results": evidence},
                    )
        return tuple(results)

    @staticmethod
    def _memory_evidence_ids(memory: Mapping[str, object]) -> tuple[str, ...]:
        """提取完整记忆视图中可读取的证据标识。"""
        summary = memory.get("evidence_summary", ())
        if not isinstance(summary, (list, tuple)):
            return ()
        return tuple(
            item["evidence_id"]
            for item in summary
            if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str)
        )

    @staticmethod
    def _search_memory_ids(results: object) -> tuple[str, ...]:
        """读取 memory_search 返回项中的真实 Memory ID。"""
        if not isinstance(results, (list, tuple)):
            return ()
        return tuple(
            item["memory_id"]
            for item in results
            if isinstance(item, Mapping)
            and isinstance(item.get("memory_id"), str)
            and item["memory_id"].strip()
        )

    @staticmethod
    def _append_protocol_error(
        observations: list[dict[str, object]],
        step_number: int,
        message: str,
        *,
        trace: Callable[[str, dict[str, object]], None] | None,
    ) -> None:
        """把可纠正的 Agent 契约问题回送到当前调查循环。"""
        observation = {
            "step": step_number,
            "tool": "PROTOCOL_ERROR",
            "message": message,
        }
        observations.append(observation)
        if trace:
            trace("protocol_error", observation)

    @staticmethod
    def _parse_search_requests(
        step: Mapping[str, object],
        candidate_contexts: Mapping[str, _CandidateContext],
    ) -> tuple[tuple[tuple[tuple[str, ...], str], ...], str | None]:
        """解析单主题或按候选主题分组的 SEARCH 请求。"""
        raw_queries = step.get("queries")
        if raw_queries is None:
            query = SleepAgentOrchestrator._optional_text(step.get("query"))
            if query is None:
                return (), "SEARCH 必须提供非空 query 或 queries。"
            raw_ids = step.get("candidate_ids", ())
            if not isinstance(raw_ids, (list, tuple)) or not all(
                isinstance(item, str) and item.strip() for item in raw_ids
            ):
                return (), "SEARCH.candidate_ids 必须是非空字符串数组。"
            candidate_ids = tuple(dict.fromkeys(item.strip() for item in raw_ids))
            unknown = tuple(
                item for item in candidate_ids if item not in candidate_contexts
            )
            if unknown:
                return (), f"SEARCH 引用了未知候选：{unknown}。"
            return ((candidate_ids, query),), None
        if not isinstance(raw_queries, (list, tuple)) or not raw_queries:
            return (), "SEARCH.queries 必须是非空数组。"
        requests: list[tuple[tuple[str, ...], str]] = []
        seen_candidate_ids: set[str] = set()
        for index, item in enumerate(raw_queries):
            if not isinstance(item, Mapping):
                return (), f"SEARCH.queries[{index}] 必须是对象。"
            query = SleepAgentOrchestrator._optional_text(item.get("query"))
            raw_ids = item.get("candidate_ids")
            if query is None:
                return (), f"SEARCH.queries[{index}].query 不能为空。"
            if not isinstance(raw_ids, (list, tuple)) or not raw_ids or not all(
                isinstance(candidate_id, str) and candidate_id.strip()
                for candidate_id in raw_ids
            ):
                return (), (
                    f"SEARCH.queries[{index}].candidate_ids 必须是非空候选 ID 数组。"
                )
            candidate_ids = tuple(dict.fromkeys(item.strip() for item in raw_ids))
            unknown = tuple(
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id not in candidate_contexts
            )
            duplicates = tuple(
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id in seen_candidate_ids
            )
            if unknown:
                return (), f"SEARCH 引用了未知候选：{unknown}。"
            if duplicates:
                return (), f"SEARCH 重复指定候选：{duplicates}。"
            seen_candidate_ids.update(candidate_ids)
            requests.append((candidate_ids, query))
        return tuple(requests), None

    @staticmethod
    def _validate_decision_actions(
        actions: tuple[object, ...],
        candidate_contexts: Mapping[str, _CandidateContext],
    ) -> tuple[tuple[dict[str, object], ...], str | None]:
        """校验动作覆盖、必填参数与底层领域输入。"""
        assigned: set[str] = set()
        normalized: list[dict[str, object]] = []
        candidate_ids = tuple(candidate_contexts)
        for index, raw_action in enumerate(actions):
            if not isinstance(raw_action, Mapping):
                return (), f"DECIDE.actions[{index}] 必须是对象。"
            action = dict(raw_action)
            retired_fields = {
                "confidence",
                "confidence_hint",
                "confidence_reason",
                "stability",
                "stability_hint",
                "stability_reason",
                "salience",
                "salience_hint",
                "salience_reason",
                "retention_reason",
                "uncertainty_note",
                "claim_basis",
                "provenance_quality",
                "evidence_role",
                "anchor_title",
                "event_time",
                "event_date",
                "event_start",
                "event_end",
                "role",
            }
            pending_values: list[object] = [action]
            retired_keys: set[str] = set()
            while pending_values:
                value = pending_values.pop()
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        if isinstance(key, str) and key.casefold() in retired_fields:
                            retired_keys.add(key)
                        pending_values.append(child)
                elif isinstance(value, (list, tuple)):
                    pending_values.extend(value)
            if retired_keys:
                return (), (
                    f"DECIDE.actions[{index}] uses retired fields: "
                    f"{tuple(sorted(retired_keys))}. Keep supported meaning in prose."
                )
            try:
                action_type = SleepAgentOrchestrator._parse_action_type(
                    action.get("action_type")
                )
            except ValueError as error:
                return (), f"DECIDE.actions[{index}].action_type 无效：{error}。"
            raw_ids = action.get("candidate_ids")
            if raw_ids is None and action_type in {
                CandidateActionType.DEFER,
                CandidateActionType.IGNORE,
            }:
                action_ids = tuple(
                    candidate_id
                    for candidate_id in candidate_ids
                    if candidate_id not in assigned
                )
            elif raw_ids is None and len(candidate_ids) == 1:
                action_ids = candidate_ids
            elif isinstance(raw_ids, (list, tuple)) and raw_ids and all(
                isinstance(item, str) and item for item in raw_ids
            ):
                action_ids = tuple(raw_ids)
            else:
                return (), (
                    f"{action_type.value} 必须提供非空 candidate_ids 数组。"
                )
            if not action_ids or len(set(action_ids)) != len(action_ids):
                return (), f"{action_type.value} candidate_ids 为空或重复。"
            unknown = tuple(
                item for item in action_ids if item not in candidate_contexts
            )
            duplicates = tuple(item for item in action_ids if item in assigned)
            if unknown or duplicates:
                return (), (
                    f"{action_type.value} candidate_ids 存在未知或重复项："
                    f"{unknown + duplicates}。"
                )
            assigned.update(action_ids)
            action["candidate_ids"] = list(action_ids)
            action_context = SleepAgentOrchestrator._combine_candidate_contexts(
                tuple(candidate_contexts[item] for item in action_ids)
            )
            required_fields = {
                CandidateActionType.CREATE_NEW: (
                    "title",
                    "content",
                    "memory_kind",
                    "source_message_ids",
                ),
                CandidateActionType.REINFORCE: (
                    "memory_id",
                    "reason",
                    "source_message_ids",
                ),
                CandidateActionType.REVISE: (
                    "memory_id",
                    "reason",
                    "title",
                    "content",
                    "memory_kind",
                    "change_reason",
                    "source_message_ids",
                ),
                CandidateActionType.MERGE: (
                    "source_memory_ids",
                    "reason",
                    "source_message_ids",
                ),
                CandidateActionType.RELATE: (
                    "source_memory_id",
                    "target_memory_id",
                    "relation_type",
                    "reason",
                ),
                CandidateActionType.DEFER: ("note",),
            }.get(action_type, ())
            missing_fields = tuple(
                field_name
                for field_name in required_fields
                if action.get(field_name) is None
                or action.get(field_name) == ""
                or action.get(field_name) == []
            )
            if missing_fields:
                return (), (
                    f"{action_type.value} 缺少必填字段：{missing_fields}。"
                )
            try:
                if action_type is CandidateActionType.CREATE_NEW:
                    model_input = SleepAgentOrchestrator._create_memory_input(
                        action_context, action
                    )
                elif action_type is CandidateActionType.REINFORCE:
                    model_input = SleepAgentOrchestrator._reinforce_memory_input(
                        action_context,
                        {
                            **action,
                            "based_on_revision_id": action.get(
                                "based_on_revision_id", "prevalidation-current-revision"
                            ),
                        },
                    )
                elif action_type is CandidateActionType.REVISE:
                    model_input = SleepAgentOrchestrator._revise_memory_input(
                        action_context,
                        {
                            **action,
                            "based_on_revision_id": action.get(
                                "based_on_revision_id", "prevalidation-current-revision"
                            ),
                        },
                    )
                    action.setdefault("note", str(action["reason"]).strip())
                elif action_type is CandidateActionType.MERGE:
                    model_input = SleepAgentOrchestrator._merge_memory_input(
                        action, action_context
                    )
                elif action_type is CandidateActionType.RELATE:
                    model_input = SleepAgentOrchestrator._relate_memory_input(action)
                else:
                    normalized.append(action)
                    continue
                model_input.validate()
                reason = SleepAgentOrchestrator._optional_text(
                    action.get("reason")
                )
                if reason and not SleepAgentOrchestrator._optional_text(
                    action.get("note")
                ):
                    action["note"] = reason
            except (TypeError, ValueError) as error:
                return (), f"{action_type.value} 参数无效：{error}。"
            if action_type in {
                CandidateActionType.CREATE_NEW,
                CandidateActionType.REINFORCE,
                CandidateActionType.REVISE,
                CandidateActionType.MERGE,
            }:
                checked = SleepAgentOrchestrator._require_human_source_messages(
                    (action,), action_context, trace=None
                )[0]
                if checked.get("action_type") == CandidateActionType.DEFER.value:
                    permitted = tuple(
                        str(source["message_id"])
                        for source in action_context.source_snapshots
                        if source.get("message_id") and source.get("person_id")
                        and str(source["person_id"]).casefold() != "bot"
                        and not source.get("redacted")
                    )
                    return (), (
                        f"{action_type.value} 的 source_message_ids 必须逐条引用"
                        "候选关联、非 Bot、直接支持内容的唯一原始消息。"
                        f"本动作 candidate_ids={action_ids!r}，允许的消息 ID={permitted!r}；"
                        "请仅从该集合选择支持正文的消息，不从同组其他候选或旧记忆借用 ID。"
                    )
            normalized.append(action)
        if assigned != set(candidate_ids):
            return (), (
                f"DECIDE.actions 未恰好覆盖本组候选："
                f"缺少 {tuple(item for item in candidate_ids if item not in assigned)}。"
            )
        return tuple(normalized), None

    @classmethod
    def _action_target_memory_ids(
        cls, actions: tuple[object, ...]
    ) -> tuple[str, ...]:
        """从最终动作中读取所有正式 Memory 目标。"""
        targets: list[str] = []
        for raw_action in actions:
            if not isinstance(raw_action, Mapping):
                continue
            action_type = cls._parse_action_type(raw_action.get("action_type"))
            fields = {
                CandidateActionType.REINFORCE: ("memory_id",),
                CandidateActionType.REVISE: ("memory_id",),
                CandidateActionType.MERGE: (
                    "canonical_memory_id",
                    "source_memory_ids",
                ),
                CandidateActionType.RELATE: (
                    "source_memory_id",
                    "target_memory_id",
                ),
            }.get(action_type, ())
            for field_name in fields:
                value = raw_action.get(field_name)
                if isinstance(value, str) and value.strip():
                    targets.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    targets.extend(
                        item.strip()
                        for item in value
                        if isinstance(item, str) and item.strip()
                    )
        return tuple(dict.fromkeys(targets))

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
        if step_type == "MESSAGE_CONTEXT_READ":
            evidence_id = self._required_text(step.get("evidence_id"), "MESSAGE_CONTEXT_READ evidence_id")
            message_id = self._required_text(step.get("message_id"), "MESSAGE_CONTEXT_READ message_id")
            counts: dict[str, int] = {}
            for name in ("before", "after"):
                value = step.get(name, 20 if name == "after" else 8)
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 30:
                    raise AgentProtocolError(f"MESSAGE_CONTEXT_READ.{name} 必须为0至30的整数")
                counts[name] = value
            return await self._tools.message_context_read(
                evidence_id, message_id, context, **counts,
            )
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
        """保留消息语义与追溯标识，减少模型输入中的重复元数据。"""
        if isinstance(value, Mapping) and isinstance(value.get("memory"), Mapping):
            memory = value["memory"]
            source_records = memory.get("evidence_metadata")
            if not isinstance(source_records, (list, tuple)):
                source_records = value.get("evidence", ())
            if not isinstance(source_records, (list, tuple)):
                source_records = ()
            originals = tuple(
                {**snapshot, "evidence_id": evidence_id}
                for evidence_id, snapshots in SleepAgentOrchestrator._snapshots_by_evidence(
                    tuple(record for record in source_records if isinstance(record, Mapping))
                ).items()
                for snapshot in snapshots
            )
            current = memory.get("current_revision")
            current_id = current.get("revision_id") if isinstance(current, Mapping) else None
            compact_memory = {
                key: item for key, item in memory.items()
                if key != "evidence_metadata"
            }
            history = compact_memory.get("history")
            if isinstance(history, (list, tuple)):
                compact_memory["history"] = tuple(
                    item for item in history
                    if not isinstance(item, Mapping) or item.get("revision_id") != current_id
                )
            return {
                "memory_id": value.get("memory_id"),
                "original_sources": SleepAgentOrchestrator._source_payload(originals),
                "memory": compact_memory,
            }
        if isinstance(value, Mapping) and isinstance(value.get("messages"), (list, tuple)):
            messages = []
            for message in value["messages"]:
                if not isinstance(message, Mapping):
                    continue
                item = {
                    key: message[key]
                    for key in (
                        "message_id", "stream_id", "time", "sender_name",
                        "sender_cardname", "person_id", "speaker_is_bot",
                        "speaker_kind", "sender_role",
                        "message_type", "reply_to",
                    )
                    if message.get(key) is not None
                }
                raw_text = str(message.get("processed_plain_text") or message.get("content") or "")
                own_text, reply_preview = split_reply_preview(
                    raw_text,
                    message.get("reply_to"),
                )
                item["text"] = own_text
                if reply_preview is not None:
                    item["reply_preview_not_speaker_text"] = reply_preview
                elif has_unseparated_reply_preview(raw_text, message.get("reply_to")):
                    item["reply_boundary_unresolved"] = True
                messages.append(item)
            return {**value, "messages": messages}
        return value

    async def _research_candidate(
        self,
        context: _CandidateContext,
        tool_context: ToolContext,
        *,
        trace: Callable[[str, dict[str, object]], None] | None = None,
    ) -> str:
        """执行 Sleep Agent 的受控全局检索、读取和证据复核。"""
        queries = tuple(
            dict.fromkeys(
                item.strip()
                for item in (
                    f"{context.rough_title}\n{context.rough_content}",
                    context.rough_title,
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
            if trace:
                trace("search", {"query": query, "results": search_results})
            for item in search_results:
                memory_id = item.get("memory_id")
                if isinstance(memory_id, str) and memory_id.strip():
                    result_by_memory_id.setdefault(memory_id, item)
            if len(result_by_memory_id) >= 8:
                break
        if not result_by_memory_id:
            return "无相关 Formal Memory 命中。"
        readable: list[dict[str, object]] = []
        if context.evidence_ids:
            readable.append(
                {
                    "candidate_source_messages": await self._tools.evidence_read(
                        context.evidence_ids, tool_context
                    )
                }
            )
        for item in tuple(result_by_memory_id.values())[:8]:
            memory_id = item.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id.strip():
                continue
            full = await self._tools.memory_read(memory_id, "full", tool_context)
            evidence_ids = self._memory_evidence_ids(full)
            evidence = (
                await self._tools.evidence_read(evidence_ids, tool_context)
                if evidence_ids
                else ()
            )
            if trace:
                trace("memory_read", {"memory_id": memory_id, "result": full})
                if evidence_ids:
                    trace("evidence_read", {"evidence_ids": evidence_ids, "result": evidence})
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
        candidate_context: _CandidateContext | None = None,
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
        if action_type in (CandidateActionType.REVISE, CandidateActionType.REINFORCE) and not intent.get("based_on_revision_id"):
            memory_id = self._required_text(intent.get("memory_id"), f"{action_type.value} memory_id")
            async with self._schema.database.session() as session:
                memory = await session.get(MemoryModel, memory_id)
                if memory is None or memory.status is not MemoryStatus.ACTIVE:
                    raise ValueError(f"{action_type.value} 目标记忆不存在或不是 ACTIVE")
                intent = {**intent, "based_on_revision_id": memory.current_revision_id}
        await self._sessions.mark_operation_executing(operation_key)
        context = candidate_context or await self._candidate_context(candidate_id)
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
        """把 CREATE_NEW 意图映射为正式记忆输入。"""
        kind = SleepAgentOrchestrator._enum_value(
            MemoryKind,
            intent.get("memory_kind"),
            context.proposed_kind or MemoryKind.EVENT,
        )
        return CreateMemoryInput(
            title=SleepAgentOrchestrator._text_or_default(
                intent.get("title"), context.rough_title
            ),
            content=SleepAgentOrchestrator._text_or_default(
                intent.get("content"), context.rough_content
            ),
            memory_kind=kind,
            subject=SleepAgentOrchestrator._trusted_subject(
                context,
                SleepAgentOrchestrator._subject_value(
                    intent.get("subject"), context.subject
                ),
            ),
            participants=SleepAgentOrchestrator._trusted_participants(
                context,
                SleepAgentOrchestrator._participants_value(
                    intent.get("participants"), context.participants
                ),
            ),
            observed_at=SleepAgentOrchestrator._datetime_value(
                intent.get("observed_at"), context.observed_at
            ),
            evidence_ids=context.evidence_ids,
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
        return ReinforceMemoryInput(
            memory_id=memory_id,
            based_on_revision_id=revision_id,
            reason=SleepAgentOrchestrator._required_text(
                intent.get("reason"), "REINFORCE reason"
            ),
            evidence_ids=context.evidence_ids,
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
            subject=SleepAgentOrchestrator._trusted_subject(
                context,
                SleepAgentOrchestrator._subject_value(
                    intent.get("subject"), context.subject
                ),
            ),
            participants=SleepAgentOrchestrator._trusted_participants(
                context,
                SleepAgentOrchestrator._participants_value(
                    intent.get("participants"), context.participants
                ),
            ),
            observed_at=SleepAgentOrchestrator._datetime_value(
                intent.get("observed_at"), context.observed_at
            ),
            change_reason=SleepAgentOrchestrator._enum_value(
                RevisionChangeReason,
                intent.get("change_reason"),
                RevisionChangeReason.DEVELOPMENT,
            ),
            evidence_ids=context.evidence_ids,
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
            evidence_ids=context.evidence_ids if context is not None and mode == "EXISTING_CANONICAL" else (),
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
