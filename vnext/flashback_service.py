"""Engram Memory vNext 闪回服务。

按 Technical Spec §121-§133 与 Core Design §61-§73 实现回复前自动联想：
复用统一 RetrievalService（不建第二套检索体系）、最近多轮拼接查询（不
调用额外 LLM，INV-022）、延迟预算内未完成本轮放弃（INV-023）、Per-Index
校准阈值（NULL 即关闭，§126）、先相关性门槛后 Salience 排序、同会话
按轮数 Cooldown（§129）、最多 0-2 条注入（§128）、只读（§131，
仅记录 FLASHBACK_EXPOSED 观察事件）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from collections import defaultdict
from uuid import uuid4

from sqlalchemy import select

from .domain import RetrievalQuery, ScoredMemory
from .enums import ActorType, MemoryEventType, VectorIndexStatus
from .models import MemoryEventModel, VectorIndexManifestModel
from .retrieval_service import RetrievalService
from .schema import VNextSchema


@dataclass(frozen=True, slots=True)
class FlashbackCandidate:
    """一条准备注入 Prompt 的闪回候选。"""

    memory_id: str
    anchor_title: str
    current_brief: str
    matched_cue: str

    def to_prompt_block(self) -> str:
        """构造注入主模型的自然联想文本（§130）。"""
        return (
            "【自然联想到的过去】\n"
            f"memory_id: {self.memory_id}\n"
            f"主题: {self.anchor_title}\n"
            f"你现在记得: {self.current_brief}\n"
            f"联想到的原因: 当前语境可能与「{self.matched_cue}」有关"
        )


class FlashbackService:
    """回复生成前的自动联想召回。"""

    def __init__(
        self,
        schema: VNextSchema,
        retrieval: RetrievalService,
        *,
        context_turns: int,
        latency_budget_ms: int,
        max_memories: int,
        cooldown_turns: int,
    ) -> None:
        """绑定检索服务与闪回参数。

        参数:
            schema: vNext Schema。
            retrieval: 统一混合检索服务（禁改认知属性）。
            context_turns: 参与查询拼接的最近对话轮数（§123）。
            latency_budget_ms: 延迟预算毫秒，超时本轮放弃（§125/INV-023）。
            max_memories: 单轮注入上限（只允许 0-2，§128）。
            cooldown_turns: 同记忆连续 N 轮不重复注入（§129）。
        """
        if context_turns <= 0:
            raise ValueError("context_turns 必须大于 0")
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms 必须大于 0")
        if not 0 <= max_memories <= 2:
            raise ValueError("max_memories 只允许 0-2")
        if cooldown_turns < 0:
            raise ValueError("cooldown_turns 不能为负")
        self._schema = schema
        self._retrieval = retrieval
        self._context_turns = context_turns
        self._latency_budget = latency_budget_ms
        self._max_memories = max_memories
        self._cooldown_turns = cooldown_turns
        self._stream_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def active_threshold(self) -> float | None:
        """读取当前 ACTIVE 索引的校准阈值；无索引或 NULL 即关闭（§126）。"""
        async with self._schema.database.session() as session:
            manifest = (
                await session.scalars(
                    select(VectorIndexManifestModel)
                    .where(VectorIndexManifestModel.status == VectorIndexStatus.ACTIVE)
                    .order_by(VectorIndexManifestModel.created_at.desc())
                )
            ).first()
        if manifest is None:
            return None
        return manifest.flashback_threshold

    async def next_turn_index(self, stream_key: str) -> int:
        """从持久化暴露事件恢复指定会话的下一轮编号。"""
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(MemoryEventModel.payload_json).where(
                            MemoryEventModel.event_type
                            == MemoryEventType.FLASHBACK_EXPOSED,
                            MemoryEventModel.stream_id == stream_key,
                        )
                    )
                ).all()
            )
        turn_indexes = tuple(
            payload.get("turn_index")
            for payload in rows
            if isinstance(payload, dict)
            and isinstance(payload.get("turn_index"), int)
        )
        return max(turn_indexes, default=-1) + 1

    async def flashback(
        self,
        recent_turns: tuple[str, ...],
        *,
        enabled: bool = True,
        stream_key: str = "default",
        turn_index: int | None = None,
        record_exposure: bool = True,
    ) -> tuple[FlashbackCandidate, ...]:
        """执行一次闪回检索并返回 0-2 条注入候选。

        参数:
            recent_turns: 最近多轮格式化文本（时间升序，可为空）。
            enabled: 运行配置 flashback.enabled 总开关。
            stream_key: Cooldown 会话键。

        返回:
            注入候选元组；关闭、无阈值、无上下文或超预算时为空。
        """
        if not enabled or self._max_memories == 0:
            return ()
        async def run_within_budget() -> tuple[FlashbackCandidate, ...]:
            """Execute threshold, cooldown and retrieval within one budget."""
            threshold = await self.active_threshold()
            if threshold is None:
                return ()
            query_text = self.build_query(recent_turns)
            if not query_text.strip():
                return ()
            exposed = await self._exposed_recently(stream_key, turn_index)
            return await self._scored(
                query_text,
                threshold,
                exposed,
                stream_key,
                turn_index,
                record_exposure,
            )
        async def locked_run() -> tuple[FlashbackCandidate, ...]:
            """Serialize cooldown read and exposure write per conversation."""
            async with self._stream_locks[stream_key]:
                return await run_within_budget()

        try:
            return await asyncio.wait_for(
                locked_run(), timeout=self._latency_budget / 1000.0
            )
        except TimeoutError:
            return ()  # INV-023：超预算放弃本轮，不拖慢回复

    def build_query(self, recent_turns: tuple[str, ...]) -> str:
        """拼接最近 N 轮为闪回查询文本（§123，不调用 LLM）。"""
        selected = recent_turns[-self._context_turns :]
        return "\n".join(piece.strip() for piece in selected if piece and piece.strip())

    async def _scored(
        self,
        query_text: str,
        threshold: float,
        excluded: frozenset[str],
        stream_key: str,
        turn_index: int | None = None,
        record_exposure: bool = True,
    ) -> tuple[FlashbackCandidate, ...]:
        """相关性门槛过滤 → Salience 辅助排序 → 限量 → 记暴露事件。"""
        results = await self._retrieval.search(RetrievalQuery(text=query_text, top_k=6))
        gated = [
            item
            for item in results
            if item.rrf_score >= threshold and item.memory_id not in excluded
        ]
        if not gated:
            return ()
        salience_by_memory = await self._salience_map(
            {item.memory_id for item in gated}
        )
        gated.sort(key=lambda item: (-item.rrf_score, -salience_by_memory.get(item.memory_id, 0), item.memory_id))
        selected = gated[: self._max_memories]
        now = datetime.now(UTC)
        candidates: list[FlashbackCandidate] = []
        for item in selected:
            candidates.append(
                FlashbackCandidate(
                    memory_id=item.memory_id,
                    anchor_title=item.anchor_title,
                    current_brief=await self._current_brief(item),
                    matched_cue="、".join(item.matched_by[:2]) or "相关经历",
                )
            )
        if record_exposure:
            await self._record_exposed_batch(
                tuple(item.memory_id for item in selected),
                stream_key,
                now,
                turn_index,
            )
        return tuple(candidates)

    async def record_exposure(
        self,
        memory_ids: tuple[str, ...],
        stream_key: str,
        turn_index: int | None = None,
    ) -> None:
        """Record exposure after candidates were actually accepted for injection."""
        await self._record_exposed_batch(
            memory_ids,
            stream_key,
            datetime.now(UTC),
            turn_index,
        )

    async def _record_exposed_batch(
        self,
        memory_ids: tuple[str, ...],
        stream_key: str,
        now: datetime,
        turn_index: int | None,
    ) -> None:
        """Atomically record all exposures selected for one flashback turn."""
        if not memory_ids:
            return
        if len(memory_ids) == 1:
            # Keep the narrow hook used by lightweight callers and tests;
            # one exposure is already an atomic database transaction.
            await self._record_exposed(memory_ids[0], stream_key, now, turn_index)
            return
        async with self._schema.database.session() as session:
            for memory_id in memory_ids:
                session.add(
                    MemoryEventModel(
                        event_id=str(uuid4()),
                        memory_id=memory_id,
                        revision_id=None,
                        event_type=MemoryEventType.FLASHBACK_EXPOSED,
                        actor_type=ActorType.SYSTEM,
                        actor_ref=None,
                        stream_id=stream_key,
                        occurred_at=now,
                        payload_json=(
                            {"turn_index": turn_index}
                            if turn_index is not None
                            else None
                        ),
                    )
                )

    async def _current_brief(self, scored: ScoredMemory) -> str:
        """读取简短当前摘要作为注入正文（不塞完整历史，§130）。"""
        from .repository import MemoryRepository

        repository = MemoryRepository(self._schema)
        revision = await repository.get_current_revision(scored.memory_id)
        if revision is None:
            return scored.anchor_title
        content = str(revision.content or "").strip()
        if not content:
            return revision.title
        return f"{revision.title}: {content[:160]}"

    async def _salience_map(self, memory_ids: frozenset[str]) -> dict[str, int]:
        """读取当前 Assessment 的 Salience 等级映射（仅供门槛后排序）。"""
        from .models import MemoryAssessmentModel, MemoryModel
        from .enums import SalienceLevel

        order = {
            SalienceLevel.LOW: 0,
            SalienceLevel.MEDIUM: 1,
            SalienceLevel.HIGH: 2,
            SalienceLevel.VERY_HIGH: 3,
        }
        if not memory_ids:
            return {}
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.execute(
                        select(MemoryModel.memory_id, MemoryAssessmentModel.salience)
                        .join(
                            MemoryAssessmentModel,
                            MemoryAssessmentModel.assessment_id
                            == MemoryModel.current_assessment_id,
                        )
                        .where(MemoryModel.memory_id.in_(memory_ids))
                    )
                ).all()
            )
        return {row.memory_id: order[row.salience] for row in rows}

    async def _exposed_recently(
        self,
        stream_key: str,
        turn_index: int | None = None,
    ) -> frozenset[str]:
        """读取本会话最近 cooldown_turns 次暴露内的记忆 ID（§129）。

        Cooldown Key 为 memory_id + 当前会话（stream_key）；
        cooldown_turns 为 0 表示不冷却。
        """
        if self._cooldown_turns == 0:
            return frozenset()
        async with self._schema.database.session() as session:
            rows = tuple(
                (
                    await session.execute(
                        select(MemoryEventModel.memory_id, MemoryEventModel.payload_json)
                        .where(
                            MemoryEventModel.event_type == MemoryEventType.FLASHBACK_EXPOSED,
                            MemoryEventModel.stream_id == stream_key,
                        )
                        .order_by(MemoryEventModel.occurred_at.desc())
                        .limit(max(self._cooldown_turns * 4, self._cooldown_turns))
                    )
                ).all()
            )
        if turn_index is None:
            return frozenset(row.memory_id for row in rows[: self._cooldown_turns])
        excluded: set[str] = set()
        for row in rows:
            payload = row.payload_json
            event_turn = payload.get("turn_index") if isinstance(payload, dict) else None
            delta = turn_index - event_turn if isinstance(event_turn, int) else None
            if delta is not None and 0 <= delta < self._cooldown_turns:
                excluded.add(row.memory_id)
        return frozenset(excluded)

    async def _record_exposed(
        self,
        memory_id: str,
        stream_key: str,
        now: datetime,
        turn_index: int | None = None,
    ) -> None:
        """记录 FLASHBACK_EXPOSED 观察事件（只读约束的最小例外，§131）。"""
        async with self._schema.database.session() as session:
            session.add(
                MemoryEventModel(
                    event_id=str(uuid4()),
                    memory_id=memory_id,
                    revision_id=None,
                    event_type=MemoryEventType.FLASHBACK_EXPOSED,
                    actor_type=ActorType.SYSTEM,
                    actor_ref=None,
                    stream_id=stream_key,
                    occurred_at=now,
                    payload_json={"turn_index": turn_index} if turn_index is not None else None,
                )
            )
