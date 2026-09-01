"""Engram Memory vNext 运行时 Owner。

本模块把 vNext 的规范数据库、领域服务、派生向量索引和后台任务
绑定到一个插件实例，保证框架组件不会各自创建隐藏的状态源。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from src.app.plugin_system.api import log_api, message_api, stream_api

from .candidate_service import CandidateService, SleepSessionService
from .doctor_service import DoctorService
from .domain import PersonaUpdateInput
from .enums import (
    ActorType,
    MemoryStatus,
    SleepSessionStatus,
    SleepTriggerType,
    VectorIndexStatus,
)
from .experience_encoder import ExperienceEncoder
from .flashback_service import FlashbackService
from .framework_bridge import (
    cancel_managed_task,
    create_managed_task,
    get_managed_task,
)
from .models import (
    CandidateEncoderCursorModel,
    MemoryModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    VectorIndexManifestModel,
)
from .retrieval_service import RetrievalService, VectorSearchBackend
from .runtime import (
    DEFAULT_EMBEDDING_MODEL_TASK,
    ChromaVectorSink,
    ExperienceEncoderDraftProducer,
    PersonaReviewProducer,
    SleepAgentStepProducer,
    VNextRuntimeAdapter,
    VectorOutboxWorker,
)
from .schema import VNextSchema
from .sleep_agent import SleepAgentOrchestrator
from .tool_service import ToolContext, VNextToolService
from .vector_service import VectorIndexService


logger = log_api.get_logger("engram_memory.vnext.runtime_owner")


class ChromaVectorSearchBackend(VectorSearchBackend):
    """通过 vNext 派生 Chroma 集合提供向量召回。"""

    def __init__(
        self,
        sink: ChromaVectorSink,
        schema: VNextSchema | None = None,
    ) -> None:
        """绑定只读向量查询端。"""
        self._sink = sink
        self._schema = schema

    async def query(
        self,
        texts: tuple[tuple[str, str], ...],
        top_k: int,
    ) -> tuple[str, ...]:
        """以第一项查询文本访问派生索引。"""
        if not texts:
            return ()
        if self._schema is None:
            return ()
        async with self._schema.database.session() as session:
            manifest = (
                await session.scalars(
                    select(VectorIndexManifestModel)
                    .where(VectorIndexManifestModel.status == VectorIndexStatus.ACTIVE)
                    .order_by(VectorIndexManifestModel.created_at.desc())
                )
            ).first()
        if manifest is None:
            return ()
        allowed_ids = {entry_id for entry_id, _ in texts[1:]}
        sink = self._sink.for_index(
            manifest.index_id,
            manifest.embedding_model_id,
            manifest.embedding_dimension,
        )
        result = await sink.query_entries(texts[0][1], top_k)
        return tuple(entry_id for entry_id in result if entry_id in allowed_ids)


class VNextRuntimeOwner:
    """持有插件范围内唯一的 vNext 运行时资源。"""

    def __init__(self, plugin: Any) -> None:
        """根据插件配置装配 vNext 资源，但不执行外部 I/O。"""
        from ..config import EngramMemoryConfig

        self.plugin = plugin
        self.config = (
            plugin.config
            if isinstance(plugin.config, EngramMemoryConfig)
            else EngramMemoryConfig()
        )
        config = self.config
        vnext = config.vnext
        self.schema = VNextSchema(config.storage.vnext_db_path)
        self.vector_sink = ChromaVectorSink(
            db_path=config.storage.vector_db_path,
            collection_name="engram_vnext_retrieval",
            embedding_task=DEFAULT_EMBEDDING_MODEL_TASK,
        )
        self._embedding_model_identity = self.vector_sink.embedding_model_identity()
        self.vector_backend = ChromaVectorSearchBackend(self.vector_sink, self.schema)
        self.retrieval = RetrievalService(
            self.schema,
            self.vector_backend,
            rrf_k=vnext.retrieval.rrf_k,
        )
        self.tools = VNextToolService(
            self.schema,
            self.vector_backend,
            embedding_model_id=self._embedding_model_identity,
            persona_max_length=vnext.persona.max_length,
            recent_memory_limit=vnext.persona.recent_memory_limit,
            default_search_limit=vnext.retrieval.default_limit,
            max_search_limit=vnext.retrieval.max_limit,
            rrf_k=vnext.retrieval.rrf_k,
        )
        self.candidate_service = CandidateService(self.schema)
        # Sleep Session is a plugin-scoped coordination service.  Keeping one
        # instance here prevents recovery and action auditing from observing
        # different runtime state.
        self.sleep_service = SleepSessionService(self.schema)
        self.encoder_producer = ExperienceEncoderDraftProducer(
            model_task=config.internal_llm.task_name,
        )
        self.sleep_producer = SleepAgentStepProducer(
            model_task=config.internal_llm.task_name,
            prompt_version="sleep-agent-v1",
        )
        self.persona_producer = PersonaReviewProducer(
            model_task=config.internal_llm.task_name,
            prompt_version="persona-review-v1",
        )
        self.runtime_adapter = VNextRuntimeAdapter(
            batch_size=vnext.candidate_encoder.message_threshold,
            encoder_producer=self.encoder_producer,
            sleep_producer=self.sleep_producer,
            vector_sink=self.vector_sink,
            vector_db_path=config.storage.vector_db_path,
        )
        self.encoder = ExperienceEncoder(
            self.candidate_service,
            message_threshold=vnext.candidate_encoder.message_threshold,
            max_wait_minutes=vnext.candidate_encoder.max_wait_minutes,
            prompt_version="experience-encoder-v1",
        )
        self.sleep_agent = SleepAgentOrchestrator(
            self.schema,
            self.tools,
            sleep_service=self.sleep_service,
            batch_size=vnext.sleep.batch_size,
            model_id=config.internal_llm.task_name,
            prompt_version="sleep-agent-v1",
        )
        self.vector_index = VectorIndexService(
            self.schema,
            self.vector_sink,
            max_attempts=vnext.vector.worker_retry_limit,
        )
        self.vector_worker = VectorOutboxWorker(
            self.vector_index,
            batch_size=vnext.sleep.batch_size,
            poll_interval_seconds=float(
                vnext.candidate_encoder.max_wait_minutes * 60
            ),
            retry_failed=True,
        )
        self.doctor: DoctorService | None = None
        self.flashback = FlashbackService(
            self.schema,
            self.retrieval,
            context_turns=vnext.flashback.context_turns,
            latency_budget_ms=vnext.flashback.latency_budget_ms,
            max_memories=vnext.flashback.max_memories,
            cooldown_turns=vnext.flashback.cooldown_turns,
        )
        self._initialized = False
        self._encoder_started_at: datetime | None = None
        self._pending_since: dict[str, datetime] = {}
        self._last_message_at: dict[str, datetime] = {}
        self._prompt_turns: dict[str, int] = {}
        self._scheduled_streams: set[str] = set()
        self._task_ids: set[str] = set()
        self._encoder_locks: dict[str, asyncio.Lock] = {}
        self._sleep_lock = asyncio.Lock()
        self._schema_initialized = False
        self._worker_started = False
        self._pending_messages: dict[str, dict[str, object]] = {}
        self._flashback_task_ids: dict[str, str] = {}
        self._flashback_results: dict[str, tuple[tuple[object, ...], int, int]] = {}
        self._flashback_generations: dict[str, int] = {}
        self._encoder_task_ids: dict[str, str] = {}
        self._task_ids_by_task: dict[asyncio.Task[Any], str] = {}
        self._flashback_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """初始化规范数据库并启动派生向量 Worker。"""
        if self._initialized:
            return
        try:
            embedding_identity, embedding_dimension = (
                await self.vector_sink.inspect_embedding_settings()
            )
            if embedding_identity != self._embedding_model_identity:
                raise RuntimeError("Embedding task identity 在初始化期间发生变化")
            self.doctor = DoctorService(
                self.schema,
                vector_service=self.vector_index,
                sleep_service=self.sleep_service,
                embedding_model_id=embedding_identity,
                embedding_dimension=embedding_dimension,
                retrieval_schema_version="engram-vnext-1",
            )
            self._schema_initialized = True
            await self.schema.initialize()
            await self.sleep_service.recover_abandoned_sessions()
            await self.vector_index.ensure_active_manifest(
                embedding_identity,
                embedding_dimension,
                "engram-vnext-1",
            )
            self._worker_started = True
            self.vector_worker.start()
            self._encoder_started_at = datetime.now(UTC)
            self._initialized = True
        except BaseException:
            await self._shutdown_resources()
            raise

    async def close(self) -> None:
        """停止后台 Worker、取消 Owner 任务并关闭规范数据库。"""
        await self._shutdown_resources()

    async def _shutdown_resources(self) -> None:
        """释放已部分或完整初始化的 Owner 资源，并保持幂等。"""
        task_ids = tuple(self._task_ids | set(self._flashback_task_ids.values()))
        task_infos = []
        for task_id in task_ids:
            try:
                task_info = get_managed_task(task_id)
            except Exception:  # noqa: BLE001
                continue
            task_infos.append(task_info)
            cancel_managed_task(task_id)
        current_task = asyncio.current_task()
        pending_tasks = tuple(
            info.task
            for info in task_infos
            if info.task is not None and info.task is not current_task
        )
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._task_ids.clear()
        self._scheduled_streams.clear()
        self._flashback_task_ids.clear()
        self._flashback_results.clear()
        self._flashback_generations.clear()
        self._encoder_task_ids.clear()
        self._task_ids_by_task.clear()
        self._pending_messages.clear()
        self._pending_since.clear()
        self._last_message_at.clear()
        self._prompt_turns.clear()
        self._encoder_locks.clear()
        self._flashback_locks.clear()
        self._encoder_started_at = None
        if self._worker_started:
            try:
                await self.vector_worker.stop()
            except Exception as error:  # noqa: BLE001
                logger.warning(f"vNext Vector Worker 停止失败: {error}")
            finally:
                self._worker_started = False
        if self._schema_initialized:
            try:
                await self.schema.close()
            except Exception as error:  # noqa: BLE001
                logger.warning(f"vNext Schema 关闭失败: {error}")
            finally:
                self._schema_initialized = False
        self._initialized = False

    def observe_message(self, message: object) -> None:
        """记录新消息并安排该流的增量经历编码。"""
        stream_id = str(self._message_value(message, "stream_id") or "").strip()
        if not stream_id:
            return
        now = datetime.now(UTC)
        self._pending_since.setdefault(stream_id, now)
        self._last_message_at[stream_id] = now
        message_id = str(
            self._message_value(message, "message_id")
            or self._message_value(message, "id")
            or ""
        ).strip()
        if message_id:
            self._pending_messages.setdefault(stream_id, {})[message_id] = message
        if self._initialized:
            self._schedule_flashback_prefetch(stream_id)
        if stream_id in self._scheduled_streams:
            return
        self._scheduled_streams.add(stream_id)
        task_info = create_managed_task(
            self._flush_stream_task(stream_id),
            name=f"engram_vnext_encode_{stream_id[:16]}",
            daemon=True,
        )
        self._task_ids.add(task_info.task_id)
        if task_info.task is not None:
            self._task_ids_by_task[task_info.task] = task_info.task_id
        self._encoder_task_ids[stream_id] = task_info.task_id

    @staticmethod
    def _message_value(message: object, field: str) -> object:
        """Read a field from a framework Message or public API mapping."""
        if isinstance(message, Mapping):
            return message.get(field)
        return getattr(message, field, None)

    def _schedule_flashback_prefetch(self, stream_id: str) -> None:
        """Replace a stream's stale flashback prefetch with a new task."""
        generation = self._flashback_generations.get(stream_id, 0) + 1
        self._flashback_generations[stream_id] = generation
        self._flashback_results.pop(stream_id, None)
        previous_id = self._flashback_task_ids.pop(stream_id, None)
        if previous_id is not None:
            cancel_managed_task(previous_id)
        task_info = create_managed_task(
            self._prefetch_flashback_task(stream_id, generation),
            name=f"engram_vnext_flashback_{stream_id[:16]}",
            daemon=True,
        )
        self._flashback_task_ids[stream_id] = task_info.task_id
        self._task_ids.add(task_info.task_id)
        if task_info.task is not None:
            self._task_ids_by_task[task_info.task] = task_info.task_id

    async def _prefetch_flashback_task(
        self,
        stream_id: str,
        generation: int,
    ) -> tuple[object, ...]:
        """Run one pre-response flashback retrieval for a stream."""
        try:
            turn_index = self._prompt_turns.get(stream_id)
            if turn_index is None:
                turn_index = await self.flashback.next_turn_index(stream_id)
            self._prompt_turns[stream_id] = turn_index + 1
            candidates = await self._flashback_for_stream(
                stream_id,
                turn_index=turn_index,
                record_exposure=False,
            )
            current_task = asyncio.current_task()
            if current_task is not None:
                task_id = self._task_ids_by_task.get(current_task)
                if task_id is not None:
                    if self._flashback_task_ids.get(stream_id) == task_id and self._flashback_generations.get(stream_id) == generation:
                        self._flashback_results[stream_id] = (candidates, turn_index, generation)
            return candidates
        finally:
            self._forget_current_task()

    def _forget_current_task(self) -> None:
        """Remove the completed Owner task handle without touching newer work."""
        current_task = asyncio.current_task()
        if current_task is None:
            return
        task_id = self._task_ids_by_task.pop(current_task, None)
        if task_id is None:
            return
        self._task_ids.discard(task_id)
        for stream_id, encoder_task_id in tuple(self._encoder_task_ids.items()):
            if encoder_task_id == task_id:
                self._encoder_task_ids.pop(stream_id, None)
        for stream_id, prefetch_task_id in tuple(self._flashback_task_ids.items()):
            if prefetch_task_id == task_id:
                self._flashback_task_ids.pop(stream_id, None)

    async def consume_flashback_prefetch(self, stream_id: str) -> tuple[object, ...]:
        """Wait for and consume the latest pre-response flashback result."""
        if not self._initialized or not stream_id.strip():
            return ()
        lock = self._flashback_locks.setdefault(stream_id, asyncio.Lock())
        async with lock:
            stored = self._flashback_results.pop(stream_id, None)
            if stored is not None:
                result, turn_index, generation = stored
                if generation != self._flashback_generations.get(stream_id):
                    return ()
                return await self._commit_flashback_result(
                    stream_id,
                    result,
                    turn_index,
                )
            task_id = self._flashback_task_ids.get(stream_id)
            if task_id is None:
                return ()
            try:
                task_info = get_managed_task(task_id)
            except Exception:  # noqa: BLE001
                if self._flashback_task_ids.get(stream_id) == task_id:
                    self._flashback_task_ids.pop(stream_id, None)
                return ()
            task = task_info.task
            if task is None:
                return ()
            budget = self.config.vnext.flashback.latency_budget_ms / 1000
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=budget)
            except TimeoutError:
                cancel_managed_task(task_id)
                await asyncio.gather(task, return_exceptions=True)
                return ()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                return ()
            finally:
                if self._flashback_task_ids.get(stream_id) == task_id:
                    self._flashback_task_ids.pop(stream_id, None)
            if not isinstance(result, tuple):
                return ()
            stored = self._flashback_results.pop(stream_id, None)
            if stored is None:
                return ()
            stored_result, turn_index, generation = stored
            if generation != self._flashback_generations.get(stream_id):
                return ()
            return await self._commit_flashback_result(
                stream_id,
                stored_result,
                turn_index,
            )

    async def _commit_flashback_result(
        self,
        stream_id: str,
        candidates: tuple[object, ...],
        turn_index: int,
    ) -> tuple[object, ...]:
        """Record exposure only after a result is accepted for Prompt injection."""
        if not candidates:
            return ()
        try:
            await self.flashback.record_exposure(
                tuple(str(getattr(candidate, "memory_id")) for candidate in candidates),
                stream_id,
                turn_index,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                f"vNext Flashback exposure 记录失败 stream={stream_id}: {error}"
            )
            return ()
        return candidates

    async def _flush_stream_task(self, stream_id: str) -> None:
        """执行一流编码并检查压力触发。"""
        try:
            await self.encode_stream(stream_id)
            await self.run_pressure_sleep()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.warning(f"vNext 流编码失败 stream={stream_id}: {error}")
        finally:
            self._scheduled_streams.discard(stream_id)
            self._forget_current_task()

    async def encode_stream(self, stream_id: str) -> tuple[str, ...]:
        """按游标加载一流新消息，并在达到触发条件时编码。"""
        if not self._initialized or not stream_id.strip():
            return ()
        lock = self._encoder_locks.setdefault(stream_id, asyncio.Lock())
        async with lock:
            async with self.schema.database.session() as session:
                cursor = await session.get(CandidateEncoderCursorModel, stream_id)
            messages = await self._load_stream_messages(stream_id)
            adapted = self.runtime_adapter.message_batcher.adapt_messages(
                messages,
                stream_id=stream_id,
            )
            pending = self.encoder.pending_messages(adapted, cursor)
            if not pending:
                self._pending_since.pop(stream_id, None)
                return ()
            now = datetime.now(UTC)
            pending_since = self._pending_since.get(stream_id) or pending[0].time
            if not self.encoder.should_encode(len(pending), pending_since, now):
                return ()
            candidate_ids: list[str] = []
            batch_size = self.config.vnext.candidate_encoder.message_threshold
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                result = await self.encoder.encode(
                    stream_id,
                    batch,
                    self.encoder_producer,
                )
                candidate_ids.extend(result.candidate_ids)
                buffered = self._pending_messages.get(stream_id)
                if buffered is not None:
                    for message in batch:
                        buffered.pop(message.message_id, None)
                    if not buffered:
                        self._pending_messages.pop(stream_id, None)
            self._pending_since.pop(stream_id, None)
            return tuple(candidate_ids)

    async def _load_stream_messages(self, stream_id: str) -> list[object]:
        """Read messages after the stream cursor and merge the receive buffer."""
        async with self.schema.database.session() as session:
            cursor = await session.get(CandidateEncoderCursorModel, stream_id)
        start_time = cursor.last_processed_message_time.timestamp() if cursor else 0.0
        if self._encoder_started_at is not None:
            start_time = max(start_time, self._encoder_started_at.timestamp())
        end_time = datetime.now(UTC).timestamp()
        rows = await message_api.get_messages_by_time_in_chat_inclusive(
            stream_id,
            start_time,
            end_time,
            limit=0,
            limit_mode="earliest",
        )
        merged: dict[str, object] = {}
        for message in rows:
            message_id = str(
                self._message_value(message, "message_id")
                or self._message_value(message, "id")
                or ""
            ).strip()
            if message_id:
                merged[message_id] = message
        for message_id, message in self._pending_messages.get(stream_id, {}).items():
            merged.setdefault(message_id, message)
        return list(merged.values())

    async def flush_all_streams(self) -> tuple[str, ...]:
        """周期性刷新所有已知聊天流，覆盖最长等待触发。"""
        stream_ids = await stream_api.get_stream_ids_from_db()
        candidate_ids: list[str] = []
        for stream_id in stream_ids:
            try:
                candidate_ids.extend(await self.encode_stream(stream_id))
            except Exception as error:  # noqa: BLE001
                logger.warning(f"vNext 周期编码失败 stream={stream_id}: {error}")
        await self.run_pressure_sleep()
        return tuple(candidate_ids)

    async def run_pressure_sleep(self) -> bool:
        """在候选压力与安静窗口均满足时运行一次 Sleep Session。"""
        async with self._sleep_lock:
            pending_count = await self.sleep_agent.actionable_count()
            settings = self.config.vnext.sleep
            if pending_count < settings.pressure_threshold:
                return False
            now = datetime.now(UTC)
            if self._last_message_at and settings.quiet_period_minutes > 0:
                latest = max(self._last_message_at.values())
                if now - latest < timedelta(minutes=settings.quiet_period_minutes):
                    return False
            await self._run_sleep_and_review(SleepTriggerType.PRESSURE)
            return True

    async def run_daily_sleep(self) -> bool:
        """在每日调度点整理当前待处理候选。"""
        async with self._sleep_lock:
            if await self.sleep_agent.actionable_count() <= 0:
                return False
            await self._run_sleep_and_review(SleepTriggerType.DAILY)
            return True

    async def _run_sleep_and_review(self, trigger_type: SleepTriggerType) -> None:
        """完成 Sleep consolidation 后运行受控的 Persona Review。"""
        status = await self.sleep_agent.run_session(
            trigger_type,
            self.sleep_producer,
        )
        session_id = self.sleep_agent.last_session_id
        if status not in {
            SleepSessionStatus.COMPLETED,
            SleepSessionStatus.PARTIAL,
        } or not session_id:
            return
        try:
            await self._review_personas(
                session_id,
                self.sleep_agent.last_changed_memory_ids,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"vNext Persona Review 失败 session={session_id}: {error}")

    async def _review_personas(
        self,
        sleep_session_id: str,
        changed_memory_ids: tuple[str, ...],
    ) -> None:
        """按本轮有意义变化的人物执行 Persona Review。"""
        if not changed_memory_ids:
            return
        person_memories = await self._person_memory_ids(changed_memory_ids)
        context = ToolContext(
            actor_type=ActorType.SLEEP_AGENT,
            sleep_session_id=sleep_session_id,
        )
        for person_id, memory_ids in person_memories.items():
            lookup = await self.tools.person_lookup(person_id, context)
            all_memory_ids = await self._all_person_memory_ids(person_id)
            changed_memories = tuple(
                await self.tools.memory_read(memory_id, "full", context)
                for memory_id in memory_ids
            )
            all_memories = tuple(
                await self.tools.memory_read(memory_id, "full", context)
                for memory_id in all_memory_ids
            )
            review = await self.persona_producer.produce(
                repr(
                    {
                        "person_id": person_id,
                        "current_persona": lookup.get("persona_impression"),
                        "recent_memories": lookup.get("recent_memories"),
                        "changed_memories": changed_memories,
                        "all_memories": all_memories,
                    }
                )
            )
            if review is None:
                continue
            await self.tools.person_impression_update(
                PersonaUpdateInput(
                    person_id=person_id,
                    impression_text=review["impression_text"],
                    reason=review["reason"],
                    memory_ids=all_memory_ids,
                    sleep_session_id=sleep_session_id,
                ),
                context,
            )

    async def _person_memory_ids(
        self,
        memory_ids: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        """从本轮 ACTIVE Memory 的当前 Subject/Participant 提取人物。"""
        async with self.schema.database.session() as session:
            subject_rows = tuple(
                (
                    await session.execute(
                        select(
                            MemoryModel.memory_id,
                            MemoryRevisionSubjectModel.person_id,
                        )
                        .join(
                            MemoryRevisionSubjectModel,
                            MemoryRevisionSubjectModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(
                            MemoryModel.memory_id.in_(memory_ids),
                            MemoryModel.status == MemoryStatus.ACTIVE,
                            MemoryRevisionSubjectModel.person_id.is_not(None),
                        )
                    )
                ).all()
            )
            participant_rows = tuple(
                (
                    await session.execute(
                        select(
                            MemoryModel.memory_id,
                            MemoryRevisionParticipantModel.person_id,
                        )
                        .join(
                            MemoryRevisionParticipantModel,
                            MemoryRevisionParticipantModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(
                            MemoryModel.memory_id.in_(memory_ids),
                            MemoryModel.status == MemoryStatus.ACTIVE,
                            MemoryRevisionParticipantModel.person_id.is_not(None),
                        )
                    )
                ).all()
            )
        grouped: dict[str, set[str]] = {}
        for memory_id, person_id in subject_rows + participant_rows:
            if person_id:
                grouped.setdefault(person_id, set()).add(memory_id)
        return {
            person_id: tuple(sorted(ids))
            for person_id, ids in sorted(grouped.items())
        }

    async def _all_person_memory_ids(self, person_id: str) -> tuple[str, ...]:
        """读取人物关联的全部 ACTIVE Formal Memory，不受近期索引限制。"""
        async with self.schema.database.session() as session:
            subject_ids = set(
                (
                    await session.scalars(
                        select(MemoryModel.memory_id)
                        .join(
                            MemoryRevisionSubjectModel,
                            MemoryRevisionSubjectModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(
                            MemoryRevisionSubjectModel.person_id == person_id,
                            MemoryModel.status == MemoryStatus.ACTIVE,
                        )
                    )
                ).all()
            )
            participant_ids = set(
                (
                    await session.scalars(
                        select(MemoryModel.memory_id)
                        .join(
                            MemoryRevisionParticipantModel,
                            MemoryRevisionParticipantModel.revision_id
                            == MemoryModel.current_revision_id,
                        )
                        .where(
                            MemoryRevisionParticipantModel.person_id == person_id,
                            MemoryModel.status == MemoryStatus.ACTIVE,
                        )
                    )
                ).all()
            )
        return tuple(sorted(subject_ids | participant_ids))

    async def recent_turns_for_flashback(
        self,
        stream_id: str,
    ) -> tuple[str, ...]:
        """从公共 API 合并持久消息与未落库消息并读取最近多轮文本。"""
        limit = self.config.vnext.flashback.context_turns
        messages = await stream_api.get_stream_messages(stream_id, limit=limit)
        merged: dict[str, object] = {}
        for index, message in enumerate(messages):
            message_id = str(
                self._message_value(message, "message_id")
                or self._message_value(message, "id")
                or f"__stream_{index}"
            ).strip()
            merged[message_id] = message
        for message_id, message in self._pending_messages.get(stream_id, {}).items():
            merged[message_id] = message

        def sort_key(message: object) -> tuple[datetime, str]:
            """Return a stable UTC ordering key for public message values."""
            value = self._message_value(message, "time")
            if isinstance(value, datetime):
                timestamp = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                timestamp = datetime.fromtimestamp(float(value), tz=UTC)
            elif isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    parsed = datetime.min.replace(tzinfo=UTC)
                timestamp = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            else:
                timestamp = datetime.min.replace(tzinfo=UTC)
            return timestamp, str(
                self._message_value(message, "message_id")
                or self._message_value(message, "id")
                or ""
            )

        turns: list[str] = []
        for message in sorted(merged.values(), key=sort_key)[-limit:]:
            text = str(
                self._message_value(message, "processed_plain_text")
                or self._message_value(message, "content")
                or ""
            ).strip()
            if not text:
                continue
            speaker = str(
                self._message_value(message, "sender_name")
                or self._message_value(message, "sender_id")
                or "未知"
            ).strip()
            turns.append(f"{speaker}: {text}")
        return tuple(turns)

    async def flashback_for_stream(self, stream_id: str) -> tuple[object, ...]:
        """在当前回复前运行统一 vNext Flashback 检索。"""
        if not self._initialized or not stream_id.strip():
            return ()
        turn_index = self._prompt_turns.get(stream_id)
        if turn_index is None:
            turn_index = await self.flashback.next_turn_index(stream_id)
        self._prompt_turns[stream_id] = turn_index + 1
        return await self._flashback_for_stream(
            stream_id,
            turn_index=turn_index,
            record_exposure=True,
        )

    async def _flashback_for_stream(
        self,
        stream_id: str,
        *,
        turn_index: int,
        record_exposure: bool,
    ) -> tuple[object, ...]:
        """Run flashback retrieval with explicit exposure recording semantics."""
        turns = await self.recent_turns_for_flashback(stream_id)
        return await self.flashback.flashback(
            turns,
            enabled=self.config.vnext.flashback.enabled,
            stream_key=stream_id,
            turn_index=turn_index,
            record_exposure=record_exposure,
        )
