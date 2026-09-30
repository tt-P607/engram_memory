"""Engram Memory vNext 运行时 Owner。

本模块把 vNext 的规范数据库、领域服务、派生向量索引和后台任务
绑定到一个插件实例，保证框架组件不会各自创建隐藏的状态源。
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy import func, select

from src.app.plugin_system.api import log_api, message_api, stream_api

from .candidate_service import CandidateService, SleepSessionService
from .doctor_service import DoctorService
from .domain import PersonaUpdateInput, SleepSessionInput
from .enums import (
    ActorType,
    CandidateStatus,
    MemoryStatus,
    OutboxObjectType,
    OutboxOperation,
    OutboxStatus,
    SleepSessionStatus,
    SleepTriggerType,
    VectorIndexStatus,
)
from .experience_encoder import EncoderMessage, ExperienceEncoder
from .flashback_service import FlashbackService
from .framework_bridge import (
    cancel_managed_task,
    create_managed_task,
    get_managed_task,
    read_message_snapshots,
)
from .models import (
    CandidateModel,
    CandidateEncoderCursorModel,
    MemoryModel,
    MemoryRetrievalEntryModel,
    MemoryRevisionParticipantModel,
    MemoryRevisionSubjectModel,
    PersonaUpdateLogModel,
    PersonaUpdateMemoryModel,
    SchemaVersionModel,
    VectorIndexManifestModel,
    VectorOutboxModel,
)
from .repository import MemoryRepository
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
from .schema import SCHEMA_KEY, SCHEMA_VERSION, VNextSchema
from .sleep_agent import SleepAgentOrchestrator
from .tool_service import ToolContext, VNextToolService
from .vector_service import VectorIndexService


logger = log_api.get_logger(
    "engram_memory.vnext.runtime_owner", display="Engram 记忆", color=log_api.COLOR.CYAN
)

_ENCODER_CONTEXT_LOOKBEHIND_LIMIT = 6
_ENCODER_CONTEXT_NEIGHBOR_COUNT = 2


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

    async def query_scored(
        self,
        texts: tuple[tuple[str, str], ...],
        top_k: int,
    ) -> tuple[tuple[str, float | None], ...]:
        """读取ACTIVE索引的排位与真实语义相似度。"""
        if not texts or self._schema is None:
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
        results = await sink.query_scored_entries(texts[0][1], top_k)
        return tuple((entry_id, score) for entry_id, score in results if entry_id in allowed_ids)


class VNextRuntimeOwner:
    """持有插件范围内唯一的 vNext 运行时资源。"""

    def __init__(self, plugin: Any) -> None:
        """根据插件配置装配 vNext 资源，但不执行外部 I/O。"""
        from ..config import EngramMemoryConfig

        self.plugin = plugin
        if not isinstance(plugin.config, EngramMemoryConfig):
            raise TypeError("Engram Runtime 必须使用已加载的插件配置")
        self.config: EngramMemoryConfig = plugin.config
        config = self.config
        vnext = config.vnext
        self.schema = VNextSchema(config.storage.vnext_db_path)
        self.repository = MemoryRepository(self.schema)
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
            on_actor_memory_changed=self._process_changed_memory_vectors,
        )
        self.candidate_service = CandidateService(self.schema)
        # Sleep Session is a plugin-scoped coordination service.  Keeping one
        # instance here prevents recovery and action auditing from observing
        # different runtime state.
        self.sleep_service = SleepSessionService(self.schema)
        self._automatic_since: datetime | None = None
        if vnext.sleep.automatic_since:
            boundary = datetime.fromisoformat(vnext.sleep.automatic_since.replace("Z", "+00:00"))
            if boundary.tzinfo is None:
                raise ValueError("自动整理起始时间必须包含时区")
            self._automatic_since = boundary.astimezone(UTC)
        self.encoder_producer = ExperienceEncoderDraftProducer(
            model_task=config.internal_llm.task_name,
        )
        self.sleep_producer = SleepAgentStepProducer(
            prompt_version="sleep-agent-v3-balanced",
        )
        self.persona_producer = PersonaReviewProducer(
            prompt_version="persona-review-v1",
            max_length=vnext.persona.max_length,
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
            prompt_version="experience-encoder-v4-context",
        )
        self.sleep_agent = SleepAgentOrchestrator(
            self.schema,
            self.tools,
            sleep_service=self.sleep_service,
            batch_size=vnext.sleep.batch_size,
            model_id=config.internal_llm.task_name,
            prompt_version="sleep-agent-v3-balanced",
            automatic_since=self._automatic_since,
        )
        self.vector_index = VectorIndexService(
            self.schema,
            self.vector_sink,
            max_attempts=vnext.vector.worker_retry_limit,
        )
        self.vector_worker = VectorOutboxWorker(
            self.vector_index,
            batch_size=vnext.sleep.batch_size,
            poll_interval_seconds=10.0,
            retry_failed=False,
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
        self._flashback_trigger_turns: dict[str, tuple[int, bool]] = {}
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
        self._pressure_task_id: str | None = None
        self._encoder_backlog_blocked = False
        self._started_at = datetime.now(UTC)

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
                retrieval_schema_version="engram-vnext-2",
            )
            self._schema_initialized = True
            await self.schema.initialize()
            await self.sleep_service.recover_abandoned_sessions(automatic_since=self._automatic_since)
            requeued = await self.sleep_service.requeue_failed_candidates(
                self.config.vnext.sleep.batch_size,
                automatic_since=self._automatic_since,
            )
            if requeued:
                logger.info(f"已重新排入 {len(requeued)} 条未形成动作计划的失败候选")
            await self.vector_index.ensure_active_manifest(
                embedding_identity,
                embedding_dimension,
                "engram-vnext-2",
            )
            self._worker_started = True
            self.vector_worker.start()
            self._encoder_started_at = datetime.now(UTC)
            self._initialized = True
            if await self.sleep_agent.actionable_count() > 0:
                self._schedule_startup_pressure_sleep()
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
        self._flashback_trigger_turns.clear()
        self._encoder_task_ids.clear()
        self._task_ids_by_task.clear()
        self._pressure_task_id = None
        self._pending_messages.clear()
        self._encoder_backlog_blocked = False
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
        self._last_message_at[stream_id] = now
        if self._initialized:
            self._schedule_flashback_prefetch(stream_id)
        if not self._candidate_chat_type_enabled(
            self._message_value(message, "chat_type")
        ):
            return
        self._pending_since.setdefault(stream_id, now)
        message_id = str(
            self._message_value(message, "message_id")
            or self._message_value(message, "id")
            or ""
        ).strip()
        if message_id:
            self._pending_messages.setdefault(stream_id, {})[message_id] = message
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

    def _candidate_chat_type_enabled(self, chat_type: object) -> bool:
        """Return whether candidate collection is enabled for a chat type."""
        normalized = str(chat_type or "").strip().lower()
        settings = self.config.vnext.candidate_encoder
        if normalized == "group":
            return settings.group_enabled
        if normalized == "private":
            return settings.private_enabled
        return settings.group_enabled and settings.private_enabled

    async def _candidate_stream_enabled(self, stream_id: str) -> bool:
        """Check candidate collection settings without loading stream messages."""
        settings = self.config.vnext.candidate_encoder
        if settings.group_enabled and settings.private_enabled:
            return True
        if not settings.group_enabled and not settings.private_enabled:
            return False
        stream = await stream_api.get_stream(stream_id)
        if stream is not None:
            return self._candidate_chat_type_enabled(
                self._message_value(stream, "chat_type")
            )
        if settings.group_enabled:
            group_stream_ids = await stream_api.get_stream_ids_from_db("group")
            if stream_id in group_stream_ids:
                return True
        if settings.private_enabled:
            private_stream_ids = await stream_api.get_stream_ids_from_db("private")
            if stream_id in private_stream_ids:
                return True
        return False

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
            self._prompt_turns.setdefault(stream_id, turn_index)
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
        if self._pressure_task_id == task_id:
            self._pressure_task_id = None
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
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                # A newer message can cancel the prefetch without cancelling
                # the Prompt that was waiting for it.
                return ()
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
        self._prompt_turns[stream_id] = max(
            self._prompt_turns.get(stream_id, turn_index), turn_index + 1
        )
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
        logger.info(f"vNext 闪回 | 为当前回复准备了 {len(candidates)} 条相关记忆")
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
        if not await self._candidate_stream_enabled(stream_id):
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
            pending_since = min(
                self._pending_since.get(stream_id) or pending[0].time,
                pending[0].time,
            )
            if not self.encoder.should_encode(len(pending), pending_since, now):
                return ()
            boundary_context = await self._load_encoder_boundary_context(
                stream_id,
                pending[0],
            )
            timeline_by_id = {message.message_id: message for message in boundary_context}
            timeline_by_id.update({message.message_id: message for message in adapted})
            timeline = tuple(
                sorted(timeline_by_id.values(), key=lambda message: message.sort_key())
            )
            timeline_indexes = {
                message.message_id: index
                for index, message in enumerate(timeline)
            }
            candidate_ids: list[str] = []
            batch_size = self.config.vnext.candidate_encoder.message_threshold
            backlog_blocked = False
            remaining_since: datetime | None = None
            for start in range(0, len(pending), batch_size):
                if await self._encoder_backlog_exceeded():
                    backlog_blocked = True
                    break
                batch = pending[start : start + batch_size]
                if not self.encoder.should_encode(len(batch), batch[0].time, now):
                    remaining_since = batch[0].time
                    break
                context_messages = self._encoder_context_for_batch(
                    batch,
                    tuple(pending[:start]),
                    boundary_context,
                    timeline,
                    timeline_by_id,
                    timeline_indexes,
                )
                missing_reply_targets = self._missing_encoder_reply_targets(
                    batch,
                    timeline_by_id,
                )
                if missing_reply_targets:
                    snapshots = await read_message_snapshots(
                        tuple((stream_id, message_id) for message_id in missing_reply_targets)
                    )
                    reply_targets = self.runtime_adapter.message_batcher.adapt_messages(
                        tuple(snapshot.to_dict() for snapshot in snapshots),
                        stream_id=stream_id,
                    )
                    current_keys = {
                        (message.stream_id, message.message_id)
                        for message in batch
                    }
                    context_by_key = {
                        (message.stream_id, message.message_id): message
                        for message in context_messages
                    }
                    context_by_key.update(
                        {
                            (message.stream_id, message.message_id): message
                            for message in reply_targets
                            if (message.stream_id, message.message_id) not in current_keys
                        }
                    )
                    context_messages = tuple(
                        sorted(context_by_key.values(), key=lambda message: message.sort_key())
                    )
                started_at = perf_counter()
                logger.info(f"[cyan]经历编码[/cyan] 开始：原始消息={len(batch)}，模型调用=1")
                result = await self.encoder.encode(
                    stream_id,
                    batch,
                    self.encoder_producer,
                    context_messages=context_messages,
                )
                candidate_ids.extend(result.candidate_ids)
                logger.info(
                    f"[green]经历编码完成[/green] 候选={len(result.candidate_ids)} "
                    f"耗时={perf_counter() - started_at:.1f}s，引用来源已保存"
                )
                buffered = self._pending_messages.get(stream_id)
                if buffered is not None:
                    for message in batch:
                        buffered.pop(message.message_id, None)
                    if not buffered:
                        self._pending_messages.pop(stream_id, None)
            if remaining_since is not None:
                self._pending_since[stream_id] = remaining_since
            elif not backlog_blocked:
                self._pending_since.pop(stream_id, None)
            return tuple(candidate_ids)

    @staticmethod
    def _missing_encoder_reply_targets(
        batch: tuple[EncoderMessage, ...],
        messages_by_id: Mapping[str, EncoderMessage],
    ) -> tuple[str, ...]:
        """Return exact reply target IDs not present in the bounded message window."""
        target_ids = tuple(
            dict.fromkeys(
                reply_to.strip()
                for message in batch
                if isinstance((reply_to := (message.snapshot or {}).get("reply_to")), str)
                and reply_to.strip()
                and reply_to.strip() not in messages_by_id
            )
        )
        return target_ids

    async def _load_encoder_boundary_context(
        self,
        stream_id: str,
        first_pending: EncoderMessage,
    ) -> tuple[EncoderMessage, ...]:
        """Load a small pre-cursor window without changing pending messages."""
        rows = await message_api.get_messages_before_time_in_chat(
            stream_id,
            first_pending.time.timestamp() + 0.000001,
            limit=_ENCODER_CONTEXT_LOOKBEHIND_LIMIT,
        )
        adapted = self.runtime_adapter.message_batcher.adapt_messages(
            rows,
            stream_id=stream_id,
        )
        first_key = first_pending.sort_key()
        return tuple(message for message in adapted if message.sort_key() < first_key)

    @staticmethod
    def _encoder_context_for_batch(
        batch: tuple[EncoderMessage, ...],
        preceding_pending: tuple[EncoderMessage, ...],
        boundary_context: tuple[EncoderMessage, ...],
        timeline: tuple[EncoderMessage, ...],
        messages_by_id: Mapping[str, EncoderMessage],
        timeline_indexes: Mapping[str, int],
    ) -> tuple[EncoderMessage, ...]:
        """Select a short prior window and available reply targets with neighbors."""
        current = batch
        prior = preceding_pending[-_ENCODER_CONTEXT_NEIGHBOR_COUNT:]
        if not prior:
            prior = boundary_context[-_ENCODER_CONTEXT_NEIGHBOR_COUNT:]
        current_keys = {(message.stream_id, message.message_id) for message in current}
        selected = {
            (message.stream_id, message.message_id): message
            for message in prior
            if (message.stream_id, message.message_id) not in current_keys
        }
        for message in current:
            snapshot = message.snapshot or {}
            reply_to = snapshot.get("reply_to")
            target_id = reply_to.strip() if isinstance(reply_to, str) else ""
            target = messages_by_id.get(target_id)
            if target is None or target.sort_key() >= message.sort_key():
                continue
            target_index = timeline_indexes.get(target.message_id)
            if target_index is None:
                continue
            for neighbor_index in range(
                max(0, target_index - 1),
                min(len(timeline), target_index + 2),
            ):
                neighbor = timeline[neighbor_index]
                key = (neighbor.stream_id, neighbor.message_id)
                if key not in current_keys:
                    selected[key] = neighbor
        return tuple(sorted(selected.values(), key=lambda message: message.sort_key()))

    async def _backlog_count(self) -> int:
        """统计自动处理范围内所有尚未完成的候选。"""
        async with self.schema.database.session() as session:
            statement = (
                select(func.count())
                .select_from(CandidateModel)
                .where(
                    CandidateModel.status.in_(
                        (
                            CandidateStatus.PENDING, CandidateStatus.PROCESSING,
                            CandidateStatus.DEFERRED, CandidateStatus.FAILED,
                        )
                    )
                )
            )
            if self._automatic_since is not None:
                statement = statement.where(CandidateModel.created_at >= self._automatic_since)
            count = await session.scalar(statement)
        return int(count or 0)

    async def _encoder_backlog_exceeded(self) -> bool:
        """Pause encoder model calls while actionable backlog exceeds its limit."""
        backlog_count = await self._backlog_count()
        pending_limit = self.config.vnext.candidate_encoder.pending_limit
        blocked = backlog_count > pending_limit
        if blocked != self._encoder_backlog_blocked:
            if blocked:
                logger.warning(
                    "vNext Candidate 积压超过上限，暂停经历编码 "
                    f"(count={backlog_count}, limit={pending_limit})"
                )
            else:
                logger.info(
                    "vNext Candidate 积压已回落，恢复经历编码 "
                    f"(count={backlog_count}, limit={pending_limit})"
                )
            self._encoder_backlog_blocked = blocked
        return blocked

    async def _load_stream_messages(self, stream_id: str) -> list[object]:
        """Read messages after the stream cursor and merge the receive buffer."""
        async with self.schema.database.session() as session:
            cursor = await session.get(CandidateEncoderCursorModel, stream_id)
        if cursor is not None:
            start_time = cursor.last_processed_message_time.timestamp()
        elif self._automatic_since is not None:
            start_time = self._automatic_since.timestamp()
        elif self._encoder_started_at is not None:
            start_time = self._encoder_started_at.timestamp()
        else:
            start_time = 0.0
        if self._automatic_since is not None:
            start_time = max(start_time, self._automatic_since.timestamp())
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
            merged[message_id] = message
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

    def _schedule_startup_pressure_sleep(self) -> None:
        """Arrange one tracked background pressure check after initialization."""
        if not self._initialized or self._pressure_task_id is not None:
            return
        task_info = create_managed_task(
            self._startup_pressure_sleep_task(),
            name="engram_vnext_startup_pressure_sleep",
            daemon=True,
        )
        self._pressure_task_id = task_info.task_id
        self._task_ids.add(task_info.task_id)
        if task_info.task is not None:
            self._task_ids_by_task[task_info.task] = task_info.task_id

    async def _startup_pressure_sleep_task(self) -> None:
        """Run startup backlog work outside the plugin initialization path."""
        try:
            await self.run_pressure_sleep()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.warning(f"vNext 启动压力整理失败: {error}")
        finally:
            self._forget_current_task()

    async def run_pressure_sleep(self) -> bool:
        """在压力与安静窗口满足时排空本轮可处理的 Pending 候选。"""
        async with self._sleep_lock:
            settings = self.config.vnext.sleep
            initial_pending = await self.sleep_agent.pending_count()
            if initial_pending < settings.pressure_threshold:
                return False
            now = datetime.now(UTC)
            latest = max(self._last_message_at.values(), default=self._started_at)
            if now - latest < timedelta(minutes=settings.quiet_period_minutes):
                return False
            candidate_ids = await self.sleep_agent.pending_candidates(
                limit=await self.sleep_agent.actionable_count(),
            )
            ran_session = False
            for start in range(0, len(candidate_ids), settings.batch_size):
                latest = max(self._last_message_at.values(), default=self._started_at)
                if datetime.now(UTC) - latest < timedelta(minutes=settings.quiet_period_minutes):
                    break
                status = await self._run_sleep_and_review(
                    SleepTriggerType.PRESSURE,
                    candidate_ids=candidate_ids[start:start + settings.batch_size],
                )
                if status is None:
                    break
                ran_session = True
                if status not in {SleepSessionStatus.COMPLETED, SleepSessionStatus.PARTIAL}:
                    break
            return ran_session

    async def run_daily_sleep(self) -> bool:
        """在每日调度点整理当前待处理候选。"""
        async with self._sleep_lock:
            initial_pending = await self.sleep_agent.actionable_count()
            initial_candidate_ids = await self.sleep_agent.pending_candidates(
                limit=initial_pending
            ) if initial_pending else ()
            if not initial_candidate_ids:
                memory_ids = await self._unreviewed_memory_ids()
                if not memory_ids:
                    return False
                started_at = perf_counter()
                persona_model_calls = [0]
                status = SleepSessionStatus.COMPLETED
                result = await self.sleep_service.start_session(
                    SleepSessionInput(
                        SleepTriggerType.DAILY, self.config.internal_llm.task_name,
                        "persona-review-v1",
                    ), (),
                )
                self.sleep_agent.last_session_id = result.sleep_session_id
                self.sleep_agent.last_changed_memory_ids = ()
                self.sleep_agent.last_llm_calls = 0
                self._record_sleep_trace("session_start", {
                    "candidate_count": 0, "changed_memory_ids": memory_ids,
                    "reason": "复核主动写入或修订的正式记忆",
                })
                try:
                    await self.sleep_service.finish_session(
                        result.sleep_session_id, status,
                    )
                    await self._review_personas(
                        result.sleep_session_id, memory_ids, persona_model_calls,
                    )
                    return True
                except Exception:
                    status = SleepSessionStatus.PARTIAL
                    raise
                finally:
                    self._record_sleep_trace("overall_summary", {
                        "trigger_type": SleepTriggerType.DAILY.value,
                        "scope": "persona_only",
                        "status": status.value,
                        "elapsed_seconds": perf_counter() - started_at,
                        "sleep_llm_calls": self.sleep_agent.last_llm_calls,
                        "persona_llm_calls": persona_model_calls[0],
                    })
            batch_size = self.config.vnext.sleep.batch_size
            ran_session = False
            for start in range(0, len(initial_candidate_ids), batch_size):
                candidate_ids = initial_candidate_ids[start : start + batch_size]
                status = await self._run_sleep_and_review(
                    SleepTriggerType.DAILY,
                    candidate_ids=candidate_ids,
                )
                if status not in {
                    SleepSessionStatus.COMPLETED,
                    SleepSessionStatus.PARTIAL,
                }:
                    break
                ran_session = True
            return ran_session

    async def _run_sleep_and_review(
        self,
        trigger_type: SleepTriggerType,
        *,
        candidate_ids: tuple[str, ...] | None = None,
    ) -> SleepSessionStatus | None:
        """完成 Sleep consolidation 后运行受控的 Persona Review。"""
        started_at = perf_counter()
        persona_model_calls = [0]
        status: SleepSessionStatus | None = None
        try:
            if candidate_ids is None:
                status = await self.sleep_agent.run_session(
                    trigger_type, self.sleep_producer, trace=self._record_sleep_trace,
                )
            else:
                status = await self.sleep_agent.run_session(
                    trigger_type, self.sleep_producer, candidate_ids=candidate_ids,
                    trace=self._record_sleep_trace,
                )
            session_id = self.sleep_agent.last_session_id
            if status not in {
                SleepSessionStatus.COMPLETED,
                SleepSessionStatus.PARTIAL,
            } or not session_id:
                return status
            await self._process_changed_memory_vectors(
                self.sleep_agent.last_changed_memory_ids
            )
            try:
                await self._review_personas(
                    session_id,
                    tuple(dict.fromkeys((
                        *self.sleep_agent.last_changed_memory_ids,
                        *await self._unreviewed_memory_ids(),
                    ))),
                    persona_model_calls,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(f"vNext Persona Review 失败 session={session_id}: {error}")
            return status
        finally:
            self._record_sleep_trace("overall_summary", {
                "trigger_type": trigger_type.value,
                "scope": "sleep_and_persona",
                "status": status.value if status is not None else None,
                "elapsed_seconds": perf_counter() - started_at,
                "sleep_llm_calls": self.sleep_agent.last_llm_calls,
                "persona_llm_calls": persona_model_calls[0],
            })

    def _record_sleep_trace(self, event: str, payload: dict[str, object]) -> None:
        """保存实际模型往返、工具结果、决定和执行后的来源回读。"""
        if not event.startswith(("persona_", "post_action_")) and event not in {
            "session_start", "session_end", "model_output", "model_error",
            "protocol_error", "group_decision", "action_start", "action_end",
            "group_start", "group_error", "model_input", "recent_formal_memories",
            "SEARCH", "MEMORY_READ", "EVIDENCE_READ", "self_check", "overall_summary",
            "search", "memory_read", "evidence_read", "source_already_covered",
            "unsafe_action_deferred", "unverified_target_deferred",
        }:
            return
        path = Path(self.config.storage.vnext_db_path).resolve().parent / "sleep-events.jsonl"
        data = dict(payload)
        if event == "model_output":
            data["raw_response"] = self.sleep_producer.last_raw_response
            data["response_metadata"] = self.sleep_producer.last_response_metadata
        if event == "persona_model_output":
            data["raw_response"] = self.persona_producer.last_raw_response
            data["response_metadata"] = self.persona_producer.last_response_metadata
        if event == "post_action_review_output":
            data["raw_response"] = self.sleep_producer.last_raw_response
            data["response_metadata"] = self.sleep_producer.last_response_metadata
        row = {
            "time": datetime.now(UTC).isoformat(),
            "sleep_session_id": self.sleep_agent.last_session_id,
            "event": event,
            "data": data,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    async def _unreviewed_memory_ids(self) -> tuple[str, ...]:
        """找出自动处理范围内仍有未审查正式变化的记忆。"""
        async with self.schema.database.session() as session:
            statement = select(MemoryModel).where(MemoryModel.status == MemoryStatus.ACTIVE)
            if self._automatic_since is not None:
                statement = statement.where(MemoryModel.updated_at >= self._automatic_since)
            memories = (await session.scalars(statement)).all()
        grouped = await self._person_memory_ids(tuple(row.memory_id for row in memories))
        updated = {row.memory_id: row.updated_at for row in memories}
        pending: set[str] = set()
        async with self.schema.database.session() as session:
            schema_v3_applied_at = await session.scalar(
                select(SchemaVersionModel.applied_at).where(
                    SchemaVersionModel.schema_key == SCHEMA_KEY,
                    SchemaVersionModel.version == SCHEMA_VERSION,
                )
            )
            for person_id, memory_ids in grouped.items():
                if schema_v3_applied_at is None:
                    pending.update(memory_ids)
                    continue
                aliases = await self.repository.resolve_person_aliases(person_id)
                reviewed = dict((await session.execute(
                    select(PersonaUpdateMemoryModel.memory_id, func.max(PersonaUpdateLogModel.created_at))
                    .join(PersonaUpdateLogModel, PersonaUpdateLogModel.update_id == PersonaUpdateMemoryModel.update_id)
                    .where(
                        PersonaUpdateLogModel.person_id.in_(aliases),
                        PersonaUpdateLogModel.created_at > schema_v3_applied_at,
                        PersonaUpdateMemoryModel.memory_id.in_(memory_ids),
                    ).group_by(PersonaUpdateMemoryModel.memory_id)
                )).all())
                pending.update(
                    memory_id for memory_id in memory_ids
                    if memory_id not in reviewed or reviewed[memory_id] < updated[memory_id]
                )
        return tuple(sorted(pending))

    async def _process_changed_memory_vectors(
        self,
        memory_ids: tuple[str, ...],
    ) -> None:
        """Deliver pending vector entries created by the completed Sleep session."""
        if not memory_ids:
            return
        async with self.schema.database.session() as session:
            outbox_ids = tuple(
                (
                    await session.scalars(
                        select(VectorOutboxModel.outbox_id)
                        .join(
                            MemoryRetrievalEntryModel,
                            VectorOutboxModel.object_id
                            == MemoryRetrievalEntryModel.entry_id,
                        )
                        .where(
                            MemoryRetrievalEntryModel.memory_id.in_(memory_ids),
                            VectorOutboxModel.object_type
                            == OutboxObjectType.RETRIEVAL_ENTRY,
                            VectorOutboxModel.operation == OutboxOperation.UPSERT,
                            VectorOutboxModel.status == OutboxStatus.PENDING,
                        )
                        .order_by(
                            VectorOutboxModel.created_at,
                            VectorOutboxModel.outbox_id,
                        )
                    )
                ).all()
            )
        if not outbox_ids:
            return
        try:
            await self.vector_index.process_pending_outbox(
                limit=len(outbox_ids),
                outbox_ids=outbox_ids,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"vNext Sleep 新记忆向量投递失败: {error}")

    async def _review_personas(
        self,
        sleep_session_id: str,
        changed_memory_ids: tuple[str, ...],
        model_call_count: list[int],
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
            review_started_at = perf_counter()
            model_calls = 0
            person_aliases = await self.repository.resolve_person_aliases(person_id)
            lookup = await self.tools.person_lookup(person_id, context)
            if not lookup.get("core_person_id"):
                self._record_sleep_trace("persona_skipped", {
                    "person_id": person_id, "reason": "核心人物记录不存在",
                })
                continue
            self._record_sleep_trace("persona_review_start", {
                "person_id": person_id, "changed_memory_ids": memory_ids,
                "current_persona": lookup.get("persona_impression"),
            })
            logger.info(f"[bold cyan]Sleep[/bold cyan] 人物印象审查 | 关联正式记忆={len(memory_ids)}")
            recent_rows = lookup.get("recent_memories")
            recent_memory_ids = tuple(
                str(item["memory_id"])
                for item in recent_rows
                if isinstance(item, dict) and item.get("memory_id")
            ) if isinstance(recent_rows, (list, tuple)) else ()
            review_memory_ids = tuple(dict.fromkeys((*memory_ids, *recent_memory_ids)))
            review_memories = {
                memory_id: self._persona_current_view(
                    await self.tools.memory_read(memory_id, "full", context),
                    person_aliases,
                )
                for memory_id in review_memory_ids
            }
            available_memory_ids = set(review_memories)
            investigation: list[dict[str, object]] = []
            review_state: dict[str, object] = {
                "person_id": person_id,
                "current_persona": lookup.get("persona_impression"),
                "changed_memories": tuple(
                    review_memories[memory_id] for memory_id in memory_ids
                ),
                "recent_memories": tuple(
                    review_memories[memory_id]
                    for memory_id in recent_memory_ids
                ),
                "investigation": investigation,
                "max_length": self.config.vnext.persona.max_length,
            }
            review: dict[str, object] | None = None
            keep_reason: str | None = None
            for step_index in range(8):
                self._record_sleep_trace("persona_model_input", {
                    "step": step_index + 1, "state": review_state,
                })
                model_calls += 1
                model_call_count[0] += 1
                decision = await self.persona_producer.produce(repr(review_state))
                self._record_sleep_trace("persona_model_output", {
                    "step": step_index + 1, "decision": decision,
                })
                if decision is None:
                    decision = {"action": "KEEP", "reason": "本轮正式记忆未改变整体人物印象"}
                action = str(decision.get("action", "UPDATE")).strip().upper()
                if action in {"UPDATE", "KEEP"} and "proposed_review" not in review_state:
                    review_state["proposed_review"] = decision
                    review_state["final_check"] = (
                        "提交前逐句对照 target_source_messages：具体经历是否属于目标本人，"
                        "是否误读 Bot 建议或未完成计划；稳定风格、亲近关系和性格是否真的"
                        "有多个独立经历，而非旧画像或同一天的零散发言；性别是否确认。"
                        "删除无依据的随和、自嘲、孩子气、鲜活亲近等概括。正文须是克制的"
                        "第一人称整体认识，不复述生日、身体不适、请假或作品剧情等流水。"
                        "UPDATE 拟稿无误也须返回最终完整 UPDATE；KEEP 只代表原印象无需修正。"
                    )
                    self._record_sleep_trace("persona_final_check", {
                        "person_id": person_id, "proposal": decision,
                    })
                    continue
                if action == "KEEP":
                    keep_reason = str(decision.get("reason") or "维持现有印象")
                    break
                if action == "SEARCH":
                    query = decision.get("query")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("Persona Review SEARCH 缺少有效 query")
                    search_results = await self.tools.memory_search(
                        query.strip(),
                        context,
                        person_ids=person_aliases,
                        limit=min(5, self.config.vnext.retrieval.max_limit),
                    )
                    summaries: list[dict[str, object]] = []
                    for result in search_results:
                        memory_id = str(result.get("memory_id") or "").strip()
                        if (
                            not memory_id
                            or result.get("status") != MemoryStatus.ACTIVE.value
                        ):
                            continue
                        available_memory_ids.add(memory_id)
                        memory = self._persona_current_view(
                            await self.tools.memory_read(memory_id, "full", context),
                            person_aliases,
                        )
                        review_memories[memory_id] = memory
                        summaries.append(memory)
                    investigation.append(
                        {
                            "step": step_index + 1,
                            "action": "SEARCH",
                            "query": query.strip(),
                            "results": tuple(summaries),
                        }
                    )
                    self._record_sleep_trace("persona_search", investigation[-1])
                    continue
                if action == "READ":
                    memory_id = decision.get("memory_id")
                    if not isinstance(memory_id, str) or not memory_id.strip():
                        raise ValueError("Persona Review READ 缺少有效 memory_id")
                    memory_id = memory_id.strip()
                    if memory_id not in available_memory_ids:
                        investigation.append(
                            {
                                "step": step_index + 1,
                                "action": "READ",
                                "memory_id": memory_id,
                                "result": "拒绝：ID 不属于本人物初始读取或定向搜索结果",
                            }
                        )
                        continue
                    memory = self._persona_current_view(
                        await self.tools.memory_read(memory_id, "full", context),
                        person_aliases,
                    )
                    if not self._persona_memory_matches(memory, person_aliases):
                        available_memory_ids.discard(memory_id)
                        investigation.append(
                            {
                                "step": step_index + 1,
                                "action": "READ",
                                "memory_id": memory_id,
                                "result": "拒绝：当前主体和参与者均不匹配目标人物",
                            }
                        )
                        continue
                    review_memories[memory_id] = memory
                    investigation.append(
                        {
                            "step": step_index + 1,
                            "action": "READ",
                            "memory_id": memory_id,
                            "memory": memory,
                        }
                    )
                    self._record_sleep_trace("persona_read", investigation[-1])
                    continue
                if action == "UPDATE":
                    text = decision.get("impression_text")
                    if isinstance(text, str) and len(text.strip()) > self.config.vnext.persona.max_length:
                        investigation.append({
                            "step": step_index + 1, "action": "UPDATE",
                            "result": f"正文超过 {self.config.vnext.persona.max_length} 字，请重新凝练完整正文。",
                        })
                        self._record_sleep_trace("persona_length_rejected", investigation[-1])
                        continue
                    review = decision
                    break
                raise ValueError(f"Persona Review action 无效: {action}")
            if review is None:
                if keep_reason is not None:
                    await self.tools.person_impression_review_complete(
                        PersonaUpdateInput(
                            person_id=person_id,
                            impression_text=str(lookup.get("persona_impression") or ""),
                            reason=keep_reason, memory_ids=tuple(review_memories),
                            sleep_session_id=sleep_session_id,
                        ), context,
                    )
                self._record_sleep_trace("persona_review_end", {
                    "person_id": person_id, "changed": False,
                    "reason": keep_reason or "调查达到上限，留待下次审查",
                    "model_calls": model_calls,
                    "elapsed_seconds": perf_counter() - review_started_at,
                })
                logger.info(f"[bold cyan]Sleep[/bold cyan] 人物印象审查 | 保持现有印象 调用={model_calls}")
                continue
            impression_text = review.get("impression_text")
            reason = review.get("reason")
            if not isinstance(impression_text, str) or not isinstance(reason, str):
                raise ValueError("Persona Review UPDATE 缺少印象正文或理由")
            update_memory_ids = tuple(
                memory_id
                for memory_id in review_memories
                if memory_id in available_memory_ids
            )
            result = await self.tools.person_impression_update(
                PersonaUpdateInput(
                    person_id=person_id,
                    impression_text=impression_text,
                    reason=reason,
                    memory_ids=update_memory_ids,
                    sleep_session_id=sleep_session_id,
                ),
                context,
            )
            reread = await self.tools.person_lookup(person_id, context)
            self._record_sleep_trace("persona_review_end", {
                "person_id": person_id, "result": result,
                "reason": reason, "impression": reread.get("persona_impression"),
                "model_calls": model_calls,
                "elapsed_seconds": perf_counter() - review_started_at,
            })
            logger.info(f"[bold cyan]Sleep[/bold cyan] 人物印象审查 | 核心印象已回读 调用={model_calls}")

    @staticmethod
    def _persona_current_view(
        memory: Mapping[str, object], person_ids: tuple[str, ...],
    ) -> dict[str, object]:
        """按当前版本及目标人物发言提供依据，保留缺少来源的记忆引用。"""
        current = {key: value for key, value in memory.items() if key not in {"history", "events"}}
        summaries = memory.get("evidence_summary") or ()
        evidence_ids = {item["evidence_id"] for item in summaries if isinstance(item, Mapping)}
        current["evidence_metadata"] = tuple(
            item for item in (memory.get("evidence_metadata") or ())
            if isinstance(item, Mapping) and item.get("evidence_id") in evidence_ids
        )
        target_messages = tuple(
            message["snapshot"]
            for item in current["evidence_metadata"]
            for message in (item.get("messages") or ())
            if isinstance(message, Mapping)
            and message.get("source_status") == "AVAILABLE"
            and isinstance(message.get("snapshot"), Mapping)
            and message["snapshot"].get("person_id") in person_ids
            and message["snapshot"].get("speaker_kind") in {"ACCOUNT", "PERSON"}
            and not message["snapshot"].get("redacted")
        )
        current["target_source_messages"] = target_messages
        current["evidence_metadata"] = tuple(
            {
                "evidence_id": item.get("evidence_id"),
                "source_type": item.get("source_type"),
                "messages": tuple(
                    message for message in (item.get("messages") or ())
                    if isinstance(message, Mapping)
                    and message.get("snapshot") in target_messages
                ),
            }
            for item in current["evidence_metadata"]
        )
        current["evidence_summary"] = ()
        current["source_limitation"] = (
            "当前正文是已整理线索，人物印象中的判断仍须由 target_source_messages 支持；"
            "账号来源不证明真人身份，不能把关于他人的叙述或旧画像当作本人事实。"
        )
        subject = current.get("current_subject")
        is_subject = isinstance(subject, Mapping) and subject.get("person_id") in person_ids
        if not target_messages or not is_subject:
            revision = current.get("current_revision")
            current["current_revision"] = (
                {"revision_id": revision.get("revision_id")}
                if isinstance(revision, Mapping) else None
            )
            current["evidence_metadata"] = ()
            current["evidence_summary"] = ()
            current["source_limitation"] = (
                "缺少可核对的目标人物发言；旧记忆仍保留，但不能据此推断人物印象。"
                if not target_messages else
                "目标人物仅为参与者；只使用 target_source_messages，不能把主体经历归给目标人物。"
            )
        return current

    @staticmethod
    def _persona_memory_matches(
        memory: object,
        person_ids: tuple[str, ...],
    ) -> bool:
        """Check exact target-person association before adding a reviewed memory."""
        if not isinstance(memory, Mapping):
            return False
        if memory.get("status") != MemoryStatus.ACTIVE.value:
            return False
        allowed_ids = set(person_ids)
        subject = memory.get("current_subject")
        subject_person_id = (
            subject.get("person_id") if isinstance(subject, Mapping) else None
        )
        if (
            isinstance(subject_person_id, str)
            and subject_person_id in allowed_ids
        ):
            return True
        participants = memory.get("current_participants")
        return isinstance(participants, (tuple, list)) and any(
            isinstance(item, Mapping)
            and isinstance(item.get("person_id"), str)
            and item.get("person_id") in allowed_ids
            for item in participants
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
        raw_grouped: dict[str, set[str]] = {}
        for memory_id, person_id in subject_rows + participant_rows:
            if person_id:
                raw_grouped.setdefault(person_id, set()).add(memory_id)
        grouped: dict[str, set[str]] = {}
        for person_id, associated_memory_ids in raw_grouped.items():
            aliases = await self.repository.resolve_person_aliases(person_id)
            canonical_id = aliases[0]
            grouped.setdefault(canonical_id, set()).update(associated_memory_ids)
        return {
            person_id: tuple(sorted(ids))
            for person_id, ids in sorted(grouped.items())
        }

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
        settings = self.config.vnext.flashback
        if not settings.enabled or settings.max_memories == 0:
            return ()
        decision = self._flashback_trigger_turns.get(stream_id)
        if decision is None or decision[0] != turn_index:
            probability = settings.trigger_probability
            decision = (turn_index, probability >= 1.0 or random.random() < probability)
            self._flashback_trigger_turns[stream_id] = decision
        if not decision[1]:
            return ()
        turns = await self.recent_turns_for_flashback(stream_id)
        return await self.flashback.flashback(
            turns,
            enabled=self.config.vnext.flashback.enabled,
            stream_key=stream_id,
            turn_index=turn_index,
            record_exposure=record_exposure,
        )
