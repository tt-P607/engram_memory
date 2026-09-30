"""Engram Memory vNext 经历编码器。

按 Technical Spec §45-§51 实现从原始聊天消息到 Candidate 的编码职责：
按 Stream 游标增量拉取、双触发（消息数阈值 + 最长等待）、允许 0 输出、
Candidate 持久化与 Evidence 消息链接。编码器只负责圈出素材，
不拥有 memory_merge / memory_revise / persona 等整理能力（§139）。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .candidate_service import CandidateService
from .domain import (
    CandidateInput,
    EvidenceInput,
    EvidenceMessageInput,
    ParticipantInput,
    SubjectInput,
)
from .enums import EvidenceSourceType, MemoryKind
from .models import CandidateEncoderCursorModel
from .message_display import has_unseparated_reply_preview, split_reply_preview


@dataclass(frozen=True, slots=True)
class EncoderMessage:
    """编码器可见的单条原始消息。"""

    message_id: str
    stream_id: str
    time: datetime
    text: str
    speaker: str | None = None
    snapshot: dict[str, object] | None = None

    def sort_key(self) -> tuple[datetime, str]:
        """返回与游标一致的复合排序键。"""
        return (self.time, self.message_id)


@dataclass(frozen=True, slots=True)
class EncoderDraft:
    """编码 LLM 输出的候选草案。

    evidence_indexes 引用本批消息的下标（按时间升序），用于持久化
    MESSAGE_SET Evidence 的原始消息快照。
    """

    rough_title: str
    rough_content: str
    evidence_indexes: tuple[int, ...]
    proposed_kind: MemoryKind | None = None
    subject: SubjectInput | None = None
    participants: tuple[ParticipantInput, ...] = ()


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """一次编码后的持久化结果与游标推进位置。"""

    candidate_ids: tuple[str, ...]
    last_processed_message_id: str
    last_processed_message_time: datetime


class DraftProducer:
    """编码 LLM 调用协议：输入批文本，输出候选草案元组（允许为空）。

    同步签名保留用于兼容测试替身；生产实现可提供同名异步
    ``produce`` 方法，由编码器直接等待其结果。
    """

    def __call__(self, prompt_version: str, batch_text: str) -> tuple[EncoderDraft, ...]:
        """按 Prompt 版本编码批文本。"""
        raise NotImplementedError


class ExperienceEncoder:
    """管理经历编码的触发判定、批组织与 Candidate 持久化。"""

    def __init__(
        self,
        candidate_service: CandidateService,
        *,
        message_threshold: int,
        max_wait_minutes: int,
        prompt_version: str,
    ) -> None:
        """绑定候选服务并设置编码触发参数。

        参数:
            candidate_service: 候选素材领域服务。
            message_threshold: 触发编码的新消息数量阈值。
            max_wait_minutes: 未达阈值时最长等待分钟数。
            prompt_version: 编码 Prompt 显式版本号（§148）。
        """
        if message_threshold <= 0:
            raise ValueError("message_threshold 必须大于 0")
        if max_wait_minutes <= 0:
            raise ValueError("max_wait_minutes 必须大于 0")
        if not prompt_version.strip():
            raise ValueError("prompt_version 不能为空")
        self._candidates = candidate_service
        self._message_threshold = message_threshold
        self._max_wait = timedelta(minutes=max_wait_minutes)
        self._prompt_version = prompt_version

    @staticmethod
    def pending_messages(
        messages: tuple[EncoderMessage, ...],
        cursor: CandidateEncoderCursorModel | None,
    ) -> tuple[EncoderMessage, ...]:
        """筛选游标之后的新消息并按 (time, message_id) 升序排列。"""
        selectable = [message for message in messages if message.stream_id]
        if cursor is not None:
            cursor_key = (cursor.last_processed_message_time, cursor.last_processed_message_id)
            selectable = [
                message for message in selectable if message.sort_key() > cursor_key
            ]
        return tuple(sorted(selectable, key=EncoderMessage.sort_key))

    def should_encode(
        self,
        pending_count: int,
        cursor_updated_at: datetime | None,
        now: datetime,
    ) -> bool:
        """按消息数量阈值或最长等待时间判断是否触发编码。

        无新消息时不触发（不调用 LLM，§51）。
        """
        if pending_count >= self._message_threshold:
            return True
        if pending_count <= 0:
            return False
        if cursor_updated_at is None:
            return False
        reference = cursor_updated_at.astimezone(UTC) if cursor_updated_at.tzinfo else cursor_updated_at
        return now - reference >= self._max_wait

    def build_batch_text(
        self,
        batch: tuple[EncoderMessage, ...],
        *,
        context_messages: tuple[EncoderMessage, ...] = (),
    ) -> str:
        """保留新消息及有限上下文的原文与来源元数据。"""
        messages = self._input_messages(batch, context_messages)
        batch_keys = {(message.stream_id, message.message_id) for message in batch}
        has_context = any(
            (message.stream_id, message.message_id) not in batch_keys
            for message in messages
        )
        lines = []
        for index, message in enumerate(messages):
            snapshot = message.snapshot or {}
            speaker = (
                snapshot.get("sender_cardname")
                or snapshot.get("sender_name")
                or message.speaker
                or "未知"
            )
            person_id = snapshot.get("person_id")
            speaker_kind = (
                "BOT"
                if snapshot.get("speaker_is_bot") is True
                or str(snapshot.get("sender_role") or "").casefold() == "bot"
                or str(snapshot.get("speaker_kind") or "").upper() == "BOT"
                or isinstance(person_id, str) and person_id.strip().casefold() == "bot"
                else "ACCOUNT"
                if person_id
                else "UNKNOWN"
            )
            if speaker_kind == "BOT":
                speaker = f"Bot ({speaker})"
            sender_id = snapshot.get("sender_id")
            message_type = snapshot.get("message_type")
            reply_to = snapshot.get("reply_to")
            own_text, reply_preview = split_reply_preview(message.text, reply_to)
            reply_metadata = (
                f"reply_to={reply_to.strip()} "
                if isinstance(reply_to, str) and reply_to.strip()
                else ""
            )
            message_time = message.time
            if message_time.tzinfo is None:
                message_time = message_time.replace(tzinfo=UTC)
            time_text = message_time.astimezone().isoformat()
            scope_name = (
                "NEW"
                if (message.stream_id, message.message_id) in batch_keys
                else "CONTEXT"
            )
            scope = f" [{scope_name}]" if has_context else ""
            reply_preview_text = (
                f" reply_preview(非本次发言)={reply_preview}"
                if reply_preview is not None
                else ""
            )
            text_label = (
                "消息全文(引用边界未分离)"
                if has_unseparated_reply_preview(message.text, reply_to)
                else "本次发言"
            )
            lines.append(
                f"[{index}] [{time_text}]{scope} message_id={message.message_id} "
                f"stream_id={message.stream_id} person_id={person_id or 'unknown'} "
                f"{reply_metadata}"
                f"sender_id={sender_id or 'unknown'} message_type={message_type or 'unknown'} "
                f"speaker_kind={speaker_kind} "
                f"speaker={speaker} {text_label}={own_text}{reply_preview_text}"
            )
        return "\n".join(lines)

    async def encode(
        self,
        stream_id: str,
        batch: tuple[EncoderMessage, ...],
        draft_producer: DraftProducer,
        *,
        context_messages: tuple[EncoderMessage, ...] = (),
    ) -> EncodeResult:
        """编码一批新消息并持久化候选与游标。

        批为空时直接跳过（不调用 LLM）；草案为空合法（0 输出，§47），
        游标仍然推进到批末尾。
        """
        if not stream_id.strip():
            raise ValueError("stream_id 不能为空")
        if not batch:
            raise ValueError("编码批不能为空")
        for message in batch:
            if message.stream_id != stream_id:
                raise ValueError("批内消息 stream_id 必须一致")
        for message in context_messages:
            if message.stream_id != stream_id:
                raise ValueError("上下文消息 stream_id 必须一致")
        last = batch[-1]
        input_messages = self._input_messages(batch, context_messages)
        batch_keys = {(message.stream_id, message.message_id) for message in batch}
        drafts = await self._produce_drafts(
            draft_producer,
            self.build_batch_text(batch, context_messages=context_messages),
        )
        if not isinstance(drafts, tuple):
            raise ValueError("draft producer 必须返回 tuple")
        inputs = tuple(
            candidate_input
            for draft in drafts
            if self._draft_has_batch_evidence(draft, input_messages, batch_keys)
            for candidate_input in (self._to_candidate_input(draft, input_messages),)
        )
        candidate_ids = await self._candidates.create_candidates(
            inputs,
            cursor=(stream_id, last.time, last.message_id),
        )
        return EncodeResult(
            candidate_ids=candidate_ids,
            last_processed_message_id=last.message_id,
            last_processed_message_time=last.time,
        )

    @staticmethod
    def _input_messages(
        batch: tuple[EncoderMessage, ...],
        context_messages: tuple[EncoderMessage, ...],
    ) -> tuple[EncoderMessage, ...]:
        """Return a time-ordered, de-duplicated prompt and evidence window."""
        messages_by_key = {
            (message.stream_id, message.message_id): message
            for message in context_messages
        }
        messages_by_key.update(
            {
                (message.stream_id, message.message_id): message
                for message in batch
            }
        )
        return tuple(sorted(messages_by_key.values(), key=EncoderMessage.sort_key))

    @staticmethod
    def _draft_has_batch_evidence(
        draft: EncoderDraft,
        input_messages: tuple[EncoderMessage, ...],
        batch_keys: set[tuple[str, str]],
    ) -> bool:
        """Require every new candidate to cite at least one newly processed message."""
        if not draft.evidence_indexes:
            raise ValueError("草案必须引用至少一条消息证据")
        for index in draft.evidence_indexes:
            if index < 0 or index >= len(input_messages):
                raise ValueError(f"草案证据下标越界: {index}")
        return any(
            (input_messages[index].stream_id, input_messages[index].message_id)
            in batch_keys
            for index in draft.evidence_indexes
        )

    async def _produce_drafts(
        self,
        draft_producer: DraftProducer,
        batch_text: str,
    ) -> tuple[EncoderDraft, ...]:
        """优先 await 异步 Producer，兼容旧的同步测试 Producer。"""
        produce = getattr(draft_producer, "produce", None)
        if callable(produce):
            result = produce(self._prompt_version, batch_text)
            if inspect.isawaitable(result):
                result = await result
        else:
            result = draft_producer(self._prompt_version, batch_text)
        if not isinstance(result, tuple):
            raise ValueError("draft producer 必须返回 tuple")
        return result

    @staticmethod
    def _to_candidate_input(
        draft: EncoderDraft,
        batch: tuple[EncoderMessage, ...],
    ) -> CandidateInput:
        """把草案映射为领域候选输入并绑定消息证据。"""
        if not draft.evidence_indexes:
            raise ValueError("草案必须引用至少一条消息证据")
        referenced: list[EncoderMessage] = []
        for index in draft.evidence_indexes:
            if index < 0 or index >= len(batch):
                raise ValueError(f"草案证据下标越界: {index}")
            referenced.append(batch[index])
        earliest = min(referenced, key=EncoderMessage.sort_key)
        messages = tuple(
            EvidenceMessageInput(
                message_id=message.message_id,
                stream_id=message.stream_id,
                snapshot=message.snapshot,
            )
            for message in referenced
        )
        evidence = EvidenceInput(
            source_type=EvidenceSourceType.MESSAGE_SET,
            observed_at=earliest.time,
            messages=messages,
        )
        return CandidateInput(
            rough_title=draft.rough_title,
            rough_content=draft.rough_content,
            observed_at=earliest.time,
            evidence=(evidence,),
            proposed_kind=draft.proposed_kind,
            subject=draft.subject,
            participants=draft.participants,
        )
