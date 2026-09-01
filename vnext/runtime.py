"""Runtime adapters for Engram Memory vNext.

The module contains the small amount of runtime glue needed by the vNext
domain services: message normalization and batching, JSON-only LLM producers,
and a derived Chroma vector sink.  It deliberately does not register a plugin
or alter the canonical vNext services.  Callers can compose these objects from
the existing plugin entry point when runtime wiring is enabled.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

from src.app.plugin_system.api import llm_api, stream_api
from src.app.plugin_system.types import LLMPayload, Message, ModelSet, ROLE, Text

from .domain import SubjectInput, VectorUpsert
from .enums import (
    CandidateActionType,
    ClaimBasis,
    ConfidenceLevel,
    MemoryKind,
    SalienceLevel,
    SubjectKind,
)
from .experience_encoder import (
    DraftProducer,
    EncoderDraft,
    EncoderMessage,
    ExperienceEncoder,
)
from .framework_bridge import (
    ManagedTaskHandle,
    VectorDatabase,
    cancel_managed_task,
    create_managed_task,
    get_vector_database,
)
from .models import CandidateEncoderCursorModel
from .schema import VNextSchema
from .sleep_agent import AgentStepProducer
from .vector_service import VectorIndexService, VectorSink

E = TypeVar("E", bound=Enum)

DEFAULT_BATCH_SIZE: int = 12
DEFAULT_ENCODER_MODEL_TASK: str = "memory"
DEFAULT_SLEEP_MODEL_TASK: str = "memory_cleanup"
DEFAULT_EMBEDDING_MODEL_TASK: str = "embedding"
DEFAULT_VECTOR_COLLECTION: str = "engram_vnext_retrieval"
DEFAULT_ENCODER_REQUEST_NAME: str = "engram_vnext_experience_encoder"
DEFAULT_SLEEP_REQUEST_NAME: str = "engram_vnext_sleep_agent"
DEFAULT_EMBEDDING_REQUEST_NAME: str = "engram_vnext_embedding"
DEFAULT_VECTOR_DB_PATH: str = "data/chroma_db"

DEFAULT_ENCODER_SYSTEM_PROMPT: str = (
    "You are the Engram Memory vNext Experience Encoder.\n"
    "Identify only experiences that may have durable cognitive value.\n"
    "Ignore greetings, repetition, ordinary one-off questions, and unsupported\n"
    "personality guesses. Do not consolidate memory, revise memory, merge\n"
    "memory, or update persona.\n"
    "Return one strict JSON array and no other text. Returning [] is valid when\n"
    "nothing is worth retaining; never invent a candidate to fill a quota.\n"
    "Each candidate object must contain:\n"
    "{\"rough_title\": string, \"rough_content\": string,\n"
    " \"retention_reason\": string, \"evidence_indexes\": integer[],\n"
    " \"proposed_kind\": one of EVENT, FACT, PREFERENCE, RELATIONSHIP,\n"
    " COMMITMENT, PERSONAL_STATE, SOCIAL_PATTERN, or null,\n"
    " \"confidence_hint\": one of UNKNOWN, LOW, MEDIUM, HIGH, or null,\n"
    " \"salience_hint\": one of LOW, MEDIUM, HIGH, CORE, or null,\n"
    " \"uncertainty_note\": string|null, \"subject\": object|null,\n"
    " \"claim_basis\": one of DIRECT_STATEMENT, OBSERVED_BEHAVIOR,\n"
    " REPORTED_BY_OTHER, INFERRED, or null}.\n"
    "When subject is not null it contains subject_kind (PERSON, GROUP, SELF,\n"
    " CONCEPT, or UNKNOWN) plus nullable person_id, subject_key, and\n"
    " subject_label strings.\n"
    "evidence_indexes refer to the zero-based messages in the supplied batch."
)

DEFAULT_SLEEP_SYSTEM_PROMPT: str = (
    "You are the Engram Memory vNext Sleep Agent.\n"
    "Review the supplied Candidate as one bounded, tool-driven Agent. Return\n"
    "one strict JSON array containing one step object at a time. A step_type\n"
    "must be SEARCH, MEMORY_READ, EVIDENCE_READ, PERSON_LOOKUP, or DECIDE.\n"
    "For SEARCH provide query; for MEMORY_READ provide memory_id and view; for\n"
    "EVIDENCE_READ provide evidence_ids; for PERSON_LOOKUP provide person_id.\n"
    "DECIDE must provide actions, an array of final memory action intents. Each\n"
    "action_type must be CREATE_NEW, REINFORCE, REVISE, MERGE, RELATE, IGNORE,\n"
    "or DEFER. Do not execute tools or write memory in the step. Returning []\n"
    "is valid only when no safe action can be formed."
)
DEFAULT_PERSONA_REVIEW_REQUEST_NAME: str = "engram_vnext_persona_review"
DEFAULT_PERSONA_REVIEW_SYSTEM_PROMPT: str = (
    "You are the Engram Memory vNext Persona Review step.\n"
    "Review only the supplied Formal Memory evidence after a completed Sleep "
    "Session. Return a JSON array with zero or one object. Return [] when the "
    "long-term impression should remain unchanged. Otherwise return one object "
    "with action UPDATE, an abstract impression_text, and a concise reason. "
    "Do not output numeric personality scores, affection, trust, or closeness; "
    "do not copy a single event as a personality conclusion."
)

SLEEP_AGENT_STEP_TYPES: frozenset[str] = frozenset(
    {"SEARCH", "MEMORY_READ", "EVIDENCE_READ", "PERSON_LOOKUP", "DECIDE"}
)


class RuntimeAdapterError(RuntimeError):
    """Base exception for vNext runtime adapter failures."""


class RuntimeProducerError(RuntimeAdapterError):
    """Failure raised by a runtime producer rather than a valid empty result."""


class LLMProducerError(RuntimeProducerError):
    """Failure while making or consuming an LLM request."""


class LLMResponseFormatError(LLMProducerError):
    """Failure caused by an invalid structured LLM response."""


class VectorSinkError(RuntimeAdapterError):
    """Failure while creating or writing a derived vector entry."""


def _normalize_datetime(value: object) -> datetime:
    """Convert a supported message time value to an aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("message.time 不是有效的时间值") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        raise ValueError("message.time 必须是 datetime、Unix timestamp 或 ISO 文本")

    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError("message.time 必须是有限时间戳")
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _normalize_text(value: object) -> str:
    """Convert a message content value to trimmed human-readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


MessageLike = Message | Mapping[str, object]


def _message_value(message: MessageLike, field: str, default: object = None) -> object:
    """Read a field from either a public Message or a public API mapping."""
    if isinstance(message, Mapping):
        return message.get(field, default)
    return getattr(message, field, default)


def message_to_encoder_message(message: MessageLike) -> EncoderMessage:
    """Adapt one public plugin-system Message to an EncoderMessage.

    ``processed_plain_text`` is preferred over ``content``.  The adapter uses
    the sender name, card name, or sender ID as the speaker and normalizes all
    supported time values to UTC.  Missing stream, ID, or usable text is an
    error here; batch helpers treat those errors as filterable input.
    """
    if message is None:
        raise ValueError("message 不能为空")

    message_id = _normalize_text(
        _message_value(message, "message_id") or _message_value(message, "id")
    )
    stream_id = _normalize_text(_message_value(message, "stream_id"))
    text = _normalize_text(_message_value(message, "processed_plain_text")) or _normalize_text(
        _message_value(message, "content")
    )
    if not message_id:
        raise ValueError("message.message_id 不能为空")
    if not stream_id:
        raise ValueError("message.stream_id 不能为空")
    if not text:
        raise ValueError("message 必须包含有效文本")

    speaker = (
        _normalize_text(_message_value(message, "sender_name"))
        or _normalize_text(_message_value(message, "sender_cardname"))
        or _normalize_text(_message_value(message, "sender_id"))
        or None
    )
    return EncoderMessage(
        message_id=message_id,
        stream_id=stream_id,
        time=_normalize_datetime(_message_value(message, "time")),
        text=text,
        speaker=speaker,
    )


def adapt_message(message: MessageLike) -> EncoderMessage:
    """Alias for :func:`message_to_encoder_message`."""
    return message_to_encoder_message(message)


def messages_to_encoder_messages(
    messages: Iterable[MessageLike],
    *,
    stream_id: str | None = None,
) -> tuple[EncoderMessage, ...]:
    """Adapt an iterable of public messages without imposing a batch size."""
    return MessageBatcher().adapt_messages(messages, stream_id=stream_id)


def adapt_messages(
    messages: Iterable[MessageLike],
    *,
    stream_id: str | None = None,
) -> tuple[EncoderMessage, ...]:
    """Adapt an iterable of public messages to encoder messages."""
    return messages_to_encoder_messages(messages, stream_id=stream_id)


def _chunk_messages(
    messages: tuple[EncoderMessage, ...],
    batch_size: int,
) -> tuple[tuple[EncoderMessage, ...], ...]:
    """Split one already-sorted stream into fixed-size immutable batches."""
    return tuple(
        messages[index : index + batch_size]
        for index in range(0, len(messages), batch_size)
    )


class MessageBatcher:
    """Normalize public messages and produce stream-safe encoder batches."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Create a batcher with a positive maximum batch size."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Return the configured maximum number of messages per batch."""
        return self._batch_size

    @staticmethod
    def adapt_one(message: MessageLike) -> EncoderMessage | None:
        """Adapt one message, returning None for unusable input."""
        try:
            return message_to_encoder_message(message)
        except (TypeError, ValueError):
            return None

    def adapt_messages(
        self,
        messages: Iterable[MessageLike],
        *,
        stream_id: str | None = None,
    ) -> tuple[EncoderMessage, ...]:
        """Adapt, filter, and deterministically sort a message iterable."""
        selected_stream = stream_id.strip() if isinstance(stream_id, str) else None
        if selected_stream == "":
            selected_stream = None
        adapted: list[EncoderMessage] = []
        for message in messages:
            item = self.adapt_one(message)
            if item is None:
                continue
            if selected_stream is not None and item.stream_id != selected_stream:
                continue
            adapted.append(item)
        return tuple(
            sorted(
                adapted,
                key=lambda item: (item.stream_id, item.time, item.message_id),
            )
        )

    def batches(
        self,
        messages: Iterable[MessageLike],
        *,
        stream_id: str | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Create batches without mixing different chat streams."""
        adapted = self.adapt_messages(messages, stream_id=stream_id)
        grouped: dict[str, list[EncoderMessage]] = {}
        for item in adapted:
            grouped.setdefault(item.stream_id, []).append(item)

        result: list[tuple[EncoderMessage, ...]] = []
        for grouped_stream_id in sorted(grouped):
            stream_messages = tuple(grouped[grouped_stream_id])
            result.extend(_chunk_messages(stream_messages, self._batch_size))
        return tuple(result)

    def batch_messages(
        self,
        messages: Iterable[MessageLike],
        *,
        stream_id: str | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Alias for :meth:`batches`."""
        return self.batches(messages, stream_id=stream_id)

    def pending_batches(
        self,
        messages: Iterable[MessageLike],
        cursor: CandidateEncoderCursorModel | None,
        *,
        stream_id: str | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Filter an encoder cursor and split the remaining messages."""
        cursor_stream = cursor.stream_id if cursor is not None else None
        selected_stream = stream_id.strip() if isinstance(stream_id, str) else None
        if selected_stream == "":
            selected_stream = None
        if cursor_stream is not None:
            if selected_stream is not None and selected_stream != cursor_stream:
                raise ValueError("stream_id 必须与 cursor.stream_id 一致")
            selected_stream = cursor_stream
        if cursor is not None and selected_stream is None:
            raise ValueError("使用 cursor 时必须指定 stream_id")

        adapted = self.adapt_messages(messages, stream_id=selected_stream)
        pending = ExperienceEncoder.pending_messages(adapted, cursor)
        if selected_stream is not None:
            return _chunk_messages(pending, self._batch_size)

        grouped: dict[str, list[EncoderMessage]] = {}
        for item in pending:
            grouped.setdefault(item.stream_id, []).append(item)
        result: list[tuple[EncoderMessage, ...]] = []
        for grouped_stream_id in sorted(grouped):
            result.extend(
                _chunk_messages(tuple(grouped[grouped_stream_id]), self._batch_size)
            )
        return tuple(result)

    def batch_encoder_messages(
        self,
        messages: Iterable[MessageLike],
        *,
        stream_id: str | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Alias for :meth:`batches` with an encoder-oriented name."""
        return self.batches(messages, stream_id=stream_id)

    async def load_stream_messages(
        self,
        stream_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Message, ...]:
        """Load messages for one stream through the public stream API."""
        normalized_stream_id = stream_id.strip()
        if not normalized_stream_id:
            raise ValueError("stream_id 不能为空")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset 必须是非负整数")
        messages = await stream_api.get_stream_messages(
            normalized_stream_id,
            limit=limit,
            offset=offset,
        )
        return tuple(messages)

    async def load_stream_batches(
        self,
        stream_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        cursor: CandidateEncoderCursorModel | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Load, optionally cursor-filter, and batch one stream's messages."""
        messages = await self.load_stream_messages(
            stream_id,
            limit=limit,
            offset=offset,
        )
        if cursor is None:
            return self.batches(messages, stream_id=stream_id)
        return self.pending_batches(messages, cursor, stream_id=stream_id)


class RuntimeMessageAdapter(MessageBatcher):
    """Named runtime facade for message normalization and batching."""


def batch_messages(
    messages: Iterable[MessageLike],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stream_id: str | None = None,
) -> tuple[tuple[EncoderMessage, ...], ...]:
    """Adapt and split public messages using a configurable batch size."""
    return MessageBatcher(batch_size=batch_size).batches(
        messages,
        stream_id=stream_id,
    )


def _run_coroutine_in_thread(coroutine_factory: Callable[[], Any]) -> Any:
    """Run a coroutine factory inside a fresh event loop in its worker thread."""
    return asyncio.run(coroutine_factory())


def _run_coroutine_sync(coroutine_factory: Callable[[], Any]) -> Any:
    """Run a coroutine from sync code, including code already on an event loop.

    The existing vNext Producer protocols are synchronous while the public LLM
    API is asynchronous.  A fresh worker thread is used only when the caller
    already owns an event loop; this keeps the compatibility call synchronous
    without attempting nested event-loop execution.  Async callers should use
    the producer's ``produce`` method directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="engram-vnext-llm",
    ) as executor:
        future = executor.submit(_run_coroutine_in_thread, coroutine_factory)
        return future.result()


def _required_text(payload: Mapping[str, object], key: str, index: int) -> str:
    """Read one required non-empty string from a decoded JSON object."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LLMResponseFormatError(
            f"candidate[{index}].{key} 必须是非空字符串"
        )
    return value.strip()


def _optional_text(
    payload: Mapping[str, object],
    key: str,
    index: int,
) -> str | None:
    """Read an optional nullable string from a decoded JSON object."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LLMResponseFormatError(f"candidate[{index}].{key} 必须是字符串或 null")
    return value.strip() or None


def _parse_enum_value(
    enum_type: type[E],
    value: object,
    field_name: str,
) -> E:
    """Parse an enum value while accepting case-insensitive wire text."""
    if not isinstance(value, str) or not value.strip():
        raise LLMResponseFormatError(f"{field_name} 必须是非空字符串")
    normalized = value.strip().upper()
    for member in enum_type:
        if member.name.upper() == normalized or str(member.value).upper() == normalized:
            return member
    allowed = ", ".join(str(member.value) for member in enum_type)
    raise LLMResponseFormatError(f"{field_name} 不是有效值，可选: {allowed}")


def _parse_optional_enum(
    enum_type: type[E],
    value: object,
    field_name: str,
) -> E | None:
    """Parse a nullable enum value."""
    if value is None:
        return None
    return _parse_enum_value(enum_type, value, field_name)


def _parse_evidence_indexes(value: object, index: int) -> tuple[int, ...]:
    """Parse and validate the message indexes attached to one draft."""
    if not isinstance(value, list) or not value:
        raise LLMResponseFormatError(
            f"candidate[{index}].evidence_indexes 必须是非空整数数组"
        )
    indexes: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise LLMResponseFormatError(
                f"candidate[{index}].evidence_indexes 只能包含非负整数"
            )
        indexes.append(item)
    if len(set(indexes)) != len(indexes):
        raise LLMResponseFormatError(
            f"candidate[{index}].evidence_indexes 不能重复"
        )
    return tuple(indexes)


def _parse_subject(value: object, index: int) -> SubjectInput | None:
    """Parse an optional structured SubjectInput from a draft object."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise LLMResponseFormatError(f"candidate[{index}].subject 必须是对象或 null")

    kind = _parse_enum_value(
        SubjectKind,
        value.get("subject_kind"),
        f"candidate[{index}].subject.subject_kind",
    )
    person_id = value.get("person_id")
    subject_key = value.get("subject_key")
    subject_label = value.get("subject_label")
    for field_name, field_value in (
        ("person_id", person_id),
        ("subject_key", subject_key),
        ("subject_label", subject_label),
    ):
        if field_value is not None and not isinstance(field_value, str):
            raise LLMResponseFormatError(
                f"candidate[{index}].subject.{field_name} 必须是字符串或 null"
            )

    subject = SubjectInput(
        subject_kind=kind,
        person_id=person_id.strip() if isinstance(person_id, str) else None,
        subject_key=subject_key.strip() if isinstance(subject_key, str) else None,
        subject_label=subject_label.strip() if isinstance(subject_label, str) else None,
    )
    try:
        subject.validate()
    except ValueError as error:
        raise LLMResponseFormatError(
            f"candidate[{index}].subject 不满足领域约束"
        ) from error
    return subject


def _strip_json_fence(raw_text: str) -> str:
    """Remove one optional Markdown JSON fence without repairing JSON."""
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
        return text
    if lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def _decode_json_array(raw_text: str, producer_name: str) -> list[object]:
    """Decode a strict JSON array, preserving [] as a valid empty result."""
    text = _strip_json_fence(raw_text)
    if not text:
        raise LLMResponseFormatError(
            f"{producer_name} 返回空文本；合法空输出必须显式返回 []"
        )
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMResponseFormatError(
            f"{producer_name} 返回的内容不是有效 JSON 数组"
        ) from error
    if not isinstance(decoded, list):
        raise LLMResponseFormatError(f"{producer_name} 的 JSON 根节点必须是数组")
    return decoded


class _LLMJsonProducer:
    """Shared request construction for the two structured LLM producers."""

    def __init__(
        self,
        *,
        model_task: str,
        request_name: str,
        system_prompt: str,
        model_set: ModelSet | None,
    ) -> None:
        """Store lazy model configuration and immutable prompt settings."""
        if not isinstance(model_task, str) or not model_task.strip():
            raise ValueError("model_task 不能为空")
        if not isinstance(request_name, str) or not request_name.strip():
            raise ValueError("request_name 不能为空")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        self._model_task = model_task.strip()
        self._request_name = request_name.strip()
        self._system_prompt = system_prompt.strip()
        self._model_set = model_set

    async def _complete_json(self, user_prompt: str, producer_name: str) -> list[object]:
        """Send one non-streaming request and decode its JSON array response."""
        try:
            model_set = self._model_set or llm_api.get_model_set_by_task(self._model_task)
            request = llm_api.create_llm_request(
                model_set,
                request_name=self._request_name,
            )
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(self._system_prompt)))
            request.add_payload(LLMPayload(ROLE.USER, Text(user_prompt)))
            send_result = request.send(stream=False)
            response: object = (
                await send_result
                if inspect.isawaitable(send_result)
                else send_result
            )
            completed: object = response
            if inspect.isawaitable(response):
                completed = await response

            raw_message: object = None
            if isinstance(response, str):
                raw_message = response
            else:
                # The public API returns LLMResponse; this also supports a
                # small injected response double without importing internals.
                raw_message = getattr(response, "message", None)
            if not isinstance(raw_message, str) and not isinstance(completed, str):
                raw_message = getattr(completed, "message", None)
            if not isinstance(raw_message, str) and isinstance(completed, str):
                raw_message = completed
            if not isinstance(raw_message, str):
                raise LLMResponseFormatError(
                    f"{producer_name} 未返回可解析的文本消息"
                )
            return _decode_json_array(raw_message, producer_name)
        except LLMProducerError:
            raise
        except Exception as error:  # noqa: BLE001
            raise LLMProducerError(f"{producer_name} LLM 请求失败") from error


class ExperienceEncoderDraftProducer(_LLMJsonProducer, DraftProducer):
    """Produce :class:`EncoderDraft` values through the public LLM API.

    A JSON ``[]`` is returned as ``()`` and is a successful zero-candidate
    result.  Blank text, malformed JSON, invalid fields, and request failures
    raise a runtime producer exception.  Consequently the existing
    ``ExperienceEncoder.encode`` method advances its cursor only for a valid
    response, including the explicit empty array case.
    """

    def __init__(
        self,
        model_task: str = DEFAULT_ENCODER_MODEL_TASK,
        *,
        request_name: str = DEFAULT_ENCODER_REQUEST_NAME,
        system_prompt: str = DEFAULT_ENCODER_SYSTEM_PROMPT,
        model_set: ModelSet | None = None,
    ) -> None:
        """Create an Experience Encoder producer with lazy model lookup."""
        _LLMJsonProducer.__init__(
            self,
            model_task=model_task,
            request_name=request_name,
            system_prompt=system_prompt,
            model_set=model_set,
        )

    async def produce(
        self,
        prompt_version: str,
        batch_text: str,
    ) -> tuple[EncoderDraft, ...]:
        """Asynchronously produce drafts for one formatted message batch."""
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version 不能为空")
        if not isinstance(batch_text, str) or not batch_text.strip():
            raise ValueError("batch_text 不能为空")

        user_prompt = (
            f"Prompt version: {prompt_version.strip()}\n"
            "Return only the JSON array described by the system contract.\n"
            "Message batch:\n"
            f"{batch_text.strip()}"
        )
        decoded = await self._complete_json(user_prompt, "Experience Encoder")
        drafts: list[EncoderDraft] = []
        for index, item in enumerate(decoded):
            if not isinstance(item, Mapping):
                raise LLMResponseFormatError(
                    f"candidate[{index}] 必须是 JSON 对象"
                )
            proposed_kind = _parse_optional_enum(
                MemoryKind,
                item.get("proposed_kind"),
                f"candidate[{index}].proposed_kind",
            )
            confidence_hint = _parse_optional_enum(
                ConfidenceLevel,
                item.get("confidence_hint"),
                f"candidate[{index}].confidence_hint",
            )
            salience_hint = _parse_optional_enum(
                SalienceLevel,
                item.get("salience_hint"),
                f"candidate[{index}].salience_hint",
            )
            claim_basis = _parse_optional_enum(
                ClaimBasis,
                item.get("claim_basis"),
                f"candidate[{index}].claim_basis",
            ) or ClaimBasis.DIRECT_STATEMENT
            subject_value = item.get("subject")
            if subject_value is None and "subject_kind" in item:
                subject_value = {
                    "subject_kind": item.get("subject_kind"),
                    "person_id": item.get("person_id"),
                    "subject_key": item.get("subject_key"),
                    "subject_label": item.get("subject_label"),
                }
            drafts.append(
                EncoderDraft(
                    rough_title=_required_text(item, "rough_title", index),
                    rough_content=_required_text(item, "rough_content", index),
                    retention_reason=_required_text(item, "retention_reason", index),
                    evidence_indexes=_parse_evidence_indexes(
                        item.get("evidence_indexes"), index
                    ),
                    proposed_kind=proposed_kind,
                    confidence_hint=confidence_hint,
                    salience_hint=salience_hint,
                    uncertainty_note=_optional_text(item, "uncertainty_note", index),
                    subject=_parse_subject(subject_value, index),
                    claim_basis=claim_basis,
                )
            )
        return tuple(drafts)

    def __call__(
        self,
        prompt_version: str,
        batch_text: str,
    ) -> tuple[EncoderDraft, ...]:
        """Synchronously bridge to :meth:`produce` for the core protocol."""
        return _run_coroutine_sync(
            lambda: self.produce(prompt_version, batch_text)
        )


class SleepAgentStepProducer(_LLMJsonProducer, AgentStepProducer):
    """Produce validated structured Sleep Agent action intents through LLM API."""

    def __init__(
        self,
        model_task: str = DEFAULT_SLEEP_MODEL_TASK,
        *,
        request_name: str = DEFAULT_SLEEP_REQUEST_NAME,
        system_prompt: str = DEFAULT_SLEEP_SYSTEM_PROMPT,
        prompt_version: str = "sleep-agent-v1",
        model_set: ModelSet | None = None,
    ) -> None:
        """Create a Sleep Agent producer with a versioned prompt context."""
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version 不能为空")
        _LLMJsonProducer.__init__(
            self,
            model_task=model_task,
            request_name=request_name,
            system_prompt=system_prompt,
            model_set=model_set,
        )
        self._prompt_version = prompt_version.strip()

    async def produce(self, candidate_payload: str) -> tuple[dict[str, object], ...]:
        """Asynchronously produce one validated Agent observation step."""
        if not isinstance(candidate_payload, str) or not candidate_payload.strip():
            raise ValueError("candidate_payload 不能为空")
        user_prompt = (
            f"Prompt version: {self._prompt_version}\n"
            "Return only the JSON array described by the system contract.\n"
            "Candidate payload:\n"
            f"{candidate_payload.strip()}"
        )
        decoded = await self._complete_json(user_prompt, "Sleep Agent")
        intents: list[dict[str, object]] = []
        for index, item in enumerate(decoded):
            if not isinstance(item, Mapping):
                raise LLMResponseFormatError(f"action[{index}] 必须是 JSON 对象")
            intent = dict(item)
            raw_step_type = item.get("step_type")
            if raw_step_type is None and item.get("action_type") is not None:
                # Keep compatibility with the original one-shot producer while
                # allowing the bounded loop to consume the new protocol.
                raw_step_type = item.get("action_type")
            if not isinstance(raw_step_type, str) or not raw_step_type.strip():
                raise LLMResponseFormatError(
                    f"step[{index}].step_type 必须是非空字符串"
                )
            normalized_step_type = raw_step_type.strip().upper()
            if normalized_step_type in SLEEP_AGENT_STEP_TYPES:
                intent["step_type"] = normalized_step_type
                if normalized_step_type == "DECIDE":
                    actions = item.get("actions")
                    if not isinstance(actions, list):
                        raise LLMResponseFormatError(
                            f"step[{index}].actions 必须是数组"
                        )
                elif normalized_step_type == "SEARCH":
                    if not isinstance(item.get("query"), str) or not str(
                        item.get("query")
                    ).strip():
                        raise LLMResponseFormatError(
                            f"step[{index}].query 必须是非空字符串"
                        )
            else:
                action_type = _parse_enum_value(
                    CandidateActionType,
                    raw_step_type,
                    f"action[{index}].action_type",
                )
                intent["action_type"] = str(action_type.value)
            intents.append(intent)
        return tuple(intents)

    def __call__(self, candidate_payload: str) -> tuple[dict[str, object], ...]:
        """Synchronously bridge to :meth:`produce` for the core protocol."""
        return _run_coroutine_sync(lambda: self.produce(candidate_payload))


class PersonaReviewProducer(_LLMJsonProducer):
    """Produce one post-Sleep Persona Review decision through the public LLM API."""

    def __init__(
        self,
        model_task: str = DEFAULT_SLEEP_MODEL_TASK,
        *,
        request_name: str = DEFAULT_PERSONA_REVIEW_REQUEST_NAME,
        system_prompt: str = DEFAULT_PERSONA_REVIEW_SYSTEM_PROMPT,
        prompt_version: str = "persona-review-v1",
        model_set: ModelSet | None = None,
    ) -> None:
        """Create a Persona Review producer with a versioned prompt."""
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version 不能为空")
        _LLMJsonProducer.__init__(
            self,
            model_task=model_task,
            request_name=request_name,
            system_prompt=system_prompt,
            model_set=model_set,
        )
        self._prompt_version = prompt_version.strip()

    async def produce(self, review_payload: str) -> dict[str, str] | None:
        """Review one person's post-Sleep evidence and return an update or None."""
        if not isinstance(review_payload, str) or not review_payload.strip():
            raise ValueError("review_payload 不能为空")
        decoded = await self._complete_json(
            (
                f"Prompt version: {self._prompt_version}\n"
                "Return only the JSON array described by the system contract.\n"
                f"Persona review payload:\n{review_payload.strip()}"
            ),
            "Persona Review",
        )
        if not decoded:
            return None
        if len(decoded) != 1 or not isinstance(decoded[0], Mapping):
            raise LLMResponseFormatError(
                "Persona Review 必须返回 [] 或只含一个对象的数组"
            )
        item = decoded[0]
        action = str(item.get("action", "")).strip().upper()
        if action == "KEEP":
            return None
        if action != "UPDATE":
            raise LLMResponseFormatError("Persona Review action 必须是 KEEP 或 UPDATE")
        impression_text = item.get("impression_text")
        reason = item.get("reason")
        if not isinstance(impression_text, str) or not impression_text.strip():
            raise LLMResponseFormatError("Persona Review impression_text 不能为空")
        if not isinstance(reason, str) or not reason.strip():
            raise LLMResponseFormatError("Persona Review reason 不能为空")
        return {
            "impression_text": impression_text.strip(),
            "reason": reason.strip(),
        }


def _validate_vector_item(item: VectorUpsert) -> None:
    """Validate the canonical fields required to derive one vector entry."""
    if not isinstance(item, VectorUpsert):
        raise TypeError("vector sink 只接受 VectorUpsert")
    for field_name, value in (
        ("entry_id", item.entry_id),
        ("memory_id", item.memory_id),
        ("text", item.text),
        ("content_hash", item.content_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"VectorUpsert.{field_name} 不能为空")
    if item.revision_id is not None and not isinstance(item.revision_id, str):
        raise ValueError("VectorUpsert.revision_id 必须是字符串或 None")


class ChromaVectorSink(VectorSink):
    """Write Canonical Retrieval Entry projections through the framework bridge.

    The sink never reads or invents cognitive fields.  Its document and
    metadata are derived solely from ``VectorUpsert``.  Since the framework vector
    interface has no upsert method, an update is implemented as delete followed
    by add; a failed add is left for the existing outbox retry path.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_VECTOR_DB_PATH,
        *,
        collection_name: str = DEFAULT_VECTOR_COLLECTION,
        embedding_task: str = DEFAULT_EMBEDDING_MODEL_TASK,
        request_name: str = DEFAULT_EMBEDDING_REQUEST_NAME,
        model_set: ModelSet | None = None,
        vector_db: VectorDatabase | None = None,
    ) -> None:
        """Create a lazy Chroma sink, optionally using an injected DB facade."""
        if not isinstance(db_path, str) or not db_path.strip():
            raise ValueError("db_path 不能为空")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name 不能为空")
        if not isinstance(embedding_task, str) or not embedding_task.strip():
            raise ValueError("embedding_task 不能为空")
        if not isinstance(request_name, str) or not request_name.strip():
            raise ValueError("request_name 不能为空")
        self._db_path = db_path.strip()
        self._collection_name = collection_name.strip()
        self._embedding_task = embedding_task.strip()
        self._request_name = request_name.strip()
        self._model_set = model_set
        self._vector_db = vector_db

    @property
    def collection_name(self) -> str:
        """Return the derived-index collection name."""
        return self._collection_name

    @property
    def supports_batch_upsert(self) -> bool:
        """Chroma sink 支持一次请求写入多条入口。"""
        return True

    @staticmethod
    def collection_name_for(
        index_id: str,
        embedding_model_id: str,
        embedding_dimension: int,
    ) -> str:
        """Derive a bounded physical collection name from manifest identity."""
        import hashlib

        if not index_id.strip() or not embedding_model_id.strip() or embedding_dimension <= 0:
            raise ValueError("索引物理参数无效")
        model_token = hashlib.sha256(embedding_model_id.encode("utf-8")).hexdigest()[:12]
        index_token = hashlib.sha256(index_id.encode("utf-8")).hexdigest()[:12]
        return f"engram_vnext_{index_token}_{model_token}_{embedding_dimension}"

    def for_index(
        self,
        index_id: str,
        embedding_model_id: str,
        embedding_dimension: int,
    ) -> ChromaVectorSink:
        """Return a sink routed to the manifest-specific collection."""
        return ChromaVectorSink(
            db_path=self._db_path,
            collection_name=self.collection_name_for(
                index_id, embedding_model_id, embedding_dimension
            ),
            embedding_task=self._embedding_task,
            request_name=self._request_name,
            model_set=self._model_set,
            vector_db=self._vector_db,
        )

    async def entry_ids(self) -> frozenset[str] | None:
        """Return all IDs currently stored in this physical collection."""
        try:
            result = await self._get_vector_db().get(
                collection_name=self._collection_name,
                include=["metadatas"],
            )
        except Exception:
            return None
        ids = result.get("ids") if isinstance(result, Mapping) else None
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
            return None
        return frozenset(str(item) for item in ids if str(item).strip())

    @staticmethod
    def derived_metadata(item: VectorUpsert) -> dict[str, str]:
        """Build scalar-only metadata from canonical retrieval identifiers."""
        _validate_vector_item(item)
        metadata: dict[str, str] = {
            "entry_id": item.entry_id,
            "memory_id": item.memory_id,
            "content_hash": item.content_hash,
        }
        if item.revision_id:
            metadata["revision_id"] = item.revision_id
        return metadata

    def _get_vector_db(self) -> VectorDatabase:
        """Resolve the injected or cached framework vector database facade."""
        if self._vector_db is None:
            self._vector_db = get_vector_database(self._db_path)
        return self._vector_db

    def embedding_model_identity(self) -> str:
        """从当前模型任务解析唯一 Embedding 模型标识。"""
        model_set = self._model_set or llm_api.get_model_set_by_task(self._embedding_task)
        if len(model_set) != 1 or not isinstance(model_set[0], Mapping):
            raise VectorSinkError("Embedding task 必须恰好配置一个模型")
        identity = model_set[0].get("model_identifier")
        if not isinstance(identity, str) or not identity.strip():
            raise VectorSinkError("Embedding task 模型缺少 model_identifier")
        return identity.strip()

    async def inspect_embedding_settings(self) -> tuple[str, int]:
        """返回模型任务 identity 与一次真实响应的向量维度。"""
        identity = self.embedding_model_identity()
        embedding = await self.embed_text("engram-vnext-runtime-preflight")
        return identity, len(embedding)

    async def embed_text(self, text: str) -> list[float]:
        """Create one embedding through the public LLM embedding API."""
        return (await self.embed_texts((text,)))[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """通过一次公开 LLM 请求生成一批按输入顺序排列的向量。"""
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding texts 只能包含非空文本")
        try:
            model_set = self._model_set or llm_api.get_model_set_by_task(
                self._embedding_task
            )
            request = llm_api.create_embedding_request(
                model_set,
                request_name=self._request_name,
                inputs=list(texts),
            )
            send_result = request.send()
            response: object = (
                await send_result
                if inspect.isawaitable(send_result)
                else send_result
            )
            # EmbeddingResponse is the public return type; the dynamic access
            # keeps lightweight injected response doubles usable in tests.
            embeddings = getattr(response, "embeddings", None)
        except Exception as error:  # noqa: BLE001
            raise VectorSinkError("embedding 请求失败") from error

        if (
            not isinstance(embeddings, Sequence)
            or isinstance(embeddings, (str, bytes, bytearray))
            or not embeddings
        ):
            raise VectorSinkError("embedding 请求返回为空")
        if len(embeddings) != len(texts):
            raise VectorSinkError("embedding 返回数量与输入数量不一致")
        vectors: list[list[float]] = []
        for embedding in embeddings:
            if (
                not isinstance(embedding, Sequence)
                or isinstance(embedding, (str, bytes, bytearray))
                or not embedding
            ):
                raise VectorSinkError("embedding 返回的向量格式无效")
            vector: list[float] = []
            for value in embedding:
                if isinstance(value, bool):
                    raise VectorSinkError("embedding 向量包含非法数值")
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise VectorSinkError("embedding 向量包含非法数值") from error
                if not math.isfinite(numeric):
                    raise VectorSinkError("embedding 向量包含非有限数值")
                vector.append(numeric)
            vectors.append(vector)
        return vectors

    async def query_entries(self, text: str, top_k: int) -> tuple[str, ...]:
        """Query the derived Chroma collection and return entry IDs only.

        The vector collection is an optional acceleration layer.  A missing
        or temporarily unavailable derived index therefore produces no vector
        hits; the canonical retrieval service can still use its other lanes.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text 不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        try:
            embedding = await self.embed_text(text)
            result = await self._get_vector_db().query(
                collection_name=self._collection_name,
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas"],
            )
        except Exception:
            return ()
        rows = result.get("ids") if isinstance(result, dict) else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
            return ()
        return tuple(str(entry_id) for entry_id in rows[0] if str(entry_id).strip())

    async def upsert(self, item: VectorUpsert) -> None:
        """Derive and replace one vector entry from canonical retrieval data."""
        await self.upsert_many((item,))

    async def upsert_many(self, items: Sequence[VectorUpsert]) -> None:
        """通过一次 embedding 请求批量替换同一物理集合中的入口。"""
        if not items:
            return
        for item in items:
            _validate_vector_item(item)
        embeddings = await self.embed_texts(tuple(item.text for item in items))
        vector_db = self._get_vector_db()
        try:
            await vector_db.delete(
                collection_name=self._collection_name,
                ids=[item.entry_id for item in items],
            )
            await vector_db.add(
                collection_name=self._collection_name,
                embeddings=embeddings,
                documents=[item.text for item in items],
                metadatas=[self.derived_metadata(item) for item in items],
                ids=[item.entry_id for item in items],
            )
        except Exception as error:  # noqa: BLE001
            raise VectorSinkError("向量派生 upsert 失败") from error

    async def delete(self, entry_id: str) -> None:
        """Delete one derived vector entry without touching canonical storage."""
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("entry_id 不能为空")
        try:
            await self._get_vector_db().delete(
                collection_name=self._collection_name,
                ids=[entry_id.strip()],
            )
        except Exception as error:  # noqa: BLE001
            raise VectorSinkError("向量派生 delete 失败") from error


@runtime_checkable
class VectorIndexServiceProtocol(Protocol):
    """Public behavior required by the outbox lifecycle wrapper."""

    async def process_pending_outbox(self, limit: int = 20) -> tuple[str, ...]:
        """Process pending vector outbox items."""
        ...

    async def retry_failed_outbox(self, limit: int = 20) -> tuple[str, ...]:
        """Reset and process failed vector outbox items."""
        ...


class VectorOutboxWorker:
    """Lifecycle wrapper for the existing vNext VectorIndexService."""

    def __init__(
        self,
        service_or_schema: VectorIndexServiceProtocol | VNextSchema,
        sink: VectorSink | None = None,
        *,
        batch_size: int = 20,
        poll_interval_seconds: float = 60.0,
        retry_failed: bool = False,
        task_name: str = "engram_vnext_vector_outbox_worker",
    ) -> None:
        """Create a worker from an index service or schema plus vector sink."""
        if isinstance(service_or_schema, VNextSchema):
            if sink is None:
                raise ValueError("使用 VNextSchema 时必须提供 sink")
            service: VectorIndexServiceProtocol = VectorIndexService(
                service_or_schema,
                sink,
            )
        elif isinstance(service_or_schema, VectorIndexServiceProtocol):
            if sink is not None:
                raise ValueError("使用 VectorIndexService 时不能再次传入 sink")
            service = service_or_schema
        else:
            raise TypeError(
                "service_or_schema 必须实现 VectorIndexServiceProtocol 或是 VNextSchema"
            )
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds 必须大于 0")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name 不能为空")
        self._service: VectorIndexServiceProtocol = service
        self._batch_size = batch_size
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._retry_failed = bool(retry_failed)
        self._task_name = task_name.strip()
        self._stop_event: asyncio.Event | None = None
        self._task_info: ManagedTaskHandle | None = None
        self._last_error: Exception | None = None

    @property
    def task_info(self) -> ManagedTaskHandle | None:
        """Return the tracked background task, if started."""
        return self._task_info

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent poll error observed by the worker."""
        return self._last_error

    async def run_once(self) -> tuple[str, ...]:
        """Process one outbox poll and return successfully indexed entry IDs."""
        succeeded = list(
            await self._service.process_pending_outbox(limit=self._batch_size)
        )
        if self._retry_failed:
            succeeded.extend(
                await self._service.retry_failed_outbox(limit=self._batch_size)
            )
        return tuple(succeeded)

    async def run_forever(self) -> None:
        """Poll until stopped, retaining errors for later inspection."""
        if self._stop_event is None:
            self._stop_event = asyncio.Event()
        stop_event = self._stop_event
        while not stop_event.is_set():
            try:
                await self.run_once()
                self._last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                self._last_error = error
            if stop_event.is_set():
                break
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                continue

    def start(self) -> ManagedTaskHandle:
        """Start one tracked daemon poll task and return its owned handle."""
        if self._task_info is not None and self._task_info.task is not None:
            if not self._task_info.task.done():
                return self._task_info
        self._stop_event = asyncio.Event()
        self._task_info = create_managed_task(
            self.run_forever(),
            name=self._task_name,
            daemon=True,
        )
        return self._task_info

    async def stop(self) -> None:
        """Stop and await the tracked poll task, if one exists."""
        if self._stop_event is not None:
            self._stop_event.set()
        task_info = self._task_info
        if task_info is None or task_info.task is None:
            return
        if not task_info.task.done():
            cancel_managed_task(task_info.task_id)
        await asyncio.gather(task_info.task, return_exceptions=True)
        self._task_info = None


class VNextRuntimeAdapter:
    """Compose message, LLM, vector, and outbox runtime adapters."""

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        encoder_producer: DraftProducer | None = None,
        sleep_producer: AgentStepProducer | None = None,
        vector_sink: VectorSink | None = None,
        vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
        vector_collection: str = DEFAULT_VECTOR_COLLECTION,
    ) -> None:
        """Create a composable runtime facade without changing plugin wiring."""
        self.message_batcher = RuntimeMessageAdapter(batch_size=batch_size)
        self.encoder_producer = encoder_producer or ExperienceEncoderDraftProducer()
        self.sleep_producer = sleep_producer or SleepAgentStepProducer()
        self.vector_sink = vector_sink or ChromaVectorSink(
            db_path=vector_db_path,
            collection_name=vector_collection,
        )

    def adapt_message(self, message: MessageLike) -> EncoderMessage:
        """Adapt one public Message through the configured batcher."""
        return message_to_encoder_message(message)

    def batch_messages(
        self,
        messages: Iterable[MessageLike],
        *,
        stream_id: str | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Batch messages through the configured runtime adapter."""
        return self.message_batcher.batches(messages, stream_id=stream_id)

    async def load_stream_batches(
        self,
        stream_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        cursor: CandidateEncoderCursorModel | None = None,
    ) -> tuple[tuple[EncoderMessage, ...], ...]:
        """Load and batch one stream through the public stream API."""
        return await self.message_batcher.load_stream_batches(
            stream_id,
            limit=limit,
            offset=offset,
            cursor=cursor,
        )

    def create_vector_worker(
        self,
        schema: VNextSchema,
        *,
        batch_size: int = 20,
        poll_interval_seconds: float = 60.0,
        retry_failed: bool = False,
    ) -> VectorOutboxWorker:
        """Create an outbox worker using this adapter's derived vector sink."""
        index_service = VectorIndexService(schema, self.vector_sink)
        return VectorOutboxWorker(
            index_service,
            batch_size=batch_size,
            poll_interval_seconds=poll_interval_seconds,
            retry_failed=retry_failed,
        )


MessageBatchAdapter = RuntimeMessageAdapter
MessageAdapter = RuntimeMessageAdapter
LLMDraftProducer = ExperienceEncoderDraftProducer
LLMExperienceEncoderDraftProducer = ExperienceEncoderDraftProducer
LLMAgentStepProducer = SleepAgentStepProducer
LLMSleepAgentStepProducer = SleepAgentStepProducer
LLMPersonaReviewProducer = PersonaReviewProducer
VectorDBSink = ChromaVectorSink
ChromaSink = ChromaVectorSink
OutboxWorker = VectorOutboxWorker
VectorWorker = VectorOutboxWorker
RuntimeAdapter = VNextRuntimeAdapter
EngramRuntimeAdapter = VNextRuntimeAdapter

__all__: list[str] = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENCODER_MODEL_TASK",
    "DEFAULT_SLEEP_MODEL_TASK",
    "DEFAULT_EMBEDDING_MODEL_TASK",
    "DEFAULT_VECTOR_COLLECTION",
    "DEFAULT_ENCODER_REQUEST_NAME",
    "DEFAULT_SLEEP_REQUEST_NAME",
    "DEFAULT_EMBEDDING_REQUEST_NAME",
    "DEFAULT_VECTOR_DB_PATH",
    "DEFAULT_ENCODER_SYSTEM_PROMPT",
    "DEFAULT_SLEEP_SYSTEM_PROMPT",
    "DEFAULT_PERSONA_REVIEW_REQUEST_NAME",
    "DEFAULT_PERSONA_REVIEW_SYSTEM_PROMPT",
    "RuntimeAdapterError",
    "RuntimeProducerError",
    "LLMProducerError",
    "LLMResponseFormatError",
    "VectorSinkError",
    "message_to_encoder_message",
    "adapt_message",
    "messages_to_encoder_messages",
    "adapt_messages",
    "batch_messages",
    "MessageBatcher",
    "RuntimeMessageAdapter",
    "MessageBatchAdapter",
    "MessageAdapter",
    "ExperienceEncoderDraftProducer",
    "LLMDraftProducer",
    "LLMExperienceEncoderDraftProducer",
    "SleepAgentStepProducer",
    "LLMAgentStepProducer",
    "LLMSleepAgentStepProducer",
    "PersonaReviewProducer",
    "LLMPersonaReviewProducer",
    "ChromaVectorSink",
    "VectorDBSink",
    "ChromaSink",
    "VectorOutboxWorker",
    "VectorIndexServiceProtocol",
    "OutboxWorker",
    "VectorWorker",
    "VNextRuntimeAdapter",
    "RuntimeAdapter",
    "EngramRuntimeAdapter",
]
