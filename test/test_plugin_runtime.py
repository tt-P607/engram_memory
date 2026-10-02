"""消息快照、插件装配、资源清理与闪回运行链测试。"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from ..config import EngramMemoryConfig
from ..vnext import runtime, runtime_owner
from ..vnext.domain import MemoryChanged, VectorUpsert
from ..vnext.enums import MemoryEventType
from ..vnext.runtime import ChromaVectorSink, MessageLike, VectorOutboxWorker
from ..vnext.runtime_owner import VNextRuntimeOwner


def _message(**values: object) -> dict[str, object]:
    """构造不包含真实身份信息的公开消息字段。"""
    message: dict[str, object] = {
        "message_id": "message-1",
        "stream_id": "stream-1",
        "time": "2026-01-01T00:00:00Z",
        "content": "原始正文",
        "processed_plain_text": "  处理后的正文  ",
        "sender_name": "示例发言者",
        "person_id": "person-a",
    }
    message.update(values)
    return message


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> VNextRuntimeOwner:
    """以资源桩装配 Owner，隔离文件、网络、模型和后台任务。"""
    schema = SimpleNamespace(initialize=AsyncMock(), close=AsyncMock())
    sink = Mock(spec=ChromaVectorSink)
    sink.embedding_model_identity.return_value = "embedding-test"
    sink.inspect_embedding_settings = AsyncMock(return_value=("embedding-test", 4))
    vector_index = SimpleNamespace(ensure_active_manifest=AsyncMock())
    worker = SimpleNamespace(start=Mock(), stop=AsyncMock())
    updater = SimpleNamespace(close=AsyncMock(), enqueue=AsyncMock(), start=Mock())
    resources = {
        "VNextSchema": schema,
        "MemoryRepository": Mock(),
        "ChromaVectorSink": sink,
        "ChromaVectorSearchBackend": Mock(),
        "RetrievalService": Mock(),
        "VNextToolService": Mock(),
        "PersonaService": Mock(),
        "PersonaUpdater": updater,
        "VectorIndexService": vector_index,
        "VectorOutboxWorker": worker,
        "FlashbackService": SimpleNamespace(record_exposure=AsyncMock()),
        "DoctorService": Mock(),
    }
    for name, resource in resources.items():
        monkeypatch.setattr(runtime_owner, name, Mock(return_value=resource))
    return VNextRuntimeOwner(SimpleNamespace(config=EngramMemoryConfig()))


@pytest.mark.parametrize("as_object", [False, True])
def test_message_snapshot_has_six_fields_and_preserves_source(as_object: bool) -> None:
    """消息映射与对象均转换为六字段快照，并保留原始正文和回复目标。"""
    message = _message(reply_to="message-parent")
    source = cast(MessageLike, SimpleNamespace(**message) if as_object else message)
    snapshot = runtime.message_to_snapshot(source)
    assert tuple(field.name for field in fields(snapshot)) == (
        "message_id", "stream_id", "time", "text", "speaker", "snapshot",
    )
    assert snapshot.message_id == "message-1"
    assert snapshot.stream_id == "stream-1"
    assert snapshot.time == datetime(2026, 1, 1, tzinfo=UTC)
    assert snapshot.text == "处理后的正文"
    assert snapshot.speaker == "示例发言者"
    assert snapshot.snapshot["content"] == "原始正文"
    assert snapshot.snapshot["reply_to"] == "message-parent"
    assert snapshot.snapshot["person_id"] == "person-a"
    json.dumps(dict(snapshot.snapshot), allow_nan=False)


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 8, tzinfo=timezone(timedelta(hours=8))),
        "2026-01-01T08:00:00+08:00",
        datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
    ],
)
def test_snapshot_normalizes_message_time(value: object) -> None:
    """支持的消息时间表示统一转换为带时区的 UTC 时间。"""
    snapshot = runtime.message_to_snapshot(_message(time=value))
    assert snapshot.time == datetime(2026, 1, 1, tzinfo=UTC)
    assert snapshot.time.tzinfo is UTC


@pytest.mark.parametrize(
    "values",
    [
        {"message_id": ""},
        {"stream_id": ""},
        {"content": "", "processed_plain_text": ""},
        {"time": True},
        {"time": float("inf")},
        {"time": "invalid-time"},
    ],
)
def test_snapshot_rejects_invalid_required_fields(values: dict[str, object]) -> None:
    """缺少来源身份、聊天流、正文或合法时间的消息直接报错。"""
    with pytest.raises(ValueError):
        runtime.message_to_snapshot(_message(**values))


def test_snapshot_uses_content_and_extra_person_identity() -> None:
    """纯文本缺失时使用原始正文，并保留附加字段中的精确人物身份。"""
    snapshot = runtime.message_to_snapshot(_message(
        processed_plain_text="", person_id=None, extra={"person_id": "person-extra"},
    ))
    assert snapshot.text == "原始正文"
    assert snapshot.snapshot["person_id"] == "person-extra"


def test_snapshot_generates_missing_core_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """人物身份缺失时仅按平台与发送账号生成核心 ID。"""
    generate = Mock(return_value="person-generated")
    monkeypatch.setattr(runtime.person_api, "generate_person_id", generate)
    snapshot = runtime.message_to_snapshot(_message(
        person_id=None, platform="test", sender_id="sender-test",
    ))
    generate.assert_called_once_with("test", "sender-test")
    assert snapshot.snapshot["person_id"] == "person-generated"


def test_snapshot_marks_bot_without_generating_account_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bot 发言保留明确身份，不生成普通账号人物 ID。"""
    generate = Mock()
    monkeypatch.setattr(runtime.person_api, "generate_person_id", generate)
    snapshot = runtime.message_to_snapshot(_message(
        person_id=None, platform="test", sender_id="sender-test", sender_role="BOT",
    ))
    generate.assert_not_called()
    assert snapshot.snapshot["person_id"] == "bot"
    assert snapshot.snapshot["speaker_is_bot"] is True


def test_runtime_exposes_only_snapshot_and_vector_interfaces() -> None:
    """公开运行接口只包含来源快照与派生向量能力。"""
    assert set(runtime.__all__) == {
        "DEFAULT_EMBEDDING_MODEL_TASK", "DEFAULT_VECTOR_COLLECTION",
        "DEFAULT_EMBEDDING_REQUEST_NAME", "DEFAULT_VECTOR_DB_PATH",
        "MessageSnapshot", "VectorSinkError", "message_to_snapshot",
        "ChromaVectorSink", "VectorOutboxWorker", "VectorIndexServiceProtocol",
    }
    for name in (
        "ExperienceEncoderDraftProducer", "SleepAgentStepProducer", "PersonaReviewProducer",
        "MessageBatcher", "RuntimeMessageAdapter", "VNextRuntimeAdapter",
        "message_to_encoder_message", "RuntimeAdapter", "LLMResponseFormatError",
    ):
        assert not hasattr(runtime, name)


def test_owner_wires_persona_and_memory_change_callback(owner: VNextRuntimeOwner) -> None:
    """Owner 共享人物服务与仓储，并向工具服务提供正式变化回调。"""
    runtime_owner.PersonaService.assert_called_once_with(  # type: ignore[attr-defined]
        owner.schema, persona_config=owner.config.vnext.persona,
    )
    runtime_owner.PersonaUpdater.assert_called_once_with(  # type: ignore[attr-defined]
        owner.persona_service, owner.repository, max_concurrency=3,
    )
    arguments = runtime_owner.VNextToolService.call_args.kwargs  # type: ignore[attr-defined]
    assert arguments["on_memory_changed"] == owner._on_memory_changed
    assert "persona_max_length" not in arguments
    assert "on_actor_memory_changed" not in arguments
    for name in (
        "encoder", "sleep_agent", "sleep_service", "persona_producer", "runtime_adapter",
        "_pending_messages", "encode_stream", "flush_all_streams", "run_daily_sleep",
        "run_pressure_sleep", "_review_personas", "_persona_current_view", "_person_memory_ids",
    ):
        assert not hasattr(owner, name)


@pytest.mark.asyncio
async def test_owner_publishes_committed_change(
    owner: VNextRuntimeOwner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式记忆变化通过公开事件 API 发布完整变化对象。"""
    publish = AsyncMock()
    monkeypatch.setattr(runtime_owner.event_api, "publish_event", publish)
    change = MemoryChanged(
        memory_id="memory-1", change_type=MemoryEventType.REVISED,
        before_person_ids=("person-a",), after_person_ids=("person-b",),
    )
    await owner._on_memory_changed(change)
    publish.assert_awaited_once_with("engram_memory:memory_changed", {"change": change})
    owner.persona_updater.enqueue.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_owner_initializes_vector_resources_once(owner: VNextRuntimeOwner) -> None:
    """初始化只建立规范库和派生索引，重复调用不启动额外 Worker。"""
    await owner.initialize()
    await owner.initialize()
    owner.schema.initialize.assert_awaited_once()  # type: ignore[attr-defined]
    owner.vector_index.ensure_active_manifest.assert_awaited_once_with(  # type: ignore[attr-defined]
        "embedding-test", 4, "engram-vnext-2",
    )
    owner.vector_worker.start.assert_called_once()  # type: ignore[attr-defined]
    owner.persona_updater.start.assert_called_once()  # type: ignore[attr-defined]
    assert owner._initialized is True


@pytest.mark.asyncio
async def test_owner_initialization_failure_releases_partial_resources(
    owner: VNextRuntimeOwner,
) -> None:
    """数据库初始化失败时关闭人物更新器和部分初始化的规范库。"""
    owner.schema.initialize.side_effect = RuntimeError("schema-test")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="schema-test"):
        await owner.initialize()
    owner.persona_updater.close.assert_awaited_once()  # type: ignore[attr-defined]
    owner.schema.close.assert_awaited_once()  # type: ignore[attr-defined]
    owner.vector_worker.start.assert_not_called()  # type: ignore[attr-defined]
    assert owner._initialized is False


@pytest.mark.asyncio
async def test_owner_close_releases_resources_and_is_idempotent(owner: VNextRuntimeOwner) -> None:
    """关闭移除闪回缓存并幂等停止向量 Worker 与规范数据库。"""
    await owner.initialize()
    owner._recent_messages["stream-1"] = {"message-1": _message()}
    owner._flashback_results["stream-1"] = ((), 1, 1)
    await owner.close()
    await owner.close()
    owner.vector_worker.stop.assert_awaited_once()  # type: ignore[attr-defined]
    owner.schema.close.assert_awaited_once()  # type: ignore[attr-defined]
    assert owner.persona_updater.close.await_count == 2  # type: ignore[attr-defined]
    assert not owner._recent_messages
    assert not owner._flashback_results
    assert owner._initialized is False


@pytest.mark.asyncio
async def test_persona_close_failure_does_not_leak_other_resources(owner: VNextRuntimeOwner) -> None:
    """人物更新器关闭失败仍释放 Worker 和规范数据库，并向调用者报错。"""
    await owner.initialize()
    owner.persona_updater.close.side_effect = RuntimeError("persona-test")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="persona-test"):
        await owner.close()
    owner.vector_worker.stop.assert_awaited_once()  # type: ignore[attr-defined]
    owner.schema.close.assert_awaited_once()  # type: ignore[attr-defined]
    assert owner._initialized is False


def test_observe_message_keeps_bounded_context_before_prefetch(
    owner: VNextRuntimeOwner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前消息进入有界闪回上下文后才触发预取。"""
    owner.config.vnext.flashback.context_turns = 2
    owner._initialized = True

    def prefetch(stream_id: str) -> None:
        """校验预取调度时最新消息已进入当前流上下文。"""
        assert owner._recent_messages[stream_id]

    schedule = Mock(side_effect=prefetch)
    monkeypatch.setattr(owner, "_schedule_flashback_prefetch", schedule)
    for index in range(3):
        owner.observe_message(_message(message_id=f"message-{index}"))
    assert tuple(owner._recent_messages["stream-1"]) == ("message-1", "message-2")
    assert schedule.call_count == 3


@pytest.mark.asyncio
async def test_flashback_context_includes_unpersisted_messages(
    owner: VNextRuntimeOwner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未落库的新消息参与闪回上下文，相同 ID 使用接收缓存中的正文。"""
    owner.config.vnext.flashback.context_turns = 2
    stored = _message(processed_plain_text="落库正文")
    load = AsyncMock(return_value=(stored,))
    monkeypatch.setattr(runtime_owner.stream_api, "get_stream_messages", load)
    owner.observe_message(_message(processed_plain_text="更新正文"))
    owner.observe_message(_message(
        message_id="message-2", processed_plain_text="最新正文", time="2026-01-01T00:00:01Z",
    ))
    assert await owner.recent_turns_for_flashback("stream-1") == (
        "示例发言者: 更新正文", "示例发言者: 最新正文",
    )
    load.assert_awaited_once_with("stream-1", limit=2)


@pytest.mark.asyncio
async def test_consumed_flashback_records_exposure_only_once(owner: VNextRuntimeOwner) -> None:
    """被当前回复消费的闪回结果只记录一次曝光并推进轮次。"""
    owner._initialized = True
    candidate = SimpleNamespace(memory_id="memory-1")
    owner._flashback_generations["stream-1"] = 1
    owner._flashback_results["stream-1"] = ((candidate,), 4, 1)
    assert await owner.consume_flashback_prefetch("stream-1") == (candidate,)
    assert await owner.consume_flashback_prefetch("stream-1") == ()
    owner.flashback.record_exposure.assert_awaited_once_with(  # type: ignore[attr-defined]
        ("memory-1",), "stream-1", 4,
    )
    assert owner._prompt_turns["stream-1"] == 5


@pytest.mark.asyncio
async def test_stale_flashback_does_not_record_exposure(owner: VNextRuntimeOwner) -> None:
    """过期预取结果不注入也不记录曝光。"""
    owner._initialized = True
    owner._flashback_generations["stream-1"] = 2
    owner._flashback_results["stream-1"] = ((SimpleNamespace(memory_id="memory-1"),), 4, 1)
    assert await owner.consume_flashback_prefetch("stream-1") == ()
    owner.flashback.record_exposure.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_vector_sink_preserves_batch_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """批量向量写入保持正式入口正文、身份和内容摘要。"""
    database = SimpleNamespace(delete=AsyncMock(), add=AsyncMock())
    sink = ChromaVectorSink(vector_db=cast(Any, database))
    embed = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(sink, "embed_texts", embed)
    items = tuple(
        VectorUpsert(
            entry_id=f"entry-{index}", memory_id=f"memory-{index}",
            revision_id=f"revision-{index}", text=f"text-{index}", content_hash=f"hash-{index}",
        )
        for index in range(2)
    )
    await sink.upsert_many(items)
    embed.assert_awaited_once_with(("text-0", "text-1"))
    database.delete.assert_awaited_once_with(
        collection_name=sink.collection_name, ids=["entry-0", "entry-1"],
    )
    payload = database.add.call_args.kwargs
    assert payload["documents"] == ["text-0", "text-1"]
    assert payload["metadatas"][0] == {
        "entry_id": "entry-0", "memory_id": "memory-0",
        "revision_id": "revision-0", "content_hash": "hash-0",
    }


@pytest.mark.asyncio
async def test_vector_worker_only_processes_pending_outbox() -> None:
    """自动向量投递遵循批量限制，不重试失败入口。"""
    service = SimpleNamespace(
        process_pending_outbox=AsyncMock(return_value=("entry-1",)),
        retry_failed_outbox=AsyncMock(),
    )
    worker = VectorOutboxWorker(service, batch_size=3)  # type: ignore[arg-type]
    assert await worker.run_once() == ("entry-1",)
    service.process_pending_outbox.assert_awaited_once_with(limit=3)
    service.retry_failed_outbox.assert_not_awaited()


def test_plugin_registers_exact_component_graph() -> None:
    """入口仅注册查询工具、写操作、变化与闪回事件、服务和管理路由。"""
    from ..plugin import EngramMemoryPlugin

    plugin = EngramMemoryPlugin(EngramMemoryConfig())
    components = plugin.get_components()
    assert {component.__name__ for component in components} == {
        "VNextMemorySearchTool", "VNextMemoryReadTool", "VNextPersonLookupTool",
        "VNextMemoryWriteAction", "VNextMemoryReviseAction", "VNextMemoryInvalidateAction",
        "VNextMemoryChangedEventHandler", "VNextFlashbackEventHandler", "VNextMemoryService",
        "VNextDoctorRouter", "VNextMemoryAdminRouter",
    }
    assert len(components) == 11
    names = {component.__name__: component.name for component in components}
    assert names["VNextMemoryChangedEventHandler"] == "memory_changed"
    assert names["VNextFlashbackEventHandler"] == "vnext_flashback_injector"
    plugin.config.plugin.enabled = False
    assert plugin.get_components() == []


@pytest.mark.asyncio
async def test_plugin_load_refreshes_routers_and_registers_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """加载共享 Owner 后刷新已挂载路由，并注册唯一全局记忆引导语。"""
    from .. import plugin as plugin_module

    plugin = plugin_module.EngramMemoryPlugin(EngramMemoryConfig())
    resource = SimpleNamespace(config=plugin.config, initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(plugin_module, "VNextRuntimeOwner", Mock(return_value=resource))
    monkeypatch.setattr(plugin_module.router_api, "get_mounted_router", Mock(return_value=object()))
    reload_router = AsyncMock()
    register_guide = Mock()
    monkeypatch.setattr(plugin_module.router_api, "reload_router", reload_router)
    monkeypatch.setattr(plugin_module.prompt_api, "add_system_reminder", register_guide)
    await plugin.on_plugin_loaded()
    resource.initialize.assert_awaited_once()
    resource.close.assert_not_awaited()
    assert plugin.runtime_owner is resource
    assert reload_router.await_count == 2
    assert {call.args[0] for call in reload_router.await_args_list} == {
        f"engram_memory:router:{plugin_module.VNextDoctorRouter.name}",
        f"engram_memory:router:{plugin_module.VNextMemoryAdminRouter.name}",
    }
    assert register_guide.call_args.kwargs["name"] == "engram_memory_guide"


@pytest.mark.asyncio
async def test_plugin_router_failure_closes_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """路由刷新失败时清理已初始化的 Owner，并保留原始异常。"""
    from .. import plugin as plugin_module

    plugin = plugin_module.EngramMemoryPlugin(EngramMemoryConfig())
    resource = SimpleNamespace(config=plugin.config, initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(plugin_module, "VNextRuntimeOwner", Mock(return_value=resource))
    monkeypatch.setattr(plugin_module.router_api, "get_mounted_router", Mock(return_value=object()))
    monkeypatch.setattr(
        plugin_module.router_api, "reload_router", AsyncMock(side_effect=RuntimeError("router-test")),
    )
    register_guide = Mock()
    monkeypatch.setattr(plugin_module.prompt_api, "add_system_reminder", register_guide)
    with pytest.raises(RuntimeError, match="router-test"):
        await plugin.on_plugin_loaded()
    resource.close.assert_awaited_once()
    assert plugin.runtime_owner is None
    register_guide.assert_not_called()


@pytest.mark.asyncio
async def test_plugin_unload_cleans_guide_when_owner_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner 关闭失败仍清除流闪回与全局引导语，并向调用者报错。"""
    from .. import plugin as plugin_module

    plugin = plugin_module.EngramMemoryPlugin(EngramMemoryConfig())
    resource = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("owner-test")))
    plugin.runtime_owner = cast(VNextRuntimeOwner, resource)
    plugin._flashback_reminder_streams["stream-1"] = {"flashback-test"}
    delete_stream = Mock()
    delete_guide = Mock()
    monkeypatch.setattr(plugin_module.prompt_api, "delete_stream_reminder", delete_stream)
    monkeypatch.setattr(plugin_module, "delete_owned_reminder", delete_guide)
    with pytest.raises(RuntimeError, match="owner-test"):
        await plugin.on_plugin_unloaded()
    delete_stream.assert_called_once_with("stream-1", "actor", "flashback-test")
    delete_guide.assert_called_once_with("actor", "engram_memory_guide")
    assert not plugin._flashback_reminder_streams
    assert plugin.runtime_owner is None