"""正式记忆变化与人物关联测试。"""

from datetime import UTC, datetime
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from plugins.engram_memory.vnext.domain import (
    CreateMemoryInput, EvidenceInput, MemoryChanged, MemoryLifecycleInput,
    ParticipantInput, ReviseMemoryInput, SubjectInput, WriteContext,
)
from plugins.engram_memory.vnext.enums import (
    ActorType, EvidenceSourceType, MemoryEventType, MemoryKind,
    ParticipantKind, RevisionChangeReason, SubjectKind,
)
from plugins.engram_memory.vnext.memory_service import MemoryService
from plugins.engram_memory.vnext.repository import MemoryRepository
from plugins.engram_memory.vnext.schema import VNextSchema
from plugins.engram_memory.vnext.tool_service import VNextToolService
from plugins.engram_memory.vnext import runtime_components


def test_affected_people_include_removed_and_added_people() -> None:
    """人物关联变化同时影响移除、保留和加入的人物。"""
    change = MemoryChanged(
        memory_id="memory-example",
        change_type=MemoryEventType.REVISED,
        before_person_ids=("person-a", "person-shared"),
        after_person_ids=("person-b", "person-shared"),
    )
    assert change.affected_person_ids == ("person-a", "person-shared", "person-b")


async def test_memory_changes_are_published_after_commit(tmp_path: Path) -> None:
    """创建、修订和作废通知包含已经提交的版本及全部受影响人物。"""
    schema = VNextSchema(str(tmp_path / "memory.db"))
    await schema.initialize()
    repository = MemoryRepository(schema)
    changes: list[MemoryChanged] = []

    async def receive(change: MemoryChanged) -> None:
        """读取提交后的当前版本并收集通知。"""
        revision = await repository.get_current_revision(change.memory_id)
        assert revision is not None
        assert revision.revision_id == change.after_revision_id
        changes.append(change)

    service = MemoryService(schema, "example-embedding", on_memory_changed=receive)
    now = datetime.now(UTC)
    context = WriteContext(ActorType.ACTOR, operation_key="example-create")
    data = CreateMemoryInput(
        title="共同项目", content="A 提出与 C 合作制作视频。",
        memory_kind=MemoryKind.COMMITMENT,
        subject=SubjectInput(SubjectKind.PERSON, person_id="person-a"),
        participants=(ParticipantInput(ParticipantKind.PERSON, person_id="person-c"),),
        observed_at=now,
        evidence=(EvidenceInput(EvidenceSourceType.ADMIN, now, note="示例来源"),),
    )
    try:
        result = await service.create_memory(data, context)
        assert changes[-1].affected_person_ids == ("person-a", "person-c")
        await service.create_memory(data, context)
        assert len(changes) == 1
        revised = await service.revise_memory(ReviseMemoryInput(
            memory_id=result.memory_id, based_on_revision_id=result.revision_id,
            title="共同项目", content="计划的发起者是 B，与 C 合作制作视频。",
            memory_kind=MemoryKind.COMMITMENT,
            subject=SubjectInput(SubjectKind.PERSON, person_id="person-b"),
            participants=data.participants, observed_at=now,
            change_reason=RevisionChangeReason.CORRECTION,
            evidence_ids=result.evidence_ids,
        ), WriteContext(ActorType.ACTOR))
        assert changes[-1].affected_person_ids == ("person-a", "person-c", "person-b")
        assert changes[-1].before_revision_id == result.revision_id
        await service.tombstone_memory(
            MemoryLifecycleInput(result.memory_id, "计划撤销"), WriteContext(ActorType.ACTOR),
        )
        assert changes[-1].affected_person_ids == ("person-b", "person-c")
        assert changes[-1].after_revision_id == revised.revision_id
        await service.restore_memory(
            MemoryLifecycleInput(result.memory_id, "恢复计划"), WriteContext(ActorType.ADMIN),
        )
        assert changes[-1].affected_person_ids == ("person-b", "person-c")
        assert [change.change_type for change in changes] == [
            MemoryEventType.CREATED, MemoryEventType.REVISED,
            MemoryEventType.TOMBSTONED, MemoryEventType.RESTORED,
        ]
    finally:
        await schema.close()


async def test_actions_preserve_people_versions_sources_and_notifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自然正文写入、主次人物调整与作废保留来源，并发布已提交的变化。"""
    schema = VNextSchema(str(tmp_path / "actions.db"))
    await schema.initialize()
    changes: list[MemoryChanged] = []

    async def receive(change: MemoryChanged) -> None:
        """收集提交后通知。"""
        changes.append(change)

    async def get_person(person_id: str) -> SimpleNamespace | None:
        """只接受示例中的准确人物标识。"""
        return SimpleNamespace(person_id=person_id) if person_id in {"person-a", "person-b", "person-c"} else None

    tools = VNextToolService(schema, cast(Any, SimpleNamespace(search=AsyncMock(return_value=()))), on_memory_changed=receive)
    repository = MemoryRepository(schema)
    owner = SimpleNamespace(tools=tools, repository=repository, persona_service=SimpleNamespace(get_core_person=get_person))
    monkeypatch.setattr(runtime_components, "_owner", lambda plugin: owner)
    first_message = {
        "message_id": "source-1", "stream_id": "stream-example", "person_id": "person-a",
        "sender_id": "account-a", "sender_name": "示例发言人", "platform": "test",
        "time": "2026-01-01T00:00:00Z", "content": "B 提过和 C 合作制作短片的计划。",
    }
    second_message = {
        **first_message, "message_id": "source-2", "time": "2026-01-02T00:00:00Z",
        "content": "发起计划的是 C，我也是参与者。现在计划已经取消。",
    }
    stream = SimpleNamespace(stream_id="stream-example", context=SimpleNamespace(
        chat_type="group", history_messages=[first_message], unread_messages=[second_message],
    ))
    plugin = SimpleNamespace()
    write = runtime_components.VNextMemoryWriteAction(cast(Any, stream), cast(Any, plugin))
    revise = runtime_components.VNextMemoryReviseAction(cast(Any, stream), cast(Any, plugin))
    invalidate = runtime_components.VNextMemoryInvalidateAction(cast(Any, stream), cast(Any, plugin))
    payload: dict[str, object] = {
        "content": "据 A 转述，B 曾提出与 C 一起制作短片的计划，尚未实施。",
        "memory_kind": "COMMITMENT", "primary_person_id": "person-b",
        "secondary_person_ids": ["person-c"], "source_message_ids": ["source-1"],
    }
    try:
        successful, raw_result = await write.execute(payload)
        assert successful, raw_result
        saved = json.loads(raw_result)
        memory_id = saved["memory_id"]
        assert changes[0].affected_person_ids == ("person-b", "person-c")
        current = await tools.memory_read(memory_id, "full", runtime_components._actor_context(write))
        assert current["primary_person_id"] == "person-b"
        assert current["secondary_person_ids"] == ["person-c"]
        assert current["current_revision"]["content"] == payload["content"]
        assert current["current_revision"]["observed_at"] == datetime(2026, 1, 1, tzinfo=UTC)
        snapshot = current["evidence_metadata"][0]["messages"][0]["snapshot"]
        assert snapshot["person_id"] == "person-a"
        assert snapshot["chat_type"] == "group"
        assert snapshot["content"] == first_message["content"]
        assert (await write.execute(payload))[0]
        assert len(changes) == 1

        successful, raw_result = await revise.execute({
            "memory_id": memory_id, "based_on_revision_id": saved["revision_id"],
            "content": "A 澄清发起者是 C，A 参与其中；这一制作计划已取消。",
            "primary_person_id": "person-c", "secondary_person_ids": ["person-a"],
            "source_message_ids": ["source-2"], "reason": "发起者澄清与计划取消",
        })
        assert successful, raw_result
        assert changes[-1].affected_person_ids == ("person-b", "person-c", "person-a")
        current = await tools.memory_read(memory_id, "full", runtime_components._actor_context(revise))
        assert len(current["history"]) == 2
        assert current["history"][0]["primary_person_id"] == "person-b"
        assert current["history"][1]["primary_person_id"] == "person-c"
        assert current["current_revision"]["memory_kind"] == "COMMITMENT"
        assert {message["message_id"] for record in current["evidence_metadata"] for message in record["messages"]} == {"source-1", "source-2"}

        successful, raw_result = await invalidate.execute(memory_id, "该计划已撤销", ["source-2"])
        assert successful, raw_result
        assert changes[-1].affected_person_ids == ("person-c", "person-a")
        assert (await invalidate.execute(memory_id, "该计划已撤销", ["source-2"]))[0]
        assert len(changes) == 3
        current = await tools.memory_read(memory_id, "full", runtime_components._actor_context(invalidate))
        assert current["status"] == "TOMBSTONED"
        assert len(current["history"]) == 2
        assert any(event["event_type"] == "TOMBSTONED" and event["payload"]["evidence_ids"] for event in current["events"])

        assert not (await write.execute({**payload, "primary_person_id": "示例昵称"}))[0]
        assert not (await write.execute({**payload, "secondary_person_ids": ["person-b"]}))[0]
        successful, raw_result = await write.execute({**payload, "source_message_ids": ["other-stream-source"]})
        assert not successful
        assert {item["message_id"] for item in json.loads(raw_result)["source_messages"]} == {"source-1", "source-2"}
        assert len(changes) == 3
    finally:
        await schema.close()


async def test_change_event_keeps_params_and_only_enqueues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """变化事件登记更新而不等待模型，并原样保留发布方参数。"""
    enqueue = AsyncMock()
    owner = SimpleNamespace(persona_updater=SimpleNamespace(enqueue=enqueue))
    monkeypatch.setattr(runtime_components, "_owner", lambda plugin: owner)
    handler = SimpleNamespace(plugin=SimpleNamespace())
    change = MemoryChanged("example-memory", MemoryEventType.CREATED, after_person_ids=("person-a",))
    params = {"change": change}
    decision, result = await runtime_components.VNextMemoryChangedEventHandler.execute(
        cast(Any, handler), "engram_memory:memory_changed", params,
    )
    assert decision is runtime_components.EventDecision.SUCCESS
    assert result is params and set(result) == {"change"}
    enqueue.assert_awaited_once_with(change)