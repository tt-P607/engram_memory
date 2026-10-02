"""人物印象更新与变化队列测试。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ..config import EngramMemoryConfig
from ..vnext import persona_service, persona_updater
from ..vnext.domain import MemoryChanged
from ..vnext.enums import MemoryEventType
from ..vnext.persona_service import (
    MEMORY_REFERENCE,
    PersonaService,
    _format_memory_footnotes,
    _inline_memory_references,
)
from ..vnext.persona_updater import PersonaUpdater
from ..vnext.framework_bridge import ManagedTaskHandle


@pytest.fixture
def managed_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[str, asyncio.Task[Any]]:
    """记录测试事件循环中的全部托管任务，支持精确取消与等待。"""
    from ..vnext import persona_updater

    tasks: dict[str, asyncio.Task[Any]] = {}

    def create_task(coro: Any, *, name: str, daemon: bool) -> ManagedTaskHandle:
        """以唯一测试 ID 创建并登记任务。"""
        task_id = f"{name}-{len(tasks)}"
        task = asyncio.create_task(coro, name=name)
        tasks[task_id] = task
        return ManagedTaskHandle(task_id, task)

    def cancel_task(task_id: str) -> bool:
        """只取消测试边界创建的精确任务。"""
        return tasks[task_id].cancel()

    monkeypatch.setattr(persona_updater, "create_managed_task", create_task)
    monkeypatch.setattr(persona_updater, "cancel_managed_task", cancel_task)
    return tasks


class _Repository:
    """在队列测试中保持示例人物的准确 ID。"""

    async def resolve_person_aliases(self, person_id: str) -> tuple[str, ...]:
        """返回示例人物的单一别名。"""
        return (person_id,)


def _change(
    memory_id: str = "memory-1",
    before: tuple[str, ...] = (),
    after: tuple[str, ...] = ("person-a",),
) -> MemoryChanged:
    """构造有前后人物关联的示例变化。"""
    return MemoryChanged(
        memory_id=memory_id,
        change_type=MemoryEventType.REVISED,
        before_person_ids=before,
        after_person_ids=after,
    )


def _async_value(value: object) -> Any:
    """构造返回固定值的异步服务桩。"""
    async def resolve(*args: object, **kwargs: object) -> object:
        """返回预设值。"""
        return value

    return resolve


@pytest.mark.parametrize("impression_body", ["喜欢一起讨论计划。", "我愿意和他一起讨论计划。" * 60])
@pytest.mark.asyncio
async def test_refresh_includes_secondary_memory_and_accepts_keep_reference(
    monkeypatch: pytest.MonkeyPatch,
    impression_body: str,
) -> None:
    """次人物记忆正文可用，保留有真实引用的印象且不限制字数。"""
    inline_impression = f"{impression_body}[Memory: memory-secondary]"
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a",
        impression=(
            f"{impression_body}\u2460\n\n记忆依据：\n"
            "\u2460 [Memory: memory-secondary]"
        ), updated_at=None,
    )
    service = PersonaService(object(), generator=None)  # type: ignore[arg-type]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.get_persona = _async_value(SimpleNamespace(
        impression_text=person.impression,
    ))  # type: ignore[method-assign]
    memories = tuple({
        "memory_id": "memory-secondary" if index == 0 else f"memory-{index}",
        "revision_id": f"revision-{index}", "title": f"共同安排 {index}",
        "content": "甲作为参与者提到自己计划下周联系对方。"
        + "双方讨论各自的安排。" * 60 + f"正文尾部 {index}",
        "target_role": "secondary", "primary_person_id": "person-other",
        "secondary_person_ids": ["person-a"],
    } for index in range(12))
    service._load_active_memories = _async_value(memories)  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    seen_payload: dict[str, Any] = {}
    service.is_current_impression = _async_value(True)  # type: ignore[method-assign]

    async def generate(payload: str) -> dict[str, object]:
        """确认模型读取全部当前依据，并要求保持原印象。"""
        seen_payload.update(json.loads(payload))
        return {
            "impression_text": seen_payload["current_impression"],
            "reason": "现有印象仍准确",
        }

    service._generator = generate
    service._append_review_log = _async_value("update-1")  # type: ignore[method-assign]
    write = AsyncMock(return_value=True)
    monkeypatch.setattr(
        persona_service.person_api, "update_user_impression", write,
    )
    result = await service.refresh("person-a", (_change(),))
    assert result is not None and result.changed is False
    write.assert_not_awaited()
    assert "计划下周联系" in seen_payload["active_memories"][0]["content"]
    assert seen_payload["active_memories"] == list(memories)
    assert seen_payload["current_impression"] == inline_impression
    assert "max_length" not in seen_payload


def test_memory_footnotes_group_references_and_reuse_numbers() -> None:
    """相邻依据共用圈号，相同依据复用，正文与引用可往返。"""
    inline_text = (
        "他让人安心[Memory: memory-1] [Memory: memory-2]。\n"
        "也有些距离[Memory: memory-3]。\n"
        "相处却不紧绷[Memory: memory-1] [Memory: memory-2]。"
    )
    expected = (
        "他让人安心\u2460。\n也有些距离\u2461。\n相处却不紧绷\u2460。"
        "\n\n记忆依据：\n"
        "\u2460 [Memory: memory-1] [Memory: memory-2]\n"
        "\u2461 [Memory: memory-3]"
    )
    assert _format_memory_footnotes(inline_text) == expected
    assert _inline_memory_references(expected) == inline_text
    assert _format_memory_footnotes(expected) == expected
    assert MEMORY_REFERENCE.findall(expected) == ["memory-1", "memory-2", "memory-3"]
    assert _format_memory_footnotes("") == ""


def test_memory_footnotes_do_not_limit_reference_count() -> None:
    """圈号覆盖二十一到五十，更多依据仍可完整往返。"""
    inline_text = "\n".join(
        f"认识{number}[Memory: memory-{number}]" for number in range(1, 53)
    )
    formatted = _format_memory_footnotes(inline_text)
    assert "认识20\u2473" in formatted
    assert "认识21\u3251" in formatted
    assert "认识35\u325f" in formatted
    assert "认识36\u32b1" in formatted
    assert "认识50\u32bf" in formatted
    assert "认识51[51]" in formatted
    assert _inline_memory_references(formatted) == inline_text
    assert _format_memory_footnotes(formatted) == formatted


@pytest.mark.asyncio
async def test_refresh_writes_footnotes_and_keeps_audit_memory_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧行内印象经校验转为尾注，审计仍记录真实记忆 ID。"""
    inline_text = "相处很自在[Memory: memory-1] [Memory: memory-2]。"
    expected = (
        "相处很自在\u2460。\n\n记忆依据：\n"
        "\u2460 [Memory: memory-1] [Memory: memory-2]"
    )
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a",
        impression=inline_text, updated_at=None,
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.get_persona = _async_value(SimpleNamespace(
        impression_text=expected,
    ))  # type: ignore[method-assign]
    service.is_current_impression = _async_value(True)  # type: ignore[method-assign]
    service._load_active_memories = _async_value((
        {"memory_id": "memory-1"}, {"memory_id": "memory-2"},
    ))  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._generator = _async_value({
        "impression_text": inline_text, "reason": "现有认识仍准确",
    })
    audit = AsyncMock(return_value="update-footnotes")
    service._append_review_log = audit  # type: ignore[method-assign]
    async def update_impression(platform: str, user_id: str, text: str) -> bool:
        """模拟核心 API 对当前正文的真实覆盖。"""
        person.impression = text
        return True

    write = AsyncMock(side_effect=update_impression)
    monkeypatch.setattr(
        persona_service.person_api, "update_user_impression", write,
    )
    result = await service.refresh("person-a", (_change(),))
    assert result is not None and result.changed is True
    write.assert_awaited_once_with("test", "a", expected)
    assert audit.await_args is not None
    assert audit.await_args.args[2] == ("memory-1", "memory-2")


@pytest.mark.parametrize("impression_text", [
    "新的认识[Memory: memory-unknown]",
    "新的认识[Memory: memory-1, memory-2]",
    "没有依据的新认识",
])
@pytest.mark.asyncio
async def test_refresh_rejects_invalid_references_before_footnote_formatting(
    monkeypatch: pytest.MonkeyPatch,
    impression_text: str,
) -> None:
    """尾注排版不放宽真实 ID 校验，也不补造缺失引用。"""
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a", impression="",
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service._load_active_memories = _async_value((
        {"memory_id": "memory-1"}, {"memory_id": "memory-2"},
    ))  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._generator = _async_value({
        "impression_text": impression_text, "reason": "补充认识",
    })
    service.is_current_impression = _async_value(False)  # type: ignore[method-assign]
    write = AsyncMock()
    monkeypatch.setattr(
        persona_service.person_api, "update_user_impression", write,
    )
    with pytest.raises(ValueError, match="Persona 引用了|非空人物印象必须引用"):
        await service.refresh("person-a", (_change(),))
    write.assert_not_awaited()


def test_persona_config_ignores_obsolete_length_limit() -> None:
    """废弃的人物长度和独立模型字段不生效，查询配置与输入保持不变。"""
    data = {
        "internal_llm": {"task_name": "example-task"},
        "vnext": {"persona": {"max_length": 500, "recent_memory_limit": 7}},
    }
    config = EngramMemoryConfig.from_dict(data)
    assert config.vnext.persona.model_dump() == {
        "recent_memory_limit": 7, "max_concurrency": 3,
        "recent_chat_days": 7, "recent_chat_max_messages": 500,
    }
    assert "internal_llm" not in config.model_dump()
    assert data["internal_llm"]["task_name"] == "example-task"
    assert data["vnext"]["persona"]["max_length"] == 500


@pytest.mark.asyncio
async def test_refresh_allows_empty_impression_without_active_memory_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最后依据撤回后可以清空人物印象和引用。"""
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a",
        impression="过去的印象 [Memory: memory-old]", updated_at=None,
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.get_persona = _async_value(SimpleNamespace(impression_text=""))  # type: ignore[method-assign]
    service.is_current_impression = _async_value(True)  # type: ignore[method-assign]
    service._load_active_memories = _async_value(())  # type: ignore[method-assign]
    service._load_change_context = _async_value(({
        "memory_id": "memory-old", "status": "TOMBSTONED", "revisions": (),
    },))  # type: ignore[method-assign]
    service._generator = _async_value({
        "impression_text": "", "reason": "唯一依据已撤回",
    })
    service._append_review_log = _async_value("update-2")  # type: ignore[method-assign]

    async def update_impression(*args: object) -> bool:
        """接受清空印象的写入。"""
        person.impression = ""
        return True

    monkeypatch.setattr(persona_service.person_api, "update_user_impression", update_impression)
    result = await service.refresh("person-a", (_change(),))
    assert result is not None and result.changed is True


@pytest.mark.asyncio
async def test_updater_splits_people_and_processes_arrivals_during_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """变化前后人物均收到通知，刷新期间的新变化进入下一批。"""
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[tuple[str, tuple[MemoryChanged, ...]]] = []

    class Service:
        """用可暂停刷新检查同人新变化的入队。"""

        async def refresh(
            self, person_id: str, changes: tuple[MemoryChanged, ...],
        ) -> object:
            """记录刷新批次并在首批等待测试继续。"""
            calls.append((person_id, changes))
            if len(calls) == 1:
                started.set()
                await release.wait()
            return object()

    def create_task(coro: Any, *, name: str, daemon: bool) -> ManagedTaskHandle:
        """将框架任务边界替换为本测试事件循环。"""
        task = asyncio.create_task(coro, name=name)
        return ManagedTaskHandle(name, task)

    monkeypatch.setattr(
        persona_updater, "create_managed_task", create_task,
    )
    updater = PersonaUpdater(Service(), _Repository(), max_concurrency=3)  # type: ignore[arg-type]
    await updater.enqueue(_change(before=("person-a",), after=("person-b",)))
    await started.wait()
    running_person = calls[0][0]
    await updater.enqueue(_change("memory-2", after=(running_person,)))
    release.set()
    assert updater._task is not None and updater._task.task is not None
    await updater._task.task
    assert {person_id for person_id, _ in calls} == {"person-a", "person-b"}
    running_batches = [batch for person_id, batch in calls if person_id == running_person]
    assert any(change.memory_id == "memory-2" for batch in running_batches for change in batch)


@pytest.mark.parametrize("source_version", [2, 3])
def test_persona_schema_migration_preserves_old_audit(
    tmp_path: Any, source_version: int,
) -> None:
    """副本扩展审计字段，旧正文及版本不凭空补造，来源字节保持不变。"""
    import sqlite3
    from contextlib import closing
    from ..scripts.migrate_schema import migrate_copy
    from ..vnext.schema import SCHEMA_KEY, SCHEMA_VERSION

    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.executescript(
            "CREATE TABLE engram_vnext_memory(memory_id TEXT);"
            "CREATE TABLE engram_vnext_memory_revision(revision_id TEXT);"
            "CREATE TABLE engram_vnext_evidence(evidence_id TEXT);"
            "CREATE TABLE engram_vnext_schema_version(schema_key TEXT,version INTEGER,applied_at TEXT);"
            "CREATE TABLE engram_vnext_persona_update_log("
            "update_id TEXT,person_id TEXT,old_content_hash TEXT,new_content_hash TEXT,"
            "reason TEXT,created_at TEXT);"
            "INSERT INTO engram_vnext_persona_update_log VALUES ('old','person-a','a','b','legacy','time');"
        )
        if source_version == 2:
            connection.execute("CREATE TABLE engram_vnext_person_persona(impression_text TEXT)")
        connection.execute(
            "INSERT INTO engram_vnext_schema_version VALUES (?,?,?)",
            (SCHEMA_KEY, source_version, "time"),
        )
        connection.commit()
    before = source.read_bytes()
    report = migrate_copy(source, target)
    assert source.read_bytes() == before
    assert report["source_unchanged"] is True
    with closing(sqlite3.connect(target)) as connection:
        row = connection.execute("SELECT * FROM engram_vnext_persona_update_log").fetchone()
        assert row == ("old", "person-a", "a", "b", "legacy", "time", None, None, None)
        assert connection.execute("SELECT version FROM engram_vnext_schema_version").fetchone() == (SCHEMA_VERSION,)


@pytest.mark.asyncio
async def test_bootstrap_excludes_legacy_baseline_and_keeps_it_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未认证的旧印象完全不进底稿，空补建结果不覆盖旧记录也不认证。"""
    from ..vnext import persona_service

    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a", impression="旧系统残留正文",
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.is_current_impression = _async_value(False)  # type: ignore[method-assign]
    service._load_active_memories = _async_value(({"memory_id": "memory-1"},))  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._load_recent_chat = _async_value(({"stream_id": "stream-a"},))  # type: ignore[method-assign]
    audit, write = AsyncMock(), AsyncMock()
    service._append_review_log = audit  # type: ignore[method-assign]
    monkeypatch.setattr(persona_service.person_api, "update_user_impression", write)

    async def generate(payload: str) -> dict[str, object]:
        """确认旧底稿排除及聊天辅助字段存在。"""
        data = json.loads(payload)
        assert data["current_impression"] == ""
        assert data["recent_chat"] == [{"stream_id": "stream-a"}]
        return {"impression_text": "", "reason": "没有形成认识"}

    service._generator = generate
    with pytest.raises(ValueError, match="不能以空正文"):
        await service.refresh("person-a")
    assert person.impression == "旧系统残留正文"
    write.assert_not_awaited()
    audit.assert_not_awaited()
    service._load_active_memories = _async_value(())  # type: ignore[method-assign]
    result = await service.refresh("person-a")
    assert result is not None and result.update_id is None
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_history_certification_and_revision_numbers(tmp_path: Any) -> None:
    """首次、保留、变更、清空分别产生 1、无新版本、2、3，认证匹配当前正文。"""
    from ..vnext.schema import VNextSchema

    schema = VNextSchema(str(tmp_path / "history.db"))
    await schema.initialize()
    try:
        service = PersonaService(schema)
        assert not await service.is_current_impression("person-a", "legacy")
        await service._append_review_log("person-a", "首次形成", (), "old", "hash-a", impression_text="alpha")
        await service._append_review_log("person-a", "认识保留", (), "hash-a", "hash-a", impression_text="alpha")
        await service._append_review_log("person-a", "认识变化", (), "hash-a", "hash-b", impression_text="beta")
        from ..vnext.persona_service import _content_hash

        await service._append_review_log("person-a", "依据撤回", (), "hash-b", _content_hash(""), impression_text="")
        history = await service.get_history("person-a")
        assert [row["revision_no"] for row in history] == [3, 2, 1]
        assert all("impression_text" not in row and row["historical"] for row in history)
        assert (await service.get_history("person-a", 1))[0]["impression_text"] == "alpha"
        assert (await service.get_history("person-a", 3))[0]["impression_text"] == ""
        assert await service.is_current_impression("person-a", "")
        assert not await service.is_current_impression("person-a", "external-edit")
    finally:
        await schema.close()


@pytest.mark.parametrize("source_version", [1, 2, 3])
@pytest.mark.asyncio
async def test_schema_copy_preserves_real_memory_and_legacy_audit(
    tmp_path: Any, source_version: int,
) -> None:
    """各旧结构的真实正式数据迁到 v4，旧审计不补造版本且旧库启动只读拒绝。"""
    import sqlite3
    from contextlib import closing
    from datetime import UTC, datetime
    from ..scripts.migrate_schema import migrate_copy
    from ..vnext.domain import CreateMemoryInput, EvidenceInput, SubjectInput, WriteContext
    from ..vnext.enums import ActorType, EvidenceSourceType, MemoryKind, SubjectKind
    from ..vnext.memory_service import MemoryService
    from ..vnext.schema import VNextSchema, SCHEMA_VERSION

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    schema = VNextSchema(str(source))
    await schema.initialize()
    now = datetime.now(UTC)
    try:
        saved = await MemoryService(schema, "test-embedding").create_memory(CreateMemoryInput(
            title="示例正式记忆", content="保持完整的原始记忆正文。", memory_kind=MemoryKind.EVENT,
            subject=SubjectInput(SubjectKind.PERSON, person_id="person-a"), observed_at=now,
            evidence=(EvidenceInput(EvidenceSourceType.ADMIN, now, note="示例来源"),),
        ), WriteContext(ActorType.ADMIN))
        await PersonaService(schema)._append_review_log(
            "person-a", "旧审查", (), "old", "new", impression_text="旧正文",
        )
    finally:
        await schema.close()
    with closing(sqlite3.connect(source)) as connection:
        connection.executescript(
            "CREATE TABLE legacy_persona_log("
            "update_id VARCHAR(36) PRIMARY KEY,person_id TEXT NOT NULL,sleep_session_id VARCHAR(36),"
            "old_content_hash TEXT NOT NULL,new_content_hash TEXT NOT NULL,"
            "reason TEXT NOT NULL,created_at DATETIME NOT NULL,"
            "FOREIGN KEY(sleep_session_id) REFERENCES engram_vnext_sleep_session(sleep_session_id));"
            "INSERT INTO legacy_persona_log SELECT update_id,person_id,sleep_session_id,"
            "old_content_hash,new_content_hash,reason,created_at FROM engram_vnext_persona_update_log;"
            "DROP TABLE engram_vnext_persona_update_log;"
            "ALTER TABLE legacy_persona_log RENAME TO engram_vnext_persona_update_log;"
            "CREATE INDEX idx_engram_vnext_persona_log_person "
            "ON engram_vnext_persona_update_log(person_id,created_at);"
        )
        connection.execute("UPDATE engram_vnext_schema_version SET version=?", (source_version,))
        if source_version == 2:
            connection.execute("CREATE TABLE engram_vnext_person_persona(person_id TEXT, impression_text TEXT)")
            connection.execute("INSERT INTO engram_vnext_person_persona VALUES ('person-a','legacy')")
        if source_version == 1:
            connection.execute("ALTER TABLE engram_vnext_memory_revision ADD COLUMN event_start_at TEXT")
            connection.execute("ALTER TABLE engram_vnext_memory_revision ADD COLUMN event_end_at TEXT")
        connection.commit()
    before = source.read_bytes()
    incompatible = VNextSchema(str(source))
    try:
        with pytest.raises(RuntimeError, match="显式副本迁移"):
            await incompatible.initialize()
    finally:
        await incompatible.close()
    assert source.read_bytes() == before
    report = migrate_copy(source, target)
    assert report["source_unchanged"] is True
    with closing(sqlite3.connect(target)) as connection:
        assert connection.execute("SELECT version FROM engram_vnext_schema_version").fetchone() == (SCHEMA_VERSION,)
    assert source.read_bytes() == before
    migrated = VNextSchema(str(target))
    await migrated.initialize()
    try:
        from ..vnext.repository import MemoryRepository
        revision = await MemoryRepository(migrated).get_current_revision(saved.memory_id)
        assert revision is not None and revision.revision_id == saved.revision_id
        assert revision.content == "保持完整的原始记忆正文。"
        service = PersonaService(migrated)
        assert not await service.is_current_impression("person-a", "旧正文")
        assert await service.get_history("person-a") == ()
    finally:
        await migrated.close()


@pytest.mark.asyncio
async def test_recent_chat_keeps_speakers_replies_and_total_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """辅助片段保留他人和 Bot，跨片段计数并标记截断。"""
    from datetime import UTC, datetime
    from ..vnext import persona_service

    now = datetime.now(UTC).timestamp() - 120
    anchors = [
        {"stream_id": "stream-a", "time": now - 1000},
        {"stream_id": "stream-b", "time": now},
    ]
    monkeypatch.setattr(persona_service.message_api, "get_messages_by_time_for_users", AsyncMock(return_value=anchors))
    messages = [{
        "message_id": f"message-{index}", "time": now + index,
        "platform": "test", "sender_id": sender, "sender_name": sender,
        "person_id": identity, "content": "完整聊天正文", "reply_to": f"message-{index - 1}",
    } for index, (sender, identity) in enumerate([
        ("a", "person-a"), ("other", "person-other"), ("bot-account", None),
    ])]
    load = AsyncMock(return_value=messages)
    monkeypatch.setattr(persona_service.message_api, "get_messages_by_time_in_chat_inclusive", load)
    monkeypatch.setattr(persona_service.adapter_api, "get_bot_info_by_platform", AsyncMock(return_value={"bot_id": "bot-account"}))
    settings = EngramMemoryConfig().vnext.persona
    settings.recent_chat_max_messages = 3
    service = PersonaService(object(), persona_config=settings)  # type: ignore[arg-type]
    blocks = await service._load_recent_chat(
        SimpleNamespace(platform="test", user_id="a"), ("person-a",),  # type: ignore[arg-type]
    )
    assert len(blocks) == 1
    assert [row["role"] for row in blocks[0]["messages"]] == ["target", "other", "bot"]
    assert len(blocks[0]["messages"]) == 3
    assert blocks[0]["messages"][2]["reply_to"] == "message-1"
    assert load.await_args.kwargs["filter_bot"] is False
    assert load.await_args.kwargs["limit"] == 4


@pytest.mark.asyncio
async def test_updater_shares_three_slots_and_deduplicates_aliases(
    managed_tasks: dict[str, asyncio.Task[Any]],
) -> None:
    """补建和变化共用三个名额，相同人物的不同身份不并行且新批次不丢失。"""
    ready, release = asyncio.Event(), asyncio.Event()
    active: set[str] = set()
    maximum = 0
    calls: list[tuple[str, tuple[MemoryChanged, ...]]] = []

    class Repository(_Repository):
        """将示例平台身份归一到同一个核心人物。"""

        async def resolve_person_aliases(self, person_id: str) -> tuple[str, ...]:
            """返回稳定核心标识和平台别名。"""
            return ("person-a", "test:a") if person_id in {"person-a", "test:a"} else (person_id,)

    class Service:
        """模拟启动目录并阻塞生成以观察真实占位。"""

        async def get_active_person_ids(self) -> tuple[str, ...]:
            """目录包含重复人物别名和一个已完成的人物。"""
            return ("person-a", "test:a", "person-b", "person-c", "person-d", "person-ready")

        async def get_persona(self, person_id: str) -> object:
            """已完成的人物无需补建。"""
            return SimpleNamespace(impression_text="新版认识" if person_id == "person-ready" else "")

        async def refresh(self, person_id: str, changes: tuple[MemoryChanged, ...]) -> object:
            """保证同人唯一，占位满三人后等待测试释放。"""
            nonlocal maximum
            assert person_id not in active
            active.add(person_id)
            maximum = max(maximum, len(active))
            calls.append((person_id, changes))
            if len(active) == 3:
                ready.set()
            try:
                await release.wait()
                return object()
            finally:
                active.remove(person_id)

    updater = PersonaUpdater(Service(), Repository(), max_concurrency=3)  # type: ignore[arg-type]
    updater.start()
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert maximum == 3
        await updater.enqueue(_change("new-memory", after=("test:a", "person-a", "person-e")))
        assert len(updater._running) == 3
        release.set()
        if updater._bootstrap is not None:
            await updater._bootstrap.task
        assert updater._task is not None
        await updater._task.task
        assert maximum == 3
        assert {person_id for person_id, _ in calls} == {
            "person-a", "person-b", "person-c", "person-d", "person-e",
        }
        assert sum(person_id == "person-a" for person_id, _ in calls) == 2
        assert any(person_id == "person-a" and len(batch) == 1 for person_id, batch in calls)
    finally:
        await updater.close()
    assert managed_tasks
    assert all(task.done() for task in managed_tasks.values())


@pytest.mark.asyncio
async def test_retry_requeues_stale_and_failing_batches_without_occupying_slot(
    managed_tasks: dict[str, asyncio.Task[Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """过期及异常保留批次，等待重试时名额可用，最多三次后停止。"""
    from ..vnext import persona_updater

    retry_waiting, release_retry = asyncio.Event(), asyncio.Event()
    calls: list[tuple[str, tuple[MemoryChanged, ...]]] = []
    healthy_finished = asyncio.Event()

    async def wait_backoff(delay: float) -> None:
        """测试控制重试等待，不消耗真实时间。"""
        retry_waiting.set()
        await release_retry.wait()

    monkeypatch.setattr(persona_updater.asyncio, "sleep", wait_backoff)

    class Service:
        """一个人物第一次过期，之后异常，另一个正常完成。"""

        async def refresh(self, person_id: str, changes: tuple[MemoryChanged, ...]) -> object:
            """记录每次批次并提供过期和异常结果。"""
            calls.append((person_id, changes))
            if person_id == "person-healthy":
                healthy_finished.set()
                return object()
            if sum(identity == person_id for identity, _ in calls) == 1:
                return None
            raise ValueError("model-test")

    updater = PersonaUpdater(Service(), _Repository(), max_concurrency=1)  # type: ignore[arg-type]
    try:
        await updater.enqueue(_change())
        await asyncio.wait_for(retry_waiting.wait(), 2)
        assert not updater._running
        await updater.enqueue(_change("healthy-memory", after=("person-healthy",)))
        await asyncio.wait_for(healthy_finished.wait(), 2)
        release_retry.set()
        for _ in range(5):
            waiting = [task for task in managed_tasks.values() if not task.done()]
            if not waiting:
                break
            await asyncio.wait_for(asyncio.gather(*waiting), 2)
        assert sum(person_id == "person-a" for person_id, _ in calls) == 3
        assert all(batch == (_change(),) for person_id, batch in calls if person_id == "person-a")
        assert updater._pending["person-a"] == [_change()]
        assert "person-a" in updater.last_errors
    finally:
        await updater.close()
    assert managed_tasks
    assert all(task.done() for task in managed_tasks.values())


@pytest.mark.asyncio
async def test_close_cancels_running_and_waiting_persona_tasks(
    managed_tasks: dict[str, asyncio.Task[Any]],
) -> None:
    """卸载等待正在生成的人物任务停止，不在关闭后再安排工作。"""
    started, stopped, never = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Service:
        """模拟正在请求模型的人物任务。"""

        async def refresh(self, person_id: str, changes: tuple[MemoryChanged, ...]) -> object:
            """取消时确认生成任务已释放。"""
            started.set()
            try:
                await never.wait()
            finally:
                stopped.set()
            return object()

    updater = PersonaUpdater(Service(), _Repository(), max_concurrency=1)  # type: ignore[arg-type]
    await updater.enqueue(_change(after=("person-a", "person-b")))
    await asyncio.wait_for(started.wait(), 2)
    await updater.close()
    assert stopped.is_set()
    assert not updater._running and not updater._pending and not updater._retries
    assert managed_tasks
    assert all(task.done() for task in managed_tasks.values())


@pytest.mark.parametrize("changed_input", ["memory", "core_impression"])
@pytest.mark.asyncio
async def test_stale_generation_does_not_write_or_certify(
    monkeypatch: pytest.MonkeyPatch, changed_input: str,
) -> None:
    """生成期间正式依据或核心正文改变时丢弃结果，不生成历史或成功标记。"""
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a", impression="旧残留",
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.is_current_impression = _async_value(False)  # type: ignore[method-assign]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    memories = ({"memory_id": "memory-1", "revision_id": "revision-1"},)
    service._load_active_memories = AsyncMock(side_effect=[
        memories,
        ({"memory_id": "memory-1", "revision_id": "revision-2"},) if changed_input == "memory" else memories,
    ])  # type: ignore[method-assign]

    async def generate(payload: str) -> dict[str, object]:
        """模拟另一个写入者在请求期间修改核心印象。"""
        if changed_input == "core_impression":
            person.impression = "其他写入者的正文"
        return {"impression_text": "感觉很自在[Memory: memory-1]", "reason": "形成认识"}

    service._generator = generate
    audit, write = AsyncMock(), AsyncMock()
    service._append_review_log = audit  # type: ignore[method-assign]
    monkeypatch.setattr(persona_service.person_api, "update_user_impression", write)
    assert await service.refresh("person-a") is None
    write.assert_not_awaited()
    audit.assert_not_awaited()


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.asyncio
async def test_core_write_failure_or_mismatched_reread_never_certifies(
    monkeypatch: pytest.MonkeyPatch, accepted: bool,
) -> None:
    """核心拒绝保存或报告成功却未回读一致时不能认证和保存历史。"""
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a", impression="旧残留",
    )
    service = PersonaService(object())  # type: ignore[arg-type]
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.is_current_impression = _async_value(False)  # type: ignore[method-assign]
    service._load_active_memories = _async_value(({"memory_id": "memory-1"},))  # type: ignore[method-assign]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._generator = _async_value({"impression_text": "认识[Memory: memory-1]", "reason": "形成认识"})
    audit = AsyncMock()
    service._append_review_log = audit  # type: ignore[method-assign]
    monkeypatch.setattr(persona_service.person_api, "update_user_impression", AsyncMock(return_value=accepted))
    with pytest.raises(ValueError, match="核心人物印象更新失败|回读与写入不一致"):
        await service.refresh("person-a")
    assert person.impression == "旧残留"
    audit.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_snapshots_reject_mutation(tmp_path: Any) -> None:
    """历史正文继续沿用追加式 ORM 约束，不允许覆盖或删除。"""
    from ..vnext.models import PersonaUpdateLogModel
    from ..vnext.schema import VNextSchema

    schema = VNextSchema(str(tmp_path / "immutable.db"))
    await schema.initialize()
    try:
        update_id = await PersonaService(schema)._append_review_log(
            "person-a", "首次认识", (), "old", "new", impression_text="历史正文",
        )
        with pytest.raises(ValueError, match="追加式历史记录"):
            async with schema.database.session() as session:
                row = await session.get(PersonaUpdateLogModel, update_id)
                row.impression_text = "替换的正文"
                await session.flush()
        with pytest.raises(ValueError, match="追加式历史记录"):
            async with schema.database.session() as session:
                row = await session.get(PersonaUpdateLogModel, update_id)
                await session.delete(row)
                await session.flush()
        assert (await PersonaService(schema).get_history("person-a", 1))[0]["impression_text"] == "历史正文"
    finally:
        await schema.close()


@pytest.mark.asyncio
async def test_chat_samples_contiguous_blocks_across_time_with_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """多片段按时间分布取样，共用消息总额且明确标记开头截断。"""
    from datetime import UTC, datetime

    end = datetime.now(UTC).timestamp() - 120
    anchors = [{"stream_id": f"stream-{index}", "time": end - (2 - index) * 1000} for index in range(3)]
    monkeypatch.setattr(persona_service.message_api, "get_messages_by_time_for_users", AsyncMock(return_value=anchors))

    async def read_block(stream_id: str, start: float, stop: float, **kwargs: Any) -> list[dict[str, object]]:
        """返回各流中的连续超额片段，模型不接收压缩或打散后的句子。"""
        timestamp = next(float(row["time"]) for row in anchors if row["stream_id"] == stream_id)
        return [{
            "message_id": f"{stream_id}-message-{index}", "time": timestamp + index * 0.5,
            "person_id": "person-a", "platform": "test", "sender_id": "a", "content": "完整正文",
        } for index in range(60)]

    monkeypatch.setattr(persona_service.message_api, "get_messages_by_time_in_chat_inclusive", read_block)
    monkeypatch.setattr(persona_service.adapter_api, "get_bot_info_by_platform", AsyncMock(return_value=None))
    settings = EngramMemoryConfig().vnext.persona
    settings.recent_chat_max_messages = 100
    blocks = await PersonaService(object(), persona_config=settings)._load_recent_chat(  # type: ignore[arg-type]
        SimpleNamespace(platform="test", user_id="a"), ("person-a",),  # type: ignore[arg-type]
    )
    assert [row["stream_id"] for row in blocks] == ["stream-0", "stream-2"]
    assert sum(len(row["messages"]) for row in blocks) == 100
    assert all(row["partial_start"] for row in blocks)
    assert all(len(row["messages"]) == 50 for row in blocks)


@pytest.mark.asyncio
async def test_formal_memory_bootstrap_history_lookup_and_withdrawal(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实正式记忆经过补建、局部保留及撤回，查询不生成也不将历史当依据。"""
    from datetime import UTC, datetime
    from ..vnext.domain import CreateMemoryInput, EvidenceInput, MemoryLifecycleInput, SubjectInput, WriteContext
    from ..vnext.enums import ActorType, EvidenceSourceType, MemoryKind, SubjectKind
    from ..vnext.memory_service import MemoryService
    from ..vnext.schema import VNextSchema
    from ..vnext.tool_service import ToolContext, VNextToolService

    schema = VNextSchema(str(tmp_path / "flow.db"))
    await schema.initialize()
    now = datetime.now(UTC)
    person = SimpleNamespace(
        person_id="person-a", platform="test", user_id="a", nickname="示例人物",
        cardname=None, impression="旧印象残留", updated_at=now.timestamp(),
    )
    memory = MemoryService(schema, "example-embedding")
    service = PersonaService(schema)
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service._load_recent_chat = _async_value(())  # type: ignore[method-assign]
    baselines: list[str] = []

    async def generate(payload: str) -> dict[str, object]:
        """首次形成后原样保留，使用真实当前 Memory ID。"""
        data = json.loads(payload)
        baselines.append(data["current_impression"])
        return {
            "impression_text": data["current_impression"] or f"相处很自在[Memory: {data['active_memories'][0]['memory_id']}]。",
            "reason": "根据正式记忆形成认识" if not data["current_impression"] else "认识仍有依据",
        }

    async def write_impression(platform: str, user_id: str, text: str) -> bool:
        """模拟核心保存和后续回读的相同对象。"""
        person.impression = text
        return True

    service._generator = generate
    write = AsyncMock(side_effect=write_impression)
    monkeypatch.setattr(persona_service.person_api, "update_user_impression", write)
    tools = VNextToolService(schema, SimpleNamespace())  # type: ignore[arg-type]
    tools._persona = service
    context = ToolContext(ActorType.ACTOR)
    try:
        assert (await tools.person_lookup("person-a", context))["persona_impression"] == "暂无人物印象"
        assert not baselines and person.impression == "旧印象残留"
        saved = await memory.create_memory(CreateMemoryInput(
            title="共同安排", content="双方已确认一起讨论项目。", memory_kind=MemoryKind.EVENT,
            subject=SubjectInput(SubjectKind.PERSON, person_id="person-a"), observed_at=now,
            evidence=(EvidenceInput(EvidenceSourceType.ADMIN, now, note="示例来源"),),
        ), WriteContext(ActorType.ADMIN))
        assert await service.get_active_person_ids() == ("person-a",)
        first = await service.refresh("person-a")
        assert first is not None and first.changed
        assert baselines == [""] and write.await_count == 1
        first_text = person.impression
        assert (await service.get_persona("person-a")).is_current
        unchanged = await service.refresh("person-a", (_change(saved.memory_id),))
        assert unchanged is not None and not unchanged.changed
        assert write.await_count == 1
        assert len(await service.get_history("person-a")) == 1
        current = await tools.person_lookup("person-a", context)
        assert current["persona_impression"] == first_text and "persona_history" not in current
        before_lookup = len(baselines)
        assert len((await tools.person_lookup("person-a", context, view="history"))["persona_history"]) == 1
        revision = await tools.person_lookup("person-a", context, view="revision", revision_no=1)
        assert revision["persona_revision"]["impression_text"] == first_text
        assert revision["persona_revision"]["historical"] is True
        assert len(baselines) == before_lookup
        person.impression = "外部修改的残留"
        assert not (await service.get_persona("person-a")).is_current
        assert (await tools.person_lookup("person-a", context))["persona_impression"] == "暂无人物印象"
        person.impression = first_text
        await memory.tombstone_memory(
            MemoryLifecycleInput(saved.memory_id, "依据撤回"), WriteContext(ActorType.ADMIN),
        )
        cleared = await service.refresh("person-a", (_change(saved.memory_id),))
        assert cleared is not None and cleared.changed and person.impression == ""
        assert len(baselines) == before_lookup
        assert [row["revision_no"] for row in await service.get_history("person-a")] == [2, 1]
        assert (await tools.person_lookup("person-a", context))["persona_impression"] == "暂无人物印象"
        assert (await service.get_history("person-a", 1))[0]["impression_text"] == first_text
        assert await service.get_active_person_ids() == ()
    finally:
        await schema.close()