"""人物印象更新与变化队列测试。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.engram_memory.config import EngramMemoryConfig
from plugins.engram_memory.vnext.domain import MemoryChanged
from plugins.engram_memory.vnext.enums import MemoryEventType
from plugins.engram_memory.vnext.persona_service import (
    MEMORY_REFERENCE,
    PersonaService,
    _format_memory_footnotes,
    _inline_memory_references,
)
from plugins.engram_memory.vnext.persona_updater import PersonaUpdater
from plugins.engram_memory.vnext.framework_bridge import ManagedTaskHandle


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
        "plugins.engram_memory.vnext.persona_service.person_api.update_user_impression",
        write,
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
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service.get_persona = _async_value(SimpleNamespace(
        impression_text=expected,
    ))  # type: ignore[method-assign]
    service._load_active_memories = _async_value((
        {"memory_id": "memory-1"}, {"memory_id": "memory-2"},
    ))  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._generator = _async_value({
        "impression_text": inline_text, "reason": "现有认识仍准确",
    })
    audit = AsyncMock(return_value="update-footnotes")
    service._append_review_log = audit  # type: ignore[method-assign]
    write = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "plugins.engram_memory.vnext.persona_service.person_api.update_user_impression",
        write,
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
    service._repository = _Repository()  # type: ignore[assignment]
    service.get_core_person = _async_value(person)  # type: ignore[method-assign]
    service._load_active_memories = _async_value((
        {"memory_id": "memory-1"}, {"memory_id": "memory-2"},
    ))  # type: ignore[method-assign]
    service._load_change_context = _async_value(())  # type: ignore[method-assign]
    service._generator = _async_value({
        "impression_text": impression_text, "reason": "补充认识",
    })
    write = AsyncMock()
    monkeypatch.setattr(
        "plugins.engram_memory.vnext.persona_service.person_api.update_user_impression",
        write,
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
    assert config.vnext.persona.model_dump() == {"recent_memory_limit": 7}
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
        return True

    from plugins.engram_memory.vnext import persona_service

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
        "plugins.engram_memory.vnext.persona_updater.create_managed_task",
        create_task,
    )
    updater = PersonaUpdater(Service(), _Repository())  # type: ignore[arg-type]
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