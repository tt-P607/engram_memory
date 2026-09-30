"""engram_memory 插件配置。"""

from __future__ import annotations

from typing import Any, ClassVar, Self

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section

from .vnext.config_sections import VNextConfig


class EngramMemoryConfig(BaseConfig):
    """Engram Memory 连续记忆配置模型。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "Engram Memory 连续记忆（经历编码、睡眠整理、人物印象与自然闪回）"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """过滤旧配置节后加载当前配置。"""
        filtered: dict[str, Any] = dict(data)
        for section in (
            "retrieval",
            "write_conflict",
            "short_term",
            "flashback",
            "journal",
            "persona",
        ):
            filtered.pop(section, None)

        storage = filtered.get("storage")
        if isinstance(storage, dict):
            filtered_storage = dict(storage)
            filtered_storage.pop("metadata_db_path", None)
            filtered["storage"] = filtered_storage

        return super().from_dict(filtered)

    @config_section("plugin", title="插件设置", tag="plugin")
    class PluginSection(SectionBase):
        """插件级开关。"""

        enabled: bool = Field(
            default=True,
            description="插件总开关",
            label="启用插件",
            tag="plugin",
        )

    @config_section("storage", title="存储配置", tag="database")
    class StorageSection(SectionBase):
        """vNext 数据库与向量索引路径。"""

        vector_db_path: str = Field(
            default="data/engram_memory/chroma",
            description="向量数据库路径（ChromaDB）",
            label="向量库路径",
            input_type="text",
            tag="file",
        )
        vnext_db_path: str = Field(
            default="data/engram_memory/vnext.db",
            description="vNext 规范认知数据库路径（与旧记忆数据库隔离）",
            label="vNext 数据库路径",
            input_type="text",
            tag="file",
        )

    @config_section("internal_llm", title="内部 LLM 配置", tag="ai")
    class InternalLLMSection(SectionBase):
        """候选编码使用的模型任务配置。"""

        task_name: str = Field(
            default="tool_use", description="候选编码模型任务名（chat）"
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    internal_llm: InternalLLMSection = Field(default_factory=InternalLLMSection)

    vnext: VNextConfig = Field(default_factory=VNextConfig)
